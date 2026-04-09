import numpy as np
import numba
from scipy.linalg import cho_solve, cho_factor, blas
from jetgp.full_degp_sparse import degp_utils as utils
from jetgp.full_degp_sparse.sparse_cholesky import (
    build_U, build_U_supernodes, nlml_from_U, alpha_from_U
)
from line_profiler import profile
import jetgp.utils as gen_utils
from jetgp.hyperparameter_optimizers import OPTIMIZERS
from jetgp.utils import matern_kernel_grad_builder


@numba.jit(nopython=True, cache=True)
def _symmetrise_upper(A):
    """Copy upper triangle to lower triangle in-place."""
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            A[j, i] = A[i, j]


@numba.jit(nopython=True, parallel=True, cache=True)
def _permute_and_subtract_outer(K_inv_ord, alpha_v, P_full, W):
    """Fused: W[P[i], P[j]] = K_inv_ord[i, j] - alpha_v[P[i]] * alpha_v[P[j]]
    Reads only the lower triangle of K_inv_ord (as produced by dsyrk with lower=1
    on a Fortran-order buffer).  In column-major layout, lower-triangle entries
    within each column are contiguous → cache-friendly sequential reads.
    W is symmetric, so we write both (pi,pj) and (pj,pi).
    """
    N = len(P_full)
    for i in numba.prange(N):
        pi = P_full[i]
        ai = alpha_v[pi]
        # Diagonal
        W[pi, pi] = K_inv_ord[i, i] - ai * ai
        # j > i  →  read lower triangle: K_inv_ord[j, i] (row > col)
        for j in range(i + 1, N):
            pj = P_full[j]
            val = K_inv_ord[j, i] - ai * alpha_v[pj]
            W[pi, pj] = val
            W[pj, pi] = val


def _build_k_index_map(plan, n_rows_func):
    """
    Build arrays that map each K-matrix index to (deriv_type, physical_point).

    Returns
    -------
    k_type : int64 array of shape (N_total,)
        Derivative type for each K index (0 = function value, 1.. = derivatives).
    k_phys : int64 array of shape (N_total,)
        Physical training point index for each K index.
    deriv_lookup : int64 array of shape (n_types, n_types)
        phi_exp_3d derivative-dimension index for block (type_i, type_j).
    sign_lookup : float64 array of shape (n_types,)
        Sign multiplier for each derivative type.
    """
    n_dt = plan['n_deriv_types']
    n_types = n_dt + 1
    N_total = n_rows_func + plan['n_pts_with_derivs']

    k_type = np.empty(N_total, dtype=np.int64)
    k_phys = np.empty(N_total, dtype=np.int64)

    # Function-value rows: type 0, phys = row index
    k_type[:n_rows_func] = 0
    k_phys[:n_rows_func] = np.arange(n_rows_func)

    # Derivative rows
    idx_flat = plan['idx_flat']
    idx_offsets = plan['idx_offsets']
    idx_sizes = plan['index_sizes']
    row_offsets = plan.get('row_offsets_abs', plan['row_offsets'] + n_rows_func)
    for j in range(n_dt):
        ro = row_offsets[j]
        sz = idx_sizes[j]
        off = idx_offsets[j]
        k_type[ro:ro + sz] = j + 1
        k_phys[ro:ro + sz] = idx_flat[off:off + sz]

    # Derivative lookup: which phi_exp_3d dimension for (type_i, type_j)
    deriv_lookup = np.empty((n_types, n_types), dtype=np.int64)
    deriv_lookup[0, 0] = 0
    fd = plan['fd_flat_indices']
    df = plan['df_flat_indices']
    dd = plan['dd_flat_indices']
    for j in range(n_dt):
        deriv_lookup[0, j + 1] = fd[j]
        deriv_lookup[j + 1, 0] = df[j]
    for i in range(n_dt):
        for j in range(n_dt):
            deriv_lookup[i + 1, j + 1] = dd[i, j]

    # Sign lookup
    signs = plan['signs']
    sign_lookup = np.empty(n_types, dtype=np.float64)
    sign_lookup[0] = signs[0]
    for j in range(n_dt):
        sign_lookup[j + 1] = signs[j + 1]

    return k_type, k_phys, deriv_lookup, sign_lookup


@numba.jit(nopython=True, cache=True)
def _extract_K_sub(phi_exp_3d, nb_type, nb_phys, deriv_lookup, sign_lookup,
                   sigma_n_sq, sigma_data_diag, m, K_sub):
    """
    Assemble K_sub directly from phi_exp_3d for a neighbourhood `nb`.

    nb_type[a], nb_phys[a] give the derivative type and physical point
    for the a-th row/column of K_sub.
    """
    for a in range(m):
        ta = nb_type[a]
        pa = nb_phys[a]
        for b in range(m):
            tb = nb_type[b]
            pb = nb_phys[b]
            d = deriv_lookup[ta, tb]
            K_sub[a, b] = phi_exp_3d[d, pa, pb] * sign_lookup[tb]
        # Add noise to diagonal
        K_sub[a, a] += sigma_n_sq + sigma_data_diag[a]


@numba.jit(nopython=True, cache=True)
def _extract_dK_sub(dphi_exp_3d, nb_type, nb_phys, deriv_lookup, sign_lookup, m, dK_sub):
    """
    Assemble dK_sub/dtheta from dphi_exp_3d for a neighbourhood.

    Same as _extract_K_sub but without noise diagonal (noise doesn't
    depend on kernel hyperparameters).
    """
    for a in range(m):
        ta = nb_type[a]
        pa = nb_phys[a]
        for b in range(m):
            tb = nb_type[b]
            pb = nb_phys[b]
            d = deriv_lookup[ta, tb]
            dK_sub[a, b] = dphi_exp_3d[d, pa, pb] * sign_lookup[tb]


@numba.jit(nopython=True, cache=True)
def _trace_term_all_blocks(dphi_exp_3d, deriv_lookup, sign_lookup,
                           block_nb_type, block_nb_phys, block_m,
                           block_V_flat, block_V_offsets, block_n_cols,
                           n_blocks):
    """
    Compute sum_blocks trace(V^T dK_sub V) in a single numba pass.

    Avoids Python-level block loop and per-block np.empty/matmul overhead.
    """
    result = 0.0
    for b in range(n_blocks):
        m = block_m[b]
        n_cols = block_n_cols[b]
        off = block_V_offsets[b]

        # For each column pair (i, j) of V, compute V[:,i]^T dK_sub V[:,j]
        # trace = sum_i V[:,i]^T dK_sub V[:,i]
        nb_type = block_nb_type[b]
        nb_phys = block_nb_phys[b]

        for col in range(n_cols):
            # Compute V[:,col]^T @ dK_sub @ V[:,col]
            # = sum_a sum_b V[a,col] * dK_sub[a,b] * V[b,col]
            for a in range(m):
                ta = nb_type[a]
                pa = nb_phys[a]
                va = block_V_flat[off + col * m + a]
                for bb in range(m):
                    tb = nb_type[bb]
                    pb = nb_phys[bb]
                    d = deriv_lookup[ta, tb]
                    dk_ab = dphi_exp_3d[d, pa, pb] * sign_lookup[tb]
                    vb = block_V_flat[off + col * m + bb]
                    result += va * dk_ab * vb
    return result


class Optimizer:
    """
    Optimizer class to perform hyperparameter tuning for derivative-enhanced Gaussian Process models
    by minimizing the negative log marginal likelihood (NLL).

    Parameters
    ----------
    model : object
        An instance of a model (e.g., ddegp) containing the necessary training data
        and kernel configuration.
    """

    def __init__(self, model):
        self.model = model
        self._kernel_plan = None
        self._deriv_buf = None
        self._deriv_buf_shape = None
        self._deriv_factors = None
        self._deriv_factors_key = None
        self._K_buf = None
        self._dK_buf = None
        self._kernel_buf_size = None
        self._W_proj_buf = None
        self._W_proj_shape = None
        self._U_buf = None
        self._K_inv_buf = None
        self._P_ix = None
        # Direct phi extraction maps (built lazily)
        self._k_index_map = None
        self._inv_P = None
        self._sigma_data_diag_mmd = None

    def _get_deriv_buf(self, phi, n_bases, order):
        """Return a pre-allocated buffer for get_all_derivs, reusing if shape matches."""
        from math import comb
        ndir = comb(n_bases + order, order)
        shape = (ndir, phi.shape[0], phi.shape[1])
        if self._deriv_buf is None or self._deriv_buf_shape != shape:
            self._deriv_buf = np.zeros(shape, dtype=np.float64)
            self._deriv_buf_shape = shape
        return self._deriv_buf

    def _expand_derivs(self, phi, n_bases, deriv_order):
        """Expand OTI derivatives, using fast struct path if available."""
        if hasattr(phi, 'get_all_derivs_fast'):
            buf = self._get_deriv_buf(phi, n_bases, deriv_order)
            factors = self._get_deriv_factors(n_bases, deriv_order)
            return phi.get_all_derivs_fast(factors, buf)
        return phi.get_all_derivs(n_bases, deriv_order)

    @staticmethod
    def _enum_factors(max_basis, ordi):
        """Enumerate derivative factors in struct memory order for a given order.

        Yields the factorial factor prod(count_b!) for each multi-index of
        the given order, enumerated in the same order as the OTI struct layout
        (last-index-major: for last=1..max_basis, recurse prefix with max=last).
        """
        from math import factorial
        from collections import Counter
        if ordi == 1:
            for _ in range(max_basis):
                yield 1.0
            return
        for last in range(1, max_basis + 1):
            if ordi == 2:
                for i in range(1, last + 1):
                    counts = Counter((i, last))
                    f = 1
                    for c in counts.values():
                        f *= factorial(c)
                    yield float(f)
            else:
                for prefix_factor, prefix_counts in Optimizer._enum_factors_with_counts(last, ordi - 1):
                    counts = dict(prefix_counts)
                    counts[last] = counts.get(last, 0) + 1
                    f = 1
                    for c in counts.values():
                        f *= factorial(c)
                    yield float(f)

    @staticmethod
    def _enum_factors_with_counts(max_basis, ordi):
        """Enumerate (factor, counts_dict) pairs in struct order."""
        from math import factorial
        from collections import Counter
        if ordi == 1:
            for i in range(1, max_basis + 1):
                yield 1.0, {i: 1}
            return
        for last in range(1, max_basis + 1):
            for _, prefix_counts in Optimizer._enum_factors_with_counts(last, ordi - 1):
                counts = dict(prefix_counts)
                counts[last] = counts.get(last, 0) + 1
                f = 1
                for c in counts.values():
                    f *= factorial(c)
                yield float(f), counts

    def _get_deriv_factors(self, n_bases, order):
        """Return cached precomputed derivative factorial factors."""
        key = (n_bases, order)
        if self._deriv_factors is not None and self._deriv_factors_key == key:
            return self._deriv_factors
        factors = [1.0]  # order 0: real part
        for ordi in range(1, order + 1):
            factors.extend(self._enum_factors(n_bases, ordi))
        self._deriv_factors = np.array(factors, dtype=np.float64)
        self._deriv_factors_key = key
        return self._deriv_factors

    def _ensure_kernel_plan(self, n_bases):
        """Lazily precompute kernel plan (once per n_bases)."""
        if self._kernel_plan is not None and self._kernel_plan_n_bases == n_bases:
            return
        if not hasattr(utils, 'precompute_kernel_plan'):
            self._kernel_plan = None
            return
        self._kernel_plan = utils.precompute_kernel_plan(
            self.model.n_order, n_bases,
            self.model.flattened_der_indices,
            self.model.powers,
            self.model.derivative_locations,
        )
        self._kernel_plan_n_bases = n_bases
        # Reset kernel buffers when plan changes
        self._K_buf = None
        self._dK_buf = None
        self._kernel_buf_size = None

    def _ensure_kernel_bufs(self, n_rows_func):
        """Pre-allocate reusable K and dK buffers (avoids repeated malloc)."""
        if self._kernel_plan is None:
            return
        total = n_rows_func + self._kernel_plan['n_pts_with_derivs']
        if self._kernel_buf_size != total:
            self._K_buf = np.empty((total, total))
            self._dK_buf = np.empty((total, total))
            self._kernel_buf_size = total
            # Cache absolute offsets in plan so rbf_kernel_fast doesn't recompute
            if 'row_offsets_abs' not in self._kernel_plan:
                self._kernel_plan['row_offsets_abs'] = self._kernel_plan['row_offsets'] + n_rows_func
                self._kernel_plan['col_offsets_abs'] = self._kernel_plan['col_offsets'] + n_rows_func

    def _ensure_phi_index_maps(self, n_rows_func):
        """Lazily build the K-index-to-phi maps and inverse permutation."""
        if self._k_index_map is not None:
            return
        plan = self._kernel_plan
        k_type, k_phys, deriv_lookup, sign_lookup = _build_k_index_map(
            plan, n_rows_func)
        self._k_index_map = (k_type, k_phys, deriv_lookup, sign_lookup)

        P_full = self.model.mmd_P_full
        inv_P = np.empty_like(P_full)
        inv_P[P_full] = np.arange(len(P_full))
        self._inv_P = inv_P

        # sigma_data diagonal in MMD order
        sd = self.model.sigma_data
        if sd.ndim == 2:
            sd_diag_orig = np.diag(sd) ** 2 if np.any(sd) else np.zeros(len(P_full))
        else:
            sd_diag_orig = np.zeros(len(P_full))
        self._sigma_data_diag_mmd = sd_diag_orig[P_full]

        # Precompute flat index arrays for phi_exp_3d gather.
        # At runtime, K_sub = phi_exp_3d.ravel()[flat_idx] * sign_mat + noise.
        stride_d = n_rows_func * n_rows_func
        stride_row = n_rows_func

        if (self.model.use_supernodes
                and self.model.sparse_supernodes_full is not None):
            for sn in self.model.sparse_supernodes_full:
                ch = sn.get('children_arr')
                if ch is None:
                    ch = np.asarray(sn['children'])
                orig_ch = P_full[ch]
                ch_type = k_type[orig_ch]
                ch_phys = k_phys[orig_ch]
                m = len(ch)

                d_mat = deriv_lookup[ch_type[:, None], ch_type[None, :]]
                pa_mat = np.broadcast_to(ch_phys[:, None], (m, m))
                pb_mat = np.broadcast_to(ch_phys[None, :], (m, m))
                sn['phi_flat_idx'] = np.ascontiguousarray(
                    d_mat * stride_d + pa_mat * stride_row + pb_mat
                )
                sn['phi_sign_mat'] = np.ascontiguousarray(
                    sign_lookup[ch_type[None, :]] * np.ones((m, 1))
                )
                sn['phi_sd_diag'] = self._sigma_data_diag_mmd[ch]

        # Same precomputation for non-supernode block path
        if (not self.model.use_supernodes
                and self.model.n_order > 0):
            N_total = len(P_full)
            block_size = self.model.n_bases + 1
            S = self.model.sparse_S_full_arr
            self._block_phi_maps = []
            for start in range(0, N_total, block_size):
                end = min(start + block_size, N_total)
                nb = S[end - 1] if isinstance(S[end - 1], np.ndarray) else np.asarray(S[end - 1])
                m = len(nb)

                orig_nb = P_full[nb]
                nb_type = k_type[orig_nb]
                nb_phys = k_phys[orig_nb]

                d_mat = deriv_lookup[nb_type[:, None], nb_type[None, :]]
                pa_mat = np.broadcast_to(nb_phys[:, None], (m, m))
                pb_mat = np.broadcast_to(nb_phys[None, :], (m, m))

                self._block_phi_maps.append({
                    'nb': nb,
                    'start': start,
                    'flat_idx': np.ascontiguousarray(
                        d_mat * stride_d + pa_mat * stride_row + pb_mat
                    ),
                    'sign_mat': np.ascontiguousarray(
                        sign_lookup[nb_type[None, :]] * np.ones((m, 1))
                    ),
                    'sd_diag': self._sigma_data_diag_mmd[nb],
                    'positions': np.searchsorted(nb, np.arange(start, end)),
                })

    @profile
    def negative_log_marginal_likelihood(self, x0):
        """
        Compute the negative log marginal likelihood (NLL) via sparse U.

        NLL = 0.5 * ||U^T y||^2 - sum(log|diag(U)|) + 0.5 * N * log(2π)

        Parameters
        ----------
        x0 : ndarray
            Vector of log-scaled hyperparameters (length scales and noise).

        Returns
        -------
        float
            Value of the negative log marginal likelihood.
        """
        try:
            if self.model._use_dense_factor:
                # Dense path: single Cholesky, no sparse U
                W, alpha, nll, *_ = self._dense_nll_and_W(x0)
                if nll > 1e6:
                    return 1e6
                return nll

            # Use direct phi path (skip full K construction) when possible
            if self.model.n_order > 0:
                alpha, U, nlml, *_ = self._sparse_nlml_direct(x0)
            else:
                K, _, _, _, _ = self._build_K_and_phi(x0)
                alpha, U, nlml = self._sparse_U_alpha_nll(K)

            # Sparse U can silently produce bad factors when K is
            # ill-conditioned (e.g. very small noise).  Clamp to 1e6
            # to match the dense fallback behaviour.
            if nlml > 1e6:
                return 1e6

            self.model._cached_U = U
            self.model._cached_P = self.model.mmd_P_full
            self.model._cached_alpha = alpha
            self.model._cached_L = None
            self.model._cached_low = None
            self.model._cached_params = x0.copy()

            return nlml
        except Exception:
            return 1e6

    def nll_wrapper(self, x0):
        """
        Wrapper function to compute NLL for optimizer.

        Parameters
        ----------
        x0 : ndarray
            Hyperparameter vector.

        Returns
        -------
        float
            NLL evaluated at x0.
        """
        return self.negative_log_marginal_likelihood(x0)

    def _compute_grad(self, x0, W, phi, n_bases, oti, diffs):
        """
        Compute the NLL gradient given pre-factorised W = K^{-1} - α α^T.

        Factoring this out allows nll_grad and nll_and_grad to share the
        expensive Cholesky decomposition instead of each rebuilding it.
        """
        ln10        = np.log(10.0)
        kernel      = self.model.kernel
        kernel_type = self.model.kernel_type
        D           = len(diffs)
        sigma_n_sq  = (10.0 ** x0[-1]) ** 2

        grad = np.zeros(len(x0))
        use_fast = self._kernel_plan is not None
        base_shape = (W.shape[0] - self._kernel_plan['n_pts_with_derivs'],) * 2 if use_fast else None

        deriv_order = 2 * self.model.n_order

        # Precompute W projected into phi_exp space to avoid assembling
        # the full dK matrix for each hyperparameter dimension.
        W_proj = None
        if use_fast and self.model.n_order > 0:
            from math import comb
            ndir = comb(n_bases + deriv_order, deriv_order)
            proj_shape = (ndir, base_shape[0], base_shape[1])
            if self._W_proj_buf is None or self._W_proj_shape != proj_shape:
                self._W_proj_buf = np.empty(proj_shape)
                self._W_proj_shape = proj_shape
            W_proj = self._W_proj_buf

            plan = self._kernel_plan
            row_off = plan.get('row_offsets_abs', plan['row_offsets'] + base_shape[0])
            col_off = plan.get('col_offsets_abs', plan['col_offsets'] + base_shape[1])

            utils._project_W_to_phi_space(
                W, W_proj, base_shape[0], base_shape[1],
                plan['fd_flat_indices'], plan['df_flat_indices'],
                plan['dd_flat_indices'],
                plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
                plan['signs'], plan['n_deriv_types'], row_off, col_off,
            )

        _use_vdot_fused = W_proj is not None and hasattr(phi, 'vdot_expand_fast')
        if _use_vdot_fused:
            _vdot_factors = self._get_deriv_factors(n_bases, deriv_order)

        def _gc(dphi):
            if _use_vdot_fused:
                return 0.5 * dphi.vdot_expand_fast(_vdot_factors, W_proj)
            if self.model.n_order == 0:
                dphi_exp = dphi.real[np.newaxis, :, :]
            else:
                dphi_exp = self._expand_derivs(dphi, n_bases, deriv_order)
            if W_proj is not None:
                dphi_3d = dphi_exp.reshape(W_proj.shape)
                return 0.5 * np.vdot(W_proj, dphi_3d)
            elif use_fast:
                dphi_3d = dphi_exp.reshape(dphi_exp.shape[0], base_shape[0], base_shape[1])
                dK = utils.rbf_kernel_fast(dphi_3d, self._kernel_plan, out=self._dK_buf)
                return 0.5 * np.vdot(W, dK)
            else:
                dK = utils.rbf_kernel(
                    dphi, dphi_exp,
                    self.model.n_order, n_bases,
                    self.model.flattened_der_indices, self.model.powers,
                    index=self.model.derivative_locations,
                )
                return 0.5 * np.vdot(W, dK)

        # ── signal variance (common: d phi/d log_sf = 2*ln10 * phi) ──────
        grad[-2] = _gc(oti.mul(2.0 * ln10, phi))

        # ── noise variance (common: dK/d log_sn = diag(2*ln10*σ_n²)) ────
        grad[-1] = ln10 * sigma_n_sq * np.trace(W)

        # ── kernel-specific hyperparameter gradients ──────────────────────

        if kernel == 'SE':
            # phi = sf² * exp(-0.5 * Σ_d ell_d² * diff_d²)
            # d phi/d log_ell_d = -ln10 * ell_d² * diff_d² * phi
            if kernel_type == 'anisotropic':
                ell = 10.0 ** x0[:D]
                if hasattr(phi, 'fused_scale_sq_mul_sparse'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul_sparse(diffs[d], phi, -ln10 * ell[d] ** 2, d)
                        grad[d] = _gc(dphi_buf)
                elif hasattr(phi, 'fused_scale_sq_mul'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul(diffs[d], phi, -ln10 * ell[d] ** 2)
                        grad[d] = _gc(dphi_buf)
                else:
                    for d in range(D):
                        d_sq   = oti.mul(diffs[d], diffs[d])
                        dphi_d = oti.mul(-ln10 * ell[d] ** 2, oti.mul(d_sq, phi))
                        grad[d] = _gc(dphi_d)
            else:  # isotropic: single ell
                ell    = 10.0 ** float(x0[0])
                if hasattr(phi, 'fused_sum_sq_sparse'):
                    sum_sq = oti.zeros(phi.shape)
                    sum_sq.fused_sum_sq_sparse(diffs)
                elif hasattr(phi, 'fused_sum_sq'):
                    sum_sq = oti.zeros(phi.shape)
                    sum_sq.fused_sum_sq(diffs)
                else:
                    sum_sq = oti.mul(diffs[0], diffs[0])
                    for d in range(1, D):
                        sum_sq = oti.sum(sum_sq, oti.mul(diffs[d], diffs[d]))
                grad[0] = _gc(oti.mul(-ln10 * ell ** 2, oti.mul(sum_sq, phi)))

        elif kernel == 'RQ':
            # phi = sf² * (1 + r²/(2α))^(-α),  r² = Σ_d (ell_d * diff_d)²
            # d phi/d log_ell_d = -ln10 * ell_d² * diff_d² * phi / base
            # d phi/d log_α     = ln10 * α * phi * [-log(base) + (1 - 1/base)]
            if kernel_type == 'anisotropic':
                ell      = 10.0 ** x0[:D]
                alpha_rq = 10.0 ** float(x0[D])
                alpha_idx = D
            else:
                ell_val  = 10.0 ** float(x0[0])
                ell      = np.full(D, ell_val)
                alpha_rq = np.exp(float(x0[1]))   # iso uses exp(x), not 10^x
                alpha_idx = 1

            # Recompute r² and base in OTI
            if hasattr(phi, 'fused_sqdist_sparse'):
                r2 = oti.zeros(phi.shape)
                ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
                r2.fused_sqdist_sparse(diffs, ell_sq)
            elif hasattr(phi, 'fused_sqdist'):
                r2 = oti.zeros(phi.shape)
                ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
                r2.fused_sqdist(diffs, ell_sq)
            else:
                r2 = oti.mul(ell[0], diffs[0])
                r2 = oti.mul(r2, r2)
                for d in range(1, D):
                    td = oti.mul(ell[d], diffs[d])
                    r2 = oti.sum(r2, oti.mul(td, td))
            base     = oti.sum(1.0, oti.mul(r2, 1.0 / (2.0 * alpha_rq)))
            inv_base = oti.pow(base, -1)
            phi_over_base = oti.mul(phi, inv_base)

            if kernel_type == 'anisotropic':
                if hasattr(phi, 'fused_scale_sq_mul_sparse'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul_sparse(diffs[d], phi_over_base, -ln10 * ell[d] ** 2, d)
                        grad[d] = _gc(dphi_buf)
                elif hasattr(phi, 'fused_scale_sq_mul'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul(diffs[d], phi_over_base, -ln10 * ell[d] ** 2)
                        grad[d] = _gc(dphi_buf)
                else:
                    for d in range(D):
                        d_sq   = oti.mul(diffs[d], diffs[d])
                        dphi_d = oti.mul(-ln10 * ell[d] ** 2, oti.mul(d_sq, phi_over_base))
                        grad[d] = _gc(dphi_d)
            else:
                if hasattr(phi, 'fused_sum_sq_sparse'):
                    sum_sq = oti.zeros(phi.shape)
                    sum_sq.fused_sum_sq_sparse(diffs)
                elif hasattr(phi, 'fused_sum_sq'):
                    sum_sq = oti.zeros(phi.shape)
                    sum_sq.fused_sum_sq(diffs)
                else:
                    sum_sq = oti.mul(diffs[0], diffs[0])
                    for d in range(1, D):
                        sum_sq = oti.sum(sum_sq, oti.mul(diffs[d], diffs[d]))
                grad[0] = _gc(oti.mul(-ln10 * ell[0] ** 2, oti.mul(sum_sq, phi_over_base)))

            # alpha gradient: phi * α_factor * [-log(base) + (1 - 1/base)]
            # aniso: alpha = 10^x  → d alpha/dx = ln10 * alpha
            # iso:   alpha = exp(x) → d alpha/dx = alpha
            log_base = oti.log(base)
            term = oti.sub(oti.sub(1.0, inv_base), log_base)
            alpha_factor = ln10 * alpha_rq if kernel_type == 'anisotropic' else alpha_rq
            grad[alpha_idx] = _gc(oti.mul(alpha_factor, oti.mul(phi, term)))

        elif kernel == 'SineExp':
            # phi = sf² * exp(-2 * Σ_d (ell_d * sin(π/p_d * diff_d))²)
            # d phi/d log_ell_d = -4*ln10 * ell_d² * sin_d² * phi
            # d phi/d log_p_d   =  4*ln10 * ell_d² * (π/p_d) * sin_d * cos_d * diff_d * phi
            if kernel_type == 'anisotropic':
                ell      = 10.0 ** x0[:D]
                p        = 10.0 ** x0[D:2 * D]
                pip      = np.pi / p          # π/p_d per dimension
                p_start  = D                  # index of first log_p in x0
            else:
                ell_val  = 10.0 ** float(x0[0])
                p_val    = 10.0 ** float(x0[1])
                pip_val  = np.pi / p_val
                ell      = np.full(D, ell_val)
                pip      = np.full(D, pip_val)
                p_start  = 1

            # Precompute sin and cos for each dimension
            sin_d = []
            cos_d = []
            for d in range(D):
                arg = oti.mul(pip[d], diffs[d])
                sin_d.append(oti.sin(arg))
                cos_d.append(oti.cos(arg))

            # Length-scale gradients: d phi/d log_ell_d = -4*ln10 * ell_d² * sin_d² * phi
            if kernel_type == 'anisotropic':
                if hasattr(phi, 'fused_scale_sq_mul'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul(sin_d[d], phi, -4.0 * ln10 * ell[d] ** 2)
                        grad[d] = _gc(dphi_buf)
                else:
                    for d in range(D):
                        sin_sq = oti.mul(sin_d[d], sin_d[d])
                        grad[d] = _gc(oti.mul(-4.0 * ln10 * ell[d] ** 2,
                                              oti.mul(sin_sq, phi)))
            else:
                if hasattr(phi, 'fused_sum_sq'):
                    sum_sin_sq = oti.zeros(phi.shape)
                    sum_sin_sq.fused_sum_sq(sin_d)
                else:
                    sum_sin_sq = oti.mul(sin_d[0], sin_d[0])
                    for d in range(1, D):
                        sum_sin_sq = oti.sum(sum_sin_sq, oti.mul(sin_d[d], sin_d[d]))
                grad[0] = _gc(oti.mul(-4.0 * ln10 * ell[0] ** 2,
                                      oti.mul(sum_sin_sq, phi)))

            # Period gradients
            if kernel_type == 'anisotropic':
                for d in range(D):
                    # d phi/d log_p_d = 4*ln10*ell_d²*(π/p_d)*sin_d*cos_d*diff_d * phi
                    sc_diff = oti.mul(sin_d[d], oti.mul(cos_d[d], diffs[d]))
                    scale   = 4.0 * ln10 * ell[d] ** 2 * pip[d]
                    grad[p_start + d] = _gc(oti.mul(scale, oti.mul(sc_diff, phi)))
            else:
                # d phi/d log_p = 4*ln10*ell²*(π/p) * Σ_d(sin_d*cos_d*diff_d) * phi
                sum_scd = oti.mul(sin_d[0], oti.mul(cos_d[0], diffs[0]))
                for d in range(1, D):
                    sum_scd = oti.sum(sum_scd,
                                      oti.mul(sin_d[d], oti.mul(cos_d[d], diffs[d])))
                scale = 4.0 * ln10 * ell[0] ** 2 * pip[0]
                grad[p_start] = _gc(oti.mul(scale, oti.mul(sum_scd, phi)))

        elif kernel == 'Matern':
            # phi = sf² * f(r),  r = sqrt(Σ_d (ell_d*(diff_d+ε))²)
            # d phi/d log_ell_d = sf² * f'(r) * ln10 * ell_d² * (diff_d+ε)² / r
            kf = self.model.kernel_factory

            # Build/cache the Matern derivative function
            if not hasattr(kf, '_matern_grad_prebuild'):
                kf._matern_grad_prebuild = matern_kernel_grad_builder(
                    kf.nu, oti_module=oti)

            if kernel_type == 'anisotropic':
                ell = 10.0 ** x0[:D]
            else:
                ell = np.full(D, 10.0 ** float(x0[0]))

            sigma_f_sq = (10.0 ** float(x0[-2])) ** 2
            _eps = 1e-10   # regularise r, not each diff (matches kernel_funcs.py)

            # Recompute r in OTI (matches matern_kernel_anisotropic/isotropic)
            if hasattr(phi, 'fused_sqdist_sparse'):
                r2 = oti.zeros(phi.shape)
                ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
                r2.fused_sqdist_sparse(diffs, ell_sq)
            elif hasattr(phi, 'fused_sqdist'):
                r2 = oti.zeros(phi.shape)
                ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
                r2.fused_sqdist(diffs, ell_sq)
            else:
                r2 = oti.mul(ell[0], diffs[0])
                r2 = oti.mul(r2, r2)
                for d in range(1, D):
                    td = oti.mul(ell[d], diffs[d])
                    r2 = oti.sum(r2, oti.mul(td, td))
            r_oti = oti.sqrt(oti.sum(r2, _eps ** 2))
            f_prime_r = kf._matern_grad_prebuild(r_oti)   # df/dr (OTI)
            inv_r     = oti.pow(r_oti, -1)

            # Precompute base = sigma_f² * f'(r) * 1/r for length-scale gradients
            # grad[d] = _gc(base * ln10 * ell_d² * diff_d²)
            base_matern = oti.mul(sigma_f_sq, oti.mul(f_prime_r, inv_r))
            if kernel_type == 'anisotropic':
                if hasattr(phi, 'fused_scale_sq_mul_sparse'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul_sparse(diffs[d], base_matern, ln10 * ell[d] ** 2, d)
                        grad[d] = _gc(dphi_buf)
                elif hasattr(phi, 'fused_scale_sq_mul'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul(diffs[d], base_matern, ln10 * ell[d] ** 2)
                        grad[d] = _gc(dphi_buf)
                else:
                    for d in range(D):
                        d_sq   = oti.mul(diffs[d], diffs[d])
                        dphi_d = oti.mul(ln10 * ell[d] ** 2, oti.mul(d_sq, base_matern))
                        grad[d] = _gc(dphi_d)
            else:
                ell_val = ell[0]
                if hasattr(phi, 'fused_sum_sq_sparse'):
                    sum_dsq = oti.zeros(phi.shape)
                    sum_dsq.fused_sum_sq_sparse(diffs)
                elif hasattr(phi, 'fused_sum_sq'):
                    sum_dsq = oti.zeros(phi.shape)
                    sum_dsq.fused_sum_sq(diffs)
                else:
                    sum_dsq = oti.mul(diffs[0], diffs[0])
                    for d in range(1, D):
                        sum_dsq = oti.sum(sum_dsq, oti.mul(diffs[d], diffs[d]))
                dphi_e = oti.mul(ln10 * ell_val ** 2, oti.mul(sum_dsq, base_matern))
                grad[0] = _gc(dphi_e)

        elif kernel == 'SI':
            # phi = sf² * Π_d (1 + ell_d * B(diff_d))
            # d phi/d log_ell_d = ln10 * ell_d * B(diff_d) / (1 + ell_d*B(diff_d)) * phi
            kf         = self.model.kernel_factory
            si_prebuild = kf.SI_kernel_prebuild

            if kernel_type == 'anisotropic':
                ell = 10.0 ** x0[:D]
            else:
                ell = np.full(D, 10.0 ** float(x0[0]))

            # Precompute SI values and factor terms for each dimension
            si_vals   = [si_prebuild(diffs[d]) for d in range(D)]
            term_vals = [oti.sum(1.0, oti.mul(ell[d], si_vals[d])) for d in range(D)]

            if kernel_type == 'anisotropic':
                for d in range(D):
                    phi_over_term = oti.div(phi, term_vals[d])
                    dphi_d = oti.mul(ln10 * ell[d],
                                     oti.mul(si_vals[d], phi_over_term))
                    grad[d] = _gc(dphi_d)
            else:
                ell_val = ell[0]
                # d phi/d log_ell = ln10 * ell * Σ_d [B(diff_d)/(1+ell*B(diff_d))] * phi
                acc = oti.mul(si_vals[0], oti.div(phi, term_vals[0]))
                for d in range(1, D):
                    acc = oti.sum(acc, oti.mul(si_vals[d],
                                               oti.div(phi, term_vals[d])))
                grad[0] = _gc(oti.mul(ln10 * ell_val, acc))

        return grad

    def _compute_grad_blockwise(self, x0, U, alpha_v, phi, n_bases, oti, diffs):
        """
        Compute the NLL gradient block-by-block using the Vecchia decomposition.

        Avoids forming the dense W = K^{-1} - αα^T.  Instead decomposes:
            grad[d] = 0.5 * Σ_j u_j^T dK_d u_j  -  0.5 * α^T dK_d α

        The trace term uses the sparse U columns within each block's
        neighbourhood (same blocks as build_U_from_phi).  The α term is
        a rank-1 quadratic form projected into phi-space once.
        """
        from math import comb

        ln10        = np.log(10.0)
        kernel      = self.model.kernel
        kernel_type = self.model.kernel_type
        D           = len(diffs)
        sigma_n_sq  = (10.0 ** x0[-1]) ** 2

        grad = np.zeros(len(x0))
        deriv_order = 2 * self.model.n_order
        plan = self._kernel_plan
        P_full = self.model.mmd_P_full
        N_total = len(P_full)
        n_func = phi.shape[0]  # number of physical training points

        # ── noise gradient: trace(W) = ||U||_F² - ||α||² ────────────
        U_frob_sq = np.sum(U * U)
        alpha_sq = np.dot(alpha_v, alpha_v)
        trace_W = U_frob_sq - alpha_sq
        grad[-1] = ln10 * sigma_n_sq * trace_W

        # ── project αα^T into phi-space (rank-1, cheap) ─────────────
        ndir = comb(n_bases + deriv_order, deriv_order)
        proj_shape = (ndir, n_func, n_func)
        if self._W_proj_buf is None or self._W_proj_shape != proj_shape:
            self._W_proj_buf = np.empty(proj_shape)
            self._W_proj_shape = proj_shape
        alpha_proj = self._W_proj_buf

        row_off = plan.get('row_offsets_abs', plan['row_offsets'] + n_func)
        col_off = plan.get('col_offsets_abs', plan['col_offsets'] + n_func)

        utils._project_alpha_to_phi_space(
            alpha_v, alpha_proj, n_func, n_func,
            plan['fd_flat_indices'], plan['df_flat_indices'],
            plan['dd_flat_indices'],
            plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
            plan['signs'], plan['n_deriv_types'], row_off, col_off,
        )

        # ── pre-compute block metadata (packed for numba) ────────────
        S = self.model.sparse_S_full_arr
        block_size = self.model.n_bases + 1
        k_type, k_phys, deriv_lookup, sign_lookup = self._k_index_map

        # Pack block data into flat arrays for the numba trace kernel
        block_nb_type_list = []
        block_nb_phys_list = []
        block_m_list = []
        block_n_cols_list = []
        V_parts = []
        for start in range(0, N_total, block_size):
            end = min(start + block_size, N_total)
            nb_union = S[end - 1] if isinstance(S[end - 1], np.ndarray) else np.asarray(S[end - 1])
            m = len(nb_union)
            orig = P_full[nb_union]
            block_nb_type_list.append(k_type[orig])
            block_nb_phys_list.append(k_phys[orig])
            block_m_list.append(m)
            n_cols = end - start
            block_n_cols_list.append(n_cols)
            # Store V column-major: V[:,0], V[:,1], ... (length m*n_cols)
            V = U[nb_union, start:end]  # (m, n_cols)
            V_parts.append(V.ravel(order='F'))

        n_blocks = len(block_m_list)
        max_m = max(block_m_list)
        # Pad nb_type/nb_phys to uniform length for numba typed array
        block_nb_type = np.zeros((n_blocks, max_m), dtype=np.int64)
        block_nb_phys = np.zeros((n_blocks, max_m), dtype=np.int64)
        for i in range(n_blocks):
            m = block_m_list[i]
            block_nb_type[i, :m] = block_nb_type_list[i]
            block_nb_phys[i, :m] = block_nb_phys_list[i]
        block_m = np.array(block_m_list, dtype=np.int64)
        block_n_cols = np.array(block_n_cols_list, dtype=np.int64)
        block_V_flat = np.concatenate(V_parts)
        block_V_offsets = np.zeros(n_blocks, dtype=np.int64)
        for i in range(1, n_blocks):
            block_V_offsets[i] = block_V_offsets[i-1] + block_m_list[i-1] * block_n_cols_list[i-1]

        # ── vdot factors for alpha term ──────────────────────────────
        _vdot_factors = self._get_deriv_factors(n_bases, deriv_order)

        # ── helper: compute grad contribution for one dphi ───────────
        def _gc_block(dphi):
            # Expand dphi to dphi_exp_3d
            dphi_exp = self._expand_derivs(dphi, n_bases, deriv_order)
            dphi_3d = dphi_exp.reshape(ndir, n_func, n_func)

            # Trace term: Σ_blocks trace(V^T dK_sub V) — single numba call
            trace_term = _trace_term_all_blocks(
                dphi_3d, deriv_lookup, sign_lookup,
                block_nb_type, block_nb_phys, block_m,
                block_V_flat, block_V_offsets, block_n_cols,
                n_blocks)

            # Alpha term: α^T dK α  (via vdot in phi-space)
            alpha_term = dphi.vdot_expand_fast(_vdot_factors, alpha_proj)

            return 0.5 * (trace_term - alpha_term)

        # ── signal variance ──────────────────────────────────────────
        grad[-2] = _gc_block(oti.mul(2.0 * ln10, phi))

        # ── kernel-specific hyperparameter gradients ─────────────────

        if kernel == 'SE':
            if kernel_type == 'anisotropic':
                ell = 10.0 ** x0[:D]
                if hasattr(phi, 'fused_scale_sq_mul_sparse'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul_sparse(diffs[d], phi, -ln10 * ell[d] ** 2, d)
                        grad[d] = _gc_block(dphi_buf)
                elif hasattr(phi, 'fused_scale_sq_mul'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul(diffs[d], phi, -ln10 * ell[d] ** 2)
                        grad[d] = _gc_block(dphi_buf)
                else:
                    for d in range(D):
                        d_sq   = oti.mul(diffs[d], diffs[d])
                        dphi_d = oti.mul(-ln10 * ell[d] ** 2, oti.mul(d_sq, phi))
                        grad[d] = _gc_block(dphi_d)
            else:
                ell = 10.0 ** float(x0[0])
                if hasattr(phi, 'fused_sum_sq_sparse'):
                    sum_sq = oti.zeros(phi.shape)
                    sum_sq.fused_sum_sq_sparse(diffs)
                elif hasattr(phi, 'fused_sum_sq'):
                    sum_sq = oti.zeros(phi.shape)
                    sum_sq.fused_sum_sq(diffs)
                else:
                    sum_sq = oti.mul(diffs[0], diffs[0])
                    for d in range(1, D):
                        sum_sq = oti.sum(sum_sq, oti.mul(diffs[d], diffs[d]))
                grad[0] = _gc_block(oti.mul(-ln10 * ell ** 2, oti.mul(sum_sq, phi)))

        elif kernel == 'RQ':
            if kernel_type == 'anisotropic':
                ell      = 10.0 ** x0[:D]
                alpha_rq = 10.0 ** float(x0[D])
                alpha_idx = D
            else:
                ell_val  = 10.0 ** float(x0[0])
                ell      = np.full(D, ell_val)
                alpha_rq = np.exp(float(x0[1]))
                alpha_idx = 1

            if hasattr(phi, 'fused_sqdist_sparse'):
                r2 = oti.zeros(phi.shape)
                ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
                r2.fused_sqdist_sparse(diffs, ell_sq)
            elif hasattr(phi, 'fused_sqdist'):
                r2 = oti.zeros(phi.shape)
                ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
                r2.fused_sqdist(diffs, ell_sq)
            else:
                r2 = oti.mul(ell[0], diffs[0])
                r2 = oti.mul(r2, r2)
                for d in range(1, D):
                    td = oti.mul(ell[d], diffs[d])
                    r2 = oti.sum(r2, oti.mul(td, td))
            base     = oti.sum(1.0, oti.mul(r2, 1.0 / (2.0 * alpha_rq)))
            inv_base = oti.pow(base, -1)
            phi_over_base = oti.mul(phi, inv_base)

            if kernel_type == 'anisotropic':
                if hasattr(phi, 'fused_scale_sq_mul_sparse'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul_sparse(diffs[d], phi_over_base, -ln10 * ell[d] ** 2, d)
                        grad[d] = _gc_block(dphi_buf)
                elif hasattr(phi, 'fused_scale_sq_mul'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul(diffs[d], phi_over_base, -ln10 * ell[d] ** 2)
                        grad[d] = _gc_block(dphi_buf)
                else:
                    for d in range(D):
                        d_sq   = oti.mul(diffs[d], diffs[d])
                        dphi_d = oti.mul(-ln10 * ell[d] ** 2, oti.mul(d_sq, phi_over_base))
                        grad[d] = _gc_block(dphi_d)
            else:
                if hasattr(phi, 'fused_sum_sq_sparse'):
                    sum_sq = oti.zeros(phi.shape)
                    sum_sq.fused_sum_sq_sparse(diffs)
                elif hasattr(phi, 'fused_sum_sq'):
                    sum_sq = oti.zeros(phi.shape)
                    sum_sq.fused_sum_sq(diffs)
                else:
                    sum_sq = oti.mul(diffs[0], diffs[0])
                    for d in range(1, D):
                        sum_sq = oti.sum(sum_sq, oti.mul(diffs[d], diffs[d]))
                grad[0] = _gc_block(oti.mul(-ln10 * ell[0] ** 2, oti.mul(sum_sq, phi_over_base)))

            log_base = oti.log(base)
            term = oti.sub(oti.sub(1.0, inv_base), log_base)
            alpha_factor = ln10 * alpha_rq if kernel_type == 'anisotropic' else alpha_rq
            grad[alpha_idx] = _gc_block(oti.mul(alpha_factor, oti.mul(phi, term)))

        elif kernel == 'SineExp':
            if kernel_type == 'anisotropic':
                ell      = 10.0 ** x0[:D]
                p        = 10.0 ** x0[D:2 * D]
                pip      = np.pi / p
                p_start  = D
            else:
                ell_val  = 10.0 ** float(x0[0])
                p_val    = 10.0 ** float(x0[1])
                pip_val  = np.pi / p_val
                ell      = np.full(D, ell_val)
                pip      = np.full(D, pip_val)
                p_start  = 1

            sin_d = []
            cos_d = []
            for d in range(D):
                arg = oti.mul(pip[d], diffs[d])
                sin_d.append(oti.sin(arg))
                cos_d.append(oti.cos(arg))

            if kernel_type == 'anisotropic':
                if hasattr(phi, 'fused_scale_sq_mul'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul(sin_d[d], phi, -4.0 * ln10 * ell[d] ** 2)
                        grad[d] = _gc_block(dphi_buf)
                else:
                    for d in range(D):
                        sin_sq = oti.mul(sin_d[d], sin_d[d])
                        grad[d] = _gc_block(oti.mul(-4.0 * ln10 * ell[d] ** 2,
                                                     oti.mul(sin_sq, phi)))
            else:
                if hasattr(phi, 'fused_sum_sq'):
                    sum_sin_sq = oti.zeros(phi.shape)
                    sum_sin_sq.fused_sum_sq(sin_d)
                else:
                    sum_sin_sq = oti.mul(sin_d[0], sin_d[0])
                    for d in range(1, D):
                        sum_sin_sq = oti.sum(sum_sin_sq, oti.mul(sin_d[d], sin_d[d]))
                grad[0] = _gc_block(oti.mul(-4.0 * ln10 * ell[0] ** 2,
                                             oti.mul(sum_sin_sq, phi)))

            if kernel_type == 'anisotropic':
                for d in range(D):
                    sc_diff = oti.mul(sin_d[d], oti.mul(cos_d[d], diffs[d]))
                    scale   = 4.0 * ln10 * ell[d] ** 2 * pip[d]
                    grad[p_start + d] = _gc_block(oti.mul(scale, oti.mul(sc_diff, phi)))
            else:
                sum_scd = oti.mul(sin_d[0], oti.mul(cos_d[0], diffs[0]))
                for d in range(1, D):
                    sum_scd = oti.sum(sum_scd,
                                       oti.mul(sin_d[d], oti.mul(cos_d[d], diffs[d])))
                scale = 4.0 * ln10 * ell[0] ** 2 * pip[0]
                grad[p_start] = _gc_block(oti.mul(scale, oti.mul(sum_scd, phi)))

        elif kernel == 'Matern':
            kf = self.model.kernel_factory
            if not hasattr(kf, '_matern_grad_prebuild'):
                from jetgp.kernel_funcs.kernel_funcs import matern_kernel_grad_builder
                kf._matern_grad_prebuild = matern_kernel_grad_builder(
                    kf.nu, oti_module=oti)

            if kernel_type == 'anisotropic':
                ell = 10.0 ** x0[:D]
            else:
                ell = np.full(D, 10.0 ** float(x0[0]))

            sigma_f_sq = (10.0 ** float(x0[-2])) ** 2
            _eps = 1e-10

            if hasattr(phi, 'fused_sqdist_sparse'):
                r2 = oti.zeros(phi.shape)
                ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
                r2.fused_sqdist_sparse(diffs, ell_sq)
            elif hasattr(phi, 'fused_sqdist'):
                r2 = oti.zeros(phi.shape)
                ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
                r2.fused_sqdist(diffs, ell_sq)
            else:
                r2 = oti.mul(ell[0], diffs[0])
                r2 = oti.mul(r2, r2)
                for d in range(1, D):
                    td = oti.mul(ell[d], diffs[d])
                    r2 = oti.sum(r2, oti.mul(td, td))
            r_oti = oti.sqrt(oti.sum(r2, _eps ** 2))
            f_prime_r = kf._matern_grad_prebuild(r_oti)
            inv_r     = oti.pow(r_oti, -1)

            base_matern = oti.mul(sigma_f_sq, oti.mul(f_prime_r, inv_r))
            if kernel_type == 'anisotropic':
                if hasattr(phi, 'fused_scale_sq_mul_sparse'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul_sparse(diffs[d], base_matern, ln10 * ell[d] ** 2, d)
                        grad[d] = _gc_block(dphi_buf)
                elif hasattr(phi, 'fused_scale_sq_mul'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul(diffs[d], base_matern, ln10 * ell[d] ** 2)
                        grad[d] = _gc_block(dphi_buf)
                else:
                    for d in range(D):
                        d_sq   = oti.mul(diffs[d], diffs[d])
                        dphi_d = oti.mul(ln10 * ell[d] ** 2, oti.mul(d_sq, base_matern))
                        grad[d] = _gc_block(dphi_d)
            else:
                ell_val = ell[0]
                if hasattr(phi, 'fused_sum_sq_sparse'):
                    sum_dsq = oti.zeros(phi.shape)
                    sum_dsq.fused_sum_sq_sparse(diffs)
                elif hasattr(phi, 'fused_sum_sq'):
                    sum_dsq = oti.zeros(phi.shape)
                    sum_dsq.fused_sum_sq(diffs)
                else:
                    sum_dsq = oti.mul(diffs[0], diffs[0])
                    for d in range(1, D):
                        sum_dsq = oti.sum(sum_dsq, oti.mul(diffs[d], diffs[d]))
                dphi_e = oti.mul(ln10 * ell_val ** 2, oti.mul(sum_dsq, base_matern))
                grad[0] = _gc_block(dphi_e)

        elif kernel == 'SI':
            kf         = self.model.kernel_factory
            si_prebuild = kf.SI_kernel_prebuild

            if kernel_type == 'anisotropic':
                ell = 10.0 ** x0[:D]
            else:
                ell = np.full(D, 10.0 ** float(x0[0]))

            si_vals   = [si_prebuild(diffs[d]) for d in range(D)]
            term_vals = [oti.sum(1.0, oti.mul(ell[d], si_vals[d])) for d in range(D)]

            if kernel_type == 'anisotropic':
                for d in range(D):
                    phi_over_term = oti.div(phi, term_vals[d])
                    dphi_d = oti.mul(ln10 * ell[d],
                                     oti.mul(si_vals[d], phi_over_term))
                    grad[d] = _gc_block(dphi_d)
            else:
                ell_val = ell[0]
                acc = oti.mul(si_vals[0], oti.div(phi, term_vals[0]))
                for d in range(1, D):
                    acc = oti.sum(acc, oti.mul(si_vals[d],
                                               oti.div(phi, term_vals[d])))
                grad[0] = _gc_block(oti.mul(ln10 * ell_val, acc))

        return grad
        
    def _build_K_and_phi(self, x0):
        """
        Shared helper: build K (with noise), phi, n_bases, oti, diffs.

        Returns (K, phi, n_bases, oti, diffs).
        """
        diffs = self.model.differences_by_dim
        oti = self.model.kernel_factory.oti
        sigma_n_sq = (10.0 ** x0[-1]) ** 2

        phi = self.model.kernel_func(diffs, x0[:-1])
        if self.model.n_order == 0:
            n_bases = 0
            phi_exp = phi.real[np.newaxis, :, :]
        else:
            n_bases = phi.get_active_bases()[-1]
            deriv_order = 2 * self.model.n_order
            phi_exp = self._expand_derivs(phi, n_bases, deriv_order)

        self._ensure_kernel_plan(n_bases)
        if self._kernel_plan is not None:
            base_shape = phi.shape
            self._ensure_kernel_bufs(base_shape[0])
            phi_3d = phi_exp.reshape(phi_exp.shape[0], base_shape[0], base_shape[1])
            K = utils.rbf_kernel_fast(phi_3d, self._kernel_plan, out=self._K_buf)
        else:
            K = utils.rbf_kernel(
                phi, phi_exp, self.model.n_order, n_bases,
                self.model.flattened_der_indices, self.model.powers,
                index=self.model.derivative_locations,
            )
        K.flat[::K.shape[0] + 1] += sigma_n_sq
        K += self.model.sigma_data ** 2
        return K, phi, n_bases, oti, diffs

    @profile
    def _sparse_nlml_direct(self, x0):
        """
        Compute sparse NLML directly from phi_exp_3d, skipping full K
        construction and permutation.

        Returns (alpha_v, U, nll, phi, n_bases, oti, diffs) where the
        last four are needed by _compute_grad.
        """
        from jetgp.full_degp_sparse.sparse_cholesky import (
            build_U_from_phi, build_U_from_phi_flat,
            build_U_supernodes_from_phi,
        )

        diffs = self.model.differences_by_dim
        oti = self.model.kernel_factory.oti
        sigma_n_sq = (10.0 ** x0[-1]) ** 2

        phi = self.model.kernel_func(diffs, x0[:-1])
        n_bases = phi.get_active_bases()[-1]
        deriv_order = 2 * self.model.n_order
        phi_exp = self._expand_derivs(phi, n_bases, deriv_order)

        self._ensure_kernel_plan(n_bases)
        base_shape = phi.shape
        phi_3d = phi_exp.reshape(phi_exp.shape[0], base_shape[0], base_shape[1])

        # Build index maps (once)
        self._ensure_phi_index_maps(base_shape[0])
        k_type, k_phys, deriv_lookup, sign_lookup = self._k_index_map

        P_full = self.model.mmd_P_full
        N_total = len(P_full)

        if self.model.use_supernodes and self.model.sparse_supernodes_full is not None:
            U, _ = build_U_supernodes_from_phi(
                phi_3d, self.model.sparse_supernodes_full, N_total,
                sigma_n_sq,
            )
        else:
            if self._U_buf is None or self._U_buf.shape[0] != N_total:
                self._U_buf = np.zeros((N_total, N_total), order='F')

            U = build_U_from_phi_flat(
                phi_3d, self._block_phi_maps, N_total,
                sigma_n_sq, out=self._U_buf,
            )

        y_ord = self.model.y_train[P_full]
        nll = nlml_from_U(U, y_ord)

        alpha_ord = alpha_from_U(U, y_ord)
        alpha_v = np.empty_like(alpha_ord)
        alpha_v[P_full] = alpha_ord

        return alpha_v, U, nll, phi, n_bases, oti, diffs

    def _dense_nll_and_W(self, x0):
        """
        Dense Cholesky path: build full K, factor once, compute NLL and W.

        Used as a fallback when the sparsity pattern is too full for the
        block-wise sparse path to be efficient.

        Returns (W, alpha_v, nll, phi, n_bases, oti, diffs).
        """
        K, phi, n_bases, oti, diffs = self._build_K_and_phi(x0)
        N = K.shape[0]

        L, low = cho_factor(K, lower=True)
        alpha_v = cho_solve((L, low), self.model.y_train)

        nll = (0.5 * np.dot(self.model.y_train, alpha_v)
               + np.sum(np.log(np.diag(L)))
               + 0.5 * N * np.log(2 * np.pi))

        K_inv = cho_solve((L, low), np.eye(N))
        W = K_inv - np.outer(alpha_v, alpha_v)

        # Cache dense factors for prediction
        self.model._cached_L = L
        self.model._cached_low = low
        self.model._cached_alpha = alpha_v
        self.model._cached_U = None
        self.model._cached_P = None
        self.model._cached_params = x0.copy()

        return W, alpha_v, nll, phi, n_bases, oti, diffs
    def _sparse_U_alpha_nll(self, K):
        """
        Build sparse U, compute alpha and NLML.  Does NOT form K^{-1}.

        Returns (alpha_v, U, nll) all in original index space.
        """
        P_full = self.model.mmd_P_full
        N_total = len(P_full)

        if self._P_ix is None:
            self._P_ix = np.ix_(P_full, P_full)
        K_ord = K[self._P_ix]
        y_ord = self.model.y_train[P_full]

        if self.model.use_supernodes and self.model.sparse_supernodes_full is not None:
            U, _ = build_U_supernodes(K_ord, self.model.sparse_supernodes_full, N_total)
        else:
            if self._U_buf is None or self._U_buf.shape[0] != N_total:
                self._U_buf = np.zeros((N_total, N_total), order='F')
            U = build_U(K_ord, self.model.sparse_S_full_arr, N_total,
                        block_size=self.model.n_bases + 1, out=self._U_buf)

        nll = nlml_from_U(U, y_ord)

        # alpha in original space
        alpha_ord = alpha_from_U(U, y_ord)
        alpha_v = np.empty_like(alpha_ord)
        alpha_v[P_full] = alpha_ord

        return alpha_v, U, nll


    def _W_from_U(self, U, alpha_v):
        """
        Compute W = K^{-1} - αα^T from a pre-built sparse U.

        U is in MMD order, alpha_v is in original index space.
        Returns W in original index space.
        """
        P_full = self.model.mmd_P_full
        N_total = len(P_full)

        # K^{-1} = U @ U.T — exploit symmetry of result via dsyrk
        # (computes upper triangle only, ~2x fewer FLOPs than dgemm)
        if self._K_inv_buf is None or self._K_inv_buf.shape[0] != N_total:
            self._K_inv_buf = np.empty((N_total, N_total), order='F')
        K_inv_ord = blas.dsyrk(1.0, U, lower=1,
                               c=self._K_inv_buf, overwrite_c=1)
        # No symmetrisation needed — _permute_and_subtract_outer reads
        # the lower triangle directly (contiguous in Fortran-order).

        W = np.empty((N_total, N_total))
        _permute_and_subtract_outer(K_inv_ord, alpha_v, P_full, W)
        return W

    def _sparse_W_and_alpha(self, K):
        """
        Compute W = K^{-1} - αα^T and alpha using the sparse U factor.

        K is in the ORIGINAL index space (size N_total × N_total).
        Returns (W, alpha_v, U, nll) all in original index space.
        """
        alpha_v, U, nll = self._sparse_U_alpha_nll(K)

        P_full = self.model.mmd_P_full
        N_total = len(P_full)

        # K^{-1} in original space: K_inv_ord = U @ U^T, then permute back.
        K_inv_ord = U @ U.T
        K_inv = np.empty_like(K_inv_ord)
        K_inv[np.ix_(P_full, P_full)] = K_inv_ord

        W = K_inv - np.outer(alpha_v, alpha_v)
        return W, alpha_v, U, nll

    def nll_grad(self, x0):
        """Analytic gradient of the NLL."""
        try:
            if self.model._use_dense_factor:
                W, alpha_v, nll, phi, n_bases, oti, diffs = self._dense_nll_and_W(x0)
            else:
                alpha_v, U, nll, phi, n_bases, oti, diffs = self._sparse_nlml_direct(x0)
                W = self._W_from_U(U, alpha_v)
        except Exception:
            return np.zeros(len(x0))
        return self._compute_grad(x0, W, phi, n_bases, oti, diffs)

    def nll_and_grad(self, x0):
        """
        Compute NLL and its gradient in a single pass.

        Routes to either the dense Cholesky path or the sparse U path
        based on the sparsity pattern fill fraction.

        Returns
        -------
        nll : float
        grad : ndarray
        """
        try:
            if self.model._use_dense_factor:
                W, alpha_v, nll, phi, n_bases, oti, diffs = self._dense_nll_and_W(x0)
                grad = self._compute_grad(x0, W, phi, n_bases, oti, diffs)
            elif self.model.n_order > 0:
                alpha_v, U, nll, phi, n_bases, oti, diffs = self._sparse_nlml_direct(x0)

                # Cache for fast prediction (reused by degp.predict)
                self.model._cached_U = U
                self.model._cached_P = self.model.mmd_P_full
                self.model._cached_alpha = alpha_v
                self.model._cached_L = None
                self.model._cached_low = None
                self.model._cached_params = x0.copy()

                W = self._W_from_U(U, alpha_v)
                grad = self._compute_grad(x0, W, phi, n_bases, oti, diffs)
            else:
                K, phi, n_bases, oti, diffs = self._build_K_and_phi(x0)
                alpha_v, U, nll = self._sparse_U_alpha_nll(K)

                self.model._cached_U = U
                self.model._cached_P = self.model.mmd_P_full
                self.model._cached_alpha = alpha_v
                self.model._cached_L = None
                self.model._cached_low = None
                self.model._cached_params = x0.copy()

                W = self._W_from_U(U, alpha_v)
                grad = self._compute_grad(x0, W, phi, n_bases, oti, diffs)
        except Exception:
            return 1e6, np.zeros(len(x0))

        if nll > 1e6:
            return 1e6, np.zeros(len(x0))
        return float(nll), grad

    def optimize_hyperparameters(self,
    optimizer="pso",
    **kwargs):
        """
        Optimize the DEGP model hyperparameters using Particle Swarm Optimization (PSO).

        Parameters:
        ----------
        n_restart_optimizer : int, default=20
            Maximum number of iterations for PSO.
        swarm_size : int, default=20
            Number of particles in the swarm.
        verbose : bool, default=True
            Controls verbosity of PSO output.

        Returns:
        -------
        best_x : ndarray
            The optimal set of hyperparameters found.
        """

        if isinstance(optimizer, str):
            if optimizer not in OPTIMIZERS:
                raise ValueError(
                    f"Unknown optimizer '{optimizer}'. Available: {list(OPTIMIZERS.keys())}"
                )
            optimizer_fn = OPTIMIZERS[optimizer]
        else:
            optimizer_fn = optimizer  # allow passing a callable directly

        bounds = self.model.bounds
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]

        # Inject nll_and_grad (single Cholesky per step) for all gradient-aware optimizers.
        if optimizer in ('lbfgs', 'jade', 'pso') and 'func_and_grad' not in kwargs and 'grad_func' not in kwargs:
            kwargs['func_and_grad'] = self.nll_and_grad

        best_x, best_val = optimizer_fn(self.nll_wrapper, lb, ub, **kwargs)

        self.model.opt_x0 = best_x
        self.model.opt_nll = best_val


        return best_x
