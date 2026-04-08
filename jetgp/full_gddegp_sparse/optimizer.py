import numpy as np
import numba
from scipy.linalg import cho_solve, cho_factor
from jetgp.full_gddegp_sparse import gddegp_utils as utils
from jetgp.full_gddegp_sparse.sparse_cholesky import (
    build_U, build_U_supernodes, nlml_from_U, alpha_from_U
)
from line_profiler import profile
import jetgp.utils as gen_utils
from jetgp.hyperparameter_optimizers import OPTIMIZERS
from jetgp.utils import matern_kernel_grad_builder


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
        Sign multiplier for each derivative type (all 1.0 for GDDEGP).
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

    # Sign lookup (all 1.0 for GDDEGP — even/odd bases handle signs internally)
    sign_lookup = np.ones(n_types, dtype=np.float64)

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


class Optimizer:
    """
    Optimizer class to perform hyperparameter tuning for sparse GDDEGP models
    by minimizing the negative log marginal likelihood (NLL).

    Parameters
    ----------
    model : object
        An instance of a model (e.g., gddegp) containing the necessary training data
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
        """Enumerate derivative factors in struct memory order for a given order."""
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
            None,  # GDDEGP uses even/odd bases, not powers
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
            if 'row_offsets_abs' not in self._kernel_plan:
                self._kernel_plan['row_offsets_abs'] = self._kernel_plan['row_offsets'] + n_rows_func
                self._kernel_plan['col_offsets_abs'] = self._kernel_plan['col_offsets'] + n_rows_func

    def _build_K(self, phi_exp, phi, n_bases):
        """Build kernel matrix using fast path if available."""
        self._ensure_kernel_plan(n_bases)
        if self._kernel_plan is not None:
            base_shape = phi.shape
            self._ensure_kernel_bufs(base_shape[0])
            phi_3d = phi_exp.reshape(phi_exp.shape[0], base_shape[0], base_shape[1])
            return utils.rbf_kernel_fast(phi_3d, self._kernel_plan, out=self._K_buf)
        return utils.rbf_kernel(
            phi, phi_exp, self.model.n_order, n_bases,
            self.model.flattened_der_indices,
            index=self.model.derivative_locations,
        )

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

    @profile
    def negative_log_marginal_likelihood(self, x0):
        """
        Compute the negative log marginal likelihood (NLL) via sparse U.

        NLL = 0.5 * ||U^T y||^2 - sum(log|diag(U)|) + 0.5 * N * log(2pi)

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
            # Use direct phi path (skip full K construction) when possible
            use_direct = (
                self._kernel_plan is not None
                and not self.model.use_supernodes
                and self.model.n_order > 0
            )
            if use_direct:
                alpha, U, nlml = self._sparse_nlml_direct(x0)
            else:
                K, _, _, _, _ = self._build_K_and_phi(x0)
                alpha, U, nlml = self._sparse_U_alpha_nll(K)

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
        Compute the NLL gradient given pre-factorised W = K^{-1} - alpha alpha^T.
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

        # Precompute W projected into phi_exp space
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

            # GDDEGP _project_W_to_phi_space does NOT take signs
            utils._project_W_to_phi_space(
                W, W_proj, base_shape[0], base_shape[1],
                plan['fd_flat_indices'], plan['df_flat_indices'],
                plan['dd_flat_indices'],
                plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
                plan['n_deriv_types'], row_off, col_off,
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
                    self.model.flattened_der_indices,
                    index=self.model.derivative_locations,
                )
                return 0.5 * np.vdot(W, dK)

        # signal variance
        grad[-2] = _gc(oti.mul(2.0 * ln10, phi))
        # noise variance
        grad[-1] = ln10 * sigma_n_sq * np.trace(W)

        if kernel == 'SE':
            if kernel_type == 'anisotropic':
                ell = 10.0 ** x0[:D]
                if hasattr(phi, 'fused_scale_sq_mul'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul(diffs[d], phi, -ln10 * ell[d] ** 2)
                        grad[d] = _gc(dphi_buf)
                else:
                    for d in range(D):
                        d_sq   = oti.mul(diffs[d], diffs[d])
                        dphi_d = oti.mul(-ln10 * ell[d] ** 2, oti.mul(d_sq, phi))
                        grad[d] = _gc(dphi_d)
            else:
                ell    = 10.0 ** float(x0[0])
                if hasattr(phi, 'fused_sum_sq'):
                    sum_sq = oti.zeros(phi.shape)
                    sum_sq.fused_sum_sq(diffs)
                else:
                    sum_sq = oti.mul(diffs[0], diffs[0])
                    for d in range(1, D):
                        sum_sq = oti.sum(sum_sq, oti.mul(diffs[d], diffs[d]))
                grad[0] = _gc(oti.mul(-ln10 * ell ** 2, oti.mul(sum_sq, phi)))

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

            if hasattr(phi, 'fused_sqdist'):
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
                if hasattr(phi, 'fused_scale_sq_mul'):
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
                if hasattr(phi, 'fused_sum_sq'):
                    sum_sq = oti.zeros(phi.shape)
                    sum_sq.fused_sum_sq(diffs)
                else:
                    sum_sq = oti.mul(diffs[0], diffs[0])
                    for d in range(1, D):
                        sum_sq = oti.sum(sum_sq, oti.mul(diffs[d], diffs[d]))
                grad[0] = _gc(oti.mul(-ln10 * ell[0] ** 2, oti.mul(sum_sq, phi_over_base)))

            log_base = oti.log(base)
            term = oti.sub(oti.sub(1.0, inv_base), log_base)
            alpha_factor = ln10 * alpha_rq if kernel_type == 'anisotropic' else alpha_rq
            grad[alpha_idx] = _gc(oti.mul(alpha_factor, oti.mul(phi, term)))

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

            if kernel_type == 'anisotropic':
                for d in range(D):
                    sc_diff = oti.mul(sin_d[d], oti.mul(cos_d[d], diffs[d]))
                    scale   = 4.0 * ln10 * ell[d] ** 2 * pip[d]
                    grad[p_start + d] = _gc(oti.mul(scale, oti.mul(sc_diff, phi)))
            else:
                sum_scd = oti.mul(sin_d[0], oti.mul(cos_d[0], diffs[0]))
                for d in range(1, D):
                    sum_scd = oti.sum(sum_scd,
                                      oti.mul(sin_d[d], oti.mul(cos_d[d], diffs[d])))
                scale = 4.0 * ln10 * ell[0] ** 2 * pip[0]
                grad[p_start] = _gc(oti.mul(scale, oti.mul(sum_scd, phi)))

        elif kernel == 'Matern':
            kf = self.model.kernel_factory
            if not hasattr(kf, '_matern_grad_prebuild'):
                kf._matern_grad_prebuild = matern_kernel_grad_builder(
                    getattr(kf, "nu", 1.5), oti_module=oti)

            if kernel_type == 'anisotropic':
                ell = 10.0 ** x0[:D]
            else:
                ell = np.full(D, 10.0 ** float(x0[0]))

            sigma_f_sq = (10.0 ** float(x0[-2])) ** 2
            _eps = 1e-10

            if hasattr(phi, 'fused_sqdist'):
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
                if hasattr(phi, 'fused_scale_sq_mul'):
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
                if hasattr(phi, 'fused_sum_sq'):
                    sum_dsq = oti.zeros(phi.shape)
                    sum_dsq.fused_sum_sq(diffs)
                else:
                    sum_dsq = oti.mul(diffs[0], diffs[0])
                    for d in range(1, D):
                        sum_dsq = oti.sum(sum_dsq, oti.mul(diffs[d], diffs[d]))
                dphi_e = oti.mul(ln10 * ell_val ** 2, oti.mul(sum_dsq, base_matern))
                grad[0] = _gc(dphi_e)

        return grad

    @profile
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
            active = phi.get_active_bases()
            n_bases = active[-1] if active else self.model.n_bases
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
                self.model.flattened_der_indices,
                index=self.model.derivative_locations,
            )
        K.flat[::K.shape[0] + 1] += sigma_n_sq
        K += self.model.sigma_data ** 2
        return K, phi, n_bases, oti, diffs

    def _sparse_nlml_direct(self, x0):
        """
        Compute sparse NLML directly from phi_exp_3d, skipping full K
        construction and permutation.

        Returns (alpha_v, U, nll) in original index space.
        """
        from jetgp.full_gddegp_sparse.sparse_cholesky import build_U_from_phi

        diffs = self.model.differences_by_dim
        sigma_n_sq = (10.0 ** x0[-1]) ** 2

        phi = self.model.kernel_func(diffs, x0[:-1])
        active = phi.get_active_bases()
        n_bases = active[-1] if active else self.model.n_bases
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

        if self._U_buf is None or self._U_buf.shape[0] != N_total:
            self._U_buf = np.zeros((N_total, N_total))

        U = build_U_from_phi(
            phi_3d, self.model.sparse_S_full_arr, N_total,
            block_size=self.model.n_bases + 1,
            k_type=k_type, k_phys=k_phys,
            deriv_lookup=deriv_lookup, sign_lookup=sign_lookup,
            inv_P=self._inv_P,
            sigma_n_sq=sigma_n_sq,
            sigma_data_diag=self._sigma_data_diag_mmd,
            out=self._U_buf,
        )

        y_ord = self.model.y_train[P_full]
        nll = nlml_from_U(U, y_ord)

        alpha_ord = alpha_from_U(U, y_ord)
        alpha_v = np.empty_like(alpha_ord)
        alpha_v[P_full] = alpha_ord

        return alpha_v, U, nll

    @profile
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
                self._U_buf = np.zeros((N_total, N_total))
            U = build_U(K_ord, self.model.sparse_S_full_arr, N_total,
                        block_size=self.model.n_bases + 1, out=self._U_buf)

        nll = nlml_from_U(U, y_ord)

        # alpha in original space
        alpha_ord = alpha_from_U(U, y_ord)
        alpha_v = np.empty_like(alpha_ord)
        alpha_v[P_full] = alpha_ord

        return alpha_v, U, nll

    def _sparse_W_and_alpha(self, K):
        """
        Compute W = K^{-1} - alpha*alpha^T and alpha using the sparse U factor.

        K is in the ORIGINAL index space (size N_total x N_total).
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
        """Analytic gradient of the NLL using the sparse U factor."""
        try:
            K, phi, n_bases, oti, diffs = self._build_K_and_phi(x0)
            W, _, _, _ = self._sparse_W_and_alpha(K)
        except Exception:
            return np.zeros(len(x0))
        return self._compute_grad(x0, W, phi, n_bases, oti, diffs)

    def nll_and_grad(self, x0):
        """
        Compute NLL and its gradient in a single pass using the sparse U factor.

        Returns
        -------
        nll : float
        grad : ndarray
        """
        try:
            K, phi, n_bases, oti, diffs = self._build_K_and_phi(x0)
            W, alpha_v, U, nll = self._sparse_W_and_alpha(K)
        except Exception:
            return 1e6, np.zeros(len(x0))

        # Cache for fast prediction
        self.model._cached_U = U
        self.model._cached_P = self.model.mmd_P_full
        self.model._cached_alpha = alpha_v
        self.model._cached_L = None
        self.model._cached_low = None
        self.model._cached_params = x0.copy()

        grad = self._compute_grad(x0, W, phi, n_bases, oti, diffs)
        return float(nll), grad

    def optimize_hyperparameters(self,
    optimizer="pso",
    **kwargs):
        """
        Optimize the GDDEGP model hyperparameters.

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
            optimizer_fn = optimizer

        bounds = self.model.bounds
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]

        # Inject nll_and_grad for gradient-aware optimizers.
        if optimizer in ('lbfgs', 'jade', 'pso') and 'func_and_grad' not in kwargs and 'grad_func' not in kwargs:
            kwargs['func_and_grad'] = self.nll_and_grad

        best_x, best_val = optimizer_fn(self.nll_wrapper, lb, ub, **kwargs)

        self.model.opt_x0 = best_x
        self.model.opt_nll = best_val

        return best_x
