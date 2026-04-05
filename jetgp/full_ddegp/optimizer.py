import numpy as np

from scipy.linalg import cho_solve, cho_factor
from jetgp.full_ddegp import ddegp_utils as utils
from line_profiler import profile
import jetgp.utils as gen_utils
from jetgp.hyperparameter_optimizers import OPTIMIZERS
from jetgp.utils import matern_kernel_grad_builder

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

    def _get_deriv_buf(self, phi, n_bases, order):
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
                for _, prefix_counts in Optimizer._enum_factors_with_counts(last, ordi - 1):
                    counts = dict(prefix_counts)
                    counts[last] = counts.get(last, 0) + 1
                    f = 1
                    for c in counts.values():
                        f *= factorial(c)
                    yield float(f)

    @staticmethod
    def _enum_factors_with_counts(max_basis, ordi):
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
        key = (n_bases, order)
        if self._deriv_factors is not None and self._deriv_factors_key == key:
            return self._deriv_factors
        factors = [1.0]
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
            self.model.flattened_der_indices, self.model.powers,
            self.model.derivative_locations,
        )

    @profile
    def negative_log_marginal_likelihood(self, x0):
        """
        Compute the negative log marginal likelihood (NLL) of the model.

        NLL = 0.5 * y^T K^-1 y + 0.5 * log|K| + 0.5 * N * log(2π)

        Parameters
        ----------
        x0 : ndarray
            Vector of log-scaled hyperparameters (length scales and noise).

        Returns
        -------
        float
            Value of the negative log marginal likelihood.
        """
        ell = x0[:-1]
        sigma_n = x0[-1]
        llhood = 0
        diffs = self.model.differences_by_dim
        phi = self.model.kernel_func(diffs, ell)
        n_bases = phi.get_active_bases()[-1]
        
        # Extract ALL derivative components
        deriv_order = 2 * self.model.n_order
        phi_exp = self._expand_derivs(phi, n_bases, deriv_order)
        K = self._build_K(phi_exp, phi, n_bases)
        K += ((10 ** sigma_n) ** 2) * np.eye(len(K))
        K += self.model.sigma_data**2

        try:
            L, low = cho_factor(K)
            alpha = cho_solve(
                (L, low),
                self.model.y_train
            )

            data_fit = 0.5 * np.dot(self.model.y_train, alpha)
            log_det_K = np.sum(np.log(np.diag(L)))
            complexity = log_det_K
            N = len(self.model.y_train)
            const = 0.5 * N * np.log(2 * np.pi)
            return data_fit + complexity + const
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

    def nll_grad(self, x0):
        """Analytic gradient of the NLL w.r.t. log10-scaled hyperparameters."""
        ln10 = np.log(10.0)

        kernel      = self.model.kernel
        kernel_type = self.model.kernel_type
        D           = len(self.model.differences_by_dim)
        sigma_n_sq  = (10.0 ** x0[-1]) ** 2
        diffs       = self.model.differences_by_dim
        oti         = self.model.kernel_factory.oti

        phi = self.model.kernel_func(diffs, x0[:-1])
        n_bases = phi.get_active_bases()[-1]
        deriv_order = 2 * self.model.n_order
        phi_exp = self._expand_derivs(phi, n_bases, deriv_order)

        K = self._build_K(phi_exp, phi, n_bases)
        K.flat[::K.shape[0] + 1] += sigma_n_sq
        K += self.model.sigma_data ** 2

        try:
            L, low  = cho_factor(K)
            alpha_v = cho_solve((L, low), self.model.y_train)
            N       = len(self.model.y_train)
            K_inv   = cho_solve((L, low), np.eye(N))
            W       = K_inv - np.outer(alpha_v, alpha_v)
        except Exception:
            return np.zeros(len(x0))

        grad = np.zeros(len(x0))
        use_fast = self._kernel_plan is not None
        base_shape = phi.shape

        W_proj = None
        if use_fast:
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
                    dphi, dphi_exp, self.model.n_order, n_bases,
                    self.model.flattened_der_indices, self.model.powers,
                    self.model.derivative_locations,
                )
                return 0.5 * np.vdot(W, dK)

        grad[-2] = _gc(oti.mul(2.0 * ln10, phi))
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
                        grad[d] = _gc(oti.mul(-ln10 * ell[d] ** 2,
                                              oti.mul(oti.mul(diffs[d], diffs[d]), phi)))
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
                ell = 10.0 ** x0[:D]; alpha_rq = 10.0 ** float(x0[D]); alpha_idx = D
            else:
                ell = np.full(D, 10.0 ** float(x0[0]))
                alpha_rq = np.exp(float(x0[1])); alpha_idx = 1
            if hasattr(phi, 'fused_sqdist'):
                r2 = oti.zeros(phi.shape)
                ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
                r2.fused_sqdist(diffs, ell_sq)
            else:
                r2 = oti.mul(ell[0], diffs[0]); r2 = oti.mul(r2, r2)
                for d in range(1, D):
                    td = oti.mul(ell[d], diffs[d]); r2 = oti.sum(r2, oti.mul(td, td))
            base = oti.sum(1.0, oti.mul(r2, 1.0 / (2.0 * alpha_rq)))
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
                        grad[d] = _gc(oti.mul(-ln10 * ell[d] ** 2,
                                              oti.mul(oti.mul(diffs[d], diffs[d]), phi_over_base)))
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
                ell = 10.0 ** x0[:D]; p = 10.0 ** x0[D:2*D]
                pip = np.pi / p; p_start = D
            else:
                ell = np.full(D, 10.0 ** float(x0[0]))
                pip = np.full(D, np.pi / 10.0 ** float(x0[1])); p_start = 1
            sin_d = [oti.sin(oti.mul(pip[d], diffs[d])) for d in range(D)]
            cos_d = [oti.cos(oti.mul(pip[d], diffs[d])) for d in range(D)]
            if kernel_type == 'anisotropic':
                if hasattr(phi, 'fused_scale_sq_mul'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul(sin_d[d], phi, -4.0 * ln10 * ell[d] ** 2)
                        grad[d] = _gc(dphi_buf)
                else:
                    for d in range(D):
                        grad[d] = _gc(oti.mul(-4.0 * ln10 * ell[d] ** 2,
                                              oti.mul(oti.mul(sin_d[d], sin_d[d]), phi)))
                for d in range(D):
                    sc = oti.mul(sin_d[d], oti.mul(cos_d[d], diffs[d]))
                    grad[p_start + d] = _gc(oti.mul(4.0 * ln10 * ell[d] ** 2 * pip[d],
                                                     oti.mul(sc, phi)))
            else:
                if hasattr(phi, 'fused_sum_sq'):
                    ss = oti.zeros(phi.shape)
                    ss.fused_sum_sq(sin_d)
                else:
                    ss = oti.mul(sin_d[0], sin_d[0])
                    for d in range(1, D):
                        ss = oti.sum(ss, oti.mul(sin_d[d], sin_d[d]))
                grad[0] = _gc(oti.mul(-4.0 * ln10 * ell[0] ** 2, oti.mul(ss, phi)))
                scd = oti.mul(sin_d[0], oti.mul(cos_d[0], diffs[0]))
                for d in range(1, D):
                    scd = oti.sum(scd, oti.mul(sin_d[d], oti.mul(cos_d[d], diffs[d])))
                grad[p_start] = _gc(oti.mul(4.0 * ln10 * ell[0] ** 2 * pip[0],
                                            oti.mul(scd, phi)))

        elif kernel == 'Matern':
            kf = self.model.kernel_factory
            if not hasattr(kf, '_matern_grad_prebuild'):
                kf._matern_grad_prebuild = matern_kernel_grad_builder(getattr(kf, "nu", 1.5), oti_module=oti)
            ell = (10.0 ** x0[:D] if kernel_type == 'anisotropic'
                   else np.full(D, 10.0 ** float(x0[0])))
            sigma_f_sq = (10.0 ** float(x0[-2])) ** 2
            _eps = 1e-10
            if hasattr(phi, 'fused_sqdist'):
                r2 = oti.zeros(phi.shape)
                ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
                r2.fused_sqdist(diffs, ell_sq)
            else:
                r2 = oti.mul(ell[0], diffs[0]); r2 = oti.mul(r2, r2)
                for d in range(1, D):
                    td = oti.mul(ell[d], diffs[d]); r2 = oti.sum(r2, oti.mul(td, td))
            r_oti = oti.sqrt(oti.sum(r2, _eps ** 2))
            f_prime_r = kf._matern_grad_prebuild(r_oti)
            inv_r = oti.pow(r_oti, -1)
            base_matern = oti.mul(sigma_f_sq, oti.mul(f_prime_r, inv_r))
            if kernel_type == 'anisotropic':
                if hasattr(phi, 'fused_scale_sq_mul'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul(diffs[d], base_matern, ln10 * ell[d] ** 2)
                        grad[d] = _gc(dphi_buf)
                else:
                    for d in range(D):
                        d_sq = oti.mul(diffs[d], diffs[d])
                        dphi_d = oti.mul(ln10 * ell[d] ** 2, oti.mul(d_sq, base_matern))
                        grad[d] = _gc(dphi_d)
            else:
                if hasattr(phi, 'fused_sum_sq'):
                    sum_dsq = oti.zeros(phi.shape)
                    sum_dsq.fused_sum_sq(diffs)
                else:
                    sum_dsq = oti.mul(diffs[0], diffs[0])
                    for d in range(1, D):
                        sum_dsq = oti.sum(sum_dsq, oti.mul(diffs[d], diffs[d]))
                dphi_e = oti.mul(ln10 * ell[0] ** 2, oti.mul(sum_dsq, base_matern))
                grad[0] = _gc(dphi_e)

        return grad

    def nll_and_grad(self, x0):
        """Compute NLL and its gradient in a single pass, sharing one Cholesky."""
        ln10 = np.log(10.0)

        kernel      = self.model.kernel
        kernel_type = self.model.kernel_type
        D           = len(self.model.differences_by_dim)
        sigma_n_sq  = (10.0 ** x0[-1]) ** 2
        diffs       = self.model.differences_by_dim
        oti         = self.model.kernel_factory.oti

        # --- shared kernel computation (done ONCE) ---
        phi = self.model.kernel_func(diffs, x0[:-1])
        if self.model.n_order == 0:
            n_bases = 0
            phi_exp = phi.real[np.newaxis, :, :]
        else:
            n_bases = phi.get_active_bases()[-1]
            deriv_order = 2 * self.model.n_order
            phi_exp = self._expand_derivs(phi, n_bases, deriv_order)

        K = self._build_K(phi_exp, phi, n_bases)
        K.flat[::K.shape[0] + 1] += sigma_n_sq
        K += self.model.sigma_data ** 2

        try:
            L, low  = cho_factor(K)
            alpha_v = cho_solve((L, low), self.model.y_train)
            N       = len(self.model.y_train)

            # NLL
            nll = (0.5 * np.dot(self.model.y_train, alpha_v)
                   + np.sum(np.log(np.diag(L)))
                   + 0.5 * N * np.log(2 * np.pi))

            # W matrix for gradient (reuse same Cholesky)
            K_inv = cho_solve((L, low), np.eye(N))
            W     = K_inv - np.outer(alpha_v, alpha_v)
        except Exception:
            return 1e6, np.zeros(len(x0))

        # --- gradient from W (no second kernel build / Cholesky) ---
        grad = np.zeros(len(x0))
        use_fast = self._kernel_plan is not None
        base_shape = phi.shape
        deriv_order_gc = 2 * self.model.n_order

        W_proj = None
        if use_fast and self.model.n_order > 0:
            from math import comb
            ndir = comb(n_bases + deriv_order_gc, deriv_order_gc)
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
            _vdot_factors = self._get_deriv_factors(n_bases, deriv_order_gc)

        def _gc(dphi):
            if _use_vdot_fused:
                return 0.5 * dphi.vdot_expand_fast(_vdot_factors, W_proj)
            if self.model.n_order == 0:
                dphi_exp = dphi.real[np.newaxis, :, :]
            else:
                dphi_exp = self._expand_derivs(dphi, n_bases, deriv_order_gc)
            if W_proj is not None:
                dphi_3d = dphi_exp.reshape(W_proj.shape)
                return 0.5 * np.vdot(W_proj, dphi_3d)
            elif use_fast:
                dphi_3d = dphi_exp.reshape(dphi_exp.shape[0], base_shape[0], base_shape[1])
                dK = utils.rbf_kernel_fast(dphi_3d, self._kernel_plan, out=self._dK_buf)
                return 0.5 * np.vdot(W, dK)
            else:
                dK = utils.rbf_kernel(
                    dphi, dphi_exp, self.model.n_order, n_bases,
                    self.model.flattened_der_indices, self.model.powers,
                    self.model.derivative_locations,
                )
                return 0.5 * np.vdot(W, dK)

        grad[-2] = _gc(oti.mul(2.0 * ln10, phi))
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
                        grad[d] = _gc(oti.mul(-ln10 * ell[d] ** 2,
                                              oti.mul(oti.mul(diffs[d], diffs[d]), phi)))
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
                ell = 10.0 ** x0[:D]; alpha_rq = 10.0 ** float(x0[D]); alpha_idx = D
            else:
                ell = np.full(D, 10.0 ** float(x0[0]))
                alpha_rq = np.exp(float(x0[1])); alpha_idx = 1
            if hasattr(phi, 'fused_sqdist'):
                r2 = oti.zeros(phi.shape)
                ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
                r2.fused_sqdist(diffs, ell_sq)
            else:
                r2 = oti.mul(ell[0], diffs[0]); r2 = oti.mul(r2, r2)
                for d in range(1, D):
                    td = oti.mul(ell[d], diffs[d]); r2 = oti.sum(r2, oti.mul(td, td))
            base = oti.sum(1.0, oti.mul(r2, 1.0 / (2.0 * alpha_rq)))
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
                        grad[d] = _gc(oti.mul(-ln10 * ell[d] ** 2,
                                              oti.mul(oti.mul(diffs[d], diffs[d]), phi_over_base)))
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
                ell = 10.0 ** x0[:D]; p = 10.0 ** x0[D:2*D]
                pip = np.pi / p; p_start = D
            else:
                ell = np.full(D, 10.0 ** float(x0[0]))
                pip = np.full(D, np.pi / 10.0 ** float(x0[1])); p_start = 1
            sin_d = [oti.sin(oti.mul(pip[d], diffs[d])) for d in range(D)]
            cos_d = [oti.cos(oti.mul(pip[d], diffs[d])) for d in range(D)]
            if kernel_type == 'anisotropic':
                if hasattr(phi, 'fused_scale_sq_mul'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul(sin_d[d], phi, -4.0 * ln10 * ell[d] ** 2)
                        grad[d] = _gc(dphi_buf)
                else:
                    for d in range(D):
                        grad[d] = _gc(oti.mul(-4.0 * ln10 * ell[d] ** 2,
                                              oti.mul(oti.mul(sin_d[d], sin_d[d]), phi)))
                for d in range(D):
                    sc = oti.mul(sin_d[d], oti.mul(cos_d[d], diffs[d]))
                    grad[p_start + d] = _gc(oti.mul(4.0 * ln10 * ell[d] ** 2 * pip[d],
                                                     oti.mul(sc, phi)))
            else:
                if hasattr(phi, 'fused_sum_sq'):
                    ss = oti.zeros(phi.shape)
                    ss.fused_sum_sq(sin_d)
                else:
                    ss = oti.mul(sin_d[0], sin_d[0])
                    for d in range(1, D):
                        ss = oti.sum(ss, oti.mul(sin_d[d], sin_d[d]))
                grad[0] = _gc(oti.mul(-4.0 * ln10 * ell[0] ** 2, oti.mul(ss, phi)))
                scd = oti.mul(sin_d[0], oti.mul(cos_d[0], diffs[0]))
                for d in range(1, D):
                    scd = oti.sum(scd, oti.mul(sin_d[d], oti.mul(cos_d[d], diffs[d])))
                grad[p_start] = _gc(oti.mul(4.0 * ln10 * ell[0] ** 2 * pip[0],
                                            oti.mul(scd, phi)))

        elif kernel == 'Matern':
            kf = self.model.kernel_factory
            if not hasattr(kf, '_matern_grad_prebuild'):
                kf._matern_grad_prebuild = matern_kernel_grad_builder(getattr(kf, "nu", 1.5), oti_module=oti)
            ell = (10.0 ** x0[:D] if kernel_type == 'anisotropic'
                   else np.full(D, 10.0 ** float(x0[0])))
            sigma_f_sq = (10.0 ** float(x0[-2])) ** 2
            _eps = 1e-10
            if hasattr(phi, 'fused_sqdist'):
                r2 = oti.zeros(phi.shape)
                ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
                r2.fused_sqdist(diffs, ell_sq)
            else:
                r2 = oti.mul(ell[0], diffs[0]); r2 = oti.mul(r2, r2)
                for d in range(1, D):
                    td = oti.mul(ell[d], diffs[d]); r2 = oti.sum(r2, oti.mul(td, td))
            r_oti = oti.sqrt(oti.sum(r2, _eps ** 2))
            f_prime_r = kf._matern_grad_prebuild(r_oti)
            inv_r = oti.pow(r_oti, -1)
            base_matern = oti.mul(sigma_f_sq, oti.mul(f_prime_r, inv_r))
            if kernel_type == 'anisotropic':
                if hasattr(phi, 'fused_scale_sq_mul'):
                    dphi_buf = oti.zeros(phi.shape)
                    for d in range(D):
                        dphi_buf.fused_scale_sq_mul(diffs[d], base_matern, ln10 * ell[d] ** 2)
                        grad[d] = _gc(dphi_buf)
                else:
                    for d in range(D):
                        d_sq = oti.mul(diffs[d], diffs[d])
                        dphi_d = oti.mul(ln10 * ell[d] ** 2, oti.mul(d_sq, base_matern))
                        grad[d] = _gc(dphi_d)
            else:
                if hasattr(phi, 'fused_sum_sq'):
                    sum_dsq = oti.zeros(phi.shape)
                    sum_dsq.fused_sum_sq(diffs)
                else:
                    sum_dsq = oti.mul(diffs[0], diffs[0])
                    for d in range(1, D):
                        sum_dsq = oti.sum(sum_dsq, oti.mul(diffs[d], diffs[d]))
                dphi_e = oti.mul(ln10 * ell[0] ** 2, oti.mul(sum_dsq, base_matern))
                grad[0] = _gc(dphi_e)

        return float(nll), grad

    def optimize_hyperparameters(    self,
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

        if optimizer in ('lbfgs', 'jade', 'pso') and 'func_and_grad' not in kwargs and 'grad_func' not in kwargs:
            kwargs['func_and_grad'] = self.nll_and_grad

        best_x, best_val = optimizer_fn(self.nll_wrapper, lb, ub, **kwargs)

        self.model.opt_x0 = best_x
        self.model.opt_nll = best_val


        return best_x

