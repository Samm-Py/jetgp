import numpy as np
from scipy.linalg import cho_solve, cho_factor
from line_profiler import profile
import jetgp.utils as gen_utils
from jetgp.hyperparameter_optimizers import OPTIMIZERS
from jetgp.utils import matern_kernel_grad_builder


class Optimizer:
    """
    Optimizer class for fitting the hyperparameters of a weighted derivative-enhanced GP model (wDEGP)
    by minimizing the negative log marginal likelihood (NLL).

    Supports DEGP, DDEGP, and GDDEGP modes.

    Attributes
    ----------
    model : object
        Instance of a weighted derivative-enhanced GP model (wDEGP) with attributes:
        x_train, y_train, n_order, n_bases, der_indices, index, bounds, submodel_type, etc.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : object
            An instance of a wDEGP model containing training data, hyperparameter bounds,
            and other model-specific structures required for kernel computation.
        """
        self.model = model

        # Import the appropriate utils module based on submodel_type
        self._setup_utils()

        # Precompute kernel plans (structural info that never changes)
        self._kernel_plans = None  # lazily initialized on first NLL call

    def _setup_utils(self):
        """Set up the correct utils module based on submodel_type."""
        submodel_type = getattr(self.model, 'submodel_type', 'degp')
        
        if submodel_type == 'degp':
            from jetgp.wdegp import wdegp_utils
            self.utils = wdegp_utils
        elif submodel_type == 'ddegp':
            from jetgp.full_ddegp import wddegp_utils
            self.utils = wddegp_utils
        elif submodel_type == 'gddegp':
            from jetgp.full_gddegp import wgddegp_utils
            self.utils = wgddegp_utils
        else:
            # Default to degp
            from jetgp.wdegp import wdegp_utils
            self.utils = wdegp_utils

    def _ensure_kernel_plans(self, n_bases):
        """Lazily precompute kernel plans for all submodels (once per n_bases)."""
        if self._kernel_plans is not None and self._kernel_plans_n_bases == n_bases:
            return
        if not hasattr(self.utils, 'precompute_kernel_plan'):
            self._kernel_plans = None
            return
        plans = []
        index = self.model.derivative_locations
        for i in range(len(index)):
            plan = self.utils.precompute_kernel_plan(
                self.model.n_order, n_bases,
                self.model.flattened_der_indices[i],
                self.model.powers[i],
                index[i],
            )
            plans.append(plan)
        self._kernel_plans = plans
        self._kernel_plans_n_bases = n_bases

    @profile
    def negative_log_marginal_likelihood(
        self,
        x0,
        x_train,
        y_train,
        n_order,
        n_bases,
        der_indices,
        index,
    ):
        """
        Computes the negative log marginal likelihood (NLL) for a given hyperparameter vector.

        NLL = 0.5 * y^T (K^-1) y + 0.5 * log|K| + 0.5*N*log(2*pi)

        Parameters
        ----------
        x0 : ndarray
            Log-scaled hyperparameter vector, where the last entry is log10(sigma_n).
        x_train : list of ndarrays
            Input training points (unused inside loop, included for general interface).
        y_train : list of ndarrays
            List of function and derivative training values for each submodel.
        n_order : int
            Maximum order of derivatives used.
        n_bases : int
            Number of Taylor bases used in the expansion.
        der_indices : list
            Multi-index derivative information.
        index : list of lists
            Indices partitioning the training data into submodels (derivative_locations).

        Returns
        -------
        float
            The computed negative log marginal likelihood.
        """
        ell = x0[:-1]
        sigma_n = x0[-1]
        llhood = 0
        # ell[0] = 0
        # ell[1] = 0
        # ell[2] = 0
        # sigma_n = -16
        diffs = self.model.differences_by_dim
        phi = self.model.kernel_func(diffs, ell)
        if self.model.n_order == 0:
            n_bases = 0
            phi_exp = phi.real
            phi_exp = phi_exp[np.newaxis, :, :]
        else:
            n_bases = phi.get_active_bases()[-1]
        
            # Extract ALL derivative components into a single flat array (highly efficient)
            phi_exp = phi.get_all_derivs(n_bases, 2 * n_order)

        # Ensure kernel plans are precomputed
        self._ensure_kernel_plans(n_bases)
        use_fast = self._kernel_plans is not None

        # Pre-reshape phi_exp to 3D once
        if use_fast:
            base_shape = phi.shape
            phi_exp_3d = phi_exp.reshape(phi_exp.shape[0], base_shape[0], base_shape[1])

        for i in range(len(index)):
            y_train_sub = y_train[i]

            if use_fast:
                K = self.utils.rbf_kernel_fast(phi_exp_3d, self._kernel_plans[i])
            else:
                K = self.utils.rbf_kernel(
                    phi, phi_exp, n_order, n_bases,
                    self.model.flattened_der_indices[i],
                    self.model.powers[i], index=index[i]
                )

            K += (10 ** sigma_n) ** 2 * np.eye(len(K))

            try:
                L, low = cho_factor(K)
                alpha = cho_solve(
                    (L, low),
                    y_train_sub
                )

                data_fit = 0.5 * np.dot(y_train_sub.flatten(), alpha.flatten())
                log_det = np.sum(np.log(np.diag(L)))
                const = 0.5 * len(y_train_sub) * np.log(2 * np.pi)

                llhood += data_fit + log_det + const
            except np.linalg.LinAlgError:
                llhood += 1e6  # Penalize badly conditioned matrices

        return llhood

    def nll_wrapper(self, x0):
        """
        Wrapper for NLL function to fit PSO optimizer interface.

        Parameters
        ----------
        x0 : ndarray
            Hyperparameter vector.

        Returns
        -------
        float
            Computed NLL value.
        """
        return self.negative_log_marginal_likelihood(
            x0,
            self.model.x_train,
            self.model.y_train_normalized,
            self.model.n_order,
            self.model.n_bases,
            self.model.der_indices,
            self.model.derivative_locations,
        )

    def nll_grad(self, x0):
        """Analytic gradient of the NLL w.r.t. log10-scaled hyperparameters."""
        ln10 = np.log(10.0)

        kernel      = self.model.kernel
        kernel_type = self.model.kernel_type
        D           = len(self.model.differences_by_dim)
        sigma_n_sq  = (10.0 ** x0[-1]) ** 2
        diffs       = self.model.differences_by_dim
        oti         = self.model.kernel_factory.oti
        index       = self.model.derivative_locations

        phi = self.model.kernel_func(diffs, x0[:-1])
        if self.model.n_order == 0:
            n_bases = 0
            phi_exp = phi.real[np.newaxis, :, :]
        else:
            n_bases = phi.get_active_bases()[-1]
            phi_exp = phi.get_all_derivs(n_bases, 2 * self.model.n_order)

        # Ensure kernel plans are precomputed
        self._ensure_kernel_plans(n_bases)
        use_fast = self._kernel_plans is not None

        # Pre-reshape phi_exp to 3D once
        if use_fast:
            base_shape = phi.shape
            phi_exp_3d = phi_exp.reshape(phi_exp.shape[0], base_shape[0], base_shape[1])

        # Build per-submodel W matrices
        W_list = []
        for i in range(len(index)):
            y_train_sub = self.model.y_train_normalized[i]
            if use_fast:
                K = self.utils.rbf_kernel_fast(phi_exp_3d, self._kernel_plans[i])
            else:
                K = self.utils.rbf_kernel(
                    phi, phi_exp, self.model.n_order, n_bases,
                    self.model.flattened_der_indices[i],
                    self.model.powers[i], index=index[i]
                )
            K.flat[::K.shape[0] + 1] += sigma_n_sq
            try:
                L, low  = cho_factor(K)
                alpha_v = cho_solve((L, low), y_train_sub)
                N       = len(y_train_sub)
                K_inv   = cho_solve((L, low), np.eye(N))
                W_list.append(K_inv - np.outer(alpha_v, alpha_v))
            except Exception:
                return np.zeros(len(x0))

        grad = np.zeros(len(x0))

        def _gc(dphi):
            if self.model.n_order == 0:
                dphi_exp = dphi.real[np.newaxis, :, :]
            else:
                dphi_exp = dphi.get_all_derivs(n_bases, 2 * self.model.n_order)
            if use_fast:
                dphi_3d = dphi_exp.reshape(dphi_exp.shape[0], base_shape[0], base_shape[1])
            total = 0.0
            for i in range(len(index)):
                if use_fast:
                    dK = self.utils.rbf_kernel_fast(dphi_3d, self._kernel_plans[i])
                else:
                    dK = self.utils.rbf_kernel(
                        dphi, dphi_exp,
                        self.model.n_order, n_bases,
                        self.model.flattened_der_indices[i],
                        self.model.powers[i],
                        index=index[i],
                    )
                total += np.sum(W_list[i] * dK)
            return 0.5 * total

        grad[-2] = _gc(oti.mul(2.0 * ln10, phi))
        grad[-1] = ln10 * sigma_n_sq * sum(np.trace(W) for W in W_list)

        if kernel == 'SE':
            if kernel_type == 'anisotropic':
                ell = 10.0 ** x0[:D]
                for d in range(D):
                    grad[d] = _gc(oti.mul(-ln10 * ell[d] ** 2,
                                          oti.mul(oti.mul(diffs[d], diffs[d]), phi)))
            else:
                ell    = 10.0 ** float(x0[0])
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
            r2 = oti.mul(ell[0], diffs[0]); r2 = oti.mul(r2, r2)
            for d in range(1, D):
                td = oti.mul(ell[d], diffs[d]); r2 = oti.sum(r2, oti.mul(td, td))
            base = oti.sum(1.0, oti.mul(r2, 1.0 / (2.0 * alpha_rq)))
            inv_base = oti.pow(base, -1)
            phi_over_base = oti.mul(phi, inv_base)
            if kernel_type == 'anisotropic':
                for d in range(D):
                    grad[d] = _gc(oti.mul(-ln10 * ell[d] ** 2,
                                          oti.mul(oti.mul(diffs[d], diffs[d]), phi_over_base)))
            else:
                sum_sq = oti.mul(diffs[0], diffs[0])
                for d in range(1, D):
                    sum_sq = oti.sum(sum_sq, oti.mul(diffs[d], diffs[d]))
                grad[0] = _gc(oti.mul(-ln10 * ell[0] ** 2, oti.mul(sum_sq, phi_over_base)))
            log_base = oti.log(base)
            term = oti.sum(oti.mul(-1.0, log_base), oti.sum(1.0, oti.mul(-1.0, inv_base)))
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
                for d in range(D):
                    grad[d] = _gc(oti.mul(-4.0 * ln10 * ell[d] ** 2,
                                          oti.mul(oti.mul(sin_d[d], sin_d[d]), phi)))
                for d in range(D):
                    sc = oti.mul(sin_d[d], oti.mul(cos_d[d], diffs[d]))
                    grad[p_start + d] = _gc(oti.mul(4.0 * ln10 * ell[d] ** 2 * pip[d],
                                                     oti.mul(sc, phi)))
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
            r_oti = oti.sqrt(sum((ell[d] * diffs[d]) ** 2 for d in range(D)) + _eps ** 2)
            f_prime_r = kf._matern_grad_prebuild(r_oti)
            inv_r = oti.pow(r_oti, -1)
            if kernel_type == 'anisotropic':
                for d in range(D):
                    d_sq = oti.mul(diffs[d], diffs[d])
                    grad[d] = _gc(oti.mul(sigma_f_sq,
                                          oti.mul(f_prime_r,
                                                  oti.mul(ln10 * ell[d] ** 2,
                                                          oti.mul(d_sq, inv_r)))))
            else:
                sum_dsq = oti.mul(diffs[0], diffs[0])
                for d in range(1, D):
                    sum_dsq = oti.sum(sum_dsq, oti.mul(diffs[d], diffs[d]))
                grad[0] = _gc(oti.mul(sigma_f_sq,
                                      oti.mul(f_prime_r,
                                              oti.mul(ln10 * ell[0] ** 2,
                                                      oti.mul(sum_dsq, inv_r)))))

        return grad

    def nll_and_grad(self, x0):
        """Compute NLL and its gradient in a single pass, sharing one Cholesky per submodel."""
        ln10 = np.log(10.0)

        kernel      = self.model.kernel
        kernel_type = self.model.kernel_type
        D           = len(self.model.differences_by_dim)
        sigma_n_sq  = (10.0 ** x0[-1]) ** 2
        diffs       = self.model.differences_by_dim
        oti         = self.model.kernel_factory.oti
        index       = self.model.derivative_locations

        # --- shared kernel computation (done ONCE) ---
        phi = self.model.kernel_func(diffs, x0[:-1])
        if self.model.n_order == 0:
            n_bases = 0
            phi_exp = phi.real[np.newaxis, :, :]
        else:
            n_bases = phi.get_active_bases()[-1]
            phi_exp = phi.get_all_derivs(n_bases, 2 * self.model.n_order)

        # Ensure kernel plans are precomputed
        self._ensure_kernel_plans(n_bases)
        use_fast = self._kernel_plans is not None

        # Pre-reshape phi_exp to 3D once
        if use_fast:
            base_shape = phi.shape
            phi_exp_3d = phi_exp.reshape(phi_exp.shape[0], base_shape[0], base_shape[1])

        # --- single loop: compute NLL and W_list simultaneously ---
        llhood = 0.0
        W_list = []
        for i in range(len(index)):
            y_train_sub = self.model.y_train_normalized[i]

            if use_fast:
                K = self.utils.rbf_kernel_fast(phi_exp_3d, self._kernel_plans[i])
            else:
                K = self.utils.rbf_kernel(
                    phi, phi_exp, self.model.n_order, n_bases,
                    self.model.flattened_der_indices[i],
                    self.model.powers[i], index=index[i]
                )
            K.flat[::K.shape[0] + 1] += sigma_n_sq

            try:
                L, low  = cho_factor(K)
                alpha_v = cho_solve((L, low), y_train_sub)
                N       = len(y_train_sub)

                # NLL contribution
                data_fit = 0.5 * np.dot(y_train_sub.flatten(), alpha_v.flatten())
                log_det  = np.sum(np.log(np.diag(L)))
                const    = 0.5 * N * np.log(2 * np.pi)
                llhood  += data_fit + log_det + const

                # W matrix for gradient (reuse same Cholesky)
                K_inv = cho_solve((L, low), np.eye(N))
                W_list.append(K_inv - np.outer(alpha_v, alpha_v))
            except np.linalg.LinAlgError:
                llhood += 1e6
                return float(llhood), np.zeros(len(x0))

        # --- gradient from W_list (no second kernel build / Cholesky) ---
        grad = np.zeros(len(x0))
        n_sub = len(index)

        def _gc(dphi):
            # Precompute dphi_exp ONCE, reshape to 3D
            if self.model.n_order == 0:
                dphi_exp = dphi.real[np.newaxis, :, :]
            else:
                dphi_exp = dphi.get_all_derivs(n_bases, 2 * self.model.n_order)
            if use_fast:
                dphi_3d = dphi_exp.reshape(dphi_exp.shape[0], base_shape[0], base_shape[1])
            total = 0.0
            for i in range(n_sub):
                if use_fast:
                    dK = self.utils.rbf_kernel_fast(dphi_3d, self._kernel_plans[i])
                else:
                    dK = self.utils.rbf_kernel(
                        dphi, dphi_exp,
                        self.model.n_order, n_bases,
                        self.model.flattened_der_indices[i],
                        self.model.powers[i],
                        index=index[i],
                    )
                total += np.sum(W_list[i] * dK)
            return 0.5 * total

        grad[-2] = _gc(oti.mul(2.0 * ln10, phi))
        grad[-1] = ln10 * sigma_n_sq * sum(np.trace(W) for W in W_list)

        if kernel == 'SE':
            if kernel_type == 'anisotropic':
                ell = 10.0 ** x0[:D]
                for d in range(D):
                    grad[d] = _gc(oti.mul(-ln10 * ell[d] ** 2,
                                          oti.mul(oti.mul(diffs[d], diffs[d]), phi)))
            else:
                ell    = 10.0 ** float(x0[0])
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
            r2 = oti.mul(ell[0], diffs[0]); r2 = oti.mul(r2, r2)
            for d in range(1, D):
                td = oti.mul(ell[d], diffs[d]); r2 = oti.sum(r2, oti.mul(td, td))
            base = oti.sum(1.0, oti.mul(r2, 1.0 / (2.0 * alpha_rq)))
            inv_base = oti.pow(base, -1)
            phi_over_base = oti.mul(phi, inv_base)
            if kernel_type == 'anisotropic':
                for d in range(D):
                    grad[d] = _gc(oti.mul(-ln10 * ell[d] ** 2,
                                          oti.mul(oti.mul(diffs[d], diffs[d]), phi_over_base)))
            else:
                sum_sq = oti.mul(diffs[0], diffs[0])
                for d in range(1, D):
                    sum_sq = oti.sum(sum_sq, oti.mul(diffs[d], diffs[d]))
                grad[0] = _gc(oti.mul(-ln10 * ell[0] ** 2, oti.mul(sum_sq, phi_over_base)))
            log_base = oti.log(base)
            term = oti.sum(oti.mul(-1.0, log_base), oti.sum(1.0, oti.mul(-1.0, inv_base)))
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
                for d in range(D):
                    grad[d] = _gc(oti.mul(-4.0 * ln10 * ell[d] ** 2,
                                          oti.mul(oti.mul(sin_d[d], sin_d[d]), phi)))
                for d in range(D):
                    sc = oti.mul(sin_d[d], oti.mul(cos_d[d], diffs[d]))
                    grad[p_start + d] = _gc(oti.mul(4.0 * ln10 * ell[d] ** 2 * pip[d],
                                                     oti.mul(sc, phi)))
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
            r_oti = oti.sqrt(sum((ell[d] * diffs[d]) ** 2 for d in range(D)) + _eps ** 2)
            f_prime_r = kf._matern_grad_prebuild(r_oti)
            inv_r = oti.pow(r_oti, -1)
            if kernel_type == 'anisotropic':
                for d in range(D):
                    d_sq = oti.mul(diffs[d], diffs[d])
                    grad[d] = _gc(oti.mul(sigma_f_sq,
                                          oti.mul(f_prime_r,
                                                  oti.mul(ln10 * ell[d] ** 2,
                                                          oti.mul(d_sq, inv_r)))))
            else:
                sum_dsq = oti.mul(diffs[0], diffs[0])
                for d in range(1, D):
                    sum_dsq = oti.sum(sum_dsq, oti.mul(diffs[d], diffs[d]))
                grad[0] = _gc(oti.mul(sigma_f_sq,
                                      oti.mul(f_prime_r,
                                              oti.mul(ln10 * ell[0] ** 2,
                                                      oti.mul(sum_dsq, inv_r)))))

        return float(llhood), grad

    def optimize_hyperparameters(
        self,
        optimizer="pso",
        **kwargs
    ):
        """
        Optimize the DEGP model hyperparameters using the specified optimizer.

        Parameters:
        ----------
        optimizer : str or callable, default="pso"
            Name of optimizer or callable. Available: 'pso', 'lbfgs', 'jade', etc.
        **kwargs : dict
            Additional arguments passed to the optimizer.

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