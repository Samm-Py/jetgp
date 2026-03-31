import numpy as np
from scipy.linalg import cho_solve, cho_factor
from jetgp.full_degp import degp_utils as utils
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
        if self.model.n_order == 0:
            n_bases = 0
            phi_exp = phi.real
            phi_exp = phi_exp[np.newaxis, :, :]
        else:
            n_bases = phi.get_active_bases()[-1]
        
        # Extract ALL derivative components into a single flat array (highly efficient)
            phi_exp = phi.get_all_derivs(n_bases, 2 * self.model.n_order)
        K = utils.rbf_kernel(
            phi,
            phi_exp,
            self.model.n_order,
            n_bases,
            self.model.flattened_der_indices,
            self.model.powers,
            index = self.model.derivative_locations,
        )
        noise_var = (10 ** sigma_n) ** 2
        K.flat[::K.shape[0] + 1] += noise_var
        K += self.model.sigma_data**2
        
        # Debug: check kernel matrix sparsity
        # near_zero = np.sum(np.abs(K) < 1e-10)
        # total = K.size
        # sparsity = near_zero / total
        # if sparsity > 0.5:
        #     print(f"  WARNING: K is {sparsity*100:.1f}% sparse | ell={10**np.array(ell)} | sigma_n={10**sigma_n:.2e} | cond={np.linalg.cond(K.real):.2e}")
        #     input('i')
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

        def _assemble_dK(dphi_oti):
            if self.model.n_order == 0:
                dphi_exp = dphi_oti.real[np.newaxis, :, :]
            else:
                dphi_exp = dphi_oti.get_all_derivs(n_bases, 2 * self.model.n_order)
            return utils.rbf_kernel(
                dphi_oti, dphi_exp,
                self.model.n_order, n_bases,
                self.model.flattened_der_indices, self.model.powers,
                index=self.model.derivative_locations,
            )

        def _gc(dphi):
            return 0.5 * np.sum(W * _assemble_dK(dphi))

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
                for d in range(D):
                    d_sq   = oti.mul(diffs[d], diffs[d])
                    dphi_d = oti.mul(-ln10 * ell[d] ** 2, oti.mul(d_sq, phi))
                    grad[d] = _gc(dphi_d)
            else:  # isotropic: single ell
                ell    = 10.0 ** float(x0[0])
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
            r2 = oti.mul(ell[0], diffs[0])
            r2 = oti.mul(r2, r2)
            for d in range(1, D):
                td = oti.mul(ell[d], diffs[d])
                r2 = oti.sum(r2, oti.mul(td, td))
            base     = oti.sum(1.0, oti.mul(r2, 1.0 / (2.0 * alpha_rq)))
            inv_base = oti.pow(base, -1)
            phi_over_base = oti.mul(phi, inv_base)

            if kernel_type == 'anisotropic':
                for d in range(D):
                    d_sq   = oti.mul(diffs[d], diffs[d])
                    dphi_d = oti.mul(-ln10 * ell[d] ** 2, oti.mul(d_sq, phi_over_base))
                    grad[d] = _gc(dphi_d)
            else:
                sum_sq = oti.mul(diffs[0], diffs[0])
                for d in range(1, D):
                    sum_sq = oti.sum(sum_sq, oti.mul(diffs[d], diffs[d]))
                grad[0] = _gc(oti.mul(-ln10 * ell[0] ** 2, oti.mul(sum_sq, phi_over_base)))

            # alpha gradient: phi * α_factor * [-log(base) + (1 - 1/base)]
            # aniso: alpha = 10^x  → d alpha/dx = ln10 * alpha
            # iso:   alpha = exp(x) → d alpha/dx = alpha
            log_base = oti.log(base)
            term = oti.sum(oti.mul(-1.0, log_base),
                           oti.sum(1.0, oti.mul(-1.0, inv_base)))
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

            # Length-scale gradients
            if kernel_type == 'anisotropic':
                for d in range(D):
                    sin_sq = oti.mul(sin_d[d], sin_d[d])
                    grad[d] = _gc(oti.mul(-4.0 * ln10 * ell[d] ** 2,
                                          oti.mul(sin_sq, phi)))
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
            r_oti = oti.sqrt(
                sum((ell[d] * diffs[d]) ** 2 for d in range(D)) + _eps ** 2
            )
            f_prime_r = kf._matern_grad_prebuild(r_oti)   # df/dr (OTI)
            inv_r     = oti.pow(r_oti, -1)

            # dr/d(log_ell_d) = ln10 * ell_d^2 * diff_d^2 / r
            # (no eps added to diff — r is regularised via r_oti above)
            if kernel_type == 'anisotropic':
                for d in range(D):
                    d_sq   = oti.mul(diffs[d], diffs[d])
                    dr_d   = oti.mul(ln10 * ell[d] ** 2, oti.mul(d_sq, inv_r))
                    dphi_d = oti.mul(sigma_f_sq, oti.mul(f_prime_r, dr_d))
                    grad[d] = _gc(dphi_d)
            else:
                ell_val    = ell[0]
                sum_dsq    = oti.mul(diffs[0], diffs[0])
                for d in range(1, D):
                    sum_dsq = oti.sum(sum_dsq, oti.mul(diffs[d], diffs[d]))
                dr_d   = oti.mul(ln10 * ell_val ** 2, oti.mul(sum_dsq, inv_r))
                dphi_e = oti.mul(sigma_f_sq, oti.mul(f_prime_r, dr_d))
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

    def nll_grad(self, x0):
        """Analytic gradient of the NLL (separate Cholesky from nll_wrapper)."""
        diffs = self.model.differences_by_dim
        oti   = self.model.kernel_factory.oti
        sigma_n_sq = (10.0 ** x0[-1]) ** 2

        phi = self.model.kernel_func(diffs, x0[:-1])
        if self.model.n_order == 0:
            n_bases = 0
            phi_exp = phi.real[np.newaxis, :, :]
        else:
            n_bases = phi.get_active_bases()[-1]
            phi_exp = phi.get_all_derivs(n_bases, 2 * self.model.n_order)

        K = utils.rbf_kernel(
            phi, phi_exp,
            self.model.n_order, n_bases,
            self.model.flattened_der_indices, self.model.powers,
            index=self.model.derivative_locations,
        )
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

        return self._compute_grad(x0, W, phi, n_bases, oti, diffs)

    def nll_and_grad(self, x0):
        """
        Compute NLL and its gradient in a single pass, sharing one Cholesky.

        Returns
        -------
        nll : float
        grad : ndarray
        """
        diffs = self.model.differences_by_dim
        oti   = self.model.kernel_factory.oti
        sigma_n_sq = (10.0 ** x0[-1]) ** 2

        phi = self.model.kernel_func(diffs, x0[:-1])
        if self.model.n_order == 0:
            n_bases = 0
            phi_exp = phi.real[np.newaxis, :, :]
        else:
            n_bases = phi.get_active_bases()[-1]
            phi_exp = phi.get_all_derivs(n_bases, 2 * self.model.n_order)

        K = utils.rbf_kernel(
            phi, phi_exp,
            self.model.n_order, n_bases,
            self.model.flattened_der_indices, self.model.powers,
            index=self.model.derivative_locations,
        )
        K.flat[::K.shape[0] + 1] += sigma_n_sq
        K += self.model.sigma_data ** 2

        try:
            L, low  = cho_factor(K)
            alpha_v = cho_solve((L, low), self.model.y_train)
            N       = len(self.model.y_train)

            nll = (0.5 * np.dot(self.model.y_train, alpha_v)
                   + np.sum(np.log(np.diag(L)))
                   + 0.5 * N * np.log(2 * np.pi))

            K_inv = cho_solve((L, low), np.eye(N))
            W     = K_inv - np.outer(alpha_v, alpha_v)
        except Exception:
            return 1e6, np.zeros(len(x0))

        grad = self._compute_grad(x0, W, phi, n_bases, oti, diffs)
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
