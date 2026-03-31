import numpy as np
from numpy.linalg import cholesky, solve
from jetgp.full_gddegp import gddegp_utils as utils
import jetgp.utils as gen_utils
from scipy.linalg import cho_solve, cho_factor
from jetgp.hyperparameter_optimizers import OPTIMIZERS
from line_profiler import profile
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
            phi_exp = phi_exp[np.newaxis,:,:]
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
            index = self.model.derivative_locations,
        )
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
        if self.model.n_order == 0:
            n_bases = 0
            phi_exp = phi.real[np.newaxis, :, :]
        else:
            n_bases = phi.get_active_bases()[-1]
            phi_exp = phi.get_all_derivs(n_bases, 2 * self.model.n_order)

        K = utils.rbf_kernel(
            phi, phi_exp,
            self.model.n_order, n_bases,
            self.model.flattened_der_indices,
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

        grad = np.zeros(len(x0))

        def _assemble_dK(dphi_oti):
            if self.model.n_order == 0:
                dphi_exp = dphi_oti.real[np.newaxis, :, :]
            else:
                dphi_exp = dphi_oti.get_all_derivs(n_bases, 2 * self.model.n_order)
            return utils.rbf_kernel(
                dphi_oti, dphi_exp,
                self.model.n_order, n_bases,
                self.model.flattened_der_indices,
                index=self.model.derivative_locations,
            )

        def _gc(dphi):
            return 0.5 * np.sum(W * _assemble_dK(dphi))

        grad[-2] = _gc(oti.mul(2.0 * ln10, phi))
        grad[-1] = ln10 * sigma_n_sq * np.trace(W)

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
        """Return (NLL, gradient) in one call for use with func_and_grad interface."""
        return self.nll_wrapper(x0), self.nll_grad(x0)

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
