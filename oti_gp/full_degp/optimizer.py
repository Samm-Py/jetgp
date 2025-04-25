import numpy as np
from numpy.linalg import cholesky, solve
from pyswarm import pso
from full_degp import degp_utils as utils


class Optimizer:
    def __init__(self, model):
        """
        model: an instance of degp
        """
        self.model = model

    def negative_log_marginal_likelihood(self, x0):
        """
        NLL for standard GP in multiple dimensions.
        NLL = 0.5 * y^T (K^-1) y + 0.5 * log|K| + 0.5*N*log(2*pi).
        """
        ell = x0[:-1]
        sigma_n = x0[-1]
        K = utils.rbf_kernel(
            self.model.differences_by_dim,
            ell,
            self.model.n_order,
            self.model.n_bases,
            self.model.kernel_func,
            self.model.flattened_der_indicies,
            self.model.powers
        )
        K += ((10 ** sigma_n) ** 2) * np.eye(len(K))

        try:
            L = cholesky(K)
            alpha = solve(L.T, solve(L, self.model.y_train))

            data_fit = 0.5 * np.dot(self.model.y_train, alpha)
            log_det_K = np.sum(np.log(np.diag(L)))
            complexity = log_det_K
            N = len(self.model.y_train)
            const = 0.5 * N * np.log(2 * np.pi)
            return data_fit + complexity + const
        except Exception:
            return 1e6

    def nll_wrapper(self, x0):
        return self.negative_log_marginal_likelihood(x0)

    def optimize_hyperparameters(self, n_restart_optimizer=20, swarm_size=20, verbose=True):
        bounds = self.model.bounds
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]

        best_x, best_nll = pso(
            self.nll_wrapper,
            lb,
            ub,
            swarmsize=swarm_size,
            maxiter=n_restart_optimizer,
            debug=verbose,
            minfunc=1e-20,
        )

        self.model.opt_x0 = best_x
        self.model.opt_nll = best_nll

        print("Best solution:", best_x)
        print("Objective value:", best_nll)

        return best_x
