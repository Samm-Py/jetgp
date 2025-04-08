import numpy as np
from numpy.linalg import cholesky, solve
from pyswarm import pso
from wdegp import wdegp_utils as utils


class Optimizer:
    def __init__(self, model):
        """
        model: an instance of wdegp
        """
        self.model = model

    def negative_log_marginal_likelihood(
        self,
        x0,
        x_train,
        y_train,
        sigma_n,
        n_order,
        n_bases,
        der_indices,
        index,
    ):
        """
        NLL = 0.5 * y^T (K^-1) y + 0.5 * log|K| + 0.5*N*log(2*pi)
        """
        ell = x0[:-1]
        sigma_n = x0[-1]
        llhood = 0

        for i in range(len(index)):
            y_train_sub = y_train[i]
            der_indices_sub = self.model.flattened_der_indicies[i]
            powers = self.model.powers[i]
            idx = index[i]
            diffs = self.model.differences_by_dim_submodels[i]

            K = utils.rbf_kernel(
                diffs, ell, n_order, n_bases, self.model.kernel_func,
                der_indices_sub, powers, index=idx
            )
            K += (10 ** sigma_n) ** 2 * np.eye(len(K))

            try:
                L = cholesky(K)
                alpha = solve(L.T, solve(L, y_train_sub))

                data_fit = 0.5 * np.dot(y_train_sub, alpha)
                log_det = np.sum(np.log(np.diag(L)))
                const = 0.5 * len(y_train_sub) * np.log(2 * np.pi)

                llhood += data_fit + log_det + const
            except np.linalg.LinAlgError:
                llhood += 1e6  # Penalize badly conditioned matrices

        return llhood

    def nll_wrapper(self, x0):
        return self.negative_log_marginal_likelihood(
            x0,
            self.model.x_train,
            self.model.y_train,
            self.model.sigma_n,
            self.model.n_order,
            self.model.n_bases,
            self.model.der_indices,
            self.model.index,
        )

    def optimize_hyperparameters(self, n_restart_optimizer=20, swarm_size=20):
        bounds = self.model.bounds
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]

        # Run PSO to minimize the NLL
        best_x, best_nll = pso(
            self.nll_wrapper,
            lb,
            ub,
            swarmsize=swarm_size,
            maxiter=n_restart_optimizer,
            debug=False,
        )

        # Optionally: store results
        self.opt_x0 = best_x
        self.opt_nll = best_nll

        print("Best solution:", best_x)
        print("Objective value:", best_nll)
        return best_x
