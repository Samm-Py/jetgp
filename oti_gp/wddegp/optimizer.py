import numpy as np
from numpy.linalg import cholesky, solve
from pyswarm import pso
from wddegp import wddegp_utils as utils


class Optimizer:
    """
    Optimizer class for Weighted Directional Derivative-Enhanced Gaussian Process (wdDEGP).

    This optimizer uses Particle Swarm Optimization (PSO) to minimize the negative log marginal likelihood (NLL)
    across submodels of the wdDEGP framework. Each submodel corresponds to a subset of data associated with
    directional derivative observations and uses a dedicated radial basis kernel to model correlations.

    Parameters
    ----------
    model : object
        An instance of the `wddegp` model containing submodel structures, training data,
        kernel functions, and noise matrices.
    """

    def __init__(self, model):
        self.model = model

    def negative_log_marginal_likelihood(self, x0, x_train, y_train, sigma_n,
                                         n_order, n_bases, der_indices, index):
        """
        Compute the total negative log marginal likelihood (NLL) across all wdDEGP submodels.

        NLL = 0.5 * y^T K^-1 y + 0.5 * log|K| + 0.5 * N * log(2 * pi)

        Parameters
        ----------
        x0 : ndarray
            Log-scaled hyperparameters including kernel parameters and noise.
        x_train : ndarray
            Training input data.
        y_train : list of ndarray
            List of outputs for each submodel.
        sigma_n : float or array-like
            Observation noise vector or matrix.
        n_order : int
            Maximum derivative order.
        n_bases : int
            Number of OTI basis terms.
        der_indices : list of lists
            Directional derivative indices per submodel.
        index : list of lists
            Index mapping of training points to submodels.

        Returns
        -------
        float
            Total NLL for all submodels.
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
                der_indices_sub, powers, index=idx, index_list=idx
            )
            K += (10 ** sigma_n) ** 2 * np.eye(len(K))
            K += self.model.sigma_data[i] ** 2

            try:
                L = cholesky(K)
                alpha = solve(L.T, solve(L, y_train_sub))

                data_fit = 0.5 * np.dot(y_train_sub, alpha)
                log_det = np.sum(np.log(np.diag(L)))
                const = 0.5 * len(y_train_sub) * np.log(2 * np.pi)

                llhood += data_fit + log_det + const
            except np.linalg.LinAlgError:
                llhood += 1e6  # Penalize poorly conditioned matrices

        return llhood

    def nll_wrapper(self, x0):
        """
        Wrapper for negative_log_marginal_likelihood using internal model state.

        Parameters
        ----------
        x0 : ndarray
            Vector of hyperparameters.

        Returns
        -------
        float
            NLL value.
        """
        return self.negative_log_marginal_likelihood(
            x0,
            self.model.x_train,
            self.model.y_train,
            self.model.sigma_data,
            self.model.n_order,
            self.model.n_bases,
            self.model.der_indices,
            self.model.index,
        )

    def optimize_hyperparameters(self, n_restart_optimizer=20, swarm_size=20, verbose=True):
        """
        Optimize kernel hyperparameters for wdDEGP using Particle Swarm Optimization (PSO).

        Parameters
        ----------
        n_restart_optimizer : int
            Number of PSO iterations.
        swarm_size : int
            Number of particles in the swarm.

        Returns
        -------
        best_x : ndarray
            Optimal hyperparameter vector.
        """
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
        )

        self.opt_x0 = best_x
        self.opt_nll = best_nll

        print("Best solution:", best_x)
        print("Objective value:", best_nll)
        return best_x
