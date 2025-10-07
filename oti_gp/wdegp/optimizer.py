import numpy as np
from scipy.linalg import cho_solve, cho_factor
from wdegp import wdegp_utils as utils
from line_profiler import profile
import utils as gen_utils


class Optimizer:
    """
    Optimizer class for fitting the hyperparameters of a weighted derivative-enhanced GP model (wDEGP)
    by minimizing the negative log marginal likelihood (NLL).

    Attributes
    ----------
    model : object
        Instance of a weighted derivative-enhanced GP model (wDEGP) with attributes:
        x_train, y_train, n_order, n_bases, der_indices, index, bounds, etc.
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
            Indices partitioning the training data into submodels.

        Returns
        -------
        float
            The computed negative log marginal likelihood.
        """
        ell = x0[:-1]
        sigma_n = x0[-1]
        llhood = 0

        diffs = self.model.differences_by_dim
        phi = self.model.kernel_func(diffs, ell, index)

        # Extract ALL derivative components into a single flat array (highly efficient)
        phi_exp = phi.get_all_derivs(n_bases, 2 * n_order)
        for i in range(len(index)):
            y_train_sub = y_train[i]
            der_indices_sub = self.model.flattened_der_indicies[i]
            powers = self.model.powers[i]
            idx = index[i]

            K = utils.rbf_kernel(
                phi, phi_exp, n_order, n_bases,
                der_indices_sub, powers, index=idx
            )
            K += (10 ** sigma_n) ** 2 * np.eye(len(K))
            K += self.model.sigma_data[i]**2

            try:
                L, low = cho_factor(K)
                alpha = cho_solve(
                    (L, low),
                    y_train_sub
                )

                data_fit = 0.5 * np.dot(y_train_sub, alpha)
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
            self.model.y_train,
            self.model.n_order,
            self.model.n_bases,
            self.model.der_indices,
            self.model.index,
        )

    def optimize_hyperparameters(self, n_restart_optimizer=20, swarm_size=20, verbose=True):
        """
        Optimizes kernel hyperparameters using Particle Swarm Optimization (PSO)
        to minimize the negative log marginal likelihood.

        Parameters
        ----------
        n_restart_optimizer : int
            Number of PSO iterations.
        swarm_size : int
            Number of particles in the swarm.
        verbose : bool
            If True, enables debug mode in PSO for progress printing.

        Returns
        -------
        best_x : ndarray
            Optimized hyperparameter vector.
        """
        bounds = self.model.bounds
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]

        best_x, best_nll = gen_utils.pso(
            self.nll_wrapper,
            lb,
            ub,
            pop_size=swarm_size,
            maxiter=n_restart_optimizer,
            debug=verbose,
        )

        self.opt_x0 = best_x
        self.opt_nll = best_nll

        print("Best solution:", best_x)
        print("Objective value:", best_nll)
        return best_x
