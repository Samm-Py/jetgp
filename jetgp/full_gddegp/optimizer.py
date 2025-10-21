import numpy as np
from numpy.linalg import cholesky, solve
from full_gddegp import gddegp_utils as utils
import utils as gen_utils
from scipy.linalg import cho_solve, cho_factor
from hyperparameter_optimizers import OPTIMIZERS

class Optimizer:
    def __init__(self, model):
        """
        Initialize the optimizer with a DEGP model instance.

        Parameters:
        ----------
        model : object
            An instance of a derivative-enhanced Gaussian process (DEGP) model.
        """
        self.model = model

    def negative_log_marginal_likelihood(self, x0):
        """
        Compute the Negative Log Marginal Likelihood (NLL) for the DEGP model.

        The NLL formula:
        NLL = 0.5 * y^T (K^-1) y + 0.5 * log|K| + 0.5 * N * log(2 * pi)

        Parameters:
        ----------
        x0 : ndarray
            Hyperparameters, where:
            - x0[:-1] are the length scales (ell).
            - x0[-1] is the log noise standard deviation (sigma_n).

        Returns:
        -------
        float
            The computed NLL value. Returns a large value (1e6) if Cholesky fails.
        """
        ell = x0[:-1]
        sigma_n = x0[-1]

        # Compute kernel matrix with current hyperparameters
        K = utils.rbf_kernel(
            self.model.differences_by_dim,
            ell,
            self.model.n_order,
            self.model.kernel_func,
            # self.model.flattened_der_indices,
            # self.model.der_indices_tr,
            # self.model.der_ind_order,
            # self.model.der_map,
        )

        # Add noise terms
        K += (10**sigma_n)**2 * np.eye(K.shape[0])
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
        Wrapper for the negative log marginal likelihood function.

        Parameters:
        ----------
        x0 : ndarray
            Hyperparameters.

        Returns:
        -------
        float
            The NLL value.
        """
        return self.negative_log_marginal_likelihood(x0)

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

        best_x, best_val = optimizer_fn(self.nll_wrapper, lb, ub, **kwargs)

        self.model.opt_x0 = best_x
        self.model.opt_nll = best_val


        return best_x
