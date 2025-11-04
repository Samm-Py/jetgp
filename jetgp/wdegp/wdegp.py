import numpy as np
from numpy.linalg import cholesky, solve
from scipy.linalg import cho_solve, cho_factor, solve_triangular
from jetgp.wdegp import wdegp_utils as wdegp_utils
import jetgp.utils as utils
from jetgp.kernel_funcs.kernel_funcs import KernelFactory
from jetgp.wdegp.optimizer import Optimizer


class wdegp:
    """
    Weighted Derivative-Enhanced Gaussian Process (wDEGP) regression model.

    Supports multiple submodels indexed by regions of the training data,
    includes normalization and automatic hyperparameter optimization.

    Parameters
    ----------
    x_train : ndarray of shape (n_samples, n_features)
        Input training points.
    y_train : list of arrays
        Each element contains function or derivative observations for a submodel.
    n_order : int
        Maximum derivative order to be supported.
    n_bases : int
        Number of OTI basis terms used.
    index : list of lists
        Submodel training data indices.
    der_indices : list of lists
        Multi-indices of derivatives for each submodel.
    normalize : bool, default=True
        If True, normalizes the input and output data.
    sigma_data : float or ndarray, optional
        Known observation noise or covariance matrix.
    kernel : str, default='SE'
        Type of kernel to use: 'SE', 'RQ', 'Matern', or 'SineExp'.
    kernel_type : str, default='anisotropic'
        Whether kernel is 'anisotropic' or 'isotropic'.
    """

    def __init__(self, x_train, y_train, n_order, n_bases, index, der_indices,
                 normalize=True, sigma_data=None, kernel="SE", kernel_type="anisotropic", smoothness_parameter = None):
        self.x_train = x_train
        self.y_train = y_train
        self.n_order = n_order
        self.n_bases = n_bases
        self.index = index
        self.num_points = len(x_train)

        self.der_indices = der_indices
        self.dim = x_train.shape[1]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.normalize = normalize

        self.flattened_der_indicies = []
        self.powers = []
        base_der_indices = utils.gen_OTI_indices(n_bases, n_order)
        flattened_base_der_indicies = [
            i for sublist in base_der_indices for i in sublist]
        if sigma_data is None:
            sigma_data = np.zeros((len(flattened_base_der_indicies)+1)
                                  * self.num_points)
        for ders in der_indices:
            self.powers.append(
                utils.build_companion_array(n_bases, n_order, ders))
            flat_indices = [i for sublist in ders for i in sublist]
            self.flattened_der_indicies.append(flat_indices)

        if normalize:
            self.y_train = []
            for k, ders in enumerate(self.der_indices):
                y_norm, self.mu_y, self.sigma_y, self.sigmas_x, self.mus_x, self.sigma_data = utils.normalize_y_data(
                    x_train, y_train[k],
                    sigma_data, self.flattened_der_indicies[k]
                )
                self.y_train.append(y_norm)
            self.x_train = utils.normalize_x_data_train(x_train)
        else:
            self.y_train = [utils.reshape_y_train(y) for y in y_train]
            self.x_train = x_train
            self.sigma_data = sigma_data

        self.differences_by_dim = wdegp_utils.differences_by_dim_func(
            self.x_train, self.x_train, self.n_order)

        self.kernel_factory = KernelFactory(
            dim=self.dim,
            normalize=self.normalize,
            n_order=self.n_order,
            differences_by_dim=self.differences_by_dim,
            smoothness_parameter=smoothness_parameter
        )
        self.kernel_func = self.kernel_factory.create_kernel(
            kernel_name=self.kernel,
            kernel_type=self.kernel_type,
        )
        self.sigma_data = np.diag(sigma_data)
        self.bounds = self.kernel_factory.bounds
        self.optimizer = Optimizer(self)

        self.sigma_data = utils.generate_submodel_noise_matricies(
            self.sigma_data, index, self.flattened_der_indicies, self.num_points, flattened_base_der_indicies)

    def optimize_hyperparameters(self, *args, **kwargs):
        """
        Optimize hyperparameters (e.g., length scales and noise) via Particle Swarm Optimization (PSO).

        Returns
        -------
        ndarray
            Optimized hyperparameter vector.
        """
        return self.optimizer.optimize_hyperparameters(*args, **kwargs)

    def predict(self, X_test, length_scales, calc_cov=False, return_submodels=False):
        """
        Compute posterior predictive mean and (optionally) covariance at test points.

        Parameters
        ----------
        X_test : ndarray of shape (n_test, n_features)
            Test input points.
        length_scales : ndarray
            Log-scaled kernel hyperparameters including noise level.
        calc_cov : bool, default=False
            If True, also compute and return predictive covariance.
        return_submodels : bool, default=False
            If True, return submodel-specific contributions.

        Returns
        -------
        y_val : ndarray of shape (n_test,)
            Predicted mean function values at the test inputs.
        y_var : ndarray of shape (n_test,), optional
            Predictive variances at the test inputs (only if calc_cov=True).
        submodel_vals : list of ndarrays, optional
            List of submodel predictive means (only if return_submodels=True).
        submodel_cov : list of ndarrays, optional
            List of submodel variances (only if calc_cov and return_submodels are True).
        """
        ell = length_scales[:-1]
        sigma_n = length_scales[-1]
        n_test = X_test.shape[0]
        n_train = self.x_train.shape[0]

        if self.normalize:
            X_test = utils.normalize_x_data_test(
                X_test, self.sigmas_x, self.mus_x)



        y_val = 0
        y_var = 0
        submodel_vals = []
        submodel_cov = []

        if len(self.index) == 1:
            i = 0
            index_i = self.index[i]
            diffs_train_test = wdegp_utils.differences_by_dim_func(
                self.x_train, X_test, self.n_order)
            diffs_train_train = self.differences_by_dim

            phi_train_train = self.kernel_func(diffs_train_train, ell)

            # Extract ALL derivative components into a single flat array (highly efficient)
            phi_exp_train_train = phi_train_train.get_all_derivs(
                self.n_bases, 2 * self.n_order)

            K = wdegp_utils.rbf_kernel(
                phi_train_train, phi_exp_train_train,  self.n_order, self.n_bases,
                self.flattened_der_indicies[i], self.powers[i], index=index_i
            )

            K += (10 ** sigma_n) ** 2 * np.eye(len(K))
            K += self.sigma_data[i]**2
            # L = cholesky(K)
            # alpha = solve(L.T, solve(L, self.y_train[i]))
            # print(K)
            try:
                cho_solve_failed = False
                L, low = cho_factor(K, lower=True)
                alpha = cho_solve(
                    (L, low),
                    self.y_train[i]
                )
            except:
                cho_solve_failed = True
                L = cholesky(K)
                alpha = np.linalg.solve(K, self.y_train[i])
                print(
                    'Warning: Cholesky decomposition failed via scipy, using standard np solve instead.')
            # If Cholesky fails, fall back to standard solve
            phi_train_test = self.kernel_func(diffs_train_test, ell)

            # Extract ALL derivative components into a single flat array (highly efficient)
            phi_exp_train_test = phi_train_test.get_all_derivs(
                self.n_bases, 2 * self.n_order)
            K_s = wdegp_utils.rbf_kernel(
                phi_train_test, phi_exp_train_test, self.n_order, self.n_bases,
                self.flattened_der_indicies[i], self.powers[i], index=index_i
            )
            f_mean = K_s[:, :n_test].T @ alpha
            if self.normalize:
                f_mean = self.mu_y + f_mean * self.sigma_y

            y_val += f_mean
            if return_submodels:
                submodel_vals.append(f_mean)

            if calc_cov:
                diffs_test_test = wdegp_utils.differences_by_dim_func(
                    X_test, X_test, self.n_order)
                # If Cholesky fails, fall back to standard solve
                phi_test_test = self.kernel_func(diffs_test_test, ell)

                # Extract ALL derivative components into a single flat array (highly efficient)
                phi_exp_test_test = phi_test_test.get_all_derivs(
                    self.n_bases, 2 * self.n_order)
                K_ss = wdegp_utils.rbf_kernel(
                    phi_test_test, phi_exp_test_test, self.n_order, self.n_bases,
                    self.flattened_der_indicies[i], self.powers[i], index=index_i
                )
                # v = solve(L, K_s[:, :n_test])
                if cho_solve_failed:
                    f_cov = (K_ss[:len(X_test), :len(X_test)] - K_s[:, :len(X_test)].T @ np.linalg.solve(K, K_s[:, :len(X_test)])
                             )
                else:
                    v = solve_triangular(L, K_s, lower=low)
                    f_cov = (K_ss[:len(X_test), :len(X_test)] - v[:, :len(X_test)].T @ v[:, :len(X_test)]
                             )

                if self.normalize:
                    f_var = utils.transform_cov(f_cov, self.sigma_y, self.sigmas_x,
                                                self.flattened_der_indicies[i], X_test)
                else:
                    f_var = np.diag(np.abs(f_cov))

                y_var += np.sqrt(f_var)
                if return_submodels:
                    submodel_cov.append(f_var)

            if return_submodels:
                return (y_val, y_var**2, submodel_vals, submodel_cov) if calc_cov else (y_val, submodel_vals)
            else:
                return (y_val, y_var**2) if calc_cov else y_val
        else:
            diffs_train_train = self.differences_by_dim
            diffs_train_test = wdegp_utils.differences_by_dim_func(
                X_test, self.x_train, 0, index=[-1])
            weights_matrix = wdegp_utils.determine_weights(
                self.differences_by_dim, diffs_train_test, ell, self.kernel_func, sigma_n)
                
            diffs_train_test = wdegp_utils.differences_by_dim_func(
                self.x_train, X_test, self.n_order)
            

            phi_train_train = self.kernel_func(
                diffs_train_train, ell)

            # Extract ALL derivative components into a single flat array (highly efficient)
            phi_exp_train_train = phi_train_train.get_all_derivs(
                self.n_bases, 2 * self.n_order)
            
            for i in range(len(self.index)):
                index_i = self.index[i]
                

                K = wdegp_utils.rbf_kernel(
                    phi_train_train, phi_exp_train_train, self.n_order, self.n_bases,
                    self.flattened_der_indicies[i], self.powers[i], index=index_i
                )

                K += (10 ** sigma_n) ** 2 * np.eye(len(K))
                K += self.sigma_data[i]**2
                # L = cholesky(K)
                # alpha = solve(L.T, solve(L, self.y_train[i]))
                # print(K)
                try:
                    cho_solve_failed = False
                    L, low = cho_factor(K, lower=True)
                    alpha = cho_solve(
                        (L, low),
                        self.y_train[i]
                    )
                except:
                    cho_solve_failed = True
                    alpha = np.linalg.solve(K, self.y_train[i])
                    print(
                        'Warning: Cholesky decomposition failed via scipy, using standard np solve instead.')
                # If Cholesky fails, fall back to standard solve

                 # If Cholesky fails, fall back to standard solve
                phi_train_test = self.kernel_func(diffs_train_test, ell)

                # Extract ALL derivative components into a single flat array (highly efficient)
                phi_exp_train_test = phi_train_test.get_all_derivs(
                    self.n_bases, 2 * self.n_order)
                K_s = wdegp_utils.rbf_kernel(
                    phi_train_test, phi_exp_train_test, self.n_order, self.n_bases,
                    self.flattened_der_indicies[i], self.powers[i], index=index_i
                )
                f_mean = K_s[:, :n_test].T @ alpha
                if self.normalize:
                    f_mean = self.mu_y + f_mean * self.sigma_y
                weight = 0
                for j in range(len(self.index[i])):
                    weight = weight + weights_matrix[:, self.index[i][j]]
                # for j in range(len(self.index[i])):
                #     y_val += weights_matrix[:, self.index[i][j]] * f_mean
                
                # if i == 0:
                #     weight = np.array([1,1,1,1,1,0,0,0,0,0])
                # else:
                #     weight = np.array([0,0,0,0,0, 1,1,1,1,1])
                    
                    
                y_val += weight * f_mean
                if return_submodels:
                    submodel_vals.append(f_mean)

                if calc_cov:
                    diffs_test_test = wdegp_utils.differences_by_dim_func(
                        X_test, X_test, self.n_order)
                    # If Cholesky fails, fall back to standard solve
                    phi_test_test = self.kernel_func(
                        diffs_test_test, ell)

                    # Extract ALL derivative components into a single flat array (highly efficient)
                    phi_exp_test_test = phi_test_test.get_all_derivs(
                        self.n_bases, 2 * self.n_order)
                    K_ss = wdegp_utils.rbf_kernel(
                        phi_test_test, phi_exp_test_test, self.n_order, self.n_bases,
                        self.flattened_der_indicies[i], self.powers[i], index=index_i
                    )
                    # v = solve(L, K_s[:, :n_test])
                    if cho_solve_failed:
                        f_cov = (K_ss[:len(X_test), :len(X_test)] - K_s[:, :len(X_test)].T @ np.linalg.solve(K, K_s[:, :len(X_test)])
                                 )
                    else:
                        v = solve_triangular(L, K_s, lower=low)
                        f_cov = (K_ss[:len(X_test), :len(X_test)] - v[:, :len(X_test)].T @ v[:, :len(X_test)]
                                 )

                    if self.normalize:
                        f_var = utils.transform_cov(f_cov, self.sigma_y, self.sigmas_x,
                                                    self.flattened_der_indicies[i], X_test)
                    else:
                        f_var = np.diag(np.abs(f_cov))

                    for j in range(len(self.index[i])):
                        y_var += weights_matrix[:,
                                                self.index[i][j]] * np.sqrt(f_var)
                    if return_submodels:
                        submodel_cov.append(f_var)

            if return_submodels:
                return (y_val, y_var**2, submodel_vals, submodel_cov) if calc_cov else (y_val, submodel_vals)
            else:
                return (y_val, y_var**2) if calc_cov else y_val
