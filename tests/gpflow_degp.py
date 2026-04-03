"""
GPflow derivative-enhanced SE kernel for first-order gradient observations.

Implements the augmented covariance matrix with analytical block formulas
for the SE kernel. Input data is augmented with a task indicator column
(0 = function value, j = df/dx_j).
"""

import numpy as np
import time
import tensorflow as tf
import gpflow
from gpflow.utilities import positive


class DerivativeSEKernel(gpflow.kernels.Kernel):
    """
    SE kernel augmented with first-order derivative observations.

    Builds the full (1+d)*n x (1+d)*n covariance matrix using analytical
    derivative formulas for the squared exponential kernel.

    Input X has shape (n_total, d+1) where:
        - columns 0..d-1 are spatial coordinates
        - column d is a task indicator: 0 = function value, j = df/dx_j

    Analytical block formulas for SE kernel k(x, x'):
        K(f, f)       = k(x, x')
        K(f, df/dx_j) = k(x, x') * (-(x_j - x'_j) / l_j^2)
        K(df/dx_i, f) = k(x, x') * ((x_i - x'_i) / l_i^2)
        K(df/dx_i, df/dx_j) = k(x, x') * (delta_ij/l_i^2
                               - (x_i - x'_i)(x_j - x'_j) / (l_i^2 * l_j^2))
    """

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = dim
        self.variance = gpflow.Parameter(1.0, transform=positive())
        self.lengthscales = gpflow.Parameter(
            np.ones(dim), transform=positive()
        )

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        d = self.input_dim
        X1_locs = X[:, :d]
        X1_tasks = tf.cast(X[:, d], tf.int32)
        X2_locs = X2[:, :d]
        X2_tasks = tf.cast(X2[:, d], tf.int32)

        # Pairwise differences: (n1, n2, d)
        diff = tf.expand_dims(X1_locs, 1) - tf.expand_dims(X2_locs, 0)
        ls2 = self.lengthscales ** 2  # (d,)

        # Base SE kernel: (n1, n2)
        scaled = diff / self.lengthscales
        Kff = self.variance * tf.exp(
            -0.5 * tf.reduce_sum(scaled ** 2, axis=-1)
        )

        # One-hot task indicators: (n1, d+1) and (n2, d+1)
        ti = tf.one_hot(X1_tasks, d + 1, dtype=X.dtype)
        tj = tf.one_hot(X2_tasks, d + 1, dtype=X.dtype)

        # Derivative task indicators (exclude task 0): (n1, d) and (n2, d)
        ti_d = ti[:, 1:]   # (n1, d)
        tj_d = tj[:, 1:]   # (n2, d)

        is_f1 = ti[:, 0]   # (n1,)
        is_f2 = tj[:, 0]   # (n2,)

        # Scaled differences: diff_k / ls2_k -> (n1, n2, d)
        scaled_diff = diff / ls2  # (n1, n2, d)

        # === ff block ===
        ff_mask = tf.expand_dims(is_f1, 1) * tf.expand_dims(is_f2, 0)  # (n1, n2)
        K = Kff * ff_mask

        # === f-dk block (vectorized over k) ===
        # K(f, df/dx_k) = Kff * scaled_diff_k, masked by is_f1 and is_dk2
        # Sum over k: sum_k [Kff * scaled_diff_k * is_f1_i * is_dk2_j]
        # = Kff * is_f1_i * sum_k [scaled_diff_k * is_dk2_j]
        # scaled_diff: (n1, n2, d), tj_d: (n2, d) -> (1, n2, d)
        fd_contrib = tf.reduce_sum(
            scaled_diff * tf.expand_dims(tj_d, 0), axis=-1
        )  # (n1, n2)
        K += Kff * fd_contrib * tf.expand_dims(is_f1, 1)

        # === dk-f block (vectorized over k) ===
        # K(df/dx_k, f) = -Kff * scaled_diff_k, masked by is_dk1 and is_f2
        # ti_d: (n1, d) -> (n1, 1, d)
        df_contrib = tf.reduce_sum(
            scaled_diff * tf.expand_dims(ti_d, 1), axis=-1
        )  # (n1, n2)
        K += Kff * (-df_contrib) * tf.expand_dims(is_f2, 0)

        # === dk-dl block (vectorized over k and l) ===
        # K(df/dx_k, df/dx_l) = Kff * (delta_kl/ls2_k - diff_k*diff_l/(ls2_k*ls2_l))
        # masked by is_dk1_i * is_dl2_j

        # Delta term: sum_k [is_dk1_i * is_dk2_j / ls2_k]
        # ti_d: (n1, d), tj_d: (n2, d), 1/ls2: (d,)
        # Outer product masked by matching dimension:
        # (n1, d) * (1/ls2) -> (n1, d), then matrix multiply with tj_d^T
        delta_term = tf.matmul(
            ti_d / ls2,  # (n1, d)
            tf.transpose(tj_d)  # (d, n2)
        )  # (n1, n2)

        # Cross term: sum_k sum_l [is_dk1_i * is_dl2_j * diff_k * diff_l / (ls2_k * ls2_l)]
        # = (sum_k is_dk1_i * diff_k / ls2_k) * (sum_l is_dl2_j * diff_l / ls2_l)
        # First factor: (n1, 1, d) * (n1, n2, d) / ls2 -> sum over d -> (n1, n2)
        cross_1 = tf.reduce_sum(
            tf.expand_dims(ti_d, 1) * scaled_diff, axis=-1
        )  # (n1, n2)
        cross_2 = tf.reduce_sum(
            tf.expand_dims(tj_d, 0) * scaled_diff, axis=-1
        )  # (n1, n2)
        cross_term = cross_1 * cross_2  # (n1, n2)

        K += Kff * (delta_term - cross_term)

        return K

    def K_diag(self, X):
        tasks = tf.cast(X[:, self.input_dim], tf.int32)
        ls2 = self.lengthscales ** 2

        # task 0: variance
        # task k: variance / ls2[k-1]
        diag_vals = tf.concat(
            [tf.expand_dims(self.variance, 0), self.variance / ls2],
            axis=0
        )
        return tf.gather(diag_vals, tasks)


def prepare_gpflow_data(X_train, y_vals, grads, dim):
    """
    Prepare training data in GPflow format with task indicator column.

    Parameters
    ----------
    X_train : ndarray (n, d)
        Training locations.
    y_vals : ndarray (n,)
        Function values.
    grads : ndarray (n, d)
        Gradient values.
    dim : int
        Input dimension.

    Returns
    -------
    X_aug : ndarray (n*(1+d), d+1)
        Augmented input with task indicators.
    Y_aug : ndarray (n*(1+d), 1)
        Augmented output.
    """
    n = X_train.shape[0]

    # Function values: task = 0
    X_func = np.column_stack([X_train, np.zeros(n)])
    Y_func = y_vals.reshape(-1, 1)

    X_list = [X_func]
    Y_list = [Y_func]

    # Derivative observations: task = j (1-indexed)
    for j in range(dim):
        X_deriv = np.column_stack([X_train, np.full(n, j + 1)])
        Y_deriv = grads[:, j].reshape(-1, 1)
        X_list.append(X_deriv)
        Y_list.append(Y_deriv)

    X_aug = np.vstack(X_list)
    Y_aug = np.vstack(Y_list)

    return X_aug, Y_aug


def run_gpflow_benchmark(func, grad_func, dim, function_name,
                         sample_sizes, n_macroreplicates=5, n_test=2000,
                         n_training_iter=200):
    """
    Run the GPflow DEGP benchmark for a given test function.
    Single Adam optimizer run (no restarts).

    Parameters
    ----------
    func : callable
        Test function.
    grad_func : callable
        Gradient function.
    dim : int
        Input dimension.
    function_name : str
        Name for output file.
    sample_sizes : list of int
        Training sample sizes.
    n_macroreplicates : int
        Number of macroreplicates.
    n_test : int
        Number of test points.
    n_training_iter : int
        Number of Adam iterations.
    """
    from benchmark_functions import generate_test_data, compute_metrics
    from scipy.stats.qmc import LatinHypercube
    import json

    results = []

    for n_train in sample_sizes:
        print(f"\n{'='*60}")
        print(f"  GPflow DEGP — {function_name} — n_train = {n_train}")
        print(f"{'='*60}")

        for rep in range(n_macroreplicates):
            seed = 1000 + rep
            print(f"\n  Macroreplicate {rep + 1}/{n_macroreplicates} (seed={seed})")

            # Generate training data
            sampler = LatinHypercube(d=dim, seed=seed)
            X_train = sampler.random(n=n_train)
            y_vals = func(X_train)
            grads = grad_func(X_train)

            # Generate test data
            X_test, y_test = generate_test_data(func, n_test, dim, seed=99)

            # Standardize outputs
            y_mean = y_vals.mean()
            y_std = y_vals.std()
            y_vals_std = (y_vals - y_mean) / y_std
            grads_std = grads / y_std

            # Prepare augmented data
            X_aug, Y_aug = prepare_gpflow_data(
                X_train, y_vals_std, grads_std, dim
            )

            # Create model
            kernel = DerivativeSEKernel(dim)
            model = gpflow.models.GPR(
                data=(X_aug, Y_aug),
                kernel=kernel,
                noise_variance=1e-4
            )

            # Single Adam training run
            opt = tf.optimizers.Adam(learning_rate=0.1)

            t_start = time.perf_counter()
            
            for i in range(n_training_iter):
                with tf.GradientTape() as tape:
                    loss = model.training_loss()
                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))
            
            t_train = time.perf_counter() - t_start

            # Predict — function values only (task = 0)
            X_test_aug = np.column_stack([
                X_test, np.zeros(X_test.shape[0])
            ])

            t_pred_start = time.perf_counter()
            f_mean, _ = model.predict_f(X_test_aug)
            t_pred = time.perf_counter() - t_pred_start

            # Denormalize
            y_pred = f_mean.numpy().flatten() * y_std + y_mean

            # Compute metrics
            metrics = compute_metrics(y_test, y_pred)
            metrics['train_time'] = t_train
            metrics['pred_time'] = t_pred
            metrics['n_train'] = n_train
            metrics['seed'] = seed
            metrics['macroreplicate'] = rep + 1
            results.append(metrics)

            print(f"    RMSE:       {metrics['rmse']:.6e}")
            print(f"    NRMSE:      {metrics['nrmse']:.6e}")
            print(f"    Train time: {metrics['train_time']:.2f}s")
            print(f"    Pred time:  {metrics['pred_time']:.4f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for n_train in sample_sizes:
        subset = [r for r in results if r['n_train'] == n_train]
        rmses = [r['rmse'] for r in subset if not np.isnan(r['rmse'])]
        times = [r['train_time'] for r in subset if r['train_time'] > 0]
        if rmses:
            print(f"\n  n = {n_train}:")
            print(f"    RMSE:  mean={np.mean(rmses):.6e}, std={np.std(rmses):.6e}")
            if times:
                print(f"    Time:  mean={np.mean(times):.2f}s, std={np.std(times):.2f}s")
        else:
            print(f"\n  n = {n_train}: All runs failed")

    # Save results
    output_file = f"results_gpflow_{function_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results
