import modules.sobol as sb
import pickle
from matplotlib import pyplot as plt
import timeit
import utils  # Utility functions, including one to generate derivative indices
import pyoti.sparse as oti
import numpy as np
# Library for automatic differentiation using hyper-complex numbers


if __name__ == "__main__":

    # Zhis function will generate random samples for parameters.
    # Zhese samples will be used for MC simulation
    def scale_samples(samples, lower_bounds, upper_bounds):
        """
        Scale each column of samples from [0, 1] to [lb_j, ub_j].

        Parameters:
            samples (ndarray): A (d, n) array of samples in [0, 1]^n.
            lower_bounds (array-like): Length-n array of lower bounds.
            upper_bounds (array-like): Length-n array of upper bounds.

        Returns:
            ndarray: A (d, n) array with each column scaled to its corresponding bounds.
        """
        samples = np.asarray(samples)
        lower_bounds = np.asarray(lower_bounds)
        upper_bounds = np.asarray(upper_bounds)

        # Ensure correct shapes
        assert (
            samples.shape[1] == len(lower_bounds) == len(upper_bounds)
        ), "Dimension mismatch between samples and bounds"

        # Reshape bounds to broadcast across rows
        lb = lower_bounds[np.newaxis, :]
        ub = upper_bounds[np.newaxis, :]

        return lb + samples * (ub - lb)

    # Set the random seed for reproducibility
    np.random.seed(1354)
    n_bases = 4

    lower_bounds = [-2.048 for i in range(4)]
    upper_bounds = [2.048 for i in range(4)]

    def true_function(x1, x2, x3, x4, alg=oti):

        return (
            100 * (x2 - x1**2) ** 2
            + (1 - x1) ** 2
            + 100 * (x3 - x2**2) ** 2
            + (1 - x2) ** 2
            + 100 * (x4 - x3**2) ** 2
            + (1 - x3) ** 2
        )

    X_test = np.array([.654, .165, -.948, 1.054]).reshape(1, -1)
    x1_real = X_test[0, 0]
    x2_real = X_test[0, 1]
    x3_real = X_test[0, 2]
    x4_real = X_test[0, 3]
    for order in range(0, 5):
        n_order = order

        # Generate indices for all derivatives up to the specified order
        # in a function with n_bases input dimensions.
        der_indices = utils.gen_OTI_indices(n_bases, n_order)

        # If the use wants only to use for example main derivatives in the
        # training process set der_indices as:
        # der_indices = [
        #     [[[1, 1]], [[2, 1]]],
        #     [[[1, 2]], [[2, 2]]],
        #     [[[1, 3]], [[2, 3]]],
        #     [[[1, 4]], [[2, 4]]],
        # ]
        # If the use wants only to use for example first and highest order derivatives in the
        # training process set der_indices as:
        # der_indices = [
        #     [[[2, 1]]],
        #     [[[2, 2]]],
        #     [[[2, 3]]],
        # ]
        # We use 5 points for this simple example. In a real case, choose
        # more or fewer points depending on the function's complexity.

        # ----- Generate Training Data -----
        with open("4d_rmse_benchmark_rosenbrock_data.pkl", "rb") as f:
            rmse_data = pickle.load(f)

        with open("4d_rmse_benchmark_rosenbrock_data_pts.pkl", "rb") as f:
            pt_data = pickle.load(f)
        pt_data
        time_data_i_hc = []
        time_data_i_real = []

        X_train = oti.array(X_test)

        # Convert training data to an OTI array that supports derivative tracking
        X_train_pert = X_train.copy()
        for i in range(1, n_bases + 1):
            X_train_pert[0, i-1] = X_train_pert[0, i-1] + oti.e(
                i, order=n_order
            )

        x1_im = X_train_pert[0, 0]
        x2_im = X_train_pert[0, 1]
        x3_im = X_train_pert[0, 2]
        x4_im = X_train_pert[0, 3]
        # Perturb the training inputs along each coordinate direction to enable derivative computation.
        # For each input dimension, add the elementary perturbation defined by oti.e

        # ----- Define the True Function -----
        # This is an arbitrarily chosen polynomial function in two variables.
        # It has nonlinear behavior and is used here for demonstration.
        # Timing function for hypercomplex (oti)

        def time_hc():
            true_function(x1_im, x2_im, x3_im, x4_im, alg=oti)

        # Timing function for numpy
        def time_np():
            true_function(x1_real, x2_real, x3_real, x4_real, alg=np)

        num_runs = 10000

        np_times = timeit.repeat(time_np, repeat=num_runs, number=1)

        hc_times = timeit.repeat(time_hc, repeat=num_runs, number=1)
        multiplier = np.mean(hc_times)/np.mean(np_times)
        print(round(1 - multiplier * pt_data[order][-1]/pt_data[0][-1], 2))
        plt.rcParams.update({"font.size": 12})
        plt.figure(12, figsize=(8, 6))

        # Plot up to that index
        if order == 0:
            plt.semilogy(
                np.array(pt_data[order]) * 1,
                rmse_data[order],
                label="Derivative Enhanced GP\nOrder {}".format(order),
            )
        else:
            plt.semilogy(
                np.array(pt_data[order]) * multiplier,
                rmse_data[order],
                label="Derivative Enhanced GP\nOrder {}".format(order),
            )

    plt.xlabel("Effective Number of Sample Points")
    plt.ylabel("NRMSE (log scale)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.minorticks_on()  # Ensure minor ticks are activated
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.000, 1.00),
        borderaxespad=0,
        frameon=False,
    )
    plt.savefig("time_plot_4d_rosenbrock_benchmark.pdf",
                bbox_inches="tight")  # Save as PDF

    plt.tight_layout()
    plt.show()
