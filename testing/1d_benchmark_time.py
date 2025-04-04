import numpy as np
# Library for automatic differentiation using hyper-complex numbers
import pyoti.sparse as oti
import utils  # Utility functions, including one to generate derivative indices
import timeit
from matplotlib import pyplot as plt
import pickle


def true_function(X, alg=oti):
    return alg.sin(10 * np.pi * X) / (2 * X) + (X - 1) ** 4


if __name__ == "__main__":
    # Set the random seed for reproducibility
    n_bases = 1
    lb_x = 0.5
    ub_x = 2.5
    time_data_real = []
    time_data_hc = []
    for order in range(0, 10):
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
        with open("1d_rmse_benchmark_data.pkl", "rb") as f:
            rmse_data = pickle.load(f)

        with open("1d_rmse_benchmark_data_pts.pkl", "rb") as f:
            pt_data = pickle.load(f)
        pt_data
        time_data_i_hc = []
        time_data_i_real = []

        num_points = 1
        lb_x = 0.5
        ub_x = 2.5

        X_train = np.linspace(lb_x, ub_x, num_points)[0]

        # Convert training data to an OTI array that supports derivative tracking
        X_train_pert = oti.number(X_train)

        # Perturb the training inputs along each coordinate direction to enable derivative computation.
        # For each input dimension, add the elementary perturbation defined by oti.e
        for i in range(1, n_bases + 1):
            X_train_pert = X_train_pert + oti.e(
                i, order=n_order
            )

        # ----- Define the True Function -----
        # This is an arbitrarily chosen polynomial function in two variables.
        # It has nonlinear behavior and is used here for demonstration.
        # Timing function for hypercomplex (oti)
        def time_hc():
            true_function(X_train_pert, alg=oti)

        # Timing function for numpy
        def time_np():
            true_function(X_train, alg=np)

        num_runs = 1000000

        np_times = timeit.repeat(time_np, repeat=num_runs, number=1)

        hc_times = timeit.repeat(time_hc, repeat=num_runs, number=1)
        multiplier = np.mean(hc_times)/np.mean(np_times)
        # print(multiplier)

        plt.rcParams.update({"font.size": 12})
        plt.figure(12, figsize=(8, 6))
        print(round(1 - multiplier * pt_data[order][-1]/pt_data[0][-1], 2))
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
    plt.savefig("time_plot_1d_benchmark.pdf",
                bbox_inches="tight")  # Save as PDF

    plt.tight_layout()
    plt.show()
