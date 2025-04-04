import modules.sobol as sb
import pickle
from matplotlib import pyplot as plt
import timeit
import utils
import pyoti.sparse as oti
import numpy as np


def true_function(x1, x2, x3, x4, x5, x6, x7, x8, alg=oti):
    return (
        1 * x1**2 +
        2 * x2**2 +
        3 * x3**2 +
        4 * x4**2 +
        5 * x5**2 +
        6 * x6**2 +
        7 * x7**2 +
        8 * x8**2
    )


if __name__ == "__main__":

    np.random.seed(1354)
    n_bases = 8

    lower_bounds = [-2.048 for _ in range(n_bases)]
    upper_bounds = [2.048 for _ in range(n_bases)]

    with open("8d_rmse_benchmark_sum_square_data.pkl", "rb") as f:
        rmse_data = pickle.load(f)

    with open("8d_rmse_benchmark_sum_square_data_pts.pkl", "rb") as f:
        pt_data = pickle.load(f)

    X_test = np.array([[0.2 * i for i in range(1, 9)]])
    x_real = X_test[0]
    time_data_i_hc = []
    time_data_i_real = []

    for order in range(0, 3):
        n_order = order
        der_indices = utils.gen_OTI_indices(n_bases, n_order)

        X_train = oti.array(X_test)
        X_train_pert = X_train.copy()

        for i in range(1, n_bases + 1):
            X_train_pert[0, i - 1] = X_train_pert[0,
                                                  i - 1] + oti.e(i, order=n_order)

        x_im = X_train_pert

        def time_hc():
            true_function(
                x_im[0, 0], x_im[0, 1], x_im[0, 2], x_im[0, 3],
                x_im[0, 4], x_im[0, 5], x_im[0, 6], x_im[0, 7],
                alg=oti
            )

        def time_np():
            true_function(
                x_real[0], x_real[1], x_real[2], x_real[3],
                x_real[4], x_real[5], x_real[6], x_real[7],
                alg=np
            )

        num_runs = 100000
        np_times = timeit.repeat(time_np, repeat=num_runs, number=1)
        hc_times = timeit.repeat(time_hc, repeat=num_runs, number=1)
        multiplier = np.mean(hc_times) / np.mean(np_times)

        print(round(1 - multiplier * pt_data[order][-1] / pt_data[0][-1], 2))

        plt.rcParams.update({"font.size": 12})
        plt.figure(12, figsize=(8, 6))

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
    plt.minorticks_on()
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.000, 1.00),
        borderaxespad=0,
        frameon=False,
    )
    plt.savefig("time_plot_8d_quadratic_benchmark.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.show()
