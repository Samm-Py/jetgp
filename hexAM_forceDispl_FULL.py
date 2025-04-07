import os
import re
import sys
import numpy as np
# Library for automatic differentiation using hyper-complex numbers
import pyoti.sparse as oti
import itertools
from oti_gp import oti_gp  # Derivative-enhanced Gaussian Process class
import utils  # Utility functions, including one to generate derivative indices
import time
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt


def TrainingOutputOTI(n_bases, n_order, sample_index):

    n_samples = len(sample_index)

    # Form TSE indices for each term
    coef = []
    for i in range(1, n_bases+1):
        coef.append([i, 0, 0])
    for i in range(1, n_bases+1):
        for j in range(1, i+1):
            coef.append([i, j, 0])
    for i in range(1, n_bases+1):
        for j in range(1, i+1):
            for k in range(1, j+1):
                coef.append([i, j, k])
    coef.append([0, 0, 0])

    # Output definition
    i = 0
    folder = '../../OUTPUT/GP/'
    n_tincs, n_terms = np.load(
        folder + 'results_reduced_' + str(sample_index[0]+1).zfill(4) + '.npy').shape
    F_oti = oti.zeros((n_tincs, n_samples))
    for ind in sample_index:
        # Load Cython output for (i+1)-th training point
        str_index = str(ind+1).zfill(4)
        F_hyp = np.load(folder + 'results_reduced_' + str_index + '.npy')

        # Create OTI variable from Cython sensitivties
        for k in range(n_terms):
            for j in range(n_tincs):
                F_oti[j, i] = F_oti[j, i] + F_hyp[j, k]*oti.e(coef[k][0], order=n_order)*oti.e(
                    coef[k][1], order=n_order)*oti.e(coef[k][2], order=n_order)
        i += 1

    return F_oti


if __name__ == "__main__":

    global_time = time.time()

# ----- Parameter Setup -----
    n_order = 2
    n_samples = 24  # For now I am using 24 samples.

    # Load all MC inputs
    filename = '../../UQ/MC_input_3000samples.mat'
    c = loadmat(filename)['c']
    n_mc_samples = c.shape[0]
    n_bases = c.shape[1]

    # plt.figure
    # plt.hist(c[:,1])
    # plt.show()
    # sys.exit()

    # Loop through all filenames from training samples to parse MC index
    k = 0
    sample_index = []
    X_train = np.zeros((n_samples, n_bases))
    folder_path = '../../GP_TrainingInputs'
    pattern = re.compile(r'run(\d+)\.inp')
    for filename in os.listdir(folder_path):
        match = pattern.fullmatch(filename)
        if match:
            index = int(match.group(1))-1
            # print(index)
            sample_index.append(index)
            X_train[k, :] = c[index, :]
            k += 1

    # print(X_train.shape)
    # sys.exit()

    # Load OTI force-displacement curves for all samples
    Y_oti = TrainingOutputOTI(n_bases, 3, sample_index)
    # Y_oti = Y_oti.truncate(n_order)
    n_tincs = Y_oti.shape[0]

    # Generate indices for all derivatives up to the specified order
    # in a function with n_bases input dimensions.
    der_indices = utils.gen_OTI_indices(n_bases, n_order)
    # der_indices = [[[[2, 1]], [[3, 1]], [[4, 1]]],
    #                [
    #     [[2, 2]],
    #     [[3, 2]],
    #     [[4, 2]]],
    #     [
    #     [[2, 3]],
    #     [[3, 3]],
    #     [[4, 3]]]]
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
    #     [[[1, 1]], [[2, 1]]],
    #     [[[1, 2]], [[2, 2]]],
    # ]
    # # We use 5 points for this simple example. In a real case, choose
    # more or fewer points depending on the function's complexity.

    # Load all MC samples for testing dataset
    filename = '../../UQ/ROM/MC_input.mat'
    X_test = loadmat(filename)['inputMC']

    # Load ground truth
    filename = '../../UQ/ROM/MC_force.mat'
    force_imported = loadmat(filename)['forceMC']

    # For now lets make a GP for the 150th time increment,
    # but eventually we will have to loop through all time
    # increments to create N GP regressions
    tincs = [50*i for i in range(1, 20)]
    val = []
    val_lb = []
    val_ub = []

    # visulaize training data

    # Set global font size
    plt.rcParams.update({'font.size': 12})

    plt.figure(1, figsize=(9, 5))

    # Plot OTI training data
    for i in range(Y_oti.shape[1]):
        Y_vals = Y_oti[:, i].real
        label = 'Training Data' if i == 0 else None
        plt.plot(np.arange(0, 1001), Y_vals.flatten(),
                 color='tab:blue', zorder=2, label=label)

    # Plot Monte Carlo training data
    for i in range(force_imported.shape[0]):
        val_mc = force_imported[i, :]
        label = 'Monte Carlo Data' if i == 0 else None
        plt.plot(np.arange(0, 1001), val_mc,
                 color='tab:red', zorder=0, label=label)

    # Plot mean of Monte Carlo data
    vals_mc = np.mean(force_imported, axis=0)
    plt.plot(np.arange(0, 1001), vals_mc, color='black',
             ls='--', lw=2, label='MC Mean Force', zorder=4)

    for tinc in tincs:

        y_true = force_imported[:, tinc]
        # spacing = 25
        # force_true = force_imported[:,range(0, n_tincs, spacing)]
        # MIN1 = np.min(force_true)
        # MAX1 = np.max(force_true)

        count = 0
        time_stamps = []
        # force_pred = np.zeros((force_true.shape))
        # for tinc in range(0, n_tincs, spacing):

        # ----- Assemble Training Data with Derivative Information -----
        # Start with the real function values
        Y_oti_current = Y_oti[tinc, :]
        y_train_real = Y_oti_current.real  # Extract just the real part

        y_train = [y_train_real.reshape(-1, 1)]
        for i in range(len(der_indices)):
            for j in range(len(der_indices[i])):
                y_train.append(Y_oti_current.get_deriv(
                    der_indices[i][j]).reshape(-1, 1))

        # Flatten the training output into a 1D array (required format for many GP implementations)

        # ----- Gaussian Process Model Setup -----
        # Create the derivative-enhanced Gaussian Process model.
        # We pass the original training inputs (X_train) along with the training outputs (y_train)
        # that include both function values and derivative information.
        gp = oti_gp(
            X_train,  # Unperturbed training inputs
            y_train,  # Training outputs (function values and derivatives)
            n_order,  # Order of derivative information used
            n_bases,  # Dimensionality of the input space
            der_indices,  # List of which derivatives to include
            kernel="SE",  # Kernel choice: Rational Quadratic (RQ) kernel
            # Anisotropic kernel to allow different length-scales per dimension
            kernel_type="anisotropic",
        )

        # Optimize the GP hyperparameters (e.g., length-scales, kernel variance) by maximizing the likelihood
        # Optimize the GP hyperparameters (e.g., length-scales, kernel variance) by maximizing the likelihood
        params = gp.optimize_hyperparameters(
            n_restart_optimizer=20, swarm_size=50
        )
        print(params)

        # ----- GP Prediction -----
        # Predict the function values on the test data using the optimized GP model.
        y_pred = gp.predict(X_test, params, calc_cov=False, return_deriv=False)

        # Assuming y_pred is a NumPy array
        mean = np.mean(y_pred)
        std = np.std(y_pred)

        lower_bound = mean - 2*std
        upper_bound = mean + 2*std
        # force_pred[:,count] = y_pred
        # count += 1

        time_iteration = time.time() - global_time
        print((tinc+1), '/', n_tincs, ' : Cumulative time ', time_iteration)
        time_stamps.append(time_iteration)
        val.append(mean)
        val_lb.append(lower_bound)
        val_ub.append(upper_bound)
        plt.figure(tinc)
        plt.plot(y_true.flatten(), y_pred[: X_test.shape[0]].flatten(), 'r.')
        plt.plot(Y_oti[tinc, :].real, Y_oti[tinc, :].real, 'b.')
        plt.show()

    plt.figure(1)
    plt.plot(tincs, val, label='Mean Prediction',
             color='tab:pink', lw=2, zorder=11)
    plt.fill_between(tincs, val_lb, val_ub, color='tab:pink',
                     alpha=0.3, label=r'Mean GP prediction $\pm 2 \sigma$', zorder=10)
    plt.xlabel('Time')
    plt.ylabel('Prediction')

    # Move legend to right margin
    plt.legend(loc='center left', bbox_to_anchor=(1.01, .90), borderaxespad=0.)

    # Adjust layout to make room for legen
    plt.grid(True)
    plt.tight_layout()
    # utils.make_plots(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_pred,
    #     true_function,
    #     X1_grid=X1_grid,
    #     X2_grid=X2_grid,
    #     n_order=n_order,
    #     n_bases=n_bases,
    #     plot_derivative_surrogates=True,
    #     der_indices=der_indices,
    # )

    # end

    # plt.figure()
    # plt.plot(force_true.flatten(), force_pred.flatten(), 'r.')
    # for j in range(0, n_tincs, spacing):
    #     plt.plot(Y_oti[j,:].real, Y_oti[j,:].real, 'b.')
    # plt.show()

    plt.figure(tinc)
    plt.plot(y_true.flatten(), y_pred[: X_test.shape[0]].flatten(), 'r.')
    plt.plot(Y_oti[tinc, :].real, Y_oti[tinc, :].real, 'b.')
    plt.show()
