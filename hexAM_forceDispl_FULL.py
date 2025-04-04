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
    tincs = [500]
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
        y_train = Y_oti_current.real

        # For each derivative index generated, extract the corresponding derivative
        # from the hyper-complex output and vertically stack it with the function values.
        for i in range(0, len(der_indices)):
            for j in range(0, len(der_indices[i])):
                y_train = np.vstack(
                    (y_train, Y_oti_current.get_deriv(der_indices[i][j])))

        # Flatten the training output into a 1D array (required format for many GP implementations)
        y_train = y_train.flatten()

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
            n_restart_optimizer=25, swarm_size=25
        )
        print(params)

        # ----- GP Prediction -----
        # Predict the function values on the test data using the optimized GP model.
        y_pred = gp.predict(X_test, params, calc_cov=False, return_deriv=True)
        # force_pred[:,count] = y_pred
        # count += 1

        time_iteration = time.time() - global_time
        print((tinc+1), '/', n_tincs, ' : Cumulative time ', time_iteration)
        time_stamps.append(time_iteration)

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
