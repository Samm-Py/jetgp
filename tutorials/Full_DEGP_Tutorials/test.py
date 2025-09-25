import numpy as np
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = "cpu"
    # n_bases: the dimensionality of the input space (1D in this example)
    n_bases = 1

    # --- Training Data ---
    # We use the same training points for all models.
    X_train = np.array([[0.0439],
                        [0.5439],
                        [0.2939],
                        [0.7939]])

    # --- Test Data ---
    # Create a dense test grid for smooth plotting.
    X_test = torch.linspace(0, 1, 252, device=device)[1:-1, None].numpy()

    # --- True Function Definition ---
    # This is the underlying function we are trying to model.
    def true_function(X, alg=oti):
        x = X[:, 0]
        return 15 * (x - 1/2)**2 * alg.sin(2 * np.pi * x)

    # Evaluate the true function on the test grid for comparison.
    y_true = true_function(X_test, alg=np)

    # --- Plotting Setup ---
    # Create a 2x2 subplot grid to display the results.
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axs = axs.flatten() # Flatten the 2x2 array for easy iteration

    # --- Main Loop for Different Orders ---
    # We will loop through orders 0, 1, 2, and 3.
    for order in range(4):
        print(f"--- Processing Order = {order} ---")
        ax = axs[order]
        
        # n_order: the maximum derivative order used for perturbation.
        n_order = order
        
        # smoothness_parameter: set according to the current order.
        smoothness_parameter = order + 1

        # der_indices: Generate indices for all derivatives up to the current order.
        der_indices = utils.gen_OTI_indices(n_bases, n_order)

        # --- Generate Training Data with Derivatives ---
        # Perturb inputs with hypercomplex numbers to compute derivatives automatically.
        X_train_pert = oti.array(X_train)
        if n_order > 0:
            for i in range(1, n_bases + 1):
                X_train_pert[:, i - 1] = X_train_pert[:, i - 1] + oti.e(i, order=n_order)

        # Evaluate the function with perturbed inputs to get derivative information.
        y_train_hc = true_function(X_train_pert)
        y_train_real = y_train_hc.real

        # Assemble the training output list: [function_values, derivative_1, derivative_2, ...]
        y_train = [y_train_real]
        if n_order > 0:
            for i in range(len(der_indices)):
                for j in range(len(der_indices[i])):
                    y_train.append(
                        y_train_hc.get_deriv(der_indices[i][j]).reshape(-1, 1)
                    )

        # --- GP Model Training ---
        # Instantiate the derivative-enhanced GP model for the current order.
        gp = degp(
            X_train,
            y_train,
            n_order,
            n_bases,
            der_indices,
            normalize=False,
            kernel="SI",
            kernel_type="anisotropic",
            smoothness_parameter=smoothness_parameter
        )

        # Optimize the model's hyperparameters.
        params = gp.optimize_hyperparameters(
            n_restart_optimizer=30,
            swarm_size=400,
            local_opt_every=30
        )

        # --- GP Prediction ---
        # Predict the mean and variance over the test grid.
        y_pred, y_var = gp.predict(
            X_test, params, calc_cov=True, return_deriv=False
        )
        y_pred = y_pred.flatten()
        y_std = np.sqrt(y_var.flatten())

        # --- Plotting for the Current Order ---
        # Plot the true function.
        ax.plot(X_test, y_true, 'r--', label='True Function')
        
        # Plot the GP's mean prediction.
        ax.plot(X_test, y_pred, 'b-', label='GP Mean')
        
        # Plot the 95% confidence interval.
        ax.fill_between(
            X_test.flatten(),
            y_pred - 1.96 * y_std,
            y_pred + 1.96 * y_std,
            color='blue',
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        # Plot the original training points.
        ax.plot(X_train, y_train[0], 'kx', markersize=10, mew=2.5, label='Training Data')
        
        # Set title and labels.
        ax.set_title(f'Order = {order}, Smoothness = {smoothness_parameter}', fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$f(x)$', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(-2.5, 2.5)

        # --- Compute and Print NRMSE Metric ---
        nrmse = utils.nrmse(y_true, y_pred.reshape(-1, 1))
        print(f"NRMSE for Order {order}: {nrmse:.4f}\n")

    # --- Final Plot Adjustments ---
    # Create a single, shared legend in the margin of the figure.
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize=12)
    
    # Adjust layout to prevent overlap and make room for the legend.
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    
    # Display the final composite plot.
    plt.show()