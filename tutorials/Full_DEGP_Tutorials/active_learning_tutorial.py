# main_script.py

import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyoti.sparse as oti
from full_degp.degp import degp
import utils  # Your utility file
from dataclasses import dataclass
from scipy.stats.qmc import Sobol

# --- Import your acquisition functions ---
from acquisition_functions import acquisition_funcs as acq

# Set plotting parameters and manage warnings
plt.rcParams.update({'font.size': 12})
warnings.filterwarnings("ignore", "invalid value encountered in sqrt")


# ==============================================================================
# 1. CONFIGURATION AND TRUE FUNCTION
# ==============================================================================
@dataclass
class ActiveLearningConfig:
    """Configuration for the Active Learning DoE experiment."""
    # --- Domain and Candidate Points ---
    lb_x: float = 0.5
    ub_x: float = 2.5
    num_integration_pts: int = 1000

    # --- Active Learning Loop Settings ---
    num_initial_points: int = 5
    num_points_to_add: int = 5
    n_order: int = 1  # The derivative order to use for this experiment

    # --- GP Model Parameters ---
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"

    # --- Optimizer Settings ---
    n_restarts: int = 15
    swarm_size: int = 150


def true_function(X, alg=oti):
    """Test function combining exponential decay, oscillations, and linear trend."""
    x1 = X[:, 0]
    return alg.sin(10 * np.pi * x1) / (2 * x1) + (x1 - 1) ** 4


# ==============================================================================
# 2. HELPER FUNCTION
# ==============================================================================

def get_training_data_for_model(X_points: np.ndarray, n_order: int):
    """Generates the list of derivative observations for a given set of points."""
    der_indices = utils.gen_OTI_indices(1, n_order)
    X_pert = oti.array(X_points) + oti.e(1, order=n_order)
    y_hc = true_function(X_pert)

    y_list = [y_hc.real]
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            derivative = y_hc.get_deriv(der_indices[i][j]).reshape(-1, 1)
            y_list.append(derivative)
    return y_list


# ==============================================================================
# 3. VISUALIZATION FUNCTION
# ==============================================================================

def create_animation(history, config, filename="active_learning_doe.gif"):
    fig, ax = plt.subplots(figsize=(12, 7))
    # Use the first history item's integration points for the plot x-axis
    candidate_points = history[0]['integration_points']
    y_true = true_function(candidate_points, alg=np)

    ax.plot(candidate_points, y_true, 'k-', lw=3, label="True Function", zorder=1)
    gp_mean, = ax.plot([], [], 'b--', lw=2, label="GP Mean")
    gp_ci = ax.fill_between(candidate_points.ravel(), 0, 0, color='blue', alpha=0.15)
    train_pts, = ax.plot([], [], 'ro', markersize=8, label="Training Points", zorder=10)
    next_pt, = ax.plot([], [], 'g*', markersize=15, label="Next Point to Add", zorder=11, markeredgecolor='k')

    ax.set_xlim(config.lb_x, config.ub_x)
    ax.set_ylim(np.min(y_true) - 1, np.max(y_true) + 1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    fig.legend(loc='upper right')

    def update(frame):
        nonlocal gp_ci
        hist_item = history[frame]
        X_train, y_pred = hist_item['X_train'], hist_item['y_pred']
        y_train = true_function(X_train, alg=np)
        std_dev = np.sqrt(hist_item['y_var'])

        gp_mean.set_data(candidate_points, y_pred)
        train_pts.set_data(X_train, y_train)

        gp_ci.remove()
        gp_ci = ax.fill_between(
            candidate_points.ravel(),
            y_pred.ravel() - 2 * std_dev.ravel(),
            y_pred.ravel() + 2 * std_dev.ravel(),
            color='blue', alpha=0.15
        )

        if hist_item['next_point'] is not None:
            next_x = hist_item['next_point']
            next_y_val = np.interp(next_x.item(), candidate_points.ravel(), y_pred.ravel())
            next_pt.set_data(next_x, [next_y_val])
        else:
            next_pt.set_data([], [])

        ax.set_title(f"Active Learning - Iteration {frame + 1} | Points: {len(X_train)}")
        return gp_mean, train_pts, next_pt, gp_ci

    ani = FuncAnimation(fig, update, frames=len(history), blit=True, interval=1000)
    ani.save(filename, writer='pillow', fps=1)
    plt.close(fig)


# ==============================================================================
# 4. MAIN SCRIPT
# ==============================================================================

def main():
    config = ActiveLearningConfig()
    
    # --- Define Distribution Parameters with EXPLICIT BOUNDS ---
    # dist = ["TN"] * 15
    dimension = 1
    # # Create the first element as a list
    # first_element = [1/4000]
    
    # # Create the next 14 elements as a list
    # next_14_elements = [1/4] * 14
    
    # # Concatenate them into a single flat list and then create the array
    # means = np.array(first_element + next_14_elements)
    
    
    # # Create the first element as a list
    # first_variance = [(1/12000)**2]
    
    # # Create the next 14 elements as a list
    # next_14_variances = [(1/12)**2] * 14
    
    # # Concatenate them into a single flat list, then create the array
    # variances = np.array(first_variance + next_14_variances)
    # lower_bounds = means - 3 * np.sqrt(variances) # Explicit lower bound for the distribution
    # upper_bounds = means + 3 * np.sqrt(variances)  # Explicit upper bound
    
    # dist_params = {
    #     'dists': dist, 
    #     'means': means, 
    #     'vars': variances,
    #     'lower_bounds': lower_bounds,
    #     'upper_bounds': upper_bounds
    # }
    
    
    
    # You could also define a Uniform distribution this way:
    dist = ["U"]
    lower_bounds = np.array([0.5])
    upper_bounds = np.array([2.5])
    dist_params = {'dists': dist, 'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds}

    
    acquisition_function_to_use = acq.imse_reduction
    print(f"Using Acquisition Function: IMSE Reduction ({acquisition_function_to_use.__name__})")

    # --- Generate candidate and training points from the distribution ---
    print(f"Generating points from a {dist[0]} distribution...")
    sampler = Sobol(d=dimension, scramble=True)
    
    # Generate integration_points
    uniform_samples_int = sampler.random(n=config.num_integration_pts)
    integration_points = utils.get_inverse(dist_params, uniform_samples_int)
    integration_points = np.sort(integration_points, axis=0)

    # Sample Initial Training Data
    uniform_samples_train = sampler.random(n=config.num_initial_points)
    X_train = utils.get_inverse(dist_params, uniform_samples_train)
    
    # --- The rest of the main function remains the same ---
    history = []
    print("-" * 60)

    # (The rest of the main function remains exactly the same as the previous version)
    for i in range(config.num_points_to_add + 1):
        num_pts = X_train.shape[0]
        print(f"Iteration {i+1}/{config.num_points_to_add + 1} | Training with {num_pts} points...")

        y_train_list = get_training_data_for_model(X_train, config.n_order)
        der_indices = utils.gen_OTI_indices(1, 1)
        gp = degp(
            X_train, y_train_list, config.n_order, n_bases=1, der_indices=der_indices,
            normalize=config.normalize_data, kernel=config.kernel, kernel_type=config.kernel_type
        )
        params = gp.optimize_hyperparameters(
            n_restart_optimizer=config.n_restarts, swarm_size=config.swarm_size)

        y_pred, y_var = gp.predict(
            integration_points, params, calc_cov=True, return_deriv=False)

        next_point = None
        if i < config.num_points_to_add:
            print("  Finding next point using two-stage optimization...")
            next_point, score = utils.find_next_point(
                gp, params, X_train, y_train_list,dist_params,acquisition_function_to_use, integration_points = integration_points,
                n_candidate_points=256, n_local_starts=10
            )
            

                
            print(f"  -> Next point chosen: x = {next_point.item():.4f}\n")
            X_train = np.vstack([X_train, next_point])

        history.append({
            "X_train": X_train.copy(),
            "y_pred": y_pred,
            "y_var": y_var,
            "next_point": next_point,
            "integration_points": integration_points
        })

    print("-" * 60)
    print("Active learning process complete.")
    create_animation(history, config)


if __name__ == "__main__":
    main()