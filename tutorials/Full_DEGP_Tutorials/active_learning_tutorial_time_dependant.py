# main_script_time_dependent.py

import warnings
import numpy as np
import matplotlib.pyplot as plt
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
from dataclasses import dataclass

# --- Import your acquisition functions ---
from acquisition_functions import acquisition_funcs as acq

# Set plotting parameters and manage warnings
plt.rcParams.update({'font.size': 12})
warnings.filterwarnings("ignore", "invalid value encountered in sqrt")


# ==============================================================================
# 1. CONFIGURATION AND TIME-DEPENDENT TRUE FUNCTION
# ==============================================================================
@dataclass
class ActiveLearningConfig:
    """Configuration for the Time-Dependent Active Learning experiment."""
    # --- Spatial Domain (x) ---
    lb_x: float = -1.5
    ub_x: float = 1.5
    num_candidate_pts: int = 200

    # --- Temporal Domain (t) ---
    t_start: float = 0.0
    t_end: float = 1.0
    t_step: float = 0.1

    # --- Active Learning Loop Settings ---
    num_initial_points: int = 4
    num_points_to_add: int = 8
    n_order: int = 1  # Derivative order for the GP models

    # --- GP Model Parameters ---
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"

    # --- Optimizer Settings ---
    n_restarts: int = 10
    swarm_size: int = 50


def true_function(X_spatial, t, alg=oti):
    """
    Time-dependent function with a pitchfork bifurcation at t=0.5.
    The function has one peak for t < 0.5 and splits into two for t > 0.5.
    """
    x = X_spatial[:, 0]
    
    # FIX: Ensure t is a 1D array for element-wise operations to prevent broadcasting errors
    t = t
    
    # The bifurcation parameter 'r' is controlled by time
    r = 4 * (t - 0.5)
    
    # Now, r and x are both 1D arrays, so multiplication is element-wise
    return r * x + 2 * alg.sin(4 * np.pi * x) - 10 * x**3


def true_function_vect(X_spatial, t, alg=oti):
    """
    Time-dependent function with a pitchfork bifurcation at t=0.5.
    The function has one peak for t < 0.5 and splits into two for t > 0.5.
    """
    x = X_spatial[:, 0]
    
    # FIX: Ensure t is a 1D array for element-wise operations to prevent broadcasting errors
    t = np.asarray(t).ravel()
    
    # The bifurcation parameter 'r' is controlled by time
    r = 4 * (t - 0.5)
    
    # Now, r and x are both 1D arrays, so multiplication is element-wise
    return r * x + 2 * alg.sin(4 * np.pi * x) - 10 * x**3

# ==============================================================================
# 2. HELPER FUNCTION
# ==============================================================================

def get_training_data_for_model(X_points: np.ndarray, t: float, n_order: int):
    """Generates derivative observations for a given set of points at a specific time t."""
    der_indices = utils.gen_OTI_indices(1, n_order)
    # Perturb only the spatial dimension 'x'
    X_pert = oti.array(X_points) + oti.e(1, order=n_order)
    y_hc = true_function(X_pert, t)

    y_list = [y_hc.real]
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            derivative = y_hc.get_deriv(der_indices[i][j]).reshape(-1, 1)
            y_list.append(derivative)
    return y_list


# ==============================================================================
# 3. VISUALIZATION FUNCTION
# ==============================================================================

def visualize_time_dependent_results(history, config, final_gps):
    """
    Creates a final summary plot with five panels, including absolute error.
    """
    # --- Extract final training data from history ---
    final_train_x = history[-1]['X_train_spatial']
    final_train_t = history[-1]['time_of_points']

    # --- Separate initial points from actively learned points ---
    num_initial = config.num_initial_points
    initial_x = final_train_x[:num_initial]
    initial_t = final_train_t[:num_initial]
    
    active_x = final_train_x[num_initial:]
    active_t = final_train_t[num_initial:]

    imse_history = [h['best_score'] for h in history if h['best_score'] is not None]

    # Create the grid for plotting
    x_grid = np.linspace(config.lb_x, config.ub_x, 200)
    t_grid = np.arange(config.t_start, config.t_end + config.t_step, config.t_step)
    xx, tt = np.meshgrid(x_grid, t_grid)
    
    # Evaluate true function and final GP predictions on the grid
    true_surface = true_function_vect(xx.reshape(-1, 1), tt, alg=np).reshape(xx.shape)
    pred_surface = np.zeros_like(xx)
    var_surface = np.zeros_like(xx)

    for i, t in enumerate(t_grid):
        gp = final_gps[i]
        params = gp.params
        pred, var = gp.predict(x_grid.reshape(-1, 1), params, calc_cov=True, return_deriv=False)
        pred_surface[i, :] = pred.ravel()
        var_surface[i, :] = var.ravel()

    # --- NEW: Calculate the absolute error surface ---
    abs_error_surface = np.abs(true_surface - pred_surface)

    # --- MODIFIED: Create a 3x2 grid for 5 plots ---
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 2])

    # --- Panel 1: True Function ---
    ax1 = fig.add_subplot(gs[0, 0])
    c1 = ax1.pcolormesh(xx, tt, true_surface, shading='auto', cmap='viridis')
    ax1.plot(initial_x, initial_t, 'x', color='cyan', markersize=10, mew=2.5, label='Initial Points')
    ax1.plot(active_x, active_t, 'o', color='red', markersize=8, markeredgecolor='white', label='Active Points')
    ax1.set_title('True Function $f(x, t)$ with Sample Points', fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('time (t)')
    ax1.legend()
    fig.colorbar(c1, ax=ax1)

    # --- Panel 2: GP Mean Prediction ---
    ax2 = fig.add_subplot(gs[0, 1])
    c2 = ax2.pcolormesh(xx, tt, pred_surface, shading='auto', cmap='viridis')
    ax2.set_title('Final GP Mean Prediction $\\mu(x, t)$', fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('time (t)')
    fig.colorbar(c2, ax=ax2)

    # --- Panel 3: GP Variance ---
    ax3 = fig.add_subplot(gs[1, 0])
    c3 = ax3.pcolormesh(xx, tt, np.log10(var_surface), shading='auto', cmap='plasma')
    ax3.plot(initial_x, initial_t, 'x', color='cyan', markersize=10, mew=2.5)
    ax3.plot(active_x, active_t, 'o', color='white', markersize=8, markeredgecolor='black')
    ax3.set_title('Final GP Log-Variance $\\log_{10}(\\sigma^2(x, t))$', fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('time (t)')
    fig.colorbar(c3, ax=ax3, label='Log Variance')
    
    # --- NEW: Panel 4: Absolute Error ---
    ax_err = fig.add_subplot(gs[1, 1])
    c_err = ax_err.pcolormesh(xx, tt, abs_error_surface, shading='auto', cmap='inferno')
    ax_err.plot(initial_x, initial_t, 'x', color='cyan', markersize=10, mew=2.5)
    ax_err.plot(active_x, active_t, 'o', color='white', markersize=8, markeredgecolor='black')
    ax_err.set_title('Absolute Error $|f(x,t) - \\mu(x,t)|$', fontweight='bold')
    ax_err.set_xlabel('x')
    ax_err.set_ylabel('time (t)')
    fig.colorbar(c_err, ax=ax_err, label='Absolute Error')

    # --- MODIFIED: Panel 5: IMSE Reduction History (spans bottom row) ---
    ax4 = fig.add_subplot(gs[2, :])
    num_points = [config.num_initial_points + i for i in range(len(imse_history))]
    ax4.plot(num_points, imse_history, 'o-', color='dodgerblue', markersize=8, lw=2.5)
    ax4.set_title('Best IMSE Reduction per Iteration', fontweight='bold')
    ax4.set_xlabel('Number of Training Points')
    ax4.set_ylabel('Max IMSE Reduction Score')
    ax4.set_yscale('log')
    ax4.grid(True, which="both", ls="--", alpha=0.7)
    if num_points:
        ax4.set_xticks(num_points)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 4. MAIN ACTIVE LEARNING LOOP
# ==============================================================================

def main():
    config = ActiveLearningConfig()
    acquisition_function_to_use = acq.imse_reduction_efficient

    # --- Setup ---
    domain_bounds = [(config.lb_x, config.ub_x)]
    time_steps = np.arange(config.t_start, config.t_end + config.t_step, config.t_step)
    integration_points = np.linspace(config.lb_x, config.ub_x, config.num_candidate_pts).reshape(-1, 1)
    
    # Initial spatial training points
    X_train_spatial = np.linspace(config.lb_x, config.ub_x, config.num_initial_points).reshape(-1, 1)
    
    # --- FIX 1: Initialize time_of_points ONCE before the loop ---
    time_of_points = np.full(X_train_spatial.shape[0], config.t_start)
    
    history = []
    print("=" * 70)
    print("Starting Time-Dependent Active Learning")
    print(f"Spatial Domain (x): [{config.lb_x}, {config.ub_x}]")
    print(f"Temporal Domain (t): [{config.t_start}, {config.t_end}] with {len(time_steps)} slices")
    print("=" * 70)

    # The loop now runs up to num_points_to_add, not +1
    for i in range(config.num_points_to_add):
        num_pts = config.num_initial_points + i
        print(f"\n--- Iteration {i+1}/{config.num_points_to_add} | Current Training Points: {num_pts} ---")

        candidate_points_info = []

        # --- Inner loop: Train a "scout" GP for each time slice ---
        for t in time_steps:
            y_train_list = get_training_data_for_model(X_train_spatial, t, config.n_order)
            
            der_indices = utils.gen_OTI_indices(1, config.n_order)
            gp = degp(
                X_train_spatial, y_train_list, config.n_order, n_bases=1, der_indices=der_indices,
                normalize=config.normalize_data, kernel=config.kernel, kernel_type=config.kernel_type
            )
            params = gp.optimize_hyperparameters(
                n_restart_optimizer=config.n_restarts, swarm_size=config.swarm_size, verbose=False
            )
            
            # Find the best point for THIS time slice
            _, y_var = gp.predict(integration_points, params, calc_cov=True, return_deriv=False)
            
            next_point, score = utils.find_next_point(
                gp, params, X_train_spatial, y_train_list, y_var, integration_points,
                domain_bounds=domain_bounds, acquisition_func=acquisition_function_to_use,
                n_coarse_points=8, n_local_starts=8
            )
            candidate_points_info.append({'t': t, 'next_x': next_point, 'score': score})

        # --- Global Point Selection ---
        best_candidate = max(candidate_points_info, key=lambda item: item['score'])
        next_point_to_add = best_candidate['next_x']
        best_overall_score = best_candidate['score']
        time_of_best_point = best_candidate['t']
        
        print(f"Highest IMSE reduction found at t={time_of_best_point:.3f}")
        print(f"-> Globally chosen next point: x = {next_point_to_add.item():.4f}\n")

        # --- FIX 2: Store the CURRENT state to history BEFORE updating ---
        history.append({
            "X_train_spatial": X_train_spatial.copy(),
            "time_of_points": time_of_points.copy(),
            "best_score": best_overall_score,
        })
        
        # --- FIX 3: Update state for the NEXT iteration AFTER saving history ---
        X_train_spatial = np.vstack([X_train_spatial, next_point_to_add])
        time_of_points = np.append(time_of_points, time_of_best_point)

    # Append the final state to the history for the final plot
    history.append({
        "X_train_spatial": X_train_spatial.copy(),
        "time_of_points": time_of_points.copy(),
        "best_score": None, # No score for the final state
    })
    
    # --- Final Model Training ---
    print("\n--- Active learning complete. Training final models for visualization... ---")
    final_gps = []
    for t in time_steps:
        y_train_list = get_training_data_for_model(X_train_spatial, t, config.n_order)
        der_indices = utils.gen_OTI_indices(1, config.n_order)
        gp = degp(X_train_spatial, y_train_list, config.n_order, n_bases=1, der_indices=der_indices,
                  normalize=config.normalize_data, kernel=config.kernel, kernel_type=config.kernel_type)
        gp.optimize_hyperparameters(n_restart_optimizer=config.n_restarts, swarm_size=config.swarm_size, verbose=False)
        final_gps.append(gp)

    print("=" * 70)
    print("Creating final visualization...")
    visualize_time_dependent_results(history, config, final_gps)
if __name__ == "__main__":
    main()