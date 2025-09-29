# main_script_time_dependent.py

import warnings
import numpy as np
import matplotlib.pyplot as plt
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
from dataclasses import dataclass
from scipy.stats.qmc import Sobol
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
    num_integration_pts: int = 1000

    # --- Temporal Domain (t) ---
    t_start: float = 0.0
    t_end: float = 1.0
    t_step: float = 0.1

    # --- Active Learning Loop Settings ---
    num_initial_points: int = 5
    num_points_to_add: int = 3
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
    ax1.plot(active_x, np.ones(active_x.shape), 'o', color='red', markersize=8, markeredgecolor='white', label='Active Points')
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
    ax3.plot(active_x, np.ones(active_x.shape), 'o', color='white', markersize=8, markeredgecolor='black')
    ax3.set_title('Final GP Log-Variance $\\log_{10}(\\sigma^2(x, t))$', fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('time (t)')
    fig.colorbar(c3, ax=ax3, label='Log Variance')
    
    # --- NEW: Panel 4: Absolute Error ---
    ax_err = fig.add_subplot(gs[1, 1])
    c_err = ax_err.pcolormesh(xx, tt, abs_error_surface, shading='auto', cmap='inferno')
    ax_err.plot(initial_x, initial_t, 'x', color='cyan', markersize=10, mew=2.5)
    ax_err.plot(active_x, np.ones(active_x.shape), 'o', color='white', markersize=8, markeredgecolor='black')
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

    # --- Spatial Domain Setup ---
    dist = ["U"]
    lower_bounds = np.array([config.lb_x])
    upper_bounds = np.array([config.ub_x])
    dist_params = {'dists': dist, 'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds}
    use_agg_al = True
    acquisition_function_to_use = acq.mse_reduction

    # --- Time Steps ---
    time_steps = np.arange(config.t_start, config.t_end + config.t_step, config.t_step)

    # --- Integration Points ---
    sampler = Sobol(d=1, scramble=True)
    uniform_samples_int = sampler.random(n=config.num_integration_pts)
    integration_points = utils.get_inverse(dist_params, uniform_samples_int)
    integration_points = np.sort(integration_points, axis=0)

    # --- Initial Training Data ---
    uniform_samples_train = sampler.random(n=config.num_initial_points)
    X_train_spatial = utils.get_inverse(dist_params, uniform_samples_train)
    time_of_points = np.full(X_train_spatial.shape[0], config.t_start)

    history = []

    print("=" * 70)
    print("Starting Time-Dependent Active Learning")
    print(f"Spatial Domain: [{config.lb_x}, {config.ub_x}]")
    print(f"Temporal Domain: [{config.t_start}, {config.t_end}] with {len(time_steps)} slices")
    print("=" * 70)

    for i in range(config.num_points_to_add):
        print(f"\n--- Iteration {i+1}/{config.num_points_to_add} | Training Points: {X_train_spatial.shape[0]} ---")

        gp_list = []
        params_list = []
        y_train_list = []

        # --- Train a GP for each time slice ---
        for t in time_steps:
            y_list = get_training_data_for_model(X_train_spatial, t, config.n_order)
            der_indices = utils.gen_OTI_indices(1, config.n_order)

            gp_t = degp(
                X_train_spatial, y_list, config.n_order,
                n_bases=1, der_indices=der_indices,
                normalize=config.normalize_data, kernel=config.kernel, kernel_type=config.kernel_type
            )

            params_t = gp_t.optimize_hyperparameters(
                n_restart_optimizer=config.n_restarts,
                swarm_size=config.swarm_size,
                verbose=False
            )

            gp_list.append(gp_t)
            params_list.append(params_t)
            y_train_list.append(y_list)

        # --- Find the next point using aggregated acquisition function ---
        next_points = utils.find_next_point_batch(
            gp=gp_list,  
            params=params_list,
            dist_params=dist_params,
            acquisition_func=acquisition_function_to_use,
            integration_points=integration_points,
            n_candidate_points=250,
            n_local_starts=1,
            n_batch_points=5,
            seed=i
        )
        

        next_points_to_add = next_points
        best_overall_score = None
        time_of_best_point = 1.0  # or any flag to indicate global selection

        # --- Save history BEFORE updating training data ---
        history.append({
            "X_train_spatial": X_train_spatial.copy(),
            "time_of_points": time_of_points.copy(),
            "best_score": best_overall_score
        })

        # --- Update training data for next iteration ---
        X_train_spatial = np.vstack([X_train_spatial, next_points_to_add])
        time_of_points = np.append(time_of_points, time_of_best_point)

    # --- Final state appended for visualization ---
    history.append({
        "X_train_spatial": X_train_spatial.copy(),
        "time_of_points": time_of_points.copy(),
        "best_score": None
    })

    # --- Train final GPs for visualization ---
    final_gps = []
    for t in time_steps:
        y_train_list = get_training_data_for_model(X_train_spatial, t, config.n_order)
        der_indices = utils.gen_OTI_indices(1, config.n_order)
        gp_t = degp(
            X_train_spatial, y_train_list, config.n_order,
            n_bases=1, der_indices=der_indices,
            normalize=config.normalize_data, kernel=config.kernel, kernel_type=config.kernel_type
        )
        gp_t.optimize_hyperparameters(n_restart_optimizer=config.n_restarts, swarm_size=config.swarm_size, verbose=False)
        final_gps.append(gp_t)

    print("=" * 70)
    print("Active learning complete. Creating final visualization...")
    visualize_time_dependent_results(history, config, final_gps)
if __name__ == "__main__":
    main()