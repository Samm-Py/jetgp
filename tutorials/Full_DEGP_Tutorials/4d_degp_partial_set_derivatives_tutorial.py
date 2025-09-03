"""
================================================================================
DEGP Tutorial: High-Dimensional Function Approximation with 2D Visualization
================================================================================

This tutorial demonstrates how to apply DEGP to high-dimensional functions
(4D in this case) and visualize the results through 2D slices. This approach
is crucial for real-world applications where functions have many inputs but
we need to understand their behavior through lower-dimensional views.

Key concepts covered:
- High-dimensional DEGP with selective derivatives
- Sobol sequence sampling for better space coverage
- Dimensional reduction through 2D slicing
- Scaling challenges and computational strategies
- Visualization techniques for high-dimensional functions
- Performance analysis in high-dimensional spaces

The example uses a 4D polynomial function with interaction terms to demonstrate
how DEGP can capture complex high-dimensional relationships while providing
interpretable 2D visualizations.
================================================================================
"""

import plotting_helper
import sobol as sb
import numpy as np
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import sys
sys.path.append("../../modules/")

def true_function(X, alg=oti):
    """
    Styblinski–Tang function in 4D.
    
    Function: f(x₁,x₂,x₃,x₄) = 0.5 * sum_{i=1}^4 (x_i^4 - 16x_i^2 + 5x_i)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 4)
        Input points with columns [x1, x2, x3, x4]
    alg : module
        Numerical library (numpy or pyoti)
        
    Returns:
    --------
    y : array-like
        Function values
    """
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return 0.5 * (x1**4 - 16*x1**2 + 5*x1 +
                  x2**4 - 16*x2**2 + 5*x2 +
                  x3**4 - 16*x3**2 + 5*x3 +
                  x4**4 - 16*x4**2 + 5*x4)

def analyze_high_dim_derivatives(n_bases: int, n_order: int) -> Dict:
    """
    Analyze the computational complexity of high-dimensional derivatives.
    
    Parameters:
    -----------
    n_bases : int
        Number of input dimensions
    n_order : int
        Maximum derivative order
        
    Returns:
    --------
    analysis : dict
        Analysis of derivative complexity
    """
    # Complete derivative set size grows exponentially
    complete_indices = utils.gen_OTI_indices(n_bases, n_order)
    complete_count = sum(len(complete_indices[i]) for i in range(len(complete_indices)))
    
    # Our selective strategy: only main derivatives
    selective_count = n_bases + n_bases  # First + second order for each dimension
    
    # Computational complexity estimates
    complete_cost = complete_count * (n_bases ** n_order)
    selective_cost = selective_count * n_bases
    
    return {
        'dimensions': n_bases,
        'max_order': n_order,
        'complete_derivative_count': complete_count,
        'selective_derivative_count': selective_count,
        'complexity_reduction': complete_count / selective_count if selective_count > 0 else float('inf'),
        'computational_savings': complete_cost / selective_cost if selective_cost > 0 else float('inf')
    }

def create_slice_strategies(n_bases: int, bounds: List) -> Dict[str, Dict]:
    """
    Create different strategies for 2D slicing of high-dimensional space.
    
    Parameters:
    -----------
    n_bases : int
        Total number of dimensions
    bounds : list
        [lower_bounds, upper_bounds] for each dimension
        
    Returns:
    --------
    strategies : dict
        Different slicing strategies with their configurations
    """
    lower_bounds, upper_bounds = bounds
    
    strategies = {
        'zero_slice': {
            'description': 'Fix unused dimensions at zero',
            'fixed_values': [0.0] * (n_bases - 2),
            'active_dims': [0, 1],
            'slice_point': 'origin'
        },
        
        'center_slice': {
            'description': 'Fix unused dimensions at domain center',
            'fixed_values': [(lower_bounds[i] + upper_bounds[i]) / 2 
                           for i in range(2, n_bases)],
            'active_dims': [0, 1],
            'slice_point': 'center'
        },
        
        'random_slice': {
            'description': 'Fix unused dimensions at random points',
            'fixed_values': [np.random.uniform(lower_bounds[i], upper_bounds[i]) 
                           for i in range(2, n_bases)],
            'active_dims': [0, 1],
            'slice_point': 'random'
        }
    }
    
    return strategies

def evaluate_slice_performance(gp, params, true_function, slice_config: Dict, 
                              bounds: List, N_grid: int = 25) -> Dict:
    """
    Evaluate DEGP performance on a specific 2D slice.
    
    Parameters:
    -----------
    gp : DEGP model
        Trained DEGP model
    params : array
        Optimized hyperparameters
    true_function : callable
        True function for comparison
    slice_config : dict
        Slice configuration
    bounds : list
        Domain bounds
    N_grid : int
        Grid resolution
        
    Returns:
    --------
    results : dict
        Performance metrics for this slice
    """
    lower_bounds, upper_bounds = bounds
    
    # Create 2D test grid
    x_lin = np.linspace(lower_bounds[0], upper_bounds[0], N_grid)
    y_lin = np.linspace(lower_bounds[1], upper_bounds[1], N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    
    # Create full test points with fixed dimensions
    X_test = np.zeros((N_grid**2, len(lower_bounds)))
    X_test[:, 0] = X1_grid.ravel()
    X_test[:, 1] = X2_grid.ravel()
    
    # Set fixed dimensions
    for i, val in enumerate(slice_config['fixed_values']):
        X_test[:, i + 2] = val
    
    # Make predictions
    y_pred, y_var = gp.predict(X_test, params, calc_cov=True, return_deriv=False)
    y_true = true_function(X_test, alg=np).flatten()
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred.flatten())**2)
    nrmse = utils.nrmse(y_true, y_pred.flatten())
    max_error = np.max(np.abs(y_true - y_pred.flatten()))
    mean_uncertainty = np.mean(np.sqrt(y_var.flatten()))
    
    return {
        'slice_type': slice_config['slice_point'],
        'fixed_values': slice_config['fixed_values'],
        'mse': mse,
        'nrmse': nrmse,
        'max_error': max_error,
        'mean_uncertainty': mean_uncertainty,
        'X_test': X_test,
        'y_pred': y_pred,
        'y_true': y_true,
        'X1_grid': X1_grid,
        'X2_grid': X2_grid
    }

def main():
    """
    Main high-dimensional DEGP tutorial with comprehensive analysis.
    """
    print("High-Dimensional DEGP Tutorial: 4D Function with 2D Visualization")
    print("=" * 75)
    
    # ==========================================================================
    # Configuration
    # ==========================================================================
    
    np.random.seed(1354)  # For reproducibility
    
    n_bases = 4           # Input dimensionality (4D function)
    n_order = 2          # Maximum derivative order
    num_points_train = 25 # Training points (sparse sampling)
    
    # Domain bounds
    lower_bounds = [-5] * n_bases
    upper_bounds = [5] * n_bases
    
    N_grid = 25          # 2D slice resolution
    
    print(f"Configuration:")
    print(f"  Function dimensionality: {n_bases}D")
    print(f"  Domain bounds: [{lower_bounds[0]}, {upper_bounds[0]}] per dimension")
    print(f"  Training points: {num_points_train}")
    print(f"  Maximum derivative order: {n_order}")
    print(f"  2D slice resolution: {N_grid}×{N_grid}")
    
    # ==========================================================================
    # High-Dimensional Derivative Analysis
    # ==========================================================================
    
    print(f"\nHigh-Dimensional Derivative Complexity Analysis:")
    print("=" * 55)
    
    complexity_analysis = analyze_high_dim_derivatives(n_bases, n_order)
    
    print(f"Complete derivative set: {complexity_analysis['complete_derivative_count']} derivatives")
    print(f"Selective strategy: {complexity_analysis['selective_derivative_count']} derivatives")
    print(f"Complexity reduction: {complexity_analysis['complexity_reduction']:.1f}x")
    print(f"Computational savings: {complexity_analysis['computational_savings']:.1f}x")
    
    # Define selective derivative strategy
    der_indices = [
        [[[1, 1]], [[2, 1]], [[3, 1]], [[4, 1]]],  # ∂f/∂x₁, ∂f/∂x₂, ∂f/∂x₃, ∂f/∂x₄
        [[[1, 2]], [[2, 2]], [[3, 2]], [[4, 2]]],  # ∂²f/∂x₁², ∂²f/∂x₂², ∂²f/∂x₃², ∂²f/∂x₄²
    ]
    
    print(f"\nSelective Derivative Strategy:")
    print(f"  First-order: ∂f/∂x₁, ∂f/∂x₂, ∂f/∂x₃, ∂f/∂x₄")
    print(f"  Second-order: ∂²f/∂x₁², ∂²f/∂x₂², ∂²f/∂x₃², ∂²f/∂x₄²")
    print(f"  Excluded: All mixed derivatives (e.g., ∂²f/∂x₁∂x₂)")
    print(f"  Rationale: Capture main directional effects without exponential cost")
    
    # ==========================================================================
    # Training Data Generation with Sobol Sampling
    # ==========================================================================
    
    print(f"\nGenerating High-Dimensional Training Data:")
    print("=" * 50)
    
    start_time = time.time()
    
    # Use Sobol sequences for better high-dimensional space coverage
    print(f"  Sampling method: Sobol sequences")
    print(f"  Advantage: Better space-filling than random sampling in high dimensions")
    
    sobol_train = sb.create_sobol_samples(num_points_train, n_bases, 1).T
    X_train = utils.scale_samples(sobol_train, lower_bounds, upper_bounds)
    
    print(f"  Training points shape: {X_train.shape}")
    print(f"  Sample coverage: Each dimension spans [{lower_bounds[0]:.3f}, {upper_bounds[0]:.3f}]")
    
    # Setup hypercomplex perturbation
    X_train_pert = oti.array(X_train)
    for i in range(n_bases):
        X_train_pert[:, i] += oti.e(i + 1, order=n_order)
    
    # Evaluate function and extract derivatives
    y_train_hc = true_function(X_train_pert)
    y_train = [y_train_hc.real]
    
    derivative_obs = 0
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            y_train.append(y_train_hc.get_deriv(der_indices[i][j]))
            derivative_obs += num_points_train
    
    data_gen_time = time.time() - start_time
    
    print(f"  Function observations: {num_points_train}")
    print(f"  Derivative observations: {derivative_obs}")
    print(f"  Total observations: {num_points_train + derivative_obs}")
    print(f"  Observation amplification: {(num_points_train + derivative_obs) / num_points_train:.1f}x")
    print(f"  Data generation time: {data_gen_time:.3f}s")
    
    # ==========================================================================
    # DEGP Model Training
    # ==========================================================================
    
    print(f"\nTraining High-Dimensional DEGP Model:")
    print("=" * 45)
    
    training_start = time.time()
    
    try:
        # Initialize DEGP
        gp = degp(
            X_train, y_train, n_order, n_bases, der_indices,
            normalize=True, kernel="SE", kernel_type="anisotropic"
        )
        
        print(f"  Model initialization: SUCCESS")
        print(f"  Kernel: SE (anisotropic) - allows different length scales per dimension")
        print(f"  Normalization: Enabled - crucial for high-dimensional stability")
        
        # Optimize hyperparameters
        print(f"  Optimizing hyperparameters...")
        opt_start = time.time()
        
        params = gp.optimize_hyperparameters(
            n_restart_optimizer=15,
            swarm_size=250,
            verbose=True  # Reduce output for cleaner tutorial
        )
        
        opt_time = time.time() - opt_start
        training_time = time.time() - training_start
        
        print(f"  Hyperparameter optimization: SUCCESS")
        print(f"  Optimization time: {opt_time:.2f}s")
        print(f"  Total training time: {training_time:.2f}s")
        
    except Exception as e:
        print(f"  Training FAILED: {e}")
        return
    
    # ==========================================================================
    # 2D Slice Analysis
    # ==========================================================================
    
    print(f"\n2D Slice Analysis: Visualizing High-Dimensional Behavior")
    print("=" * 60)
    
    # Create different slicing strategies
    slice_strategies = create_slice_strategies(n_bases, [lower_bounds, upper_bounds])
    
    print(f"Available slicing strategies:")
    for name, config in slice_strategies.items():
        print(f"  {name.upper()}: {config['description']}")
        if 'fixed_values' in config:
            fixed_str = ", ".join([f"x_{i+3}={v:.3f}" for i, v in enumerate(config['fixed_values'])])
            print(f"    Fixed dimensions: {fixed_str}")
    
    # Analyze performance on different slices
    print(f"\nEvaluating performance on different 2D slices...")
    
    slice_results = {}
    
    # Focus on zero slice for main analysis (most interpretable)
    slice_config = slice_strategies['zero_slice']
    print(f"\nMain Analysis: {slice_config['description']}")
    
    result = evaluate_slice_performance(
        gp, params, true_function, slice_config, 
        [lower_bounds, upper_bounds], N_grid
    )
    
    slice_results['zero_slice'] = result
    
    print(f"  2D Slice Performance (x₁-x₂ plane, x₃=x₄=0):")
    print(f"    NRMSE: {result['nrmse']:.6f}")
    print(f"    Max Error: {result['max_error']:.6f}")
    print(f"    Mean Uncertainty: {result['mean_uncertainty']:.6f}")
    
    # ==========================================================================
    # Comparative Analysis: Different Slices
    # ==========================================================================
    
    print(f"\nComparative Slice Analysis:")
    print("-" * 35)
    
    for slice_name, slice_config in slice_strategies.items():
        if slice_name != 'zero_slice':  # Already computed
            result = evaluate_slice_performance(
                gp, params, true_function, slice_config,
                [lower_bounds, upper_bounds], N_grid
            )
            slice_results[slice_name] = result
        
        result = slice_results[slice_name]
        print(f"{slice_name:<15} NRMSE: {result['nrmse']:.6f}, "
              f"Max Error: {result['max_error']:.6f}")
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    
    print(f"\nGenerating 2D Slice Visualization...")
    
    try:
        # Use the zero slice for visualization (most interpretable)
        main_result = slice_results['zero_slice']
        
        plotting_helper.make_plots(
            X_train,
            y_train,
            main_result['X_test'],
            main_result['y_pred'],
            true_function,
            X1_grid=main_result['X1_grid'],
            X2_grid=main_result['X2_grid'],
            n_order=n_order,
            n_bases=n_bases,
            plot_derivative_surrogates=False,
            der_indices=der_indices,
        )
        
        print("  2D Visualization: SUCCESS")
        
    except Exception as e:
        print(f"  Visualization: FAILED - {e}")
    
    # ==========================================================================
    # High-Dimensional DEGP Tutorial Summary
    # ==========================================================================
    
    total_time = time.time() - start_time
    main_nrmse = slice_results['zero_slice']['nrmse']
    
    print(f"\nHigh-Dimensional DEGP Tutorial Summary:")
    print("=" * 55)
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Final NRMSE (x₁-x₂ slice): {main_nrmse:.8f}")
    

if __name__ == "__main__":
    main()