"""
================================================================================
DEGP Tutorial: 2D Derivative-Enhanced Gaussian Process Regression
================================================================================

This tutorial demonstrates how to apply derivative-enhanced Gaussian Process 
regression to two-dimensional functions. We'll explore how partial derivatives
in multiple dimensions can dramatically improve function approximation with
limited training data.

Key concepts covered:
- 2D hypercomplex automatic differentiation
- Partial derivative extraction (∂f/∂x₁, ∂f/∂x₂, ∂²f/∂x₁∂x₂, etc.)
- Multi-dimensional GP regression with derivative constraints
- Performance comparison strategies
- 2D visualization of results

The example uses a complex 2D function combining polynomial and oscillatory
terms to showcase DEGP capabilities in higher dimensions.
================================================================================
"""

import numpy as np
import pyoti.sparse as oti
import itertools
from full_degp.degp import degp
import utils
import plotting_helper
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict

def true_function(X, alg=oti):
    """
    Complex 2D test function with polynomial and oscillatory components.
    
    This function combines:
    - Polynomial term: x₁²x₂ (provides smooth variation)
    - High-frequency oscillations: cos(10x₁) + cos(10x₂) (challenging features)
    
    This combination tests the GP's ability to handle both smooth trends
    and rapid oscillations in 2D space.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Input points with columns [x1, x2]
    alg : module
        Numerical library (numpy or pyoti for automatic differentiation)
        
    Returns:
    --------
    y : array-like
        Function values
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    return x1**2 * x2 + alg.cos(10 * x1) + alg.cos(10 * x2)

def analyze_derivative_structure(der_indices: List, n_bases: int) -> Dict:
    """
    Analyze the structure of derivatives being used.
    
    Parameters:
    -----------
    der_indices : list
        Derivative indices structure
    n_bases : int
        Number of input dimensions
        
    Returns:
    --------
    analysis : dict
        Analysis of derivative structure
    """
    total_derivatives = 0
    derivative_types = []
    
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            deriv_spec = der_indices[i][j]
            total_derivatives += 1
            
            # Parse derivative specification
            if len(deriv_spec) == 1:
                # Single variable derivative
                var_idx, order = deriv_spec[0]
                derivative_types.append(f"∂^{order}/∂x_{var_idx}^{order}")
            else:
                # Mixed partial derivative  
                desc = "∂^{len(deriv_spec)}/"
                for var_idx, order in deriv_spec:
                    desc += f"∂x_{var_idx}^{order}"
                derivative_types.append(desc)
    
    return {
        'total_count': total_derivatives,
        'types': derivative_types,
        'derivatives_per_point': total_derivatives
    }

def evaluate_2d_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                           X_test: np.ndarray, grid_shape: Tuple[int, int]) -> Dict:
    """
    Compute comprehensive performance metrics for 2D function approximation.
    
    Parameters:
    -----------
    y_true : array
        True function values
    y_pred : array
        Predicted values  
    X_test : array
        Test input points
    grid_shape : tuple
        Shape of the evaluation grid (N_grid, N_grid)
        
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    # Flatten arrays for consistent computation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Basic error metrics
    mse = np.mean((y_true_flat - y_pred_flat)**2)
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    rmse = np.sqrt(mse)
    nrmse = utils.nrmse(y_true_flat, y_pred_flat)
    max_error = np.max(np.abs(y_true_flat - y_pred_flat))
    
    # Spatial error distribution
    errors = np.abs(y_true_flat - y_pred_flat).reshape(grid_shape)
    mean_error_by_region = {
        'corners': np.mean([errors[0,0], errors[0,-1], errors[-1,0], errors[-1,-1]]),
        'edges': np.mean([np.mean(errors[0,1:-1]), np.mean(errors[-1,1:-1]), 
                         np.mean(errors[1:-1,0]), np.mean(errors[1:-1,-1])]),
        'center': np.mean(errors[grid_shape[0]//4:3*grid_shape[0]//4, 
                                grid_shape[1]//4:3*grid_shape[1]//4])
    }
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse, 
        'nrmse': nrmse,
        'max_error': max_error,
        'spatial_errors': mean_error_by_region,
        'error_std': np.std(y_true_flat - y_pred_flat)
    }

def create_training_grid(lb_x: float, ub_x: float, lb_y: float, ub_y: float, 
                        num_points: int, strategy: str = 'uniform') -> np.ndarray:
    """
    Create 2D training grid with different sampling strategies.
    
    Parameters:
    -----------
    lb_x, ub_x : float
        X-dimension bounds
    lb_y, ub_y : float  
        Y-dimension bounds
    num_points : int
        Number of points per dimension
    strategy : str
        Sampling strategy ('uniform', 'chebyshev', 'random')
        
    Returns:
    --------
    X_train : array
        Training input points
    """
    if strategy == 'uniform':
        x_vals = np.linspace(lb_x, ub_x, num_points)
        y_vals = np.linspace(lb_y, ub_y, num_points)
        return np.array(list(itertools.product(x_vals, y_vals)))
    
    elif strategy == 'chebyshev':
        # Chebyshev nodes for better conditioning
        k = np.arange(1, num_points + 1)
        x_cheb = 0.5 * (lb_x + ub_x) + 0.5 * (ub_x - lb_x) * np.cos((2*k - 1) * np.pi / (2 * num_points))
        y_cheb = 0.5 * (lb_y + ub_y) + 0.5 * (ub_y - lb_y) * np.cos((2*k - 1) * np.pi / (2 * num_points))
        return np.array(list(itertools.product(x_cheb, y_cheb)))
    
    elif strategy == 'random':
        np.random.seed(42)  # For reproducibility
        n_total = num_points ** 2
        x_rand = np.random.uniform(lb_x, ub_x, n_total)
        y_rand = np.random.uniform(lb_y, ub_y, n_total)
        return np.column_stack([x_rand, y_rand])
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

def main():
    """
    Main 2D DEGP tutorial execution with comprehensive analysis.
    """
    print("2D DEGP Tutorial: Multi-Dimensional Function Approximation")
    print("=" * 70)
    
    # ==========================================================================
    # Configuration Parameters
    # ==========================================================================
    
    # Derivative configuration
    n_order = 2      # Maximum derivative order to include
    n_bases = 2      # Number of input dimensions (x₁, x₂)
    
    # Domain configuration  
    lb_x, ub_x = -1, 1
    lb_y, ub_y = -1, 1
    
    # Training configuration
    num_points = 5   # Points per dimension (total: num_points²)
    sampling_strategy = 'uniform'  # 'uniform', 'chebyshev', 'random'
    
    # Test grid configuration
    N_grid = 25      # Grid resolution for evaluation
    
    print(f"Configuration:")
    print(f"  Domain: [{lb_x}, {ub_x}] × [{lb_y}, {ub_y}]")
    print(f"  Training grid: {num_points}×{num_points} = {num_points**2} points")
    print(f"  Sampling strategy: {sampling_strategy}")
    print(f"  Maximum derivative order: {n_order}")
    print(f"  Test grid resolution: {N_grid}×{N_grid}")
    
    # ==========================================================================
    # Derivative Analysis
    # ==========================================================================
    
    # Generate all derivative indices up to n_order
    der_indices = utils.gen_OTI_indices(n_bases, n_order)
    deriv_analysis = analyze_derivative_structure(der_indices, n_bases)
    
    print(f"\nDerivative Structure Analysis:")
    print(f"  Total derivative types: {deriv_analysis['total_count']}")
    print(f"  Derivatives per training point: {deriv_analysis['derivatives_per_point']}")
    print(f"  Included derivatives:")
    for i, deriv_type in enumerate(deriv_analysis['types'], 1):
        print(f"    {i}. {deriv_type}")
    
    # ==========================================================================
    # Training Data Generation
    # ==========================================================================
    
    print(f"\nGenerating Training Data...")
    start_time = time.time()
    
    # Create training grid
    X_train = create_training_grid(lb_x, ub_x, lb_y, ub_y, num_points, sampling_strategy)
    print(f"  Training points shape: {X_train.shape}")
    
    # Setup hypercomplex perturbation for automatic differentiation
    X_train_pert = oti.array(X_train)
    for i in range(n_bases):
        X_train_pert[:, i] += oti.e(i + 1, order=n_order)
    
    # Evaluate function with all derivatives up to n_order
    y_train_hc = true_function(X_train_pert)
    y_train_real = y_train_hc.real
    
    # Extract function values and all derivatives
    y_train = [y_train_real]
    total_derivative_obs = 0
    
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            deriv = y_train_hc.get_deriv(der_indices[i][j])
            y_train.append(deriv)
            total_derivative_obs += len(deriv)
    
    data_gen_time = time.time() - start_time
    print(f"  Function observations: {len(y_train[0])}")
    print(f"  Derivative observations: {total_derivative_obs}")
    print(f"  Total training observations: {len(y_train[0]) + total_derivative_obs}")
    print(f"  Data generation time: {data_gen_time:.3f}s")
    
    # ==========================================================================
    # DEGP Model Setup and Training
    # ==========================================================================
    
    print(f"\nSetting up 2D DEGP Model...")
    
    try:
        # Initialize DEGP model
        gp = degp(
            X_train,           # Training inputs (2D)
            y_train,           # Training outputs (function + all derivatives)
            n_order,           # Maximum derivative order
            n_bases,           # Input dimensionality (2D)
            der_indices,       # All derivatives up to n_order
            normalize=True,    # Normalize inputs/outputs
            kernel="SE",       # Squared Exponential kernel
            kernel_type="anisotropic"  # Different length scales per dimension
        )
        
        print("  Model initialization: SUCCESS")
        print(f"  Kernel: Squared Exponential (anisotropic)")
        print(f"  Input normalization: Enabled")
        
    except Exception as e:
        print(f"  Model initialization: FAILED")
        print(f"  Error: {e}")
        return
    
    # ==========================================================================
    # Hyperparameter Optimization
    # ==========================================================================
    
    print(f"\nOptimizing Hyperparameters...")
    optimization_start = time.time()
    
    try:
        # Optimize using particle swarm optimization
        # Increased swarm size for 2D complexity
        params = gp.optimize_hyperparameters(
            n_restart_optimizer=15,  # Number of optimization restarts
            swarm_size=300          # Larger swarm for 2D problems
        )
        
        optimization_time = time.time() - optimization_start
        print(f"  Optimization time: {optimization_time:.2f}s")
        print(f"  Optimization: SUCCESS")
        
    except Exception as e:
        print(f"  Optimization: FAILED")
        print(f"  Error: {e}")
        return
    
    # ==========================================================================
    # Model Prediction
    # ==========================================================================
    
    print(f"\nGenerating 2D Predictions...")
    prediction_start = time.time()
    
    # Create 2D test grid
    x_lin = np.linspace(lb_x, ub_x, N_grid)
    y_lin = np.linspace(lb_y, ub_y, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    
    print(f"  Test grid points: {X_test.shape[0]} ({N_grid}×{N_grid})")
    
    try:
        # Generate predictions
        y_pred = gp.predict(
            X_test,              # Test inputs (2D grid)
            params,              # Optimized hyperparameters
            calc_cov=False,      # Skip covariance for speed in 2D
            return_deriv=False   # Only return function predictions
        )
        
        prediction_time = time.time() - prediction_start
        print(f"  Prediction time: {prediction_time:.3f}s")
        print(f"  Predictions shape: {y_pred.shape}")
        
    except Exception as e:
        print(f"  Prediction: FAILED")
        print(f"  Error: {e}")
        return
    
    # ==========================================================================
    # Performance Evaluation
    # ==========================================================================
    
    print(f"\nEvaluating 2D Model Performance...")
    
    # Compute true function values for comparison
    y_true = true_function(X_test, alg=np)
    
    # Calculate comprehensive 2D metrics
    metrics = evaluate_2d_performance(y_true, y_pred, X_test, (N_grid, N_grid))
    
    print(f"\nPerformance Metrics:")
    print(f"  NRMSE:           {metrics['nrmse']:.6f}")
    print(f"  RMSE:            {metrics['rmse']:.6f}")
    print(f"  MAE:             {metrics['mae']:.6f}")
    print(f"  Max Error:       {metrics['max_error']:.6f}")
    print(f"  Error Std:       {metrics['error_std']:.6f}")
    
    print(f"\nSpatial Error Analysis:")
    print(f"  Corner regions:  {metrics['spatial_errors']['corners']:.6f}")
    print(f"  Edge regions:    {metrics['spatial_errors']['edges']:.6f}")
    print(f"  Center region:   {metrics['spatial_errors']['center']:.6f}")
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    
    print(f"\nGenerating 2D Visualizations...")
    
    try:
        # Create plots comparing predictions with ground truth
        plotting_helper.make_plots(
            X_train,                    # Training inputs
            y_train,                    # Training outputs  
            X_test,                     # Test inputs
            y_pred,                     # Predictions
            true_function,              # True function
            X1_grid=X1_grid,           # X mesh for plotting
            X2_grid=X2_grid,           # Y mesh for plotting
            n_order=n_order,            # Derivative order
            n_bases=n_bases,            # Input dimensionality
            plot_derivative_surrogates=False,  # Focus on function approximation
            der_indices=der_indices     # Derivative configuration
        )
        
        print("  2D Visualization: SUCCESS")
        
    except Exception as e:
        print(f"  Visualization: FAILED")
        print(f"  Error: {e}")
        print("  Continuing without plots...")
    
    # ==========================================================================
    # Tutorial Summary and Insights
    # ==========================================================================
    
    total_time = time.time() - start_time
    observations_per_point = 1 + deriv_analysis['derivatives_per_point']
    total_observations = X_train.shape[0] * observations_per_point
    
    print(f"\n2D DEGP Tutorial Summary:")
    print(f"=" * 60)
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Final NRMSE: {metrics['nrmse']:.6f}")
    print(f"Training efficiency: {total_observations} observations from {X_train.shape[0]} points")
    print(f"Observation multiplier: {observations_per_point}x (due to derivatives)")
    
    print(f"\n2D-Specific Insights:")
    print(f"• Partial derivatives provide directional information")
    print(f"• Mixed derivatives (∂²f/∂x₁∂x₂) capture interaction effects")
    print(f"• {num_points}²={num_points**2} points → {total_observations} total observations")
    print(f"• Spatial error varies: center vs. boundary performance")
    
    print(f"\nKey Advantages of 2D DEGP:")
    print(f"• Captures anisotropic function behavior")  
    print(f"• Handles complex interaction terms")
    print(f"• Efficient use of limited training data")
    print(f"• Provides gradient information for optimization")
    
    print(f"\nNext Steps for 2D DEGP:")
    print(f"• Try different sampling strategies (Chebyshev, random)")
    print(f"• Experiment with selective derivative inclusion")
    print(f"• Apply to real 2D engineering/science problems")
    print(f"• Compare with standard 2D GP regression")

if __name__ == "__main__":
    main()