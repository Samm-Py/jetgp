"""
================================================================================
DEGP Tutorial: Derivative-Enhanced Gaussian Process for 1D Function Approximation
================================================================================

This tutorial demonstrates how to use derivative information to enhance Gaussian 
Process regression. We'll show how including specific derivative orders can 
dramatically improve predictions, especially with limited training data.

Key concepts covered:
- Hypercomplex automatic differentiation for derivative computation
- Selective derivative inclusion in GP training
- Hyperparameter optimization
- Performance evaluation and visualization

The example uses a complex 1D function with exponential decay, oscillations, 
and linear trends to showcase DEGP capabilities.
================================================================================
"""

import numpy as np
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
import plotting_helper
import time
from typing import List, Tuple, Optional

def true_function(X, alg=oti):
    """
    Complex test function combining multiple mathematical components.
    
    This function is designed to be challenging for standard GP regression
    due to its combination of:
    - Exponential decay: exp(-x)
    - Oscillatory behavior: sin(x) + cos(3x)  
    - Linear trend: 0.2x + 1.0
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 1)
        Input points
    alg : module
        Numerical library (numpy or pyoti for automatic differentiation)
        
    Returns:
    --------
    y : array-like
        Function values
    """
    x = X[:, 0]
    f = alg.exp(-x) + alg.sin(x) + alg.cos(3 * x) + 0.2 * x + 1.0
    return f

def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                             y_var: np.ndarray) -> dict:
    """
    Compute comprehensive performance metrics.
    
    Parameters:
    -----------
    y_true : array
        True function values
    y_pred : array  
        Predicted values
    y_var : array
        Prediction variances
        
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    # Flatten arrays for consistent computation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    y_var_flat = y_var.flatten()
    
    # Basic error metrics
    mse = np.mean((y_true_flat - y_pred_flat)**2)
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    rmse = np.sqrt(mse)
    
    # Normalized metrics
    nrmse = utils.nrmse(y_true_flat, y_pred_flat)
    max_error = np.max(np.abs(y_true_flat - y_pred_flat))
    
    # Uncertainty metrics
    mean_uncertainty = np.mean(np.sqrt(y_var_flat))
    uncertainty_range = np.max(np.sqrt(y_var_flat)) - np.min(np.sqrt(y_var_flat))
    
    # Coverage probability (approximate)
    std_pred = np.sqrt(y_var_flat)
    within_2sigma = np.mean(np.abs(y_true_flat - y_pred_flat) <= 2 * std_pred)
    
    return {
        'mse': mse,
        'mae': mae, 
        'rmse': rmse,
        'nrmse': nrmse,
        'max_error': max_error,
        'mean_uncertainty': mean_uncertainty,
        'uncertainty_range': uncertainty_range,
        'coverage_2sigma': within_2sigma
    }

def main():
    """
    Main tutorial execution with detailed explanations and error handling.
    """
    print("DEGP Tutorial: Enhanced Gaussian Process with Derivatives")
    print("=" * 65)
    
    # ==========================================================================
    # Configuration Parameters
    # ==========================================================================
    
    # Derivative order configuration
    n_order = 4  # Maximum derivative order for perturbation
    n_bases = 1  # Input dimensionality (1D example)
    
    # Domain configuration  
    lb_x, ub_x = 0.2, 6.0  # Domain bounds
    num_points = 6          # Number of training points
    
    # Test grid for evaluation
    N_grid = 100
    
    print(f"Configuration:")
    print(f"  Domain: [{lb_x}, {ub_x}]")
    print(f"  Training points: {num_points}")
    print(f"  Maximum derivative order: {n_order}")
    print(f"  Test grid points: {N_grid}")
    
    # ==========================================================================
    # Derivative Selection Strategy
    # ==========================================================================
    
    # Custom derivative indices: include 1st and 4th order derivatives
    # Format: [[[[variable_index, derivative_order]]]]
    # This selective approach reduces computational cost while capturing
    # both local slope information (1st) and higher-order curvature (4th)
    der_indices = [[[[1, 1]], [[1, 4]]]]
    
    print(f"\nDerivative Selection:")
    print(f"  Including derivatives: 1st order (local slope)")
    print(f"                        4th order (higher curvature)")
    print(f"  Derivative indices format: {der_indices}")
    
    # ==========================================================================
    # Training Data Generation
    # ==========================================================================
    
    print(f"\nGenerating Training Data...")
    start_time = time.time()
    
    # Create evenly spaced training points
    X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)
    print(f"  Training locations: {X_train.ravel()}")
    
    # Setup hypercomplex perturbation for automatic differentiation
    X_train_pert = oti.array(X_train)
    for i in range(1, n_bases + 1):
        X_train_pert[:, i - 1] = X_train_pert[:, i - 1] + oti.e(i, order=n_order)
    
    # Evaluate function with all derivatives up to n_order
    y_train_hc = true_function(X_train_pert)
    
    # Extract function values and selected derivatives
    y_train = [y_train_hc.real]  # Function values
    total_derivative_observations = 0
    
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            derivative_data = y_train_hc.get_deriv(der_indices[i][j]).reshape(-1, 1)
            y_train.append(derivative_data)
            total_derivative_observations += len(derivative_data)
    
    data_gen_time = time.time() - start_time
    print(f"  Function observations: {len(y_train[0])}")
    print(f"  Derivative observations: {total_derivative_observations}")
    print(f"  Total training observations: {len(y_train[0]) + total_derivative_observations}")
    print(f"  Data generation time: {data_gen_time:.3f}s")
    
    # ==========================================================================
    # DEGP Model Setup and Training
    # ==========================================================================
    
    print(f"\nSetting up DEGP Model...")
    
    try:
        # Initialize DEGP model
        gp = degp(
            X_train,           # Training inputs
            y_train,           # Training outputs (function + derivatives)
            n_order,           # Maximum derivative order
            n_bases,           # Input dimensionality  
            der_indices,       # Which derivatives to include
            normalize=True,    # Normalize inputs/outputs
            kernel="SE",       # Squared Exponential kernel
            kernel_type="anisotropic"  # Allow different length scales per dimension
        )
        
        print("  Model initialization: SUCCESS")
        print(f"  Kernel: Squared Exponential (anisotropic)")
        print(f"  Normalization: Enabled")
        
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
        params = gp.optimize_hyperparameters(
            n_restart_optimizer=25,  # Number of optimization restarts
            swarm_size=100          # PSO swarm size
        )
        
        optimization_time = time.time() - optimization_start
        print(f"  Optimization time: {optimization_time:.2f}s")
        print(f"  Optimization: SUCCESS")
        
        # Display optimized parameters (if accessible)
        if hasattr(params, '__len__') and len(params) > 0:
            print(f"  Optimized parameters: {len(params)} values")
        
    except Exception as e:
        print(f"  Optimization: FAILED")
        print(f"  Error: {e}")
        return
    
    # ==========================================================================
    # Model Prediction
    # ==========================================================================
    
    print(f"\nMaking Predictions...")
    prediction_start = time.time()
    
    # Create test grid
    X_test = np.linspace(lb_x, ub_x, N_grid).reshape(-1, 1)
    
    try:
        # Generate predictions with uncertainty quantification
        y_pred, y_var = gp.predict(
            X_test,              # Test inputs
            params,              # Optimized hyperparameters
            calc_cov=True,       # Calculate full covariance
            return_deriv=False   # Only return function predictions
        )
        
        prediction_time = time.time() - prediction_start
        print(f"  Prediction time: {prediction_time:.3f}s")
        print(f"  Predictions generated: {len(y_pred)}")
        
    except Exception as e:
        print(f"  Prediction: FAILED")
        print(f"  Error: {e}")
        return
    
    # ==========================================================================
    # Performance Evaluation
    # ==========================================================================
    
    print(f"\nEvaluating Model Performance...")
    
    # Compute true function values for comparison
    y_true = true_function(X_test, alg=np)
    
    # Calculate comprehensive metrics
    metrics = evaluate_model_performance(y_true, y_pred, y_var)
    
    print(f"\nPerformance Metrics:")
    print(f"  NRMSE:           {metrics['nrmse']:.6f}")
    print(f"  RMSE:            {metrics['rmse']:.6f}")
    print(f"  MAE:             {metrics['mae']:.6f}")
    print(f"  Max Error:       {metrics['max_error']:.6f}")
    print(f"  Mean Uncertainty: {metrics['mean_uncertainty']:.6f}")
    print(f"  95% Coverage:    {metrics['coverage_2sigma']:.3f}")
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    
    print(f"\nGenerating Visualizations...")
    
    try:
        # Create plots comparing predictions with ground truth
        plotting_helper.make_plots(
            X_train,                    # Training inputs
            y_train,                    # Training outputs
            X_test,                     # Test inputs  
            y_pred.flatten(),           # Predictions
            true_function,              # True function
            cov=y_var,                  # Prediction variance
            n_order=n_order,            # Derivative order
            n_bases=n_bases,            # Input dimensionality
            plot_derivative_surrogates=False,  # Focus on function approximation
            der_indices=der_indices     # Derivative configuration
        )
        
        print("  Visualization: SUCCESS")
        
    except Exception as e:
        print(f"  Visualization: FAILED") 
        print(f"  Error: {e}")
    
    # ==========================================================================
    # Tutorial Summary
    # ==========================================================================
    
    total_time = time.time() - start_time
    
    print(f"\nTutorial Summary:")
    print(f"=" * 50)
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Final NRMSE: {metrics['nrmse']:.6f}")
    print(f"\nKey Takeaways:")
    print(f"• DEGP leverages derivative information for better predictions")
    print(f"• Selective derivative inclusion balances accuracy vs. cost") 
    print(f"• With {num_points} training points, achieved {metrics['nrmse']:.4f} NRMSE")
    print(f"• Uncertainty quantification provides confidence bounds")
    print(f"\nNext steps:")
    print(f"• Try different derivative combinations")
    print(f"• Experiment with various kernel types")
    print(f"• Apply to your own functions/datasets")

if __name__ == "__main__":
    main()