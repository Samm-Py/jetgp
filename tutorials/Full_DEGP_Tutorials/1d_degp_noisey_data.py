"""
================================================================================
DEGP Tutorial: Robustness of Derivative Information Under Noisy Conditions
================================================================================

This tutorial demonstrates the robustness of derivative-enhanced Gaussian Processes
(DEGPs) when derivative observations are corrupted by measurement noise. We compare
DEGP performance against standard GPs to quantify the value of derivative information
even under challenging measurement conditions.

Key concepts demonstrated:
- Impact of derivative measurement noise on GP performance
- Fair comparison methodology: DEGP vs Standard GP with identical function data
- Noise modeling through sigma_data parameter configuration
- Performance quantification under realistic measurement scenarios
- Visualization of prediction uncertainty under noisy derivative conditions

The tutorial uses a challenging test function f(x) = x*sin(x) with rich derivative
structure, perfect function observations, but deliberately corrupted derivative
measurements to simulate realistic experimental conditions.

Scientific Question: How much derivative noise can a DEGP tolerate while still
outperforming a standard GP trained on perfect function observations?
================================================================================
"""

import numpy as np
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
import plotting_helper
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

def configure_experiment():
    """
    Configure experimental parameters for the DEGP robustness study.
    
    Returns:
    --------
    config : dict
        Experimental configuration parameters
    """
    config = {
        # Domain and discretization
        'lb_x': 0,
        'ub_x': 10,
        'num_training_pts': 6,
        'num_test_pts': 100,
        
        # DEGP configuration
        'n_order': 1,      # Include first derivatives
        'n_bases': 1,      # 1D function
        'kernel': "SE",
        'kernel_type': "anisotropic",
        
        # Noise configuration
        'function_noise_ratio': 0.0,    # Perfect function observations
        'derivative_noise_ratio': 0.25,  # 25% noise on derivatives
        
        # Optimization settings
        'n_restarts': 15,
        'swarm_size': 150,
        
        # Reproducibility
        'random_seed': 1
    }
    return config

def true_function(X, alg=oti):
    """
    Test function with challenging derivative structure.
    
    f(x) = x * sin(x)
    
    Properties:
    - f'(x) = sin(x) + x*cos(x)    (complex first derivative)
    - f''(x) = 2*cos(x) - x*sin(x) (rich second derivative structure)
    - Combines polynomial growth with oscillatory behavior
    - Tests GP ability to learn from noisy derivative information
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 1)
        Input points
    alg : module
        Numerical library (numpy for standard evaluation, oti for derivatives)
        
    Returns:
    --------
    y : array-like
        Function values
    """
    x = X[:, 0]
    return x * alg.sin(x)

def generate_training_data(config: Dict, rng: np.random.RandomState) -> Tuple[np.ndarray, List, np.ndarray, List]:
    """
    Generate training data with controlled noise injection.
    
    Parameters:
    -----------
    config : dict
        Experimental configuration
    rng : RandomState
        Random number generator for reproducibility
        
    Returns:
    --------
    X_train : array
        Training input points
    y_train : list
        Training observations [function_values, noisy_derivatives...]
    noise_std : array
        Noise standard deviations for each observation type
    der_indices : list
        OTI derivative index structure
    """
    # Generate candidate training points
    X_candidate = np.linspace(config['lb_x'], config['ub_x'], 1000).reshape(-1, 1)
    training_indices = rng.choice(
        np.arange(X_candidate.shape[0]), 
        size=config['num_training_pts'], 
        replace=False
    )
    X_train = np.sort(X_candidate[training_indices], axis=0)
    X_train[0] = 1.0  # Ensure coverage near boundary
    
    # Setup OTI for automatic differentiation
    X_train_pert = oti.array(X_train)
    for i in range(config['n_bases']):
        X_train_pert[:, i] += oti.e(i + 1, order=config['n_order'])
    
    # Evaluate function and derivatives exactly
    y_hc = true_function(X_train_pert)
    y_func_clean = y_hc.real
    
    # Generate derivative indices
    der_indices = utils.gen_OTI_indices(config['n_bases'], config['n_order'])
    
    # Initialize noise standard deviations
    total_obs = (len(der_indices) + 1) * len(X_train)
    noise_std = np.zeros(total_obs)
    
    # Add noise to function observations
    func_noise_scale = config['function_noise_ratio']
    noise_std[:len(X_train)] = np.abs(y_func_clean.flatten()) * func_noise_scale
    
    y_func_noisy = y_func_clean.copy()
    for i in range(len(y_func_clean)):
        y_func_noisy[i] = y_func_noisy[i] + \
            rng.normal(loc=0.0, scale=np.abs(y_func_noisy[i]) * func_noise_scale, size=1)
    
    # Package training data
    y_train = [y_func_noisy]
    
    # Add noise to derivatives
    deriv_noise_scale = config['derivative_noise_ratio']
    obs_idx = len(X_train)
    
    for i in range(len(der_indices)):
        derivative_order = i + 1
        for j in range(len(der_indices[i])):
            # Extract clean derivative
            deriv_clean = y_hc.get_deriv(der_indices[i][j]).reshape(-1, 1)
            deriv_noisy = deriv_clean.copy()
            
            # Set noise level (scaled by derivative order)
            noise_level = deriv_noise_scale * derivative_order
            noise_std[obs_idx:obs_idx + len(X_train)] = \
                np.abs(deriv_clean.flatten()) * noise_level
            
            # Inject noise
            for k in range(len(deriv_noisy)):
                deriv_noisy[k] = deriv_noisy[k] + \
                    rng.normal(loc=0.0, scale=np.abs(deriv_noisy[k] * noise_level), size=1)
            
            y_train.append(deriv_noisy)
            obs_idx += len(X_train)
    
    return X_train, y_train, noise_std, der_indices

def train_degp_model(X_train: np.ndarray, y_train: List, noise_std: np.ndarray, 
                    der_indices: List, config: Dict) -> Tuple[object, Dict]:
    """
    Train derivative-enhanced Gaussian Process model.
    
    Parameters:
    -----------
    X_train : array
        Training input points
    y_train : list
        Training observations
    noise_std : array
        Noise standard deviations
    der_indices : list
        Derivative index structure
    config : dict
        Configuration parameters
        
    Returns:
    --------
    gp : DEGP model object
    params : dict
        Optimized hyperparameters
    """
    print("Training DEGP with noisy derivatives...")
    
    # Initialize DEGP
    gp = degp(
        X_train,
        y_train,
        config['n_order'],
        config['n_bases'],
        der_indices,
        normalize=True,
        sigma_data=noise_std,  # Critical: informs model about noise levels
        kernel=config['kernel'],
        kernel_type=config['kernel_type'],
    )
    
    # Optimize hyperparameters
    params = gp.optimize_hyperparameters(
        n_restart_optimizer=config['n_restarts'],
        swarm_size=config['swarm_size']
    )
    
    return gp, params

def train_standard_gp_model(X_train: np.ndarray, y_func: np.ndarray, 
                           func_noise_std: np.ndarray, config: Dict) -> Tuple[object, Dict]:
    """
    Train standard Gaussian Process model for comparison.
    
    Uses identical function observations as DEGP but without derivative information.
    
    Parameters:
    -----------
    X_train : array
        Training input points
    y_func : array
        Function observations (same as used in DEGP)
    func_noise_std : array
        Function noise standard deviations
    config : dict
        Configuration parameters
        
    Returns:
    --------
    std_gp : Standard GP model object
    std_params : dict
        Optimized hyperparameters
    """
    print("Training Standard GP for comparison...")
    
    # Initialize standard GP (no derivatives)
    std_gp = degp(
        X_train,
        [y_func],  # Only function values
        0,  # n_order=0 (no derivatives)
        config['n_bases'],
        [],  # empty derivative indices
        normalize=True,
        sigma_data=func_noise_std,
        kernel=config['kernel'],
        kernel_type=config['kernel_type'],
    )
    
    # Optimize hyperparameters
    std_params = std_gp.optimize_hyperparameters(
        n_restart_optimizer=config['n_restarts'],
        swarm_size=config['swarm_size']
    )
    
    return std_gp, std_params

def evaluate_models(X_test: np.ndarray, degp_model: object, degp_params: Dict,
                   std_gp_model: object, std_params: Dict) -> Dict:
    """
    Evaluate both models on test data and compute performance metrics.
    
    Parameters:
    -----------
    X_test : array
        Test input points
    degp_model, std_gp_model : GP model objects
    degp_params, std_params : dict
        Model hyperparameters
        
    Returns:
    --------
    results : dict
        Comprehensive performance comparison
    """
    # DEGP predictions
    y_pred_degp, y_var_degp = degp_model.predict(
        X_test, degp_params, calc_cov=True, return_deriv=False
    )
    
    # Standard GP predictions
    y_pred_std, y_var_std = std_gp_model.predict(
        X_test, std_params, calc_cov=True, return_deriv=False
    )
    
    # True function values for evaluation
    y_true = true_function(X_test, alg=np)
    
    # Compute performance metrics
    degp_nrmse = utils.nrmse(y_true, y_pred_degp)
    std_nrmse = utils.nrmse(y_true, y_pred_std)
    improvement = (std_nrmse - degp_nrmse) / std_nrmse * 100
    
    # Uncertainty quantification metrics
    degp_mean_uncertainty = np.mean(np.sqrt(y_var_degp.flatten()))
    std_mean_uncertainty = np.mean(np.sqrt(y_var_std.flatten()))
    
    return {
        'degp_nrmse': degp_nrmse,
        'std_nrmse': std_nrmse,
        'improvement_percent': improvement,
        'degp_predictions': y_pred_degp,
        'std_predictions': y_pred_std,
        'degp_variance': y_var_degp,
        'std_variance': y_var_std,
        'true_values': y_true,
        'degp_mean_uncertainty': degp_mean_uncertainty,
        'std_mean_uncertainty': std_mean_uncertainty
    }

def create_comparison_visualization(X_train: np.ndarray, X_test: np.ndarray,
                                  y_train_func: np.ndarray, results: Dict, config: Dict):
    """
    Generate comprehensive comparison visualization.
    
    Parameters:
    -----------
    X_train, X_test : array
        Training and test input points
    y_train_func : array
        Training function observations
    results : dict
        Model evaluation results
    config : dict
        Experimental configuration
    """
    plt.figure(figsize=(16, 8))
    
    # DEGP subplot
    plt.subplot(2, 2, 1)
    degp_pred = results['degp_predictions'].flatten()
    degp_std = np.sqrt(results['degp_variance'].flatten())
    y_true = results['true_values'].flatten()
    
    plt.plot(X_test, y_true, 'k-', label='True function', linewidth=2)
    plt.plot(X_test, degp_pred, 'b-', label='DEGP prediction', linewidth=2)
    plt.fill_between(X_test.flatten(), 
                    degp_pred - 2*degp_std, 
                    degp_pred + 2*degp_std,
                    alpha=0.3, color='blue', label='±2σ uncertainty')
    plt.scatter(X_train, y_train_func, c='red', s=80, 
               zorder=5, label='Training data', edgecolors='darkred')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'DEGP (with {config["derivative_noise_ratio"]*100:.0f}% derivative noise)\n'
              f'NRMSE: {results["degp_nrmse"]:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Standard GP subplot
    plt.subplot(2, 2, 2)
    std_pred = results['std_predictions'].flatten()
    std_std = np.sqrt(results['std_variance'].flatten())
    
    plt.plot(X_test, y_true, 'k-', label='True function', linewidth=2)
    plt.plot(X_test, std_pred, 'g-', label='Standard GP prediction', linewidth=2)
    plt.fill_between(X_test.flatten(),
                    std_pred - 2*std_std,
                    std_pred + 2*std_std,
                    alpha=0.3, color='green', label='±2σ uncertainty')
    plt.scatter(X_train, y_train_func, c='red', s=80,
               zorder=5, label='Training data', edgecolors='darkred')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Standard GP (function only)\n'
              f'NRMSE: {results["std_nrmse"]:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Prediction error comparison
    plt.subplot(2, 2, 3)
    degp_error = np.abs(y_true - degp_pred)
    std_error = np.abs(y_true - std_pred)
    
    plt.plot(X_test, degp_error, 'b-', label='DEGP error', linewidth=2)
    plt.plot(X_test, std_error, 'g-', label='Standard GP error', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title('Prediction Error Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Uncertainty comparison
    plt.subplot(2, 2, 4)
    plt.plot(X_test, degp_std, 'b-', label='DEGP uncertainty (σ)', linewidth=2)
    plt.plot(X_test, std_std, 'g-', label='Standard GP uncertainty (σ)', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Prediction Uncertainty')
    plt.title('Uncertainty Quantification Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_comprehensive_results(results: Dict, config: Dict, X_train: np.ndarray):
    """
    Print comprehensive experimental results and analysis.
    """
    print("\n" + "="*70)
    print("EXPERIMENTAL RESULTS")
    print("="*70)
    
    print(f"Performance Comparison:")
    print(f"  DEGP NRMSE:        {results['degp_nrmse']:.6f}")
    print(f"  Standard GP NRMSE: {results['std_nrmse']:.6f}")
    print(f"  DEGP Improvement:  {results['improvement_percent']:.1f}%")
    
    print(f"\nUncertainty Analysis:")
    print(f"  DEGP Mean σ:       {results['degp_mean_uncertainty']:.4f}")
    print(f"  Standard GP Mean σ: {results['std_mean_uncertainty']:.4f}")
    
    print(f"\nObservation Count Analysis:")
    degp_obs = len(X_train) * (1 + config['n_order'])  # function + derivatives
    std_obs = len(X_train)  # function only
    print(f"  DEGP observations:     {degp_obs} ({degp_obs//std_obs}x multiplier)")
    print(f"  Standard GP observations: {std_obs} (baseline)")
    print(f"  Information efficiency: {results['improvement_percent']:.1f}% improvement")
    print(f"                         for {degp_obs//std_obs}x more observations")
    
    print(f"\nNoise Robustness Assessment:")
    print(f"  Function noise level:   {config['function_noise_ratio']*100:.1f}%")
    print(f"  Derivative noise level: {config['derivative_noise_ratio']*100:.1f}%")
    print(f"  DEGP still outperforms: {'YES' if results['improvement_percent'] > 0 else 'NO'}")

def main():
    """
    Main tutorial execution: DEGP robustness under noisy derivative conditions.
    """
    print("DEGP Tutorial: Derivative Information Robustness Analysis")
    print("="*65)
    
    # Configure experiment
    config = configure_experiment()
    rng = np.random.RandomState(config['random_seed'])
    
    print(f"Experimental Configuration:")
    print(f"  Test function: f(x) = x * sin(x)")
    print(f"  Domain: [{config['lb_x']}, {config['ub_x']}]")
    print(f"  Training points: {config['num_training_pts']}")
    print(f"  Function noise: {config['function_noise_ratio']*100:.1f}% (perfect)")
    print(f"  Derivative noise: {config['derivative_noise_ratio']*100:.1f}%")
    print(f"  Derivative order: {config['n_order']}")
    print()
    
    # Generate training data
    print("Generating training data with controlled noise injection...")
    X_train, y_train, noise_std, der_indices = generate_training_data(config, rng)
    
    # Create test points
    X_test = np.linspace(config['lb_x'], config['ub_x'], config['num_test_pts']).reshape(-1, 1)
    
    print(f"Data generation complete:")
    print(f"  Training observations: {len(y_train)} arrays")
    print(f"  Test points: {len(X_test)}")
    
    # Train models
    degp_model, degp_params = train_degp_model(X_train, y_train, noise_std, der_indices, config)
    std_gp_model, std_params = train_standard_gp_model(
        X_train, y_train[0], noise_std[:len(X_train)], config
    )
    
    # Evaluate models
    print("Evaluating model performance...")
    results = evaluate_models(X_test, degp_model, degp_params, std_gp_model, std_params)
    
    # Display results
    print_comprehensive_results(results, config, X_train)
    
    # Generate visualizations
    print(f"\nGenerating comparison visualization...")
    create_comparison_visualization(X_train, X_test, y_train[0], results, config)
    
    # Conclusion
    print(f"\n" + "="*70)
    print("TUTORIAL CONCLUSIONS")
    print("="*70)
    
    if results['improvement_percent'] > 0:
        print(f"✓ DEGP demonstrates robustness to derivative noise")
        print(f"✓ {results['improvement_percent']:.1f}% performance improvement despite {config['derivative_noise_ratio']*100:.0f}% derivative noise")
        print(f"✓ Derivative information remains valuable under realistic measurement conditions")
    else:
        print(f"✗ High derivative noise ({config['derivative_noise_ratio']*100:.0f}%) overwhelms information content")
        print(f"✗ Standard GP outperforms DEGP in this noise regime")
        print(f"✗ Consider reducing derivative noise or using different regularization")
    
    print(f"\nThis tutorial demonstrates the trade-off between derivative information")
    print(f"content and measurement noise in practical DEGP applications.")

if __name__ == "__main__":
    main()