"""
================================================================================
DEGP Tutorial: Selective Derivative Strategy in 2D Function Approximation
================================================================================

This tutorial demonstrates an advanced DEGP technique: selective derivative 
inclusion. Instead of using ALL possible derivatives up to a given order,
we strategically select only the most informative derivatives to balance
accuracy and computational efficiency.

Key concepts covered:
- Strategic derivative selection (directional vs. mixed derivatives)
- Computational trade-offs in high-dimensional derivative spaces
- Performance comparison: selective vs. complete derivative sets
- 2D visualization of selective DEGP results

The selective approach focuses on "main" derivatives (∂f/∂x₁, ∂f/∂x₂, ∂²f/∂x₁², ∂²f/∂x₂²)
while excluding mixed derivatives (∂²f/∂x₁∂x₂) to reduce computational cost.
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
from typing import List, Tuple, Dict, Optional


def true_function(X, alg=oti):
    """
    Complex 2D test function with polynomial and oscillatory components.

    The function f(x₁,x₂) = x₁²x₂ + cos(10x₁) + cos(10x₂) contains:
    - Polynomial interaction: x₁²x₂ (smooth, contains mixed derivatives)
    - High-frequency oscillations: cos(10x₁) + cos(10x₂) (separable terms)

    This design allows us to test how well selective derivatives capture
    both interaction effects and separable high-frequency features.

    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Input points with columns [x1, x2]
    alg : module
        Numerical library (numpy or pyoti)

    Returns:
    --------
    y : array-like
        Function values
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    return x1**2 * x2 + alg.cos(10 * x1) + alg.cos(10 * x2)


def create_derivative_strategies(n_order: int, n_bases: int) -> Dict[str, List]:
    """
    Create different derivative selection strategies for comparison.

    Parameters:
    -----------
    n_order : int
        Maximum derivative order
    n_bases : int
        Number of input dimensions

    Returns:
    --------
    strategies : dict
        Dictionary of different derivative selection approaches
    """
    strategies = {}

    # Strategy 1: Only first-order derivatives (gradients)
    strategies['gradient_only'] = [
        [[[1, 1]], [[2, 1]]]  # ∂f/∂x₁, ∂f/∂x₂
    ]

    # Strategy 2: Main derivatives (no mixed terms) - Current approach
    strategies['main_derivatives'] = [
        [[[1, 1]], [[2, 1]]],  # First-order: ∂f/∂x₁, ∂f/∂x₂
        [[[1, 2]], [[2, 2]]],  # Second-order: ∂²f/∂x₁², ∂²f/∂x₂²
    ]

    # Strategy 3: All derivatives (complete set)
    strategies['complete'] = utils.gen_OTI_indices(n_bases, n_order)

    return strategies


def analyze_derivative_strategy(der_indices: List, strategy_name: str) -> Dict:
    """
    Analyze a derivative selection strategy.

    Parameters:
    -----------
    der_indices : list
        Derivative indices structure
    strategy_name : str
        Name of the strategy

    Returns:
    --------
    analysis : dict
        Analysis of the derivative strategy
    """
    derivative_count = 0
    derivative_descriptions = []

    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            derivative_count += 1
            deriv_spec = der_indices[i][j]

            # Create human-readable description
            if len(deriv_spec) == 1:
                var_idx, order = deriv_spec[0]
                if order == 1:
                    desc = f"∂f/∂x_{var_idx}"
                else:
                    desc = f"∂^{order}f/∂x_{var_idx}^{order}"
            else:
                # Mixed derivative
                total_order = sum(spec[1] for spec in deriv_spec)
                vars_involved = [f"∂x_{spec[0]}" + (f"^{spec[1]}" if spec[1] > 1 else "")
                                 for spec in deriv_spec]
                desc = f"∂^{total_order}f/" + "".join(vars_involved)

            derivative_descriptions.append(desc)

    return {
        'strategy_name': strategy_name,
        'derivative_count': derivative_count,
        'descriptions': derivative_descriptions,
        'derivatives_per_point': derivative_count
    }


def run_degp_experiment(X_train: np.ndarray, der_indices: List, n_order: int,
                        n_bases: int, X_test: np.ndarray, strategy_name: str) -> Dict:
    """
    Run a complete DEGP experiment with a specific derivative strategy.

    Parameters:
    -----------
    X_train : array
        Training input points
    der_indices : list
        Derivative indices for this strategy
    n_order, n_bases : int
        DEGP configuration parameters
    X_test : array
        Test points for evaluation
    strategy_name : str
        Name of the derivative strategy

    Returns:
    --------
    results : dict
        Comprehensive results from the experiment
    """
    print(f"  Running {strategy_name} strategy...")
    start_time = time.time()

    try:
        # Setup perturbed training inputs
        # Note: In practice, derivatives would typically come from the user
        # rather than being computed this way
        X_train_pert = oti.array(X_train)
        for i in range(n_bases):
            X_train_pert[:, i] += oti.e(i + 1, order=n_order)

        # Evaluate function and selected derivatives
        y_train_hc = true_function(X_train_pert)
        # The derivative information in y_train must match the exact order of der_indices
        y_train = [y_train_hc.real]  # Function values always come first

        derivative_obs_count = 0
        # Add derivatives in the same order as specified by der_indices
        for i in range(len(der_indices)):
            for j in range(len(der_indices[i])):
                # Order must match der_indices
                y_train.append(y_train_hc.get_deriv(der_indices[i][j]))
                derivative_obs_count += len(X_train)

        # Initialize and train DEGP
        gp = degp(
            X_train, y_train, n_order, n_bases, der_indices,
            normalize=True, kernel="SE", kernel_type="anisotropic"
        )

        # Optimize hyperparameters using particle swarm optimization
        # n_restart_optimizer: number of generations for the particle swarm optimizer
        # swarm_size: number of particles in the swarm
        opt_start = time.time()
        params = gp.optimize_hyperparameters(
            n_restart_optimizer=15, swarm_size=100)
        opt_time = time.time() - opt_start

        # Make predictions
        pred_start = time.time()
        y_pred = gp.predict(X_test, params, calc_cov=False, return_deriv=False)
        pred_time = time.time() - pred_start

        # Evaluate performance
        y_true = true_function(X_test, alg=np).flatten()
        mse = np.mean((y_true - y_pred.flatten())**2)
        nrmse = utils.nrmse(y_true, y_pred.flatten())
        max_error = np.max(np.abs(y_true - y_pred.flatten()))

        total_time = time.time() - start_time

        return {
            'strategy_name': strategy_name,
            'success': True,
            'nrmse': nrmse,
            'mse': mse,
            'max_error': max_error,
            'total_time': total_time,
            'optimization_time': opt_time,
            'prediction_time': pred_time,
            'function_obs': len(X_train),
            'derivative_obs': derivative_obs_count,
            'total_obs': len(X_train) + derivative_obs_count,
            'y_pred': y_pred,
            'params': params
        }

    except Exception as e:
        return {
            'strategy_name': strategy_name,
            'success': False,
            'error': str(e),
            'total_time': time.time() - start_time,
            'common_causes': [
                'Inconsistent data: mismatch between y_train and der_indices',
                'Cholesky decomposition failure due to numerical instability',
                'Insufficient computational resources',
                'Unsupported kernel type'
            ]
        }


def main():
    """
    Main selective derivative DEGP tutorial with strategy comparison.
    """
    print("Selective Derivative DEGP Tutorial: Strategic Choice of Derivatives")
    print("=" * 75)

    # ==========================================================================
    # Configuration
    # ==========================================================================

    n_order = 2      # Maximum derivative order
    n_bases = 2      # Input dimensions
    lb_x, ub_x = -1, 1
    lb_y, ub_y = -1, 1
    num_points = 5   # Points per dimension
    N_grid = 25      # Test grid resolution

    print(f"Configuration:")
    print(f"  Domain: [{lb_x}, {ub_x}] × [{lb_y}, {ub_y}]")
    print(
        f"  Training grid: {num_points}×{num_points} = {num_points**2} points")
    print(f"  Maximum derivative order: {n_order}")
    print(f"  Test grid resolution: {N_grid}×{N_grid}")

    # ==========================================================================
    # Strategy Analysis
    # ==========================================================================

    print(f"\nDerivative Strategy Analysis:")
    print("=" * 50)

    # Generate different derivative strategies
    strategies = create_derivative_strategies(n_order, n_bases)

    print(f"Strategy Structure Examples for 2D, 2nd order:")
    print(f"  gradient_only:    {strategies['gradient_only']}")
    print(f"  main_derivatives: {strategies['main_derivatives']}")
    print(f"  complete:         {strategies['complete']}")
    print(f"\nThis shows:")
    print(f"  • gradient_only: Only ∂f/∂x₁, ∂f/∂x₂")
    print(f"  • main_derivatives: Adds ∂²f/∂x₁², ∂²f/∂x₂² (excludes mixed ∂²f/∂x₁∂x₂)")
    print(f"  • complete: Includes all derivatives including mixed terms")

    strategy_analyses = {}
    for name, der_indices in strategies.items():
        analysis = analyze_derivative_strategy(der_indices, name)
        strategy_analyses[name] = analysis

        print(f"\nStrategy: {name.upper()}")
        print(f"  Derivatives per point: {analysis['derivative_count']}")
        print(
            f"  Computational trade-off: {'Low' if analysis['derivative_count'] <= 2 else 'Medium' if analysis['derivative_count'] <= 4 else 'High'}")
        print(f"  Included derivatives:")
        for i, desc in enumerate(analysis['descriptions'], 1):
            print(f"    {i}. {desc}")

    # ==========================================================================
    # Training Data Setup
    # ==========================================================================

    print(f"\nSetting up Training Data...")

    # Create training grid
    x_vals = np.linspace(lb_x, ub_x, num_points)
    y_vals = np.linspace(lb_y, ub_y, num_points)
    X_train = np.array(list(itertools.product(x_vals, y_vals)))

    # Create test grid
    x_lin = np.linspace(lb_x, ub_x, N_grid)
    y_lin = np.linspace(lb_y, ub_y, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    print(f"  Training points: {X_train.shape}")
    print(f"  Test points: {X_test.shape}")

    # ==========================================================================
    # Strategy Comparison Experiments
    # ==========================================================================

    print(f"\nRunning Strategy Comparison Experiments:")
    print("=" * 55)

    results = {}

    # Run experiment for each strategy
    for strategy_name, der_indices in strategies.items():
        result = run_degp_experiment(
            X_train, der_indices, n_order, n_bases, X_test, strategy_name
        )
        results[strategy_name] = result

        if result['success']:
            print(f"    {strategy_name}: NRMSE = {result['nrmse']:.6f}, "
                  f"Time = {result['total_time']:.2f}s, "
                  f"Obs = {result['total_obs']}")
        else:
            print(f"    {strategy_name}: FAILED - {result['error']}")
            print(
                f"      Common causes: {', '.join(result.get('common_causes', []))}")

    # ==========================================================================
    # Results Analysis and Comparison
    # ==========================================================================

    print(f"\nStrategy Comparison Results:")
    print("=" * 60)

    successful_results = {k: v for k, v in results.items() if v['success']}

    if len(successful_results) > 0:
        print(
            f"{'Strategy':<20} {'NRMSE':<12} {'Time(s)':<10} {'Obs':<8} {'Efficiency'}")
        print("-" * 70)

        for strategy_name in ['gradient_only', 'main_derivatives', 'complete']:
            if strategy_name in successful_results:
                r = successful_results[strategy_name]
                efficiency = r['nrmse'] * r['total_time']  # Lower is better
                print(f"{strategy_name:<20} {r['nrmse']:<12.6f} {r['total_time']:<10.2f} "
                      f"{r['total_obs']:<8} {efficiency:.4f}")

    # ==========================================================================
    # Detailed Analysis of Main Derivatives Strategy
    # ==========================================================================

    print(f"\nFocused Analysis: Main Derivatives Strategy")
    print("=" * 50)

    # Use the main derivatives strategy for detailed analysis
    main_der_indices = strategies['main_derivatives']
    main_result = results['main_derivatives']

    if main_result['success']:
        print(f"Selected Strategy: Main Derivatives (∂f/∂x₁, ∂f/∂x₂, ∂²f/∂x₁², ∂²f/∂x₂²)")
        print(f"  Rationale: Captures directional information and curvature")
        print(f"             without expensive mixed derivative computation")
        print(f"  Performance: NRMSE = {main_result['nrmse']:.6f}")
        print(
            f"  Efficiency: {main_result['total_obs']} observations from {num_points**2} points")
        print(
            f"  Observation multiplier: {main_result['total_obs']//(num_points**2)}x")

        # ==========================================================================
        # Visualization
        # ==========================================================================

        print(f"\nGenerating Visualization...")

        try:
            plotting_helper.make_plots(
                X_train,
                # Adjust based on plotting_helper requirements
                [main_result['y_pred']],
                X_test,
                main_result['y_pred'],
                true_function,
                X1_grid=X1_grid,
                X2_grid=X2_grid,
                n_order=n_order,
                n_bases=n_bases,
                plot_derivative_surrogates=False,
                der_indices=main_der_indices,
            )
            print("  Visualization: SUCCESS")

        except Exception as e:
            print(f"  Visualization: FAILED - {e}")

    # ==========================================================================
    # Tutorial Summary and Guidelines
    # ==========================================================================

    print(f"\nTutorial Summary: Selective Derivative Strategies")
    print("=" * 60)

    if len(successful_results) > 0:
        print(f"Key Findings:")

        # Performance comparison
        if 'gradient_only' in successful_results and 'main_derivatives' in successful_results:
            grad_nrmse = successful_results['gradient_only']['nrmse']
            main_nrmse = successful_results['main_derivatives']['nrmse']
            improvement = (grad_nrmse - main_nrmse) / grad_nrmse * 100
            print(
                f"  • Main derivatives vs gradient-only: {improvement:.1f}% improvement")

        if 'main_derivatives' in successful_results and 'complete' in successful_results:
            main_nrmse = successful_results['main_derivatives']['nrmse']
            comp_nrmse = successful_results['complete']['nrmse']
            main_time = successful_results['main_derivatives']['total_time']
            comp_time = successful_results['complete']['total_time']
            print(
                f"  • Main vs complete NRMSE: {main_nrmse:.6f} vs {comp_nrmse:.6f}")
            print(
                f"  • Computational savings: {(comp_time - main_time)/comp_time*100:.1f}% time reduction")

        print(f"\nStrategy Selection Guidelines:")
        print(f"  • Use gradient_only for: Simple functions, limited computational budget")
        print(f"  • Use main_derivatives for: Balanced accuracy/efficiency trade-off")
        print(f"  • Use complete set for: Complex interaction terms, maximum accuracy")

        print(f"\nComputational Trade-offs:")
        for name, analysis in strategy_analyses.items():
            if name in successful_results:
                # +1 for function
                obs_per_point = analysis['derivatives_per_point'] + 1
                total_obs = num_points**2 * obs_per_point
                print(
                    f"  • {name}: {obs_per_point} obs/point = {total_obs} total observations")

    if main_result['success']:
        print(f"\nFinal Result with Main Derivatives Strategy:")
        print(f"NRMSE: {main_result['nrmse']:.6f}")
        print(f"This demonstrates effective selective derivative inclusion for")
        print(f"balancing computational efficiency with prediction accuracy.")


if __name__ == "__main__":
    main()
