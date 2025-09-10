"""
Individual Submodel Derivative-Enhanced Gaussian Process Implementation
=====================================================================

This module implements a derivative-enhanced Gaussian Process where each training
point forms its own individual submodel. All submodels use the same comprehensive
set of directional derivatives up to a specified order. The final prediction is
obtained by combining these individual submodels through a weighted GP framework.

Key Features:
- One submodel per training point for maximum local information
- Uniform derivative structure across all submodels
- Comprehensive directional derivative computation
- Weighted combination of individual submodel predictions
- 1D function approximation with full derivative utilization
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
import pyoti.sparse as oti
from wdegp.wdegp import wdegp
import utils
import plotting_helper
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class IndividualSubmodelConfig:
    """Configuration for individual submodel derivative-enhanced GP."""
    n_order: int = 2                    # Maximum derivative order
    n_bases: int = 1                    # Number of input dimensions (1D case)
    lb_x: float = 0.5                   # Lower bound of domain
    ub_x: float = 2.5                   # Upper bound of domain
    num_points: int = 10                # Number of training points
    kernel: str = "SE"                  # Kernel type (Squared Exponential)
    kernel_type: str = "anisotropic"    # Kernel parameterization
    normalize: bool = True              # Whether to normalize training data
    n_restart_optimizer: int = 15       # Hyperparameter optimization restarts
    swarm_size: int = 50               # Particle swarm optimization size
    test_points: int = 250             # Number of test points for evaluation
    random_seed: Optional[int] = None   # Random seed for reproducibility


class IndividualSubmodelGP:
    """
    Derivative-enhanced GP with individual submodels for each training point.

    This implementation creates a separate submodel for each training point,
    where each submodel contains full derivative information at its location.
    The approach maximizes local information content while maintaining a 
    structured framework for combining predictions.
    """

    def __init__(self, config: IndividualSubmodelConfig, true_function: Callable):
        """
        Initialize the individual submodel GP experiment.

        Args:
            config: Configuration object with experiment parameters
            true_function: Target function to approximate
        """
        self.config = config
        self.true_function = true_function
        self.gp_model = None
        self.training_data = None
        self.results = None

        if config.random_seed is not None:
            np.random.seed(config.random_seed)

    def generate_training_points(self) -> np.ndarray:
        """
        Generate uniformly distributed training points across the domain.

        Returns:
            Array of training point coordinates
        """
        X_train = np.linspace(
            self.config.lb_x,
            self.config.ub_x,
            self.config.num_points
        ).reshape(-1, 1)
        return X_train

    def create_individual_submodel_structure(self, num_points: int) -> Tuple[List[List[int]], List]:
        """
        Create submodel structure where each training point forms its own submodel.

        Args:
            num_points: Number of training points

        Returns:
            Tuple of (submodel_indices, derivative_specifications)
        """
        # Each training point gets its own submodel
        submodel_indices = [[i] for i in range(num_points)]

        # Generate comprehensive derivative indices for the specified order
        base_derivative_indices = utils.gen_OTI_indices(
            self.config.n_bases, self.config.n_order)

        # All submodels use the same comprehensive derivative structure
        derivative_specifications = [
            base_derivative_indices for _ in range(num_points)]

        return submodel_indices, derivative_specifications, base_derivative_indices

    def compute_derivatives_at_point(self, X_point: np.ndarray,
                                     base_derivative_indices: List) -> List[np.ndarray]:
        """
        Compute all specified derivatives at a single training point.

        Args:
            X_point: Single training point coordinate
            base_derivative_indices: Derivative index specifications

        Returns:
            List of derivative values
        """
        # Create OTI array with perturbations for automatic differentiation
        X_oti = oti.array(X_point)
        for i in range(self.config.n_bases):
            for j in range(X_oti.shape[0]):
                X_oti[j, i] += oti.e(i + 1, order=self.config.n_order)

        # Evaluate function with OTI to obtain derivative information
        y_with_derivatives = oti.array(
            [self.true_function(x, alg=oti) for x in X_oti])

        # Extract all specified derivatives
        derivatives = []
        for i in range(len(base_derivative_indices)):
            for j in range(len(base_derivative_indices[i])):
                derivative_value = y_with_derivatives.get_deriv(
                    base_derivative_indices[i][j]
                ).reshape(-1, 1)
                derivatives.append(derivative_value)

        return derivatives

    def prepare_individual_submodel_data(self, X_train: np.ndarray,
                                         submodel_indices: List[List[int]],
                                         base_derivative_indices: List) -> List[List[np.ndarray]]:
        """
        Prepare training data for each individual submodel.

        Args:
            X_train: All training point coordinates
            submodel_indices: Point indices for each submodel
            base_derivative_indices: Derivative specifications

        Returns:
            Training data arrays for each submodel
        """
        # Compute function values at all training points
        y_function_values = self.true_function(X_train, alg=np)

        submodel_data = []

        print(
            f"Preparing data for {len(submodel_indices)} individual submodels...")
        for k, point_indices in enumerate(submodel_indices):
            # Each submodel contains one point, so point_indices = [k]
            point_index = point_indices[0]
            X_point = X_train[point_indices]

            # Start with function values at all training points
            submodel_training_data = [y_function_values]

            # Add all derivatives computed at this specific point
            derivatives = self.compute_derivatives_at_point(
                X_point, base_derivative_indices)
            submodel_training_data.extend(derivatives)

            submodel_data.append(submodel_training_data)

            if (k + 1) % 5 == 0 or k == len(submodel_indices) - 1:
                print(f"  Processed {k + 1}/{len(submodel_indices)} submodels")

        return submodel_data

    def build_weighted_gp_model(self, X_train: np.ndarray, submodel_data: List,
                                submodel_indices: List[List[int]],
                                derivative_specifications: List) -> wdegp:
        """
        Construct the weighted derivative-enhanced GP model.

        Args:
            X_train: Training point coordinates
            submodel_data: Training data for each submodel
            submodel_indices: Point indices for each submodel
            derivative_specifications: Derivative specs for each submodel

        Returns:
            Constructed GP model
        """
        print("Building weighted derivative-enhanced GP model...")
        print(f"  Number of submodels: {len(submodel_indices)}")
        print(f"  Derivative order: {self.config.n_order}")
        print(f"  Total derivatives per submodel: {len(submodel_data[0]) - 1}")

        self.gp_model = wdegp(
            X_train,
            submodel_data,
            self.config.n_order,
            self.config.n_bases,
            submodel_indices,
            derivative_specifications,
            normalize=self.config.normalize,
            kernel=self.config.kernel,
            kernel_type=self.config.kernel_type,
        )

        return self.gp_model

    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """
        Optimize GP hyperparameters using particle swarm optimization.

        Returns:
            Optimized hyperparameters
        """
        if self.gp_model is None:
            raise ValueError(
                "GP model not built. Call build_weighted_gp_model() first.")

        print("Optimizing hyperparameters...")
        print(f"  Optimization restarts: {self.config.n_restart_optimizer}")
        print(f"  Swarm size: {self.config.swarm_size}")

        params = self.gp_model.optimize_hyperparameters(
            n_restart_optimizer=self.config.n_restart_optimizer,
            swarm_size=self.config.swarm_size
        )

        print("Hyperparameter optimization completed.")
        return params

    def evaluate_model_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model performance on a dense test set.

        Args:
            params: Optimized hyperparameters

        Returns:
            Dictionary containing predictions and performance metrics
        """
        if self.gp_model is None:
            raise ValueError("GP model not built.")

        # Generate dense test set
        X_test = np.linspace(
            self.config.lb_x,
            self.config.ub_x,
            self.config.test_points
        ).reshape(-1, 1)

        print(
            f"Making predictions on {self.config.test_points} test points...")

        # Generate predictions with uncertainty and submodel contributions
        y_pred, y_cov, submodel_vals, submodel_cov = self.gp_model.predict(
            X_test,
            params,
            calc_cov=True,
            return_submodels=True
        )

        # Compute true function values and error metrics
        y_true = self.true_function(X_test, alg=np)
        nrmse = utils.nrmse(y_true, y_pred)

        results = {
            'X_test': X_test,
            'y_pred': y_pred,
            'y_true': y_true,
            'y_cov': y_cov,
            'submodel_vals': submodel_vals,
            'submodel_cov': submodel_cov,
            'nrmse': nrmse
        }

        print(f"Model evaluation completed. NRMSE: {nrmse:.6f}")
        return results

    def create_comprehensive_visualization(self, results: Dict[str, Any],
                                           show_submodels: bool = True):
        """
        Generate comprehensive visualization including individual submodel contributions.

        Args:
            results: Evaluation results dictionary
            show_submodels: Whether to plot individual submodel contributions
        """
        plotting_helper.make_submodel_plots(
            self.training_data['X_train'],
            self.training_data['submodel_data'],
            results['X_test'],
            results['y_pred'],
            self.true_function,
            cov=results['y_cov'],
            n_order=self.config.n_order,
            n_bases=self.config.n_bases,
            plot_submodels=show_submodels,
            submodel_vals=results['submodel_vals'],
            submodel_cov=results['submodel_cov'],
        )

    def analyze_submodel_contributions(self, results: Dict[str, Any]):
        """
        Analyze and visualize individual submodel contributions.

        Args:
            results: Evaluation results dictionary
        """
        X_test = results['X_test']
        X_train = self.training_data['X_train']
        submodel_vals = results['submodel_vals']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot individual submodel predictions
        colors = plt.cm.tab10(np.linspace(0, 1, len(submodel_vals)))
        for i, (submodel_pred, color) in enumerate(zip(submodel_vals, colors)):
            ax1.plot(X_test.ravel(), submodel_pred.ravel(),
                     color=color, alpha=0.6, linewidth=1,
                     label=f'Submodel {i+1} (x={X_train[i,0]:.2f})')

        # Highlight training points
        ax1.scatter(X_train.ravel(),
                    self.true_function(X_train, alg=np),
                    color='red', s=50, edgecolor='black',
                    label='Training Points', zorder=10)

        ax1.set_title('Individual Submodel Predictions')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot final combined prediction vs true function
        ax2.plot(X_test.ravel(), results['y_true'].ravel(),
                 'b-', linewidth=2, label='True Function')
        ax2.plot(X_test.ravel(), results['y_pred'].ravel(),
                 'r--', linewidth=2, label='Combined GP Prediction')

        # Add uncertainty bands
        std_dev = np.sqrt(results['y_cov'])
        ax2.fill_between(X_test.ravel(),
                         (results['y_pred'] - 2*std_dev).ravel(),
                         (results['y_pred'] + 2*std_dev).ravel(),
                         alpha=0.3, color='red', label='95% Confidence')

        ax2.scatter(X_train.ravel(),
                    self.true_function(X_train, alg=np),
                    color='red', s=50, edgecolor='black',
                    label='Training Points', zorder=10)

        ax2.set_title('Combined Model Prediction with Uncertainty')
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_complete_experiment(self, show_submodels: bool = True,
                                analyze_contributions: bool = True) -> Dict[str, Any]:
        """
        Execute the complete individual submodel GP experiment.

        Args:
            show_submodels: Whether to show individual submodel plots
            analyze_contributions: Whether to analyze submodel contributions

        Returns:
            Complete experimental results
        """
        print("=== Individual Submodel Derivative-Enhanced GP Experiment ===")

        # Step 1: Generate training points
        print("Generating uniformly distributed training points...")
        X_train = self.generate_training_points()

        # Step 2: Create individual submodel structure
        print("Creating individual submodel structure...")
        submodel_indices, derivative_specs, base_der_indices = \
            self.create_individual_submodel_structure(len(X_train))

        print(f"Created {len(submodel_indices)} individual submodels")
        print(f"Each submodel uses {len(base_der_indices)} derivative types")

        # Step 3: Prepare training data with derivatives
        print("Computing derivatives at each training point...")
        submodel_data = self.prepare_individual_submodel_data(
            X_train, submodel_indices, base_der_indices
        )

        # Store training data for later use
        self.training_data = {
            'X_train': X_train,
            'submodel_data': submodel_data,
            'submodel_indices': submodel_indices,
            'derivative_specs': derivative_specs
        }

        # Step 4: Build and optimize GP model
        self.build_weighted_gp_model(
            X_train, submodel_data, submodel_indices, derivative_specs)
        params = self.optimize_hyperparameters()

        # Step 5: Evaluate model performance
        results = self.evaluate_model_performance(params)

        # Step 6: Generate visualizations
        print("Creating comprehensive visualizations...")
        self.create_comprehensive_visualization(results, show_submodels)

        if analyze_contributions:
            self.analyze_submodel_contributions(results)

        # Step 7: Display summary
        self.display_experiment_summary(results)

        return results

    def display_experiment_summary(self, results: Dict[str, Any]):
        """
        Display comprehensive experiment summary.

        Args:
            results: Experimental results
        """
        print(f"\n=== Experiment Summary ===")
        print(f"Domain: [{self.config.lb_x}, {self.config.ub_x}]")
        print(f"Training points: {self.config.num_points}")
        print(f"Test points: {self.config.test_points}")
        print(f"Individual submodels: {self.config.num_points}")
        print(f"Maximum derivative order: {self.config.n_order}")
        print(
            f"Derivatives per submodel: {len(self.training_data['submodel_data'][0]) - 1}")
        print(f"Final NRMSE: {results['nrmse']:.6f}")

        print(f"\n=== Submodel Configuration ===")
        for i in range(min(3, self.config.num_points)):  # Show first 3 as examples
            x_coord = self.training_data['X_train'][i, 0]
            n_derivatives = len(self.training_data['submodel_data'][i]) - 1
            print(
                f"Submodel {i+1}: x = {x_coord:.3f}, {n_derivatives} derivatives")

        if self.config.num_points > 3:
            print(f"... and {self.config.num_points - 3} more submodels")


def oscillatory_function_with_trend(X, alg=oti):
    """
    Example function with both oscillatory behavior and polynomial trend.

    This function combines high-frequency oscillations with a polynomial trend,
    making it challenging to approximate and ideal for demonstrating the benefits
    of derivative information in individual submodels.

    Args:
        X: Input points
        alg: Algorithm library (numpy or pyoti)

    Returns:
        Function values
    """
    x1 = X[:, 0]
    return alg.sin(10 * np.pi * x1) / (2 * x1) + (x1 - 1) ** 4


def main():
    """
    Demonstrate individual submodel derivative-enhanced GP implementation.

    This example shows how to:
    1. Create individual submodels for each training point
    2. Compute comprehensive derivatives at each location
    3. Combine submodels through weighted GP framework
    4. Analyze individual submodel contributions
    """
    print("Individual Submodel Derivative-Enhanced GP Demo")
    print("=" * 50)

    # Configure experiment
    config = IndividualSubmodelConfig(
        n_order=2,
        n_bases=1,
        lb_x=0.5,
        ub_x=2.5,
        num_points=10,
        test_points=250,
        n_restart_optimizer=15,
        swarm_size=50
    )

    # Create and run experiment
    experiment = IndividualSubmodelGP(config, oscillatory_function_with_trend)
    results = experiment.run_complete_experiment(
        show_submodels=False,
        analyze_contributions=True
    )

    print(f"\n=== Derivative Information Benefits ===")
    print(f"Individual submodels capture local function behavior through derivatives")
    print(f"Each training point contributes its local derivative information")
    print(f"Weighted combination provides global approximation with local accuracy")
    print(f"Final model achieves NRMSE of {results['nrmse']:.6f}")


if __name__ == "__main__":
    main()
