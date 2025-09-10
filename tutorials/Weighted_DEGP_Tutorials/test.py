"""
Grouped Submodel Derivative-Enhanced Gaussian Process Implementation
==================================================================

This module demonstrates a derivative-enhanced Gaussian Process where training
points are grouped into a predefined number of submodels (m submodels from n
training points, where m < n). Each submodel is informed by the function values
and derivatives from its assigned points. The final prediction is obtained by
combining these submodels through a weighted GP framework.

Key Features:
- Groups of training points form distinct submodels.
- Each submodel can have its own unique derivative structure.
- Combines regional information into a global prediction.
- 1D function approximation using grouped derivative information.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
import pyoti.sparse as oti
from wdegp.wdegp import wdegp
import utils
import plotting_helper
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


@dataclass
class GroupedSubmodelConfig:
    """Configuration for grouped submodel derivative-enhanced GP."""
    n_order: int = 2                     # Maximum derivative order
    n_bases: int = 1                     # Number of input dimensions (1D case)
    lb_x: float = 0.5                    # Lower bound of domain
    ub_x: float = 10.0                    # Upper bound of domain
    num_points: int = 10                 # Number of training points
    kernel: str = "SE"                   # Kernel type (Squared Exponential)
    kernel_type: str = "anisotropic"     # Kernel parameterization
    normalize: bool = True               # Whether to normalize training data
    n_restart_optimizer: int = 15        # Hyperparameter optimization restarts
    swarm_size: int = 50                 # Particle swarm optimization size
    test_points: int = 250               # Number of test points for evaluation
    random_seed: Optional[int] = None    # Random seed for reproducibility
    # Defines how points are grouped, e.g., [[0,1,2],[3,4,5]]
    submodel_groups: List[List[int]] = field(default_factory=list)
    # Defines derivative orders for each group, e.g., [2, 1]
    submodel_orders: List[int] = field(default_factory=list)


class GroupedSubmodelGP:
    """
    Derivative-enhanced GP with grouped submodels.

    This implementation creates submodels from groups of training points.
    Each group can be configured with a different set of derivatives,
    allowing for flexible model construction based on regional function complexity.
    """

    def __init__(self, config: GroupedSubmodelConfig, true_function: Callable):
        """
        Initialize the grouped submodel GP experiment.

        Args:
            config: Configuration object with experiment parameters.
            true_function: Target function to approximate.
        """
        self.config = config
        self.true_function = true_function
        self.gp_model = None
        self.training_data = None

        if not config.submodel_groups or not config.submodel_orders:
            raise ValueError(
                "submodel_groups and submodel_orders must be defined in the config.")
        if len(config.submodel_groups) != len(config.submodel_orders):
            raise ValueError(
                "The number of submodel groups must match the number of submodel orders.")

        if config.random_seed is not None:
            np.random.seed(config.random_seed)

    def generate_training_points(self) -> np.ndarray:
        """
        Generate uniformly distributed training points across the domain.

        Returns:
            Array of training point coordinates.
        """
        self.config.random_seed = 2
        self.rng = np.random.RandomState(self.config.random_seed)

        # 1. Generate Training Points
        X_candidates = np.linspace(
            self.config.lb_x, self.config.ub_x, 1000).reshape(-1, 1)
        training_indices = self.rng.choice(
            np.arange(X_candidates.shape[0]), size=self.config.num_points, replace=False)
        X_train = np.sort(X_candidates[training_indices], axis=0)
        return X_train

    def create_grouped_submodel_structure(self) -> Tuple[List[List[int]], List]:
        """
        Create the submodel structure based on predefined groups and derivative orders.

        Returns:
            A tuple containing (submodel_indices, derivative_specifications).
        """
        submodel_indices = self.config.submodel_groups

        # Generate derivative specifications for each submodel based on its assigned order
        derivative_specifications = []
        for order in self.config.submodel_orders:
            der_indices = utils.gen_OTI_indices(self.config.n_bases, order)
            derivative_specifications.append(der_indices)

        return submodel_indices, derivative_specifications

    def prepare_grouped_submodel_data(
        self,
        X_train: np.ndarray,
        submodel_indices: List[List[int]],
        derivative_specs: List
    ) -> List[List[np.ndarray]]:
        """
        Prepare training data for each grouped submodel.

        Args:
            X_train: All training point coordinates.
            submodel_indices: Point indices for each submodel group.
            derivative_specs: Derivative specifications for each submodel.

        Returns:
            A list of training data arrays, one for each submodel.
        """
        y_function_values = self.true_function(X_train, alg=np)
        submodel_data = []

        print(
            f"Preparing data for {len(submodel_indices)} grouped submodels...")
        for k, point_indices in enumerate(submodel_indices):
            # Select the points for the current submodel
            X_sub_oti = oti.array(X_train[point_indices])

            # Create OTI array with perturbations for automatic differentiation
            # The max order is used to ensure all needed derivatives can be computed
            for i in range(self.config.n_bases):
                for j in range(X_sub_oti.shape[0]):
                    X_sub_oti[j, i] += oti.e(i + 1, order=self.config.n_order)

            # Evaluate the function with OTI inputs
            y_with_derivatives = oti.array(
                [self.true_function(x, alg=oti) for x in X_sub_oti])

            # Start with function values at ALL training points
            submodel_training_data = [y_function_values]

            # Extract the derivatives specified for THIS submodel
            current_der_spec = derivative_specs[k]
            for i in range(len(current_der_spec)):
                for j in range(len(current_der_spec[i])):
                    derivative_value = y_with_derivatives.get_deriv(
                        current_der_spec[i][j]
                    ).reshape(-1, 1)
                    submodel_training_data.append(derivative_value)

            submodel_data.append(submodel_training_data)
            print(
                f"  Processed Submodel {k+1} with {len(point_indices)} points and derivative order {self.config.submodel_orders[k]}.")

        return submodel_data

    def build_and_optimize_gp(self, X_train: np.ndarray, submodel_data: List,
                              submodel_indices: List[List[int]],
                              derivative_specs: List) -> Dict[str, Any]:
        """
        Construct the GP model and optimize its hyperparameters.

        Args:
            X_train: Training point coordinates.
            submodel_data: Training data for each submodel.
            submodel_indices: Point indices for each submodel.
            derivative_specs: Derivative specs for each submodel.

        Returns:
            A dictionary of optimized hyperparameters.
        """
        print("\nBuilding and optimizing the weighted GP model...")
        self.gp_model = wdegp(
            X_train,
            submodel_data,
            self.config.n_order,
            self.config.n_bases,
            submodel_indices,
            derivative_specs,
            normalize=self.config.normalize,
            kernel=self.config.kernel,
            kernel_type=self.config.kernel_type,
        )

        params = self.gp_model.optimize_hyperparameters(
            n_restart_optimizer=self.config.n_restart_optimizer,
            swarm_size=self.config.swarm_size
        )
        print("Hyperparameter optimization complete.")
        return params

    def evaluate_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model performance on a dense test set.

        Args:
            params: Optimized hyperparameters.

        Returns:
            A dictionary containing predictions and performance metrics.
        """
        if self.gp_model is None:
            raise ValueError("GP model has not been built.")

        print(
            f"\nMaking predictions on {self.config.test_points} test points...")
        X_test = np.linspace(
            self.config.lb_x, self.config.ub_x, self.config.test_points
        ).reshape(-1, 1)

        y_pred, y_cov, submodel_vals, submodel_cov = self.gp_model.predict(
            X_test,
            params,
            calc_cov=True,
            return_submodels=True
        )

        y_true = self.true_function(X_test, alg=np)
        nrmse = utils.nrmse(y_true, y_pred)
        print(f"Model evaluation complete. NRMSE: {nrmse:.6f}")

        return {
            'X_test': X_test,
            'y_pred': y_pred,
            'y_true': y_true,
            'y_cov': y_cov,
            'submodel_vals': submodel_vals,
            'submodel_cov': submodel_cov,
            'nrmse': nrmse
        }

    def visualize_results(self, results: Dict[str, Any]):
        """
        Generate comprehensive visualizations of the model and submodels.

        Args:
            results: The evaluation results dictionary.
        """
        print("\nCreating standard visualizations...")
        plotting_helper.make_submodel_plots(
            self.training_data['X_train'],
            self.training_data['submodel_data'],
            results['X_test'],
            results['y_pred'],
            self.true_function,
            cov=results['y_cov'],
            n_order=self.config.n_order,
            n_bases=self.config.n_bases,
            plot_submodels=True,
            submodel_vals=results['submodel_vals'],
            submodel_cov=results['submodel_cov'],
        )

    def analyze_submodel_contributions(self, results: Dict[str, Any]):
        """
        Analyze and visualize individual submodel contributions.

        Args:
            results: Evaluation results dictionary
        """
        print("Creating submodel contribution analysis plot...")
        X_test = results['X_test']
        X_train = self.training_data['X_train']
        submodel_vals = results['submodel_vals']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # --- Plot 1: Individual Submodel Predictions ---
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(submodel_vals)))
        for i, submodel_pred in enumerate(submodel_vals):
            # Label now correctly describes the group of points
            point_group = self.config.submodel_groups[i]
            label = f'Submodel {i+1} (Points {point_group})'
            ax1.plot(X_test.ravel(), submodel_pred.ravel(),
                     color=colors[i], alpha=0.8, linewidth=1.5, label=label)

        # Highlight training points
        ax1.scatter(X_train.ravel(), self.true_function(X_train, alg=np),
                    color='red', s=50, edgecolor='black',
                    label='Training Points', zorder=10)

        ax1.set_title('Individual Submodel Predictions')
        ax1.set_ylabel('f(x)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # --- Plot 2: Combined Prediction vs. True Function ---
        ax2.plot(X_test.ravel(), results['y_true'].ravel(),
                 'b-', linewidth=2, label='True Function')
        ax2.plot(X_test.ravel(), results['y_pred'].ravel(),
                 'r--', linewidth=2, label='Combined GP Prediction')

        # Add uncertainty bands
        std_dev = np.sqrt(results['y_cov'])
        ax2.fill_between(X_test.ravel(),
                         (results['y_pred'].ravel() - 2 * std_dev),
                         (results['y_pred'].ravel() + 2 * std_dev),
                         alpha=0.2, color='red', label='95% Confidence')

        ax2.scatter(X_train.ravel(), self.true_function(X_train, alg=np),
                    color='red', s=50, edgecolor='black',
                    label='Training Points', zorder=10)

        ax2.set_title('Combined Model Prediction with Uncertainty')
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def display_experiment_summary(self, results: Dict[str, Any]):
        """
        Display a summary of the experiment configuration and results.

        Args:
            results: The experimental results.
        """
        print("\n=== Experiment Summary ===")
        print(f"Domain: [{self.config.lb_x}, {self.config.ub_x}]")
        print(f"Total training points: {self.config.num_points}")
        print(f"Number of submodels: {len(self.config.submodel_groups)}")
        print(f"Final NRMSE: {results['nrmse']:.6f}")

        print("\n=== Submodel Configuration ===")
        for i, (indices, order) in enumerate(zip(self.config.submodel_groups, self.config.submodel_orders)):
            num_derivs = sum(len(sublist)
                             for sublist in self.training_data['derivative_specs'][i])
            print(
                f"Submodel {i+1}: Points {indices}, Max Derivative Order: {order}, Total Derivatives: {num_derivs}")

    def run_complete_experiment(self, visualize: bool = True, analyze: bool = True) -> Dict[str, Any]:
        """
        Execute the complete grouped submodel GP experiment from start to finish.

        Args:
            visualize (bool): If True, shows the standard plots from plotting_helper.
            analyze (bool): If True, shows the detailed submodel contribution plot.

        Returns:
            A dictionary containing the complete experimental results.
        """
        print("=== Grouped Submodel Derivative-Enhanced GP Experiment ===")

        # Step 1: Generate training points
        X_train = self.generate_training_points()

        # Step 2: Create submodel structure
        submodel_indices, derivative_specs = self.create_grouped_submodel_structure()

        # Step 3: Prepare training data with derivatives
        submodel_data = self.prepare_grouped_submodel_data(
            X_train, submodel_indices, derivative_specs
        )

        self.training_data = {
            'X_train': X_train,
            'submodel_data': submodel_data,
            'submodel_indices': submodel_indices,
            'derivative_specs': derivative_specs
        }

        # Step 4: Build and optimize GP model
        params = self.build_and_optimize_gp(
            X_train, submodel_data, submodel_indices, derivative_specs)

        # Step 5: Evaluate model performance
        results = self.evaluate_model(params)

        # Step 6: Generate visualizations and summary
        if visualize:
            self.visualize_results(results)
        if analyze:
            self.analyze_submodel_contributions(results)

        self.display_experiment_summary(results)

        return results


def true_function(X, alg=oti):
    """Test function f(x) = x * sin(x)."""
    x = X[:, 0]
    return x * alg.sin(x)


def main():
    """Main execution block to demonstrate the grouped submodel GP."""
    print("Grouped Submodel Derivative-Enhanced GP Demo")
    print("=" * 50)

    # 1. Configure the experiment
    # We will create 2 submodels from 10 points.
    # - The first submodel uses points 0-4 with 2nd order derivatives.
    # - The second submodel uses points 5-9 with 1st order derivatives.
    config = GroupedSubmodelConfig(
        n_order=3,  # Max order needed for any submodel
        num_points=6,
        submodel_groups=[[0, 1, 2], [3, 4, 5]],
        submodel_orders=[2, 3],  # Order for submodel 1, order for submodel 2
        n_restart_optimizer=15,
        swarm_size=200
    )

    # 2. Create and run the experiment
    experiment = GroupedSubmodelGP(config, true_function)
    # Run with both the standard visualization and the new analysis plot
    results = experiment.run_complete_experiment(visualize=True, analyze=True)

    print("\n=== Demo Complete ===")
    print("This example showed how to group points into submodels with different derivative orders.")
    print(
        f"The final model accurately captured the function with an NRMSE of {results['nrmse']:.6f}.")


if __name__ == "__main__":
    main()
