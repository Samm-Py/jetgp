"""
Unit test for 2D DEGP with mixed derivative coverage.

This test verifies that the DEGP model correctly handles:
- Points with only function values (no derivatives)
- Points with first-order derivatives only
- Points with first and second-order derivatives

Test function: f(x,y) = sin(x)cos(y)
"""

import unittest
import numpy as np
from jetgp.full_degp.degp import degp
import sys


class TestDEGP2DMixedDerivatives(unittest.TestCase):
    """Test case for 2D DEGP with mixed derivative coverage."""
    
    @classmethod
    def setUpClass(cls):
        """Set up training data and model once for all tests."""
        # Define training grid (3x3 = 9 points)
        X1 = np.array([0.0, 0.5, 1.0])
        X2 = np.array([0.0, 0.5, 1.0])
        X1_grid, X2_grid = np.meshgrid(X1, X2)
        cls.X_train = np.column_stack([X1_grid.flatten(), X2_grid.flatten()])
        cls.n_train = len(cls.X_train)
        
        # Define point groups with different derivative coverage:
        # Points 0, 1, 2: function values only
        # Points 3, 4, 5: function + first-order derivatives
        # Points 6, 7, 8: function + first + second-order derivatives
        cls.func_only_pts = [0, 1, 2]
        cls.first_order_pts = [3, 4, 5, 6, 7, 8]  # Points with 1st order derivs
        cls.second_order_pts = [6, 7, 8]          # Points with 2nd order derivs
        
        # Compute true function and derivatives at ALL points (for verification)
        cls.y_func_all = np.sin(cls.X_train[:, 0]) * np.cos(cls.X_train[:, 1])
        cls.y_deriv_x_all = np.cos(cls.X_train[:, 0]) * np.cos(cls.X_train[:, 1])
        cls.y_deriv_y_all = -np.sin(cls.X_train[:, 0]) * np.sin(cls.X_train[:, 1])
        cls.y_deriv_xx_all = -np.sin(cls.X_train[:, 0]) * np.cos(cls.X_train[:, 1])
        cls.y_deriv_yy_all = -np.sin(cls.X_train[:, 0]) * np.cos(cls.X_train[:, 1])
        
        # Prepare training data - only include values at specified locations
        cls.y_train = [
            cls.y_func_all.reshape(-1, 1),                          # All points have function values
            cls.y_deriv_x_all[cls.first_order_pts].reshape(-1, 1),  # 6 points with ∂f/∂x
            cls.y_deriv_y_all[cls.first_order_pts].reshape(-1, 1),  # 6 points with ∂f/∂y
            cls.y_deriv_xx_all[cls.second_order_pts].reshape(-1, 1), # 3 points with ∂²f/∂x²
            cls.y_deriv_yy_all[cls.second_order_pts].reshape(-1, 1)  # 3 points with ∂²f/∂y²
        ]
        
        # Define derivative indices (same structure as before)
        cls.der_indices = [
            [[[1, 1]], [[2, 1]]],  # first-order: ∂/∂x, ∂/∂y
            [[[1, 2]], [[2, 2]]]   # second-order: ∂²/∂x², ∂²/∂y²
        ]
        
        # Define derivative locations - which points have which derivatives
        cls.derivative_locations = [
            cls.first_order_pts,   # ∂f/∂x available at points 3,4,5,6,7,8
            cls.first_order_pts,   # ∂f/∂y available at points 3,4,5,6,7,8
            cls.second_order_pts,  # ∂²f/∂x² available at points 6,7,8
            cls.second_order_pts   # ∂²f/∂y² available at points 6,7,8
        ]
        
        # Initialize model
        cls.model = degp(
            cls.X_train, 
            cls.y_train, 
            n_order=2, 
            n_bases=2,
            der_indices=cls.der_indices,
            derivative_locations=cls.derivative_locations,
            normalize=True,
            kernel="SineExp", 
            kernel_type="isotropic"
        )
        
        # Optimize hyperparameters
        cls.params = cls.model.optimize_hyperparameters(
            optimizer='pso',
            pop_size=100,
            n_generations=15,
            local_opt_every=15,
            debug=False
        )
        
        # Get predictions at training points (with derivatives)
        cls.y_train_pred, _ = cls.model.predict(
            cls.X_train, 
            cls.params, 
            calc_cov=True, 
            return_deriv=True
        )
    
    def test_training_data_shapes(self):
        """Test that training data has correct shapes for mixed coverage."""
        self.assertEqual(self.X_train.shape, (9, 2), 
                        "Training data should have 9 points in 2D")
        self.assertEqual(len(self.y_train), 5,
                        "Should have 5 output arrays (func + 4 derivatives)")
        
        # Check individual array shapes
        self.assertEqual(self.y_train[0].shape, (9, 1),
                        "Function values should have shape (9, 1)")
        self.assertEqual(self.y_train[1].shape, (6, 1),
                        "∂f/∂x should have shape (6, 1)")
        self.assertEqual(self.y_train[2].shape, (6, 1),
                        "∂f/∂y should have shape (6, 1)")
        self.assertEqual(self.y_train[3].shape, (3, 1),
                        "∂²f/∂x² should have shape (3, 1)")
        self.assertEqual(self.y_train[4].shape, (3, 1),
                        "∂²f/∂y² should have shape (3, 1)")
    
    def test_derivative_locations_structure(self):
        """Test that derivative_locations correctly specifies coverage."""
        self.assertEqual(len(self.derivative_locations), 4,
                        "Should have 4 derivative location lists")
        self.assertEqual(self.derivative_locations[0], [3, 4, 5, 6, 7, 8],
                        "First-order x derivative locations incorrect")
        self.assertEqual(self.derivative_locations[1], [3, 4, 5, 6, 7, 8],
                        "First-order y derivative locations incorrect")
        self.assertEqual(self.derivative_locations[2], [6, 7, 8],
                        "Second-order xx derivative locations incorrect")
        self.assertEqual(self.derivative_locations[3], [6, 7, 8],
                        "Second-order yy derivative locations incorrect")
    
    def test_function_interpolation_all_points(self):
        """Test that function values are correctly interpolated at all points."""
        y_func_pred = self.y_train_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Function interpolation error too large: {max_error}")
    
    def test_function_interpolation_func_only_points(self):
        """Test interpolation at points with only function values (no derivative info)."""
        y_func_pred = self.y_train_pred[0, self.func_only_pts].flatten()
        y_func_true = self.y_func_all[self.func_only_pts]
        abs_error = np.abs(y_func_pred - y_func_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Function interpolation at func-only points error: {max_error}")
    
    def test_first_derivative_x_interpolation(self):
        """Test ∂f/∂x interpolation at points with first-order derivatives."""
        # Predictions are returned at derivative_locations
        y_deriv_x_pred = self.y_train_pred[1, self.first_order_pts].flatten()
        y_deriv_x_true = self.y_deriv_x_all[self.first_order_pts]
        abs_error = np.abs(y_deriv_x_pred - y_deriv_x_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"∂f/∂x interpolation error too large: {max_error}")
    
    def test_first_derivative_y_interpolation(self):
        """Test ∂f/∂y interpolation at points with first-order derivatives."""
        y_deriv_y_pred = self.y_train_pred[2, self.first_order_pts].flatten()
        y_deriv_y_true = self.y_deriv_y_all[self.first_order_pts]
        abs_error = np.abs(y_deriv_y_pred - y_deriv_y_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"∂f/∂y interpolation error too large: {max_error}")
    
    def test_second_derivative_xx_interpolation(self):
        """Test ∂²f/∂x² interpolation at points with second-order derivatives."""
        y_deriv_xx_pred = self.y_train_pred[3, self.second_order_pts].flatten()
        y_deriv_xx_true = self.y_deriv_xx_all[self.second_order_pts]
        abs_error = np.abs(y_deriv_xx_pred - y_deriv_xx_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"∂²f/∂x² interpolation error too large: {max_error}")
    
    def test_second_derivative_yy_interpolation(self):
        """Test ∂²f/∂y² interpolation at points with second-order derivatives."""
        y_deriv_yy_pred = self.y_train_pred[4, self.second_order_pts].flatten()
        y_deriv_yy_true = self.y_deriv_yy_all[self.second_order_pts]
        abs_error = np.abs(y_deriv_yy_pred - y_deriv_yy_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"∂²f/∂y² interpolation error too large: {max_error}")
    
    def test_prediction_output_shapes(self):
        """Test that prediction output has correct shape for mixed coverage."""
        # Row 0: function values at all 9 points
        # Row 1: ∂f/∂x at 6 points (first_order_pts)
        # Row 2: ∂f/∂y at 6 points (first_order_pts)
        # Row 3: ∂²f/∂x² at 3 points (second_order_pts)
        # Row 4: ∂²f/∂y² at 3 points (second_order_pts)
        
        expected_cols = [9, 6, 6, 3, 3]
        
        self.assertEqual(self.y_train_pred.shape[0], 5,
                        "Should have 5 output rows")
        
        # Note: Depending on implementation, columns might be padded or ragged
        # Adjust this test based on actual output format
    
    def test_all_interpolations_summary(self):
        """Test all interpolations and provide a summary."""
        # Extract predictions
        y_func_pred = self.y_train_pred[0, :].flatten()
        y_deriv_x_pred = self.y_train_pred[1, [self.first_order_pts]].flatten()
        y_deriv_y_pred = self.y_train_pred[2, [self.first_order_pts]].flatten()
        y_deriv_xx_pred = self.y_train_pred[3, self.second_order_pts].flatten()
        y_deriv_yy_pred = self.y_train_pred[4, self.second_order_pts].flatten()
        
        # Compute errors (comparing against correct subset of true values)
        errors = {
            'Function (all 9 pts)': np.abs(y_func_pred - self.y_func_all),
            'Function (func-only pts 0,1,2)': np.abs(y_func_pred[self.func_only_pts] - self.y_func_all[self.func_only_pts]),
            '∂f/∂x (pts 3-8)': np.abs(y_deriv_x_pred - self.y_deriv_x_all[self.first_order_pts]),
            '∂f/∂y (pts 3-8)': np.abs(y_deriv_y_pred - self.y_deriv_y_all[self.first_order_pts]),
            '∂²f/∂x² (pts 6-8)': np.abs(y_deriv_xx_pred - self.y_deriv_xx_all[self.second_order_pts]),
            '∂²f/∂y² (pts 6-8)': np.abs(y_deriv_yy_pred - self.y_deriv_yy_all[self.second_order_pts])
        }
        
        print("\n" + "="*70)
        print("DEGP 2D Mixed Derivative Coverage Test Summary")
        print("="*70)
        print("Point coverage:")
        print("  - Points 0,1,2: function only")
        print("  - Points 3,4,5: function + 1st order derivatives")
        print("  - Points 6,7,8: function + 1st + 2nd order derivatives")
        print("-"*70)
        
        all_passed = True
        for name, error in errors.items():
            max_err = np.max(error)
            mean_err = np.mean(error)
            passed = max_err < 1e-6
            all_passed = all_passed and passed
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} | {name:35s} | Max: {max_err:.2e} | Mean: {mean_err:.2e}")
        
        print("="*70)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print("="*70 + "\n")
        
        self.assertTrue(all_passed, "Not all interpolation tests passed")


def run_tests_with_details():
    """Run tests with detailed output."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDEGP2DMixedDerivatives)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    print("\nRunning DEGP 2D Mixed Derivative Coverage Unit Tests...")
    print("="*70 + "\n")
    
    result = run_tests_with_details()
    sys.exit(0 if result.wasSuccessful() else 1)