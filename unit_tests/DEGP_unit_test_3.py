"""
Unit test for 2D DEGP with second-order derivatives.

This test verifies that the DEGP model correctly interpolates training data
including function values, first-order derivatives, and second-order main 
derivatives for f(x,y) = sin(x)cos(y).
"""

import unittest
import numpy as np
from jetgp.full_degp.degp import degp
import sys

class TestDEGP2DSecondOrderV2(unittest.TestCase):
    """Test case for 2D DEGP with second-order derivatives."""
    
    @classmethod
    def setUpClass(cls):
        """Set up training data and model once for all tests."""
        # Define training grid
        X1 = np.array([0.0, 0.5, 1.0])
        X2 = np.array([0.0, 0.5, 1.0])
        X1_grid, X2_grid = np.meshgrid(X1, X2)
        cls.X_train = np.column_stack([X1_grid.flatten(), X2_grid.flatten()])
        # Define derivative indices
        cls.der_indices = [
            [[[1, 1]], [[2, 1]]],  # first-order derivatives
            [[[1, 2]], [[2, 2]]]   # second-order derivatives
        ]
        cls.derivative_locations = [[0,1,2,3,7,8],[4,5,6],[0,1,2,3,4],[5,6,7,8]]
        # Compute true function and derivatives
        cls.y_func = np.sin(cls.X_train[:, 0]) * np.cos(cls.X_train[:, 1])
        cls.y_deriv_x = np.cos(cls.X_train[cls.derivative_locations[0], 0]) * np.cos(cls.X_train[cls.derivative_locations[0],1])
        cls.y_deriv_y = -np.sin(cls.X_train[cls.derivative_locations[1],0]) * np.sin(cls.X_train[cls.derivative_locations[1],1])
        cls.y_deriv_xx = -np.sin(cls.X_train[cls.derivative_locations[2],0]) * np.cos(cls.X_train[cls.derivative_locations[2],1])
        cls.y_deriv_yy = -np.sin(cls.X_train[cls.derivative_locations[3],0]) * np.cos(cls.X_train[cls.derivative_locations[3],1])
        
        # Prepare training data list
        cls.y_train = [
            cls.y_func.reshape(-1, 1),
            cls.y_deriv_x.reshape(-1, 1),
            cls.y_deriv_y.reshape(-1, 1),
            cls.y_deriv_xx.reshape(-1, 1),
            cls.y_deriv_yy.reshape(-1, 1)
        ]
        

        
        # Initialize model
        cls.model = degp(
            cls.X_train, 
            cls.y_train, 
            n_order=2, 
            n_bases=2,
            der_indices=cls.der_indices, 
            derivative_locations=cls.derivative_locations,
            normalize=False,
            kernel="RQ", 
            kernel_type="anisotropic"
        )
        
        # Optimize hyperparameters
        cls.params = cls.model.optimize_hyperparameters(
            optimizer='pso',
            pop_size=100,
            n_generations=15,
            local_opt_every=15,
            debug=False
        )
        
        # Get predictions at training points
        cls.y_train_pred, _ = cls.model.predict(
            cls.X_train, 
            cls.params, 
            calc_cov=True, 
            return_deriv=True
        )
        
        cls.n_train = len(cls.X_train)
    
    def test_function_interpolation(self):
        """Test that function values are correctly interpolated."""
        y_func_pred = self.y_train_pred[0,:].flatten()
        abs_error = np.abs(y_func_pred - self.y_func)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Function interpolation error too large: {max_error}")
        
        # Also check mean error
        mean_error = np.mean(abs_error)
        self.assertLess(mean_error, 1e-7,
                       f"Mean function interpolation error too large: {mean_error}")
    
    def test_first_derivative_x_interpolation(self):
        """Test that first derivative w.r.t. x is correctly interpolated."""
        y_deriv_x_pred = self.y_train_pred[1,:].flatten()
        abs_error = np.abs(y_deriv_x_pred[self.derivative_locations[0]] - self.y_deriv_x)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"First derivative (x) interpolation error too large: {max_error}")
        
        mean_error = np.mean(abs_error)
        self.assertLess(mean_error, 1e-7,
                       f"Mean first derivative (x) interpolation error too large: {mean_error}")
    
    def test_first_derivative_y_interpolation(self):
        """Test that first derivative w.r.t. y is correctly interpolated."""
        y_deriv_y_pred = self.y_train_pred[2,:].flatten()
        abs_error = np.abs(y_deriv_y_pred[self.derivative_locations[1]] - self.y_deriv_y)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"First derivative (y) interpolation error too large: {max_error}")
        
        mean_error = np.mean(abs_error)
        self.assertLess(mean_error, 1e-7,
                       f"Mean first derivative (y) interpolation error too large: {mean_error}")
    
    def test_second_derivative_xx_interpolation(self):
        """Test that second derivative w.r.t. x² is correctly interpolated."""
        y_deriv_xx_pred = self.y_train_pred[3,:].flatten()
        abs_error = np.abs(y_deriv_xx_pred[self.derivative_locations[2]] - self.y_deriv_xx)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Second derivative (xx) interpolation error too large: {max_error}")
        
        mean_error = np.mean(abs_error)
        self.assertLess(mean_error, 1e-7,
                       f"Mean second derivative (xx) interpolation error too large: {mean_error}")
    
    def test_second_derivative_yy_interpolation(self):
        """Test that second derivative w.r.t. y² is correctly interpolated."""
        y_deriv_yy_pred = self.y_train_pred[4,:].flatten()
        abs_error = np.abs(y_deriv_yy_pred[self.derivative_locations[3]] - self.y_deriv_yy)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Second derivative (yy) interpolation error too large: {max_error}")
        
        mean_error = np.mean(abs_error)
        self.assertLess(mean_error, 1e-7,
                       f"Mean second derivative (yy) interpolation error too large: {mean_error}")
    
    def test_all_interpolations_summary(self):
        """Test all interpolations and provide a summary."""
        # Extract all predictions
        y_func_pred = self.y_train_pred[0,:].flatten()
        y_deriv_x_pred = self.y_train_pred[1,:].flatten()
        y_deriv_y_pred = self.y_train_pred[2,:].flatten()
        y_deriv_xx_pred = self.y_train_pred[3,:].flatten()
        y_deriv_yy_pred = self.y_train_pred[4,:].flatten()
        
        # Compute errors
        errors = {
            'Function': np.abs(y_func_pred - self.y_func),
            'Derivative ∂f/∂x': np.abs(y_deriv_x_pred[self.derivative_locations[0]] - self.y_deriv_x),
            'Derivative ∂f/∂y': np.abs(y_deriv_y_pred[self.derivative_locations[1]] - self.y_deriv_y),
            'Second derivative ∂²f/∂x²': np.abs(y_deriv_xx_pred[self.derivative_locations[2]] - self.y_deriv_xx),
            'Second derivative ∂²f/∂y²': np.abs(y_deriv_yy_pred[self.derivative_locations[3]]- self.y_deriv_yy)
        }
        
        print("\n" + "="*60)
        print("DEGP 2D Second-Order Interpolation Test Summary")
        print("="*60)
        
        all_passed = True
        for name, error in errors.items():
            max_err = np.max(error)
            mean_err = np.mean(error)
            passed = max_err < 1e-6
            all_passed = all_passed and passed
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} | {name:30s} | Max: {max_err:.2e} | Mean: {mean_err:.2e}")
        
        print("="*60)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print("="*60 + "\n")
        
        self.assertTrue(all_passed, "Not all interpolation tests passed")
    


def run_tests_with_details():
    """Run tests with detailed output."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDEGP2DSecondOrderV2)
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    # Run tests
    print("\nRunning DEGP 2D Second-Order Interpolation Unit Tests...")
    print("="*60 + "\n")
    
    result = run_tests_with_details()
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)