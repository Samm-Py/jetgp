"""
Unit test for 1D Weighted DEGP with individual submodels.

This test verifies that the WDEGP model correctly interpolates training data
for a 1D oscillatory function with trend. Each training point has its own submodel
with function values and up to 2nd order derivatives.

Test function: sin(10πx)/(2x) + (x-1)^4
"""

import sys
import unittest
import numpy as np
import sympy as sp
from jetgp.wdegp.wdegp import wdegp
import jetgp.utils as utils


class TestWDEGP1DIndividual(unittest.TestCase):
    """Test case for 1D WDEGP with individual submodels (one per training point)."""
    
    @classmethod
    def setUpClass(cls):
        """Set up training data and model once for all tests."""
        # Configuration
        cls.n_bases = 1
        cls.n_order = 2
        cls.lb_x = 0.5
        cls.ub_x = 2.5
        cls.num_points = 10
        cls.kernel = "SE"
        cls.kernel_type = "isotropic"
        cls.normalize = True
        
        # Generate training points
        cls.X_train = np.linspace(cls.lb_x, cls.ub_x, cls.num_points).reshape(-1, 1)
        
        # Each submodel corresponds to one training point
        cls.submodel_indices = [[i] for i in range(cls.num_points)]
        
        # Compute derivatives using SymPy
        cls.submodel_data, cls.y_function_values = cls._compute_symbolic_derivatives()
        
        # Create index array for submodels (each submodel has 1 point)
        cls.index = np.array([1 for _ in range(cls.num_points)])
        
        # Generate derivative specifications
        cls.base_derivative_indices = utils.gen_OTI_indices(cls.n_bases, cls.n_order)
        cls.derivative_specs = [cls.base_derivative_indices for _ in range(cls.num_points)]
        
        # Initialize WDEGP model
        cls.model = wdegp(
            cls.X_train,
            cls.submodel_data,
            cls.n_order,
            cls.n_bases,
            cls.submodel_indices,
            cls.derivative_specs,
            normalize=cls.normalize,
            kernel=cls.kernel,
            kernel_type=cls.kernel_type,
        )
        
        # Optimize hyperparameters
        cls.params = cls.model.optimize_hyperparameters(
            optimizer='lbfgs',
            n_restart_optimizer = 10,
            debug=False
        )
    
    @classmethod
    def _oscillatory_function_symbolic(cls):
        """Define the oscillatory function symbolically."""
        x = sp.symbols('x', real=True)
        f = sp.sin(10 * sp.pi * x) / (2 * x) + (x - 1)**4
        return x, f
    
    @classmethod
    def _compute_symbolic_derivatives(cls):
        """Compute derivatives symbolically using SymPy."""
        x_sym, f_sym = cls._oscillatory_function_symbolic()
        
        # Compute derivatives
        df_dx = sp.diff(f_sym, x_sym)
        d2f_dx2 = sp.diff(df_dx, x_sym)
        
        # Lambdify for numerical evaluation
        f_func_raw = sp.lambdify(x_sym, f_sym, 'numpy')
        df_dx_func_raw = sp.lambdify(x_sym, df_dx, 'numpy')
        d2f_dx2_func_raw = sp.lambdify(x_sym, d2f_dx2, 'numpy')
        
        # Wrap to ensure array output (handle constants)
        def make_array_func(func_raw):
            def wrapped(x):
                result = func_raw(x)
                result = np.atleast_1d(result)
                # If result is a scalar (constant function), broadcast to input size
                if result.size == 1 and np.atleast_1d(x).size > 1:
                    result = np.full_like(x, result[0])
                return result
            return wrapped
        
        f_func = make_array_func(f_func_raw)
        df_dx_func = make_array_func(df_dx_func_raw)
        d2f_dx2_func = make_array_func(d2f_dx2_func_raw)
        
        # Compute function values at ALL training points
        y_function_values = f_func(cls.X_train.flatten()).reshape(-1, 1)
        
        # Prepare data for each submodel
        # Each submodel gets: [all function values] + [derivatives at its own point]
        submodel_data = []
        
        for k, idx in enumerate(cls.submodel_indices):
            x_point = cls.X_train[idx].flatten()[0]
            
            # Compute derivatives at this submodel's point
            d1 = df_dx_func(np.array([x_point])).reshape(-1, 1)
            d2 = d2f_dx2_func(np.array([x_point])).reshape(-1, 1)
            
            # Append [all function values] + [local derivatives]
            submodel_data.append([y_function_values, d1, d2])
        
        return submodel_data, y_function_values
    
    def test_training_data_structure(self):
        """Test that training data has correct structure."""
        self.assertEqual(self.X_train.shape, (10, 1),
                        "Training data should have 10 points in 1D")
        self.assertEqual(len(self.submodel_data), 10,
                        "Should have 10 submodels")
        
        # Check each submodel
        for i in range(10):
            self.assertEqual(len(self.submodel_data[i]), 3,
                            f"Submodel {i} should have func + 2 derivatives")
            self.assertEqual(self.submodel_data[i][0].shape, (10, 1),
                            f"Submodel {i} function values should be for ALL 10 points")
            self.assertEqual(self.submodel_data[i][1].shape, (1, 1),
                            f"Submodel {i} first derivative should be for 1 point only")
            self.assertEqual(self.submodel_data[i][2].shape, (1, 1),
                            f"Submodel {i} second derivative should be for 1 point only")
    
    def test_function_value_interpolation(self):
        """Test that function values are correctly interpolated at all training points."""
        y_pred_train,_ = self.model.predict(self.X_train, self.params, calc_cov=True)
        
        abs_errors = np.abs(y_pred_train.flatten() - self.y_function_values.flatten())
        max_error = np.max(abs_errors)
        mean_error = np.mean(abs_errors)
        
        self.assertLess(max_error, 1e-6,
                       f"Function value interpolation error too large: {max_error}")
        self.assertLess(mean_error, 1e-7,
                       f"Mean function value interpolation error too large: {mean_error}")
    
    def test_first_derivative_interpolation(self):
        """Test first derivative interpolation at all training points."""
        h = 1e-6
        max_errors = []
        
        for i, idx in enumerate(self.submodel_indices):
            x_point = self.X_train[idx].flatten()[0]
            

            f_mean = self.model.predict(
                np.array([x_point]), self.params, calc_cov=False, return_submodels=False, return_deriv=True
            )

            
            # First derivative via central difference on the i-th submodel
            fd_first_deriv = f_mean[1,0]
            analytic_first_deriv = self.submodel_data[i][1][0, 0]
            
            error = abs(fd_first_deriv - analytic_first_deriv)
            max_errors.append(error)
            
            self.assertLess(error, 1e-4,
                           f"First derivative error at point {i} too large: {error}")
        
        max_error_overall = max(max_errors)
        self.assertLess(max_error_overall, 1e-4,
                       f"Maximum first derivative error: {max_error_overall}")
    
    def test_second_derivative_interpolation(self):
        """Test second derivative interpolation at all training points."""
        h = 1e-6
        max_errors = []
        
        for i, idx in enumerate(self.submodel_indices):
            x_point = self.X_train[idx].flatten()[0]
            

            X_center = self.X_train[idx].reshape(1, -1)
            
            f_mean = self.model.predict(
                X_center, self.params, calc_cov=False, return_submodels=False, return_deriv=True
            )
            
            # Second derivative via central difference on the i-th submodel
            fd_second_deriv = f_mean[2,0]
            analytic_second_deriv = self.submodel_data[i][2][0, 0]
            
            error = abs(fd_second_deriv - analytic_second_deriv)
            max_errors.append(error)
            
            self.assertLess(error, 1e-2,
                           f"Second derivative error at point {i} too large: {error}")
        
        max_error_overall = max(max_errors)
        self.assertLess(max_error_overall, 1e-2,
                       f"Maximum second derivative error: {max_error_overall}")
    
    def test_comprehensive_summary(self):
        """Comprehensive test with detailed summary."""
        print("\n" + "="*80)
        print("WDEGP 1D Individual Submodels Interpolation Test Summary")
        print("="*80)
        
        all_tests_passed = True
        
        # Test function values
        print("\nFunction Value Interpolation")
        print("-" * 80)
        y_pred_train = self.model.predict(self.X_train, self.params, calc_cov=False)
        abs_errors = np.abs(y_pred_train.flatten() - self.y_function_values.flatten())
        max_error = np.max(abs_errors)
        mean_error = np.mean(abs_errors)
        
        passed = max_error < 1e-6
        all_tests_passed = all_tests_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Function values | Max: {max_error:.2e} | Mean: {mean_error:.2e}")
        
        # Test first derivatives
        print("\nFirst Derivative Interpolation (via finite differences)")
        print("-" * 80)
        h = 1e-6
        first_deriv_errors = []
        
        for i, idx in enumerate(self.submodel_indices):
            x_point = self.X_train[idx].flatten()[0]
            

            f_mean = self.model.predict(
                np.array([x_point]), self.params, calc_cov=False, return_submodels=False, return_deriv=True
            )

            
            # First derivative via central difference on the i-th submodel
            fd_first_deriv = f_mean[1,0]
            analytic_first_deriv = self.submodel_data[i][1][0, 0]
            error = abs(fd_first_deriv - analytic_first_deriv)
            first_deriv_errors.append(error)
        
        max_first_error = max(first_deriv_errors)
        mean_first_error = np.mean(first_deriv_errors)
        
        passed = max_first_error < 1e-4
        all_tests_passed = all_tests_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | First derivatives | Max: {max_first_error:.2e} | Mean: {mean_first_error:.2e}")
        
        # Test second derivatives
        print("\nSecond Derivative Interpolation (via finite differences)")
        print("-" * 80)
        second_deriv_errors = []
        
        for i, idx in enumerate(self.submodel_indices):
            x_point = self.X_train[idx].flatten()[0]
            

            X_center = self.X_train[idx].reshape(1, -1)
            
            f_mean = self.model.predict(
                X_center, self.params, calc_cov=False, return_submodels=False, return_deriv=True
            )
            
            # Second derivative via central difference on the i-th submodel
            fd_second_deriv = f_mean[2,0]
            analytic_second_deriv = self.submodel_data[i][2][0, 0]
            error = abs(fd_second_deriv - analytic_second_deriv)
            second_deriv_errors.append(error)
        
        max_second_error = max(second_deriv_errors)
        mean_second_error = np.mean(second_deriv_errors)
        
        passed = max_second_error < 1e-2
        all_tests_passed = all_tests_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Second derivatives | Max: {max_second_error:.2e} | Mean: {mean_second_error:.2e}")
        
        print("\n" + "="*80)
        print("Summary:")
        print(f"  - Total training points: {self.num_points}")
        print(f"  - Submodels: {len(self.submodel_data)} (one per training point)")
        print(f"  - Derivative order: up to 2nd order")
        print(f"  - Test function: sin(10πx)/(2x) + (x-1)^4")
        print(f"  - Derivatives computed symbolically with SymPy")
        print(f"  - Verification: finite differences on individual submodels")
        print("="*80)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_tests_passed else '✗ SOME TESTS FAILED'}")
        print("="*80 + "\n")
        
        self.assertTrue(all_tests_passed, "Not all interpolation tests passed")


def run_tests_with_details():
    """Run tests with detailed output."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWDEGP1DIndividual)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    print("\nRunning WDEGP 1D Individual Submodels Interpolation Unit Tests...")
    print("="*80 + "\n")
    
    result = run_tests_with_details()
    
    sys.exit(0 if result.wasSuccessful() else 1)