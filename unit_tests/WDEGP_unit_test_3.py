"""
Unit tests for Sparse Weighted DEGP (Example 2)
Tests the 1D Sparse WDEGP with selective derivative observations
"""

import unittest
import numpy as np
import sympy as sp
from jetgp.wdegp.wdegp import wdegp
import jetgp.utils as utils


class TestSparseWDEGP1D(unittest.TestCase):
    """Test sparse WDEGP with derivatives at subset of training points"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests"""
        # Parameters
        cls.n_order = 2
        cls.n_bases = 1
        cls.lb_x = 0.5
        cls.ub_x = 2.5
        cls.num_points = 10
        cls.derivative_indices = [2, 3, 4, 5]
        
        # Define symbolic function
        x = sp.symbols('x')
        f_sym = sp.sin(10 * sp.pi * x) / (2 * x) + (x - 1)**4
        f1_sym = sp.diff(f_sym, x)
        f2_sym = sp.diff(f_sym, x, 2)
        
        # Convert to callable functions and vectorize
        cls.f_fun = np.vectorize(sp.lambdify(x, f_sym, "numpy"))
        cls.f1_fun = np.vectorize(sp.lambdify(x, f1_sym, "numpy"))
        cls.f2_fun = np.vectorize(sp.lambdify(x, f2_sym, "numpy"))
        
        # Generate training data
        cls.X_train = np.linspace(cls.lb_x, cls.ub_x, cls.num_points).reshape(-1, 1)
        
        # Compute function values at all points
        cls.y_function_values = cls.f_fun(cls.X_train.flatten()).reshape(-1, 1)
        
        # Compute derivatives only at selected points
        cls.d1_all = np.array([[cls.f1_fun(cls.X_train[idx, 0]) 
                                for idx in cls.derivative_indices]]).T
        cls.d2_all = np.array([[cls.f2_fun(cls.X_train[idx, 0]) 
                                for idx in cls.derivative_indices]]).T
        
        # Define sparse derivative structure
        cls.submodel_indices = [cls.derivative_indices]
        cls.base_derivative_indices = utils.gen_OTI_indices(cls.n_bases, cls.n_order)
        cls.derivative_specs = [cls.base_derivative_indices]
        cls.submodel_data = [[cls.y_function_values, cls.d1_all, cls.d2_all]]
        
        # Build and optimize model
        cls.gp_model = wdegp(
            cls.X_train,
            cls.submodel_data,
            cls.n_order,
            cls.n_bases,
            cls.submodel_indices,
            cls.derivative_specs,
            normalize=True,
            kernel="Matern",
            kernel_type="isotropic",
            smoothness_parameter=4
        )
        
        cls.params = cls.gp_model.optimize_hyperparameters(
            optimizer='pso',
            pop_size=200,
            n_generations=15,
            local_opt_every=15,
            debug=False
        )
        print(f"\nOptimized parameters: {cls.params}")
    
    def test_configuration(self):
        """Test that sparse configuration is correct"""
        self.assertEqual(len(self.submodel_indices), 1, 
                        "Should have exactly one submodel")
        self.assertEqual(len(self.derivative_indices), 4,
                        "Should have derivatives at 4 points")
        self.assertEqual(self.num_points, 10,
                        "Should have 10 total training points")
        self.assertLess(len(self.derivative_indices), self.num_points,
                       "Sparse: derivatives at fewer points than total")
    
    def test_training_data_structure(self):
        """Test that training data has correct structure"""
        # Function values at all points
        self.assertEqual(self.y_function_values.shape, (self.num_points, 1),
                        "Function values should be at all training points")
        
        # Derivatives only at sparse points
        self.assertEqual(self.d1_all.shape, (len(self.derivative_indices), 1),
                        "First derivatives only at sparse points")
        self.assertEqual(self.d2_all.shape, (len(self.derivative_indices), 1),
                        "Second derivatives only at sparse points")
    
    def test_function_value_interpolation(self):
        """Test that function values are interpolated at all training points"""
        y_pred_train = self.gp_model.predict(self.X_train, self.params, calc_cov=False)
        
        errors = np.abs(y_pred_train.flatten() - self.y_function_values.flatten())
        max_error = np.max(errors)
        
        # Should interpolate function values exactly (within tolerance)
        self.assertLess(max_error, 1e-6,
                       f"Function interpolation error too large: {max_error:.2e}")
        
        # Check each point individually
        for i in range(self.num_points):
            error = abs(y_pred_train[0, i] - self.y_function_values[i, 0])
            self.assertLess(error, 1e-6,
                           f"Point {i} interpolation error: {error:.2e}")
    
    def test_first_derivative_interpolation_sparse(self):
        """Test first derivative interpolation at sparse points only"""
        max_error = 0.0
        
        for idx, train_idx in enumerate(self.derivative_indices):
            X_point = self.X_train[train_idx].reshape(1, -1)
            
            # Get prediction with derivatives
            # f_mean is 1D: [f, df/dx, d2f/dx2]
            f_mean = self.gp_model.predict(
                X_point, self.params, calc_cov=False, return_submodels=False, return_deriv=True
            )
            
            # Extract first derivative
            predicted_first_deriv = f_mean[1]
            analytic_first_deriv = self.d1_all[idx, 0]
            
            error = abs(predicted_first_deriv - analytic_first_deriv)[0]
            max_error = max(max_error, error)
            
            self.assertLess(error, 1e-6,
                           f"First derivative error at point {train_idx}: {error:.2e}")
        
        print(f"Max first derivative error at sparse points: {max_error:.2e}")
    
    def test_second_derivative_interpolation_sparse(self):
        """Test second derivative interpolation at sparse points only"""
        max_error = 0.0
        
        for idx, train_idx in enumerate(self.derivative_indices):
            X_point = self.X_train[train_idx].reshape(1, -1)
            
            # Get prediction with derivatives
            # f_mean is 1D: [f, df/dx, d2f/dx2]
            f_mean = self.gp_model.predict(
                X_point, self.params, calc_cov=False, return_submodels=False, return_deriv=True
            )
            
            # Extract second derivative
            predicted_second_deriv = f_mean[2]
            analytic_second_deriv = self.d2_all[idx, 0]
            
            error = abs(predicted_second_deriv - analytic_second_deriv)[0]
            max_error = max(max_error, error)
            
            self.assertLess(error, 1e-6,
                           f"Second derivative error at point {train_idx}: {error:.2e}")
        
        print(f"Max second derivative error at sparse points: {max_error:.2e}")
    
    def test_single_submodel_structure(self):
        """Test that sparse formulation uses single submodel"""
        X_test = np.linspace(self.lb_x, self.ub_x, 50).reshape(-1, 1)
        _, _, submodel_vals, _ = self.gp_model.predict(
            X_test, self.params, calc_cov=True, return_submodels=True
        )
        
        self.assertEqual(len(submodel_vals), 1,
                        "Sparse formulation should have exactly one submodel")
        self.assertEqual(submodel_vals[0].shape[1], X_test.shape[0],
                        "Submodel predictions should match test set size")
    
    def test_confidence_intervals(self):
        """Test that confidence intervals are reasonable"""
        X_test = np.linspace(self.lb_x, self.ub_x, 100).reshape(-1, 1)
        y_pred, y_cov = self.gp_model.predict(X_test, self.params, calc_cov=True)
        
        # Variance should be positive
        self.assertTrue(np.all(y_cov >= 0),
                       "Covariance should be non-negative")
        
        # Uncertainty should be smaller near training points
        distances_to_train = np.min([np.abs(X_test - x) for x in self.X_train], axis=0)
        
        # Points close to training should have lower variance
        close_points = distances_to_train < 0.1
        far_points = distances_to_train > 0.5
        
        if np.any(close_points) and np.any(far_points):
            mean_var_close = np.mean(y_cov[close_points.flatten()])
            mean_var_far = np.mean(y_cov[far_points.flatten()])
            
            self.assertLess(mean_var_close, mean_var_far,
                           "Variance should be lower near training points")
    
    def test_comprehensive_summary(self):
        """Print comprehensive summary of all tests"""
        print("\n" + "="*80)
        print("SPARSE WDEGP COMPREHENSIVE TEST SUMMARY")
        print("="*80)
        print(f"Configuration:")
        print(f"  - Total training points: {self.num_points}")
        print(f"  - Sparse derivative points: {len(self.derivative_indices)} {self.derivative_indices}")
        print(f"  - Derivative order: {self.n_order}")
        print(f"  - Number of submodels: {len(self.submodel_indices)}")
        print(f"  - Formulation: Single global model with sparse derivatives")
        
        # Function interpolation
        y_pred_train = self.gp_model.predict(self.X_train, self.params, calc_cov=False)
        func_errors = np.abs(y_pred_train.flatten() - self.y_function_values.flatten())
        print(f"\nFunction Interpolation (all {self.num_points} points):")
        print(f"  - Max absolute error: {np.max(func_errors):.2e}")
        print(f"  - Mean absolute error: {np.mean(func_errors):.2e}")
        
        # Derivative interpolation at sparse points using return_deriv=True
        first_deriv_errors = []
        second_deriv_errors = []
        
        for idx, train_idx in enumerate(self.derivative_indices):
            X_point = self.X_train[train_idx].reshape(1, -1)
            
            # Get prediction with derivatives
            f_mean = self.gp_model.predict(
                X_point, self.params, calc_cov=False, return_submodels=False, return_deriv=True
            )
            
            # Extract derivatives from 1D array: [f, df/dx, d2f/dx2]
            predicted_first = f_mean[1]
            predicted_second = f_mean[2]
            
            first_deriv_errors.append(abs(predicted_first - self.d1_all[idx, 0]))
            second_deriv_errors.append(abs(predicted_second - self.d2_all[idx, 0]))
        
        print(f"\nFirst Derivative Interpolation (sparse points only):")
        print(f"  - Max absolute error: {np.max(first_deriv_errors):.2e}")
        print(f"  - Mean absolute error: {np.mean(first_deriv_errors):.2e}")
        
        print(f"\nSecond Derivative Interpolation (sparse points only):")
        print(f"  - Max absolute error: {np.max(second_deriv_errors):.2e}")
        print(f"  - Mean absolute error: {np.mean(second_deriv_errors):.2e}")
        
        print("="*80)
        print("All tests completed successfully using return_deriv=True")
        print("="*80 + "\n")


if __name__ == '__main__':
    unittest.main(verbosity=2)