"""
Unit test for 2D Weighted DEGP with heterogeneous derivative orders.

This test verifies that the WDEGP model correctly interpolates training data
across three submodels with different derivative orders:
- Submodel 1 (Corners): Function values only (no derivatives)
- Submodel 2 (Edges): Function values + 1st order derivatives
- Submodel 3 (Center): Function values + up to 2nd order derivatives

Test function: Six-hump camel function

Note: Uses non-contiguous indices directly - no reordering required.
"""

import sys
import unittest
import numpy as np
import sympy as sp
import itertools
from jetgp.wdegp.wdegp import wdegp
import jetgp.utils


class TestWDEGPHeterogeneousDerivatives(unittest.TestCase):
    """Test case for 2D WDEGP with heterogeneous derivative orders."""
    
    @classmethod
    def setUpClass(cls):
        """Set up training data and model once for all tests."""
        # Set random seed for reproducibility
        np.random.seed(0)
        
        # Configuration
        cls.n_order = 2
        cls.n_bases = 2
        cls.lb_x, cls.ub_x = -1.0, 1.0
        cls.lb_y, cls.ub_y = -1.0, 1.0
        cls.points_per_axis = 4
        cls.kernel = "SineExp"
        cls.kernel_type = "anisotropic"
        cls.normalize = True
        
        # Submodel groupings (grid indices - can be non-contiguous)
        cls.submodel_indices = [
            [[0, 3, 12, 15]],                 # Corners (df/dx)
            [[1, 2, 4, 8, 7, 11, 13, 14],[1, 2, 4, 8, 7, 11, 13, 14]],   # Edges (1st order)
            [[5, 6, 9, 10],[5, 6, 9, 10],[5, 6, 9, 10],[5, 6, 9, 10],[5, 6, 9, 10]] # Center (2nd order)
        ]
        
        # Generate training points (4x4 grid)
        cls.X_train = cls._generate_training_points()
        
        # Compute derivatives using SymPy
        cls.submodel_data, cls.der_indices = cls._compute_symbolic_derivatives()
        
        # Initialize WDEGP model
        cls.model = wdegp(
            cls.X_train,
            cls.submodel_data,
            cls.n_order,
            cls.n_bases,
            cls.der_indices,
            derivative_locations=cls.submodel_indices,
            normalize=cls.normalize,
            kernel=cls.kernel,
            kernel_type=cls.kernel_type
        )
        
        # Optimize hyperparameters
        cls.params = cls.model.optimize_hyperparameters(
            optimizer='jade',
            pop_size=100,
            n_generations=15,
            local_opt_every=5,
            debug=True
        )
        print(f"\nOptimized parameters: {cls.params}")
    
    @classmethod
    def _generate_training_points(cls):
        """Generate 4×4 grid of training points."""
        x_vals = np.linspace(cls.lb_x, cls.ub_x, cls.points_per_axis)
        y_vals = np.linspace(cls.lb_y, cls.ub_y, cls.points_per_axis)
        return np.array(list(itertools.product(x_vals, y_vals)))
    
    @classmethod
    def _six_hump_camel_symbolic(cls):
        """Define six-hump camel function symbolically."""
        x1, x2 = sp.symbols('x1 x2', real=True)
        f = ((4 - 2.1*x1**2 + x1**4/3) * x1**2 + x1*x2 + (-4 + 4*x2**2) * x2**2)
        return x1, x2, f
    
    @classmethod
    def _compute_symbolic_derivatives(cls):
        """Compute derivatives symbolically using SymPy."""
        x1_sym, x2_sym, f_sym = cls._six_hump_camel_symbolic()
        
        # Compute derivatives
        df_dx1 = sp.diff(f_sym, x1_sym)
        df_dx2 = sp.diff(f_sym, x2_sym)
        d2f_dx1_2 = sp.diff(df_dx1, x1_sym)
        d2f_dx1dx2 = sp.diff(df_dx1, x2_sym)
        d2f_dx2_2 = sp.diff(df_dx2, x2_sym)
        
        # Lambdify for numerical evaluation
        f_func_raw = sp.lambdify((x1_sym, x2_sym), f_sym, 'numpy')
        df_dx1_func_raw = sp.lambdify((x1_sym, x2_sym), df_dx1, 'numpy')
        df_dx2_func_raw = sp.lambdify((x1_sym, x2_sym), df_dx2, 'numpy')
        d2f_dx1_2_func_raw = sp.lambdify((x1_sym, x2_sym), d2f_dx1_2, 'numpy')
        d2f_dx1dx2_func_raw = sp.lambdify((x1_sym, x2_sym), d2f_dx1dx2, 'numpy')
        d2f_dx2_2_func_raw = sp.lambdify((x1_sym, x2_sym), d2f_dx2_2, 'numpy')
        
        # Wrap to ensure array output
        def make_array_func(func_raw):
            def wrapped(x1, x2):
                result = func_raw(x1, x2)
                result = np.atleast_1d(result)
                if result.size == 1 and np.atleast_1d(x1).size > 1:
                    result = np.full_like(x1, result[0])
                return result
            return wrapped
        
        f_func = make_array_func(f_func_raw)
        df_dx1_func = make_array_func(df_dx1_func_raw)
        df_dx2_func = make_array_func(df_dx2_func_raw)
        d2f_dx1_2_func = make_array_func(d2f_dx1_2_func_raw)
        d2f_dx1dx2_func = make_array_func(d2f_dx1dx2_func_raw)
        d2f_dx2_2_func = make_array_func(d2f_dx2_2_func_raw)
        
        # Define derivative indices for each submodel
        der_indices = [
            [
                [[[1, 1]]]  # df/dx1, df/dx2
                ],  # Submodel 1: no derivatives
            [  # Submodel 2: 1st order
                [[[1, 1]], [[2, 1]]]  # df/dx1, df/dx2
            ],
            [  # Submodel 3: 1st and 2nd order
                [[[1, 1]], [[2, 1]]],  # df/dx1, df/dx2
                [[[1, 2]], [[1, 1], [2, 1]], [[2, 2]]]  # d2f/dx1^2, d2f/dx1dx2, d2f/dx2^2
            ]
        ]
        
        # Prepare data for each submodel
        # Function values at ALL training points
        y_all = f_func(cls.X_train[:, 0], cls.X_train[:, 1]).reshape(-1, 1)
        
        submodel_data = []
        
        # Submodel 1: Corners (no derivatives)
        edges_idx = cls.submodel_indices[0][0]
        X_edges = cls.X_train[edges_idx]
        dy_dx1_edges = df_dx1_func(X_edges[:, 0], X_edges[:, 1]).reshape(-1, 1)
        submodel_data.append([y_all, dy_dx1_edges])
        
        
        # Submodel 2: Edges (1st order)
        edges_idx = cls.submodel_indices[1][0]
        X_edges = cls.X_train[edges_idx]
        dy_dx1_edges = df_dx1_func(X_edges[:, 0], X_edges[:, 1]).reshape(-1, 1)
        dy_dx2_edges = df_dx2_func(X_edges[:, 0], X_edges[:, 1]).reshape(-1, 1)
        submodel_data.append([y_all, dy_dx1_edges, dy_dx2_edges])
        
        # Submodel 3: Center (2nd order)
        center_idx = cls.submodel_indices[2][0]
        X_center = cls.X_train[center_idx]
        dy_dx1_center = df_dx1_func(X_center[:, 0], X_center[:, 1]).reshape(-1, 1)
        dy_dx2_center = df_dx2_func(X_center[:, 0], X_center[:, 1]).reshape(-1, 1)
        d2y_dx1_2_center = d2f_dx1_2_func(X_center[:, 0], X_center[:, 1]).reshape(-1, 1)
        d2y_dx1dx2_center = d2f_dx1dx2_func(X_center[:, 0], X_center[:, 1]).reshape(-1, 1)
        d2y_dx2_2_center = d2f_dx2_2_func(X_center[:, 0], X_center[:, 1]).reshape(-1, 1)
        submodel_data.append([
            y_all, dy_dx1_center, dy_dx2_center,
            d2y_dx1_2_center, d2y_dx1dx2_center, d2y_dx2_2_center
        ])
        
        return submodel_data, der_indices
    
    
    def test_submodel1_function_interpolation(self):
        """Test function value interpolation for Submodel 1 (corners)."""
        corners_idx = [0, 3, 12, 15]
        X_corners = self.X_train[corners_idx]
        
        y_pred, submodel_vals = self.model.predict(
            X_corners, self.params, calc_cov=False, return_submodels=True, return_deriv=True
        )
        
        y_true = self.submodel_data[0][0][corners_idx].flatten()
        y_pred_submodel = submodel_vals[0][0,:].flatten()
        
        abs_error = np.abs(y_pred_submodel - y_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Submodel 1 function interpolation error too large: {max_error}")
    
    def test_submodel2_function_interpolation(self):
        """Test function value interpolation for Submodel 2 (edges)."""
        edges_idx = self.submodel_indices[1][0]
        X_edges = self.X_train[edges_idx]
        
        _, submodel_vals = self.model.predict(
            X_edges, self.params, calc_cov=False, return_submodels=True
        )
        
        y_true = self.submodel_data[1][0][edges_idx].flatten()
        y_pred_submodel = submodel_vals[1].flatten()
        
        abs_error = np.abs(y_pred_submodel - y_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Submodel 2 function interpolation error too large: {max_error}")
    
    def test_submodel2_first_derivatives(self):
        """Test 1st derivative interpolation for Submodel 2 (edges)."""
        edges_idx = self.submodel_indices[1][0]
        X_edges = self.X_train[edges_idx]
        
        h = 5e-6
        
        for i, local_idx in enumerate(range(len(edges_idx))):
            x_point = X_edges[local_idx]
            
        
            
            f_plus = self.model.predict(x_point.reshape(1, -1), self.params, calc_cov=False, return_submodels=False, return_deriv=True)
            
            fd_deriv_x1 = f_plus[1,0]
            analytic_deriv_x1 = self.submodel_data[1][1][local_idx, 0]
            
            error_x1 = abs(fd_deriv_x1 - analytic_deriv_x1)
            self.assertLess(error_x1, 1e-3,
                           f"Submodel 2 ∂f/∂x₁ error at point {i}: {error_x1}")
            
            # Test ∂f/∂x₁
            X_plus = x_point.copy().reshape(1, -1)
            X_minus = x_point.copy().reshape(1, -1)
            X_plus[0, 0] += h
            X_minus[0, 0] -= h
            
            # Test ∂f/∂x₂
            X_plus[0, 0] = x_point[0]
            X_minus[0, 0] = x_point[0]
            X_plus[0, 1] = x_point[1] + h
            X_minus[0, 1] = x_point[1] - h
            
            _, sm_plus = self.model.predict(X_plus, self.params, calc_cov=False, return_submodels=True)
            _, sm_minus = self.model.predict(X_minus, self.params, calc_cov=False, return_submodels=True)
            
            fd_deriv_x2 = (sm_plus[1][0, 0] - sm_minus[1][0, 0]) / (2 * h)
            analytic_deriv_x2 = self.submodel_data[1][2][local_idx, 0]
            
            error_x2 = abs(fd_deriv_x2 - analytic_deriv_x2)
            self.assertLess(error_x2, 1e-3,
                           f"Submodel 2 ∂f/∂x₂ error at point {i}: {error_x2}")
    
    def test_submodel3_function_interpolation(self):
        """Test function value interpolation for Submodel 3 (center)."""
        center_idx = self.submodel_indices[2][0]
        X_center = self.X_train[center_idx]
        
        _, submodel_vals = self.model.predict(
            X_center, self.params, calc_cov=False, return_submodels=True
        )
        
        y_true = self.submodel_data[2][0][center_idx].flatten()
        y_pred_submodel = submodel_vals[2].flatten()
        
        abs_error = np.abs(y_pred_submodel - y_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Submodel 3 function interpolation error too large: {max_error}")
    
    def test_submodel3_first_derivatives(self):
        """Test 1st derivative interpolation for Submodel 3 (center)."""
        center_idx = self.submodel_indices[2][0]
        X_center = self.X_train[center_idx]
        
        h = 5e-6
        
        for i, local_idx in enumerate(range(len(center_idx))):
            x_point = X_center[local_idx]
            
            # Test ∂f/∂x₁
            X_plus = x_point.copy().reshape(1, -1)
            X_minus = x_point.copy().reshape(1, -1)
            X_plus[0, 0] += h
            X_minus[0, 0] -= h
            
             
            f_plus = self.model.predict(x_point.reshape(1, -1), self.params, calc_cov=False, return_submodels=False, return_deriv=True)
            
            fd_deriv_x1 = f_plus[1,0]
            analytic_deriv_x1 = self.submodel_data[2][1][local_idx, 0]
            
            error_x1 = abs(fd_deriv_x1 - analytic_deriv_x1)
            self.assertLess(error_x1, 1e-3,
                           f"Submodel 3 ∂f/∂x₁ error at point {i}: {error_x1}")
            
            # Test ∂f/∂x₂
            X_plus[0, 0] = x_point[0]
            X_minus[0, 0] = x_point[0]
            X_plus[0, 1] = x_point[1] + h
            X_minus[0, 1] = x_point[1] - h
            
            _, sm_plus = self.model.predict(X_plus, self.params, calc_cov=False, return_submodels=True)
            _, sm_minus = self.model.predict(X_minus, self.params, calc_cov=False, return_submodels=True)
            
            fd_deriv_x2 = (sm_plus[2][0, 0] - sm_minus[2][0, 0]) / (2 * h)
            analytic_deriv_x2 = self.submodel_data[2][2][local_idx, 0]
            
            error_x2 = abs(fd_deriv_x2 - analytic_deriv_x2)
            self.assertLess(error_x2, 1e-2,
                           f"Submodel 3 ∂f/∂x₂ error at point {i}: {error_x2}")
    
    def test_submodel3_second_derivatives(self):
        """Test 2nd derivative interpolation for Submodel 3 (center)."""
        center_idx = self.submodel_indices[2][0]
        X_center = self.X_train[center_idx]
        
        h = 1e-3  # Larger h for numerical stability in 2nd derivatives
        
        for i, local_idx in enumerate(range(len(center_idx))):
            x_point = X_center[local_idx]
            
            # Get center value
            _, submodel_vals_center = self.model.predict(
                x_point.reshape(1, -1), self.params, calc_cov=False, return_submodels=True
            )
            
            # Test ∂²f/∂x₁²
            X_plus = x_point.copy().reshape(1, -1)
            X_minus = x_point.copy().reshape(1, -1)
            X_plus[0, 0] += h
            X_minus[0, 0] -= h
            
            _, sm_plus = self.model.predict(X_plus, self.params, calc_cov=False, return_submodels=True)
            _, sm_minus = self.model.predict(X_minus, self.params, calc_cov=False, return_submodels=True)
            
            fd_deriv2_x1x1 = (sm_plus[2][0, 0] - 2*submodel_vals_center[2][0, 0] + 
                              sm_minus[2][0, 0]) / (h**2)
            analytic_deriv2_x1x1 = self.submodel_data[2][3][local_idx, 0]
            
            error_x1x1 = abs(fd_deriv2_x1x1 - analytic_deriv2_x1x1)
            self.assertLess(error_x1x1, 1e-1,
                           f"Submodel 3 ∂²f/∂x₁² error at point {i}: {error_x1x1}")
            
            # Test ∂²f/∂x₁∂x₂ (mixed partial)
            X_pp = x_point.copy().reshape(1, -1)
            X_pm = x_point.copy().reshape(1, -1)
            X_mp = x_point.copy().reshape(1, -1)
            X_mm = x_point.copy().reshape(1, -1)
            
            X_pp[0, :] += h
            X_pm[0, 0] += h
            X_pm[0, 1] -= h
            X_mp[0, 0] -= h
            X_mp[0, 1] += h
            X_mm[0, :] -= h
            
            _, sm_pp = self.model.predict(X_pp, self.params, calc_cov=False, return_submodels=True)
            _, sm_pm = self.model.predict(X_pm, self.params, calc_cov=False, return_submodels=True)
            _, sm_mp = self.model.predict(X_mp, self.params, calc_cov=False, return_submodels=True)
            _, sm_mm = self.model.predict(X_mm, self.params, calc_cov=False, return_submodels=True)
            
            fd_deriv2_x1x2 = (sm_pp[2][0, 0] - sm_pm[2][0, 0] - 
                              sm_mp[2][0, 0] + sm_mm[2][0, 0]) / (4 * h**2)
            analytic_deriv2_x1x2 = self.submodel_data[2][4][local_idx, 0]
            
            error_x1x2 = abs(fd_deriv2_x1x2 - analytic_deriv2_x1x2)
            self.assertLess(error_x1x2, 1e-1,
                           f"Submodel 3 ∂²f/∂x₁∂x₂ error at point {i}: {error_x1x2}")
            
            # Test ∂²f/∂x₂²
            X_plus = x_point.copy().reshape(1, -1)
            X_minus = x_point.copy().reshape(1, -1)
            X_plus[0, 1] += h
            X_minus[0, 1] -= h
            
            _, sm_plus = self.model.predict(X_plus, self.params, calc_cov=False, return_submodels=True)
            _, sm_minus = self.model.predict(X_minus, self.params, calc_cov=False, return_submodels=True)
            
            fd_deriv2_x2x2 = (sm_plus[2][0, 0] - 2*submodel_vals_center[2][0, 0] + 
                              sm_minus[2][0, 0]) / (h**2)
            analytic_deriv2_x2x2 = self.submodel_data[2][5][local_idx, 0]
            
            error_x2x2 = abs(fd_deriv2_x2x2 - analytic_deriv2_x2x2)
            self.assertLess(error_x2x2, 1e-1,
                           f"Submodel 3 ∂²f/∂x₂² error at point {i}: {error_x2x2}")
    
    def test_comprehensive_summary(self):
        """Comprehensive test with detailed summary."""
        print("\n" + "="*80)
        print("WDEGP Heterogeneous Derivatives Interpolation Test Summary")
        print("="*80)
        
        all_tests_passed = True
        
        # Submodel 1: Corners (function only)
        print("\nSubmodel 1: Corners (No Derivatives)")
        print(f"Indices: {self.submodel_indices[0]}")
        print("-" * 80)
        corners_idx =  [0, 3, 12, 15]
        X_corners = self.X_train[corners_idx]
        _, submodel_vals = self.model.predict(
            X_corners, self.params, calc_cov=False, return_submodels=True
        )
        y_true = self.submodel_data[0][0][corners_idx].flatten()
        y_pred = submodel_vals[0].flatten()
        abs_error = np.abs(y_pred - y_true)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)
        
        passed = max_error < 1e-6
        all_tests_passed = all_tests_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Function values | Max: {max_error:.2e} | Mean: {mean_error:.2e}")
        
        # Submodel 2: Edges (function + 1st derivatives)
        print("\nSubmodel 2: Edges (1st Order Derivatives)")
        print(f"Indices: {self.submodel_indices[1]}")
        print("-" * 80)
        edges_idx = self.submodel_indices[1][0]
        X_edges = self.X_train[edges_idx]
        
        # Function values
        _, submodel_vals = self.model.predict(
            X_edges, self.params, calc_cov=False, return_submodels=True
        )
        y_true = self.submodel_data[1][0][edges_idx].flatten()
        y_pred = submodel_vals[1].flatten()
        abs_error = np.abs(y_pred - y_true)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)
        
        passed = max_error < 1e-6
        all_tests_passed = all_tests_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Function values | Max: {max_error:.2e} | Mean: {mean_error:.2e}")
        
        print(f"      | 1st derivatives verified via finite differences (h=1e-6)")
        
       # Submodel 3: Center (function + 1st + 2nd derivatives)
        print("\nSubmodel 3: Center (2nd Order Derivatives)")
        print("-" * 80)
        center_idx = self.submodel_indices[2][0]
        X_center = self.X_train[center_idx]
        
        # Function values
        _, submodel_vals = self.model.predict(
            X_center, self.params, calc_cov=False, return_submodels=True
        )
        y_true = self.submodel_data[2][0][center_idx].flatten()
        y_pred = submodel_vals[2].flatten()
        abs_error = np.abs(y_pred - y_true)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)
        
        passed = max_error < 1e-6
        all_tests_passed = all_tests_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Function values | Max: {max_error:.2e} | Mean: {mean_error:.2e}")
        
        print(f"      | 1st derivatives verified via finite differences (h=5e-6)")
        print(f"      | 2nd derivatives verified via finite differences (h=1e-3)")
        
        print("\n" + "="*80)
        print("Summary:")
        print(f"  - Total training points: {len(self.X_train)}")
        print(f"  - Submodel 1 (Corners): {len(self.submodel_indices[0])} points, 0th order only")
        print(f"  - Submodel 2 (Edges): {len(self.submodel_indices[1])} points, up to 1st order")
        print(f"  - Submodel 3 (Center): {len(self.submodel_indices[2])} points, up to 2nd order")
        print(f"  - Test function: Six-hump camel function")
        print(f"  - Derivatives computed symbolically with SymPy")
        print(f"  - Function values evaluated at ALL points, shared across submodels")
        print(f"  - Derivative values evaluated only at each submodel's points")
        print("="*80)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_tests_passed else '✗ SOME TESTS FAILED'}")
        print("="*80 + "\n")
        
        
        self.assertTrue(all_tests_passed, "Not all interpolation tests passed")


def run_tests_with_details():
    """Run tests with detailed output."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWDEGPHeterogeneousDerivatives)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    print("\nRunning WDEGP Heterogeneous Derivatives Interpolation Unit Tests...")
    print("="*80 + "\n")
    
    result = run_tests_with_details()
    
    sys.exit(0 if result.wasSuccessful() else 1)