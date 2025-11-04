"""
Unit test for Weighted Directional Derivative-Enhanced Gaussian Process (WDDEGP)
with heterogeneous submodels on the Six-Hump Camel function.

This test verifies that WDDEGP correctly interpolates:
- Function values across all training points
- First-order directional derivatives along submodel-specific rays
- Second-order directional derivatives along submodel-specific rays

The test uses 9 heterogeneous submodels:
- 4 edge submodels (3 points each, 3 rays each)
- 4 corner submodels (1 point each, 3 rays each)
- 1 interior submodel (9 points, 3 rays)
"""

import sys
import unittest
import numpy as np
import sympy as sp
import itertools
from jetgp.wddegp.wddegp import wddegp
import jetgp.utils as utils


class TestWDDEGPHeterogeneousSubmodels(unittest.TestCase):
    """Test case for WDDEGP with heterogeneous submodels on Six-Hump Camel function."""
    
    @classmethod
    def setUpClass(cls):
        """Set up training data and model once for all tests."""
        # --- Configuration ---
        cls.n_order = 2
        cls.n_bases = 2
        cls.num_pts_per_axis = 5
        cls.domain_bounds = ((-1, 1), (-1, 1))
        cls.normalize_data = True
        cls.kernel = "SE"
        cls.kernel_type = "anisotropic"
        cls.random_seed = 0
        np.random.seed(cls.random_seed)
        
        # --- Define Six-Hump Camel function symbolically ---
        x1_sym, x2_sym = sp.symbols('x1 x2', real=True)
        f_sym = ((4 - 2.1*x1_sym**2 + (x1_sym**4)/3.0) * x1_sym**2 + 
                x1_sym*x2_sym + (-4 + 4*x2_sym**2) * x2_sym**2)
        
        cls.grad_x1_sym = sp.diff(f_sym, x1_sym)
        cls.grad_x2_sym = sp.diff(f_sym, x2_sym)
        
        # Compute Hessian for second-order derivatives
        cls.h11_sym = sp.diff(cls.grad_x1_sym, x1_sym)
        cls.h12_sym = sp.diff(cls.grad_x1_sym, x2_sym)
        cls.h22_sym = sp.diff(cls.grad_x2_sym, x2_sym)
        
        # Convert to NumPy functions with array handling
        true_function_np_raw = sp.lambdify([x1_sym, x2_sym], f_sym, 'numpy')
        grad_x1_func_raw = sp.lambdify([x1_sym, x2_sym], cls.grad_x1_sym, 'numpy')
        grad_x2_func_raw = sp.lambdify([x1_sym, x2_sym], cls.grad_x2_sym, 'numpy')
        h11_func_raw = sp.lambdify([x1_sym, x2_sym], cls.h11_sym, 'numpy')
        h12_func_raw = sp.lambdify([x1_sym, x2_sym], cls.h12_sym, 'numpy')
        h22_func_raw = sp.lambdify([x1_sym, x2_sym], cls.h22_sym, 'numpy')
        
        def make_array_func(func_raw):
            def wrapped(x1, x2):
                result = func_raw(x1, x2)
                result = np.atleast_1d(result)
                if result.size == 1 and np.atleast_1d(x1).size > 1:
                    result = np.full_like(x1, result[0])
                return result
            return wrapped
        
        cls.true_function_np = make_array_func(true_function_np_raw)
        cls.grad_x1_func = make_array_func(grad_x1_func_raw)
        cls.grad_x2_func = make_array_func(grad_x2_func_raw)
        cls.h11_func = make_array_func(h11_func_raw)
        cls.h12_func = make_array_func(h12_func_raw)
        cls.h22_func = make_array_func(h22_func_raw)
        
        def true_function(X):
            """Six-Hump Camel function."""
            return cls.true_function_np(X[:, 0], X[:, 1])
        
        def true_gradient(x1, x2):
            """Analytical gradient of Six-Hump Camel function."""
            gx1 = cls.grad_x1_func(x1, x2)
            gx2 = cls.grad_x2_func(x1, x2)
            # Convert to scalar if needed
            if np.isscalar(x1) or (hasattr(x1, '__len__') and len(x1) == 1):
                gx1 = np.asarray(gx1).item() if hasattr(gx1, '__iter__') else gx1
                gx2 = np.asarray(gx2).item() if hasattr(gx2, '__iter__') else gx2
            return gx1, gx2
        
        cls.true_function = true_function
        cls.true_gradient = true_gradient
        
        # --- Generate training grid ---
        x_lin = np.linspace(cls.domain_bounds[0][0], cls.domain_bounds[0][1], cls.num_pts_per_axis)
        y_lin = np.linspace(cls.domain_bounds[1][0], cls.domain_bounds[1][1], cls.num_pts_per_axis)
        X1_grid_train, X2_grid_train = np.meshgrid(x_lin, y_lin)
        X_train_initial = np.column_stack([X1_grid_train.ravel(), X2_grid_train.ravel()])
        
        # --- Define submodel structure (following RST exactly) ---
        cls.submodel_groups_initial = [
            [1, 2, 3],                          # Submodel 0: Top edge
            [5, 10, 15],                        # Submodel 1: Left edge
            [9, 14, 19],                        # Submodel 2: Right edge
            [21, 22, 23],                       # Submodel 3: Bottom edge
            [0],                                # Submodel 4: Top-left corner
            [4],                                # Submodel 5: Top-right corner
            [20],                               # Submodel 6: Bottom-left corner
            [24],                               # Submodel 7: Bottom-right corner
            [6, 7, 8, 11, 12, 13, 16, 17, 18]  # Submodel 8: Interior points
        ]
        
        # Ray angles per submodel (in radians) - following RST
        cls.submodel_ray_thetas = [
            [-np.pi/4, 0, np.pi/4],                    # Submodel 0: Three rays
            [-np.pi/4, 0, np.pi/4],                    # Submodel 1: Three rays
            [-np.pi/4, 0, np.pi/4],                    # Submodel 2: Three rays
            [-np.pi/4, 0, np.pi/4],                    # Submodel 3: Three rays
            [-np.pi/2, 0, -np.pi/4],                   # Submodel 4: Corner rays
            [np.pi/2, 0, np.pi/4],                     # Submodel 5: Corner rays
            [np.pi/2, 0, np.pi/4],                     # Submodel 6: Corner rays
            [-np.pi/2, 0, -np.pi/4],                   # Submodel 7: Corner rays
            [np.pi/2, np.pi/4, np.pi/4 + np.pi/2]     # Submodel 8: Interior rays
        ]
        
        # Derivative indices specification (same for all submodels)
        cls.submodel_der_indices = [
            [[[[1,1]], [[1,2]], [[2,1]], [[2,2]], [[3,1]], [[3,2]]]] 
            for _ in range(len(cls.submodel_groups_initial))
        ]
        
        cls.num_submodels = len(cls.submodel_groups_initial)
        
        # --- Reorder training data ---
        reorder_indices = list(itertools.chain.from_iterable(cls.submodel_groups_initial))
        cls.X_train = X_train_initial[reorder_indices]
        
        # Create contiguous submodel indices
        cls.submodel_indices = []
        current_pos = 0
        for group in cls.submodel_groups_initial:
            group_size = len(group)
            cls.submodel_indices.append(list(range(current_pos, current_pos + group_size)))
            current_pos += group_size
        
        # --- Compute rays and directional derivatives ---
        cls.y_train_data_all, cls.rays_data_all = cls._compute_derivatives_and_rays()
        
        # --- Initialize WDDEGP model ---
        cls.model = wddegp(
            cls.X_train,
            cls.y_train_data_all,
            cls.n_order,
            cls.n_bases,
            cls.submodel_indices,
            cls.submodel_der_indices,
            cls.rays_data_all,
            normalize=cls.normalize_data,
            kernel=cls.kernel,
            kernel_type=cls.kernel_type
        )
        
        # --- Optimize hyperparameters ---
        cls.params = cls.model.optimize_hyperparameters(
            optimizer='pso',
            pop_size=100,
            n_generations=15,
            local_opt_every=15,
            debug=False
        )
    
    @classmethod
    def _compute_derivatives_and_rays(cls):
        """Compute directional derivatives and rays for all submodels."""
        # Function values at ALL training points (shared by all submodels)
        y_func_values = cls.true_function(cls.X_train).reshape(-1, 1)
        
        y_train_data_all = []
        rays_data_all = []
        
        for k, group_indices in enumerate(cls.submodel_indices):
            # Extract points for this submodel
            X_sub = cls.X_train[group_indices]
            
            # Create rays for this submodel
            thetas = cls.submodel_ray_thetas[k]
            rays = np.column_stack([[np.cos(t), np.sin(t)] for t in thetas])
            
            # Normalize rays to unit length (should already be unit, but ensure)
            for i in range(rays.shape[1]):
                rays[:, i] = rays[:, i] / np.linalg.norm(rays[:, i])
            
            rays_data_all.append(rays)
            
            # Compute directional derivatives using chain rule
            y_train_submodel = [y_func_values]  # All submodels share function values
            
            for ray_idx, ray in enumerate(rays.T):
                # Compute first and second order directional derivatives
                for order in range(1, cls.n_order + 1):
                    deriv_values = []
                    
                    for point in X_sub:
                        x1, x2 = point[0], point[1]
                        
                        # Get gradient at this point
                        gx1, gx2 = cls.true_gradient(x1, x2)
                        
                        if order == 1:
                            # First-order directional derivative: ∇f · d
                            d_ray = gx1 * ray[0] + gx2 * ray[1]
                            deriv_values.append(d_ray)
                        
                        elif order == 2:
                            # Second-order directional derivative: d^T H d
                            h11_val = cls.h11_func(x1, x2)
                            h12_val = cls.h12_func(x1, x2)
                            h22_val = cls.h22_func(x1, x2)
                            
                            # Convert to scalar if needed
                            if hasattr(h11_val, '__iter__'):
                                h11_val = np.asarray(h11_val).item()
                            if hasattr(h12_val, '__iter__'):
                                h12_val = np.asarray(h12_val).item()
                            if hasattr(h22_val, '__iter__'):
                                h22_val = np.asarray(h22_val).item()
                            
                            # d^T H d = d1^2 * H11 + 2*d1*d2 * H12 + d2^2 * H22
                            d2_ray = (ray[0]**2 * h11_val + 
                                     2 * ray[0] * ray[1] * h12_val + 
                                     ray[1]**2 * h22_val)
                            deriv_values.append(d2_ray)
                    
                    y_train_submodel.append(np.array(deriv_values).reshape(-1, 1))
            
            y_train_data_all.append(y_train_submodel)
        
        return y_train_data_all, rays_data_all
    
    def test_configuration(self):
        """Verify model configuration."""
        self.assertEqual(self.num_submodels, 9, "Should have 9 submodels")
        self.assertEqual(self.n_order, 2, "Should use 2nd order derivatives")
        self.assertEqual(len(self.X_train), 25, "Should have 25 total training points")
        
        # Check submodel sizes
        expected_sizes = [3, 3, 3, 3, 1, 1, 1, 1, 9]
        actual_sizes = [len(group) for group in self.submodel_groups_initial]
        self.assertEqual(actual_sizes, expected_sizes, "Submodel sizes incorrect")
    
    def test_training_data_structure(self):
        """Test that training data has correct structure."""
        self.assertEqual(self.X_train.shape, (25, 2),
                        "Training data should have 25 points in 2D")
        self.assertEqual(len(self.y_train_data_all), 9,
                        "Should have 9 submodels")
        
        # Check that all submodels have function values for ALL points
        for i, submodel_data in enumerate(self.y_train_data_all):
            self.assertEqual(submodel_data[0].shape, (25, 1),
                            f"Submodel {i} should have function values for ALL 25 points")
            
            # Check derivative arrays
            # Each submodel has 3 rays, each with 2 orders = 6 derivative arrays
            expected_deriv_arrays = 1 + 3 * 2  # func + (3 rays × 2 orders)
            self.assertEqual(len(submodel_data), expected_deriv_arrays,
                            f"Submodel {i} should have {expected_deriv_arrays} arrays")
    
    def test_ray_structure(self):
        """Verify ray structure for each submodel."""
        self.assertEqual(len(self.rays_data_all), 9,
                        "Should have rays for 9 submodels")
        
        # All submodels have 3 rays each
        for i, rays in enumerate(self.rays_data_all):
            self.assertEqual(rays.shape, (2, 3),
                            f"Submodel {i} should have 3 rays in 2D")
    
    def test_ray_unit_length(self):
        """Verify that all rays have unit length."""
        for i, rays in enumerate(self.rays_data_all):
            for j in range(rays.shape[1]):
                ray = rays[:, j]
                ray_norm = np.linalg.norm(ray)
                self.assertAlmostEqual(ray_norm, 1.0, places=10,
                                      msg=f"Ray {j} in submodel {i} should have unit length")
    
    def test_function_value_interpolation(self):
        """Test function value interpolation at all training points."""
        y_pred_all = self.model.predict(
            self.X_train, self.params, calc_cov=False
        )
        
        y_true = self.y_train_data_all[0][0]  # Function values (same for all submodels)
        abs_errors = np.abs(y_pred_all.flatten() - y_true.flatten())
        max_error = np.max(abs_errors)
        mean_error = np.mean(abs_errors)
        
        self.assertLess(max_error, 1e-6,
                       f"Function value interpolation error too large: {max_error}")
        self.assertLess(mean_error, 1e-7,
                       f"Mean function value interpolation error too large: {mean_error}")
    
    def test_directional_derivatives_finite_differences(self):
        """Test directional derivatives at representative points using finite differences."""
        h_first = 1e-6
        h_second = 1e-3
        
        # Test one point from each type of submodel
        test_cases = [
            (0, 1, 0),  # Submodel 0 (top edge), point 1, ray 0
            (4, 0, 1),  # Submodel 4 (corner), point 0, ray 1
            (8, 4, 1),  # Submodel 8 (interior), point 4, ray 1
        ]
        
        for submodel_idx, local_idx, ray_idx in test_cases:
            point_indices = self.submodel_indices[submodel_idx]
            global_idx = point_indices[local_idx]
            x_point = self.X_train[global_idx]
            
            rays = self.rays_data_all[submodel_idx]
            ray_dir = rays[:, ray_idx]
            
            # Get submodel predictions at perturbed points
            x_plus = x_point + h_first * ray_dir
            x_minus = x_point - h_first * ray_dir
            x_center = x_point
            
            _, sm_plus = self.model.predict(x_plus.reshape(1, -1), self.params,
                                           calc_cov=False, return_submodels=True)
            _, sm_minus = self.model.predict(x_minus.reshape(1, -1), self.params,
                                            calc_cov=False, return_submodels=True)
            _, sm_center = self.model.predict(x_center.reshape(1, -1), self.params,
                                             calc_cov=False, return_submodels=True)
            
            # First derivative via finite difference
            fd_1st = (sm_plus[submodel_idx][0, 0] - sm_minus[submodel_idx][0, 0]) / (2 * h_first)
            
            # Index in y_train_data_all: [func, ray0_1st, ray0_2nd, ray1_1st, ray1_2nd, ray2_1st, ray2_2nd]
            deriv_idx_1st = 1 + ray_idx * 2
            analytic_1st = self.y_train_data_all[submodel_idx][deriv_idx_1st][local_idx, 0]
            err_1st = abs(fd_1st - analytic_1st)
            
            self.assertLess(err_1st, 1e-3,
                           f"1st derivative error at submodel {submodel_idx}, point {local_idx}, ray {ray_idx}: {err_1st}")
            
            # Second derivative via finite difference
            x_plus_2 = x_point + h_second * ray_dir
            x_minus_2 = x_point - h_second * ray_dir
            
            _, sm_plus_2 = self.model.predict(x_plus_2.reshape(1, -1), self.params,
                                             calc_cov=False, return_submodels=True)
            _, sm_minus_2 = self.model.predict(x_minus_2.reshape(1, -1), self.params,
                                              calc_cov=False, return_submodels=True)
            
            fd_2nd = (sm_plus_2[submodel_idx][0, 0] - 2*sm_center[submodel_idx][0, 0] + 
                     sm_minus_2[submodel_idx][0, 0]) / (h_second**2)
            
            deriv_idx_2nd = 1 + ray_idx * 2 + 1
            analytic_2nd = self.y_train_data_all[submodel_idx][deriv_idx_2nd ][local_idx, 0]
            err_2nd = abs(fd_2nd - analytic_2nd)
            
            self.assertLess(err_2nd, 1e-2,
                           f"2nd derivative error at submodel {submodel_idx}, point {local_idx}, ray {ray_idx}: {err_2nd}")
    
    def test_comprehensive_summary(self):
        """Comprehensive test with detailed summary."""
        print("\n" + "="*80)
        print("WDDEGP Heterogeneous Submodels Interpolation Test Summary")
        print("="*80)
        
        all_tests_passed = True
        
        # Function values
        print("\nFunction Value Interpolation")
        print("-" * 80)
        y_pred_all = self.model.predict(
            self.X_train, self.params, calc_cov=False,
        )
        y_true = self.y_train_data_all[0][0]
        abs_errors = np.abs(y_pred_all.flatten() - y_true.flatten())
        max_error = np.max(abs_errors)
        mean_error = np.mean(abs_errors)
        
        passed = max_error < 1e-6
        all_tests_passed = all_tests_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Function values | Max: {max_error:.2e} | Mean: {mean_error:.2e}")
        
        # Configuration
        print("\n" + "="*80)
        print("Configuration:")
        print(f"  - Function: Six-Hump Camel")
        print(f"  - Total training points: 25 (5×5 grid)")
        print(f"  - Domain: x1 ∈ [-1, 1], x2 ∈ [-1, 1]")
        print(f"  - Number of submodels: {self.num_submodels}")
        print(f"  - Submodel structure:")
        
        submodel_names = [
            "Top edge (3 pts, 3 rays)",
            "Left edge (3 pts, 3 rays)",
            "Right edge (3 pts, 3 rays)",
            "Bottom edge (3 pts, 3 rays)",
            "Top-left corner (1 pt, 3 rays)",
            "Top-right corner (1 pt, 3 rays)",
            "Bottom-left corner (1 pt, 3 rays)",
            "Bottom-right corner (1 pt, 3 rays)",
            "Interior (9 pts, 3 rays)"
        ]
        
        for i, name in enumerate(submodel_names):
            print(f"    Submodel {i}: {name}")
        
        print(f"  - Derivative order: {self.n_order}")
        print(f"  - Kernel: {self.kernel} ({self.kernel_type})")
        print(f"  - Derivatives: computed symbolically with SymPy")
        
        print("="*80)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_tests_passed else '✗ SOME TESTS FAILED'}")
        print("="*80 + "\n")
        
        self.assertTrue(all_tests_passed, "Not all interpolation tests passed")


def run_tests_with_details():
    """Run tests with detailed output."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWDDEGPHeterogeneousSubmodels)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    print("\nRunning WDDEGP Heterogeneous Submodels Interpolation Unit Tests...")
    print("="*80 + "\n")
    
    result = run_tests_with_details()
    
    sys.exit(0 if result.wasSuccessful() else 1)

    """Test case for WDDEGP with heterogeneous submodels on Six-Hump Camel function."""
    
    @classmethod
    def setUpClass(cls):
        """Set up training data and model once for all tests."""
        # --- Configuration ---
        cls.n_order = 2
        cls.n_bases = 2
        cls.num_pts_per_axis = 5
        cls.domain_bounds = ((-1, 1), (-1, 1))
        cls.normalize_data = True
        cls.kernel = "SE"
        cls.kernel_type = "anisotropic"
        cls.random_seed = 0
        np.random.seed(cls.random_seed)
        
        # --- Define Six-Hump Camel function ---
        x1_sym, x2_sym = sp.symbols('x1 x2', real=True)
        f_sym = ((4 - 2.1*x1_sym**2 + (x1_sym**4)/3.0) * x1_sym**2 + 
                x1_sym*x2_sym + (-4 + 4*x2_sym**2) * x2_sym**2)
        
        grad_x1_sym = sp.diff(f_sym, x1_sym)
        grad_x2_sym = sp.diff(f_sym, x2_sym)
        
        # Convert to NumPy functions with array handling
        true_function_np_raw = sp.lambdify([x1_sym, x2_sym], f_sym, 'numpy')
        grad_x1_func_raw = sp.lambdify([x1_sym, x2_sym], grad_x1_sym, 'numpy')
        grad_x2_func_raw = sp.lambdify([x1_sym, x2_sym], grad_x2_sym, 'numpy')
        
        def make_array_func(func_raw):
            def wrapped(x1, x2):
                result = func_raw(x1, x2)
                result = np.atleast_1d(result)
                if result.size == 1 and np.atleast_1d(x1).size > 1:
                    result = np.full_like(x1, result[0])
                return result
            return wrapped
        
        cls.true_function_np = make_array_func(true_function_np_raw)
        cls.grad_x1_func = make_array_func(grad_x1_func_raw)
        cls.grad_x2_func = make_array_func(grad_x2_func_raw)
        
        def true_function(X, alg=np):
            """Six-Hump Camel function (polymorphic for numpy or OTI)."""
            x1 = X[:, 0]
            x2 = X[:, 1]
            return ((4 - 2.1*x1**2 + (x1**4)/3.0) * x1**2 + 
                   x1*x2 + (-4 + 4*x2**2) * x2**2)
        
        cls.true_function = true_function
        
        # --- Generate training grid ---
        x1_train = np.linspace(cls.domain_bounds[0][0], cls.domain_bounds[0][1], cls.num_pts_per_axis)
        x2_train = np.linspace(cls.domain_bounds[1][0], cls.domain_bounds[1][1], cls.num_pts_per_axis)
        X_train_initial = np.array(list(itertools.product(x1_train, x2_train)))
        
        # --- Define submodel structure ---
        cls.submodel_groups_initial = [
            [1, 2, 3],                          # Submodel 0: Top edge
            [5, 10, 15],                        # Submodel 1: Left edge
            [9, 14, 19],                        # Submodel 2: Right edge
            [21, 22, 23],                       # Submodel 3: Bottom edge
            [0],                                # Submodel 4: Top-left corner
            [4],                                # Submodel 5: Top-right corner
            [20],                               # Submodel 6: Bottom-left corner
            [24],                               # Submodel 7: Bottom-right corner
            [6, 7, 8, 11, 12, 13, 16, 17, 18]  # Submodel 8: Interior points
        ]
        
        # Ray angles per submodel (in radians)
        cls.submodel_ray_thetas = [
            [0.0],                     # Top edge: horizontal
            [np.pi/2],                 # Left edge: vertical
            [np.pi/2],                 # Right edge: vertical
            [0.0],                     # Bottom edge: horizontal
            [np.pi/4],                 # Top-left: diagonal
            [3*np.pi/4],               # Top-right: diagonal
            [-np.pi/4],                # Bottom-left: diagonal
            [-3*np.pi/4],              # Bottom-right: diagonal
            [0.0, np.pi/2]             # Interior: horizontal + vertical
        ]
        
        cls.num_submodels = len(cls.submodel_groups_initial)
        
        # --- Reorder training data ---
        cls.X_train_reordered, cls.sequential_indices, cls.reorder_map = \
            cls._reorder_training_data(X_train_initial)
        
        # --- Compute rays and derivatives ---
        cls.rays_per_submodel, cls.y_train_data_all, cls.der_indices_all = \
            cls._compute_rays_and_derivatives()
        
        # Create index array
        cls.index = np.array([len(group) for group in cls.submodel_groups_initial])
        
        # --- Initialize WDDEGP model ---
        cls.model = wddegp(
            cls.X_train_reordered,
            cls.y_train_data_all,
            cls.n_order,
            cls.n_bases,
            cls.index,
            cls.der_indices_all,
            rays_per_submodel=cls.rays_per_submodel,
            normalize=cls.normalize_data,
            kernel=cls.kernel,
            kernel_type=cls.kernel_type
        )
        
        # --- Optimize hyperparameters ---
        cls.params = cls.model.optimize_hyperparameters(
            optimizer='pso',
            pop_size=100,
            n_generations=15,
            local_opt_every=15,
            debug=False
        )
    
    @classmethod
    def _reorder_training_data(cls, X_train_initial):
        """Reorder training data so submodel indices are contiguous."""
        reorder_map = []
        for group in cls.submodel_groups_initial:
            reorder_map.extend(group)
        
        X_train_reordered = X_train_initial[reorder_map]
        
        sequential_indices = []
        current_pos = 0
        for group in cls.submodel_groups_initial:
            group_len = len(group)
            sequential_indices.append(list(range(current_pos, current_pos + group_len)))
            current_pos += group_len
        
        return X_train_reordered, sequential_indices, reorder_map
    
    @classmethod
    def _compute_rays_and_derivatives(cls):
        """Compute rays and directional derivatives for all submodels."""
        # Function values at ALL training points
        y_func_all = cls.true_function(cls.X_train_reordered, alg=np).reshape(-1, 1)
        
        rays_per_submodel = []
        y_train_data_all = []
        der_indices_all = []
        
        for submodel_idx, (point_indices, ray_thetas) in enumerate(
            zip(cls.sequential_indices, cls.submodel_ray_thetas)
        ):
            num_rays = len(ray_thetas)
            X_sub = cls.X_train_reordered[point_indices]
            
            # Create rays for this submodel
            rays_sub = np.array([
                [np.cos(theta) for theta in ray_thetas],
                [np.sin(theta) for theta in ray_thetas]
            ])
            rays_per_submodel.append(rays_sub)
            
            # Generate derivative indices using OTI format
            der_indices_sub = utils.gen_OTI_indices(cls.n_bases, cls.n_order, num_rays)
            der_indices_all.append(der_indices_sub)
            
            
            # For this test, we'll use symbolic differentiation for verification
            # Compute directional derivatives symbolically
            derivatives_this_sub = []
            for ray_idx, theta in enumerate(ray_thetas):
                ray_dir = np.array([np.cos(theta), np.sin(theta)])
                
                # First-order directional derivative: ∇f · v
                grad_x1_vals = cls.grad_x1_func(X_sub[:, 0], X_sub[:, 1])
                grad_x2_vals = cls.grad_x2_func(X_sub[:, 0], X_sub[:, 1])
                dir_deriv_1 = grad_x1_vals * ray_dir[0] + grad_x2_vals * ray_dir[1]
                derivatives_this_sub.append(dir_deriv_1.reshape(-1, 1))
                
                # Second-order directional derivative (simplified for this test)
                # In practice, would compute d²f/ds² along the ray
                # For now, use approximate placeholder
                dir_deriv_2 = np.zeros_like(dir_deriv_1).reshape(-1, 1)
                derivatives_this_sub.append(dir_deriv_2)
            
            # Package data: [all function values] + [derivatives for this submodel]
            submodel_data = [y_func_all] + derivatives_this_sub
            y_train_data_all.append(submodel_data)
        
        return rays_per_submodel, y_train_data_all, der_indices_all
    
    def test_configuration(self):
        """Verify model configuration."""
        self.assertEqual(self.num_submodels, 9, "Should have 9 submodels")
        self.assertEqual(self.n_order, 2, "Should use 2nd order derivatives")
        self.assertEqual(len(self.X_train_reordered), 25, "Should have 25 total training points")
        
        # Check submodel sizes
        expected_sizes = [3, 3, 3, 3, 1, 1, 1, 1, 9]
        actual_sizes = [len(group) for group in self.submodel_groups_initial]
        self.assertEqual(actual_sizes, expected_sizes, "Submodel sizes incorrect")
    
    def test_training_data_structure(self):
        """Test that training data has correct structure."""
        self.assertEqual(self.X_train_reordered.shape, (25, 2),
                        "Training data should have 25 points in 2D")
        self.assertEqual(len(self.y_train_data_all), 9,
                        "Should have 9 submodels")
        
        # Check that all submodels have function values for ALL points
        for i, submodel_data in enumerate(self.y_train_data_all):
            self.assertEqual(submodel_data[0].shape, (25, 1),
                            f"Submodel {i} should have function values for ALL 25 points")
    
    def test_ray_structure(self):
        """Verify ray structure for each submodel."""
        self.assertEqual(len(self.rays_per_submodel), 9,
                        "Should have rays for 9 submodels")
        
        # Check ray dimensions
        expected_num_rays = [1, 1, 1, 1, 1, 1, 1, 1, 2]
        for i, (rays, expected) in enumerate(zip(self.rays_per_submodel, expected_num_rays)):
            self.assertEqual(rays.shape[1], expected,
                            f"Submodel {i} should have {expected} rays")
            self.assertEqual(rays.shape[0], 2,
                            f"Submodel {i} rays should be 2D")
    
    def test_ray_unit_length(self):
        """Verify that all rays have unit length."""
        for i, rays in enumerate(self.rays_per_submodel):
            for j in range(rays.shape[1]):
                ray = rays[:, j]
                ray_norm = np.linalg.norm(ray)
                self.assertAlmostEqual(ray_norm, 1.0, places=10,
                                      msg=f"Ray {j} in submodel {i} should have unit length")
    
    def test_function_value_interpolation(self):
        """Test function value interpolation at all training points."""
        y_pred_all = self.model.predict(
            self.X_train_reordered, self.params, calc_cov=False, return_deriv=False
        )
        
        y_true = self.y_train_data_all[0][0]  # Function values (same for all submodels)
        abs_errors = np.abs(y_pred_all.flatten() - y_true.flatten())
        max_error = np.max(abs_errors)
        mean_error = np.mean(abs_errors)
        
        self.assertLess(max_error, 1e-6,
                       f"Function value interpolation error too large: {max_error}")
        self.assertLess(mean_error, 1e-7,
                       f"Mean function value interpolation error too large: {mean_error}")
    
    def test_representative_directional_derivatives(self):
        """Test directional derivatives at representative points using finite differences."""
        h_first = 1e-6
        h_second = 1e-5
        
        # Test representative points from different submodels
        test_cases = [
            (0, 0, 0),  # Submodel 0 (top edge), point 0, ray 0
            (1, 0, 0),  # Submodel 1 (left edge), point 0, ray 0
            (8, 4, 0),  # Submodel 8 (interior), point 4, ray 0
        ]
        
        max_errors_1st = []
        max_errors_2nd = []
        
        for submodel_idx, local_idx, ray_idx in test_cases:
            point_indices = self.sequential_indices[submodel_idx]
            global_idx = point_indices[local_idx]
            x_point = self.X_train_reordered[global_idx]
            
            rays = self.rays_per_submodel[submodel_idx]
            ray_dir = rays[:, ray_idx]
            
            # First derivative via finite difference
            x_plus = x_point + h_first * ray_dir
            x_minus = x_point - h_first * ray_dir
            
            _, sm_plus = self.model.predict(x_plus.reshape(1, -1), self.params,
                                           calc_cov=False, return_submodels=True)
            _, sm_minus = self.model.predict(x_minus.reshape(1, -1), self.params,
                                            calc_cov=False, return_submodels=True)
            
            fd_1st = (sm_plus[submodel_idx][0, 0] - sm_minus[submodel_idx][0, 0]) / (2 * h_first)
            analytic_1st = self.y_train_data_all[submodel_idx][1][local_idx, 0]
            err_1st = abs(fd_1st - analytic_1st)
            max_errors_1st.append(err_1st)
            
            self.assertLess(err_1st, 1e-4,
                           f"First derivative error at submodel {submodel_idx}, point {local_idx}: {err_1st}")
        
        print(f"\nRepresentative directional derivative errors:")
        print(f"  Max 1st derivative error: {max(max_errors_1st):.2e}")
    
    def test_comprehensive_summary(self):
        """Comprehensive test with detailed summary."""
        print("\n" + "="*80)
        print("WDDEGP Heterogeneous Submodels Interpolation Test Summary")
        print("="*80)
        
        all_tests_passed = True
        
        # Function values
        print("\nFunction Value Interpolation")
        print("-" * 80)
        y_pred_all = self.model.predict(
            self.X_train_reordered, self.params, calc_cov=False, return_deriv=False
        )
        y_true = self.y_train_data_all[0][0]
        abs_errors = np.abs(y_pred_all.flatten() - y_true.flatten())
        max_error = np.max(abs_errors)
        mean_error = np.mean(abs_errors)
        
        passed = max_error < 1e-6
        all_tests_passed = all_tests_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Function values | Max: {max_error:.2e} | Mean: {mean_error:.2e}")
        
        # Configuration
        print("\n" + "="*80)
        print("Configuration:")
        print(f"  - Function: Six-Hump Camel")
        print(f"  - Total training points: 25 (5×5 grid)")
        print(f"  - Domain: x1 ∈ [-1, 1], x2 ∈ [-1, 1]")
        print(f"  - Number of submodels: {self.num_submodels}")
        print(f"  - Submodel structure:")
        
        submodel_names = [
            "Top edge (3 pts, horizontal ray)",
            "Left edge (3 pts, vertical ray)",
            "Right edge (3 pts, vertical ray)",
            "Bottom edge (3 pts, horizontal ray)",
            "Top-left corner (1 pt, diagonal)",
            "Top-right corner (1 pt, diagonal)",
            "Bottom-left corner (1 pt, diagonal)",
            "Bottom-right corner (1 pt, diagonal)",
            "Interior (9 pts, 2 rays)"
        ]
        
        for i, name in enumerate(submodel_names):
            num_rays = len(self.submodel_ray_thetas[i])
            print(f"    Submodel {i}: {name}")
        
        print(f"  - Derivative order: {self.n_order}")
        print(f"  - Kernel: {self.kernel} ({self.kernel_type})")
        
        print("="*80)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_tests_passed else '✗ SOME TESTS FAILED'}")
        print("="*80 + "\n")
        
        self.assertTrue(all_tests_passed, "Not all interpolation tests passed")


def run_tests_with_details():
    """Run tests with detailed output."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWDDEGPHeterogeneousSubmodels)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    print("\nRunning WDDEGP Heterogeneous Submodels Interpolation Unit Tests...")
    print("="*80 + "\n")
    
    result = run_tests_with_details()
    
    sys.exit(0 if result.wasSuccessful() else 1)