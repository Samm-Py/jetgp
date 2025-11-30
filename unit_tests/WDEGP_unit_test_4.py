"""
Unit test for WDEGP with DDEGP submodels (directional derivatives).

This test verifies that the WDEGP model correctly handles:
- Two DDEGP submodels with DISJOINT derivative coverage
- Global directional derivatives at 45-degree angles
- Proper interpolation at training points

Test function: f(x,y) = sin(x)cos(y)
"""

import unittest
import numpy as np
from jetgp.wdegp.wdegp import wdegp
import sys


class TestWDEGPWithDDEGPSubmodels(unittest.TestCase):
    """Test case for WDEGP with 2 DDEGP submodels using 45-degree rays."""
    
    @classmethod
    def setUpClass(cls):
        """Set up training data and model once for all tests."""
        np.random.seed(42)
        
        # Generate 2D training data on a grid
        X1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        X2 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        X1_grid, X2_grid = np.meshgrid(X1, X2)
        cls.X_train = np.column_stack([X1_grid.flatten(), X2_grid.flatten()])
        cls.n_train = len(cls.X_train)
        
        # Define global rays at 45-degree angles
        sqrt2_inv = 1.0 / np.sqrt(2.0)
        cls.rays = np.array([
            [sqrt2_inv, -sqrt2_inv],   # x-components: ray1 = +45°, ray2 = +135°
            [sqrt2_inv,  sqrt2_inv]    # y-components
        ])
        
        # Split points into two DISJOINT submodels based on x-coordinate
        # Submodel 1: left half (x < 0) - strictly less than
        # Submodel 2: right half (x >= 0)
        cls.sm1_indices = [i for i in range(cls.n_train) if cls.X_train[i, 0] < 0]
        cls.sm2_indices = [i for i in range(cls.n_train) if cls.X_train[i, 0] >= 0]
        
        # Verify disjoint
        assert len(set(cls.sm1_indices) & set(cls.sm2_indices)) == 0, \
            "Submodel indices must be disjoint!"
        
        # Compute true function: f(x,y) = sin(x)cos(y)
        x = cls.X_train[:, 0]
        y = cls.X_train[:, 1]
        cls.y_func_all = np.sin(x) * np.cos(y)
        
        # Gradient: ∇f = [cos(x)cos(y), -sin(x)sin(y)]
        grad_x = np.cos(x) * np.cos(y)
        grad_y = -np.sin(x) * np.sin(y)
        
        # Directional derivatives: D_v f = ∇f · v
        cls.dy_ray1_all = grad_x * cls.rays[0, 0] + grad_y * cls.rays[1, 0]
        cls.dy_ray2_all = grad_x * cls.rays[0, 1] + grad_y * cls.rays[1, 1]
        
        # Closed form for verification
        cls.dy_ray1_closed = sqrt2_inv * np.cos(x + y)
        cls.dy_ray2_closed = -sqrt2_inv * np.cos(x - y)
        
        # Prepare training data for each submodel
        y_vals = cls.y_func_all.reshape(-1, 1)
        
        dy_ray1_sm1 = cls.dy_ray1_all[cls.sm1_indices].reshape(-1, 1)
        dy_ray2_sm1 = cls.dy_ray2_all[cls.sm1_indices].reshape(-1, 1)
        
        dy_ray1_sm2 = cls.dy_ray1_all[cls.sm2_indices].reshape(-1, 1)
        dy_ray2_sm2 = cls.dy_ray2_all[cls.sm2_indices].reshape(-1, 1)
        
        cls.y_train = [
            [y_vals, dy_ray1_sm1, dy_ray2_sm1],  # Submodel 1
            [y_vals, dy_ray1_sm2, dy_ray2_sm2]   # Submodel 2
        ]
        
        # Define derivative indices for each submodel
        cls.der_indices = [
            [[[[1, 1]], [[2, 1]]]],  # SM1: 2 directional derivatives, order 1
            [[[[1, 1]], [[2, 1]]]]   # SM2: 2 directional derivatives, order 1
        ]
        
        # Define derivative locations - MUST BE DISJOINT between submodels
        cls.derivative_locations = [
            [cls.sm1_indices, cls.sm1_indices],  # SM1: both rays at left points (x < 0)
            [cls.sm2_indices, cls.sm2_indices]   # SM2: both rays at right points (x >= 0)
        ]
        
        # Initialize WDEGP model with DDEGP submodels
        cls.model = wdegp(
            cls.X_train,
            cls.y_train,
            n_order=1,
            n_bases=2,
            der_indices=cls.der_indices,
            derivative_locations=cls.derivative_locations,
            submodel_type='ddegp',
            rays=cls.rays,
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
        
        # Get predictions at training points
        cls.y_train_pred, _ = cls.model.predict(
            cls.X_train,
            cls.params,
            calc_cov=True,
            return_deriv=True
        )
    
    def test_submodels_are_disjoint(self):
        """Test that submodel derivative locations are disjoint."""
        overlap = set(self.sm1_indices) & set(self.sm2_indices)
        self.assertEqual(len(overlap), 0,
                        f"Submodel indices must be disjoint, but found overlap: {overlap}")
        
        # Also verify they cover all points
        all_covered = set(self.sm1_indices) | set(self.sm2_indices)
        self.assertEqual(len(all_covered), self.n_train,
                        "Submodels should cover all training points")
    
    def test_rays_are_45_degrees(self):
        """Test that rays are correctly oriented at 45-degree angles."""
        sqrt2_inv = 1.0 / np.sqrt(2.0)
        
        np.testing.assert_array_almost_equal(
            self.rays[:, 0], [sqrt2_inv, sqrt2_inv],
            err_msg="Ray 1 should be at 45°"
        )
        np.testing.assert_array_almost_equal(
            self.rays[:, 1], [-sqrt2_inv, sqrt2_inv],
            err_msg="Ray 2 should be at 135°"
        )
        
        # Verify orthogonality
        dot_product = np.dot(self.rays[:, 0], self.rays[:, 1])
        self.assertAlmostEqual(dot_product, 0.0, places=10,
                              msg="Rays should be orthogonal")
    
    def test_training_data_structure(self):
        """Test that training data has correct structure."""
        self.assertEqual(self.X_train.shape, (25, 2),
                        "Training data should have 25 points in 2D")
        self.assertEqual(len(self.y_train), 2,
                        "Should have 2 submodels")
        
        # Check derivative array sizes match submodel point counts
        self.assertEqual(self.y_train[0][1].shape[0], len(self.sm1_indices),
                        f"SM1 derivatives should have {len(self.sm1_indices)} rows")
        self.assertEqual(self.y_train[1][1].shape[0], len(self.sm2_indices),
                        f"SM2 derivatives should have {len(self.sm2_indices)} rows")
    
    def test_function_interpolation_all_points(self):
        """Test that function values are correctly interpolated at all points."""
        y_func_pred = self.y_train_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Function interpolation error too large: {max_error}")
    
    def test_directional_derivative_ray1_sm1(self):
        """Test ray 1 (45°) directional derivative interpolation at SM1 points."""
        y_deriv_pred = self.y_train_pred[1, self.sm1_indices].flatten()
        y_deriv_true = self.dy_ray1_all[self.sm1_indices]
        abs_error = np.abs(y_deriv_pred - y_deriv_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Ray 1 (45°) derivative at SM1 points error: {max_error}")
    
    def test_directional_derivative_ray2_sm1(self):
        """Test ray 2 (135°) directional derivative interpolation at SM1 points."""
        y_deriv_pred = self.y_train_pred[2, self.sm1_indices].flatten()
        y_deriv_true = self.dy_ray2_all[self.sm1_indices]
        abs_error = np.abs(y_deriv_pred - y_deriv_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Ray 2 (135°) derivative at SM1 points error: {max_error}")
    
    def test_directional_derivative_ray1_sm2(self):
        """Test ray 1 (45°) directional derivative interpolation at SM2 points."""
        y_deriv_pred = self.y_train_pred[1, self.sm2_indices].flatten()
        y_deriv_true = self.dy_ray1_all[self.sm2_indices]
        abs_error = np.abs(y_deriv_pred - y_deriv_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Ray 1 (45°) derivative at SM2 points error: {max_error}")
    
    def test_directional_derivative_ray2_sm2(self):
        """Test ray 2 (135°) directional derivative interpolation at SM2 points."""
        y_deriv_pred = self.y_train_pred[2, self.sm2_indices].flatten()
        y_deriv_true = self.dy_ray2_all[self.sm2_indices]
        abs_error = np.abs(y_deriv_pred - y_deriv_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Ray 2 (135°) derivative at SM2 points error: {max_error}")
    
    def test_all_interpolations_summary(self):
        """Test all interpolations and provide a summary."""
        y_func_pred = self.y_train_pred[0, :].flatten()
        y_ray1_pred_sm1 = self.y_train_pred[1, self.sm1_indices].flatten()
        y_ray2_pred_sm1 = self.y_train_pred[2, self.sm1_indices].flatten()
        y_ray1_pred_sm2 = self.y_train_pred[1, self.sm2_indices].flatten()
        y_ray2_pred_sm2 = self.y_train_pred[2, self.sm2_indices].flatten()
        
        errors = {
            'Function (all 25 pts)': np.abs(y_func_pred - self.y_func_all),
            'D_ray1 (45°) SM1': np.abs(y_ray1_pred_sm1 - self.dy_ray1_all[self.sm1_indices]),
            'D_ray2 (135°) SM1': np.abs(y_ray2_pred_sm1 - self.dy_ray2_all[self.sm1_indices]),
            'D_ray1 (45°) SM2': np.abs(y_ray1_pred_sm2 - self.dy_ray1_all[self.sm2_indices]),
            'D_ray2 (135°) SM2': np.abs(y_ray2_pred_sm2 - self.dy_ray2_all[self.sm2_indices])
        }
        
        print("\n" + "=" * 70)
        print("WDEGP with DDEGP Submodels Test Summary (45° Rays, Disjoint)")
        print("=" * 70)
        print("Configuration:")
        print(f"  - Total points: {self.n_train}")
        print(f"  - Ray 1: [{self.rays[0, 0]:.4f}, {self.rays[1, 0]:.4f}] (45°)")
        print(f"  - Ray 2: [{self.rays[0, 1]:.4f}, {self.rays[1, 1]:.4f}] (135°)")
        print(f"  - Submodel 1 (x < 0): {len(self.sm1_indices)} points")
        print(f"  - Submodel 2 (x >= 0): {len(self.sm2_indices)} points")
        print(f"  - Overlap: 0 points (disjoint)")
        print("-" * 70)
        
        all_passed = True
        for name, error in errors.items():
            max_err = np.max(error)
            mean_err = np.mean(error)
            passed = max_err < 1e-6
            all_passed = all_passed and passed
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} | {name:25s} | Max: {max_err:.2e} | Mean: {mean_err:.2e}")
        
        print("=" * 70)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print("=" * 70 + "\n")
        
        self.assertTrue(all_passed, "Not all interpolation tests passed")


class TestWDEGPWithDDEGPMixedOrders(unittest.TestCase):
    """Test case for WDEGP with DDEGP submodels having different derivative orders."""
    
    @classmethod
    def setUpClass(cls):
        """Set up training data with mixed derivative orders across submodels."""
        np.random.seed(123)
        
        # Generate training grid
        X1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        X2 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        X1_grid, X2_grid = np.meshgrid(X1, X2)
        cls.X_train = np.column_stack([X1_grid.flatten(), X2_grid.flatten()])
        cls.n_train = len(cls.X_train)
        
        # Global rays at 45-degree angles
        sqrt2_inv = 1.0 / np.sqrt(2.0)
        cls.rays = np.array([
            [sqrt2_inv, -sqrt2_inv],   # x-components
            [sqrt2_inv,  sqrt2_inv]    # y-components
        ])
        
        # DISJOINT submodels:
        # Submodel 1: outer points (|x| > 0.5 OR |y| > 0.5), first-order only
        # Submodel 2: center points (|x| <= 0.5 AND |y| <= 0.5), first and second order
        cls.sm1_indices = [i for i in range(cls.n_train) 
                          if np.abs(cls.X_train[i, 0]) > 0.5 or np.abs(cls.X_train[i, 1]) > 0.5]
        cls.sm2_indices = [i for i in range(cls.n_train) 
                          if np.abs(cls.X_train[i, 0]) <= 0.5 and np.abs(cls.X_train[i, 1]) <= 0.5]
        
        # Verify disjoint
        assert len(set(cls.sm1_indices) & set(cls.sm2_indices)) == 0, \
            "Submodel indices must be disjoint!"
        
        # Compute true values: f(x,y) = sin(x)cos(y)
        x = cls.X_train[:, 0]
        y = cls.X_train[:, 1]
        cls.y_func_all = np.sin(x) * np.cos(y)
        
        # First-order directional derivatives
        cls.dy_ray1_all = sqrt2_inv * np.cos(x + y)
        cls.dy_ray2_all = -sqrt2_inv * np.cos(x - y)
        
        # Second-order directional derivatives
        cls.d2y_ray1_all = -np.sin(x + y)
        cls.d2y_ray2_all = -np.sin(x - y)
        
        # Prepare y_train
        y_vals = cls.y_func_all.reshape(-1, 1)
        
        # SM1: function + first-order derivatives (outer points)
        dy_ray1_sm1 = cls.dy_ray1_all[cls.sm1_indices].reshape(-1, 1)
        dy_ray2_sm1 = cls.dy_ray2_all[cls.sm1_indices].reshape(-1, 1)
        
        # SM2: function + first-order + second-order derivatives (center points)
        dy_ray1_sm2 = cls.dy_ray1_all[cls.sm2_indices].reshape(-1, 1)
        dy_ray2_sm2 = cls.dy_ray2_all[cls.sm2_indices].reshape(-1, 1)
        d2y_ray1_sm2 = cls.d2y_ray1_all[cls.sm2_indices].reshape(-1, 1)
        d2y_ray2_sm2 = cls.d2y_ray2_all[cls.sm2_indices].reshape(-1, 1)
        
        cls.y_train = [
            [y_vals, dy_ray1_sm1, dy_ray2_sm1],
            [y_vals, dy_ray1_sm2, dy_ray2_sm2, d2y_ray1_sm2, d2y_ray2_sm2]
        ]
        
        # Derivative indices
        cls.der_indices = [
            [[[[1, 1]], [[2, 1]]]],  # SM1: first-order only
            [[[[1, 1]], [[2, 1]]], [[[1, 2]], [[2, 2]]]]  # SM2: first and second order
        ]
        
        # Derivative locations - DISJOINT between submodels
        cls.derivative_locations = [
            [cls.sm1_indices, cls.sm1_indices],  # SM1: outer points
            [cls.sm2_indices, cls.sm2_indices, cls.sm2_indices, cls.sm2_indices]  # SM2: center points
        ]
        
        # Initialize model
        cls.model = wdegp(
            cls.X_train,
            cls.y_train,
            n_order=2,
            n_bases=2,
            der_indices=cls.der_indices,
            derivative_locations=cls.derivative_locations,
            submodel_type='ddegp',
            rays=cls.rays,
            normalize=True,
            kernel="SineExp",
            kernel_type="isotropic"
        )
        
        # Optimize
        cls.params = cls.model.optimize_hyperparameters(
            optimizer='pso',
            pop_size=100,
            n_generations=15,
            local_opt_every=15,
            debug=False
        )
        
        # Predict
        cls.y_train_pred, _ = cls.model.predict(
            cls.X_train,
            cls.params,
            calc_cov=True,
            return_deriv=True
        )
    
    def test_submodels_are_disjoint(self):
        """Test that submodel derivative locations are disjoint."""
        overlap = set(self.sm1_indices) & set(self.sm2_indices)
        self.assertEqual(len(overlap), 0,
                        f"Submodel indices must be disjoint, but found overlap: {overlap}")
    
    def test_submodel_coverage(self):
        """Test that submodels cover expected regions."""
        # SM1: outer points
        for idx in self.sm1_indices:
            x, y = self.X_train[idx]
            self.assertTrue(np.abs(x) > 0.5 or np.abs(y) > 0.5,
                           f"SM1 point {idx} should be outer (|x|>0.5 or |y|>0.5)")
        
        # SM2: center points
        for idx in self.sm2_indices:
            x, y = self.X_train[idx]
            self.assertTrue(np.abs(x) <= 0.5 and np.abs(y) <= 0.5,
                           f"SM2 point {idx} should be center (|x|<=0.5 and |y|<=0.5)")
    
    def test_function_interpolation(self):
        """Test function interpolation with mixed-order submodels."""
        y_func_pred = self.y_train_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Function interpolation error: {max_error}")
    
    def test_first_order_derivatives_sm1(self):
        """Test first-order derivative interpolation at SM1 (outer) points."""
        y_ray1_pred = self.y_train_pred[1, self.sm1_indices].flatten()
        y_ray1_true = self.dy_ray1_all[self.sm1_indices]
        max_error = np.max(np.abs(y_ray1_pred - y_ray1_true))
        
        self.assertLess(max_error, 1e-6,
                       f"First-order ray1 (45°) at SM1 error: {max_error}")
        
        y_ray2_pred = self.y_train_pred[2, self.sm1_indices].flatten()
        y_ray2_true = self.dy_ray2_all[self.sm1_indices]
        max_error = np.max(np.abs(y_ray2_pred - y_ray2_true))
        
        self.assertLess(max_error, 1e-6,
                       f"First-order ray2 (135°) at SM1 error: {max_error}")
    
    def test_first_order_derivatives_sm2(self):
        """Test first-order derivative interpolation at SM2 (center) points."""
        y_ray1_pred = self.y_train_pred[1, self.sm2_indices].flatten()
        y_ray1_true = self.dy_ray1_all[self.sm2_indices]
        max_error = np.max(np.abs(y_ray1_pred - y_ray1_true))
        
        self.assertLess(max_error, 1e-6,
                       f"First-order ray1 (45°) at SM2 error: {max_error}")
    
    def test_all_interpolations_summary(self):
        """Summary of mixed-order interpolation tests with 45° rays."""
        sqrt2_inv = 1.0 / np.sqrt(2.0)
        
        print("\n" + "=" * 70)
        print("WDEGP Mixed Order Submodels Test Summary (45° Rays, Disjoint)")
        print("=" * 70)
        print("Configuration:")
        print(f"  - Ray 1: [{sqrt2_inv:.4f}, {sqrt2_inv:.4f}] (45°)")
        print(f"  - Ray 2: [{-sqrt2_inv:.4f}, {sqrt2_inv:.4f}] (135°)")
        print(f"  - SM1 (outer, 1st order): {len(self.sm1_indices)} points")
        print(f"  - SM2 (center, 1st+2nd order): {len(self.sm2_indices)} points")
        print(f"  - Overlap: 0 points (disjoint)")
        print("-" * 70)
        
        errors = {
            'Function': np.abs(self.y_train_pred[0, :].flatten() - self.y_func_all),
            'D_ray1 (SM1, outer)': np.abs(self.y_train_pred[1, self.sm1_indices].flatten() - self.dy_ray1_all[self.sm1_indices]),
            'D_ray2 (SM1, outer)': np.abs(self.y_train_pred[2, self.sm1_indices].flatten() - self.dy_ray2_all[self.sm1_indices]),
            'D_ray1 (SM2, center)': np.abs(self.y_train_pred[1, self.sm2_indices].flatten() - self.dy_ray1_all[self.sm2_indices]),
            'D_ray2 (SM2, center)': np.abs(self.y_train_pred[2, self.sm2_indices].flatten() - self.dy_ray2_all[self.sm2_indices]),
        }
        
        all_passed = True
        for name, error in errors.items():
            max_err = np.max(error)
            mean_err = np.mean(error)
            passed = max_err < 1e-6
            all_passed = all_passed and passed
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} | {name:25s} | Max: {max_err:.2e} | Mean: {mean_err:.2e}")
        
        print("=" * 70)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print("=" * 70 + "\n")
        
        self.assertTrue(all_passed, "Not all interpolation tests passed")


def run_tests_with_details():
    """Run all test cases with detailed output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestWDEGPWithDDEGPSubmodels))
    suite.addTests(loader.loadTestsFromTestCase(TestWDEGPWithDDEGPMixedOrders))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    print("\nRunning WDEGP with DDEGP Submodels Unit Tests (45° Rays, Disjoint)...")
    print("=" * 70 + "\n")
    
    result = run_tests_with_details()
    sys.exit(0 if result.wasSuccessful() else 1)
