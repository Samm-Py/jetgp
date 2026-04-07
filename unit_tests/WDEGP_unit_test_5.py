"""
Unit test for WDEGP with GDDEGP submodels (point-wise directional derivatives).

This test verifies that the WDEGP model correctly handles:
- Two GDDEGP submodels with DISJOINT derivative coverage
- Point-wise directional derivatives (each point has unique ray directions)
- Proper interpolation at training points

Test function: f(x,y) = sin(x)cos(y)
"""

import unittest
import numpy as np
from jetgp.wdegp.wdegp import wdegp
import sys
import warnings
from unittest.mock import patch
from scipy.linalg import LinAlgError
import io


class TestWDEGPWithGDDEGPSubmodels(unittest.TestCase):
    """Test case for WDEGP with 2 GDDEGP submodels using point-wise rays."""
    
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
        
        # Split into DISJOINT submodels based on distance from origin
        distances = np.linalg.norm(cls.X_train, axis=1)
        median_dist = np.median(distances)
        
        cls.sm1_indices = [i for i in range(cls.n_train) if distances[i] < median_dist]
        cls.sm2_indices = [i for i in range(cls.n_train) if distances[i] >= median_dist]
        
        # Verify disjoint
        assert len(set(cls.sm1_indices) & set(cls.sm2_indices)) == 0, \
            "Submodel indices must be disjoint!"
        
        cls.n_sm1 = len(cls.sm1_indices)
        cls.n_sm2 = len(cls.sm2_indices)
        
        # Compute true function: f(x,y) = sin(x)cos(y)
        x = cls.X_train[:, 0]
        y = cls.X_train[:, 1]
        cls.y_func_all = np.sin(x) * np.cos(y)
        
        # Gradient: ∇f = [cos(x)cos(y), -sin(x)sin(y)]
        cls.grad_x_all = np.cos(x) * np.cos(y)
        cls.grad_y_all = -np.sin(x) * np.sin(y)
        
        # Build point-wise rays for ALL points (needed for prediction)
        # Also store per-submodel rays for training
        cls.rays_dir1_all = np.zeros((2, cls.n_train))
        cls.rays_dir2_all = np.zeros((2, cls.n_train))
        cls.dy_dir1_all = np.zeros(cls.n_train)
        cls.dy_dir2_all = np.zeros(cls.n_train)
        
        for idx in range(cls.n_train):
            grad = np.array([cls.grad_x_all[idx], cls.grad_y_all[idx]])
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < 1e-10:
                cls.rays_dir1_all[:, idx] = [1.0, 0.0]
                cls.rays_dir2_all[:, idx] = [0.0, 1.0]
            else:
                cls.rays_dir1_all[:, idx] = grad / grad_norm
                cls.rays_dir2_all[:, idx] = [-cls.rays_dir1_all[1, idx], cls.rays_dir1_all[0, idx]]
            
            cls.dy_dir1_all[idx] = np.dot(grad, cls.rays_dir1_all[:, idx])
            cls.dy_dir2_all[idx] = np.dot(grad, cls.rays_dir2_all[:, idx])
        
        # Extract submodel-specific rays and derivatives for training
        cls.rays_dir1_sm1 = cls.rays_dir1_all[:, cls.sm1_indices]
        cls.rays_dir2_sm1 = cls.rays_dir2_all[:, cls.sm1_indices]
        cls.dy_dir1_sm1 = cls.dy_dir1_all[cls.sm1_indices].reshape(-1, 1)
        cls.dy_dir2_sm1 = cls.dy_dir2_all[cls.sm1_indices].reshape(-1, 1)
        
        cls.rays_dir1_sm2 = cls.rays_dir1_all[:, cls.sm2_indices]
        cls.rays_dir2_sm2 = cls.rays_dir2_all[:, cls.sm2_indices]
        cls.dy_dir1_sm2 = cls.dy_dir1_all[cls.sm2_indices].reshape(-1, 1)
        cls.dy_dir2_sm2 = cls.dy_dir2_all[cls.sm2_indices].reshape(-1, 1)
        
        # rays_list structure for training: [sm1_rays, sm2_rays]
        cls.rays_list = [
            [cls.rays_dir1_sm1, cls.rays_dir2_sm1],  # SM1 rays
            [cls.rays_dir1_sm2, cls.rays_dir2_sm2]   # SM2 rays
        ]
        
        # rays_predict for prediction at all training points
        cls.rays_predict = [cls.rays_dir1_all, cls.rays_dir2_all]
        
        # Prepare y_train
        y_vals = cls.y_func_all.reshape(-1, 1)
        
        cls.y_train = [
            [y_vals, cls.dy_dir1_sm1, cls.dy_dir2_sm1],  # Submodel 1
            [y_vals, cls.dy_dir1_sm2, cls.dy_dir2_sm2]   # Submodel 2
        ]
        
        # Define derivative indices
        cls.der_indices = [
            [[[[1, 1]], [[2, 1]]]],  # SM1: 2 directional derivatives, order 1
            [[[[1, 1]], [[2, 1]]]]   # SM2: 2 directional derivatives, order 1
        ]
        
        # Define derivative locations - DISJOINT
        cls.derivative_locations = [
            [cls.sm1_indices, cls.sm1_indices],  # SM1: inner points
            [cls.sm2_indices, cls.sm2_indices]   # SM2: outer points
        ]
        
        # Initialize WDEGP model with GDDEGP submodels
        cls.model = wdegp(
            cls.X_train,
            cls.y_train,
            n_order=1,
            n_bases=2,
            der_indices=cls.der_indices,
            derivative_locations=cls.derivative_locations,
            submodel_type='gddegp',
            rays_list=cls.rays_list,
            normalize=True,
            kernel="SE",
            kernel_type="isotropic"
        )
        
        # Optimize hyperparameters
        cls.params = cls.model.optimize_hyperparameters(
            optimizer='pso',
            pop_size=100,
            n_generations=15,
            local_opt_every=15,
            debug=True
        )
        
        # Get predictions at training points with rays_predict
        cls.y_train_pred, _ = cls.model.predict(
            cls.X_train,
            cls.params,
            rays_predict=cls.rays_predict,
            calc_cov=True,
            return_deriv=True,
            derivs_to_predict=[[[1, 1]], [[2, 1]]]
        )

    def test_submodels_are_disjoint(self):
        """Test that submodel derivative locations are disjoint."""
        overlap = set(self.sm1_indices) & set(self.sm2_indices)
        self.assertEqual(len(overlap), 0,
                        f"Submodel indices must be disjoint, but found overlap: {overlap}")
        
        # Verify they cover all points
        all_covered = set(self.sm1_indices) | set(self.sm2_indices)
        self.assertEqual(len(all_covered), self.n_train,
                        "Submodels should cover all training points")
    
    def test_rays_predict_structure(self):
        """Test that rays_predict has correct structure."""
        self.assertEqual(len(self.rays_predict), 2,
                        "rays_predict should have 2 directions")
        self.assertEqual(self.rays_predict[0].shape, (2, self.n_train),
                        f"rays_predict[0] should have shape (2, {self.n_train})")
        self.assertEqual(self.rays_predict[1].shape, (2, self.n_train),
                        f"rays_predict[1] should have shape (2, {self.n_train})")
    
    def test_rays_are_point_wise(self):
        """Test that rays vary per point (not global)."""
        # Check rays have correct shape
        self.assertEqual(self.rays_dir1_sm1.shape, (2, self.n_sm1),
                        f"SM1 ray1 should have shape (2, {self.n_sm1})")
        self.assertEqual(self.rays_dir2_sm2.shape, (2, self.n_sm2),
                        f"SM2 ray2 should have shape (2, {self.n_sm2})")
        
        # Verify rays are unit vectors at each point
        for idx in range(self.n_train):
            norm1 = np.linalg.norm(self.rays_dir1_all[:, idx])
            norm2 = np.linalg.norm(self.rays_dir2_all[:, idx])
            self.assertAlmostEqual(norm1, 1.0, places=10,
                                  msg=f"Ray1 at point {idx} should be unit length")
            self.assertAlmostEqual(norm2, 1.0, places=10,
                                  msg=f"Ray2 at point {idx} should be unit length")
        
        # Verify rays are orthogonal at each point
        for idx in range(self.n_train):
            dot = np.dot(self.rays_dir1_all[:, idx], self.rays_dir2_all[:, idx])
            self.assertAlmostEqual(dot, 0.0, places=10,
                                  msg=f"Rays at point {idx} should be orthogonal")
    
    def test_rays_list_structure(self):
        """Test that rays_list has correct structure for GDDEGP."""
        self.assertEqual(len(self.rays_list), 2,
                        "rays_list should have 2 submodel entries")
        self.assertEqual(len(self.rays_list[0]), 2,
                        "SM1 should have 2 ray directions")
        self.assertEqual(len(self.rays_list[1]), 2,
                        "SM2 should have 2 ray directions")
    
    def test_perpendicular_derivative_is_zero(self):
        """Test that derivative perpendicular to gradient is ~zero."""
        for idx in range(self.n_train):
            self.assertAlmostEqual(self.dy_dir2_all[idx], 0.0, places=10,
                                  msg=f"Perpendicular derivative at point {idx} should be ~0")
    
    def test_gradient_derivative_is_gradient_magnitude(self):
        """Test that derivative along gradient direction equals |∇f|."""
        for idx in range(self.n_train):
            grad = np.array([self.grad_x_all[idx], self.grad_y_all[idx]])
            grad_norm = np.linalg.norm(grad)
            self.assertAlmostEqual(self.dy_dir1_all[idx], grad_norm, places=10,
                                  msg=f"Gradient derivative at point {idx} should equal |∇f|")
    
    def test_function_interpolation_all_points(self):
        """Test that function values are correctly interpolated at all points."""
        y_func_pred = self.y_train_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Function interpolation error too large: {max_error}")
    
    def test_directional_derivative_dir1_sm1(self):
        """Test gradient-direction derivative interpolation at SM1 points."""
        y_deriv_pred = self.y_train_pred[1, self.sm1_indices].flatten()
        y_deriv_true = self.dy_dir1_all[self.sm1_indices]
        abs_error = np.abs(y_deriv_pred - y_deriv_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Gradient-direction derivative at SM1 error: {max_error}")
    
    def test_directional_derivative_dir2_sm1(self):
        """Test perpendicular-direction derivative interpolation at SM1 points."""
        y_deriv_pred = self.y_train_pred[2, self.sm1_indices].flatten()
        y_deriv_true = self.dy_dir2_all[self.sm1_indices]
        abs_error = np.abs(y_deriv_pred - y_deriv_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Perpendicular-direction derivative at SM1 error: {max_error}")
    
    def test_directional_derivative_dir1_sm2(self):
        """Test gradient-direction derivative interpolation at SM2 points."""
        y_deriv_pred = self.y_train_pred[1, self.sm2_indices].flatten()
        y_deriv_true = self.dy_dir1_all[self.sm2_indices]
        abs_error = np.abs(y_deriv_pred - y_deriv_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Gradient-direction derivative at SM2 error: {max_error}")
    
    def test_directional_derivative_dir2_sm2(self):
        """Test perpendicular-direction derivative interpolation at SM2 points."""
        y_deriv_pred = self.y_train_pred[2, self.sm2_indices].flatten()
        y_deriv_true = self.dy_dir2_all[self.sm2_indices]
        abs_error = np.abs(y_deriv_pred - y_deriv_true)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Perpendicular-direction derivative at SM2 error: {max_error}")

    # =========================================================================
    # Prediction mode tests
    # =========================================================================

    def test_predict_no_deriv_with_cov(self):
        """Test prediction with return_deriv=False, calc_cov=True."""
        y_pred, cov = self.model.predict(
            self.X_train,
            self.params,
            calc_cov=True,
            return_deriv=False
        )
        
        # Should only return function values (no derivatives)
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row (function only), got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.n_train,
                        f"Expected {self.n_train} prediction points, got {y_pred.shape[1]}")
        
        # Covariance should be returned
        self.assertIsNotNone(cov, "Covariance should be returned when calc_cov=True")
        
        # Function values should still be accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Function interpolation error (no deriv): {max_error}")

    def test_predict_no_deriv_no_cov(self):
        """Test prediction with return_deriv=False, calc_cov=False."""
        y_pred = self.model.predict(
            self.X_train,
            self.params,
            calc_cov=False,
            return_deriv=False
        )
        
        # Should only return function values (no derivatives)
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row (function only), got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.n_train,
                        f"Expected {self.n_train} prediction points, got {y_pred.shape[1]}")
        
        # Function values should still be accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Function interpolation error (no deriv, no cov): {max_error}")

    def test_predict_with_deriv_no_cov(self):
        """Test prediction with return_deriv=True, calc_cov=False."""
        y_pred = self.model.predict(
            self.X_train,
            self.params,
            rays_predict=self.rays_predict,
            calc_cov=False,
            return_deriv=True,
            derivs_to_predict=[[[1, 1]], [[2, 1]]]
        )

        # Should return function + 2 derivative outputs
        self.assertEqual(y_pred.shape[0], 3,
                        f"Expected 3 output rows (func + 2 derivs), got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.n_train,
                        f"Expected {self.n_train} prediction points, got {y_pred.shape[1]}")
        
        # Function values should be accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Function interpolation error (with deriv, no cov): {max_error}")

    # =========================================================================
    # Warning tests for rays_predict
    # =========================================================================

    def test_predict_no_deriv_with_rays_warns(self):
        """Test that providing rays_predict with return_deriv=False issues a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            y_pred = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=self.rays_predict,  # Rays provided but not needed
                calc_cov=False,
                return_deriv=False  # Not returning derivatives
            )
            
            # Check that a warning was issued
            self.assertGreater(len(w), 0, 
                              "Expected a warning when rays_predict provided with return_deriv=False")
        
        # Predictions should still work (rays ignored)
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row (function only), got {y_pred.shape[0]}")

    # =========================================================================
    # Cholesky fallback tests
    # =========================================================================

    def test_cholesky_fallback_predict_with_cov(self):
        """Test that prediction works when Cholesky decomposition fails (with covariance)."""
        with patch('jetgp.wdegp.wdegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
            y_pred, cov = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=self.rays_predict,
                calc_cov=True,
                return_deriv=True,
                derivs_to_predict=[[[1, 1]], [[2, 1]]]
            )
        
        # Predictions should still work via fallback
        self.assertEqual(y_pred.shape[0], 3,
                        f"Expected 3 output rows, got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.n_train,
                        f"Expected {self.n_train} prediction points, got {y_pred.shape[1]}")
        
        # Covariance should still be computed
        self.assertIsNotNone(cov, "Covariance should be returned even with Cholesky fallback")
        
        # Function values should still be reasonably accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-1,
                       f"Function interpolation error with Cholesky fallback: {max_error}")

    def test_cholesky_fallback_predict_no_cov(self):
        """Test that prediction works when Cholesky decomposition fails (without covariance)."""
        with patch('jetgp.wdegp.wdegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
            y_pred = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=self.rays_predict,
                calc_cov=False,
                return_deriv=True,
                derivs_to_predict=[[[1, 1]], [[2, 1]]]
            )
        
        # Predictions should still work via fallback
        self.assertEqual(y_pred.shape[0], 3,
                        f"Expected 3 output rows, got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.n_train,
                        f"Expected {self.n_train} prediction points, got {y_pred.shape[1]}")

    def test_cholesky_vs_fallback_consistency(self):
        """Test that Cholesky and fallback methods produce consistent results."""
        # Get predictions with normal Cholesky
        y_pred_chol, cov_chol = self.model.predict(
            self.X_train,
            self.params,
            rays_predict=self.rays_predict,
            calc_cov=True,
            return_deriv=True,
            derivs_to_predict=[[[1, 1]], [[2, 1]]]
        )

        # Get predictions with fallback (mocked Cholesky failure)
        with patch('jetgp.wdegp.wdegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
            y_pred_fallback, cov_fallback = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=self.rays_predict,
                calc_cov=True,
                return_deriv=True,
                derivs_to_predict=[[[1, 1]], [[2, 1]]]
            )
        
        # Results should be very similar (within numerical tolerance)
        np.testing.assert_array_almost_equal(
            y_pred_chol, y_pred_fallback, decimal=6,
            err_msg="Cholesky and fallback predictions should match"
        )

    def test_all_interpolations_summary(self):
        """Test all interpolations and provide a summary."""
        y_func_pred = self.y_train_pred[0, :].flatten()
        y_dir1_pred_sm1 = self.y_train_pred[1, self.sm1_indices].flatten()
        y_dir2_pred_sm1 = self.y_train_pred[2, self.sm1_indices].flatten()
        y_dir1_pred_sm2 = self.y_train_pred[1, self.sm2_indices].flatten()
        y_dir2_pred_sm2 = self.y_train_pred[2, self.sm2_indices].flatten()
        
        errors = {
            'Function (all pts)': np.abs(y_func_pred - self.y_func_all),
            'D_grad (SM1, inner)': np.abs(y_dir1_pred_sm1 - self.dy_dir1_all[self.sm1_indices]),
            'D_perp (SM1, inner)': np.abs(y_dir2_pred_sm1 - self.dy_dir2_all[self.sm1_indices]),
            'D_grad (SM2, outer)': np.abs(y_dir1_pred_sm2 - self.dy_dir1_all[self.sm2_indices]),
            'D_perp (SM2, outer)': np.abs(y_dir2_pred_sm2 - self.dy_dir2_all[self.sm2_indices])
        }
        
        print("\n" + "=" * 70)
        print("WDEGP with GDDEGP Submodels Test Summary (Point-Wise Rays)")
        print("=" * 70)
        print("Configuration:")
        print(f"  - Total points: {self.n_train}")
        print(f"  - Submodel 1 (inner, r < median): {self.n_sm1} points")
        print(f"  - Submodel 2 (outer, r >= median): {self.n_sm2} points")
        print(f"  - Ray directions: gradient + perpendicular (unique per point)")
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


class TestWDEGPWithGDDEGPMixedOrders(unittest.TestCase):
    """Test case for WDEGP with GDDEGP submodels having different derivative orders."""
    
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
        
        # DISJOINT submodels: outer (1st order) vs center (1st + 2nd order)
        cls.sm1_indices = [i for i in range(cls.n_train) 
                          if np.abs(cls.X_train[i, 0]) > 0.5 or np.abs(cls.X_train[i, 1]) > 0.5]
        cls.sm2_indices = [i for i in range(cls.n_train) 
                          if np.abs(cls.X_train[i, 0]) <= 0.5 and np.abs(cls.X_train[i, 1]) <= 0.5]
        
        assert len(set(cls.sm1_indices) & set(cls.sm2_indices)) == 0, \
            "Submodel indices must be disjoint!"
        
        cls.n_sm1 = len(cls.sm1_indices)
        cls.n_sm2 = len(cls.sm2_indices)
        
        # Compute true function: f(x,y) = sin(x)cos(y)
        x = cls.X_train[:, 0]
        y = cls.X_train[:, 1]
        cls.y_func_all = np.sin(x) * np.cos(y)
        
        # Gradient
        cls.grad_x_all = np.cos(x) * np.cos(y)
        cls.grad_y_all = -np.sin(x) * np.sin(y)
        
        # Hessian components for second derivatives
        cls.H_xx = -np.sin(x) * np.cos(y)
        cls.H_xy = -np.cos(x) * np.sin(y)
        cls.H_yy = -np.sin(x) * np.cos(y)
        
        # Build point-wise rays for ALL points
        cls.rays_dir1_all = np.zeros((2, cls.n_train))
        cls.rays_dir2_all = np.zeros((2, cls.n_train))
        cls.dy_dir1_all = np.zeros(cls.n_train)
        cls.dy_dir2_all = np.zeros(cls.n_train)
        cls.d2y_dir1_all = np.zeros(cls.n_train)
        cls.d2y_dir2_all = np.zeros(cls.n_train)
        
        for idx in range(cls.n_train):
            grad = np.array([cls.grad_x_all[idx], cls.grad_y_all[idx]])
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < 1e-10:
                cls.rays_dir1_all[:, idx] = [1.0, 0.0]
                cls.rays_dir2_all[:, idx] = [0.0, 1.0]
            else:
                cls.rays_dir1_all[:, idx] = grad / grad_norm
                cls.rays_dir2_all[:, idx] = [-cls.rays_dir1_all[1, idx], cls.rays_dir1_all[0, idx]]
            
            # First-order directional derivatives
            cls.dy_dir1_all[idx] = np.dot(grad, cls.rays_dir1_all[:, idx])
            cls.dy_dir2_all[idx] = np.dot(grad, cls.rays_dir2_all[:, idx])
            
            # Second-order directional derivatives: D²_v f = v^T H v
            v1 = cls.rays_dir1_all[:, idx]
            v2 = cls.rays_dir2_all[:, idx]
            H = np.array([[cls.H_xx[idx], cls.H_xy[idx]],
                         [cls.H_xy[idx], cls.H_yy[idx]]])
            
            cls.d2y_dir1_all[idx] = v1 @ H @ v1
            cls.d2y_dir2_all[idx] = v2 @ H @ v2
        
        # Extract submodel-specific rays and derivatives
        cls.rays_dir1_sm1 = cls.rays_dir1_all[:, cls.sm1_indices]
        cls.rays_dir2_sm1 = cls.rays_dir2_all[:, cls.sm1_indices]
        cls.dy_dir1_sm1 = cls.dy_dir1_all[cls.sm1_indices].reshape(-1, 1)
        cls.dy_dir2_sm1 = cls.dy_dir2_all[cls.sm1_indices].reshape(-1, 1)
        
        cls.rays_dir1_sm2 = cls.rays_dir1_all[:, cls.sm2_indices]
        cls.rays_dir2_sm2 = cls.rays_dir2_all[:, cls.sm2_indices]
        cls.dy_dir1_sm2 = cls.dy_dir1_all[cls.sm2_indices].reshape(-1, 1)
        cls.dy_dir2_sm2 = cls.dy_dir2_all[cls.sm2_indices].reshape(-1, 1)
        cls.d2y_dir1_sm2 = cls.d2y_dir1_all[cls.sm2_indices].reshape(-1, 1)
        cls.d2y_dir2_sm2 = cls.d2y_dir2_all[cls.sm2_indices].reshape(-1, 1)
        
        # rays_list structure for training
        cls.rays_list = [
            [cls.rays_dir1_sm1, cls.rays_dir2_sm1],  # SM1 rays
            [cls.rays_dir1_sm2, cls.rays_dir2_sm2]   # SM2 rays
        ]
        
        # rays_predict for prediction at all training points
        cls.rays_predict = [cls.rays_dir1_all, cls.rays_dir2_all]
        
        # Prepare y_train
        y_vals = cls.y_func_all.reshape(-1, 1)
        
        cls.y_train = [
            [y_vals, cls.dy_dir1_sm1, cls.dy_dir2_sm1],  # SM1: 1st order only
            [y_vals, cls.dy_dir1_sm2, cls.dy_dir2_sm2, 
             cls.d2y_dir1_sm2, cls.d2y_dir2_sm2]  # SM2: 1st + 2nd order
        ]
        
        # Derivative indices
        cls.der_indices = [
            [[[[1, 1]], [[2, 1]]]],  # SM1: first-order only
            [[[[1, 1]], [[2, 1]]], [[[1, 2]], [[2, 2]]]]  # SM2: first and second order
        ]
        
        # Derivative locations - DISJOINT
        cls.derivative_locations = [
            [cls.sm1_indices, cls.sm1_indices],  # SM1: outer points
            [cls.sm2_indices, cls.sm2_indices, cls.sm2_indices, cls.sm2_indices]  # SM2: center
        ]
        
        # Initialize model
        cls.model = wdegp(
            cls.X_train,
            cls.y_train,
            n_order=2,
            n_bases=2,
            der_indices=cls.der_indices,
            derivative_locations=cls.derivative_locations,
            submodel_type='gddegp',
            rays_list=cls.rays_list,
            normalize=True,
            kernel="SE",
            kernel_type="isotropic"
        )
        
        # Optimize
        cls.params = cls.model.optimize_hyperparameters(
            optimizer='pso',
            pop_size=100,
            n_generations=15,
            local_opt_every=15,
            debug=True
        )
        
        # Predict with rays_predict
        cls.y_train_pred, _ = cls.model.predict(
            cls.X_train,
            cls.params,
            rays_predict=cls.rays_predict,
            calc_cov=True,
            return_deriv=True,
            derivs_to_predict=[[[1, 1]], [[2, 1]]]
        )
    
    def test_submodels_are_disjoint(self):
        """Test that submodel derivative locations are disjoint."""
        overlap = set(self.sm1_indices) & set(self.sm2_indices)
        self.assertEqual(len(overlap), 0,
                        f"Submodel indices must be disjoint, but found overlap: {overlap}")
    
    def test_mixed_order_structure(self):
        """Test that submodels have different derivative orders."""
        self.assertEqual(len(self.y_train[0]), 3,
                        "SM1 should have 3 outputs")
        self.assertEqual(len(self.y_train[1]), 5,
                        "SM2 should have 5 outputs")
    
    def test_rays_predict_matches_training_rays(self):
        """Test that rays_predict contains the same rays used for training."""
        # Check SM1 rays are correctly extracted
        for j, idx in enumerate(self.sm1_indices):
            np.testing.assert_array_almost_equal(
                self.rays_predict[0][:, idx], self.rays_dir1_sm1[:, j],
                err_msg=f"rays_predict[0] at point {idx} should match SM1 training ray"
            )
        
        # Check SM2 rays are correctly extracted
        for j, idx in enumerate(self.sm2_indices):
            np.testing.assert_array_almost_equal(
                self.rays_predict[0][:, idx], self.rays_dir1_sm2[:, j],
                err_msg=f"rays_predict[0] at point {idx} should match SM2 training ray"
            )
    
    def test_function_interpolation(self):
        """Test function interpolation with mixed-order submodels."""
        y_func_pred = self.y_train_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Function interpolation error: {max_error}")
    
    def test_first_order_derivatives_sm1(self):
        """Test first-order derivative interpolation at SM1 (outer) points."""
        y_dir1_pred = self.y_train_pred[1, self.sm1_indices].flatten()
        y_dir1_true = self.dy_dir1_all[self.sm1_indices]
        max_error = np.max(np.abs(y_dir1_pred - y_dir1_true))
        
        self.assertLess(max_error, 1e-6,
                       f"First-order dir1 at SM1 error: {max_error}")
        
        y_dir2_pred = self.y_train_pred[2, self.sm1_indices].flatten()
        y_dir2_true = self.dy_dir2_all[self.sm1_indices]
        max_error = np.max(np.abs(y_dir2_pred - y_dir2_true))
        
        self.assertLess(max_error, 1e-6,
                       f"First-order dir2 at SM1 error: {max_error}")
    
    def test_first_order_derivatives_sm2(self):
        """Test first-order derivative interpolation at SM2 (center) points."""
        y_dir1_pred = self.y_train_pred[1, self.sm2_indices].flatten()
        y_dir1_true = self.dy_dir1_all[self.sm2_indices]
        max_error = np.max(np.abs(y_dir1_pred - y_dir1_true))
        
        self.assertLess(max_error, 1e-6,
                       f"First-order dir1 at SM2 error: {max_error}")

    # =========================================================================
    # Prediction mode tests
    # =========================================================================

    def test_predict_no_deriv_with_cov(self):
        """Test prediction with return_deriv=False, calc_cov=True."""
        y_pred, cov = self.model.predict(
            self.X_train,
            self.params,
            calc_cov=True,
            return_deriv=False
        )
        
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row (function only), got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.n_train,
                        f"Expected {self.n_train} prediction points, got {y_pred.shape[1]}")
        
        self.assertIsNotNone(cov, "Covariance should be returned when calc_cov=True")
        
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Function interpolation error (no deriv): {max_error}")

    def test_predict_no_deriv_no_cov(self):
        """Test prediction with return_deriv=False, calc_cov=False."""
        y_pred = self.model.predict(
            self.X_train,
            self.params,
            calc_cov=False,
            return_deriv=False
        )
        
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row (function only), got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.n_train,
                        f"Expected {self.n_train} prediction points, got {y_pred.shape[1]}")
        
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-6,
                       f"Function interpolation error (no deriv, no cov): {max_error}")

    def test_predict_with_deriv_no_cov(self):
        """Test prediction with return_deriv=True, calc_cov=False."""
        y_pred = self.model.predict(
            self.X_train,
            self.params,
            rays_predict=self.rays_predict,
            calc_cov=False,
            return_deriv=True,
            derivs_to_predict=[[[1, 1]], [[2, 1]]]
        )

        # Function values should be accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)

        self.assertLess(max_error, 1e-6,
                       f"Function interpolation error (with deriv, no cov): {max_error}")

    # =========================================================================
    # Cholesky fallback tests
    # =========================================================================

    def test_cholesky_fallback_mixed_orders(self):
        """Test Cholesky fallback for mixed-order model."""
        with patch('jetgp.wdegp.wdegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
            y_pred, cov = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=self.rays_predict,
                calc_cov=True,
                return_deriv=True,
                derivs_to_predict=[[[1, 1]], [[2, 1]]]
            )
        
        # Predictions should still work
        self.assertEqual(y_pred.shape[1], self.n_train,
                        f"Expected {self.n_train} prediction points, got {y_pred.shape[1]}")
        
        # Function values should be reasonably accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-1,
                       f"Function interpolation error with Cholesky fallback: {max_error}")
    
    def test_all_interpolations_summary(self):
        """Summary of mixed-order interpolation tests with point-wise rays."""
        print("\n" + "=" * 70)
        print("WDEGP GDDEGP Mixed Order Test Summary (Point-Wise Rays)")
        print("=" * 70)
        print("Configuration:")
        print(f"  - SM1 (outer, 1st order): {self.n_sm1} points")
        print(f"  - SM2 (center, 1st+2nd order): {self.n_sm2} points")
        print(f"  - Ray directions: gradient + perpendicular (unique per point)")
        print("-" * 70)
        
        errors = {
            'Function': np.abs(self.y_train_pred[0, :].flatten() - self.y_func_all),
            'D_grad (SM1, outer)': np.abs(self.y_train_pred[1, self.sm1_indices].flatten() - self.dy_dir1_all[self.sm1_indices]),
            'D_perp (SM1, outer)': np.abs(self.y_train_pred[2, self.sm1_indices].flatten() - self.dy_dir2_all[self.sm1_indices]),
            'D_grad (SM2, center)': np.abs(self.y_train_pred[1, self.sm2_indices].flatten() - self.dy_dir1_all[self.sm2_indices]),
            'D_perp (SM2, center)': np.abs(self.y_train_pred[2, self.sm2_indices].flatten() - self.dy_dir2_all[self.sm2_indices]),
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


class TestWDEGPWithGDDEGPZeroOrder(unittest.TestCase):
    """Test case for WDEGP with GDDEGP submodels using n_order=0 (function values only)."""
    
    @classmethod
    def setUpClass(cls):
        """Set up training data and model with n_order=0."""
        np.random.seed(42)
        
        # Generate 2D training data on a grid
        X1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        X2 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        X1_grid, X2_grid = np.meshgrid(X1, X2)
        cls.X_train = np.column_stack([X1_grid.flatten(), X2_grid.flatten()])
        cls.n_train = len(cls.X_train)
        
        # Split into DISJOINT submodels based on distance from origin
        distances = np.linalg.norm(cls.X_train, axis=1)
        median_dist = np.median(distances)
        
        cls.sm1_indices = [i for i in range(cls.n_train) if distances[i] < median_dist]
        cls.sm2_indices = [i for i in range(cls.n_train) if distances[i] >= median_dist]
        
        # Verify disjoint
        assert len(set(cls.sm1_indices) & set(cls.sm2_indices)) == 0, \
            "Submodel indices must be disjoint!"
        
        cls.n_sm1 = len(cls.sm1_indices)
        cls.n_sm2 = len(cls.sm2_indices)
        
        # Compute true function: f(x,y) = sin(x)cos(y)
        x = cls.X_train[:, 0]
        y = cls.X_train[:, 1]
        cls.y_func_all = np.sin(x) * np.cos(y)
        
        # Prepare y_train (function values only)
        y_vals = cls.y_func_all.reshape(-1, 1)
        
        cls.y_train = [
            [y_vals],  # Submodel 1: function only
            [y_vals]   # Submodel 2: function only
        ]
        
        # No derivative indices for n_order=0
        cls.der_indices = [
            [],  # SM1: no derivatives
            []   # SM2: no derivatives
        ]
        
        # No derivative locations for n_order=0
        cls.derivative_locations = [
            [],  # SM1
            []   # SM2
        ]
        
        # No rays for n_order=0
        cls.rays_list = [
            [],  # SM1 rays
            []   # SM2 rays
        ]
        
        # Initialize WDEGP model with GDDEGP submodels and n_order=0
        cls.model = wdegp(
            cls.X_train,
            cls.y_train,
            n_order=0,
            n_bases=2,
            der_indices=cls.der_indices,
            derivative_locations=cls.derivative_locations,
            submodel_type='gddegp',
            rays_list=cls.rays_list,
            normalize=True,
            kernel="SE",
            kernel_type="isotropic"
        )
        
        # Optimize hyperparameters
        cls.params = cls.model.optimize_hyperparameters(
            optimizer='adam',
            n_restart_optimizer=5,
            debug=True
        )
    
    def test_configuration_zero_order(self):
        """Verify model configuration is correct for n_order=0."""
        self.assertEqual(len(self.der_indices[0]), 0, "SM1 should have no derivative indices")
        self.assertEqual(len(self.der_indices[1]), 0, "SM2 should have no derivative indices")
        self.assertEqual(len(self.derivative_locations[0]), 0, "SM1 should have no derivative locations")
        self.assertEqual(len(self.derivative_locations[1]), 0, "SM2 should have no derivative locations")
        self.assertEqual(len(self.rays_list[0]), 0, "SM1 should have no rays")
        self.assertEqual(len(self.rays_list[1]), 0, "SM2 should have no rays")
    
    def test_training_shapes_zero_order(self):
        """Ensure training data has correct shapes for n_order=0."""
        self.assertEqual(self.X_train.shape, (self.n_train, 2),
                        f"Training points should be {self.n_train}×2")
        self.assertEqual(len(self.y_train[0]), 1,
                        "SM1 y_train should only contain function values")
        self.assertEqual(len(self.y_train[1]), 1,
                        "SM2 y_train should only contain function values")
    
    def test_predict_zero_order_with_cov(self):
        """Test prediction with n_order=0, calc_cov=True, return_deriv=False."""
        y_pred, cov = self.model.predict(
            self.X_train,
            self.params,
            calc_cov=True,
            return_deriv=False
        )
        
        # Should only return function values
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row (function only), got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.n_train,
                        f"Expected {self.n_train} prediction points, got {y_pred.shape[1]}")
        
        # Covariance should be returned
        self.assertIsNotNone(cov, "Covariance should be returned when calc_cov=True")
        
        # Function values should be accurate (interpolation)
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-2,
                       f"Function interpolation error: {max_error}")
    
    def test_predict_zero_order_no_cov(self):
        """Test prediction with n_order=0, calc_cov=False, return_deriv=False."""
        y_pred = self.model.predict(
            self.X_train,
            self.params,
            calc_cov=False,
            return_deriv=False
        )
        
        # Should only return function values
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row (function only), got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.n_train,
                        f"Expected {self.n_train} prediction points, got {y_pred.shape[1]}")
        
        # Function values should be accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-2,
                       f"Function interpolation error: {max_error}")
    
    def test_function_interpolation_zero_order(self):
        """Verify that the WDEGP with n_order=0 interpolates function values."""
        y_pred, _ = self.model.predict(
            self.X_train,
            self.params,
            calc_cov=True,
            return_deriv=False
        )
        
        y_pred_func = y_pred[0, :].flatten()
        abs_err = np.abs(y_pred_func - self.y_func_all)
        max_err = np.max(abs_err)
        mean_err = np.mean(abs_err)
        
        self.assertLess(max_err, 1e-2, f"Function max error too large: {max_err}")
        self.assertLess(mean_err, 1e-2, f"Function mean error too large: {mean_err}")
    
    # =========================================================================
    # Warning tests for rays_predict (zero order)
    # =========================================================================
    
    def test_predict_no_deriv_with_rays_warns_zero_order(self):
        """Test warning when rays_predict provided with return_deriv=False (n_order=0)."""
        # Create dummy rays for testing
        dummy_rays = [np.ones((2, self.n_train)), np.ones((2, self.n_train))]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            y_pred = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=dummy_rays,  # Rays provided but not needed
                calc_cov=False,
                return_deriv=False
            )
            
            # Check that a warning was issued
            self.assertGreater(len(w), 0, 
                              "Expected a warning when rays_predict provided with return_deriv=False")
        
        # Predictions should still work
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row, got {y_pred.shape[0]}")
    
    # =========================================================================
    # Cholesky fallback tests (zero order)
    # =========================================================================
    
    def test_cholesky_fallback_zero_order(self):
        """Test Cholesky fallback for n_order=0 model."""
        with patch('jetgp.wdegp.wdegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
            y_pred, cov = self.model.predict(
                self.X_train,
                self.params,
                calc_cov=True,
                return_deriv=False
            )
        
        # Predictions should still work
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row, got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.n_train,
                        f"Expected {self.n_train} prediction points, got {y_pred.shape[1]}")
        
        # Function values should be reasonably accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func_all)
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-1,
                       f"Function interpolation error with Cholesky fallback: {max_error}")
    
    def test_cholesky_vs_fallback_consistency_zero_order(self):
        """Test Cholesky and fallback consistency for n_order=0 model."""
        # Get predictions with normal Cholesky
        y_pred_chol, cov_chol = self.model.predict(
            self.X_train,
            self.params,
            calc_cov=True,
            return_deriv=False
        )
        
        # Get predictions with fallback
        with patch('jetgp.wdegp.wdegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
            y_pred_fallback, cov_fallback = self.model.predict(
                self.X_train,
                self.params,
                calc_cov=True,
                return_deriv=False
            )
        
        # Results should match
        np.testing.assert_array_almost_equal(
            y_pred_chol, y_pred_fallback, decimal=6,
            err_msg="Cholesky and fallback predictions should match (n_order=0)"
        )
    
    def test_comprehensive_summary_zero_order(self):
        """Print a comprehensive summary for n_order=0 model."""
        y_pred, cov = self.model.predict(
            self.X_train,
            self.params,
            calc_cov=True,
            return_deriv=False
        )
        
        y_pred_func = y_pred[0, :].flatten()
        func_err = np.abs(y_pred_func - self.y_func_all)
        
        print("\n" + "=" * 70)
        print("WDEGP GDDEGP Zero-Order (n_order=0) Test Summary")
        print("=" * 70)
        print("Configuration:")
        print(f"  - Total points: {self.n_train}")
        print(f"  - Submodel 1 (inner): {self.n_sm1} points")
        print(f"  - Submodel 2 (outer): {self.n_sm2} points")
        print(f"  - n_order: 0 (function values only)")
        print("-" * 70)
        
        all_passed = True
        
        # Function interpolation
        max_func_err = np.max(func_err)
        mean_func_err = np.mean(func_err)
        passed = max_func_err < 1e-2
        all_passed = all_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Function values | Max: {max_func_err:.2e} | Mean: {mean_func_err:.2e}")
        
        print("=" * 70)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print("=" * 70 + "\n")
        
        self.assertTrue(all_passed, "Not all interpolation tests passed")


def run_tests_with_details():
    """Run all test cases with detailed output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestWDEGPWithGDDEGPSubmodels))
    suite.addTests(loader.loadTestsFromTestCase(TestWDEGPWithGDDEGPMixedOrders))
    suite.addTests(loader.loadTestsFromTestCase(TestWDEGPWithGDDEGPZeroOrder))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    print("\nRunning WDEGP with GDDEGP Submodels Unit Tests (Point-Wise Rays)...")
    print("=" * 70 + "\n")
    
    result = run_tests_with_details()
    sys.exit(0 if result.wasSuccessful() else 1)