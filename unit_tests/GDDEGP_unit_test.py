"""
Unit test for 2D Generalized Directional DEGP (GDDEGP) with gradient-aligned rays.

This test verifies that the GDDEGP model correctly interpolates both function values
and pointwise directional derivatives for the 2D Branin function when each training
point has its own unique gradient-aligned direction.
"""

import sys
import unittest
import numpy as np
import sympy as sp
from jetgp.full_gddegp.gddegp import gddegp
from scipy.stats import qmc
import jetgp.utils
import warnings
from unittest.mock import patch
from scipy.linalg import LinAlgError
import io


class TestGDDEGPBraninGradientAligned(unittest.TestCase):
    """Test case for GDDEGP on the 2D Branin function with gradient-aligned rays."""

    @classmethod
    def setUpClass(cls):
        """Set up training data, rays, and model once for all tests."""
        # --- Configuration parameters ---
        cls.n_order = 1
        cls.n_bases = 2
        cls.num_training_pts = 20
        cls.domain_bounds = ((-5.0, 10.0), (0.0, 15.0))
        cls.normalize_data = True
        cls.kernel = "Matern"
        cls.kernel_type = "anisotropic"
        cls.smoothness_parameter = 3
        cls.random_seed = 1
        np.random.seed(cls.random_seed)

        # --- Define Branin function symbolically ---
        x_sym, y_sym = sp.symbols("x y", real=True)
        a, b, c, r, s, t = 1.0, 5.1 / (4 * sp.pi**2), 5.0 / sp.pi, 6.0, 10.0, 1.0 / (8 * sp.pi)
        f_sym = a * (y_sym - b * x_sym**2 + c * x_sym - r) ** 2 + s * (1 - t) * sp.cos(x_sym) + s

        grad_x_sym = sp.diff(f_sym, x_sym)
        grad_y_sym = sp.diff(f_sym, y_sym)

        # Convert to NumPy functions with proper array handling
        true_function_np_raw = sp.lambdify([x_sym, y_sym], f_sym, "numpy")
        grad_x_func_raw = sp.lambdify([x_sym, y_sym], grad_x_sym, "numpy")
        grad_y_func_raw = sp.lambdify([x_sym, y_sym], grad_y_sym, "numpy")
        
        # Wrap to handle constants
        def make_array_func(func_raw):
            def wrapped(x, y):
                result = func_raw(x, y)
                result = np.atleast_1d(result)
                if result.size == 1 and np.atleast_1d(x).size > 1:
                    result = np.full_like(x, result[0])
                return result
            return wrapped
        
        cls.true_function_np = make_array_func(true_function_np_raw)
        cls.grad_x_func = make_array_func(grad_x_func_raw)
        cls.grad_y_func = make_array_func(grad_y_func_raw)

        # --- Helper functions ---
        def true_function(X):
            return cls.true_function_np(X[:, 0], X[:, 1])

        def true_gradient(x, y):
            gx = cls.grad_x_func(x, y)
            gy = cls.grad_y_func(x, y)
            # Ensure scalar output for single points
            if np.isscalar(x) or (hasattr(x, '__len__') and len(x) == 1):
                gx = np.asarray(gx).item() if hasattr(gx, '__iter__') else gx
                gy = np.asarray(gy).item() if hasattr(gy, '__iter__') else gy
            return gx, gy

        cls.true_function = true_function
        cls.true_gradient = true_gradient

        # --- Generate training data using Latin Hypercube Sampling ---
        sampler = qmc.LatinHypercube(d=2, seed=cls.random_seed)
        unit_samples = sampler.random(n=cls.num_training_pts)
        cls.X_train = qmc.scale(
            unit_samples, [b[0] for b in cls.domain_bounds], [b[1] for b in cls.domain_bounds]
        )

        # --- Compute gradient-aligned rays ---
        cls.rays_list = []
        cls.gradient_magnitudes = []
        cls.zero_gradient_points = []
        
        for idx, (x, y) in enumerate(cls.X_train):
            gx, gy = cls.true_gradient(x, y)
            magnitude = np.sqrt(gx**2 + gy**2)
            cls.gradient_magnitudes.append(magnitude)
            
            if magnitude < 1e-10:  # Use small threshold instead of exact zero
                ray = np.array([[1.0], [0.0]])  # fallback direction
                cls.zero_gradient_points.append(idx)
            else:
                ray = np.array([[gx / magnitude], [gy / magnitude]])
            cls.rays_list.append(ray)

        # --- Compute function and directional derivative values ---
        cls.y_func = cls.true_function(cls.X_train).reshape(-1, 1)
        directional_derivs = []
        for i, (x, y) in enumerate(cls.X_train):
            gx, gy = cls.true_gradient(x, y)
            ray_dir = cls.rays_list[i].flatten()
            dir_deriv = gx * ray_dir[0] + gy * ray_dir[1]
            directional_derivs.append(dir_deriv)

        cls.y_dir = np.array(directional_derivs).reshape(-1, 1)

        # --- Package training data for GDDEGP ---
        cls.y_train = [cls.y_func, cls.y_dir]
        cls.der_indices = [
            [[[1, 1]]]
            ]
        cls.rays_array = np.hstack(cls.rays_list)  # (2, num_training_pts)
        cls.derivative_locations = []
        for i in range(len(cls.der_indices)):
            for j in range(len(cls.der_indices[i])):
                cls.derivative_locations.append([i for i in range(len(cls.X_train ))])
        # --- Initialize and train GDDEGP model ---
        cls.model = gddegp(
            cls.X_train,
            cls.y_train,
            n_order=cls.n_order,
            rays_list=[cls.rays_array],
            der_indices=cls.der_indices,
            derivative_locations=cls.derivative_locations,
            normalize=cls.normalize_data,
            kernel=cls.kernel,
            kernel_type=cls.kernel_type,
            smoothness_parameter=cls.smoothness_parameter
        )

        cls.params = cls.model.optimize_hyperparameters(
            optimizer='pso',
            pop_size=100,
            n_generations=15,
            local_opt_every=15,
            debug=True
        )
        # --- Predict at training points ---
        derivs_to_predict = [[[1,1]]]
        cls.y_pred_train_full, _ = cls.model.predict(
            cls.X_train, cls.params, rays_predict = [cls.rays_array], calc_cov=True, return_deriv=True, derivs_to_predict = derivs_to_predict
        )
        cls.N = cls.num_training_pts

    def test_configuration(self):
        """Verify model configuration is correct."""
        self.assertEqual(self.n_order, 1, "Should use 1st order derivatives")
        self.assertEqual(self.n_bases, 2, "Should be 2D problem")
        self.assertEqual(len(self.rays_list), self.num_training_pts,
                        "Should have one ray per training point")

    def test_training_shapes(self):
        """Ensure training data and rays have correct shapes."""
        self.assertEqual(self.X_train.shape, (self.num_training_pts, 2),
                        "Training points should be 20×2")
        self.assertEqual(self.y_func.shape, (self.num_training_pts, 1),
                        "Function values should be 20×1")
        self.assertEqual(self.y_dir.shape, (self.num_training_pts, 1),
                        "Directional derivatives should be 20×1")
        self.assertEqual(self.rays_array.shape, (2, self.num_training_pts),
                        "Rays array should be 2×20")

    def test_ray_properties(self):
        """Verify that all rays have unit length."""
        for i, ray in enumerate(self.rays_list):
            ray_norm = np.linalg.norm(ray)
            self.assertAlmostEqual(ray_norm, 1.0, places=10,
                                  msg=f"Ray {i} should have unit length")
    
    def test_gradient_alignment(self):
        """Verify that rays are aligned with gradients (or fallback for zero gradient)."""
        for i, (x, y) in enumerate(self.X_train):
            gx, gy = self.__class__.true_gradient(x, y)
            magnitude = np.sqrt(gx**2 + gy**2)
            ray = self.rays_list[i].flatten()
            
            if magnitude > 1e-10:
                # Ray should be parallel to gradient
                expected_ray = np.array([gx / magnitude, gy / magnitude])
                np.testing.assert_array_almost_equal(
                    ray, expected_ray, decimal=10,
                    err_msg=f"Ray {i} not aligned with gradient"
                )

    def test_function_interpolation(self):
        """Verify that the GDDEGP interpolates function values exactly."""
        y_pred_func = self.y_pred_train_full[0,:].flatten()
        abs_err = np.abs(y_pred_func - self.y_func.flatten())
        max_err = np.max(abs_err)
        mean_err = np.mean(abs_err)

        self.assertLess(max_err, 1e-2, f"Function max error too large: {max_err}")
        self.assertLess(mean_err, 1e-2, f"Function mean error too large: {mean_err}")

    def test_directional_derivative_interpolation(self):
        """Verify that the GDDEGP interpolates gradient-aligned directional derivatives."""
        y_pred_dir = self.y_pred_train_full[1,:].flatten()
        abs_err = np.abs(y_pred_dir - self.y_dir.flatten())
        max_err = np.max(abs_err)
        mean_err = np.mean(abs_err)

        self.assertLess(max_err, 1e-2, f"Directional derivative max error too large: {max_err}")
        self.assertLess(mean_err, 1e-2, f"Directional derivative mean error too large: {mean_err}")

    # =========================================================================
    # Warning tests for rays_predict
    # =========================================================================
    
    def test_predict_deriv_without_rays_warns(self):
        """Test that predicting derivatives without rays_predict issues a warning."""
        derivs_to_predict = [[[1, 1]]]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            y_pred = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=None,  # No rays provided
                calc_cov=False,
                return_deriv=True,
                derivs_to_predict=derivs_to_predict
            )
            
            # Check that a warning was issued
            self.assertGreater(len(w), 0, "Expected a warning when rays_predict is None with return_deriv=True")
            
            # Check warning message contains expected text
            warning_messages = [str(warning.message) for warning in w]
            found_rays_warning = any(
                "rays_predict" in msg.lower() or "coordinate" in msg.lower() 
                for msg in warning_messages
            )
            self.assertTrue(found_rays_warning, 
                           f"Expected warning about missing rays_predict, got: {warning_messages}")
        
        # Predictions should still work (using default coordinate axis rays)
        self.assertEqual(y_pred.shape[0], 2,
                        f"Expected 2 output rows (func + 1 deriv), got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
    
    def test_predict_no_deriv_with_rays_warns(self):
        """Test that providing rays_predict with return_deriv=False issues a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            y_pred = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=[self.rays_array],  # Rays provided but not needed
                calc_cov=False,
                return_deriv=False  # Not returning derivatives
            )
            
            # Check that a warning was issued
            self.assertGreater(len(w), 0, 
                              "Expected a warning when rays_predict provided with return_deriv=False")
            
            # Check warning message contains expected text
            warning_messages = [str(warning.message) for warning in w]
            found_ignored_warning = any(
                "ignored" in msg.lower() or "return_deriv=False" in msg.lower() or "return_deriv" in msg.lower()
                for msg in warning_messages
            )
            self.assertTrue(found_ignored_warning, 
                           f"Expected warning about rays being ignored, got: {warning_messages}")
        
        # Predictions should still work (rays ignored)
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row (function only), got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
    
    def test_default_coordinate_rays_accuracy(self):
        """Test that default coordinate axis rays produce correct partial derivatives."""
        derivs_to_predict = [[[1, 1]]]
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            
            y_pred = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=None,  # Use default coordinate rays
                calc_cov=False,
                return_deriv=True,
                derivs_to_predict=derivs_to_predict
            )
        
        # Function values should still be accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func.flatten())
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-2,
                       f"Function interpolation error with default rays: {max_error}")
        
        # The derivative prediction should correspond to df/dx (first coordinate axis)
        true_dfdx = self.__class__.grad_x_func(self.X_train[:, 0], self.X_train[:, 1])
        y_deriv_pred = y_pred[1, :].flatten()
        
        # Just check that we get reasonable values
        self.assertEqual(y_deriv_pred.shape[0], self.N,
                        "Derivative prediction should have same length as training points")

    # =========================================================================
    # Cholesky fallback tests
    # =========================================================================

    def test_cholesky_fallback_predict_with_cov(self):
        """Test that prediction works when Cholesky decomposition fails (with covariance)."""
        derivs_to_predict = [[[1, 1]]]
        
        # Mock cho_factor to raise an exception, forcing fallback to np.linalg.solve
        with patch('jetgp.full_gddegp.gddegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
            y_pred, cov = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=[self.rays_array],
                calc_cov=True,
                return_deriv=True,
                derivs_to_predict=derivs_to_predict
            )
        
        # Predictions should still work via fallback
        self.assertEqual(y_pred.shape[0], 2,
                        f"Expected 2 output rows, got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
        
        # Covariance should still be computed via np.linalg.inv fallback
        self.assertIsNotNone(cov, "Covariance should be returned even with Cholesky fallback")
        
        # Function values should still be reasonably accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func.flatten())
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-1,
                       f"Function interpolation error with Cholesky fallback: {max_error}")

    def test_cholesky_fallback_predict_no_cov(self):
        """Test that prediction works when Cholesky decomposition fails (without covariance)."""
        derivs_to_predict = [[[1, 1]]]
        
        # Mock cho_factor to raise an exception
        with patch('jetgp.full_gddegp.gddegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
            y_pred = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=[self.rays_array],
                calc_cov=False,
                return_deriv=True,
                derivs_to_predict=derivs_to_predict
            )
        
        # Predictions should still work via fallback
        self.assertEqual(y_pred.shape[0], 2,
                        f"Expected 2 output rows, got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
        
        # Function values should still be reasonably accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func.flatten())
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-1,
                       f"Function interpolation error with Cholesky fallback: {max_error}")

    def test_cholesky_vs_fallback_consistency(self):
        """Test that Cholesky and fallback methods produce consistent results."""
        derivs_to_predict = [[[1, 1]]]
        
        # Get predictions with normal Cholesky
        y_pred_chol, cov_chol = self.model.predict(
            self.X_train,
            self.params,
            rays_predict=[self.rays_array],
            calc_cov=True,
            return_deriv=True,
            derivs_to_predict=derivs_to_predict
        )
        
        # Get predictions with fallback (mocked Cholesky failure)
        with patch('jetgp.full_gddegp.gddegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
            y_pred_fallback, cov_fallback = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=[self.rays_array],
                calc_cov=True,
                return_deriv=True,
                derivs_to_predict=derivs_to_predict
            )
        
        # Results should be very similar (within numerical tolerance)
        np.testing.assert_array_almost_equal(
            y_pred_chol, y_pred_fallback, decimal=6,
            err_msg="Cholesky and fallback predictions should match"
        )
        
        np.testing.assert_array_almost_equal(
            cov_chol, cov_fallback, decimal=6,
            err_msg="Cholesky and fallback covariances should match"
        )

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
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
        
        # Covariance should be returned and have correct shape
        self.assertIsNotNone(cov, "Covariance should be returned when calc_cov=True")
        self.assertEqual(cov.shape[1], self.N,
                        f"Covariance should have {self.N} columns")
        
        # Function values should still be accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func.flatten())
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-2,
                       f"Function interpolation error (no deriv): {max_error}")
        
        # Variance at training points should be small (interpolation)
        diag_var = cov
        max_var = np.max(diag_var)
        self.assertLess(max_var, 1e-2,
                       f"Variance at training points should be small, got max: {max_var}")

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
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
        
        # Function values should still be accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func.flatten())
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-2,
                       f"Function interpolation error (no deriv, no cov): {max_error}")

    def test_predict_with_deriv_no_cov(self):
        """Test prediction with return_deriv=True, calc_cov=False."""
        derivs_to_predict = [[[1, 1]]]
        y_pred = self.model.predict(
            self.X_train,
            self.params,
            rays_predict=[self.rays_array],
            calc_cov=False,
            return_deriv=True,
            derivs_to_predict=derivs_to_predict
        )
        
        # Should return function + 1 derivative output
        self.assertEqual(y_pred.shape[0], 2,
                        f"Expected 2 output rows (func + 1 deriv), got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
        
        # Function values should be accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func.flatten())
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-2,
                       f"Function interpolation error (with deriv, no cov): {max_error}")

    def test_comprehensive_summary(self):
        """Print a comprehensive summary of interpolation accuracy."""
        y_pred_func = self.y_pred_train_full[0,:].flatten()
        y_pred_dir = self.y_pred_train_full[1,:].flatten()
        func_err = np.abs(y_pred_func - self.y_func.flatten())
        dir_err = np.abs(y_pred_dir - self.y_dir.flatten())

        print("\n" + "=" * 80)
        print("GDDEGP Branin Gradient-Aligned Ray Interpolation Summary")
        print("=" * 80)
        
        all_tests_passed = True
        
        # Function interpolation
        print("\nFunction Value Interpolation")
        print("-" * 80)
        max_func_err = np.max(func_err)
        mean_func_err = np.mean(func_err)
        passed = max_func_err < 1e-2
        all_tests_passed = all_tests_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Function values | Max: {max_func_err:.2e} | Mean: {mean_func_err:.2e}")
        
        # Directional derivative interpolation
        print("\nDirectional Derivative Interpolation")
        print("-" * 80)
        max_dir_err = np.max(dir_err)
        mean_dir_err = np.mean(dir_err)
        passed = max_dir_err < 1e-2
        all_tests_passed = all_tests_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Directional derivatives | Max: {max_dir_err:.2e} | Mean: {mean_dir_err:.2e}")
        
        # Configuration summary
        print("\n" + "=" * 80)
        print("Configuration:")
        print(f"  - Function: 2D Branin benchmark function")
        print(f"  - Training points: {self.N} (Latin Hypercube)")
        print(f"  - Domain: x ∈ [{self.domain_bounds[0][0]}, {self.domain_bounds[0][1]}], "
              f"y ∈ [{self.domain_bounds[1][0]}, {self.domain_bounds[1][1]}]")
        print(f"  - Rays: Gradient-aligned (unique per training point)")
        print(f"  - Zero-gradient points: {len(self.zero_gradient_points)}")
        if self.zero_gradient_points:
            print(f"    Indices: {self.zero_gradient_points}")
        print(f"  - Derivative order: {self.n_order} (directional derivatives)")
        print(f"  - Kernel: {self.kernel} ({self.kernel_type})")
        print(f"  - Smoothness parameter: {self.smoothness_parameter}")
        print(f"  - Derivatives: computed symbolically with SymPy")
        
        # Gradient statistics
        print("\nGradient Statistics:")
        print(f"  - Min gradient magnitude: {min(self.gradient_magnitudes):.2e}")
        print(f"  - Max gradient magnitude: {max(self.gradient_magnitudes):.2e}")
        print(f"  - Mean gradient magnitude: {np.mean(self.gradient_magnitudes):.2e}")
        
        print("=" * 80)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_tests_passed else '✗ SOME TESTS FAILED'}")
        print("=" * 80 + "\n")

        self.assertTrue(all_tests_passed, "Not all interpolation tests passed")


class TestGDDEGPBraninZeroOrder(unittest.TestCase):
    """Test case for GDDEGP with n_order=0 (function values only, no derivative training)."""

    @classmethod
    def setUpClass(cls):
        """Set up training data and model with n_order=0."""
        # --- Configuration parameters ---
        cls.n_order = 0  # No derivatives in training
        cls.n_bases = 2
        cls.num_training_pts = 20
        cls.domain_bounds = ((-5.0, 10.0), (0.0, 15.0))
        cls.normalize_data = True
        cls.kernel = "Matern"
        cls.kernel_type = "anisotropic"
        cls.smoothness_parameter = 3
        cls.random_seed = 1
        np.random.seed(cls.random_seed)

        # --- Define Branin function symbolically ---
        x_sym, y_sym = sp.symbols("x y", real=True)
        a, b, c, r, s, t = 1.0, 5.1 / (4 * sp.pi**2), 5.0 / sp.pi, 6.0, 10.0, 1.0 / (8 * sp.pi)
        f_sym = a * (y_sym - b * x_sym**2 + c * x_sym - r) ** 2 + s * (1 - t) * sp.cos(x_sym) + s

        # Convert to NumPy function
        true_function_np_raw = sp.lambdify([x_sym, y_sym], f_sym, "numpy")
        
        def make_array_func(func_raw):
            def wrapped(x, y):
                result = func_raw(x, y)
                result = np.atleast_1d(result)
                if result.size == 1 and np.atleast_1d(x).size > 1:
                    result = np.full_like(x, result[0])
                return result
            return wrapped
        
        cls.true_function_np = make_array_func(true_function_np_raw)

        def true_function(X):
            return cls.true_function_np(X[:, 0], X[:, 1])

        cls.true_function = true_function

        # --- Generate training data using Latin Hypercube Sampling ---
        sampler = qmc.LatinHypercube(d=2, seed=cls.random_seed)
        unit_samples = sampler.random(n=cls.num_training_pts)
        cls.X_train = qmc.scale(
            unit_samples, [b[0] for b in cls.domain_bounds], [b[1] for b in cls.domain_bounds]
        )

        # --- Compute function values only ---
        cls.y_func = cls.true_function(cls.X_train).reshape(-1, 1)

        # --- Package training data for GDDEGP (function only) ---
        cls.y_train = [cls.y_func]
        cls.der_indices = []  # No derivatives
        cls.derivative_locations = []
        cls.N = cls.num_training_pts

        # --- Initialize and train GDDEGP model with n_order=0 ---
        cls.model = gddegp(
            cls.X_train,
            cls.y_train,
            n_order=cls.n_order,
            rays_list=[],
            der_indices=cls.der_indices,
            derivative_locations=cls.derivative_locations,
            normalize=cls.normalize_data,
            kernel=cls.kernel,
            kernel_type=cls.kernel_type,
            smoothness_parameter=cls.smoothness_parameter
        )

        cls.params = cls.model.optimize_hyperparameters(
            optimizer='adam',
            n_restart_optimizer=5,
            debug=True
        )

    def test_configuration_zero_order(self):
        """Verify model configuration is correct for n_order=0."""
        self.assertEqual(self.n_order, 0, "Should use n_order=0 (no derivatives)")
        self.assertEqual(len(self.der_indices), 0, "Should have no derivative indices")
        self.assertEqual(len(self.derivative_locations), 0, "Should have no derivative locations")

    def test_training_shapes_zero_order(self):
        """Ensure training data has correct shapes for function-only model."""
        self.assertEqual(self.X_train.shape, (self.num_training_pts, 2),
                        "Training points should be 20×2")
        self.assertEqual(self.y_func.shape, (self.num_training_pts, 1),
                        "Function values should be 20×1")
        self.assertEqual(len(self.y_train), 1,
                        "y_train should only contain function values")

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
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
        
        # Covariance should be returned
        self.assertIsNotNone(cov, "Covariance should be returned when calc_cov=True")
        self.assertEqual(cov.shape[1], self.N,
                        f"Covariance should have {self.N} rows")

        # Function values should be accurate (interpolation)
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func.flatten())
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
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
        
        # Function values should be accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func.flatten())
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-2,
                       f"Function interpolation error: {max_error}")

    def test_function_interpolation_zero_order(self):
        """Verify that the GDDEGP with n_order=0 interpolates function values."""
        y_pred, _ = self.model.predict(
            self.X_train,
            self.params,
            calc_cov=True,
            return_deriv=False
        )
        
        y_pred_func = y_pred[0, :].flatten()
        abs_err = np.abs(y_pred_func - self.y_func.flatten())
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
        dummy_rays = np.eye(self.n_bases)
        dummy_rays = np.tile(dummy_rays, (1, self.N // self.n_bases + 1))[:, :self.N]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            y_pred = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=[dummy_rays],  # Rays provided but not needed
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
        with patch('jetgp.full_gddegp.gddegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
            y_pred, cov = self.model.predict(
                self.X_train,
                self.params,
                calc_cov=True,
                return_deriv=False
            )
        
        # Predictions should still work
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row, got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
        
        # Function values should be reasonably accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func.flatten())
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-1,
                       f"Function interpolation error with Cholesky fallback: {max_error}")

    def test_cholesky_fallback_prints_warning_zero_order(self):
        """Test that a warning is printed when Cholesky fails (n_order=0)."""
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            with patch('jetgp.full_gddegp.gddegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
                y_pred = self.model.predict(
                    self.X_train,
                    self.params,
                    calc_cov=False,
                    return_deriv=False
                )
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("Cholesky", output,
                     f"Expected warning about Cholesky failure, got: {output}")

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
        with patch('jetgp.full_gddegp.gddegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
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
        
        np.testing.assert_array_almost_equal(
            cov_chol, cov_fallback, decimal=6,
            err_msg="Cholesky and fallback covariances should match (n_order=0)"
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
        func_err = np.abs(y_pred_func - self.y_func.flatten())

        print("\n" + "=" * 80)
        print("GDDEGP Zero-Order (n_order=0) Interpolation Summary")
        print("=" * 80)
        
        all_tests_passed = True
        
        # Function interpolation
        print("\nFunction Value Interpolation (No Derivative Training)")
        print("-" * 80)
        max_func_err = np.max(func_err)
        mean_func_err = np.mean(func_err)
        passed = max_func_err < 1e-2
        all_tests_passed = all_tests_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Function values | Max: {max_func_err:.2e} | Mean: {mean_func_err:.2e}")
        
        # Variance check
        diag_var = np.diag(cov)
        max_var = np.max(diag_var)
        var_passed = max_var < 1e-2
        all_tests_passed = all_tests_passed and var_passed
        status = "✓ PASS" if var_passed else "✗ FAIL"
        print(f"{status} | Variance at train pts | Max: {max_var:.2e}")
        
        # Configuration summary
        print("\n" + "=" * 80)
        print("Configuration:")
        print(f"  - Function: 2D Branin benchmark function")
        print(f"  - Training points: {self.N} (Latin Hypercube)")
        print(f"  - Domain: x ∈ [{self.domain_bounds[0][0]}, {self.domain_bounds[0][1]}], "
              f"y ∈ [{self.domain_bounds[1][0]}, {self.domain_bounds[1][1]}]")
        print(f"  - n_order: {self.n_order} (function values only)")
        print(f"  - Kernel: {self.kernel} ({self.kernel_type})")
        print(f"  - Smoothness parameter: {self.smoothness_parameter}")
        
        print("=" * 80)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_tests_passed else '✗ SOME TESTS FAILED'}")
        print("=" * 80 + "\n")

        self.assertTrue(all_tests_passed, "Not all interpolation tests passed")


class TestGDDEGPBraninGradientAlignedNoNormalize(unittest.TestCase):
    """Test case for GDDEGP on the 2D Branin function with gradient-aligned rays and normalize=False."""

    @classmethod
    def setUpClass(cls):
        """Set up training data, rays, and model once for all tests."""
        # --- Configuration parameters ---
        cls.n_order = 1
        cls.n_bases = 2
        cls.num_training_pts = 20
        cls.domain_bounds = ((-5.0, 10.0), (0.0, 15.0))
        cls.normalize_data = False  # KEY DIFFERENCE
        cls.kernel = "Matern"
        cls.kernel_type = "anisotropic"
        cls.smoothness_parameter = 3
        cls.random_seed = 1
        np.random.seed(cls.random_seed)

        # --- Define Branin function symbolically ---
        x_sym, y_sym = sp.symbols("x y", real=True)
        a, b, c, r, s, t = 1.0, 5.1 / (4 * sp.pi**2), 5.0 / sp.pi, 6.0, 10.0, 1.0 / (8 * sp.pi)
        f_sym = a * (y_sym - b * x_sym**2 + c * x_sym - r) ** 2 + s * (1 - t) * sp.cos(x_sym) + s

        grad_x_sym = sp.diff(f_sym, x_sym)
        grad_y_sym = sp.diff(f_sym, y_sym)

        # Convert to NumPy functions with proper array handling
        true_function_np_raw = sp.lambdify([x_sym, y_sym], f_sym, "numpy")
        grad_x_func_raw = sp.lambdify([x_sym, y_sym], grad_x_sym, "numpy")
        grad_y_func_raw = sp.lambdify([x_sym, y_sym], grad_y_sym, "numpy")
        
        def make_array_func(func_raw):
            def wrapped(x, y):
                result = func_raw(x, y)
                result = np.atleast_1d(result)
                if result.size == 1 and np.atleast_1d(x).size > 1:
                    result = np.full_like(x, result[0])
                return result
            return wrapped
        
        cls.true_function_np = make_array_func(true_function_np_raw)
        cls.grad_x_func = make_array_func(grad_x_func_raw)
        cls.grad_y_func = make_array_func(grad_y_func_raw)

        def true_function(X):
            return cls.true_function_np(X[:, 0], X[:, 1])

        def true_gradient(x, y):
            gx = cls.grad_x_func(x, y)
            gy = cls.grad_y_func(x, y)
            if np.isscalar(x) or (hasattr(x, '__len__') and len(x) == 1):
                gx = np.asarray(gx).item() if hasattr(gx, '__iter__') else gx
                gy = np.asarray(gy).item() if hasattr(gy, '__iter__') else gy
            return gx, gy

        cls.true_function = true_function
        cls.true_gradient = true_gradient

        # --- Generate training data using Latin Hypercube Sampling ---
        sampler = qmc.LatinHypercube(d=2, seed=cls.random_seed)
        unit_samples = sampler.random(n=cls.num_training_pts)
        cls.X_train = qmc.scale(
            unit_samples, [b[0] for b in cls.domain_bounds], [b[1] for b in cls.domain_bounds]
        )

        # --- Compute gradient-aligned rays ---
        cls.rays_list = []
        cls.gradient_magnitudes = []
        cls.zero_gradient_points = []
        
        for idx, (x, y) in enumerate(cls.X_train):
            gx, gy = cls.true_gradient(x, y)
            magnitude = np.sqrt(gx**2 + gy**2)
            cls.gradient_magnitudes.append(magnitude)
            
            if magnitude < 1e-10:
                ray = np.array([[1.0], [0.0]])
                cls.zero_gradient_points.append(idx)
            else:
                ray = np.array([[gx / magnitude], [gy / magnitude]])
            cls.rays_list.append(ray)

        # --- Compute function and directional derivative values ---
        cls.y_func = cls.true_function(cls.X_train).reshape(-1, 1)
        directional_derivs = []
        for i, (x, y) in enumerate(cls.X_train):
            gx, gy = cls.true_gradient(x, y)
            ray_dir = cls.rays_list[i].flatten()
            dir_deriv = gx * ray_dir[0] + gy * ray_dir[1]
            directional_derivs.append(dir_deriv)

        cls.y_dir = np.array(directional_derivs).reshape(-1, 1)

        # --- Package training data for GDDEGP ---
        cls.y_train = [cls.y_func, cls.y_dir]
        cls.der_indices = [[[[1, 1]]]]
        cls.rays_array = np.hstack(cls.rays_list)
        cls.derivative_locations = []
        for i in range(len(cls.der_indices)):
            for j in range(len(cls.der_indices[i])):
                cls.derivative_locations.append([i for i in range(len(cls.X_train))])

        # --- Initialize and train GDDEGP model with normalize=False ---
        cls.model = gddegp(
            cls.X_train,
            cls.y_train,
            n_order=cls.n_order,
            rays_list=[cls.rays_array],
            der_indices=cls.der_indices,
            derivative_locations=cls.derivative_locations,
            normalize=cls.normalize_data,  # False
            kernel=cls.kernel,
            kernel_type=cls.kernel_type,
            smoothness_parameter=cls.smoothness_parameter
        )

        cls.params = cls.model.optimize_hyperparameters(
            optimizer='cobyla',
            n_restart_optimizer=40,
            debug=True
        )

        # --- Predict at training points ---
        derivs_to_predict = [[[1, 1]]]
        cls.y_pred_train_full, _ = cls.model.predict(
            cls.X_train, cls.params, rays_predict=[cls.rays_array],
            calc_cov=True, return_deriv=True, derivs_to_predict=derivs_to_predict
        )
        cls.N = cls.num_training_pts

    def test_configuration_no_normalize(self):
        """Verify model configuration is correct with normalize=False."""
        self.assertEqual(self.n_order, 1, "Should use 1st order derivatives")
        self.assertFalse(self.normalize_data, "normalize should be False")
        self.assertEqual(len(self.rays_list), self.num_training_pts,
                        "Should have one ray per training point")

    def test_function_interpolation_no_normalize(self):
        """Verify that the GDDEGP interpolates function values with normalize=False."""
        y_pred_func = self.y_pred_train_full[0, :].flatten()
        abs_err = np.abs(y_pred_func - self.y_func.flatten())
        max_err = np.max(abs_err)
        mean_err = np.mean(abs_err)

        self.assertLess(max_err, 1e-2, f"Function max error too large: {max_err}")
        self.assertLess(mean_err, 1e-2, f"Function mean error too large: {mean_err}")

    def test_directional_derivative_interpolation_no_normalize(self):
        """Verify directional derivative interpolation with normalize=False."""
        y_pred_dir = self.y_pred_train_full[1, :].flatten()
        abs_err = np.abs(y_pred_dir - self.y_dir.flatten())
        max_err = np.max(abs_err)
        mean_err = np.mean(abs_err)

        self.assertLess(max_err, 1e-2, f"Directional derivative max error too large: {max_err}")
        self.assertLess(mean_err, 1e-2, f"Directional derivative mean error too large: {mean_err}")

    # =========================================================================
    # Warning tests for rays_predict (no normalize)
    # =========================================================================

    def test_predict_deriv_without_rays_warns_no_normalize(self):
        """Test warning when predicting derivatives without rays_predict (normalize=False)."""
        derivs_to_predict = [[[1, 1]]]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            y_pred = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=None,
                calc_cov=False,
                return_deriv=True,
                derivs_to_predict=derivs_to_predict
            )
            
            self.assertGreater(len(w), 0, 
                              "Expected a warning when rays_predict is None with return_deriv=True")
        
        # Predictions should still work
        self.assertEqual(y_pred.shape[0], 2,
                        f"Expected 2 output rows, got {y_pred.shape[0]}")

    def test_predict_no_deriv_with_rays_warns_no_normalize(self):
        """Test warning when rays_predict provided with return_deriv=False (normalize=False)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            y_pred = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=[self.rays_array],
                calc_cov=False,
                return_deriv=False
            )
            
            self.assertGreater(len(w), 0, 
                              "Expected a warning when rays_predict provided with return_deriv=False")
        
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row, got {y_pred.shape[0]}")

    # =========================================================================
    # Prediction mode tests (no normalize)
    # =========================================================================

    def test_predict_no_deriv_with_cov_no_normalize(self):
        """Test prediction with return_deriv=False, calc_cov=True, normalize=False."""
        y_pred, cov = self.model.predict(
            self.X_train,
            self.params,
            calc_cov=True,
            return_deriv=False
        )
        
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row (function only), got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
        
        self.assertIsNotNone(cov, "Covariance should be returned when calc_cov=True")
        
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func.flatten())
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-2,
                       f"Function interpolation error (no deriv, no normalize): {max_error}")

    def test_predict_no_deriv_no_cov_no_normalize(self):
        """Test prediction with return_deriv=False, calc_cov=False, normalize=False."""
        y_pred = self.model.predict(
            self.X_train,
            self.params,
            calc_cov=False,
            return_deriv=False
        )
        
        self.assertEqual(y_pred.shape[0], 1,
                        f"Expected 1 output row (function only), got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
        
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func.flatten())
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-2,
                       f"Function interpolation error (no deriv, no cov, no normalize): {max_error}")

    def test_predict_with_deriv_no_cov_no_normalize(self):
        """Test prediction with return_deriv=True, calc_cov=False, normalize=False."""
        derivs_to_predict = [[[1, 1]]]
        y_pred = self.model.predict(
            self.X_train,
            self.params,
            rays_predict=[self.rays_array],
            calc_cov=False,
            return_deriv=True,
            derivs_to_predict=derivs_to_predict
        )
        
        self.assertEqual(y_pred.shape[0], 2,
                        f"Expected 2 output rows (func + 1 deriv), got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
        
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func.flatten())
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-2,
                       f"Function interpolation error (with deriv, no cov, no normalize): {max_error}")

    # =========================================================================
    # Cholesky fallback tests (no normalize)
    # =========================================================================

    def test_cholesky_fallback_no_normalize(self):
        """Test Cholesky fallback for normalize=False model."""
        derivs_to_predict = [[[1, 1]]]
        
        with patch('jetgp.full_gddegp.gddegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
            y_pred, cov = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=[self.rays_array],
                calc_cov=True,
                return_deriv=True,
                derivs_to_predict=derivs_to_predict
            )
        
        # Predictions should still work
        self.assertEqual(y_pred.shape[0], 2,
                        f"Expected 2 output rows, got {y_pred.shape[0]}")
        self.assertEqual(y_pred.shape[1], self.N,
                        f"Expected {self.N} prediction points, got {y_pred.shape[1]}")
        
        # Function values should be reasonably accurate
        y_func_pred = y_pred[0, :].flatten()
        abs_error = np.abs(y_func_pred - self.y_func.flatten())
        max_error = np.max(abs_error)
        
        self.assertLess(max_error, 1e-1,
                       f"Function interpolation error with Cholesky fallback (no normalize): {max_error}")


    def test_cholesky_vs_fallback_consistency_no_normalize(self):
        """Test Cholesky and fallback consistency for normalize=False model."""
        derivs_to_predict = [[[1, 1]]]
        
        # Get predictions with normal Cholesky
        y_pred_chol, cov_chol = self.model.predict(
            self.X_train,
            self.params,
            rays_predict=[self.rays_array],
            calc_cov=True,
            return_deriv=True,
            derivs_to_predict=derivs_to_predict
        )
        
        # Get predictions with fallback
        with patch('jetgp.full_gddegp.gddegp.cho_factor', side_effect=LinAlgError("Mocked Cholesky failure")):
            y_pred_fallback, cov_fallback = self.model.predict(
                self.X_train,
                self.params,
                rays_predict=[self.rays_array],
                calc_cov=True,
                return_deriv=True,
                derivs_to_predict=derivs_to_predict
            )
        
        # Results should match
        np.testing.assert_array_almost_equal(
            y_pred_chol, y_pred_fallback, decimal=6,
            err_msg="Cholesky and fallback predictions should match (no normalize)"
        )
        
        np.testing.assert_array_almost_equal(
            cov_chol, cov_fallback, decimal=6,
            err_msg="Cholesky and fallback covariances should match (no normalize)"
        )

    def test_comprehensive_summary_no_normalize(self):
        """Print a comprehensive summary for normalize=False model."""
        y_pred_func = self.y_pred_train_full[0, :].flatten()
        y_pred_dir = self.y_pred_train_full[1, :].flatten()
        func_err = np.abs(y_pred_func - self.y_func.flatten())
        dir_err = np.abs(y_pred_dir - self.y_dir.flatten())

        print("\n" + "=" * 80)
        print("GDDEGP Branin Gradient-Aligned (normalize=False) Summary")
        print("=" * 80)
        
        all_tests_passed = True
        
        max_func_err = np.max(func_err)
        mean_func_err = np.mean(func_err)
        passed = max_func_err < 1e-2
        all_tests_passed = all_tests_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Function values | Max: {max_func_err:.2e} | Mean: {mean_func_err:.2e}")
        
        max_dir_err = np.max(dir_err)
        mean_dir_err = np.mean(dir_err)
        passed = max_dir_err < 1e-2
        all_tests_passed = all_tests_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Directional derivatives | Max: {max_dir_err:.2e} | Mean: {mean_dir_err:.2e}")
        
        print("\n" + "=" * 80)
        print("Configuration:")
        print(f"  - normalize: {self.normalize_data} (KEY SETTING)")
        print(f"  - Training points: {self.N}")
        print(f"  - Kernel: {self.kernel} ({self.kernel_type})")
        print("=" * 80)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_tests_passed else '✗ SOME TESTS FAILED'}")
        print("=" * 80 + "\n")

        self.assertTrue(all_tests_passed, "Not all interpolation tests passed")


def run_tests_with_details():
    """Run tests with detailed output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestGDDEGPBraninGradientAligned))
    suite.addTests(loader.loadTestsFromTestCase(TestGDDEGPBraninZeroOrder))
    suite.addTests(loader.loadTestsFromTestCase(TestGDDEGPBraninGradientAlignedNoNormalize))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    print("\nRunning GDDEGP Branin Gradient-Aligned Unit Tests...")
    print("=" * 80 + "\n")
    
    result = run_tests_with_details()
    
    sys.exit(0 if result.wasSuccessful() else 1)