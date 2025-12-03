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


def run_tests_with_details():
    """Run tests with detailed output."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGDDEGPBraninGradientAligned)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    print("\nRunning GDDEGP Branin Gradient-Aligned Unit Tests...")
    print("=" * 80 + "\n")
    
    result = run_tests_with_details()
    
    sys.exit(0 if result.wasSuccessful() else 1)