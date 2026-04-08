"""
Unit test for the Sparse Directional-Derivative Enhanced Gaussian Process (DDEGP)
on the 2D Branin function using a global set of directional rays.

This test verifies that the sparse DDEGP model correctly interpolates:
- Function values at training points
- Directional derivatives along specified rays (45°, 90°, 135°)

Uses sparse Cholesky (Vecchia approximation) for the kernel inverse.
"""

import sys
import unittest
import numpy as np
import sympy as sp
from jetgp.full_ddegp_sparse.ddegp import ddegp
from scipy.stats import qmc
import jetgp.utils


class TestSparseGlobalDirectionalBranin(unittest.TestCase):
    """Test case for sparse DDEGP with global directional rays on the Branin function."""

    @classmethod
    def setUpClass(cls):
        """Setup global configuration and training data for the sparse DDEGP test."""
        # --- Step 1: Configuration ---
        cls.n_order = 1
        cls.n_bases = 2
        cls.num_training_pts = 16
        cls.domain_bounds = ((-5.0, 10.0), (0.0, 15.0))

        # Global directional rays (45°, 90°, 135°)
        cls.rays = np.array([
            [np.cos(np.pi/4), np.cos(np.pi/2), np.cos(3*np.pi/4)],
            [np.sin(np.pi/4), np.sin(np.pi/2), np.sin(3*np.pi/4)]
        ])
        cls.num_rays = cls.rays.shape[1]

        cls.normalize_data = True
        cls.kernel = "RQ"
        cls.kernel_type = "isotropic"
        cls.random_seed = 1
        np.random.seed(cls.random_seed)

        # Sparse Cholesky parameters
        cls.rho = 3.0
        cls.use_supernodes = True

        # --- Step 2: Branin function setup ---
        x1_sym, x2_sym = sp.symbols("x1 x2", real=True)
        a, b, c, r, s, t = (
            1.0,
            5.1 / (4.0 * sp.pi**2),
            5.0 / sp.pi,
            6.0,
            10.0,
            1.0 / (8.0 * sp.pi),
        )
        f_sym = a * (x2_sym - b * x1_sym**2 + c * x1_sym - r) ** 2 + \
                s * (1 - t) * sp.cos(x1_sym) + s
        grad_x1 = sp.diff(f_sym, x1_sym)
        grad_x2 = sp.diff(f_sym, x2_sym)

        # Convert to fast NumPy-compatible functions
        f_func_raw = sp.lambdify([x1_sym, x2_sym], f_sym, "numpy")
        grad_x1_func_raw = sp.lambdify([x1_sym, x2_sym], grad_x1, "numpy")
        grad_x2_func_raw = sp.lambdify([x1_sym, x2_sym], grad_x2, "numpy")

        # Wrap to handle constants
        def make_array_func(func_raw):
            def wrapped(x1, x2):
                result = func_raw(x1, x2)
                result = np.atleast_1d(result)
                if result.size == 1 and np.atleast_1d(x1).size > 1:
                    result = np.full_like(x1, result[0])
                return result
            return wrapped

        cls.f_func = make_array_func(f_func_raw)
        cls.grad_x1_func = make_array_func(grad_x1_func_raw)
        cls.grad_x2_func = make_array_func(grad_x2_func_raw)

        # --- Step 3: Training data ---
        sampler = qmc.LatinHypercube(d=cls.n_bases, seed=cls.random_seed)
        unit_samples = sampler.random(n=cls.num_training_pts)
        cls.X_train = qmc.scale(
            unit_samples,
            [b[0] for b in cls.domain_bounds],
            [b[1] for b in cls.domain_bounds],
        )

        # Compute function and gradient values
        y_func = cls.f_func(cls.X_train[:, 0], cls.X_train[:, 1]).reshape(-1, 1)
        grad_x1_vals = cls.grad_x1_func(cls.X_train[:, 0], cls.X_train[:, 1]).reshape(-1, 1)
        grad_x2_vals = cls.grad_x2_func(cls.X_train[:, 0], cls.X_train[:, 1]).reshape(-1, 1)

        # Store for verification
        cls.grad_x1_vals = grad_x1_vals
        cls.grad_x2_vals = grad_x2_vals

        # Directional derivatives for each ray
        directional_derivs = []
        for i in range(cls.rays.shape[1]):
            ray = cls.rays[:, i]
            d_ray = grad_x1_vals * ray[0] + grad_x2_vals * ray[1]
            directional_derivs.append(d_ray)

        cls.y_train_list = [y_func] + directional_derivs
        cls.der_indices = [[[[1, 1]], [[2, 1]], [[3, 1]]]]
        cls.derivative_locations = []
        for i in range(len(cls.der_indices)):
            for j in range(len(cls.der_indices[i])):
                cls.derivative_locations.append([i for i in range(len(cls.X_train))])

        # --- Step 4: Initialize and train sparse model ---
        cls.model = ddegp(
            cls.X_train,
            cls.y_train_list,
            n_order=cls.n_order,
            der_indices=cls.der_indices,
            rays=cls.rays,
            derivative_locations=cls.derivative_locations,
            normalize=cls.normalize_data,
            kernel=cls.kernel,
            kernel_type=cls.kernel_type,
            rho=cls.rho,
            use_supernodes=cls.use_supernodes,
        )

        cls.params = cls.model.optimize_hyperparameters(
            optimizer="powell",
            n_restart_optimizer=10
        )

    # ------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------
    def test_configuration(self):
        """Verify model configuration is correct."""
        self.assertEqual(self.n_order, 1, "Should use 1st order derivatives")
        self.assertEqual(self.n_bases, 2, "Should be 2D problem")
        self.assertEqual(self.num_rays, 3, "Should have 3 directional rays")

        # Check ray directions
        expected_angles = [np.pi/4, np.pi/2, 3*np.pi/4]
        for i, angle in enumerate(expected_angles):
            expected_ray = np.array([np.cos(angle), np.sin(angle)])
            actual_ray = self.rays[:, i]
            np.testing.assert_array_almost_equal(
                actual_ray, expected_ray, decimal=10,
                err_msg=f"Ray {i} direction incorrect"
            )

    def test_shapes(self):
        """Ensure training data dimensions are consistent."""
        self.assertEqual(self.X_train.shape, (self.num_training_pts, 2),
                        "Training points should be 16x2")
        # 1 function + 3 directional derivatives
        self.assertEqual(len(self.y_train_list), 4,
                        "Should have function + 3 directional derivatives")
        for i, y in enumerate(self.y_train_list):
            self.assertEqual(y.shape, (self.num_training_pts, 1),
                           f"Training data {i} should have shape (16, 1)")

    def test_function_interpolation(self):
        """Verify function interpolation at training points."""
        y_true = self.y_train_list[0]
        y_pred, _ = self.model.predict(
            self.X_train, self.params, calc_cov=True, return_deriv=False
        )

        abs_err = np.abs(y_pred.flatten() - y_true.flatten())
        max_err = np.max(abs_err)
        mean_err = np.mean(abs_err)

        self.assertLess(max_err, 1e-4,
                       f"Function interpolation max error too large: {max_err}")
        self.assertLess(mean_err, 1e-5,
                       f"Function interpolation mean error too large: {mean_err}")

    def test_directional_derivative_interpolation(self):
        """Verify interpolation of directional derivatives along each global ray."""
        derivs_to_predict = [[[1,1]], [[2,1]], [[3,1]]]
        y_pred_all = self.model.predict(
            self.X_train, self.params, calc_cov=False, return_deriv=True, derivs_to_predict=derivs_to_predict
        )

        # Partition prediction vector: f + 3 rays
        y_pred_func = y_pred_all[0,:]
        y_pred_rays = [y_pred_all[(i + 1), :]
                       for i in range(self.num_rays)]

        # Analytic values
        y_true_rays = self.y_train_list[1:]

        for i, (y_pred, y_true) in enumerate(zip(y_pred_rays, y_true_rays)):
            abs_err = np.abs(y_pred.flatten() - y_true.flatten())
            mean_err = np.mean(abs_err)
            max_err = np.max(abs_err)

            self.assertLess(max_err, 1e-4,
                           f"Ray {i+1} (angle={np.degrees(np.arctan2(self.rays[1,i], self.rays[0,i])):.1f}) "
                           f"max error too large: {max_err}")
            self.assertLess(mean_err, 1e-5,
                           f"Ray {i+1} mean error too large: {mean_err}")

    def test_ray_orthogonality(self):
        """Verify that rays have unit length (approximately)."""
        for i in range(self.num_rays):
            ray = self.rays[:, i]
            ray_norm = np.linalg.norm(ray)
            self.assertAlmostEqual(ray_norm, 1.0, places=10,
                                  msg=f"Ray {i} should have unit length")

    def test_comprehensive_summary(self):
        """Comprehensive test with detailed summary."""
        print("\n" + "="*80)
        print("Sparse DDEGP Global Directional Rays (Branin Function) Test Summary")
        print("="*80)

        all_tests_passed = True

        # Test function values
        print("\nFunction Value Interpolation")
        print("-" * 80)
        y_true = self.y_train_list[0]
        y_pred = self.model.predict(
            self.X_train, self.params, calc_cov=False, return_deriv=False
        )
        abs_err = np.abs(y_pred.flatten() - y_true.flatten())
        max_err = np.max(abs_err)
        mean_err = np.mean(abs_err)

        passed = max_err < 1e-4
        all_tests_passed = all_tests_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"{status} | Function values | Max: {max_err:.2e} | Mean: {mean_err:.2e}")

        # Test directional derivatives
        print("\nDirectional Derivative Interpolation")
        print("-" * 80)
        y_pred_all = self.model.predict(
            self.X_train, self.params, calc_cov=False, return_deriv=True
        )

        y_pred_rays = [y_pred_all[(i + 1), :]
                       for i in range(self.num_rays)]
        y_true_rays = self.y_train_list[1:]

        ray_angles = [45, 90, 135]  # degrees
        for i, (y_pred, y_true, angle) in enumerate(zip(y_pred_rays, y_true_rays, ray_angles)):
            abs_err = np.abs(y_pred.flatten() - y_true.flatten())
            max_err = np.max(abs_err)
            mean_err = np.mean(abs_err)

            passed = max_err < 1e-4
            all_tests_passed = all_tests_passed and passed
            status = "PASS" if passed else "FAIL"
            print(f"{status} | Ray {i+1} ({angle:3d} deg) | Max: {max_err:.2e} | Mean: {mean_err:.2e}")

        print("\n" + "="*80)
        print("Configuration:")
        print(f"  - Function: 2D Branin benchmark function")
        print(f"  - Training points: {self.num_training_pts} (Latin Hypercube)")
        print(f"  - Domain: x1 in [{self.domain_bounds[0][0]}, {self.domain_bounds[0][1]}], "
              f"x2 in [{self.domain_bounds[1][0]}, {self.domain_bounds[1][1]}]")
        print(f"  - Directional rays: {self.num_rays} (45 deg, 90 deg, 135 deg)")
        print(f"  - Derivative order: {self.n_order} (directional derivatives)")
        print(f"  - Kernel: {self.kernel} ({self.kernel_type})")
        print(f"  - Sparse Cholesky: rho={self.rho}, use_supernodes={self.use_supernodes}")
        print(f"  - Derivatives: computed symbolically with SymPy")
        print("="*80)
        print(f"Overall: {'ALL TESTS PASSED' if all_tests_passed else 'SOME TESTS FAILED'}")
        print("="*80 + "\n")

        self.assertTrue(all_tests_passed, "Not all interpolation tests passed")


def run_tests_with_details():
    """Run tests with detailed output."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSparseGlobalDirectionalBranin)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    print("\nRunning Sparse DDEGP Global Directional Rays Interpolation Unit Tests...")
    print("="*80 + "\n")

    result = run_tests_with_details()

    sys.exit(0 if result.wasSuccessful() else 1)
