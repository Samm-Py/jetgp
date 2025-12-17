"""
Unit tests for the Directional-Derivative Enhanced Gaussian Process (DDEGP)
to achieve full code coverage.

These tests target specific uncovered code paths:
- normalize=False initialization path
- Cholesky decomposition fallback path
- Covariance transformation with derivatives
- Covariance without normalization
- calc_cov=True and return_deriv=False edge case in rbf_kernel_predictions
- n_order=0 edge case in differences_by_dim_func
"""

import sys
import unittest
import numpy as np
import sympy as sp
from unittest.mock import patch
from jetgp.full_ddegp.ddegp import ddegp
from jetgp.full_ddegp import ddegp_utils
from scipy.stats import qmc
from jetgp.kernel_funcs.kernel_funcs import KernelFactory, get_oti_module

class TestDDEGPNormalizeOff(unittest.TestCase):
    """Test case for DDEGP with normalize=False (covers lines 69-70)."""
    
    @classmethod
    def setUpClass(cls):
        """Setup configuration and training data for normalize=False test."""
        # --- Configuration ---
        cls.n_order = 1
        cls.n_bases = 2
        cls.num_training_pts = 10
        cls.domain_bounds = ((-5.0, 10.0), (0.0, 15.0))
        
        # Simple directional rays
        cls.rays = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ]).T
        
        cls.random_seed = 42
        np.random.seed(cls.random_seed)
        
        # --- Simple quadratic function for testing ---
        x1_sym, x2_sym = sp.symbols("x1 x2", real=True)
        f_sym = x1_sym**2 + x2_sym**2
        grad_x1 = sp.diff(f_sym, x1_sym)  # 2*x1
        grad_x2 = sp.diff(f_sym, x2_sym)  # 2*x2
        
        cls.f_func = sp.lambdify([x1_sym, x2_sym], f_sym, "numpy")
        cls.grad_x1_func = sp.lambdify([x1_sym, x2_sym], grad_x1, "numpy")
        cls.grad_x2_func = sp.lambdify([x1_sym, x2_sym], grad_x2, "numpy")
        
        # --- Training data ---
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
        
        cls.y_train_list = [y_func, grad_x1_vals, grad_x2_vals]
        cls.der_indices = [[[[1, 1]], [[2, 1]]]]
        cls.derivative_locations = [
            list(range(cls.num_training_pts)),
            list(range(cls.num_training_pts))
        ]
        
        # --- Initialize model with normalize=False ---
        cls.model = ddegp(
            cls.X_train,
            cls.y_train_list,
            n_order=cls.n_order,
            der_indices=cls.der_indices,
            rays=cls.rays,
            derivative_locations=cls.derivative_locations,
            normalize=False,  # KEY: This triggers lines 69-70
            kernel="SE",
            kernel_type="isotropic",
        )
        
        cls.params = cls.model.optimize_hyperparameters(
            optimizer="powell",
            n_restart_optimizer=5
        )
    
    def test_normalize_false_initialization(self):
        """Verify model initializes correctly with normalize=False."""
        self.assertFalse(self.model.normalize, 
                        "Model should have normalize=False")
        
        # x_train should be unchanged (not normalized)
        np.testing.assert_array_almost_equal(
            self.model.x_train, self.X_train, decimal=10,
            err_msg="x_train should be unchanged when normalize=False"
        )
    
    def test_prediction_without_normalization(self):
        """Verify predictions work correctly without normalization."""
        y_pred = self.model.predict(
            self.X_train, self.params, calc_cov=False, return_deriv=False
        )
        
        y_true = self.y_train_list[0]
        abs_err = np.abs(y_pred.flatten() - y_true.flatten())
        max_err = np.max(abs_err)
        
        self.assertLess(max_err, 1e-3,
                       f"Function interpolation error too large: {max_err}")
    
    def test_covariance_without_normalization(self):
        """Test covariance calculation with normalize=False (covers line 234)."""
        y_pred, y_var = self.model.predict(
            self.X_train, self.params, calc_cov=True, return_deriv=False
        )
        
        self.assertIsNotNone(y_var, "Variance should be computed")
        self.assertEqual(y_var.shape[1], self.num_training_pts,
                        "Variance should have correct shape")
        
        # Variance should be non-negative
        self.assertTrue(np.all(y_var >= 0),
                       "Variance should be non-negative")


class TestDDEGPCholeskyFallback(unittest.TestCase):
    """Test case for Cholesky decomposition fallback (covers lines 170-173, 216)."""
    
    @classmethod
    def setUpClass(cls):
        """Setup configuration for Cholesky fallback test."""
        cls.n_order = 1
        cls.n_bases = 2
        cls.num_training_pts = 8
        cls.domain_bounds = ((0.0, 1.0), (0.0, 1.0))
        
        cls.rays = np.array([[1.0], [0.0]])
        
        np.random.seed(123)
        
        # Simple linear function
        cls.X_train = np.random.rand(cls.num_training_pts, 2)
        y_func = (cls.X_train[:, 0] + cls.X_train[:, 1]).reshape(-1, 1)
        y_deriv = np.ones((cls.num_training_pts, 1))  # derivative is 1
        
        cls.y_train_list = [y_func, y_deriv]
        cls.der_indices = [[[[1, 1]]]]
        cls.derivative_locations = [list(range(cls.num_training_pts))]
        
        cls.model = ddegp(
            cls.X_train,
            cls.y_train_list,
            n_order=cls.n_order,
            der_indices=cls.der_indices,
            rays=cls.rays,
            derivative_locations=cls.derivative_locations,
            normalize=True,
            kernel="SE",
            kernel_type="isotropic",
        )
        
        cls.params = cls.model.optimize_hyperparameters(
            optimizer="powell",
            n_restart_optimizer=3
        )
    
    def test_cholesky_fallback_mean_only(self):
        """Test prediction when Cholesky fails (mean only, covers lines 170-173)."""
        X_test = np.array([[0.5, 0.5]])
        
        with patch('jetgp.full_ddegp.ddegp.cho_factor') as mock_cho_factor:
            mock_cho_factor.side_effect = np.linalg.LinAlgError(
                "Matrix not positive definite"
            )
            
            # Should fall back to np.linalg.solve without crashing
            y_pred = self.model.predict(
                X_test, self.params, calc_cov=False, return_deriv=False
            )
            
            self.assertIsNotNone(y_pred, 
                               "Prediction should succeed even if Cholesky fails")
    
    def test_cholesky_fallback_with_covariance(self):
        """Test prediction with covariance when Cholesky fails (covers line 216)."""
        X_test = np.array([[0.5, 0.5]])
        
        with patch('jetgp.full_ddegp.ddegp.cho_factor') as mock_cho_factor:
            mock_cho_factor.side_effect = np.linalg.LinAlgError(
                "Matrix not positive definite"
            )
            
            # Should compute covariance using np.linalg.inv fallback
            y_pred, y_var = self.model.predict(
                X_test, self.params, calc_cov=True, return_deriv=False
            )
            
            self.assertIsNotNone(y_pred, "Mean prediction should succeed")
            self.assertIsNotNone(y_var, "Variance should be computed via fallback")


class TestDDEGPCovarianceWithDerivatives(unittest.TestCase):
    """Test case for covariance with derivatives (covers lines 228-230)."""
    
    @classmethod
    def setUpClass(cls):
        """Setup for covariance with derivatives test."""
        cls.n_order = 1
        cls.n_bases = 2
        cls.num_training_pts = 12
        cls.domain_bounds = ((-2.0, 2.0), (-2.0, 2.0))
        
        # Two directional rays
        cls.rays = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ]).T
        
        np.random.seed(456)
        
        # Quadratic function: f = x1^2 + x2^2
        x1_sym, x2_sym = sp.symbols("x1 x2", real=True)
        f_sym = x1_sym**2 + x2_sym**2
        
        cls.f_func = sp.lambdify([x1_sym, x2_sym], f_sym, "numpy")
        cls.grad_x1_func = sp.lambdify([x1_sym, x2_sym], sp.diff(f_sym, x1_sym), "numpy")
        cls.grad_x2_func = sp.lambdify([x1_sym, x2_sym], sp.diff(f_sym, x2_sym), "numpy")
        
        sampler = qmc.LatinHypercube(d=cls.n_bases, seed=456)
        unit_samples = sampler.random(n=cls.num_training_pts)
        cls.X_train = qmc.scale(
            unit_samples,
            [b[0] for b in cls.domain_bounds],
            [b[1] for b in cls.domain_bounds],
        )
        
        y_func = cls.f_func(cls.X_train[:, 0], cls.X_train[:, 1]).reshape(-1, 1)
        grad_x1_vals = cls.grad_x1_func(cls.X_train[:, 0], cls.X_train[:, 1]).reshape(-1, 1)
        grad_x2_vals = cls.grad_x2_func(cls.X_train[:, 0], cls.X_train[:, 1]).reshape(-1, 1)
        
        cls.y_train_list = [y_func, grad_x1_vals, grad_x2_vals]
        cls.der_indices = [[[[1, 1]], [[2, 1]]]]
        cls.derivative_locations = [
            list(range(cls.num_training_pts)),
            list(range(cls.num_training_pts))
        ]
        
        cls.model = ddegp(
            cls.X_train,
            cls.y_train_list,
            n_order=cls.n_order,
            der_indices=cls.der_indices,
            rays=cls.rays,
            derivative_locations=cls.derivative_locations,
            normalize=True,  # KEY: normalize=True for transform_cov_directional
            kernel="SE",
            kernel_type="isotropic",
        )
        
        cls.params = cls.model.optimize_hyperparameters(
            optimizer="powell",
            n_restart_optimizer=5
        )
    
    def test_covariance_with_derivatives_normalized(self):
        """Test covariance with return_deriv=True and normalize=True (covers lines 228-230)."""
        X_test = self.X_train[:3]  # Use subset of training points
        
        derivs_to_predict = [[[1, 1]], [[2, 1]]]
        y_pred, y_var = self.model.predict(
            X_test, self.params, 
            calc_cov=True, 
            return_deriv=True,
            derivs_to_predict=derivs_to_predict
        )
        
        self.assertIsNotNone(y_pred, "Mean with derivatives should be computed")
        self.assertIsNotNone(y_var, "Variance with derivatives should be computed")
        
        # Check shapes: should include function + derivatives
        n_test = X_test.shape[0]
        n_derivs = len(derivs_to_predict)
        expected_rows = 1 + n_derivs  # function + derivatives
        
        self.assertEqual(y_pred.shape[0], expected_rows,
                        f"Prediction should have {expected_rows} rows (f + {n_derivs} derivs)")
        self.assertEqual(y_pred.shape[1], n_test,
                        f"Prediction should have {n_test} columns")


class TestDDEGPUtilsCalcCovOnly(unittest.TestCase):
    """Test case for calc_cov=True and return_deriv=False edge case."""
    
    def test_rbf_kernel_predictions_calc_cov_no_deriv(self):
        """Test rbf_kernel_predictions when calc_cov=True and return_deriv=False."""
        n_test = 5
        n_train = 5
        
        np.random.seed(202)
        
        # Create mock phi object with .real and .shape attributes
        class MockPhi:
            def __init__(self, data):
                self.real = data
                self.shape = data.shape
        
        # Create mock phi_exp (not used in this path, but needed for signature)
        phi_data = np.random.randn(n_test, n_train)
        phi = MockPhi(phi_data)
        phi_exp = np.random.randn(10, n_test * n_train)  # Placeholder
        
        n_order = 1
        n_bases = 2
        der_indices = [[[1, 1]]]
        powers = [0, 1]
        return_deriv = False
        index = [list(range(n_train))]
        common_derivs = []
        calc_cov = True  # KEY: calc_cov=True and return_deriv=False triggers early return
        
        result = ddegp_utils.rbf_kernel_predictions(
            phi=phi,
            phi_exp=phi_exp,
            n_order=n_order,
            n_bases=n_bases,
            der_indices=der_indices,
            powers=powers,
            return_deriv=return_deriv,
            index=index,
            common_derivs=common_derivs,
            calc_cov=calc_cov,
            powers_predict=None
        )
        
        self.assertIsNotNone(result, "Should return kernel matrix for calc_cov=True, return_deriv=False")
        self.assertEqual(result.shape, (n_test, n_train),
                        "Result should have shape (n_test, n_train)")
        np.testing.assert_array_almost_equal(
            result, phi_data, decimal=10,
            err_msg="Result should equal phi.real for early return path"
        )


class TestDDEGPUtilsDifferencesByDim(unittest.TestCase):
    """Test cases for differences_by_dim_func edge cases."""
    
    def test_differences_by_dim_n_order_zero(self):
        """Test differences_by_dim_func when n_order=0 (no hypercomplex perturbation)."""
        n1 = 5
        n2 = 4
        d = 2
        
        np.random.seed(303)
        X1 = np.random.randn(n1, d)
        X2 = np.random.randn(n2, d)
        
        # Simple rays (identity directions)
        rays = np.eye(d)
        
        n_order = 0  # KEY: n_order=0 triggers the first branch
        oti_module = get_oti_module(d, n_order)
        result = ddegp_utils.differences_by_dim_func(
            X1, X2, rays, n_order,oti_module=oti_module, return_deriv=True
        )
        
        # Verify result structure
        self.assertEqual(len(result), d, f"Should return {d} difference arrays")
        
        for k in range(d):
            self.assertEqual(result[k].shape, (n1, n2),
                           f"Dimension {k} differences should have shape ({n1}, {n2})")
            
            # For n_order=0, result should be pure real differences
            # Check that the real part matches expected differences
            expected_diffs = X1[:, k].reshape(-1, 1) - X2[:, k].reshape(1, -1)
            np.testing.assert_array_almost_equal(
                result[k].real, expected_diffs, decimal=10,
                err_msg=f"Dimension {k} differences should match expected values for n_order=0"
            )
    



class TestDDEGPCoverageSummary(unittest.TestCase):
    """Comprehensive test with coverage summary."""
    
    @classmethod
    def setUpClass(cls):
        """Setup for comprehensive coverage test."""
        cls.n_order = 1
        cls.n_bases = 2
        cls.num_training_pts = 10
        cls.domain_bounds = ((-1.0, 1.0), (-1.0, 1.0))
        
        cls.rays = np.array([[1.0, 0.0], [0.0, 1.0]]).T
        
        np.random.seed(999)
        
        # Simple function: f = x1 + x2
        cls.X_train = np.random.rand(cls.num_training_pts, 2) * 2 - 1
        y_func = (cls.X_train[:, 0] + cls.X_train[:, 1]).reshape(-1, 1)
        grad_x1 = np.ones((cls.num_training_pts, 1))
        grad_x2 = np.ones((cls.num_training_pts, 1))
        
        cls.y_train_list = [y_func, grad_x1, grad_x2]
        cls.der_indices = [[[[1, 1]], [[2, 1]]]]
        cls.derivative_locations = [
            list(range(cls.num_training_pts)),
            list(range(cls.num_training_pts))
        ]
        
        # Create both normalized and non-normalized models
        cls.model_normalized = ddegp(
            cls.X_train.copy(),
            [y.copy() for y in cls.y_train_list],
            n_order=cls.n_order,
            der_indices=cls.der_indices,
            rays=cls.rays.copy(),
            derivative_locations=cls.derivative_locations,
            normalize=True,
            kernel="SE",
            kernel_type="isotropic",
        )
        
        cls.model_unnormalized = ddegp(
            cls.X_train.copy(),
            [y.copy() for y in cls.y_train_list],
            n_order=cls.n_order,
            der_indices=cls.der_indices,
            rays=cls.rays.copy(),
            derivative_locations=cls.derivative_locations,
            normalize=False,
            kernel="SE",
            kernel_type="isotropic",
        )
        
        cls.params_norm = cls.model_normalized.optimize_hyperparameters(
            optimizer="powell", n_restart_optimizer=3
        )
        cls.params_unnorm = cls.model_unnormalized.optimize_hyperparameters(
            optimizer="powell", n_restart_optimizer=3
        )
    
    def test_comprehensive_coverage_summary(self):
        """Run comprehensive tests and print coverage summary."""
        print("\n" + "=" * 80)
        print("DDEGP Coverage Test Summary")
        print("=" * 80)
        
        all_tests_passed = True
        X_test = np.array([[0.0, 0.0], [0.5, 0.5]])
        
        # Test 1: normalize=False path (lines 69-70)
        print("\n1. Normalize=False Initialization (lines 69-70)")
        print("-" * 80)
        try:
            self.assertFalse(self.model_unnormalized.normalize)
            np.testing.assert_array_almost_equal(
                self.model_unnormalized.x_train, self.X_train, decimal=10
            )
            print("✓ PASS | normalize=False initialization works correctly")
        except AssertionError as e:
            print(f"✗ FAIL | {e}")
            all_tests_passed = False
        
        # Test 2: Covariance without normalization (line 234)
        print("\n2. Covariance Without Normalization (line 234)")
        print("-" * 80)
        try:
            _, y_var = self.model_unnormalized.predict(
                X_test, self.params_unnorm, calc_cov=True, return_deriv=False
            )
            self.assertIsNotNone(y_var)
            self.assertTrue(np.all(y_var >= 0))
            print("✓ PASS | Covariance computed correctly without normalization")
        except AssertionError as e:
            print(f"✗ FAIL | {e}")
            all_tests_passed = False
        
        # Test 3: Covariance with derivatives, normalized (lines 228-230)
        print("\n3. Covariance with Derivatives, Normalized (lines 228-230)")
        print("-" * 80)
        try:
            derivs_to_predict = [[[1, 1]], [[2, 1]]]
            _, y_var = self.model_normalized.predict(
                X_test, self.params_norm, 
                calc_cov=True, return_deriv=True,
                derivs_to_predict=derivs_to_predict
            )
            self.assertIsNotNone(y_var)
            print("✓ PASS | Covariance with derivatives computed correctly")
        except AssertionError as e:
            print(f"✗ FAIL | {e}")
            all_tests_passed = False
        
        # Test 4: Cholesky fallback (lines 170-173, 216)
        print("\n4. Cholesky Fallback Path (lines 170-173, 216)")
        print("-" * 80)
        try:
            with patch('jetgp.full_ddegp.ddegp.cho_factor') as mock:
                mock.side_effect = np.linalg.LinAlgError("Test failure")
                y_pred, y_var = self.model_normalized.predict(
                    X_test, self.params_norm, calc_cov=True, return_deriv=False
                )
                self.assertIsNotNone(y_pred)
                self.assertIsNotNone(y_var)
            print("✓ PASS | Cholesky fallback works correctly")
        except AssertionError as e:
            print(f"✗ FAIL | {e}")
            all_tests_passed = False
        
        # Test 5: differences_by_dim_func with n_order=0
        print("\n5. differences_by_dim_func with n_order=0")
        print("-" * 80)
        try:
            X1 = np.random.randn(5, 2)
            X2 = np.random.randn(4, 2)
            rays = np.eye(2)
            oti_module = get_oti_module(2, 0)
            result = ddegp_utils.differences_by_dim_func(X1, X2, rays, n_order=0, oti_module=oti_module)
            self.assertEqual(len(result), 2)
            print("✓ PASS | differences_by_dim_func works with n_order=0")
        except AssertionError as e:
            print(f"✗ FAIL | {e}")
            all_tests_passed = False
        
        # Test 6: calc_cov=True and return_deriv=False early return
        print("\n6. rbf_kernel_predictions early return (calc_cov=True, return_deriv=False)")
        print("-" * 80)
        try:
            class MockPhi:
                def __init__(self, data):
                    self.real = data
                    self.shape = data.shape
            
            phi_data = np.random.randn(5, 5)
            phi = MockPhi(phi_data)
            phi_exp = np.random.randn(10, 25)
            
            result = ddegp_utils.rbf_kernel_predictions(
                phi=phi, phi_exp=phi_exp, n_order=1, n_bases=2,
                der_indices=[[[1, 1]]], powers=[0, 1],
                return_deriv=False, index=[list(range(5))],
                common_derivs=[], calc_cov=True, powers_predict=None
            )
            np.testing.assert_array_almost_equal(result, phi_data, decimal=10)
            print("✓ PASS | rbf_kernel_predictions early return works correctly")
        except AssertionError as e:
            print(f"✗ FAIL | {e}")
            all_tests_passed = False
        
        print("\n" + "=" * 80)
        print("Coverage Targets:")
        print("  - Lines 69-70:   normalize=False initialization")
        print("  - Lines 170-173: Cholesky decomposition fallback")
        print("  - Line 216:      Covariance with Cholesky fallback")
        print("  - Lines 228-230: transform_cov_directional")
        print("  - Line 234:      Covariance without normalization")
        print("  - ddegp_utils:   n_order=0 branch in differences_by_dim_func")
        print("  - ddegp_utils:   calc_cov=True, return_deriv=False early return")
        print("=" * 80)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_tests_passed else '✗ SOME TESTS FAILED'}")
        print("=" * 80 + "\n")
        
        self.assertTrue(all_tests_passed, "Not all coverage tests passed")


def run_tests_with_details():
    """Run tests with detailed output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDDEGPNormalizeOff))
    suite.addTests(loader.loadTestsFromTestCase(TestDDEGPCholeskyFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestDDEGPCovarianceWithDerivatives))
    suite.addTests(loader.loadTestsFromTestCase(TestDDEGPUtilsCalcCovOnly))
    suite.addTests(loader.loadTestsFromTestCase(TestDDEGPUtilsDifferencesByDim))
    suite.addTests(loader.loadTestsFromTestCase(TestDDEGPCoverageSummary))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    print("\nRunning DDEGP Coverage Unit Tests...")
    print("=" * 80 + "\n")
    
    result = run_tests_with_details()
    
    sys.exit(0 if result.wasSuccessful() else 1)