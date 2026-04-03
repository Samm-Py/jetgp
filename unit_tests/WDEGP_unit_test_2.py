"""
Unit test for 2D Weighted DEGP with heterogeneous derivative orders.

This test verifies that the WDEGP model correctly interpolates training data
across three submodels with different derivative orders:
- Submodel 1 (Corners): Function values only (no derivatives)
- Submodel 2 (Edges): Function values + 1st order derivatives
- Submodel 3 (Center): Function values + up to 2nd order derivatives

Test function: Six-hump camel function

Derivatives are verified by predicting them directly from the GP
(return_deriv=True with derivs_to_predict) and comparing against
the analytic values computed via SymPy.

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
            # Edges (1st order)
            [[1, 2, 4, 8, 7, 11, 13, 14], [1, 2, 4, 8, 7, 11, 13, 14]],
            [[5, 6, 9, 10], [5, 6, 9, 10], [5, 6, 9, 10], [
                5, 6, 9, 10], [5, 6, 9, 10]]  # Center (2nd order)
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
            pop_size=200,
            n_generations=15,
            local_opt_every=15,
            debug=True
        )
        print(f"\nOptimized parameters: {cls.params}")

        # Store analytic derivative functions for test-time evaluation
        cls._setup_analytic_functions()

    @classmethod
    def _generate_training_points(cls):
        """Generate 4x4 grid of training points."""
        x_vals = np.linspace(cls.lb_x, cls.ub_x, cls.points_per_axis)
        y_vals = np.linspace(cls.lb_y, cls.ub_y, cls.points_per_axis)
        return np.array(list(itertools.product(x_vals, y_vals)))

    @classmethod
    def _six_hump_camel_symbolic(cls):
        """Define six-hump camel function symbolically."""
        x1, x2 = sp.symbols('x1 x2', real=True)
        f = ((4 - 2.1*x1**2 + x1**4/3) * x1 **
             2 + x1*x2 + (-4 + 4*x2**2) * x2**2)
        return x1, x2, f

    @classmethod
    def _setup_analytic_functions(cls):
        """Set up lambdified analytic derivative functions."""
        x1_sym, x2_sym, f_sym = cls._six_hump_camel_symbolic()

        df_dx1 = sp.diff(f_sym, x1_sym)
        df_dx2 = sp.diff(f_sym, x2_sym)
        d2f_dx1_2 = sp.diff(df_dx1, x1_sym)
        d2f_dx1dx2 = sp.diff(df_dx1, x2_sym)
        d2f_dx2_2 = sp.diff(df_dx2, x2_sym)

        def make_array_func(func_raw):
            def wrapped(x1, x2):
                result = func_raw(x1, x2)
                result = np.atleast_1d(result)
                if result.size == 1 and np.atleast_1d(x1).size > 1:
                    result = np.full_like(x1, result[0])
                return result
            return wrapped

        # Store as a dict to avoid Python's descriptor protocol binding 'self'
        cls._analytic = {
            'f': make_array_func(sp.lambdify((x1_sym, x2_sym), f_sym, 'numpy')),
            'dx1': make_array_func(sp.lambdify((x1_sym, x2_sym), df_dx1, 'numpy')),
            'dx2': make_array_func(sp.lambdify((x1_sym, x2_sym), df_dx2, 'numpy')),
            'd2x1': make_array_func(sp.lambdify((x1_sym, x2_sym), d2f_dx1_2, 'numpy')),
            'd2x1x2': make_array_func(sp.lambdify((x1_sym, x2_sym), d2f_dx1dx2, 'numpy')),
            'd2x2': make_array_func(sp.lambdify((x1_sym, x2_sym), d2f_dx2_2, 'numpy')),
        }

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
        def make_array_func(func_raw):
            def wrapped(x1, x2):
                result = func_raw(x1, x2)
                result = np.atleast_1d(result)
                if result.size == 1 and np.atleast_1d(x1).size > 1:
                    result = np.full_like(x1, result[0])
                return result
            return wrapped

        f_func = make_array_func(sp.lambdify((x1_sym, x2_sym), f_sym, 'numpy'))
        df_dx1_func = make_array_func(sp.lambdify((x1_sym, x2_sym), df_dx1, 'numpy'))
        df_dx2_func = make_array_func(sp.lambdify((x1_sym, x2_sym), df_dx2, 'numpy'))
        d2f_dx1_2_func = make_array_func(sp.lambdify((x1_sym, x2_sym), d2f_dx1_2, 'numpy'))
        d2f_dx1dx2_func = make_array_func(sp.lambdify((x1_sym, x2_sym), d2f_dx1dx2, 'numpy'))
        d2f_dx2_2_func = make_array_func(sp.lambdify((x1_sym, x2_sym), d2f_dx2_2, 'numpy'))

        # Define derivative indices for each submodel
        der_indices = [
            [
                [[[1, 1]]]  # df/dx1
            ],  # Submodel 1: 1st order x1 only
            [  # Submodel 2: 1st order
                [[[1, 1]], [[2, 1]]]  # df/dx1, df/dx2
            ],
            [  # Submodel 3: 1st and 2nd order
                [[[1, 1]], [[2, 1]]],  # df/dx1, df/dx2
                # d2f/dx1^2, d2f/dx1dx2, d2f/dx2^2
                [[[1, 2]], [[1, 1], [2, 1]], [[2, 2]]]
            ]
        ]

        # Prepare data for each submodel
        # Function values at ALL training points
        y_all = f_func(cls.X_train[:, 0], cls.X_train[:, 1]).reshape(-1, 1)

        submodel_data = []

        # Submodel 1: Corners
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

    # ------------------------------------------------------------------
    # Function interpolation tests
    # ------------------------------------------------------------------

    def test_submodel1_function_interpolation(self):
        """Test function value interpolation for Submodel 1 (corners)."""
        corners_idx = [0, 3, 12, 15]
        X_corners = self.X_train[corners_idx]

        y_pred, submodel_vals = self.model.predict(
            X_corners, self.params, calc_cov=False, return_submodels=True, return_deriv=False
        )

        y_true = self.submodel_data[0][0][corners_idx].flatten()
        y_pred_submodel = submodel_vals[0][0, :].flatten()

        abs_error = np.abs(y_pred_submodel - y_true)
        max_error = np.max(abs_error)

        self.assertLess(max_error, 1e-5,
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

        self.assertLess(max_error, 1e-5,
                        f"Submodel 2 function interpolation error too large: {max_error}")

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

        self.assertLess(max_error, 1e-4,
                        f"Submodel 3 function interpolation error too large: {max_error}")

    # ------------------------------------------------------------------
    # 1st derivative tests — predict directly from GP
    # ------------------------------------------------------------------

    def test_submodel2_first_derivatives(self):
        """Test 1st derivative predictions for Submodel 2 (edges) using GP predict."""
        edges_idx = self.submodel_indices[1][0]
        X_edges = self.X_train[edges_idx]

        # Predict df/dx1 and df/dx2 directly from the GP
        derivs_to_predict = [[[1, 1]], [[2, 1]]]
        pred, _ = self.model.predict(
            X_edges, self.params, calc_cov=True,
            return_deriv=True, derivs_to_predict=derivs_to_predict
        )
        # pred shape: (3, n_points) -> row 0: f, row 1: df/dx1, row 2: df/dx2

        analytic_dx1 = self._analytic['dx1'](X_edges[:, 0], X_edges[:, 1])
        analytic_dx2 = self._analytic['dx2'](X_edges[:, 0], X_edges[:, 1])

        error_dx1 = np.max(np.abs(pred[1, :] - analytic_dx1))
        error_dx2 = np.max(np.abs(pred[2, :] - analytic_dx2))

        self.assertLess(error_dx1, 5e-2,
                        f"Submodel 2 max |df/dx1| error: {error_dx1}")
        self.assertLess(error_dx2, 5e-2,
                        f"Submodel 2 max |df/dx2| error: {error_dx2}")

    def test_submodel3_first_derivatives(self):
        """Test 1st derivative predictions for Submodel 3 (center) using GP predict."""
        center_idx = self.submodel_indices[2][0]
        X_center = self.X_train[center_idx]

        derivs_to_predict = [[[1, 1]], [[2, 1]]]
        pred, _ = self.model.predict(
            X_center, self.params, calc_cov=True,
            return_deriv=True, derivs_to_predict=derivs_to_predict
        )

        analytic_dx1 = self._analytic['dx1'](X_center[:, 0], X_center[:, 1])
        analytic_dx2 = self._analytic['dx2'](X_center[:, 0], X_center[:, 1])

        error_dx1 = np.max(np.abs(pred[1, :] - analytic_dx1))
        error_dx2 = np.max(np.abs(pred[2, :] - analytic_dx2))

        self.assertLess(error_dx1, 5e-2,
                        f"Submodel 3 max |df/dx1| error: {error_dx1}")
        self.assertLess(error_dx2, 5e-2,
                        f"Submodel 3 max |df/dx2| error: {error_dx2}")

    # ------------------------------------------------------------------
    # 2nd derivative tests — predict directly from GP
    # ------------------------------------------------------------------

    def test_submodel3_second_derivatives(self):
        """Test 2nd derivative predictions for Submodel 3 (center) using GP predict."""
        center_idx = self.submodel_indices[2][0]
        X_center = self.X_train[center_idx]

        derivs_to_predict = [[[1, 2]], [[1, 1], [2, 1]], [[2, 2]]]
        pred, _ = self.model.predict(
            X_center, self.params, calc_cov=True,
            return_deriv=True, derivs_to_predict=derivs_to_predict
        )
        # pred shape: (4, n_points) -> row 0: f, row 1: d2f/dx1^2, row 2: d2f/dx1dx2, row 3: d2f/dx2^2

        analytic_d2x1 = self._analytic['d2x1'](X_center[:, 0], X_center[:, 1])
        analytic_d2x1x2 = self._analytic['d2x1x2'](X_center[:, 0], X_center[:, 1])
        analytic_d2x2 = self._analytic['d2x2'](X_center[:, 0], X_center[:, 1])

        error_d2x1 = np.max(np.abs(pred[1, :] - analytic_d2x1))
        error_d2x1x2 = np.max(np.abs(pred[2, :] - analytic_d2x1x2))
        error_d2x2 = np.max(np.abs(pred[3, :] - analytic_d2x2))

        self.assertLess(error_d2x1, 5e-1,
                        f"Submodel 3 max |d2f/dx1^2| error: {error_d2x1}")
        self.assertLess(error_d2x1x2, 5e-1,
                        f"Submodel 3 max |d2f/dx1dx2| error: {error_d2x1x2}")
        self.assertLess(error_d2x2, 5e-1,
                        f"Submodel 3 max |d2f/dx2^2| error: {error_d2x2}")

    # ------------------------------------------------------------------
    # Comprehensive summary
    # ------------------------------------------------------------------

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
        corners_idx = [0, 3, 12, 15]
        X_corners = self.X_train[corners_idx]
        _, submodel_vals = self.model.predict(
            X_corners, self.params, calc_cov=False, return_submodels=True
        )
        y_true = self.submodel_data[0][0][corners_idx].flatten()
        y_pred = submodel_vals[0].flatten()
        abs_error = np.abs(y_pred - y_true)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)

        passed = max_error < 1e-5
        all_tests_passed = all_tests_passed and passed
        status = "PASS" if passed else "FAIL"
        print(
            f"{status} | Function values | Max: {max_error:.2e} | Mean: {mean_error:.2e}")

        # Submodel 2: Edges (function + 1st derivatives)
        print("\nSubmodel 2: Edges (1st Order Derivatives)")
        print(f"Indices: {self.submodel_indices[1]}")
        print("-" * 80)
        edges_idx = self.submodel_indices[1][0]
        X_edges = self.X_train[edges_idx]

        _, submodel_vals = self.model.predict(
            X_edges, self.params, calc_cov=False, return_submodels=True
        )
        y_true = self.submodel_data[1][0][edges_idx].flatten()
        y_pred = submodel_vals[1].flatten()
        abs_error = np.abs(y_pred - y_true)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)

        passed = max_error < 1e-5
        all_tests_passed = all_tests_passed and passed
        status = "PASS" if passed else "FAIL"
        print(
            f"{status} | Function values | Max: {max_error:.2e} | Mean: {mean_error:.2e}")

        # Predict 1st derivatives directly from GP
        derivs_to_predict = [[[1, 1]], [[2, 1]]]
        pred, _ = self.model.predict(
            X_edges, self.params, calc_cov=True,
            return_deriv=True, derivs_to_predict=derivs_to_predict
        )
        analytic_dx1 = self._analytic['dx1'](X_edges[:, 0], X_edges[:, 1])
        analytic_dx2 = self._analytic['dx2'](X_edges[:, 0], X_edges[:, 1])
        err_dx1 = np.max(np.abs(pred[1, :] - analytic_dx1))
        err_dx2 = np.max(np.abs(pred[2, :] - analytic_dx2))
        print(f"      | df/dx1 max error: {err_dx1:.2e}")
        print(f"      | df/dx2 max error: {err_dx2:.2e}")

        # Submodel 3: Center (function + 1st + 2nd derivatives)
        print("\nSubmodel 3: Center (2nd Order Derivatives)")
        print("-" * 80)
        center_idx = self.submodel_indices[2][0]
        X_center = self.X_train[center_idx]

        _, submodel_vals = self.model.predict(
            X_center, self.params, calc_cov=False, return_submodels=True
        )
        y_true = self.submodel_data[2][0][center_idx].flatten()
        y_pred = submodel_vals[2].flatten()
        abs_error = np.abs(y_pred - y_true)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)

        passed = max_error < 1e-4
        all_tests_passed = all_tests_passed and passed
        status = "PASS" if passed else "FAIL"
        print(
            f"{status} | Function values | Max: {max_error:.2e} | Mean: {mean_error:.2e}")

        # Predict 1st + 2nd derivatives directly from GP
        derivs_1st = [[[1, 1]], [[2, 1]]]
        pred_1st, _ = self.model.predict(
            X_center, self.params, calc_cov=True,
            return_deriv=True, derivs_to_predict=derivs_1st
        )
        analytic_dx1 = self._analytic['dx1'](X_center[:, 0], X_center[:, 1])
        analytic_dx2 = self._analytic['dx2'](X_center[:, 0], X_center[:, 1])
        print(f"      | df/dx1 max error: {np.max(np.abs(pred_1st[1, :] - analytic_dx1)):.2e}")
        print(f"      | df/dx2 max error: {np.max(np.abs(pred_1st[2, :] - analytic_dx2)):.2e}")

        derivs_2nd = [[[1, 2]], [[1, 1], [2, 1]], [[2, 2]]]
        pred_2nd, _ = self.model.predict(
            X_center, self.params, calc_cov=True,
            return_deriv=True, derivs_to_predict=derivs_2nd
        )
        analytic_d2x1 = self._analytic['d2x1'](X_center[:, 0], X_center[:, 1])
        analytic_d2x1x2 = self._analytic['d2x1x2'](X_center[:, 0], X_center[:, 1])
        analytic_d2x2 = self._analytic['d2x2'](X_center[:, 0], X_center[:, 1])
        print(f"      | d2f/dx1^2 max error: {np.max(np.abs(pred_2nd[1, :] - analytic_d2x1)):.2e}")
        print(f"      | d2f/dx1dx2 max error: {np.max(np.abs(pred_2nd[2, :] - analytic_d2x1x2)):.2e}")
        print(f"      | d2f/dx2^2 max error: {np.max(np.abs(pred_2nd[3, :] - analytic_d2x2)):.2e}")

        print("\n" + "="*80)
        print("Summary:")
        print(f"  - Total training points: {len(self.X_train)}")
        print(
            f"  - Submodel 1 (Corners): {len(self.submodel_indices[0])} points, 0th order only")
        print(
            f"  - Submodel 2 (Edges): {len(self.submodel_indices[1])} points, up to 1st order")
        print(
            f"  - Submodel 3 (Center): {len(self.submodel_indices[2])} points, up to 2nd order")
        print(f"  - Test function: Six-hump camel function")
        print(f"  - Derivatives predicted directly from GP (no finite differences)")
        print(f"  - Analytic derivatives computed with SymPy")
        print("="*80)
        print(
            f"Overall: {'ALL TESTS PASSED' if all_tests_passed else 'SOME TESTS FAILED'}")
        print("="*80 + "\n")

        self.assertTrue(all_tests_passed, "Not all interpolation tests passed")


def run_tests_with_details():
    """Run tests with detailed output."""
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestWDEGPHeterogeneousDerivatives)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    print("\nRunning WDEGP Heterogeneous Derivatives Interpolation Unit Tests...")
    print("="*80 + "\n")

    result = run_tests_with_details()

    sys.exit(0 if result.wasSuccessful() else 1)
