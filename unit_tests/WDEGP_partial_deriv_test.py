"""
Unit test for WDEGP predicting a derivative not common to all submodels.

Function:   f(x1, x2) = sin(x1) + x2^2
Submodel 0 (corners): f at all points + df/dx1 at corner points
Submodel 1 (edges):   f at all points + df/dx1 + df/dx2 at edge/centre points

Previously, requesting df/dx2 would raise a ValueError because df/dx2 is
not in submodel 0 (not common to ALL submodels).  After the guard was
removed, the cross-covariance K_* is constructed from kernel derivatives
directly and each submodel handles the request on its own.
"""

import sys
import unittest
import warnings
import numpy as np
from jetgp.wdegp.wdegp import wdegp


def f(X):      return np.sin(X[:, 0]) + X[:, 1] ** 2
def df_dx1(X): return np.cos(X[:, 0])
def df_dx2(X): return 2.0 * X[:, 1]


class TestWDEGPNonCommonDerivative(unittest.TestCase):
    """WDEGP with heterogeneous derivative coverage; predicts non-common df/dx2."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(7)

        # 3x3 grid → 9 training points
        x_vals = np.linspace(0, 1, 3)
        cls.X_train = np.array([[x1, x2] for x1 in x_vals for x2 in x_vals])

        y_all = f(cls.X_train).reshape(-1, 1)

        corners = [0, 2, 6, 8]
        edges   = [1, 3, 4, 5, 7]   # includes centre

        # Submodel 0: f everywhere + df/dx1 only at corners
        y_dx1_corners = df_dx1(cls.X_train[corners]).reshape(-1, 1)
        submodel_0 = [y_all, y_dx1_corners]

        # Submodel 1: f everywhere + df/dx1 and df/dx2 at edges
        y_dx1_edges = df_dx1(cls.X_train[edges]).reshape(-1, 1)
        y_dx2_edges = df_dx2(cls.X_train[edges]).reshape(-1, 1)
        submodel_1 = [y_all, y_dx1_edges, y_dx2_edges]

        y_train = [submodel_0, submodel_1]

        # Derivative indices: submodel 0 has dx1 only; submodel 1 has dx1 and dx2
        der_indices = [
            [[[[1, 1]]]],             # submodel 0
            [[[[1, 1]], [[2, 1]]]],   # submodel 1
        ]
        derivative_locations = [
            [corners],           # submodel 0: dx1 at corners
            [edges, edges],      # submodel 1: dx1 and dx2 at edges
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.model = wdegp(
                cls.X_train, y_train,
                n_order=1, n_bases=2,
                der_indices=der_indices,
                derivative_locations=derivative_locations,
                normalize=True,
                kernel="SE",
                kernel_type="anisotropic",
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.params = cls.model.optimize_hyperparameters(
                optimizer="powell", n_restart_optimizer=5, debug=False
            )

        np.random.seed(99)
        n_test = 30
        cls.X_test = np.random.uniform(0, 1, (n_test, 2))
        cls.f_true   = f(cls.X_test)
        cls.dx2_true = df_dx2(cls.X_test)

    # ------------------------------------------------------------------

    def test_no_error_non_common_deriv(self):
        """Predicting a non-common derivative must not raise ValueError."""
        try:
            self.model.predict(
                self.X_test, self.params, calc_cov=False,
                return_deriv=True, derivs_to_predict=[[[2, 1]]],
            )
        except ValueError as e:
            self.fail(f"predict raised ValueError for non-common derivative: {e}")

    def test_output_shape_non_common(self):
        """Output shape must be (2, n_test): rows for [f, dx2]."""
        pred = self.model.predict(
            self.X_test, self.params, calc_cov=False,
            return_deriv=True, derivs_to_predict=[[[2, 1]]],
        )
        self.assertEqual(pred.shape, (2, len(self.X_test)))

    def test_no_error_common_deriv(self):
        """Predicting the truly common derivative (df/dx1) must still work."""
        try:
            pred = self.model.predict(
                self.X_test, self.params, calc_cov=False,
                return_deriv=True, derivs_to_predict=[[[1, 1]]],
            )
        except Exception as e:
            self.fail(f"predict raised exception for common derivative: {e}")
        self.assertEqual(pred.shape, (2, len(self.X_test)))

    def test_no_error_both_derivs(self):
        """Requesting both dx1 and dx2 together must not raise."""
        try:
            pred = self.model.predict(
                self.X_test, self.params, calc_cov=False,
                return_deriv=True,
                derivs_to_predict=[[[1, 1]], [[2, 1]]],
            )
        except ValueError as e:
            self.fail(f"predict raised ValueError for both derivs: {e}")
        self.assertEqual(pred.shape, (3, len(self.X_test)))

    def test_function_sanity(self):
        """Function predictions must remain accurate when predicting non-common derivs."""
        pred = self.model.predict(
            self.X_test, self.params, calc_cov=False,
            return_deriv=True, derivs_to_predict=[[[2, 1]]],
        )
        f_pred = pred[0, :].flatten()
        rmse = float(np.sqrt(np.mean((f_pred - self.f_true) ** 2)))
        self.assertLess(rmse, 0.5, f"Function RMSE too large: {rmse:.4e}")

    def test_non_common_deriv_positive_correlation(self):
        """Predicted df/dx2 should be positively correlated with truth.

        Submodel 1 has direct df/dx2 training data, so the weighted ensemble
        should produce a prediction correlated with 2*x2.
        """
        pred = self.model.predict(
            self.X_test, self.params, calc_cov=False,
            return_deriv=True, derivs_to_predict=[[[2, 1]]],
        )
        dx2_pred = pred[1, :].flatten()
        corr = float(np.corrcoef(dx2_pred, self.dx2_true.flatten())[0, 1])
        self.assertGreater(corr, 0.3,
            f"Predicted non-common df/dx2 correlation with truth too low: {corr:.3f}")

    def test_comprehensive_summary(self):
        """Print a summary table for both derivatives."""
        def rmse(a, b):
            return float(np.sqrt(np.mean((a.flatten() - b.flatten()) ** 2)))

        pred = self.model.predict(
            self.X_test, self.params, calc_cov=False,
            return_deriv=True,
            derivs_to_predict=[[[1, 1]], [[2, 1]]],
        )
        dx1_pred = pred[1, :].flatten()
        dx2_pred = pred[2, :].flatten()
        dx1_true = df_dx1(self.X_test).flatten()

        corr_dx1 = float(np.corrcoef(dx1_pred, dx1_true)[0, 1])
        corr_dx2 = float(np.corrcoef(dx2_pred, self.dx2_true.flatten())[0, 1])

        print("\n" + "=" * 65)
        print("WDEGP Non-Common Derivative Test — Summary")
        print("=" * 65)
        print(f"  Function RMSE                   : {rmse(pred[0, :], self.f_true):.4e}")
        print(f"  df/dx1 RMSE  (common)           : {rmse(dx1_pred, dx1_true):.4e}"
              f"  |  r = {corr_dx1:.3f}")
        print(f"  df/dx2 RMSE  (non-common)       : {rmse(dx2_pred, self.dx2_true):.4e}"
              f"  |  r = {corr_dx2:.3f}")
        print("=" * 65)

        self.assertGreater(corr_dx2, 0.3,
            f"df/dx2 correlation too low: {corr_dx2:.3f}")


def run_tests_with_details():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWDEGPNonCommonDerivative)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    print("\nRunning WDEGP Non-Common Derivative Prediction Tests...")
    print("=" * 65 + "\n")
    result = run_tests_with_details()
    sys.exit(0 if result.wasSuccessful() else 1)
