"""
Unit test for WDEGP calc_cov=True with heterogeneous derivative coverage.

Function:  f(x1, x2) = sin(x1) + x2^2
Submodel 0 (corners): f at all points + df/dx1 at corners only
Submodel 1 (all pts): f at all points + df/dx1 + df/dx2 at all points

Predict df/dx1 with calc_cov=True.

Key assertions
--------------
- No error on calc_cov=True when requesting derivatives.
- Output shapes are correct for both mean and variance.
- Function variance is strictly positive.
- Derivative variance is strictly positive (tests the transform_cov /
  transform_cov_directional fix: previously used
  self.flattened_der_indices[submodel_idx] instead of common_derivs,
  so derivative variances were wrongly scaled or zeroed).
- K_ss computes without IndexError (tests the der_ind_order_pred K_dd fix
  in wdegp_utils.rbf_kernel_predictions).
"""

import sys
import unittest
import warnings
import numpy as np
from jetgp.wdegp.wdegp import wdegp


def f(X):      return np.sin(X[:, 0]) + X[:, 1] ** 2
def df_dx1(X): return np.cos(X[:, 0])
def df_dx2(X): return 2.0 * X[:, 1]


class TestWDEGPFunctionOnlyCovariance(unittest.TestCase):
    """WDEGP with heterogeneous coverage; verifies calc_cov=True output."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)

        # 3x3 grid — 9 training points
        x_vals = np.linspace(0, 1, 3)
        cls.X_train = np.array([[x1, x2] for x1 in x_vals for x2 in x_vals])
        n_train = len(cls.X_train)

        y_all = f(cls.X_train).reshape(-1, 1)

        corners = [0, 2, 6, 8]
        all_pts = list(range(n_train))

        # Submodel 0: f everywhere + df/dx1 at corners
        y_dx1_corners = df_dx1(cls.X_train[corners]).reshape(-1, 1)
        submodel_0 = [y_all, y_dx1_corners]

        # Submodel 1: f everywhere + df/dx1 + df/dx2 at all points
        y_dx1_all = df_dx1(cls.X_train).reshape(-1, 1)
        y_dx2_all = df_dx2(cls.X_train).reshape(-1, 1)
        submodel_1 = [y_all, y_dx1_all, y_dx2_all]

        y_train = [submodel_0, submodel_1]

        der_indices = [
            [[[[1, 1]]]],             # submodel 0: dx1 only
            [[[[1, 1]], [[2, 1]]]],   # submodel 1: dx1 and dx2
        ]
        derivative_locations = [
            [corners],           # submodel 0: dx1 at corners
            [all_pts, all_pts],  # submodel 1: dx1 and dx2 everywhere
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

        np.random.seed(7)
        n_test = 30
        cls.X_test = np.random.uniform(0, 1, (n_test, 2))
        cls.f_true    = f(cls.X_test)
        cls.dx1_true  = df_dx1(cls.X_test)

    # ------------------------------------------------------------------

    def test_no_error_calc_cov(self):
        """calc_cov=True with derivative prediction must not raise."""
        try:
            self.model.predict(
                self.X_test, self.params,
                calc_cov=True, return_deriv=True,
                derivs_to_predict=[[[1, 1]]],
            )
        except Exception as e:
            self.fail(f"predict raised {type(e).__name__}: {e}")

    def test_output_shape(self):
        """Mean and variance must both have shape (2, n_test)."""
        mean, var = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )
        n = len(self.X_test)
        self.assertEqual(mean.shape, (2, n))
        self.assertEqual(var.shape,  (2, n))

    def test_function_variance_positive(self):
        """Predictive variance for f must be strictly positive."""
        _, var = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )
        self.assertTrue(np.all(var[0, :] > 0),
            f"Function variance contains non-positive values: min={var[0,:].min():.3e}")

    def test_derivative_variance_positive(self):
        """Predictive variance for df/dx1 must be strictly positive.

        Tests the transform_cov fix: previously self.flattened_der_indices[i]
        was passed to transform_cov instead of common_derivs, causing
        derivative variances to be incorrectly scaled after denormalization.
        """
        _, var = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )
        self.assertTrue(np.all(var[1, :] > 0),
            f"Derivative variance contains non-positive values: min={var[1,:].min():.3e}")

    def test_function_accuracy(self):
        """Function RMSE at test points must be reasonable."""
        mean, _ = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )
        rmse = float(np.sqrt(np.mean((mean[0, :] - self.f_true) ** 2)))
        self.assertLess(rmse, 0.5, f"Function RMSE too large: {rmse:.4e}")

    def test_derivative_positive_correlation(self):
        """Predicted df/dx1 should be positively correlated with truth."""
        mean, _ = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )
        corr = float(np.corrcoef(mean[1, :].flatten(), self.dx1_true.flatten())[0, 1])
        self.assertGreater(corr, 0.3,
            f"df/dx1 correlation too low: {corr:.3f}")

    def test_comprehensive_summary(self):
        """Print a summary table."""
        mean, var = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )

        def rmse(a, b):
            return float(np.sqrt(np.mean((a.flatten() - b.flatten()) ** 2)))

        corr = float(np.corrcoef(mean[1, :].flatten(), self.dx1_true.flatten())[0, 1])

        print("\n" + "=" * 65)
        print("WDEGP Heterogeneous Coverage — Covariance Test Summary")
        print("=" * 65)
        print(f"  Function  RMSE          : {rmse(mean[0,:], self.f_true):.4e}")
        print(f"  df/dx1    RMSE          : {rmse(mean[1,:], self.dx1_true):.4e}")
        print(f"  df/dx1    Pearson r     : {corr:.3f}")
        print(f"  Function  var  (mean)   : {var[0,:].mean():.4e}")
        print(f"  Derivative var (mean)   : {var[1,:].mean():.4e}")
        print("=" * 65)

        self.assertGreater(corr, 0.3,
            f"df/dx1 correlation too low: {corr:.3f}")


def run_tests_with_details():
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestWDEGPFunctionOnlyCovariance)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    print("\nRunning WDEGP Heterogeneous Coverage Covariance Tests...")
    print("=" * 65 + "\n")
    result = run_tests_with_details()
    sys.exit(0 if result.wasSuccessful() else 1)
