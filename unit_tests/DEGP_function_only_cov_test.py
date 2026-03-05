"""
Unit test for DEGP calc_cov=True when training on function values only.

Function: f(x1, x2, x3) = sin(x1) + cos(x2) + x3^2
Training: function values ONLY — no derivative observations.
Predict:  f and df/dx3 with calc_cov=True.

Key assertions
--------------
- No error on calc_cov=True with untrained derivatives.
- Output shapes are (2, n_test) for both mean and variance.
- Derivative variance is strictly positive (tests the transform_cov fix that
  previously passed self.flattened_der_indices instead of common_derivs).
- K_ss computes without IndexError (tests the der_ind_order_pred K_dd fix).
"""

import sys
import unittest
import warnings
import numpy as np
from jetgp.full_degp.degp import degp


def f(X):      return np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2] ** 2
def df_dx3(X): return 2.0 * X[:, 2]


class TestDEGPFunctionOnlyCovariance(unittest.TestCase):
    """DEGP trained on f values only; predicts df/dx3 with covariance."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_train = 20
        cls.X_train = np.random.uniform(0, 1, (n_train, 3))

        y_train = [f(cls.X_train).reshape(-1, 1)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.model = degp(
                cls.X_train, y_train,
                n_order=1, n_bases=3,
                der_indices=[],
                normalize=True,
                kernel="SE", kernel_type="anisotropic",
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.params = cls.model.optimize_hyperparameters(
                optimizer="powell", n_restart_optimizer=5, debug=False
            )

        np.random.seed(7)
        n_test = 40
        cls.X_test = np.random.uniform(0, 1, (n_test, 3))
        cls.f_true    = f(cls.X_test)
        cls.dx3_true  = df_dx3(cls.X_test)

    # ------------------------------------------------------------------

    def test_no_error_calc_cov(self):
        """calc_cov=True with untrained derivative must not raise."""
        try:
            self.model.predict(
                self.X_test, self.params,
                calc_cov=True, return_deriv=True,
                derivs_to_predict=[[[3, 1]]],
            )
        except Exception as e:
            self.fail(f"predict raised {type(e).__name__}: {e}")

    def test_output_shape(self):
        """Mean and variance must both have shape (2, n_test)."""
        mean, var = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[3, 1]]],
        )
        n = len(self.X_test)
        self.assertEqual(mean.shape, (2, n))
        self.assertEqual(var.shape,  (2, n))

    def test_function_variance_positive(self):
        """Predictive variance for f must be strictly positive."""
        _, var = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[3, 1]]],
        )
        self.assertTrue(np.all(var[0, :] > 0),
            f"Function variance contains non-positive values: min={var[0,:].min():.3e}")

    def test_derivative_variance_positive(self):
        """Predictive variance for df/dx3 must be strictly positive.

        This directly tests the transform_cov fix: previously common_derivs
        was replaced by self.flattened_der_indices (empty for function-only
        training), causing derivative variances to be zero.
        """
        _, var = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[3, 1]]],
        )
        self.assertTrue(np.all(var[1, :] > 0),
            f"Derivative variance contains non-positive values: min={var[1,:].min():.3e}")

    def test_function_accuracy(self):
        """Function RMSE at test points must be reasonable."""
        mean, _ = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[3, 1]]],
        )
        rmse = float(np.sqrt(np.mean((mean[0, :] - self.f_true) ** 2)))
        self.assertLess(rmse, 0.5, f"Function RMSE too large: {rmse:.4e}")

    def test_derivative_positive_correlation(self):
        """Predicted df/dx3 must be positively correlated with truth."""
        mean, _ = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[3, 1]]],
        )
        corr = float(np.corrcoef(mean[1, :], self.dx3_true)[0, 1])
        self.assertGreater(corr, 0.3,
            f"df/dx3 correlation with truth too low: {corr:.3f}")

    def test_comprehensive_summary(self):
        """Print a summary table."""
        mean, var = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[3, 1]]],
        )

        def rmse(a, b):
            return float(np.sqrt(np.mean((a.flatten() - b.flatten()) ** 2)))

        corr = float(np.corrcoef(mean[1, :], self.dx3_true)[0, 1])

        print("\n" + "=" * 65)
        print("DEGP Function-Only Training — Covariance Test Summary")
        print("=" * 65)
        print(f"  Function  RMSE          : {rmse(mean[0,:], self.f_true):.4e}")
        print(f"  df/dx3    RMSE          : {rmse(mean[1,:], self.dx3_true):.4e}")
        print(f"  df/dx3    Pearson r     : {corr:.3f}")
        print(f"  Function  var  (mean)   : {var[0,:].mean():.4e}")
        print(f"  Derivative var (mean)   : {var[1,:].mean():.4e}")
        print("=" * 65)

        self.assertGreater(corr, 0.3,
            f"df/dx3 correlation too low: {corr:.3f}")


def run_tests_with_details():
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestDEGPFunctionOnlyCovariance)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    print("\nRunning DEGP Function-Only Covariance Tests...")
    print("=" * 65 + "\n")
    result = run_tests_with_details()
    sys.exit(0 if result.wasSuccessful() else 1)
