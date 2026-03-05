"""
Unit test for DDEGP calc_cov=True when training on function values only.

Function:  f(x1, x2) = x1^2 + x2^2
Training:  function values ONLY — no directional derivative observations.
           Three ray directions are defined in the OTI space but no training
           derivative data is provided (der_indices=[]).
Predict:   f and directional derivative along ray 1 (45°) with calc_cov=True.

Key assertions
--------------
- No error on calc_cov=True with an entirely untrained derivative space.
- Output shapes are (2, n_test) for both mean and variance.
- Derivative variance is strictly positive (tests transform_cov_directional fix
  that previously passed self.flattened_der_indices instead of common_derivs).
- K_ss computes without IndexError (tests the der_ind_order_pred K_dd fix in
  ddegp_utils.rbf_kernel_predictions).
"""

import sys
import unittest
import warnings
import numpy as np
from jetgp.full_ddegp.ddegp import ddegp


def f(X):
    return X[:, 0] ** 2 + X[:, 1] ** 2


def dir_deriv(X, ray):
    grad = np.column_stack([2.0 * X[:, 0], 2.0 * X[:, 1]])
    return (grad @ ray).flatten()


# Three ray directions: 45°, 90°, 135°
_angles = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
RAYS = np.array(
    [[np.cos(a) for a in _angles],
     [np.sin(a) for a in _angles]]
)  # shape (2, 3)


class TestDDEGPFunctionOnlyCovariance(unittest.TestCase):
    """DDEGP trained on f values only; predicts directional derivative with cov."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_train = 15
        cls.X_train = np.random.uniform(-1, 1, (n_train, 2))

        y_train = [f(cls.X_train).reshape(-1, 1)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.model = ddegp(
                cls.X_train, y_train,
                n_order=1,
                der_indices=[],
                rays=RAYS,
                normalize=True,
                kernel="SE",
                kernel_type="isotropic",
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.params = cls.model.optimize_hyperparameters(
                optimizer="powell", n_restart_optimizer=5, debug=False
            )

        np.random.seed(7)
        n_test = 40
        cls.X_test = np.random.uniform(-1, 1, (n_test, 2))
        cls.f_true    = f(cls.X_test)
        cls.dir1_true = dir_deriv(cls.X_test, RAYS[:, 0])   # 45° direction

    # ------------------------------------------------------------------

    def test_no_error_calc_cov(self):
        """calc_cov=True with untrained directional derivative must not raise."""
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
        """Predictive variance for f must be non-negative with positive mean."""
        _, var = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )
        self.assertTrue(np.all(var[0, :] >= 0),
            f"Function variance contains negative values: min={var[0,:].min():.3e}")
        self.assertGreater(float(var[0, :].mean()), 0,
            "Function variance is zero everywhere — transform_cov may be broken")

    def test_derivative_variance_positive(self):
        """Predictive variance for the directional derivative must be positive.

        This tests the transform_cov_directional fix: previously used
        self.flattened_der_indices (empty) instead of common_derivs, causing
        derivative variances to be zero after denormalization.
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
        """Predicted 45° directional derivative should correlate with truth."""
        mean, _ = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )
        corr = float(np.corrcoef(mean[1, :].flatten(), self.dir1_true)[0, 1])
        self.assertGreater(corr, 0.3,
            f"Directional derivative correlation too low: {corr:.3f}")

    def test_comprehensive_summary(self):
        """Print a summary table."""
        mean, var = self.model.predict(
            self.X_test, self.params,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )

        def rmse(a, b):
            return float(np.sqrt(np.mean((a.flatten() - b.flatten()) ** 2)))

        corr = float(np.corrcoef(mean[1, :].flatten(), self.dir1_true)[0, 1])

        print("\n" + "=" * 65)
        print("DDEGP Function-Only Training — Covariance Test Summary")
        print("=" * 65)
        print(f"  Function  RMSE          : {rmse(mean[0,:], self.f_true):.4e}")
        print(f"  Dir-deriv RMSE (45°)    : {rmse(mean[1,:], self.dir1_true):.4e}")
        print(f"  Dir-deriv Pearson r     : {corr:.3f}")
        print(f"  Function  var  (mean)   : {var[0,:].mean():.4e}")
        print(f"  Derivative var (mean)   : {var[1,:].mean():.4e}")
        print("=" * 65)

        self.assertGreater(corr, 0.3,
            f"Directional derivative correlation too low: {corr:.3f}")


def run_tests_with_details():
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestDDEGPFunctionOnlyCovariance)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    print("\nRunning DDEGP Function-Only Covariance Tests...")
    print("=" * 65 + "\n")
    result = run_tests_with_details()
    sys.exit(0 if result.wasSuccessful() else 1)
