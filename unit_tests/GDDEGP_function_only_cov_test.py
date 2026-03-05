"""
Unit test for GDDEGP calc_cov=True when training on function values only.

Function:  f(x1, x2) = sin(x1) * cos(x2)
Training:  function values ONLY — no directional derivative observations.
           n_bases=2 is passed explicitly so the OTI space can represent
           one prediction direction type, even though der_indices=[].
Predict:   f and a directional derivative along the gradient (45° ray) with
           calc_cov=True.

Why n_bases must be set explicitly
------------------------------------
By default GDDEGP computes n_bases = 2 * n_direction_types_in_training.
With der_indices=[] that gives n_bases=0, leaving no OTI space for derivative
predictions.  Passing n_bases=2 reserves space for one prediction direction
type without requiring any training derivatives.

Key assertions
--------------
- No error on calc_cov=True with untrained directional derivative.
- Output shapes are (2, n_test) for both mean and variance.
- Function variance is non-negative with positive mean.
- Derivative variance is strictly positive (tests the transform_cov_directional
  fix and the der_indices_tr_odd_pred K_fd/K_dd fix in gddegp_utils).
"""

import sys
import unittest
import warnings
import numpy as np
from jetgp.full_gddegp.gddegp import gddegp


def f(X):
    return np.sin(X[:, 0]) * np.cos(X[:, 1])


def dir_deriv(X, ray):
    dfdx1 = np.cos(X[:, 0]) * np.cos(X[:, 1])
    dfdx2 = -np.sin(X[:, 0]) * np.sin(X[:, 1])
    grad = np.column_stack([dfdx1, dfdx2])
    return (grad @ ray).flatten()


class TestGDDEGPFunctionOnlyCovariance(unittest.TestCase):
    """GDDEGP trained on f values only; predicts directional deriv with cov."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_train = 20
        cls.X_train = np.random.uniform(0, 1, (n_train, 2))

        y_train = [f(cls.X_train).reshape(-1, 1)]

        # n_bases=2 reserves OTI space for 1 prediction direction type.
        # rays_list=[] and der_indices=[] because no training derivatives.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.model = gddegp(
                cls.X_train, y_train,
                n_order=1,
                rays_list=[],
                der_indices=[],
                derivative_locations=[],
                n_bases=2,
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
        cls.f_true = f(cls.X_test)

        # 45° ray for all test points
        ray_45 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
        cls.rays_predict = [np.tile(ray_45.reshape(2, 1), (1, n_test))]
        cls.dir_true = dir_deriv(cls.X_test, ray_45)

    # ------------------------------------------------------------------

    def test_no_error_calc_cov(self):
        """calc_cov=True with untrained directional derivative must not raise."""
        try:
            self.model.predict(
                self.X_test, self.params,
                rays_predict=self.rays_predict,
                calc_cov=True, return_deriv=True,
                derivs_to_predict=[[[1, 1]]],
            )
        except Exception as e:
            self.fail(f"predict raised {type(e).__name__}: {e}")

    def test_output_shape(self):
        """Mean and variance must both have shape (2, n_test)."""
        mean, var = self.model.predict(
            self.X_test, self.params,
            rays_predict=self.rays_predict,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )
        n = len(self.X_test)
        self.assertEqual(mean.shape, (2, n))
        self.assertEqual(var.shape,  (2, n))

    def test_function_variance_nonnegative(self):
        """Predictive variance for f must be non-negative with positive mean."""
        _, var = self.model.predict(
            self.X_test, self.params,
            rays_predict=self.rays_predict,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )
        self.assertTrue(np.all(var[0, :] >= 0),
            f"Function variance contains negative values: min={var[0,:].min():.3e}")
        self.assertGreater(float(var[0, :].mean()), 0,
            "Function variance is zero everywhere")

    def test_derivative_variance_positive(self):
        """Predictive variance for the directional derivative must be positive.

        Tests the transform_cov_directional fix (gddegp.py) and the
        der_indices_tr_odd_pred K_fd/K_dd fix (gddegp_utils.py).
        """
        _, var = self.model.predict(
            self.X_test, self.params,
            rays_predict=self.rays_predict,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )
        self.assertTrue(np.all(var[1, :] > 0),
            f"Derivative variance contains non-positive values: min={var[1,:].min():.3e}")

    def test_function_accuracy(self):
        """Function RMSE at test points must be reasonable."""
        mean, _ = self.model.predict(
            self.X_test, self.params,
            rays_predict=self.rays_predict,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )
        rmse = float(np.sqrt(np.mean((mean[0, :] - self.f_true) ** 2)))
        self.assertLess(rmse, 0.5, f"Function RMSE too large: {rmse:.4e}")

    def test_derivative_positive_correlation(self):
        """Predicted directional derivative should correlate with truth."""
        mean, _ = self.model.predict(
            self.X_test, self.params,
            rays_predict=self.rays_predict,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )
        corr = float(np.corrcoef(mean[1, :].flatten(), self.dir_true)[0, 1])
        self.assertGreater(corr, 0.3,
            f"Directional derivative correlation too low: {corr:.3f}")

    def test_comprehensive_summary(self):
        """Print a summary table."""
        mean, var = self.model.predict(
            self.X_test, self.params,
            rays_predict=self.rays_predict,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=[[[1, 1]]],
        )

        def rmse(a, b):
            return float(np.sqrt(np.mean((a.flatten() - b.flatten()) ** 2)))

        corr = float(np.corrcoef(mean[1, :].flatten(), self.dir_true)[0, 1])

        print("\n" + "=" * 65)
        print("GDDEGP Function-Only Training — Covariance Test Summary")
        print("=" * 65)
        print(f"  Function  RMSE          : {rmse(mean[0,:], self.f_true):.4e}")
        print(f"  Dir-deriv RMSE (45°)    : {rmse(mean[1,:], self.dir_true):.4e}")
        print(f"  Dir-deriv Pearson r     : {corr:.3f}")
        print(f"  Function  var  (mean)   : {var[0,:].mean():.4e}")
        print(f"  Derivative var (mean)   : {var[1,:].mean():.4e}")
        print("=" * 65)

        self.assertGreater(corr, 0.3,
            f"Directional derivative correlation too low: {corr:.3f}")


def run_tests_with_details():
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGDDEGPFunctionOnlyCovariance)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    print("\nRunning GDDEGP Function-Only Covariance Tests...")
    print("=" * 65 + "\n")
    result = run_tests_with_details()
    sys.exit(0 if result.wasSuccessful() else 1)
