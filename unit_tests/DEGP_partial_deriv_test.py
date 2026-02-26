"""
Unit test for DEGP predicting derivatives not seen during training.

Function: f(x1, x2, x3) = sin(x1) + cos(x2) + x3^2
Training: f values + df/dx1 + df/dx2  (NOT df/dx3)
Test:     predict df/dx3 at test points — not in the training set.

Key assertion: the code path that previously raised a ValueError for
"derivative index not in training set" no longer blocks valid predictions.
"""

import sys
import unittest
import warnings
import numpy as np
from jetgp.full_degp.degp import degp


def f(X):      return np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2] ** 2
def df_dx1(X): return  np.cos(X[:, 0])
def df_dx2(X): return -np.sin(X[:, 1])
def df_dx3(X): return  2.0 * X[:, 2]


class TestDEGPUntrainedDerivative(unittest.TestCase):
    """DEGP trained on f, df/dx1, df/dx2; predicts untrained df/dx3."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(7)
        n_train = 20
        cls.X_train = np.random.uniform(0, 1, (n_train, 3))

        y_train = [
            f(cls.X_train).reshape(-1, 1),
            df_dx1(cls.X_train).reshape(-1, 1),
            df_dx2(cls.X_train).reshape(-1, 1),
        ]
        # Train with ONLY df/dx1 and df/dx2 — NOT df/dx3
        der_indices = [[[[1, 1]], [[2, 1]]]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.model = degp(
                cls.X_train, y_train,
                n_order=1, n_bases=3,
                der_indices=der_indices,
                normalize=True, kernel="SE", kernel_type="anisotropic",
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.params = cls.model.optimize_hyperparameters(
                optimizer="powell", n_restart_optimizer=5, debug=False
            )

        np.random.seed(99)
        n_test = 50
        cls.X_test = np.random.uniform(0, 1, (n_test, 3))
        cls.f_true   = f(cls.X_test)
        cls.dx3_true = df_dx3(cls.X_test)

    # ------------------------------------------------------------------

    def test_no_error_single_untrained_deriv(self):
        """Predicting an untrained derivative must not raise ValueError."""
        try:
            self.model.predict(
                self.X_test, self.params, calc_cov=False,
                return_deriv=True, derivs_to_predict=[[[3, 1]]],
            )
        except ValueError as e:
            self.fail(f"predict raised ValueError for untrained derivative: {e}")

    def test_output_shape_untrained(self):
        """Output shape must be (1 + n_derivs_requested, n_test)."""
        pred = self.model.predict(
            self.X_test, self.params, calc_cov=False,
            return_deriv=True, derivs_to_predict=[[[3, 1]]],
        )
        # Row 0: f, row 1: dx3
        self.assertEqual(pred.shape, (2, len(self.X_test)))

    def test_no_error_mixed_trained_untrained(self):
        """Requesting trained and untrained derivatives together must not raise."""
        try:
            pred = self.model.predict(
                self.X_test, self.params, calc_cov=False,
                return_deriv=True,
                derivs_to_predict=[[[1, 1]], [[2, 1]], [[3, 1]]],
            )
        except ValueError as e:
            self.fail(f"predict raised ValueError for mixed derivs: {e}")
        # Row 0: f, rows 1-3: dx1, dx2, dx3
        self.assertEqual(pred.shape, (4, len(self.X_test)))

    def test_function_sanity_under_untrained_call(self):
        """Function values must remain accurate when predicting untrained derivs."""
        pred = self.model.predict(
            self.X_test, self.params, calc_cov=False,
            return_deriv=True, derivs_to_predict=[[[3, 1]]],
        )
        f_pred = pred[0, :]
        rmse = float(np.sqrt(np.mean((f_pred - self.f_true) ** 2)))
        self.assertLess(rmse, 0.5, f"Function RMSE too large: {rmse:.4e}")

    def test_untrained_deriv_positive_correlation(self):
        """Predicted df/dx3 should be positively correlated with truth.

        The GP posterior for df/dx3 is informed by function values at
        different x3 values through the kernel derivative structure.
        """
        pred = self.model.predict(
            self.X_test, self.params, calc_cov=False,
            return_deriv=True, derivs_to_predict=[[[3, 1]]],
        )
        dx3_pred = pred[1, :].flatten()
        corr = float(np.corrcoef(dx3_pred, self.dx3_true.flatten())[0, 1])
        self.assertGreater(corr, 0.3,
            f"Predicted untrained df/dx3 correlation with truth too low: {corr:.3f}")

    def test_comprehensive_summary(self):
        """Print a comparison table of predictions."""
        pred_untrained = self.model.predict(
            self.X_test, self.params, calc_cov=False,
            return_deriv=True, derivs_to_predict=[[[3, 1]]],
        )
        pred_all = self.model.predict(
            self.X_test, self.params, calc_cov=False,
            return_deriv=True,
            derivs_to_predict=[[[1, 1]], [[2, 1]], [[3, 1]]],
        )

        def rmse(a, b):
            return float(np.sqrt(np.mean((a.flatten() - b.flatten()) ** 2)))

        corr_dx3 = float(np.corrcoef(
            pred_untrained[1, :].flatten(), self.dx3_true.flatten())[0, 1])

        print("\n" + "=" * 65)
        print("DEGP Untrained Derivative Test — Summary")
        print("=" * 65)
        print(f"  Function RMSE (untrained-deriv call) : "
              f"{rmse(pred_untrained[0, :], self.f_true):.4e}")
        print(f"  df/dx3 RMSE  (untrained)             : "
              f"{rmse(pred_untrained[1, :], self.dx3_true):.4e}")
        print(f"  df/dx3 Pearson r                     : {corr_dx3:.3f}")
        print(f"  Function RMSE (all-derivs call)      : "
              f"{rmse(pred_all[0, :], self.f_true):.4e}")
        print(f"  df/dx3 RMSE  (all-derivs call)       : "
              f"{rmse(pred_all[3, :], self.dx3_true):.4e}")
        print("=" * 65)
        self.assertGreater(corr_dx3, 0.3,
            f"df/dx3 correlation too low: {corr_dx3:.3f}")


def run_tests_with_details():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDEGPUntrainedDerivative)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    print("\nRunning DEGP Untrained Derivative Prediction Tests...")
    print("=" * 65 + "\n")
    result = run_tests_with_details()
    sys.exit(0 if result.wasSuccessful() else 1)
