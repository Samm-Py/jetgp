"""
Unit test for DDEGP predicting a directional derivative along an untrained ray.

Function:  f(x1, x2) = x1^2 + x2^2
Rays:      4 directions (45°, 90°, 135°, 180°)
Training:  f values + directional derivatives along rays 1, 2, 3 only
Test:      predict directional derivative along ray 4 (180°) — not in training.

The 180° directional derivative equals -2*x1, which is well-constrained
by the trained directional derivatives (particularly the 45° and 135° ones).
Expected correlation with truth: > 0.5.
"""

import sys
import unittest
import warnings
import numpy as np
from jetgp.full_ddegp.ddegp import ddegp


def f(X):
    return X[:, 0] ** 2 + X[:, 1] ** 2


def grad(X):
    return np.column_stack([2.0 * X[:, 0], 2.0 * X[:, 1]])


def dir_deriv(X, ray):
    """Directional derivative of f along unit vector `ray`."""
    return (grad(X) @ ray).flatten()


# Four ray directions: 45°, 90°, 135°, 180°
_angles = [np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
RAYS = np.array(
    [[np.cos(a) for a in _angles],
     [np.sin(a) for a in _angles]]
)  # shape (2, 4)


class TestDDEGPUntrainedRay(unittest.TestCase):
    """DDEGP trained with 3 directional rays; predicts untrained 4th ray."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(7)
        n_train = 15
        cls.X_train = np.random.uniform(-1, 1, (n_train, 2))

        # Build training data: f + directional derivs along rays 1, 2, 3
        y_func = f(cls.X_train).reshape(-1, 1)
        y_dirs = [dir_deriv(cls.X_train, RAYS[:, i]).reshape(-1, 1)
                  for i in range(3)]
        y_train = [y_func] + y_dirs

        # n_rays=4 (from RAYS shape), but only train with 3 der_indices
        der_indices = [[[[1, 1]], [[2, 1]], [[3, 1]]]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.model = ddegp(
                cls.X_train, y_train,
                n_order=1,
                der_indices=der_indices,
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

        np.random.seed(99)
        n_test = 50
        cls.X_test = np.random.uniform(-1, 1, (n_test, 2))
        cls.dir4_true = dir_deriv(cls.X_test, RAYS[:, 3])   # 180° direction

    # ------------------------------------------------------------------

    def test_no_error_untrained_ray(self):
        """Predicting along an untrained ray must not raise ValueError."""
        try:
            self.model.predict(
                self.X_test, self.params, calc_cov=False,
                return_deriv=True, derivs_to_predict=[[[4, 1]]],
            )
        except ValueError as e:
            self.fail(f"predict raised ValueError for untrained ray: {e}")

    def test_output_shape_untrained_ray(self):
        """Output shape must be (2, n_test): rows for [f, dir4]."""
        pred = self.model.predict(
            self.X_test, self.params, calc_cov=False,
            return_deriv=True, derivs_to_predict=[[[4, 1]]],
        )
        self.assertEqual(pred.shape, (2, len(self.X_test)))

    def test_no_error_trained_subset(self):
        """Predicting a strict subset of trained rays must still work."""
        try:
            pred = self.model.predict(
                self.X_test, self.params, calc_cov=False,
                return_deriv=True,
                derivs_to_predict=[[[1, 1]], [[3, 1]]],
            )
        except ValueError as e:
            self.fail(f"predict raised ValueError for trained subset: {e}")
        self.assertEqual(pred.shape, (3, len(self.X_test)))

    def test_no_error_all_four_rays(self):
        """Requesting all four rays (3 trained + 1 untrained) must not raise."""
        try:
            pred = self.model.predict(
                self.X_test, self.params, calc_cov=False,
                return_deriv=True,
                derivs_to_predict=[[[1, 1]], [[2, 1]], [[3, 1]], [[4, 1]]],
            )
        except ValueError as e:
            self.fail(f"predict raised ValueError for all four rays: {e}")
        self.assertEqual(pred.shape, (5, len(self.X_test)))

    def test_untrained_ray_positive_correlation(self):
        """Predicted 4th directional derivative should correlate with truth.

        The 180° (i.e. -x1 axis) derivative is -2*x1.
        The trained rays (45°, 90°, 135°) provide enough gradient information
        for the GP to reconstruct it well.
        """
        pred = self.model.predict(
            self.X_test, self.params, calc_cov=False,
            return_deriv=True, derivs_to_predict=[[[4, 1]]],
        )
        dir4_pred = pred[1, :].flatten()
        corr = float(np.corrcoef(dir4_pred, self.dir4_true)[0, 1])
        self.assertGreater(corr, 0.5,
            f"Predicted untrained ray-4 derivative has low correlation: {corr:.3f}")

    def test_comprehensive_summary(self):
        """Print a comparison table."""
        def rmse(a, b):
            return float(np.sqrt(np.mean((a.flatten() - b.flatten()) ** 2)))

        pred = self.model.predict(
            self.X_test, self.params, calc_cov=False,
            return_deriv=True,
            derivs_to_predict=[[[1, 1]], [[2, 1]], [[3, 1]], [[4, 1]]],
        )

        true_dirs = [dir_deriv(self.X_test, RAYS[:, i]) for i in range(4)]
        labels = ["45°", "90°", "135°", "180° (untrained)"]

        print("\n" + "=" * 65)
        print("DDEGP Untrained Ray Test — Summary")
        print("=" * 65)
        for i, (label, true) in enumerate(zip(labels, true_dirs)):
            pred_i = pred[i + 1, :].flatten()
            corr = float(np.corrcoef(pred_i, true)[0, 1])
            print(f"  Ray {i + 1} ({label:20s}) | RMSE: {rmse(pred_i, true):.4e} | r: {corr:.3f}")
        print("=" * 65)

        dir4_pred = pred[4, :].flatten()
        corr_4 = float(np.corrcoef(dir4_pred, self.dir4_true)[0, 1])
        self.assertGreater(corr_4, 0.5,
            f"Untrained ray-4 correlation too low: {corr_4:.3f}")


def run_tests_with_details():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDDEGPUntrainedRay)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    print("\nRunning DDEGP Untrained Ray Prediction Tests...")
    print("=" * 65 + "\n")
    result = run_tests_with_details()
    sys.exit(0 if result.wasSuccessful() else 1)
