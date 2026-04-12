"""
Unit test for 1D SDEGP — reproduces Example 4.1 from Cheng & Zimmermann (2024).

Test function (eq. 26):
    g(x) = e^{-x} + sin(5x) + cos(5x) + 0.2x + 4,  x in [0, 6]

Tests:
1. Partition correctness: slices are balanced, non-overlapping, and cover all points.
2. Submodel structure: correct number of submodels (2m-3), correct weights,
   correct function_locations and derivative_locations sizes.
3. NLL consistency: for a fixed hyperparameter, the sliced NLL is close to
   the full (non-sliced) DEGP NLL (they share the same optimum location).
4. Prediction accuracy: SDEGP with optimized hyperparameters achieves low
   NRMSE on test data, comparable to full DEGP.
"""

import unittest
import numpy as np
from scipy.stats.qmc import LatinHypercube

from jetgp.full_degp.degp import degp
from jetgp.sdegp.sdegp import sdegp
from jetgp.sdegp.sliced_partition import partition_indices, build_sliced_submodels
import jetgp.utils as utils


# --- Paper's 1D test function (eq. 26) ---
def g_1d(x):
    """g(x) = e^{-x} + sin(5x) + cos(5x) + 0.2x + 4"""
    x = np.asarray(x).ravel()
    return np.exp(-x) + np.sin(5 * x) + np.cos(5 * x) + 0.2 * x + 4.0


def g_1d_grad(x):
    """dg/dx = -e^{-x} + 5cos(5x) - 5sin(5x) + 0.2"""
    x = np.asarray(x).ravel()
    return (-np.exp(-x) + 5.0 * np.cos(5 * x) - 5.0 * np.sin(5 * x) + 0.2).reshape(-1, 1)


def g_1d_grad2(x):
    """d2g/dx2 = e^{-x} - 25sin(5x) - 25cos(5x)"""
    x = np.asarray(x).ravel()
    return (np.exp(-x) - 25.0 * np.sin(5 * x) - 25.0 * np.cos(5 * x)).reshape(-1, 1)


class TestPartitionIndices(unittest.TestCase):
    """Test the slice partitioning logic."""

    def setUp(self):
        np.random.seed(42)
        self.N = 10
        self.X = np.linspace(0, 6, self.N).reshape(-1, 1)
        self.grads = g_1d_grad(self.X)

    def test_slice_count(self):
        m = 5
        slices, _ = partition_indices(self.X, self.grads, m)
        self.assertEqual(len(slices), m)

    def test_slices_cover_all_points(self):
        m = 5
        slices, _ = partition_indices(self.X, self.grads, m)
        all_idx = sorted(idx for s in slices for idx in s)
        self.assertEqual(all_idx, list(range(self.N)))

    def test_slices_non_overlapping(self):
        m = 5
        slices, _ = partition_indices(self.X, self.grads, m)
        all_idx = [idx for s in slices for idx in s]
        self.assertEqual(len(all_idx), len(set(all_idx)))

    def test_slices_balanced(self):
        m = 5
        slices, _ = partition_indices(self.X, self.grads, m)
        sizes = [len(s) for s in slices]
        self.assertTrue(max(sizes) - min(sizes) <= 1)

    def test_m_less_than_3_raises(self):
        with self.assertRaises(ValueError):
            partition_indices(self.X, self.grads, 2)

    def test_1d_slice_dim_is_zero(self):
        """For 1D, the only coordinate is dim 0."""
        m = 5
        _, slice_dim = partition_indices(self.X, self.grads, m)
        self.assertEqual(slice_dim, 0)


class TestBuildSlicedSubmodels(unittest.TestCase):
    """Test the submodel construction."""

    def setUp(self):
        self.N = 10
        self.DIM = 1
        self.M = 5
        self.X = np.linspace(0, 6, self.N).reshape(-1, 1)
        self.y = g_1d(self.X)
        self.grads = g_1d_grad(self.X)

    def test_num_submodels(self):
        result = build_sliced_submodels(self.X, self.y, self.grads, self.M)
        submodel_data = result[0]
        self.assertEqual(len(submodel_data), 2 * self.M - 3)

    def test_weights(self):
        result = build_sliced_submodels(self.X, self.y, self.grads, self.M)
        weights = result[4]
        # First m-1 are pair blocks (+1), next m-2 are corrections (-1)
        expected = [+1.0] * (self.M - 1) + [-1.0] * (self.M - 2)
        np.testing.assert_array_equal(weights, expected)

    def test_function_locations_sizes(self):
        result = build_sliced_submodels(self.X, self.y, self.grads, self.M)
        func_locs = result[3]
        slices = result[6]
        # Pair blocks: union of two adjacent slices
        for i in range(self.M - 1):
            expected_size = len(slices[i]) + len(slices[i + 1])
            self.assertEqual(len(func_locs[i]), expected_size)
        # Single-slice corrections
        for i in range(self.M - 2):
            sm_idx = (self.M - 1) + i
            expected_size = len(slices[i + 1])
            self.assertEqual(len(func_locs[sm_idx]), expected_size)

    def test_y_train_matches_function_locations(self):
        """Each submodel's y_train[0] should have the same length as its
        function_locations."""
        result = build_sliced_submodels(self.X, self.y, self.grads, self.M)
        submodel_data = result[0]
        func_locs = result[3]
        for i, (sd, fl) in enumerate(zip(submodel_data, func_locs)):
            self.assertEqual(
                len(sd[0]), len(fl),
                f"Submodel {i}: y_train[0] length {len(sd[0])} != "
                f"function_locations length {len(fl)}"
            )

    def test_function_values_correct(self):
        """Check that sliced function values match the full y at the
        corresponding global indices."""
        result = build_sliced_submodels(self.X, self.y, self.grads, self.M)
        submodel_data = result[0]
        func_locs = result[3]
        for i, (sd, fl) in enumerate(zip(submodel_data, func_locs)):
            expected = self.y[fl].reshape(-1, 1)
            np.testing.assert_array_almost_equal(
                sd[0], expected,
                err_msg=f"Submodel {i}: function values mismatch"
            )


class TestSDEGP1DModel(unittest.TestCase):
    """Test SDEGP model construction and NLL on the 1D example."""

    @classmethod
    def setUpClass(cls):
        """Build both a full DEGP and a sliced SDEGP on the same data."""
        cls.N = 10
        cls.DIM = 1
        cls.M = 5

        # Equally-spaced points (deterministic, reproducible)
        cls.X = np.linspace(0, 6, cls.N).reshape(-1, 1)
        cls.y = g_1d(cls.X)
        cls.grads = g_1d_grad(cls.X)

        # --- Full DEGP ---
        der_specs = utils.gen_OTI_indices(cls.DIM, 1)
        y_train_full = [
            cls.y.reshape(-1, 1),
            cls.grads,
        ]
        all_pts = list(range(cls.N))
        cls.degp_model = degp(
            cls.X, y_train_full,
            n_order=1, n_bases=cls.DIM,
            der_indices=der_specs,
            derivative_locations=[all_pts],
            normalize=True, kernel='SE', kernel_type='anisotropic',
        )

        # --- Sliced SDEGP (new simplified API) ---
        cls.sdegp_model = sdegp(
            cls.X, cls.y, cls.grads,
            n_order=1, m=cls.M,
            kernel='SE', kernel_type='anisotropic',
        )

    def test_model_builds(self):
        """Smoke test: model construction doesn't crash."""
        self.assertIsNotNone(self.sdegp_model)

    def test_nll_evaluates(self):
        """NLL evaluation returns a finite number at a reasonable theta."""
        x0 = np.array([0.7, 0.0, -6.0])
        nll = self.sdegp_model.optimizer.nll_wrapper(x0)
        self.assertTrue(np.isfinite(nll), f"NLL is not finite: {nll}")

    def test_nll_grad_evaluates(self):
        """NLL gradient returns finite values."""
        x0 = np.array([0.7, 0.0, -6.0])
        grad = self.sdegp_model.optimizer.nll_grad(x0)
        self.assertTrue(np.all(np.isfinite(grad)), f"Gradient not finite: {grad}")

    def test_nll_and_grad_consistent(self):
        """nll_and_grad returns same NLL as nll_wrapper."""
        x0 = np.array([0.7, 0.0, -6.0])
        nll_only = self.sdegp_model.optimizer.nll_wrapper(x0)
        nll_combined, grad = self.sdegp_model.optimizer.nll_and_grad(x0)
        self.assertAlmostEqual(
            nll_only, nll_combined, places=6,
            msg="nll_wrapper and nll_and_grad return different NLL values"
        )

    def test_sliced_nll_minimum_near_full(self):
        """The sliced NLL minimum should be near the full DEGP NLL minimum
        (same optimal theta location, as shown in Figs 5-6 of the paper)."""
        # Sweep log10(theta) near the minimum region
        thetas = np.linspace(-0.5, 1.5, 50)
        nll_full = []
        nll_sliced = []
        for t in thetas:
            x0 = np.array([t, 0.0, -6.0])
            nll_full.append(self.degp_model.optimizer.nll_wrapper(x0))
            nll_sliced.append(self.sdegp_model.optimizer.nll_wrapper(x0))

        theta_opt_full = thetas[np.argmin(nll_full)]
        theta_opt_sliced = thetas[np.argmin(nll_sliced)]

        self.assertAlmostEqual(
            theta_opt_full, theta_opt_sliced, delta=0.2,
            msg=f"Optimal theta differs: full={theta_opt_full:.3f}, "
                f"sliced={theta_opt_sliced:.3f}"
        )

    def test_sliced_nll_converges_to_full_near_optimum(self):
        """Near the optimum, SDEGP and full DEGP NLL values should be very close."""
        x0 = np.array([0.75, 0.0, -6.0])
        nll_full = self.degp_model.optimizer.nll_wrapper(x0)
        nll_sliced = self.sdegp_model.optimizer.nll_wrapper(x0)
        # Should be within ~1% of each other near the optimum
        rel_diff = abs(nll_sliced - nll_full) / abs(nll_full)
        self.assertLess(
            rel_diff, 0.02,
            f"NLL values diverge near optimum: full={nll_full:.4f}, "
            f"sliced={nll_sliced:.4f}, rel_diff={rel_diff:.4f}"
        )


class TestSDEGP1DPrediction(unittest.TestCase):
    """Test that SDEGP predictions are accurate on the 1D example."""

    @classmethod
    def setUpClass(cls):
        cls.N = 10
        cls.DIM = 1
        cls.M = 5
        cls.N_TEST = 500

        cls.X = np.linspace(0, 6, cls.N).reshape(-1, 1)
        cls.y = g_1d(cls.X)
        cls.grads = g_1d_grad(cls.X)

        cls.model = sdegp(
            cls.X, cls.y, cls.grads,
            n_order=1, m=cls.M,
            kernel='SE', kernel_type='anisotropic',
        )

        # Use known-good hyperparameters near the NLL minimum
        cls.params = np.array([0.75, 0.0, -6.0])

        # Test data
        cls.X_test = np.linspace(0, 6, cls.N_TEST).reshape(-1, 1)
        cls.y_test = g_1d(cls.X_test)

    def test_predict(self):
        """Prediction returns correct shape and reasonable NRMSE."""
        y_pred = self.model.predict(
            self.X_test, self.params,
            calc_cov=False, return_deriv=False
        )
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        y_pred = np.asarray(y_pred).flatten()

        self.assertEqual(len(y_pred), self.N_TEST)

        # Compute NRMSE
        rmse = np.sqrt(np.mean((self.y_test - y_pred) ** 2))
        nrmse = rmse / np.std(self.y_test)
        self.assertLess(
            nrmse, 0.15,
            f"NRMSE too high: {nrmse:.4f} (expected < 0.15 for this problem)"
        )


class TestSDEGPMetadata(unittest.TestCase):
    """Test that SDEGP exposes slicing metadata."""

    def test_slicing_attributes(self):
        N, M = 10, 5
        X = np.linspace(0, 6, N).reshape(-1, 1)
        y = g_1d(X)
        grads = g_1d_grad(X)

        model = sdegp(X, y, grads, n_order=1, m=M,
                       kernel='SE', kernel_type='anisotropic')

        self.assertEqual(model.m, M)
        self.assertEqual(len(model.slices), M)
        self.assertEqual(model.slice_dim, 0)
        self.assertEqual(model.num_submodels, 2 * M - 3)
        self.assertEqual(len(model.submodel_weights), 2 * M - 3)

        # NLL should evaluate without error
        x0 = np.array([0.7, 0.0, -6.0])
        nll = model.optimizer.nll_wrapper(x0)
        self.assertTrue(np.isfinite(nll))


class TestSDEGP1DSecondOrder(unittest.TestCase):
    """Test SDEGP with second-order derivatives on the 1D example."""

    @classmethod
    def setUpClass(cls):
        cls.N = 10
        cls.DIM = 1
        cls.M = 5
        cls.N_TEST = 500

        cls.X = np.linspace(0, 6, cls.N).reshape(-1, 1)
        cls.y = g_1d(cls.X)
        cls.g1 = g_1d_grad(cls.X)
        cls.g2 = g_1d_grad2(cls.X)
        # Stack first- and second-order derivatives as columns
        cls.grads_2 = np.hstack([cls.g1, cls.g2])

        # --- Full second-order DEGP ---
        der_specs = utils.gen_OTI_indices(cls.DIM, 2)
        all_pts = list(range(cls.N))
        n_deriv_types = sum(len(group) for group in der_specs)
        cls.degp_model = degp(
            cls.X,
            [cls.y.reshape(-1, 1), cls.g1, cls.g2],
            n_order=2, n_bases=cls.DIM,
            der_indices=der_specs,
            derivative_locations=[all_pts] * n_deriv_types,
            normalize=True, kernel='SE', kernel_type='anisotropic',
        )

        # --- Second-order SDEGP ---
        cls.sdegp_model = sdegp(
            cls.X, cls.y, cls.grads_2,
            n_order=2, m=cls.M,
            kernel='SE', kernel_type='anisotropic',
        )

        cls.params = np.array([0.75, 0.0, -6.0])
        cls.X_test = np.linspace(0, 6, cls.N_TEST).reshape(-1, 1)
        cls.y_test = g_1d(cls.X_test)

    def test_model_builds(self):
        self.assertIsNotNone(self.sdegp_model)
        self.assertEqual(self.sdegp_model.num_submodels, 2 * self.M - 3)

    def test_nll_evaluates(self):
        nll = self.sdegp_model.optimizer.nll_wrapper(self.params)
        self.assertTrue(np.isfinite(nll), f"NLL is not finite: {nll}")

    def test_nll_grad_evaluates(self):
        grad = self.sdegp_model.optimizer.nll_grad(self.params)
        self.assertTrue(np.all(np.isfinite(grad)), f"Gradient not finite: {grad}")

    def test_nll_and_grad_consistent(self):
        nll_only = self.sdegp_model.optimizer.nll_wrapper(self.params)
        nll_combined, grad = self.sdegp_model.optimizer.nll_and_grad(self.params)
        self.assertAlmostEqual(
            nll_only, nll_combined, places=6,
            msg="nll_wrapper and nll_and_grad return different NLL values"
        )

    def test_sliced_nll_minimum_near_full(self):
        """The sliced NLL minimum should be near the full DEGP NLL minimum."""
        thetas = np.linspace(-0.5, 1.5, 50)
        nll_full = []
        nll_sliced = []
        for t in thetas:
            x0 = np.array([t, 0.0, -6.0])
            nll_full.append(self.degp_model.optimizer.nll_wrapper(x0))
            nll_sliced.append(self.sdegp_model.optimizer.nll_wrapper(x0))

        theta_opt_full = thetas[np.argmin(nll_full)]
        theta_opt_sliced = thetas[np.argmin(nll_sliced)]

        self.assertAlmostEqual(
            theta_opt_full, theta_opt_sliced, delta=0.2,
            msg=f"Optimal theta differs: full={theta_opt_full:.3f}, "
                f"sliced={theta_opt_sliced:.3f}"
        )

    def test_predict(self):
        """Prediction returns correct shape and reasonable NRMSE."""
        y_pred = self.sdegp_model.predict(
            self.X_test, self.params,
            calc_cov=False, return_deriv=False
        )
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        y_pred = np.asarray(y_pred).flatten()

        self.assertEqual(len(y_pred), self.N_TEST)

        rmse = np.sqrt(np.mean((self.y_test - y_pred) ** 2))
        nrmse = rmse / np.std(self.y_test)
        self.assertLess(
            nrmse, 0.15,
            f"NRMSE too high: {nrmse:.4f} (expected < 0.15 for this problem)"
        )


if __name__ == '__main__':
    unittest.main()
