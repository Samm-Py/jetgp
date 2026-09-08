"""The closed-form GVR must agree with an explicit fantasy rebuild.

Adding an observation to a GP with fixed hyperparameters reduces the posterior
variance by an amount that does not depend on the value observed, which is why
``global_variance`` can score candidates without ever building a fantasy model.
These tests build the fantasy models anyway and check that the two agree.

Everything here uses ``normalize=False``. With normalisation on, appending a
point changes ``mus_x`` / ``sigmas_x`` and ``mu_y`` / ``sigma_y``, which changes
the kernel in real space and makes a like-for-like comparison impossible.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Prefer the jetgp source tree in this repository over any other copy the
# environment has registered; conftest.py only puts active_learning/ on the path.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from jetgp.full_gddegp.gddegp import gddegp

import doe_utils
import global_variance as gv
from posterior_queries import query_function_posterior_batched


BOUNDS = np.array([[-2.0, 2.0], [-2.0, 2.0]])
DIM = 2
N_BASES = 2 * DIM
RTOL = 1e-6


def _f(X):
    X = np.atleast_2d(X)
    return (np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2).reshape(-1, 1)


def _grad(X):
    X = np.atleast_2d(X)
    return np.column_stack([np.cos(X[:, 0]), X[:, 1]])


def _build(X, y, observations=()):
    """Unnormalised GDDEGP over ``X``, ``y`` and directional observations.

    Mirrors ``gp_builders._construct_directional_gp`` (observations sharing a
    slot become one direction type) but keeps ``normalize=False``.
    """
    slots = {}
    for obs in observations:
        slots.setdefault(obs["slot"], []).append(obs)

    y_blocks = [np.asarray(y, dtype=float)]
    rays_list = []
    der_locs = []
    for slot in sorted(slots):
        group = slots[slot]
        y_blocks.append(np.array([[o["value"]] for o in group], dtype=float))
        rays_list.append(np.column_stack([
            np.asarray(o["direction"], dtype=float)
            / np.linalg.norm(o["direction"]) for o in group]))
        der_locs.append([int(o["x_index"]) for o in group])

    n_types = len(rays_list)
    der_indices = [[[[i + 1, 1]] for i in range(n_types)]] if n_types else []
    return gddegp(
        np.asarray(X, dtype=float), y_blocks,
        n_order=1 if n_types else 0,
        rays_list=rays_list,
        der_indices=der_indices,
        derivative_locations=der_locs,
        n_bases=N_BASES,
        normalize=False,
        kernel="SE",
        kernel_type="anisotropic",
    )


def _integrated_variance(model, params, Z):
    _, var = query_function_posterior_batched(model, params, Z, batch_size=250)
    return float(np.mean(var))


@pytest.fixture(scope="module")
def setup():
    X = doe_utils.lhs_design(6, BOUNDS, seed=3)
    y = _f(X)
    Z = doe_utils.lhs_design(15, BOUNDS, seed=4)
    model = _build(X, y)
    # Mid-range hyperparameters: valid by construction, and the run is about
    # internal consistency rather than fit quality.
    params = np.array([0.5 * (lo + hi) for lo, hi in model.bounds], dtype=float)
    noise_var = gv.noise_variance(model, params)
    base_gv = _integrated_variance(model, params, Z)
    return {"X": X, "y": y, "Z": Z, "model": model, "params": params,
            "noise_var": noise_var, "base_gv": base_gv}


def test_option_a_matches_fantasy_rebuild(setup):
    """A: a directional derivative at an existing design site."""
    x_idx = 2
    predicted, v = gv.gvr_derivative_at_site(
        setup["model"], setup["params"], setup["X"][x_idx], setup["Z"],
        setup["noise_var"])
    assert v is not None and predicted > 0.0

    fantasy = _build(setup["X"], setup["y"], [{
        "x_index": x_idx, "direction": v, "slot": 0,
        "value": float(_grad(setup["X"][x_idx])[0] @ v)}])
    realised = setup["base_gv"] - _integrated_variance(
        fantasy, setup["params"], setup["Z"])

    assert realised == pytest.approx(predicted, rel=RTOL)


def test_option_b_matches_fantasy_rebuild(setup):
    """B: a function value at a new site."""
    x_new = np.array([0.4, -1.1])
    predicted, _, _ = gv.gvr_at_new_site(
        setup["model"], setup["params"], x_new, setup["Z"],
        setup["noise_var"])
    assert predicted > 0.0

    X_aug = np.vstack([setup["X"], x_new[None, :]])
    y_aug = np.vstack([setup["y"], _f(x_new)])
    realised = setup["base_gv"] - _integrated_variance(
        _build(X_aug, y_aug), setup["params"], setup["Z"])

    assert realised == pytest.approx(predicted, rel=RTOL)


def test_option_c_matches_fantasy_rebuild(setup):
    """C: a function value and a directional derivative at a new site."""
    x_new = np.array([0.4, -1.1])
    gvr_b, gvr_c, v = gv.gvr_at_new_site(
        setup["model"], setup["params"], x_new, setup["Z"],
        setup["noise_var"])
    assert v is not None and gvr_c > 0.0

    X_aug = np.vstack([setup["X"], x_new[None, :]])
    y_aug = np.vstack([setup["y"], _f(x_new)])
    new_idx = X_aug.shape[0] - 1
    fantasy = _build(X_aug, y_aug, [{
        "x_index": new_idx, "direction": v, "slot": 0,
        "value": float(_grad(x_new)[0] @ v)}])
    realised = setup["base_gv"] - _integrated_variance(
        fantasy, setup["params"], setup["Z"])

    assert realised == pytest.approx(gvr_c, rel=RTOL)
    assert gvr_c >= gvr_b


def test_reduction_is_independent_of_the_fantasy_values(setup):
    """The whole point: the value observed does not change the variance."""
    x_new = np.array([-0.7, 0.9])
    X_aug = np.vstack([setup["X"], x_new[None, :]])

    reductions = []
    for f_val, d_val in ((0.0, 0.0), (1e3, -250.0)):
        y_aug = np.vstack([setup["y"], [[f_val]]])
        fantasy = _build(X_aug, y_aug, [{
            "x_index": X_aug.shape[0] - 1, "direction": np.array([1.0, 0.0]),
            "slot": 0, "value": d_val}])
        reductions.append(setup["base_gv"] - _integrated_variance(
            fantasy, setup["params"], setup["Z"]))

    assert reductions[0] == pytest.approx(reductions[1], rel=1e-10)


def test_direction_is_optimal(setup):
    """The Rayleigh-quotient direction must beat any other unit direction."""
    x_idx = 1
    best, v = gv.gvr_derivative_at_site(
        setup["model"], setup["params"], setup["X"][x_idx], setup["Z"],
        setup["noise_var"])
    assert v is not None

    *_, g, Sigma_g = gv.joint_blocks(
        setup["model"], setup["params"], setup["X"][x_idx], setup["Z"])
    Sigma = Sigma_g + setup["noise_var"] * np.eye(DIM)
    for theta in np.linspace(0.0, np.pi, 37):
        u = np.array([np.cos(theta), np.sin(theta)])
        value = np.mean((u @ g) ** 2) / float(u @ Sigma @ u)
        assert value <= best * (1.0 + 1e-9)
