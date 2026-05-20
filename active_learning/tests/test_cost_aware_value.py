"""Integration checks that cost-aware acquisition responds to the cost model."""

import numpy as np

from adaptive_doe import AdaptiveDirectionalGP
from doe_utils import lhs_design
from posterior_queries import query_function_posterior_batched


_A = 1.0
_B = 5.1 / (4 * np.pi**2)
_C = 5.0 / np.pi
_R = 6.0
_S = 10.0
_T = 1.0 / (8.0 * np.pi)

BOUNDS = np.array([[-5.0, 10.0], [0.0, 15.0]])
OPT_KWARGS = {
    "pop_size": 10,
    "n_generations": 5,
    "local_opt_every": 5,
    "debug": False,
}


def branin(X):
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    term = x2 - _B * x1**2 + _C * x1 - _R
    y = _A * term**2 + _S * (1 - _T) * np.cos(x1) + _S
    return y.reshape(-1, 1)


def branin_grad(X):
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    term = x2 - _B * x1**2 + _C * x1 - _R
    df_dx1 = 2 * _A * term * (-2 * _B * x1 + _C) - _S * (1 - _T) * np.sin(x1)
    df_dx2 = 2 * _A * term
    return np.column_stack([df_dx1, df_dx2])


def make_grid(n_per_axis=18):
    x1 = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], n_per_axis)
    x2 = np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], n_per_axis)
    X1, X2 = np.meshgrid(x1, x2)
    return np.column_stack([X1.ravel(), X2.ravel()])


class CandidateFilteredAdaptiveGP(AdaptiveDirectionalGP):
    """Restrict the policy to one candidate kind for baseline comparisons."""

    allowed_kind = None

    def _build_cost_aware_candidates(self):
        candidates = super()._build_cost_aware_candidates()
        if self.allowed_kind is None:
            return candidates
        return [c for c in candidates if c.kind == self.allowed_kind]


class FunctionOnlyAdaptiveGP(CandidateFilteredAdaptiveGP):
    allowed_kind = "f"


class DerivativeOnlyAdaptiveGP(CandidateFilteredAdaptiveGP):
    allowed_kind = "d"


def run_policy(policy_cls, c_f, c1, budget, n_iter):
    X_test = make_grid()
    y_test = branin(X_test)
    X_init = lhs_design(5, BOUNDS, seed=123)

    al = policy_cls(
        func=branin,
        grad_func=branin_grad,
        bounds=BOUNDS,
        n_init=5,
        rel_tol=0.001,
        n_iter=n_iter,
        seed=5,
        c_f=c_f,
        c1=c1,
        cost_budget=budget,
        max_directions=2,
        X_init=X_init,
        test_set=(X_test, y_test),
        predict_batch_size=128,
        verbose=False,
        optimizer_kwargs=OPT_KWARGS,
    )
    history = al.run()
    mean, _ = query_function_posterior_batched(
        al.gp_model, al.params, X_test, batch_size=128)
    rmse = float(np.sqrt(np.mean((mean - y_test.reshape(-1)) ** 2)))
    return al, history, rmse


def test_cost_aware_beats_function_only_when_derivatives_are_cheaper():
    cost_aware, cost_history, cost_rmse = run_policy(
        AdaptiveDirectionalGP, c_f=10.0, c1=1.0, budget=10.0, n_iter=10)
    function_only, function_history, function_rmse = run_policy(
        FunctionOnlyAdaptiveGP, c_f=10.0, c1=1.0, budget=10.0, n_iter=10)

    assert all(rec["chosen_type"] == "d" for rec in cost_history)
    assert len(cost_aware.directional_observations) > 0
    assert len(function_only.directional_observations) == 0
    assert function_history[-1]["cumulative_cost"] == 10.0
    assert cost_rmse < 0.8 * function_rmse


def test_cost_aware_beats_derivative_only_when_functions_are_cheaper():
    cost_aware, cost_history, cost_rmse = run_policy(
        AdaptiveDirectionalGP, c_f=1.0, c1=10.0, budget=4.0, n_iter=4)
    derivative_only, derivative_history, derivative_rmse = run_policy(
        DerivativeOnlyAdaptiveGP, c_f=1.0, c1=10.0, budget=4.0, n_iter=4)

    assert all(rec["chosen_type"] == "f" for rec in cost_history)
    assert cost_aware.X_train.shape[0] > derivative_only.X_train.shape[0]
    assert len(derivative_history) == 0
    assert len(cost_aware.directional_observations) == 0
    assert cost_rmse < derivative_rmse - 1.0
