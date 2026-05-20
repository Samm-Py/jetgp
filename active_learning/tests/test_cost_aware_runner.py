import numpy as np

import adaptive_doe
import gp_builders
import posterior_queries
from adaptive_doe import AdaptiveDirectionalGP
from candidates import CostAwareCandidate


def bare_runner():
    runner = object.__new__(AdaptiveDirectionalGP)
    runner.X_train = np.array([[0.0, 0.0]])
    runner.y_train = np.array([[0.0]])
    runner.directional_observations = []
    runner.second_order_observations = []
    runner.params = np.array([1.0, 2.0])
    runner.test_set = None
    runner.verbose = True
    runner.log_weight_fn = None
    return runner


class FakeFunctionGP:
    def __init__(self):
        self.batch_sizes = []

    def predict(self, X, params, calc_cov=True, return_deriv=False):
        self.batch_sizes.append(X.shape[0])
        mean = np.sum(X, axis=1)
        var = np.full(X.shape[0], 0.5)
        return [mean], [var]


def test_choose_candidate_uses_highest_score():
    low = CostAwareCandidate(kind="f", score=0.5, rho=0.5, cost=1.0)
    high = CostAwareCandidate(kind="d", score=1.25, rho=2.5, cost=2.0)

    chosen = AdaptiveDirectionalGP._choose_candidate([low, high])

    assert chosen is high


def test_candidates_within_budget_filters_by_remaining_cost():
    runner = bare_runner()
    runner.cost_budget = 1.0
    affordable = CostAwareCandidate(kind="d", score=0.5, rho=0.5, cost=0.25)
    too_expensive = CostAwareCandidate(kind="f", score=10.0, rho=10.0, cost=1.0)

    candidates = runner._candidates_within_budget(
        [too_expensive, affordable], cumulative_cost=0.75)

    assert candidates == [affordable]


def test_candidates_within_budget_returns_all_without_budget():
    runner = bare_runner()
    runner.cost_budget = None
    candidates = [
        CostAwareCandidate(kind="d", score=0.5, rho=0.5, cost=0.25),
        CostAwareCandidate(kind="f", score=10.0, rho=10.0, cost=1.0),
    ]

    assert runner._candidates_within_budget(candidates, 0.75) == candidates


def test_query_function_posterior_batched_splits_large_inputs():
    gp_model = FakeFunctionGP()
    X = np.arange(10.0).reshape(5, 2)

    mean, var = posterior_queries.query_function_posterior_batched(
        gp_model, params=np.array([1.0]), X_test=X, batch_size=2)

    assert gp_model.batch_sizes == [2, 2, 1]
    assert np.allclose(mean, np.sum(X, axis=1))
    assert np.allclose(var, np.full(5, 0.5))


def test_test_rmse_uses_predict_batch_size():
    runner = bare_runner()
    runner.gp_model = FakeFunctionGP()
    runner.predict_batch_size = 2
    X_test = np.arange(10.0).reshape(5, 2)
    y_test = np.sum(X_test, axis=1)
    runner.test_set = (X_test, y_test)

    assert runner._test_rmse() == 0.0
    assert runner.gp_model.batch_sizes == [2, 2, 1]


def test_make_ray_rejects_zero_direction():
    with np.testing.assert_raises(ValueError):
        posterior_queries._make_ray(np.array([0.0, 0.0]), n_points=3)


def test_make_ray_rejects_nonfinite_direction():
    with np.testing.assert_raises(ValueError):
        posterior_queries._make_ray(np.array([1.0, np.nan]), n_points=3)


def test_apply_function_candidate_updates_training_arrays_and_returns_cost():
    runner = bare_runner()
    runner.c_f = 2.5
    runner.func = lambda X: np.sum(X, axis=1, keepdims=True)
    candidate = CostAwareCandidate(
        kind="f", x=np.array([1.0, 2.0]), score=1.0, rho=1.0, cost=2.5)

    cost = runner._apply_candidate(candidate)

    assert cost == 2.5
    assert np.allclose(runner.X_train, [[0.0, 0.0], [1.0, 2.0]])
    assert np.allclose(runner.y_train, [[0.0], [3.0]])


def test_verbose_false_suppresses_apply_candidate_output(capsys):
    runner = bare_runner()
    runner.verbose = False
    runner.c_f = 2.5
    runner.func = lambda X: np.sum(X, axis=1, keepdims=True)
    candidate = CostAwareCandidate(
        kind="f", x=np.array([1.0, 2.0]), score=1.0, rho=1.0, cost=2.5)

    cost = runner._apply_candidate(candidate)

    assert cost == 2.5
    assert capsys.readouterr().out == ""


def test_apply_second_order_derivative_records_first_and_second_observations():
    runner = bare_runner()
    runner.grad_func = lambda X: np.array([[3.0, 4.0]])
    runner.hess_func = lambda X: np.array([[2.0, 0.0], [0.0, 6.0]])
    candidate = CostAwareCandidate(
        kind="d",
        x_idx=0,
        direction=np.array([1.0, 0.0]),
        order=2,
        score=1.0,
        rho=1.0,
        cost=4.0,
    )

    cost = runner._apply_candidate(candidate)

    assert cost == 4.0
    assert len(runner.directional_observations) == 1
    assert len(runner.second_order_observations) == 1
    assert runner.directional_observations[0]["x_index"] == 0
    assert runner.directional_observations[0]["slot"] == 0
    assert runner.directional_observations[0]["value"] == 3.0
    assert runner.second_order_observations[0]["slot"] == 0
    assert runner.second_order_observations[0]["value"] == 2.0


def test_history_record_reports_candidate_and_current_counts():
    runner = bare_runner()
    runner.directional_observations.append({
        "x_index": 0,
        "direction": np.array([1.0, 0.0]),
        "value": 3.0,
        "slot": 0,
    })
    candidate = CostAwareCandidate(
        kind="d",
        x_idx=0,
        direction=np.array([1.0, 0.0]),
        order=1,
        score=0.75,
        rho=1.5,
        cost=2.0,
    )

    record = runner._history_record(step=3, candidate=candidate, cumulative_cost=5.0)

    assert record["step"] == 3
    assert record["chosen_type"] == "d"
    assert record["chosen_score"] == 0.75
    assert record["chosen_rho"] == 1.5
    assert record["chosen_cost"] == 2.0
    assert record["chosen_x_idx"] == 0
    assert np.allclose(record["chosen_direction"], [1.0, 0.0])
    assert record["chosen_order"] == 1
    assert record["cumulative_cost"] == 5.0
    assert record["n_train"] == 1
    assert record["n_directional_obs"] == 1
    assert record["n_second_order_obs"] == 0
    assert np.allclose(record["params"], [1.0, 2.0])
    assert record["rmse_test"] is None


def test_derivative_candidates_apply_threshold_costs_and_duplicate_filter():
    runner = bare_runner()
    runner.rel_tol = 0.2
    runner.c1 = 2.0
    runner.c2 = 5.0
    runner.acquire_second_order = True
    runner.hess_func = lambda X: np.eye(2)
    runner.directional_observations.append({
        "x_index": 0,
        "direction": np.array([0.0, 1.0]),
        "value": 0.0,
        "slot": 0,
    })
    runner._get_spectrum = lambda x_idx: {
        "eigvals": np.array([6.0, 4.0]),
        "eigvecs": np.eye(2),
        "var2": np.array([9.0, 20.0]),
    }

    candidates = runner._derivative_candidates_at(
        x_idx=0, lam_prior_grad=10.0, lam_prior_hess=10.0)

    assert [(c.kind, c.order, c.score, c.rho) for c in candidates] == [
        ("d", 1, 0.3, 0.6),
    ]
    assert np.allclose(candidates[0].direction, [1.0, 0.0])


def test_construct_directional_gp_preserves_second_order_slot_indices(monkeypatch):
    captured = {}

    def fake_gddegp(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(gp_builders, "gddegp", fake_gddegp)

    X_train = np.array([[0.0, 0.0]])
    y_train = np.array([[0.0]])
    directional_observations = [
        {
            "x_index": 0,
            "direction": np.array([1.0, 0.0]),
            "value": 1.0,
            "slot": 0,
        },
        {
            "x_index": 0,
            "direction": np.array([0.0, 1.0]),
            "value": 2.0,
            "slot": 1,
        },
    ]
    second_order_observations = [
        {
            "x_index": 0,
            "direction": np.array([0.0, 1.0]),
            "value": 3.0,
            "slot": 1,
        },
    ]

    gp_builders._construct_directional_gp(
        X_train,
        y_train,
        directional_observations,
        second_order_observations=second_order_observations,
    )

    assert captured["kwargs"]["der_indices"] == [
        [[[1, 1]], [[2, 1]], [[2, 2]]]
    ]
