"""
Example 3 - Efficient derivative-enhanced GP construction on a 6D active-subspace problem.

This example compares three surrogate construction strategies on a 6D
problem with a known 2D active subspace:

1. Function-only GP.
2. Full-gradient derivative-enhanced GP (DEGP).
3. Eigenbasis directional derivative GDDEGP.

All methods use the same function evaluation locations. The locations are
selected by maximum predictive variance from the directional GDDEGP, while the
full-gradient and function-only models receive the same function values. This
isolates how efficiently each method uses derivative information at the same
expensive sample locations.
"""

from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube

from adaptive_doe import (
    find_next_point_mpv,
    fit_directional_gp,
    fit_function_only_gp,
    sequential_initial_derivative_enrichment,
    select_derivatives_at_xnew,
)
from jetgp.full_degp.degp import degp
from plotting_utils.example_3_plotting_utils import (
    configure_plotting,
    save_example_3_figures,
)


ACTIVE6_BOUNDS = np.array([[-1.0, 1.0]] * 6)

# Two orthonormal active directions in R^6.
_w1 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_w1 = _w1 / np.linalg.norm(_w1)

_w2 = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 1.0])
_w2 = _w2 / np.linalg.norm(_w2)


OPTIMIZER_KWARGS = {
    "optimizer": "jade",
    "pop_size": 50,
    "n_generations": 15,
    "local_opt_every": 15,
    "debug": False,
}


def active_subspace6(X):
    """
    Smooth 6D function with a known 2D active subspace.

    The response depends only on two linear combinations of the six inputs:
        z1 = w1^T x
        z2 = w2^T x

    f(x) = sin(2 pi z1) + 0.3 cos(pi z2) + 0.2 z1 z2
    """
    X = np.atleast_2d(X)
    z1 = X @ _w1
    z2 = X @ _w2

    values = (
        np.sin(2.0 * np.pi * z1)
        + 0.3 * np.cos(np.pi * z2)
        + 0.2 * z1 * z2
    )
    return values.reshape(-1, 1)


def active_subspace6_grad(X):
    """
    Gradient of the 6D active-subspace test function.
    """
    X = np.atleast_2d(X)
    z1 = X @ _w1
    z2 = X @ _w2

    grad = (
        (2.0 * np.pi * np.cos(2.0 * np.pi * z1) + 0.2 * z2)[:, None] * _w1[None, :]
        + (-0.3 * np.pi * np.sin(np.pi * z2) + 0.2 * z1)[:, None] * _w2[None, :]
    )
    return grad


def lhs_design(n_points, bounds, seed=42):
    d = bounds.shape[0]
    sampler = LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n_points)
    lb, ub = bounds[:, 0], bounds[:, 1]
    return lb + unit_samples * (ub - lb)


def fit_full_gradient_degp(X_train, y_train, gradient_observations,
                           kernel="SE", kernel_type="anisotropic"):
    """
    Fit a DEGP with coordinate derivative observations at selected locations.

    gradient_observations is a list of dicts:
        {"x_index": int, "gradient": ndarray shape (d,)}
    """
    d = X_train.shape[1]
    if len(gradient_observations) == 0:
        return fit_function_only_gp(
            X_train,
            y_train,
            n_dir_types=d,
            kernel=kernel,
            kernel_type=kernel_type,
            optimizer_kwargs=OPTIMIZER_KWARGS,
        )

    y_blocks = [y_train]
    derivative_locations = []
    for j in range(d):
        values = np.array([[obs["gradient"][j]] for obs in gradient_observations])
        y_blocks.append(values)
        derivative_locations.append([obs["x_index"] for obs in gradient_observations])

    der_indices = [[[[j + 1, 1]] for j in range(d)]]
    model = degp(
        X_train,
        y_blocks,
        n_order=1,
        n_bases=d,
        der_indices=der_indices,
        derivative_locations=derivative_locations,
        normalize=True,
        kernel=kernel,
        kernel_type=kernel_type,
    )
    params = model.optimize_hyperparameters(**OPTIMIZER_KWARGS)
    return model, params


def fit_directional_model(X_train, y_train, directional_observations,
                          kernel="SE", kernel_type="anisotropic"):
    """Fit function-only initially, then GDDEGP once directions exist."""
    if len(directional_observations) == 0:
        return fit_function_only_gp(
            X_train,
            y_train,
            n_dir_types=X_train.shape[1],
            kernel=kernel,
            kernel_type=kernel_type,
            optimizer_kwargs=OPTIMIZER_KWARGS,
        )
    return fit_directional_gp(
        X_train,
        y_train,
        directional_observations,
        kernel=kernel,
        kernel_type=kernel_type,
        optimizer_kwargs=OPTIMIZER_KWARGS,
    )


def predict_function_mean(model, params, X):
    pred = model.predict(np.atleast_2d(X), params, calc_cov=False, return_deriv=False)
    if isinstance(pred, tuple):
        pred = pred[0]
    return np.asarray(pred)[0].reshape(-1)


def evaluate_model(model, params, X_val, y_val):
    pred = predict_function_mean(model, params, X_val)
    err = pred - y_val.reshape(-1)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.max(np.abs(err))),
        "mean_abs_error": float(np.mean(np.abs(err))),
    }


def append_metrics(results, method, model, params, X_val, y_val,
                   n_function_evals, n_derivative_obs, iteration):
    metrics = evaluate_model(model, params, X_val, y_val)
    results.append({
        "method": method,
        "iteration": iteration,
        "n_function_evals": n_function_evals,
        "n_derivative_obs": n_derivative_obs,
        **metrics,
    })


def print_iteration_summary(results, iteration):
    print(f"\nValidation metrics after iteration {iteration}:")
    for row in [r for r in results if r["iteration"] == iteration]:
        print(
            f"  {row['method']:<32} "
            f"RMSE={row['rmse']:.5f}, "
            f"max|err|={row['max_abs_error']:.5f}, "
            f"N_f={row['n_function_evals']}, "
            f"N_deriv={row['n_derivative_obs']}"
        )


if __name__ == "__main__":
    configure_plotting()

    d = ACTIVE6_BOUNDS.shape[0]
    n_init = 10 * d
    n_iter = 25
    tau = 0.025
    n_val = 10000
    seed = 123

    print("=" * 72)
    print("Example 3 - 6D active-subspace derivative-efficiency demonstration")
    print("=" * 72)

    X_initial = lhs_design(n_init, ACTIVE6_BOUNDS, seed=seed)
    y_initial = active_subspace6(X_initial)
    X_val = lhs_design(n_val, ACTIVE6_BOUNDS, seed=seed + 1000)
    y_val = active_subspace6(X_val)

    X_function = X_initial.copy()
    y_function = y_initial.copy()

    X_full = X_initial.copy()
    y_full = y_initial.copy()
    full_gradient_observations = []

    X_directional = X_initial.copy()
    y_directional = y_initial.copy()
    directional_observations = []

    results = []
    direction_counts = []

    print(f"Initial DOE: {n_init} points in {d}D")
    print(f"Validation set: {n_val} LHS points")

    print("\nFitting initial models...")
    function_model, function_params = fit_function_only_gp(
        X_function,
        y_function,
        n_dir_types=d,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        normalize=True,
    )
    directional_model, directional_params = fit_directional_model(
        X_directional, y_directional, directional_observations)

    print("Selecting directional derivatives on the initial DOE...")
    directional_observations, initial_selection_records = sequential_initial_derivative_enrichment(
        directional_model,
        directional_params,
        X_directional,
        y_directional,
        active_subspace6_grad,
        tau=tau,
        kernel="SE",
        kernel_type="anisotropic",
    )
    initial_direction_counts = [
        len(record["selected_directions"]) for record in initial_selection_records
    ]
    full_gradients_initial = active_subspace6_grad(X_full)
    full_gradient_observations = [
        {"x_index": i, "gradient": full_gradients_initial[i].copy()}
        for i in range(X_full.shape[0])
    ]

    print(f"Initial DOE directional counts per point: {initial_direction_counts}")
    print(f"Initial DOE directional observations: {len(directional_observations)}")
    print(f"Initial DOE full-gradient observations: {d * len(full_gradient_observations)}")

    full_model, full_params = fit_full_gradient_degp(
        X_full, y_full, full_gradient_observations)
    directional_model, directional_params = fit_directional_model(
        X_directional, y_directional, directional_observations)

    append_metrics(
        results, "Function-only GP", function_model, function_params,
        X_val, y_val, n_init, 0, iteration=0)
    append_metrics(
        results, "Full-gradient DEGP", full_model, full_params,
        X_val, y_val, n_init, d * len(full_gradient_observations), iteration=0)
    append_metrics(
        results, "Eigenbasis directional GDDEGP", directional_model, directional_params,
        X_val, y_val, n_init, len(directional_observations), iteration=0)
    print_iteration_summary(results, iteration=0)

    for iteration in range(1, n_iter + 1):
        print(f"\n{'-' * 72}")
        print(f"Adaptive iteration {iteration}/{n_iter}")
        print(f"{'-' * 72}")

        x_new, mpv = find_next_point_mpv(
            directional_model,
            directional_params,
            ACTIVE6_BOUNDS,
            n_restarts=14,
            seed=seed + iteration,
        )
        selection = select_derivatives_at_xnew(
            directional_model,
            directional_params,
            X_directional,
            y_directional,
            x_new,
            tau=tau,
        )
        f_new = float(active_subspace6(np.atleast_2d(x_new))[0, 0])
        grad_new = active_subspace6_grad(np.atleast_2d(x_new))[0]

        print(f"Selected x_new = {np.round(x_new, 4)}")
        print(f"sigma_f^2(x_new) = {mpv:.6f}")
        print(f"Selected {len(selection['selected_directions'])} directional derivatives")
        print(f"lambda_j/lambda_1 = {np.round(selection['variance_ratios'], 4)}")

        # Update shared function locations.
        new_index = X_function.shape[0]
        X_function = np.vstack([X_function, np.atleast_2d(x_new)])
        y_function = np.vstack([y_function, np.array([[f_new]])])
        X_full = np.vstack([X_full, np.atleast_2d(x_new)])
        y_full = np.vstack([y_full, np.array([[f_new]])])
        X_directional = np.vstack([X_directional, np.atleast_2d(x_new)])
        y_directional = np.vstack([y_directional, np.array([[f_new]])])

        full_gradient_observations.append({
            "x_index": new_index,
            "gradient": grad_new.copy(),
        })

        for slot, direction in enumerate(selection["selected_directions"]):
            d_val = float(grad_new @ direction)
            directional_observations.append({
                "x_index": new_index,
                "direction": direction.copy(),
                "value": d_val,
                "slot": slot,
            })
        direction_counts.append(len(selection["selected_directions"]))

        print("Refitting models...")
        function_model, function_params = fit_function_only_gp(
            X_function, y_function, n_dir_types=d,
            optimizer_kwargs=OPTIMIZER_KWARGS,
            normalize=True,
        )
        full_model, full_params = fit_full_gradient_degp(
            X_full, y_full, full_gradient_observations)
        directional_model, directional_params = fit_directional_model(
            X_directional, y_directional, directional_observations)

        n_function_evals = X_function.shape[0]
        append_metrics(
            results, "Function-only GP", function_model, function_params,
            X_val, y_val, n_function_evals, 0, iteration)
        append_metrics(
            results, "Full-gradient DEGP", full_model, full_params,
            X_val, y_val, n_function_evals,
            d * len(full_gradient_observations), iteration)
        append_metrics(
            results, "Eigenbasis directional GDDEGP",
            directional_model, directional_params,
            X_val, y_val, n_function_evals,
            len(directional_observations), iteration)
        print_iteration_summary(results, iteration)

    figure_dir = Path("example_3_figures")
    save_example_3_figures(results, direction_counts, d, figure_dir)
    print(f"\nSaved example 3 figures to: {figure_dir.resolve()}")
