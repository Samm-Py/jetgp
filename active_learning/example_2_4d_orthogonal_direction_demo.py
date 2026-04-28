"""
Example 2 - 4D orthogonal direction demonstration.

This example uses the AdaptiveDirectionalGP class to select an infill point
and the local eigenbasis of directional derivative uncertainty in a 4D test
problem. It then compares those eigen-directions against directions obtained
by direct optimization of the directional derivative variance in successive
orthogonal subspaces.
"""

import numpy as np
from pathlib import Path
from scipy.optimize import minimize

from adaptive_doe import AdaptiveDirectionalGP
from plotting_utils.example_2_plotting_utils import (
    configure_plotting,
    save_direction_validation_figures,
)


ACKLEY4_BOUNDS = np.array([[-2.0, 2.0]] * 4)


def ackley4(X):
    """Four-dimensional Ackley function, returned as shape (n, 1)."""
    X = np.atleast_2d(X)
    d = X.shape[1]
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi

    sum_sq = np.sum(X**2, axis=1)
    mean_sq = sum_sq / d
    mean_cos = np.mean(np.cos(c * X), axis=1)
    values = -a * np.exp(-b * np.sqrt(mean_sq)) - np.exp(mean_cos) + a + np.e
    return values.reshape(-1, 1)


def ackley4_grad(X):
    """Analytic gradient of the four-dimensional Ackley function."""
    X = np.atleast_2d(X)
    d = X.shape[1]
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi

    sum_sq = np.sum(X**2, axis=1)
    mean_sq = sum_sq / d
    sqrt_mean_sq = np.sqrt(mean_sq)
    exp_radial = np.exp(-b * sqrt_mean_sq)
    mean_cos = np.mean(np.cos(c * X), axis=1)
    exp_cos = np.exp(mean_cos)

    grad = np.zeros_like(X)
    nonzero = sqrt_mean_sq > 1e-14
    grad[nonzero] += (
        a * b * exp_radial[nonzero, None]
        * X[nonzero]
        / (d * sqrt_mean_sq[nonzero, None])
    )
    grad += exp_cos[:, None] * (c / d) * np.sin(c * X)
    return grad


def normalize(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("Cannot normalize the zero vector.")
    return v / norm


def orthogonal_complement_basis(selected, dim):
    """Return an orthonormal basis for the subspace orthogonal to selected."""
    if len(selected) == 0:
        return np.eye(dim)

    q_selected, _ = np.linalg.qr(np.column_stack(selected))
    project_away = np.eye(dim) - q_selected @ q_selected.T
    u, singular_values, _ = np.linalg.svd(project_away)
    rank = np.sum(singular_values > 1e-10)
    return u[:, :rank]


def canonicalize_direction(direction, reference):
    direction = normalize(direction)
    reference = normalize(reference)
    return direction if np.dot(direction, reference) >= 0.0 else -direction


def directional_variance(cov, direction):
    direction = normalize(direction)
    return float(direction @ cov @ direction)


def optimize_direction_in_subspace(cov, basis, n_restarts=32, seed=2026):
    """
    Maximize v^T C v over unit vectors v in span(basis).

    The solution should align with the leading eigenvector of C restricted to
    this subspace. We use numerical optimization here only as a validation
    comparison for the eigenbasis method.
    """
    dim = basis.shape[1]
    if dim == 1:
        v = normalize(basis[:, 0])
        return v, directional_variance(cov, v)

    rng = np.random.default_rng(seed)
    starts = rng.normal(size=(n_restarts, dim))
    starts /= np.linalg.norm(starts, axis=1, keepdims=True)

    best_y = None
    best_obj = np.inf

    def objective(y):
        y = normalize(y)
        v = basis @ y
        return -directional_variance(cov, v)

    constraint = {"type": "eq", "fun": lambda y: np.dot(y, y) - 1.0}
    for start in starts:
        result = minimize(
            objective,
            x0=start,
            method="SLSQP",
            constraints=[constraint],
            options={"ftol": 1e-12, "maxiter": 300, "disp": False},
        )
        y_candidate = normalize(result.x)
        obj_candidate = objective(y_candidate)
        if obj_candidate < best_obj:
            best_obj = obj_candidate
            best_y = y_candidate

    v = normalize(basis @ best_y)
    return v, -best_obj


def format_vector(v):
    return "[" + ", ".join(f"{value: .4f}" for value in v) + "]"


def print_matrix(matrix, precision=4):
    with np.printoptions(precision=precision, suppress=True):
        print(matrix)


if __name__ == "__main__":
    configure_plotting()

    print("=" * 72)
    print("Example 2 - 4D orthogonal direction demonstration")
    print("=" * 72)

    d = ACKLEY4_BOUNDS.shape[0]
    al = AdaptiveDirectionalGP(
        func=ackley4,
        grad_func=ackley4_grad,
        bounds=ACKLEY4_BOUNDS,
        n_init=5 * d,
        tau=0.0,
        n_iter=1,
        kernel="SE",
        kernel_type="anisotropic",
        seed=11,
    )
    history = al.run()
    rec = history[0]

    cov = rec["derivative_covariance"]
    eigvals = rec["eigenvalues"]
    eigvecs = rec["eigenvectors"]
    selected_dirs = rec["selected_directions"]

    print("\nSelected infill point:")
    print(f"  x_new = {format_vector(rec['x_new'])}")
    print(f"  sigma_f^2(x_new) = {rec['mpv']:.6f}")

    print("\nLocal derivative covariance eigenvalues:")
    print(f"  lambda = {format_vector(eigvals)}")
    print(f"  lambda_j/lambda_1 = {format_vector(rec['variance_ratios'])}")

    print("\nEigen-directions selected by AdaptiveDirectionalGP:")
    for idx, (direction, variance) in enumerate(
            zip(selected_dirs, rec["selected_variances"]), start=1):
        print(f"  q{idx} = {format_vector(direction)}")
        print(f"       Var[d_q{idx} f(x_new)] = {variance:.6f}")

    gram = np.abs(np.column_stack(selected_dirs).T @ np.column_stack(selected_dirs))
    print("\nAbsolute dot-product matrix of selected directions:")
    print_matrix(gram)

    print("\nOptimizer validation in successive orthogonal subspaces:")
    optimizer_dirs = []
    for idx in range(d):
        basis = orthogonal_complement_basis(optimizer_dirs, d)
        opt_dir, opt_var = optimize_direction_in_subspace(
            cov,
            basis,
            seed=3000 + idx,
        )
        eig_dir = eigvecs[:, idx]
        opt_dir = canonicalize_direction(opt_dir, eig_dir)
        optimizer_dirs.append(opt_dir)

        print(f"\n  Direction {idx + 1}:")
        print(f"    optimizer direction = {format_vector(opt_dir)}")
        print(f"    eigen direction     = {format_vector(eig_dir)}")
        print(f"    optimizer variance  = {opt_var:.6f}")
        print(f"    eigenvalue          = {eigvals[idx]:.6f}")
        print(f"    |dot(opt, eig)|     = {abs(float(np.dot(opt_dir, eig_dir))):.6f}")
        for prev_idx, prev_dir in enumerate(optimizer_dirs[:-1], start=1):
            print(
                f"    |dot(opt, opt_{prev_idx})| = "
                f"{abs(float(np.dot(opt_dir, prev_dir))):.6f}"
            )

    opt_gram = np.abs(np.column_stack(optimizer_dirs).T @ np.column_stack(optimizer_dirs))
    print("\nAbsolute dot-product matrix of optimizer directions:")
    print_matrix(opt_gram)

    figure_dir = Path("example_2_figures")
    save_direction_validation_figures(
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        optimizer_directions=optimizer_dirs,
        figure_dir=figure_dir,
    )
    print(f"\nSaved example 2 figures to: {figure_dir.resolve()}")

    print("\nConclusion:")
    print("  The numerical optimizer recovers the same orthogonal directions as")
    print("  the eigenvectors of the local derivative covariance matrix.")
