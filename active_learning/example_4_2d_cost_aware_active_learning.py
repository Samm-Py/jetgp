"""
Example 4 - Cost-aware active learning on a 2D Branin function.

This script exercises the current active-learning implementation end to end:
initial DOE, cost-aware acquisition, Lanczos-selected derivative candidates,
model refits, final grid RMSE, and simple diagnostic figures.
"""

import os
from pathlib import Path

# The local JetGP checkout can fail at import time when numba tries to create
# cache files from this source layout. The example does not rely on numba JIT.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptive_doe import AdaptiveDirectionalGP
from posterior_queries import query_function_posterior_batched


# ---------------------------------------------------------------------------
# Branin-Hoo test function
# ---------------------------------------------------------------------------

_A = 1.0
_B = 5.1 / (4 * np.pi**2)
_C = 5.0 / np.pi
_R = 6.0
_S = 10.0
_T = 1.0 / (8.0 * np.pi)

BOUNDS = np.array([[-5.0, 10.0], [0.0, 15.0]])


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


def make_grid(n_per_axis):
    x1 = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], n_per_axis)
    x2 = np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], n_per_axis)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack([X1.ravel(), X2.ravel()])
    return X, X1, X2


def rmse(y_pred, y_true):
    return float(np.sqrt(np.mean((y_pred.reshape(-1) - y_true.reshape(-1)) ** 2)))


def summarize_history(history):
    print("\nCost-aware iteration summary:")
    print(f"  {'step':>4}  {'type':>4}  {'order':>5}  {'score':>10}  "
          f"{'rho':>10}  {'cost':>8}  {'cum_cost':>10}  {'n_f':>5}  "
          f"{'n_d1':>6}  {'n_d2':>6}")
    for rec in history:
        order = "-" if rec["chosen_order"] is None else str(rec["chosen_order"])
        print(f"  {rec['step']:4d}  {rec['chosen_type']:>4}  {order:>5}  "
              f"{rec['chosen_score']:10.4f}  {rec['chosen_rho']:10.4f}  "
              f"{rec['chosen_cost']:8.3f}  {rec['cumulative_cost']:10.3f}  "
              f"{rec['n_train']:5d}  {rec['n_directional_obs']:6d}  "
              f"{rec['n_second_order_obs']:6d}")


def save_design_figure(al, figure_dir):
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    X_grid, X1, X2 = make_grid(120)
    values = branin(X_grid).reshape(X1.shape)
    contour = ax.contourf(X1, X2, values, levels=35, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="f(x)")
    ax.scatter(al.X_train[:, 0], al.X_train[:, 1], c="white", s=42,
               edgecolor="black", label="function sites")
    if al.directional_observations:
        for obs in al.directional_observations:
            x = al.X_train[obs["x_index"]]
            v = obs["direction"]
            ax.arrow(x[0], x[1], 0.65 * v[0], 0.65 * v[1],
                     width=0.025, head_width=0.18, color="tab:red",
                     length_includes_head=True, alpha=0.75)
    ax.set_xlim(BOUNDS[0])
    ax.set_ylim(BOUNDS[1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Final cost-aware design")
    ax.legend(loc="upper right")
    fig.savefig(figure_dir / "final_design.png", dpi=180)
    plt.close(fig)


def save_prediction_figure(al, figure_dir, n_grid, batch_size):
    X_grid, X1, X2 = make_grid(n_grid)
    mean_pred, var_pred = query_function_posterior_batched(
        al.gp_model, al.params, X_grid, batch_size)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    fields = [
        (mean_pred.reshape(X1.shape), "Posterior mean", "viridis"),
        (var_pred.reshape(X1.shape), "Posterior variance", "magma"),
    ]
    for ax, (field, title, cmap) in zip(axes, fields):
        contour = ax.contourf(X1, X2, field, levels=35, cmap=cmap)
        fig.colorbar(contour, ax=ax)
        ax.scatter(al.X_train[:, 0], al.X_train[:, 1], c="white", s=28,
                   edgecolor="black")
        ax.set_xlim(BOUNDS[0])
        ax.set_ylim(BOUNDS[1])
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(title)
    fig.savefig(figure_dir / "posterior_mean_variance.png", dpi=180)
    plt.close(fig)


def run_example(args):
    X_test, _, _ = make_grid(args.test_grid)
    y_test = branin(X_test)

    al = AdaptiveDirectionalGP(
        func=branin,
        grad_func=branin_grad,
        bounds=BOUNDS,
        n_init=args.n_init,
        rel_tol=args.rel_tol,
        n_iter=args.n_iter,
        kernel="SE",
        kernel_type="anisotropic",
        seed=args.seed,
        c_f=args.cost_function,
        c1=args.cost_gradient,
        max_directions=args.max_directions,
        cost_budget=args.cost_budget,
        test_set=(X_test, y_test),
        predict_batch_size=args.predict_batch_size,
        optimizer_kwargs={
            "pop_size": args.pop_size,
            "n_generations": args.n_generations,
            "local_opt_every": args.n_generations,
            "debug": False,
        },
    )

    history = al.run()
    summarize_history(history)

    mean_pred, _ = query_function_posterior_batched(
        al.gp_model, al.params, X_test, args.predict_batch_size)
    final_rmse = rmse(mean_pred, y_test)
    print(f"\nFinal grid RMSE: {final_rmse:.6f}")

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_design_figure(al, figure_dir)
    save_prediction_figure(al, figure_dir, args.plot_grid,
                           args.predict_batch_size)
    print(f"Saved figures to: {figure_dir.resolve()}")

    return al, history, final_rmse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cost-aware active learning on a 2D Branin function.")
    parser.add_argument("--n-init", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=6)
    parser.add_argument("--rel-tol", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cost-function", type=float, default=1.0)
    parser.add_argument("--cost-gradient", type=float, default=0.25)
    parser.add_argument("--max-directions", type=int, default=1)
    parser.add_argument("--cost-budget", type=float, default=None)
    parser.add_argument("--test-grid", type=int, default=35)
    parser.add_argument("--plot-grid", type=int, default=90)
    parser.add_argument("--predict-batch-size", type=int, default=250)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--n-generations", type=int, default=5)
    parser.add_argument("--figure-dir", default="example_4_2d_figures")
    return parser.parse_args()


if __name__ == "__main__":
    run_example(parse_args())
