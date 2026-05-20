"""Generate figures for the Branin-Hoo active-learning tutorial."""

import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

THIS_FILE = Path(__file__).resolve()
# THIS_FILE: docs/source/JetGP_module_examples/active_learning/<script>.py
# parents:    [0]active_learning  [1]JetGP_module_examples  [2]source
#             [3]docs             [4]<repo root>
REPO_ROOT = THIS_FILE.parents[4]
ACTIVE_LEARNING_DIR = REPO_ROOT / "active_learning"
if str(ACTIVE_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVE_LEARNING_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptive_doe import AdaptiveDirectionalGP
from posterior_queries import query_function_posterior_batched


A = 1.0
B = 5.1 / (4 * np.pi**2)
C = 5.0 / np.pi
R = 6.0
S = 10.0
T = 1.0 / (8.0 * np.pi)

BOUNDS = np.array([[-5.0, 10.0], [0.0, 15.0]])


def branin(X):
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    term = x2 - B * x1**2 + C * x1 - R
    y = A * term**2 + S * (1 - T) * np.cos(x1) + S
    return y.reshape(-1, 1)


def branin_grad(X):
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    term = x2 - B * x1**2 + C * x1 - R
    df_dx1 = 2 * A * term * (-2 * B * x1 + C) - S * (1 - T) * np.sin(x1)
    df_dx2 = 2 * A * term
    return np.column_stack([df_dx1, df_dx2])


def branin_hess(X):
    X = np.atleast_2d(X)
    x1, x2 = X[0, 0], X[0, 1]
    term = x2 - B * x1**2 + C * x1 - R
    dterm_dx1 = -2 * B * x1 + C
    d2term_dx1 = -2 * B
    h11 = 2 * A * (dterm_dx1**2 + term * d2term_dx1) - S * (1 - T) * np.cos(x1)
    h12 = 2 * A * dterm_dx1
    h22 = 2 * A
    return np.array([[h11, h12], [h12, h22]])


def make_grid(n_per_axis):
    x1 = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], n_per_axis)
    x2 = np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], n_per_axis)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack([X1.ravel(), X2.ravel()])
    return X, X1, X2


def capture_run(al, n_iter):
    """Run the cost-aware loop while storing design snapshots."""
    al._initialize_design()
    snapshots = [{
        "label": "Initial design",
        "X": al.X_train.copy(),
        "d1": [],
        "d2": [],
    }]

    cumulative_cost = 0.0
    for step in range(1, n_iter + 1):
        candidates = al._candidates_within_budget(
            al._build_cost_aware_candidates(), cumulative_cost)
        if not candidates:
            break
        best = al._choose_candidate(candidates)
        cumulative_cost += al._apply_candidate(best)
        al._refit()
        al.history.append(al._history_record(step, best, cumulative_cost))
        snapshots.append({
            "label": f"After step {step}: {best.kind}^{best.order or ''}",
            "X": al.X_train.copy(),
            "d1": [
                {"x": al.X_train[o["x_index"]].copy(),
                 "v": o["direction"].copy()}
                for o in al.directional_observations
            ],
            "d2": [
                {"x": al.X_train[o["x_index"]].copy(),
                 "v": o["direction"].copy()}
                for o in al.second_order_observations
            ],
        })
    return snapshots


def plot_sequence(snapshots, figure_path):
    X_grid, X1, X2 = make_grid(140)
    truth = branin(X_grid).reshape(X1.shape)
    selected = [snapshots[i] for i in [0, 1, 2, min(4, len(snapshots) - 1),
                                       len(snapshots) - 1]]
    # Drop duplicate snapshots when the run is short.
    deduped = []
    seen = set()
    for snap in selected:
        key = snap["label"]
        if key not in seen:
            deduped.append(snap)
            seen.add(key)

    fig, axes = plt.subplots(1, len(deduped), figsize=(4.0 * len(deduped), 3.7),
                             constrained_layout=True)
    if len(deduped) == 1:
        axes = [axes]
    for ax, snap in zip(axes, deduped):
        ax.contourf(X1, X2, truth, levels=32, cmap="viridis")
        ax.scatter(snap["X"][:, 0], snap["X"][:, 1], c="white", s=28,
                   edgecolor="black", linewidth=0.8)
        for obs in snap["d1"]:
            x, v = obs["x"], obs["v"]
            ax.arrow(x[0], x[1], 0.55 * v[0], 0.55 * v[1],
                     width=0.018, head_width=0.15, color="tab:red",
                     length_includes_head=True, alpha=0.75)
        for obs in snap["d2"]:
            x, v = obs["x"], obs["v"]
            ax.plot(x[0], x[1], marker="s", markersize=6,
                    markerfacecolor="none", markeredgecolor="cyan",
                    markeredgewidth=1.2)
            ax.arrow(x[0], x[1], 0.35 * v[0], 0.35 * v[1],
                     width=0.012, head_width=0.11, color="cyan",
                     length_includes_head=True, alpha=0.85)
        ax.set_title(snap["label"], fontsize=10)
        ax.set_xlim(BOUNDS[0])
        ax.set_ylim(BOUNDS[1])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def plot_surrogate(al, figure_path):
    X_grid, X1, X2 = make_grid(75)
    truth = branin(X_grid).reshape(X1.shape)
    mean, var = query_function_posterior_batched(
        al.gp_model, al.params, X_grid, batch_size=250)
    mean = mean.reshape(X1.shape)
    var = var.reshape(X1.shape)
    error = np.abs(mean - truth)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    fields = [
        (truth, "True Branin-Hoo function", "viridis"),
        (mean, "Final surrogate mean", "viridis"),
        (var, "Final posterior variance", "magma"),
    ]
    for ax, (field, title, cmap) in zip(axes, fields):
        contour = ax.contourf(X1, X2, field, levels=32, cmap=cmap)
        fig.colorbar(contour, ax=ax)
        ax.scatter(al.X_train[:, 0], al.X_train[:, 1], c="white", s=22,
                   edgecolor="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlim(BOUNDS[0])
        ax.set_ylim(BOUNDS[1])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    axes[1].contour(X1, X2, error, levels=6, colors="white", linewidths=0.4,
                    alpha=0.5)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def main():
    # docs/source/_static — one level above JetGP_module_examples/
    static_dir = THIS_FILE.parents[2] / "_static"
    static_dir.mkdir(parents=True, exist_ok=True)

    X_test, _, _ = make_grid(25)
    al = AdaptiveDirectionalGP(
        func=branin,
        grad_func=branin_grad,
        hess_func=branin_hess,
        acquire_second_order=True,
        bounds=BOUNDS,
        n_init=10,
        rel_tol=0.00,
        n_iter=10,
        seed=11,
        c_f=1.0,
        c1=0.25,
        c2=0.35,
        max_directions=2,
        test_set=(X_test, branin(X_test)),
        predict_batch_size=250,
        verbose=False,
        optimizer_kwargs={
            "pop_size": 40,
            "n_generations": 20,
            "local_opt_every": 20,
            "debug": False,
        },
    )

    snapshots = capture_run(al, n_iter=15)
    plot_sequence(snapshots, static_dir / "active_learning_branin_sequence.png")
    plot_surrogate(al, static_dir / "active_learning_branin_surrogate.png")

    n_second = len(al.second_order_observations)
    print("Generated active-learning Branin-Hoo tutorial figures.")
    print(f"Function sites: {al.X_train.shape[0]}")
    print(f"First-order observations: {len(al.directional_observations)}")
    print(f"Second-order observations: {n_second}")
    if al.history:
        print(f"Final recorded RMSE: {al.history[-1]['rmse_test']:.6f}")


if __name__ == "__main__":
    main()
