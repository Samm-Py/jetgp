"""Generate figures for the cost-aware policy comparison tutorial."""

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
from doe_utils import lhs_design
from posterior_queries import query_function_posterior_batched


A = 1.0
B = 5.1 / (4 * np.pi**2)
C = 5.0 / np.pi
R = 6.0
S = 10.0
T = 1.0 / (8.0 * np.pi)

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


def make_grid(n_per_axis=18):
    x1 = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], n_per_axis)
    x2 = np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], n_per_axis)
    X1, X2 = np.meshgrid(x1, x2)
    return np.column_stack([X1.ravel(), X2.ravel()])


class CandidateFilteredAdaptiveGP(AdaptiveDirectionalGP):
    """Restrict a run to one candidate kind for baseline comparisons."""

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
        rel_tol=0.0,
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
    types = "".join(rec["chosen_type"] for rec in history) or "-"
    cost = history[-1]["cumulative_cost"] if history else 0.0
    return {
        "rmse": rmse,
        "cost": cost,
        "types": types,
        "n_f": al.X_train.shape[0],
        "n_d": len(al.directional_observations),
    }


def run_comparison():
    # n_iter is sized so the cost budget is the binding stopping condition,
    # not the iteration cap. With B=20 and a minimum per-step cost of 1,
    # n_iter=20 suffices for any policy to fully spend its budget.
    n_iter = 20
    return {
        "Derivatives cheap": {
            "cost_model": "c_f=10, c1=1, B=20",
            "Cost-aware": run_policy(
                AdaptiveDirectionalGP, c_f=10.0, c1=1.0, budget=20.0, n_iter=n_iter),
            "Function-only": run_policy(
                FunctionOnlyAdaptiveGP, c_f=10.0, c1=1.0, budget=20.0, n_iter=n_iter),
        },
        "Functions cheap": {
            "cost_model": "c_f=1, c1=10, B=20",
            "Cost-aware": run_policy(
                AdaptiveDirectionalGP, c_f=1.0, c1=10.0, budget=20.0, n_iter=n_iter),
            "Derivative-only": run_policy(
                DerivativeOnlyAdaptiveGP, c_f=1.0, c1=10.0, budget=20.0, n_iter=n_iter),
        },
    }


def plot_results(results, figure_path):
    regimes = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    colors = {
        "Cost-aware": "#4c78a8",
        "Function-only": "#f58518",
        "Derivative-only": "#e45756",
    }

    for ax, regime in zip(axes, regimes):
        policies = [name for name in results[regime] if name != "cost_model"]
        rmses = [results[regime][name]["rmse"] for name in policies]
        ax.bar(policies, rmses, color=[colors[name] for name in policies])
        ax.set_title(f"{regime}\n{results[regime]['cost_model']}")
        ax.set_ylabel("Grid RMSE")
        ax.grid(True, axis="y", alpha=0.25)
        for i, (name, rmse) in enumerate(zip(policies, rmses)):
            info = results[regime][name]
            ax.text(i, rmse, f"{rmse:.1f}\n{info['types']}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(rmses) * 1.25)

    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def print_rst_table(results):
    print(".. list-table:: Cost-aware policy comparison")
    print("   :header-rows: 1")
    print("")
    print("   * - Regime")
    print("     - Policy")
    print("     - Selected observations")
    print("     - Cost used")
    print("     - Grid RMSE")
    for regime, entries in results.items():
        for policy, values in entries.items():
            if policy == "cost_model":
                continue
            print(f"   * - {regime} ({entries['cost_model']})")
            print(f"     - {policy}")
            print(f"     - ``{values['types']}``")
            print(f"     - {values['cost']:.1f}")
            print(f"     - {values['rmse']:.2f}")


def main():
    # docs/source/_static — one level above JetGP_module_examples/
    static_dir = THIS_FILE.parents[2] / "_static"
    static_dir.mkdir(parents=True, exist_ok=True)
    results = run_comparison()
    plot_results(results, static_dir / "active_learning_cost_comparison.png")
    print_rst_table(results)


if __name__ == "__main__":
    main()
