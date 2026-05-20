"""
Plot HYPAD-UQ cost-vs-error learning curves from
`data/hypad_learning_curves.json`.

For each policy, every per-seed trajectory is interpolated onto a common cost
grid (held-last semantics — once a policy exhausts its budget its curve stays
flat), then the mean and ±1 std band across seeds are plotted on a log-y axis.
Individual seed trajectories are drawn as faint lines for transparency.

Output: docs/source/_static/hypad_learning_curves.png
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


POLICY_ORDER = ["CostAware", "FunctionOnly", "DerivativeOnly"]
POLICY_COLOURS = {
    "CostAware":      "#1f77b4",
    "FunctionOnly":   "#d62728",
    "DerivativeOnly": "#2ca02c",
}


def trajectory_arrays(traj):
    cost = np.array([p["cumulative_cost"] for p in traj], dtype=float)
    rmse = np.array([p["rmse_test"] for p in traj], dtype=float)
    return cost, rmse


def interpolate_held_last(cost, rmse, grid):
    """Step-function interpolation: rmse[i] is the value AFTER spending
    cost[i]. For grid points before the first observation, return NaN.
    Beyond the last observation, hold the last value (policy exhausted budget
    but cost grid continues)."""
    out = np.full_like(grid, np.nan, dtype=float)
    if cost.size == 0:
        return out
    # Forward iteration so each successive (larger cost, newer rmse) value
    # overwrites earlier ones on the suffix of the grid it covers.
    for i in range(cost.size):
        out[grid >= cost[i]] = rmse[i]
    return out


def aggregate(rows, grid):
    """Group rows by (regime, policy) and return per-cell stacked arrays."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["regime"], row["policy"])].append(row)
    out = {}
    for key, group in grouped.items():
        stacks = np.stack([
            interpolate_held_last(*trajectory_arrays(r["trajectory"]), grid)
            for r in group
        ])
        out[key] = stacks
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/hypad_learning_curves.json")
    parser.add_argument("--output",
                        default="../docs/source/_static/hypad_learning_curves.png")
    parser.add_argument("--n-grid", type=int, default=200)
    args = parser.parse_args()

    with open(args.input) as f:
        payload = json.load(f)
    rows = payload["rows"]
    if not rows:
        raise SystemExit("No rows in JSON.")

    regimes = sorted({r["regime"] for r in rows})
    budget = max(r["budget"] for r in rows)
    grid = np.linspace(0.0, budget, args.n_grid)

    cells = aggregate(rows, grid)
    n_seeds = len(next(iter(cells.values())))

    fig, axes = plt.subplots(
        1, len(regimes), figsize=(6 * len(regimes), 4.5),
        squeeze=False, sharey=True)

    all_finals = []
    for regime in regimes:
        for policy in POLICY_ORDER:
            stack = cells.get((regime, policy))
            if stack is None:
                continue
            all_finals.append(stack[:, -1])
    final_arr = np.concatenate(all_finals)
    y_hi = 10.0 ** np.ceil(np.log10(np.nanmax(final_arr) * 30.0))
    y_lo = 10.0 ** np.floor(np.log10(np.nanmin(final_arr) * 0.5))

    x_lo = max(1.0, grid[0])

    for ax, regime in zip(axes[0], regimes):
        for policy in POLICY_ORDER:
            stack = cells.get((regime, policy))
            if stack is None:
                continue
            p25 = np.nanpercentile(stack, 25, axis=0)
            p50 = np.nanpercentile(stack, 50, axis=0)
            p75 = np.nanpercentile(stack, 75, axis=0)
            colour = POLICY_COLOURS[policy]
            for trace in stack:
                ax.plot(grid, trace, color=colour, alpha=0.18, lw=0.7)
            ax.plot(grid, p50, color=colour, lw=2.2, label=policy)
            ax.fill_between(grid, p25, p75, color=colour, alpha=0.20,
                            linewidth=0)
        ax.set_yscale("log")
        ax.set_xlim(x_lo, grid[-1])
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("cumulative cost")
        ax.set_title(f"regime: {regime}")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
    axes[0, 0].set_ylabel("test RMSE")
    axes[0, 0].legend(loc="upper right", framealpha=0.9)
    fig.suptitle(
        f"HYPAD-UQ active learning — median + IQR across {n_seeds} seeds",
        fontsize=11)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
