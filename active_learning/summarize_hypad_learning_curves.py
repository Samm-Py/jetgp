"""
Summarize hypad_learning_curves.json.

Per (regime, policy) reports:
  - n_seeds
  - final RMSE: median, IQR, mean, std
  - "cost to reach RMSE = epsilon" per seed (NaN if never reached), then
    median and IQR of that distribution.
  - average f/d split in the chosen sequence.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def cost_to_reach(traj, epsilon):
    for rec in traj:
        if rec["rmse_test"] <= epsilon:
            return rec["cumulative_cost"]
    return np.nan


def fmt_rmse(x):
    if not np.isfinite(x):
        return "      nan"
    return f"{x:9.3e}"


def fmt_cost(x):
    if not np.isfinite(x):
        return "  never"
    return f"{x:7.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/hypad_learning_curves.json")
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[1e-3, 5e-4, 2e-4])
    args = parser.parse_args()

    payload = json.load(open(args.input))
    rows = payload["rows"]

    groups = defaultdict(list)
    for r in rows:
        groups[(r["regime"], r["policy"])].append(r)

    regimes = sorted({r["regime"] for r in rows})
    policies = ["CostAware", "FunctionOnly", "DerivativeOnly"]

    for regime in regimes:
        print(f"\n{'=' * 88}\nregime: {regime}\n{'=' * 88}")
        print(f"{'policy':<16s} {'n':>3s} "
              f"{'rmse median':>11s} {'rmse IQR':>20s} "
              f"{'avg n_f':>8s} {'avg n_d':>8s}")
        for policy in policies:
            group = groups.get((regime, policy), [])
            if not group:
                continue
            finals = np.array([r["rmse_test"] for r in group])
            n_f_iter = np.array([
                sum(1 for s in r["sequence"] if s == "f") for r in group])
            n_d_iter = np.array([
                sum(1 for s in r["sequence"] if s == "d") for r in group])
            med = float(np.median(finals))
            p25, p75 = (float(np.percentile(finals, 25)),
                        float(np.percentile(finals, 75)))
            print(f"{policy:<16s} {len(group):3d} "
                  f"{fmt_rmse(med)} [{fmt_rmse(p25)}, {fmt_rmse(p75)}] "
                  f"{n_f_iter.mean():8.2f} {n_d_iter.mean():8.2f}")

        for epsilon in args.epsilons:
            print(f"\n  cost to reach RMSE <= {epsilon:g}:")
            print(f"  {'policy':<16s} {'median':>9s} "
                  f"{'IQR':>22s} {'n_reached':>10s}")
            for policy in policies:
                group = groups.get((regime, policy), [])
                if not group:
                    continue
                costs = np.array([
                    cost_to_reach(r["trajectory"], epsilon) for r in group])
                n_reached = int(np.sum(np.isfinite(costs)))
                if n_reached == 0:
                    print(f"  {policy:<16s} {'never':>9s} "
                          f"{'':>22s} {n_reached:10d}")
                    continue
                med = float(np.nanmedian(costs))
                p25 = float(np.nanpercentile(costs, 25))
                p75 = float(np.nanpercentile(costs, 75))
                print(f"  {policy:<16s} {fmt_cost(med)} "
                      f"[{fmt_cost(p25)}, {fmt_cost(p75)}] {n_reached:10d}")


if __name__ == "__main__":
    main()
