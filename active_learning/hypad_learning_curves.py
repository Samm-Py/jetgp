"""
Cost-vs-error learning curves for the HYPAD-UQ cost-aware policy.

For each cost regime, each policy, and each seed, this script records the
(cumulative_cost, rmse_test) trajectory across the run. The resulting JSON is
intended to be plotted as mean ± std bands per policy on a single axis —
showing whether CostAware Pareto-dominates the single-modality baselines
across the *whole* budget range, not just at the final budget.

To keep curves comparable across policies that consume budget at different
rates, we let every policy run until either its budget is exhausted or
`--n-iter` steps elapse, then save the raw per-step (cost, rmse) pairs.
Aggregation onto a common cost grid is left to the plotting code.

Output: data/hypad_learning_curves.json
"""

import argparse
import json
import os

from adaptive_doe import AdaptiveDirectionalGP
from hypad_cost_aware_probe import (
    DerivativeOnlyAdaptiveGP,
    FunctionOnlyAdaptiveGP,
    run_policy,
    setup_problem,
)


REGIMES = [
    # Central FD: one directional derivative = 2 extra function evals.
    # Worst case for CostAware (derivatives strictly more expensive).
    ("fd_central", 1.0, 2.0, 6.0),
    # Forward FD at an existing point: function value f(x) is reused, so
    # only one perturbed eval is needed. c_d == c_f.
    ("fd_forward", 1.0, 1.0, 6.0),
    # AD / OTI: directional derivative obtained alongside the function
    # value with modest extra arithmetic. c_d = 0.5 is illustrative;
    # replace with a measured OTI/function timing ratio once available.
    ("ad",         1.0, 0.5, 6.0),
]
POLICIES = [AdaptiveDirectionalGP, FunctionOnlyAdaptiveGP, DerivativeOnlyAdaptiveGP]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=5000)
    parser.add_argument("--case", type=int, choices=(1, 2), default=1)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--n-init", type=int, default=6,
                        help="Initial DOE size. With max_directions=7, this "
                             "caps DerivativeOnly at n_init*7 directional "
                             "observations; 6 gives 42 of headroom.")
    parser.add_argument("--n-iter", type=int, default=20,
                        help="Max iterations per run. Higher than the probe so "
                             "policies have headroom past the budget point.")
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--n-generations", type=int, default=20)
    parser.add_argument("--budget-scale", type=float, default=1.0,
                        help="Multiply each regime's nominal budget by this. "
                             "Lets the learning curves extend further than the "
                             "probe's headline budget.")
    parser.add_argument("--out", default="data/hypad_learning_curves.json")
    args = parser.parse_args()

    optimizer_kwargs = {
        "pop_size": args.pop_size,
        "n_generations": args.n_generations,
        "local_opt_every": args.n_generations,
        "debug": False,
    }

    rows = []
    seeds = [args.seed_base + i for i in range(args.n_seeds)]
    for seed in seeds:
        X_init, Z_val, y_val = setup_problem(
            seed, args.n_init, args.n_val, args.case, args.time)
        for regime_name, c_f, c1, nominal_budget in REGIMES:
            budget = nominal_budget * args.budget_scale
            for policy in POLICIES:
                result = run_policy(
                    policy, X_init, Z_val, y_val,
                    c_f=c_f, c1=c1, budget=budget,
                    n_iter=args.n_iter, seed=seed,
                    optimizer_kwargs=optimizer_kwargs,
                    show_optimizer_output=False,
                )
                result.update({"regime": regime_name, "seed": seed,
                               "c_f": c_f, "c1": c1, "budget": budget,
                               "nominal_budget": nominal_budget})
                rows.append(result)
                traj = result["trajectory"]
                n_pts = len(traj)
                final_cost = traj[-1]["cumulative_cost"] if traj else 0.0
                print(f"[{regime_name:<18s} seed={seed} {result['policy']:<15s}] "
                      f"steps={n_pts:2d} final_cost={final_cost:5.1f} "
                      f"final_rmse={result['rmse_test']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "args": vars(args)}, f, indent=2)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
