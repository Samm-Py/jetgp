"""
Compare cost-aware active learning to a naive gradient-enhanced baseline.

Naive: take a fresh LHS design of n points and evaluate the function value
       and the *full* gradient at each. No active learning, no iteration.
       Total cost = n * (c_f + d * c_d).

For HYPAD-UQ Case 1 in the AD regime (c_f = 1, c_d = 0.5, d = 7), each LHS
point with a full gradient costs c_f + d * c_d = 4.5 cost units. So
n = budget / 4.5 LHS points exhaust a budget B.

The cost-aware result at the matching cumulative budget is read directly from
the existing data/hypad_learning_curves.json (no re-run). If the requested
budget exceeds the trajectory's final cumulative cost, the cost-aware result
at the final point is used instead.

Output: data/hypad_naive_vs_costaware.json
"""

import argparse
import json
import os

import numpy as np

import example_3_hypad_fin_active_learning as fin
from adaptive_doe import AdaptiveDirectionalGP
from doe_utils import lhs_design
from gp_builders import fit_directional_gp
from hypad_cost_aware_probe import (fin_func, fin_grad, run_policy,
                                     setup_problem)
from posterior_queries import query_function_posterior_batched


def run_naive(seed, n, case, time, n_val, optimizer_kwargs):
    """Build an LHS-of-n design with f + ∇f at each point, return RMSE."""
    if n < 2:
        raise ValueError(
            f"naive baseline requires n >= 2 (got n={n}); "
            "a single training point makes y_train constant and the "
            "GP normalisation divides by zero.")
    fin.configure_active_case(case)
    fin.configure_active_time(time)
    rng = np.random.default_rng(seed)

    X_lhs = lhs_design(n, fin.BOUNDS_Z, seed=seed)
    Z_val = fin.sample_active_case_z(n_val, rng)
    y_val = fin.T_tip_vec(Z_val)

    y_lhs = fin_func(X_lhs)
    grads = fin_grad(X_lhs)

    d = X_lhs.shape[1]
    directional_observations = []
    for i in range(n):
        for k in range(d):
            v = np.zeros(d)
            v[k] = 1.0
            directional_observations.append({
                "x_index": i,
                "direction": v,
                "value": float(grads[i, k]),
                "slot": k,
            })

    gp_model, params = fit_directional_gp(
        X_lhs, y_lhs, directional_observations,
        optimizer_kwargs=optimizer_kwargs,
        max_directions=d,
    )
    pred, _ = query_function_posterior_batched(
        gp_model, params, Z_val, batch_size=128)
    rmse = float(np.sqrt(np.mean(
        (np.asarray(pred).ravel() - np.asarray(y_val).ravel()) ** 2)))
    cost = float(n) * (1.0 + d * 0.5)  # c_f + d*c_d in AD regime
    return {"n": n, "cost": cost, "rmse": rmse}


def run_costaware_inline(seed, n_init, budget, c_f, c1, case, time, n_val,
                          n_iter, optimizer_kwargs):
    """Run a fresh CostAware policy with matching budget and n_init.

    Returns the full row including final RMSE.
    """
    X_init, Z_val, y_val = setup_problem(seed, n_init, n_val, case, time)
    result = run_policy(
        AdaptiveDirectionalGP, X_init, Z_val, y_val,
        c_f=c_f, c1=c1, budget=budget,
        n_iter=n_iter, seed=seed,
        optimizer_kwargs=optimizer_kwargs,
        show_optimizer_output=False,
    )
    return result


def extract_costaware_at_budget(learning_curves_path, budget,
                                 regime="ad", policy="CostAware"):
    """Per-seed RMSE from the cost-aware trajectory at cumulative_cost <= budget.

    Returns a list of dicts keyed by seed with the last-trajectory-point at or
    below the requested budget.
    """
    payload = json.load(open(learning_curves_path))
    out = []
    for row in payload["rows"]:
        if row["regime"] != regime or row["policy"] != policy:
            continue
        traj = row["trajectory"]
        rec_at_budget = None
        for rec in traj:
            if rec["cumulative_cost"] <= budget + 1e-9:
                rec_at_budget = rec
        if rec_at_budget is None:
            continue
        out.append({
            "seed": row["seed"],
            "cumulative_cost": rec_at_budget["cumulative_cost"],
            "rmse": rec_at_budget["rmse_test"],
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=5000)
    parser.add_argument("--ns", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 6],
                        help="Number of naive LHS points (per-point cost 4.5 "
                             "in the AD regime → budget = n*4.5).")
    parser.add_argument("--case", type=int, choices=(1, 2), default=1)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--n-generations", type=int, default=20)
    parser.add_argument("--n-init-costaware", type=int, default=2,
                        help="Initial-DOE size for the inline CostAware run. "
                             "Set to match naive's training-set size for an "
                             "apples-to-apples comparison.")
    parser.add_argument("--inline-costaware", action="store_true",
                        default=True,
                        help="Run CostAware fresh per (seed,n) with matching "
                             "budget and n_init instead of reading from the "
                             "learning-curves JSON.")
    parser.add_argument("--n-iter", type=int, default=30,
                        help="Iteration cap for the inline CostAware run.")
    parser.add_argument("--learning-curves",
                        default="data/hypad_learning_curves.json")
    parser.add_argument("--out",
                        default="data/hypad_naive_vs_costaware.json")
    args = parser.parse_args()

    optimizer_kwargs = {
        "pop_size": args.pop_size,
        "n_generations": args.n_generations,
        "local_opt_every": args.n_generations,
        "debug": False,
    }

    seeds = [args.seed_base + i for i in range(args.n_seeds)]

    naive_rows = []
    for seed in seeds:
        for n in args.ns:
            res = run_naive(seed, n, args.case, args.time, args.n_val,
                            optimizer_kwargs)
            res.update({"seed": seed})
            naive_rows.append(res)
            print(f"[naive    seed={seed} n={n} cost={res['cost']:5.1f}] "
                  f"RMSE={res['rmse']:.4e}")

    costaware_rows = []
    if args.inline_costaware:
        for seed in seeds:
            for n in args.ns:
                budget = float(n) * 4.5
                result = run_costaware_inline(
                    seed, args.n_init_costaware, budget,
                    c_f=1.0, c1=0.5,
                    case=args.case, time=args.time, n_val=args.n_val,
                    n_iter=args.n_iter,
                    optimizer_kwargs=optimizer_kwargs,
                )
                row = {
                    "seed": seed,
                    "n_naive_equivalent": n,
                    "target_budget": budget,
                    "cumulative_cost": result["cost"],
                    "rmse": result["rmse_test"],
                    "sequence": result["sequence"],
                    "n_init": args.n_init_costaware,
                }
                costaware_rows.append(row)
                print(f"[costaware seed={seed} n={n} n_init={args.n_init_costaware} "
                      f"budget={budget:5.1f}] "
                      f"RMSE={row['rmse']:.4e} seq={row['sequence']}")
    elif os.path.exists(args.learning_curves):
        for n in args.ns:
            budget = float(n) * 4.5
            rows = extract_costaware_at_budget(
                args.learning_curves, budget,
                regime="ad", policy="CostAware")
            for r in rows:
                r["target_budget"] = budget
                r["n_naive_equivalent"] = n
                costaware_rows.append(r)
            print(f"[costaware target_budget={budget:5.1f}] "
                  f"got {len(rows)} seeds from trajectory")
    else:
        print(f"warning: {args.learning_curves} not found; "
              f"cost-aware comparison rows are empty")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "naive": naive_rows,
            "costaware": costaware_rows,
            "args": vars(args),
        }, f, indent=2)
    print(f"\nWrote naive: {len(naive_rows)}, "
          f"costaware: {len(costaware_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
