"""
Explore cost-aware active learning on the HYPAD-UQ heated fin.

This script is intentionally a probe for documentation design rather than a
formal benchmark. It compares the cost-aware policy against naive policies that
are forced to select only function values or only directional derivatives under
the same budget.
"""

import argparse
import contextlib
import io
import os


import numpy as np

import example_3_hypad_fin_active_learning as fin
from acquisition import make_weighted_mpv
from adaptive_doe import AdaptiveDirectionalGP
from doe_utils import lhs_design
from posterior_queries import query_function_posterior_batched


def fin_func(Z):
    return fin.T_tip_vec(np.atleast_2d(Z)).reshape(-1, 1)


def fin_grad(Z):
    Z = np.atleast_2d(Z)
    grads = np.empty((Z.shape[0], fin.D))
    for i, z in enumerate(Z):
        _, grads[i] = fin.T_tip_val_and_grad_z(z)
    return grads


class CandidateFilteredAdaptiveGP(AdaptiveDirectionalGP):
    """Restrict a policy to one candidate kind for baseline comparisons."""

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


def nrmse(y_pred, y_true):
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    scale = np.ptp(y_true)
    return float(rmse / scale) if scale > 0.0 else float(rmse)


POLICY_LABELS = {
    "AdaptiveDirectionalGP": "CostAware",
    "FunctionOnlyAdaptiveGP": "FunctionOnly",
    "DerivativeOnlyAdaptiveGP": "DerivativeOnly",
}


def run_policy(policy_cls, X_init, Z_val, y_val, c_f, c1, budget,
               n_iter, seed, optimizer_kwargs, show_optimizer_output,
               weighted=True):
    """Run an active-learning policy on the fin problem.

    weighted=True (default) selects the PDF-weighted MPV acquisition keyed off
    the currently-configured fin case (case1_log_pdf or case2_log_pdf). Set
    weighted=False for plain MPV — useful for back-compatibility with the
    earlier unweighted experiments.
    """
    if weighted:
        acquisition_func = make_weighted_mpv(fin.active_case_log_pdf)
        log_weight_fn = fin.active_case_log_pdf
    else:
        acquisition_func = None
        log_weight_fn = None
    al = policy_cls(
        func=fin_func,
        grad_func=fin_grad,
        bounds=fin.BOUNDS_Z,
        n_init=X_init.shape[0],
        rel_tol=0.00,
        n_iter=n_iter,
        seed=seed,
        c_f=c_f,
        c1=c1,
        cost_budget=budget,
        max_directions=7,
        X_init=X_init,
        test_set=(Z_val, y_val),
        predict_batch_size=128,
        verbose=False,
        optimizer_kwargs=optimizer_kwargs,
        acquisition_func=acquisition_func,
        log_weight_fn=log_weight_fn,
    )
    if show_optimizer_output:
        history = al.run()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            history = al.run()
    pred, _ = query_function_posterior_batched(
        al.gp_model, al.params, Z_val, batch_size=128)
    sequence = "".join(rec["chosen_type"] for rec in history) or "-"
    cost = history[-1]["cumulative_cost"] if history else 0.0
    trajectory = [
        {"step": rec["step"],
         "cumulative_cost": rec["cumulative_cost"],
         "rmse_test": rec["rmse_test"],
         "chosen_type": rec["chosen_type"]}
        for rec in history
    ]
    return {
        "policy": POLICY_LABELS.get(policy_cls.__name__, policy_cls.__name__),
        "nrmse": nrmse(pred, y_val),
        "rmse_test": history[-1]["rmse_test"] if history else None,
        "cost": cost,
        "sequence": sequence,
        "n_function": al.X_train.shape[0],
        "n_derivative": len(al.directional_observations),
        "trajectory": trajectory,
    }


def print_table(title, rows):
    print("\n" + title)
    print("-" * len(title))
    print(f"{'policy':<18s} {'seq':<12s} {'cost':>7s} "
          f"{'N_f':>5s} {'N_d':>5s} {'NRMSE':>10s}")
    for row in rows:
        print(f"{row['policy']:<18s} {row['sequence']:<12s} "
              f"{row['cost']:7.2f} {row['n_function']:5d} "
              f"{row['n_derivative']:5d} {row['nrmse']:10.4g}")
    cost_aware = next(row for row in rows if row["policy"] == "CostAware")
    for row in rows:
        if row["policy"] == "CostAware":
            continue
        delta = row["nrmse"] - cost_aware["nrmse"]
        rel = 100.0 * delta / row["nrmse"] if row["nrmse"] > 0.0 else 0.0
        print(f"  vs {row['policy']:<14s}: NRMSE improvement = "
              f"{delta:.4g} ({rel:.1f}%)")


def setup_problem(seed, n_init, n_val, case, time):
    """Configure the fin and build the (X_init, Z_val, y_val) triple for a seed."""
    fin.configure_active_case(case)
    fin.configure_active_time(time)
    rng = np.random.default_rng(seed)
    X_lhs = lhs_design(n_init - 1, fin.BOUNDS_Z, seed=seed)
    X_init = np.vstack([np.zeros((1, fin.D)), X_lhs])
    Z_val = fin.sample_active_case_z(n_val, rng)
    y_val = fin.T_tip_vec(Z_val)
    return X_init, Z_val, y_val


def run_probe(args):
    X_init, Z_val, y_val = setup_problem(
        args.seed, args.n_init, args.n_val, args.case, args.time)

    optimizer_kwargs = {
        "pop_size": args.pop_size,
        "n_generations": args.n_generations,
        "local_opt_every": args.n_generations,
        "debug": False,
    }

    experiments = [
        (
            "Derivatives cheap",
            args.function_expensive_cost,
            1.0,
            args.function_expensive_budget,
            [AdaptiveDirectionalGP, FunctionOnlyAdaptiveGP, DerivativeOnlyAdaptiveGP],
        ),
        (
            "Functions cheap",
            1.0,
            args.derivative_expensive_cost,
            args.derivative_expensive_budget,
            [AdaptiveDirectionalGP, FunctionOnlyAdaptiveGP, DerivativeOnlyAdaptiveGP],
        ),
    ]

    print(f"HYPAD-UQ fin probe: case={args.case}, t={args.time:g}s, "
          f"n_init={args.n_init}, n_iter={args.n_iter}")
    print(f"Validation samples: {args.n_val}")

    all_results = {}
    for title, c_f, c1, budget, policies in experiments:
        rows = [
            run_policy(
                policy, X_init, Z_val, y_val,
                c_f=c_f, c1=c1, budget=budget,
                n_iter=args.n_iter, seed=args.seed,
                optimizer_kwargs=optimizer_kwargs,
                show_optimizer_output=args.show_optimizer_output,
            )
            for policy in policies
        ]
        all_results[title] = rows
        print_table(f"{title}: c_f={c_f:g}, c1={c1:g}, B={budget:g}", rows)
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe HYPAD-UQ cost-aware active learning vs naive baselines.")
    parser.add_argument("--case", type=int, choices=(1, 2), default=1)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--n-init", type=int, default=2)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--n-generations", type=int, default=20)
    parser.add_argument("--function-expensive-cost", type=float, default=10.0)
    parser.add_argument("--function-expensive-budget", type=float, default=20.0)
    parser.add_argument("--derivative-expensive-cost", type=float, default=10.0)
    parser.add_argument("--derivative-expensive-budget", type=float, default=20.0)
    parser.add_argument("--show-optimizer-output", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_probe(parse_args())
