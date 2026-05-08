"""
Example 6 — HYPAD-UQ fin: derivative-strategy comparison at t = 10 s.

Compares four active-learning surrogates (Case 2, t=10 s, n_init=2, n_iter=10):
  1. Full-gradient DEGP         — all D gradient components at every point
  2. Full-Hessian DEGP          — ∇f + full symmetric Hessian (D + D(D+1)/2 derivs)
  3. Adaptive 1st-order GDDEGP  — eigenbasis directional, first order only
  4. Adaptive 2nd-order GDDEGP  — eigenbasis directional, first + second order

Usage
-----
    conda run -n pyoti_2 python example_6_derivative_strategy_comparison.py
    conda run -n pyoti_2 python example_6_derivative_strategy_comparison.py --n-iter 10 --seed 42
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configure problem before importing mutable globals
import example_5_hypad_fin_active_learning as ex5
ex5.configure_active_case(1)
ex5.configure_active_time(10.0)

from example_5_hypad_fin_active_learning import (
    T_tip_val_and_grad_z,
    T_tip_vec,
    sample_active_case_z,
    find_next_point_weighted_mpv,
    active_case_log_pdf,
    D,
)
from adaptive_doe import (
    AdaptiveDirectionalGP,
    lhs_design,
    make_as_basis_provider,
)
from jetgp.full_degp.degp import degp
from functools import partial

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

N_INIT = 2
N_ITER = 10
SEED   = 42
N_VAL  = 3000
REL_TOL = 0.05
LAMBDA_TOL = 1e-8

OPT_KWARGS = dict(
    optimizer="jade",
    pop_size=50,
    n_generations=40,
    local_opt_every=20,
    debug=False,
)

OUTDIR = None  # set at runtime by run()


# ---------------------------------------------------------------------------
# Problem wrappers
# ---------------------------------------------------------------------------

def _bounds():
    return ex5.BOUNDS_Z


def _func(Z):
    """f(Z: (n, D)) -> (n, 1)."""
    Z = np.atleast_2d(Z)
    return np.array([[T_tip_val_and_grad_z(Z[i])[0]] for i in range(len(Z))])


def _grad(Z):
    """grad f(Z: (n, D)) -> (n, D)."""
    Z = np.atleast_2d(Z)
    return np.array([T_tip_val_and_grad_z(Z[i])[1] for i in range(len(Z))])


def _hess(Z, eps=1e-5):
    """Full D×D Hessian via central FD on gradient. Returns (D, D) for single point."""
    Z = np.atleast_2d(Z)
    n = len(Z)
    out = np.zeros((n, D, D))
    for k in range(n):
        z = Z[k]
        H = np.zeros((D, D))
        for j in range(D):
            zp, zm = z.copy(), z.copy()
            zp[j] += eps
            zm[j] -= eps
            _, gp = T_tip_val_and_grad_z(zp)
            _, gm = T_tip_val_and_grad_z(zm)
            H[:, j] = (gp - gm) / (2 * eps)
        out[k] = 0.5 * (H + H.T)
    return out[0] if n == 1 else out


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _opt_kwargs(model, initial_params=None):
    kw = dict(OPT_KWARGS)
    if initial_params is not None:
        ip = np.asarray(initial_params, dtype=float).reshape(-1)
        if ip.size == len(model.bounds) and np.all(np.isfinite(ip)):
            kw["initial_positions"] = np.atleast_2d(ip)
    return kw


def _predict_mean(model, params, Z):
    Z = np.atleast_2d(Z)
    pred = model.predict(Z, params, calc_cov=False, return_deriv=False)
    if isinstance(pred, tuple):
        pred = pred[0]
    return np.asarray(pred)[0].reshape(-1)


def _nrmse(model, params, Z_val, y_val):
    y_pred = _predict_mean(model, params, Z_val)
    err = y_pred - y_val
    rmse = float(np.sqrt(np.mean(err ** 2)))
    scale = float(np.max(y_val) - np.min(y_val))
    return rmse / max(scale, 1e-30)


def _initial_design(n_init, seed):
    """Mean point (z=0) + (n_init-1) LHS points, matching example_5."""
    z_lhs = lhs_design(n_init - 1, _bounds(), seed=seed)
    return np.vstack([np.zeros((1, D)), z_lhs])


# ---------------------------------------------------------------------------
# Full-gradient DEGP
# ---------------------------------------------------------------------------

def _fit_full_gradient(X, y, grads, ip=None):
    n_train = X.shape[0]
    all_idx = list(range(n_train))
    y_blocks = [y] + [grads[:, j:j+1] for j in range(D)]
    der_indices = [[[[j + 1, 1]] for j in range(D)]]
    derivative_locations = [all_idx] * D
    m = degp(X, y_blocks, n_order=1, n_bases=D,
             der_indices=der_indices,
             derivative_locations=derivative_locations,
             normalize=True,
             kernel="SE", kernel_type="anisotropic")
    p = m.optimize_hyperparameters(**_opt_kwargs(m, ip))
    return m, p


def run_full_gradient(Z_val, y_val, n_init=N_INIT, n_iter=N_ITER, seed=SEED, Z0=None):
    print("\n" + "=" * 60)
    print("Full-gradient DEGP")
    print("=" * 60)

    if Z0 is None:
        Z0 = _initial_design(n_init, seed)
    Y0 = _func(Z0)
    G0 = _grad(Z0)

    X, y, grads = Z0.copy(), Y0.copy(), G0.copy()
    m, p = _fit_full_gradient(X, y, grads)

    nrmse_hist  = [_nrmse(m, p, Z_val, y_val)]
    n_train_hist = [n_init]
    n_deriv_hist = [D * n_init]
    print(f"  init  n_train={n_init:2d}  n_deriv={n_deriv_hist[0]:3d}  NRMSE={nrmse_hist[0]:.4g}")

    for step in range(1, n_iter + 1):
        z_new, _ = find_next_point_weighted_mpv(m, p, _bounds(),
                                                log_weight_fn=active_case_log_pdf,
                                                seed=seed + step)
        f_new, g_new = T_tip_val_and_grad_z(z_new)
        X     = np.vstack([X, z_new[None, :]])
        y     = np.vstack([y, [[f_new]]])
        grads = np.vstack([grads, g_new[None, :]])
        m, p  = _fit_full_gradient(X, y, grads, p)
        nrmse_hist.append(_nrmse(m, p, Z_val, y_val))
        n_train_hist.append(X.shape[0])
        n_deriv_hist.append(D * X.shape[0])
        print(f"  iter {step:2d}  n_train={X.shape[0]:2d}  "
              f"n_deriv={n_deriv_hist[-1]:3d}  NRMSE={nrmse_hist[-1]:.4g}")

    return {
        "label":    "Full-gradient DEGP",
        "nrmse":    nrmse_hist,
        "n_train":  n_train_hist,
        "n_deriv":  n_deriv_hist,
    }


# ---------------------------------------------------------------------------
# Full-Hessian DEGP (all D + D(D+1)/2 derivatives per point: ∇f and full Hessian)
# ---------------------------------------------------------------------------

# Upper-triangle (i<j) indices for the off-diagonal Hessian entries.
_OFF_I, _OFF_J = np.triu_indices(D, k=1)
N_OFF = len(_OFF_I)                       # = D*(D-1)//2
N_DERIV_PER_POINT = D + D + N_OFF         # ∇f + diag(H) + off-diag(H) = D(D+3)/2


def _hess_blocks(H):
    """Split (D, D) symmetric Hessian into (diag (D,), off-diag (N_OFF,))."""
    H = np.asarray(H)
    return np.diag(H).copy(), H[_OFF_I, _OFF_J].copy()


def _fit_full_hessian(X, y, grads, hess_diags, hess_off, ip=None):
    """
    Full-Hessian DEGP: function value + full gradient + full symmetric Hessian.
    OTI specs use [[i+1, 2]] for the diagonal d²f/dx_i² entries and
    [[i+1, 1], [j+1, 1]] for the mixed d²f/dx_i dx_j entries (i < j).
    """
    n_train = X.shape[0]
    all_idx = list(range(n_train))

    y_blocks = ([y]
                + [grads[:, j:j+1] for j in range(D)]
                + [hess_diags[:, j:j+1] for j in range(D)]
                + [hess_off[:, k:k+1] for k in range(N_OFF)])

    first_order  = [[[j + 1, 1]] for j in range(D)]
    second_diag  = [[[j + 1, 2]] for j in range(D)]
    second_mixed = [[[int(_OFF_I[k]) + 1, 1], [int(_OFF_J[k]) + 1, 1]]
                    for k in range(N_OFF)]
    der_indices  = [first_order, second_diag + second_mixed]

    derivative_locations = [all_idx] * (D + D + N_OFF)

    m = degp(X, y_blocks, n_order=2, n_bases=D,
             der_indices=der_indices,
             derivative_locations=derivative_locations,
             normalize=True,
             kernel="SE", kernel_type="anisotropic")
    p = m.optimize_hyperparameters(**_opt_kwargs(m, ip))
    return m, p


def _hess_data(Z):
    """Compute full Hessian for each row of Z; return (diag, off) blocks."""
    Z = np.atleast_2d(Z)
    n = len(Z)
    diag = np.zeros((n, D))
    off  = np.zeros((n, N_OFF))
    Hs = _hess(Z)
    if Hs.ndim == 2:                       # single point
        diag[0], off[0] = _hess_blocks(Hs)
    else:
        for i in range(n):
            diag[i], off[i] = _hess_blocks(Hs[i])
    return diag, off


def run_full_hessian(Z_val, y_val, n_init=N_INIT, n_iter=N_ITER, seed=SEED, Z0=None):
    print("\n" + "=" * 60)
    print("Full-Hessian DEGP")
    print("=" * 60)

    if Z0 is None:
        Z0 = _initial_design(n_init, seed)
    Y0 = _func(Z0)
    G0 = _grad(Z0)
    HD0, HO0 = _hess_data(Z0)

    X, y, grads, hdiags, hoff = Z0.copy(), Y0.copy(), G0.copy(), HD0.copy(), HO0.copy()
    m, p = _fit_full_hessian(X, y, grads, hdiags, hoff)

    nrmse_hist   = [_nrmse(m, p, Z_val, y_val)]
    n_train_hist = [n_init]
    n_deriv_hist = [N_DERIV_PER_POINT * n_init]
    print(f"  init  n_train={n_init:2d}  n_deriv={n_deriv_hist[0]:3d}  NRMSE={nrmse_hist[0]:.4g}")

    for step in range(1, n_iter + 1):
        z_new, _ = find_next_point_weighted_mpv(m, p, _bounds(),
                                                log_weight_fn=active_case_log_pdf,
                                                seed=seed + step)
        f_new, g_new = T_tip_val_and_grad_z(z_new)
        hd_new, ho_new = _hess_data(z_new[None, :])
        X      = np.vstack([X, z_new[None, :]])
        y      = np.vstack([y, [[f_new]]])
        grads  = np.vstack([grads, g_new[None, :]])
        hdiags = np.vstack([hdiags, hd_new])
        hoff   = np.vstack([hoff, ho_new])
        m, p   = _fit_full_hessian(X, y, grads, hdiags, hoff, p)
        nrmse_hist.append(_nrmse(m, p, Z_val, y_val))
        n_train_hist.append(X.shape[0])
        n_deriv_hist.append(N_DERIV_PER_POINT * X.shape[0])
        print(f"  iter {step:2d}  n_train={X.shape[0]:2d}  "
              f"n_deriv={n_deriv_hist[-1]:3d}  NRMSE={nrmse_hist[-1]:.4g}")

    return {
        "label":   "Full-Hessian DEGP",
        "nrmse":   nrmse_hist,
        "n_train": n_train_hist,
        "n_deriv": n_deriv_hist,
    }


# ---------------------------------------------------------------------------
# Adaptive GDDEGP (1st-order and 2nd-order)
# ---------------------------------------------------------------------------

def run_adaptive(order, Z_val, y_val, Z0, n_init=N_INIT, n_iter=N_ITER, seed=SEED,
                 lambda_abs_tol=LAMBDA_TOL):
    label = ("Adaptive 2nd-order GDDEGP"
             if order == 2 else "Adaptive 1st-order GDDEGP")
    print(f"\n{'='*60}\n{label}\n{'='*60}")

    acq = partial(find_next_point_weighted_mpv,
                  log_weight_fn=active_case_log_pdf)

    rng_as = np.random.default_rng(seed + 991)
    Z_AS = sample_active_case_z(200, rng_as)
    as_basis_provider = make_as_basis_provider(Z_AS)

    al = AdaptiveDirectionalGP(
        func=_func,
        grad_func=_grad,
        hess_func=_hess if order == 2 else None,
        bounds=_bounds(),
        n_init=n_init,
        rel_tol=REL_TOL,
        n_iter=n_iter,
        kernel="SE",
        kernel_type="anisotropic",
        seed=seed,
        lambda_abs_tol=lambda_abs_tol,
        acquire_second_order=(order == 2),
        acquisition_func=acq,
        optimizer_kwargs=OPT_KWARGS,
        X_init=Z0,
        as_basis_provider=as_basis_provider,
    )
    al.enable_initial_derivative_enrichment(True)
    al_history = al.run()

    # --- post-enrichment (initial state before active loop) ---
    n_init_first  = sum(len(r["selected_directions"])
                        for r in al.initial_derivative_history)
    n_init_second = sum(len(r.get("second_order_selected_directions", []))
                        for r in al.initial_derivative_history)

    nrmse_hist   = [_nrmse(al.post_enrichment_gp_model,
                            al.post_enrichment_params, Z_val, y_val)]
    n_train_hist = [n_init]
    n_deriv_hist = [n_init_first + n_init_second]

    print(f"  post-enrichment  n_train={n_init:2d}  "
          f"n_deriv={n_deriv_hist[0]:3d}  NRMSE={nrmse_hist[0]:.4g}")

    for h in al_history:
        nrmse_hist.append(_nrmse(h["post_update_gp_model"],
                                  h["post_update_params"], Z_val, y_val))
        n_train_hist.append(h["n_train"])
        n_deriv_hist.append(h["n_directional_obs"] + h["n_second_order_obs"])
        print(f"  iter {h['step']:2d}  n_train={h['n_train']:2d}  "
              f"n_deriv={n_deriv_hist[-1]:3d}  NRMSE={nrmse_hist[-1]:.4g}")

    return {
        "label":   label,
        "nrmse":   nrmse_hist,
        "n_train": n_train_hist,
        "n_deriv": n_deriv_hist,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
MARKERS = ["s", "^", "o", "D"]


def plot_nrmse_vs_n_train(results, outdir, tag=""):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for res, col, mk in zip(results, COLORS, MARKERS):
        ax.semilogy(res["n_train"], res["nrmse"],
                    mk + "-", color=col, label=res["label"])
    ax.set_xlabel("# function evaluations")
    ax.set_ylabel("Validation NRMSE")
    ax.set_title(f"Active-learning comparison — t = 10 s, {ex5.CASE_LABEL}{tag}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = outdir / "nrmse_vs_n_train.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"\nSaved: {path}")


def plot_nrmse_vs_n_deriv(results, outdir, tag=""):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for res, col, mk in zip(results, COLORS, MARKERS):
        total_obs = [nt + nd for nt, nd in zip(res["n_train"], res["n_deriv"])]
        ax.semilogy(total_obs, res["nrmse"],
                    mk + "-", color=col, label=res["label"])
    ax.set_xlabel("# function + derivative scalar observations")
    ax.set_ylabel("Validation NRMSE")
    ax.set_title(f"Cost-normalized comparison — t = 10 s, {ex5.CASE_LABEL}{tag}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = outdir / "nrmse_vs_total_obs.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary_table(results):
    print("\n" + "=" * 80)
    print(f"{'Method':<30s} {'final NRMSE':>12s} {'n_train':>8s} {'n_deriv':>9s}")
    print("-" * 80)
    for res in results:
        print(f"  {res['label']:<28s} "
              f"{res['nrmse'][-1]:>12.4g} "
              f"{res['n_train'][-1]:>8d} "
              f"{res['n_deriv'][-1]:>9d}")
    print("=" * 80)


def print_nrmse_table(results):
    """Print NRMSE at each function-evaluation count."""
    # All methods should share the same n_train progression.
    n_trains = results[0]["n_train"]
    print("\nNRMSE per iteration:")
    labels = [r["label"] for r in results]
    header = f"  {'n_f':>4s}" + "".join(f"  {lbl[:22]:>22s}" for lbl in labels)
    print(header)
    print("  " + "-" * (4 + 24 * len(labels)))
    for i, nt in enumerate(n_trains):
        row = f"  {nt:>4d}"
        for res in results:
            if i < len(res["nrmse"]):
                row += f"  {res['nrmse'][i]:>22.4g}"
            else:
                row += f"  {'—':>22s}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n_init=N_INIT, n_iter=N_ITER, seed=SEED, lambda_abs_tol=LAMBDA_TOL,
        verbose=True):
    case_tag = ex5.CASE_LABEL.lower().replace(" ", "_")
    time_tag = f"{ex5.ACTIVE_TIME_SECONDS:g}s".replace(".", "p")
    outdir = Path(__file__).parent / f"example_6_{case_tag}_t{time_tag}_figures"
    outdir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)
    Z_val = sample_active_case_z(N_VAL, rng)
    y_val = T_tip_vec(Z_val)
    print(f"Validation set: {N_VAL} points, "
          f"T_tip in [{y_val.min():.2f}, {y_val.max():.2f}] K")
    print(f"lambda_abs_tol = {lambda_abs_tol:.1e}")

    # Shared initial design for all four methods
    Z0 = _initial_design(n_init, seed)
    print(f"Initial design: {n_init} points (mean + LHS, seed={seed})")

    results = []
    results.append(run_full_gradient(Z_val, y_val, n_init, n_iter, seed, Z0=Z0))
    results.append(run_full_hessian(Z_val, y_val, n_init, n_iter, seed, Z0=Z0))
    results.append(run_adaptive(1, Z_val, y_val, Z0, n_init, n_iter, seed,
                                lambda_abs_tol=lambda_abs_tol))
    results.append(run_adaptive(2, Z_val, y_val, Z0, n_init, n_iter, seed,
                                lambda_abs_tol=lambda_abs_tol))

    print_nrmse_table(results)
    print_summary_table(results)

    plot_nrmse_vs_n_train(results, outdir)
    plot_nrmse_vs_n_deriv(results, outdir)

    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare derivative-acquisition strategies on the HYPAD fin.")
    p.add_argument("--n-init", type=int, default=N_INIT)
    p.add_argument("--n-iter", type=int, default=N_ITER)
    p.add_argument("--seed",   type=int, default=SEED)
    p.add_argument("--time",   type=float, default=10.0,
                   help="Transient time in seconds.")
    p.add_argument("--case",        type=int,   choices=(1, 2), default=1)
    p.add_argument("--lambda-tol",  type=float, default=LAMBDA_TOL,
                   help="Absolute leading-eigenvalue gate for second-order selection.")
    p.add_argument("--rel-tol",     type=float, default=REL_TOL,
                   help="Prior-relative gate threshold ρⱼ = λⱼ_post / vⱼᵀ K_prior vⱼ "
                        "for first-order direction selection.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ex5.configure_active_case(args.case)
    ex5.configure_active_time(args.time)
    REL_TOL = args.rel_tol
    run(n_init=args.n_init, n_iter=args.n_iter, seed=args.seed,
        lambda_abs_tol=args.lambda_tol)
