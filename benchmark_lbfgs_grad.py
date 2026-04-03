"""
Benchmark: L-BFGS-B with vs without analytical gradients.

Test problem : DEGP on the 10D Dixon-Price function with all 10 partial derivatives.
Kernel       : SE anisotropic  (12 hyperparameters: 10 length scales + sf + sn)
Optimizer    : lbfgs, 10 restarts (LHS starting points)

Metrics reported per run:
  - Wall-clock time (s)
  - Final NLL
  - Total NLL function evaluations
  - Prediction RMSE on held-out test set

Run with:
    python benchmark_lbfgs_grad.py [--n_train N] [--n_test N] [--n_seeds N]
"""
import time
import argparse
import numpy as np
from scipy.stats import qmc

from jetgp.full_degp.degp import degp


DIM = 6
N_RESTARTS = 1


# ─────────────────────────────────────────────────────────────────────────────
# Dixon-Price 10D
# ─────────────────────────────────────────────────────────────────────────────

def dixon_price(X):
    """Dixon-Price, domain [-2, 2]^10. X shape (n, 10)."""
    x1   = X[:, 0]
    rest = X[:, 1:]
    prev = X[:, :-1]
    i    = np.arange(2, DIM + 1, dtype=float)
    return (x1 - 1.0) ** 2 + (i * (2.0 * rest**2 - prev**2)**2).sum(axis=1)


def dixon_price_grad(X):
    """Gradient of Dixon-Price. Returns shape (n, 10)."""
    grad = np.zeros_like(X)
    grad[:, 0] += 2.0 * (X[:, 0] - 1.0)
    for j in range(1, DIM):
        i = j + 1
        grad[:, j] += i * 2.0 * (2.0 * X[:, j]**2 - X[:, j-1]**2) * 4.0 * X[:, j]
    for j in range(0, DIM - 1):
        i = j + 2
        grad[:, j] += i * 2.0 * (2.0 * X[:, j+1]**2 - X[:, j]**2) * (-2.0 * X[:, j])
    return grad


# ─────────────────────────────────────────────────────────────────────────────
# Data and model
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(n_train, n_test, seed=42):
    lb = np.full(DIM, -2.0)
    ub = np.full(DIM,  2.0)
    X_train = qmc.scale(qmc.LatinHypercube(d=DIM, seed=seed).random(n=n_train), lb, ub)
    X_test  = qmc.scale(qmc.LatinHypercube(d=DIM, seed=seed+1).random(n=n_test), lb, ub)

    f_train = dixon_price(X_train).reshape(-1, 1)
    g_train = dixon_price_grad(X_train)
    y_train = [f_train] + [g_train[:, d].reshape(-1, 1) for d in range(DIM)]
    y_test  = dixon_price(X_test)
    return X_train, y_train, X_test, y_test


def build_model(X_train, y_train):
    # One first-order partial per dimension; OTI basis indices are 1-based
    der_indices = [[[[d + 1, 1]]] for d in range(DIM)]
    return degp(
        X_train, y_train,
        n_order=1, n_bases=DIM,
        der_indices=der_indices,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_one(model, X_test, y_test, use_analytic_grad, debug=False):
    """Run L-BFGS-B optimization and return (params, time, nll, nfev, rmse).

    nfev counts kernel-build + Cholesky evaluations (one per optimizer step),
    which is equivalent for both variants:
      - analytic: each nll_and_grad call = 1 Cholesky
      - numerical FD: each nll_wrapper call = 1 Cholesky
    """
    nfev_counter = [0]

    if use_analytic_grad:
        # Count nll_and_grad calls (each = 1 Cholesky)
        orig_nag = model.optimizer.nll_and_grad
        def counted_nag(x):
            nfev_counter[0] += 1
            return orig_nag(x)
        model.optimizer.nll_and_grad = counted_nag
        kwargs = dict(optimizer="lbfgs", n_restart_optimizer=N_RESTARTS, debug=debug)
    else:
        # Count nll_wrapper calls (each = 1 Cholesky)
        orig_nll = model.optimizer.nll_wrapper
        def counted_nll(x):
            nfev_counter[0] += 1
            return orig_nll(x)
        model.optimizer.nll_wrapper = counted_nll
        kwargs = dict(optimizer="lbfgs", n_restart_optimizer=N_RESTARTS,
                      func_and_grad=None, grad_func=None, debug=debug)

    t0 = time.perf_counter()
    params = model.optimize_hyperparameters(**kwargs)
    elapsed = time.perf_counter() - t0

    # Restore originals
    if use_analytic_grad:
        model.optimizer.nll_and_grad = orig_nag
    else:
        model.optimizer.nll_wrapper = orig_nll

    nll  = float(model.optimizer.nll_wrapper(params))
    nfev = nfev_counter[0]
    y_pred = model.predict(X_test, params, calc_cov=False, return_deriv=False)
    rmse = float(np.sqrt(np.mean((y_pred[0].ravel() - y_test) ** 2)))

    return params, elapsed, nll, nfev, rmse


def run_benchmark(n_train=100, n_test=500, n_seeds=3, debug=False):
    print(f"\n{'='*70}")
    print(f"  L-BFGS-B Gradient Benchmark — SE Anisotropic, Dixon-Price 10D")
    print(f"  n_train={n_train}  n_test={n_test}  restarts={N_RESTARTS}  seeds={n_seeds}")
    print(f"  Hyperparameters: {DIM + 2}  (10 ell + sf + sn)")
    print(f"{'='*70}\n")

    variants = [
        ("numerical FD", False),
        ("analytic grad", True),
    ]

    results = {name: dict(times=[], nlls=[], nfevs=[], rmses=[])
               for name, _ in variants}

    for seed in range(n_seeds):
        print(f"── seed {seed} {'─'*55}")
        X_train, y_train, X_test, y_test = make_dataset(n_train, n_test, seed=seed * 7)

        for name, use_grad in variants:
            model = build_model(X_train, y_train)
            try:
                _, t, nll, nfev, rmse = run_one(model, X_test, y_test,
                                                  use_analytic_grad=use_grad,
                                                  debug=debug)
                results[name]["times"].append(t)
                results[name]["nlls"].append(nll)
                results[name]["nfevs"].append(nfev)
                results[name]["rmses"].append(rmse)
                print(f"  {name:<16}  t={t:6.1f}s  NLL={nll:9.3f}  "
                      f"nfev={nfev:5d}  RMSE={rmse:.4f}")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  {name:<16}  ERROR: {e}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary (mean ± std over {n_seeds} seeds)")
    print(f"{'='*70}")
    print(f"  {'Variant':<16}  {'Time(s)':>12}  {'NLL':>12}  "
          f"{'nfev':>8}  {'RMSE':>10}")
    print(f"  {'-'*64}")
    for name, _ in variants:
        r = results[name]
        if not r["times"]:
            print(f"  {name:<16}  (all failed)")
            continue
        t_m, t_s   = np.mean(r["times"]),  np.std(r["times"])
        nll_m       = np.mean(r["nlls"])
        nfev_m      = np.mean(r["nfevs"])
        rmse_m, rmse_s = np.mean(r["rmses"]), np.std(r["rmses"])
        print(f"  {name:<16}  {t_m:7.1f}±{t_s:.1f}s  {nll_m:>12.3f}  "
              f"{nfev_m:>8.0f}  {rmse_m:.4f}±{rmse_s:.4f}")

    speedup = None
    if results["numerical FD"]["times"] and results["analytic grad"]["times"]:
        speedup = np.mean(results["numerical FD"]["times"]) / \
                  np.mean(results["analytic grad"]["times"])
        print(f"\n  Speedup (analytic vs FD): {speedup:.2f}x")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train",  type=int, default=120)
    parser.add_argument("--n_test",   type=int, default=500)
    parser.add_argument("--n_seeds",  type=int, default=3)
    parser.add_argument("--debug",    action="store_true")
    args = parser.parse_args()

    run_benchmark(n_train=args.n_train, n_test=args.n_test,
                  n_seeds=args.n_seeds, debug=args.debug)
