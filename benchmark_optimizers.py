"""
Benchmark: optimizer comparison for JetGP hyperparameter optimization.

Test problem: DDEGP on the 2D Branin function with 3 directional rays.
Compares lbfgs, pso, jade, spsa, pso_spsa, jade_spsa on:
  - Wall-clock optimization time
  - Final NLL
  - Prediction RMSE (function values)

NOTE: Requires static OTI modules with get_all_derivs compiled.
      Run after recompile_all_static.py completes successfully.
"""

import sys
import time
import numpy as np
import sympy as sp
from scipy.stats import qmc

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, '/root/working_folder/build')   # compiled static modules
sys.path.insert(0, '/root/jetgp')

from jetgp.full_ddegp.ddegp import ddegp

# ── Branin function ───────────────────────────────────────────────────────────

def make_branin():
    x1s, x2s = sp.symbols("x1 x2", real=True)
    a, b, c, r, s, t = (1.0, 5.1/(4*sp.pi**2), 5/sp.pi, 6.0, 10.0, 1/(8*sp.pi))
    f_sym = a*(x2s - b*x1s**2 + c*x1s - r)**2 + s*(1-t)*sp.cos(x1s) + s
    g1 = sp.diff(f_sym, x1s)
    g2 = sp.diff(f_sym, x2s)

    def _wrap(fn):
        raw = sp.lambdify([x1s, x2s], fn, "numpy")
        def wrapped(x1, x2):
            v = np.atleast_1d(raw(x1, x2))
            if v.size == 1 and np.atleast_1d(x1).size > 1:
                v = np.full_like(np.asarray(x1, float), float(v[0]))
            return v
        return wrapped

    return _wrap(f_sym), _wrap(g1), _wrap(g2)


# ── data generation ───────────────────────────────────────────────────────────

def make_dataset(n_train, n_test, rays, seed=42):
    f_func, g1_func, g2_func = make_branin()
    domain = [(-5.0, 10.0), (0.0, 15.0)]

    sampler = qmc.LatinHypercube(d=2, seed=seed)
    X_train = qmc.scale(sampler.random(n=n_train),
                        [b[0] for b in domain], [b[1] for b in domain])

    y_func = f_func(X_train[:, 0], X_train[:, 1]).reshape(-1, 1)

    # Directional derivatives: rays shape (2, n_rays)
    g1 = g1_func(X_train[:, 0], X_train[:, 1])
    g2 = g2_func(X_train[:, 0], X_train[:, 1])
    grad = np.stack([g1, g2], axis=0)  # (2, n_train)

    y_dirs = []
    for r in range(rays.shape[1]):
        dd = (grad * rays[:, r:r+1]).sum(axis=0).reshape(-1, 1)
        y_dirs.append(dd)

    y_train = [y_func] + y_dirs

    # Test grid
    sampler_test = qmc.LatinHypercube(d=2, seed=seed + 1)
    X_test = qmc.scale(sampler_test.random(n=n_test),
                       [b[0] for b in domain], [b[1] for b in domain])
    y_test = f_func(X_test[:, 0], X_test[:, 1])

    return X_train, y_train, X_test, y_test


# ── model builder ─────────────────────────────────────────────────────────────

def build_model(X_train, y_train, rays, n_order=1):
    n_rays = rays.shape[1]
    der_indices = [[[[r + 1, 1]] for r in range(n_rays)]]
    return ddegp(
        X_train, y_train,
        n_order=n_order,
        der_indices=der_indices,
        rays=rays,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic",
    )


# ── benchmark runner ──────────────────────────────────────────────────────────

OPTIMIZERS = {
    "lbfgs":     dict(optimizer="lbfgs",     n_restart_optimizer=5),
    "spsa":      dict(optimizer="spsa",      n_restart_optimizer=5, maxiter=300),
    "pso":       dict(optimizer="pso",       pop_size=20, n_generations=30,
                      local_opt_every=10),
    "jade":      dict(optimizer="jade",      pop_size=20, n_generations=30,
                      local_opt_every=10),
    "pso_spsa":  dict(optimizer="pso_spsa",  pop_size=20, n_generations=30,
                      local_opt_every=10, spsa_maxiter=150),
    "jade_spsa": dict(optimizer="jade_spsa", pop_size=20, n_generations=30,
                      local_opt_every=10, spsa_maxiter=150),
}


def run_benchmark(n_train=20, n_test=200, n_seeds=3):
    rays = np.array([
        [np.cos(np.pi/4), np.cos(np.pi/2), np.cos(3*np.pi/4)],
        [np.sin(np.pi/4), np.sin(np.pi/2), np.sin(3*np.pi/4)],
    ])

    print(f"\n{'='*65}")
    print(f"  JetGP Optimizer Benchmark — Branin 2D DDEGP")
    print(f"  n_train={n_train}  n_test={n_test}  n_seeds={n_seeds}")
    print(f"{'='*65}\n")

    # Header
    print(f"{'Optimizer':<14} {'Time(s)':>10} {'NLL':>12} {'RMSE':>12}")
    print("-" * 52)

    results = {}

    for name, opt_kwargs in OPTIMIZERS.items():
        times, nlls, rmses = [], [], []

        for seed in range(n_seeds):
            X_train, y_train, X_test, y_test = make_dataset(
                n_train, n_test, rays, seed=seed * 100
            )
            model = build_model(X_train, y_train, rays)

            kwargs = {k: v for k, v in opt_kwargs.items() if k != "optimizer"}
            opt_name = opt_kwargs["optimizer"]

            t0 = time.perf_counter()
            try:
                params = model.optimize_hyperparameters(
                    optimizer=opt_name, **kwargs
                )
                elapsed = time.perf_counter() - t0

                nll = model.optimizer.negative_log_marginal_likelihood(params)
                y_pred = model.predict(X_test, params, calc_cov=False,
                                       return_deriv=False)
                # predict returns shape (n_outputs, n_test); row 0 = function
                rmse = float(np.sqrt(np.mean((y_pred[0] - y_test)**2)))

                times.append(elapsed)
                nlls.append(nll)
                rmses.append(rmse)

            except Exception as e:
                print(f"  {name:<14} seed={seed} ERROR: {e}")
                times.append(np.nan)
                nlls.append(np.nan)
                rmses.append(np.nan)

        t_mean, t_std   = np.nanmean(times), np.nanstd(times)
        nll_mean        = np.nanmean(nlls)
        rmse_mean       = np.nanmean(rmses)

        results[name] = dict(time=t_mean, time_std=t_std,
                             nll=nll_mean, rmse=rmse_mean)

        print(f"{name:<14} {t_mean:>8.1f}s  {nll_mean:>12.3f}  {rmse_mean:>12.4f}"
              f"   (±{t_std:.1f}s)")

    print("\n" + "="*65)
    print("  Summary (ranked by RMSE)")
    print("="*65)
    ranked = sorted(results.items(), key=lambda x: x[1]['rmse'])
    for rank, (name, r) in enumerate(ranked, 1):
        print(f"  {rank}. {name:<14}  RMSE={r['rmse']:.4f}  "
              f"NLL={r['nll']:.2f}  t={r['time']:.1f}s")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train",  type=int, default=20)
    parser.add_argument("--n_test",   type=int, default=200)
    parser.add_argument("--n_seeds",  type=int, default=3)
    args = parser.parse_args()

    run_benchmark(n_train=args.n_train, n_test=args.n_test, n_seeds=args.n_seeds)
