"""
Benchmark: optimizer comparison for JetGP hyperparameter optimization.

Test problem: DEGP on the 10D Dixon-Price function with all 10 partial derivatives.
With anisotropic SE kernel: 10 length scales + 1 signal var + 1 noise = 12 hyperparameters.

Compares lbfgs, pso, jade, spsa, pso_spsa, jade_spsa on:
  - Wall-clock optimization time
  - Final NLL
  - Prediction RMSE (function values)
"""

from jetgp.full_degp.degp import degp
import sys
import time
import numpy as np
from scipy.stats import qmc

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, '/root/working_folder/build')   # compiled static modules
sys.path.insert(0, '/root/jetgp')


DIM = 10

# ── Dixon-Price function ──────────────────────────────────────────────────────


def dixon_price(X):
    """Dixon-Price function, X shape (n, 10). Domain [-2, 2]^10."""
    x1 = X[:, 0]
    rest = X[:, 1:]
    prev = X[:, :-1]
    term1 = (x1 - 1.0) ** 2
    i = np.arange(2, DIM + 1, dtype=float)          # shape (9,)
    term2 = (i * (2.0 * rest**2 - prev**2)**2).sum(axis=1)
    return term1 + term2


def dixon_price_grad(X):
    """Returns gradient, shape (n, 10)."""
    n = X.shape[0]
    grad = np.zeros_like(X)
    # df/dx1 = 2*(x1 - 1) - 2*2*(x2^2 - x1^2)*(-2*x1)*1
    # General: df/dx_i = i*2*(2*x_i^2 - x_{i-1}^2)*4*x_i  (self term as x_i)
    #                  + (i+1)*2*(2*x_{i+1}^2 - x_i^2)*(-2*x_i)  (as x_{i-1})

    # Contribution as x_j (self) for j = 2..D
    for j in range(1, DIM):   # 0-indexed, j=1 is x_2
        i = j + 1             # 1-indexed i
        prev_val = X[:, j-1]
        self_val = X[:, j]
        grad[:, j] += i * 2.0 * (2.0 * self_val**2 -
                                 prev_val**2) * 4.0 * self_val

    # Contribution as x_{i-1} for i = 2..D  (i.e. x_j appears as "prev" for i=j+1)
    for j in range(0, DIM - 1):   # x_j appears as prev in term i=j+2
        i = j + 2
        prev_val = X[:, j]
        self_val = X[:, j+1]
        grad[:, j] += i * 2.0 * (2.0 * self_val**2 -
                                 prev_val**2) * (-2.0 * prev_val)

    # x_1 term: 2*(x1 - 1)
    grad[:, 0] += 2.0 * (X[:, 0] - 1.0)

    return grad


# ── data generation ───────────────────────────────────────────────────────────

def make_dataset(n_train, n_test, seed=42):
    domain_lb = np.full(DIM, -2.0)
    domain_ub = np.full(DIM,  2.0)

    sampler = qmc.LatinHypercube(d=DIM, seed=seed)
    X_train = qmc.scale(sampler.random(n=n_train), domain_lb, domain_ub)

    y_func = dixon_price(X_train).reshape(-1, 1)
    grad = dixon_price_grad(X_train)           # (n_train, 10)

    # y_train: [f, df/dx1, ..., df/dx10]
    y_train = [y_func] + [grad[:, d].reshape(-1, 1) for d in range(DIM)]

    sampler_test = qmc.LatinHypercube(d=DIM, seed=seed + 1)
    X_test = qmc.scale(sampler_test.random(n=n_test), domain_lb, domain_ub)
    y_test = dixon_price(X_test)

    return X_train, y_train, X_test, y_test


# ── model builder ─────────────────────────────────────────────────────────────

def build_model(X_train, y_train, n_order=1):
    # der_indices: one entry per derivative, each is [[[basis_idx, order]]]
    # OTI basis indices are 1-based, so d+1
    der_indices = [[[[d + 1, 1]]]
                   for d in range(DIM)]   # 10 first-order partials
    return degp(
        X_train, y_train,
        n_order=n_order,
        n_bases=DIM,
        der_indices=der_indices,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic",
    )


# ── benchmark runner ──────────────────────────────────────────────────────────

OPTIMIZERS = {
    # "lbfgs":     dict(optimizer="lbfgs",     n_restart_optimizer=5),
    # "spsa":      dict(optimizer="spsa",      n_restart_optimizer=5, maxiter=500),
    "pso":       dict(optimizer="pso",       pop_size=30, n_generations=50,
                      local_opt_every=50, debug=True),
    "jade":      dict(optimizer="jade",      pop_size=30, n_generations=50,
                      local_opt_every=50, debug=True),
    "pso_spsa":  dict(optimizer="pso_spsa",  pop_size=30, n_generations=50,
                      local_opt_every=50, spsa_maxiter=300, debug=True),
    "jade_spsa": dict(optimizer="jade_spsa", pop_size=30, n_generations=50,
                      local_opt_every=50, spsa_maxiter=300, debug=True),
}


def run_benchmark(n_train=100, n_test=500, n_seeds=1):
    print(f"\n{'='*65}")
    print(f"  JetGP Optimizer Benchmark — Dixon-Price 10D DEGP")
    print(f"  n_train={n_train}  n_test={n_test}  n_seeds={n_seeds}")
    print(f"  Hyperparameters: 12 (10 length scales + signal var + noise)")
    print(f"{'='*65}\n")

    print(f"{'Optimizer':<14} {'Time(s)':>10} {'NLL':>12} {'RMSE':>12}")
    print("-" * 52)

    results = {}

    for name, opt_kwargs in OPTIMIZERS.items():
        times, nlls, rmses = [], [], []

        for seed in range(n_seeds):
            X_train, y_train, X_test, y_test = make_dataset(
                n_train, n_test, seed=seed * 100
            )
            model = build_model(X_train, y_train)

            kwargs = {k: v for k, v in opt_kwargs.items() if k != "optimizer"}
            opt_name = opt_kwargs["optimizer"]

            t0 = time.perf_counter()
            try:
                params = model.optimize_hyperparameters(
                    optimizer=opt_name, **kwargs)
                elapsed = time.perf_counter() - t0

                nll = model.optimizer.negative_log_marginal_likelihood(params)
                y_pred = model.predict(X_test, params, calc_cov=False,
                                       return_deriv=False)
                rmse = float(np.sqrt(np.mean((y_pred[0] - y_test)**2)))

                times.append(elapsed)
                nlls.append(nll)
                rmses.append(rmse)

            except Exception as e:
                print(f"  {name:<14} seed={seed} ERROR: {e}")
                times.append(np.nan)
                nlls.append(np.nan)
                rmses.append(np.nan)

        t_mean, t_std = np.nanmean(times), np.nanstd(times)
        nll_mean = np.nanmean(nlls)
        rmse_mean = np.nanmean(rmses)

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
    parser.add_argument("--n_train",  type=int, default=100)
    parser.add_argument("--n_test",   type=int, default=500)
    parser.add_argument("--n_seeds",  type=int, default=1)
    args = parser.parse_args()

    run_benchmark(n_train=args.n_train,
                  n_test=args.n_test, n_seeds=args.n_seeds)
