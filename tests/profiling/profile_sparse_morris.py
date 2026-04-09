"""
Profile sparse vs dense DEGP on Morris 20D with N=20 training points.

K matrix size: 20 * (1 + 20) = 420 x 420

This is a realistic high-dimensional benchmark where the O(N_d^3) cost
of dense Cholesky should start to matter.
"""

import os, sys, time
import numpy as np
from scipy.stats.qmc import LatinHypercube
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_functions import (
    morris, morris_gradient,
    generate_test_data, compute_metrics,
)
from jetgp.full_degp.degp import degp as DenseDEGP
from jetgp.full_degp_sparse.degp import degp as SparseDEGP


DIM = 20
N_TRAIN = 50
N_TEST = 500
DER_INDICES = [[[[i, 1]] for i in range(1, DIM + 1)]]

JADE_KWARGS = dict(
    n_generations=1,
    local_opt_every=1,
    pop_size=10,
    debug=True,
)


def time_nlml(model, x0, n_calls=5):
    """Time NLML evaluation (average over n_calls after warm-up)."""
    model.optimizer.negative_log_marginal_likelihood(x0)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        model.optimizer.negative_log_marginal_likelihood(x0)
    return (time.perf_counter() - t0) / n_calls


def time_nlml_and_grad(model, x0, n_calls=5):
    """Time joint NLML+gradient evaluation."""
    model.optimizer.nll_and_grad(x0)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        model.optimizer.nll_and_grad(x0)
    return (time.perf_counter() - t0) / n_calls


def main():
    print("=" * 90)
    print(f"  Morris 20D — N={N_TRAIN} training points")
    print(f"  K matrix: {N_TRAIN * (1 + DIM)} x {N_TRAIN * (1 + DIM)}")
    print("=" * 90)

    # Generate data
    sampler = LatinHypercube(d=DIM, seed=42)
    X_train = sampler.random(n=N_TRAIN)
    y_vals = morris(X_train)
    grads = morris_gradient(X_train)

    y_train_list = [y_vals.reshape(-1, 1)]
    for j in range(DIM):
        y_train_list.append(grads[:, j].reshape(-1, 1))

    X_test, y_test = generate_test_data(morris, N_TEST, DIM, seed=99)

    # Dummy hyperparams for timing NLML
    x0 = np.zeros(DIM + 2)
    x0[-1] = -3.0

    # ── Dense ───────────────────────────────────────────────────────────
    # print("\n  Building dense model...", flush=True)
    # t0 = time.perf_counter()
    # dense = DenseDEGP(
    #     X_train, y_train_list,
    #     n_order=1, n_bases=DIM,
    #     der_indices=DER_INDICES,
    #     normalize=True, kernel='SE', kernel_type='anisotropic',
    # )
    # t_build_dense = time.perf_counter() - t0

    # #t_nlml_dense = time_nlml(dense, x0)
    # #t_nllg_dense = time_nlml_and_grad(dense, x0)

    # #print(f"  Dense build:       {t_build_dense:.3f}s")
    # #print(f"  Dense NLML:        {t_nlml_dense:.4f}s")
    # #print(f"  Dense NLML+grad:   {t_nllg_dense:.4f}s")

    # print("  Dense optimising...", flush=True)
    # t0 = time.perf_counter()
    # dp = dense.optimize_hyperparameters(optimizer='lbfgs', **JADE_KWARGS)
    # t_opt_dense = time.perf_counter() - t0
    # print(f"  Dense opt time:    {t_opt_dense:.2f}s")

    # y_pred_d = dense.predict(X_test, dp).flatten()
    # m_d = compute_metrics(y_test, y_pred_d)
    # print(f"  Dense RMSE:        {m_d['rmse']:.4e}")
    # print(f"  Dense NRMSE:       {m_d['nrmse']:.4e}")

    # # ── Sparse at various rho ──────────────────────────────────────────
    rho_values = [1]

    print(f"\n  {'rho':>5} │ {'sp%':>6} {'#SN':>4} │ "
          f"{'build':>7} {'NLML':>8} {'NLL+g':>8} {'opt':>8} │ "
          f"{'RMSE':>10} {'NRMSE':>10}")
    print("  " + "─" * 85)

    for rho in rho_values:
        t0 = time.perf_counter()
        sparse = SparseDEGP(
            X_train, y_train_list,
            n_order=1, n_bases=DIM,
            der_indices=DER_INDICES,
            normalize=True, kernel='SE', kernel_type='anisotropic',
            rho=rho, use_supernodes=False,
        )
        t_build = time.perf_counter() - t0

        # Sparsity info
        S_full = sparse.sparse_S_full
        N_total = len(sparse.mmd_P_full)
        nnz = sum(len(v) for v in S_full.values())
        max_nnz = N_total * (N_total + 1) // 2
        sp_pct = 1 - nnz / max_nnz
        n_sn = len(sparse.sparse_supernodes) if sparse.sparse_supernodes else N_TRAIN

        #t_nlml = time_nlml(sparse, x0)
        #t_nllg = time_nlml_and_grad(sparse, x0)

        t0 = time.perf_counter()
        sp_params = sparse.optimize_hyperparameters(optimizer='jade', **JADE_KWARGS)
        t_opt = time.perf_counter() - t0

        y_pred_s = sparse.predict(X_test, sp_params).flatten()
        m_s = compute_metrics(y_test, y_pred_s)

        #print(f"  {rho:>5} │ {sp_pct:>5.0%} {n_sn:>4} │ "
        #      f"{t_build:>6.3f}s {t_nlml:>7.4f}s {t_nllg:>7.4f}s {t_opt:>7.2f}s │ "
        #      f"{m_s['rmse']:>10.4e} {m_s['nrmse']:>10.4e}")

    # Dense summary line
    #print("  " + "─" * 85)
    #print(f"  {'dense':>5} │ {'0%':>6} {'—':>4} │ "
    #      f"{t_build_dense:>6.3f}s {t_nlml_dense:>7.4f}s {t_nllg_dense:>7.4f}s {t_opt_dense:>7.2f}s │ "
    #      f"{m_d['rmse']:>10.4e} {m_d['nrmse']:>10.4e}")


if __name__ == '__main__':
    main()
