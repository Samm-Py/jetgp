"""
PVR (predictive variance reduction) sanity spike.

Validates that a basic function-only PVR scoring agrees with — or sensibly
diverges from — the current max-posterior-variance (MPV) policy when costs are
equal. Specifically, sets up the same fin GP at seed 5000, asks the existing
MPV optimizer for its next pick `x_mpv`, then computes PVR_f for `x_mpv` and
a batch of LHS candidates, and reports:

  - PVR_f(x_mpv) and where it ranks among the LHS candidates by PVR.
  - σ²(x_mpv) and where it ranks among them by raw posterior variance.

If MPV's pick is near the top of the PVR ranking, the formula is at least
consistent with the existing scoring under equal costs. If it ranks low, the
two criteria genuinely diverge — which is exactly the gap we suspect causes
CostAware's underperformance.
"""

import numpy as np

from acquisition import find_next_point_mpv
from doe_utils import lhs_design
from hypad_cost_aware_probe import fin_grad, fin_func, setup_problem
import example_3_hypad_fin_active_learning as fin
from adaptive_doe import AdaptiveDirectionalGP


def pvr_function_candidate(gp_model, params, x_candidate, Z_star):
    """Predictive variance reduction at Z_star from observing f(x_candidate).

    Returns a non-negative scalar; larger = candidate is more informative
    about the GP's posterior at Z_star.
    """
    x_candidate = np.atleast_2d(x_candidate)
    test = np.vstack([Z_star, x_candidate])
    mean, full_cov = gp_model.predict(
        test, params, calc_cov=True, return_full_cov=True)
    # full_cov is the full predictive covariance, shape (n_test, n_test)
    C = np.asarray(full_cov)
    if C.ndim == 3:
        C = C[0]
    sigma_n = 10.0 ** float(params[-1])
    sigma2_x = float(C[-1, -1])
    cov_zx = C[:-1, -1]
    denom = sigma2_x + sigma_n ** 2
    if denom <= 0.0:
        return np.nan
    return float(np.sum(cov_zx ** 2) / denom)


def build_gp(seed=5000, n_init=6):
    X_init, _, _ = setup_problem(seed=seed, n_init=n_init, n_val=10,
                                 case=2, time=1.0)
    al = AdaptiveDirectionalGP(
        func=fin_func, grad_func=fin_grad, bounds=fin.BOUNDS_Z,
        n_init=n_init, rel_tol=0.0, n_iter=0, seed=seed,
        c_f=1.0, c1=1.0, cost_budget=1.0,
        max_directions=7, X_init=X_init, verbose=False,
    )
    al._initialize_design()
    al._refit()
    return al


def main():
    rng = np.random.default_rng(123)
    al = build_gp(seed=5000, n_init=6)
    gp, params = al.gp_model, al.params

    x_mpv, var_mpv = find_next_point_mpv(gp, params, fin.BOUNDS_Z,
                                         n_restarts=12, seed=42)
    print(f"MPV pick:       x={x_mpv}")
    print(f"MPV pick var:   σ²(x)={var_mpv:.4e}")

    Z_star = lhs_design(100, fin.BOUNDS_Z, seed=int(rng.integers(1 << 30)))
    candidates = lhs_design(20, fin.BOUNDS_Z, seed=int(rng.integers(1 << 30)))
    candidates = np.vstack([x_mpv.reshape(1, -1), candidates])

    rows = []
    for i, x in enumerate(candidates):
        _, var_x = gp.predict(np.atleast_2d(x), params, calc_cov=True,
                              return_deriv=False)
        var_x = float(var_x[0])
        pvr = pvr_function_candidate(gp, params, x, Z_star)
        rows.append({"i": i, "x": x, "var": var_x, "pvr": pvr,
                     "is_mpv": i == 0})

    rows_by_pvr = sorted(rows, key=lambda r: -r["pvr"])
    rows_by_var = sorted(rows, key=lambda r: -r["var"])
    mpv_rank_by_pvr = next(j for j, r in enumerate(rows_by_pvr)
                            if r["is_mpv"]) + 1
    mpv_rank_by_var = next(j for j, r in enumerate(rows_by_var)
                            if r["is_mpv"]) + 1

    print(f"\n{'rank':>4s} {'is_mpv':>7s} {'σ²(x)':>12s} {'PVR_f(x;Z*)':>14s}")
    print("-" * 42)
    for j, r in enumerate(rows_by_pvr):
        tag = "MPV" if r["is_mpv"] else ""
        print(f"{j+1:4d} {tag:>7s} {r['var']:12.4e} {r['pvr']:14.4e}")

    print(f"\nMPV pick ranks #{mpv_rank_by_pvr} / {len(rows)} by PVR")
    print(f"MPV pick ranks #{mpv_rank_by_var} / {len(rows)} by raw σ²")
    print(f"(MPV pick should be near #1 by σ², as that's what MPV optimises.)")


if __name__ == "__main__":
    main()
