"""
Example 5 - Active learning on the HYPAD-UQ heated fin.

Starting from a small LHS design (plus the mean point), iteratively enrich
three GP surrogates (function-only, full-gradient DEGP, eigenbasis directional
GDDEGP) by adaptive MPV infill. After a few iterations, propagate the selected
input distribution through the analytical steady-state fin model and through
each GP surrogate, and compare output distributions.

Training, prediction and hyperparameter optimization are all performed in
standardized input space z = (x - mu) / sigma, where mu and sigma are taken
from Table 1 of Balcer et al. (2023). The GPs also use normalize=True.
"""

import argparse
from pathlib import Path

import numpy as np
import pyoti.sparse as oti
from scipy.stats import wasserstein_distance, gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from adaptive_doe import (
    find_next_point_mpv,
    fit_directional_gp,
    fit_function_only_gp,
    lhs_design,
    query_function_posterior,
    select_derivatives_at_observed_point,
)
from jetgp.full_degp.degp import degp


# ---------------------------------------------------------------------------
# Problem definition (matches example 4)
# ---------------------------------------------------------------------------

VAR_NAMES = [r"$k$", r"$C_p$", r"$\rho$", r"$h_U$",
             r"$T_\infty$", r"$T_W$", r"$b$"]

MEANS = np.array([7.1, 580.0, 4430.0, 114.0, 283.0, 389.0, 51.0e-3])
DELTA_CASE1 = np.array([0.10, 0.03, 0.03, 0.10, 0.001, 0.05, 0.01])
DELTA_CASE2 = np.array([0.20, 0.20, 0.20, 0.20, 0.01, 0.20, 0.20])

DELTA_THICK = 4.75e-3
L_DEPTH = 1.0

D = 7

DEFAULT_ACTIVE_CASE = 2
DEFAULT_TIMES = [10.0, 100.0, 200.0]
DEFAULT_DERIVATIVE_VARIANCE_TOL = 1e-8
TRANSIENT_SERIES_TERMS = 100
# The paper assumes the fin starts at ambient temperature, so
# h_0 = (T_0 - T_infinity) / (T_W - T_infinity) = 0.
NORMALIZED_INITIAL_TEMPERATURE = 0.0
ACTIVE_CASE = None
CASE_LABEL = None
SIGMAS = None
BOUNDS_Z = None
ACTIVE_TIME_SECONDS = None
DERIVATIVE_VARIANCE_TOL = DEFAULT_DERIVATIVE_VARIANCE_TOL


def z_to_x(z):
    z = np.atleast_2d(z)
    return MEANS[None, :] + SIGMAS[None, :] * z


def x_to_z(x):
    x = np.atleast_2d(x)
    return (x - MEANS[None, :]) / SIGMAS[None, :]


def T_tip_real(x):
    k, Cp, rho, hU, Tinf, TW, b = x
    omega = np.sqrt(2.0 * hU * b * b / (k * DELTA_THICK * L_DEPTH))
    tau = ACTIVE_TIME_SECONDS * k / (b * b * rho * Cp)
    h = 1.0 / np.cosh(omega)
    for j in range(1, TRANSIENT_SERIES_TERMS + 1):
        kj = np.pi * (2 * j - 1) / 2.0
        denom = omega * omega + kj * kj
        coeff = NORMALIZED_INITIAL_TEMPERATURE / kj - kj / denom
        h += 2.0 * ((-1.0) ** (j + 1)) * coeff * np.exp(-denom * tau)
    return Tinf + (TW - Tinf) * h


def T_tip_val_and_grad_z(z):
    """Evaluate T_tip and its gradient wrt z (standardized inputs) via pyoti."""
    x = MEANS + SIGMAS * z
    k_o  = x[0] + oti.e(1) * SIGMAS[0]
    Cp_o = x[1] + oti.e(2) * SIGMAS[1]
    rh_o = x[2] + oti.e(3) * SIGMAS[2]
    hU_o = x[3] + oti.e(4) * SIGMAS[3]
    Ti_o = x[4] + oti.e(5) * SIGMAS[4]
    TW_o = x[5] + oti.e(6) * SIGMAS[5]
    b_o  = x[6] + oti.e(7) * SIGMAS[6]
    omega = ((2.0 * hU_o * b_o * b_o) / (k_o * DELTA_THICK * L_DEPTH)) ** 0.5
    tau = ACTIVE_TIME_SECONDS * k_o / (b_o * b_o * rh_o * Cp_o)
    h = 1.0 / oti.cosh(omega)
    for j in range(1, TRANSIENT_SERIES_TERMS + 1):
        kj = np.pi * (2 * j - 1) / 2.0
        denom = omega ** 2 + kj ** 2
        coeff = NORMALIZED_INITIAL_TEMPERATURE / kj - kj / denom
        h = h + 2.0 * ((-1.0) ** (j + 1)) * coeff * oti.exp(-denom * tau)
    T = Ti_o + (TW_o - Ti_o) * h
    value = T.real
    grad = np.array([T.get_deriv([i]) for i in range(1, D + 1)])
    return value, grad


def T_tip_vec(z_array):
    """Vectorized real evaluation on an (n, D) z-array."""
    z_array = np.atleast_2d(z_array)
    x = z_to_x(z_array)
    return np.array([T_tip_real(x[i]) for i in range(x.shape[0])])


# ---------------------------------------------------------------------------
# GP fitting helpers (mirror example 3 conventions)
# ---------------------------------------------------------------------------

OPTIMIZER_KWARGS = {
    "optimizer": "jade",
    "pop_size": 50,
    "n_generations": 40,
    "local_opt_every": 20,
    "debug": True,
}


def optimizer_kwargs_with_warm_start(initial_params, n_params):
    """Use the previous optimum as JADE's first candidate when compatible."""
    opt_kwargs = dict(OPTIMIZER_KWARGS)
    if initial_params is None:
        return opt_kwargs

    initial_params = np.asarray(initial_params, dtype=float).reshape(-1)
    if initial_params.size == n_params and np.all(np.isfinite(initial_params)):
        opt_kwargs["initial_positions"] = np.atleast_2d(initial_params)
    return opt_kwargs


def active_case_bounds():
    if ACTIVE_CASE == 1:
        return np.array([[-3.0, 3.0]] * D)
    bounds = np.array([[-3.0, 3.0]] * D)
    bounds[4] = [-np.sqrt(6.0), np.sqrt(6.0)]
    bounds[5] = [-np.sqrt(3.0), np.sqrt(3.0)]
    bounds[6] = [-np.sqrt(3.0), np.sqrt(3.0)]
    return bounds


def configure_active_case(case_id):
    """Configure global standardized coordinates for Table 1 Case 1 or Case 2."""
    global ACTIVE_CASE, CASE_LABEL, SIGMAS, BOUNDS_Z
    if case_id not in (1, 2):
        raise ValueError("case_id must be 1 or 2.")
    ACTIVE_CASE = int(case_id)
    CASE_LABEL = f"Case {ACTIVE_CASE}"
    deltas = DELTA_CASE1 if ACTIVE_CASE == 1 else DELTA_CASE2
    SIGMAS = deltas * MEANS
    BOUNDS_Z = active_case_bounds()


def configure_active_time(time_seconds):
    """Configure the transient time at which the tip temperature is modeled."""
    global ACTIVE_TIME_SECONDS
    ACTIVE_TIME_SECONDS = float(time_seconds)


def configure_derivative_variance_tol(tol):
    """Configure the absolute leading-eigenvalue gate for derivative selection."""
    global DERIVATIVE_VARIANCE_TOL
    DERIVATIVE_VARIANCE_TOL = float(tol)


def format_time_label(time_seconds):
    return f"{time_seconds:g}s"


def study_label():
    return f"{CASE_LABEL}, t={ACTIVE_TIME_SECONDS:g} s"


configure_active_case(DEFAULT_ACTIVE_CASE)
configure_active_time(DEFAULT_TIMES[0])


def fit_full_gradient_degp(X_train, y_train, gradient_observations,
                           kernel="SE", kernel_type="anisotropic",
                           initial_params=None):
    d = X_train.shape[1]
    if len(gradient_observations) == 0:
        return fit_function_only_gp(
            X_train, y_train, n_dir_types=d,
            kernel=kernel, kernel_type=kernel_type,
            optimizer_kwargs=OPTIMIZER_KWARGS,
            normalize=True,
            initial_params=initial_params,
        )
    y_blocks = [y_train]
    derivative_locations = []
    for j in range(d):
        values = np.array([[obs["gradient"][j]] for obs in gradient_observations])
        y_blocks.append(values)
        derivative_locations.append([obs["x_index"] for obs in gradient_observations])
    der_indices = [[[[j + 1, 1]] for j in range(d)]]
    model = degp(
        X_train, y_blocks,
        n_order=1, n_bases=d,
        der_indices=der_indices,
        derivative_locations=derivative_locations,
        normalize=True, kernel=kernel, kernel_type=kernel_type,
    )
    opt_kwargs = optimizer_kwargs_with_warm_start(
        initial_params, len(model.bounds))
    params = model.optimize_hyperparameters(**opt_kwargs)
    return model, params


def fit_directional_model(X_train, y_train, directional_observations,
                          kernel="SE", kernel_type="anisotropic",
                          initial_params=None):
    if len(directional_observations) == 0:
        return fit_function_only_gp(
            X_train, y_train, n_dir_types=X_train.shape[1],
            kernel=kernel, kernel_type=kernel_type,
            optimizer_kwargs=OPTIMIZER_KWARGS,
            normalize=True,
            initial_params=initial_params,
        )
    return fit_directional_gp(
        X_train, y_train, directional_observations,
        kernel=kernel, kernel_type=kernel_type,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        initial_params=initial_params,
    )


def predict_function_mean(model, params, Z):
    pred = model.predict(np.atleast_2d(Z), params, calc_cov=False, return_deriv=False)
    if isinstance(pred, tuple):
        pred = pred[0]
    return np.asarray(pred)[0].reshape(-1)


# ---------------------------------------------------------------------------
# PDF-weighted MPV acquisition
#
# Standard MPV drives infills to the domain corners, which have negligible
# input-density for Gaussian inputs and therefore contribute almost nothing
# to the output distribution. We instead maximise
#     a(z) = sigma_f^2(z) * p(z),
# which is the greedy (pointwise) approximation of the weighted IMSE
# criterion (Sacks et al. 1989; Picheny et al. 2010; Gramacy 2020).
# For Case 1, p(z) is the standard multivariate normal density in z-space.
# For Case 2, p(z) is induced by the Table 1 log-normal/triangular/uniform
# input distributions after the standardization z = (x - mean) / sigma.
# ---------------------------------------------------------------------------

def case1_log_pdf(z):
    z = np.atleast_2d(z)
    return -0.5 * np.sum(z ** 2, axis=1)


def case2_log_pdf(z):
    z = np.atleast_2d(z)
    out = np.zeros(z.shape[0])

    # k, Cp, rho, hU are log-normal in physical x-space.
    cv = DELTA_CASE2[:4]
    sigma_x = cv * MEANS[:4]
    x = MEANS[:4][None, :] + sigma_x[None, :] * z[:, :4]
    valid = np.all(x > 0.0, axis=1)
    out[~valid] = -np.inf
    if np.any(valid):
        sigma_ln = np.sqrt(np.log1p(cv ** 2))
        mu_ln = np.log(MEANS[:4]) - 0.5 * sigma_ln ** 2
        xv = x[valid]
        log_pdf_x = (
            -np.log(xv)
            - np.log(sigma_ln[None, :])
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * ((np.log(xv) - mu_ln[None, :]) / sigma_ln[None, :]) ** 2
        )
        out[valid] += np.sum(log_pdf_x + np.log(sigma_x[None, :]), axis=1)

    # T_inf is symmetric triangular with sigma = dx * mean.
    a_tri = np.sqrt(6.0)
    z_tri = z[:, 4]
    valid_tri = np.abs(z_tri) <= a_tri
    tri_pdf = np.where(
        z_tri < 0.0,
        (z_tri + a_tri) / a_tri ** 2,
        (a_tri - z_tri) / a_tri ** 2,
    )
    out[~valid_tri] = -np.inf
    valid = np.isfinite(out) & valid_tri & (tri_pdf > 0.0)
    out[valid] += np.log(tri_pdf[valid])

    # T_W and b are uniform with sigma = dx * mean.
    a_uni = np.sqrt(3.0)
    z_uni = z[:, 5:7]
    valid_uni = np.all(np.abs(z_uni) <= a_uni, axis=1)
    out[~valid_uni] = -np.inf
    valid = np.isfinite(out) & valid_uni
    out[valid] += 2.0 * np.log(1.0 / (2.0 * a_uni))

    return out


def active_case_log_pdf(z):
    return case1_log_pdf(z) if ACTIVE_CASE == 1 else case2_log_pdf(z)


def find_next_point_weighted_mpv(gp_model, params, bounds,
                                 log_weight_fn=active_case_log_pdf,
                                 n_restarts=12, seed=123):
    """
    Argmax over z in bounds of  log(sigma_f^2(z)) + log_weight_fn(z).

    Working in log space avoids catastrophic underflow of the density term
    near the edges of the bounded domain.
    """
    lb, ub = bounds[:, 0], bounds[:, 1]
    starts = lhs_design(n_restarts, bounds, seed=seed)

    def neg_acq(z):
        _, var = query_function_posterior(gp_model, params, np.atleast_2d(z))
        v = float(var[0])
        if not np.isfinite(v) or v <= 0.0:
            return np.inf
        return -(np.log(v) + float(log_weight_fn(z)))

    best_z, best_val = None, np.inf
    for z0 in starts:
        res = minimize(neg_acq, x0=z0, method="L-BFGS-B",
                       bounds=list(zip(lb, ub)))
        z_cand = np.clip(res.x, lb, ub)
        val = neg_acq(z_cand)
        if val < best_val:
            best_val = val
            best_z = z_cand
    return best_z, -best_val


# ---------------------------------------------------------------------------
# Distribution comparison
# ---------------------------------------------------------------------------

def central_moments(values):
    v = np.asarray(values, dtype=float)
    mu = v.mean()
    s2 = v.var(ddof=0)
    s = np.sqrt(s2)
    skew = np.mean(((v - mu) / s) ** 3) if s > 0 else 0.0
    kurt = np.mean(((v - mu) / s) ** 4) if s > 0 else 0.0
    return dict(mean=mu, variance=s2, skewness=skew, kurtosis=kurt)


def compare_distributions(analytic, gp_predictions):
    """Return a dict of per-GP summary stats vs the analytic reference."""
    ref = central_moments(analytic)
    rows = {"Analytic (truth)": ref | dict(W1_to_truth=0.0)}
    for name, vals in gp_predictions.items():
        mom = central_moments(vals)
        w1 = wasserstein_distance(analytic, vals)
        rows[name] = mom | dict(W1_to_truth=w1)
    return rows


# ---------------------------------------------------------------------------
# Active learning
# ---------------------------------------------------------------------------

def run_active_learning(n_iter=5, n_init=2, seed=2026, verbose=True):
    rng = np.random.default_rng(seed)

    # Training data (z-space). Mean point + (n_init - 1) LHS points.
    if n_init < 2:
        raise ValueError("n_init must be >= 2 for normalize=True to work.")
    z_lhs = lhs_design(n_init - 1, BOUNDS_Z, seed=seed)
    Z0 = np.vstack([np.zeros((1, D)), z_lhs])
    Y0 = np.empty((n_init, 1))
    grads0 = np.empty((n_init, D))
    for i in range(n_init):
        f_i, g_i = T_tip_val_and_grad_z(Z0[i])
        Y0[i, 0] = f_i
        grads0[i] = g_i

    X_func = Z0.copy()
    y_func = Y0.copy()

    X_full = Z0.copy()
    y_full = Y0.copy()
    full_gradient_observations = [
        {"x_index": i, "gradient": grads0[i].copy()} for i in range(n_init)
    ]

    X_dir = Z0.copy()
    y_dir = Y0.copy()

    directional_observations = []

    history = []

    def fit_all(iter_label, func_initial=None, full_initial=None, dir_initial=None):
        func_model, func_params = fit_function_only_gp(
            X_func, y_func, n_dir_types=D,
            optimizer_kwargs=OPTIMIZER_KWARGS,
            normalize=True,
            initial_params=func_initial,
        )
        full_model, full_params = fit_full_gradient_degp(
            X_full, y_full, full_gradient_observations,
            initial_params=full_initial)
        dir_model, dir_params = fit_directional_model(
            X_dir, y_dir, directional_observations,
            initial_params=dir_initial)
        if verbose:
            print(f"  [fit@{iter_label}] "
                  f"N_func={X_func.shape[0]}, "
                  f"N_full_grads={len(full_gradient_observations)}, "
                  f"N_dir_obs={len(directional_observations)}")
        return (func_model, func_params,
                full_model, full_params,
                dir_model, dir_params)

    if verbose:
        print(f"Initial fits from {n_init} points (mean + LHS), "
              f"with no directional derivatives in the eigenbasis model")
    (func_model, func_params,
     full_model, full_params,
     dir_model, dir_params) = fit_all("init")

    # Initial derivative enrichment: at each DOE point, pick eigenbasis
    # directions from the current posterior, append the real projected
    # directional derivatives, refit, and then move to the next point.
    for i in range(n_init):
        selection = select_derivatives_at_observed_point(
            dir_model, dir_params, x_point=Z0[i], tau=0.05,
            lambda_abs_tol=DERIVATIVE_VARIANCE_TOL,
        )
        n_picked = 0
        for slot, direction in enumerate(selection["selected_directions"]):
            d_val = float(grads0[i] @ direction)
            directional_observations.append({
                "x_index": i,
                "direction": direction.copy(),
                "value": d_val,
                "slot": slot,
            })
            n_picked += 1
        if verbose:
            print(f"  [init] point {i}: picked {n_picked} directions "
                  f"(lambda_1={selection['var1']:.3g}, "
                  f"tol={DERIVATIVE_VARIANCE_TOL:.1e}, "
                  f"cumulative={len(directional_observations)})")
        if directional_observations and i < n_init - 1:
            dir_model, dir_params = fit_directional_model(
                X_dir, y_dir, directional_observations,
                initial_params=dir_params)
    if verbose:
        print(f"  [init] seeded {len(directional_observations)} "
              f"directional derivatives; final enrichment refit")
    dir_model, dir_params = fit_directional_model(
        X_dir, y_dir, directional_observations,
        initial_params=dir_params)

    history.append(dict(
        iteration=0,
        X_func=X_func.copy(), y_func=y_func.copy(),
        func_model=func_model, func_params=func_params,
        full_model=full_model, full_params=full_params,
        dir_model=dir_model, dir_params=dir_params,
        n_full_grads=len(full_gradient_observations),
        n_dir_obs=len(directional_observations),
    ))

    for it in range(1, n_iter + 1):
        if verbose:
            print(f"\n--- iter {it}/{n_iter} ---")

        # PDF-weighted MPV acquisition: max over z of sigma_f^2(z) * p(z),
        # where p(z) is the active input density in standardized z-space.
        z_new, log_acq = find_next_point_weighted_mpv(
            dir_model, dir_params, BOUNDS_Z,
            log_weight_fn=active_case_log_pdf,
            n_restarts=12, seed=seed + it,
        )
        _, var_at_new = query_function_posterior(
            dir_model, dir_params, np.atleast_2d(z_new)
        )
        mpv = float(var_at_new[0])
        if verbose:
            print(f"  z_new = {np.round(z_new, 3)}  |z|={np.linalg.norm(z_new):.2f}"
                  f"  sigma_f^2={mpv:.4g}  log(a)={log_acq:.3g}")

        # New scheme: evaluate the truth at z_new first, register it as an
        # observed function point in the directional GP, refit, then pick
        # eigenbasis directions from the REAL posterior at z_new (no fantasy).
        # Finally, project the already-computed true gradient onto those
        # directions to obtain the directional-derivative observations.
        f_new, g_new = T_tip_val_and_grad_z(z_new)

        new_idx_func = X_func.shape[0]
        X_func = np.vstack([X_func, z_new[None, :]])
        y_func = np.vstack([y_func, [[f_new]]])

        X_full = np.vstack([X_full, z_new[None, :]])
        y_full = np.vstack([y_full, [[f_new]]])
        full_gradient_observations.append(
            {"x_index": new_idx_func, "gradient": g_new.copy()}
        )

        X_dir = np.vstack([X_dir, z_new[None, :]])
        y_dir = np.vstack([y_dir, [[f_new]]])

        # Refit directional GP with z_new as a new function observation
        # (existing directional obs preserved) so z_new is a real training
        # point when we pick directions at it.
        dir_model, dir_params = fit_directional_model(
            X_dir, y_dir, directional_observations,
            initial_params=dir_params)

        selection = select_derivatives_at_observed_point(
            dir_model, dir_params, x_point=z_new, tau=0.05,
            lambda_abs_tol=DERIVATIVE_VARIANCE_TOL,
        )
        if verbose:
            print(f"  selected {len(selection['selected_directions'])} direction(s) "
                  f"(lambda_1={selection['var1']:.3g}, "
                  f"tol={DERIVATIVE_VARIANCE_TOL:.1e})")
        for slot, direction in enumerate(selection["selected_directions"]):
            d_val = float(g_new @ direction)
            directional_observations.append({
                "x_index": new_idx_func,
                "direction": direction.copy(),
                "value": d_val,
                "slot": slot,
            })

        (func_model, func_params,
         full_model, full_params,
         dir_model, dir_params) = fit_all(
             f"iter{it}",
             func_initial=func_params,
             full_initial=full_params,
             dir_initial=dir_params,
         )

        history.append(dict(
            iteration=it,
            X_func=X_func.copy(), y_func=y_func.copy(),
            func_model=func_model, func_params=func_params,
            full_model=full_model, full_params=full_params,
            dir_model=dir_model, dir_params=dir_params,
            n_full_grads=len(full_gradient_observations),
            n_dir_obs=len(directional_observations),
        ))

    # Annotate each history entry with a copy of the cumulative directional
    # observations at that point, so we can post-hoc inspect which directions
    # were acquired (and when).
    # We rebuild the cumulative list from the final obs list, assuming new
    # observations were appended in iteration order (this is the case here).
    cum = []
    for entry in history:
        target = entry["n_dir_obs"]
        cum = [dict(o, direction=o["direction"].copy())
               for o in directional_observations[:target]]
        entry["directional_observations"] = cum

    return history, directional_observations


# ---------------------------------------------------------------------------
# Validation + distribution propagation
# ---------------------------------------------------------------------------

def sample_case1_z(n, rng):
    """Case 1 is all-normal, so samples in z-space are N(0, 1)."""
    return rng.normal(size=(n, D))


def sample_case2_z(n, rng):
    """
    Draw Case 2 samples in standardized z-space, z=(x-mean)/(dx*mean).

    The physical distributions follow Table 1: log-normal for k, Cp, rho, hU;
    symmetric triangular for T_inf; and uniform for T_W and b.
    """
    z = np.empty((n, D))

    cv = DELTA_CASE2[:4]
    sigma_ln = np.sqrt(np.log1p(cv ** 2))
    mu_ln = np.log(MEANS[:4]) - 0.5 * sigma_ln ** 2
    x_logn = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=(n, 4))
    z[:, :4] = (x_logn - MEANS[:4][None, :]) / (cv[None, :] * MEANS[:4][None, :])

    z[:, 4] = rng.triangular(-np.sqrt(6.0), 0.0, np.sqrt(6.0), size=n)
    z[:, 5] = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=n)
    z[:, 6] = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=n)
    return z


def sample_active_case_z(n, rng):
    return sample_case1_z(n, rng) if ACTIVE_CASE == 1 else sample_case2_z(n, rng)


def gp_rmse(model, params, Z_val, y_val):
    y_pred = predict_function_mean(model, params, Z_val)
    err = y_pred - y_val
    return float(np.sqrt(np.mean(err ** 2)))


def gp_nrmse(model, params, Z_val, y_val):
    """Validation RMSE normalized by the truth range on the validation set."""
    scale = float(np.max(y_val) - np.min(y_val))
    if scale <= 0.0:
        scale = max(float(np.max(np.abs(y_val))), 1.0)
    return gp_rmse(model, params, Z_val, y_val) / scale


METHOD_KEYS = [
    ("Function-only GP",   "func_model", "func_params"),
    ("Full-gradient DEGP", "full_model", "full_params"),
    ("Eigenbasis GDDEGP",  "dir_model",  "dir_params"),
]


def _predict_all(history_entry, Z_mc):
    preds = {}
    for label, model_key, param_key in METHOD_KEYS:
        preds[label] = predict_function_mean(
            history_entry[model_key], history_entry[param_key], Z_mc
        )
    return preds


def propagate_and_summarize(history, n_mc=5000, seed=7):
    rng = np.random.default_rng(seed)
    Z_mc = sample_active_case_z(n_mc, rng)
    y_truth = T_tip_vec(Z_mc)
    preds = _predict_all(history[-1], Z_mc)
    stats = compare_distributions(y_truth, preds)
    return dict(Z_mc=Z_mc, y_truth=y_truth, preds=preds, stats=stats)


METHOD_KINDS = {
    "Function-only GP": "gddegp",
    "Full-gradient DEGP": "degp",
    "Eigenbasis GDDEGP": "gddegp",
}


def gp_active_subspace_history(history, AS_eigvals_truth, AS_eigvecs_truth,
                               k_active=5, n_samples=1500, seed=23):
    """
    For each iteration and each surrogate, compute the active subspace from
    the GP's analytic posterior-mean gradient sampled from the active input
    distribution, so the AS is weighted by the actual input density, and
    compare to the ground-truth active subspace.

    Returns a dict keyed by method label with per-iteration lists of:
        eigvals, activity_scores, alignment_k (with top-k_active subspace),
        dominant_cosine (|w1_GP . w1_truth|).
    """
    rng = np.random.default_rng(seed)
    Z = sample_active_case_z(n_samples, rng)

    W_truth_k = AS_eigvecs_truth[:, :k_active]

    out = {label: dict(iterations=[], n_func=[],
                       eigvals=[], activity=[],
                       align_k=[], dom_cos=[]) for label in METHOD_KINDS}

    for h in history:
        for label, kind in METHOD_KINDS.items():
            model_key = {"Function-only GP": "func_model",
                         "Full-gradient DEGP": "full_model",
                         "Eigenbasis GDDEGP": "dir_model"}[label]
            param_key = {"Function-only GP": "func_params",
                         "Full-gradient DEGP": "full_params",
                         "Eigenbasis GDDEGP": "dir_params"}[label]
            try:
                grads = gp_gradients(h[model_key], h[param_key], Z, kind)
                ev, W, _ = active_subspace_from_grads(grads)
            except Exception as exc:
                print(f"  [warn] {label} gradient extraction failed "
                      f"at iter {h['iteration']}: {exc}")
                ev = np.full(D, np.nan)
                W = np.eye(D)

            activity = np.sum(ev[None, :] * W ** 2, axis=1)
            W_k = W[:, :k_active]
            align = subspace_alignment(W_k, W_truth_k)
            w1_gp = W[:, 0]
            w1_tr = AS_eigvecs_truth[:, 0]
            dom_cos = float(abs(w1_gp @ w1_tr))

            out[label]["iterations"].append(h["iteration"])
            out[label]["n_func"].append(h["X_func"].shape[0])
            out[label]["eigvals"].append(ev)
            out[label]["activity"].append(activity)
            out[label]["align_k"].append(align)
            out[label]["dom_cos"].append(dom_cos)

    for label in out:
        out[label]["eigvals"] = np.array(out[label]["eigvals"])
        out[label]["activity"] = np.array(out[label]["activity"])

    return out


def plot_gp_active_subspace_history(as_hist, AS_eigvals_truth,
                                    AS_eigvecs_truth, outdir,
                                    case_label="Case 1"):
    labels = list(as_hist.keys())
    n_iters = len(next(iter(as_hist.values()))["iterations"])

    # Fig 1: eigenvalue spectrum at final iteration vs truth.
    fig, axes = plt.subplots(1, len(labels) + 1, figsize=(4 * (len(labels) + 1), 3.8),
                             sharey=True)
    axes[0].semilogy(np.arange(1, D + 1),
                     np.maximum(AS_eigvals_truth, 1e-30), "ko-")
    axes[0].set_title("Truth")
    axes[0].set_xlabel("index")
    axes[0].set_ylabel(r"$\lambda_j$")
    axes[0].grid(True, which="both", alpha=0.3)
    for ax, label in zip(axes[1:], labels):
        ev_final = as_hist[label]["eigvals"][-1]
        ax.semilogy(np.arange(1, D + 1),
                    np.maximum(ev_final, 1e-30), "o-")
        ax.set_title(label)
        ax.set_xlabel("index")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"{case_label}: GP-derived AS eigenvalue spectrum (final iter)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / "gp_as_spectrum_final.png", dpi=180)
    plt.close(fig)

    # Fig 2: subspace alignment and |w1_GP . w1_truth| over iterations.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    markers = {"Function-only GP": "o",
               "Full-gradient DEGP": "s",
               "Eigenbasis GDDEGP": "^"}
    for label in labels:
        n_f = as_hist[label]["n_func"]
        axes[0].plot(n_f, as_hist[label]["align_k"],
                     markers[label] + "-", label=label)
        axes[1].plot(n_f, as_hist[label]["dom_cos"],
                     markers[label] + "-", label=label)
    axes[0].set_xlabel("# function evaluations")
    axes[0].set_ylabel("top-5 subspace alignment (cos² avg)")
    axes[0].set_ylim(0, 1.02)
    axes[0].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[0].set_title("Active-subspace convergence")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("# function evaluations")
    axes[1].set_ylabel(r"$|w_1^{GP} \cdot w_1^{truth}|$")
    axes[1].set_ylim(0, 1.02)
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set_title("Leading-direction alignment")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"{case_label}: GP active subspace vs truth")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / "gp_as_alignment.png", dpi=180)
    plt.close(fig)

    # Fig 3: per-variable activity score bar chart at final iteration.
    truth_activity = np.sum(AS_eigvals_truth[None, :] * AS_eigvecs_truth ** 2,
                            axis=1)
    truth_activity = truth_activity / truth_activity.sum()
    fig, axes = plt.subplots(1, len(labels) + 1, figsize=(4 * (len(labels) + 1), 3.8),
                             sharey=True)
    axes[0].bar(np.arange(D), truth_activity, color="k", alpha=0.7)
    axes[0].set_title("Truth")
    axes[0].set_xticks(np.arange(D))
    axes[0].set_xticklabels(VAR_NAMES, fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("normalized activity score")
    for ax, label in zip(axes[1:], labels):
        a = as_hist[label]["activity"][-1]
        a = a / a.sum() if a.sum() > 0 else a
        ax.bar(np.arange(D), a)
        ax.set_xticks(np.arange(D))
        ax.set_xticklabels(VAR_NAMES, fontsize=8)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"{case_label}: per-variable activity at final iter")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / "gp_as_activity_final.png", dpi=180)
    plt.close(fig)


def metric_history(history, n_mc=5000, seed=7):
    """
    Propagate the fixed MC sample through each surrogate at every iteration,
    returning per-iteration distribution metrics.

    Returns dict:
        iterations : list of iteration indices
        n_func     : list of # function evaluations at each entry
        n_deriv    : dict {method_label: list of derivative-observation count}
        metrics    : dict {method_label: {metric_key: list_of_values}}
    """
    rng = np.random.default_rng(seed)
    Z_mc = sample_active_case_z(n_mc, rng)
    y_truth = T_tip_vec(Z_mc)
    truth_moments = central_moments(y_truth)

    metric_keys = ["mean", "variance", "skewness", "kurtosis", "W1_to_truth"]
    iterations = []
    n_func_list = []
    n_deriv = {lbl: [] for lbl, _, _ in METHOD_KEYS}
    metrics = {lbl: {k: [] for k in metric_keys} for lbl, _, _ in METHOD_KEYS}

    d = history[0]["X_func"].shape[1]

    for h in history:
        iterations.append(h["iteration"])
        n_func_list.append(h["X_func"].shape[0])
        preds = _predict_all(h, Z_mc)

        # Derivative-observation counts per method at this iteration.
        n_deriv["Function-only GP"].append(0)
        n_deriv["Full-gradient DEGP"].append(d * h["n_full_grads"])
        n_deriv["Eigenbasis GDDEGP"].append(h["n_dir_obs"])

        for label, _, _ in METHOD_KEYS:
            vals = preds[label]
            mom = central_moments(vals)
            w1 = wasserstein_distance(y_truth, vals)
            metrics[label]["mean"].append(mom["mean"])
            metrics[label]["variance"].append(mom["variance"])
            metrics[label]["skewness"].append(mom["skewness"])
            metrics[label]["kurtosis"].append(mom["kurtosis"])
            metrics[label]["W1_to_truth"].append(w1)

    return dict(
        iterations=iterations,
        n_func=n_func_list,
        n_deriv=n_deriv,
        metrics=metrics,
        truth_moments=truth_moments,
        metric_keys=metric_keys,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curves(history, Z_val, y_val, outdir, case_label=None):
    if case_label is None:
        case_label = CASE_LABEL
    its = [h["iteration"] for h in history]
    n_func = [h["X_func"].shape[0] for h in history]
    func_nrmse = [gp_nrmse(h["func_model"], h["func_params"], Z_val, y_val) for h in history]
    full_nrmse = [gp_nrmse(h["full_model"], h["full_params"], Z_val, y_val) for h in history]
    dir_nrmse = [gp_nrmse(h["dir_model"], h["dir_params"], Z_val, y_val) for h in history]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(n_func, func_nrmse, "o-", label="Function-only GP")
    ax.semilogy(n_func, full_nrmse, "s-", label="Full-gradient DEGP")
    ax.semilogy(n_func, dir_nrmse, "^-", label="Eigenbasis GDDEGP")
    ax.set_xlabel("# function evaluations")
    ax.set_ylabel("Validation NRMSE")
    ax.set_title(f"Active-learning progress ({case_label})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "learning_curves.png", dpi=180)
    plt.close(fig)
    return dict(its=its, n_func=n_func,
                func=func_nrmse, full=full_nrmse, dir=dir_nrmse)


def plot_learning_curves_vs_cost(history, Z_val, y_val, outdir,
                                 case_label="Case 1"):
    """
    Cost-normalized learning curve: NRMSE vs total information budget per
    method, where the cost is the number of scalar observations ingested
    (function values + derivative components).

        Function-only GP   : cost = N_f
        Full-gradient DEGP : cost = N_f + D * N_f_with_grads
        Eigenbasis GDDEGP  : cost = N_f + N_directional_observations

    Each method gets its own x-axis (total scalar observations); all three
    curves share the same y-axis (validation NRMSE).
    """
    d = history[0]["X_func"].shape[1]

    n_func = np.array([h["X_func"].shape[0] for h in history])
    func_nrmse = np.array([gp_nrmse(h["func_model"], h["func_params"], Z_val, y_val)
                           for h in history])
    full_nrmse = np.array([gp_nrmse(h["full_model"], h["full_params"], Z_val, y_val)
                           for h in history])
    dir_nrmse = np.array([gp_nrmse(h["dir_model"], h["dir_params"], Z_val, y_val)
                          for h in history])

    cost_func = n_func
    cost_full = n_func + d * np.array([h["n_full_grads"] for h in history])
    cost_dir = n_func + np.array([h["n_dir_obs"] for h in history])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogy(cost_full, full_nrmse, "s-", label="Full-gradient DEGP")
    ax.semilogy(cost_dir, dir_nrmse, "^-", label="Eigenbasis GDDEGP")
    ax.set_xlabel("# function evaluations + # derivative observations")
    ax.set_ylabel("Validation NRMSE")
    ax.set_title(f"{case_label}: cost-normalized learning curve")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "learning_curves_vs_cost.png", dpi=180)
    plt.close(fig)

    print("\nCost-normalized learning curve (NRMSE vs N_f + N_deriv):")
    print(f"  {'iter':>4s} {'func (cost, NRMSE)':>26s} "
          f"{'DEGP (cost, NRMSE)':>26s} {'GDDEGP (cost, NRMSE)':>26s}")
    for i, h in enumerate(history):
        print(f"  {h['iteration']:>4d} "
              f"{cost_func[i]:>12d}, {func_nrmse[i]:>10.4g}  "
              f"{cost_full[i]:>12d}, {full_nrmse[i]:>10.4g}  "
              f"{cost_dir[i]:>12d}, {dir_nrmse[i]:>10.4g}")

    return dict(cost_func=cost_func, cost_full=cost_full, cost_dir=cost_dir,
                func_nrmse=func_nrmse, full_nrmse=full_nrmse, dir_nrmse=dir_nrmse)


def plot_relative_metric_errors(mh, outdir, case_label="Case 1"):
    """
    Relative error of each moment (mean, variance, skewness, kurtosis) vs
    the number of derivative observations per surrogate, on a log y-axis.
    W1 is plotted as an absolute distance (no truth reference = 0).

        rel_err = |metric_gp - metric_truth| / max(|metric_truth|, eps)

    Per method the x-axis is its own total # of scalar derivative
    observations; a shared MC reference sample is used throughout.
    """
    truth = mh["truth_moments"]
    truth_lookup = dict(
        mean=truth["mean"], variance=truth["variance"],
        skewness=truth["skewness"], kurtosis=truth["kurtosis"],
    )
    eps = 1e-30
    markers = {"Function-only GP": "o",
               "Full-gradient DEGP": "s",
               "Eigenbasis GDDEGP": "^"}
    moment_keys = ["mean", "variance", "skewness", "kurtosis"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
    for ax, key in zip(axes[:4], moment_keys):
        denom = max(abs(truth_lookup[key]), eps)
        for label in mh["metrics"]:
            x = mh["n_deriv"][label]
            y_abs = np.array(mh["metrics"][label][key])
            rel = np.abs(y_abs - truth_lookup[key]) / denom
            rel = np.maximum(rel, eps)
            ax.semilogy(x, rel, markers[label] + "-", label=label)
        ax.set_xlabel("# derivative observations")
        ax.set_ylabel(f"relative error in {key}")
        ax.set_title(f"{key} (truth={truth_lookup[key]:.4g})")
        ax.grid(True, which="both", alpha=0.3)

    # W1 stays absolute (truth reference is 0 so relative error is undefined).
    ax = axes[4]
    for label in mh["metrics"]:
        x = mh["n_deriv"][label]
        y = np.array(mh["metrics"][label]["W1_to_truth"])
        y = np.maximum(y, eps)
        ax.semilogy(x, y, markers[label] + "-", label=label)
    ax.set_xlabel("# derivative observations")
    ax.set_ylabel(r"$W_1$ to truth")
    ax.set_title("W1 (absolute)")
    ax.grid(True, which="both", alpha=0.3)

    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{case_label}: relative error in distribution moments vs "
                 f"# derivative observations")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / "relative_metric_errors_vs_n_deriv.png", dpi=180)
    plt.close(fig)

    # Also a version against # function evaluations for apples-to-apples.
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
    for ax, key in zip(axes[:4], moment_keys):
        denom = max(abs(truth_lookup[key]), eps)
        for label in mh["metrics"]:
            y_abs = np.array(mh["metrics"][label][key])
            rel = np.abs(y_abs - truth_lookup[key]) / denom
            rel = np.maximum(rel, eps)
            ax.semilogy(mh["n_func"], rel, markers[label] + "-", label=label)
        ax.set_xlabel("# function evaluations")
        ax.set_ylabel(f"relative error in {key}")
        ax.set_title(f"{key} (truth={truth_lookup[key]:.4g})")
        ax.grid(True, which="both", alpha=0.3)
    ax = axes[4]
    for label in mh["metrics"]:
        y = np.array(mh["metrics"][label]["W1_to_truth"])
        y = np.maximum(y, eps)
        ax.semilogy(mh["n_func"], y, markers[label] + "-", label=label)
    ax.set_xlabel("# function evaluations")
    ax.set_ylabel(r"$W_1$ to truth")
    ax.set_title("W1 (absolute)")
    ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{case_label}: relative error in distribution moments vs "
                 f"# function evaluations")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / "relative_metric_errors_vs_n_func.png", dpi=180)
    plt.close(fig)


def plot_output_distributions(y_truth, preds, outdir, case_label="Case 1"):
    lo = min(y_truth.min(), *(p.min() for p in preds.values()))
    hi = max(y_truth.max(), *(p.max() for p in preds.values()))
    grid = np.linspace(lo, hi, 400)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(grid, gaussian_kde(y_truth)(grid), "k-", lw=2.5,
            label="Analytic truth")
    for name, vals in preds.items():
        ax.plot(grid, gaussian_kde(vals)(grid), "--", lw=1.6, label=name)
    ax.set_xlabel(r"$T_\mathrm{tip}$ (K)")
    ax.set_ylabel("PDF")
    ax.set_title(f"{case_label}: output distribution from each surrogate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "output_distributions.png", dpi=180)
    plt.close(fig)


def plot_metric_evolution(mh, outdir, case_label="Case 1"):
    """
    Plot how each distribution metric evolves with total scalar observation
    budget, N_f + N_deriv. Each method follows its own x-axis, so curves are
    not at a shared x but share the same MC reference sample.
    """
    metric_keys = mh["metric_keys"]
    truth = mh["truth_moments"]
    truth_lookup = dict(
        mean=truth["mean"], variance=truth["variance"],
        skewness=truth["skewness"], kurtosis=truth["kurtosis"],
        W1_to_truth=0.0,
    )

    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
    markers = {"Function-only GP": "o",
               "Full-gradient DEGP": "s",
               "Eigenbasis GDDEGP": "^"}
    derivative_labels = [
        "Full-gradient DEGP",
        "Eigenbasis GDDEGP",
    ]

    for ax, key in zip(axes, metric_keys):
        plotted_values = []
        for label in derivative_labels:
            x = np.asarray(mh["n_func"]) + np.asarray(mh["n_deriv"][label])
            y = mh["metrics"][label][key]
            plotted_values.extend(y)
            ax.plot(x, y, markers[label] + "-", label=label)
        truth_value = truth_lookup[key]
        if truth_value > 0.0:
            ax.axhline(truth_value, color="k", lw=1.2, ls="--",
                       label="Analytic truth" if key == metric_keys[0] else None)
        ax.set_xlabel("# function + derivative observations")
        ax.set_title(key)
        if plotted_values and all(v > 0 for v in plotted_values):
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{case_label}: output-distribution metrics vs "
                 f"total scalar observations per surrogate")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / "metric_evolution_vs_n_deriv.png", dpi=180)
    plt.close(fig)

    # Also plot vs # function evaluations (shared x-axis across methods).
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
    for ax, key in zip(axes, metric_keys):
        for label in mh["metrics"]:
            ax.plot(mh["n_func"], mh["metrics"][label][key],
                    markers[label] + "-", label=label)
        ax.axhline(truth_lookup[key], color="k", lw=1.2, ls="--")
        ax.set_xlabel("# function evaluations")
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
        if key == "W1_to_truth" and all(v > 0 for v in
                                        mh["metrics"]["Full-gradient DEGP"][key]):
            ax.set_yscale("log")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{case_label}: output-distribution metrics vs # function evaluations")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / "metric_evolution_vs_n_func.png", dpi=180)
    plt.close(fig)


def print_metric_history(mh):
    print("\nMetric evolution (per iteration):")
    header = ("  iter  N_f  " +
              "".join(f"{lbl[:18]:>20s} N_deriv " for lbl in mh["metrics"]))
    print(header)
    for i, it in enumerate(mh["iterations"]):
        line = f"  {it:>3d} {mh['n_func'][i]:>5d}"
        for lbl in mh["metrics"]:
            w1 = mh["metrics"][lbl]["W1_to_truth"][i]
            nd = mh["n_deriv"][lbl][i]
            line += f"  W1={w1:>8.4g} nD={nd:>4d}"
        print(line)


# ---------------------------------------------------------------------------
# GP-derived active subspace via the analytic derivative prediction
# ---------------------------------------------------------------------------

def _gp_gradients_degp(model, params, Z):
    """Coordinate gradients from the full-gradient DEGP at each row of Z."""
    derivs = [[[j + 1, 1]] for j in range(D)]
    mean = model.predict(
        np.atleast_2d(Z), params,
        calc_cov=False, return_deriv=True,
        derivs_to_predict=derivs,
    )
    # mean shape: (num_derivs + 1, n_test). Row 0 is f_mean; rows 1..D are grads.
    return np.asarray(mean)[1:, :].T


def _gp_gradients_gddegp(model, params, Z):
    """Coordinate gradients from a (directional / function-only) GDDEGP."""
    Z = np.atleast_2d(Z)
    n = Z.shape[0]
    eye = np.eye(D)
    rays_predict = [np.tile(eye[j:j + 1].T, (1, n)) for j in range(D)]
    derivs = [[[j + 1, 1]] for j in range(D)]
    mean = model.predict(
        Z, params,
        rays_predict=rays_predict,
        calc_cov=False, return_deriv=True,
        derivs_to_predict=derivs,
    )
    return np.asarray(mean)[1:, :].T


def gp_gradients(model, params, Z, kind):
    if kind == "degp":
        return _gp_gradients_degp(model, params, Z)
    return _gp_gradients_gddegp(model, params, Z)


def active_subspace_from_grads(grads):
    C = grads.T @ grads / grads.shape[0]
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order], C


def principal_angles(W_a, W_b):
    Q_a, _ = np.linalg.qr(W_a)
    Q_b, _ = np.linalg.qr(W_b)
    s = np.linalg.svd(Q_a.T @ Q_b, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def subspace_alignment(W_a, W_b):
    """Scalar in [0, 1]: mean cos^2 of principal angles (1 = identical k-subspace)."""
    return float(np.mean(np.cos(principal_angles(W_a, W_b)) ** 2))


# ---------------------------------------------------------------------------
# Active-subspace reference + direction-acquisition analysis
# ---------------------------------------------------------------------------

def compute_global_active_subspace(n_samples=2000, seed=17):
    """
    Reference active subspace of T_tip computed in z-space by averaging
    gradient outer products over samples from the active input distribution.

    Returns (eigvals, eigvecs) sorted in descending order of eigvals.
    """
    rng = np.random.default_rng(seed)
    Z = sample_active_case_z(n_samples, rng)
    grads = np.empty_like(Z)
    for i in range(n_samples):
        _, g = T_tip_val_and_grad_z(Z[i])
        grads[i] = g
    C = grads.T @ grads / n_samples
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def analyze_acquired_directions(directional_observations, AS_eigvecs):
    """
    Project each acquired directional-derivative unit vector onto the global
    active-subspace basis. Returns an (n_obs, D) matrix of squared projections
    (each row sums to 1) and the raw directions stacked as (n_obs, D).
    """
    if len(directional_observations) == 0:
        return np.zeros((0, D)), np.zeros((0, D))
    V = np.vstack([o["direction"] for o in directional_observations])
    # Re-normalize to guard against numerical drift.
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    projections = (V @ AS_eigvecs) ** 2   # (n_obs, D)
    return projections, V


def plot_active_subspace_reference(eigvals, eigvecs, outdir, case_label="Case 1"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Spectrum.
    ax = axes[0]
    ax.semilogy(np.arange(1, D + 1), np.maximum(eigvals, 1e-30), "o-")
    ax.set_xlabel("Index $j$")
    ax.set_ylabel(r"$\lambda_j$")
    ax.set_title(f"{case_label}: global active-subspace spectrum")
    ax.grid(True, which="both", alpha=0.3)

    # Top-2 eigenvector composition.
    ax = axes[1]
    width = 0.35
    x = np.arange(D)
    w1 = eigvecs[:, 0]
    w2 = eigvecs[:, 1]
    if w1[np.argmax(np.abs(w1))] < 0: w1 = -w1
    if w2[np.argmax(np.abs(w2))] < 0: w2 = -w2
    ax.bar(x - width / 2, w1, width, label=rf"$w_1$ ($\lambda_1$={eigvals[0]:.3g})")
    ax.bar(x + width / 2, w2, width, label=rf"$w_2$ ($\lambda_2$={eigvals[1]:.3g})")
    ax.set_xticks(x)
    ax.set_xticklabels(VAR_NAMES)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"{case_label}: leading active-subspace directions")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "active_subspace_reference.png", dpi=180)
    plt.close(fig)


def plot_acquired_directions(directional_observations, AS_eigvals, AS_eigvecs,
                             outdir, case_label="Case 1"):
    projections, V = analyze_acquired_directions(
        directional_observations, AS_eigvecs
    )
    if V.shape[0] == 0:
        print("  (no directional observations to plot)")
        return

    n_obs = V.shape[0]
    idx = np.arange(1, n_obs + 1)

    # Figure 1: stacked bars of squared projection onto each AS eigenvector.
    fig, ax = plt.subplots(figsize=(max(6.5, 0.25 * n_obs + 2), 4))
    bottom = np.zeros(n_obs)
    cmap = plt.get_cmap("viridis")
    for k in range(D):
        heights = projections[:, k]
        ax.bar(idx, heights, bottom=bottom,
               color=cmap(k / max(D - 1, 1)),
               label=rf"$w_{k + 1}$ ($\lambda$={AS_eigvals[k]:.2g})")
        bottom += heights
    ax.set_xlabel("Acquisition order")
    ax.set_ylabel(r"$(v \cdot w_k)^2$")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{case_label}: acquired directions projected onto global AS")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "acquired_directions_projection.png", dpi=180)
    plt.close(fig)

    # Figure 2: mean squared projection per AS eigenvector (summary bar).
    fig, ax = plt.subplots(figsize=(6.5, 4))
    mean_proj = projections.mean(axis=0)
    ax.bar(np.arange(1, D + 1), mean_proj,
           color=[cmap(k / max(D - 1, 1)) for k in range(D)])
    ax.set_xticks(np.arange(1, D + 1))
    ax.set_xticklabels([rf"$w_{k + 1}$" for k in range(D)])
    ax.set_ylabel(r"mean $(v \cdot w_k)^2$")
    ax.set_title(f"{case_label}: average projection of acquired directions "
                 f"(N={n_obs})")
    ax.axhline(1.0 / D, color="k", ls="--", lw=1,
               label=f"isotropic reference = 1/{D}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "acquired_directions_mean_projection.png", dpi=180)
    plt.close(fig)

    # Figure 3: each acquired direction as a bar over the 7 physical variables.
    # Only show at most the last 16 to keep the figure readable.
    n_show = min(n_obs, 16)
    cols = 4
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.2 * rows),
                             sharey=True)
    axes = np.atleast_2d(axes).ravel()
    for i in range(n_show):
        v = V[-(n_show - i)]
        axes[i].bar(np.arange(D), v)
        axes[i].set_xticks(np.arange(D))
        axes[i].set_xticklabels(VAR_NAMES, fontsize=7)
        axes[i].axhline(0, color="k", lw=0.5)
        axes[i].set_title(f"acq #{n_obs - n_show + i + 1}", fontsize=9)
        axes[i].grid(True, axis="y", alpha=0.3)
    for j in range(n_show, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{case_label}: last {n_show} acquired directions (z-space)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "acquired_directions_bars.png", dpi=180)
    plt.close(fig)

    print(f"  Acquired {n_obs} directions total")
    print(f"  Mean squared projection per AS eigenvector:")
    for k in range(D):
        print(f"    w{k + 1} (lambda={AS_eigvals[k]:.3g}): "
              f"{mean_proj[k]:.4f}")


def print_stats_table(stats):
    keys = ["mean", "variance", "skewness", "kurtosis", "W1_to_truth"]
    header = "  " + "Method".ljust(22) + "".join(f"{k:>14s}" for k in keys)
    print(header)
    print("  " + "-" * (22 + 14 * len(keys)))
    for name, row in stats.items():
        line = "  " + name.ljust(22)
        for k in keys:
            line += f"{row[k]:>14.5g}"
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the HYPAD-UQ heated-fin active-learning example."
    )
    parser.add_argument(
        "--case",
        type=int,
        choices=(1, 2),
        default=DEFAULT_ACTIVE_CASE,
        help="Table 1 input case to use: 1=all-normal low-COV, 2=non-Gaussian large-COV.",
    )
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=DEFAULT_TIMES,
        help="Transient time point(s), in seconds, to study.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of active-learning iterations per time point.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for DOE, validation, and Monte Carlo diagnostics.",
    )
    parser.add_argument(
        "--deriv-var-tol",
        type=float,
        default=DEFAULT_DERIVATIVE_VARIANCE_TOL,
        help=(
            "Absolute tolerance on the leading local derivative-variance "
            "eigenvalue. If lambda_1 is below this value, no directional "
            "derivatives are acquired at that point."
        ),
    )
    return parser.parse_args()


def run_complete_study(n_iter=20, seed=42, verbose=True):
    label = study_label()
    print("=" * 72)
    print(f"Example 5 - HYPAD-UQ fin ({label}): active learning from two points")
    print(f"Directional derivative variance tolerance: {DERIVATIVE_VARIANCE_TOL:.1e}")
    print("=" * 72)

    case_tag = CASE_LABEL.lower().replace(" ", "_")
    time_tag = format_time_label(ACTIVE_TIME_SECONDS).replace(".", "p")
    outdir = Path(__file__).parent / f"example_5_{case_tag}_t{time_tag}_figures"
    outdir.mkdir(exist_ok=True)

    history, directional_observations = run_active_learning(
        n_iter=n_iter, seed=seed, verbose=verbose
    )

    # Validation set in z-space sampled from the active input distribution.
    rng_val = np.random.default_rng(seed)
    Z_val = sample_active_case_z(3000, rng_val)
    y_val = T_tip_vec(Z_val)

    print("\nLearning curves (NRMSE vs iteration):")
    curves = plot_learning_curves(history, Z_val, y_val, outdir,
                                  case_label=label)
    for i, n in enumerate(curves["n_func"]):
        print(f"  N_f={n:2d} | func={curves['func'][i]:.4g} | "
              f"DEGP={curves['full'][i]:.4g} | GDDEGP={curves['dir'][i]:.4g}")

    plot_learning_curves_vs_cost(history, Z_val, y_val, outdir,
                                 case_label=label)

    print(f"\nPropagating {label} input distribution through each surrogate (N=5000):")
    result = propagate_and_summarize(history, n_mc=5000, seed=seed)
    print_stats_table(result["stats"])
    plot_output_distributions(result["y_truth"], result["preds"], outdir,
                              case_label=label)

    print("\nTracking metrics across all iterations...")
    mh = metric_history(history, n_mc=5000, seed=seed)
    print_metric_history(mh)
    plot_metric_evolution(mh, outdir, case_label=label)

    print("\nComputing global active subspace for comparison...")
    AS_eigvals, AS_eigvecs = compute_global_active_subspace(
        n_samples=2000, seed=seed + 17
    )
    print("  Global AS eigenvalues (descending):")
    print("    " + np.array2string(AS_eigvals, precision=4))
    plot_active_subspace_reference(AS_eigvals, AS_eigvecs, outdir,
                                   case_label=label)

    print("\nAnalyzing acquired directional derivatives vs active subspace...")
    plot_acquired_directions(directional_observations, AS_eigvals, AS_eigvecs,
                             outdir, case_label=label)

    print("\nTracking GP-derived active subspace across iterations...")
    as_hist = gp_active_subspace_history(
        history, AS_eigvals, AS_eigvecs,
        k_active=5, n_samples=1500, seed=seed + 23,
    )
    for label in as_hist:
        final_align = as_hist[label]["align_k"][-1]
        final_dom = as_hist[label]["dom_cos"][-1]
        print(f"  {label:<22s}  top-5 cos^2 align = {final_align:.4f}   "
              f"|w1_GP . w1_truth| = {final_dom:.4f}")
    plot_gp_active_subspace_history(as_hist, AS_eigvals, AS_eigvecs, outdir,
                                    case_label=label)

    print(f"\nFigures written to: {outdir.resolve()}")


if __name__ == "__main__":
    args = parse_args()
    configure_active_case(args.case)
    configure_derivative_variance_tol(args.deriv_var_tol)
    for time_seconds in args.times:
        configure_active_time(time_seconds)
        run_complete_study(n_iter=args.n_iter, seed=args.seed, verbose=True)
