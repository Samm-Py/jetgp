import numpy as np
from scipy.optimize import minimize

from doe_utils import lhs_design
from posterior_queries import query_function_posterior


_MPV_DEBUG_FIRED = False


def _diagnose_predict_failure(gp_model, params, x, label, exc):
    """Print extensive diagnostic info about a failed predict call."""
    global _MPV_DEBUG_FIRED
    if _MPV_DEBUG_FIRED:
        return
    _MPV_DEBUG_FIRED = True
    print("=" * 72)
    print(f"[DIAG] first predict failure in {label}")
    print(f"  Exception: {type(exc).__name__}: {exc}")
    print(f"  x shape = {np.shape(x)}, x = {np.asarray(x).ravel()}")
    print(f"  x finite = {np.all(np.isfinite(np.asarray(x)))}")
    p = np.asarray(params)
    print(f"  params shape = {p.shape}")
    print(f"  params = {p}")
    print(f"  params finite = {np.all(np.isfinite(p))}, "
          f"any nan = {np.any(np.isnan(p))}, any inf = {np.any(np.isinf(p))}")
    for attr in ("x_train", "y_train", "n_bases", "n_order",
                  "max_order", "normalize", "sigmas_x", "mus_x"):
        try:
            val = getattr(gp_model, attr, "<missing>")
            if isinstance(val, np.ndarray):
                print(f"  gp.{attr}.shape={val.shape}  "
                      f"nan={np.any(np.isnan(val))}  inf={np.any(np.isinf(val))}")
            else:
                print(f"  gp.{attr} = {val}")
        except Exception as e:
            print(f"  gp.{attr} -> error: {e}")
    try:
        x_safe = np.mean(gp_model.x_train, axis=0)
        if getattr(gp_model, "normalize", False):
            x_safe = x_safe * gp_model.sigmas_x + gp_model.mus_x
        _, v_safe = gp_model.predict(
            np.atleast_2d(x_safe), params, calc_cov=True, return_deriv=False)
        print(f"  predict at training-mean: var={float(v_safe[0]):.3e}, "
              f"finite={np.isfinite(float(v_safe[0]))}")
    except Exception as e:
        print(f"  predict at training-mean ALSO fails: "
              f"{type(e).__name__}: {e}")
    print("=" * 72)


def find_next_point_mpv(gp_model, params, bounds, n_restarts=12, seed=123):
    """
    Stage 2: argmax_x log(sigma^2_f(x)) over the domain.
    Returns (x_new, max_var).
    """
    lb, ub = bounds[:, 0], bounds[:, 1]
    starts = lhs_design(n_restarts, bounds, seed=seed)

    # Large finite penalty so scipy's finite-difference gradient stays
    # well-defined when consecutive evals hit the degenerate region
    # (returning np.inf here causes inf - inf = NaN in scipy's _numdiff).
    _BAD = 1e20

    def neg_log_var(x):
        if not np.all(np.isfinite(x)):
            return _BAD
        try:
            _, var = query_function_posterior(
                gp_model, params, np.atleast_2d(x))
        except (ValueError, np.linalg.LinAlgError) as exc:
            _diagnose_predict_failure(gp_model, params, x, "neg_log_var", exc)
            return _BAD
        v = float(var[0])
        if not np.isfinite(v) or v <= 0.0:
            return _BAD
        return -np.log(v)

    best_x, best_var = None, -np.inf
    for r_idx, x0 in enumerate(starts):
        try:
            res = minimize(
                neg_log_var, x0=x0, method="L-BFGS-B",
                bounds=list(zip(lb, ub)),
            )
            if not np.all(np.isfinite(res.x)):
                continue
            x_cand = np.clip(res.x, lb, ub)
            var_cand = float(query_function_posterior(
                gp_model, params, np.atleast_2d(x_cand))[1][0])
        except (ValueError, np.linalg.LinAlgError) as exc:
            _diagnose_predict_failure(
                gp_model, params, x0, f"MPV restart {r_idx}", exc)
            continue
        if np.isfinite(var_cand) and var_cand > best_var:
            best_x, best_var = x_cand, var_cand

    if best_x is None:
        raise ValueError(
            f"all {n_restarts} MPV restarts failed in find_next_point_mpv "
            f"(see diagnostics above)")
    return best_x, best_var


def find_next_point_weighted_mpv(gp_model, params, bounds, log_weight_fn,
                                 n_restarts=12, seed=123):
    """argmax_x [log sigma^2_f(x) + log_weight_fn(x)] over the domain.

    PDF-weighted variant of MPV: prioritises high-variance regions whose
    inputs also have non-negligible probability under log_weight_fn (the log
    input density). Returns (x_new, sigma^2_f at that point).
    """
    lb, ub = bounds[:, 0], bounds[:, 1]
    starts = lhs_design(n_restarts, bounds, seed=seed)
    _BAD = 1e20

    def neg_log_acq(x):
        if not np.all(np.isfinite(x)):
            return _BAD
        try:
            _, var = query_function_posterior(
                gp_model, params, np.atleast_2d(x))
        except (ValueError, np.linalg.LinAlgError) as exc:
            _diagnose_predict_failure(
                gp_model, params, x, "neg_log_acq (wMPV)", exc)
            return _BAD
        v = float(var[0])
        if not np.isfinite(v) or v <= 0.0:
            return _BAD
        log_w = float(log_weight_fn(x))
        if not np.isfinite(log_w):
            return _BAD
        return -(np.log(v) + log_w)

    best_x, best_var, best_acq = None, -np.inf, np.inf
    for r_idx, x0 in enumerate(starts):
        try:
            res = minimize(
                neg_log_acq, x0=x0, method="L-BFGS-B",
                bounds=list(zip(lb, ub)),
            )
            if not np.all(np.isfinite(res.x)):
                continue
            x_cand = np.clip(res.x, lb, ub)
            acq_cand = neg_log_acq(x_cand)
            var_cand = float(query_function_posterior(
                gp_model, params, np.atleast_2d(x_cand))[1][0])
        except (ValueError, np.linalg.LinAlgError) as exc:
            _diagnose_predict_failure(
                gp_model, params, x0, f"wMPV restart {r_idx}", exc)
            continue
        if np.isfinite(acq_cand) and acq_cand < best_acq:
            best_acq, best_var, best_x = acq_cand, var_cand, x_cand

    if best_x is None:
        raise ValueError(
            f"all {n_restarts} wMPV restarts failed in "
            f"find_next_point_weighted_mpv (see diagnostics above)")
    return best_x, best_var


def make_weighted_mpv(log_weight_fn):
    """Bind a log-weight function to produce an acquisition_func compatible
    with AdaptiveDirectionalGP (signature gp, params, bounds, seed)."""
    def _acq(gp_model, params, bounds, n_restarts=12, seed=123):
        return find_next_point_weighted_mpv(
            gp_model, params, bounds, log_weight_fn,
            n_restarts=n_restarts, seed=seed)
    _acq.__name__ = f"weighted_mpv[{getattr(log_weight_fn, '__name__', 'fn')}]"
    return _acq
