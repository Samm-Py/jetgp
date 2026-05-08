"""
Adaptive Sequential DOE
=======================

Implements the four-stage adaptive framework:
  Stage 2 — Infill location via maximum predictive variance (MPV).
  Stage 3 — Eigenbasis directional derivative selection at x_new via the
            global-prior gate: keep vⱼ where λⱼ_post / λ_max(K_prior) > rel_tol.
  Stage 4a — Evaluate first-order directional derivatives, update, refit.
  Stage 4b — (optional) For each Stage 4a direction v, check Var[d²f/dv² | data].
              If above lambda_abs_tol, evaluate the pure second directional derivative
              in that same direction and refit. Directions are shared with Stage 4a,
              not independently re-selected.

Usage
-----
    from adaptive_doe import AdaptiveDirectionalGP

    al = AdaptiveDirectionalGP(
        func=my_func, grad_func=my_grad,
        bounds=np.array([[lb0, ub0], [lb1, ub1]]),
        n_init=20, rel_tol=0.05, n_iter=10,
    )
    history = al.run()

    # With second-order derivatives:
    al = AdaptiveDirectionalGP(
        func=my_func, grad_func=my_grad, hess_func=my_hess,
        bounds=..., n_init=20, rel_tol=0.05, n_iter=10,
        acquire_second_order=True,
    )
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

from jetgp.full_gddegp.gddegp import gddegp


def _optimizer_kwargs_with_warm_start(optimizer_kwargs, initial_params, n_params):
    """Add a previous optimum as the first JADE candidate when dimensions match."""
    opt_kwargs = {
        "optimizer": "pso",
        "pop_size": 60,
        "n_generations": 20,
        "local_opt_every": 20,
        "debug": False,
    }
    if optimizer_kwargs is not None:
        opt_kwargs.update(optimizer_kwargs)
    if initial_params is None:
        return opt_kwargs

    initial_params = np.asarray(initial_params, dtype=float).reshape(-1)
    if initial_params.size == n_params and np.all(np.isfinite(initial_params)):
        opt_kwargs["initial_positions"] = np.atleast_2d(initial_params)
    return opt_kwargs


# ---------------------------------------------------------------------------
# LHS design
# ---------------------------------------------------------------------------

def lhs_design(n_points, bounds, seed=42):
    """Latin Hypercube Sample scaled to bounds. Returns (n_points, d)."""
    d = bounds.shape[0]
    sampler = LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n_points)
    lb, ub = bounds[:, 0], bounds[:, 1]
    return lb + unit_samples * (ub - lb)


# ---------------------------------------------------------------------------
# GP construction helpers
# ---------------------------------------------------------------------------

def fit_function_only_gp(X_train, y_train, n_dir_types=None,
                         kernel="SE", kernel_type="anisotropic",
                         optimizer_kwargs=None, normalize=True,
                         initial_params=None):
    """

    n_order=0 (pure GP). Reserve enough basis directions so that the
    coordinate derivative covariance can be queried at prediction time.
    """
    d = X_train.shape[1]
    if n_dir_types is None:
        n_dir_types = d
    gp_model = gddegp(
        X_train,
        [y_train],
        n_order=0,
        rays_list=[],
        der_indices=[],
        derivative_locations=[],
        n_bases=max(2 * d, 2 * max(1, n_dir_types)),
        normalize=normalize,
        kernel=kernel,
        kernel_type=kernel_type,
    )
    opt_kwargs = _optimizer_kwargs_with_warm_start(
        optimizer_kwargs, initial_params, len(gp_model.bounds))
    params = gp_model.optimize_hyperparameters(**opt_kwargs)
    return gp_model, params


def _construct_directional_gp(X_train, y_train, directional_observations,
                              second_order_observations=None,
                              kernel="SE", kernel_type="anisotropic"):
    """Construct a GDDEGP with first- and optional second-order observations.

    Paired observations in direction v at slot j use specs [[j+1, 1]] (df/dv)
    and [[j+1, 2]] (d²f/dv²) — the SAME basis index, different powers.  This
    means second-order slots add no new OTI bases, keeping n_bases = 2*D for
    any D-dimensional problem regardless of how many directions are selected.
    Training values equal actual derivatives (OTI derivative convention, no 1/p!).
    Requires n_order=2 when any second-order observations are present.
    """
    # --- First-order slots ---
    first_slots = {}
    for obs in directional_observations:
        first_slots.setdefault(obs["slot"], []).append(obs)

    n_first_slots = max(first_slots.keys()) + 1 if first_slots else 0
    y_blocks = [y_train]
    rays_list = []
    derivative_locations = []

    for s in range(n_first_slots):
        slot_obs = first_slots.get(s, [])
        if not slot_obs:
            continue
        values = np.array([[o["value"]] for o in slot_obs])
        rays = np.hstack([_make_ray(o["direction"], 1) for o in slot_obs])
        locs = [o["x_index"] for o in slot_obs]
        y_blocks.append(values)
        rays_list.append(rays)
        derivative_locations.append(locs)

    n_first_types = len(rays_list)

    # --- Second-order slots (pure second directional: d²f/dv²) ---
    n_second_types = 0
    if second_order_observations:
        second_slots = {}
        for obs in second_order_observations:
            second_slots.setdefault(obs["slot"], []).append(obs)

        n_second_slots = max(second_slots.keys()) + 1 if second_slots else 0
        for s in range(n_second_slots):
            slot_obs = second_slots.get(s, [])
            if not slot_obs:
                continue
            values = np.array([[o["value"]] for o in slot_obs])
            rays = np.hstack([_make_ray(o["direction"], 1) for o in slot_obs])
            locs = [o["x_index"] for o in slot_obs]
            y_blocks.append(values)
            rays_list.append(rays)
            derivative_locations.append(locs)
            n_second_types += 1

    # --- Derivative index specs ---
    # First-order: [[j+1, 1]]
    # Second-order: [[j+1, 2]] — same slot j as the paired first-order derivative.
    # e_{j+1}^1 and e_{j+1}^2 share the same basis index, so second-order
    # observations do not require additional OTI bases beyond n_first_types.
    first_der = [[[i + 1, 1]] for i in range(n_first_types)]
    second_der = [[[j + 1, 2]] for j in range(n_second_types)]

    total_types = n_first_types + n_second_types
    if total_types > 0:
        der_indices = [first_der + second_der]
    else:
        der_indices = []

    n_order = 2 if n_second_types > 0 else (1 if n_first_types > 0 else 0)
    # n_bases is driven by the number of unique basis indices = n_first_types,
    # since second-order slots reuse first-order slot indices.
    n_unique_slots = max(n_first_types, n_second_types)
    n_bases = max(2 * X_train.shape[1], 2 * max(1, n_unique_slots))

    return gddegp(
        X_train, y_blocks,
        n_order=n_order,
        rays_list=rays_list,
        der_indices=der_indices,
        derivative_locations=derivative_locations,
        n_bases=n_bases,
        normalize=True,
        kernel=kernel,
        kernel_type=kernel_type,
    )


def fit_directional_gp(X_train, y_train, directional_observations,
                       second_order_observations=None,
                       kernel="SE", kernel_type="anisotropic",
                       optimizer_kwargs=None, initial_params=None):
    """
    First-order observations are dicts:
        {"x_index": int, "direction": ndarray (d,), "value": float, "slot": int}

    Second-order observations (optional) are pure second directional derivatives:
        {"x_index": int, "direction": ndarray (d,), "value": float, "slot": int}
    where value = d²f/dv².

    Observations are grouped by slot: all observations with slot=0
    share OTI base pair 1, slot=1 shares pair 2, etc.
    """
    gp_model = _construct_directional_gp(
        X_train,
        y_train,
        directional_observations,
        second_order_observations=second_order_observations,
        kernel=kernel,
        kernel_type=kernel_type,
    )
    opt_kwargs = _optimizer_kwargs_with_warm_start(
        optimizer_kwargs, initial_params, len(gp_model.bounds))
    params = gp_model.optimize_hyperparameters(**opt_kwargs)
    return gp_model, params


# ---------------------------------------------------------------------------
# GP query helpers
# ---------------------------------------------------------------------------

def _make_ray(direction, n_points):
    v = np.asarray(direction, dtype=float).reshape(-1)
    v = v / np.linalg.norm(v)
    return np.tile(v.reshape(-1, 1), (1, n_points))


def query_function_posterior(gp_model, params, X_test):
    """Returns (mean, var) each shape (n_test,)."""
    mean, var = gp_model.predict(
        np.atleast_2d(X_test), params, calc_cov=True, return_deriv=False)
    return mean[0], var[0]


def query_directional_variance(gp_model, params, X_test, direction):
    """
    Posterior variance of the directional derivative along `direction` at X_test.
    Uses derivs_to_predict=[[[1,1]]] — direction-type 1.
    Returns var shape (n_test,).
    """
    ray = _make_ray(direction, X_test.shape[0])
    _, var = gp_model.predict(
        X_test, params,
        rays_predict=[ray],
        calc_cov=True,
        return_deriv=True,
        derivs_to_predict=[[[1, 1]]],
    )
    return var[1]


def query_directional_mean(gp_model, params, X_test, direction):
    """Posterior mean of directional derivative along `direction`."""
    ray = _make_ray(direction, X_test.shape[0])
    mean, _ = gp_model.predict(
        X_test, params,
        rays_predict=[ray],
        calc_cov=True,
        return_deriv=True,
        derivs_to_predict=[[[1, 1]]],
    )
    return mean[1]


# ---------------------------------------------------------------------------
# Acquisition function: maximum predictive variance (MPV)
# ---------------------------------------------------------------------------

def find_next_point_mpv(gp_model, params, bounds, n_restarts=12, seed=123):
    """
    Stage 2: argmax_x sigma^2_f(x) over the domain.
    Returns (x_new, max_var).
    """
    lb, ub = bounds[:, 0], bounds[:, 1]
    starts = lhs_design(n_restarts, bounds, seed=seed)

    best_x, best_var = None, -np.inf
    for x0 in starts:
        res = minimize(
            lambda x: -float(query_function_posterior(
                gp_model, params, np.atleast_2d(x))[1][0]),
            x0=x0, method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
        )
        x_cand = np.clip(res.x, lb, ub)
        var_cand = float(query_function_posterior(
            gp_model, params, np.atleast_2d(x_cand))[1][0])
        if var_cand > best_var:
            best_x, best_var = x_cand, var_cand

    return best_x, best_var


# ---------------------------------------------------------------------------
# Greedy directional derivative selection (Stage 3)
# ---------------------------------------------------------------------------

def _build_fantasy_gp(X_train, y_train, x_new, f_new,
                      directions, deriv_values,
                      kernel, kernel_type):
    """
    plus any provided directional derivative observations at x_new.
    Hyperparameters are NOT re-optimised (fantasy model).
    """
    X_aug = np.vstack([X_train, np.atleast_2d(x_new)])
    y_func_aug = np.vstack([y_train, np.array([[f_new]])])

    y_blocks = [y_func_aug]
    rays_list = []
    der_locs = []
    new_idx = X_aug.shape[0] - 1

    for val, v in zip(deriv_values, directions):
        y_blocks.append(np.array([[val]]))
        rays_list.append(_make_ray(v, 1))
        der_locs.append([new_idx])

    n_obs = len(directions)
    der_indices = [[[[i + 1, 1]] for i in range(n_obs)]] if n_obs > 0 else []
    n_bases = max(2 * X_aug.shape[1], 2 * max(1, n_obs))

    return gddegp(
        X_aug, y_blocks,
        n_order=1 if n_obs > 0 else 0,
        rays_list=rays_list,
        der_indices=der_indices,
        derivative_locations=der_locs,
        n_bases=n_bases,
        normalize=True,
        kernel=kernel,
        kernel_type=kernel_type,
    )


def coordinate_derivative_covariance(gp_model, params, x_new):
    """
    Full posterior covariance of coordinate directional derivatives at x_new.

    Returns Cov[grad f(x_new) | data] in the original output scale.
    """
    X_new = np.atleast_2d(x_new)
    d = X_new.shape[1]
    rays = []
    derivs = []
    for i in range(d):
        ray = np.zeros((d, 1))
        ray[i, 0] = 1.0
        rays.append(ray)
        derivs.append([[i + 1, 1]])

    _, _, full_cov = gp_model.predict(
        X_new,
        params,
        rays_predict=rays,
        calc_cov=True,
        return_deriv=True,
        derivs_to_predict=derivs,
        return_full_cov=True,
    )
    return full_cov[1:, 1:]


def coordinate_hessian_covariance(gp_model, params, x_new):
    """
    Posterior covariance of all unique second-order coordinate derivatives at x_new.

    Uses d(d+1)/2 unique pairs (i,j) with i<=j, spec [[i+1,1],[j+1,1]], and
    d coordinate-axis rays — fitting within the existing n_bases=2d budget.
    The GP upgrades to predict_order=2 on the fly if trained at n_order=1.

    Returns
    -------
    cov : ndarray, shape (n_pairs, n_pairs)
        Posterior covariance of the unique Hessian entries.
    pairs : list of (int, int)
        Index pairs (i, j) corresponding to rows/columns of cov.
    """
    X_new = np.atleast_2d(x_new)
    d = X_new.shape[1]

    rays = []
    for i in range(d):
        ray = np.zeros((d, 1))
        ray[i, 0] = 1.0
        rays.append(ray)

    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    derivs = [[[i + 1, 1], [j + 1, 1]] for (i, j) in pairs]

    _, _, full_cov = gp_model.predict(
        X_new, params,
        rays_predict=rays,
        calc_cov=True,
        return_deriv=True,
        derivs_to_predict=derivs,
        return_full_cov=True,
    )
    return full_cov[1:, 1:], pairs


def sorted_eigendecomposition(cov):
    """Eigenpairs sorted in descending eigenvalue order."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def gp_posterior_mean_gradient(gp_model, params, Z):
    """
    GP posterior mean gradient ∇μ(z) at each row of Z.

    Returns (n, D) ndarray. Works for any GDDEGP (function-only or with
    directional observations) by querying d_v f along the D coordinate axes.
    """
    Z = np.atleast_2d(Z)
    n, D = Z.shape
    eye = np.eye(D)
    rays_predict = [np.tile(eye[j:j + 1].T, (1, n)) for j in range(D)]
    derivs = [[[j + 1, 1]] for j in range(D)]
    mean = gp_model.predict(
        Z, params,
        rays_predict=rays_predict,
        calc_cov=False,
        return_deriv=True,
        derivs_to_predict=derivs,
    )
    if isinstance(mean, tuple):
        mean = mean[0]
    # mean[0] holds f-values; mean[1..D] hold the gradient components.
    return np.column_stack([mean[j + 1] for j in range(D)])


def gp_active_subspace_basis(gp_model, params, Z_AS, return_eigvals=False):
    """
    GP-derived active subspace eigenvectors at the current iteration.

    Computes ``C_AS = (1/M) Σ_m ∇μ(z_m) ∇μ(z_m)ᵀ`` over a fixed Monte-Carlo
    sample ``Z_AS`` from the input distribution, then eigendecomposes. Returns
    eigenvectors in descending eigenvalue order — these are the directions
    that capture most of the function's gradient activity *globally*, as
    opposed to the local Cov[∇f(x_new) | data] eigenvectors.

    Parameters
    ----------
    gp_model, params : the trained GP and its hyperparameters.
    Z_AS : ndarray, shape (M, D)
        Monte-Carlo sample from the input distribution; fixed across calls.
    return_eigvals : bool

    Returns
    -------
    eigvecs : ndarray, shape (D, D), columns are wⱼ in descending order
    eigvals : ndarray, shape (D,), only if return_eigvals=True
    """
    grads = gp_posterior_mean_gradient(gp_model, params, Z_AS)  # (M, D)
    M = grads.shape[0]
    C_AS = (grads.T @ grads) / M
    eigvals, eigvecs = np.linalg.eigh(C_AS)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if return_eigvals:
        return eigvecs, eigvals
    return eigvecs


def make_as_basis_provider(Z_AS, gamma_as=0.99):
    """
    Closure that returns ``(W, k_active)`` for a fixed Monte-Carlo sample.

    ``W`` is the GP-AS eigenvector matrix (D × D, columns sorted by descending
    eigenvalue). ``k_active`` is the smallest k such that
    ``Σⱼ≤k λⱼ_AS / Σⱼ λⱼ_AS ≥ gamma_as`` — the size of the active set, which
    selection logic *always* acquires regardless of the local rel_tol gate.
    Set ``gamma_as`` to None or ≥1.0 to disable the active-set floor (then
    selection falls back to the pure rel_tol gate).

    The cumulative-variance criterion (Jolliffe, 2002, §6.1.1) is applied to
    the *global* AS spectrum here — that's the right quantity to threshold,
    because AS eigenvalues directly measure how much gradient activity each
    direction carries. Local-posterior eigenvalues, by contrast, conflate
    kernel scale with activity.
    """
    Z_AS = np.asarray(Z_AS)

    def _provider(gp_model, params):
        W, eigvals = gp_active_subspace_basis(
            gp_model, params, Z_AS, return_eigvals=True)
        if gamma_as is None or gamma_as >= 1.0:
            return W, 0
        eig_pos = np.maximum(eigvals, 0.0)
        total = float(np.sum(eig_pos))
        if total <= 0.0:
            return W, 0
        cumsum = 0.0
        k_active = 0
        for i, lam in enumerate(eig_pos):
            cumsum += float(lam)
            k_active = i + 1
            if cumsum / total >= gamma_as:
                break
        return W, k_active

    return _provider


def prior_gradient_covariance(gp_model, params):
    """
    Prior gradient covariance ``K_∇∇(x, x)`` under the model's current kernel
    and hyperparameters, with no data conditioning.

    Stationary-kernel trick: for SE/Matérn/RQ etc. the prior gradient
    covariance is independent of x, so we evaluate the existing model's
    gradient covariance at a point placed far from all training data. The
    cross-covariance ``K(x_far, X_train)`` decays to zero, so the posterior
    reduces to the prior. Reuses the same predict pipeline as
    ``coordinate_derivative_covariance``, so input/output scaling is
    automatically consistent between prior and posterior.

    Returns
    -------
    K_prior : (D, D) ndarray
    """
    # Reconstruct training data in original (unnormalized) input space.
    if getattr(gp_model, "normalize", False):
        x_train_orig = (gp_model.x_train * gp_model.sigmas_x) + gp_model.mus_x
    else:
        x_train_orig = gp_model.x_train

    # Far point: many standard deviations beyond every training coordinate.
    span = np.maximum(
        np.std(x_train_orig, axis=0),
        np.maximum(np.abs(x_train_orig).max(axis=0), 1.0),
    )
    x_far = x_train_orig.mean(axis=0) + 1e3 * span

    return coordinate_derivative_covariance(
        gp_model, params, np.atleast_2d(x_far))


def _select_basis_and_project(cov, as_basis):
    """
    Choose the basis for direction selection.

    ``as_basis`` may be:
        None         — eigendecompose ``cov`` (legacy local-eigenvector mode);
                        sigmas returned in descending order. k_active = 0.
        (D, D) array — project onto its columns; preserve their order
                        (treat as already sorted by global activity).
                        k_active = 0 (no active-set floor).
        (W, k_active) tuple — same as above, with k_active > 0 indicating that
                        the first k_active columns of W are the active set
                        and must be force-selected by the caller.

    Returns
    -------
    sigmas    : ndarray, shape (D,) — diagonal values in the chosen basis
    basis     : ndarray, shape (D, D) — columns are the basis directions
    k_active  : int — size of the active-set floor (0 if disabled)
    """
    if as_basis is None:
        sigmas, basis = sorted_eigendecomposition(cov)
        return sigmas, basis, 0

    if isinstance(as_basis, tuple):
        W, k_active = as_basis
    else:
        W, k_active = as_basis, 0

    W = np.asarray(W)
    sigmas = np.array([float(W[:, j] @ cov @ W[:, j])
                       for j in range(W.shape[1])])
    return sigmas, W, int(k_active)


def _relative_variance_selection(eigenvalues, eigenvectors, K_prior, rel_tol,
                                  force_first=0):
    """
    Global-prior selection (Option A) with optional active-set floor.

    Pure gate:  select j  if  ρⱼ = λⱼ_post / λ_max(K_prior) > rel_tol.
    Active-set floor: also force-include indices 0 … force_first-1 (intended
    to be the leading AS eigenvectors when ``eigenvectors`` is the AS basis).

    Returns
    -------
    selected_indices : list of int (sorted ascending)
    rhos             : ndarray of shape (D,) — ρⱼ for every direction
    """
    n = len(eigenvalues)
    max_prior_eig = float(np.max(np.linalg.eigvalsh(K_prior)))
    if max_prior_eig <= 0.0:
        # No prior signal — only the active-set floor (if any) survives.
        rhos = np.full(n, np.nan)
        if force_first > 0:
            return list(range(min(force_first, n))), rhos
        return [], rhos
    rhos = np.maximum(eigenvalues, 0.0) / max_prior_eig
    selected = [j for j in range(n)
                if (j < force_first) or rhos[j] > rel_tol]
    return selected, rhos


def _cumulative_variance_selection(eigenvalues, gamma, lambda_abs_tol):
    """
    Cumulative-variance truncation: keep the smallest k eigenvectors such that
    Σⱼ≤k λⱼ / Σⱼ λⱼ ≥ gamma. Standard PCA dimension-selection rule (Jolliffe,
    2002, *Principal Component Analysis*, 2nd ed., §6.1.1) applied here to the
    posterior gradient covariance.

    The lambda_abs_tol threshold acts as a floor on the *total* trace: if the
    full posterior gradient covariance is essentially zero, no directions are
    selected.

    Returns the selected indices (sorted, descending eigenvalue order).
    """
    eig_pos = np.maximum(eigenvalues, 0.0)
    total = float(np.sum(eig_pos))
    if total <= lambda_abs_tol:
        return []
    cumsum = 0.0
    selected = []
    for i, lam in enumerate(eig_pos):
        cumsum += float(lam)
        selected.append(i)
        if cumsum / total >= gamma:
            break
    return selected


def select_derivatives_at_xnew(gp_model, params, X_train, y_train,
                                x_new, rel_tol=0.05, as_basis=None):
    """
    Stage 3: eigenbasis directional derivative selection at x_new.

    Two basis options:
      * ``as_basis=None`` (legacy): eigendecompose Cov[∇f(x_new) | data]
        locally and use those eigenvectors. Fragile — local eigenvectors
        often don't track the function's global active subspace.
      * ``as_basis=W`` (preferred): use the columns of W as the selection
        basis (e.g. the GP's current active-subspace estimate). The
        directions are then globally informed.

    For each candidate direction wⱼ, compute σⱼ = wⱼᵀ Cov[∇f(x_new)|data] wⱼ
    and gate at ρⱼ = σⱼ / λ_max(K_prior) > ``rel_tol``.

    Returns
    -------
    dict with keys:
        selected_directions   : list of unit ndarray (d,)
        selected_variances    : list of float (eigenvalues at selection)
        fantasy_f_value       : float
        eigenvalues           : ndarray, descending
        eigenvectors          : ndarray, columns matched to eigenvalues
        variance_ratios       : ndarray, eigenvalue / leading eigenvalue
        rho                   : ndarray, ρⱼ per eigenvector
        K_prior               : ndarray (d, d), prior gradient covariance
    """
    x_new = np.asarray(x_new).reshape(-1)
    X_new = np.atleast_2d(x_new)

    # Fantasy function value at x_new (posterior mean — does not affect variance)
    f_new = float(query_function_posterior(gp_model, params, X_new)[0][0])

    # Fantasy GP with no derivatives yet
    gp_fantasy0 = _build_fantasy_gp(
        X_train, y_train, x_new, f_new,
        directions=[], deriv_values=[],
        kernel=gp_model.kernel, kernel_type=gp_model.kernel_type,
    )

    cov = coordinate_derivative_covariance(gp_fantasy0, params, X_new)
    K_prior = prior_gradient_covariance(gp_fantasy0, params)

    eigenvalues, eigenvectors, k_active = _select_basis_and_project(
        cov, as_basis)
    leading = eigenvalues[0]
    variance_ratios = (
        eigenvalues / leading if leading > 0.0 else np.full_like(eigenvalues, np.nan)
    )

    selected_indices, rho = _relative_variance_selection(
        eigenvalues, eigenvectors, K_prior, rel_tol, force_first=k_active)

    selected_directions = [eigenvectors[:, i] for i in selected_indices]
    selected_variances = [float(eigenvalues[i]) for i in selected_indices]

    return {
        "selected_directions": selected_directions,
        "selected_variances": selected_variances,
        "fantasy_f_value": f_new,
        "derivative_covariance": cov,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "variance_ratios": variance_ratios,
        "rho": rho,
        "K_prior": K_prior,
        "selected_indices": selected_indices,
        "rel_tol": float(rel_tol),
        "k_active": int(k_active),
        "basis_source": "AS" if as_basis is not None else "local",
        "Ad0": float(leading),
        "v1": eigenvectors[:, 0],
        "var1": float(eigenvalues[0]),
    }


def select_derivatives_at_observed_point(
        gp_model, params, x_point, rel_tol=0.05, as_basis=None):
    """
    Select directional derivatives at an existing function-observation site.

    See ``select_derivatives_at_xnew`` for the basis options. When
    ``as_basis`` is the GP-derived active subspace eigenvectors, the gated
    quantity σⱼ = wⱼᵀ Cov[∇f(x)|data] wⱼ measures posterior gradient
    uncertainty along *globally-informative* directions rather than the
    eigenvectors of the local covariance (which can rotate to arbitrary
    orientations even when a true active subspace exists).
    """
    x_point = np.asarray(x_point).reshape(-1)
    X_point = np.atleast_2d(x_point)

    cov = coordinate_derivative_covariance(gp_model, params, X_point)
    K_prior = prior_gradient_covariance(gp_model, params)

    eigenvalues, eigenvectors, k_active = _select_basis_and_project(
        cov, as_basis)
    leading = eigenvalues[0]
    variance_ratios = (
        eigenvalues / leading if leading > 0.0 else np.full_like(eigenvalues, np.nan)
    )

    selected_indices, rho = _relative_variance_selection(
        eigenvalues, eigenvectors, K_prior, rel_tol, force_first=k_active)

    selected_directions = [eigenvectors[:, i] for i in selected_indices]
    selected_variances = [float(eigenvalues[i]) for i in selected_indices]

    return {
        "selected_directions": selected_directions,
        "selected_variances": selected_variances,
        "derivative_covariance": cov,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "variance_ratios": variance_ratios,
        "rho": rho,
        "K_prior": K_prior,
        "selected_indices": selected_indices,
        "rel_tol": float(rel_tol),
        "k_active": int(k_active),
        "basis_source": "AS" if as_basis is not None else "local",
        "Ad0": float(leading),
        "v1": eigenvectors[:, 0],
        "var1": float(eigenvalues[0]),
    }


def second_order_variance_in_direction(gp_model, params, x_point, direction):
    """
    Posterior variance of d²f/dv² at x_point along unit direction v.

    Uses derivs_to_predict=[[[1, 2]]] with a single ray v.  Works even when
    gp_model was trained with n_order=1 (predict auto-upgrades to order 2).

    Returns
    -------
    float : Var[d²f/dv² | data] at x_point.
    """
    v = np.asarray(direction, dtype=float).reshape(-1)
    v = v / np.linalg.norm(v)
    ray = _make_ray(v, 1)
    _, var = gp_model.predict(
        np.atleast_2d(x_point), params,
        rays_predict=[ray],
        calc_cov=True,
        return_deriv=True,
        derivs_to_predict=[[[1, 2]]],
    )
    return float(var[1][0])


def sequential_initial_derivative_enrichment(
        gp_model, params, X_train, y_train, grad_func,
        rel_tol=0.05, kernel="SE", kernel_type="anisotropic",
        max_selected_directions=None, optimizer_kwargs=None,
        lambda_abs_tol=0.0,
        hess_func=None, acquire_second_order=False,
        as_basis_provider=None):
    """
    Process the initial DOE sequentially with real derivative updates.

    At each existing DOE point x^(i):
      - Stage 4a: select first-order directions, evaluate, refit.
      - Stage 4b (optional): check Hessian covariance under updated posterior;
        if informative, evaluate pure second directional derivatives, refit.
    """
    working_model = gp_model
    working_params = np.asarray(params).copy()
    gradients = grad_func(X_train)

    actual_observations = []
    second_order_observations = []
    selection_records = []

    for x_index, x_point in enumerate(X_train):
        # --- Stage 4a: first-order ---
        as_basis = (as_basis_provider(working_model, working_params)
                    if as_basis_provider is not None else None)
        selection = select_derivatives_at_observed_point(
            working_model, working_params, x_point, rel_tol=rel_tol,
            as_basis=as_basis,
        )

        if max_selected_directions is not None:
            keep = min(len(selection["selected_directions"]), max_selected_directions)
            selection["selected_directions"] = selection["selected_directions"][:keep]
            selection["selected_variances"] = selection["selected_variances"][:keep]
            selection["selected_indices"] = selection["selected_indices"][:keep]

        true_derivs = []
        for slot, direction in enumerate(selection["selected_directions"]):
            true_val = float(gradients[x_index] @ direction)
            true_derivs.append(true_val)
            actual_observations.append({
                "x_index": x_index,
                "direction": direction.copy(),
                "value": true_val,
                "slot": slot,
            })

        if selection["selected_directions"]:
            working_model, working_params = fit_directional_gp(
                X_train, y_train, actual_observations,
                second_order_observations=second_order_observations or None,
                kernel=kernel, kernel_type=kernel_type,
                optimizer_kwargs=optimizer_kwargs,
                initial_params=working_params,
            )

        # --- Stage 4b: second-order along the same Stage 4a directions ---
        # For each direction v selected in Stage 4a, check Var[d²f/dv² | data].
        # If above lambda_abs_tol, evaluate and record the pure second derivative.
        second_order_dirs = []
        second_order_vars = []
        true_second_derivs = []
        if acquire_second_order and hess_func is not None and selection["selected_directions"]:
            H = None
            for slot, v in enumerate(selection["selected_directions"]):
                var_d2 = second_order_variance_in_direction(
                    working_model, working_params, x_point, v)
                if var_d2 > lambda_abs_tol:
                    if H is None:
                        H = hess_func(np.atleast_2d(x_point))
                        if H.ndim == 3:
                            H = H[0]
                    true_val2 = float(v @ H @ v)
                    true_second_derivs.append(true_val2)
                    second_order_dirs.append(v.copy())
                    second_order_vars.append(var_d2)
                    second_order_observations.append({
                        "x_index": x_index,
                        "direction": v.copy(),
                        "value": true_val2,
                        "slot": slot,
                    })
            if second_order_dirs:
                working_model, working_params = fit_directional_gp(
                    X_train, y_train, actual_observations,
                    second_order_observations=second_order_observations,
                    kernel=kernel, kernel_type=kernel_type,
                    optimizer_kwargs=optimizer_kwargs,
                    initial_params=working_params,
                )

        record = {
            "x_index": x_index,
            "x_point": np.asarray(x_point).copy(),
            "selected_directions": [v.copy() for v in selection["selected_directions"]],
            "selected_variances": list(selection["selected_variances"]),
            "true_derivs": list(true_derivs),
            "derivative_covariance": selection["derivative_covariance"].copy(),
            "eigenvalues": selection["eigenvalues"].copy(),
            "eigenvectors": selection["eigenvectors"].copy(),
            "variance_ratios": selection["variance_ratios"].copy(),
            "rho": selection["rho"].copy(),
            "K_prior": selection["K_prior"].copy(),
            "selected_indices": list(selection["selected_indices"]),
            "k_active": selection.get("k_active", 0),
            "second_order_selected_directions": second_order_dirs,
            "second_order_selected_variances": second_order_vars,
            "second_order_true_derivs": list(true_second_derivs),
        }
        selection_records.append(record)

    return actual_observations, second_order_observations, selection_records


# ---------------------------------------------------------------------------
# AdaptiveDirectionalGP — main class
# ---------------------------------------------------------------------------

class AdaptiveDirectionalGP:
    """

    Implements Stages 1-4 of the white paper for directional derivatives,
    with an optional Stage 4b for pure second-order directional derivatives.

    Parameters
    ----------
    func : callable
        f(X: ndarray (n, d)) -> ndarray (n, 1).
    grad_func : callable
        grad_f(X: ndarray (n, d)) -> ndarray (n, d).
    bounds : ndarray, shape (d, 2)
    n_init : int
    rel_tol : float
        Global-prior threshold for eigenvector selection: keep direction vⱼ
        if ρⱼ = λⱼ_post / λ_max(K_prior) exceeds rel_tol. ρⱼ ∈ [0, 1] reads as
        "absolute posterior gradient variance in direction vⱼ as a fraction
        of the kernel's most-uncertain prior direction." Default 0.05.
    n_iter : int
    kernel : str
    kernel_type : str
    seed : int
    lambda_abs_tol : float
        Reserved for second-order Stage 4b variance gating. First-order
        selection no longer uses an absolute floor (handled by rel_tol via
        prior-relative ρⱼ).
    hess_func : callable or None
        hess_f(X: ndarray (1, d)) -> ndarray (d, d).
        Required when acquire_second_order=True.
    acquire_second_order : bool
        If True, run Stage 4b after each Stage 4a refit.
    """

    def __init__(self, func, grad_func, bounds, n_init, rel_tol, n_iter,
                 kernel="SE", kernel_type="anisotropic",
                 seed=42, lambda_abs_tol=0.0,
                 hess_func=None, acquire_second_order=False,
                 acquisition_func=None, optimizer_kwargs=None,
                 X_init=None, as_basis_provider=None):
        self.func = func
        self.grad_func = grad_func
        self.hess_func = hess_func
        self.acquire_second_order = acquire_second_order
        self.bounds = np.asarray(bounds)
        self.n_init = n_init
        self.rel_tol = rel_tol
        self.n_iter = n_iter
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.seed = seed
        self.lambda_abs_tol = lambda_abs_tol
        # acquisition_func(gp_model, params, bounds, seed=s) -> (x_new, val)
        self.acquisition_func = acquisition_func or find_next_point_mpv
        self.optimizer_kwargs = optimizer_kwargs
        # Callable (gp_model, params) -> (D, D) basis matrix for selection.
        # If None, falls back to local-eigenvector mode.
        self.as_basis_provider = as_basis_provider
        # Optional pre-built initial design (n_init, d); overrides LHS if given
        self.X_init = np.asarray(X_init) if X_init is not None else None

        # State — populated by run()
        self.X_train = None
        self.y_train = None
        self.directional_observations = []
        self.second_order_observations = []
        self.gp_model = None
        self.params = None
        self.history = []
        self.initial_function_gp_model = None
        self.initial_function_params = None
        self.post_enrichment_gp_model = None
        self.post_enrichment_params = None
        self.initial_derivative_history = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval_directional(self, x, v):
        """True directional derivative of func at x along unit vector v."""
        grad = self.grad_func(np.atleast_2d(x))
        v = np.asarray(v, dtype=float).reshape(-1)
        return float(grad[0] @ v)

    def _eval_second_order_directional(self, x, v):
        """True pure second directional derivative d²f/dv² at x."""
        H = self.hess_func(np.atleast_2d(x))   # (d, d) or (1, d, d)
        v = np.asarray(v, dtype=float).reshape(-1)
        if H.ndim == 3:
            H = H[0]
        return float(v @ H @ v)

    def _refit(self):
        """Refit the GP from the current training set."""
        previous_params = None if self.params is None else self.params.copy()
        has_second = bool(self.second_order_observations)
        if len(self.directional_observations) == 0 and not has_second:
            self.gp_model, self.params = fit_function_only_gp(
                self.X_train, self.y_train,
                n_dir_types=self.bounds.shape[0],
                kernel=self.kernel, kernel_type=self.kernel_type,
                optimizer_kwargs=self.optimizer_kwargs,
                initial_params=previous_params,
            )
        else:
            self.gp_model, self.params = fit_directional_gp(
                self.X_train, self.y_train,
                self.directional_observations,
                second_order_observations=self.second_order_observations or None,
                kernel=self.kernel, kernel_type=self.kernel_type,
                optimizer_kwargs=self.optimizer_kwargs,
                initial_params=previous_params,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enable_initial_derivative_enrichment(self, enabled=True):
        """Toggle derivative acquisition on the initial DOE sites."""
        self.enrich_initial_doe = bool(enabled)

    def _initial_derivative_enrichment(self):
        """
        Acquire directional derivatives at the existing initial DOE sites.

        Points are processed sequentially. At each DOE site:
          Stage 4a: first-order directions selected, evaluated, model refit.
          Stage 4b: (if acquire_second_order) Hessian covariance checked under
                    updated posterior; second-order derivatives acquired if
                    informative, model refit again before next DOE point.
        """
        print("\n" + "=" * 60)
        print("Stage 1.5 - Initial directional enrichment")
        if self.acquire_second_order:
            print("         (first- and second-order)")
        print("=" * 60)

        (self.directional_observations,
         self.second_order_observations,
         selection_records) = sequential_initial_derivative_enrichment(
            self.gp_model,
            self.params,
            self.X_train,
            self.y_train,
            self.grad_func,
            rel_tol=self.rel_tol,
            kernel=self.kernel,
            kernel_type=self.kernel_type,
            lambda_abs_tol=self.lambda_abs_tol,
            hess_func=self.hess_func,
            acquire_second_order=self.acquire_second_order,
            optimizer_kwargs=self.optimizer_kwargs,
            as_basis_provider=self.as_basis_provider,
        )

        total_first = len(self.directional_observations)
        total_second = len(self.second_order_observations)
        for record in selection_records:
            rho = np.round(record["rho"], 4)
            n_sel1 = len(record["selected_directions"])
            n_sel2 = len(record["second_order_selected_directions"])
            k_active = record.get("k_active", 0)
            print(f"  x_{record['x_index']:03d} = {np.round(record['x_point'], 4)}")
            print(f"    1st: rho = {rho}  "
                  f"(rel_tol={self.rel_tol}, k_active={k_active})  "
                  f"->  {n_sel1} dir(s)")
            for k, v in enumerate(record["selected_directions"]):
                print(f"          v{k+1} = {np.round(v, 4)}")
            if n_sel2 > 0:
                vars2 = np.round(record["second_order_selected_variances"], 5)
                print(f"    2nd: Var[d²f/dv²] = {vars2}  "
                      f"(tol={self.lambda_abs_tol:g})  ->  {n_sel2} dir(s)")

        print(f"  Added {total_first} first-order, {total_second} second-order "
              f"observations across {self.X_train.shape[0]} initial DOE point(s)")

        if total_first > 0 or total_second > 0:
            self._refit()
            print(f"  params = {self.params}")

        self.post_enrichment_gp_model = self.gp_model
        self.post_enrichment_params = self.params.copy()
        self.initial_derivative_history = selection_records

    def run(self):
        """
        Execute the full adaptive loop.

        Stage 1   — Initial LHS DOE.
        Stage 1.5 — (optional) Enrich initial DOE with directional derivatives.
        Per iteration:
          Stage 2  — Select x_new via MPV.
          Stage 3  — Evaluate f(x_new), add, refit.
          Stage 4a — Select/evaluate first-order directional derivatives, refit.
          Stage 4b — (optional) Select/evaluate second-order derivatives, refit.

        Returns
        -------
        history : list of dict, one entry per iteration.
        """
        # --- Stage 1: Initial DOE ---
        print("=" * 60)
        print("Stage 1 — Initial DOE")
        print("=" * 60)
        if self.X_init is not None:
            self.X_train = self.X_init.copy()
        else:
            self.X_train = lhs_design(self.n_init, self.bounds, seed=self.seed)
        self.y_train = self.func(self.X_train)
        self.directional_observations = []
        self.second_order_observations = []
        print(f"  {self.n_init} initial points, f in "
              f"[{self.y_train.min():.3f}, {self.y_train.max():.3f}]")

        self._refit()
        self.initial_function_gp_model = self.gp_model
        self.initial_function_params = self.params.copy()
        print(f"  params = {self.params}")

        if getattr(self, "enrich_initial_doe", False):
            self._initial_derivative_enrichment()
        else:
            self.post_enrichment_gp_model = self.gp_model
            self.post_enrichment_params = self.params.copy()

        # --- Stages 2-4: active learning loop ---
        for step in range(1, self.n_iter + 1):
            print(f"\n{'─'*60}")
            print(f"Iteration {step}/{self.n_iter}")
            print(f"{'─'*60}")

            # Stage 2 — infill location
            x_new, mpv = self.acquisition_func(
                self.gp_model, self.params, self.bounds,
                seed=self.seed + step,
            )
            print(f"  Stage 2: x_new = {np.round(x_new, 4)},  "
                  f"sigma^2_f = {mpv:.5f}")

            pre_update_gp_model = self.gp_model
            pre_update_params = self.params.copy()
            pre_update_X_train = self.X_train.copy()
            pre_update_y_train = self.y_train.copy()
            pre_update_directional_observations = [
                {"x_index": o["x_index"], "direction": o["direction"].copy(),
                 "value": o["value"], "slot": o["slot"]}
                for o in self.directional_observations
            ]

            # Stage 3 — observe f(x_new), add, refit.
            f_new = float(self.func(np.atleast_2d(x_new))[0, 0])
            new_index = self.X_train.shape[0]
            self.X_train = np.vstack([self.X_train, np.atleast_2d(x_new)])
            self.y_train = np.vstack([self.y_train, np.array([[f_new]])])
            self._refit()
            print(f"  Stage 3: f(x_new) = {f_new:.4f}; refit before derivative selection")

            # Stage 4a — first-order directional derivatives.
            as_basis = (self.as_basis_provider(self.gp_model, self.params)
                        if self.as_basis_provider is not None else None)
            selection = select_derivatives_at_observed_point(
                self.gp_model, self.params, x_new, rel_tol=self.rel_tol,
                as_basis=as_basis,
            )
            n_sel1 = len(selection["selected_directions"])
            rho = np.round(selection["rho"], 4)
            print(f"  Stage 4a [{selection['basis_source']} basis]:  "
                  f"eigenvalues = {np.round(selection['eigenvalues'], 5)}")
            print(f"            rho = {rho}  "
                  f"(rel_tol={self.rel_tol}, k_active={selection['k_active']})"
                  f"  ->  {n_sel1} dir(s)")
            for k, (v, var) in enumerate(
                    zip(selection["selected_directions"],
                        selection["selected_variances"]), start=1):
                print(f"            v{k} = {np.round(v, 4)},  "
                      f"Var[d_v{k} f] = {var:.5f}")

            true_derivs = []
            for idx, v in enumerate(selection["selected_directions"]):
                d_val = self._eval_directional(x_new, v)
                true_derivs.append(d_val)
                self.directional_observations.append({
                    "x_index": new_index, "direction": v,
                    "value": d_val, "slot": idx,
                })
            print(f"            true derivs = {[round(d, 4) for d in true_derivs]}")

            self._refit()
            print(f"  params after 4a = {self.params}")

            # Stage 4b — d²f/dvⱼ² along each Stage 4a direction vⱼ.
            # Gate: Var[d²f/dvⱼ² | data] > lambda_abs_tol.
            second_order_dirs = []
            second_order_vars = []
            true_second_derivs = []
            if self.acquire_second_order and self.hess_func is not None and n_sel1 > 0:
                print(f"  Stage 4b: checking d²f/dv² variance in {n_sel1} Stage 4a direction(s)")
                for idx, v in enumerate(selection["selected_directions"]):
                    var_d2 = second_order_variance_in_direction(
                        self.gp_model, self.params, x_new, v)
                    print(f"            v{idx+1}: Var[d²f/dv²] = {var_d2:.5f}  "
                          f"(tol={self.lambda_abs_tol:g})  "
                          f"{'-> acquire' if var_d2 > self.lambda_abs_tol else '-> skip'}")
                    if var_d2 > self.lambda_abs_tol:
                        d2_val = self._eval_second_order_directional(x_new, v)
                        true_second_derivs.append(d2_val)
                        second_order_dirs.append(v)
                        second_order_vars.append(var_d2)
                        self.second_order_observations.append({
                            "x_index": new_index, "direction": v,
                            "value": d2_val, "slot": idx,
                        })
                n_sel2 = len(second_order_dirs)
                print(f"            true 2nd derivs = "
                      f"{[round(d, 4) for d in true_second_derivs]}"
                      f"  ({n_sel2} acquired)")
                if n_sel2 > 0:
                    self._refit()
                    print(f"  params after 4b = {self.params}")
            else:
                n_sel2 = 0

            record = {
                "step": step,
                "x_new": x_new.copy(),
                "f_new": f_new,
                "mpv": mpv,
                "pre_update_gp_model": pre_update_gp_model,
                "pre_update_params": pre_update_params,
                "pre_update_X_train": pre_update_X_train,
                "pre_update_y_train": pre_update_y_train,
                "pre_update_directional_observations": pre_update_directional_observations,
                "post_update_gp_model": self.gp_model,
                "post_update_params": self.params.copy(),
                "post_update_X_train": self.X_train.copy(),
                "post_update_y_train": self.y_train.copy(),
                # Stage 4a
                "selected_directions": selection["selected_directions"],
                "selected_variances": selection["selected_variances"],
                "true_derivs": true_derivs,
                "n_selected": n_sel1,
                "derivative_covariance": selection["derivative_covariance"].copy(),
                "eigenvalues": selection["eigenvalues"].copy(),
                "eigenvectors": selection["eigenvectors"].copy(),
                "variance_ratios": selection["variance_ratios"].copy(),
                "selected_indices": list(selection["selected_indices"]),
                "v1": selection["v1"].copy(),
                "var1": selection["var1"],
                # Stage 4b
                "second_order_selected_directions": second_order_dirs,
                "second_order_selected_variances": second_order_vars,
                "second_order_true_derivs": true_second_derivs,
                "n_second_order_selected": n_sel2,
                # Counts
                "n_train": self.X_train.shape[0],
                "n_directional_obs": len(self.directional_observations),
                "n_second_order_obs": len(self.second_order_observations),
                "params": self.params.copy(),
            }
            self.history.append(record)

        print(f"\n{'='*60}")
        print("Active learning loop complete.")
        print(f"  Final training set:  {self.X_train.shape[0]} points")
        print(f"  First-order obs:     {len(self.directional_observations)}")
        print(f"  Second-order obs:    {len(self.second_order_observations)}")
        print("=" * 60)
        return self.history


