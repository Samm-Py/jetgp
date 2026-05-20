import numpy as np

from gp_builders import _build_fantasy_gp, fit_directional_gp
from lanczos_selection import (
    prior_lambda_max_gradient as prior_lambda_max_gradient_iterative,
    prior_lambda_max_hessian as prior_lambda_max_hessian_iterative,
    second_order_variance_in_direction,
    select_eigenpairs as select_eigenpairs_iterative,
)
from posterior_queries import query_function_posterior


def coordinate_derivative_covariance(gp_model, params, x_new):
    """Full posterior covariance of coordinate directional derivatives at x_new."""
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
    """Posterior covariance of unique second-order coordinate derivatives."""
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
    """GP posterior mean gradient at each row of Z."""
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
    return np.column_stack([mean[j + 1] for j in range(D)])


def gp_active_subspace_basis(gp_model, params, Z_AS, return_eigvals=False):
    """GP-derived active subspace eigenvectors at the current iteration."""
    grads = gp_posterior_mean_gradient(gp_model, params, Z_AS)
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
    """Closure that returns ``(W, k_active)`` for a fixed Monte-Carlo sample."""
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
    """Prior gradient covariance using a far-point posterior probe."""
    if getattr(gp_model, "normalize", False):
        x_train_orig = (gp_model.x_train * gp_model.sigmas_x) + gp_model.mus_x
    else:
        x_train_orig = gp_model.x_train

    span = np.maximum(
        np.std(x_train_orig, axis=0),
        np.maximum(np.abs(x_train_orig).max(axis=0), 1.0),
    )
    x_far = x_train_orig.mean(axis=0) + 1e3 * span
    return coordinate_derivative_covariance(
        gp_model, params, np.atleast_2d(x_far))


def prior_hessian_covariance(gp_model, params):
    """Prior Hessian covariance using a far-point posterior probe."""
    if getattr(gp_model, "normalize", False):
        x_train_orig = (gp_model.x_train * gp_model.sigmas_x) + gp_model.mus_x
    else:
        x_train_orig = gp_model.x_train

    span = np.maximum(
        np.std(x_train_orig, axis=0),
        np.maximum(np.abs(x_train_orig).max(axis=0), 1.0),
    )
    x_far = x_train_orig.mean(axis=0) + 1e3 * span
    cov, _ = coordinate_hessian_covariance(
        gp_model, params, np.atleast_2d(x_far))
    return cov


def _build_order_decisions(gp_model, params, x_point, selected_directions,
                            selected_rhos, rel_tol, c1, c2,
                            acquire_second_order, max_directions=None):
    """Build the unified (direction, order) acquisition list."""
    d = np.asarray(x_point).reshape(-1).size
    cap = max_directions if max_directions is not None else d
    cap = min(cap, d)

    decisions = []
    for v, rho1 in zip(selected_directions, selected_rhos):
        if len(decisions) >= cap:
            break
        decisions.append({
            "direction": np.asarray(v).reshape(-1).copy(),
            "order": 1,
            "rho1": float(rho1),
            "rho2": None,
        })

    if not (acquire_second_order and c2 is not None and c2 > 0):
        return decisions

    lam2 = prior_lambda_max_hessian_iterative(gp_model, params)
    if lam2 <= 0.0:
        return decisions

    for dec in decisions:
        var2 = second_order_variance_in_direction(
            gp_model, params, x_point, dec["direction"])
        rho2 = max(var2, 0.0) / (c2 * lam2)
        dec["rho2"] = float(rho2)
        if rho2 > rel_tol:
            dec["order"] = 2

    return decisions


def select_derivatives_at_xnew(gp_model, params, X_train, y_train,
                                x_new, rel_tol=0.05, as_basis=None,
                                c1=1.0, c2=None,
                                acquire_second_order=False,
                                max_directions=None, lanczos_m=None, seed=0):
    """Stage 3 derivative selection at x_new using Lanczos eigenpairs."""
    x_new = np.asarray(x_new).reshape(-1)
    X_new = np.atleast_2d(x_new)
    d = x_new.size

    if as_basis is not None:
        raise ValueError("as_basis projection is not supported with Lanczos selection")

    f_new = float(query_function_posterior(gp_model, params, X_new)[0][0])
    gp_fantasy0 = _build_fantasy_gp(
        X_train, y_train, x_new, f_new,
        directions=[], deriv_values=[],
        kernel=gp_model.kernel, kernel_type=gp_model.kernel_type,
    )

    k = min(max_directions if max_directions is not None else d, d)
    eigenvalues, eigenvectors = select_eigenpairs_iterative(
        gp_fantasy0, params, x_new, k=k, m=lanczos_m, seed=seed)
    lam_prior = prior_lambda_max_gradient_iterative(
        gp_fantasy0, params, seed=seed)
    leading = eigenvalues[0] if eigenvalues.size > 0 else 0.0
    variance_ratios = (
        eigenvalues / leading if leading > 0.0
        else np.full_like(eigenvalues, np.nan))
    rho = (np.maximum(eigenvalues, 0.0) / lam_prior
           if lam_prior > 0.0 else np.full_like(eigenvalues, np.nan))
    selected_indices = [j for j in range(len(eigenvalues))
                         if rho[j] > rel_tol]

    selected_directions = [eigenvectors[:, i] for i in selected_indices]
    selected_variances = [float(eigenvalues[i]) for i in selected_indices]
    selected_rhos = [float(rho[i]) for i in selected_indices]

    order_decisions = _build_order_decisions(
        gp_model, params, x_new, selected_directions, selected_rhos,
        rel_tol=rel_tol, c1=c1, c2=c2,
        acquire_second_order=acquire_second_order,
        max_directions=max_directions,
    )

    return {
        "selected_directions": selected_directions,
        "selected_variances": selected_variances,
        "order_decisions": order_decisions,
        "fantasy_f_value": f_new,
        "derivative_covariance": None,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "variance_ratios": variance_ratios,
        "rho": rho,
        "K_prior": None,
        "selected_indices": selected_indices,
        "rel_tol": float(rel_tol),
        "k_active": 0,
        "basis_source": "lanczos",
        "Ad0": float(leading),
        "v1": eigenvectors[:, 0],
        "var1": float(eigenvalues[0]),
    }


def select_derivatives_at_observed_point(
        gp_model, params, x_point, rel_tol=0.05, as_basis=None,
        c1=1.0, c2=None, acquire_second_order=False,
        max_directions=None, iterative=True,
        lanczos_m=None, seed=0):
    """Select directional derivatives at an existing function-observation site."""
    x_point = np.asarray(x_point).reshape(-1)
    d = x_point.size

    if as_basis is not None:
        raise ValueError("as_basis projection is not supported with Lanczos selection")
    if not iterative:
        print("  [warn] iterative=False ignored; using Lanczos selection")

    k = min(max_directions if max_directions is not None else d, d)
    eigenvalues, eigenvectors = select_eigenpairs_iterative(
        gp_model, params, x_point, k=k, m=lanczos_m, seed=seed)
    lam_prior = prior_lambda_max_gradient_iterative(
        gp_model, params, seed=seed)
    leading = eigenvalues[0] if eigenvalues.size > 0 else 0.0
    variance_ratios = (
        eigenvalues / leading if leading > 0.0
        else np.full_like(eigenvalues, np.nan))
    rho_full = (np.maximum(eigenvalues, 0.0) / lam_prior
                if lam_prior > 0.0 else np.full_like(eigenvalues, np.nan))
    selected_indices = [j for j in range(len(eigenvalues))
                         if rho_full[j] > rel_tol]

    selected_directions = [eigenvectors[:, i] for i in selected_indices]
    selected_variances = [float(eigenvalues[i]) for i in selected_indices]
    selected_rhos = [float(rho_full[i]) for i in selected_indices]

    order_decisions = _build_order_decisions(
        gp_model, params, x_point, selected_directions, selected_rhos,
        rel_tol=rel_tol, c1=c1, c2=c2,
        acquire_second_order=acquire_second_order,
        max_directions=max_directions,
    )

    return {
        "selected_directions": selected_directions,
        "selected_variances": selected_variances,
        "order_decisions": order_decisions,
        "derivative_covariance": None,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "variance_ratios": variance_ratios,
        "rho": rho_full,
        "K_prior": None,
        "selected_indices": selected_indices,
        "rel_tol": float(rel_tol),
        "k_active": 0,
        "basis_source": "lanczos",
        "Ad0": float(leading),
        "v1": eigenvectors[:, 0],
        "var1": float(eigenvalues[0]),
    }


def sequential_initial_derivative_enrichment(
        gp_model, params, X_train, y_train, grad_func,
        rel_tol=0.05, kernel="SE", kernel_type="anisotropic",
        max_selected_directions=None, optimizer_kwargs=None,
        lambda_abs_tol=0.0,
        hess_func=None, acquire_second_order=False,
        c1=1.0, c2=2.0,
        max_directions=None,
        iterative=True,
        as_basis_provider=None):
    """Process the initial DOE sequentially with real derivative updates."""
    working_model = gp_model
    working_params = np.asarray(params).copy()
    gradients = grad_func(X_train)

    actual_observations = []
    second_order_observations = []
    selection_records = []

    for x_index, x_point in enumerate(X_train):
        selection = select_derivatives_at_observed_point(
            working_model, working_params, x_point, rel_tol=rel_tol,
            as_basis=None,
            c1=c1, c2=c2,
            acquire_second_order=acquire_second_order and hess_func is not None,
            max_directions=max_directions,
            iterative=iterative,
        )

        decisions = selection["order_decisions"]
        if max_selected_directions is not None:
            decisions = decisions[:max_selected_directions]

        grad_at_x = gradients[x_index]
        H_at_x = None
        true_derivs = []
        true_second_derivs = []
        second_order_dirs = []
        second_order_vars = []
        for slot, dec in enumerate(decisions):
            v = dec["direction"]
            d_val = float(grad_at_x @ v)
            true_derivs.append(d_val)
            actual_observations.append({
                "x_index": x_index,
                "direction": v.copy(),
                "value": d_val,
                "slot": slot,
            })
            if dec["order"] == 2:
                if H_at_x is None:
                    H_at_x = hess_func(np.atleast_2d(x_point))
                    if H_at_x.ndim == 3:
                        H_at_x = H_at_x[0]
                d2_val = float(v @ H_at_x @ v)
                true_second_derivs.append(d2_val)
                second_order_dirs.append(v.copy())
                second_order_vars.append(dec["rho2"])
                second_order_observations.append({
                    "x_index": x_index,
                    "direction": v.copy(),
                    "value": d2_val,
                    "slot": slot,
                })

        if decisions:
            working_model, working_params = fit_directional_gp(
                X_train, y_train, actual_observations,
                second_order_observations=second_order_observations or None,
                kernel=kernel, kernel_type=kernel_type,
                optimizer_kwargs=optimizer_kwargs,
                initial_params=working_params,
                max_directions=max_directions,
            )

        record = {
            "x_index": x_index,
            "x_point": np.asarray(x_point).copy(),
            "selected_directions": [dec["direction"].copy() for dec in decisions],
            "selected_variances": list(selection["selected_variances"]),
            "order_decisions": decisions,
            "true_derivs": list(true_derivs),
            "derivative_covariance": None,
            "eigenvalues": selection["eigenvalues"].copy(),
            "eigenvectors": selection["eigenvectors"].copy(),
            "variance_ratios": selection["variance_ratios"].copy(),
            "rho": selection["rho"].copy(),
            "K_prior": None,
            "selected_indices": list(selection["selected_indices"]),
            "k_active": selection.get("k_active", 0),
            "second_order_selected_directions": second_order_dirs,
            "second_order_selected_variances": second_order_vars,
            "second_order_true_derivs": list(true_second_derivs),
        }
        selection_records.append(record)

    return actual_observations, second_order_observations, selection_records
