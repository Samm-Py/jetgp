"""
Adaptive Sequential DOE
=======================

Implements the four-stage adaptive framework:
  Stage 2 — Infill location via maximum predictive variance (MPV).
  Stage 3 — Greedy directional derivative selection at x_new (tau threshold).
  Stage 4 — Evaluate true derivatives, update training set, refit GP, repeat.

Usage
-----
    from adaptive_doe import AdaptiveDirectionalGP

    al = AdaptiveDirectionalGP(
        func=my_func, grad_func=my_grad,
        bounds=np.array([[lb0, ub0], [lb1, ub1]]),
        n_init=20, tau=0.05, n_iter=10,
    )
    history = al.run()
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
                              kernel="SE", kernel_type="anisotropic"):
    """Construct a GDDEGP with directional derivative observations."""
    # Group observations by slot
    slots = {}
    for obs in directional_observations:
        s = obs["slot"]
        slots.setdefault(s, []).append(obs)

    n_slots = max(slots.keys()) + 1 if slots else 0

    # Build per-slot arrays
    y_blocks = [y_train]
    rays_list = []
    derivative_locations = []

    for s in range(n_slots):
        slot_obs = slots.get(s, [])
        if not slot_obs:
            continue
        values = np.array([[o["value"]] for o in slot_obs])      # (k, 1)
        rays = np.hstack([_make_ray(o["direction"], 1)
                          for o in slot_obs])                     # (d, k)
        locs = [o["x_index"] for o in slot_obs]

        y_blocks.append(values)
        rays_list.append(rays)
        derivative_locations.append(locs)

    n_dir_types = len(rays_list)
    der_indices = [[[[i + 1, 1]] for i in range(n_dir_types)]] if n_dir_types > 0 else []
    n_bases = max(2 * X_train.shape[1], 2 * max(1, n_dir_types))

    return gddegp(
        X_train, y_blocks,
        n_order=1,
        rays_list=rays_list,
        der_indices=der_indices,
        derivative_locations=derivative_locations,
        n_bases=n_bases,
        normalize=True,
        kernel=kernel,
        kernel_type=kernel_type,
    )


def fit_directional_gp(X_train, y_train, directional_observations,
                       kernel="SE", kernel_type="anisotropic",
                       optimizer_kwargs=None, initial_params=None):
    """

    Each observation is a dict:
        {"x_index": int, "direction": ndarray (d,), "value": float, "slot": int}

    Observations are grouped by ``slot``: all observations with slot=0
    share OTI base pair 1, slot=1 shares pair 2, etc. Reserve enough
    basis directions to query the full coordinate derivative covariance.
    """
    gp_model = _construct_directional_gp(
        X_train,
        y_train,
        directional_observations,
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


def sorted_eigendecomposition(cov):
    """Eigenpairs sorted in descending eigenvalue order."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def select_derivatives_at_xnew(gp_model, params, X_train, y_train,
                                x_new, tau=0.05, lambda_abs_tol=0.0):
    """
    Stage 3: eigenbasis directional derivative selection at x_new.

    The posterior covariance of coordinate directional derivatives at x_new is
    the local gradient covariance matrix. Its eigenvectors are the principal
    directions of directional-derivative uncertainty, and its eigenvalues are
    the corresponding variances. Select eigenvectors while lambda_j/lambda_1
    exceeds tau.

    Returns
    -------
    dict with keys:
        selected_directions  : list of unit ndarray (d,)
        selected_variances   : list of float (eigenvalues at selection)
        fantasy_f_value      : float
        eigenvalues          : ndarray, descending
        eigenvectors         : ndarray, columns matched to eigenvalues
        variance_ratios      : ndarray, eigenvalue / leading eigenvalue
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
    eigenvalues, eigenvectors = sorted_eigendecomposition(cov)
    leading = eigenvalues[0]
    variance_ratios = (
        eigenvalues / leading if leading > 0.0 else np.full_like(eigenvalues, np.nan)
    )

    selected_indices = []
    if leading > lambda_abs_tol:
        selected_indices = [0]
        for i in range(1, len(eigenvalues)):
            if (not np.isnan(variance_ratios[i])) and variance_ratios[i] > tau:
                selected_indices.append(i)
            else:
                break

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
        "selected_indices": selected_indices,
        "lambda_abs_tol": float(lambda_abs_tol),
        "Ad0": float(leading),
        "v1": eigenvectors[:, 0],
        "var1": float(eigenvalues[0]),
    }


def select_derivatives_at_observed_point(
        gp_model, params, x_point, tau=0.05, lambda_abs_tol=0.0):
    """
    Select directional derivatives at an existing function-observation site.

    Unlike ``select_derivatives_at_xnew``, this does not build a fantasy model,
    because the function value at ``x_point`` is already in the training set.
    """
    x_point = np.asarray(x_point).reshape(-1)
    X_point = np.atleast_2d(x_point)

    cov = coordinate_derivative_covariance(gp_model, params, X_point)
    eigenvalues, eigenvectors = sorted_eigendecomposition(cov)
    leading = eigenvalues[0]
    variance_ratios = (
        eigenvalues / leading if leading > 0.0 else np.full_like(eigenvalues, np.nan)
    )

    selected_indices = []
    if leading > lambda_abs_tol:
        selected_indices = [0]
        for i in range(1, len(eigenvalues)):
            if (not np.isnan(variance_ratios[i])) and variance_ratios[i] > tau:
                selected_indices.append(i)
            else:
                break

    selected_directions = [eigenvectors[:, i] for i in selected_indices]
    selected_variances = [float(eigenvalues[i]) for i in selected_indices]

    return {
        "selected_directions": selected_directions,
        "selected_variances": selected_variances,
        "derivative_covariance": cov,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "variance_ratios": variance_ratios,
        "selected_indices": selected_indices,
        "lambda_abs_tol": float(lambda_abs_tol),
        "Ad0": float(leading),
        "v1": eigenvectors[:, 0],
        "var1": float(eigenvalues[0]),
    }


def sequential_initial_derivative_enrichment(
        gp_model, params, X_train, y_train, grad_func,
        tau=0.05, kernel="SE", kernel_type="anisotropic",
        max_selected_directions=None, optimizer_kwargs=None,
        lambda_abs_tol=0.0):
    """
    Process the initial DOE sequentially with real derivative updates.

    At each existing DOE point x^(i), select all directions whose relative
    eigenvalue exceeds ``tau`` under the current model, evaluate and add the
    true directional derivatives, refit, and then continue to the next DOE
    point.
    """
    working_model = gp_model
    working_params = np.asarray(params).copy()
    gradients = grad_func(X_train)

    actual_observations = []
    selection_records = []

    for x_index, x_point in enumerate(X_train):
        selection = select_derivatives_at_observed_point(
            working_model, working_params, x_point, tau=tau,
            lambda_abs_tol=lambda_abs_tol,
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
                X_train,
                y_train,
                actual_observations,
                kernel=kernel,
                kernel_type=kernel_type,
                optimizer_kwargs=optimizer_kwargs,
                initial_params=working_params,
            )

        selection_records.append({
            "x_index": x_index,
            "x_point": np.asarray(x_point).copy(),
            "selected_directions": [v.copy() for v in selection["selected_directions"]],
            "selected_variances": list(selection["selected_variances"]),
            "true_derivs": list(true_derivs),
            "derivative_covariance": selection["derivative_covariance"].copy(),
            "eigenvalues": selection["eigenvalues"].copy(),
            "eigenvectors": selection["eigenvectors"].copy(),
            "variance_ratios": selection["variance_ratios"].copy(),
            "selected_indices": list(selection["selected_indices"]),
        })

    return actual_observations, selection_records


# ---------------------------------------------------------------------------
# AdaptiveDirectionalGP — main class
# ---------------------------------------------------------------------------

class AdaptiveDirectionalGP:
    """

    Implements Stages 1-4 of the white paper for directional derivatives.
    At each new point, directions are chosen from the eigendecomposition of
    the local posterior gradient covariance matrix.

    Parameters
    ----------
    func : callable
        f(X: ndarray (n, d)) -> ndarray (n, 1).
    grad_func : callable
        grad_f(X: ndarray (n, d)) -> ndarray (n, d).
        Full gradient; used to evaluate the selected directional derivatives.
    bounds : ndarray, shape (d, 2)
        [[lb0, ub0], [lb1, ub1], ...].
    n_init : int
        Initial LHS size (paper uses 10*d).
    tau : float
        Diminishing-returns threshold for additional directional derivatives.
    n_iter : int
        Number of active learning iterations.
    kernel : str
    kernel_type : str
    seed : int
        Random seed for LHS and optimizer restarts.
    """

    def __init__(self, func, grad_func, bounds, n_init, tau, n_iter,
                 kernel="SE", kernel_type="anisotropic",
                 seed=42, lambda_abs_tol=0.0):
        self.func = func
        self.grad_func = grad_func
        self.bounds = np.asarray(bounds)
        self.n_init = n_init
        self.tau = tau
        self.n_iter = n_iter
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.seed = seed
        self.lambda_abs_tol = lambda_abs_tol

        # State — populated by run()
        self.X_train = None
        self.y_train = None
        self.directional_observations = []
        self.gp_model = None
        self.params = None
        self.history = []          # one dict per iteration
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
        grad = self.grad_func(np.atleast_2d(x))   # (1, d)
        v = np.asarray(v, dtype=float).reshape(-1)
        return float(grad[0] @ v)

    def _refit(self):
        """Refit the GP from the current training set."""
        previous_params = None if self.params is None else self.params.copy()
        if len(self.directional_observations) == 0:
            self.gp_model, self.params = fit_function_only_gp(
                self.X_train, self.y_train,
                n_dir_types=self.bounds.shape[0],
                kernel=self.kernel, kernel_type=self.kernel_type,
                initial_params=previous_params,
            )
        else:
            self.gp_model, self.params = fit_directional_gp(
                self.X_train, self.y_train,
                self.directional_observations,
                kernel=self.kernel, kernel_type=self.kernel_type,
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

        Points are processed sequentially. At each DOE site, all directions
        passing the eigenvalue-ratio threshold are selected from the current
        real-data model, added as observations, refit, and then the loop moves to the
        next DOE point.
        """
        print("\n" + "=" * 60)
        print("Stage 1.5 - Initial directional enrichment")
        print("=" * 60)

        self.directional_observations, selection_records = (
            sequential_initial_derivative_enrichment(
                self.gp_model,
                self.params,
                self.X_train,
                self.y_train,
                self.grad_func,
                tau=self.tau,
                kernel=self.kernel,
                kernel_type=self.kernel_type,
                lambda_abs_tol=self.lambda_abs_tol,
            )
        )

        total_selected = len(self.directional_observations)
        for record in selection_records:
            ratios = np.round(record["variance_ratios"], 4)
            n_selected = len(record["selected_directions"])
            print(f"  x_{record['x_index']:03d} = {np.round(record['x_point'], 4)}")
            print(f"           lambda_j/lambda_1 = {ratios}  "
                  f"(tau = {self.tau}, lambda_tol = {self.lambda_abs_tol:g})"
                  f"  ->  {n_selected} deriv(s) selected")

        print(f"  Added {total_selected} directional derivative observations "
              f"across {self.X_train.shape[0]} initial DOE point(s)")

        if total_selected > 0:
            self._refit()
            print(f"  params = {self.params}")

        self.post_enrichment_gp_model = self.gp_model
        self.post_enrichment_params = self.params.copy()
        self.initial_derivative_history = selection_records

    def run(self):
        """
        Execute the full adaptive loop.

        Stage 1 — Initial DOE.
        Optional Stage 1.5 — enrich the initial DOE with directional derivatives.
        For each of n_iter iterations:
          Stage 2 — Select x_new via MPV.
          Stage 3 — Evaluate f(x_new), add it, and refit.
          Stage 4 — Select/evaluate directional derivatives and refit.

        Returns
        -------
        history : list of dict, one entry per iteration.
        """
        # --- Stage 1: Initial DOE ---
        print("=" * 60)
        print("Stage 1 — Initial LHS DOE")
        print("=" * 60)
        self.X_train = lhs_design(self.n_init, self.bounds, seed=self.seed)
        self.y_train = self.func(self.X_train)
        self.directional_observations = []
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
            x_new, mpv = find_next_point_mpv(
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
                {
                    "x_index": obs["x_index"],
                    "direction": obs["direction"].copy(),
                    "value": obs["value"],
                    "slot": obs["slot"],
                }
                for obs in self.directional_observations
            ]

            # Stage 3 — observe the function value, add it to the data, and refit.
            f_new = float(self.func(np.atleast_2d(x_new))[0, 0])
            new_index = self.X_train.shape[0]
            self.X_train = np.vstack([self.X_train, np.atleast_2d(x_new)])
            self.y_train = np.vstack([self.y_train, np.array([[f_new]])])
            self._refit()
            print(f"  Stage 3: f(x_new) = {f_new:.4f}; refit before derivative selection")

            # Stage 4 — select derivatives at the observed point, evaluate, refit.
            selection = select_derivatives_at_observed_point(
                self.gp_model, self.params, x_new, tau=self.tau,
                lambda_abs_tol=self.lambda_abs_tol,
            )
            n_selected = len(selection["selected_directions"])
            ratios = np.round(selection["variance_ratios"], 4)
            print(f"  Stage 4: eigenvalues = "
                  f"{np.round(selection['eigenvalues'], 5)}")
            print(f"           lambda_j/lambda_1 = {ratios}  "
                  f"(tau = {self.tau}, lambda_tol = {self.lambda_abs_tol:g})"
                  f"  ->  {n_selected} deriv(s) selected")
            for idx, (v, var) in enumerate(
                    zip(selection["selected_directions"],
                        selection["selected_variances"]), start=1):
                print(f"           v{idx} = {np.round(v, 4)},  "
                      f"Var[d_v{idx} f] = {var:.5f}")

            true_derivs = []
            for idx, v in enumerate(selection["selected_directions"]):
                d_val = self._eval_directional(x_new, v)
                true_derivs.append(d_val)
                self.directional_observations.append({
                    "x_index": new_index,
                    "direction": v,
                    "value": d_val,
                    "slot": idx,
                })
            print(f"           true derivs = {[round(v, 4) for v in true_derivs]}")

            self._refit()
            print(f"  params = {self.params}")

            # Diagnostics
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
                "selected_directions": selection["selected_directions"],
                "selected_variances": selection["selected_variances"],
                "true_derivs": true_derivs,
                "n_selected": n_selected,
                "derivative_covariance": selection["derivative_covariance"].copy(),
                "eigenvalues": selection["eigenvalues"].copy(),
                "eigenvectors": selection["eigenvectors"].copy(),
                "variance_ratios": selection["variance_ratios"].copy(),
                "selected_indices": list(selection["selected_indices"]),
                "v1": selection["v1"].copy(),
                "var1": selection["var1"],
                "n_train": self.X_train.shape[0],
                "n_directional_obs": len(self.directional_observations),
                "params": self.params.copy(),
            }
            self.history.append(record)

        print(f"\n{'='*60}")
        print("Active learning loop complete.")
        print(f"  Final training set:  {self.X_train.shape[0]} points")
        print(f"  Directional obs:     {len(self.directional_observations)}")
        print("=" * 60)
        return self.history


