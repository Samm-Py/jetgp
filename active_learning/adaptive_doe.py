"""
Adaptive Sequential DOE
=======================

Implements the four-stage adaptive framework:
  Stage 1 — Initial DOE.
  Stage 2 — Build a cost-aware candidate set containing one function-evaluation
            candidate and derivative candidates at existing DOE sites.
  Stage 3 — Score candidates by prior-relative uncertainty per cost.
  Stage 4 — Evaluate the best candidate, update, and refit.

Derivative candidate directions are always selected with Lanczos top-eigenpair
queries against the posterior gradient-covariance matvec operator; the dense
gradient covariance matrix is not materialized for direction selection.

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

import acquisition
from candidates import CostAwareCandidate
import doe_utils
import gp_builders
from lanczos_selection import (
    prior_function_variance,
    prior_lambda_max_gradient as prior_lambda_max_gradient_iterative,
    prior_lambda_max_hessian as prior_lambda_max_hessian_iterative,
    second_order_variance_in_direction,
    select_eigenpairs as select_eigenpairs_iterative,
)
import posterior_queries
import selection


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
        Prior-relative uncertainty-per-cost threshold. Function and derivative
        candidates are kept only when their normalized score exceeds rel_tol.
        Derivative scores use Lanczos-estimated gradient-covariance eigenpairs.
    n_iter : int
    kernel : str
    kernel_type : str
    seed : int
    lambda_abs_tol : float
        Reserved for second-order Stage 4b variance gating. First-order
        selection no longer uses an absolute floor (handled by rel_tol via
        prior-relative rho_j).
    hess_func : callable or None
        hess_f(X: ndarray (1, d)) -> ndarray (d, d).
        Required when acquire_second_order=True.
    acquire_second_order : bool
        If True, include second directional derivative candidates in the
        cost-aware candidate set when hess_func is available.
    """

    def __init__(self, func, grad_func, bounds, n_init, rel_tol, n_iter,
                 kernel="SE", kernel_type="anisotropic",
                 seed=42, lambda_abs_tol=0.0,
                 hess_func=None, acquire_second_order=False,
                 c1=1.0, c2=2.0,
                 max_directions=None, iterative=True,
                 cost_aware=True, c_f=1.0, cost_budget=None,
                 test_set=None, predict_batch_size=250,
                 warm_start_optimizer=True,
                 acquisition_func=None, optimizer_kwargs=None,
                 log_weight_fn=None,
                 X_init=None, as_basis_provider=None, verbose=True):
        self.func = func
        self.grad_func = grad_func
        self.hess_func = hess_func
        self.acquire_second_order = acquire_second_order
        self.c1 = float(c1)
        self.c2 = float(c2) if c2 is not None else None
        self.max_directions = (int(max_directions)
                                if max_directions is not None else None)
        # The adaptive loop is intentionally cost-aware and Lanczos-based.
        # Keep the legacy keyword arguments for caller compatibility, but do
        # not allow them to switch back to the dense/non-cost-aware paths.
        self.iterative = True
        self.cost_aware = True
        self.c_f = float(c_f)
        self.cost_budget = (float(cost_budget) if cost_budget is not None
                              else None)
        self._spectra_cache = {}
        # Optional (X_test, y_test) for inline RMSE tracking in cost-aware mode.
        self.test_set = test_set
        self.predict_batch_size = predict_batch_size
        self.warm_start_optimizer = bool(warm_start_optimizer)
        self.bounds = np.asarray(bounds)
        self.n_init = n_init
        self.rel_tol = rel_tol
        self.n_iter = n_iter
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.seed = seed
        self.lambda_abs_tol = lambda_abs_tol
        # acquisition_func(gp_model, params, bounds, seed=s) -> (x_new, val)
        self.acquisition_func = acquisition_func or acquisition.find_next_point_mpv
        # log_weight_fn(x) -> log p(x). Used to PDF-weight cost-aware scores
        # for both function and derivative candidates so the mixer compares
        # them in test-distribution-aware units. Default uniform (= no
        # weighting; identical to previous behaviour).
        self.log_weight_fn = log_weight_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.verbose = bool(verbose)
        # Retained for constructor compatibility. The active loop now always
        # uses Lanczos-selected local directions, so external bases are ignored.
        self.as_basis_provider = None
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

    def _log(self, message=""):
        if self.verbose:
            print(message)

    def _weight_at(self, x):
        """exp(log_weight_fn(x)) with safeguards. Returns 1.0 if no weight
        function is configured (uniform weighting / unchanged behaviour)."""
        if self.log_weight_fn is None:
            return 1.0
        log_w = float(self.log_weight_fn(np.atleast_2d(x)))
        if not np.isfinite(log_w):
            return 0.0
        return float(np.exp(log_w))

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
        if self.warm_start_optimizer and self.params is not None:
            previous_params = self.params.copy()
        else:
            previous_params = None
        self._spectra_cache.clear()
        has_second = bool(self.second_order_observations)
        if len(self.directional_observations) == 0 and not has_second:
            self.gp_model, self.params = gp_builders.fit_function_only_gp(
                self.X_train, self.y_train,
                kernel=self.kernel, kernel_type=self.kernel_type,
                optimizer_kwargs=self.optimizer_kwargs,
                initial_params=previous_params,
                max_directions=self.max_directions,
            )
        else:
            self.gp_model, self.params = gp_builders.fit_directional_gp(
                self.X_train, self.y_train,
                self.directional_observations,
                second_order_observations=self.second_order_observations or None,
                kernel=self.kernel, kernel_type=self.kernel_type,
                optimizer_kwargs=self.optimizer_kwargs,
                initial_params=previous_params,
                max_directions=self.max_directions,
            )

    def _initialize_design(self):
        """Create the initial DOE and fit the initial function-only model."""
        if self.X_init is not None:
            self.X_train = self.X_init.copy()
        else:
            self.X_train = doe_utils.lhs_design(
                self.n_init, self.bounds, seed=self.seed)
        self.y_train = self.func(self.X_train)
        self.directional_observations = []
        self.second_order_observations = []
        self.history = []

        self._log(f"  Stage 1: {self.X_train.shape[0]} initial points, f in "
                  f"[{self.y_train.min():.3f}, {self.y_train.max():.3f}]")

        self._refit()
        self.initial_function_gp_model = self.gp_model
        self.initial_function_params = self.params.copy()
        self.post_enrichment_gp_model = self.gp_model
        self.post_enrichment_params = self.params.copy()

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
        self._log("\n" + "=" * 60)
        self._log("Stage 1.5 - Initial directional enrichment")
        if self.acquire_second_order:
            self._log("         (first- and second-order)")
        self._log("=" * 60)

        (self.directional_observations,
         self.second_order_observations,
         selection_records) = selection.sequential_initial_derivative_enrichment(
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
            c1=self.c1, c2=self.c2,
            max_directions=self.max_directions,
            iterative=self.iterative,
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
            self._log(
                f"  x_{record['x_index']:03d} = "
                f"{np.round(record['x_point'], 4)}")
            self._log(f"    1st: rho = {rho}  "
                      f"(rel_tol={self.rel_tol}, k_active={k_active})  "
                      f"->  {n_sel1} dir(s)")
            for k, v in enumerate(record["selected_directions"]):
                self._log(f"          v{k+1} = {np.round(v, 4)}")
            if n_sel2 > 0:
                vars2 = np.round(record["second_order_selected_variances"], 5)
                self._log(f"    2nd: Var[d²f/dv²] = {vars2}  "
                          f"(tol={self.lambda_abs_tol:g})  ->  "
                          f"{n_sel2} dir(s)")

        self._log(f"  Added {total_first} first-order, {total_second} "
                  f"second-order observations across "
                  f"{self.X_train.shape[0]} initial DOE point(s)")

        if total_first > 0 or total_second > 0:
            self._refit()
            self._log(f"  params = {self.params}")

        self.post_enrichment_gp_model = self.gp_model
        self.post_enrichment_params = self.params.copy()
        self.initial_derivative_history = selection_records

    def run(self):
        """
        Execute the cost-aware adaptive loop.

        Each iteration builds a unified candidate set containing one possible
        function evaluation and Lanczos-selected derivative evaluations at the
        existing design sites. The highest uncertainty-per-cost candidate is
        evaluated, then the GP is refit.

        Returns
        -------
        history : list of dict, one entry per iteration.
        """
        return self._run_cost_aware()

    # ------------------------------------------------------------------
    # Cost-aware acquisition (unified argmax over function-eval + derivatives)
    # ------------------------------------------------------------------

    def _n_directions_at(self, x_idx):
        return sum(1 for o in self.directional_observations
                   if o["x_index"] == x_idx)

    def _next_slot_at(self, x_idx):
        return self._n_directions_at(x_idx)

    def _direction_already_observed(self, x_idx, v, cos_tol=0.999):
        v = np.asarray(v, dtype=float).reshape(-1)
        for o in self.directional_observations:
            if o["x_index"] == x_idx:
                if abs(float(o["direction"] @ v)) > cos_tol:
                    return True
        return False

    def _get_spectrum(self, x_idx):
        """Cached top-k spectrum at x_train[x_idx]. Invalidated by _refit()."""
        if x_idx in self._spectra_cache:
            return self._spectra_cache[x_idx]
        used = self._n_directions_at(x_idx)
        cap = self.max_directions if self.max_directions is not None else self.bounds.shape[0]
        k = max(0, min(cap, self.bounds.shape[0]) - used)
        if k == 0:
            entry = None
        else:
            x_i = self.X_train[x_idx]
            eigvals, eigvecs = select_eigenpairs_iterative(
                self.gp_model, self.params, x_i, k=k)
            var2 = None
            if self.acquire_second_order and self.hess_func is not None:
                var2 = np.array([
                    second_order_variance_in_direction(
                        self.gp_model, self.params, x_i, eigvecs[:, j])
                    for j in range(eigvecs.shape[1])
                ])
            entry = {"eigvals": eigvals, "eigvecs": eigvecs, "var2": var2}
        self._spectra_cache[x_idx] = entry
        return entry

    def _function_candidate(self):
        """Build the MPV function-evaluation candidate, if it clears the gate."""
        try:
            x_new, var_f = self.acquisition_func(
                self.gp_model, self.params, self.bounds, seed=self.seed)
            lam_prior_f = prior_function_variance(
                self.gp_model, self.params)
            if (np.isfinite(var_f) and np.isfinite(lam_prior_f)
                    and lam_prior_f > 0.0 and self.c_f > 0.0):
                rho_f = max(float(var_f), 0.0) / lam_prior_f
                weight_f = self._weight_at(x_new)
                score_f = rho_f * weight_f / self.c_f
                if score_f > self.rel_tol:
                    return CostAwareCandidate(
                        kind="f",
                        x=np.asarray(x_new).reshape(-1).copy(),
                        score=score_f,
                        rho=rho_f,
                        cost=self.c_f,
                    )
        except (ValueError, np.linalg.LinAlgError) as e:
            self._log(f"  [warn] f-eval candidate skipped: "
                      f"{type(e).__name__}: {e}")
        return None

    def _prior_derivative_scales(self):
        """Return prior scales used to normalize derivative candidate scores."""
        try:
            lam_prior_grad = prior_lambda_max_gradient_iterative(
                self.gp_model, self.params)
        except (ValueError, np.linalg.LinAlgError) as e:
            self._log(f"  [warn] derivative candidates skipped: "
                      f"{type(e).__name__}: {e}")
            return None, None
        do_2nd = self.acquire_second_order and self.hess_func is not None
        try:
            lam_prior_hess = (prior_lambda_max_hessian_iterative(
                self.gp_model, self.params) if do_2nd else 0.0)
        except (ValueError, np.linalg.LinAlgError):
            lam_prior_hess = 0.0
        return lam_prior_grad, lam_prior_hess

    def _derivative_candidates_at(self, x_idx, lam_prior_grad, lam_prior_hess):
        """Build all derivative candidates at one existing function site."""
        candidates = []
        try:
            spec = self._get_spectrum(x_idx)
        except (ValueError, np.linalg.LinAlgError):
            return candidates
        if spec is None:
            return candidates

        eigvals = spec["eigvals"]
        eigvecs = spec["eigvecs"]
        var2 = spec["var2"]
        do_2nd = self.acquire_second_order and self.hess_func is not None
        # Weight all derivative candidates at this site by p(x_i): derivatives
        # at a low-density training point have little impact on test RMSE.
        weight_xi = self._weight_at(self.X_train[x_idx])

        for j in range(len(eigvals)):
            v = eigvecs[:, j]
            if self._direction_already_observed(x_idx, v):
                continue
            if lam_prior_grad is not None and lam_prior_grad > 0.0 and self.c1 > 0.0:
                rho_1 = max(float(eigvals[j]), 0.0) / lam_prior_grad
                score_1 = rho_1 * weight_xi / self.c1
                if score_1 > self.rel_tol:
                    candidates.append(CostAwareCandidate(
                        kind="d",
                        x_idx=int(x_idx),
                        direction=v.copy(),
                        order=1,
                        score=score_1,
                        rho=rho_1,
                        cost=self.c1,
                    ))
            if do_2nd and lam_prior_hess > 0.0 and self.c2 > 0.0:
                rho_2 = max(float(var2[j]), 0.0) / lam_prior_hess
                score_2 = rho_2 * weight_xi / self.c2
                if score_2 > self.rel_tol:
                    candidates.append(CostAwareCandidate(
                        kind="d",
                        x_idx=int(x_idx),
                        direction=v.copy(),
                        order=2,
                        score=score_2,
                        rho=rho_2,
                        cost=self.c2,
                    ))
        return candidates

    def _build_cost_aware_candidates(self):
        candidates = []

        f_candidate = self._function_candidate()
        if f_candidate is not None:
            candidates.append(f_candidate)

        lam_prior_grad, lam_prior_hess = self._prior_derivative_scales()
        if lam_prior_grad is None:
            return candidates
        for x_idx in range(self.X_train.shape[0]):
            candidates.extend(self._derivative_candidates_at(
                x_idx, lam_prior_grad, lam_prior_hess))
        return candidates

    @staticmethod
    def _choose_candidate(candidates):
        return max(candidates, key=lambda c: c.score)

    def _candidates_within_budget(self, candidates, cumulative_cost):
        if self.cost_budget is None:
            return candidates
        remaining = self.cost_budget - cumulative_cost
        return [c for c in candidates if c.cost <= remaining]

    def _apply_candidate(self, candidate):
        """Evaluate one candidate, mutate observations, and return its cost."""
        if candidate.kind == "f":
            x_new = candidate.x
            f_new = float(self.func(np.atleast_2d(x_new))[0, 0])
            self.X_train = np.vstack([self.X_train, x_new[None, :]])
            self.y_train = np.vstack([self.y_train, [[f_new]]])
            self._log(f"  evaluated f at x_new = {np.round(x_new, 4)} -> "
                      f"f={f_new:.4f}")
            return self.c_f

        x_idx = candidate.x_idx
        v = candidate.direction
        order = candidate.order
        slot = self._next_slot_at(x_idx)
        d_val = self._eval_directional(self.X_train[x_idx], v)
        self.directional_observations.append({
            "x_index": x_idx, "direction": v.copy(),
            "value": d_val, "slot": slot,
        })
        msg = (f"  evaluated d^{order}f at x_idx={x_idx}, "
               f"slot={slot}, v={np.round(v, 4)}: df/dv={d_val:.4f}")
        if order == 2:
            d2_val = self._eval_second_order_directional(
                self.X_train[x_idx], v)
            self.second_order_observations.append({
                "x_index": x_idx, "direction": v.copy(),
                "value": d2_val, "slot": slot,
            })
            msg += f", d²f/dv²={d2_val:.4f}"
        self._log(msg)
        return candidate.cost

    def _test_rmse(self):
        if self.test_set is None:
            return None
        X_test, y_test = self.test_set
        mean_pred, _ = posterior_queries.query_function_posterior_batched(
            self.gp_model, self.params, np.atleast_2d(X_test),
            self.predict_batch_size)
        return float(np.sqrt(np.mean(
            (mean_pred - np.asarray(y_test).reshape(-1)) ** 2)))

    def _history_record(self, step, candidate, cumulative_cost):
        return {
            "step": step,
            "chosen_type": candidate.kind,
            "chosen_score": candidate.score,
            "chosen_rho": candidate.rho,
            "chosen_cost": candidate.cost,
            "chosen_x_idx": candidate.x_idx,
            "chosen_x": candidate.x,
            "chosen_direction": candidate.direction,
            "chosen_order": candidate.order,
            "cumulative_cost": cumulative_cost,
            "n_train": self.X_train.shape[0],
            "n_directional_obs": len(self.directional_observations),
            "n_second_order_obs": len(self.second_order_observations),
            "params": self.params.copy(),
            "rmse_test": self._test_rmse(),
        }

    def _run_cost_aware(self):
        """Single-observation-per-iteration loop with unified argmax."""
        self._log("=" * 60)
        self._log("Cost-aware active learning")
        self._log(f"  c_f={self.c_f}, c1={self.c1}, c2={self.c2}, "
                  f"budget={self.cost_budget}, n_iter={self.n_iter}")
        self._log("=" * 60)

        self._initialize_design()

        cumulative_cost = 0.0
        stopped_reason = "completed n_iter"

        for step in range(1, self.n_iter + 1):
            if (self.cost_budget is not None
                    and cumulative_cost >= self.cost_budget):
                stopped_reason = "cost budget exhausted"
                break

            candidates = self._build_cost_aware_candidates()
            candidates = self._candidates_within_budget(
                candidates, cumulative_cost)
            if not candidates:
                if self.cost_budget is not None:
                    stopped_reason = "no affordable candidate above rel_tol"
                else:
                    stopped_reason = "no candidate above rel_tol"
                break

            best = self._choose_candidate(candidates)
            self._log(f"\n--- step {step}/{self.n_iter} "
                      f"(cum cost {cumulative_cost:.3f}) ---")
            self._log(f"  best: type={best.kind}, "
                      f"score={best.score:.4f}, rho={best.rho:.4f}, "
                      f"cost={best.cost:.4f}")

            cumulative_cost += self._apply_candidate(best)
            self._refit()
            self.history.append(
                self._history_record(step, best, cumulative_cost))

        self._log(f"\n{'='*60}")
        self._log(f"Cost-aware loop stopped: {stopped_reason}")
        self._log(f"  Final training set:  {self.X_train.shape[0]} points")
        self._log(f"  First-order obs:     {len(self.directional_observations)}")
        self._log(f"  Second-order obs:    {len(self.second_order_observations)}")
        self._log(f"  Cumulative cost:     {cumulative_cost:.3f}")
        self._log("=" * 60)
        return self.history
