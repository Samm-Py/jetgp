"""Cost-aware active learning driven by global variance reduction (A / B / C).

Same shape as ``example_1_branin_hoo.py``, but every candidate is scored with a
single *global* criterion instead of the local MPV / Lanczos pair used by
``AdaptiveDirectionalGP``: the reduction in the integrated posterior variance of
the **function value** over the domain (see ``global_variance.py``).

Three observation types compete at every step:

    A ("d")   a directional derivative  d_v f(x_i)  at an existing design site
    B ("f")   a function value          f(x_new)    at a new site
    C ("fd")  both                      f(x_new) and d_v f(x_new) at a new site

each with its own cost, ranked by GVR per unit cost. The optimal direction for
A and C is available in closed form (a generalized Rayleigh quotient), so the
only search is over the new site ``x``.

Run::

    python active_learning/example_5_global_variance_abc.py

Reading the results
-------------------
With the default costs (``C_F = 1.0``, ``C_D = 0.25``) option A wins every step
on the Branin-Hoo problem, and option C never wins at all. That is a property of
the criterion rather than an artefact. Writing ``G_A``, ``G_B``, ``G_C`` for the
three raw reductions at one step:

    C wins   <=>   G_A / (G_C - G_A)  <  c_d  <  G_C / G_B - 1

On this problem that interval is empty -- at the first step the lower bound is
about 2.2 and the upper about 0.24 -- so no choice of ``C_D`` makes C the
winner. The reason is that ``f(x)`` and ``d_v f(x)`` at the *same* point are
largely redundant for reducing variance elsewhere: conditioning on the function
value first leaves the derivative only a small marginal contribution
(``G_C - G_B``), while a derivative at an *existing* site (option A) buys a
comparable reduction without paying for a function evaluation at all.

So option C matters when option A is not actually available -- when the
derivative can only be obtained together with a fresh function evaluation, as
with an adjoint or HYPAD solve, or once every site has hit ``MAX_DIRECTIONS``.
Set ``ENABLE_KINDS = ("f", "fd")`` to run that comparison.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

THIS_FILE = Path(__file__).resolve()
ACTIVE_LEARNING_DIR = THIS_FILE.parent
REPO_ROOT = ACTIVE_LEARNING_DIR.parent
# Prefer the jetgp source tree this script ships with. Running the file by path
# puts only its own directory on sys.path, so without this the import would fall
# through to whatever other jetgp copy the environment has registered.
for entry in (str(ACTIVE_LEARNING_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import doe_utils
import gp_builders
import global_variance as gv
from candidates import CostAwareCandidate
from posterior_queries import query_function_posterior_batched


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BOUNDS = np.array([[-5.0, 10.0], [0.0, 15.0]])

N_INIT = 10             # initial LHS design size
N_ITER = 12             # acquisition steps
SEED = 11

C_F = 1.0               # cost of one function evaluation
C_D = 0.25              # cost of one directional derivative evaluation
COST_BUDGET = None      # optional cap on cumulative cost

N_INTEGRATION = 120     # |Z|; the GVR integral is a mean over this set
N_CANDIDATES = 96       # coarse LHS sweep for the new-site search
N_POLISH = 3            # coarse winners refined with a short local search
POLISH_MAXITER = 12

MAX_DIRECTIONS = 2      # distinct directions observable at any one site
COS_TOL = 0.999         # two directions closer than this count as the same

# Which of the three observation types may be acquired. Drop "d" to model a
# solver that only hands back a derivative alongside a fresh function
# evaluation (an adjoint or HYPAD solve), which is the regime where option C
# earns its place -- see "Reading the results" in the module docstring.
ENABLE_KINDS = ("d", "f", "fd")

VALIDATE = True         # check each step's realised drop against its prediction
VALIDATE_TOL = 1e-6

OPTIMIZER_KWARGS = {
    "pop_size": 40,
    "n_generations": 20,
    "local_opt_every": 20,
    "debug": False,
}


# --------------------------------------------------------------------------
# Branin-Hoo test problem (same constants as the tutorial)
# --------------------------------------------------------------------------

A = 1.0
B = 5.1 / (4 * np.pi**2)
C = 5.0 / np.pi
R = 6.0
S = 10.0
T = 1.0 / (8.0 * np.pi)


def branin(X):
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    term = x2 - B * x1**2 + C * x1 - R
    y = A * term**2 + S * (1 - T) * np.cos(x1) + S
    return y.reshape(-1, 1)


def branin_grad(X):
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    term = x2 - B * x1**2 + C * x1 - R
    df_dx1 = 2 * A * term * (-2 * B * x1 + C) - S * (1 - T) * np.sin(x1)
    df_dx2 = 2 * A * term
    return np.column_stack([df_dx1, df_dx2])


def make_grid(n_per_axis):
    x1 = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], n_per_axis)
    x2 = np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], n_per_axis)
    X1, X2 = np.meshgrid(x1, x2)
    return np.column_stack([X1.ravel(), X2.ravel()]), X1, X2


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

KIND_LABEL = {"d": "A  d_v f @ existing", "f": "B  f     @ new     ",
              "fd": "C  f+d_v f @ new    "}
KIND_LETTER = {"d": "A", "f": "B", "fd": "C"}
KIND_LEGEND = {"d": "A  derivative at existing site",
               "f": "B  function at new site",
               "fd": "C  function + derivative at new site"}
KIND_COLOR = {"d": "tab:red", "f": "tab:blue", "fd": "tab:green"}


class GlobalVarianceLoop:
    """Sequential DOE that always picks the best GVR-per-cost observation."""

    def __init__(self, func, grad_func, bounds, seed=SEED):
        self.func = func
        self.grad_func = grad_func
        self.bounds = np.asarray(bounds, dtype=float)
        self.seed = int(seed)
        self.dim = self.bounds.shape[0]

        # The integration set defining the "global" in global variance
        # reduction. Fixed for the whole run so successive GVR values are
        # directly comparable.
        self.Z = doe_utils.lhs_design(N_INTEGRATION, self.bounds,
                                      seed=self.seed + 1)

        self.X_train = None
        self.y_train = None
        self.directional_observations = []
        self.gp_model = None
        self.params = None
        self.history = []
        self.snapshots = []

    # -- model management ---------------------------------------------------

    def _refit(self):
        previous = self.params.copy() if self.params is not None else None
        common = dict(optimizer_kwargs=OPTIMIZER_KWARGS,
                      initial_params=previous,
                      max_directions=MAX_DIRECTIONS)
        if self.directional_observations:
            self.gp_model, self.params = gp_builders.fit_directional_gp(
                self.X_train, self.y_train, self.directional_observations,
                **common)
        else:
            self.gp_model, self.params = gp_builders.fit_function_only_gp(
                self.X_train, self.y_train, **common)

    def initialize(self):
        self.X_train = doe_utils.lhs_design(N_INIT, self.bounds, seed=self.seed)
        self.y_train = self.func(self.X_train)
        self.directional_observations = []
        self._refit()
        self.snapshots.append(self._snapshot("Initial design"))

    # -- candidate construction --------------------------------------------

    def _observed_directions(self, x_idx):
        return [o["direction"] for o in self.directional_observations
                if o["x_index"] == x_idx]

    def _candidate_a(self, noise_var):
        """Best derivative-only candidate over all existing design sites."""
        best = None
        for x_idx in range(self.X_train.shape[0]):
            observed = self._observed_directions(x_idx)
            if len(observed) >= MAX_DIRECTIONS:
                continue
            # `exclude` is only a numerical guard: once d_v f has been
            # observed at a site the refit posterior already has almost no
            # remaining variance along v, so the top eigenvector moves away
            # from it on its own.
            try:
                gvr, v = gv.gvr_derivative_at_site(
                    self.gp_model, self.params, self.X_train[x_idx],
                    self.Z, noise_var, exclude=observed, cos_tol=COS_TOL)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if v is None or gvr <= 0.0:
                continue
            score = gvr / C_D
            if best is None or score > best.score:
                best = CostAwareCandidate(
                    kind="d", score=score, rho=gvr, cost=C_D,
                    x=self.X_train[x_idx].copy(), x_idx=int(x_idx),
                    direction=v.copy(), order=1)
        return best

    def _candidates_bc(self, noise_var):
        """Best function-only and function+derivative candidates at new sites.

        A coarse LHS sweep followed by a short local polish. Both options are
        read off the same posterior query, so scoring a site costs one call
        regardless of how many kinds it feeds.
        """
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        # (score, x, gvr, direction) incumbents, one per kind
        best = {"f": (-np.inf, None, 0.0, None),
                "fd": (-np.inf, None, 0.0, None)}

        def evaluate(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            if not np.all(np.isfinite(x)):
                return None
            try:
                gvr_b, gvr_c, v_c = gv.gvr_at_new_site(
                    self.gp_model, self.params, x, self.Z, noise_var)
            except (ValueError, np.linalg.LinAlgError):
                return None
            if not (np.isfinite(gvr_b) and np.isfinite(gvr_c)):
                return None
            score_b = gvr_b / C_F
            score_c = gvr_c / (C_F + C_D) if v_c is not None else -np.inf
            if score_b > best["f"][0]:
                best["f"] = (score_b, x.copy(), gvr_b, None)
            if score_c > best["fd"][0]:
                best["fd"] = (score_c, x.copy(), gvr_c, v_c.copy())
            return score_b, score_c

        sweep = doe_utils.lhs_design(N_CANDIDATES, self.bounds,
                                     seed=self.seed + 100 + len(self.history))
        scored = []
        for x in sweep:
            out = evaluate(x)
            if out is not None:
                scored.append((max(out), x))

        # Polish the most promising coarse sites; every evaluation inside the
        # local search updates both incumbents, so one run serves both kinds.
        scored.sort(key=lambda t: -t[0])
        _BAD = 1e20

        def neg_score(x, idx):
            out = evaluate(x)
            if out is None or not np.isfinite(out[idx]):
                return _BAD
            return -out[idx]

        for _, x0 in scored[:N_POLISH]:
            for idx in (0, 1):
                try:
                    minimize(neg_score, x0=x0, args=(idx,), method="L-BFGS-B",
                             bounds=list(zip(lb, ub)),
                             options={"maxiter": POLISH_MAXITER})
                except (ValueError, np.linalg.LinAlgError):
                    continue

        out = []
        for kind, cost in (("f", C_F), ("fd", C_F + C_D)):
            score, x, gvr, v = best[kind]
            if x is None or not np.isfinite(score) or score <= 0.0:
                continue
            out.append(CostAwareCandidate(
                kind=kind, score=score, rho=gvr, cost=cost,
                x=x.copy(), direction=None if v is None else v.copy(),
                order=None if kind == "f" else 1))
        return out

    # -- applying a choice --------------------------------------------------

    def _next_slot_at(self, x_idx):
        return sum(1 for o in self.directional_observations
                   if o["x_index"] == x_idx)

    def _apply(self, cand):
        if cand.kind in ("f", "fd"):
            x_new = cand.x
            self.X_train = np.vstack([self.X_train, x_new[None, :]])
            self.y_train = np.vstack(
                [self.y_train, self.func(np.atleast_2d(x_new))])
            x_idx = self.X_train.shape[0] - 1
        else:
            x_idx = cand.x_idx

        if cand.kind in ("d", "fd"):
            v = cand.direction
            value = float(self.grad_func(
                np.atleast_2d(self.X_train[x_idx]))[0] @ v)
            self.directional_observations.append({
                "x_index": x_idx, "direction": v.copy(),
                "value": value, "slot": self._next_slot_at(x_idx),
            })
        return cand.cost

    # -- diagnostics --------------------------------------------------------

    def _integrated_variance(self, gp_model=None, params=None):
        gp_model = self.gp_model if gp_model is None else gp_model
        params = self.params if params is None else params
        _, var = query_function_posterior_batched(
            gp_model, params, self.Z, batch_size=250)
        return float(np.mean(var))

    def _test_rmse(self, X_test, y_test):
        mean, _ = query_function_posterior_batched(
            self.gp_model, self.params, X_test, batch_size=250)
        return float(np.sqrt(np.mean((mean.reshape(-1, 1) - y_test) ** 2)))

    def _snapshot(self, label):
        return {
            "label": label,
            "X": self.X_train.copy(),
            "d": [{"x": self.X_train[o["x_index"]].copy(),
                   "v": o["direction"].copy()}
                  for o in self.directional_observations],
        }

    # -- main loop ----------------------------------------------------------

    def run(self, n_iter=N_ITER, test_set=None):
        self.initialize()
        cumulative_cost = 0.0
        gv_before = self._integrated_variance()
        print(f"Initial integrated variance over |Z|={len(self.Z)}: "
              f"{gv_before:.6e}\n")

        for step in range(1, n_iter + 1):
            t0 = time.time()
            noise_var = gv.noise_variance(self.gp_model, self.params)

            candidates = []
            if "d" in ENABLE_KINDS:
                a = self._candidate_a(noise_var)
                if a is not None:
                    candidates.append(a)
            if "f" in ENABLE_KINDS or "fd" in ENABLE_KINDS:
                candidates.extend(self._candidates_bc(noise_var))
            candidates = [c for c in candidates if c.kind in ENABLE_KINDS]
            if COST_BUDGET is not None:
                remaining = COST_BUDGET - cumulative_cost
                candidates = [c for c in candidates if c.cost <= remaining]
            if not candidates:
                print(f"step {step}: no admissible candidate, stopping.")
                break

            winner = max(candidates, key=lambda c: c.score)
            self._report_step(step, candidates, winner)

            cumulative_cost += self._apply(winner)
            gv_predicted = winner.rho

            gv_realised = None
            if VALIDATE:
                gv_realised = self._validate(
                    gv_before, gv_predicted, winner.kind)

            self._refit()
            gv_after = self._integrated_variance()
            rmse = (self._test_rmse(*test_set) if test_set is not None
                    else None)

            self.history.append({
                "step": step, "kind": winner.kind, "cost": winner.cost,
                "cumulative_cost": cumulative_cost,
                "gvr_predicted": gv_predicted, "gvr_realised": gv_realised,
                "integrated_variance": gv_after, "rmse_test": rmse,
                "seconds": time.time() - t0,
            })
            self.snapshots.append(self._snapshot(
                f"After step {step}: {KIND_LETTER[winner.kind]}"))

            print(f"    integrated variance {gv_before:.6e} -> {gv_after:.6e}"
                  f"   cumulative cost {cumulative_cost:.2f}"
                  + (f"   test RMSE {rmse:.4f}" if rmse is not None else "")
                  + f"   ({time.time() - t0:.1f}s)\n")
            gv_before = gv_after

        return self.history

    def _validate(self, gv_before, gv_predicted, kind):
        """Realised drop at *fixed* hyperparameters, against the predicted GVR.

        Must be measured before ``_refit`` moves the hyperparameters. The two
        agree to machine precision only when the rebuilt model uses the same
        real-space kernel, i.e. for option A: ``gp_builders`` always sets
        ``normalize=True``, so adding a *point* (options B and C) shifts
        ``mus_x`` / ``sigmas_x`` and ``mu_y`` / ``sigma_y`` and thereby the
        effective length scales. For those the two numbers are reported side by
        side but not compared. ``tests/test_global_variance.py`` checks all
        three kinds exactly, using unnormalised models.
        """
        try:
            model = gp_builders._construct_directional_gp(
                self.X_train, self.y_train, self.directional_observations,
                max_directions=MAX_DIRECTIONS)
            gv_after = self._integrated_variance(model, self.params)
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"    [warn] validation skipped: {type(exc).__name__}: {exc}")
            return None
        realised = gv_before - gv_after
        if kind == "d":
            rel = abs(realised - gv_predicted) / max(abs(gv_predicted), 1e-30)
            flag = "ok" if rel < VALIDATE_TOL else f"MISMATCH rel={rel:.2e}"
        else:
            flag = "renormalised on rebuild, not compared"
        print(f"    predicted GVR {gv_predicted:.6e} | "
              f"realised {realised:.6e} | {flag}")
        return realised

    def _report_step(self, step, candidates, winner):
        print(f"step {step}")
        by_kind = {}
        for c in candidates:
            if c.kind not in by_kind or c.score > by_kind[c.kind].score:
                by_kind[c.kind] = c
        for kind in ("d", "f", "fd"):
            c = by_kind.get(kind)
            if c is None:
                print(f"  {KIND_LABEL[kind]}  --")
                continue
            where = (f"x[{c.x_idx}]=" if c.kind == "d" else "")
            where += np.array2string(c.x, precision=3, suppress_small=True)
            direction = ("  v=" + np.array2string(c.direction, precision=3,
                                                  suppress_small=True)
                         if c.direction is not None else "")
            mark = "  <-- selected" if c is winner else ""
            print(f"  {KIND_LABEL[kind]}  {where}{direction}"
                  f"  GVR={c.rho:.4e}  cost={c.cost:.2f}"
                  f"  score={c.score:.4e}{mark}")


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_sequence(snapshots, figure_path):
    X_grid, X1, X2 = make_grid(140)
    truth = branin(X_grid).reshape(X1.shape)
    picks = sorted({0, 1, 2, min(4, len(snapshots) - 1), len(snapshots) - 1})
    selected = [snapshots[i] for i in picks]

    fig, axes = plt.subplots(1, len(selected),
                             figsize=(4.0 * len(selected), 3.7),
                             constrained_layout=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, snap in zip(axes, selected):
        ax.contourf(X1, X2, truth, levels=32, cmap="viridis")
        ax.scatter(snap["X"][:, 0], snap["X"][:, 1], c="white", s=28,
                   edgecolor="black", linewidth=0.8)
        for obs in snap["d"]:
            x, v = obs["x"], obs["v"]
            ax.arrow(x[0], x[1], 0.9 * v[0], 0.9 * v[1], width=0.02,
                     head_width=0.18, color="tab:red",
                     length_includes_head=True, alpha=0.8)
        ax.set_title(snap["label"], fontsize=10)
        ax.set_xlim(BOUNDS[0])
        ax.set_ylim(BOUNDS[1])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def plot_surrogate(loop, figure_path):
    X_grid, X1, X2 = make_grid(75)
    truth = branin(X_grid).reshape(X1.shape)
    mean, var = query_function_posterior_batched(
        loop.gp_model, loop.params, X_grid, batch_size=250)
    mean = mean.reshape(X1.shape)
    var = var.reshape(X1.shape)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8),
                             constrained_layout=True)
    fields = [(truth, "True Branin-Hoo function", "viridis"),
              (mean, "Final surrogate mean", "viridis"),
              (var, "Final posterior variance", "magma")]
    for ax, (field, title, cmap) in zip(axes, fields):
        contour = ax.contourf(X1, X2, field, levels=32, cmap=cmap)
        fig.colorbar(contour, ax=ax)
        ax.scatter(loop.X_train[:, 0], loop.X_train[:, 1], c="white", s=22,
                   edgecolor="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlim(BOUNDS[0])
        ax.set_ylim(BOUNDS[1])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    axes[1].contour(X1, X2, np.abs(mean - truth), levels=6, colors="white",
                    linewidths=0.4, alpha=0.5)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def plot_progress(history, figure_path):
    if not history:
        return
    cost = [h["cumulative_cost"] for h in history]
    igv = [h["integrated_variance"] for h in history]
    kinds = [h["kind"] for h in history]
    rmse = [h["rmse_test"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0),
                            constrained_layout=True)
    axes[0].plot(cost, igv, "-", color="0.4", zorder=1)
    for kind in ("d", "f", "fd"):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        if idx:
            axes[0].scatter([cost[i] for i in idx], [igv[i] for i in idx],
                            s=45, color=KIND_COLOR[kind], zorder=2,
                            label=KIND_LEGEND[kind])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("cumulative cost")
    axes[0].set_ylabel(r"integrated variance  $\frac{1}{M}\sum_z \sigma_f^2(z)$")
    axes[0].set_title("Global function-value uncertainty vs cost")
    axes[0].legend(fontsize=9)

    if all(r is not None for r in rmse):
        axes[1].plot(cost, rmse, "o-", color="tab:purple")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("cumulative cost")
        axes[1].set_ylabel("test RMSE")
        axes[1].set_title("Accuracy vs cost")
    else:
        axes[1].axis("off")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------

def main():
    static_dir = REPO_ROOT / "docs" / "source" / "_static"
    static_dir.mkdir(parents=True, exist_ok=True)

    X_test, _, _ = make_grid(25)
    y_test = branin(X_test)

    loop = GlobalVarianceLoop(branin, branin_grad, BOUNDS, seed=SEED)
    history = loop.run(n_iter=N_ITER, test_set=(X_test, y_test))

    plot_sequence(loop.snapshots,
                  static_dir / "gvr_active_learning_sequence.png")
    plot_surrogate(loop, static_dir / "gvr_active_learning_surrogate.png")
    plot_progress(history, static_dir / "gvr_active_learning_progress.png")

    counts = {k: sum(1 for h in history if h["kind"] == k)
              for k in ("d", "f", "fd")}
    print("=" * 72)
    print(f"Function sites:              {loop.X_train.shape[0]}")
    print(f"Directional observations:    {len(loop.directional_observations)}")
    print(f"Selections  A(d)={counts['d']}  B(f)={counts['f']}  "
          f"C(fd)={counts['fd']}")
    if history:
        print(f"Cumulative cost:             "
              f"{history[-1]['cumulative_cost']:.2f}")
        print(f"Final integrated variance:   "
              f"{history[-1]['integrated_variance']:.6e}")
        if history[-1]["rmse_test"] is not None:
            print(f"Final test RMSE:             "
                  f"{history[-1]['rmse_test']:.6f}")
    print(f"Figures written to {static_dir}")


if __name__ == "__main__":
    main()
