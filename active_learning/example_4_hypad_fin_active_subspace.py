"""
Example 4 - Active subspace of the HYPAD-UQ heated-fin benchmark (steady state).

Reproduces the steady-state solution from Balcer et al. (2023), "HYPAD-UQ: A
Derivative-Based Uncertainty Quantification Method Using a Hypercomplex Finite
Element Method" (ASME JVVUQ, 8, 021002), and performs an active-subspace
analysis for the two random-variable cases in Table 1 of that paper.

Steady-state normalized tip temperature (paper Eq. 70 at X = 1):
    theta_SS(1) = 1 / cosh(omega),
    omega = sqrt(2 * h_U * b^2 / (k * delta * L))
    T_tip = T_inf + (T_W - T_inf) / cosh(omega)

Cp and rho do not enter the steady-state solution (they only appear in the
transient term), so a valid active-subspace analysis should assign them zero
activity -- a built-in sanity check.

Gradients with respect to the 7 random variables are obtained with
pyoti.sparse (first-order OTI: one function evaluation yields all 7 partials).
"""

import numpy as np
import pyoti.sparse as oti
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VAR_NAMES = [r"$k$", r"$C_p$", r"$\rho$", r"$h_U$",
             r"$T_\infty$", r"$T_W$", r"$b$"]
VAR_KEYS = ["k", "Cp", "rho", "hU", "Tinf", "TW", "b"]

MEANS = np.array([7.1, 580.0, 4430.0, 114.0, 283.0, 389.0, 51.0e-3])

# delta_x = sigma / mu (paper Table 1)
DELTA_CASE1 = np.array([0.10, 0.03, 0.03, 0.10, 0.001, 0.05, 0.01])
DELTA_CASE2 = np.array([0.20, 0.20, 0.20, 0.20, 0.01,  0.20, 0.20])

# Fixed deterministic parameters (paper Sec. 3.4)
DELTA_THICK = 4.75e-3   # fin thickness (m)
L_DEPTH = 1.0           # fin depth out of plane (m)


# ---------------------------------------------------------------------------
# Steady-state model
# ---------------------------------------------------------------------------

def T_tip_real(x):
    """Real-valued steady-state fin-tip temperature for a single input vector."""
    k, Cp, rho, hU, Tinf, TW, b = x
    omega = np.sqrt(2.0 * hU * b * b / (k * DELTA_THICK * L_DEPTH))
    return Tinf + (TW - Tinf) / np.cosh(omega)


def T_tip_grad_oti(x):
    """
    Evaluate T_tip and its gradient at x using first-order OTI.

    Returns (value, grad) where grad has shape (7,) and grad[i] = dT_tip/dx_i.
    """
    # Perturb each input by the i-th imaginary basis with unit coefficient.
    # Note: pyoti uses 1-indexed bases.
    k_o   = x[0] + oti.e(1)
    Cp_o  = x[1] + oti.e(2)   # does not enter steady state; grad should be 0
    rho_o = x[2] + oti.e(3)   # does not enter steady state; grad should be 0
    hU_o  = x[3] + oti.e(4)
    Ti_o  = x[4] + oti.e(5)
    TW_o  = x[5] + oti.e(6)
    b_o   = x[6] + oti.e(7)

    omega = ((2.0 * hU_o * b_o * b_o) / (k_o * DELTA_THICK * L_DEPTH)) ** 0.5
    T = Ti_o + (TW_o - Ti_o) / oti.cosh(omega)

    # Ensure Cp and rho terms are present (they should cancel to zero
    # derivative, but keep the OTI graph aware of them):
    T = T + 0.0 * Cp_o + 0.0 * rho_o

    value = T.real
    grad = np.array([T.get_deriv([i]) for i in range(1, 8)])
    return value, grad


# ---------------------------------------------------------------------------
# Sampling the random-variable distributions (Table 1)
# ---------------------------------------------------------------------------

def sample_case1(n, rng):
    """All variables Normal with (mu, sigma = delta * mu)."""
    sigma = DELTA_CASE1 * MEANS
    return rng.normal(MEANS, sigma, size=(n, 7))


def sample_case2(n, rng):
    """
    Case 2 (paper Table 1):
        k, Cp, rho, hU : Log-normal, delta = 0.20
        T_inf          : Symmetric Triangle, delta = 0.01
        T_W            : Uniform, delta = 0.20
        b              : Uniform, delta = 0.20
    All have mean equal to MEANS and std = delta * mean.
    """
    X = np.empty((n, 7))
    deltas = DELTA_CASE2
    sigmas = deltas * MEANS

    # Log-normal for k, Cp, rho, hU.
    for j, idx in enumerate([0, 1, 2, 3]):
        mu, sig = MEANS[idx], sigmas[idx]
        cov2 = (sig / mu) ** 2
        sig_log = np.sqrt(np.log(1.0 + cov2))
        mu_log = np.log(mu) - 0.5 * sig_log ** 2
        X[:, idx] = rng.lognormal(mean=mu_log, sigma=sig_log, size=n)

    # Symmetric triangular for T_inf. For symmetric triangle on [mu-h, mu+h],
    # variance = h^2 / 6, so h = sqrt(6) * sigma.
    mu, sig = MEANS[4], sigmas[4]
    h = np.sqrt(6.0) * sig
    X[:, 4] = rng.triangular(mu - h, mu, mu + h, size=n)

    # Uniform for T_W and b. For Uniform on [mu - s, mu + s], variance = s^2/3,
    # so s = sqrt(3) * sigma.
    for idx in [5, 6]:
        mu, sig = MEANS[idx], sigmas[idx]
        s = np.sqrt(3.0) * sig
        X[:, idx] = rng.uniform(mu - s, mu + s, size=n)

    return X


# ---------------------------------------------------------------------------
# Active subspace via gradient-covariance eigendecomposition
# ---------------------------------------------------------------------------

def active_subspace(samples, sigmas):
    """
    Compute the active-subspace decomposition of grad f.

    Gradients are scaled by input std-devs so directions are directly
    comparable across variables of very different physical units:
        C = (1/N) sum g_z g_z^T,  with g_z[i] = (df/dx_i) * sigma_i

    Returns dict with 'values', 'grads_x', 'grads_z', 'C', 'eigvals',
    'eigvecs', 'activity_scores'.
    """
    n = samples.shape[0]
    values = np.empty(n)
    grads = np.empty((n, 7))
    for i in range(n):
        v, g = T_tip_grad_oti(samples[i])
        values[i] = v
        grads[i] = g

    grads_z = grads * sigmas[None, :]
    C = grads_z.T @ grads_z / n
    eigvals, eigvecs = np.linalg.eigh(C)
    # Sort descending.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Activity score for variable i: sum_j lambda_j * W[i, j]^2 (over the
    # dominant eigenpairs -- here we sum over all since there are only 7).
    activity = np.sum(eigvals[None, :] * eigvecs ** 2, axis=1)

    return dict(
        values=values,
        grads_x=grads,
        grads_z=grads_z,
        C=C,
        eigvals=eigvals,
        eigvecs=eigvecs,
        activity_scores=activity,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectrum(res1, res2, outdir):
    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.arange(1, 8)
    ax.semilogy(idx, res1["eigvals"], "o-", label="Case 1 (Normal, low var.)")
    ax.semilogy(idx, np.maximum(res2["eigvals"], 1e-30),
                "s--", label="Case 2 (mixed, larger var.)")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel(r"$\lambda_j$ of $E[g g^\top]$ (scaled grads)")
    ax.set_title("Active-subspace eigenvalue spectrum: heated-fin tip T")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "eigenvalue_spectrum.png", dpi=180)
    plt.close(fig)


def plot_activity_scores(res1, res2, outdir):
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.4
    x = np.arange(7)
    # Normalize each case's scores so they sum to 1 for fair comparison.
    a1 = res1["activity_scores"] / res1["activity_scores"].sum()
    a2 = res2["activity_scores"] / res2["activity_scores"].sum()
    ax.bar(x - width / 2, a1, width, label="Case 1")
    ax.bar(x + width / 2, a2, width, label="Case 2")
    ax.set_xticks(x)
    ax.set_xticklabels(VAR_NAMES)
    ax.set_ylabel("Normalized activity score")
    ax.set_title("Per-variable activity (sum of $\\lambda_j W_{ij}^2$)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "activity_scores.png", dpi=180)
    plt.close(fig)


def plot_active_directions(res, case_label, outdir):
    """Bar chart of the first two active directions (eigenvectors)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for k, ax in enumerate(axes):
        w = res["eigvecs"][:, k]
        # Sign-canonicalize to largest-magnitude component positive.
        if w[np.argmax(np.abs(w))] < 0.0:
            w = -w
        ax.bar(np.arange(7), w)
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels(VAR_NAMES)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{case_label}: $w_{k + 1}$  "
                     f"($\\lambda_{k + 1}$ = {res['eigvals'][k]:.3g})")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Component value")
    fig.tight_layout()
    fig.savefig(outdir / f"active_directions_{case_label.lower().replace(' ', '_')}.png",
                dpi=180)
    plt.close(fig)


def plot_sufficient_summary(samples, res, case_label, outdir):
    """1D and 2D sufficient-summary plots along the leading active directions."""
    z = (samples - MEANS[None, :]) / (res["_sigmas"][None, :])
    y = res["values"]
    w1 = res["eigvecs"][:, 0]
    w2 = res["eigvecs"][:, 1]
    u1 = z @ w1
    u2 = z @ w2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(u1, y, s=4, alpha=0.4)
    axes[0].set_xlabel(r"$w_1^\top z$")
    axes[0].set_ylabel(r"$T_\mathrm{tip}$ (K)")
    axes[0].set_title(f"{case_label}: 1D sufficient summary")
    axes[0].grid(True, alpha=0.3)

    sc = axes[1].scatter(u1, u2, c=y, s=5, cmap="viridis")
    axes[1].set_xlabel(r"$w_1^\top z$")
    axes[1].set_ylabel(r"$w_2^\top z$")
    axes[1].set_title(f"{case_label}: 2D sufficient summary")
    fig.colorbar(sc, ax=axes[1], label=r"$T_\mathrm{tip}$ (K)")
    fig.tight_layout()
    fname = f"sufficient_summary_{case_label.lower().replace(' ', '_')}.png"
    fig.savefig(outdir / fname, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("Example 4 - HYPAD-UQ heated fin: steady-state active subspace")
    print("=" * 72)

    outdir = Path(__file__).parent / "example_4_figures"
    outdir.mkdir(exist_ok=True)

    rng = np.random.default_rng(2026)
    N = 4000

    # Sanity check: deterministic T_tip at the mean.
    T0 = T_tip_real(MEANS)
    print(f"\nT_tip at mean inputs: {T0:.6f} K")

    # ----- Case 1 -----
    print("\nCase 1 (Normal, low coefficients of variation)")
    samples1 = sample_case1(N, rng)
    sigmas1 = DELTA_CASE1 * MEANS
    res1 = active_subspace(samples1, sigmas1)
    res1["_sigmas"] = sigmas1

    print("  eigenvalues (descending):")
    print("    " + np.array2string(res1["eigvals"], precision=4))
    print("  normalized activity scores:")
    scores1 = res1["activity_scores"] / res1["activity_scores"].sum()
    for name, s in zip(VAR_KEYS, scores1):
        print(f"    {name:>6s}: {s:.4f}")

    # ----- Case 2 -----
    print("\nCase 2 (log-normal / triangle / uniform, larger coeff. of variation)")
    samples2 = sample_case2(N, rng)
    sigmas2 = DELTA_CASE2 * MEANS
    res2 = active_subspace(samples2, sigmas2)
    res2["_sigmas"] = sigmas2

    print("  eigenvalues (descending):")
    print("    " + np.array2string(res2["eigvals"], precision=4))
    print("  normalized activity scores:")
    scores2 = res2["activity_scores"] / res2["activity_scores"].sum()
    for name, s in zip(VAR_KEYS, scores2):
        print(f"    {name:>6s}: {s:.4f}")

    # ----- Plots -----
    plot_eigenvalue_spectrum(res1, res2, outdir)
    plot_activity_scores(res1, res2, outdir)
    plot_active_directions(res1, "Case 1", outdir)
    plot_active_directions(res2, "Case 2", outdir)
    plot_sufficient_summary(samples1, res1, "Case 1", outdir)
    plot_sufficient_summary(samples2, res2, "Case 2", outdir)

    print(f"\nFigures written to: {outdir.resolve()}")
    print("\nExpected behavior: Cp and rho carry ~zero activity at steady state.")
