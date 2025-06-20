# ishigami_ddgp_demo.py
# ------------------------------------------------------------
# Directional–Derivative GP demo on the 3-D Ishigami function
# ------------------------------------------------------------
import numpy as np
import itertools
import pyoti.sparse as oti
from scipy.stats import qmc
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from full_gddegp.gddegp import gddegp     # ← your DD-GP class
import utils                              # nrmse helper (unchanged)

# --- Ishigami constants -------------------------------------------------
a_ish, b_ish = 7.0, 0.1
pi = np.pi

# --- true function & gradient ------------------------------------------


def true_function(X, alg=np):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (alg.sin(x1)
            + a_ish * alg.sin(x2)**2
            + b_ish * x3**4 * alg.sin(x1))


def grad_ishigami(x1, x2, x3):
    gx1 = np.cos(x1) + b_ish * x3**4 * np.cos(x1)
    gx2 = 2 * a_ish * np.sin(x2) * np.cos(x2)     # a·sin2x2
    gx3 = 4 * b_ish * x3**3 * np.sin(x1)
    return gx1, gx2, gx3

# -----------------------------------------------------------------------
# 1) Latin-hypercube points on [-π,π]³  + per-point gradient rays
# -----------------------------------------------------------------------


def sobol_points(n_samples, seed=None, scramble=True):
    """
    Sobol sampling on [-π, π]^3.

    Parameters
    ----------
    n_samples : int
        Number of points you want.  
        • If `scramble=False` you must supply a power-of-two (2, 4, 8, …).  
        • With `scramble=True` any positive integer works.
    seed : int or None
        Random seed for reproducible scrambling.
    scramble : bool
        Scramble the sequence (Owen scrambling). Recommended unless you
        truly need the deterministic Sobol order.

    Returns
    -------
    X : ndarray, shape (n_samples, 3)
        Points mapped to [-π, π] in each coordinate.
    """
    sampler = qmc.Sobol(d=3, scramble=scramble, seed=seed)

    # if scramble:
    #     # Sobol.random works for any n when scrambled
    #     unit = sampler.random(n_samples)                # (n,3) in [0,1)
    # else:
    # must draw a power-of-two with random_base2
    m = int(np.ceil(np.log2(n_samples)))
    unit = sampler.random_base2(m)[:n_samples]

    X = -pi + 2 * pi * unit        # affine map to [-π, π]^3
    return X


def generate_pointwise_rays(n_samples=24, n_order=1, seed=1):
    """Generate random rays instead of gradient-based ones"""
    X = sobol_points(n_samples, seed)
    rays, tags = [], []

    # Set random seed for reproducible random rays
    np.random.seed(seed)

    for idx, (x1, x2, x3) in enumerate(X):
        # Generate random unit vector instead of gradient
        g = np.random.randn(3)  # Random normal vector
        g /= np.linalg.norm(g)  # Normalize to unit vector
        rays.append(g[:, None])  # (3,1)
        tags.append(idx + 1)     # unique tag
    return X, rays, tags

# -----------------------------------------------------------------------
# 2) apply hyper-complex perturbation (one tag per point)
# -----------------------------------------------------------------------


def apply_pointwise_perturb(X, rays, tags, n_order):
    X_hc = oti.array(X)
    for i, (ray, tag) in enumerate(zip(rays, tags)):
        e_tag = oti.e(1, order=n_order)            # dual unit
        X_hc[i, :] += (oti.array(ray)*e_tag).T
    return X_hc

# -----------------------------------------------------------------------
# 3) build training targets  f  and  ∂_v f
# -----------------------------------------------------------------------


def generate_training_data(n_samples=24, n_order=1):
    X, rays, tags = generate_pointwise_rays(n_samples, n_order)
    X_hc = apply_pointwise_perturb(X, rays, tags, n_order)
    f_hc = true_function(X_hc, alg=oti)

    # drop cross-tag terms
    for pair in itertools.combinations(tags, 2):
        f_hc = f_hc.truncate(pair)

    # ---- assemble output blocks ------------------------------------------
    y_blocks = [f_hc.real.reshape(-1, 1)]          # function values

    der_indices = [[[1, i+1]] for i in range(n_order)]

    for idx in der_indices:
        y_blocks.append(
            f_hc.get_deriv(idx).reshape(-1, 1)
        )

    return X, rays, y_blocks

# -----------------------------------------------------------------------
# 4) GP fit + 2-D slice prediction  (x3 = 0)
# -----------------------------------------------------------------------


def main():
    np.random.seed(0)
    n_order = 4

    X_train, rays_list, y_blocks = generate_training_data(n_samples=66,
                                                          n_order=n_order)
    rays_array = np.hstack(rays_list)

    gp = gddegp(X_train, y_blocks,
                n_order=n_order,
                rays_array=rays_array,
                normalize=True,
                kernel="SE",
                kernel_type="anisotropic")

    params = gp.optimize_hyperparameters(n_restart_optimizer=30,
                                         swarm_size=150, verbose=True)

    # --- 2-D slice grid (x3 = 0) ------------------------------------
    gx = np.linspace(-pi,  pi, 100)
    gy = np.linspace(-pi,  pi, 100)
    X1, X2 = np.meshgrid(gx, gy)
    X_pred = np.column_stack([X1.ravel(), X2.ravel(),
                              np.pi + np.zeros_like(X1).ravel()])

    # # fixed prediction direction  u = e₁  (cosθ=1,sinθ=0)
    u = np.array([1., 0., 0.])
    u_col = u[:, None]
    rays_pred = np.hstack([u_col] * X_pred.shape[0])

    y_star = gp.predict(X_pred, rays_pred, params,
                        calc_cov=False, return_deriv=False)

    N = X_pred.shape[0]
    f_gp = y_star[:N]
    # df_gp = y_star[N:2*N].reshape(X1.shape)

    # # analytic directional derivative along e₁
    # df_true = grad_ishigami(X_pred[:, 0],
    #                         X_pred[:, 1],
    #                         X_pred[:, 2])[0].reshape(X1.shape)

    # # absolute error map
    # abs_err = np.abs(df_gp - df_true)
    # eps = 1e-8
    # abs_err_clip = np.clip(abs_err, eps, None)

    # # -------------------------------------------------- plotting -----
    # fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    # cf0 = ax[0].contourf(X1, X2, df_true, 40, cmap="viridis")
    # fig.colorbar(cf0, ax=ax[0])
    # ax[0].set_title("Analytic ∂ₓ₁ f")

    # cf1 = ax[1].contourf(X1, X2, df_gp, 40, cmap="viridis")
    # fig.colorbar(cf1, ax=ax[1])
    # ax[1].set_title("GP prediction")

    # levels = np.logspace(np.log10(abs_err_clip.min()), 1, 100)
    # cf2 = ax[2].contourf(X1, X2, abs_err_clip,
    #                      levels=levels, norm=LogNorm(),
    #                      cmap="magma_r")
    # fig.colorbar(cf2, ax=ax[2], format="%.1e")
    # ax[2].set_title("Absolute error (log scale)")

    # for a in ax:
    #     a.set_xlabel("$x_1$")
    #     a.set_ylabel("$x_2$")
    #     a.set_aspect("equal")

    # plt.suptitle("DD-GP on Ishigami slice  (x₃ = 0, direction = e₁)")
    # plt.tight_layout()
    # plt.show()

    # quick scalar NRMSE on function (optional)
    y_pred_f = f_gp
    y_true_f = true_function(X_pred).ravel()
    print("NRMSE(f):", utils.nrmse(y_true_f, y_pred_f))

    # ------------------------------------------------------------
    # 2-D slice – GP vs True vs log-|error|   (no arrows)
    # ------------------------------------------------------------

    # ---- absolute error (log-scale) -----------------------------
    gp_map = f_gp.reshape(X1.shape)
    true_map = y_true_f.reshape(X1.shape)
    abs_err = np.abs(gp_map - true_map)
    eps = 1e-8
    abs_clip = np.clip(abs_err, eps, None)
    err_min, err_max = abs_clip.min(), abs_clip.max()
    log_lvls = np.logspace(np.log10(err_min), 1, 200)

    # ---- figure --------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # (A) GP prediction
    cf1 = axs[0].contourf(X1, X2, gp_map, levels=30, cmap='viridis')
    # --- show only training points with |x1| < tol -----------------
    tol = 0.5                          # half-width of the vertical band
    mask = np.abs(X_train[:, -1] - np.pi) < tol
    axs[0].scatter(X_train[mask, 0], X_train[mask, 1],
                   c='red', s=40, label='Train pts (|x3|<0.5)')
    axs[0].set_title("GP prediction")
    axs[0].legend()
    fig.colorbar(cf1, ax=axs[0])

    # (B) True function
    cf2 = axs[1].contourf(X1, X2, true_map, levels=30, cmap='viridis')
    axs[1].set_title("True function")
    fig.colorbar(cf2, ax=axs[1])

    # (C) log-abs error
    cf3 = axs[2].contourf(X1, X2, abs_clip,
                          levels=log_lvls,
                          norm=LogNorm(vmin=err_min, vmax=err_max),
                          cmap="magma_r")
    fig.colorbar(cf3, ax=axs[2], format="%.1e")
    axs[2].set_title("Absolute error (log scale)")

    # consistent axes
    for ax in axs:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")
        ax.set_xlim(gx[0], gx[-1])
        ax.set_ylim(gy[0], gy[-1])

    plt.show()


if __name__ == "__main__":
    main()
