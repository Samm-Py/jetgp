"""Plotting utilities for the Branin-Hoo adaptive DOE example."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, MaxNLocator

from adaptive_doe import query_function_posterior


DIRECTION_COLORS = ["#ff4fa3", "#00a6d6", "#58a55c"]
INITIAL_ENRICHMENT_GREY = "#ffffff"
PREVIOUS_INFILl_GREY = "#7f7f7f"


def configure_plotting():
    """Set presentation-friendly Matplotlib defaults."""
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        "mathtext.fontset": "cm",
        "font.family": "serif",
    })


def make_grid(bounds, resolution):
    lb, ub = bounds[:, 0], bounds[:, 1]
    x1g = np.linspace(lb[0], ub[0], resolution)
    x2g = np.linspace(lb[1], ub[1], resolution)
    X1, X2 = np.meshgrid(x1g, x2g)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    return X1, X2, X_grid


def query_function_posterior_batched(gp_model, params, X_grid, batch_size=250):
    """Evaluate GP function posterior on a grid without one huge covariance call."""
    means = []
    variances = []
    for start in range(0, X_grid.shape[0], batch_size):
        stop = min(start + batch_size, X_grid.shape[0])
        mean_batch, var_batch = query_function_posterior(
            gp_model, params, X_grid[start:stop])
        means.append(mean_batch)
        variances.append(var_batch)
    return np.concatenate(means), np.concatenate(variances)


def positive_log_limits(arrays):
    """Return stable positive limits for log-scaled error plots."""
    positive_values = np.concatenate([
        np.asarray(array)[np.asarray(array) > 0.0].ravel()
        for array in arrays
    ])
    if positive_values.size == 0:
        return 1e-12, 1.0
    vmin = max(float(np.percentile(positive_values, 1)), 1e-12)
    vmax = max(float(np.max(positive_values)), vmin * 10.0)
    return vmin, vmax


def apply_log_floor(error, vmin):
    """Avoid zeros in arrays shown with LogNorm."""
    return np.maximum(error, vmin)


def add_design_points(ax, X_train, n_init):
    """Overlay initial DOE and previously selected infill points."""
    ax.scatter(X_train[:n_init, 0], X_train[:n_init, 1],
               c="white", edgecolor="black", s=38, linewidth=0.8,
               label="Initial DOE", zorder=5)
    if X_train.shape[0] > n_init:
        ax.scatter(X_train[n_init:, 0], X_train[n_init:, 1],
                   c="#d9d9d9", edgecolor="black", s=44,
                   label="Previous infill", zorder=6)


def add_current_point(ax, rec):
    """Overlay the current maximum-variance point."""
    x_new = rec["x_new"]
    ax.scatter(x_new[0], x_new[1], marker="*", s=320,
               c="#ff4fa3", edgecolor="black", linewidth=0.8,
               label=r"$x_{\mathrm{new}}$", zorder=8, clip_on=False)


def add_current_directions(ax, rec, arrow_scale=1.4):
    """Overlay the directions selected at the current infill point."""
    x_new = rec["x_new"]
    for idx, direction in enumerate(rec["selected_directions"], start=1):
        v = np.asarray(direction)
        color = DIRECTION_COLORS[(idx - 1) % len(DIRECTION_COLORS)]
        ax.arrow(x_new[0], x_new[1],
                 arrow_scale * v[0], arrow_scale * v[1],
                 width=0.035, head_width=0.28, head_length=0.35,
                 length_includes_head=True, facecolor=color,
                 edgecolor="black", linewidth=0.6, zorder=9,
                 label=rf"$v_{idx}$",
                 clip_on=False)


def add_direction_history(ax, history_subset, arrow_scale=1.1, grey=False):
    """Overlay all selected directional observations in a history prefix."""
    used_labels = set()
    for rec in history_subset:
        x_new = rec["x_new"]
        for idx, direction in enumerate(rec["selected_directions"], start=1):
            v = np.asarray(direction)
            if grey:
                color = PREVIOUS_INFILl_GREY
                label = "Previous direction" if "previous" not in used_labels else "_nolegend_"
                used_labels.add("previous")
            else:
                color = DIRECTION_COLORS[(idx - 1) % len(DIRECTION_COLORS)]
                label = rf"$v_{idx}$" if idx not in used_labels else "_nolegend_"
                used_labels.add(idx)
            ax.arrow(x_new[0], x_new[1],
                     arrow_scale * v[0], arrow_scale * v[1],
                     width=0.028, head_width=0.22, head_length=0.28,
                     length_includes_head=True, facecolor=color,
                     edgecolor="black", linewidth=0.5, zorder=8,
                     label=label, clip_on=False)


def add_initial_enrichment_directions(ax, initial_derivative_history, arrow_scale=1.1):
    """Overlay enrichment directions selected on the initial DOE."""
    used_labels = set()
    for rec in initial_derivative_history:
        x_point = rec["x_point"]
        for idx, direction in enumerate(rec["selected_directions"], start=1):
            v = np.asarray(direction)
            color = DIRECTION_COLORS[(idx - 1) % len(DIRECTION_COLORS)]
            label = rf"$v_{idx}$" if idx not in used_labels else "_nolegend_"
            used_labels.add(idx)
            ax.arrow(x_point[0], x_point[1],
                     arrow_scale * v[0], arrow_scale * v[1],
                     width=0.028, head_width=0.22, head_length=0.28,
                     length_includes_head=True, facecolor=color,
                     edgecolor="black", linewidth=0.5, zorder=8,
                     label=label, clip_on=False)


def add_initial_enrichment_history(ax, initial_derivative_history, arrow_scale=1.1,
                                   grey=False):
    """Overlay all initial-enrichment directions, optionally greyed out."""
    used_labels = set()
    for rec in initial_derivative_history:
        x_point = rec["x_point"]
        for idx, direction in enumerate(rec["selected_directions"], start=1):
            v = np.asarray(direction)
            if grey:
                color = INITIAL_ENRICHMENT_GREY
                label = ("Initial enrichment direction"
                         if "initial" not in used_labels else "_nolegend_")
                used_labels.add("initial")
            else:
                color = DIRECTION_COLORS[(idx - 1) % len(DIRECTION_COLORS)]
                label = rf"$v_{idx}$" if idx not in used_labels else "_nolegend_"
                used_labels.add(idx)
            ax.arrow(x_point[0], x_point[1],
                     arrow_scale * v[0], arrow_scale * v[1],
                     width=0.028, head_width=0.22, head_length=0.28,
                     length_includes_head=True, facecolor=color,
                     edgecolor="black", linewidth=0.5, zorder=7,
                     label=label, clip_on=False)


def save_initial_enrichment_figure(al, figure_dir, resolution=40):
    """Save GP mean/variance with initial enrichment directions over the DOE."""
    if not getattr(al, "initial_derivative_history", None):
        return
    if getattr(al, "initial_function_gp_model", None) is None:
        return

    X1, X2, X_grid = make_grid(al.bounds, resolution)
    mean_f, var_f = query_function_posterior_batched(
        al.initial_function_gp_model,
        al.initial_function_params,
        X_grid,
    )
    gp_mean = mean_f.reshape(resolution, resolution)
    f_var = var_f.reshape(resolution, resolution)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), sharex=True, sharey=True)
    panels = [
        (axes[0], gp_mean, r"GP mean $\mu_f(x)$", r"$\mu_f(x)$"),
        (axes[1], f_var, r"Predictive variance $\sigma_f^2(x)$", r"$\sigma_f^2(x)$"),
    ]

    for ax, Z, panel_title, cbar_label in panels:
        cp = ax.contourf(X1, X2, Z, levels=40, cmap="viridis")
        cbar = fig.colorbar(cp, ax=ax)
        cbar.set_label(cbar_label)
        cbar.locator = MaxNLocator(nbins=7)
        cbar.update_ticks()

        ax.scatter(al.X_train[:al.n_init, 0], al.X_train[:al.n_init, 1],
                   c="white", edgecolor="black", s=38, linewidth=0.8,
                   label="Initial DOE", zorder=5)
        add_initial_enrichment_directions(ax, al.initial_derivative_history)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_title(panel_title)
        ax.set_xlim(al.bounds[0])
        ax.set_ylim(al.bounds[1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Initial DOE enrichment: selected directional derivatives")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(figure_dir / "initial_enrichment_directions.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_initial_doe_figure(al, figure_dir, resolution=40):
    """Save GP mean/variance after the initial DOE, without direction overlays."""
    if getattr(al, "initial_function_gp_model", None) is None:
        return

    X1, X2, X_grid = make_grid(al.bounds, resolution)
    mean_f, var_f = query_function_posterior_batched(
        al.initial_function_gp_model,
        al.initial_function_params,
        X_grid,
    )
    gp_mean = mean_f.reshape(resolution, resolution)
    f_var = var_f.reshape(resolution, resolution)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), sharex=True, sharey=True)
    panels = [
        (axes[0], gp_mean, r"GP mean $\mu_f(x)$", r"$\mu_f(x)$"),
        (axes[1], f_var, r"Predictive variance $\sigma_f^2(x)$", r"$\sigma_f^2(x)$"),
    ]

    for ax, Z, panel_title, cbar_label in panels:
        cp = ax.contourf(X1, X2, Z, levels=40, cmap="viridis")
        cbar = fig.colorbar(cp, ax=ax)
        cbar.set_label(cbar_label)
        cbar.locator = MaxNLocator(nbins=7)
        cbar.update_ticks()

        ax.scatter(al.X_train[:al.n_init, 0], al.X_train[:al.n_init, 1],
                   c="white", edgecolor="black", s=38, linewidth=0.8,
                   label="Initial DOE", zorder=5)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_title(panel_title)
        ax.set_xlim(al.bounds[0])
        ax.set_ylim(al.bounds[1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Initial DOE: GP mean and predictive variance")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(figure_dir / "initial_doe_mean_variance.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_post_enrichment_figure(al, figure_dir, resolution=40):
    """Save GP mean/variance after the initial enrichment refit."""
    if not getattr(al, "initial_derivative_history", None):
        return
    if (getattr(al, "post_enrichment_gp_model", None) is None or
            getattr(al, "post_enrichment_params", None) is None):
        return

    X1, X2, X_grid = make_grid(al.bounds, resolution)
    mean_f, var_f = query_function_posterior_batched(
        al.post_enrichment_gp_model,
        al.post_enrichment_params,
        X_grid,
    )
    gp_mean = mean_f.reshape(resolution, resolution)
    f_var = var_f.reshape(resolution, resolution)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), sharex=True, sharey=True)
    panels = [
        (axes[0], gp_mean, r"GP mean $\mu_f(x)$", r"$\mu_f(x)$"),
        (axes[1], f_var, r"Predictive variance $\sigma_f^2(x)$", r"$\sigma_f^2(x)$"),
    ]

    for ax, Z, panel_title, cbar_label in panels:
        cp = ax.contourf(X1, X2, Z, levels=40, cmap="viridis")
        cbar = fig.colorbar(cp, ax=ax)
        cbar.set_label(cbar_label)
        cbar.locator = MaxNLocator(nbins=7)
        cbar.update_ticks()

        ax.scatter(al.X_train[:al.n_init, 0], al.X_train[:al.n_init, 1],
                   c="white", edgecolor="black", s=38, linewidth=0.8,
                   label="Initial DOE", zorder=5)
        add_initial_enrichment_directions(ax, al.initial_derivative_history)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_title(panel_title)
        ax.set_xlim(al.bounds[0])
        ax.set_ylim(al.bounds[1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Post-enrichment GP mean and predictive variance")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(figure_dir / "post_enrichment_mean_variance.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_iteration_figures(al, history, figure_dir, resolution=40):
    """Save three slide-ready GP mean/variance figures per iteration."""
    X1, X2, X_grid = make_grid(al.bounds, resolution)

    for rec in history:
        step = rec["step"]
        previous_history = history[:step - 1]
        mean_f, var_f = query_function_posterior_batched(
            rec["pre_update_gp_model"],
            rec["pre_update_params"],
            X_grid,
        )
        gp_mean = mean_f.reshape(resolution, resolution)
        f_var = var_f.reshape(resolution, resolution)

        figure_specs = [
            ("gp_mean_variance", "GP mean and predictive variance", False, False),
            ("max_variance_point", "Maximum predictive variance point", True, False),
            ("selected_direction", "Selected directional derivative(s)", True, True),
        ]

        for suffix, title, show_current_point, show_current_directions in figure_specs:
            fig, axes = plt.subplots(
                1, 2, figsize=(13.8, 5.8), sharex=True, sharey=True)
            panels = [
                (axes[0], gp_mean, r"GP mean $\mu_f(x)$", r"$\mu_f(x)$"),
                (axes[1], f_var, r"Predictive variance $\sigma_f^2(x)$",
                 r"$\sigma_f^2(x)$"),
            ]

            for ax, Z, panel_title, cbar_label in panels:
                cp = ax.contourf(X1, X2, Z, levels=40, cmap="viridis")
                cbar = fig.colorbar(cp, ax=ax)
                cbar.set_label(cbar_label)
                cbar.locator = MaxNLocator(nbins=7)
                cbar.update_ticks()

                add_design_points(ax, rec["pre_update_X_train"], al.n_init)
                if getattr(al, "initial_derivative_history", None):
                    add_initial_enrichment_history(
                        ax, al.initial_derivative_history, grey=True)
                if previous_history:
                    add_direction_history(ax, previous_history, grey=True)
                if show_current_point:
                    add_current_point(ax, rec)
                if show_current_directions:
                    add_current_directions(ax, rec)

                ax.set_xlabel(r"$x_1$")
                ax.set_ylabel(r"$x_2$")
                ax.set_title(panel_title)
                ax.set_xlim(al.bounds[0])
                ax.set_ylim(al.bounds[1])

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
            fig.suptitle(f"Iteration {step}: {title}")
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.18)
            fig.savefig(figure_dir / f"iter_{step:02d}_{suffix}.png",
                        dpi=220, bbox_inches="tight")
            plt.close(fig)


def save_eigen_spectrum_figures(history, tau, figure_dir):
    """Save one eigenvalue-ratio bar chart per iteration."""
    for rec in history:
        ratios = rec["variance_ratios"]
        indices = np.arange(1, len(ratios) + 1)
        selected = set(rec["selected_indices"])
        colors = ["#ff4fa3" if i in selected else "#4c78a8"
                  for i in range(len(ratios))]

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(indices, ratios, color=colors, edgecolor="black", linewidth=0.7)
        ax.axhline(tau, color="black", linestyle="--", linewidth=1.3,
                   label=rf"$\tau = {tau:g}$")
        ax.set_xlabel("Eigen-direction index")
        ax.set_ylabel(r"$\lambda_j / \lambda_1$")
        ax.set_title(f"Iteration {rec['step']}: derivative covariance spectrum")
        ax.set_xticks(indices)
        ax.set_ylim(bottom=0.0, top=max(1.05, float(np.nanmax(ratios)) * 1.1))
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / f"iter_{rec['step']:02d}_eigen_spectrum.png",
                    dpi=220, bbox_inches="tight")
        plt.close(fig)


def save_final_design_figure(al, history, figure_dir, true_func, resolution=200):
    """Save true function, final GP mean, and absolute error."""
    X1, X2, X_grid = make_grid(al.bounds, resolution)
    f_true = true_func(X_grid).reshape(resolution, resolution)
    gp_mean, _ = query_function_posterior_batched(al.gp_model, al.params, X_grid)
    gp_mean = gp_mean.reshape(resolution, resolution)
    abs_error = np.abs(gp_mean - f_true)
    vmin, vmax = positive_log_limits([abs_error])
    abs_error = apply_log_floor(abs_error, vmin)
    error_levels = np.geomspace(vmin, vmax, 40)

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), sharex=True, sharey=True)
    panels = [
        (axes[0], f_true, "True function", r"$f(x)$", None, None),
        (axes[1], gp_mean, "Final GP mean", r"$\mu_f(x)$", None, None),
        (axes[2], abs_error, "Absolute error", r"$|\mu_f(x)-f(x)|$",
         error_levels, LogNorm(vmin=vmin, vmax=vmax)),
    ]

    for ax, Z, title, cbar_label, levels, norm in panels:
        contour_levels = 40 if levels is None else levels
        cp = ax.contourf(X1, X2, Z, levels=contour_levels, cmap="viridis",
                         norm=norm)
        cbar = fig.colorbar(cp, ax=ax)
        cbar.set_label(cbar_label)
        if norm is None:
            cbar.locator = MaxNLocator(nbins=7)
        else:
            cbar.locator = LogLocator(base=10, numticks=8)
        cbar.update_ticks()

        ax.scatter(al.X_train[:al.n_init, 0], al.X_train[:al.n_init, 1],
                   c="white", edgecolor="black", s=38, linewidth=0.8,
                   label="Initial DOE", zorder=5)
        ax.scatter(al.X_train[al.n_init:, 0], al.X_train[al.n_init:, 1],
                   c="#ff4fa3", edgecolor="black", s=95, marker="*",
                   linewidth=0.8, label="Adaptive infill", zorder=6,
                   clip_on=False)
        if getattr(al, "initial_derivative_history", None):
            add_initial_enrichment_history(ax, al.initial_derivative_history,
                                           grey=True)
        add_direction_history(ax, history)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        f"Final adaptive design: {al.X_train.shape[0]} points, "
        f"{len(al.directional_observations)} directional observations"
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(figure_dir / "final_design.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
