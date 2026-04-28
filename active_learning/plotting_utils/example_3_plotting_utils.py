"""Plotting utilities for the 6D Hartmann derivative-efficiency example."""

import matplotlib.pyplot as plt


def configure_plotting():
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


def save_error_vs_function_evaluations(results, figure_dir):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for label, style in [
        ("Function-only GP", {"color": "#4c78a8", "marker": "o"}),
        ("Full-gradient DEGP", {"color": "#58a55c", "marker": "s"}),
        ("Eigenbasis directional GDDEGP", {"color": "#ff4fa3", "marker": "^"}),
    ]:
        rows = [row for row in results if row["method"] == label]
        ax.plot(
            [row["n_function_evals"] for row in rows],
            [row["rmse"] for row in rows],
            label=label,
            linewidth=2.0,
            markersize=6.0,
            **style,
        )
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Surrogate accuracy vs. function evaluations")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "example_3_rmse_vs_function_evals.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_error_vs_derivative_observations(results, figure_dir):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for label, style in [
        ("Full-gradient DEGP", {"color": "#58a55c", "marker": "s"}),
        ("Eigenbasis directional GDDEGP", {"color": "#ff4fa3", "marker": "^"}),
    ]:
        rows = [row for row in results if row["method"] == label]
        ax.plot(
            [row["n_derivative_obs"] for row in rows],
            [row["rmse"] for row in rows],
            label=label,
            linewidth=2.0,
            markersize=6.0,
            **style,
        )
    ax.set_xlabel("Derivative observations acquired")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Surrogate accuracy vs. derivative information")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "example_3_rmse_vs_derivative_obs.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_max_error_vs_function_evaluations(results, figure_dir):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for label, style in [
        ("Function-only GP", {"color": "#4c78a8", "marker": "o"}),
        ("Full-gradient DEGP", {"color": "#58a55c", "marker": "s"}),
        ("Eigenbasis directional GDDEGP", {"color": "#ff4fa3", "marker": "^"}),
    ]:
        rows = [row for row in results if row["method"] == label]
        ax.plot(
            [row["n_function_evals"] for row in rows],
            [row["max_abs_error"] for row in rows],
            label=label,
            linewidth=2.0,
            markersize=6.0,
            **style,
        )
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Validation max absolute error")
    ax.set_title("Worst-case validation error")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "example_3_max_error_vs_function_evals.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_derivative_count_per_iteration(direction_counts, dim, figure_dir):
    iterations = list(range(1, len(direction_counts) + 1))
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(iterations, direction_counts, color="#ff4fa3",
           edgecolor="black", linewidth=0.7,
           label="Selected directional derivatives")
    ax.axhline(dim, color="#58a55c", linestyle="--", linewidth=2.0,
               label=f"Full gradient ({dim} derivatives)")
    ax.set_xlabel("Adaptive iteration")
    ax.set_ylabel("Derivative observations")
    ax.set_title("Derivative information acquired per infill point")
    ax.set_xticks(iterations)
    ax.set_ylim(0, max(dim, max(direction_counts, default=0)) + 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "example_3_derivatives_per_iteration.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_example_3_figures(results, direction_counts, dim, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_error_vs_function_evaluations(results, figure_dir)
    save_error_vs_derivative_observations(results, figure_dir)
    save_max_error_vs_function_evaluations(results, figure_dir)
    save_derivative_count_per_iteration(direction_counts, dim, figure_dir)
