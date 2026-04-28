"""Plotting utilities for the 4D orthogonal direction demonstration."""

import matplotlib.pyplot as plt
import numpy as np


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


def annotate_heatmap(ax, data, fmt=".2f", threshold=0.5):
    """Write values into heatmap cells."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if abs(data[i, j]) > threshold else "black"
            ax.text(j, i, format(data[i, j], fmt),
                    ha="center", va="center", color=color, fontsize=11)


def save_direction_component_heatmap(eigenvectors, figure_dir):
    """Visualize each 4D eigen-direction by coordinate components."""
    data = eigenvectors.T
    d = data.shape[1]

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    im = ax.imshow(data, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Direction component")
    ax.set_xticks(np.arange(d))
    ax.set_yticks(np.arange(d))
    ax.set_xticklabels([rf"$x_{i+1}$" for i in range(d)])
    ax.set_yticklabels([rf"$q_{i+1}$" for i in range(d)])
    ax.set_xlabel("Coordinate")
    ax.set_ylabel("Eigen-direction")
    ax.set_title("Components of selected eigen-directions")
    annotate_heatmap(ax, data, fmt=".2f", threshold=0.55)

    fig.tight_layout()
    fig.savefig(figure_dir / "example_2_direction_components.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_optimizer_direction_component_heatmap(optimizer_directions, figure_dir):
    """Visualize each optimized 4D direction by coordinate components."""
    data = np.column_stack(optimizer_directions).T
    d = data.shape[1]

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    im = ax.imshow(data, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Direction component")
    ax.set_xticks(np.arange(d))
    ax.set_yticks(np.arange(d))
    ax.set_xticklabels([rf"$x_{i+1}$" for i in range(d)])
    ax.set_yticklabels([rf"$v_{i+1}^{{opt}}$" for i in range(d)])
    ax.set_xlabel("Coordinate")
    ax.set_ylabel("Optimizer direction")
    ax.set_title("Components of optimized directions")
    annotate_heatmap(ax, data, fmt=".2f", threshold=0.55)

    fig.tight_layout()
    fig.savefig(figure_dir / "example_2_optimizer_direction_components.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_optimizer_alignment_heatmap(eigenvectors, optimizer_directions, figure_dir):
    """Show that optimized directions align with covariance eigenvectors."""
    eig = eigenvectors
    opt = np.column_stack(optimizer_directions)
    data = np.abs(eig.T @ opt)
    d = data.shape[0]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$|\langle q_i, v_j^{\mathrm{opt}}\rangle|$")
    ax.set_xticks(np.arange(d))
    ax.set_yticks(np.arange(d))
    ax.set_xticklabels([rf"$v_{j+1}^{{opt}}$" for j in range(d)])
    ax.set_yticklabels([rf"$q_{i+1}$" for i in range(d)])
    ax.set_xlabel("Optimizer direction")
    ax.set_ylabel("Eigen-direction")
    ax.set_title("Optimizer directions recover the eigenbasis")
    annotate_heatmap(ax, data, fmt=".3f", threshold=0.55)

    fig.tight_layout()
    fig.savefig(figure_dir / "example_2_optimizer_alignment.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_direction_validation_figures(eigenvalues, eigenvectors,
                                      optimizer_directions, figure_dir):
    """Save all figures for the 4D direction validation example."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_direction_component_heatmap(eigenvectors, figure_dir)
    save_optimizer_direction_component_heatmap(optimizer_directions, figure_dir)
    save_optimizer_alignment_heatmap(eigenvectors, optimizer_directions, figure_dir)
