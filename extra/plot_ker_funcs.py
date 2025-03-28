import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize
import pyoti.sparse as oti
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import pandas as pd


def compute_nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    norm = np.max(y_true) - np.min(y_true)
    return rmse / norm if norm != 0 else 0.0


def se_kernel(x, ell):
    return np.exp(-0.5 * (ell * x) ** 2)


def se_kernel_derivative(x, ell, order):
    """Analytic derivatives of SE kernel up to 5th order"""
    k = se_kernel(x, ell)
    l2 = ell**2
    l4 = l2**2
    l6 = l2**3
    l8 = l4**2
    l10 = l4 * l6

    if order == 0:
        return k
    elif order == 1:
        return -x * l2 * k
    elif order == 2:
        return (x**2 * l4 - l2) * k
    elif order == 3:
        return (-(x**3) * l6 + 3 * x * l4) * k
    elif order == 4:
        return (x**4 * l8 - 6 * x**2 * l6 + 3 * l4) * k
    elif order == 5:
        return (-(x**5) * l10 + 10 * x**3 * l8 - 15 * x * l6) * k
    else:
        raise ValueError("Only orders 0 through 5 supported.")


def se_kernel_anisotropic(X1, X2, length_scales, n_order, index=-1):
    X1 = oti.array(X1)
    X2 = oti.array(X2)
    n1, d = X1.shape
    n2, d = X2.shape

    ell = np.exp(length_scales[0:-1])
    sigma_f = length_scales[-1]

    differences_by_dim = []
    for k in range(d):
        diffs_k = oti.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                diffs_k[i, j] = (
                    X1[i, k]
                    + oti.e(2 * k + 2, order=2 * n_order)
                    - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                )
        differences_by_dim.append(diffs_k)

    sqdist = 0
    for i in range(d):
        sqdist = sqdist + (ell[i] * (differences_by_dim[i])) ** 2

    kernel_vals = sigma_f**2 * oti.exp(-0.5 * sqdist)
    return differences_by_dim, kernel_vals


# --- Plotting ---

length_scales = [[np.log(0.1), 1], [np.log(0.5), 1], [np.log(1), 1]]
ells = [[0.1, 1], [0.5, 1], [1, 1]]
X1 = np.array([0]).reshape(-1, 1)
X2 = np.linspace(-3, 3, 1000).reshape(-1, 1)
linestyles = ["--", "-.", ":"]
titles = [
    "Kernel Value",
    "1st Derivative",
    "2nd Derivative",
    "3rd Derivative",
    "4th Derivative",
    "5th Derivative",
]
plt.rcParams.update({"font.size": 12})
nrmse_results = []
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

for i, ax in enumerate(axes.flat):
    for j, length_scale in enumerate(length_scales):
        diff_by_dim, vals = se_kernel_anisotropic(
            X1, X2, length_scale, i + 2, index=-1
        )
        diff = diff_by_dim[0].real

        # Get AD value
        if i == 0:
            ad_val = vals.real
        else:
            ad_val = vals.get_deriv([[2, i]])

        # Get analytic value
        x = diff
        ell = ells[j][0]
        analytic_val = se_kernel_derivative(x, ell, i)

        # Compute NRMSE
        nrmse = compute_nrmse(analytic_val.flatten(), ad_val.flatten())
        nrmse_results.append(
            {"Derivative Order": i, "Theta": ells[j][0], "NRMSE": nrmse}
        )

        # Plot both
        ax.plot(
            diff.flatten(),
            ad_val.flatten(),
            color="black",
            linestyle=linestyles[j],
            lw=2,
            zorder=10,
        )
        ax.plot(x.flatten(), analytic_val.flatten(), color="red", lw=2)

    ax.set_title(titles[i])
    ax.set_xlabel(r"$x^{(i)} - x^{(j)}$")
    ax.set_ylabel("Value")
    ax.grid(True)

# Global legend
theta_lines = [
    Line2D(
        [0],
        [0],
        color="black",
        linestyle=linestyles[j],
        lw=2,
        label=rf"$\theta = {ells[j][0]}$",
    )
    for j in range(len(ells))
]
custom_lines = [
    Line2D([0], [0], color="black", lw=2, label="Hypercomplex AD"),
    Line2D([0], [0], color="red", lw=2, label="Analytic derivative"),
]


fig.legend(
    handles=theta_lines + custom_lines,
    loc="upper right",
    bbox_to_anchor=(1.00, 0.95),
)

plt.tight_layout(rect=[0, 0, 0.82, 1.0])
plt.show()


df_nrmse = pd.DataFrame(nrmse_results)

# Format theta values to strings for clearer display
df_nrmse["Theta"] = df_nrmse["Theta"].apply(lambda x: f"{x:.2f}")

# Pivot for clean table format: Derivative Order as rows, Theta as columns
nrmse_table = df_nrmse.pivot(
    index="Derivative Order", columns="Theta", values="NRMSE"
)

# Print the table
print("\nNormalized Root Mean Squared Error (NRMSE) between AD and Analytic:")
print(nrmse_table.to_string(float_format="%.2e"))


# Convert NRMSE table to LaTeX format
latex_table = nrmse_table.to_latex(
    float_format="%.2e",
    index=True,
    caption="Normalized Root Mean Squared Error (NRMSE) between analytic derivatives and hypercomplex AD.",
    label="tab:nrmse_ad_vs_analytic",
    column_format="c" * (len(nrmse_table.columns) + 1),
    header=True,
)

# Print the LaTeX table
print("\nLaTeX Table:")
print(latex_table)
