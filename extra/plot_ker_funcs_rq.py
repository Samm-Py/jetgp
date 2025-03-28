import numpy as np
import pandas as pd
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pyoti.sparse as oti

# ------------------------------------------------------------------------------------
# RQ Kernel: Analytic Derivatives via SymPy
# ------------------------------------------------------------------------------------

# Symbolic variables
r, ell, alpha = sp.symbols("r ell alpha", real=True, positive=True)
z = 1 + r**2 / (2 * alpha * ell**2)
k_rq = z ** (-alpha)

# Derivatives up to 5th order
rq_derivatives = [k_rq]
for _ in range(5):
    rq_derivatives.append(sp.diff(rq_derivatives[-1], r))

# Lambdify (convert to NumPy functions)
rq_deriv_funcs = [
    sp.lambdify((r, ell, alpha), expr, modules="numpy")
    for expr in rq_derivatives
]


def rq_kernel_derivative(x, ell, alpha, order):
    if order > 5:
        raise ValueError("Only derivatives up to order 5 are supported.")
    return rq_deriv_funcs[order](x, ell, alpha)


# ------------------------------------------------------------------------------------
# RQ Kernel Evaluation via Hypercomplex AD
# ------------------------------------------------------------------------------------


def rq_kernel_anisotropic(X1, X2, length_scales, n_order, index=-1):
    X1 = oti.array(X1)
    X2 = oti.array(X2)
    n1, d = X1.shape
    n2, d = X2.shape

    ell = np.exp(length_scales[0:-2])
    alpha = np.exp(length_scales[-2])
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
        sqdist += (differences_by_dim[i]) ** 2

    z = 1 + sqdist / (2 * alpha * ell[0] ** 2)
    kernel_vals = sigma_f**2 * z**-alpha
    return differences_by_dim, kernel_vals


# ------------------------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------------------------


def compute_nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    norm = np.max(y_true) - np.min(y_true)
    return rmse / norm if norm != 0 else 0.0


# ------------------------------------------------------------------------------------
# Setup for Plotting and Comparison
# ------------------------------------------------------------------------------------

length_scales = [
    [np.log(0.5), np.log(1.0), 1],
    [np.log(1), np.log(1.0), 1],
    [np.log(2), np.log(1.0), 1],
]
ells = [[0.5, 1.0], [1, 1.0], [2, 1.0]]  # [ell, alpha]

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
        diff_by_dim, vals = rq_kernel_anisotropic(
            X1, X2, length_scale, i + 2, index=-1
        )
        diff = diff_by_dim[0].real

        # Hypercomplex AD
        ad_val = vals.real if i == 0 else vals.get_deriv([[2, i]])

        # Analytic
        x = diff
        ell, alpha_val = ells[j]
        analytic_val = rq_kernel_derivative(x, ell, alpha_val, i)

        # NRMSE
        nrmse = compute_nrmse(analytic_val.flatten(), ad_val.flatten())
        nrmse_results.append(
            {
                "Derivative Order": i,
                "Theta": f"{ells[j][0]:.2f}",
                "NRMSE": nrmse,
            }
        )

        # Plot
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

# ------------------------------------------------------------------------------------
# Global Legend
# ------------------------------------------------------------------------------------

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
    bbox_to_anchor=(1.0, 0.95),
)
plt.tight_layout(rect=[0, 0, 0.82, 1.0])
plt.show()

# ------------------------------------------------------------------------------------
# NRMSE Table Output
# ------------------------------------------------------------------------------------

df_nrmse = pd.DataFrame(nrmse_results)

# Format theta values to strings for clearer display
df_nrmse["Theta"] = df_nrmse["Theta"].apply(lambda x: f"{x}")

# Pivot for clean table format
nrmse_table = df_nrmse.pivot(
    index="Derivative Order", columns="Theta", values="NRMSE"
)

print("\nNormalized Root Mean Squared Error (NRMSE) between AD and Analytic:")
print(nrmse_table.to_string(float_format="%.2e"))

# LaTeX Export
latex_table = nrmse_table.to_latex(
    float_format="%.2e",
    index=True,
    caption="Normalized Root Mean Squared Error (NRMSE) between analytic derivatives and hypercomplex AD for the Rational Quadratic kernel.",
    label="tab:nrmse_rq_ad_vs_analytic",
    column_format="c" * (len(nrmse_table.columns) + 1),
    header=True,
)

print("\nLaTeX Table:")
print(latex_table)
