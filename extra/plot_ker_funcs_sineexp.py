import numpy as np
import pandas as pd
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pyoti.sparse as oti  # make sure pyoti is installed in your environment

# ------------------------------------------------------------------------------------
# Symbolic Derivatives of the Periodic Kernel
# ------------------------------------------------------------------------------------
r, ell, p, sigma_f = sp.symbols("r ell p sigma_f", real=True, positive=True)
cos_term = sp.cos(2 * sp.pi * r / p)
k_per = sigma_f**2 * sp.exp((cos_term - 1) * ell**2)

per_derivatives = [k_per]
for _ in range(5):
    per_derivatives.append(sp.diff(per_derivatives[-1], r))

# Convert to numpy-callable functions
per_deriv_funcs = [
    sp.lambdify((r, ell, p, sigma_f), expr, modules="numpy")
    for expr in per_derivatives
]


def periodic_kernel_derivative(x, ell, p, sigma_f, order):
    if order > 5:
        raise ValueError("Only derivatives up to order 5 are supported.")
    return per_deriv_funcs[order](x, ell, p, sigma_f)


# ------------------------------------------------------------------------------------
# Hypercomplex Automatic Differentiation for Periodic Kernel
# ------------------------------------------------------------------------------------


def sine_exp_kernel_anisotropic(X1, X2, length_scales, n_order, index=-1):
    X1 = oti.array(X1)
    X2 = oti.array(X2)

    n1, d = X1.shape

    n2, d = X2.shape

    ell = np.exp(length_scales[0])
    p = length_scales[1]
    sigma_f = length_scales[-1]

    # Prepare the output: a list of d arrays, each of shape (n, m)
    differences_by_dim = []

    # Loop over each dimension k
    for k in range(d):
        # Create an empty (n, m) array for this dimension
        diffs_k = oti.zeros((n1, n2))

        # Nested loops to fill diffs_k
        for i in range(n1):
            for j in range(n2):
                diffs_k[i, j] = (
                    X1[i, k]
                    + oti.e(2 * k + 2, order=2 * n_order)
                    - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                )

        # Append to our list
        differences_by_dim.append(diffs_k)

    # Distances scaled by each dimension's length scale

    sqdist = np.sum(
        (ell * oti.sin(np.pi / p * differences_by_dim[i])) ** 2
        for i in range(d)
    )
    return differences_by_dim, sigma_f**2 * oti.exp(-2 * sqdist)


# ------------------------------------------------------------------------------------
# Setup for Comparison and Plotting
# ------------------------------------------------------------------------------------

length_scales = [
    [np.log(0.1), 2.0, 1.0],
    [np.log(0.5), 2.0, 1.0],
    [np.log(1.0), 2.0, 1.0],
]
params = [
    [0.1, 2.0, 1.0],
    [0.5, 2.0, 1.0],
    [1.0, 2.0, 1.0],
]  # [ell, p, sigma_f]

X1 = np.array([[0]])
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

nrmse_results = []
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

for i, ax in enumerate(axes.flat):
    for j, length_scale in enumerate(length_scales):
        diff_by_dim, vals = sine_exp_kernel_anisotropic(
            X1, X2, length_scale, i + 2
        )
        diff = diff_by_dim[0].real

        ad_val = vals.real if i == 0 else vals.get_deriv([[2, i]])

        ell, p_val, sigma = params[j]
        analytic_val = periodic_kernel_derivative(diff, ell, p_val, sigma, i)

        # NRMSE
        def compute_nrmse(y_true, y_pred):
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            norm = np.max(y_true) - np.min(y_true)
            return rmse / norm if norm != 0 else 0.0

        nrmse = compute_nrmse(analytic_val.flatten(), ad_val.flatten())
        nrmse_results.append(
            {
                "Derivative Order": i,
                "Theta": f"{params[j][0]:.2f}",
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
            zorder=100,
        )
        ax.plot(diff.flatten(), analytic_val.flatten(), color="red", lw=2)

    ax.set_title(titles[i])
    ax.set_xlabel(r"$x^{(i)} - x^{(j)}$")
    ax.set_ylabel("Value")
    ax.grid(True)

# Legend
theta_lines = [
    Line2D(
        [0],
        [0],
        color="black",
        linestyle=linestyles[j],
        lw=2,
        label=rf"$\theta = {params[j][0]}$",
    )
    for j in range(len(params))
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

# NRMSE Table
df_nrmse = pd.DataFrame(nrmse_results)
df_nrmse["Theta"] = df_nrmse["Theta"].apply(lambda x: f"{x}")
nrmse_table = df_nrmse.pivot(
    index="Derivative Order", columns="Theta", values="NRMSE"
)

print("\nNRMSE between AD and Analytic:")
print(nrmse_table.to_string(float_format="%.2e"))

# LaTeX Export
latex_table = nrmse_table.to_latex(
    float_format="%.2e",
    index=True,
    caption="NRMSE between analytic and hypercomplex AD derivatives for the periodic kernel.",
    label="tab:nrmse_periodic_ad_vs_analytic",
    column_format="c" * (len(nrmse_table.columns) + 1),
    header=True,
)

print("\nLaTeX Table:")
print(latex_table)
