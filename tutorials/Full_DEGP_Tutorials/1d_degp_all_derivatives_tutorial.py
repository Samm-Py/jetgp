import numpy as np
import matplotlib.pyplot as plt
import pyoti.sparse as oti
from full_degp.degp import degp
import utils

plt.rcParams.update({'font.size': 12})

def true_function(X, alg=oti):
    x = X[:, 0]
    return alg.exp(-x) + alg.sin(2*x) + alg.cos(3 * x) + 0.2 * x + 1.0

lb_x, ub_x = 0.2, 5.0
num_points = 3
X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)
X_test = np.linspace(lb_x, ub_x, 100).reshape(-1, 1)
y_true = true_function(X_test, alg=np)

orders = [0, 1, 2, 4]
titles = [
    r"Order 0: $f(x)$",
    r"Order 1: $f(x)$, $f'(x)$",
    r"Order 2: $f(x)$, $f'(x)$, $f''(x)$",
    r"Order 4: $f(x)$, ..., $f^{(4)}(x)$"
]

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axs = axs.flatten()

for idx, n_order in enumerate(orders):
    n_bases = 1
    der_indices = utils.gen_OTI_indices(n_bases, n_order)
    # Hypercomplex perturb
    X_train_pert = oti.array(X_train)
    for i in range(1, n_bases + 1):
        X_train_pert[:, i - 1] = X_train_pert[:, i - 1] + oti.e(i, order=n_order)

    y_train_hc = true_function(X_train_pert)
    y_train = [y_train_hc.real]
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            y_train.append(y_train_hc.get_deriv(der_indices[i][j]).reshape(-1, 1))

    gp = degp(
        X_train, y_train, n_order, n_bases, der_indices,
        normalize=False, kernel="SE", kernel_type="anisotropic"
    )
    params = gp.optimize_hyperparameters(n_restart_optimizer=10, swarm_size=100)
    y_pred, y_var = gp.predict(X_test, params, calc_cov=True, return_deriv=False)

    ax = axs[idx]
    l0, = ax.plot(X_test, y_true, 'k-', lw=2, label="True $f(x)$")
    l1, = ax.plot(X_test, y_pred, 'b--', lw=2, label="GP mean")
    l2 = ax.fill_between(
        X_test.ravel(),
        y_pred.ravel() - 2*np.sqrt(y_var.ravel()),
        y_pred.ravel() + 2*np.sqrt(y_var.ravel()),
        color='blue', alpha=0.15, label='GP 95% CI'
    )
    l3 = ax.scatter(X_train, y_train[0], c='red', s=40, zorder=5, label="Train pts")
    ax.set_title(titles[idx])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")

# Call tight_layout BEFORE adding legend!
plt.tight_layout(rect=[0, 0.11, 1, 1])
# Extra space for bottom legend
plt.subplots_adjust(bottom=0.17)

# To avoid missing handles, collect from all axes
handles, labels = [], []
for ax in axs:
    h, l = ax.get_legend_handles_labels()
    for handle, label in zip(h, l):
        if label not in labels:
            handles.append(handle)
            labels.append(label)

fig.legend(
    handles, labels,
    loc='lower center', bbox_to_anchor=(0.5, 0.02),
    ncol=4, frameon=False, fontsize=12
)

plt.savefig("derivative_gp_comparison.png", dpi=300)
plt.show()
