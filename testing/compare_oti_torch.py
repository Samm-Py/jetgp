from utils import nrmse
import matplotlib.pyplot as plt
import utils
from full_degp.degp import degp
import pyoti.sparse as oti
import os
import torch
import gpytorch
from matplotlib import cm
import numpy as np


def franke(X, Y):
    term1 = .75*torch.exp(-((9*X - 2).pow(2) + (9*Y - 2).pow(2))/4)
    term2 = .75*torch.exp(-((9*X + 1).pow(2))/49 - (9*Y + 1)/10)
    term3 = .5*torch.exp(-((9*X - 7).pow(2) + (9*Y - 3).pow(2))/4)
    term4 = .2*torch.exp(-(9*X - 4).pow(2) - (9*Y - 7).pow(2))

    f = term1 + term2 + term3 - term4
    dfx = -2*(9*X - 2)*9/4 * term1 - 2*(9*X + 1)*9/49 * term2 + \
          -2*(9*X - 7)*9/4 * term3 + 2*(9*X - 4)*9 * term4
    dfy = -2*(9*Y - 2)*9/4 * term1 - 9/10 * term2 + \
          -2*(9*Y - 3)*9/4 * term3 + 2*(9*Y - 7)*9 * term4

    return f, dfx, dfy


xv, yv = torch.meshgrid(torch.linspace(0, 1, 4),
                        torch.linspace(0, 1, 4), indexing="ij")
train_x = torch.cat((
    xv.contiguous().view(xv.numel(), 1),
    yv.contiguous().view(yv.numel(), 1)),
    dim=1
)

f, dfx, dfy = franke(train_x[:, 0], train_x[:, 1])
train_y = torch.stack([f, dfx, dfy], -1).squeeze(1)


class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(
            train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    num_tasks=3)  # Value + x-derivative + y-derivative
model = GPModelWithDerivatives(train_x, train_y, likelihood)

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 100


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
# Includes GaussianLikelihood parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print("Iter %d/%d - Loss: %.3f   lengthscales: %.3f, %.3f   noise: %.3f" % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.squeeze()[0],
        model.covar_module.base_kernel.lengthscale.squeeze()[1],
        model.likelihood.noise.item()
    ))
    optimizer.step()


# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
fig, ax = plt.subplots(2, 3, figsize=(14, 10))

# Test points
n1, n2 = 50, 50
xv, yv = torch.meshgrid(torch.linspace(0, 1, n1),
                        torch.linspace(0, 1, n2), indexing="ij")
f, dfx, dfy = franke(xv, yv)

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
    test_x = torch.stack(
        [xv.reshape(n1*n2, 1), yv.reshape(n1*n2, 1)], -1).squeeze(1)
    predictions = likelihood(model(test_x))
    mean = predictions.mean


true_vals = [f, dfx, dfy]
pred_vals = [mean[:, i].detach().numpy().reshape(n1, n2) for i in range(3)]

print("NRMSE between model and true function: {:.6f}".format(
    nrmse(pred_vals[0].flatten(), f.flatten())))

titles = [
    "True values", "True x-derivatives", "True y-derivatives",
    "Predicted values", "Predicted x-derivatives", "Predicted y-derivatives"
]

for i in range(3):
    c0 = ax[0, i].contourf(xv.numpy(), yv.numpy(),
                           true_vals[i].numpy(), levels=40, cmap=cm.jet)
    fig.colorbar(c0, ax=ax[0, i])
    ax[0, i].set_title(titles[i])

    c1 = ax[1, i].contourf(xv.numpy(), yv.numpy(),
                           pred_vals[i], levels=40, cmap=cm.jet)
    fig.colorbar(c1, ax=ax[1, i])
    ax[1, i].set_title(titles[i + 3])


"""
--------------------------------------------------------------------------------
This script demonstrates a derivative-enhanced Gaussian Process (GP) model
using pyOTI-based automatic differentiation and the Franke function in 2D.
Function values and derivatives up to a specified order are computed, and
a GP model is trained on this enriched dataset. Predictions and contours
are compared against the true function.
--------------------------------------------------------------------------------
"""


# ----- Problem Configuration -----
n_order = 1      # Max derivative order to include
n_bases = 2      # Number of input dimensions
lb_x, ub_x = 0, 1
lb_y, ub_y = 0, 1

# Derivative indices: include all derivatives up to n_order
der_indices = utils.gen_OTI_indices(n_bases, n_order)

# ----- Generate Training Inputs (10x10 grid like torch version) -----
x_vals = np.linspace(lb_x, ub_x, 4)
y_vals = np.linspace(lb_y, ub_y, 4)
X1, X2 = np.meshgrid(x_vals, y_vals, indexing="ij")
X_train = np.column_stack([X1.ravel(), X2.ravel()])

# Create perturbed inputs for hypercomplex AD
X_train_pert = oti.array(X_train)
for i in range(n_bases):
    X_train_pert[:, i] += oti.e(i + 1, order=n_order)

# ----- Define Franke Function -----


def franke(X, alg=oti):
    x, y = X[:, 0], X[:, 1]
    term1 = 0.75 * alg.exp(-((9 * x - 2)**2 + (9 * y - 2)**2) / 4)
    term2 = 0.75 * alg.exp(-((9 * x + 1)**2) / 49 - (9 * y + 1) / 10)
    term3 = 0.5 * alg.exp(-((9 * x - 7)**2 + (9 * y - 3)**2) / 4)
    term4 = 0.2 * alg.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return term1 + term2 + term3 - term4


# Evaluate function and derivatives at training points
y_train_hc = franke(X_train_pert)
y_train_real = y_train_hc.real

y_train = [y_train_real]
for i in range(len(der_indices)):
    for j in range(len(der_indices[i])):
        deriv = y_train_hc.get_deriv(der_indices[i][j])
        y_train.append(deriv)

# ----- GP Model Setup -----
gp = degp(
    X_train,
    y_train,
    n_order,
    n_bases,
    der_indices,
    normalize=True,
    kernel="SE",
    kernel_type="anisotropic",
)

# ----- Hyperparameter Optimization -----
params = gp.optimize_hyperparameters(
    n_restart_optimizer=100,
    swarm_size=10*n_bases
)

# ----- Generate Test Data -----
N_grid = 50
x_lin = np.linspace(lb_x, ub_x, N_grid)
y_lin = np.linspace(lb_y, ub_y, N_grid)
X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

# ----- GP Prediction -----
y_pred = gp.predict(
    X_test,
    params,
    calc_cov=False,
    return_deriv=False
)

# ----- Evaluate True Function on Test Points -----
y_true = franke(X_test, alg=np).reshape(X1_grid.shape)
y_pred_grid = y_pred.reshape(X1_grid.shape)

# ----- Plot Results -----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot true function
c0 = axes[0].contourf(X1_grid, X2_grid, y_true, levels=40, cmap="viridis")
fig.colorbar(c0, ax=axes[0])
axes[0].set_title("True Franke Function")

# Plot predicted function
c1 = axes[1].contourf(X1_grid, X2_grid, y_pred_grid, levels=40, cmap="viridis")
fig.colorbar(c1, ax=axes[1])
axes[1].set_title("Predicted GP Mean")

plt.tight_layout()
plt.show()

# ----- Compute Error -----
print("NRMSE between model and true function: {:.6f}".format(
    nrmse(y_true.flatten(), y_pred)))
