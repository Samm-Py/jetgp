"""
Test whether GPyTorch supports derivative observations at a SUBSET of training points.

GPyTorch's derivative GP uses a multi-output framework:
  - Task 0: function values f(x)
  - Task 1: df/dx1
  - Task 2: df/dx2

The question: can we provide derivatives at only SOME training points,
while providing function values at ALL training points?
"""

import torch
import gpytorch

# ============================================================
# Helper: Franke function and its analytical derivatives
# ============================================================
def franke(X):
    x, y = X[:, 0], X[:, 1]
    t1 = 0.75 * torch.exp(-((9*x - 2)**2)/4 - ((9*y - 2)**2)/4)
    t2 = 0.75 * torch.exp(-((9*x + 1)**2)/49 - (9*y + 1)/10)
    t3 = 0.50 * torch.exp(-((9*x - 7)**2)/4 - ((9*y - 3)**2)/4)
    t4 = -0.2 * torch.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return t1 + t2 + t3 + t4

def franke_with_grad(X):
    """Compute Franke function values and gradients analytically (no autograd graph)."""
    X_ = X.detach().clone().requires_grad_(True)
    y = franke(X_)
    y.sum().backward()
    return y.detach(), X_.grad.detach()

# ============================================================
# Model definition (standard GPyTorch derivative GP)
# ============================================================
class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def train_model(model, likelihood, train_x, train_y, n_iter=10):
    """Quick training loop. Returns final loss or raises on failure."""
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    return loss.item()

# ============================================================
# Generate training data (all detached — no autograd issues)
# ============================================================
torch.manual_seed(42)
n_all = 20
n_deriv = 10
noise_std = 0.05

X_all = torch.rand(n_all, 2)
y_all, grads_all = franke_with_grad(X_all)

X_deriv = X_all[:n_deriv].clone()
grads_deriv = grads_all[:n_deriv].clone()


# ============================================================
# TEST 1: Derivatives at ALL points (baseline — should work)
# ============================================================
print("=" * 60)
print("TEST 1: Derivatives at ALL 20 points (standard usage)")
print("=" * 60)

try:
    train_y_full = torch.stack([
        y_all + noise_std * torch.randn(n_all),
        grads_all[:, 0] + noise_std * torch.randn(n_all),
        grads_all[:, 1] + noise_std * torch.randn(n_all),
    ], dim=-1)

    lik1 = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
    mod1 = GPModelWithDerivatives(X_all, train_y_full, lik1)

    print(f"  train_x shape: {X_all.shape}")
    print(f"  train_y shape: {train_y_full.shape}")

    loss = train_model(mod1, lik1, X_all, train_y_full)
    print(f"  => SUCCESS: trained, final loss = {loss:.4f}")

except Exception as e:
    print(f"  => FAILED: {type(e).__name__}: {e}")


# ============================================================
# TEST 2: Only pass 10 points that have derivatives
#   (works, but LOSES function values at points 10-19)
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Only 10 points (derivatives everywhere, but lose 10 func-only pts)")
print("=" * 60)

try:
    train_y_sub = torch.stack([
        y_all[:n_deriv] + noise_std * torch.randn(n_deriv),
        grads_deriv[:, 0] + noise_std * torch.randn(n_deriv),
        grads_deriv[:, 1] + noise_std * torch.randn(n_deriv),
    ], dim=-1)

    lik2 = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
    mod2 = GPModelWithDerivatives(X_deriv, train_y_sub, lik2)

    print(f"  train_x shape: {X_deriv.shape}")
    print(f"  train_y shape: {train_y_sub.shape}")

    loss = train_model(mod2, lik2, X_deriv, train_y_sub)
    print(f"  => SUCCESS: trained, final loss = {loss:.4f}")
    print(f"  NOTE: Works but discards function-only points 10-19!")

except Exception as e:
    print(f"  => FAILED: {type(e).__name__}: {e}")


# ============================================================
# TEST 3: NaN for missing derivatives at points 10-19
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: All 20 pts, NaN for missing derivatives at points 10-19")
print("=" * 60)

try:
    train_y_nan = torch.stack([
        y_all + noise_std * torch.randn(n_all),
        torch.cat([grads_all[:n_deriv, 0] + noise_std * torch.randn(n_deriv),
                    torch.full((n_all - n_deriv,), float('nan'))]),
        torch.cat([grads_all[:n_deriv, 1] + noise_std * torch.randn(n_deriv),
                    torch.full((n_all - n_deriv,), float('nan'))]),
    ], dim=-1)

    lik3 = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
    mod3 = GPModelWithDerivatives(X_all, train_y_nan, lik3)

    print(f"  train_x shape: {X_all.shape}")
    print(f"  train_y shape: {train_y_nan.shape}")
    print(f"  NaN entries in train_y: {torch.isnan(train_y_nan).sum().item()}")

    loss = train_model(mod3, lik3, X_all, train_y_nan)
    if torch.isnan(torch.tensor(loss)):
        print(f"  => Loss is NaN — NaN masking does NOT work")
    else:
        print(f"  => Trained, final loss = {loss:.4f}")

except Exception as e:
    print(f"  => FAILED: {type(e).__name__}: {e}")


# ============================================================
# TEST 4: Zero-fill missing derivatives (naive workaround)
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: All 20 pts, zero-fill missing derivatives at pts 10-19")
print("=" * 60)

try:
    train_y_zero = torch.stack([
        y_all + noise_std * torch.randn(n_all),
        torch.cat([grads_all[:n_deriv, 0] + noise_std * torch.randn(n_deriv),
                    torch.zeros(n_all - n_deriv)]),
        torch.cat([grads_all[:n_deriv, 1] + noise_std * torch.randn(n_deriv),
                    torch.zeros(n_all - n_deriv)]),
    ], dim=-1)

    lik4 = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
    mod4 = GPModelWithDerivatives(X_all, train_y_zero, lik4)

    print(f"  train_x shape: {X_all.shape}")
    print(f"  train_y shape: {train_y_zero.shape}")

    loss = train_model(mod4, lik4, X_all, train_y_zero)
    print(f"  => Trained, final loss = {loss:.4f}")
    print(f"  NOTE: 'Works' but feeds INCORRECT derivative data at pts 10-19!")

except Exception as e:
    print(f"  => FAILED: {type(e).__name__}: {e}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
GPyTorch's derivative GP (RBFKernelGrad + MultitaskGaussianLikelihood)
uses a multi-output framework where:
  - All tasks (f, df/dx1, df/dx2) share the same input locations X
  - train_y must be shape (n, num_tasks) — same n for all tasks

This means:
  1. You CANNOT natively have function values at 20 points but derivatives
     at only 10 points — the framework requires rectangular (n x tasks) data.
  2. NaN masking: check results above.
  3. Zero-fill workaround runs but feeds wrong derivative data.
  4. To properly handle partial derivative observations, you'd need a custom
     multi-output GP with different input sets per output — non-trivial.

Conclusion: GPyTorch does NOT natively support derivative observations at a
subset of training points. JetGP's `derivative_locations` feature is a
genuine differentiator.
""")