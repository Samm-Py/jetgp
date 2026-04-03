"""
Benchmark: GPyTorch DEGP on the Dette-Pepelyshev function (8D)
First-order gradient-enhanced GP with RBFKernelGrad (SE equivalent), ARD.
CPU only for fair comparison with JetGP/GEKPLS.

Follows the methodology of Erickson et al. (2018) with sample sizes
n = 5d = 40 and n = 10d = 80, using 5 macroreplicates.
"""

import numpy as np
import time
import json
import sys
sys.path.insert(0, '.')

from benchmark_functions import (
    dette_pepelyshev, dette_pepelyshev_gradient,
    generate_test_data, compute_metrics
)
from scipy.stats.qmc import LatinHypercube

import torch
import gpytorch

# Force CPU
torch.set_default_dtype(torch.float64)
device = torch.device('cpu')

# ============================================================================
# Configuration
# ============================================================================
FUNCTION_NAME = "dette_pepelyshev"
DIM = 8
SAMPLE_SIZES = [5 * DIM, 10 * DIM]  # 40, 80
N_MACROREPLICATES = 5
N_TEST = 2000
N_TRAINING_ITER = 200
N_RESTARTS = 10


# ============================================================================
# GPyTorch model definition
# ============================================================================
class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=DIM)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def run_single(n_train, seed):
    """
    Run a single GPyTorch DEGP benchmark.
    """
    # Generate training data
    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = dette_pepelyshev(X_train)
    grads = dette_pepelyshev_gradient(X_train)

    # Generate test data (fixed seed for consistency across runs)
    X_test, y_test = generate_test_data(dette_pepelyshev, N_TEST, DIM, seed=99)

    # Standardize outputs
    y_mean = y_vals.mean()
    y_std = y_vals.std()
    y_vals_std = (y_vals - y_mean) / y_std
    grads_std = grads / y_std

    # Package training data: (n, 1+d) tensor
    train_y_np = np.column_stack([y_vals_std] + [grads_std[:, j] for j in range(DIM)])
    train_x = torch.tensor(X_train, dtype=torch.float64, device=device)
    train_y = torch.tensor(train_y_np, dtype=torch.float64, device=device)
    test_x = torch.tensor(X_test, dtype=torch.float64, device=device)

    # Initialize model
    t_start = time.perf_counter()

    best_loss = float('inf')
    best_state = None

    for restart in range(N_RESTARTS):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=1 + DIM
        ).to(device)
        model = GPModelWithDerivatives(train_x, train_y, likelihood).to(device)

        # Randomize initial hyperparameters for restarts > 0
        if restart > 0:
            with torch.no_grad():
                for param in model.parameters():
                    param.data = param.data + torch.randn_like(param.data) * 0.5
                for param in likelihood.parameters():
                    param.data = param.data + torch.randn_like(param.data) * 0.5

        # Training
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        final_loss = float('inf')
        for i in range(N_TRAINING_ITER):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        if final_loss < best_loss:
            best_loss = final_loss
            best_state = {
                'model': model.state_dict(),
                'likelihood': likelihood.state_dict(),
            }

    # Restore best model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=1 + DIM
    ).to(device)
    model = GPModelWithDerivatives(train_x, train_y, likelihood).to(device)
    model.load_state_dict(best_state['model'])
    likelihood.load_state_dict(best_state['likelihood'])

    t_train = time.perf_counter() - t_start

    # Prediction
    model.eval()
    likelihood.eval()

    t_pred_start = time.perf_counter()
    with torch.no_grad(), gpytorch.settings.fast_computations(
        log_prob=False, covar_root_decomposition=False
    ):
        predictions = likelihood(model(test_x))
        mean = predictions.mean

    t_pred = time.perf_counter() - t_pred_start

    # Extract function value predictions (column 0) and denormalize
    y_pred = mean[:, 0].cpu().numpy() * y_std + y_mean

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    metrics['train_time'] = t_train
    metrics['pred_time'] = t_pred
    metrics['n_train'] = n_train
    metrics['seed'] = seed

    return metrics


def main():
    results = []

    for n_train in SAMPLE_SIZES:
        print(f"\n{'='*60}")
        print(f"  GPyTorch DEGP — Dette-Pepelyshev — n_train = {n_train}")
        print(f"{'='*60}")

        for rep in range(N_MACROREPLICATES):
            seed = 1000 + rep
            print(f"\n  Macroreplicate {rep + 1}/{N_MACROREPLICATES} (seed={seed})")

            result = run_single(n_train, seed)
            result['macroreplicate'] = rep + 1
            results.append(result)

            print(f"    RMSE:       {result['rmse']:.6e}")
            print(f"    NRMSE:      {result['nrmse']:.6e}")
            print(f"    Train time: {result['train_time']:.2f}s")
            print(f"    Pred time:  {result['pred_time']:.4f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for n_train in SAMPLE_SIZES:
        subset = [r for r in results if r['n_train'] == n_train]
        rmses = [r['rmse'] for r in subset]
        times = [r['train_time'] for r in subset]
        print(f"\n  n = {n_train}:")
        print(f"    RMSE:  mean={np.mean(rmses):.6e}, std={np.std(rmses):.6e}")
        print(f"    Time:  mean={np.mean(times):.2f}s, std={np.std(times):.2f}s")

    # Save results
    output_file = f"results_gpytorch_{FUNCTION_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
