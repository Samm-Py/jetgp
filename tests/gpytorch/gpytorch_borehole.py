"""
Benchmark: GPyTorch DEGP on the Borehole function (8D)
First-order gradient-enhanced GP with RBFKernelGrad (SE equivalent), ARD.
Adam optimizer (standard PyTorch approach), single run.
CPU only, single thread for fair comparison.
"""

import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import (
    borehole, borehole_gradient,
    generate_test_data, compute_metrics
)
from scipy.stats.qmc import LatinHypercube

import torch
import gpytorch

torch.set_default_dtype(torch.float64)
# torch.set_num_threads(1)
device = torch.device('cpu')

FUNCTION_NAME = "borehole"
DIM = 8
SAMPLE_SIZES = [DIM, 5 * DIM, 10 * DIM]
N_MACROREPLICATES = 5
N_TEST = 2000
if len(sys.argv) > 1 and sys.argv[1] == '--single':
    N_TRAINING_ITER = int(sys.argv[2])
else:
    N_TRAINING_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 200

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TESTS_DIR, 'data')


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
    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = borehole(X_train)
    grads = borehole_gradient(X_train)
    X_test, y_test = generate_test_data(borehole, N_TEST, DIM, seed=99)

    # Standardize
    y_mean = y_vals.mean()
    y_std = y_vals.std()
    y_vals_std = (y_vals - y_mean) / y_std
    grads_std = grads / y_std

    train_y_np = np.column_stack([y_vals_std] + [grads_std[:, j] for j in range(DIM)])
    train_x = torch.tensor(X_train, dtype=torch.float64, device=device)
    train_y = torch.tensor(train_y_np, dtype=torch.float64, device=device)
    test_x = torch.tensor(X_test, dtype=torch.float64, device=device)

    # Single Adam training run
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=1 + DIM
    ).to(device)
    model = GPModelWithDerivatives(train_x, train_y, likelihood).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters())
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    t_start = time.perf_counter()

    for i in range(N_TRAINING_ITER):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

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

    # Denormalize
    y_pred = mean[:, 0].cpu().numpy() * y_std + y_mean

    metrics = compute_metrics(y_test, y_pred)
    metrics['train_time'] = t_train
    metrics['pred_time'] = t_pred
    metrics['n_train'] = n_train
    metrics['seed'] = seed
    metrics['n_training_iter'] = N_TRAINING_ITER
    return metrics


def main():
    results = []
    for n_train in SAMPLE_SIZES:
        print(f"\n{'='*60}")
        print(f"  GPyTorch DEGP — Borehole — n_train = {n_train}")
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

    outfile = os.path.join(DATA_DIR, f"results_gpytorch_{FUNCTION_NAME}_{N_TRAINING_ITER}iter.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")


def single():
    # sys.argv: script.py --single <n_iter> <n_train> <seed> [rep]
    n_train = int(sys.argv[3])
    seed = int(sys.argv[4])
    rep = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    os.makedirs(DATA_DIR, exist_ok=True)
    outfile = os.path.join(DATA_DIR, f"results_gpytorch_{FUNCTION_NAME}_{N_TRAINING_ITER}iter.json")
    print(f"  GPyTorch — {FUNCTION_NAME} ({N_TRAINING_ITER} iter) — n_train={n_train}, seed={seed}")
    result = run_single(n_train, seed)
    result['macroreplicate'] = rep
    print(f"    NRMSE:      {result['nrmse']:.6e}")
    print(f"    Train time: {result['train_time']:.2f}s")
    if os.path.exists(outfile):
        with open(outfile) as f:
            results = json.load(f)
    else:
        results = []
    results.append(result)
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {outfile} ({len(results)} total)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--single':
        single()
    else:
        main()
