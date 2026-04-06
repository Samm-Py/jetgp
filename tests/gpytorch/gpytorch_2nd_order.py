"""
Benchmark: GPyTorch 2nd-order derivative-enhanced GP.

Uses RBFKernelGradGrad for covariance between function values,
first derivatives, and diagonal second derivatives.
Adam optimizer, CPU only.

Benchmark functions: Borehole (8D), OTL Circuit (6D), Morris (20D).
Sample sizes: DIM, 3*DIM, 5*DIM per function.
"""

import os
import numpy as np
import time
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import (
    borehole, borehole_gradient, borehole_hessian_diag,
    otl_circuit, otl_circuit_gradient, otl_circuit_hessian_diag,
    morris, morris_gradient, morris_hessian_diag,
    generate_test_data, compute_metrics
)
from scipy.stats.qmc import LatinHypercube

import torch
import gpytorch

torch.set_default_dtype(torch.float64)
device = torch.device('cpu')

N_MACROREPLICATES = 5
N_TEST = 2000
if len(sys.argv) > 1 and sys.argv[1] == '--single':
    N_TRAINING_ITER = int(sys.argv[2])
else:
    N_TRAINING_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 200

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TESTS_DIR, 'data')

BENCHMARKS = {
    'borehole': {
        'func': borehole,
        'grad_func': borehole_gradient,
        'hess_func': borehole_hessian_diag,
        'dim': 8,
    },
    'otl_circuit': {
        'func': otl_circuit,
        'grad_func': otl_circuit_gradient,
        'hess_func': otl_circuit_hessian_diag,
        'dim': 6,
    },
    'morris': {
        'func': morris,
        'grad_func': morris_gradient,
        'hess_func': morris_hessian_diag,
        'dim': 20,
    },
}


class GPModel2ndOrder(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dim):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGradGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGradGrad(ard_num_dims=dim)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def run_single(func_name, n_train, seed):
    cfg = BENCHMARKS[func_name]
    dim = cfg['dim']
    func = cfg['func']
    grad_func = cfg['grad_func']
    hess_func = cfg['hess_func']

    sampler = LatinHypercube(d=dim, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = func(X_train)
    grads = grad_func(X_train)
    hess_diag = hess_func(X_train)
    X_test, y_test = generate_test_data(func, N_TEST, dim, seed=99)

    # Standardize
    y_mean = y_vals.mean()
    y_std = y_vals.std()
    y_vals_std = (y_vals - y_mean) / y_std
    grads_std = grads / y_std
    hess_std = hess_diag / y_std

    # Training targets: [f, df/dx1,...,df/dxD, d2f/dx1^2,...,d2f/dxD^2]
    # RBFKernelGradGrad expects n_tasks = 1 + 2*D
    train_y_np = np.column_stack(
        [y_vals_std]
        + [grads_std[:, j] for j in range(dim)]
        + [hess_std[:, j] for j in range(dim)]
    )
    train_x = torch.tensor(X_train, dtype=torch.float64, device=device)
    train_y = torch.tensor(train_y_np, dtype=torch.float64, device=device)
    test_x = torch.tensor(X_test, dtype=torch.float64, device=device)

    num_tasks = 1 + 2 * dim
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks
    ).to(device)
    model = GPModel2ndOrder(train_x, train_y, likelihood, dim).to(device)

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

    # Denormalize (first task column is function values)
    y_pred = mean[:, 0].cpu().numpy() * y_std + y_mean

    metrics = compute_metrics(y_test, y_pred)
    metrics['train_time'] = t_train
    metrics['pred_time'] = t_pred
    metrics['n_train'] = n_train
    metrics['seed'] = seed
    metrics['n_training_iter'] = N_TRAINING_ITER
    metrics['function'] = func_name
    metrics['dim'] = dim
    return metrics


def main():
    all_results = {}

    for func_name, cfg in BENCHMARKS.items():
        dim = cfg['dim']
        sample_sizes = [dim, 3 * dim, 5 * dim]
        results = []

        for n_train in sample_sizes:
            print(f"\n{'='*60}")
            print(f"  GPyTorch 2nd Order — {func_name} {dim}D — n_train={n_train}")
            print(f"  Matrix size: {n_train} x {1 + 2*dim} = {n_train * (1 + 2*dim)}")
            print(f"{'='*60}")
            for rep in range(N_MACROREPLICATES):
                seed = 1000 + rep
                print(f"\n  Macroreplicate {rep + 1}/{N_MACROREPLICATES} (seed={seed})")
                result = run_single(func_name, n_train, seed)
                result['macroreplicate'] = rep + 1
                results.append(result)
                print(f"    RMSE:       {result['rmse']:.6e}")
                print(f"    NRMSE:      {result['nrmse']:.6e}")
                print(f"    Train time: {result['train_time']:.2f}s")
                print(f"    Pred time:  {result['pred_time']:.4f}s")

        print(f"\n{'='*60}")
        print(f"  Summary — {func_name}")
        print(f"{'='*60}")
        for n_train in sample_sizes:
            subset = [r for r in results if r['n_train'] == n_train]
            nrmses = [r['nrmse'] for r in subset]
            times = [r['train_time'] for r in subset]
            print(f"\n  n={n_train}:")
            print(f"    NRMSE: mean={np.mean(nrmses):.6e}, std={np.std(nrmses):.6e}")
            print(f"    Time:  mean={np.mean(times):.2f}s, std={np.std(times):.2f}s")

        outfile = os.path.join(DATA_DIR, f"results_gpytorch_2nd_order_{func_name}_{N_TRAINING_ITER}iter.json")
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(outfile, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {outfile}")
        all_results[func_name] = results

    return all_results


def single():
    # sys.argv: script.py --single <n_iter> <func_name> <n_train> <seed> [rep]
    func_name = sys.argv[3]
    n_train = int(sys.argv[4])
    seed = int(sys.argv[5])
    rep = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    os.makedirs(DATA_DIR, exist_ok=True)
    outfile = os.path.join(DATA_DIR, f"results_gpytorch_2nd_order_{func_name}_{N_TRAINING_ITER}iter.json")
    print(f"  GPyTorch 2nd Order — {func_name} ({N_TRAINING_ITER} iter) — n_train={n_train}, seed={seed}")
    result = run_single(func_name, n_train, seed)
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
