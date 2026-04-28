"""
Profile: GDDEGP on random 20D data with n_train=100 and 1..5 directional
derivative channels.

Examples:
    kernprof -l -v profile_gddegp_large.py
    kernprof -l -v profile_gddegp_large.py -- --n-dirs 3
    kernprof -l -v profile_gddegp_large.py -- --max-dirs 5 --iters 10
"""

import argparse
import numpy as np

from jetgp.full_gddegp.gddegp import gddegp
from jetgp.full_gddegp.optimizer import Optimizer

try:
    profile
except NameError:
    def profile(func):
        return func


DEFAULT_SEED = 42
DEFAULT_DIM = 20
DEFAULT_N_TRAIN = 200
DEFAULT_ITERS = 10
DEFAULT_MAX_DIRS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile GDDEGP with varying numbers of directional derivatives."
    )
    parser.add_argument(
        "--n-dirs",
        type=int,
        nargs="+",
        help="Specific directional-derivative counts to profile (e.g. --n-dirs 1 3 5).",
    )
    parser.add_argument(
        "--max-dirs",
        type=int,
        default=DEFAULT_MAX_DIRS,
        help=f"Maximum number of directional derivatives to sweep when --n-dirs is omitted (default: {DEFAULT_MAX_DIRS}).",
    )
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM, help="Input dimension.")
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN, help="Training set size.")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="Iterations per profiled call.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    return parser.parse_args()


def resolve_n_dirs(args):
    if args.n_dirs:
        counts = args.n_dirs
    else:
        counts = list(range(1, args.max_dirs + 1))

    if not counts:
        raise ValueError("At least one directional-derivative count must be provided.")
    if min(counts) < 1:
        raise ValueError("Directional-derivative counts must be positive.")
    if max(counts) > args.dim:
        raise ValueError(
            f"Requested {max(counts)} directional derivatives, but dim={args.dim}. "
            "Each point can support at most dim mutually orthonormal directions."
        )
    return counts


def generate_orthonormal_rays(n_train, dim, n_dirs, rng):
    """Return n_dirs per-point orthonormal direction fields, one array per direction type."""
    rays = np.empty((n_dirs, dim, n_train))
    for point_idx in range(n_train):
        q, _ = np.linalg.qr(rng.standard_normal((dim, n_dirs)))
        rays[:, :, point_idx] = q.T
    return [rays[k] for k in range(n_dirs)]


def build_case(dim, n_train, n_dirs, seed):
    rng = np.random.default_rng(seed)
    X = rng.random((n_train, dim))

    y_train = [rng.random((n_train, 1))]
    for _ in range(n_dirs):
        y_train.append(rng.random((n_train, 1)))

    rays_list = generate_orthonormal_rays(n_train, dim, n_dirs, rng)
    der_indices = [[[[k + 1, 1]]] for k in range(n_dirs)]
    deriv_locs = [list(range(n_train)) for _ in range(n_dirs)]

    model = gddegp(
        X,
        y_train,
        n_order=1,
        rays_list=rays_list,
        der_indices=der_indices,
        derivative_locations=deriv_locs,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic",
    )

    opt = Optimizer(model)
    x0 = np.array([0.1] * (len(model.bounds) - 2) + [0.5, -3.0])

    return model, opt, x0


def register_profiled_functions(opt):
    try:
        profile.add_function(opt.nll_and_grad)
        profile.add_function(opt.negative_log_marginal_likelihood)
        profile.add_function(opt.nll_grad)
    except AttributeError:
        pass


@profile
def run_nll_and_grad(opt, x0, n_iters):
    for _ in range(n_iters):
        opt.nll_and_grad(x0)


@profile
def run_nlml(opt, x0, n_iters):
    for _ in range(n_iters):
        opt.nll_wrapper(x0)


def main():
    args = parse_args()
    n_dirs_counts = resolve_n_dirs(args)

    print(
        f"GDDEGP profiling: dim={args.dim}, n_train={args.n_train}, "
        f"n_dirs={n_dirs_counts}, iters={args.iters}, seed={args.seed}"
    )

    for n_dirs in n_dirs_counts:
        print(f"\n{'=' * 72}")
        print(f"Profiling GDDEGP with {n_dirs} directional derivative(s) per point")
        print(f"{'=' * 72}")

        model, opt, x0 = build_case(args.dim, args.n_train, n_dirs, args.seed + n_dirs)
        register_profiled_functions(opt)

        print(
            f"K size: {len(model.y_train)} x {len(model.y_train)} "
            f"(func={args.n_train}, dir blocks={n_dirs})"
        )
        print("Warming up...")
        opt.nll_and_grad(x0)
        opt.nll_wrapper(x0)

        print(f"Profiling nll_and_grad ({args.iters} iterations)...")
        run_nll_and_grad(opt, x0, args.iters)

        print(f"Profiling negative_log_marginal_likelihood ({args.iters} iterations)...")
        run_nlml(opt, x0, args.iters)


if __name__ == "__main__":
    main()
