"""
Top-down profile of build_U_from_phi.

Breaks it down into: _extract_K_sub, cho_factor/cho_solve, numpy overhead.
"""

import os, sys, time, cProfile, pstats
import numpy as np
from scipy.stats.qmc import LatinHypercube
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_functions import morris, morris_gradient
from jetgp.full_degp_sparse.degp import degp as SparseDEGP
from jetgp.full_degp_sparse.sparse_cholesky import build_U_from_phi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
DIM = 20
N_TRAIN = 100
DER_INDICES = [[[[i, 1]] for i in range(1, DIM + 1)]]

sampler = LatinHypercube(d=DIM, seed=42)
X_train = sampler.random(n=N_TRAIN)
y_vals = morris(X_train)
grads = morris_gradient(X_train)

y_train_list = [y_vals.reshape(-1, 1)]
for j in range(DIM):
    y_train_list.append(grads[:, j].reshape(-1, 1))

model = SparseDEGP(
    X_train, y_train_list,
    n_order=1, n_bases=DIM,
    der_indices=DER_INDICES,
    normalize=True, kernel='SE', kernel_type='anisotropic',
    rho=1.0, use_supernodes=False,
)

opt = model.optimizer
x0 = np.array([0.1] * (len(model.bounds) - 2) + [0.5, -3.0])
P_full = model.mmd_P_full
N_total = len(P_full)

# Run one NLML to trigger lazy init
opt.negative_log_marginal_likelihood(x0)

# Ensure kernel plan + phi index maps are built
diffs = model.differences_by_dim
phi_tmp = model.kernel_func(diffs, x0[:-1])
n_bases_tmp = phi_tmp.get_active_bases()[-1]
opt._ensure_kernel_plan(n_bases_tmp)
opt._ensure_phi_index_maps(N_TRAIN)

# Extract ingredients for build_U_from_phi
sigma_n_sq = (10.0 ** x0[-1]) ** 2
phi = model.kernel_func(diffs, x0[:-1])
n_bases = phi.get_active_bases()[-1]
deriv_order = 2 * model.n_order
phi_exp = opt._expand_derivs(phi, n_bases, deriv_order)
base_shape = phi.shape
phi_3d = phi_exp.reshape(phi_exp.shape[0], base_shape[0], base_shape[1])

k_type, k_phys, deriv_lookup, sign_lookup = opt._k_index_map
U_buf = np.zeros((N_total, N_total))
block_size = model.n_bases + 1

print(f"K size: {N_total}x{N_total}")
print(f"phi_3d shape: {phi_3d.shape}")
print(f"block_size: {block_size}")
print(f"Number of blocks: {(N_total + block_size - 1) // block_size}")

# Check block sizes
S = model.sparse_S_full_arr
block_sizes = []
for start in range(0, N_total, block_size):
    end = min(start + block_size, N_total)
    nb = S[end - 1]
    block_sizes.append(len(nb))
block_sizes = np.array(block_sizes)
print(f"Neighbourhood sizes: min={block_sizes.min()}, max={block_sizes.max()}, "
      f"mean={block_sizes.mean():.1f}, median={np.median(block_sizes):.0f}")
print()

# Warm up
build_U_from_phi(
    phi_3d, S, N_total,
    block_size=block_size,
    k_type=k_type, k_phys=k_phys,
    deriv_lookup=deriv_lookup, sign_lookup=sign_lookup,
    P_full=P_full,
    sigma_n_sq=sigma_n_sq,
    sigma_data_diag=opt._sigma_data_diag_mmd,
    out=U_buf,
)

# ---------------------------------------------------------------------------
# Overall timing
# ---------------------------------------------------------------------------
N_ITER = 30

def bench(label, func, n=N_ITER):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000
    print(f"  {label:50s}  mean={times.mean():7.2f}  median={np.median(times):7.2f}  min={times.min():7.2f} ms")
    return np.median(times)

print(f"{'='*70}")
print(f"Overall timing ({N_ITER} calls)")
print(f"{'='*70}")

bench("build_U_from_phi",
      lambda: build_U_from_phi(
          phi_3d, S, N_total,
          block_size=block_size,
          k_type=k_type, k_phys=k_phys,
          deriv_lookup=deriv_lookup, sign_lookup=sign_lookup,
          P_full=P_full,
          sigma_n_sq=sigma_n_sq,
          sigma_data_diag=opt._sigma_data_diag_mmd,
          out=U_buf,
      ))

# ---------------------------------------------------------------------------
# cProfile top-down
# ---------------------------------------------------------------------------
N_PROF = 30
print(f"\n{'='*70}")
print(f"cProfile top-down ({N_PROF} calls)")
print(f"{'='*70}")

pr = cProfile.Profile()
pr.enable()
for _ in range(N_PROF):
    build_U_from_phi(
        phi_3d, S, N_total,
        block_size=block_size,
        k_type=k_type, k_phys=k_phys,
        deriv_lookup=deriv_lookup, sign_lookup=sign_lookup,
        P_full=P_full,
        sigma_n_sq=sigma_n_sq,
        sigma_data_diag=opt._sigma_data_diag_mmd,
        out=U_buf,
    )
pr.disable()
pstats.Stats(pr).strip_dirs().sort_stats('tottime').print_stats(30)

# ---------------------------------------------------------------------------
# Component benchmarks: isolate _extract_K_sub vs cho_factor/cho_solve
# ---------------------------------------------------------------------------
from jetgp.full_degp_sparse.optimizer import _extract_K_sub
from scipy.linalg import cho_factor, cho_solve

print(f"\n{'='*70}")
print(f"Component benchmarks ({N_ITER} calls, summed over all blocks)")
print(f"{'='*70}")

# Time each phase separately across all blocks
def time_extract_only():
    for start in range(0, N_total, block_size):
        end = min(start + block_size, N_total)
        nb_union = S[end - 1]
        m_union = len(nb_union)
        orig_nb_union = P_full[nb_union]
        nb_type_union = k_type[orig_nb_union]
        nb_phys_union = k_phys[orig_nb_union]
        sd_diag_union = opt._sigma_data_diag_mmd[nb_union]
        K_sub_union = np.empty((m_union, m_union))
        _extract_K_sub(phi_3d, nb_type_union, nb_phys_union,
                       deriv_lookup, sign_lookup,
                       sigma_n_sq, sd_diag_union, m_union, K_sub_union)

def time_cho_only():
    for start in range(0, N_total, block_size):
        end = min(start + block_size, N_total)
        nb_union = S[end - 1]
        m_union = len(nb_union)
        orig_nb_union = P_full[nb_union]
        nb_type_union = k_type[orig_nb_union]
        nb_phys_union = k_phys[orig_nb_union]
        sd_diag_union = opt._sigma_data_diag_mmd[nb_union]
        K_sub_union = np.empty((m_union, m_union))
        _extract_K_sub(phi_3d, nb_type_union, nb_phys_union,
                       deriv_lookup, sign_lookup,
                       sigma_n_sq, sd_diag_union, m_union, K_sub_union)
        cols = list(range(start, end))
        n_cols = len(cols)
        positions = np.searchsorted(nb_union, cols)
        E = np.zeros((m_union, n_cols))
        E[positions, np.arange(n_cols)] = 1.0
        L_u, low_u = cho_factor(K_sub_union, lower=True)
        V = cho_solve((L_u, low_u), E)

def time_indexing_only():
    """Just the P_full lookups, k_type/k_phys gathering, etc."""
    for start in range(0, N_total, block_size):
        end = min(start + block_size, N_total)
        nb_union = S[end - 1]
        orig_nb_union = P_full[nb_union]
        nb_type_union = k_type[orig_nb_union]
        nb_phys_union = k_phys[orig_nb_union]
        sd_diag_union = opt._sigma_data_diag_mmd[nb_union]

def time_scatter_only():
    """Just the U[ix_(nb, cols)] = V scatter."""
    U_buf[:] = 0.0
    for start in range(0, N_total, block_size):
        end = min(start + block_size, N_total)
        nb_union = S[end - 1]
        cols = list(range(start, end))
        n_cols = len(cols)
        V = np.ones((len(nb_union), n_cols))
        U_buf[np.ix_(nb_union, cols)] = V

bench("_extract_K_sub (all blocks)", time_extract_only)
bench("cho_factor + cho_solve (all blocks)", time_cho_only)
bench("Index gathering (all blocks)", time_indexing_only)
bench("U scatter (all blocks)", time_scatter_only)

# ---------------------------------------------------------------------------
# Per-block size analysis
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"Per-block breakdown (single pass, timed per block)")
print(f"{'='*70}")

t_extract = []
t_cho = []
t_scatter = []
t_total = []
sizes = []

for start in range(0, N_total, block_size):
    end = min(start + block_size, N_total)
    nb_union = S[end - 1]
    m_union = len(nb_union)
    sizes.append(m_union)

    # Extract
    t0 = time.perf_counter()
    orig_nb_union = P_full[nb_union]
    nb_type_union = k_type[orig_nb_union]
    nb_phys_union = k_phys[orig_nb_union]
    sd_diag_union = opt._sigma_data_diag_mmd[nb_union]
    K_sub_union = np.empty((m_union, m_union))
    _extract_K_sub(phi_3d, nb_type_union, nb_phys_union,
                   deriv_lookup, sign_lookup,
                   sigma_n_sq, sd_diag_union, m_union, K_sub_union)
    t1 = time.perf_counter()
    t_extract.append(t1 - t0)

    # Cho
    cols = list(range(start, end))
    n_cols = len(cols)
    positions = np.searchsorted(nb_union, cols)
    E = np.zeros((m_union, n_cols))
    E[positions, np.arange(n_cols)] = 1.0
    L_u, low_u = cho_factor(K_sub_union, lower=True)
    V = cho_solve((L_u, low_u), E)
    diag_vals = V[positions, np.arange(n_cols)]
    np.maximum(diag_vals, 1e-30, out=diag_vals)
    np.sqrt(diag_vals, out=diag_vals)
    V /= diag_vals
    t2 = time.perf_counter()
    t_cho.append(t2 - t1)

    # Scatter
    U_buf[np.ix_(nb_union, cols)] = V
    t3 = time.perf_counter()
    t_scatter.append(t3 - t2)
    t_total.append(t3 - t0)

t_extract = np.array(t_extract) * 1000
t_cho = np.array(t_cho) * 1000
t_scatter = np.array(t_scatter) * 1000
t_total = np.array(t_total) * 1000
sizes = np.array(sizes)

print(f"  {'Phase':<20s} {'Total (ms)':>10s} {'Mean/block':>12s} {'Max/block':>12s}")
print(f"  {'─'*20} {'─'*10} {'─'*12} {'─'*12}")
print(f"  {'_extract_K_sub':<20s} {t_extract.sum():10.2f} {t_extract.mean():12.3f} {t_extract.max():12.3f}")
print(f"  {'cho_factor+solve':<20s} {t_cho.sum():10.2f} {t_cho.mean():12.3f} {t_cho.max():12.3f}")
print(f"  {'U scatter':<20s} {t_scatter.sum():10.2f} {t_scatter.mean():12.3f} {t_scatter.max():12.3f}")
print(f"  {'TOTAL':<20s} {t_total.sum():10.2f} {t_total.mean():12.3f} {t_total.max():12.3f}")

# Show correlation with neighbourhood size
print(f"\n  Neighbourhood size vs time (top 5 most expensive blocks):")
order = np.argsort(-t_total)[:5]
for i in order:
    print(f"    block {i:3d}: size={sizes[i]:3d}, extract={t_extract[i]:.3f}ms, "
          f"cho={t_cho[i]:.3f}ms, scatter={t_scatter[i]:.3f}ms, total={t_total[i]:.3f}ms")

print(f"\n  Size distribution of blocks:")
for pct in [25, 50, 75, 90, 95, 100]:
    print(f"    p{pct:>3d}: {np.percentile(sizes, pct):.0f}")

print("\nDone.")
