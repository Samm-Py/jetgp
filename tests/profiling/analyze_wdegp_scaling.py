"""
Analyze optimal number of WDEGP submodels for Morris function.

Each submodel gets ALL n_train function values + gradients at (n_train/N) points.
Submodel K size: m = n_train + D * (n_train / N)
Total cost ~ N * m^3
"""
import os
import numpy as np
import matplotlib.pyplot as plt

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(TESTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

D = 20  # Morris dimension

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, n_train in zip(axes, [20, 100, 200]):
    # N must divide n_train evenly, and N >= 1
    N_vals = [n for n in range(1, n_train + 1) if n_train % n == 0]

    costs = []
    m_vals = []
    for N in N_vals:
        grads_per_sub = n_train // N
        m = n_train + D * grads_per_sub
        cost = N * m**3
        costs.append(cost)
        m_vals.append(m)

    costs = np.array(costs, dtype=float)
    m_vals = np.array(m_vals)

    # Normalize cost for readability
    min_idx = np.argmin(costs)

    ax.semilogy(N_vals, costs, 'b.-', markersize=8)
    ax.semilogy(N_vals[min_idx], costs[min_idx], 'r*', markersize=15,
                label=f'Optimal: N={N_vals[min_idx]}, m={m_vals[min_idx]}')
    ax.set_xlabel('Number of submodels (N)')
    ax.set_ylabel('Total cost ~ N * m^3')
    ax.set_title(f'n_train = {n_train}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Print details
    print(f"\nn_train = {n_train}, D = {D}")
    print(f"{'N':>5} {'grads/sub':>10} {'m':>6} {'N*m^3':>15} {'vs current':>10}")
    print("-" * 55)
    for i, N in enumerate(N_vals):
        grads_per_sub = n_train // N
        ratio = costs[i] / costs[-1]  # vs current (N=n_train)
        marker = " <-- optimal" if i == min_idx else ""
        marker2 = " <-- current" if N == n_train else ""
        print(f"{N:>5} {grads_per_sub:>10} {m_vals[i]:>6} {costs[i]:>15.0f} {ratio:>10.2f}x{marker}{marker2}")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'wdegp_optimal_submodels.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'wdegp_optimal_submodels.png'), dpi=150)
print("\nPlots saved.")
