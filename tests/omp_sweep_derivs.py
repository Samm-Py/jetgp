"""
OpenMP threshold sweep for get_all_derivs across module sizes and matrix dimensions.
Single config per invocation — OMP_NUM_THREADS must be set BEFORE launching python.

Usage:
  python omp_sweep_derivs.py <module> <N>
  e.g.: python omp_sweep_derivs.py m6n2 420

Or run all via: bash omp_sweep_derivs.sh
"""
import sys
import time
import json

MODULES = {
    'm6n2':  ('pyoti.static.onumm6n2',  6, 2, 28),
    'm10n2': ('pyoti.static.onumm10n2', 10, 2, 66),
    'm15n2': ('pyoti.static.onumm15n2', 15, 2, 136),
    'm20n2': ('pyoti.static.onumm20n2', 20, 2, 231),
}

N_WARMUP = 2
N_REPS = 5

mod_name = sys.argv[1]
N = int(sys.argv[2])

mod_path, nbasis, order, n_coeffs = MODULES[mod_name]
mod = __import__(mod_path, fromlist=[''])

elem_bytes = n_coeffs * 8
n_elements = N * N
total_mb = n_elements * elem_bytes / (1024 * 1024)

A = mod.zeros((N, N))

# Warmup
for _ in range(N_WARMUP):
    _ = A.get_all_derivs(nbasis, order)

# Benchmark
t0 = time.perf_counter()
for _ in range(N_REPS):
    _ = A.get_all_derivs(nbasis, order)
elapsed_ms = (time.perf_counter() - t0) / N_REPS * 1000

result = {
    'mod': mod_name,
    'N': N,
    'elements': n_elements,
    'elem_bytes': elem_bytes,
    'total_mb': round(total_mb, 1),
    'time_ms': round(elapsed_ms, 2),
}
print(json.dumps(result))
