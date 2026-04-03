"""
OpenMP threshold sweep: mul operation across multiple module sizes and matrix dimensions.
Runs a single configuration (module + matrix size) per invocation to avoid memory issues.
OMP_NUM_THREADS must be set BEFORE launching python.

Usage:
  python omp_sweep.py <module> <N>
  e.g.: python omp_sweep.py m6n2 420

Or run all via the shell wrapper: bash omp_sweep.sh
"""
import sys
import time
import json

MODULES = {
    'm6n2':  ('pyoti.static.onumm6n2',  28),
    'm10n2': ('pyoti.static.onumm10n2', 66),
    'm15n2': ('pyoti.static.onumm15n2', 136),
    'm20n2': ('pyoti.static.onumm20n2', 231),
}

N_WARMUP = 3
# Scale reps inversely with expected time to get stable measurements
# Small matrices need many more reps since individual calls are sub-ms
N_REPS_BASE = 5

mod_name = sys.argv[1]
N = int(sys.argv[2])

mod_path, n_coeffs = MODULES[mod_name]
mod = __import__(mod_path, fromlist=[''])

elem_bytes = n_coeffs * 8
n_elements = N * N
total_mb = n_elements * elem_bytes / (1024 * 1024)

A = mod.zeros((N, N))
B = mod.zeros((N, N))
C = mod.zeros((N, N))

# Scale reps: target ~1s of measurement time minimum
# Estimate with a single warmup call
for _ in range(N_WARMUP):
    mod.mul(A, B, out=C)

t_est = time.perf_counter()
mod.mul(A, B, out=C)
t_est = time.perf_counter() - t_est
N_REPS = max(N_REPS_BASE, int(1.0 / max(t_est, 1e-9)))  # aim for ~1s total
N_REPS = min(N_REPS, 10000)  # cap it

# Benchmark
t0 = time.perf_counter()
for _ in range(N_REPS):
    mod.mul(A, B, out=C)
elapsed_ms = (time.perf_counter() - t0) / N_REPS * 1000

# Output as JSON for easy parsing
result = {
    'mod': mod_name,
    'N': N,
    'elements': n_elements,
    'elem_bytes': elem_bytes,
    'total_mb': round(total_mb, 1),
    'time_ms': round(elapsed_ms, 2),
}
print(json.dumps(result))
