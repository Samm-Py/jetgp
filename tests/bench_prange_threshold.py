"""
Benchmark get_all_derivs at various matrix sizes to find the prange crossover point.
Tests square OTI matrices from 5x5 to 300x300.
"""
import numpy as np
import time
import pyoti.static.onumm20n2 as oti

NBASES = 20
ORDER = 2
N_REPEATS = 20

sizes = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300]

print(f"{'size':>6} {'elements':>10} {'time_ms':>10} {'per_elem_us':>12}")
print("-" * 45)

for n in sizes:
    # Create an OTI matrix of size n x n
    arr = oti.zeros((n, n))
    # Put some data in it so it's not trivial
    for i in range(min(n, NBASES)):
        real_data = np.random.randn(n, n)
        arr = arr + oti.array(real_data)

    # Warm up
    _ = arr.get_all_derivs(NBASES, ORDER)

    # Time it
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        _ = arr.get_all_derivs(NBASES, ORDER)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = np.median(times) * 1000
    elements = n * n
    per_elem_us = avg_ms * 1000 / elements

    print(f"{n:>6} {elements:>10} {avg_ms:>10.3f} {per_elem_us:>12.3f}")
