"""
Verify mul actually uses multiple threads by timing a large operation
where the difference should be unambiguous.
"""
import sys
import time
import ctypes

lib = ctypes.CDLL('libgomp.so.1')
omp_get = lib.omp_get_max_threads
omp_get.restype = ctypes.c_int
print(f"OMP_MAX_THREADS = {omp_get()}")

from pyoti.static.onumm20n2 import *

N = 800
A = zeros((N, N))
B = zeros((N, N))
C = zeros((N, N))

# warmup
for _ in range(3):
    mul(A, B, out=C)

# time
t0 = time.perf_counter()
for _ in range(10):
    mul(A, B, out=C)
elapsed = (time.perf_counter() - t0) / 10

print(f"m20n2 {N}x{N}: {elapsed*1000:.2f} ms")
