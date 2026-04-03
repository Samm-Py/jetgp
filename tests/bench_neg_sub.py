import sys
sys.path.insert(0, '.')
import numpy as np, time
from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.full_degp.degp import degp

DIM=20; N=200
sampler = LatinHypercube(d=DIM, seed=42)
X = sampler.random(n=N)
y = morris(X); g = morris_gradient(X)
di = [[[[i+1,1]]] for i in range(DIM)]
dl = [list(range(N)) for _ in range(DIM)]
yt = [y.reshape(-1,1)] + [g[:,j:j+1] for j in range(DIM)]
m = degp(X, yt, n_order=1, n_bases=DIM, der_indices=di, derivative_locations=dl, normalize=True, kernel="RQ", kernel_type="anisotropic")
oti = m.oti
diffs = m.differences_by_dim
x0 = np.zeros(len(m.bounds))
for i,b in enumerate(m.bounds): x0[i]=0.5*(b[0]+b[1])
m.optimizer.nll_and_grad(x0)

phi = m.kernel_func(diffs, x0[:-1])
a = oti.mul(1.5, phi)

def bench(name, fn, n=10):
    times = []
    for _ in range(n):
        t0=time.perf_counter(); fn(); t1=time.perf_counter()
        times.append(t1-t0)
    med = np.median(times)*1000
    print(f"  {name:40s} {med:8.2f}ms")

print("=== Negate approaches ===")
bench("mul(-1, a)  [negate via mul]", lambda: oti.mul(-1.0, a))
bench("neg(a)      [dedicated neg]", lambda: oti.neg(a))
bench("sub(0, a)   [negate via sub]", lambda: oti.sub(0.0, a))

print("\n=== 1-x approaches ===")
bench("sum(1, mul(-1,a)) [old: 2 ops]", lambda: oti.sum(1.0, oti.mul(-1.0, a)))
bench("sub(1, a)         [new: 1 op]", lambda: oti.sub(1.0, a))

print("\n=== Verify correctness ===")
r1 = oti.sum(1.0, oti.mul(-1.0, a))
r2 = oti.sub(1.0, a)
r3 = oti.neg(a)
r4 = oti.mul(-1.0, a)
print(f"  sum(1,mul(-1,a)) vs sub(1,a) max diff: {np.max(np.abs(r1.real - r2.real)):.2e}")
print(f"  neg(a) vs mul(-1,a) max diff:           {np.max(np.abs(r3.real - r4.real)):.2e}")
