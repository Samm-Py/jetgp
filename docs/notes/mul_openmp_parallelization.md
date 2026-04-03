# OpenMP Parallelization of OTI `mul` — Investigation Summary

## Status: DISABLED (as of 2026-04-01)

OMP was removed from `mul` entirely. See `otilib_openmp_changes.md` for full details.

---

## Key Findings

### `mul` is memory-bandwidth bound

Each OTI element (m=20, n=2) has 231 coefficients x 8 bytes = ~1.8KB.
Per `mul` call on a 200x200 matrix (40,000 elements):
- Read: 2 x 40,000 x 1.8KB = ~144MB
- Write: 1 x 40,000 x 1.8KB = ~72MB
- Total memory traffic: ~216MB per call

At DDR4 bandwidth of ~40 GB/s, adding more threads doesn't help when the
memory bus is saturated. Max observed speedup: ~1.8x on 12 threads.

### `get_all_derivs` is compute-bound but OMP hurts in practice

In isolation, `get_all_derivs` shows 3-5x speedup with OMP on large matrices.
But in real workloads (890 calls per borehole optimization), thread spawn
overhead (~10-50us per spawn x 890 calls) dominated and caused 5x slowdown.

### Thread spawn overhead kills tight loops

The optimization loop calls `get_all_derivs` and `mul` hundreds of times.
Each OMP parallel region spawns/wakes threads. This overhead accumulates
and far exceeds any computation savings.

---

## Potential Future Approaches (not OMP)

1. **Fast C `get_all_derivs`** — cast struct to `double*` for direct memory access,
   avoiding `onumm_get_item()` switch overhead. See `get_derivs_m10n2.c` for example.
2. **Cache-blocking / tiling** for mul — process elements in L2/L3-sized chunks
3. **Fused operations** — combine mul+sum into single pass to reduce memory round-trips
4. **SIMD vectorization** — AVX2/AVX-512 for multiple coefficients per cycle
5. **Struct-of-arrays layout** — better streaming vs current array-of-structs
