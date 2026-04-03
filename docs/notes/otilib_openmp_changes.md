# OTIlib OpenMP Changes — Final State

**Date:** 2026-04-01

---

## Summary

OpenMP parallelization was **disabled entirely** for both `mul` and `get_all_derivs` operations.
Benchmarking showed that thread spawn overhead in tight optimization loops (hundreds of calls
per benchmark) caused significant regressions that outweighed any gains from parallelism.

---

## Current State

### `mul` (C-level)
- **OMP pragmas removed** from C template
- Template: `src/python/pyoti/python/source_conv/src/c/static/number/array/algebra_elementwise_to.c`
- Also synced to: `build/pyoti/source_conv/src/c/static/number/array/algebra_elementwise_to.c`

### `get_all_derivs` (Cython-level)
- **`prange` replaced with plain `range`** — no OpenMP at all
- File: `src/python/pyoti/python/cmod_writer.py` (line ~3588)
- Also in: `build/pyoti/cmod_writer.py` and `jetgp/jetgp/cmod_writer.py`

### `static.c`
- Trimmed to only include modules needed for JetGP benchmarks:
  m2n2, m4n2, m6n2, m8n2, m10n2, m20n2 (+ get_derivs_m10n2)
- **Warning:** `regenerate_all_c.py` overwrites `static.c` to include ALL modules.
  Must re-edit `static.c` after running regeneration.

---

## Build Procedure

```bash
cd /home/sam/research_head/otilib-master/build

# 1. Regenerate source files (only needed modules)
python regenerate_all_c.py m2n2 m4n2 m6n2 m8n2 m10n2 m20n2

# 2. Fix static.c (regenerate overwrites it with all 58 modules)
#    Edit src/c/static.c to only include the 6 modules you need

# 3. Rebuild liboti.a (delete stale .a first — make won't detect regenerated .c changes)
rm -f lib/liboti.a
make oti

# 4. Rebuild module .a libraries
make m2n2 m4n2 m6n2 m8n2 m10n2 m20n2

# 5. Build Cython modules
for m in m2n2 m4n2 m6n2 m8n2 m10n2 m20n2; do
  python build_static.py $m
done
```

---

## Verification

After building, verify no OMP parallelization in mul:
```bash
nm pyoti/static/onumm8n2.cpython-39-x86_64-linux-gnu.so | grep 'oarrm8n2_mul.*omp_fn'
# Should return nothing
```

Note: `get_all_derivs._omp_fn.0` symbols may still appear — this is just Cython boilerplate
from using `with nogil:` blocks. With plain `range` (not `prange`), no threads are spawned.

---

## Module-to-Benchmark Mapping

| Module | Benchmarks |
|---|---|
| m2n2 | GDDEGP 1 direction, DDEGP active subspace |
| m4n2 | GDDEGP 2 directions |
| m6n2 | GDDEGP 3 directions, DEGP OTL Circuit |
| m8n2 | DEGP Borehole, DEGP Dette-Pepelyshev |
| m10n2 | DEGP Active Subspace |
| m20n2 | DEGP Morris |

---

## Why OMP Was Disabled

1. **`mul` is memory-bandwidth bound** — max ~1.8x speedup on 12 threads even for large matrices.
   Not worth the overhead.
2. **`get_all_derivs` has good compute scaling** (3-5x on 12 threads) in isolation,
   but in real workloads (890 calls per borehole optimization), thread spawn overhead
   dominated and caused 5x slowdown.
3. **`OMP_NUM_THREADS` cannot be changed mid-process** — OpenMP runtime reads it only at
   initialization, making adaptive thresholding impractical.

---

## Future Optimization Ideas

### Fast C `get_all_derivs` for all modules (no OMP)

A fast C implementation exists for m10n2: `src/c/static/get_derivs_m10n2.c`.
It achieves speedup by casting the `onumm_t` struct to `double*` for direct contiguous
memory access, avoiding the per-coefficient `onumm_get_item()` switch/lookup overhead.

**TODO:** Write equivalent fast C `get_all_derivs` for m2n2, m4n2, m6n2, m8n2, m20n2.
Remove the `#pragma omp parallel for` from the m10n2 version as well.

Key idea from `get_derivs_m10n2.c`:
```c
/* Cast struct to double array for direct access */
double* coeffs = (double*)&p_data[kk];

/* Copy all coefficients with scaling — no switch overhead */
for (uint64_t d = 0; d < ndir; d++) {
    result[d * plane_size + ij_offset] = factors[d] * coeffs[d];
}
```

This should give a significant speedup for `get_all_derivs`, which is typically the
#1 bottleneck (30-50% of total optimization time).

---

## Key Gotchas

1. **`make` doesn't detect regenerated `.c` files** — must `rm -f lib/libXXX.a` before `make`
2. **`regenerate_all_c.py` overwrites `static.c`** — must re-edit after running
3. **Build/source template desync** — `regenerate_all_c.py` reads templates from `build/pyoti/source_conv/`
   but imports `cmod_writer.py` from `src/python/pyoti/python/`. Both must be kept in sync.
4. **Three copies of `cmod_writer.py`** — otilib source, otilib build, and jetgp. All must match.
