# Static Module Build Guide

Guide for compiling OTI static modules (`onummXnY`) used by JetGP.

**Updated:** 2026-04-01 — OMP disabled, streamlined build procedure.

## Prerequisites

- Conda environment: `pyoti_2`
- Working directory: `~/research_head/otilib-master/build`

```bash
conda activate pyoti_2
cd ~/research_head/otilib-master/build
```

## Modules Used by JetGP Benchmarks

| Module | Benchmarks |
|--------|------------|
| m2n2   | GDDEGP 1 direction, DDEGP active subspace |
| m4n2   | GDDEGP 2 directions |
| m6n2   | GDDEGP 3 directions, DEGP OTL Circuit |
| m8n2   | DEGP Borehole, DEGP Dette-Pepelyshev |
| m10n2  | DEGP Active Subspace |
| m20n2  | DEGP Morris |

The module is selected at runtime by `jetgp.utils.get_oti_module(n_bases, n_order)`.

## Full Rebuild Procedure

### 1. Regenerate source files from templates

```bash
python regenerate_all_c.py m2n2 m4n2 m6n2 m8n2 m10n2 m20n2
```

**Warning:** This overwrites `src/c/static.c` to include ALL 58 modules.
You MUST edit it afterward to only include the modules you need:

```c
#include "static/onumm2n2.c"
#include "static/onumm4n2.c"
#include "static/onumm6n2.c"
#include "static/onumm8n2.c"
#include "static/onumm10n2.c"
#include "static/get_derivs_m10n2.c"
#include "static/onumm20n2.c"
```

### 2. Build base libraries and module libraries

```bash
# Delete stale .a files (make won't detect regenerated .c changes!)
rm -f lib/liboti.a

# Build base libraries
make oti oticwrap otistatic

# Build module libraries
make m2n2 m4n2 m6n2 m8n2 m10n2 m20n2
```

**Important:** If `make oti` tries to compile all 58 modules, you forgot to trim `static.c`.

### 3. Build Cython .so files

```bash
for m in m2n2 m4n2 m6n2 m8n2 m10n2 m20n2; do
  python build_static.py $m
done
```

Output goes to `otilib-master/build/pyoti/static/`.
Since pyoti is installed in dev mode, no copy is needed.

## Building a Single Module

Example: rebuild just `m8n2` after a template change.

```bash
# Must delete stale .a or make won't recompile
rm -f lib/libm8n2.a
make m8n2
python build_static.py m8n2
```

If base libraries (`liboti.a`) also need updating:
```bash
rm -f lib/liboti.a
make oti
```

## Verification

After building, verify no OMP parallelization in mul:

```bash
nm pyoti/static/onumm8n2.cpython-39-x86_64-linux-gnu.so | grep 'oarrm8n2_mul.*omp_fn'
# Should return nothing (exit code 1)
```

Quick import test:
```python
python -c "from pyoti.static.onumm8n2 import onumm8n2; print('OK')"
```

## Key Files

### Templates (source of truth)
- **C template:** `src/python/pyoti/python/source_conv/src/c/static/number/array/algebra_elementwise_to.c`
- **Cython generator:** `src/python/pyoti/python/cmod_writer.py`

### Build copies (must be synced from source)
- `build/pyoti/source_conv/src/c/static/number/array/algebra_elementwise_to.c`
- `build/pyoti/cmod_writer.py`

### JetGP copy (must also be synced)
- `jetgp/jetgp/cmod_writer.py`

**All three copies of `cmod_writer.py` must match.**

## Gotchas

1. **`make` doesn't detect regenerated `.c` files** — always `rm -f lib/libXXX.a` first
2. **`regenerate_all_c.py` overwrites `static.c`** — must re-edit after running
3. **Build/source template desync** — `regenerate_all_c.py` reads C templates from
   `build/pyoti/source_conv/` but imports `cmod_writer.py` from `src/python/pyoti/python/`.
   Both must be kept in sync.
4. **Building all 58 modules takes 30-60+ min** — only build what you need
