# otilib_mods

Modified files from `otilib-master` required to build the version of pyoti used by JetGP.
Use `setup_otilib` to apply all patches automatically — see the [otilib Setup](../docs/source/getting_started/otilib_setup.rst) docs or run:

```bash
python -m jetgp.setup_otilib --otilib /path/to/otilib-master --build
```

## File Map

| File in this folder | Destination in otilib-master |
|---|---|
| `src_CMakeLists.txt` | `src/CMakeLists.txt` |
| `src_python_pyoti_CMakeLists.txt` | `src/python/pyoti/CMakeLists.txt` |
| `regenerate_all_c.py` | `build/regenerate_all_c.py` |
| `build_static.py` | `build/build_static.py` |
| `cmod_writer.py` | `build/pyoti/cmod_writer.py` |
| `cmod_writer.py` | `src/python/pyoti/python/cmod_writer.py` |
| `creators.pxi` | `src/python/pyoti/python/source_conv/src/python/pyoti/cython/static/number/creators.pxi` |
| `include.pxi` | `src/python/pyoti/python/source_conv/src/python/pyoti/cython/static/number/include.pxi` |
| `array_base.pxi` | `src/python/pyoti/python/source_conv/src/python/pyoti/cython/static/number/array/base.pxi` |

## What Each Change Does

- **`cmod_writer.py`** — JetGP's patched `cmod_writer`. Deployed to both `build/pyoti/` and `src/python/pyoti/python/` in otilib, and also copied into the active conda env's `pyoti` package by `setup_otilib`.
- **`regenerate_all_c.py`** — JetGP-authored script that regenerates all static module C/Cython sources from templates. Copied to `build/` and path-patched by `setup_otilib`.
- **`build_static.py`** — Batch Cython module build script used by `rebuild_all_static.sh`. Copied to `build/` and path-patched by `setup_otilib` (updates `PROJECT_ROOT`).
- **`src/CMakeLists.txt`** — Adds `-fopenmp` to the static module compile flags so OpenMP is enabled for all `onummXnY` C libraries.
- **`src/python/pyoti/CMakeLists.txt`** — Changes the Cython build command from `python` to `python3`.
- **`creators.pxi`** — Adds `oti.empty()`: an uninitialized array allocator (like `zeros()` but skips zero-fill). Required for JetGP's fused difference path performance.
- **`include.pxi`** — Adds `from openmp cimport omp_get_num_procs, omp_get_max_threads` for OpenMP thread query support.
- **`array_base.pxi`** — Adds `{arr_get_all_derivs}` expansion to the array base template.

## Build Workflow

After copying the files:

```bash
# 1. Regenerate all C/Cython source from templates
cd /path/to/otilib-master
python build/regenerate_all_c.py

# 2. CMake configure and build core libraries
cd build
cmake ..
make -j$(nproc)
make gendata

# 3. Build all Cython static modules (parallel, 4 workers)
bash rebuild_all_static.sh 4
```

> **Note:** All hardcoded paths in the build scripts are rewritten automatically by `setup_otilib`. If running manually, update `BASE_DIR`/`PROJECT_ROOT` in `regenerate_all_c.py`, `build_static.py`, and `rebuild_all_static.py`, and the `ls` glob in `rebuild_all_static.sh`.
