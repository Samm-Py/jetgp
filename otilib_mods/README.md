# otilib_mods

Modified files from `otilib-master` required to build the version of pyoti used by JetGP.
Copy each file to its destination in your local `otilib-master` before running the build workflow.

## File Map

| File in this folder | Destination in otilib-master |
|---|---|
| `src_CMakeLists.txt` | `src/CMakeLists.txt` |
| `src_python_pyoti_CMakeLists.txt` | `src/python/pyoti/CMakeLists.txt` |
| `regenerate_all_c.py` | `build/regenerate_all_c.py` |
| `creators.pxi` | `src/python/pyoti/python/source_conv/src/python/pyoti/cython/static/number/creators.pxi` |
| `include.pxi` | `src/python/pyoti/python/source_conv/src/python/pyoti/cython/static/number/include.pxi` |
| `array_base.pxi` | `src/python/pyoti/python/source_conv/src/python/pyoti/cython/static/number/array/base.pxi` |

## What Each Change Does

- **`regenerate_all_c.py`** — JetGP-authored script that regenerates all static module C/Cython sources from templates. Copied to `build/` and path-patched by `setup_otilib.py`.
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

> **Note:** `build/regenerate_all_c.py`, `build/build_static.py`, `build/rebuild_all_static.py`,
> and `build/rebuild_all_static.sh` contain hardcoded absolute paths that must be updated
> to match your machine before running.
