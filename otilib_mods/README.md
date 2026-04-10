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
| `rebuild_all_static.py` | `build/rebuild_all_static.py` |
| `rebuild_all_static.sh` | `build/rebuild_all_static.sh` |

## What Each Change Does

- **`cmod_writer.py`** — JetGP's patched `cmod_writer`. Deployed to both `build/pyoti/` and `src/python/pyoti/python/` in otilib.
- **`regenerate_all_c.py`** — Regenerates static module C/Cython sources from templates. Cleans up shipped sources not in the target module list before generating.
- **`build_static.py`** — Builds a single Cython static module (e.g. `python build_static.py m10n2`).
- **`rebuild_all_static.py`** — Rebuilds all existing Cython static modules sequentially.
- **`rebuild_all_static.sh`** — Rebuilds all existing Cython static modules in parallel.
- **`src_CMakeLists.txt`** — Adds auto-detected `make mXnY` targets with `-fopenmp` for all `onummXnY` C libraries.
- **`src_python_pyoti_CMakeLists.txt`** — Changes the Cython build command from `python` to `python3`.
- **`creators.pxi`** — Adds `oti.empty()`: an uninitialized array allocator (like `zeros()` but skips zero-fill).
- **`include.pxi`** — Adds `from openmp cimport omp_get_num_procs, omp_get_max_threads` for OpenMP thread query support.
- **`array_base.pxi`** — Adds `{arr_get_all_derivs}` expansion to the array base template.

> **Note:** All hardcoded paths in the build scripts are rewritten automatically by `setup_otilib`.
