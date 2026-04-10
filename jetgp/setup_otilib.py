#!/usr/bin/env python3
"""
setup_otilib.py — Patch otilib-master with JetGP's required modifications
and optionally run the full build workflow.

Usage:
    python setup_otilib.py                        # interactive
    python setup_otilib.py --otilib /path/to/otilib-master
    python setup_otilib.py --otilib /path/to/otilib-master --build
    python setup_otilib.py --otilib /path/to/otilib-master --build --workers 8
"""

import argparse
import os
import re
import shutil
import site
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config file — persists the otilib path for get_oti_module() auto-detection
# ---------------------------------------------------------------------------
JETGP_CONFIG_FILE = Path.home() / ".config" / "jetgp" / "otilib_path"

# ---------------------------------------------------------------------------
# File map: list of (source relative to otilib_mods/, destination relative to otilib root)
# A source file may appear more than once to copy it to multiple destinations.
# ---------------------------------------------------------------------------
FILE_MAP = [
    ("src_CMakeLists.txt",             "src/CMakeLists.txt"),
    ("src_python_pyoti_CMakeLists.txt", "src/python/pyoti/CMakeLists.txt"),
    ("regenerate_all_c.py",            "build/regenerate_all_c.py"),
    ("build_static.py",               "build/build_static.py"),
    ("cmod_writer.py",                 "build/pyoti/cmod_writer.py"),
    ("cmod_writer.py",                 "src/python/pyoti/python/cmod_writer.py"),
    ("creators.pxi",
        "src/python/pyoti/python/source_conv/"
        "src/python/pyoti/cython/static/number/creators.pxi"),
    ("include.pxi",
        "src/python/pyoti/python/source_conv/"
        "src/python/pyoti/cython/static/number/include.pxi"),
    ("array_base.pxi",
        "src/python/pyoti/python/source_conv/"
        "src/python/pyoti/cython/static/number/array/base.pxi"),
    ("rebuild_all_static.py",          "build/rebuild_all_static.py"),
    ("rebuild_all_static.sh",          "build/rebuild_all_static.sh"),
]

# ---------------------------------------------------------------------------
# Build scripts whose hardcoded paths need to be rewritten
# ---------------------------------------------------------------------------
SCRIPT_PATCHES = [
    (
        "build/regenerate_all_c.py",
        [
            (
                r'BASE_DIR\s*=\s*"[^"]+"',
                'BASE_DIR = "{otilib}"',
            ),
            (
                r'sys\.path\.insert\(0,\s*"[^"]+"\)',
                'sys.path.insert(0, "{otilib}/src/python/pyoti/python")',
            ),
            (
                r'(print\(f?"  1\. cd )[^"]+(")',
                r'\g<1>{otilib}/build\g<2>',
            ),
        ],
    ),
    (
        "build/build_static.py",
        [
            (
                r'PROJECT_ROOT\s*=\s*"[^"]+"',
                'PROJECT_ROOT = "{otilib}"',
            ),
        ],
    ),
    (
        "build/rebuild_all_static.py",
        [
            (
                r'BUILD_DIR\s*=\s*"[^"]+"',
                'BUILD_DIR = "{otilib}/build"',
            ),
            (
                r'PYTHON\s*=\s*"[^"]+"',
                'PYTHON = "{python}"',
            ),
        ],
    ),
    (
        "build/rebuild_all_static.sh",
        [
            (
                r'ls [^\s]+/src/c/static/onumm\*\.c',
                'ls {otilib}/src/c/static/onumm*.c',
            ),
        ],
    ),
]


def resolve_otilib(path_arg):
    """Resolve and validate the otilib root directory."""
    p = Path(path_arg).expanduser().resolve()
    if not p.is_dir():
        sys.exit(f"Error: otilib path does not exist: {p}")
    if not (p / "src" / "CMakeLists.txt").exists():
        sys.exit(
            f"Error: {p} does not look like an otilib-master root "
            "(missing src/CMakeLists.txt)"
        )
    return p


def copy_mod_files(mods_dir: Path, otilib: Path):
    print("\n[1/3] Copying mod files to otilib-master...")
    for src_name, dst_rel in FILE_MAP:
        src = mods_dir / src_name
        dst = otilib / dst_rel
        if not src.exists():
            sys.exit(f"Error: mod file not found: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {src_name}  ->  {dst_rel}")
    print("  Done.")


def patch_build_scripts(otilib: Path, python_exe: str):
    print("\n[2/3] Patching hardcoded paths in build scripts...")
    for rel_path, patches in SCRIPT_PATCHES:
        script = otilib / rel_path
        if not script.exists():
            print(f"  WARNING: {rel_path} not found, skipping.")
            continue
        text = script.read_text()
        original = text
        for pattern, template in patches:
            replacement = template.format(otilib=str(otilib), python=python_exe)
            text = re.sub(pattern, replacement, text)
        if text != original:
            script.write_text(text)
            print(f"  Patched: {rel_path}")
        else:
            print(f"  Already up to date: {rel_path}")
    print("  Done.")


def save_config(otilib: Path):
    """Write otilib path to ~/.config/jetgp/otilib_path."""
    JETGP_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    JETGP_CONFIG_FILE.write_text(str(otilib))
    print(f"  Saved otilib path to {JETGP_CONFIG_FILE}")


def install_pyoti_to_path(otilib: Path):
    """
    Make pyoti importable in the active environment by writing a .pth file
    pointing to otilib-master/build/ into site-packages.
    This replicates what `conda develop .` does from the build directory.
    """
    build_dir = str(otilib / "build")
    site_packages = site.getsitepackages()[0]
    pth_file = Path(site_packages) / "otilib.pth"
    pth_file.write_text(build_dir + "\n")
    print(f"  Wrote {pth_file} -> {build_dir}")


def run_build(otilib: Path, workers: int):
    build_dir = otilib / "build"
    python = sys.executable

    # Clean up shipped static module files not needed by JetGP before
    # the bootstrap build (avoids Cythonizing unwanted .pxd headers).
    # Parse ALL_MODULES from the deployed regenerate_all_c.py.
    print("\n  >> Cleaning up unwanted shipped static modules...")
    regen_text = (build_dir / "regenerate_all_c.py").read_text()
    match = re.search(r'ALL_MODULES\s*=\s*\[([^\]]+)\]', regen_text)
    wanted = set()
    if match:
        for m, n in re.findall(r'\((\d+)\s*,\s*(\d+)\)', match.group(1)):
            wanted.add(f"onumm{m}n{n}")
    clean_dirs = [
        otilib / "src" / "c" / "static",
        otilib / "src" / "python" / "pyoti" / "cython" / "static",
        otilib / "include" / "pyoti" / "static",
    ]
    for d in clean_dirs:
        for ext in (".c", ".pyx", ".pxd"):
            for f in d.glob(f"onumm*{ext}"):
                name = f.stem
                if name not in wanted:
                    f.unlink()
                    companion = f.with_suffix("")
                    if companion.is_dir():
                        shutil.rmtree(companion)
                    print(f"  Removed: {f.relative_to(otilib)}")

    # Bootstrap: build pyoti.core first so that regenerate_all_c.py can
    # import cmod_writer (which depends on pyoti.core).
    bootstrap_steps = [
        (
            "Running cmake .. (bootstrap)",
            ["cmake", ".."],
            str(build_dir),
        ),
        (
            f"Running make oticython -j{workers} (bootstrap — builds pyoti.core only)",
            ["make", "oticython", f"-j{workers}"],
            str(build_dir),
        ),
    ]

    print("\n[3/3] Running build workflow...")
    for description, cmd, cwd in bootstrap_steps:
        print(f"\n  >> {description}")
        print(f"     {' '.join(cmd)}  (cwd: {cwd})")
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode != 0:
            sys.exit(f"\nError: step failed: {description}")

    print(f"\n  >> Making pyoti importable in active environment...")
    install_pyoti_to_path(otilib)

    # Now that pyoti.core is available, regenerate sources and rebuild.
    steps = [
        (
            "Regenerating C/Cython sources from templates",
            [python, "regenerate_all_c.py"],
            str(build_dir),
        ),
        (
            "Running cmake .. (picks up regenerated sources)",
            ["cmake", ".."],
            str(build_dir),
        ),
        (
            f"Running make -j{workers}",
            ["make", f"-j{workers}"],
            str(build_dir),
        ),
        (
            "Running make gendata",
            ["make", "gendata"],
            str(build_dir),
        ),
    ]

    for description, cmd, cwd in steps:
        print(f"\n  >> {description}")
        print(f"     {' '.join(cmd)}  (cwd: {cwd})")
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode != 0:
            sys.exit(f"\nError: step failed: {description}")

    print(f"\n  >> Building all Cython static modules ({workers} workers)...")
    result = subprocess.run(
        ["bash", "rebuild_all_static.sh", str(workers)],
        cwd=str(build_dir),
    )
    if result.returncode != 0:
        sys.exit("\nError: static module build failed.")

    print("\n  Build complete. Static modules are in otilib-master/build/pyoti/static/")
    print("  pyoti is importable via the .pth file written to site-packages.")


def main():
    parser = argparse.ArgumentParser(
        description="Patch otilib-master with JetGP modifications and optionally build."
    )
    parser.add_argument(
        "--otilib",
        metavar="PATH",
        help="Path to your otilib-master directory",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run the full build workflow after patching",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for the Cython build step (default: 4)",
    )
    args = parser.parse_args()

    # Locate otilib_mods/ relative to this script (repo root, one level up)
    mods_dir = Path(__file__).parent.parent / "otilib_mods"
    if not mods_dir.is_dir():
        sys.exit(f"Error: otilib_mods/ not found next to this script ({mods_dir})")

    # Resolve otilib path
    if args.otilib:
        otilib = resolve_otilib(args.otilib)
    elif JETGP_CONFIG_FILE.exists():
        candidate = JETGP_CONFIG_FILE.read_text().strip()
        if Path(candidate).is_dir():
            print(f"Using otilib path from config: {candidate}")
            otilib = Path(candidate)
        else:
            sys.exit(
                f"Error: saved otilib path '{candidate}' no longer exists.\n"
                f"Run again with --otilib /path/to/otilib-master"
            )
    else:
        default = Path.home() / "otilib-master"
        prompt = f"Path to otilib-master [{default}]: "
        answer = input(prompt).strip()
        otilib = resolve_otilib(answer if answer else default)

    python_exe = sys.executable
    print(f"\notilib root : {otilib}")
    print(f"Python      : {python_exe}")

    copy_mod_files(mods_dir, otilib)
    patch_build_scripts(otilib, python_exe)
    save_config(otilib)

    if args.build:
        run_build(otilib, args.workers)
    else:
        print(
            "\nPatching complete. To build, run:\n"
            f"  python -m jetgp.setup_otilib --otilib {otilib} --build --workers 8\n"
            "Or manually:\n"
            f"  cd {otilib}/build\n"
            "  python regenerate_all_c.py\n"
            "  cmake .. && make -j$(nproc) && make gendata\n"
            "  conda develop .\n"
            "  bash rebuild_all_static.sh 4"
        )


if __name__ == "__main__":
    main()
