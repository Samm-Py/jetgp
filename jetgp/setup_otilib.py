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
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# File map: list of (source relative to otilib_mods/, destination relative to otilib root)
# A source file may appear more than once to copy it to multiple destinations.
# ---------------------------------------------------------------------------
FILE_MAP = [
    ("src_CMakeLists.txt",             "src/CMakeLists.txt"),
    ("src_python_pyoti_CMakeLists.txt", "src/python/pyoti/CMakeLists.txt"),
    ("regenerate_all_c.py",            "build/regenerate_all_c.py"),
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
]

# ---------------------------------------------------------------------------
# Build scripts whose hardcoded paths need to be rewritten
# ---------------------------------------------------------------------------
# Each entry: (path relative to otilib root, list of (pattern, replacement_template))
# Replacement templates may contain {otilib} and {python} placeholders.
SCRIPT_PATCHES = [
    (
        "build/regenerate_all_c.py",
        [
            # BASE_DIR = "/hard/coded/path"
            (
                r'BASE_DIR\s*=\s*"[^"]+"',
                'BASE_DIR = "{otilib}"',
            ),
            # sys.path.insert(0, "/hard/coded/path/src/python/pyoti/python")
            (
                r'sys\.path\.insert\(0,\s*"[^"]+"\)',
                'sys.path.insert(0, "{otilib}/src/python/pyoti/python")',
            ),
            # print lines that echo the build dir
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
            # ls /hard/coded/path/src/c/static/onumm*.c
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
    # Sanity check — look for a known file
    if not (p / "src" / "CMakeLists.txt").exists():
        sys.exit(
            f"Error: {p} does not look like an otilib-master root "
            "(missing src/CMakeLists.txt)"
        )
    return p


def copy_mod_files(mods_dir: Path, otilib: Path):
    print("\n[1/3] Copying mod files...")
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


def run_build(otilib: Path, workers: int):
    build_dir = otilib / "build"
    python = sys.executable

    steps = [
        (
            "Regenerating C/Cython sources from templates",
            [python, "regenerate_all_c.py"],
            str(build_dir),
        ),
        (
            "Running cmake ..",
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
        (
            f"Building all Cython static modules ({workers} workers)",
            ["bash", "rebuild_all_static.sh", str(workers)],
            str(build_dir),
        ),
    ]

    print("\n[3/3] Running build workflow...")
    for description, cmd, cwd in steps:
        print(f"\n  >> {description}")
        print(f"     {' '.join(cmd)}  (cwd: {cwd})")
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode != 0:
            sys.exit(f"\nError: step failed: {description}")
    print("\n  Build complete.")


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

    # Locate otilib_mods/ relative to this script (sits at repo root, one level up)
    mods_dir = Path(__file__).parent.parent / "otilib_mods"
    if not mods_dir.is_dir():
        sys.exit(f"Error: otilib_mods/ not found next to this script ({mods_dir})")

    # Resolve otilib path — prompt if not given
    if args.otilib:
        otilib = resolve_otilib(args.otilib)
    else:
        default = Path.home() / "research_head" / "otilib-master"
        prompt = f"Path to otilib-master [{default}]: "
        answer = input(prompt).strip()
        otilib = resolve_otilib(answer if answer else default)

    python_exe = sys.executable
    print(f"\notilib root : {otilib}")
    print(f"Python      : {python_exe}")

    copy_mod_files(mods_dir, otilib)
    patch_build_scripts(otilib, python_exe)

    if args.build:
        run_build(otilib, args.workers)
    else:
        print(
            "\nPatching complete. To build, run:\n"
            f"  python setup_otilib.py --otilib {otilib} --build [--workers N]\n"
            "Or manually:\n"
            f"  cd {otilib}/build\n"
            "  python regenerate_all_c.py\n"
            "  cmake .. && make -j$(nproc) && make gendata\n"
            "  bash rebuild_all_static.sh 4"
        )


if __name__ == "__main__":
    main()
