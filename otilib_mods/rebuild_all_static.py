#!/usr/bin/env python3
"""Rebuild all static onumm modules sequentially."""
import subprocess
import sys
import os
import re
import glob

BUILD_DIR = "/home/sam/research_head/otilib-master/build"
SO_DIR = os.path.join(BUILD_DIR, "pyoti", "static")
PYTHON = "/home/sam/anaconda3/envs/pyoti_2/bin/python"

# Find all built modules
so_files = glob.glob(os.path.join(SO_DIR, "onumm*.so"))
modules = []
for f in so_files:
    name = os.path.basename(f).split(".")[0]  # e.g. onumm10n2
    match = re.match(r'onumm(\d+)n(\d+)', name)
    if match:
        modules.append(f"m{match.group(1)}n{match.group(2)}")

modules.sort()
print(f"Rebuilding {len(modules)} modules...")

failed = []
for i, mn in enumerate(modules):
    print(f"\n[{i+1}/{len(modules)}] Building {mn}...")
    result = subprocess.run(
        [PYTHON, "build_static.py", mn],
        cwd=BUILD_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  FAILED!")
        print(result.stderr[-500:] if result.stderr else "no stderr")
        failed.append(mn)
    else:
        print(f"  OK")

print(f"\n{'='*60}")
if failed:
    print(f"FAILED ({len(failed)}): {', '.join(failed)}")
else:
    print(f"All {len(modules)} modules rebuilt successfully!")
