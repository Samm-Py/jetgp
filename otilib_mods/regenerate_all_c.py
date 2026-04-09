#!/usr/bin/env python3
"""
Regenerate all static module C/H/Cython files from templates.
Usage: python regenerate_all_c.py [mXnY ...]
  No args = regenerate all modules
  With args = regenerate only specified modules (e.g., m20n2 m10n2)
"""
import sys
import re
import os

# All modules: (nbases, order)
ALL_MODULES = [
    (1,1), (1,2), (1,3), (1,4), (1,6), (1,8), (1,10), (1,20), (1,30), (1,66),
    (2,1), (2,2), (2,3), (2,4), (2,6),
    (3,1), (3,2), (3,3), (3,4), (3,8),
    (4,1), (4,2), (4,3), (4,4),
    (5,1), (5,2), (5,3), (5,4),
    (6,1), (6,2), (6,3), (6,4),
    (7,1), (7,2), (7,3), (7,4),
    (8,1), (8,2), (8,3), (8,4),
    (9,1), (9,2), (9,3), (9,4),
    (10,1), (10,2), (10,3), (10,4),
    (15,1), (15,2), (15,3), (15,4),
    (20,2), (30,2), (50,2), (51,2), (60,2), (61,2),
]

BASE_DIR = "/home/sam/research_head/otilib-master"

def parse_mn(s):
    m = re.match(r'm(\d+)n(\d+)', s)
    if not m:
        print(f"Error: invalid format '{s}', expected mXnY")
        sys.exit(1)
    return int(m.group(1)), int(m.group(2))

if __name__ == "__main__":
    sys.path.insert(0, "/home/sam/research_head/otilib-master/src/python/pyoti/python")
    from cmod_writer import writer

    if len(sys.argv) > 1:
        modules = [parse_mn(a) for a in sys.argv[1:]]
    else:
        modules = ALL_MODULES

    print(f"Regenerating {len(modules)} modules...")
    for nbases, order in modules:
        name = f"m{nbases}n{order}"
        print(f"\n{'='*60}")
        print(f"  Generating {name} (nbases={nbases}, order={order})")
        print(f"{'='*60}")
        w = writer(nbases, order)
        w.write_files(base_dir=BASE_DIR)
        print(f"  Done: {name}")

    print(f"\nAll {len(modules)} modules regenerated.")
    print("Next steps:")
    print("  1. cd /home/sam/research_head/otilib-master/build")
    print("  2. cmake .. && make -j$(nproc)")
    print("  3. bash rebuild_all_static.sh 4")
