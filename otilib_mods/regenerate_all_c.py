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
import glob

# Modules required to build documentation and run getting-started examples.
# Additional modules can be compiled on demand via JetGP's auto_compile feature.
ALL_MODULES = [
    (1,2), (1,4), (1,6), (1,8),
    (2,2), (2,4), (2,6),
    (3,2), (3,4),
    (4,2), (4,4),
    (6,2),
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

        # Clean up shipped static sources not in ALL_MODULES (only for default builds)
        wanted = {f"onumm{nb}n{od}" for nb, od in modules}
        static_dir = os.path.join(BASE_DIR, "src", "c", "static")
        for f in glob.glob(os.path.join(static_dir, "onumm*.c")):
            name = os.path.splitext(os.path.basename(f))[0]
            if name not in wanted:
                companion_dir = os.path.join(static_dir, name)
                os.remove(f)
                if os.path.isdir(companion_dir):
                    import shutil
                    shutil.rmtree(companion_dir)
                print(f"  Removed unwanted static source: {name}")

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
