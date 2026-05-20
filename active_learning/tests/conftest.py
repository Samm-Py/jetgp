"""Pytest configuration for active-learning unit tests."""

import os
import sys
from pathlib import Path


# Importing JetGP in this local checkout can fail during numba cache setup.
# Disabling JIT is enough for import-level tests and does not affect the small
# fake-object unit tests in this directory.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

ACTIVE_LEARNING_DIR = Path(__file__).resolve().parents[1]
if str(ACTIVE_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVE_LEARNING_DIR))
