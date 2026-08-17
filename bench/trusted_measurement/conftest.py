"""Make ``trusted_measurement`` importable when pytest is run from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

_BENCH_DIR = str(Path(__file__).resolve().parent.parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
