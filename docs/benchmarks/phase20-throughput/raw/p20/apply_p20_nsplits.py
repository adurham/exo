#!/usr/bin/env python3
"""apply_p20_nsplits.py -- apply the P20 split-K probe hook to a gemv_metal.py copy.

The hook is a CALL-TIME env override of the auto n_splits loop in _run_inner_gem
(gemv_metal.py L1438-1443).  Stock behaviour when the env vars are unset.  No kernel
source is touched (n_splits is a runtime dims value, not a compile key) -> no recompile,
and the resulting numerics change only in the partial-sum ORDER (battery-gated).
"""
import sys

PATH = sys.argv[1]
SRC = open(PATH).read()

OLD = """    n_splits = 1
    while (
        out_tiles * m_groups * n_splits < 8192
        and in_tiles // (n_splits * 2) >= min_split_tiles
    ):
        n_splits *= 2
"""
NEW = """    n_splits = 1
    while (
        out_tiles * m_groups * n_splits < 8192
        and in_tiles // (n_splits * 2) >= min_split_tiles
    ):
        n_splits *= 2
    # --- P20 split-K probe hook: CALL-TIME env override (stock when unset) ---
    _p20_ms = int(os.environ.get("P20_MIN_SPLIT_TILES", "0"))
    _p20_x = int(os.environ.get("P20_XSPLIT", "0"))
    if _p20_x > 0:
        _p20_ms = max(1, in_tiles // _p20_x)
    if _p20_ms > 0:
        n_splits = 1
        while in_tiles // (n_splits * 2) >= _p20_ms:
            n_splits *= 2
    # --- end P20 hook ---
"""

assert SRC.count(OLD) == 1, f"anchor count = {SRC.count(OLD)} (need exactly 1)"
open(PATH, "w").write(SRC.replace(OLD, NEW))
print("patched", PATH)
