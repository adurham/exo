#!/usr/bin/env python3
"""apply_v4_alu_optim.py -- make the two bit-identical kernel wins permanent.

Applies to the PonyExl3 working tree on node 1:

  gemv_metal.py
    _decode_expr: env-gated EXL3_DECODE_SWAR (default 1) -- the mul1 codebook
    decode's 4-iteration byte-sum loop becomes two packed half-sums (SWAR).
    Algebraically identical (sum of 4 bytes, no overflow); md5-verified.

  exl3_moe.py
    The four MoE kernel source builders get wrapped so that
    ``simd_shuffle(x_lane, ushort(row_[jw]))`` becomes a direct indexed read
    of the same element (env-gated EXL3_XDIRECT, default 1). Lane s's x_lane
    holds x[..., (s & 15)]; the shuffle broadcasts lane row_[jw]'s value =
    x[..., (row_[jw] & 15)]. Byte-identical; md5-verified.

Both changes measured on node 1, E=384, layer 1 (2026-09-28):
    control        R=1 1.031 ms (2.650x)   R=4 3.169 (3.052)  R=8 6.043 (3.082)
    DIRECT+SWAR    R=1 0.756 ms (1.937x)   R=4 2.088 (1.986)  R=8 3.941 (2.015)
    md5(py output) identical at R=1/4/6/8.

Usage: python3 apply_v4_alu_optim.py [--revert]
"""
from __future__ import annotations

import hashlib
import os
import py_compile
import re
import shutil
import subprocess
import sys

PKG = os.path.expanduser("~/repos/ref/PonyExl3/ponyexl3/mlx")
GM = os.path.join(PKG, "gemv_metal.py")
MOE = os.path.join(PKG, "exl3_moe.py")
REVERT = "--revert" in sys.argv


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def read(p):
    return open(p).read()


def write(p, s):
    open(p, "w").write(s)


print(f"[pre] gemv_metal.py md5={md5(GM)}")
print(f"[pre] exl3_moe.py  md5={md5(MOE)}")

if REVERT:
    for p in (GM, MOE):
        bak = p + ".bak-v4"
        if os.path.exists(bak):
            shutil.copy(bak, p)
            print(f"[revert] {p} restored from .bak-v4")
        else:
            print(f"[revert] no backup for {p}")
    sys.exit(0)

# backups (from the CURRENT, i.e. v3-patched, files)
for p in (GM, MOE):
    bak = p + ".bak-v4"
    if not os.path.exists(bak):
        shutil.copy(p, bak)
        print(f"[backup] {bak}")

# ---------------------------------------------------------------- gemv_metal
src = read(GM)
MARK = "_EXL3_DECODE_SWAR"
if MARK in src:
    print("[gemv] SWAR gate already present -- skipping")
else:
    anchor = '            float dq_val = float(dq_h * dq_k_inv + dq_k_bias);\n"""\n\n\n_HAD_SCALE_LIT'
    if src.count(anchor) != 1:
        sys.exit(f"[gemv] anchor count = {src.count(anchor)} (want 1); aborting")
    insert = '''            float dq_val = float(dq_h * dq_k_inv + dq_k_bias);
"""


# --- phase 9 spike: env-gated, bit-identical mul1 decode (EXL3_DECODE_SWAR,
# default on). The 4-iteration byte-sum loop becomes two packed half-sums;
# integer addition is associative and nothing overflows (max 0x6400 + 4*0xFF).
# Verified md5-identical to the loop form on node 1, E=384, R=1/4/6/8;
# measured R=1 1.031 -> 1.000 ms alone (see phase9 doc).
_EXL3_DECODE_SWAR = __import__("os").environ.get("EXL3_DECODE_SWAR", "1") == "1"
_decode_expr_stock = _decode_expr


def _decode_expr(cb: CodebookMode, *, cw_in: str = "cw") -> str:  # noqa: F811
    """Env-gated mul1 fast path; otherwise the stock implementation."""
    if _EXL3_DECODE_SWAR and cb == CodebookMode.MUL1:
        return f"""
            uint dq_cw = {cw_in} * 0x83DCD12Du;
            uint dq_t = (dq_cw & 0x00FF00FFu) + ((dq_cw >> 8u) & 0x00FF00FFu);
            uint dq_sum = 0x6400u + (dq_t & 0xFFFFu) + (dq_t >> 16u);
            half dq_h = as_type<half>(ushort(dq_sum & 0xFFFFu));
            half dq_k_inv = as_type<half>(ushort(0x1EEEu));
            half dq_k_bias = as_type<half>(ushort(0xC931u));
            float dq_val = float(dq_h * dq_k_inv + dq_k_bias);
"""
    return _decode_expr_stock(cb, cw_in=cw_in)


_HAD_SCALE_LIT'''
    src = src.replace(anchor, insert, 1)
    write(GM, src)
    print("[gemv] SWAR gate inserted")

# ------------------------------------------------------------------ exl3_moe
src = read(MOE)
MARK = "_XDIRECT"
if MARK in src:
    print("[moe] XDIRECT gate already present -- skipping")
else:
    anchor = "\n\nclass EXL3SwitchGLU("
    if src.count(anchor) != 1:
        sys.exit(f"[moe] anchor count = {src.count(anchor)} (want 1); aborting")
    insert = '''

# --- phase 9 spike: env-gated, bit-identical cross-lane removal
# (EXL3_XDIRECT, default on). ``simd_shuffle(x_lane, ushort(row_[jw]))``
# broadcasts lane row_[jw]'s x_lane, which holds x[..., (row_[jw] & 15)] --
# so a direct indexed read computes the same fp16 value with no cross-lane
# op. md5-identical over R=1/4/6/8; measured R=1 1.031 -> 0.823 ms (2.65x ->
# 2.11x vs production), R=4-8 -> ~2.25-2.33x (see phase9 doc).
_XDIRECT = __import__("os").environ.get("EXL3_XDIRECT", "1") == "1"


def _xdirect_src(src: str) -> str:
    """Rewrite shuffle-broadcast sites to direct indexed reads."""
    import re as _re
    if not _XDIRECT:
        return src
    exprs = _re.findall(r"float x_lane = (.+);", src)
    if not exprs:
        return src
    it = iter(exprs)

    def _sub(_m):
        return next(it).replace("(lane & 15u)", "(row_[jw] & 15u)")
    out, n = _re.subn(
        _re.escape("simd_shuffle(x_lane, ushort(row_[jw]))"), _sub, src)
    if n != len(exprs):
        raise ValueError(
            f"xdirect: {n} shuffle sites vs {len(exprs)} x_lane defs")
    return out


if _XDIRECT:
    def _mk_xdirect(_fn):
        def _wrapped(*a, **kw):
            return _xdirect_src(_fn(*a, **kw))
        return _wrapped
    for _n in ("_moe_gateup_source", "_moe_down_source",
               "_moe_gateup2_source", "_moe_down2_source"):
        globals()[_n] = _mk_xdirect(globals()[_n])


class EXL3SwitchGLU('''
    src = src.replace(anchor, insert, 1)
    write(MOE, src)
    print("[moe] XDIRECT gate inserted")

# ------------------------------------------------------------------- verify
try:
    py_compile.compile(GM, doraise=True)
    py_compile.compile(MOE, doraise=True)
    print("[verify] both files compile")
except py_compile.PyCompileError as e:
    sys.exit(f"[verify] COMPILE FAILED: {e}")

print(f"[post] gemv_metal.py md5={md5(GM)}")
print(f"[post] exl3_moe.py  md5={md5(MOE)}")

# patch file
r = subprocess.run(
    ["diff", "-u", "gemv_metal.py.bak-v4", "gemv_metal.py"],
    cwd=PKG, capture_output=True, text=True)
r2 = subprocess.run(
    ["diff", "-u", "exl3_moe.py.bak-v4", "exl3_moe.py"],
    cwd=PKG, capture_output=True, text=True)
patch = (r.stdout + r2.stdout)
out = os.path.expanduser("~/exl3-moe-v4-alu-optim.patch")
open(out, "w").write(patch)
print(f"[patch] {out} ({len(patch)} chars, "
      f"{patch.count(chr(10))} lines)")
print("APPLY-V4-DONE")
