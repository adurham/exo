#!/usr/bin/env python3
"""RED/GREEN proof for the DSpark guard fix.

Runs the SAME assertions against:
  (a) the OLD comparison (raw set-diff of quantized vs unquantized)  -> must FAIL
  (b) the NEW comparison (quant-normalized on both sides)           -> must PASS

If (a) passes, the test does not actually capture the bug.
If (b) fails, the fix does not work.
"""
from __future__ import annotations

import sys

# --- the fixture, identical to the regression test ---------------------
_UNQUANTIZED = {
    "main_proj.weight", "main_norm.weight", "norm.weight",
    "markov_embed.weight", "markov_head.weight", "confidence_proj.weight",
    "stages.0.attn.wkv.weight", "stages.0.attn.wo_a.weight",
    "stages.0.attn.wo_b.weight", "stages.0.attn.wq_a.weight",
    "stages.0.attn.wq_b.weight", "stages.0.attn.kv_norm.weight",
    "stages.0.attn.q_norm.weight",
    "stages.0.ffn.shared_experts.down_proj.weight",
    "stages.0.ffn.shared_experts.gate_proj.weight",
    "stages.0.ffn.shared_experts.up_proj.weight",
    "stages.0.ffn.experts.gate_proj.weight",
    "stages.0.ffn.experts.up_proj.weight",
    "stages.0.ffn.experts.down_proj.weight",
    "stages.0.norm.weight",
    "stages.1.attn.wkv.weight",
    "stages.1.ffn.shared_experts.gate_proj.weight",
    "stages.2.attn.wkv.weight",
    "stages.2.ffn.shared_experts.down_proj.weight",
}


def _loaded_side():
    out = set(_UNQUANTIZED)
    for k in _UNQUANTIZED:
        if k.endswith(".weight") and not k.endswith(("_norm.weight", ".norm.weight")):
            if (".attn." in k or "shared_experts" in k or "main_proj" in k
                    or "markov" in k or "confidence" in k):
                out.add(k[: -len(".weight")] + ".scales")
    return out


def _norm_quant(keys, _suffixes=(".scales", ".biases")):
    out, n = set(), 0
    for k in keys:
        for sfx in _suffixes:
            if k.endswith(sfx):
                out.add(k[: -len(sfx)] + ".weight")
                n += 1
                break
        else:
            out.add(k)
    return out, n


def old_compare(loaded, expected):
    """PRE-FIX: raw diff, no normalization."""
    missing = expected - loaded
    extra = loaded - expected
    return (not missing and not extra), missing, extra


def new_compare(loaded, expected):
    """POST-FIX: normalize both sides."""
    ln, _ = _norm_quant(loaded)
    en, _ = _norm_quant(expected)
    missing, extra = en - ln, ln - en
    return (not missing and not extra), missing, extra


loaded = _loaded_side()
expected = set(_UNQUANTIZED)
print(f"fixture: loaded={len(loaded)} expected={len(expected)} "
      f"(diff={len(loaded)-len(expected)})")
print(f"  all raw extras are quant keys: "
      f"{all(k.endswith(('.scales', '.biases')) for k in loaded - expected)}")
print()

fails = 0

print("=== (a) OLD logic against the regression assertion ===")
ok_old, miss_old, extra_old = old_compare(loaded, expected)
print(f"  tree_ok = {ok_old}")
print(f"  missing = {len(miss_old)}  extra = {len(extra_old)}")
if ok_old:
    print("  !! RED CHECK FAILED: old logic passes, so the test does not capture the bug")
    fails += 1
else:
    print("  RED OK: old logic reports FAIL (the bug is captured)")

print()
print("=== (b) NEW logic against the same assertion ===")
ok_new, miss_new, extra_new = new_compare(loaded, expected)
print(f"  tree_ok = {ok_new}")
print(f"  missing = {sorted(miss_new)}")
print(f"  extra   = {sorted(extra_new)}")
if ok_new:
    print("  GREEN OK: new logic reports PASS")
else:
    print("  !! GREEN FAILED: the fix does not clear the false alarm")
    fails += 1

print()
print("=== (c) NEW logic must still catch GENUINE mismatches ===")
for label, l, e in (
    ("genuine extra", loaded | {"stages.0.attn.wq_z.weight"}, expected),
    ("genuine missing", loaded, expected | {"stages.0.attn.wq_z.weight"}),
):
    ok, m, x = new_compare(l, e)
    print(f"  {label:<16} tree_ok={ok}  missing={len(m)} extra={len(x)}")
    if ok:
        print(f"  !! {label} was NOT detected — the fix over-normalizes")
        fails += 1

print()
print("RESULT:", "ALL CHECKS PASS" if fails == 0 else f"{fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
