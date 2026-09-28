#!/usr/bin/env python3
"""Wire step 8: capture DSpark tap hiddens in the body forward.

Production's mechanism (deepseek_v4.py:7878-7930) is:

    for i, (layer, layer_cache) in enumerate(zip(self.pipeline_layers, cache)):
        h = layer(h, mask, layer_cache, inputs)
        if _dspark_tap is not None and _global_i in _dspark_tap:
            _DSPARK_CTX["hiddens"][_global_i] = h.mean(axis=2)

i.e. the hc-MEAN of the block's hidden state right after each tapped layer, then
``get_dspark_ctx`` concatenates them in ``dspark_target_layer_ids`` order.

This port does the same thing but returns them from the forward instead of
stashing them in a module-global: ``return_taps=True`` gives
``(logits, {layer_id: hc_mean})``. Default off, so existing callers and timings
are untouched.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

PKG = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/deepseek_v41_mlx")
STAMP = ".pretaps.bak"


def edit(path, old, new, label):
    full = os.path.join(PKG, path)
    src = open(full).read()
    if new in src and old not in src:
        print(f"  [{label}] already applied")
        return True
    if old not in src:
        print(f"  [{label}] !! ANCHOR NOT FOUND")
        for line in old.splitlines()[:10]:
            print("        " + line)
        return False
    bak = full + STAMP
    if not os.path.exists(bak):
        shutil.copy2(full, bak)
    open(full, "w").write(src.replace(old, new, 1))
    print(f"  [{label}] applied")
    return True


ok = True

print("=== model.py: signature ===")
ok &= edit(
    "model.py",
    """    def __call__(self, input_ids: mx.array, cache: ModelCache,
                 last_logit_only: bool = False) -> mx.array:
        \"\"\"input_ids [b, n] continue the sequence at cache.offset. Advances the cache.\"\"\"""",
    """    def __call__(self, input_ids: mx.array, cache: ModelCache,
                 last_logit_only: bool = False, return_taps: bool = False):
        \"\"\"input_ids [b, n] continue the sequence at cache.offset. Advances the cache.

        ``return_taps`` additionally returns ``{layer_id: hc_mean_hidden}`` for
        every layer in ``args.dspark_target_layer_ids`` -- the DSpark context
        feed. Production captures the same quantity (``h.mean(axis=2)`` after
        each tapped layer) into a module-level side channel; returning it is the
        same information without the global.
        \"\"\"""",
    "signature",
)

print()
print("=== model.py: capture point ===")
ok &= edit(
    "model.py",
    """        pre_mix = make_identity_pre_mix(b, n, self.hc_mult)
        shared = SharedState()
        for layer in self.layers:""",
    """        pre_mix = make_identity_pre_mix(b, n, self.hc_mult)
        shared = SharedState()
        tap_ids = set(self.args.dspark_target_layer_ids) if return_taps else set()
        taps = {}
        for layer in self.layers:""",
    "tap init",
)

ok &= edit(
    "model.py",
    """                h, pre_mix = layer(h, pre_mix, start_pos, cache, shared_use)
            else:
                h, pre_mix = layer(h, pre_mix, start_pos, cache, shared)

        h = hc_pre(h, pre_mix)                       # collapse with the last ffn_pre""",
    """                h, pre_mix = layer(h, pre_mix, start_pos, cache, shared_use)
            else:
                h, pre_mix = layer(h, pre_mix, start_pos, cache, shared)
            if layer.layer_id in tap_ids:
                taps[layer.layer_id] = h.mean(axis=2)

        h = hc_pre(h, pre_mix)                       # collapse with the last ffn_pre""",
    "capture",
)

print()
print("=== model.py: return ===")
ok &= edit(
    "model.py",
    """        logits = self.head(h.astype(mx.float32))   # fp32 logits, as the reference
        cache.offset = start_pos + n
        return logits""",
    """        logits = self.head(h.astype(mx.float32))   # fp32 logits, as the reference
        cache.offset = start_pos + n
        if return_taps:
            return logits, taps
        return logits""",
    "return",
)

print()
r = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(PKG, "model.py")],
                   capture_output=True, text=True)
print(f"model.py: {'OK' if r.returncode == 0 else r.stderr[:400]}")
ok &= r.returncode == 0

if ok:
    # confirm no existing caller passes a 4th positional arg
    src = open(os.path.join(PKG, "model.py")).read()
    print()
    print("=== tap capture present in file? ===")
    for line in src.splitlines():
        if "tap_ids" in line or "return_taps" in line or "taps[layer.layer_id]" in line:
            print(f"    {line.rstrip()}")
    print()
    for f in ("generate.py", "load.py", "mtp.py"):
        s = open(os.path.join(PKG, f)).read()
        hits = [l.strip() for l in s.splitlines() if "last_logit_only" in l]
        print(f"  {f}: {len(hits)} call sites mentioning last_logit_only")

print()
print("done" if ok else "FAILED")
sys.exit(0 if ok else 1)
