#!/usr/bin/env python3
"""Wire step 7: fix two mapping bugs found by the strict loader.

BUG 1 -- shared_experts must NOT be renamed.
  I translated `ffn.shared_experts.w1` -> `gate_proj` (copied from production's
  own sanitizer). But this PORT's SharedExpert module owns `w1/w2/w3`, and the
  port's converter does not rename them. Result: the head expected `w1.weight`
  while the mapper delivered `gate_proj.weight` -- 27 params missing, 27
  unexpected. Fix: drop the shared_experts rename entirely.

BUG 2 -- module-level tensors only matched their exact `.weight` name.
  `mtp.2.markov_head.embed.scales` was not in the table (only .weight was), so
  it fell through to `mtp.stages.2.markov_head.embed.scales` and the head's
  `mtp.markov_embed.scales` never arrived. Fix: match module-level tensors by
  PREFIX so weight/scales/biases all route correctly.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/deepseek_v41_mlx")
STAMP = ".premtpmap3.bak"


def edit(path, old, new, label):
    full = os.path.join(REPO, path)
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

print("=== load.py: rewrite the mapping tables + function ===")
ok &= edit(
    "load.py",
    """# draft-stack projections use the body's w1/w3/w2 names
_MTP_PROJ = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
# module-level (non-stage-scoped) draft tensors
_MTP_TOP = {
    "main_proj.weight": "mtp.main_proj.weight",
    "main_norm.weight": "mtp.main_norm.weight",
    "norm.weight": "mtp.norm.weight",
    "markov_head.embed.weight": "mtp.markov_embed.weight",
    "markov_head.head.weight": "mtp.markov_head.weight",
    "confidence_head.proj.weight": "mtp.confidence_proj.weight",
}""",
    """# Module-level (not stage-scoped) draft sub-modules, as PREFIX -> runtime
# prefix. Matched by prefix so .weight/.scales/.biases all route the same way
# (the draft's embeddings and heads are quantized, so scales/biases exist).
#
# NOTE: shared_experts keep their w1/w2/w3 names -- this port's SharedExpert
# owns w1/w2/w3 and its converter does not rename them. (Production's own
# sanitizer does rename them; that is a different codebase.)
_MTP_TOP_PREFIX = {
    "main_proj.": "mtp.main_proj.",
    "main_norm.": "mtp.main_norm.",
    "norm.": "mtp.norm.",
    "markov_head.embed.": "mtp.markov_embed.",
    "markov_head.head.": "mtp.markov_head.",
    "confidence_head.proj.": "mtp.confidence_proj.",
}
# bare module-path forms (no trailing component) for the quant module map
_MTP_TOP_M = {
    "main_proj": "mtp.main_proj",
    "main_norm": "mtp.main_norm",
    "norm": "mtp.norm",
    "markov_head.embed": "mtp.markov_embed",
    "markov_head.head": "mtp.markov_head",
    "confidence_head.proj": "mtp.confidence_proj",
}""",
    "tables",
)

ok &= edit(
    "load.py",
    """    parts = k.split(".")
    if len(parts) < 3:
        return None
    stage, tail = parts[1], ".".join(parts[2:])
    for w, p in _MTP_PROJ.items():
        tail = tail.replace(f"ffn.shared_experts.{w}.", f"ffn.shared_experts.{p}.")
    has_weight = tail.endswith(".weight")
    bare = tail[:-len(".weight")] if has_weight else tail
    if bare in _MTP_TOP_M:
        return _MTP_TOP_M[bare] + (".weight" if has_weight else "")
    return f"mtp.stages.{stage}.{tail}\"""",
    """    parts = k.split(".")
    if len(parts) < 3:
        return None
    stage, tail = parts[1], ".".join(parts[2:])
    # module-level tensors, by prefix (covers .weight/.scales/.biases)
    for pre, repl in _MTP_TOP_PREFIX.items():
        if tail.startswith(pre):
            return repl + tail[len(pre):]
    if tail in _MTP_TOP_M:
        return _MTP_TOP_M[tail]
    return f"mtp.stages.{stage}.{tail}\"""",
    "mapper",
)

print()
r = subprocess.run([sys.executable, "-m", "py_compile",
                    os.path.join(REPO, "load.py")], capture_output=True, text=True)
print(f"load.py: {'OK' if r.returncode == 0 else r.stderr[:400]}")
ok &= r.returncode == 0

if ok:
    sys.path.insert(0, os.path.dirname(REPO))
    from deepseek_v41_mlx.load import map_mtp_name as m
    print()
    print("=== mapping (final) ===")
    for c in [
        "mtp.0.ffn.shared_experts.w1.weight",
        "mtp.0.ffn.shared_experts.w1.scales",
        "mtp.2.markov_head.embed.weight",
        "mtp.2.markov_head.embed.scales",
        "mtp.2.markov_head.head.weight",
        "mtp.2.markov_head.head.biases",
        "mtp.2.confidence_head.proj.weight",
        "mtp.0.main_proj.scales",
        "mtp.0.main_norm.weight",
        "mtp.2.norm.weight",
        "mtp.0.ffn.experts.gate_proj",
        "mtp.2.markov_head.embed",     # module-path form
        "mtp.2.confidence_head.proj",  # module-path form
        "mtp.stages.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.weight",
    ]:
        print(f"    {c:<44} -> {m(c)}")

print()
print("done" if ok else "FAILED")
sys.exit(0 if ok else 1)
