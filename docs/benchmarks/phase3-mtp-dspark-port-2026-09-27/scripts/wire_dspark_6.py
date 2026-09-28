#!/usr/bin/env python3
"""Wire step 6: fix the module-path (no .weight) branch of map_mtp_name.

Observed:
    mtp.0.main_proj        -> mtp.main_proj.weight   WRONG (should be mtp.main_proj)
    mtp.2.confidence_head.proj -> mtp.confidence_proj.weight   WRONG

Cause: `_MTP_TOP_M` was built by stripping only the KEY's suffix, leaving the
VALUE with ".weight". quant_predicate looks up module paths, so a value ending
in ".weight" never matches a module. Fix: make the table map bare -> bare, and
append ".weight" only when the input had it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/deepseek_v41_mlx")
STAMP = ".premtpmap2.bak"


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

print("=== load.py: bare -> bare table ===")
ok &= edit(
    "load.py",
    """# module-level draft modules, without the ".weight" suffix (module-map keys)
_MTP_TOP_M = {k[:-len(".weight")]: v for k, v in _MTP_TOP.items()}
# markov_head.embed / .head appear as module paths too
_MTP_TOP_M["markov_head.embed"] = "mtp.markov_embed"
_MTP_TOP_M["markov_head.head"] = "mtp.markov_head\"""",
    """# module-level draft modules, keyed AND valued WITHOUT ".weight", so the
# same table serves both checkpoint tensor names and module-map paths.
_MTP_TOP_M = {
    "main_proj": "mtp.main_proj",
    "main_norm": "mtp.main_norm",
    "norm": "mtp.norm",
    "markov_head.embed": "mtp.markov_embed",
    "markov_head.head": "mtp.markov_head",
    "confidence_head.proj": "mtp.confidence_proj",
}""",
    "bare table",
)

ok &= edit(
    "load.py",
    """    # module paths carry no ".weight"; the lookup table does, so try both
    if tail in _MTP_TOP:
        return _MTP_TOP[tail]
    bare = tail[:-len(".weight")] if tail.endswith(".weight") else tail
    if bare in _MTP_TOP_M:
        return _MTP_TOP_M[bare]
    return f"mtp.stages.{stage}.{tail}\"""",
    """    has_weight = tail.endswith(".weight")
    bare = tail[:-len(".weight")] if has_weight else tail
    if bare in _MTP_TOP_M:
        return _MTP_TOP_M[bare] + (".weight" if has_weight else "")
    return f"mtp.stages.{stage}.{tail}\"""",
    "bare branch",
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
    print("=== mapping behaviour (final) ===")
    for c in [
        "mtp.0.attn.wq_a.weight",
        "mtp.0.ffn.shared_experts.w3.biases",
        "mtp.0.main_proj.weight",
        "mtp.0.main_proj",
        "mtp.2.markov_head.embed.weight",
        "mtp.2.markov_head.embed",
        "mtp.2.markov_head.head",
        "mtp.2.confidence_head.proj",
        "mtp.0.norm.weight",
        "mtp.0.norm",
        "mtp.0.ffn.experts.gate_proj",
        "mtp.2.ffn.gate",
        "mtp.stages.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.weight",
    ]:
        print(f"    {c:<44} -> {m(c)}")

print()
print("done" if ok else "FAILED")
sys.exit(0 if ok else 1)
