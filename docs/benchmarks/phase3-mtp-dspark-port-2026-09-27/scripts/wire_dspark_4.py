#!/usr/bin/env python3
"""Wire step 4: build the DSpark head in Model and map its checkpoint names.

The build stores BARE names (`layers.0.attn.wq_a.weight`) and load.py passes
them straight to `model.load_weights(items)`, so the mapping must also produce
bare names relative to the Model root. Name translation:

  mtp.{i}.attn.wq_a.weight        -> mtp.stages.{i}.attn.wq_a.weight
  mtp.{i}.ffn.experts.{p}.weight  -> mtp.stages.{i}.ffn.experts.{proj}.weight
  mtp.{i}.ffn.shared_experts.w1.  -> mtp.stages.{i}.ffn.shared_experts.gate_proj.
  mtp.{i}.hc_attn_fn              -> mtp.stages.{i}.hc_attn_fn
  mtp.0.main_proj.weight          -> mtp.main_proj.weight
  mtp.0.main_norm.weight          -> mtp.main_norm.weight
  mtp.2.norm.weight               -> mtp.norm.weight
  mtp.2.markov_head.embed.weight  -> mtp.markov_embed.weight
  mtp.2.markov_head.head.weight   -> mtp.markov_head.weight
  mtp.2.confidence_head.proj.weight -> mtp.confidence_proj.weight
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/deepseek_v41_mlx")
STAMP = ".predsparkmodel.bak"


def edit(path, old, new, label):
    full = os.path.join(REPO, path)
    src = open(full).read()
    if new in src and old not in src:
        print(f"  [{label}] already applied")
        return True
    if old not in src:
        print(f"  [{label}] !! ANCHOR NOT FOUND")
        for line in old.splitlines()[:8]:
            print("        " + line)
        return False
    bak = full + STAMP
    if not os.path.exists(bak):
        shutil.copy2(full, bak)
    open(full, "w").write(src.replace(old, new, 1))
    print(f"  [{label}] applied")
    return True


ok = True

print("=== model.py: build the head ===")
ok &= edit(
    "model.py",
    """        self.engram_hasher = None
        if args.engram_layer_ids and token_map is not None:
            self.set_token_map(token_map)""",
    """        self.mtp = None
        if args.n_mtp_layers:
            from .mtp import DSparkHead
            self.mtp = DSparkHead(args)
        self.engram_hasher = None
        if args.engram_layer_ids and token_map is not None:
            self.set_token_map(token_map)""",
    "build head",
)

print()
print("=== load.py: name mapping ===")
ok &= edit(
    "load.py",
    """from .config import ModelArgs
from .convert import VISION_PREFIXES, bits_for, is_quant_target
from .model import Model""",
    """from .config import ModelArgs
from .convert import VISION_PREFIXES, bits_for, is_quant_target
from .model import Model

# draft-stack projections use the body's w1/w3/w2 names
_MTP_PROJ = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
# module-level (non-stage-scoped) draft tensors
_MTP_TOP = {
    "main_proj.weight": "mtp.main_proj.weight",
    "main_norm.weight": "mtp.main_norm.weight",
    "norm.weight": "mtp.norm.weight",
    "markov_head.embed.weight": "mtp.markov_embed.weight",
    "markov_head.head.weight": "mtp.markov_head.weight",
    "confidence_head.proj.weight": "mtp.confidence_proj.weight",
}


def map_mtp_name(k: str):
    \"\"\"Checkpoint ``mtp.*`` name -> bare module path, or None if unrelated.\"\"\"
    if not k.startswith("mtp."):
        return None
    parts = k.split(".")
    if len(parts) < 3:
        return None
    stage, tail = parts[1], ".".join(parts[2:])
    for w, p in _MTP_PROJ.items():
        tail = tail.replace(f"ffn.shared_experts.{w}.", f"ffn.shared_experts.{p}.")
    if tail in _MTP_TOP:
        return _MTP_TOP[tail]
    return f"mtp.stages.{stage}.{tail}\"""",
    "map_mtp_name",
)

ok &= edit(
    "load.py",
    """            items.append((k, v))
        model.load_weights(items, strict=False)""",
    """            mapped = map_mtp_name(k)
            items.append((mapped if mapped else k, v))
        model.load_weights(items, strict=False)""",
    "apply mapping",
)

print()
for f in ("model.py", "load.py", "mtp.py"):
    full = os.path.join(REPO, f)
    if not os.path.exists(full):
        print(f"  {f}: MISSING")
        ok = False
        continue
    r = subprocess.run([sys.executable, "-m", "py_compile", full],
                       capture_output=True, text=True)
    print(f"  {f}: {'OK' if r.returncode == 0 else r.stderr[:400]}")
    if r.returncode != 0:
        ok = False

print()
print("done" if ok else "FAILED")
sys.exit(0 if ok else 1)
