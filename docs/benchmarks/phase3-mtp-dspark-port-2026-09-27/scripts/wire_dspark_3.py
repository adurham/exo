#!/usr/bin/env python3
"""Wire step 3: group tag hygiene + make the build's config advertise the head.

  a) `_group_tag("mtp.0")` currently returns "mtp.0", producing filenames with
     a dot. Normalize to "mtp-0" for consistency with "layers-00".
  b) The converted build's config.json must carry n_mtp_layers and the dspark_*
     fields, otherwise `ModelArgs.from_dict` reads 0 stages and the head is
     never built. The source config already has them (num_nextn_predict_layers=3,
     dspark_n_routed_experts=128, dspark_num_experts_per_tok=3, ...) but
     convert() builds `out_cfg` from `cfg` and only overrides model_type +
     quantization, so if the release config nests them under text_config they
     must be flattened the same way ModelArgs.from_dict reads them.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

REPO = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/deepseek_v41_mlx")
STAMP = ".premtpconv2.bak"


def edit(path, old, new, label, count=1):
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
    open(full, "w").write(src.replace(old, new, count))
    print(f"  [{label}] applied")
    return True


ok = True
print("=== a) _group_tag ===")
ok &= edit(
    "convert.py",
    """def _group_tag(g: str) -> str:
    if g.startswith("layers."):
        return f"layers-{int(g.split('.')[1]):02d}"
    return g""",
    """def _group_tag(g: str) -> str:
    if g.startswith("layers."):
        return f"layers-{int(g.split('.')[1]):02d}"
    if g.startswith("mtp."):
        return f"mtp-{int(g.split('.')[1])}"
    return g""",
    "_group_tag",
)

print("=== b) config passthrough: show what convert writes ===")
src = open(os.path.join(REPO, "convert.py")).read()
i = src.find("cfg = json.load(open(cfg_path))")
print(src[i:i + 700] if i >= 0 else "  (anchor not found)")

print()
print("=== what does ModelArgs.from_dict read from? ===")
csrc = open(os.path.join(REPO, "config.py")).read()
j = csrc.find("def from_dict")
print(csrc[j:j + 700] if j >= 0 else "  (not found)")

print()
r = subprocess.run([sys.executable, "-m", "py_compile",
                    os.path.join(REPO, "convert.py")], capture_output=True, text=True)
print(f"convert.py syntax: {'OK' if r.returncode == 0 else r.stderr[:300]}")
sys.exit(0 if ok else 1)
