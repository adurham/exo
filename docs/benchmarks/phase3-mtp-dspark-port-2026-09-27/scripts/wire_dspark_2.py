#!/usr/bin/env python3
"""Wire step 2: make the converter KEEP mtp.* instead of dropping it.

Currently convert.py has:
    if name.startswith("mtp."):
        return "mtp"
  ...
    accounting = {"kept": 0, "dropped_mtp": len(groups.pop("mtp", [])), ...}
i.e. the draft stack is popped off and discarded. Four changes:

  a) `owner()` returns "mtp.0" / "mtp.1" / "mtp.2" (per stage) rather than a
     single "mtp", so each stage becomes its own shard group;
  b) don't pop it out of `groups`;
  c) add the mtp groups to the output `order`;
  d) report mtp accounting as `mtp_kept` instead of `dropped_mtp`.

Also: sanitize_group already handles the shapes we need (fp8 weight+scale
dequant, expert stacking) but its expert-stacking regex is `layers\\.\\d+\\.`
only. The draft's experts live at `mtp.{i}.ffn.experts.{N}.w{1,2,3}.weight`, so
that regex must be widened to accept `mtp\\.\\d+` too — otherwise 128 experts x 3
projections would be emitted as separate tensors no module owns.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/deepseek_v41_mlx")
STAMP = ".premtpconv.bak"


def edit(path, old, new, label):
    full = os.path.join(REPO, path)
    src = open(full).read()
    if new in src and old not in src:
        print(f"  [{label}] already applied")
        return True
    if old not in src:
        print(f"  [{label}] !! ANCHOR NOT FOUND in {path}")
        for line in old.splitlines()[:8]:
            print("        " + line)
        return False
    bak = full + STAMP
    if not os.path.exists(bak):
        shutil.copy2(full, bak)
    open(full, "w").write(src.replace(old, new, 1))
    print(f"  [{label}] applied (backup {os.path.basename(bak)})")
    return True


ok = True
print("=== convert.py: keep mtp.* ===")

# (a) owner(): per-stage mtp groups
ok &= edit(
    "convert.py",
    """        if name.startswith("mtp."):
            return "mtp\"""",
    """        if name.startswith("mtp."):
            # one group per stage so each draft block is its own shard
            m = re.match(r"mtp\\.(\\d+)\\.", name)
            return f"mtp.{m.group(1)}" if m else "mtp\"""",
    "owner() per-stage",
)

# (b) stop popping mtp out of groups
ok &= edit(
    "convert.py",
    """    accounting = {"kept": 0, "dropped_mtp": len(groups.pop("mtp", [])),
                  "vision_passthrough": 0}""",
    """    mtp_groups = [g for g in groups if g.startswith("mtp")]
    accounting = {"kept": 0, "mtp_kept": sum(len(groups[g]) for g in mtp_groups),
                  "vision_passthrough": 0}""",
    "accounting",
)

# (c) include mtp groups in the output order
ok &= edit(
    "convert.py",
    """    order = ["top"] + sorted((g for g in groups if g.startswith("layers.")),
                             key=lambda s: int(s.split(".")[1]))
    if "vision" in groups:
        order.append("vision")""",
    """    order = ["top"] + sorted((g for g in groups if g.startswith("layers.")),
                             key=lambda s: int(s.split(".")[1]))
    order += sorted((g for g in groups if g.startswith("mtp")),
                    key=lambda s: int(s.split(".")[1]) if "." in s else 0)
    if "vision" in groups:
        order.append("vision")""",
    "order",
)

# (d) the expert-stacking regex must accept mtp.<i>.ffn.experts too
ok &= edit(
    "convert.py",
    """        m = re.match(r"(layers\\.\\d+\\.ffn\\.experts)\\.(\\d+)\\.(w[123])\\.weight$", name)""",
    """        m = re.match(r"((?:layers|mtp)\\.\\d+\\.ffn\\.experts)\\.(\\d+)\\.(w[123])\\.weight$",
                     name)""",
    "expert regex",
)

# (e) _group_tag must handle "mtp.N"
print()
print("=== _group_tag: does it handle mtp.N? ===")
src = open(os.path.join(REPO, "convert.py")).read()
i = src.find("def _group_tag")
print(src[i:i + 500] if i >= 0 else "  (not found)")

print()
print("=== syntax check ===")
r = subprocess.run([sys.executable, "-m", "py_compile",
                    os.path.join(REPO, "convert.py")],
                   capture_output=True, text=True)
print(f"  convert.py: {'OK' if r.returncode == 0 else r.stderr[:400]}")
ok &= r.returncode == 0

print()
print("mtp_kept" if ok else "FAILED")
sys.exit(0 if ok else 1)
