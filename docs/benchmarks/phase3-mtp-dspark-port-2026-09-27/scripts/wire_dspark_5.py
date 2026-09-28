#!/usr/bin/env python3
"""Wire step 5: finish the MTP name mapping.

Two gaps remain in load.py's map_mtp_name:

  a) IDEMPOTENCY. If the build already uses runtime names
     (`mtp.stages.0.attn.wq_a.weight`), the current function mangles them into
     `mtp.stages.stages.0...` because it only checks the `mtp.` prefix. Guard it.
  b) QUANT MODULE MAP. convert.py writes `quantization.modules` keyed by the
     SANITIZED (release-style) module path, e.g. `mtp.0.ffn.experts.gate_proj`
     and `mtp.2.markov_head.embed`. quant_predicate looks them up by RUNTIME
     module path (`mtp.stages.0.ffn.experts`, `mtp.markov_embed`). Those never
     match, so quantized draft weights would load as raw packed uint32 into
     unquantized Linear modules -> shape errors or silent corruption.
     Fix: translate the map's keys through the same function before use.

The module-level tensors also need their `.weight`-less forms handled, since
module paths carry no `.weight` suffix.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/deepseek_v41_mlx")
STAMP = ".premtpmap.bak"


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

print("=== load.py: robust map_mtp_name ===")
ok &= edit(
    "load.py",
    """def map_mtp_name(k: str):
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
    """def map_mtp_name(k: str):
    \"\"\"Checkpoint ``mtp.*`` name -> bare runtime module path (or None).

    Idempotent: a name already in runtime form is returned unchanged, so this
    can be applied to checkpoint keys and to module-map keys alike.
    \"\"\"
    if not k.startswith("mtp."):
        return None
    if k.startswith("mtp.stages.") or k in _MTP_TOP_M:
        return k                          # already runtime form
    parts = k.split(".")
    if len(parts) < 3:
        return None
    stage, tail = parts[1], ".".join(parts[2:])
    for w, p in _MTP_PROJ.items():
        tail = tail.replace(f"ffn.shared_experts.{w}.", f"ffn.shared_experts.{p}.")
    # module paths carry no ".weight"; the lookup table does, so try both
    if tail in _MTP_TOP:
        return _MTP_TOP[tail]
    bare = tail[:-len(".weight")] if tail.endswith(".weight") else tail
    if bare in _MTP_TOP_M:
        return _MTP_TOP_M[bare]
    return f"mtp.stages.{stage}.{tail}\"""",
    "map_mtp_name robust",
)

ok &= edit(
    "load.py",
    """def map_mtp_name(k: str):
    \"\"\"Checkpoint ``mtp.*`` name -> bare runtime module path (or None).""",
    """# module-level draft modules, without the ".weight" suffix (module-map keys)
_MTP_TOP_M = {k[:-len(".weight")]: v for k, v in _MTP_TOP.items()}
# markov_head.embed / .head appear as module paths too
_MTP_TOP_M["markov_head.embed"] = "mtp.markov_embed"
_MTP_TOP_M["markov_head.head"] = "mtp.markov_head"


def map_mtp_name(k: str):
    \"\"\"Checkpoint ``mtp.*`` name -> bare runtime module path (or None).""",
    "_MTP_TOP_M",
)

print()
print("=== load.py: translate the quant module map ===")
ok &= edit(
    "load.py",
    """    q = cfg.get("quantization")
    if q:
        if q.get("bits"):
            module_map = q.get("modules")""",
    """    q = cfg.get("quantization")
    if q:
        if q.get("bits"):
            module_map = q.get("modules")
            if module_map:
                # convert.py keys this map by sanitized (release-style) module
                # paths; quant_predicate looks up runtime paths. Translate the
                # mtp entries so quantized draft weights land correctly.
                module_map = {(map_mtp_name(k) or k): v
                              for k, v in module_map.items()}""",
    "quant map translate",
)

print()
r = subprocess.run([sys.executable, "-m", "py_compile",
                    os.path.join(REPO, "load.py")], capture_output=True, text=True)
print(f"load.py: {'OK' if r.returncode == 0 else r.stderr[:400]}")
ok &= r.returncode == 0

if ok:
    # verify the mapping behaviour
    sys.path.insert(0, os.path.dirname(REPO))
    from deepseek_v41_mlx.load import map_mtp_name as m
    print()
    print("=== mapping behaviour ===")
    cases = [
        "mtp.0.attn.wq_a.weight",              # -> stages.0
        "mtp.1.ffn.experts.gate_proj.weight",  # -> stages.1
        "mtp.0.ffn.shared_experts.w1.biases",  # w1 -> gate_proj
        "mtp.0.main_proj.weight",              # -> mtp.main_proj
        "mtp.2.markov_head.embed.weight",      # -> mtp.markov_embed
        "mtp.0.main_proj",                     # module-path form
        "mtp.2.markov_head.embed",             # module-path form
        "mtp.2.confidence_head.proj",          # module-path form
        "mtp.0.ffn.experts.gate_proj",         # module-path form
        "mtp.stages.0.attn.wq_a.weight",       # already runtime -> unchanged
        "layers.0.attn.wq_a.weight",           # not mtp -> None
    ]
    for c in cases:
        print(f"    {c:<44} -> {m(c)}")

print()
print("done" if ok else "FAILED")
sys.exit(0 if ok else 1)
