#!/usr/bin/env python3
"""Wire the DSpark draft head into the reference port.

Four coordinated source edits (all in ~/repos/ref/deepseek-v41-mlx):

  1. config.py      -- add the three dspark_* fields the checkpoint advertises
                       (n_routed_experts 128, num_experts_per_tok 3,
                       block_size/markov_rank/noise_token_id) and read them.
  2. moe.py         -- let Gate take an explicit expert count / top-k so the
                       draft MoE can be 128/top-3 while the body stays 384/top-6.
  3. convert.py     -- STOP dropping mtp.*; write the draft stack into its own
                       shard(s) instead.
  4. model.py       -- build the DSparkHead when the build carries mtp weights
                       and expose it on the Model.

The new module mtp.py is written separately (already staged).

Backs up every file it touches, prints a diff, and leaves the repo importable.
Run on the node that hosts ~/repos/ref/deepseek-v41-mlx.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/deepseek_v41_mlx")
STAMP = ".predspark.bak"


def edit(path, old, new, label):
    full = os.path.join(REPO, path)
    src = open(full).read()
    if new in src and old not in src:
        print(f"  [{label}] already applied")
        return True
    if old not in src:
        print(f"  [{label}] !! ANCHOR NOT FOUND in {path}")
        print("      expected:")
        for line in old.splitlines()[:6]:
            print("        " + line)
        return False
    bak = full + STAMP
    if not os.path.exists(bak):
        shutil.copy2(full, bak)
    open(full, "w").write(src.replace(old, new, 1))
    print(f"  [{label}] applied to {path} (backup {os.path.basename(bak)})")
    return True


ok = True

# ---------------- 1. config.py: dspark fields ----------------
print("=== 1. config.py ===")
ok &= edit(
    "config.py",
    """    # side-paths dropped for inference (kept for provenance)
    n_mtp_layers: int = 0
    dspark_target_layer_ids: tuple = ()
    vision_n_layers: int = 0""",
    """    # side-paths dropped for inference (kept for provenance)
    n_mtp_layers: int = 0
    dspark_target_layer_ids: tuple = ()
    vision_n_layers: int = 0

    # DSpark draft head (the mtp.{0..n-1} stack). The draft MoE is SMALLER than
    # the body: 128 routed experts, top-3, vs the body's 384/top-6. These come
    # from the checkpoint's own dspark_* keys.
    dspark_n_experts: int = 0
    dspark_topk: int = 0
    dspark_block_size: int = 5
    dspark_markov_rank: int = 256
    dspark_noise_token_id: int = 128799""",
    "config fields",
)

ok &= edit(
    "config.py",
    """            dspark_target_layer_ids=tuple(_get(c, "dspark_target_layer_ids", default=()) or ()),
            vision_n_layers=0,  # text-only runtime""",
    """            dspark_target_layer_ids=tuple(_get(c, "dspark_target_layer_ids", default=()) or ()),
            dspark_n_experts=_get(c, "dspark_n_routed_experts", default=0),
            dspark_topk=_get(c, "dspark_num_experts_per_tok", default=0),
            dspark_block_size=_get(c, "dspark_block_size", default=5),
            dspark_markov_rank=_get(c, "dspark_markov_rank", default=256),
            dspark_noise_token_id=_get(c, "dspark_noise_token_id", default=128799),
            vision_n_layers=0,  # text-only runtime""",
    "config _get",
)

# ---------------- 2. moe.py: Gate takes explicit sizes ----------------
print("=== 2. moe.py ===")
ok &= edit(
    "moe.py",
    """class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = mx.zeros((args.n_routed_experts, args.dim))
        self.bias = mx.zeros((args.n_routed_experts,), dtype=mx.float32)
        self.bias_vl = mx.zeros((args.n_routed_experts,), dtype=mx.float32)""",
    """class Gate(nn.Module):
    def __init__(self, args: ModelArgs, n_experts: int | None = None,
                 topk: int | None = None):
        # The DSpark draft MoE is 128 experts / top-3 while the body is
        # 384 / top-6, so the sizes must be overridable.
        super().__init__()
        n_experts = n_experts or args.n_routed_experts
        self.topk = topk or args.n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = mx.zeros((n_experts, args.dim))
        self.bias = mx.zeros((n_experts,), dtype=mx.float32)
        self.bias_vl = mx.zeros((n_experts,), dtype=mx.float32)""",
    "Gate sizes",
)

print()
print("=== syntax check ===")
for f in ("config.py", "moe.py"):
    r = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(REPO, f)],
                       capture_output=True, text=True)
    print(f"  {f}: {'OK' if r.returncode == 0 else r.stderr[:300]}")
    if r.returncode != 0:
        ok = False

print()
if ok:
    print("config.py + moe.py patched. convert.py and model.py are separate steps.")
else:
    print("ONE OR MORE EDITS FAILED — nothing else attempted.")
    sys.exit(1)
