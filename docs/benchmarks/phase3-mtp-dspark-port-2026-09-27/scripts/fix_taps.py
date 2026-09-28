#!/usr/bin/env python3
"""Point the reduced build's DSpark taps at real layers.

The build script filtered ``dspark_target_layer_ids`` with ``x < NL`` and NL=4,
so the release's [37, 38, 39] collapsed to [] -- the head then fell back to
``len(ids) or 3``, which is why main_proj still had the right shape (3 taps) but
the model never captured anything to feed it.

No tensor depends on this field (taps are body layer hiddens), so patching the
config is equivalent to a rebuild and far cheaper. Taps become the last three
kept layers, mirroring the release's [37,38,39] of 40 = the last three.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

CONV = os.path.expanduser("~/v41-mtp-converted")
CFG = os.path.join(CONV, "config.json")

cfg = json.load(open(CFG))
tc = cfg.get("text_config", cfg)
NL = tc["num_hidden_layers"]
taps = [NL - 3, NL - 2, NL - 1]

print(f"=== before ===")
print(f"  num_hidden_layers       = {NL}")
print(f"  dspark_target_layer_ids = {tc.get('dspark_target_layer_ids')}")
print(f"  num_nextn_predict_layers= {tc.get('num_nextn_predict_layers')}")

shutil.copy2(CFG, CFG + ".pretaps.bak")
tc["dspark_target_layer_ids"] = taps
if "text_config" in cfg:
    cfg["text_config"] = tc
json.dump(cfg, open(CFG, "w"), indent=2)

chk = json.load(open(CFG))
ctc = chk.get("text_config", chk)
print()
print(f"=== after ===")
print(f"  dspark_target_layer_ids = {ctc.get('dspark_target_layer_ids')}")
print(f"  dspark_block_size       = {ctc.get('dspark_block_size')}")
print(f"  dspark_n_routed_experts = {ctc.get('dspark_n_routed_experts')} / topk "
      f"{ctc.get('dspark_num_experts_per_tok')}")
print()
print("TAPS-DONE")
