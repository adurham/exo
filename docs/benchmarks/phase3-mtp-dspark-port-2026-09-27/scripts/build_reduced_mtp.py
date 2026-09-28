#!/usr/bin/env python3
"""Build a reduced-layer V4.1 model WITH the DSpark draft head, then smoke-test it.

TWO BUGS FIXED from the first attempt:

  1. ENGRAM LEAK. The filter kept `layers.{0..3}.*` wholesale, and layer 1 is
     an engram layer -- so `layers.1.engram.embed.weight` [384006168, 256] fp8
     = ~98 GB came along and blew the Metal buffer limit. Engram is disabled in
     a reduced build, so every `.engram.` tensor must be dropped explicitly.

  2. OUTPUT TAG COLLISION. The tag was derived from the tensor names, so several
     different source shards all mapped to "top" and overwrote each other
     (visible as "model-top.safetensors" written twice). The tag is now
     `{position}-{source_shard}` so every source shard gets its own file.

Scope: body layers 0..3, all three draft stages verbatim, engram off, vision off.

Usage: python3 build_reduced_mtp.py <DST>
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

SRC = os.path.expanduser("~/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram")
DST = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/v41-build-mtp")
NL = 4

print("=== reduced+MTP build ===")
print(f"  src {SRC}")
print(f"  dst {DST}")
print(f"  body layers 0..{NL-1}, all mtp stages, engram OFF")
print()

src_cfg = json.load(open(os.path.join(SRC, "config.json")))
tc = dict(src_cfg.get("text_config", src_cfg))
index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
print(f"  index: {len(index)} tensors")

keep = {}
dropped_engram = 0
for name, shard in index.items():
    if name.startswith("mtp."):
        keep[name] = shard
        continue
    if "engram" in name or "hash" in name:
        dropped_engram += 1
        continue
    if name.startswith("vision.") or name.startswith("aligner."):
        continue
    if name.startswith("layers."):
        try:
            li = int(name.split(".")[1])
        except (IndexError, ValueError):
            continue
        if li < NL:
            keep[name] = shard
        continue
    if any(name.startswith(p) for p in ("embed", "head", "norm",
                                        "image_start", "image_end", "image_newline")):
        keep[name] = shard
        continue

mtp_n = sum(1 for k in keep if k.startswith("mtp."))
print(f"  keeping {len(keep)} tensors  (mtp.*: {mtp_n}, engram dropped: {dropped_engram})")
if dropped_engram == 0:
    print("  !! WARNING: no engram tensors dropped — filter may be wrong")
print()

os.makedirs(DST, exist_ok=True)
# clear stale outputs from the failed run so no empty/partial file survives
for f in glob.glob(os.path.join(DST, "*")):
    if os.path.isfile(f):
        os.remove(f)
    else:
        shutil.rmtree(f, ignore_errors=True)

by_shard = {}
for name, shard in keep.items():
    by_shard.setdefault(shard, []).append(name)

import mlx.core as mx  # noqa: E402

out_index = {}
order = 0
for shard, names in sorted(by_shard.items()):
    src_path = os.path.join(SRC, shard)
    full = mx.load(src_path)
    tensors = {n: full[n] for n in names}
    if any(n.startswith("mtp.") for n in names):
        kinds = "mtp"
    elif any(n.startswith("layers.") for n in names):
        kinds = "layers"
    else:
        kinds = "top"
    snum = shard.replace("model-", "").replace("-of-00048.safetensors", "")
    out_name = f"model-{order:02d}-{kinds}-{snum}.safetensors"
    order += 1
    mx.save_safetensors(os.path.join(DST, out_name), tensors)
    for n in tensors:
        out_index[n] = out_name
    print(f"    {out_name:<40} {len(tensors):>5} tensors  "
          f"{sum(v.nbytes for v in tensors.values())/1e9:6.2f} GB")
    del full, tensors
    mx.clear_cache()

print()
print(f"  wrote {len(out_index)} tensors across {order} files")
missing_mtp = [k for k in keep if k.startswith("mtp.") and k not in out_index]
print(f"  mtp tensors missing from output: {len(missing_mtp)}")

# ---- config ------------------------------------------------------------
tc["num_hidden_layers"] = NL
tc["engram_layer_ids"] = []
tc["engram_num_embeddings"] = []
tc["num_nextn_predict_layers"] = int(tc.get("num_nextn_predict_layers", 3))
cr = tc.get("compress_ratios") or []
tc["compress_ratios"] = cr[:NL] if cr else []
for key in ("kv_source_layer_ids", "index_source_layer_ids"):
    if key in tc and isinstance(tc[key], list):
        tc[key] = [x for x in tc[key] if x < NL]
tc["dspark_target_layer_ids"] = [x for x in (tc.get("dspark_target_layer_ids") or []) if x < NL]

out_cfg = {k: v for k, v in src_cfg.items() if k != "text_config"}
out_cfg.update(tc)
out_cfg["model_type"] = "deepseek_v41"
json.dump(out_cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
json.dump({"metadata": {}, "weight_map": out_index},
          open(os.path.join(DST, "model.safetensors.index.json"), "w"))

for f in ("tokenizer.json", "tokenizer_config.json"):
    p = os.path.join(SRC, f)
    if os.path.exists(p):
        shutil.copyfile(p, os.path.join(DST, f))

tot = sum(os.path.getsize(f) for f in glob.glob(os.path.join(DST, "*")) if os.path.isfile(f))
print(f"  config: {NL} layers, mtp={tc['num_nextn_predict_layers']}, "
      f"dspark {tc.get('dspark_n_routed_experts')}/{tc.get('dspark_num_experts_per_tok')}")
print(f"  dir total: {tot/1e9:.2f} GB")
print("BUILD-MTP-DONE")
