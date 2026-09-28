#!/usr/bin/env python3
"""Exact expert byte accounting from the checkpoints' own safetensors headers.

Sizes Plan B's residency question with real numbers instead of the '4.25 bpw'
approximation:
  * per-expert bytes (w1+w3, w2) for the NATIVE (MXFP4) checkpoint
  * routed-expert bytes per layer and for all 40 layers
  * what fits resident within a given budget, and therefore how many bytes/token
    must be streamed at the measured 6.5 GB/s cold SSD rate

Nothing is loaded — only the header JSON is read (8-byte length prefix then the
JSON map), so this runs in seconds even on a 244 GB directory.

Usage: python3 p8a_expert_bytes.py
"""
from __future__ import annotations

import json
import struct
from collections import defaultdict
from pathlib import Path

NATIVE = Path.home() / ".exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
EXL3 = Path.home() / ".exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def scan(d: Path, label: str):
    """Return per-layer routed-expert bytes and other category totals."""
    files = sorted(d.glob("*.safetensors"))
    if not files:
        print(f"=== {label}: NO SHARDS at {d} ===")
        return None
    cats: dict[str, int] = defaultdict(int)
    per_layer_experts: dict[int, int] = defaultdict(int)
    per_layer_other: dict[int, int] = defaultdict(int)
    examples: dict[str, str] = {}
    n_tensors = 0

    for f in files:
        try:
            h = read_header(f)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! header error {f.name}: {exc}")
            continue
        for k, v in h.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            nb = b - a
            n_tensors += 1
            # layer index
            layer = None
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer = int(parts[i + 1])
                    except ValueError:
                        layer = None
                    break
            if k.startswith("mtp."):
                cats["mtp"] += nb
                examples.setdefault("mtp", k)
                continue
            if "engram" in k:
                cats["engram"] += nb
                examples.setdefault("engram", k)
                continue
            if ".experts." in k or (len(k.split(".")) >= 2 and k.split(".")[-2] == "experts"):
                if layer is not None:
                    per_layer_experts[layer] += nb
                cats["routed_experts"] += nb
                examples.setdefault("routed_experts", k)
                continue
            if "shared_experts" in k:
                cats["shared_experts"] += nb
                examples.setdefault("shared_experts", k)
                continue
            if "embed" in k or "lm_head" in k:
                cats["embed_head"] += nb
                examples.setdefault("embed_head", k)
                continue
            cats["other_dense"] += nb
            examples.setdefault("other_dense", k)
            if layer is not None:
                per_layer_other[layer] += nb

    print(f"=== {label} ===")
    print(f"  dir          : {d}")
    print(f"  shards       : {len(files)}   tensors: {n_tensors}")
    tot = sum(cats.values())
    print(f"  total        : {tot / 1e9:8.2f} GB")
    print()
    print(f"  {'category':<18} {'GB':>9}   {'share':>6}   example key")
    for c, nb in sorted(cats.items(), key=lambda x: -x[1]):
        ex = examples.get(c, "")
        print(f"  {c:<18} {nb / 1e9:9.3f}   {100 * nb / max(tot, 1):5.1f}%   {ex[:60]}")
    print()
    if per_layer_experts:
        print(f"  routed-expert bytes: {len(per_layer_experts)} layers present")
        ks = sorted(per_layer_experts)
        print(f"    layers {ks[0]}..{ks[-1]}")
        for L in ks[:4]:
            print(f"      layer {L:>3}: {per_layer_experts[L] / 1e9:7.3f} GB")
        if len(ks) > 4:
            print("      ...")
        # per-expert for one full layer
        n_exp = 384
        per_expert = per_layer_experts[ks[0]] / n_exp
        print(f"    per-expert (assuming {n_exp} experts): "
              f"{per_expert / 1e6:.2f} MB")
    print()
    return {"cats": dict(cats), "per_layer_experts": dict(per_layer_experts)}


nat = scan(NATIVE, "NATIVE (engram) — MXFP4 experts")
if EXL3.exists():
    ex = scan(EXL3, "EXL3 2.9bpw")

if nat and nat.get("per_layer_experts"):
    ple = nat["per_layer_experts"]
    full_layers = [L for L in ple if ple[L] > 1e9]  # real (non-partial) layers
    med = sorted(ple[L] for L in full_layers)[len(full_layers) // 2] if full_layers else 0
    N_LAYERS = 40
    TOPK = 6
    N_EXPERTS = 384
    per_expert = med / N_EXPERTS
    print("=== Plan B sizing (NATIVE format) ===")
    print(f"  median routed-expert bytes per full layer: {med / 1e9:.3f} GB")
    print(f"  per-expert (w1+w3+w2)                   : {per_expert / 1e6:.2f} MB")
    print(f"  x {N_LAYERS} layers                            : "
          f"{med * N_LAYERS / 1e9:.2f} GB  <- all routed experts, one rank")
    print()
    print(f"  Per TOKEN the model makes {N_LAYERS} layers x {TOPK} picks = "
          f"{N_LAYERS * TOPK} expert-selections.")
    print("  Cold bytes/token = selections x miss_rate x per_expert_bytes.")
    print("  Worst case (every pick cold): "
          f"{N_LAYERS * TOPK * per_expert / 1e9:.2f} GB/token.")
    print()
    budget = 95e9
    frac = budget / (med * N_LAYERS)
    print(f"  resident fraction at {budget / 1e9:.0f} GB of experts: {frac * 100:.1f}%")
    print(f"  i.e. ~{frac * N_EXPERTS:.0f} of {N_EXPERTS} experts per layer")
    print()
    print("  COLD-MISS BYTES/TOKEN vs LRU HIT RATE at that resident set")
    print("  (a MISS is a pick landing on a non-resident expert):")
    print()
    print(f"    {'hit_rate':>8}  {'miss%':>6}  {'MB/token':>9}  {'ceiling tok/s':>13}")
    ssd = 6.5e9
    for h in (0.80, 0.90, 0.94, 0.95, 0.97, 0.98, 0.99, 1.00):
        bpt = N_LAYERS * TOPK * (1.0 - h) * per_expert
        ceil = (ssd / bpt) if bpt > 0 else float("inf")
        print(f"    {h:8.2f}  {(1 - h) * 100:5.1f}%  {bpt / 1e6:9.2f}  {ceil:13.1f}")
    # the bar the plan sets
    print()
    target = 25.0
    need_bpt = ssd / target
    need_miss = need_bpt / (N_LAYERS * TOPK * per_expert)
    print(f"  PLAN B's BAR: to beat {target:.0f} tok/s the SSD ceiling needs")
    print(f"  < {need_bpt / 1e6:.1f} MB/token  =>  hit rate > "
          f"{(1 - need_miss) * 100:.1f}% at the resident set above.")
    print()
    print("  So Plan B lives or dies on ROUTING CONCENTRATION: what share of")
    print("  picks land on the resident experts. Uniform routing would give")
    print(f"  hit_rate={frac:.3f} ({frac * 100:.1f}%) -> "
          f"{N_LAYERS * TOPK * (1 - frac) * per_expert / 1e9:.2f} GB/token -> "
          f"{ssd / (N_LAYERS * TOPK * (1 - frac) * per_expert):.1f} tok/s = DEAD.")
    print("  Real MoE routing is Zipf-like; the actual concentration must be")
    print("  MEASURED from a routing trace (see p8b).")
print("DONE")
