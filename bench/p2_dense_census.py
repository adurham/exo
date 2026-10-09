#!/usr/bin/env python3
"""p2_dense_census.py -- exact per-rank dense-EXL3 trellis byte census for the
real DSv4.1 checkpoint (all 40 layers). Host-side struct/json only, no mlx, no GPU.

Reports, per layer, the dense (non-expert) trellis bytes at full shape and at the
world=2 rank-0 slice (attn.wq_b/wo_b, shared w1/w2/w3 halved; everything else
replicated), then the model totals and the per-PASS dense bytes.
"""
import json, os, struct, sys

CK = os.environ.get(
    "PD_CK", os.path.expanduser("~") + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
)
NL = 40
w = json.load(open(os.path.join(CK, "model.safetensors.index.json")))["weight_map"]

def hdr(name):
    p = os.path.join(CK, w[name])
    with open(p, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        h = json.loads(fh.read(n))
    return h[name]

SHARD_HALVE = ("attn.wq_b", "attn.wo_b")
SHARD_HALVE_FFN = ("ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3")

def is_dense(k):
    return (k.startswith("layers.") and ".ffn.experts." not in k and k.endswith(".trellis"))

tot_full = 0
tot_rank = 0
rows = []
ks = sorted([k for k in w if is_dense(k)], key=lambda s: (int(s.split(".")[1]), s))
per_layer = {}
for k in ks:
    L = int(k.split(".")[1])
    e = hdr(k)
    nb = e["data_offsets"][1] - e["data_offsets"][0]
    group = k
    halve = any(g in k for g in SHARD_HALVE) or any(g in k for g in SHARD_HALVE_FFN)
    rank_b = nb // 2 if halve else nb
    tot_full += nb
    tot_rank += rank_b
    per_layer.setdefault(L, [0, 0])
    per_layer[L][0] += nb
    per_layer[L][1] += rank_b
    rows.append((k, nb, rank_b))

for L in sorted(per_layer):
    f, r = per_layer[L]
    print("layer %2d  full %8.2f MB  rank0 %8.2f MB" % (L, f / 1e6, r / 1e6))
print()
print("DENSE-ONLY (attn+shared+indexer+compressor, no routed experts):")
print("  full-model  trellis bytes = %.3f GB" % (tot_full / 1e9))
print("  per-rank    trellis bytes = %.3f GB   (world=2)" % (tot_rank / 1e9))
print("  per-RANK per-PASS dense bytes = %.4f GB" % (tot_rank / 1e9))

# routed-expert bytes for context (packed varies by layer)
exp_full = 0
for k in w:
    if k.startswith("layers.") and ".ffn.experts." in k and k.endswith(".trellis"):
        e = hdr(k)
        exp_full += e["data_offsets"][1] - e["data_offsets"][0]
print("  (routed-expert trellis full = %.3f GB, unactivated; not part of dense slice)" % (exp_full / 1e9))

json.dump({"per_layer_full_bytes": {str(L): per_layer[L][0] for L in per_layer},
           "per_layer_rank_bytes": {str(L): per_layer[L][1] for L in per_layer},
           "total_full_bytes": tot_full,
           "total_rank_bytes": tot_rank,
           "total_rank_gb": tot_rank / 1e9},
          open(os.environ.get("PD_OUT", os.path.expanduser("~") + "/p5-dense-ws/dense_census.json"), "w"), indent=1)
