#!/usr/bin/env python3
"""Q1D Task A -- EXACT per-rank memory-fit arithmetic for the experts-requant arms.

Pure CPU / stdlib + header reads only (read-only pread of the 39 safetensors headers).
No MLX import, no GPU, no writes under ~/repos/exo.

Computes, for the EXL3 checkpoint dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw:
  * per-rank ROUTED-EXPERT bytes for all 40 MoE layers (trellis + suh/svh signs),
    applying the TP=2 intermediate-width slice exactly as mlx_lm/models/exl3/loader.py
    ::load_experts / _slice_intermediate does it, split by layer.
  * the non-expert resident rest per rank (MTP + shared expert + replicated roster),
    with the running code's sharding rule (routed experts / shared_experts / mtp ffn
    are WIDTH-sharded; attention/embed/head/router/vision replicated).
  * projected native-affine (q4g64/q5g64/q4g32/q6g64 + mixed gu q4/dn q6) routed-expert
    bytes for the SAME rank weight count.
  * KV bytes per rank from the model code formula (cache.py: bf16-stored quantized grid;
    comp_kv + index_k on the 4 kv_source layers; window ring constant).
Fit is evaluated against 128 GiB (137,438,953,472 B).
"""
import json, os, re, struct
from math import prod

MODEL = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
DT = {"F16": 2, "BF16": 2, "F32": 4, "I16": 2, "I32": 4, "I64": 8,
      "U8": 1, "I8": 1, "U16": 2, "U32": 4, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}

idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
cfg = json.load(open(os.path.join(MODEL, "config.json")))
tc = cfg["text_config"]
NL = tc["num_hidden_layers"]      # 40
E = tc["n_routed_experts"]        # 384
D = tc["hidden_size"]             # 5120
H = tc["moe_intermediate_size"]   # 2304
WORLD = 2

_hdrs = {}
def hdr(shard):
    if shard not in _hdrs:
        with open(os.path.join(MODEL, shard), "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            _hdrs[shard] = json.loads(f.read(n))
    return _hdrs[shard]

def ent(name):
    h = hdr(idx[name])[name]
    return h, prod(h["shape"]) * DT[h["dtype"]]

# ---- classify + per-rank split factor -------------------------------------------------
RE_EXP = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(\w+)$")
def split_factor(k, suffix, sub):
    """Per-rank multiplier: how much of THIS stored tensor rank r keeps."""
    if suffix == "trellis":
        return 1.0 / WORLD                       # intermediate axis halved
    if suffix == "mul1":
        return 1.0 / WORLD                       # scalar flag per expert, 4 B -> negligible
    # sign vectors (loader _slice_intermediate):
    #   gu_suh = {w1.suh,w3.suh}            REPLICATED (full D)
    #   gu_svh = {w1.svh,w3.svh}            SLICED on H  -> H/2
    #   dn_suh = {w2.suh}                  SLICED on H  -> H/2
    #   dn_svh = {w2.svh}                  REPLICATED (full D)
    if suffix in ("suh", "svh"):
        if (sub == "w1" and suffix == "svh") or (sub == "w3" and suffix == "svh"):
            return 1.0 / WORLD
        if (sub == "w2" and suffix == "suh"):
            return 1.0 / WORLD
        return 1.0                                # w1.suh, w3.suh, w2.svh replicated
    return 1.0 / WORLD

CAT_REST = ("shared_expert", "mtp_ffn")
def rest_factor(k):
    if ".ffn.shared_experts." in k:
        return 1.0 / WORLD                        # width-sharded
    if k.startswith("mtp.") and ".ffn." in k:
        return 1.0 / WORLD                        # width-sharded (DSPARK_TP_SHARD=1)
    return 1.0                                    # replicated

expert_full = 0            # both-ranks routed-expert bytes (all 40 layers)
expert_rank = 0            # rank-0 routed-expert bytes (intermediate slice)
trellis_rank = 0          # rank-0 trellis component
sign_rank = 0             # rank-0 suh/svh sign component
per_layer = {}             # layer -> {"full":..,"rank":..,"k":..}
rest_rank = 0             # per-rank non-routed-expert bytes
rest_full = 0
total_repo = 0
cat_full = {}

for k, shard in idx.items():
    h, nb = ent(k)
    total_repo += nb
    m = RE_EXP.match(k)
    if m:
        L = int(m.group(1)); wid = int(m.group(2)); sub = m.group(3); suf = m.group(4)
        pl = per_layer.setdefault(L, {"full": 0, "rank": 0, "k": None, "E": 0})
        pl["full"] += nb
        expert_full += nb
        rf = split_factor(k, suf, sub)
        # all sliced dims (tiles of 16) are divisible by WORLD, so scaling by
        # 1/WORLD on the byte count is exact.
        expert_rank += nb * rf
        pl["rank"] += nb * rf
        if suf == "trellis":
            trellis_rank += nb * rf
        elif suf in ("suh", "svh"):
            sign_rank += nb * rf
        if suf == "trellis" and sub == "w1":
            packed = h["shape"][2]
            pl["k"] = packed * 16 // 256
        continue
    c = ("shared_expert" if ".ffn.shared_experts." in k
         else "mtp_ffn" if (k.startswith("mtp.") and ".ffn." in k)
         else "mtp_other" if k.startswith("mtp.")
         else "replicated_other")
    cat_full[c] = cat_full.get(c, 0) + nb
    rest_full += nb
    rest_rank += nb * rest_factor(k)

GiB = 1024 ** 3
GB = 1e9
def g(x): return x / GB
def gib(x): return x / GiB

# ---- layer table ----------------------------------------------------------------------
layer_rows = []
for L in sorted(per_layer):
    d = per_layer[L]
    layer_rows.append((L, d["k"], d["full"] / GB, d["rank"] / GB))

# ---- native affine arms ---------------------------------------------------------------
N_rank = E * 3 * (H // WORLD) * D * NL        # routed-expert weights per rank (no signs)
def aff(bpw): return N_rank * bpw / 8.0
gu = E * 2 * (H // WORLD) * D * NL            # w1+w3 weights/rank
dn = E * (H // WORLD) * D * NL                # w2 weights/rank
arms = {
    "EXL3 (2.9bpw trellis)": expert_rank,
    "q4g64 (4.25 bpw)": aff(4.25),
    "q5g64 (5.25 bpw)": aff(5.25),
    "q4g32 (4.50 bpw)": aff(4.50),
    "mixed gu q4 / dn q6": gu * 4.25 / 8 + dn * 6.25 / 8,
    "q6g64 (6.25 bpw)": aff(6.25),
}

# ---- KV cache (per rank; attention is REPLICATED -> each rank holds it all) ----------
# cache.py: win_kv every layer (window=128, head_dim, bf16) -> constant
#           comp_kv + index_k only on kv_source_layers [2,8,14,20], bf16, rows=ceil(ctx/ratio)
WIN = tc["sliding_window"]
ratio = {i: r for i, r in enumerate(tc["compress_ratios"][:NL])}
kvsrc = list(tc["kv_source_layer_ids"])
idxsrc = set(tc["index_source_layer_ids"])
HD = tc["head_dim"]        # == wkv out dim == 512 (attention.py: self.wkv = Linear(dim, head_dim))
INDEX_HD = tc["index_head_dim"]
BF16 = 2
win = NL * WIN * HD * BF16                                  # constant window ring (head_dim)
def kv_bytes(ctx):
    main = sum(((ctx + ratio[L] - 1) // ratio[L]) * HD * BF16 for L in kvsrc)
    ix = sum(((ctx + ratio[L] - 1) // ratio[L]) * INDEX_HD * BF16
             for L in kvsrc if L in idxsrc)
    return win + main + ix, win, main, ix
CTX_AGENTIC = 91043
CTX_MAX = tc["max_position_embeddings"]
kv91, w91, m91, i91 = kv_bytes(CTX_AGENTIC)
kvmax, wmax, mmax, imax = kv_bytes(CTX_MAX)
kv_per_token = (m91 + i91) / CTX_AGENTIC                      # depth-linear slope

# ---- fit table vs 128 GiB -------------------------------------------------------------
LIMIT_B = 128 * GiB
print("=" * 108)
print("Q1D TASK A — per-rank memory-fit arithmetic  (EXL3 experts-requant arms)")
print("=" * 108)
print(f"model        : {os.path.basename(MODEL)}")
print(f"n_layers={NL} n_routed_experts={E} hidden D={D} moe_intermediate H={H} TP world={WORLD}")
print(f"per-rank routed-expert weights (no signs) N_rank = {N_rank:,}  ({g(N_rank):.3f} G)")
print(f"repo total (39 headers)            = {total_repo/GB:.3f} GB  ({gib(total_repo):.3f} GiB)")
print()
print("EXL3 routed-expert trellis bit-width per layer (k = packed*16//256):")
ks = sorted({r[1] for r in layer_rows})
print(f"   distinct k present: {ks}")
for kk in ks:
    ls = [r[0] for r in layer_rows if r[1] == kk]
    tot_full = sum(r[2] for r in layer_rows if r[1] == kk)
    tot_rank = sum(r[3] for r in layer_rows if r[1] == kk)
    print(f"   k={kk}: layers {ls}  full={tot_full:.3f} GB  per-rank={tot_rank:.3f} GB")
print()
print(f"EXL3 routed experts  FULL(all layers,both ranks) = {g(expert_full):.3f} GB")
print(f"EXL3 routed experts  PER-RANK (TP=2 slice)       = {g(expert_rank):.3f} GB "
      f"({gib(expert_rank):.3f} GiB)")
print(f"   of which trellis = {g(trellis_rank):.3f} GB ; suh/svh signs = "
      f"{g(sign_rank):.3f} GB")
print()
print("non-routed-expert RESIDENT rest per rank (running-code sharding):")
for c, v in sorted(cat_full.items(), key=lambda x: -x[1]):
    print(f"   {c:20s} full={g(v):8.3f} GB   per-rank={g(v if c=='replicated_other' else v/2):8.3f} GB")
print(f"   {'REST TOTAL':20s} full={g(rest_full):8.3f} GB   per-rank={g(rest_rank):8.3f} GB")
print()
print("KV cache per rank (attention replicated; cache.py bf16-stored quantized grid):")
print(f"   depth-linear slope = {kv_per_token:.2f} B/token  (comp_kv+index_k on kv_source "
      f"layers {kvsrc})")
print(f"   constant window ring = {w91/1e6:.2f} MB")
print(f"   @ agentic 91,043 tok : {kv91/1e6:8.2f} MB  (0.0{100*gib(kv91):.0f} GiB)")
print(f"   @ config max {CTX_MAX:,} : {kvmax/GB:8.3f} GB  ({gib(kvmax):.3f} GiB)")
print()

hdr_fmt = f"{'arm':30s} {'experts':>10s} {'rest':>9s} {'KV@91K':>9s} {'TOTAL':>10s} {'GiB':>8s}  {'vs128GiB':>9s}  verdict"
print(hdr_fmt); print("-" * len(hdr_fmt))
rows = []
for name, ex_b in arms.items():
    tot = ex_b + rest_rank + kv91
    frac = tot / LIMIT_B
    verdict = "FITS" if frac <= 0.92 else ("TIGHT" if frac <= 1.0 else "DOES-NOT-FIT")
    print(f"{name:30s} {g(ex_b):10.2f} {g(rest_rank):9.2f} {g(kv91):9.3f} {g(tot):10.2f} "
          f"{gib(tot):8.2f}  {100*frac:8.1f}%  {verdict}")
    rows.append((name, ex_b, rest_rank, kv91, tot, frac, verdict))
print()
print(f"128 GiB limit = {LIMIT_B/GB:.2f} GB ; production wired limit iogpu.wired_limit_mb=115000 "
      f"= {115000*1024**2/GB:.2f} GB")
print()
print("CROSS-CHECKS")
print(f"  header per-rank total weights (experts+rest) = {g(expert_rank+rest_rank):.3f} GB "
      f"(memory-budget-correction doc: 108.76 GB)")
print(f"  measured live EXL3 per-rank active anchor      = 105.5 GB")
print(f"  anchor-derived rest = 105.5 - experts = {105.5-g(expert_rank):.3f} GB")
print(f"  doc decomposition: routed 98.02 | MTP 3.84 | shared 0.40 | replicated 6.50 = 108.76")
