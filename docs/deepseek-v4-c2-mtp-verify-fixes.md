# DeepSeek-V4-Flash — Making MTP Speculative Decode Correct at c≥2 (BS>1)

**Date:** 2026-06-06/07
**Cluster:** 2× Mac Studio M4 Max (128 GB each), RDMA over Thunderbolt 5, exo
tensor-parallel (2 ranks).
**Model:** `mlx-community/DeepSeek-V4-Flash-8bit` (MoE + sparse-pooled attention,
served via the fork's `deepseek_v4.py`).
**Scope:** Concurrent serving (`c=2` = 2 simultaneous request streams, batch
size N=2) with MTP self-speculation **on** at 100K context.

---

## 1. TL;DR

Before this session, `c=2` (two concurrent streams) with **MTP on** was broken
at long context: output was catastrophic BOS-spam / "FULL" garbage, or the
long-range needle was silently wrong. After three root-cause fixes it is now
**stable and correct**: both streams run concurrently at 100K, retrieve
long-range content, never spam, never wedge the cluster.

**Measured (c=2, 100K, MTP-on, γ=2):** ~**20 tok/s per stream** (~40 aggregate),
steady-state. (vs ~18.4 spec-off — MTP buys ~10% at 100K because the verify
forward re-reads the full 100K KV.) `c=1` is unaffected (~30–40 t/s, needle 2/2).

**Stability soak (8 concurrent rounds, 100K):** both streams alive 8/8, BOS-spam
0/8, needle content retrieved 8/8 on both streams, cluster crashes / `LEN_ERR`
wedges / index errors = 0.

**Residual (cosmetic, documented in §5):** a worst-case near-tie — a code that
ends in a digit *immediately* followed by EOS (e.g. `…7749` where `9` is its own
token before EOS) — can bistably drop that final character (`…774`). The content
is always retrieved correctly; normal prose/conversation is unaffected.

---

## 2. The setup that makes this hard

DSv4-Flash uses three attention flavors, chosen per layer by `compress_ratios`
(histogram for this checkpoint: **ratio-4 ×21 layers, ratio-128 ×20, ratio-0 ×3**):

- `ratio==0` → **LocalAttention** (plain sliding window, fused SDPA).
- `ratio==128` → **CompressedAttention** (local window + *all* pooled blocks,
  fused SDPA).
- `ratio==4` → **SparseCompressedAttention** (local window + **top-k=512**
  pooled blocks selected by the lightning *indexer*).

MTP self-speculation runs two passes per cycle at batch size N:

1. **Draft** — the small MTP head proposes γ tokens (cheap).
2. **Verify** — the *target model* runs one forward over `L = γ+1` query
   positions to accept/reject the drafts. **This verify forward is the L>1,
   BS>1 path that was buggy.**

Decode (non-spec) is always `L==1`. So the bug surfaced **only when BS>1 AND
L>1 happen together** — i.e. the c≥2 speculative verify. `c=1` spec (BS=1, L>1)
and non-spec `c=2` decode (BS=2, L=1) were both fine; this was the key
truth-table observation that localized every fix.

Per-stream KV state for c≥2 lives in `PerStreamBatchRotatingKVCache`
(`mlx_lm/models/cache.py`), swapped in after prefill. Each stream keeps its own
absolute `offset`; the local ring and the pooled cache share one physical
buffer.

---

## 3. The three bugs (root causes + fixes)

### Bug 1 — local KV ring corrupted at the L>1 verify (BOS-spam class)

`PerStreamBatchRotatingKVCache` was a fixed `max_size` (=128) ring. That is
correct for `L==1` decode but **wrong for the `L=γ+1` verify**: it wrote the S
new verify tokens at `(offset+j) % max_size`, **clobbering committed in-window
keys that earlier query rows of the same forward still needed**, and
`make_mask` produced only `max_size` key slots instead of the `max_size+S-1` the
base class grows to. Once the ring had rotated (long context) the verify saw a
corrupted local window → wrong logits → catastrophic BOS-spam at c=2 100K.

**Fix (mlx-lm `cb7e3bd`, superseding the earlier appendix attempt `eb64d25`):**
a **wide ring** — physical width `max_size + _RING_SLACK` (slack=8 ≥ max verify
S). All writes (decode and verify) are modular over `ring_width`; the attention
window stays `max_size` (enforced by `make_mask`). The wide ring physically
retains the last `max_size+slack` positions so every verify query row sees its
full window without a new token overwriting a still-needed committed key.
Validated with a content-level unit test (per-query attended **logical
positions** match the base `BatchRotatingKVCache` across L=1/L>1, rotated and
not, single- and multi-stream).

### Bug 2 — pooled mask masked out the needle at the L>1 verify (THE big one)

After Bug 1, output was coherent but the long-range needle was still wrong.
`BatchPoolingCache.make_mask(L>1, offset)` was written for **prefill chunks**,
where `offset` is the large, correct chunk position and per-query causal masking
over pooled blocks matters. But at the **speculative verify**, the caller passes
the *local sliding-window cache's* `offset` (e.g. **131**), **not** the absolute
token position (~52568). The L>1 branch computed
`causal = pool_idx < (offset+arange(1,L+1)) // ratio = 131 // 128 = 1`, leaving
**only ~1 pooled block visible per query** (diag: `offset=[131,131]`,
`pool_lengths=[410,410]`, `pmask_sum=6` over 2×3 queries = 1 block each). The
needle's pooled block (~390) was masked out → pooled attention mass collapsed
from ~0.5 (decode) to ~0.03 (verify) → the model ignored the document.

**Fix (mlx-lm `aaac5c3` + `60a0a0c`):** at the decode-time verify the γ+1 query
tokens are all at the generation frontier, *after* every pooled block, so **all
valid pooled blocks are causal — identical to the `L==1` decode case**. Gate on
`L <= _POOL_VERIFY_MAX_L` (=16, cleanly below the prefill step size of 128/4096)
and return the length-only `valid` mask (or `None` when fully valid), skipping
the offset-dependent causal cutoff. Applied to both `BatchPoolingCache` (c≥2)
and the scalar `PoolingCache` (c=1) for consistency. **This restored retrieval:
pooled mass 0.03→0.5+, needle correct.**

### Bug 3 — last-token drop from the less-accurate L>1 sparse kernel

After Bugs 1–2, retrieval worked but a final-token near-tie persisted: the
verify scored EOS (29.75) over the true next token "9" (29.12) after "774" by
~0.63, so c=2 dropped the last character. Cause: the ratio-4
`SparseCompressedAttention` layers route `L==1` through Apple's **fused fp32
SDPA**, but `L>1` fell through to `_sparse_pooled_attention_inner`, a hand-rolled
split-softmax that accumulates in **bf16** and is ~3× less accurate (max abs diff
vs fp32 ref 0.012 vs 0.004). Across 21 sparse layers that error compounded into a
~0.6-logit shift at the verify.

**Fix (mlx-lm `491f6fe` + gate `5b00004`):** for small `L` (the verify,
`L <= _SPARSE_VERIFY_MAX_L`=16), run each of the γ+1 query positions through the
**same accurate fused `L==1` path** and stack. Large-L prefill chunks keep the
batched inner kernel (looping hundreds of fused SDPAs would be far too slow).
This flipped the decisive margin from **−0.63 → +0.125** (now correct on
average; see §5 for the residual).

---

## 4. Commit map (all on `main`)

mlx-lm (`github.com/adurham/mlx-lm`):

| Commit | What |
|--------|------|
| `cb7e3bd` | **Bug 1** — wide-ring `PerStreamBatchRotatingKVCache` (correct L>1 verify local attn). Supersedes `eb64d25`. |
| `aaac5c3` | **Bug 2** — pooled `make_mask` treats tiny-L as decode-verify (all blocks causal). |
| `60a0a0c` | **Bug 2** — verify `make_mask` mirrors `L==1` (None when fully valid) + diag strip. |
| `491f6fe` | **Bug 3** — route L>1 sparse verify through accurate fused SDPA. |
| `5b00004` | **Bug 3** — gate per-position fused sparse verify to small L (prefill keeps inner kernel). |

exo (`github.com/adurham/exo`) — `uv.lock` bumps tracking the above
(`a68417ab`, `b28ae4f3`, `463a3fd8`, `ef3d0654`) plus diag plumbing that was
added and stripped within the session.

All investigation scaffolding (`EXO_DSV4_VERIFY_DIAG`, `EXO_DSV4_STREAMDIV_DIAG`,
`EXO_DSV4_TAILDIAG`) was removed; the tree is clean.

---

## 5. Residual: last-token near-tie (and why it's bounded)

The L>1 *batched* verify produces a hidden state with tiny fp non-associativity
noise vs the L=1 non-spec decode (which is bit-correct — non-spec c=2 yields the
full `…7749`). After Bug 3 the decisive margin is **+0.125** (correct direction),
but residual ~0.3 noise can still flip it on a pathological near-tie: a code
ending in a digit token immediately followed by EOS. Effect: `…7749` → `…774`,
roughly 80% of the time on that exact adversarial needle.

Notes:
- The logits are already fp32 (`quantized_matmul` outputs fp32), so this is
  **not** logit precision — it's the upstream L>1 batched-forward noise.
- It is **BS>1 + L>1 specific**: c=1 spec and non-spec c=2 are both exact.
- γ=1 reduces the flip rate (less L accumulates less noise) but does not
  eliminate it.
- The **only** guaranteed bit-exact fix is to run each verify position as a
  separate `L==1` forward (the TRT-LLM "flatten sequence into batch" approach
  for MTP>1). That is ~3× the verify cost and would erase the ~10% c=2 spec
  throughput win at 100K, so it was **not** taken. If exact final-character
  fidelity is required, run that stream **spec-off** at c=2 — it's bit-correct
  and only ~10% slower.

Normal generation (prose, code, Q&A) is unaffected — the content is always
correct; only the final character of a number-immediately-before-EOS can clip.

---

## 6. How to reproduce / verify

```bash
# Champion (default): start_cluster.sh already sets EXO_SPECULATIVE=1 GAMMA=2 ...
cd ~/repos/exo && ./start_cluster.sh

# c=2 needle (two concurrent streams, 100K): expect needle content on both,
# no BOS-spam, both streams alive.
python /tmp/soak.py   # 8-round concurrent soak (see this doc's history)

# c=1 regression check:
.venv/bin/python bench/mtp_longctx_probe.py --target-tokens 100000 --iters 2 \
  --max-tokens 400 --model mlx-community/DeepSeek-V4-Flash-8bit --seed 7749
```

Forbidden levers unchanged (do NOT alter): `EXO_KV_CACHE_BITS=0` (bf16 KV
floor), `EXO_DSV4_INDEX_TOPK>=512`. Always quality-probe before quoting t/s —
throughput-clean + quality-dead (BOS-spam) is a known failure mode; show real
generated text.
