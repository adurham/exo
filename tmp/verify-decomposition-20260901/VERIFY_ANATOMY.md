# VERIFY_ANATOMY — Decomposition of the MTP 'verify' phase (DeepSeek-V4-Flash / exo)

**Date:** 2026-09-01
**Repo:** `/Users/adam.durham/repos/exo`
**Model file:** `mlx-lm/mlx_lm/models/deepseek_v4.py` (submodule)
**Verify driver:** `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`
**Measured context:** verify 56.1 ms / 81.4 % of a 68.85 ms cycle at 89,408 prompt tokens, 34 t/s (given in task).
**This is a READ-ONLY source analysis.** No code was modified; nothing was run. All "READ" claims are quotable; all "INFER" claims are labeled. Anything not determinable is in §6.

> **Headline (answer to the motivating question):** At 89 K context, verify is **structurally tied to context length.** The dominant, reproducible context-scaling component is attention itself: 21 *sparse-indexer* layers (ratio 4) each do a full **O(context) score GEMM + O(context) top-k** on a pool of **P ≈ 22,250** keys (≈ 89 K / 4), plus 20 *compressed-attention* layers (ratio 128) each attend a pool of **P ≈ 700** keys that also grows with context. The verify forward is **BATCHED** (M=4 rows in one call) under the live env, so these per-layer costs apply once per cycle, not 4×, but they all scale with the KV/pool size. The MoE (gate / switch_mlp / shared_experts / combine) and lm_head are **fixed-cost** (batch/block/dim-bound only). There are no per-layer or jaccl comms counters already recording inside verify; the 56 ms is currently a monolithic wall-time bracket.

---

## 1. The 'verify' timer's exact bracket

### 1.1 Where the profiler records 'verify'
Three sites, one per generator path — all bracket **only the single batched target forward** (`dsv4_speculative_forward` → `Model(inputs, cache=cache)`):

| Site | Generator | Line (dsv4_mtp.py) |
|---|---|---|
| `_speculative_next_batch` (BS>1) | `prof.record("verify", ...)` | 2722 |
| `_speculative_next` (single-uid linear) | `prof.record("verify", ...)` | 4242 |
| `_speculative_next_tree` (BS=1 tree) | `prof.record("verify", ...)` | 5592 |

The live serving path is single-uid (`len(uids) == 1 → self._speculative_next` at dsv4_mtp.py:2472-2474), so the measured 56 ms comes from **site 4242** (though all three bracket the same forward). The tree path (`_speculative_next_tree`, 5592) is gated behind `EXO_DSV4_TREE_DRAFT=1` and is **not** in the live env (not in preflight), so it does **not** fire.

### 1.2 Start of the bracket (what it EXCLUDES)
For site 4242 (`_speculative_next`, dsv4_mtp.py:4098-4242):

- **Timer starts at** `t_after_draft` (dsv4_mtp.py:4101), recorded immediately before the verify block. `t_after_draft` is set **after** `mx.eval(*draft_ids)` (4100) — i.e. the draft phase's cost (MTP draft head forward + sampling) is fully drained out of verify.
- **Verify block opens** at `_t_rb_snap0` (4138, only when `_RB_PROFILE`), then `_pool_caches`/snapshot/arm bookkeeping (4139-4162) runs, then `dsv4_speculative_forward(...)` (4178-4183). The `_RB_PROFILE` bracket (`EXO_DSV4_RB_PROFILE=1` is live) isolates the snapshot cost as `rb_snap` so it is NOT charged to verify (this is the 0.150 ms already closed as irrelevant).
- **Timer stops at** `t_after_verify` (4240-4242): `mx.eval(verify_pre_norm, verify_logits)` then `prof.record("verify", (t_after_verify - t_after_draft) * 1000.0)`.

> **IMPORTANT measurement caveat (already documented, v3 RESULTS.md; confirmed in-source):** The `prof.record("verify", ...)` value is `perf_counter` wall time bracketed by an `mx.eval` (dsv4_mtp.py:4100, 4240). The source comment at dsv4_mtp.py:238-241 states *"Inserts evals at phase boundaries which serialises pipelining — measurements are upper bounds on real production walls."* The `[MTP-PROF]` dump (dsv4_mtp.py:819-845) prints **cumulative running means**, not per-interval.

### 1.3 What is INSIDE the bracket (the full code span)
The entire verify phase is `dsv4_speculative_forward(self.model, verify_input, gen_batch.prompt_cache, self._captured)` (dsv4_mtp.py:4178), which executes, per the model `Model.__call__` (deepseek_v4.py:7221-7227) → `DeepseekV4Model._forward_steps` (6980-7110):

1. **Embedding lookup** (input_ids → hidden).
2. **DSpark tap** — `EXO_DSV4_DSPARK=1` is live; at each `dspark_target_layer_ids` (`[40,41,42]`, config:883) it runs `h.mean(axis=2)` (deepseek_v4.py:7049). *Not part of the target's per-layer forward path per se but runs inside the same model call.*
3. **43 per-layer block forwards** (`DeepseekV4Block.__call__`, 5134): attn_hc / attn_norm / attention / attn_residual / ffn_hc / ffn_norm / MoE / ffn_residual.
4. **Model tail** (7100+): `hc_head` + `norm`, then `lm_head` (`span("model.lm_head")`, 7270).
5. **Cross-rank collectives**, interleaved in the block loop: `all_sum` per MoE layer (3076), `all_gather`/`send` at PP boundaries (7079-7089), plus the `mx.eval`/fence per fence-event.

The EOS ban (4223-4228) and the pool snapshot are outside the strict forward but inside the `t_after_draft→t_after_verify` window (verify *timer* bracket), so they are charged to verify in the profile. The post-verify accept/rollback (4244+) is **excluded** (recorded separately).

---

## 2. Verify component enumeration + context-scaling classification

All shapes below are for the live config (43 layers, hidden 4096, head_dim 512, q_lora_rank 1024, sliding_window 128, 64 heads, vocab 129,280 — config.json, both nodes). Verify input = `(B, L=4)` rows (γ+1; γ=3).

| # | Component | Where (code) | What it computes | Shape it reads/loops over | CLASSIFICATION |
|---|---|---|---|---|---|
| 1 | **local (sliding-window) attention** — Local + Sparse layers | `LocalAttention.__call__` (4186ff); Sparse local branch 4867-4876 | SDPA over rotating local cache | local KV fixed at ≤`sliding_window`=128 | **FIXED-COST** (cache window is clamped at 128; never grows) |
| 2 | **compressor + pool write** — all pooled layers (ratio-4 & ratio-128) | `Compressor.__call__` (3182ff) → `accumulate_windows` (cache.py:1429) + `update_and_fetch` (1494) | project_kv_gate → window accumulate → compress+norm+rope → append pooled | per-layer: `OutDim × head_dim` MLP on 4 rows; pool appends 0–1 col/cycle; `pooled.shape[1]` **grows** but is only *written*, not read, here | **FIXED-COST** for the forward itself (batch/dim-bound). The pool **growth** is what makes later reads scale. |
| 3 | **CompressedAttention SDPA** — 20 layers, ratio 128 (odd 3..41) | `CompressedAttention.__call__` 4478+; concat `[kv, pooled[:,None]]` 4452 | dense SDPA over **local + full pooled** | `pooled.shape[1] ≈ context/128` (≈ 700 at 89 K) — **grows with ctx** | **CONTEXT-SCALING** (Q attends every pooled key; KV width ∝ ctx). No Indexer, no top-k — this is the ratio-128 correction. |
| 4 | **Indexer q-proj / weights_proj / score GEMM** — 21 sparse layers (even 2..42) | `Indexer.__call__` (3964ff); `_indexer_score` (3784) | `q_weighted (L,D) @ pooled (D,P) → (L,P)` | **P = pooled.shape[1] ≈ context/4** (≈ 22,250 at 89 K) — grows with ctx | **CONTEXT-SCALING** — the single GEMM is (D=512) × **(P)**; flops ∝ P. |
| 5 | **Indexer pmask apply** | Indexer 4039-4086 (tail-restricted `EXO_DSV4_TAIL_PMASK`, live) | O(L,P) mask `where` → tail-band restricted | band only `≈ L/ratio+1` cols (OPT-12) despite P huge → **≈ FIXED per row** in practice | **FIXED-COST** (band-restricted; see deepseek_v4.py:4040-4080). *Note: non-band full-P would scale; the tail gate limits it.* |
| 6 | **Indexer top-k** | Indexer 4087-4152 (`_exact_topk` kernel; `EXO_DSV4_EXACT_TOPK` default 1, live) | select k=min(512, P) of P scores, per query row | iterates P per row → **O(P)** | **CONTEXT-SCALING** (top-k scans all P keys per query row) |
| 7 | **Sparse (indexed) SDPA** — 21 sparse layers | `SparseCompressedAttention` 4893-5017 → `_sparse_pooled_attention` (2544) → `_sparse_pooled_attention_inner` (1481) | gather k=512 pooled keys + local, split-softmax SDPA | gathered tensor **(B,H,L_q,k=512,D)** fixed by `k=index_topk`; per-row gather **reads P** for `take_along_axis`/gather indices but touches only k entries (OPT-10 `reshape+gather`, 2595-2616 — "does NOT scale with P" per comment) | **FIXED-COST** per-row once top-k is done (k capped at 512). BUT it **depends on the O(P) top-k** above, so end-to-end it inherits context scaling. |
| 8 | **MoE — gate** | `DeepseekV4MoE` 2979-2992, `MoEGate.__call__` (2843) | top-k expert selection per token | batch x hidden, batch=4 | **FIXED-COST** (4 rows × dims; no context) |
| 9 | **MoE — switch_mlp (experts)** | 2994-3039, `SwitchGLU` | routed-expert GEMMs | batch=4 × expert dims (moe_inter 2048 × 256 experts) | **FIXED-COST** (batch/dim-bound; no context) |
| 10 | **MoE — shared_experts** | 3041-3056, `DeepseekV4MLP` | dense MLP | batch=4 × dims | **FIXED-COST** |
| 11 | **MoE — post_combine** | 3057-3072 `_moe_post_combine` | weighted_reduce + shared add | batch=4 × dims | **FIXED-COST** |
| 12 | **MoE — all_sum** | 3074-3151 | TP cross-rank collective on `y`; `mx.eval`/`mx.async_eval` fence | tensor full block width × dims (batch=4) | **FIXED-COST** per layer (bandwidth ∝ block dims, not ctx). Frequency governed by `EXO_DSV4_FENCE_EVERY_N_LAYERS=4` (live). |
| 13 | **lm_head** | model.lm_head (7219) `span("model.lm_head")` (7270) | hidden→vocab projection | batch=4 × hidden 4096 × vocab 129,280; mxfp8 `EXO_DSV4_LMHEAD_MXFP8=1` live | **FIXED-COST** (vocab-bound, batch 4; no context) |
| 14 | **cross-rank p2p / all_gather / send** | `model.send` (7079), `model.all_gather` (7088) | PP shard handoff | full-L hidden (L=4) | **FIXED-COST** (block width 4; PP is 2 nodes) |
| 15 | **mx.eval / sync points** | per-fence fence site (3150/3130), verify `mx.eval` (4240) | graph materialization + rank lockstep | depends on accumulated lazy graph | **FIXED-COST** (sync/drain overhead; not proportional to ctx as a kernel, though the *drained work* is — see §6 INFER) |

**Bottom line for §2:** The context-scaling components are exactly the **attention path over the compressed KV pool**: Component 3 (CompressedAttention over P≈ctx/128) and Components **4+6** (Indexer O(P) score GEMM + O(P) top-k over P≈ctx/4), which together gate the sparse gather (7). Everything else (local 128-window attention, compressor-write, MoE, lm_head, collectives) is fixed-cost at verify's small batch.

---

## 3. CRITICAL — Indexer branch analysis (cheap dense vs expensive sparse)

### 3.1 The branch, exactly
In `SparseCompressedAttention.__call__` (deepseek_v4.py:4918-5017):

```
if pooled.shape[1] == 0:                          # 4867 — local-only
    out = scaled_dot_product_attention(q, kv, kv, cache=local_cache, ...)
elif pooled.shape[1] <= self.indexer.index_topk:  # 4879 — CHEAP dense path
    full_kv = mx.concatenate([kv, pooled[:, None]], axis=2)   # 4880
    ... scaled_dot_product_attention(q, full_kv, full_kv, ...) # 4882
else:                                              # 4893 — EXPENSIVE sparse path
    ... Indexer top-k + _sparse_pooled_attention ...
```

So the branch **flips when `pooled.shape[1] > index_topk`.** Below/at that, the layer does a single dense SDPA over `local(≤128) + pooled(P)`; above it, it runs the Indexer (see §1: score GEMM `_indexer_score` at 3784 and top-k at 4087-4152) then the sparse gathered SDPA.

### 3.2 `index_topk` value
- Config default: **`index_topk: int = 512`** (deepseek_v4.py:870).
- Live override: **`EXO_DSV4_INDEX_TOPK=512`** (preflight both nodes) → `Indexer.__init__` reads it (deepseek_v4.py:3928-3929). **Effective `index_topk` = 512.**
- k actually used = `min(self.index_topk, pooled.shape[1])` (deepseek_v4.py:4087).

### 3.3 compress_ratios (live config, resolved)
Both live config.json files are identical after truncation to `num_hidden_layers=43`:

- **Layer 0,1:** `ratio = 0` → `LocalAttention` (factory 5112-5113). No pool, no indexer.
- **Layers 2,4,6,…,42 (21 layers, all even):** `ratio = 4` → **`SparseCompressedAttention`** (WITH Indexer, index_topk=512) (factory 5116).
- **Layers 3,5,7,…,41 (20 layers, all odd):** `ratio = 128` → **`CompressedAttention`** (NO Indexer) (factory 5114-5115).

> **CORRECTION to the prior finding (2026-08-04, DSpark/ROWSEQ_FULLBLOCK):** The prior note said "~21 sparse layers … flip around context≈2048 for compress_ratio=4 layers." **Current source confirms** the 21 ratio-4 sparse layers and index_topk=512. But the prior framing implied all ~21 alternate 4/128 under one indexer concept. **Current source makes the ratio-128 layers `CompressedAttention` with NO Indexer and NO branch** — they are dense-over-pooled attention and scale with ctx (P≈ctx/128) with no knee. The branch flip (knee) exists **only** on the 21 ratio-4 sparse layers.

### 3.4 Computing the flip point (per compress_ratio)

**Pool size vs context:** `PoolingCache.pooled.shape[1]` (=P) grows by 1 every `compress_ratio` decode tokens on each pooled layer — "grows by 1 every `compress_ratio` decode tokens" (deepseek_v4.py:3793); `accumulate_windows` flushes/compresses once per full window of `ratio` (cache.py:1440-1492). So **P ≈ floor(context / compress_ratio)** for that layer (with small remainder effects; overlap ratio-4 layers store overlapping windows, so P ≈ context/4 is the leading term).

**Flip condition:** `pooled.shape[1] > index_topk` ⟺ `context / compress_ratio > 512` ⟺ `context > 512 × compress_ratio`.

| compress_ratio | # layers | attention class | has Indexer? | P at 89 K | FLIP context (P>512) | State at 89,408 ctx |
|---|---|---|---|---|---|---|
| 0 | 2 (0,1) | LocalAttention | no | — | — | local-only always |
| 4 | 21 (even 2..42) | SparseCompressedAttention | **yes** | ≈ **22,350** | **ctx > 512×4 = 2048** | **EXPENSIVE (always past flip)** |
| 128 | 20 (odd 3..41) | CompressedAttention | no | ≈ **699** | no flip (no indexer) | dense-over-pool always (grows with ctx) |

**Therefore:** every ratio-4 sparse layer flipped from the cheap dense-SDPA branch to the expensive Indexer branch at **context ≈ 2,048**, and has been in the expensive branch throughout the entire 89 K measurement window (and at 150 K). Verify cost from the indexer path grows **smoothly** with P (linear in P for score GEMM + top-k) past ≈2 K — there is **no second knee** at deeper context for the ratio-4 layers. The ratio-128 layers have no knee at all, but their cost grows linearly with context (P≈ctx/128) because they dense-attend every pooled key.

> **INFER (clearly labeled):** "≈2,048" assumes P = context/4 exactly with zero initial offset. There is an initialization/remainder-dependent constant (pool starts empty, remainder starts 0), so the flip is more precisely at `context ≈ 512×4 + small_const` tokens. At any realistic serving depth (≥ 89 K) this constant is immaterial — all 21 sparse layers are unambiguously past the flip. I did not instrument the exact pooled-length arithmetic (read-only constraint), so the precise integer flip token count is not pinned to the last token; the 2,048 figure is the analytical value.

**Measured-context implication:** At 89,408 ctx the verify work includes 21×(full O(ctx/4) indexer GEMM + O(ctx/4) top-k + k=512 sparse SDPA) + 20×(dense SDPA over P≈699) — all context-scaling — plus fixed batch-4 MoE/lm_head. This is the mechanism half of "why verify dominates and grows with context"; it predicts **verify scales approximately linearly with context** (dominated by the 21 indexer layers' O(ctx) score+topk), with a soft knee near 2 K (already passed).

---

## 4. Available instrumentation to attribute the 56 ms (no new code)

### 4.1 Already inside verify under the live env
The live env already has these ON (preflight both nodes; task's list PLUS `EXO_DSV4_VERIFY_BATCH=1` which the task list omitted):

| Env var (live) | Gated instrumentation | What it would reveal | Already ON? |
|---|---|---|---|
| `EXO_DSV4_MTP_PROFILE=50` | `prof.record` + `[MTP-PROF]` dump every 50 cycles (dsv4_mtp.py:816, 823-845); phases `draft, verify, accept, commit, rollback, total` (+ RB sub-phases) | verify mean/min/max wall + share vs draft/accept/rollback (cumulative running mean, serialized eval) | **YES** |
| `EXO_DSV4_RB_PROFILE=1` | RB sub-phase boundaries incl. `rb_snap` (dsv4_mtp.py:258, 4136-4138) | separates snapshot/arm from true verify; already closed at 0.15 ms | **YES** |
| `EXO_DSV4_SECTION_TIME` | GPU-time per-section `attn/ffn/other` acc (deepseek_v4.py:174-191, 7189); needs `EXO_DSV4_SECTION_TIME_LOG_EVERY` | attn vs MoE true GPU share inside verify — **NOT currently on** (var absent) | **NO** |
| `EXO_DSV4_SECTION_TIME_LOG_EVERY` | dump cadence for the above | — | **NO** |
| `EXO_DSV4_TAIL_PMASK=1` (default ON) | band-restricted pmask (4040-4080) | already limits O(L,P) mask cost; not a diagnostic | env ON (inert w/o SECTION_TIME) |

### 4.2 Off / would-need-turning-on sub-phase diagnostics (all in model file)
The profiler **already has the sub-phase spans** (`span("attn")`, `span("indexer")`, `span("attn.gather")`, `span("moe.gate")`, `span("moe.switch_mlp")`, `span("moe.post_combine")`, `span("model.lm_head")`, `span("model.final_norm")`, etc. — deepseek_v4.py throughout) and `_ATTN_SUB_ACC` (`compressor/proj_qkv/qk_prep/indexer/sdpa/out_proj`, deepseek_v4.py:212-215, populated at 4770/4797/4826/4854/5019). These are the **exact** per-component timers that would attribute the 56 ms — but they are gated behind `_SECTION_TIME_ENABLED`:

- **`EXO_DSV4_SECTION_TIME=1`** → enables `span`/`finalize` profiler hooks + the section accumulators (deepseek_v4.py:186). **This is the single most useful existing knob: it yields true GPU kernel shares for indexer vs sdpa vs moe inside verify.** Currently OFF.
- Set **`EXO_DSV4_SECTION_TIME_LOG_EVERY=N`** (e.g. 1) to dump per-forward (deepseek_v4.py:187, 7189-7193).
- Cost caveat (source): when ON it inserts ~4 `mx.synchronize()`/layer (deepseek_v4.py:182-183) — serializes pipeline; **shares accurate, absolute totals are upper bounds** (same caveat as EXO_DSV4_MTP_PROFILE).

### 4.3 Other off diagnostics that bear on verify components
| Env var | Reveals | Currently? |
|---|---|---|
| `EXO_DSV4_ROUTE_HIST=1` (+ `_DECODE_ONLY`) | expert routing histogram per layer (2940-2955) — validates MoE batch path | OFF |
| `EXO_MOE_EXPERT_HIST_DIAG=1` | captures lazy `inds`; consumer in pp_speculation.py:491 computes expert-assignment histogram riding the real eval | OFF |
| `EXO_MOE_GPUTRACE_DIAG=1` | wraps `all_sum`/collective start_capture/stop_capture for Metal gputrace (pp_speculation.py:2643-2673) — isolates all_sum cost | OFF |
| `EXO_DSV4_FENCE_GATE_DIAG=1` | logs why the async-fence gate fails (3156-3149) — diagnostic for the all_sum fence cost | OFF |
| `EXO_DSV4_TOPK_OVERLAP_LOG=1` | per-step Jaccard overlap of consecutive top-k sets (3963-3981) — quantifies whether full O(P) rescoring is redundant | OFF |
| `JACCL_TRACE_PROGRESS=1` (set by start_cluster.sh, per pp_speculation.py:64) | C++-side jaccl `[jaccl-prog]`/`[jaccl-p2p]` wire traces | (set by cluster script; not a Python counter) |
| `EXO_DSV4_SPEC_TRACE=1` | per-cycle committed-token + cache-offset divergence trace (1659-1666) | OFF |
| `EXO_DSV4_C2_TRACE=1` | per-cycle per-chain-step JSONL trace (892) | OFF |
| `EXO_DSV4_MTP_TRANSITION_TRACE=1` | uid-transition bookkeeping trace (1849) | OFF |

### 4.4 Honest limitation
There is **no existing per-layer or per-attention-class timer** that is on by default, and **no jaccl comms counter exposed to the engine that already sums inside verify**. The 56 ms is currently a monolithic upper-bound wall bracket. To attribute it without new code you must turn ON **`EXO_DSV4_SECTION_TIME=1` (+ `EXO_DSV4_SECTION_TIME_LOG_EVERY`)** — that is the pre-built instrument that splits verify into attention vs MoE and, with `_ATTN_SUB_ACC`, into compressor/indexer/sdpa within attention.

> Note: enabling SECTION_TIME is a **relaunch-level** change (reads env at import). The task says do not relaunch — this is flagged as the *instrumentation inventory only*, not something done here.

---

## 5. Batched vs per-row verdict (γ=3 → verify batch ≈ 4 rows)

**Verdict: BATCHED.** Under the live env, the verify forward processes the 4 rows (γ+1, γ=3) as **one batched M=4 model call** — NOT per-row loop — at the measured 89 K context.

**Deciding code (deepseek_v4.py):**
- `_VERIFY_BATCH = os.environ.get("EXO_DSV4_VERIFY_BATCH", "0") == "1"` (1648). **Live: EXO_DSV4_VERIFY_BATCH=1** (preflight both nodes). This var was **omitted from the task's env list** but is present in the actual running config.
- `_VERIFY_BATCH_MIN_CTX = int(...) or 8192` (1661). Live uses default 8192.
- Activation (7012-7044): `_vb_active = _VERIFY_BATCH and h.shape[0]==1 and 2<=h.shape[1]<=_VERIFY_ROWSEQ_MAX_L and _vb_ctx_len >= _VERIFY_BATCH_MIN_CTX`. At 89,408 ctx ≥ 8192 → **active=True** → `_set_verify_batch_ctx(active=True)` (7044).
- `DeepseekV4Block` gates: the rowseq paths are suppressed while `(_VERIFY_BATCH and _VERIFY_BATCH_CTX["active"])` (5236, 5383), falling through to the **batched `self.attn(normed, mask, cache)` (5435) and batched `self.ffn(...)` (5448)** over all L rows.
- The single-uid verify input is `(B=1, L=γ+1=4)` (dsv4_mtp.py:4108-4110). With `B=1` and `L=4` the block forward is one batched attention (fused SDPA `L_q=4`) + one batched MoE.

**What WOULD change it to per-row (for completeness):**
- If `_VERIFY_BATCH` were off (`EXO_DSV4_VERIFY_BATCH=0`) and `EXO_DSV4_VERIFY_ROWSEQ=1` + `EXO_DSV4_ROWSEQ_FULLBLOCK=1` (both default-off but **live**: preflight lines 39, 50), then with `_VERIFY_ROWSEQ_MIN_CTX=0` (live 51) the FULLBLOCK per-row loop (5244-5368) runs each row's attn individually and batches only the MoE.
- Seq-split (`EXO_DSV4_SEQ_SPLIT=1`, live) is length-gated at `L >= _SEQ_SPLIT_MIN_L=16` (230-231, 4742-4744); verify L=4 < 16 so seq-split **does not fire** for verify regardless.
- `EXO_DSV4_SPARSE_SDPA_TILE=128` (live) tiles the sparse SDPA only when `_Lq > _tile` (4986/4943); L=4 < 128 → **single tile, no tiling loop** for verify.

**So the batch is genuinely M=4 end-to-end** (attention fused SDPA L_q=4 + batched MoE + batched lm_head), giving the fixed-cost components a batch-4 denominator rather than 4×batch-1.

---

## 6. Explicitly could-not-determine (read-only limits)

1. **Exact integer flip token for each ratio-4 layer.** I derived `context > 512×4 = 2048` analytically (P≈ctx/4), but did not pin the exact pooled-length offset/remainder arithmetic (would require reading `Compressor`/`update_and_fetch` growth constants exhaustively or instrumenting). The conclusion "all 21 sparse layers are past the flip at ≥89 K" is unaffected.
2. **True GPU-vs-wall split inside the 56 ms.** The live profile is a serialized upper-bound wall bracket; the actual share of indexer-GEMM vs sdpa vs MoE vs sync is only obtainable by enabling `EXO_DSV4_SECTION_TIME` (relaunch — not done here).
3. **All_sum / jaccl comms measured cost within verify.** Existence and gating are confirmed in source; exact microseconds under the live TP-2 topology are not (no live counter on by default; jaccl wire traces are C++-side via `JACCL_TRACE_PROGRESS=1` set by cluster script).
4. **Whether the sync/fence drain (Component 15) scales with ctx.** INFER it reflects the drained O(ctx) indexer work, but its fixed per-cycle overhead vs context-proportional component was not separable from source alone.
5. **Exact `pooled.shape[1]` at 89,408 for each ratio** — I used P≈ctx/ratio (leading term). Remainder/window-boundary quantization at a specific position is not pinned without runtime state.
6. **The precise `DSpark` tap cost** (~`h.mean` at layers 40-42 under `EXO_DSV4_DSPARK=1`) — runs inside the model call and thus inside the verify bracket, but its microseconds are not isolated by any live counter.

---

## §7 (reference) Live env relevant to verify — as measured
From `tmp/restore-default-20260901/preflight_node1_ps_eww.txt` (node2 identical), the vars governing verify:
```
EXO_DSV4_VERIFY_BATCH=1            # <-- omitted from task list; KEY for §5
EXO_DSV4_VERIFY_BATCH_MIN_CTX=8192 # default anyway
EXO_DSV4_VERIFY_ROWSEQ=1           # inert while VERIFY_BATCH active ≥8192
EXO_DSV4_ROWSEQ_FULLBLOCK=1        # inert while VERIFY_BATCH active ≥8192
EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX=0
EXO_DSV4_ROWSEQ_ROWMASK=1
EXO_DSV4_VERIFY_ROWSEQ_VEC=1
EXO_DSV4_INDEX_TOPK=512            # == model default; k used
EXO_DSV4_SEQ_SPLIT=1               # inert for verify L=4 < 16
EXO_DSV4_SPARSE_SDPA_TILE=128      # inert for verify L=4 < 128
EXO_DSV4_LMHEAD_MXFP8=1
EXO_DSV4_DSPARK=1
EXO_DSV4_MTP=1
EXO_DSV4_MTP_PROFILE=50
EXO_DSV4_RB_PROFILE=1
EXO_SPECULATIVE=1
EXO_SPECULATIVE_GAMMA=3
EXO_KV_CACHE_BITS=0
EXO_COMPUTE_DTYPE=bf16
EXO_DSV4_FENCE_EVERY_N_LAYERS=4
EXO_DSV4_FENCE_ASYNC=1
EXO_DSV4_EXACT_TOPK_PREFILL=1
EXO_DSV4_EXACT_TOPK=1 (default; not listed but default on)
EXO_DSV4_TAIL_PMASK=1 (default on)
EXO_DSV4_ARGPARTITION_MIN_P=8192
EXO_DSV4_QUERY_TILED_SDPA=1
```
`EXO_DSV4_SECTION_TIME` is **absent** from live env (off). `EXO_DSV4_SPEC_EOS_BAN`, `EXO_DSV4_TREE_DRAFT`, `EXO_DSV4_INDEXER_PBLOCK`, `EXO_DSV4_SPARSE_FUSED_SDPA`, `EXO_DSV4_SINGLE_GATHER` are all off/at defaults per preflight + source defaults.
