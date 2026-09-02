# Item 6 — Indexer sliding-window bound: premise verification + correctness/memory harness inventory

Repo: `~/repos/exo`, branch `main`, HEAD `cb1f91903`. Read-only verification; no code/commit/SSH/bench executed.
Model: DSv4-Flash (TP, 2 nodes). All file:line citations are against HEAD.

---

## VERDICT (summary)

| # | Premise / gate | Finding |
|---|---|---|
| a | Indexer lookback is unbounded? | **YES — unbounded.** `_indexer_score` scores every query row over the FULL pooled prefix `P` (`deepseek_v4.py:3849`, `(B,L,P)`). No window anywhere on the score path. |
| b | Indexer is the only O(P) term? | **NO.** Main sparse SDPA is also O(P) (`(B,H,Lq,k,D)` gather with `k=index_topk=512` fixed — this is O(P) in *work per token* only in the sense of being fixed-k; more importantly **the indexer is only ~4.0–4.7% of prefill wall time**, so bounding it alone can flatten at most that. The depth curve is dominated by attention SDPA (~13.6%), compressed SDPA (~11.8%), MoE (~26.9%), collectives (~18%) — NOT the indexer. |
| c | Existing knob? | **NO functional knob.** `EXO_DSV4_INDEXER_WINDOW` exists in `start_cluster.sh:26` and a test fixture, but has **no reader anywhere in model/generator code** — it is dead on current `main`. No code change is required to A/B it (start_cluster comment documents the intent) but it will have ZERO effect until wired into `_indexer_score`. A sliding/bounded window would be a **code change**, not a config flip. |
| d | Best correctness harness | `bench/ab_probe_tier1.py` — self-contained needle-in-haystack: builds a fresh ~`target_tokens` (default 100K) prompt with a unique random run seed + cached-token-buster, needle `FALCON-MERCURY-7749` at 40% depth, asserts `needle_hit` and BOS-spam, also reports prefill/decode t/s and reads `reasoning_content`. Depth-agnostic (any `target_tokens`). Caveats: uses `~4 chars/token` heuristic (NOT the real ~5.68 ratio for prose) and hardcoded `API=192.168.86.201`. |
| e | Memory evidence supports clear-cache interval=8 safety? | **NO — and `interval=8` is currently impossible.** The knob `EXO_PREFILL_CLEAR_CACHE_INTERVAL` is **dead too** (no Python reader; `generate.py:1527` calls `mx.clear_cache()` unconditionally every chunk under `prefill_batched`). So "76% headroom at 150K" cannot be mapped to any interval-8 experiment today. The best real headroom measurement (Section 27) shows 2×concurrent-150K at peak resident **87.3 GB (node1) / 84.1 GB (node2)** vs a 115 GB wired ceiling (~25+ GB headroom), 0 swap — but it was measured with the standing config (clear-cache-every-chunk). It does **not** bound the allocator-pool growth if cache is skipped 7 of 8 chunks. |

**Bottom line:** The reviewer's indexer premise **fails on cost** (indexer is <5% of wall; bounding it cannot flatten a 426→406 t/s depth curve), and the two other gates (clear-cache interval, memory at 150K) rest on **dead env knobs and a non-matching measurement** respectively. Do not fund an engineering project; at most a cheap argpartition A/B already exists (Section below).

---

## 1. Is the lookback actually unbounded? — YES

**Indexer scoring path (read-only):**
- `Indexer.__call__` → `_indexer_score(...)` at `deepseek_v4.py:3980` (`pooled = self.compressor(x, pool_cache, offset)`), then `:4014-4031` calls the score function with the full `pooled` tensor.
- `_indexer_score` (`deepseek_v4.py:3784-3849`): the core is line **3849**:
  ```python
  return q_weighted @ pooled.swapaxes(-1, -2)  # (B, L, D) @ (B, D, P) -> (B, L, P)
  ```
  `pooled` **is the entire compressed-prefix pool** (`P = ceil(ctx / compress_ratio)`). There is **no slicing, no window, no cap** on `P` at the score call.
- Top-k (`:4087`): `k = min(self.index_topk, pooled.shape[1])` — `index_topk` caps the *selected* set size (512 default) but does NOT bound the scores tensor; the full `(B,L,P)` scores are computed and then argpartition/argsort over ALL of P. `EXO_DSV4_TOPK_FUSED` cannot engage at prefill (`scores.shape[1]==1` decode-only, `:4104`).
- `EXO_DSV4_INDEXER_PBLOCK` (`:333`, `_indexer_score_tiled` `:3869`) tiles P for **allocation** only — it concatenates the per-tile collapses back to the FULL `(B,L,P)` (`:3893-3913`). It is a memory-bounding tile, NOT a lookback bound. Bit-identical, prefill-neutral (see Item 7 note below).

**Conclusion: unbounded.** The reviewer's "currently unbounded lookback" is factually correct at the code level. The premise that it is thereby "the only linear-in-P term worth fixing" is what fails (below).

---

## 2. Is it the only linear-in-P term? — NO

Per-token costs that grow with prefix length P on the prefill path:

1. **Indexer score GEMM** — `(L,D=128) @ (D,128,P)`, O(P) per query token (OPT-6 folded, `:3835-3849`). Grows with P.
2. **Indexer top-k** — argsort over `(L,P)` scores = O(P log P) per query (worse than linear), `:4152`. Bounded only by `index_topk` on the *output*.
3. **Compressed-layer SDPA** — `CompressedAttention.__call__` concatenates the full pooled KV onto local KV (`:4452` `kv = mx.concatenate([kv, pooled[:, None]], axis=2); :4453 mask = _extend_mask(...)`) then full SDPA over `kv.shape[2]` which includes ALL pooled columns. **O(P) work per token** on the 20 compressed layers (ratio=128). This is NOT bounded by `index_topk` at all.
4. **Sparse-layer SDPA** — `SparseCompressedAttention.__call__` gathers `(B,Lq,k,D)` where `k=index_topk=512` per row (`:4959-4964` `_pooled_flat` + `_topk_flat` gather, then `_sparse_pooled_attention`), so **per-row attention work is fixed-k** (not O(P)) — but it scales with Lq and is the biggest single attention span.
5. **KV gather cost / pool read** — the pooled KV tensor is read for SDPA (O(P) bytes per layer, ~32 GB/chunk per the roofline note).

**Which dominates (measured span profile, 220K, `docs/dsv4-220k-prefill-span-profile-2026-08-18.md:79-115`):**
```
attn                 58.4%   (all attention)
ffn                  41.6%
  attn.sdpa          13.6%
  attn.sdpa.compressed 11.8%
  attn.o_proj        10.0%
  moe.all_sum         9.5%
  attn.proj_qkv       8.9%
  attn.all_gather     8.5%
  attn.indexer        4.0%   <—— the reviewer's target
  ...
  indexer.score       0.0%  (lazy-graph build only, not GPU)
  indexer.topk        0.0%
```
Also `docs/indexer-prefill-decomposition-2026-08-24.md` re-attributes `attn.indexer` = **4.0% of wall** (→ ~4.7% by the doc's own recount) and concludes **"CLOSED — no un-tried fused-kernel candidate with predicted e2e ≥ 1% exists"** (`:199-230`). FLOP-share table (`:100-104`): at 100K/220K/500K, the score GEMM is 19.9%/35.4%/55.5% *of the indexer's own GFLOPs* — but that is a fraction of a 4% span, so even eliminating it entirely tops out at a few percent e2e, and the depth curve (426→406) spans ~5% of which indexer is <1/5.

**Quantify:** 4.0% of wall at 220K is the ceiling any indexer-bounding idea can address. Even a *perfectly free* indexer (both score and topk) buys at most ~4-4.7% e2e — roughly the width of the whole 426→406 t/s decline, and only if P is where that decline lives (it mostly lives in SDPA/MoE, which the same profile shows scaling). **Bounding the indexer cannot flatten the depth curve.**

---

## 3. Is there an existing knob? — NO functional one

- **`EXO_DSV4_INDEXER_WINDOW`** — declared in `start_cluster.sh:26` (default unset `:=`), commented as "W=8192 caps the indexer at ~65K raw tokens of lookback" and "8192 was the validated winning combo per the dsv4 sliding-indexer plan." **BUT: `git grep -rn "EXO_DSV4_INDEXER_WINDOW"` finds it ONLY in** `start_cluster.sh`, `bench/trusted_measurement/fingerprint.py:220-223`, and a test fixture (`test_registry_reconciliation.py:175,188`). **There is NO reader in any model/generator `.py`.** It is a dead knob on current `main` — a vestigial plan/schema artifact.
- **`EXO_DSV4_INDEX_TOPK`** — live (`deepseek_v4.py:3928-3929`, `:4087`). Caps the *selected set* k (512 default), not lookback. Lowering it shrinks sparse-SDPA per-token k, not the score scan.
- **`EXO_DSV4_PREFILL_ARGPARTITION` / `EXO_DSV4_ARGPARTITION_MIN_P`** — live (`:4144-4148`), changes topk sort from O(P log P) argsort to O(P) argpartition. Same top-k set, softmax-order-invariant. The prefill-indexer doc's own recommendation (`indexer-prefill-decomposition...:224-242`) is to run this single live A/B (no new code) before any kernel work.

**Implication:** An indexer sliding-window bound requires wiring a real reader into `_indexer_score`. That is an engineering project (adds a correctness risk), not a config A/B. The one cheap existing A/B is `EXO_DSV4_PREFILL_ARGPARTITION=1` at 220K/500K, projected ≤0.4% e2e (`:166-174`).

---

## 4. Correctness harness inventory (for a window-bounding ablation)

The design's requirement: bounding lookback must not silently destroy long-range recall. Existing harnesses:

**A. `bench/ab_probe_tier1.py` (best fit, self-contained).** `ab_probe_tier1.py:35-55` `build_prompt` → fresh random-run prompt (~`target_tokens`×4 chars, `uuid` cache-buster `:41-44`), seeds needle `FALCON-MERCURY-7749` at 40% depth (`:49`), asks for the code only. Asserts `needle_hit` (`:118`) and `bos_spam` (`:117`), reports `prefill_tps`, `decode_tps`, `output_head/tail`. Invoke: `python3 bench/ab_probe_tier1.py [TARGET_TOKENS] [--max-tokens N] [--tag X]`. Note: needle depth is a **single fixed 40% position** — a window-bound ablation that wants to test recall at varying distances would need multiple needle positions (40% only tests one distance; a windowed indexer that can still reach 40% but not deep/early content would pass this harness). Hardcoded API host `192.168.86.201:52415` (`:20`). `--max-tokens 1500` default may be truncated by reasoning (`reasoning_content` budget — see pitfall in exo-cluster-operations #60).

**B. `bench/quality_probe_dsv4.py`** — per skill docs (`exo-cluster-operations` #41): the "quality gate" harness. `--concurrency N --iters M`, gates on 0 special-token-leak + 0 bistability; reads `content` (expects real answer, e.g. "Paris"), reads `reasoning_content` + `finish_reason`, flags `reasoning_truncated`. Path exists in repo. Not a deep-context recall probe — good supplementary but not needle-driven.

**C. **Needle-recall methodology used in live perf runs** (not a named script, pattern): fresh random-word-salad prompt with unique embedded secret code, `cached_tokens: 0` to prove fresh prefill, `finish_reason="stop"` + code-recall as the pass (e.g. `docs/dsv4-220k-prefill-span-profile-2026-08-18.md:52-53`, `docs/dsv4-clear-cache-interval-2-test-2026-08-19.md`). `bench/context_stress.py` also ships a memory-pressure + needle loop (`:190` reads `memory_pressure`).

**Rec depth support:** `ab_probe_tier1.py` is depth-agnostic via `target_tokens` positional (tested to 100K-500K); needle depth is the limit — need a param to vary needle position for a genuine long-range-recall ablation. No harness currently sweeps needle-at-N-distances.

---

## 5. Memory-headroom evidence for a clear-cache interval sweep

**The knob is dead → the stated gate cannot be tested today.**
- `EXO_PREFILL_CLEAR_CACHE_INTERVAL`: declared `start_cluster.sh:107` (`:=1`), referenced in `bench/fingerprint.py:153` and two bench comments, **but no read anywhere in `src/`**. Under `prefill_batched`, `mx.clear_cache()` is called **unconditionally every chunk** at `generate.py:1527` (and once before the loop `:1386`). There is no `_next_count % interval` gate the way `EXO_MLX_CLEAR_CACHE_INTERVAL` gates decode (`batch_generate.py:103,4736-4739`). So an "interval sweep" is currently a **no-op** — any value is ignored; the standing behavior is clear-every-chunk regardless.
- `EXO_MLX_CLEAR_CACHE_INTERVAL` (decode, `batch_generate.py:103`, default 0) IS live and is the mechanism that actually varies clear cadence — the recent DSpark work recommends `=64` (`docs/dspark-352k-residency-analysis-2026-08-27.md:35`). Any "skip N chunks" experiment must target this decode-time knob or re-wire a new prefill-side gate.

**Best existing 150K measurement (Section 27, `hybrid-pp...2026-08-04.md:5368-5383`):**
- 2× concurrent **150K-token** requests, peak resident **87.3 GB (node1) / 84.1 GB (node2)**, vs a 115 GB/node `iogpu.wired_limit_mb` ceiling → **~25+ GB headroom each node**; wired memory flat at 3.1-3.5 GB (all growth is `active` = KV+activations); `vm.swapusage` ~0. Verdict PASS on the wired-memory ceiling.
- **Date:** 2026-08-09 (doc Section 27, attempt #6). The user's "76% headroom / ~11.7 KB/token" likely derives from Section 90/91's **97% free by `memory_pressure`** (`:12215,12220`) and the **11.7 KB/token KV slope** measured 2026-08-09-16 (`:12287-12296`, n=17 samples, 11,264→92,003 tokens, floor 85.68 GB). Note the honest figures (97% free, ~25+ GB headroom on the wired metric, 0 swap) are strictly *better* than "76%", so the reviewer's number is a conservative (low) reading.
- **CLEAR_CACHE_INTERVAL state at measurement:** Standing prod config = clear-cache-every-chunk (the only prefill behavior available). So this headroom was measured WITH the allocator being periodically reaped (line 1527 semantics).

**The nuance the task flags — headroom measured with cache-clear running does NOT bound interval=8:**
- `mx.clear_cache()` releases MLX's *caching allocator* back to the Metal heap; skipping it lets dead buffers accumulate in the pool. The 87.3/84.1 GB figures are the resident (active+wired) footprint at 150K *including whatever pool accumulation existed after up-to-73 clears* (150K/2048 ≈ 73 chunks). Skipping 7 of 8 clears removes ~64 of those reap points → the allocator pool can grow by roughly the per-chunk transient working set that clear_cache currently frees. The model's steady-state KV is ~11.7 KB/token (small vs weights) but the *allocator transient* during each chunk's forward (indexer `(B,H,L,P)` transients avoided by PBLOCK; SDPA gathers; MoE dispatches) is what pool accumulation would retain.
- **What the measurement supports:** 2×150K with clear-every-chunk has ~25+ GB wired headroom and 0 swap → there is genuinely spare RAM to absorb *some* pool growth.
- **What it does NOT support:** that skipping 7/8 clears stays under the 115 GB ceiling. The un-freed pool size at interval=8 is not measured in any artifact found; it depends on per-chunk peak transient (L-dependent) that the 2026-08-18 indexer-PBLOCK work found spikes at depth (`~22ms spikes at 360K`, `:329`). An interval=8 experiment **requires a fresh headroom measurement under interval=8** (via a re-wired knob) before trusting it.
- KV growth rate documented: **11.7 KB/token** cold-prefill slope (`hybrid...:12292`), independently reproducing the ~12 KB/token figure (`:12296`); a "growing-session" 45 KB/token rate exists for multi-turn (`:3708`) but that's decode-side warm memory, not prefill.

---

## Caveats / honest flags
- All percentages are from the 220K span profile (2026-08-18, profiled — a ~10-15% throughput tax applies to absolute tok/s but relative span shares are the usable quantity). `attn.indexer` is 4.0% there.
- `EXO_DSV4_INDEXER_WINDOW` and `EXO_PREFILL_CLEAR_CACHE_INTERVAL` being dead is a HEAD-verification (`git grep` on `cb1f91903`); a live cluster runner might run different code but that is not this task's remit.
- The 2×150K memory run's own HTTP layer returned `finish_reason=None`/`needle_found=False` on both streams (`hybrid...:5385-5403`) — the memory numbers are real but the needle-recall assertion in that one run did not complete cleanly.
- No artifact matching an exact "76% headroom" string or a clean "CLEAR_CACHE_INTERVAL=1-at-150K→interval-8" A/B was found; `76%` and `interval=8` appear to be the reviewer's paraphrase, not a citation.
