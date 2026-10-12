# DSv4.1 — prefill levers RE-PRICE (post-qsplit): what's left, priced (2026-10-12)

**Date:** 2026-10-12 · **Author:** p20 campaign (delegated, offline) · **Branch:**
`p20/prefill-levers-reprice-2026-10-12` (worktree `/private/tmp/p20-reprice`, off `origin/main`
`06d134c95`).
**Live tree:** exo `deploy/next23b-qsplit @ 60008939d` + mlx-lm `6a5c78f`, 2-rank TP (`worldSize=2`,
jaccl/RDMA, one TB5 pair), `DSV41_INDEXER_QSPLIT=1` + `EXO_DSV4_CKPT=1`.
**Trigger:** the indexer query-split shipped live (+10.5 % / +11.9 % @50 / 105 K `[MEASURED]`;
token-identity 5/5; battery CLEAN). The owner asks for *more* prefill speed in the same family
("break things up more / two work streams per node in parallel?").
**Labels:** every number is `[MEASURED]`, `[DERIVED]`, `[BOUND]` or `[MODELED]`. **OFFLINE: 0 boots,
0 config changes, 0 node edits, no cluster contact.** All code citations are to the live tree above.

---

## VERDICT

- **Dual-microbatch comm-hiding: PARK PERMANENTLY.** Modeled **centre ≈ +1.5 % e2e prefill**
  (range ≈ [0 %, +3 %]), depth-flat over 25–130 K — **<5 % at every owner depth**. Engine work
  ≈ **13–22 eng-days** (centre 16). The family's overlap class was already shipped once and measured
  FLAT, and the async analog is already live at +1 %.
- **Remaining replicated set: CONFIRM "thin/diminishing."** Best standalone item is HC at
  **≈2.7 % `[MODELED]`** (perfect 2-way shard of 5.2 %, before sync) — under both the 3 % floor and
  the 5 % gate. Everything else is <2 % raw or structurally unshardable. Perfectly sharding the
  *entire* itemized set (8.25 %) yields ≈ **4.3 % `[MODELED]`** — still <5 %.
- **Indexer absolute-cost follow-up: the one direction with a real ceiling** (the indexer is
  31.1 % @100 K / 42.9–44.2 % deep `[MEASURED]`). Priced at **≈3–6 eng-days**, mostly
  offline-analyzable. **Recommended next slot** (scoping — not started here).
- **Un-run ABs: both closed.** Fence sweep = **CLOSED-ON-EVIDENCE** (vestigial knob + neutral live
  analog); `ASYNC_EVAL` = **NEVER-RAN** (deprioritized; eval-boundary class already closed at depth).
  Neither is worth a test window.
- **Honest bottom line:** with the ≥5 % bar applied, **no lever re-priced in this doc clears it.
  The "break it up more" family is exhausted** (the indexer split was the last large splittable
  term). The next slot should go to the indexer's *absolute* cost (§3), where the wall actually is.

---

## 1. Dual-microbatch comm-hiding — RE-PRICE

**Idea** (parked twice; `PH:11419` calls it "unpriced"): split each prefill chunk (2048 rows) into
**2 microbatches**; overlap microbatch **A**'s TP `all_reduce` (`all_sum`) with microbatch **B**'s
compute. Owner phrasing: "two work streams per node."
**Governing fact:** both microbatches share **ONE** GPU — this hides **waits/collectives only, never
compute**. The ceiling of the idea is the comm term itself; total compute is unchanged.

### 1(a) Modeled e2e band vs the ≥5 % bar

| quantity | value | label / source |
|---|---|---|
| TP `all_sum` share of chunk wall (the target) | central **~6–9 %**; BOUND **1–24 %** | `[MEASURED/BOUND]` measure-pack Task B (all_sum unmeasurable via `gpu_time`, bounded 0.08–1.9 s/chunk); `PH:10533` "all_sum 6–9 % of deep-prefill wall"; M2 sync-span attribution `attn.all_sum` 5.5–6.9 % @~340 K (`PH:10460`) |
| analytic RDMA-payload **floor** of that collective | 84–336 ms/chunk = **1.05–4.2 %** of the 7975 ms chunk wall | `[DERIVED]` measure-pack Task B (chunk wall 7975 ms `[MEASURED]`) |
| `dispatch / GPU-idle` bucket (`WALL − Σ(bracketed)`) | **~9.6 %** — *separate* term; a split adds CPU dispatch, does not remove it | `[BOUND]` measure-pack Task B |
| perfect-overlap ceiling (whole comm/overlap class) | **~10.5 %** best case from a real 9.5 %-of-wall measurement; "realistically 5–7 % after imperfect overlap/scheduling overhead" — *flagged as possibly not worth a risky live deployment* | `[MEASURED/BOUND]` `PH:415-421` |

**Ceiling, stated plainly:** a term of share `S` yields ≤~`S` e2e even if perfectly hidden. Target
term ~6–9 % ⇒ **absolute ceiling ≈ +6–10 %**, i.e. barely at the bar and only at φ≈1. It is not
reachable:

1. **Already partly live.** `EXO_DSV4_FENCE_ASYNC=1` (non-blocking `async_eval` at the per-layer
   `all_sum`) is the production default and **this IS the comm/compute-overlap design**
   (`docs/comm-compute-overlap-already-exists-2026-08-21.md`); re-measured **+1.04 % decode** (8 reps,
   1.24×σ), historic +28 % **UNRECONCILED/unsupported**. Note **prefill is excluded from it by
   construction**: the async branch is gated `y.shape[0] <= _FENCE_ASYNC_MAX_B and y.shape[1] <= 8`
   (`deepseek_v4.py:3811-3817`; comment: *"Unarmed cases (prefill, B over limit, transitions) keep
   the blocking fence"*) — prefill L=2048 runs the blocking fence today. So the measured +1.04 % is
   the *decode* analog of exactly this mechanism and it is small.
2. **The escape hatch is dead.** Overlapping compute with an in-flight `all_sum` needs either
   (1) pre-evaluating the collective input, or (2) issuing the overlapping compute on a **second GPU
   stream** (`docs/phase0b-collective-overlap-gate-2026-08-20.md`). Hatch (2) is measured **NEUTRAL**
   ("stream overlap: neutral — MLX serializes streams"). Hatch (1) *is* the blocking drain. Worse,
   ring **and** jaccl pin **one** CPU collective stream per rank (`ring.cpp:465` `to_stream(cpu)`;
   `jaccl.cpp:88-95`), so A's and B's `all_sum`s **FIFO-serialize on that single stream** — the
   structure cannot keep two useful collectives airborne.
3. **The direct empirical analog is FLAT.** The comm/compute-overlap class shipped once as
   `EXO_PREFILL_CHUNK_OVERLAP` and measured **FLAT live**, ceiling ~1–3 %, likely ~0 % — **CLOSED**
   (`docs/prefill-chunk-overlap-race-fix-2026-08-20.md`, `docs/prefill-chunk-overlap-live-test-2026-08-20.md`).
   And the **fence-spacing A/B** (`EXO_PREFILL_FENCE_EVERY=8` vs `2`; 15 fewer blocking drains/chunk)
   was **NEUTRAL ±0.3 %** at 100–200 K (`PH:10537-10558`) — *"if 15 fewer blocking drains per chunk
   move nothing, the eval-boundary class is closed at this depth."*

**Band.** Model `Δe2e ≈ C(depth) × φ` (`C` = `all_sum` share of wall; `φ` = recoverable fraction,
centre **0.20**, range 0.10–0.30 `[MODELED]`; `C` tilts down with depth as the O(context) indexer
dilutes the denominator):

| depth | `C` = all_sum share `[BOUND]` | CEILING (φ=1) | Δe2e centre `[MODELED]` | Δe2e range |
|---:|---:|---:|---:|---:|
| 25 K | 8–11 % | +8–11 % | **+1.9 %** | +0.8 … +3.3 % |
| 50 K | 7–9.5 % | +7–9.5 % | **+1.7 %** | +0.7 … +2.9 % |
| 100 K | 6–9 % | +6–9 % | **+1.5 %** | +0.6 … +2.7 % |
| 130 K | 5.5–8.5 % | +5.5–8.5 % | **+1.4 %** | +0.6 … +2.6 % |

**Centre ≈ +1.5 % (depth-flat); range ≈ [0 %, +3 %]. <5 % at every depth. PARK PERMANENTLY.**
Do not confuse with the ~9.6 % dispatch/idle `[BOUND]` bucket — that is a CPU-dispatch lever,
unaffected by a microbatch split.

### 1(b) Engine-work estimate ≈ 13–22 eng-days (centre 16)

| # | item | code locus | days |
|---|---|---|---:|
| 1 | Split chunk → 2 microbatches; restructure the layer loop (today one `model(...)` per 2048-row chunk as a single graph) | `src/exo/worker/engines/mlx/generator/generate.py:1762-1892` | 3–5 |
| 2 | Two activation buffers + fit vs the transient-budget chunk policy | `generate.py` chunk planner; `deepseek_v4.py` builder | 1–2 |
| 3 | Per-microbatch `all_sum` scheduling on the **shared CPU collective stream** — and **two in-flight collectives FIFO-serialize** (single pinned stream), which reshapes the scheme. Load-bearing unknown. | `deepseek_v4.py:3772` (MoE all_sum); `:5026-5031` (attn all_sum) | 2–3 |
| 4 | KV / prefix-cache + M2/HIER-indexer interaction; qsplit bands query rows and keys the tiled route on `n_full` (D3); 1024-row microbatch changes band geometry and the `n > 16` / `_FENCE_MIN_ROWS` gates | `mlx_lm/models/deepseek_v41/indexer.py`; `_gates.py:43` | 2–3 |
| 5 | `EXO_DSV4_FENCE_ASYNC` / cache-gating re-derivation if prefill is to be added; must re-avoid the 2026-07-02 corruption class and the 2026-08-20 cross-stream ordering race (which this structure *reintroduces*) | `deepseek_v4.py:3811-3847`; race doc | 2–4 |
| 6 | Correctness / token-identity gate (a row-count change can flip a 1-ulp top-k tie — exactly what aborted qsplit round-1 at `identity_3`) | `docs/dsv41-indexer-qsplit-live-ab2-2026-10-12.md` §3 | 2–3 |
| 7 | Live A/B + quality battery (needs ≥1 boot; priced for completeness) | — | 1–2 |

### 1(c) Smallest validation experiment — NOT required (band <5 %)

Band <5 % ⇒ **PARK PERMANENTLY**; the smallest experiment is demoted to an *optional ≤1-day offline
confirmation* (described, not run): extend `bench/phase0b_collective_overlap_probe.py` to the **real**
chunk shape (2048-row MoE `all_sum` payload, attn `(B,n,nb)`) on the current build, plus the
**two-in-flight** arm (issue A's then B's `all_sum` back-to-back, measure serialize-or-not) on
`mlx.launch -n 2 --backend ring` loopback. Headline number = `overlap_achievable_ms / 7975 ms`.
Expected: overlap reachable only via `PREEVAL` (a drain) ⇒ confirms <5 %. Optional because the live
`EXO_PREFILL_CHUNK_OVERLAP` FLAT result and the neutral fence-spacing A/B already bound it.

---

## 2. Remaining REPLICATED (rank-indivisible) set — enumerate + price

Headline: **remaining replicated ≈ 9.7 % of per-rank wall @100 K `[DERIVED]`** (tp3-addendum §1).
Method: per-item share `S` of the per-rank wall; a **perfect, sync-free 2-way shard removes `S/2`** ⇒
`Δe2e = f/(1−f)`, `f = S/2` `[MODELED]`. A term can never yield more than its own share, and in
practice ≤~half of it minus sync. Nothing here is measured as an e2e win.

> **Reconciliation caveat (flagged).** The itemized set sums to **8.25 % `[DERIVED]`**; the headline is
> **9.7 %**. The **≈1.45 pp gap is unattributed** — the only unquantified term is **KV keys**
> ("not separately resolved", folded into `sdpa`/dense-qkv) `[BOUND]`. Treat 8.25 % as the priced set,
> 9.7 % as the ceiling.

| item | share @100 K | shardable? and how | modeled Δe2e if split | added sync/comm cost | effort (eng-d) | clears ≥5 %? |
|---|---:|---|---:|---|---:|---|
| **hyper-connections `hc_*`** | **5.2 %** `[MEASURED]` | **Only inside phase-2 SP** — cross-token mix (4 full-width residual copies `[b,s,4,d]`, `_mix_proj` + 20-sweep Sinkhorn collapse/expand); a rank band splits the residual stream, **not** the op; **no head axis** | **≈2.7 % `[MODELED]`** | piggybacks the SP residual exchange (net ≈ wash +~2 % wire); needs the collapsed 5120-wide residual reduce-scattered | ~0 alone; only as part of SP **12–25 d** | **No** (2.7 % < 3 % < 5 %) |
| **RMSNorm** (attn/ffn/final/q/kv) | 1.7 % `[MEASURED]` | row-local elementwise — band-splits perfectly **iff** the surrounding attention/MoE path is also band-split (phase-2 SP only) | **≈0.86 % `[MODELED]`** | none of its own (rides SP) | ~0 alone (SP) | **No** |
| **`attn.wq_a`** (shared q latent) | ~0.55 % `[DERIVED]` | **Not shardable.** No head axis (shared `[.,1280]` q latent consumed whole by every head + the indexer); sharding needs a *new* per-layer 1280-wide all_gather; blocked by `_block_bounds` uniform-`world` tiling | **≈0.28 % `[MODELED]`** | **adds** a per-layer latent all_gather (q 5.24 MB/layer) — **new cost > gain** | 2–4 d, **negative ROI** | **No** |
| **`attn.wkv`** (single MQA KV head) | ~0.22 % `[DERIVED]` | **Not shardable at all.** `head_dim=512` *is* the whole KV vector — one head, no head axis | **≈0.11 %** `[MODELED]` | — | n/a | **No** |
| **router gate `ffn.gate`** (score + topk) | 0.5 % `[MEASURED]` | row-local — band-splits iff the MoE expert path is band-split (SP) | **≈0.25 %** `[MODELED]` | none of its own (SP) | ~0 alone (SP) | **No** |
| **compressor (CSA/HCA)** | 0.08 % `[MEASURED]` | seq/row shardable in principle; already negligible | **≈0.04 %** `[MODELED]` | none of its own (SP) | ~0 (SP) | **No** |
| **KV keys** (rope + fp8/fp4 fake-quant + window/comp cache) | **not separately resolved** — folded into `sdpa` 13.6 / dense-qkv 4.3 `[BOUND]` | row-band only (each key row written once per rank); per-row **bandwidth-bound** at/near the DRAM-write floor ⇒ halving written bytes ≠ halving time | **not quantifiable** — plausibly **<1 %** `[BOUND]` | no new collective (cache rank-local) | ~0 (SP) | **No** (only unquantified ≥3 %-*hider*; no evidence) |
| **embeddings** `[129280,5120]` = 1.318 GB bf16 `[DERIVED]` | **0 compute** (row-indexed gather) | trivially partitionable (vocab rows) — but **memory, not time** | **0 %** | none | 0.5–1 d to shard, **0 perf** | **No** (memory-only) |
| **`dispatch / GPU-idle`** | ~9.6 % `[BOUND]` | not a replicated *compute* item — the fence/drain/all_sum-wait stall residue; **not addressable by rank-sharding** | n/a by sharding | n/a | n/a | **No** — belongs to §1 (comm-hiding) |

**Confirm or refute "thin/diminishing" ⇒ CONFIRM.**
1. Only two rows exceed 3 % raw: **HC 5.2 %** (the only replicated *compute* class) and
   **dispatch/idle ~9.6 %** (a stall bucket, not rank-splittable). Every other item is **<2 %** raw.
2. **HC's realistic post-sync e2e delta does not clear 3 %:** its perfect, sync-free 2-way shard is
   **≈2.7 % `[MODELED]`** — *below 3 % before any sync cost* — and it is shardable only as a
   **by-product of phase-2 SP** (a residual-layout change across all 40 layers, 12–25 eng-d). Task G
   §5 independently finds it "mostly irreducible per-layer bandwidth; not a high-value lever."
3. **Whole-set ceiling is at the gate, not over it:** perfectly/ costlessly sharding the entire
   itemized set (8.25 %) ⇒ **≈4.3 % `[MODELED]`**; even the full 9.7 % headline ⇒ **≈5.1 %** — the
   unattainable max, with zero realistic path. *Achievable* ≈ **4.3 % < 5 %**.
   ⇒ **No standalone engineering slot;** the set is capturable only folded into phase-2 SP (12–25 d),
   itself gated on a fresh deep attribution still showing the replicated set ≳30 % — **currently
   closed** (post-split it is 9.7 %).

---

## 3. Indexer absolute-cost follow-up — scoping (PRICE ONLY; not started)

**Knowns.** The indexer is a real `O(context)` class and, pre-split, the single largest deep bucket:
**31.1 % @100 K** (Task G §1, reproduces the pack's 2241.4 / 7207.6 ms exactly) and 13.5 % @16 K at the
deepest chunk `[MEASURED]`; the 3.6 % figure is a *pre-HIER, 8-chunk-averaged* frame (frame+path
mismatch, not a contradiction). In-repo depth attribution: **14.6 % → 42.9 % over a 0→350 K feed,
~15 % @100 K** `[MEASURED]` (`PH:10487`, `PH:10526-27`). The **query-split already shipped**
(+10.5 / +11.9 % @50 / 105 K). What remains is the indexer's *absolute* cost. Task G names two
directions, then STOPs:

1. **nb-based HIER floor.** The HIER gate is a **row-count** test, not a context test:
   `if _HIER and n > _FENCE_MIN_ROWS` (`indexer.py:728`, current tree; `_FENCE_MIN_ROWS = 16`,
   `_gates.py:43`). Prefill rows (2048) always exceed 16, so **HIER is ON for every prefill
   regardless of context** — a 16 K prefill pays HIER's coarse+exact machinery for no memory
   benefit. HIER's benefit is nb-shaped: laptop microbench **hier/tiled = 1.71–2.05× @ nb=16384**
   (pure overhead) but **0.38× = 2.6× faster @ nb=100352** `[MEASURED/laptop]`; in-repo A/B agrees
   (HIER −0.9 % neutral at fresh 100 K; +15.2 % on the 100→200 K delta; +26.7 % on 160→500 K)
   `[MEASURED]`.
2. **Deep-context top-k/scan.** The score GEMM itself is already ~99 % of ceiling (OPT-6 fold), so
   the residual lever is the top-k/scan, not the GEMM `[DERIVED]`.

**What a bounded investigation would TEST.** (i) *HIER floor:* locate the nb crossover and confirm a
**`nb`-based floor** (engage HIER only when nb exceeds the crossover) removes the small-nb overhead
while keeping the depth win — the *ratio* is the transferable quantity (laptop absolute ms will not
transfer to the M4-Max pair). (ii) *Top-k/scan:* a **per-chunk, eval-fenced attribution under HIER-ON
at 16 K** (Task G's "Recommended follow-up", explicitly **not run**) to size the non-GEMM indexer
residual *before* any build — sizing only.

**Price — bounded, mostly offline-analyzable, ≈ 3–6 eng-days (centre ≈ 4) `[MODELED]`:**

| direction | offline vs boot | eng-days | Δe2e claim |
|---|---|---|---|
| (i) nb-based HIER floor | ~1–2 d offline (extend the hi er/tiled microbench to find the crossover; predicate edit ≈1 line) **+ 1 quiet-cluster A/B boot** to confirm laptop→node ratio | **1–3** | unquantified; bounded by the small-nb HIER overhead on the shallow indexer share (~7 % avg @16 K / 13.5 % deepest chunk) — **upper-bounded well under the 5 % e2e gate; no ≥5 % claim supportable offline** `[BOUND]` |
| (ii) top-k/scan | ~2–4 d; offline **if** a covering trace exists, else 1 attributed boot | **2–4** | **not quantifiable** until the attribution runs — a sizing exercise, not a priced lever `[BOUND]` |

**Risk / token-identity.** Both directions **change which path computes the indexer**, so a **1-ulp
top-k tie flip is live** — the exact failure that aborted qsplit round-1 at `identity_3`, cleared only
by the fused-GEMM row-count fix. Any build must run the **5-prompt greedy token-identity gate** plus
the 3-gate battery. **Bottom line: priced, not started.** On current evidence (i) is a small,
offline-first item plausibly worth a few % of *small-nb* prefill wall (still sub-gate); (ii) is
un-sized until the missing per-chunk attribution runs. Neither is a *confirmed* ≥5 % lever on its own.

---

## 4. Un-run candidate ABs — status

Two candidates from the 2026-10-05 GPU-utilization investigation were once listed and never priced.
Both land on the closed list.

| # | candidate | status | decisive line | live knob today? | one window? |
|---|---|---|---|---|---|
| **A** | `EXO_DSV4_FENCE_EVERY_N_LAYERS` sweep (2/4/8) at depth | **CLOSED-ON-EVIDENCE** | `PH:8841` — *"FENCE_EVERY_N_LAYERS cadence — CLOSE ON EVIDENCE … single-digit us per fence op over ~43 layers is <0.1 t/s."* | **No — vestigial** | **No** |
| **B** | `DSV41_ASYNC_EVAL=0` — the "bubble A/B" | **NEVER-RAN (deprioritized)** | `PH:10553` — *"The queued 'bubble A/B' (DSV41_ASYNC_EVAL=0 class) is deprioritized: if 15 fewer blocking drains per chunk move nothing, the eval-boundary class is closed at this depth."* | yes, real — **default 1 (ON)** (`deepseek_v41/model.py:45`) | **No** |

**A — the dead-vs-live contradiction, resolved against the current tree.** `PH:462` and `PH:8661`
("DEAD KNOB … zero read sites tree-wide") say dead; `PH:8804` (a Round-5 correction) says that is
wrong and a live reader exists. Tree-wide grep of the live tree settles it:
`EXO_DSV4_FENCE_EVERY_N_LAYERS` is read into `self._fence_every_n` at **`deepseek_v4.py:3595-3596`**,
and that attribute is then **never read** — the only other occurrence is the provenance comment at
`:3784` ("OPT-7 … tested and REVERTED — it made B=2 prefill 23 % slower"). So `PH:8661` is **correct in
substance**: the knob **gates no eval today — vestigial.** The cadence lever is closed regardless
(`PH:8841`). The **live** prefill fence is a *different* knob — `EXO_PREFILL_FENCE_EVERY` (default 2;
`src/exo/worker/engines/mlx/dsv41/session.py:137`) — and **its own A/B already ran: `=8` vs `=2`,
NEUTRAL ±0.3 % at 100–200 K** (`PH:10537-10547`), fence stays at 2. **Not worth a window:** on the
named knob it is a no-op read, and the live analog is already measured neutral at depth.

**B — price of a clean re-test (if it were worth running).** The prefill-side arm exists and is
symmetric across ranks: `EXO_PREFILL_ASYNC_EVAL=1` (default 0) swaps `mx.eval`→`mx.async_eval` on the
cache state (`generate.py:504-513`), so chunk N+1's graph build overlaps chunk N's GPU execution —
the comment states it targets the **7 % GPU-idle gap** measured by `EXO_PREFILL_GPU_TIME`. Caveat: the
async branch is taken **only when `EXO_PREFILL_EVAL_INTERVAL > 1`** (`generate.py:510-514`), so on
defaults the knob is **inert even when =1**. A clean run would need **(a) arm** =
`EXO_PREFILL_ASYNC_EVAL=1` **with `EXO_PREFILL_EVAL_INTERVAL>1`**; control = shipped defaults;
**(b) gate** = **token-identity**, because `async_eval` order is matched *positionally* across ranks
— a rank-divergent schedule is **deterministic silent corruption** with no crash/NaN
(`PH:359-363`) — plus the needle battery; **(c) cost** ≈ **2 boots** (1 treatment + 1 rollback
reserve), ~30–60 min quiet-cluster wall `[BOUND]`. **Not worth spending it:** the eval-boundary class
it targets is already **closed at depth** (`PH:10553-54`), and the sibling decode-side async fence is
already live at **+1.04 %**. **Recommend do-not-run.**

---

## 5. Verdict + recommended order

**Applied bar: ≥5 % e2e prefill.** Nothing re-priced in this doc clears it.

**Recommended order**
1. **Indexer absolute-cost bounded investigation (§3)** — the only remaining direction with a real
   ceiling (the indexer is 31.1 % @100 K / 42.9–44.2 % deep), same family as the shipped split,
   priced **≈3–6 eng-days, offline-first**. Do (i) the **nb-based HIER floor** first (1–3 d: offline
   crossover hunt + a 1-line predicate edit + 1 quiet-cluster A/B boot), then (ii) the per-chunk
   HIER-ON attribution to *size* the top-k/scan residual before building anything. Token-identity
   gate mandatory.
2. *If the owner wants a measured win now:* there is **none open.** The only ≥5 %-class **measured**
   prefill lever on the bench is **native-format experts** (measure-pack Task A: +22 % on the expert
   block ⇒ ~7 % e2e per chunk at a ~33 % block share), but it is **owner-CLOSED** (declined
   2026-10-11): no arm fits 2 nodes (q4g64 = 155.4 GB = 113 % of 128 GiB; q3g128 fits but is lossier,
   cosine 0.9423). Sole reopen = a **third node (TP3)** + the DSpark-taps reclaim + a fresh quality
   battery — itself gated on the TP3 readiness package (≈15–32 eng-days).
3. **Phase-2 SP (sequence/band-parallel residual)** — only if a fresh deep attribution re-opens the
   replicated set ≳30 % of wall; today it is 9.7 %, so the gate is closed.

**CLOSED — do NOT re-propose**
- **Dual-microbatch comm-hiding** (this doc: centre +1.5 %, <5 % at every depth) — parked permanently.
- **`EXO_DSV4_FENCE_EVERY_N_LAYERS` 2/4/8 sweep** — vestigial knob (no consumer) + cadence closed.
- **`EXO_PREFILL_FENCE_EVERY` spacing** — 8 vs 2 neutral at 100–200 K; fence stays at 2.
- **`DSV41_ASYNC_EVAL` / `EXO_PREFILL_ASYNC_EVAL` bubble A/B** — never run; eval-boundary class closed
  at depth; prefill-side knob inert without `EXO_PREFILL_EVAL_INTERVAL>1`.
- **Comm/compute overlap for `moe.all_sum`** — already implemented and live as `EXO_DSV4_FENCE_ASYNC=1`.
- **`EXO_PREFILL_CHUNK_OVERLAP`** — shipped once, measured FLAT live, closed.
- **Intra-GPU stream overlap on MLX/Metal** — "stream overlap: neutral — MLX serializes streams."
- **Async I/O overlap with compute** — measured dead (2026-07/08).
- **MoE kernel family** — parked, ≤3 %. **sdpa micro-levers** — exhausted. **chunk-size** — closed.
- **c=2 concurrency** — owner-dropped.
- **Standalone sharding of any remaining replicated item** (§2) — best (HC) ≈2.7 %; whole set ≈4.3 %.
- **Embeddings sharding** — memory-only, 0 perf. **lm_head vocab sharding** — below the floor.
- **Native-format experts (requant)** — owner-declined 2026-10-11 (no arm fits 2 nodes; the only
  fitting arm q3g128 is lossy). Reopen only via TP3 + DSpark-taps reclaim + a fresh quality battery.

---

## Provenance / sources

- **OFFLINE: 0 boots, 0 config changes, 0 node edits, no cluster contact.**
- Indexer split (live): `docs/dsv41-indexer-qsplit-live-ab2-2026-10-12.md`, `origin/main @ 06d134c95`.
- Replicated-vs-splittable partition: `docs/dsv41-tp3-scaling-readiness-addendum-2026-10-12.md`
  (`p20/tp3-scaling-readiness @ 5868aa4a4`).
- Per-op-class attribution @100 K: `docs/dsv41-prefill-measure-pack-2026-10-11.md`
  (`p20/prefill-measure-pack @ 302421ebe`).
- Indexer-share reconciliation + HC classification (Task G):
  `dsv41-pmp-tasb-reconciliation-2026-10-11.md` (`p20/pmp-tasb-reconciliation @ d323964c9`).
- Replicated-set shardability: `dsv41-attention-sharding-scope-2026-10-11.md`
  (`p20/attention-sharding-scope @ 0adf3f0d`).
- Overlap evidence: `comm-compute-overlap-already-exists-2026-08-21.md`;
  `phase0b-collective-overlap-gate-2026-08-20.md`; `prefill-chunk-overlap-live-test-2026-08-20.md`,
  `prefill-chunk-overlap-race-fix-2026-08-20.md`; `prefill-fence-gate-audit-2026-08-22.md`.
- Timeline: `docs/PERFORMANCE_HISTORY.md` (`PH:` line cites above) — fence dead-knob `:462`,`:8661`,
  `:8804`,`:8841`; async-fence `:359-363`,`:415-421`,`:598-612`; fence-spacing A/B `:10537-10558`;
  M2/HIER attribution `:10448-10560`; dual-microbatch unpriced `:11419`.
- Code (live tree): `mlx-lm/mlx_lm/models/deepseek_v4.py:3595-3596,3784,3811-3817`;
  `mlx_lm/models/deepseek_v41/{model.py:45,indexer.py:728,_gates.py:43}`;
  `mlx_lm/generate.py:504-514`; `src/exo/worker/engines/mlx/dsv41/session.py:137`;
  `src/exo/worker/engines/mlx/generator/generate.py:1762-1892`.
