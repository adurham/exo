# PERF CAMPAIGN 2 — ROUND 1: where does the 1.75-2.16x decode headroom live?

**Date:** 2026-09-03 · **Scope:** I6, I1-prereq, I3, I1, I4, I2 (audit half)
**Cluster:** 2x Mac Studio M4 Max, TB5 RDMA, TP=2, DeepSeek-V4-Flash-0731, 43 layers

---

## THE ANSWER

**The 2x does not live in the collectives, and it does not live in the MoE expert kernel.
Both were measured this round and both are exonerated. It lives in local GPU
drain + dispatch overhead around the per-layer sync points — the collective is merely
where the sync lands, not where the time goes.**

The numbers: the MoE expert kernel achieves **83.9% of the 546 GB/s ceiling at the M=4
verify shape** (458.0 GB/s, 350.3us/call, chained-graph method) — above the ≥80% band, so
kernels are fine. Cross-rank communication is **2.6% of the per-layer budget by median**
(jaccl transport 36.1/36.0us median per rank against a ~1400us per-layer all_sum drain;
RDMA poll loop 5.00us = 0.36%). That leaves **~95% of the per-layer collective budget as
local GPU drain and dispatch** — the local rank at the fence is waiting on *its own GPU*
to finish the preceding MoE, not on the peer. Expert-read duplication is real but small:
the corrected shared-vs-distinct ratio is **1.42x**, only 14% of the way from no-dedup
toward full 4x dedup, worth roughly 4-8ms of the 56ms verify. So the residual gap is an
**overlap/pipelining problem**, not a comm problem and not a kernel problem — which is a
different lever than any of the three this round set out to test.

---

## RECONCILIATION WITH PRIOR RECORD

This section exists because a supervisor correction caught that two of this round's
questions were **already answered and on record**. Both of this round's original
measurements reproduced a **known, already-retracted artifact**. Recording it explicitly so
it is not re-litigated a fourth time.

### I3 — this round initially reproduced the retracted serial-sync artifact

| | Value | Verdict |
|---|---|---|
| Prior record (2026-08-22, **retracted**) | 27.7% of peak | artifact: `mx.eval` inside the per-iteration loop, ~172us/call host overhead charged to the kernel |
| Prior record (2026-08-22, **corrected**) | 116-117us/call, ~404 GB/s, **74% of peak** | chained-graph, one eval per ~300 calls, rotated indices. Verdict on record: *"NO kernel optimization lever exists here."* |
| This round, run 1 | 284.9 GB/s, 52.2% | **same artifact** (wrong `affine` mode AND per-iteration eval) |
| This round, run 2 | 287.7 GB/s, 52.7% | **same artifact** — fixed the quantization mode to mxfp4 but kept `mx.eval` in the loop (`i3_microbench_rerun.py:67-70`) |
| This round, run 3 (chained) | **350.3us/call, 458.0 GB/s, 83.9% of peak** at M=4 | **agrees with the record's method**; timing at M=1 (114.4us/call) matches the record's 116-117us/call at noise level |

**Which is the artifact:** runs 1 and 2. Confirmed directly — the chained harness ran its
own serial-sync control and reproduced 53.1%, within 1% of run 2's number, from the same
machine in the same session. The artifact is fully accounted for.

**Consequence — a verdict is WITHDRAWN.** This round's earlier
`<60% → MLX KERNEL WORK IS FUNDED` firing was an artifact and does not stand. The
corrected number fires `≥80% → kernels are fine`. **Do not fund MLX kernel work off this
round.** This agrees with the record's standing "no kernel lever exists here."

One residual disagreement, stated rather than smoothed: this round computes 350.7 GB/s at
M=1 where the record has ~404 GB/s. Traced to **byte accounting** (40.1 MB/token here vs
47.19 MB/token on record, ~13%), not to kernel speed — the us/call figures agree. The
byte-accounting discrepancy is unresolved and is a loose end.

### I1 — the "collective share" measurement was already on record; only the re-scope was new

Already established, not re-derived:
- Per-layer all_sum drain **~1.4ms x 43 layers**, uniform (the apparent 35% spread was
  iter-0 cold-compile-cache contamination).
- jaccl/RDMA is **not** the lever: poll wall 1.9% of verify (mean 8.15us, median 5.00us);
  ~98% of the cost attributable to the `mx.eval` fence itself.
- The chained-collective peer-CQE-tail hypothesis was **falsified** — `FENCE=1` made verify
  **10.6% worse** (63.16ms vs 57.10ms), the opposite of its prediction.
- `moe.all_sum` at 61-64% of wall is known, with three failed fusion/reordering attempts.

**What this round added (the genuinely open question):** since ~98% of the cost is the
`mx.eval` *wait*, what is the rank waiting *on* — the peer, or its own GPU? The record did
not distinguish these, and they have different fixes. **Answer: the local GPU.** See below.
The relaunch for the original "measure collective share" framing was **correctly skipped**
as a repeat.

---

## I6 — expert reads: per verify or per row?

**Verdict: per-(row,expert) pair — the kernel does NO dedup. But it is NOT the 2x.**

- `gather_qmm` performs no grouping/sorting of rows by expert; each (row, expert) pair
  dispatches its own weight-tile load. Bytes multiplier vs M=1: **2.37x measured**
  (24 pairs / ~6 distinct at realistic routing), not the clean 4x the hypothesis assumed.
- **Live measurement confirms it** (this is the decisive number, and it survived the
  artifact correction): 4 rows routed to the *same* 6 experts vs 24 *distinct* experts —
  chained-graph ratio **1.42x** (the earlier serial-sync run reported 1.21x; removing the
  fixed per-call overhead from both arms raises the ratio, as predicted).
  A dedup-capable cache would approach 4x. It does not.
- Consistent with the record's ablation finding that efficiency is **flat in B** (B=1..32) —
  a deduplicating kernel would show super-linear gains with B.
- **Size: ~4-8ms of the 56ms verify.** Real, worth noting, not the headline.

Two footguns found in passing:
- `EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=1` would take collectives 43 → 172.
- The sorted/run-length `gather_qmv_rhs` kernel — built precisely to stream each expert
  once — is **unreachable at decode**: `do_sort` needs `indices.size>=64` (verify has 24)
  and its own `B/E>=2` gate fails at B=24/E=256. Lowering the sort threshold alone will not
  reach it.

---

## Collective count per forward (from code, independently verified)

**43, not 86. The `start_cluster.sh` comment is wrong.** Confirmed live (`EXO_DSV4_ATTN_ALLSUM=0`
on both nodes) and by two independent code reads.

- The apparent second per-layer all_sum is `sum_gradients` (`deepseek_v4.py:2959`), whose
  **forward is identity** — `return x` at `mlx/python/mlx/nn/layers/distributed.py:21`, with
  the collective only in the `@f.vjp` backward at `:25`. Zero at inference. Almost certainly
  the source of the doubling in the comment.
- **Attention is REPLICATED, not head-sharded** — confirmed three ways. Production uses
  `DeepseekV4ShardingStrategy` and never `Model.shard`, so the `n_heads //= N` code at
  `deepseek_v4.py:7545` is **live-looking but dead**. A casual grep concludes the opposite.
  **I1b is therefore moot** — there is no post-attention all_sum to eliminate.
- Payload: **(1,4,4096) bf16 = 32 KiB** per call, 43 x 32 KiB = 1.375 MiB/forward.
  Latency-bound, not bandwidth-bound. Coord collectives are on a separate process group and
  fire per *cycle*, not per forward.

A near-miss worth recording: `ShardedToAllLinear.__call__` *does* have a forward `all_sum`
(`distributed.py:333`, `:585`), but DSv4 uses `shard_inplace` (parameter-dict rewrite), not
`shard_linear` — so it never fires. That check could have overturned the count.

---

## I3 — achieved bandwidth / TFLOPS (corrected, chained-graph)

| Measurement | Value | % of peak |
|---|---|---|
| MoE expert kernel, **M=4 (verify shape)** | 350.3 us/call, **458.0 GB/s** | **83.9%** of 546 GB/s |
| MoE expert kernel, M=1 (draft shape) | 114.4 us/call, 350.7 GB/s | 64.2% |
| Prefill GEMM M=2048 (dense bf16) | 14.23 TFLOPS | ~79% of ~18 TFLOPS |
| Serial-sync control, M=4 (artifact reproduction) | 552.9 us/call | 53.1% — matches the withdrawn number |

**Deployed quantization** (verified, and it corrects a campaign-wide assumption): routed
experts are **mxfp4, group_size=32, bits=4** — *not* the 6-bit the campaign's byte accounting
assumed, and not the `fp8` in the checkpoint's top-level `quantization_config` (that is the
upstream storage format, not the runtime quantization `make_quantization_config()` applies).
Shared experts + attention: mxfp8 b8 g32. **lm_head: mxfp8 b8 g32** (an earlier claim of
"affine g64" was wrong). MoE router: unquantized bf16.

**BAND FIRED: ≥80% → kernels are fine; the gap is overhead.** Applied verbatim to the
corrected M=4 number. Note M=1 alone lands in the 60-80% band, so draft-step traffic is not
fully exonerated — flagged rather than glossed.

**Kernel-sum x43 vs 56ms verify: NOT REPORTED as a clean number.** The per-op forced-eval
method produced 123.71ms against a 56ms bracket — i.e. it *exceeds* the thing it is meant to
decompose, because forcing ~473 eval/sync boundaries destroys the graph fusion production
gets from one eval per verify. Reporting this as "overhead" would be fabrication. A
trustworthy version needs per-layer (not per-op) eval granularity. Open.

---

## I1 — what the fence is actually waiting on

**Verdict: (b) LOCAL-COMPUTE-WAIT. The lever is overlap/pipelining, NOT comm.**

Against the ~1400us per-layer budget:

| Component | Value | Share |
|---|---|---|
| jaccl transport (peer-dependent), median | 36.1 / 36.0 us per rank (n=45,666/rank) | **2.6%** |
| jaccl transport, mean | 66.3 / 58.9 us | 4.7% |
| RDMA poll loop, median | 5.00 us | 0.36% |
| **Residual: local GPU drain + dispatch** | — | **~95%** |

Three structural arguments, not just the arithmetic:
1. **No-peer control (strongest):** the same 2.66x stream-boundary penalty reproduces with a
   plain non-collective CPU-stream op — *no peer exists in that experiment*.
2. **Falsified prediction with a signed fingerprint:** `FENCE=1` made verify *worse* (+10.6%),
   opposite of the peer-tail prediction; per-layer cost alternates deterministically with each
   layer's own local `compress_ratio` (1.50 vs 1.28ms) — peer skew has no reason to track that.
3. **Documented misattribution mechanism:** `mx.synchronize()` drains MLX's entire pending
   lazy graph, billing upstream *local* compute to the collective.

Cross-rank waits are **symmetric** (medians differ 0.3%; p95/p99 asymmetries point in
*opposite* directions) — there is no chronically-late rank.

**Band:** the pre-registered share bands (≥30% fund collective reduction / 10-30% fund jaccl
/ <10% closed) are answered by the transport figure of **2.6% median → CLOSED**. Consistent
with the record's prior falsification of the jaccl lever. **I1c (jaccl latency work) is
CLOSED. I1b is moot** (attention already replicated).

**Caveat, stated wherever the share is quoted:** production runs `EXO_DSV4_FENCE_ASYNC=1`
(non-blocking `async_eval`); all bracketed sources force blocking evals, so share figures are
**upper bounds** and no profiled forward time may be quoted as a decode-latency number. The
verdict rests on a ~39x separation (36us vs 1400us), which survives that caveat comfortably.

---

## I4 — Fix B re-test at ≥3 chunks

**Verdict: the round-4 blocker STANDS. Fix B is NOT re-opened.** The band's own words:
"diverging variants ~0 at ≥3 chunks → blocker STANDS."

| Request | Prompt tokens | Divergence | HTTP `cached_tokens` |
|---|---|---|---|
| A baseline | 7524 | — | 0 |
| B exact repeat | 7524 | — | **7522** |
| C/D/E diverge past 3rd boundary | 8292 | ~6491 | **0** |
| F control, early divergence | 12704 | ~1503 | 0 |

The executing agent initially called this **AMBIGUOUS** by substituting a runner-log
`shared_prefix` figure (6491) for `cached_tokens`. **An independent audit refuted that** and I
am applying the stricter reading:

- `shared_prefix` is emitted by `add_kv_cache` — the **insert** path (`cache.py:823-825`).
  It measures trie overlap of a session being *stored*, not a hit.
- Three confirmations no hit occurred: `add_kv_cache` (not `update_kv_cache`) ran for all
  four; `cache.py:1725-1726` states DSv4 hits its path on *every* partial hit and it did not;
  and request C took **26.4s / 3.18 ms-per-token vs the cold baseline's 2.68** — *slower than
  cold*, which a real 78% cache hit cannot produce.
- **`cached_tokens=0` is not an observability gap** — it is wired straight to
  `prefix_hit_length`, the same variable that decides whether prefill is skipped.

**Separately, and clearly labelled as analysis rather than band:** round 4's 381-token *test*
**was** structurally invalid (zero chunk boundaries, partial hits impossible). Its
*conclusion* is nonetheless independently confirmed by this round's better experiment. The
premise was wrong; the answer was right.

**Root cause of the no-partial-hit behavior identified:** `cache.py:496` treats trimmable
`CacheList`s as non-sliceable (snapshot required) while `cache.py:356` refuses to snapshot
them (none available). That contradiction — not chunk granularity — is why partial hits do
not serve. **No Fix B design is written, per the band.** If Fix B is ever revisited, that
contradiction is the thing to fix first.

---

## I2 — c=2 tax audit (no changes made)

10 rows audited. Full table in `I2-C2-TAX-AUDIT.md`. Headline:

| Knob | Current | c=1-optimal | Documented c=1 cost | Read at |
|---|---|---|---|---|
| `EXO_DSV4_FENCE_EVERY_N_LAYERS` | 4 | 8 | ~0.7 t/s | process start |
| `MLX_STEEL_BATCH_INVARIANT` | 1 | 0 | ~5% decode | process start (C++ static init) |
| `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` | 200 | 0 | **200ms TTFT, every request** | process start (module import) |
| `MLX_GEMV_BATCH_INVARIANT` | 1 | 0 | none quoted | process start |
| `EXO_DSV4_MTP_C2_MAX_CTX` | — | — | none quoted | **runtime** (per decode cycle) |
| `EXO_DSV4_FUSED_MOE` / `COMPILE_FFN` / `COMPILE_LAYER` | 0 | — | ~3-4% claimed | **dormant — see below** |

**Total documented tax** (the comments' own claims, not measurements): **~0.7 t/s + ~5%
decode, and 200ms TTFT on every request.** Reported separately, not summed — different units
and baselines.

**Discrepancies found:**
1. **`FUSED_MOE`/`COMPILE_FFN`/`COMPILE_LAYER` are dead knobs.** The wiring was *removed from
   source* on 2026-06-18 (`auto_parallel.py:110-118`), not merely defaulted off — a repo-wide
   grep finds zero live reads. Setting them to 1 today does nothing. `start_cluster.sh`'s
   comment "set any of these =1 to re-enable" is **stale**. Their claimed ~3-4% is excluded
   from the total.
2. The `MLX_STEEL_BATCH_INVARIANT=0` ⟺ `EXO_DSV4_VERIFY_ROWSEQ_VEC=0` interaction is
   **comment-only, not code-enforced** — no cross-check exists between the two.
3. `EXO_DSV4_MTP_C2_MAX_CTX`'s comment is stale; the gate it describes was removed 2026-06-24.
4. `EXO_MAX_ACTIVE_TASKS` and `MLX_GEMV_BATCH_INVARIANT` quote no c=1 cost;
   `EXO_DSV4_SPEC_CACHE_ROLLBACK_C2` is c=2-vs-c=2 with no c=1 cost — all three arguably do
   not belong in this audit's inclusion criterion.

**No knob was changed this round**, per instructions.

---

## Process findings worth keeping

- **A real bug in this round's own instrumentation patch was caught before it produced data.**
  Re-indenting the Phase-H fence block pulled `return y` from indent 12 to 16, moving it
  inside `if self.sharding_group is not None:`. On the DSpark draft head (`sharding_group is
  None`) the MoE `__call__` fell through and returned `None`, crashing the runner. It was
  invisible to `git diff -w`, `ast.parse`, and a single-rank self-test, because the broken
  path is never exercised single-rank. **Any re-indentation patch must be checked on both the
  sharded and unsharded paths.**
- **node2's runner was found DEAD** mid-round (no screen session, log frozen 9 min, node1
  `/state` showing `"instances":{}`). The restore relaunch fixed it. Worth noting the cluster
  can be half-down while the API still answers.
- **`i1_workload.py` has a live bug** (left unfixed, another agent's deliverable): it reads
  only `delta.content`, but DSv4 emits `reasoning_content` first, so `ttft_s=None` →
  `TypeError` at line 204. **It would read as a serving failure on a perfectly healthy cluster.**
- The `exo-cluster-operations` skill references `scripts/all_sum_latency_probe.py`, which
  **does not exist** — stale reference, worth patching.
- The `exo/mlx` vs `repos/mlx` split is a live trap: the deployed mlx is built from
  **`~/repos/exo/mlx` @ `e40a416b2`**, while this MacBook's venv uses `~/repos/mlx` @
  `ac73d0c9e`. Local mlx behavioral benches silently test the wrong build.

---

## Cluster health verification (post-round, verified by the PM directly)

```
API:            http=200  (curl http://192.168.86.201:52415/v1/models)
node1 runners:  4 procs, screen 31240.exorun (Detached)
node2 runners:  4 procs, screen 35647.exorun (Detached)
completion:     finish_reason=stop, content="The capital of France is Paris."
                usage: prompt 17 / completion 50 (reasoning 41)
profiler symbol in installed site-packages, both Studios:  0 (absent)
EXO_DSV4_COLL_PROFILE in runner env, both nodes:           absent
git tracked-diff (start_cluster.sh, dsv4_mtp.py, deepseek_v4.py): CLEAN
```

**Cluster is healthy on shipped production config. All instrumentation reverted.**
(An earlier probe with `max_tokens=40` returned empty `content` — that is the known DSv4
`reasoning_content` behavior at low token budgets, not a fault; re-probed at 300 and it
returns correctly.)

---

## Highest-EV next round

**Attack the local GPU drain / dispatch overhead around the per-layer sync points — the
overlap/pipelining lever.** That is where the residual gap actually is, and it is the one
lever this round newly localized rather than closed. Concretely: production already runs
`EXO_DSV4_FENCE_ASYNC=1`, so the question is why ~95% of a ~1400us per-layer budget is still
local drain when the transport is 36us. Note the record contains **three failed**
fusion/reordering attempts at this — so the next round must first read those three and state
what it is doing *differently*, or it will be a fourth repeat.

**Do not fund:** MLX kernel work (I3 closed at 83.9%), jaccl/comm work (I1c closed at 2.6%),
I1b (moot — attention already replicated), Fix B (blocker stands).

**Cheap and unblocked:** the c=2 tax is a genuine ~5% + 0.7 t/s + 200ms TTFT that costs
nothing algorithmically to reclaim at c=1 — but 3 of its 10 knobs are stale/dead and the
audit's own inclusion criterion needs tightening before any A/B.
