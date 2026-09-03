# PERF CAMPAIGN 2 — ROUND 2: the MoE decode stall, attacked at the MLX layer

**Date:** 2026-09-03
**Scope:** Phase A (mlx source archaeology). Phase B = **structural close**, by design.
**Cluster:** 2x Mac Studio M4 Max, TB5 RDMA, TP=2, DeepSeek-V4-Flash-0731, 43 layers
**mlx** `~/repos/exo/mlx` @ `e40a416b2` · **mlx-lm** `~/repos/exo/mlx-lm` @ `37260bbd6`
**Companion:** `MECHANISM.md` (the five answers with cites) — this file is the verdict,
the reconciliation, and the disposition.

---

## THE ANSWER

**The per-layer stall is a TRUE GPU→CPU DATA DEPENDENCY created by the fact that MLX has
no GPU collective at all. It is not an over-broad fence, and it is not command-buffer
churn. The fix that the "true dependency" verdict prescribes — cross-layer pipelining —
is independently blocked by DSv4's hyper-connection structure. This round therefore
closes the stall structurally, with the exact reason, and scopes the one genuinely
untested root-cause fix instead of attempting a fourth rearrangement.**

Two source facts carry the whole result, both re-verified by the PM at the primary
source:

- `mlx/distributed/jaccl/jaccl.cpp:88` — the collective stream is
  `new_stream(Device::cpu)`. A **CPU** stream.
- `mlx/backend/metal/distributed.cpp:17-19` — `AllReduce::eval_gpu` **unconditionally
  throws**: *"has no GPU implementation."*

So every one of the 43 per-layer `all_sum`s is a CPU worker thread doing an RDMA
transfer, which must read `y` out of host-visible memory. A CPU thread therefore *must*
observe GPU completion before the transfer can start. MLX implements that as a
whole-payload `input_coherent` kernel plus a device-wide `memoryBarrier`
(`fence.cpp:128-158`) and a **host busy-spin** (`fence.cpp:58-77`). That is the
~1400 µs/layer "local drain" round 1 measured — and it is a dependency, not a missed
overlap.

---

## WHAT IS DIFFERENT FROM THE THREE FAILURES

The 2026-06-26 record documents three attempts, all of which **rearranged work around a
fence whose semantics they did not control**. This round did not rearrange anything. It
went into the mlx fork and established the semantics. That produced three findings none
of the three attempts could have reached by reading exo/mlx-lm:

1. **There is no GPU collective to overlap with.** Every prior attempt assumed the
   "overlap primitive exists" (the 06-26 doc says so at its lines 88-89). It does not.
   `eval_gpu` throws. The comm stream is a CPU stream. The textbook
   collective-overlap model was never applicable.
2. **The event primitive the 06-26 doc wished for already exists — and cannot help.**
   `Event` wraps `MTLSharedEvent` with a genuine GPU-side `encodeWait`, and MLX
   *already inserts the cross-stream edge automatically* (`transforms.cpp:159-168`,
   `:263-277`). But the GPU-side wait branch is **structurally unreachable** for a
   GPU→CPU edge. So the hypothesis behind verdict (b) — "they used `mx.eval` where an
   event was needed" — is false at the source level.
3. **Attempt 3 did not break bit-equivalence. It broke the algebra.** See the record
   correction below. This matters because the "fence position is load-bearing for
   bit-equiv" conclusion has been deterring work in this neighborhood for two months, and
   it is not what happened.

**No exo-level fence rearrangement is proposed. No recorded attempt was re-run.**

---

## THE (a)/(b)/(c) VERDICT

| Verdict | Status |
|---|---|
| **(a) true dependency, pipelinable only across layers** | **CONFIRMED — and its fix is independently BLOCKED** |
| (b) over-broad fence a stream-scoped event would fix | **RULED OUT** — the event exists, MLX already inserts it, and it cannot span a GPU→CPU handoff |
| (c) command-buffer churn at stream boundaries | **RULED OUT as dominant** — real, but ≤~7% at decode payload |

### Why (a)'s own fix is blocked — the exact reason

Cross-layer pipelining (layer N's `all_sum` overlapping layer N+1's attention) requires
layer N+1 to not depend on layer N's collective output. **It does depend on it, totally.**

`hyper_connection.py:486-489` (`_hc_expand_op`) mixes the MoE output into **all four**
hyper-connection streams via a broadcast plus a dense `matmul` against the residual.
There is **no clean residual stream that bypasses the FFN** — DSv4's structure is
*stronger* than the standard transformer framing, not weaker. Layer N+1's first op is an
RMSNorm (full-axis reduction), so no partial ordering survives either. The model loop
(`deepseek_v4.py:7046`) is a straight-line rebind with no lookahead.

**And there is no independent GPU work anywhere to overlap with instead** — expert
dequantization has no separable node (it happens inside `gather_qmm`), the MTP/DSpark
draft head consumes hidden states from *after* all 43 layers, and the attention
indexer/compressor consume the current layer's own normed hidden.

This is the "structural, here is the exact reason" close the round-2 brief explicitly
authorizes as being worth more than a fourth failed attempt.

---

## RECONCILIATION WITH PRIOR RECORD

### Correction 1 — the fence is NOT load-bearing for cross-rank bit-equivalence

`docs/dsv4-decode-stall-2026-06-26.md:90-94` and the production comment at
`deepseek_v4.py:3077-3092` both assert the `mx.eval(y)` fence is required for cross-rank
bit-equivalence, citing attempt 3's near-zero output as proof. **The evidence supports a
simpler explanation: attempt 3 had an algebra bug.**

- `auto_parallel.py:1128` shards `shared_experts.down_proj` **sharded-to-all**.
- `shard_inplace` (mlx `nn/layers/distributed.py`) inserts **no collective** — its own
  docstring states the module must natively support communication for any to happen.
- `auto_parallel.py:~758-762` divides sharded-to-all **biases** by `n` (`weight /= n`),
  which is only correct if a later `all_sum` re-adds them `n` times.

⇒ `shared_out` is a **partial sum**. The single `all_sum` after `_moe_post_combine`
(`deepseek_v4.py:3072` → `:3076`) is what reduces **both** it and the routed `y`.
Attempt 3 (`mlx-lm 9bc2206`) moved the `all_sum` *before* the combine, leaving the
shared-expert partial **never reduced on all 43 layers**. Near-zero garbage is exactly
what that predicts.

Competing hypotheses ruled out with cites: nondeterministic reduction order is **OUT**
(jaccl assembles peer bytes into a sequence-indexed buffer and applies `reduce_op`
exactly once — `mesh_impl.h:1137`, `:1423`, `:1495`); buffer aliasing is **OUT**
(synchronous under `collective_mutex_`, zero outstanding recv WRs); an intra-process race
on `y` is **OUT** (MLX inserts the edge itself).

**Confidence, stated rather than smoothed.** That attempt 3 failed for algebra reasons:
**HIGH**. That the fence has *zero* residual numeric role: **MEDIUM-HIGH** — it is an
argument from absence, and the production comment asserts a cross-rank lockstep purpose
MLX's intra-process edge does not provide. Two facts push against the comment: the
fence's real cross-rank job (pinning collective call order) fails **loudly** with a
`DESYNC` throw (`mesh_impl.h:3573-3592`), not as silent wrong numbers; and production
already runs `mx.async_eval` (`EXO_DSV4_FENCE_ASYNC=1`), so a per-layer *blocking* fence
is already not what holds the ranks in lockstep.

**This correction is NOT acted on this round** — no fence was touched, and the settling
experiment below was deliberately not run (see "Why Phase B was not executed").

### Correction 2 — the ~1400 µs figure was being quoted in the wrong regime

The "~1400 µs per-layer all_sum drain" is **per verify-forward-pass** (the MTP
speculative-decode verify cycle), **not per output token**. ×43 = **60.2 ms**, which
matches the independently measured **56–57 ms verify bracket**. Read as per-output-token
it would be 60.2 ms against a real ~27 ms/token budget at 37 t/s — **impossible, 2.23x
over**. The number is sound; the regime label was not. **Any future citation must say
"per verify cycle."** (Primary source: `.hermes/plans/2026-05-19_allsum_tail_findings.md`
— *"~1.4ms per layer × 43 layers = 60ms drain time"* and *"Sum of per-layer p50s =
59.68ms (close to FENCE=43 verify mean 57ms)"*.)

### Correction 3 — the 2.66x does not transfer to decode scale

The 2.66x (`docs/phase0a-allsum-boundary-decomposition-2026-08-20.md:61-62`) was measured
at a **16.8 MB** payload, and that doc's own sweep is **linear at ~7 GB/s** — it states
explicitly that a fixed latency *"would be flat across this sweep."* This **falsifies the
"86 crossings × fixed commit cost"** framing in the round-2 brief. At decode's real
**32 KiB** payload the same mechanism predicts **4.7 µs** (pure-linear) to **~100 µs**
(including the ~97 µs floor, which independently matches round 1's measured 96.8 µs ±4.6
CPU→GPU dispatch latency) — **at most ~7%** of the ~1400 µs.

**Convergence worth recording:** Q5's payload-linear ~7 GB/s term and Q1's source reading
land on the same line of code — the **`input_coherent` kernel sweeping the whole buffer**
(`fence.cpp:131-140`). An independent measurement and an independent code read agreeing on
one mechanism is the strongest single result of this round.

### Correction 4 — two per-layer collectives, not one

`sharding_group` is set on Compressed/SparseCompressed attention
(`auto_parallel.py:1122-1126`), so **41 of 43 layers carry TWO collectives**
(`attn.all_gather` then `moe.all_sum`). Round 1's "43 collectives per forward" counted
the MoE collective only. Both are noted; neither changes the verdict.

### Correction 5 — a number NOT used, and why

`docs/dsv4-decode-stall-2026-06-26.md`'s "~7 ms of real GPU compute" and "2935 ms
envelope" are **ambiguous in their own source** (per-call vs summed-over-run) and were
captured with the sync-span instrumentation the record elsewhere documents as
misattributing cost. Round 1's cleaner measurement (350 µs/call `gather_qmm`) disagrees
with the June doc's 2.3 ms by 6.6x. **These are cited in this report only directionally
("the envelope is mostly stall, not compute"), never as absolute figures.**

### Hypothesis grep against the record

| Hypothesis | Already tested? | Result |
|---|---|---|
| H-a: remove/gate the per-layer fence | **YES** | OPT-7 → −23% prefill, reverted. Safe variant `EXO_DSV4_FENCE_ASYNC` shipped. |
| H-b: stream overlap of all_sum with independent work | **YES** | Attempt 3 → broke output (root cause corrected above). |
| H-c: cross-layer pipelining | **NO** (never implemented) | **Now closed structurally by this round — P1.** |
| H-d: command-buffer / commit-frequency tuning | Partially (prefill only) | Not validated for decode; ≤~7% ceiling per Correction 3. |
| **H-e: GPU-resident collective** | **NO — genuinely untested** | **The root-cause fix. Scoped below, not implemented.** |
| H-f: fuse/shrink the 43 collectives | **NO — genuinely untested** | Secondary; see below. |

---

## THE SCOPED FIX (design only — deliberately NOT implemented this round)

**H-e: make `AllReduce::eval_gpu` stop throwing — a GPU-resident collective.**

This is the only candidate that attacks the actual root cause. If the collective can run
as a Metal kernel against GPU-resident memory, the mandatory GPU→CPU handoff — the
`input_coherent` sweep, the device-wide `memoryBarrier`, and the host busy-spin —
**disappears entirely**, rather than being rearranged around. It is also the one thing
none of the three prior attempts could have done, because all three were confined to
exo/mlx-lm.

- **Where:** `mlx/backend/metal/distributed.cpp:17-19` (the throw), plus a jaccl transport
  path that can source/sink GPU buffers.
- **Blast radius:** large and genuinely uncertain — it depends on whether TB5/RDMA can DMA
  from a Metal buffer without a host bounce. **If it cannot, this fix does not exist**, and
  that question should be answered *first*, cheaply, before any code is written.
- **Ceiling:** bounded by the ~8–17% envelope share the collectives occupy — a real but
  not transformative lever, and it should be scoped against that number, not against the
  ">99% GPU idle" framing (which Correction 5 flags as unreliable).
- **Gate:** bit-equivalence FIRST, as a hard gate, before any throughput measurement.

**Recommended cheap precursor (one experiment, not a campaign):** the settling test for
Correction 1 — re-apply attempt 3's reorder **plus** a second `all_sum` on
`shared_experts(x)`. If quality is restored, H4 is confirmed and the "fence is
load-bearing for bit-equiv" claim can be retired from the record with evidence rather
than inference. It is mlx-lm-only, needs no rebuild, and is one needle run.

### Why Phase B was not executed

The brief gates Phase B on the Phase A verdict, and Phase A ruled out both (b) and (c) as
the dominant mechanism and showed (a)'s prescribed fix is structurally blocked. **There
was no Phase B left that the verdict authorizes.** Running the H-e work would mean
starting a research-grade MLX backend change on a 3-hour time-box with an unanswered
DMA-feasibility question in front of it — and running the Correction-1 settling test would
mean touching the bit-equivalence surface at the very end of a time-box, against a hard
quality gate, with no time to properly verify a failure. Both were deliberately declined
rather than rushed. **Phase A alone is a valid round per the brief.**

---

## CLUSTER HEALTH (verified by the PM directly, post-round)

```
API (192.168.86.201:52415/v1/models):  http=200
/state instance:                       DeepSeek-V4-Flash-0731, worldSize=2,
                                       deviceRank 0 + 1, 43 layers, TP sharded
completion (temp=0):                   finish_reason=stop
                                       content="The capital of France is Paris."
                                       usage: prompt 11 / completion 71 (reasoning 62)
git tracked-diff (exo):                CLEAN  (untracked tmp/ artifacts only)
submodules:                            mlx e40a416b2, mlx-lm 37260bbd6 — both unchanged
```

**Cluster is healthy and serving on the shipped production config.** This round was
**read-only**: no source file, no env knob, and no cluster config was modified; no
relaunch was performed; no benchmark was run. Nothing to revert.

*(Note: a first API probe returned no response and a second returned http=200 within
seconds — transient LAN flakiness on this MacBook's path to the Studios, a known issue,
not a cluster fault. An initial completion probe returned empty because it used the wrong
model id; re-probed with `deepseek-ai/DeepSeek-V4-Flash-0731` it returns correctly.)*

---

## DISPOSITION

**CLOSED — structural.** The per-layer decode stall is a true GPU→CPU data dependency
arising from MLX having no GPU collective. It cannot be fenced, evented, or reordered
away at the exo/mlx-lm level, and cross-layer pipelining is blocked by `hc_expand`.

**Do not fund:** any further exo-level fence rearrangement (mechanism now says why all
three failed and why a fourth would too); stream-scoped-event work (verdict b, ruled out);
command-buffer batching as a decode lever (≤~7% at real payload).

**Genuinely open, in priority order:** (1) the DMA-feasibility question gating H-e — can
TB5/RDMA source from a Metal buffer without a host bounce; (2) the Correction-1 settling
experiment, which is cheap and would let the record's bit-equiv claim be retired on
evidence; (3) H-f (fusing the 43 per-layer collectives), untested but secondary.

**Record corrections to land in `docs/`:** the fence/bit-equiv claim
(`dsv4-decode-stall-2026-06-26.md:90-94`), the "overlap primitive exists" claim (same doc,
:88-89), and the per-verify-cycle regime label on the ~1400 µs figure. These are stale
claims that have already misdirected one investigation each; they should be fixed at the
source, not just noted here.

---

## Round 2 artifacts

| File | Contents |
|---|---|
| `MECHANISM.md` | The five answers with mlx cites + the (a)/(b)/(c) verdict |
| `REPORT.md` | This file — verdict, reconciliation, scoped fix, health |
| `A1-EVAL-SEMANTICS.md` | Q1/Q4 raw archaeology |
| `A2-JACCL-DEPENDENCY.md` | Q2/Q3 raw archaeology |
| `A3-STREAM-BOUNDARY.md` | Q5 raw archaeology + round-1 harness audit |
| `A4-RECONCILIATION.md` | Number-regime reconciliation + hypothesis grep |
| `A5-PIPELINING-FEASIBILITY.md` | The P1 dependency-chain analysis |
