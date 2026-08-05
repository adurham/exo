# Batched-PP Sharding for DeepSeek-V4-Flash on the exo Cluster
(originally titled "Hybrid PP-Prefill / TP-Decode" — RENAMED 2026-08-04
after a second review flagged the old title as misleading: the design
keeps PP's layer-split topology for BOTH prefill AND decode, it does not
switch to TP for decode. "Hybrid" refers to combining PP's topology with
TP-derived REQUIREMENTS/capabilities — concurrency, cancellation,
decode throughput — not to switching sharding schemes by phase. See
Section 7 for why phase-disaggregation [PP-prefill → TP-decode, the
approach the old title implied] was considered and set aside.)

Status: DESIGN PROPOSAL — no code written yet. This document exists to be
reviewed and revised before implementation starts.

Date: 2026-08-04
Cluster: 2x Mac Studio M4 Max, 128GB each, RDMA over Thunderbolt 5 (jaccl)
Model: DeepSeek-V4-Flash (native TP and PP sharding both already exist in
this fork; see "Prior art" below)

## 1. Motivation

On this cluster, under matched thermal conditions, PP (pipeline parallel,
splitting model LAYERS across the two nodes) consistently beats TP
(tensor parallel, splitting the MoE expert weights across nodes with
attention replicated) on raw prefill throughput — roughly +14% at 500K
context and a substantially larger gap at short/medium context (measured
~490 vs ~427 tok/s at 1K tokens; see "Prior art" for citations). The
mechanism is structural: PP's point-to-point activation handoff between
pipeline stages avoids TP's per-layer `all_sum` reduction (MoE expert
output) and per-chunk `all_gather` (sparse attention indexer under
`EXO_DSV4_SEQ_SPLIT`), and TP additionally pays for fully REPLICATED
attention compute on both nodes (DSv4's LoRA-decomposed Q/output
projections can't be head-sharded without breaking `mx.quantized_matmul`,
so TP has no choice but to replicate attention entirely).

But PP has a hard limitation that makes it unusable as exo's default
mode: it is **strictly single-request-only**. exo disables ALL
coordination collectives under PP (`EXO_PP_NO_COORD_COLLECTIVE=1` —
`mx_any`, `agree_on_tasks`, `agree_on_cancellations`) because those
collectives would contend on the same physical p2p wire link PP's own
pipeline send/recv uses, and because PP's pipeline layer objects
(`PipelineFirstLayer`/`PipelineLastLayer` in `auto_parallel.py`) hold
MUTABLE per-request sequencing state (`is_prefill`, `queue_sends`)
directly on the layer instance — there is currently no way to
distinguish "whose turn is it" across two concurrent requests sharing
the same layer objects. A second concurrent request today would corrupt
the first request's pipeline state. There's also no mid-decode
cancellation — rank 1 blocks on `recv` forever if rank 0's request is
cancelled, because rank 1 has no way to know a cancellation happened.

TP, in contrast, already supports real concurrent requests today (stateless
collectives, no shared mutable per-layer request state) and already has
a working BATCHED forward path (`prefill_batched` / `BatchGenerator`)
that handles multiple simultaneous requests with heterogeneous context
lengths via right-padding + per-stream cache masking.

Goal of this design: get PP's prefill throughput advantage AND TP's
concurrent-decode capability, on the SAME cluster, for the SAME model,
without a full rewrite of MLX's distributed primitives (send/recv/
collectives) or hand-rolling network-level request tagging.

## 2.5 Hard requirements (user-specified, 2026-08-04) — THIS DESIGN MUST HIT ALL FOUR

These are not aspirational — they are the acceptance bar. A version of
this design that doesn't clear all four is not done. Restated here in one
place so Section 9's phase plan has an unambiguous target to validate
against, instead of "measure throughput" with no number attached.

1. **Concurrent requests (TP-derived).** Multiple simultaneous requests
   must work, matching TP's existing capability. STATUS: architecturally
   addressed by Section 6.2 items 1-3 (rank-0 scheduler + per-request
   cache routing); TP's own `prefill_batched`/`BatchGenerator` prove the
   underlying batching machinery already works today. NOT YET BUILT for
   the PP-batched path — this is what Phases 1-4 exist to deliver and
   validate.

2. **Instant cancellation (TP-derived).** A request must be cancellable
   and stop immediately. STATUS: real, existing caveat that must not get
   lost — TP's cancellation today works cleanly for DECODE (checked
   per-step, so effectively near-instant) but does NOT interrupt an
   in-flight PREFILL; a large prefill runs to completion regardless of a
   cancel request (warm memory facts 649/656, confirmed via direct code
   read: `POST /v1/cancel/{command_id}` closes the stream but "only
   decode checks the cancel flag, so full prefill runs to completion
   regardless"). This hybrid design's Section 6.2 item 5 (chunked
   prefill interleaved with decode) is actually a DIRECT opportunity to
   FIX this long-standing gap, not just preserve it — because prefill
   becomes a sequence of small scheduler-visible chunks instead of one
   giant blocking call, rank 0 can check for cancellation BETWEEN chunks
   and stop a prefill mid-flight for the first time in this fork's
   history. This should be treated as a first-class requirement of
   Section 6.2 item 6 ("cancellation"), not left as "same as before":
   cancellation must work during BOTH prefill and decode once this design
   ships, which is a strictly higher bar than what TP already does today.

3. **Decode throughput: ≥30 tok/s AT 500K CONTEXT — REVISED 2026-08-04,
   the goal below SUPERSEDES the original "37.5 tok/s at c=2" framing.**
   User's own words, stated directly: "My real goal here is 30 tok/s of
   decode at 500K of context, IDEALLY with that being N=2, but that may
   not be feasible... I just know we had much better decode throughput
   on TP and I want to work to get that back without losing the 400+
   tok/s of prefill we get with PP." This is a SIMPLER, more honest
   target than the original number, and changes what needs validating:
   - **The core bar: 30 tok/s decode AT 500K context.** N=2 concurrency
     is explicitly a NICE-TO-HAVE on top of that, not a hard gate —
     if 500K-depth decode throughput and N=2 concurrency turn out to be
     in tension, the 30 tok/s / 500K number wins.
   - **STATUS: UNMEASURED, for either sharding scheme, at real depth.**
     This is a real gap that needs to be named plainly, not glossed
     over: there is NO existing measurement anywhere in this fork's
     history of decode throughput specifically AT 500K context, under
     ANY configuration — PP or TP, with or without speculation. Every
     number cited in this doc's prior revisions (24.68 tok/s PP
     no-speculation, 37.5 tok/s TP+MTP, the 15.5 tok/s c=1 100K figure)
     was measured at SHORT prompts or, at best, 100K context — NOT 500K.
     Decode cost scales with context depth (KV-cache read grows), so
     none of these numbers can be assumed to hold at 500K; they're
     starting points for extrapolation, not answers.
   - **"We had much better decode throughput on TP" — TRUSTED, NOT
     RE-VERIFIED.** User's explicit direction (2026-08-04): "I don't
     think we need to re-baseline at all, we have our requirements and
     my memory says the baseline numbers we do have SHOULD be valid."
     This fork's existing TP+MTP numbers at short/100K context (fact
     745's 37.5 tok/s aggregate at c=2 short-prompt, the ~30 tok/s c=1
     100K champion figures cited elsewhere in this fork's history) are
     taken as trustworthy AS-IS — NOT scheduled for a fresh re-measurement
     pass, despite real code changes since they were measured (DSpark
     self-doubt-loop fixes, FULLBLOCK/FULLBLOCK_MOE, ROWSEQ work, the
     -0731 checkpoint switch — none of which are known/confirmed to
     have touched TP+MTP's code path specifically, but also not
     independently re-verified not to have). This is an explicit,
     deliberate scope decision: spend effort on the genuinely NEW
     measurement gap (500K depth) rather than re-litigating numbers
     already trusted from memory.
   - **BUT this does NOT resolve the actual gap — 500K decode
     throughput was never measured under ANY configuration, ever, by
     anyone, so there is nothing to "re-baseline" there in the first
     place.** "Don't re-baseline" applies to numbers that already exist
     (short/100K context data) and are being kept as-is; it does not
     and cannot apply to a number that has simply never been taken. The
     30 tok/s @ 500K target remains genuinely unmeasured ground for both
     PP and TP — this is new territory to explore when the time comes,
     not old territory to re-walk.
   - MTP vs DSpark: per Section 13.2's finding, MTP (TP-only mechanism)
     and DSpark (PP-only mechanism) are structurally different
     code paths that cannot be swapped for each other under a fixed
     sharding scheme. Whichever real 500K measurement is taken must be
     honest about which mechanism produced it and under which sharding
     scheme, not conflated.

   **CONCRETE NEXT STEP arising from this revision:** at whatever point
   this design reaches a phase that needs a real throughput number
   (per Section 9's phased plan, that's Phase 3 at the earliest — NOT
   now, not as a prerequisite to starting design/Phase-0 work), take
   ONE fresh measurement: PP no-speculation decode at 500K context
   (the "what does this design's foundation actually deliver at the
   depth we care about" number). This is NOT a TP re-baseline — per
   the correction above, TP's existing numbers are trusted as-is and
   not being re-measured. It's simply the one number that has never
   existed for PP at real depth and is needed to know how large the
   gap to 30 tok/s actually is before building the batching layer on
   top of it.

4. **Prefill throughput: 400+ tok/s (PP-derived), reduced RDMA syncs
   (PP-derived).** STATUS: PARTIALLY confirmed, with a real gap that must
   be stated plainly, not glossed over. Warm memory fact 1018's
   fresh-restart, thermally-matched PP numbers: 1K=490, 10K=512,
   94K=485, 200K=431, 400K=377, 500K=364 tok/s. **400+ tok/s holds
   cleanly from short context through roughly 200K, but the confirmed
   number AT 500K (364 tok/s) is BELOW the 400 tok/s bar.** This design
   must either (a) hit 400+ at 500K specifically — which nothing in
   today's PP numbers demonstrates yet, so this is a real, live gap this
   design needs to close, not just preserve — or (b) the requirement
   needs an explicit sign-off that "400+ at short/mid context, best-
   effort beyond that" is acceptable, which has NOT been given. Treat
   this as unresolved until Phase 3/5's real measurement, not assumed
   satisfied. On "reduced RDMA syncs": this is a real, measurable,
   already-partially-quantified TP cost this design removes by
   construction — TP pays an explicit per-chunk `mx_barrier(group)` sync
   during prefill (Section 3's `generate.py` citation) AND a per-layer
   `all_sum` collective during decode that warm memory fact 752 measured
   directly via GPU-idle profiling as a real bottleneck (302ms / 7.5% of
   decode wall time is GPU sitting idle waiting on the MoE `all_sum`
   collective to complete, not compute). PP's point-to-point send/recv
   has no equivalent per-chunk-barrier-then-collective pattern in either
   phase — this is a structural win of the design, not something that
   needs a new benchmark to prove exists, though Phase 5's real
   comparison should still quantify it (e.g. count/measure actual sync
   points per request under both modes) rather than leave it purely
   qualitative.

## 3. Prior art / cited measurements (this fork, this cluster)

- Fresh-restart, thermally-matched PP vs TP prefill comparison
  (warm memory fact 1018, 2026-07-17): PP beats TP at EVERY tested
  context depth once thermal throttling is controlled for — 1K:
  490(PP) vs 427(TP), 10K: 512 vs 379, 94K: 485 vs 358, 200K: 431 vs
  353, 400K: 377 vs 332, 500K: 364 vs 319. (+14.2% at 500K, larger gaps
  at shorter context.)
- TP architecture (warm memory facts 612, 657, 661): DeepseekV4ShardingStrategy
  in `auto_parallel.py` (~line 862) replicates attention on both ranks,
  shards only the MoE block (each rank holds half of 256 experts,
  `all_sum` reduction after). `EXO_DSV4_SEQ_SPLIT` (default on) splits
  prefill query rows across ranks + `all_gather`s the result back for
  the sparse/compressed attention indexer.
- PP architecture (warm memory fact 1014, this doc's own code reads):
  `pipeline_auto_parallel` (`auto_parallel.py` ~line 399) splits LAYERS
  across ranks. `PipelineFirstLayer`/`PipelineLastLayer` (lines 184-283+)
  do point-to-point `mx.distributed.send`/`recv_like` between adjacent
  pipeline stages, with an explicit `mx.eval()` before each send to
  materialize the tensor and isolate the lazy graph. Confirmed via direct
  code read: neither class hard-codes a batch-size-1 assumption in its
  own logic — both operate on `x.shape` generically and call
  `recv_like`/`send` on whatever tensor they're given. The batch-size-1
  constraint today comes from HOW they're driven (the decode loop and
  request scheduler), not a structural block in the layer classes
  themselves.
- PP's single-request-only limitation (warm memory session context, this
  doc's own code reads): `EXO_PP_NO_COORD_COLLECTIVE=1` is
  auto-set under Pipeline sharding (`utils_mlx.py` ~line 1618) and
  disables `mx_any`/`agree_on_tasks`/`agree_on_cancellations` because
  those collectives use `group.split()`, which the MlxRing transport
  throws on, and because they'd contend with PP's own p2p traffic on
  jaccl/RDMA. `PipelineFirstLayer.is_prefill` and
  `PipelineLastLayer.is_prefill`/`.queue_sends` are plain instance
  attributes mutated externally per-request — i.e., genuinely ambient
  per-request state living on a shared layer object, confirmed by direct
  code read (`auto_parallel.py` lines 194, 233-234).
- TP's existing heterogeneous-length BATCHED prefill machinery (this
  doc's own code reads, `generate.py` `prefill_batched()` ~line 792-1012):
  already handles N concurrent requests of DIFFERENT lengths in one
  batched forward via right-padding to `max_length`, `cache.prepare(
  lengths=..., right_padding=...)` so the per-chunk attention mask zeroes
  out padded positions, and `cache.finalize()` to roll padding off after.
  Falls back to serial prefill only for ArraysCache/SSM cache types
  (irrelevant to DSv4, which uses `CacheList(RotatingKVCache, PoolingCache,
  PoolingCache)`) or single-token prompts.
- TP's existing per-stream batched CACHE machinery (this doc's own code
  reads, `cache.py`): `BatchPoolingCache` (line 1770) and
  `BatchRotatingKVCache` (line 2795) are real, tested implementations —
  NOT batch-size-1 stubs — with genuine per-request state (`_pool_lengths`
  as a per-stream Python list, `_pending_bumps` staged per-stream,
  `left_padding` per-stream). Exercised throughout
  `dsv4_mtp.py`/`pp_speculation.py`/`batch_generate.py` and covered by
  `tests/test_pp_speculation_cache_snapshot.py`. `PerStreamBatchRotatingKVCache`
  additionally exists as a further specialization for per-stream spec-decode
  rollback.
- TP's real per-chunk collective-sync cost, confirmed via code comment
  (`generate.py` line 965-969): every `prefill_batched` chunk calls
  `mx_barrier(group)` explicitly, described in the code itself as a
  "TP-rank synchronization point... guards against rank drift before the
  next chunk's all_sum collectives fire." This is the concrete mechanism
  behind PP's prefill throughput edge — PP's point-to-point handoff has
  no equivalent per-chunk barrier-then-collective cost.

## 4. Second-opinion consult (2026-08-04, before this doc was written)

Consulted an external reference model on this exact question before
starting the code-reading gate-check above. Its answer, summarized:

- The premise "PP concurrency needs wire-level request tags" is
  overstated for THIS topology specifically. A 2-stage, single-writer,
  ORDERED point-to-point channel is fully self-describing if the sender
  (rank 0) always tells the receiver what's coming via a small metadata
  frame before the activation tensor — no MLX primitive changes needed,
  this is application-level framing on top of existing `send`/`recv`.
- Proposed design: rank 0 becomes the authoritative per-step scheduler
  (decides batch composition, phase, chunk sizes; sends a metadata
  tensor before each activation tensor); concurrency is achieved via
  BATCHING concurrent requests into one send/recv per step (not
  interleaving separate streams on the wire); 2-stage micro-batch
  pipelining (split concurrent requests into 2 micro-batches, rank 1
  works batch A while rank 0 works batch B) to fill the ~50% idle time
  single-stream PP decode leaves on a 2-node pipeline; chunked
  (Sarathi-style) prefill so large prefills don't starve concurrent
  decode requests; cancellation falls out for free (rank 0 just omits a
  cancelled request from the next step's batch).
- Named risk: whether DSv4's attention/indexer code can handle a real
  batched, heterogeneous-length request — flagged as gating the whole
  design. THIS RISK IS NOW RESOLVED (see Section 3 above and Section 5
  below) — the machinery already exists in TP's `prefill_batched` path
  and can very likely be reused rather than rebuilt.
- Also flagged: DSpark speculative decode + batching interact badly
  (divergent accept lengths per request) — plan to run DSpark only at
  concurrency=1, fall back to plain batched decode at concurrency>=2.
  Still strictly better than today's full serialization.
- Alternative it also raised: phase disaggregation (PP for prefill, TP
  for decode, hand off the tiny MLA-latent KV state between them since
  it's small enough to transfer fast even at high context). Flagged as
  possibly SIMPLER to prototype (no batched-attention refactor needed)
  but with a real memory cost: running the union of PP's per-node full
  expert set for its owned layers AND TP's half-expert-set-per-layer
  would push per-node expert-weight residency from ~50% to ~75%.
  NOT adopted as the primary direction for this doc — and, per Section 7,
  subsequently REJECTED and PULLED entirely after the refined memory math
  (Section 13.4) confirmed it's infeasible; not kept as a fallback.

## 5. Gate-check result: the "can DSv4 batch with heterogeneous lengths"
   question is PARTIALLY answered by this fork's own TP code —
   CORRECTED 2026-08-04 after a second independent review (see Section 12)

**CORRECTION: this section originally claimed the batching risk was
"ALREADY ANSWERED YES" / "ALREADY RESOLVED." A second independent design
review (Section 12) correctly identified this as overstated and
self-contradicting Risk #1 in Section 8 — the claim is retracted to the
narrower, accurate version below. Do not read this section as "risk
closed"; read Section 12 first.**

What TP's existing code actually proves, precisely: the per-layer
batched-attention/cache-masking MATH exists and works correctly WHEN
EVERY RANK RUNS EVERY ATTENTION LAYER AND SEES THE FULL SEQUENCE (TP's
actual execution model — attention fully replicated on both ranks).
This is still a real, valuable finding — the consult's stated biggest
unknown ("does DSv4 batch at all, or is it hard-coded to batch=1") is
answered: it batches, and the padding/masking/cache-prepare mechanism is
real production code, not vaporware:

- `prefill_batched()` right-pads N different-length prompts to a common
  `max_length`, builds one `(B, L_chunk)` tensor, and drives the WHOLE
  batch through the model in one forward per chunk.
- The per-layer cache objects (`BatchRotatingKVCache`, `BatchPoolingCache`)
  track per-stream real lengths independently (`_pool_lengths` list,
  `left_padding` list) and mask the attention computation so padded
  positions are invisible — the padding is a WIRE-FORMAT convenience
  (fixed tensor shape for one XLA-style forward call), not something the
  math sees as real content.
- `cache.prepare(lengths=..., right_padding=...)` / `cache.finalize()`
  is the exact mechanism: prepare tells each cache layer about this
  chunk's real vs padded lengths so the attention mask is built
  correctly; finalize rolls the padding back off the cache after.

CONSEQUENCE for the hybrid design, corrected: what's proven is narrower
than originally claimed — the per-layer batched math is proven; its
correctness when driven from metadata frames ACROSS a NEW PP rank
boundary (rank 1 reconstructing per-stream masking/lengths/offsets
purely from a metadata frame it didn't derive itself, never exercised by
TP's validation) is UNPROVEN. This is exactly Risk #1 in Section 8, and
per the second review (Section 12), it should be tested with a
hardcoded two-request batch through the PP split BEFORE any scheduler
code exists — not assumed solved. This does still mean the new work is
primarily a PP SCHEDULING/transport-and-masking-across-a-rank-boundary
problem rather than inventing batched-attention math from nothing, which
somewhat bounds the problem — but "somewhat bounds" is a materially
weaker claim than the original "resolved."

## 6. Proposed architecture

### 6.1 High-level

Keep PP's layer-split topology (rank 0 = first ~half of layers, rank 1 =
second ~half) for BOTH prefill and decode — do not disaggregate PP vs TP
by phase (see Section 7 for why phase disaggregation was considered and
set aside as the primary direction). Instead, make PP itself support
batching multiple concurrent requests through its existing point-to-point
pipeline, using the batched-attention machinery TP's `prefill_batched`
already proved out.

### 6.2 Components (numbered, not necessarily an implementation order —
    see Section 9 for the actual phased plan)

1. **Rank-0 step scheduler.** Rank 0 (today's request-accepting node)
   decides, once per pipeline step, which in-flight requests are
   included in this step's batch, what phase each is in (prefill chunk
   N, decode step N), and composes ONE batched tensor
   `(B_step, L_chunk, hidden)` covering all of them — using
   `prefill_batched`'s existing right-padding logic for prefill chunks,
   and a decode-equivalent (all decode requests contribute exactly 1
   token each per step, so no padding needed there — a decode batch is
   naturally uniform-length).

2. **Metadata-framed send/recv.** Before each activation tensor crosses
   the rank0→rank1 (or rank1→rank0) wire, rank 0 sends a small,
   FIXED-SHAPE metadata tensor: which request UIDs are in this step (as
   a fixed-width array, padded with a sentinel), each one's real
   (unpadded) length this chunk, and per-request phase flags (prefill
   vs decode, is this request's LAST chunk). Rank 1 is purely reactive —
   it reads the metadata frame, then knows exactly what shape/semantics
   the following activation tensor has and how to route/cache each
   request. This eliminates the layer-object ambient-mutable-state
   problem (`is_prefill`/`queue_sends` as instance attributes) by turning
   per-step decisions into an explicit per-step message rather than an
   externally-mutated instance flag — `PipelineFirstLayer`/
   `PipelineLastLayer` become dumb executors of whatever the metadata
   frame says, not independent holders of "current mode" state.

3. **Per-request cache routing on both ranks.** Each rank maintains its
   OWN half of each in-flight request's KV cache (rank 0 caches for its
   layers, rank 1 for its layers) — this is unchanged from today's PP,
   just needs to become a dict keyed by request UID instead of a single
   active cache, with `BatchRotatingKVCache`/`BatchPoolingCache`'s
   existing per-stream tracking reused for the within-step batch
   dimension.

4. **Micro-batch interleaving for decode.** With exactly 2 pipeline
   stages, naive single-request decode leaves each rank ~50% idle
   (rank 0 computes stage 1 while rank 1 sits idle waiting for the
   send, then vice versa). With 2+ concurrent decode requests, split
   them into 2 alternating micro-batches so rank 1 processes
   micro-batch A's stage-2 work WHILE rank 0 is already computing
   micro-batch B's stage-1 work for the NEXT step — classic 2-stage
   pipeline bubble-filling, and the 2-node case is the easiest possible
   instance of it (only 2 micro-batches needed to fully fill the
   pipeline, not N for an N-stage pipeline).

5. **Chunked prefill interleaved with decode steps.** A large prefill
   (e.g. 100K+ tokens) currently blocks everything until it's done. With
   the rank-0 scheduler, prefill for one request can be split into
   `EXO_PREFILL_STEP_SIZE`-sized chunks (the existing knob) and
   interleaved: one prefill chunk gets scheduled, then a batch of
   pending decode steps for OTHER in-flight requests, alternating — so a
   long prefill doesn't starve concurrent decode users. This directly
   reuses `prefill_batched`'s existing chunking loop structure.

6. **Cancellation.** Falls out of the scheduler design for free — rank 0
   just stops including a cancelled request's UID in future step
   metadata frames. Rank 1 never blocks indefinitely because it only
   ever `recv`s what rank 0's metadata frame told it to expect.

7. **DSpark speculative decode gating.** Per the consult's flagged risk
   (divergent per-request accept lengths under batching): DSpark stays
   gated to concurrency=1 requests only. At concurrency>=2, decode falls
   back to plain (non-speculative) batched decode. This is still a
   strict win over today's PP (concurrency=1 is IMPOSSIBLE today; DSpark
   works today only because there's never a second request to conflict
   with) — but is flagged as a real scope decision, not a minor detail:
   see Section 8.

## 7. Alternative REJECTED and PULLED (2026-08-04, user decision): phase
   disaggregation (PP-prefill → TP-decode handoff)

**STATUS: PULLED. Not a fallback, not a documented option to fall back
on — the user reviewed the refined memory math in Section 13.4 (dual
weight-layout residency leaves only ~2.7GB/node headroom, effectively
unusable) and explicitly rejected this as "not doable at all." This
section is kept only as a historical record of what was considered and
why it doesn't work — do not revisit this as a fallback option without
new information that changes the memory math.**

The consult had raised, as a possibly-cheaper alternative: use PP only
for the prefill phase (get its throughput win), then hand the completed
KV state off to a TP-mode decode process for the actual decode phase
(get TP's concurrency win). The intermediate state (MLA latents) is
small enough — the consult estimated roughly ~1KB/token, ~500MB at 500K
context — that the handoff itself should be sub-second on Thunderbolt 5.

This was NOT selected as this design's primary direction, for these
reasons:

- It requires running BOTH sharding schemes' weight layouts resident at
  once during the handoff window: PP's per-node "all experts for my
  layers" residency AND TP's per-node "half experts for every layer"
  residency are structurally different weight PARTITIONS, not the same
  weights viewed two ways — so the union would need real memory headroom.
  **CONFIRMED INFEASIBLE (Section 13.4): refined math using this fork's
  own measured per-layer weight distribution puts dual-layout residency
  at ~115.3GB/node against a ~118GB budget — only ~2.7GB left for KV
  cache and margin, below even a single request's real KV footprint at
  any meaningful depth. This is not a tight-but-workable number; it's
  categorically unusable.**
- It introduces a genuinely NEW class of engineering (a live process
  handoff / weight-layout conversion mid-request) that doesn't reuse any
  existing exo machinery, whereas Section 6's batched-PP design reuses
  `prefill_batched`'s already-proven padding/masking/cache logic almost
  directly.
- It would still leave PP's per-request KV cache format (RotatingKVCache/
  PoolingCache halves split by LAYER across ranks) needing conversion
  into TP's format (same cache types but split by nothing — TP replicates
  attention, so TP's KV cache is NOT split across ranks at all, just
  duplicated) — a real, nontrivial cache-format translation on the hot
  path of every single request.

**This alternative is NOT a fallback.** If Section 6's batched-PP
scheduler design hits a wall that makes it structurally infeasible, this
is NOT where the project goes next — the memory math rules it out
regardless of how Section 6 turns out. A genuinely different approach
would need fresh design work (see Section 14 for the related open
question this raises about Requirement 3's exact scope, since this was
the option that would have let the design keep TP's already-proven
37.5 tok/s number outright).

## 8. Real risks (not glossed over)

1. **DSv4 attention/indexer numerics under a NEW batching pattern.**
   Section 5 establishes that DSv4 CAN batch heterogeneous lengths — but
   `prefill_batched` was built and validated for TP's forward pass
   (attention replicated on both ranks, full sequence visible to both).
   Running the SAME padded/masked batch through a PP-split forward
   (rank 0 sees only its layers' worth of the batch, rank 1 only its
   layers') has not been tested and may surface new edge cases,
   particularly around the Indexer's pooled-KV top-k search (which reads
   `Compressor`/`PoolingCache` state that must now also be correctly
   per-request-masked across a PP layer boundary, not just within a
   single rank's forward as `prefill_batched` does today). Needs a
   dedicated correctness test (batched-PP output vs known-good serial
   output, per request, byte-for-byte at temp=0) before trusting this
   for anything beyond a lab environment.
2. **DSpark + batching scope cut.** Restricting DSpark to concurrency=1
   is a real product/performance tradeoff, not a free lunch — it means
   the fastest decode mode (DSpark) and the concurrency win are mutually
   exclusive in this design's first version. Worth flagging explicitly
   as a decision to revisit later (a batched-DSpark verify with
   per-request accept-length handling is a real but separate, harder,
   follow-on project), not something to silently accept as permanent.
3. **The existing DSpark-specific ~18-20% catastrophic-stall bug is
   STILL UNRESOLVED** (see `docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md`,
   this fork, ongoing separate investigation as of this doc's writing).
   This hybrid design does not depend on DSpark being fixed (item 7 in
   Section 6.2 already scopes DSpark to concurrency=1-only), but if
   DSpark's own reliability bug is still open when this design reaches
   implementation, concurrency=1 requests running DSpark under the new
   batched-PP scheduler will STILL be exposed to that separate,
   unrelated bug. Not a reason to block this design, but should not be
   conflated with it either — two independent DSpark-adjacent problems
   in flight at once.
4. **Per-step metadata-frame overhead at small step counts / short
   requests.** The consult flagged this as a "measure, might need to
   fuse meta into a header row" risk — an extra small `mx.eval`-forcing
   send/recv per step could dominate wall time for very short requests
   or very small batches. Needs real benchmarking, not assumed away.
5. **This is genuinely new, nontrivial concurrency-control code** in a
   part of the system (`PipelineFirstLayer`/`PipelineLastLayer` +
   whatever new rank-0 scheduler component gets built) that currently
   has ZERO concept of multiple simultaneous requests. Bugs here are the
   kind that show up as silent cross-request data corruption (request A's
   tokens leaking into request B's output) rather than a clean crash —
   this needs a correctness-first test suite (adversarial concurrent-
   request tests, not just a throughput benchmark) before this could ever
   be trusted as a production default, given this fork's own standing
   priority on correctness over speed for exactly this class of bug (see
   the DSpark self-doubt-loop and FULLBLOCK investigations, both examples
   of subtle batched-verify correctness bugs that took real investigation
   to find).
6. **Effort/timeline is genuinely substantial**, not a quick patch — this
   touches the model's cache-preparation path, the PP layer classes, and
   requires a new scheduler component with real state machine logic. The
   user has explicitly asked for this to be done right, not as a cheap
   prototype; Section 9's phased plan reflects that by front-loading
   correctness validation before any throughput claims are made.
7. **Decode throughput is completely unquantified for PP — the single
   biggest de-risking gap in this doc (found by second review, Section
   12).** Section 3 has detailed PP PREFILL numbers across the full
   context range. There are ZERO PP DECODE numbers anywhere in this doc
   or the cited fork history. Requirement 3 (≥37.5 tok/s aggregate at
   c=2) is confirmed ONLY under TP. Single-request PP decode pays a
   real per-token wire hop plus, on exactly 2 pipeline stages, a ~50%
   idle bubble per rank absent interleaving — Section 6.2 item 4's
   micro-batch interleaving is the ONLY mechanism proposed to recover
   that, and under the phased plan as originally written, this isn't
   even MEASURED until Phase 3, the most expensive phase to discover a
   hard-requirement failure in. See the new "Pre-Phase-0 checks" in
   Section 9 for the fix.
8. **Requirement 3 (37.5 tok/s) may conflict with item 7's DSpark
   scope cut (found by second review).** The 37.5 tok/s TP benchmark
   (fact 745) was measured "MTP on." This design's item 7 gates
   speculative decode (DSpark) to concurrency=1, falling back to plain
   single-token batched decode at concurrency≥2. If fact 745's 37.5
   tok/s number depended on multi-token speculative acceptance (MTP),
   and this design's c≥2 decode is plain (no speculation), the design
   may be structurally incapable of hitting 37.5 tok/s at c=2 REGARDLESS
   of how well micro-batch interleaving works — a scope decision, not an
   engineering bug, but one made accidentally rather than deliberately if
   left unexamined. NEEDS RESOLUTION: confirm whether fact 745's "MTP on"
   config is the SAME mechanism as this design's DSpark gate, or a
   different (always-batchable) speculative path, before accepting item 7
   as written. See Section 9's new pre-work step.
9. **Requirement 4 (400+ tok/s prefill) is not met by the numbers already
   in this doc, even before adding new overhead (found by second
   review).** Section 3's PP prefill numbers (364 tok/s at 500K) are
   SINGLE-REQUEST measurements. This design's metadata-frame overhead
   (item 2) and prefill-interleaved-with-concurrent-decode (item 5) will
   make hybrid prefill-under-load strictly SLOWER than the already-
   sub-400 single-request 500K number, not faster. No phase in the
   original plan explicitly closes this gap — it must not be allowed to
   die silently in Phase 5's final comparison. Either this gets
   explicit engineering attention, or Requirement 4 needs an explicit,
   deliberate renegotiation (e.g. "400+ through 200K, best-effort
   beyond") — NOT a default that happens by omission.
10. **Cancellation-by-omission (item 6) is underspecified and leaks
    memory (found by second review).** As written, "rank 0 just stops
    including a cancelled request's UID in future step metadata frames"
    is AMBIGUOUS to rank 1: it cannot distinguish "this request simply
    wasn't scheduled THIS step" (normal — happens constantly under
    chunked prefill interleaving, Section 6.2 item 5) from "this request
    was CANCELLED — free its KV/PoolingCache state now." Item 6 needs
    an explicit EVICTION entry in the metadata frame (not just omission),
    plus an explicit idle/shutdown frame so rank 1 is never left blocked
    on `recv` with zero real work scheduled.
11. **No wire-protocol state machine or deadlock analysis exists yet
    (found by second review).** Micro-batch interleaving (item 4) means
    TWO pipeline steps in flight on one physical link at once, combined
    with MLX's lazy evaluation — exactly the condition class where
    from-scratch distributed code deadlocks silently. The metadata-frame
    protocol (item 2) needs to be written out as an explicit state
    machine (what each rank sends/expects/blocks-on, in what order, for
    every phase combination) BEFORE Phase 1 starts, not discovered
    empirically while debugging a hang. Also needs an explicit check
    (not yet done) of whether jaccl's `recv` can accept a dynamically-
    shaped activation tensor purely from a metadata frame's declared
    shape, or whether MLX/jaccl requires shapes to be collectively
    pre-agreed — if the latter, that would be a hidden change to
    transport-level assumptions this doc currently declares out of scope
    (Section 10).
12. **No KV-cache memory budget exists for concurrent requests at real
    context depth (found by second review).** Requirement 3 must be
    validated at 100K-500K context per item 3 of Section 2.5. Two (or
    more) simultaneous deep-context KV caches, split by LAYER across
    128GB nodes, alongside each node's resident expert weights, may
    simply not fit — this is unchecked arithmetic, not yet done anywhere
    in this doc. Its absence risks discovering an OOM wall at Phase 4's
    realistic-concurrency load test rather than on paper beforehand.

## 9. Proposed phased plan (no code written yet — this is the plan to
   review, not a commitment to start immediately)

**PRE-PHASE-0 CHECKS (added 2026-08-04 after second review, Section 12
— these are cheap, mostly arithmetic/measurement, and each one can
independently invalidate or reshape the whole design; do ALL of them
before Phase 0's correctness harness work begins):**

- **Decode-throughput ceiling estimate (addresses Risk #7).** Measure
  TODAY's single-request PP decode tok/s (no new code needed — this
  already exists and runs). Compute the theoretical BEST-CASE aggregate
  at c=2 assuming perfect micro-batch interleaving (i.e., what would
  2-stage bubble-filling get you if it worked flawlessly). Compare
  against the 37.5 tok/s bar. If the theoretical ceiling doesn't clear
  37.5 with real margin, Requirement 3 is unreachable by this design as
  structured and that needs to surface NOW, not after Phase 3's
  multi-week investment.
- **MTP/DSpark disambiguation (addresses Risk #8).** Determine whether
  fact 745's "MTP on" 37.5 tok/s config is the SAME speculative
  mechanism this design's item 7 gates to concurrency=1 (DSpark), or a
  distinct, always-batchable path. This directly determines whether item
  7's scope cut is compatible with Requirement 3 at all.
- **KV-cache memory budget at depth (addresses Risk #12).** Compute,
  don't guess: per-node resident expert weights (known, ~77-89GB) +
  N concurrent requests' KV cache size at 100K/500K context (KV/token
  is a previously-measured, cited figure in this fork's history — reuse
  it, don't re-derive) — does N=2 fit in 128GB with room for runtime
  overhead? Does N=4? This bounds how much real concurrency this design
  can ever support at deep context, independent of whether the
  scheduling logic itself works.
- **Fallback memory math for Section 7's phase-disaggregation
  alternative — DONE, RESULT NEGATIVE, ALTERNATIVE PULLED.** See
  Section 13.4: refined math shows only ~2.7GB/node headroom after
  dual-layout residency, effectively unusable. Per explicit user
  decision (2026-08-04), this alternative is REJECTED and no longer a
  documented fallback — Section 7 updated accordingly. Do not revisit
  without new information that changes the underlying memory math.

If any of the first three checks come back negative (ceiling doesn't
clear 37.5, DSpark/MTP are the same mechanism and item 7 breaks
Requirement 3, or KV memory doesn't fit even N=2 at real depth), STOP
and revise this doc's approach before writing Phase 0 code — these are
exactly the checks a second review identified as cheap enough to do
first and expensive to discover late.

**Phase 0 — Correctness baseline (before any new scheduler code) —
METHODOLOGY CORRECTED 2026-08-04 after second review (Section 12):**
Original methodology (diff TP batched output vs PP serial output,
byte-for-byte) is WRONG and was flagged by the second review as likely
to produce constant false alarms: TP's `all_sum` reduction order and
quantized-matmul partitioning produce genuinely different float
accumulation than PP's single-device compute, and batched/padded
kernels differ numerically from unbatched ones even within the SAME
sharding scheme — byte-equality across TWO DIFFERENT sharding schemes
is not a meaningful correctness bar and would either cause constant
false-positive failures or force loosening tolerances until real bugs
hide behind them. CORRECTED baseline: diff **serial single-request PP**
(today's already-trusted, already-shipped code) against **new batched
PP** (once Phase 1+ exists), SAME sharding scheme throughout, using
greedy-token-identical output or a tight logit-tolerance comparison —
NOT cross-sharding byte equality. Phase 0's actual deliverable, given
this correction, is establishing that harness/tooling and confirming
what "correct" means for the SAME-sharding comparison, since there is
no batched-PP code yet to diff against — this phase is largely
tooling + the Pre-Phase-0 checks above, not a diff that can fully run
until Phase 0.5 exists.

**Phase 0 tooling — DONE, 2026-08-05.** Built and validated the
correctness-diff harness this phase exists to deliver, on the local
dev machine (no cluster relaunch needed — pure single-process MLX/CPU
work): `src/exo/worker/engines/mlx/pp_batched_correctness.py`
(`SimPipelineTransport`, `build_two_rank_split`, `run_two_rank_pp_forward`,
`compare_logits`) plus
`src/exo/worker/engines/mlx/tests/test_pp_batched_correctness_harness.py`
(7 tests, all passing, basedpyright/ruff clean). Reviewed via `consult`
before writing code — the reviewer flagged three real correctness risks
in the original plan (real OS threads directly touching MLX's lazy
graph/eval machinery is not documented thread-safe; anchoring against
*simulated* serial PP instead of a plain unsharded forward would
validate one unproven thing against another; passing the same `mx.array`
object across simulated ranks would hide real transport bugs), all
addressed in the shipped harness: a global `_MLX_CALL_LOCK` serializes
actual MLX op execution across the two simulated-rank threads (released
only while blocked inside the fake transport's `recv_like`); the golden
reference is a PLAIN unsharded forward, not "trust the split"; the fake
transport copies (numpy roundtrip) rather than aliasing.

One real, expected finding surfaced while validating the harness itself
(not a harness bug): a REAL 2-rank simulated split does NOT match a
plain unsharded forward at float-tolerance precision, because the real
`PipelineFirstLayer`/`PipelineLastLayer` classes cast activations to
bf16 before every cross-rank send (a genuine JACCL/RDMA requirement) —
a cost the plain forward never pays. Measured directly: max logit diff
~0.18-0.20 for a small random-weight test model, well outside a
byte-equality-style tolerance, but with ZERO greedy-token (argmax)
mismatches across all tested decode steps. This means the design doc's
own "greedy-token-identical output OR a tight logit-tolerance
comparison" framing (this section, above) needs to be read as an
EITHER/OR, not both simultaneously satisfiable at a tight tolerance —
greedy-token agreement is the meaningful correctness bar for a
cross-transport-hop comparison; a tight float tolerance is only
meaningful for a SAME-transport-cost comparison (e.g. serial-PP vs
batched-PP, Phase 0.5+, which both pay the identical bf16 cast cost and
so CAN reuse a tight tolerance). Test file docstrings capture this
finding in detail for whoever picks up Phase 0.5+.

Not yet done: the actual Phase 0.5 diff (serial-PP-through-this-harness
vs real batched-PP output) — there is still no batched-PP code to diff
against; that's Phase 0.5+'s job once the metadata-framed transport
exists.

**Phase 0.5 — Transport-only refactor at concurrency=1 (NEW phase added
2026-08-04 after second review, Section 12):** The original Phase 1
conflated THREE distinct pieces of new work — (a) the metadata-framed
transport protocol replacing today's ambient mutable per-layer state,
(b) the rank-0 scheduler, and (c) actual batching of multiple requests
— into one phase. Given Risk #11 (silent-cross-request-corruption is
this design's most dangerous failure mode), isolate (a) first: run a
SINGLE request through the NEW metadata-framed send/recv protocol (no
scheduler, no batching, concurrency still =1) and verify EXACT parity
against today's existing PP transport. This isolates "did I break the
transport" bugs from "did I break the batching" bugs before they can
compound — a transport bug discovered under concurrency=2 batched load
is much harder to localize than one caught here in isolation.

**Phase 0.5 — DONE, 2026-08-05 (simulated, local — real-cluster A/B
still pending).** Built `src/exo/worker/engines/mlx/pp_metaframe.py`
(`MetaFramedPipelineFirstLayer`/`MetaFramedPipelineLastLayer`,
`encode_metaframe`/`send_metaframe`/`recv_metaframe`,
`handshake_metaframe_protocol`,
`install_metaframed_pipeline_layers`) plus its test suite
`src/exo/worker/engines/mlx/tests/test_pp_metaframe.py` (9 tests, all
passing, stable across 5 repeated runs, basedpyright/ruff clean).
Reviewed via `consult` before writing code (2026-08-05) — the reviewer
shaped five real design decisions: (1) the frame is a fixed HEADER +
per-request TABLE, not a flat tuple, specifically so Phase 1's
scheduler can add rows without a second protocol-shape change; (2) a
`version` field in the header, checked on every frame; (3) a startup
`handshake_metaframe_protocol` so a per-node env-var mismatch
(`EXO_PP_METAFRAME` set on one node, not the other) fails loudly at
warmup instead of hanging on the first real request; (4) the new
`MetaFramedPipelineLastLayer` reuses today's `_pending_prefill_sends`/
`flush_prefill_sends` queue directly rather than reimplementing
`queue_sends`' timing semantics, since a byte-identical-token
comparison alone would never catch a queuing/deadlock regression; (5)
brand-new classes (not an in-place flag on `PipelineFirstLayer`/
`PipelineLastLayer`), so today's shipped transport is provably
untouched by this work.

Validated via the Phase 0 harness's simulated-2-rank machinery
(`pp_batched_correctness.py`), not the real cluster yet — per Phase
0.5's own isolation rationale, proving this locally first is strictly
cheaper than discovering a transport bug during a live A/B. Exact
parity (argmax mismatches==0, max logit diff < 1e-4 — both transports
pay the identical bf16 cast cost, so a tight tolerance is the correct
bar here, unlike Phase 0's plain-forward-vs-split comparison) confirmed
across three coverage cases the consult review specifically asked for:
a multi-chunk prefill + 8-step decode, a single-token-prompt edge case,
and a 24-step long decode (stresses the phase-transition/`is_last_chunk`
boundary where the old ambient-flag toggling and the new explicit
per-step framing would be most likely to disagree).

One real bug found and fixed while building this, not just written
blind: the first draft of `MetaFramedPipelineLastLayer` omitted the
decode-only final-hidden-state handoff (last pipeline stage sending its
output back to rank 0 for sampling) — incorrectly reasoned as
"out of scope" for a transport-only phase, when in fact it's required
for decode to function at all. Phase 0.5's own parity tests caught
this immediately (decode diverged/deadlocked without it) — exactly the
kind of bug this isolated phase exists to surface before it could
compound with Phase 1's scheduler/batching code.

**Not yet done: the real 2-node cluster A/B.** Local simulation proves
the protocol's NUMERICS and control flow are correct; it does NOT
prove real jaccl/RDMA transport behavior (actual wire bytes, real
timing, real multi-process semantics) — that requires the actual
`EXO_PP_METAFRAME=1` env var wired into `start_cluster.sh`'s allow-list
(not yet done — `pp_metaframe.py` is not called from any production
code path yet, purely additive/dormant) and a live restart. That real
A/B, plus wiring the metaframe classes into
`pipeline_auto_parallel`/`mlx_generate` behind the flag, is the
concrete next step whenever cluster time is available — needs the
user's separate explicit go-ahead per standing rules, not implied by
this local validation work being complete.

**Wiring + FIRST REAL CLUSTER ATTEMPT — 2026-08-05, found and fixed a
real bug the local simulation could not have caught.** Wired
`EXO_PP_METAFRAME=1` into `set_pipeline_prefill`/`set_pipeline_queue_sends`
(`auto_parallel.py`, lazy-imports the metaframe counterparts, no
circular-import issue, no new basedpyright/ruff errors vs the
unmodified baseline — verified by stash-diffing both files), into the
`PipelineShardMetadata` model-loading branch (`utils_mlx.py`, installs
the metaframe layers via `install_metaframed_pipeline_layers` right
after `pipeline_auto_parallel` returns, calls
`handshake_metaframe_protocol` on BOTH the enabled and disabled path so
a config mismatch fails loudly either way), and into
`start_cluster.sh`'s env allow-list (default `0`, both nodes always get
the identical value via the shared `EXO_ENV` variable — same
propagation mechanism already proven safe by `EXO_PP_NO_COORD_COLLECTIVE`
right above it). Full worker test suite (628 tests) run before touching
the cluster: 3 pre-existing failures, all confirmed present on
unmodified `main` via the same stash-diff method — zero regressions
introduced.

Relaunched the real cluster with `EXO_PP_METAFRAME=1` (with the user's
separate explicit go-ahead, per standing rules) — placement succeeded,
the `EXO_PP_METAFRAME=1` startup banner printed correctly on both
nodes, but BOTH runners then failed during warmup with
`RunnerFailed: ValueError: not enough values to unpack (expected 4, got 3)`
inside `hyper_connection.py`. Root cause: DSv4-Flash broadcasts its
residual stream to 4D `(batch, seq_len, hc_mult, hidden_dim)`
immediately after embedding (hyper-connections — see
`HyperHead`/`mlx_lm.models.hyper_connection`) and keeps it 4D through
EVERY layer, only collapsing back to 3D at the final `hc_head`/`norm`.
The metaframe protocol's v1 `activation_template_shape()` hardcoded a
3D `(batch, seq_len, hidden_dim)` `recv_like` template — silently wrong
for DSv4, since `mx.distributed.recv_like` needs a template matching
the ACTUAL rank of the tensor being received, not just its last-dim
size. **This gap was invisible to Phase 0.5's local test suite for a
structural reason, not an oversight in test coverage per se**: that
suite (deliberately, per this doc's own Phase 0/0.5 methodology)
uses `mlx-lm`'s plain `Llama` as the test model to keep validation fast
and independent of the real 166GB DSv4 checkpoint — and plain Llama has
no hyper-connections, so it never leaves 3D. The bug only exists in the
intersection of "the new transport code" and "DSv4's specific residual-
stream shape," which by construction the local harness never exercises.
**This is exactly why Phase 0.5 exists as an isolated, cheap-to-fail
step before Phase 1** — the bug surfaced immediately on the very first
real-model exercise, cleanly isolated to the transport layer alone (no
scheduler/batching code yet to also debug), and cost one relaunch
cycle to find, not a multi-week investigation compounded with Phase 1
work.

Fix (protocol v2, same day): added an `extra_dim` field to the fixed
header (0 = the common 3D case; a positive value = the size of the
extra middle dimension for a 4D residual stream like DSv4's
`hc_mult`). `MetaFramedPipelineLastLayer` derives this from the ACTUAL
tensor being sent (`tensor.ndim == 4` check) at both its `encode_metaframe`
call sites — never a static per-model-type assumption, so this
generalizes to any future architecture with an analogous extra
dimension without a further protocol change. `activation_template_shape()`
returns the correct 3-tuple or 4-tuple accordingly. Three new
regression tests added (encode/decode roundtrip at DSv4's real
`hc_mult=4`, an integration test driving a real
`MetaFramedPipelineLastLayer` instance with a 4D-output stand-in layer
through the actual send/encode call path, and a 3D-still-default guard
proving the fix doesn't change behavior for every non-DSv4 model
already covered) — 15/15 tests passing, stable across 3 repeated runs,
basedpyright/ruff clean. **The v2 fix itself has NOT yet been re-run
against the real cluster** — after the failure, the cluster was
immediately restored to today's known-good transport
(`EXO_PP_METAFRAME=0`) and the CORRECT model
(`deepseek-ai/DeepSeek-V4-Flash-0731` — a first restore attempt
mistakenly loaded the wrong default, `mlx-community/DeepSeek-V4-Flash`,
caught and corrected before declaring the cluster safe), both runners
confirmed `RunnerReady` on the right model. The next real-cluster A/B
with the v2 fix is the concrete next step, still gated on the user's
separate explicit go-ahead per standing rules.

**SECOND REAL CLUSTER ATTEMPT — 2026-08-05, v2 fix verified, found and
fixed a second real deadlock (v3 fix).** With the user's separate
explicit go-ahead, relaunched with `EXO_PP_METAFRAME=1` +
`DSV4_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-0731` (the correct model
this time). The 4D hyper-connection shape bug was confirmed FIXED — no
more `ValueError: not enough values to unpack`. But both runners then
failed with a NEW error: `RuntimeError: [jaccl] recv() deadline in
drain` — a two-sided deadlock (rank 0 stuck in its own decode-gather
`recv_metaframe` call, rank 1 stuck in `MetaFramedPipelineFirstLayer`'s
forward-hop recv).

Root cause (found via a `consult` review of the exact two-rank failure
trace, not blind trial-and-error): in rank 0's
`MetaFramedPipelineLastLayer.__call__` during decode, the forward-hop
block builds a lazy `mx.distributed.send(...)` node and assigns it to
`output` — then the decode-only handoff block a few lines later
IMMEDIATELY overwrites that same `output` variable with the
decode-gather recv result, discarding the send's only reference before
anything ever calls `mx.eval()` on it. MLX distributed ops are lazy:
building the graph node transmits no bytes, only `eval()` does. The
activation therefore never actually left rank 0 — rank 1 blocked
forever waiting for it that would never arrive, and rank 0 (having sent
nothing real) then blocked forever waiting for rank 1's gather reply
that rank 1 could never produce, since rank 1 itself never got past its
own stuck recv.

Fix (v3): explicitly `mx.eval()` the forward-hop send immediately,
before `output` can be reassigned — matching every other send call in
the file. Added a regression test using GENUINELY lazy hand-rolled
send/recv fakes rather than `pp_batched_correctness.py`'s
`SimPipelineTransport` — the latter eagerly calls `mx.eval()`
internally by design (documented, needed for its own OS-thread
synchronization), which was EMPIRICALLY CONFIRMED to mask this exact
bug: an earlier draft of the regression test built on
`SimPipelineTransport` passed even against the unfixed code, defeating
its own purpose. The final test was verified BOTH ways — explicitly
reverted the fix via `git stash` and confirmed the test fails against
the buggy code, then restored the fix and confirmed it passes — not
just "wrote a test and it happened to pass once." 13/13 tests passing,
stable across 3 repeated runs, basedpyright/ruff clean, full worker
suite shows the same single pre-existing unrelated failure as before
(no new regressions).

Cluster restored again to `EXO_PP_METAFRAME=0` with the correct model,
both runners confirmed `RunnerReady`, before any further code work.
**Two real, structurally distinct bugs found on two consecutive real
cluster attempts — this is exactly what Phase 0.5's isolation rationale
predicted:** local simulation validates numerics and control-flow
logic but cannot by construction exercise the real RDMA transport's
lazy-eval/deadline semantics or a specific model's real tensor shapes.
Each bug was found, fixed, tested, and pushed in isolation, at
transport-scope, before any Phase 1 scheduler/batching code exists to
compound the debugging surface — the whole point of running this phase
separately. The v3 fix has NOT yet been re-run against the real
cluster; that's the next step, still gated on the user's separate
explicit go-ahead per standing rules.

**Phase 1 — Rank-0 scheduler skeleton, decode-only, 2 concurrent
requests, NO speculative decode:** Build the metadata-frame protocol and
the rank-0 scheduler for the SIMPLEST case first — 2 concurrent plain
(no DSpark) decode-only requests (both already prefilled via today's
existing serial PP prefill, just testing the NEW concurrent decode
path). Validate byte-for-byte correctness against 2 serial single-
request PP runs before touching throughput at all. (This phase now adds
ONLY the scheduler+batching delta on top of Phase 0.5's already-verified
transport, per the isolation rationale above.)

**Phase 2 — Extend to prefill batching + chunked interleaving:** Add
batched/chunked prefill through the new scheduler, reusing
`prefill_batched`'s padding/masking logic adapted for the PP split.
Validate correctness (Phase 0-style diff) for BOTH the prefill and
decode halves together, at 2 concurrent requests with different context
lengths.

**Phase 3 — Micro-batch interleaving for decode throughput:** Once
correctness is established at 2 concurrent requests, add the 2-stage
micro-batch interleaving from Section 6.2 item 4, and THEN measure
throughput — this is the first point in the plan where a throughput
number should be trusted/reported, because correctness is already
locked in by Phases 0-2.

**Phase 4 — Cancellation, DSpark gating, N=2 realistic-load validation
(SCOPE CONFIRMED 2026-08-04: N=2 is the real target, N>2 is explicitly
NOT a goal — see Section 13.3):** Wire up cancellation (item 6) and add
the DSpark concurrency=1-only gate (item 7). Full correctness +
throughput validation at N=2 concurrency under realistic load (mirroring
actual Hermes usage patterns — the existing `/tmp/hermes_stress_test.py`
corpus-replay harness built for the DSpark-off stability soak test is a
natural fit to reuse/extend here for a mixed prefill+decode concurrent-
load validation at N=2). Extending beyond N=2 concurrent requests is
explicitly OUT of scope for this design per the user's confirmed target
— if ever revisited later, Section 13.3's growing-session KV ceiling
(N=4 doesn't fit at 500K context) is the first thing to re-check before
attempting it.

**Phase 5 — Production readiness:** basedpyright/ruff/nix fmt compliance
per this repo's standing pre-commit requirements, full test suite
addition under the appropriate `tests/` directory, decision on whether
this becomes exo's new default sharding mode for DSv4 or an opt-in flag
alongside existing PP/TP, and a real load-bearing throughput comparison
against both existing modes under matched thermal/load conditions
(reusing the "fresh-restart, no back-to-back thermal contamination"
discipline from fact 1018/1017's own hard-won methodology lesson).

## 10. Explicitly out of scope for this doc

- Fixing the existing, separately-tracked DSpark ~18-20% catastrophic
  stall bug (`docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md`).
  Related in that both touch PP-mode code, but a different bug, being
  investigated separately.
- Any change to jaccl/RDMA transport internals, MLX's C++
  `mx.distributed.send`/`recv`/collective primitives, or the mesh
  completion-tracking layer. Everything in this design operates entirely
  at the Python application layer, on top of existing MLX primitives.
- Extending this hybrid approach to any model other than DSv4-Flash on
  this specific 2-node cluster topology. The design as written assumes
  2 pipeline stages specifically (Section 6.2 item 4's "exactly 2
  micro-batches" simplification would need generalizing for N>2 stages).
- Supporting more than N=2 concurrent requests (CONFIRMED explicitly out
  of scope by the user, 2026-08-04 — "I wouldn't ever target more than
  N=2 there"). Phase 4 (Section 9) is scoped to N=2 accordingly; the
  growing-session KV-memory ceiling at N=4+ (Section 13.3) is a real,
  quantified constraint but not one this design needs to solve.
- Phase disaggregation (PP-prefill → TP-decode handoff) as a fallback or
  alternative approach — REJECTED and PULLED (Section 7, per user
  decision after reviewing Section 13.4's refined memory math showing
  it's infeasible, ~2.7GB/node headroom). Kept in the doc only as a
  historical record of what was considered and ruled out.

## 11. Open questions for review before implementation starts —
    ANSWERED 2026-08-04 by the second review (Section 12); answers below,
    original questions kept for record

1. **Fuzzing before Phase 1?** ANSWER: yes, before Phase 1 — but at the
   LOGIC level (property-based/randomized testing of batch composition,
   length permutations, request join/leave/cancel sequences against the
   cache-routing and mask-construction code specifically), not
   end-to-end cluster-level fuzzing, which should wait for Phase 2/4
   once there's a real system to fuzz against. Logic-level fuzzing is
   cheap and targets exactly Risk #5/#11 (silent cross-request
   corruption, deadlock) directly.
2. **DSpark gated to concurrency=1 — acceptable permanent v1 boundary?**
   ANSWER: acceptable ONLY AFTER Risk #8 is resolved (confirm whether
   fact 745's "MTP on" 37.5 tok/s baseline used the SAME mechanism this
   design gates off). Do not answer this question until the Pre-Phase-0
   MTP/DSpark disambiguation check (Section 9) has real data — if
   Requirement 3 depended on the gated mechanism, batched speculative
   decode must be pulled INTO scope, not deferred.
3. **Check phase-disaggregation's memory math now?** ANSWER: yes, do it
   now, as part of the Pre-Phase-0 checks (Section 9) — cheap (~30 min),
   and do the PRIMARY design's own KV-memory math (Risk #12) at the same
   time, for the same reason: if the primary design's pre-checks reveal
   a wall, you want the fallback's viability already known rather than
   discovered mid-crisis.
4. **Timeline expectation?** ANSWER: this is realistically MULTI-WEEK,
   not multi-day — from-scratch distributed concurrency-control code
   with a silent-data-corruption failure mode (Risk #5/#11) is
   consistently underestimated when treated as a quick build. Structure
   this with per-phase estimates and explicit kill/checkpoint criteria,
   with the FIRST checkpoint being the Pre-Phase-0 checks (Section 9)
   and the SECOND being the end of Phase 0.5/Phase 1 (transport +
   scheduler correctness, before any throughput work begins).

## 12. Second independent design review (2026-08-04)

A second, independent review of this doc (as it stood at commit
`4a212d7ea`, before the corrections in this revision) was obtained via
the `consult` tool at the user's explicit request, after Section 2.5's
hard requirements were added. Full review record kept here for
traceability; findings have been incorporated throughout this doc
(Section 5's retracted claim, Section 8's new risks #7-12, Section 9's
Pre-Phase-0 checks + Phase 0 methodology fix + new Phase 0.5, Section
11's answered open questions, and the title itself).

**Review findings, verbatim structure preserved:**

1. **The "already resolved" claim in Section 5 was overstated and
   self-contradicted the doc's own Risk #1.** What TP's code actually
   proves is narrower than originally claimed: the per-layer batched
   math works when every rank runs every attention layer and sees the
   full sequence (TP's actual execution model). Under the proposed PP
   design, rank 1 must reconstruct per-stream masking/lengths/offsets
   purely from metadata frames it didn't derive itself — new
   coordination code, unproven, and precisely the class of risk the
   doc's own Risk #5 (silent cross-request corruption) already warned
   about. CORRECTED in Section 5 and the CONSEQUENCE paragraph
   immediately following it.
2. **Decode throughput is completely unquantified for PP** — detailed
   prefill numbers exist across the full context range; zero PP decode
   numbers exist anywhere. This is flagged as the single biggest
   de-risking omission, since Requirement 3 is a hard bar and nothing in
   the original phase plan measured it until the expensive Phase 3.
   Recommendation: estimate the theoretical best-case ceiling TODAY
   (near-free, no new code) before investing in Phase 0. ADDED as Risk
   #7 and the first Pre-Phase-0 check.
3. **Requirement 3 may conflict with the DSpark concurrency=1 scope
   cut** if fact 745's "MTP on" baseline used the same speculative
   mechanism being gated off — unresolved ambiguity in the original
   doc. ADDED as Risk #8 and a Pre-Phase-0 disambiguation check.
4. **Requirement 4 (400+ tok/s prefill) is not met by the numbers
   already in the doc**, before any new overhead from this design is
   even added (the cited 500K single-request PP number, 364 tok/s, is
   already below 400) — flagged as something the original doc's Phase 5
   would have silently absorbed without any phase explicitly addressing
   it. ADDED as Risk #9, with an explicit instruction not to let this die
   silently.
5. **Cancellation-by-omission (item 6) is underspecified and leaks
   memory** — rank 1 cannot distinguish "not scheduled this step" from
   "cancelled, free the state" from omission alone. ADDED as Risk #10,
   with the fix (explicit eviction entry + idle/shutdown frame) folded
   directly into the risk description for Section 6.2 item 6 to
   implement against.
6. **No wire-protocol state machine or deadlock analysis exists**,
   which matters specifically because micro-batch interleaving keeps two
   pipeline steps in flight on one physical link at once, combined with
   MLX's lazy evaluation — flagged as needing to be written out
   explicitly BEFORE Phase 1, not discovered empirically while debugging
   a hang. Also flagged an unverified assumption (whether jaccl's `recv`
   can accept a dynamically-shaped tensor from a metadata frame, or
   requires shapes collectively pre-agreed — a potential hidden
   transport-primitive dependency this doc otherwise declares out of
   scope). ADDED as Risk #11.
7. **No KV-cache memory budget exists for concurrent requests at real
   context depth** — Requirement 3 must be validated at 100K-500K
   context, and whether even N=2 concurrent deep-context KV caches fit
   in 128GB alongside resident expert weights was simply never
   calculated. ADDED as Risk #12 and a Pre-Phase-0 check.
8. **Phase 0's original correctness methodology (diff TP batched output
   against PP serial output byte-for-byte) was flagged as methodologically
   wrong** — different sharding schemes produce genuinely different
   float accumulation (all_sum reduction order, quantized-matmul
   partitioning, batched-vs-unbatched kernel differences) independent of
   any real bug, so a cross-sharding byte-diff would either produce
   constant false alarms or force tolerances loose enough to hide real
   bugs. CORRECTED to same-sharding-scheme diff (serial PP vs batched PP)
   with greedy-token or logit-tolerance comparison, in Section 9.
9. **Phase 1 as originally written conflated three distinct new things**
   (metadata-framed transport, the rank-0 scheduler, and batching) into
   one phase, which is risky given the silent-corruption failure mode —
   recommended inserting an isolated transport-only-at-concurrency=1
   step first. ADDED as Phase 0.5 in Section 9.
10. Answered all four of the doc's original open questions — see the
    updated Section 11 above.

**Overall verdict from the review (direct quote preserved):** "the
architecture is reasonable and the correctness-first phasing is the
right instinct, but the doc is stronger on prefill (where you have data)
than decode (where you have none), the 'already resolved' claim should
be retracted to match your own Risk #1, and three cheap pre-Phase-0
checks — decode throughput ceiling, KV memory math, MTP/DSpark
disambiguation — could each invalidate the design and should happen
before any code."

## 13. Pre-Phase-0 check RESULTS (2026-08-04) — TWO OF FOUR CAME BACK
    NEGATIVE. Read this section before doing anything else with this doc.

All four Pre-Phase-0 checks from Section 9 were run. Two are genuinely
concerning findings that go BEYOND what the second review anticipated —
not just "needs validation," but "the assumption behind Requirement 3's
DSpark-gating scope decision (item 7) may be categorically wrong, not
just imprecisely benchmarked." Full arithmetic and citations below.

### 13.1 Decode-throughput ceiling estimate — MARGINAL, not a clean pass

Using warm memory fact 1154 (PP mode, `EXO_SPECULATIVE=0`, no
speculation, c=1, 6 runs): mean single-request PP decode = **24.68
tok/s** (range 22.93-25.72).

- Worst case at c=2 (zero benefit from micro-batch interleaving, pure
  time-sharing): aggregate stays at ~24.68 tok/s.
- Best case at c=2 (perfectly ideal 2-stage bubble-filling, fully hides
  the pipeline idle time): aggregate ≈ 2× single-stream = **49.36
  tok/s**.
- Target: **37.5 tok/s**.
- This means micro-batch interleaving needs to capture roughly **52% of
  its theoretical maximum benefit** to hit the target — not a small
  margin, but not obviously unreachable either. This is a real,
  non-trivial engineering bar for Section 6.2 item 4 to clear, not a
  free win. TREAT AS: proceed, but Phase 3's throughput validation
  (Section 9) is now known to be a real risk point, not a formality —
  if real-world interleaving efficiency lands meaningfully below ~50%,
  Requirement 3 fails even with a working implementation.

### 13.2 MTP/DSpark disambiguation — NEGATIVE. This is the serious one.

Confirmed via direct code read, not assumption:

- Fact 745 (the 37.5 tok/s benchmark) explicitly used **"seq-split
  on."** `EXO_DSV4_SEQ_SPLIT` is a TP-only mechanism (Section 3's own
  citation: it "splits prefill query rows across ranks + `all_gather`s
  the result back" — this concept has no meaning under PP's layer-split
  topology, PP has no query-row splitting to speak of).
- Fact 1014 (this doc's own Section 3 citation, PP architecture)
  states directly: **"MTP speculation disabled (`pp_speculation.py`
  send/recv conflicts with `PipelineLastLayer` handoff)"** under PP mode
  today.
- CONCLUSION: fact 745's exact benchmarked configuration — TP sharding +
  seq-split + MTP — **cannot run under PP as architected today, at
  all, for ANY concurrency level, independent of this design's item 7
  DSpark-gating decision.** This is not "the benchmark used a different
  mechanism than what we're gating off" (Risk #8's original framing,
  which implied a possible but uncertain conflict) — it's "the exact
  benchmarked configuration is TP-specific and structurally
  inapplicable to PP, full stop." Requirement 3's 37.5 tok/s number was
  achieved in a regime (TP + MTP + seq-split) that has NOTHING to do
  with the PP-only DSpark mechanism this design's item 7 discusses
  gating — they are not even the same speculative-decode implementation
  colliding with each other; PP-mode decode speculation, when it works
  at all, goes through DSpark specifically (`pp_dspark_decode_loop` in
  `pp_speculation.py`), a DIFFERENT code path from TP+MTP's mechanism
  (`dsv4_mtp.py`) entirely.
- **IMPLICATION FOR THE DESIGN:** Requirement 3 (37.5 tok/s aggregate at
  c=2) was validated in a sharding mode (TP) this design is explicitly
  NOT using — this design keeps PP's topology throughout (Section 6.1).
  There is no historical measurement anywhere in this fork of PP c=2
  MTP-on decode throughput, because that configuration has never
  existed / has been actively disabled. The 13.1 ceiling estimate above
  (24.68-49.36 tok/s, built from PP's own no-speculation baseline) is
  therefore the ONLY grounded data point this design can lean on for
  Requirement 3 — not fact 745, which measured a different sharding
  scheme's different speculative mechanism. This should be treated as
  a REQUIREMENT-SCOPING QUESTION for the user, not something this doc
  can resolve unilaterally: is Requirement 3 "37.5 tok/s via WHATEVER
  mechanism gets there" (in which case PP-mode's own eventual DSpark
  batching, once/if built, is the actual candidate — not TP+MTP), or is
  it specifically "match what TP+MTP already does" (in which case this
  design's entire premise of staying on PP topology for decode may need
  re-examination, since TP already meets this requirement TODAY with
  zero new engineering)? See Section 14 for the concrete question to
  put to the user.

### 13.3 KV-cache memory budget at depth — MOSTLY FITS, with one real ceiling

Per-node headroom after weights (~85GB, midpoint of measured 77-89GB
range) + runtime overhead (~10GB) on a 128GB node: **~33GB.**

PP shards KV cache BY LAYER across the 2 ranks, so each rank holds
roughly half of any one request's total KV footprint.

Using the ARCHITECTURE-CORRECT/cold-prefill KV rate (14 KB/token, warm
memory facts 639/655/635, directly cited in this fork's own measured
history): at 500K context, one request's KV ≈ 6.68GB total → ~3.34GB
per node. N=2 concurrent requests at 500K: ~6.68GB/node — comfortably
fits. Even N=8 concurrent 500K-context requests (~26.70GB/node) fits
within the ~33GB headroom.

BUT using the REALISTIC GROWING-SESSION rate (45 KB/token, warm memory
fact 650/651 — the rate that ACTUALLY applies to real multi-turn
conversations, not a cold single-shot prefill, and directly relevant
since Requirement 3 must be validated against realistic usage per
Section 2.5 item 3): at 500K context, one request's KV ≈ 21.46GB total
→ ~10.73GB per node. N=2 fits (~21.46GB/node, under the 33GB headroom)
but **N=4 concurrent growing-session requests at 500K context does NOT
fit (~42.92GB/node needed vs ~33GB available).**

CONCLUSION: N=2 concurrent requests at real depth is safe under either
KV-rate assumption — matches this design's Phase 1-3 scope (which is
built around 2 concurrent requests throughout). **RESOLVED by explicit
user confirmation (2026-08-04): the design's real target is N=2
concurrency, not N=4+ — "I wouldn't ever target more than N=2 there."
This removes the N>2 growing-session memory ceiling as an open risk for
this design's actual scope: it's a real, quantified limit if someone
ever tried to push this beyond N=2 at deep context, but that's
explicitly not a goal here.** Phase 4's original framing ("N > 2
concurrent requests") should be read as the LOWEST priority / optional
stretch extension given this scope confirmation, not a required
deliverable — see the updated Phase 4 description in Section 9.

### 13.4 Phase-disaggregation fallback memory math — REFINED, and WORSE
    than the original rough estimate

The original Section 7 estimate (from the first consult, not yet
verified against real per-layer numbers) was "~50% → ~75% per-node
expert-weight residency." Refined using this fork's own measured
per-layer weight distribution (155GB total / 43 layers ≈ 3.60GB/layer,
consistent with the measured ~75.7GB for a ~21-layer PP rank residency
against the actual measured 77-89GB range):

- A PP rank already holds 100% of the weights for its own ~21-22
  layers (~75.7GB, matches measured).
- To ALSO run TP-style decode for its non-owned ~21-22 layers, it would
  need TP's half-expert share for just those layers: ~39.7GB
  incremental.
- **Total dual-layout residency: ~115.3GB per node**, against a
  ~118GB budget (128GB - ~10GB runtime overhead) — leaving only
  **~2.7GB of headroom for KV cache and any margin.** This is
  effectively unusable in practice: 2.7GB is far below even a SINGLE
  request's KV cache at any meaningful context depth (per Section
  13.3's own numbers, even a short/shallow request needs more than
  this once real KV growth is accounted for).
- CONCLUSION: this refined math is WORSE than the original rough
  estimate suggested, not better — phase disaggregation (Section 7) is
  now more confidently ruled out as a fallback than before, not less.
  If Section 13.2's finding forces a real re-think of this design's
  premise, phase disaggregation is NOT a viable fallback to fall back
  on — a genuinely different approach would be needed.

## 14. Concrete question for the user, arising from Section 13.2 — this
    needs an answer before Phase 0 work begins

**Requirement 3 (≥37.5 tok/s aggregate at c=2, MTP on) was validated in
a sharding mode (TP) and mechanism (MTP via `dsv4_mtp.py`) that has
NOTHING to do with this design, which keeps PP topology throughout and
would use PP's own DSpark mechanism (`pp_speculation.py`) if any
speculation runs at all. TP already meets this requirement TODAY, with
zero new engineering, by definition — it's where the number came from.**

This means one of two very different projects is actually being asked
for, and this doc cannot resolve which without the user's input:

**(a) "Give me PP's prefill advantage AND concurrent decode AND
cancellation, at whatever aggregate decode throughput the resulting
system achieves — 37.5 tok/s was just a rough proxy for 'good enough,'
not a literal must-match-TP's-number bar."** If this is the real intent,
this design proceeds mostly as planned, with Section 13.1's 24.68-49.36
tok/s ceiling estimate as the honest expectation range, and the actual
bar re-stated as something like "meaningfully better than PP's current
single-request-only ~24.68 tok/s, ideally approaching TP's 37.5 as a
stretch goal, not a hard gate."

**(b) "I specifically want 37.5+ tok/s aggregate at c=2, and I don't
care whether that comes from PP or TP under the hood — get me PP's
prefill win WITHOUT giving up TP's already-proven decode number."** If
this is the real intent, this design's core premise (keep PP topology
for decode, Section 6.1) needs to be reconsidered — the phase-
disaggregation alternative (Section 7), which was the option that would
have let the design keep TP's decode number directly, is now REJECTED
and PULLED (confirmed infeasible in Section 13.4, ~2.7GB/node headroom
— not a fallback, the user explicitly agreed "dual residency is not
doable at all"). This means option (b), if it's the real intent, has NO
known viable path via a fallback already in this doc — it would need
fresh design work not yet attempted here (e.g. running BOTH shardings
as separate loaded instances and routing prefill-heavy vs decode-heavy
traffic between them at the request level, accepting the cost of
loading the model twice in some form, rather than trying to make one
process serve both regimes).

**This doc's own recommendation, pending the user's answer: option (a)
is more consistent with what's actually achievable given Section 13's
findings, and with the spirit of Section 1's original goal ("PP's
prefill advantage AND TP's concurrent-decode CAPABILITY" — capability,
not necessarily identical throughput). But this is the user's call to
make explicitly, not something to assume silently, given how central
Requirement 3's exact number was to this doc's own Section 2.5.**


