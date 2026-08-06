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

   **Update (2026-08-05): a real MLX/DSv4-specific blocker discovered
   while attempting this.** This fork's existing 2-rank correctness
   harness (`pp_batched_correctness.py`'s `SimPipelineTransport`) uses
   TWO REAL OS THREADS to simulate the two PP ranks in one process
   (needed for the decode-only handoff's genuine send-then-block-on-
   recv dependency). This harness is validated and works correctly for
   plain Llama-proxy models (all of this session's Phase 1 tests) but
   **crashes hard (fatal, non-catchable process crash) when the model
   under test is real DSv4-Flash** — confirmed via a minimal repro: any
   `@mx.compile`'d function with 2+ return values crashes MLX
   ("RuntimeError: no Stream(gpu,0) in current thread" followed by a
   fatal `PyThreadState_Get`/GIL crash) if invoked from any thread
   OTHER than the thread that executed its `@mx.compile` DECORATION.
   DSv4's `HyperConnection` (`hyper_connection.py`'s
   `_hc_split_sinkhorn_ops`, 3 outputs) and at least two more functions
   in `deepseek_v4.py` itself (`_gate_route`, `_split_softmax`, both
   2-output) hit this. `mx.disable_compile()` fixes the ISOLATED
   minimal repro when called before the decorating module is ever
   imported, but did NOT fully resolve the real DSv4 model forward
   cross-thread (still throws the same stream error, non-fatal this
   time, from some remaining thread-bound compiled path not yet
   isolated). Llama has no multi-output compiled functions in its
   forward path and is entirely unaffected.

   This is a TEST-HARNESS-ONLY limitation, not a production concern —
   the real cluster runs each PP rank as a genuinely separate OS
   PROCESS, not simulated threads in one process, so this exact
   thread-affinity class of bug structurally cannot occur there.
   Practical consequence for THIS item: the "dedicated correctness
   test... byte-for-byte at temp=0" this risk calls for cannot be built
   as an in-process simulated-2-rank test the way this fork's other
   Phase 0/0.5/1 correctness tests were — it needs either a genuine
   multi-process harness (heavier to build, more faithful to
   production) or must be deferred to the real 2-node cluster A/B step
   already planned for Phase 1's completion. Decision: stop here rather
   than keep sinking time into isolating the exact remaining thread-
   bound DSv4 mechanism — the workaround space (subprocess ranks vs.
   further compile-patching) is well-understood even though the final
   fix isn't chosen yet.

   **RESOLVED, 2026-08-05 (later the same session):** built the
   genuine multi-process harness flagged above as the real fix.
   `test_pp_batched_decode_subprocess.py` launches two real
   `subprocess.Popen` Python processes connected via MLX's own real
   `ring` distributed backend (localhost TCP loopback -- no cluster or
   RDMA hardware needed; `mlx.launch`'s CLI was avoided since it shells
   out via ssh-like plumbing even for 127.0.0.1 and is fiddly with
   venv/PATH resolution for local dev -- direct `MLX_HOSTFILE`/
   `MLX_RANK` env vars per subprocess work cleanly instead). Confirmed:
   the MLX cross-thread multi-output-`@mx.compile` bug does NOT apply
   across real process boundaries (each process has its own single
   main thread == its own compile-decoration thread). A real (small)
   DSv4-Flash model -- including its `HyperConnection` multi-output
   compiled Sinkhorn function, the exact mechanism that crashed the
   in-process 2-thread harness -- runs a real batched-PP forward pass
   cleanly across two genuinely separate processes with zero errors.
   Two real bugs were found and fixed while BUILDING this harness (not
   in the already-verified session/protocol code): a layer-installation
   ordering bug (prefill must run before batched metaframe layers are
   installed) and a `subprocess.Popen` call that silently dropped its
   constructed `env=` dict. Both tests pass, stable across 3 repeated
   runs. This closes Risk #1's own required correctness test.
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
separately.

**PHASE 0.5 COMPLETE — 2026-08-05, v3 fix confirmed working end-to-end
on the real cluster.** Third real cluster attempt, same-day, with the
user's separate explicit go-ahead: relaunched with
`EXO_PP_METAFRAME=1` + `DSV4_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-0731`
+ the v3 fix. Both runners reached `RunnerReady` on the first attempt
(warmup itself exercises prefill, decode, AND the decode-gather
handoff — the exact code path both prior bugs lived in, so a clean
warmup is real evidence, not a coincidence). Two real generations
confirmed correct end-to-end behavior over the metaframe transport, not
just "the process didn't crash":
- Short: "What is the capital of France? Answer in one word." →
  `content: "Paris"`, `finish_reason: "stop"`, clean single-step decode.
- Longer: "Count from 1 to 10, one number per line." → correct 1-10
  output, 77 completion tokens, clean `finish_reason: "stop"` — a real
  multi-step decode loop (repeatedly exercising the exact forward-hop +
  decode-gather sequence the v3 fix touches) with no BOS-spam, no
  garbage, no hang.

This confirms the metadata-framed transport (Phase 0.5's actual
deliverable) is now a WORKING, validated alternative to today's
ambient-mutable-flag transport, at concurrency=1, on real 2-node RDMA
hardware — not just locally simulated. Phase 0.5 is DONE. The cluster
was left running on this validated `EXO_PP_METAFRAME=1` config after
the successful verification (no reason to revert to `=0`, since this
IS the now-proven-working target state) — see the next session's state
before assuming which transport is currently live.

**Next step: Phase 1** (rank-0 scheduler skeleton, 2 concurrent
decode-only requests) can now build on a metaframe transport that has
been proven correct against the real DSv4 model on real hardware, not
just against a plain-Llama local simulation.

**Phase 1 — Rank-0 scheduler skeleton, decode-only, 2 concurrent
requests, NO speculative decode:** Build the metadata-frame protocol and
the rank-0 scheduler for the SIMPLEST case first — 2 concurrent plain
(no DSpark) decode-only requests (both already prefilled via today's
existing serial PP prefill, just testing the NEW concurrent decode
path). Validate byte-for-byte correctness against 2 serial single-
request PP runs before touching throughput at all. (This phase now adds
ONLY the scheduler+batching delta on top of Phase 0.5's already-verified
transport, per the isolation rationale above.)

**Phase 1, step 1 (wire-protocol state machine + fuzzing) — DONE,
2026-08-05.** Per Risk #11's explicit requirement ("needs to be written
out as an explicit state machine ... BEFORE Phase 1 starts") and
Section 11 open question #1's answer (logic-level fuzzing before Phase
1), built `src/exo/worker/engines/mlx/pp_scheduler_protocol.py` — a
pure, zero-MLX, zero-I/O core (`SchedulerCore` for rank 0's decisions,
`RankOneMirror` as rank 1's independent reactive validator, sharing one
implementation per a `consult` review's guidance) plus
`src/exo/worker/engines/mlx/tests/test_pp_scheduler_protocol.py` (14
directed unit tests + a 2000-seed uniform-random fuzzer + 2 targeted
abort/reuse-race tests + a 1000-seed targeted abort-cycle stress
fuzzer, 24/24 passing, basedpyright/ruff clean). No `hypothesis`
dependency (confirmed absent from this project; a hand-rolled seeded
`random`-module fuzzer used instead — deterministic exact-reproduction
matters more here than automatic shrinking for a state machine this
small).

Directly implements the fixes this doc already specified for Risk #10
(cancellation-by-omission ambiguity — explicit `DRAINING` slot state +
`EvictMessage`/`EvictAckMessage`, not silent omission) and Risk #11
(no deadlock analysis — `RankOneMirror` rejects any illegal message
loudly via `ProtocolViolationError` instead of silently corrupting
cache state or blocking forever). Fail-stop throughout, per the
consult's explicit warning that auto-correction/repair code is where
silent corruption hides — nothing in this module attempts to recover
from a violated invariant.

**The fuzzer caught two real bugs in this module before it was ever
committed** — exactly the outcome Section 11's fuzzing recommendation
was for, not a formality:
1. `RankOneMirror` double-counted the cache-length advance (validated
   the claimed length via raw equality without accounting for
   `n_tokens`, then separately re-added `n_tokens` on top of its own
   tracked state).
2. `SchedulerCore._active_batch_entries` hardcoded `n_tokens=1` for
   EVERY co-listed active request in a step snapshot, not just the
   one whose event actually fired — silently claiming every OTHER
   concurrent request also advanced a token in lockstep. This one only
   surfaced once a fuzzed sequence had more than one concurrently
   active request; the earlier directed single-request-only unit tests
   never exercised the code path where it mattered. Fixed by threading
   through exactly which `request_id` advanced on each `handle()` call.

Both bugs are the exact class the design doc's Risk #5/#11 warned
about — silent cross-request state divergence, not a clean crash — and
both were found and fixed at the pure-logic level, with zero cluster
time spent, before any real scheduler code exists to compound the
debugging surface. This is Section 11's fuzzing recommendation paying
for itself immediately, not a box-ticking exercise.

**Phase 1, step 2 (per-request cache routing) — DONE, 2026-08-05.**
Per Section 6.2 item 3 ("each rank maintains its OWN half of each
in-flight request's KV cache ... needs to become a dict keyed by
request UID instead of a single active cache"), built
`src/exo/worker/engines/mlx/pp_batched_cache_router.py`
(`BatchedCacheRouter`, `merge_request_caches`, `extract_request_cache`)
plus `src/exo/worker/engines/mlx/tests/test_pp_batched_cache_router.py`
(18 tests, all passing, stable across 3 repeated runs,
basedpyright/ruff clean). Reviewed via `consult` before writing code
(2026-08-05) — the reviewer shaped the design directly:

1. **Slot-indexed, matching `SchedulerCore`'s existing slot numbering
   exactly** — not a second independent request-UID-keyed structure
   that could drift out of sync with the protocol layer's own slot
   tracking.
2. **The batched cache IS canonical storage; this router tracks
   METADATA (occupancy/length) only.** Physically merging/extracting
   per-request cache lists on every decode step would be O(total cache
   bytes) per token — flagged by the consult as a real perf mistake to
   design around up front, not discover later. `merge()`/`extract()`
   (mlx-lm's existing, already-proven-by-`prefill_batched` machinery)
   are used only at real boundaries — constructing the initial batched
   cache, and extracting a finished request's cache back out — not on
   every step.
3. **No generation counter needed for the classic ABA slot-reuse
   race** (a corruption vector the consult explicitly flagged as
   likely under-tested by naive designs): `SchedulerCore`'s `DRAINING`
   slot state, from Phase 1 step 1, already structurally prevents a
   new request being assigned to a slot before the prior occupant's
   eviction is acknowledged. This router deliberately trusts that
   single upstream source of truth rather than re-deriving the same
   guarantee a second time, which the consult noted would risk the two
   invariants silently diverging under a future edit.
4. **Reset-on-assign, never trim-on-release** — a released slot's
   stale KV bytes are left in the buffer; every consumer (attention
   mask construction, in particular) must derive visibility strictly
   from tracked length, never physical buffer extent. Accepted
   trade-off, documented not silently incurred: a slot's buffer
   capacity ratchets up to the longest request that ever occupied it
   and never shrinks — bounded, but a real Phase 1 cost.

Verified against REAL mlx-lm cache objects, not just mocked/pure
Python: `KVCache` (plain, Llama-style) and DSv4's actual
`CacheList(RotatingKVCache, PoolingCache)` structure, at both matched
and DELIBERATELY HETEROGENEOUS per-request lengths (mixed
prefill/decode-progress state — the realistic Phase 1 scenario), with
explicit data-content assertions (not just shape checks) confirming
request A's tokens never leak into request B's extracted cache after a
merge/extract round-trip — a real check of the design doc's own Risk
#5 (silent cross-request corruption), not a shape-only smoke test.

**Phase 1, step 3 (batched-decode layers + real correctness
checkpoint) — DONE, 2026-08-05.** Built
`src/exo/worker/engines/mlx/pp_batched_decode_layers.py`
(`BatchedMetaFramedPipelineFirstLayer`,
`BatchedMetaFramedPipelineLastLayer`, `BatchStepContext`,
`batch_step_scope`) — new, separate classes serving N=2 concurrent
decode-only requests through ONE layer instance across different
calls, using the metaframe protocol's new `batch_axis=1` stacking
(step 3's prerequisite, also this session). Never touches Phase 0.5's
already-cluster-verified `MetaFramedPipelineFirstLayer`/`LastLayer`.

Design question this step answers: how does a single layer instance
serve DIFFERENT request sets on different calls without falling back
to the ambient-mutable-instance-flag anti-pattern
(`is_prefill`/`queue_sends`) `pp_metaframe.py` was explicitly built to
eliminate? Resolved via two `consult` reviews: per-call context is
required (instance flags fail precisely in the case PP creates —
multiple steps potentially in flight through one layer instance); a
`contextvars.ContextVar` (`BatchStepContext`) scoped by a context
manager around exactly one `model(...)` call per step is the right
mechanism here (not embedding context in a cache-slot object, the
alternative for callers who don't own the forward loop — this fork
does own it, and the lifetime match is exact: per-step batch
composition is per-CALL data, not per-REQUEST persistent state that
goes stale as batch membership changes across steps); no default
value on the ContextVar (a scoping bug fails loudly, never silently
routes to the wrong/empty request set); explicit ordering/identity
assertions in both layers (nothing else structurally ties context
ordering to actual batch-tensor row order — a mismatch now fails
loudly instead of silently swapping tokens between requests).

**Real bug #1** (found by this step's new correctness test, not by
`pp_batched_cache_router.py`'s existing 18 unit tests, which never
crossed a thread boundary): `merge_request_caches`'s underlying
`merge()` builds a lazy MLX graph not materialized until evaluated;
handing an un-eval'd merged cache to a DIFFERENT thread than the one
that built it (this fork's own 2-rank correctness-harness pattern)
raises `RuntimeError: There is no Stream(gpu, N) in current thread.`
Fixed by force-evaluating the merged cache's full `.state` before
`merge_request_caches` returns, closing the hazard structurally rather
than requiring every future caller to remember it.

**Real bug #2** (found via the same test, in the test harness itself):
`BatchKVCache.offset`/`left_padding` are lazy scalars advanced via
`+=` each call — not necessarily on the model output's own dependency
graph (output is computed from the PRE-increment offset). A harness
that spawns a brand-new thread per decode step needs the cache's OWN
state force-eval'd before the thread exits, or the next step's thread
inherits lazy graph nodes still bound to a dead thread's MLX stream
context. Both bugs are the SAME MLX-thread-interaction hazard class
`pp_batched_correctness.py`'s module docstring already flagged (its
own `mx.eval(tokens)`-before-dispatch fix) recurring in a new spot
(cache state, not input tokens) — not a new discovery, now documented
at both sites.

**THE real checkpoint**
(`test_pp_batched_decode_correctness.py`, 3/3 passing, stable across 3
repeated runs, basedpyright/ruff clean): 2 concurrent decode-only
requests through the FULL new scheduler-adjacent stack (batched
metaframe transport + `BatchStepContext` + cache router's real
merge/extract) via simulated 2-rank PP, compared against 2 SERIAL
single-request PLAIN (unsharded) forward passes over 5 real decode
steps each — greedy tokens match EXACTLY. Golden reference is the
plain forward, not another PP path, per this fork's own established
Phase 0 methodology (`pp_batched_correctness.py`'s module docstring
point 2). Also covers the degenerate N=1 batch-of-one case and a
direct unit test of the context-mismatch guard.

**Phase 1, step 4 (scheduler/cache-router driver glue + full-stack
lifecycle checkpoint) — DONE, 2026-08-05.** Built
`src/exo/worker/engines/mlx/pp_batched_decode_driver.py`
(`BatchedDecodeDriver` for rank 0, `RankOneMirrorDriver` for rank 1) --
the actual thing wiring `SchedulerCore`/`RankOneMirror` and
`BatchedCacheRouter` into what a real decode loop calls
(`admit_request`/`on_tokens_generated`/`evict_request`/`on_evict_ack`
on rank 0; `on_step_message`/`on_evict_message` on rank 1). Reviewed
via `consult` before writing code: the risk was introducing a SECOND,
driftable encoding of batch composition (one inside the driver, one
inside the metaframe/`BatchStepContext` plumbing). Resolved by making
`StepMessage.entries` the single source of truth --
`batch_step_context_from_step_message` is the ONE function that
derives a `BatchStepContext` from a `StepMessage`, used identically by
both rank 0's and rank 1's driver classes, never re-derived
independently by either side. Per the same review:
`RankOneMirrorDriver` contains ZERO decision logic -- it only
validates rank 0's claims and mirrors bookkeeping to match.

A real gap surfaced while starting this wiring: `TokenGeneratedEvent`
was strictly single-request, but a real batched decode step advances
N=2 requests SIMULTANEOUSLY in one model forward call -- fixed (via a
separate `consult` review) by generalizing the event to carry
`request_ids: tuple[int, ...]` rather than adding a parallel event
type, since splitting a real batched step into N single-request events
would make `RankOneMirror` pass through intermediate states that never
existed on rank 0 -- exactly the core/mirror divergence surface this
module's fail-stop design exists to prevent. `_handle_tokens_generated`
validates ALL request_ids before mutating ANY state, so a bad id
anywhere in a batch can't leave some requests advanced and others not.
27/27 tests passing (24 existing + 3 new) on `pp_scheduler_protocol.py`
after this change, 12/12 new tests on the driver itself.

**The final local-verification checkpoint**
(`test_pp_batched_decode_driver_full_stack.py`, 1/1 passing, stable
across 3 repeated runs, basedpyright/ruff clean): the full request
lifecycle -- two requests admitted, both decode together for 4 real
batched steps (real batch of 2, all composition decided by the actual
driver, not a hand-picked tuple), one is evicted (the real
DRAINING/ack cycle through `evict_request`/`on_evict_ack`/
`on_evict_message`), the other continues decoding alone for 2 more
steps (real batch of 1 -- the "some slots active, others empty"
mixed-step case) -- with greedy tokens for both requests matching
their serial plain-forward golden references across the ENTIRE
lifecycle, not just steady-state batching. This is the complete answer
to what `test_pp_batched_decode_correctness.py` (Phase 1 step 3) had
deliberately left open: every piece built this session -- protocol
state machine, cache router, batched metaframe layers, and now the
scheduler/cache-router driver glue -- verified together, end to end,
at the CPU/simulated-transport level.

**Phase 1, step 5 (mid-stream admission verified) — DONE, 2026-08-05.**
The full-stack test above (step 4) only covered requests admitted
UPFRONT (both before any decode step) -- the realistic case of a NEW
request joining an ALREADY-IN-PROGRESS batch (one request has been
decoding for a while, nonzero cache offset) was left open. Verified
via a throwaway repro before writing any test code (per this session's
established "test the primitive in isolation before building on it"
habit): `mlx-lm`'s own `BatchKVCache.merge` already handles this
heterogeneous-offset case correctly -- no new primitive needed.

Added `test_merge_supports_mid_stream_admission_advanced_plus_fresh`
(`test_pp_batched_cache_router.py`) confirming the cache-merge
PRIMITIVE handles an advanced (nonzero-offset) request merged with a
fresh (offset=0) one, and
`test_mid_stream_admission_matches_serial_plain_forwards`
(`test_pp_batched_decode_driver_full_stack.py`) confirming the actual
attention math produces CORRECT results from this shape, end to end,
through the real driver: request A decodes solo for 2 real steps
(genuinely advancing its cache), THEN request B is admitted mid-stream
and merges into the same in-progress batch, both continue decoding
together for 2 more real steps -- greedy tokens for both requests
match their serial plain-forward golden references across the whole
lifecycle. 21/21 tests passing across both files, stable across 3
repeated runs, basedpyright/ruff clean.

This closes the LAST open question in Phase 1's local verification:
admission (upfront and mid-stream), steady-state batched decode,
eviction/slot-release, and solo continuation are all now proven
correct at the CPU/simulated-transport level, through the real
scheduler/cache-router/metaframe stack, not hand-picked test
shortcuts.

Not yet done at the time of step 5: the real 2-node cluster A/B for
this batched path (everything above is simulated-2-rank, proven
correct at the CPU/logic level per this fork's established Phase 0
rationale for keeping GPU/cluster time off correctness questions the
CPU can answer just as definitively).

**Phase 1, step 6 (real decode-loop runtime session) — DONE,
2026-08-05.** Everything built through step 5 was pure scheduling/
transport glue (`SchedulerCore`/`RankOneMirror` protocol,
`BatchedCacheRouter`, `BatchedMetaFramedPipeline*Layer`,
`BatchedDecodeDriver`/`RankOneMirrorDriver`) -- nothing yet DROVE an
actual `model(...)` call with real per-request sampling, generation
state, and eviction as a usable session object. Built
`pp_batched_decode_runtime.py`: `BatchedDecodeSession` (rank 0) owns
per-request generation state (next_token, sampler, done flag) OUTSIDE
the driver (which only owns protocol/cache-slot bookkeeping, never
generation policy), and a three-phase `prepare_step`/`run_forward`/
`finish_step` API -- split out from a single `step()` call
SPECIFICALLY so a real 2-rank caller can hand rank 1 the `StepMessage`
BEFORE either rank's forward pass starts, matching the real
transport's actual ordering constraint (this also eliminated a race
that an earlier single-call design hit in the 2-thread simulated-
transport test harness, which needed a busy-poll workaround before the
split; the split removes the race structurally instead of masking it).
`RankOneMirrorSession` is the symmetric rank-1 side: zero sampling,
zero generation-policy decisions, only validation + its own half of
the identical forward pass.

Verified via `test_pp_batched_decode_runtime.py` (3 new tests): the
full multi-step lifecycle (admit both, batch decode, evict one,
continue solo) through the REAL session API end to end, matching
serial plain-forward golden references exactly -- plus a single-rank
convenience-wrapper test and a `batch_step_scope` regression guard
confirming `run_forward` really does activate the SAME
`BatchStepContext` `prepare_step` computed. 3/3 passing, stable across
3 repeated runs, basedpyright/ruff clean. Full worker suite: 211
passing, same 1 pre-existing unrelated failure, 0 new regressions.

**Phase 1, step 7 (real wire encoding for scheduler control messages)
— DONE, 2026-08-05.** Everything through step 6 used in-process
Python object sharing for `StepMessage`/`EvictMessage`/
`EvictAckMessage` between the two simulated-thread "ranks" -- fine for
a same-process test harness, but a real 2-process deployment needs
these control messages to actually cross the wire. Built
`pp_scheduler_wire.py`: real `mx.distributed.send`/`recv_like`-based
encoding for all three message kinds, unified under one fixed 5-field
header (a real design bug was found and fixed here -- the first
attempt gave each kind a different header shape, so a kind mismatch
crashed the transport with a raw shape error instead of a clean
`SchedulerWireProtocolError`; fixed by matching `pp_metaframe.py`'s
own fixed-header discipline). Per a `consult` review: the cache_slot
mapping is the dangerous part, not the uid list -- this carries
`BatchEntry`'s full field set (request_id, cache_slot, phase,
expected_cache_len, n_tokens), not just the bare uids
`encode_batched_decode_metaframe` already carries. Verified via 11
tests over the real 2-thread `SimPipelineTransport`, including the
actual production header-first dispatch pattern (receive the header
with no prior kind knowledge, branch on `msg_kind`) handling all three
kinds correctly in sequence. 11/11 passing, basedpyright/ruff clean,
full suite 222 passing with the same 1 pre-existing unrelated failure.

**Phase 1, step 8 (real-wire integration test found a real bug) —
DONE, 2026-08-05.** Built
`test_pp_batched_decode_over_real_wire.py`: the SAME full-lifecycle
correctness test as step 6's, but with EVERY control message
genuinely crossing `pp_scheduler_wire.py`'s real transport instead of
being shared in-process as a live Python object -- rank 1 reconstructs
each `StepMessage`/`EvictMessage`/`EvictAckMessage` purely from wire
bytes, matching a real 2-process deployment's actual constraint. This
immediately found a REAL, previously-hidden bug: `RankOneMirrorSession`
held a separate `_slot_caches` dict, populated once at `admit_request`
and never refreshed as `step()` advanced the real `batched_cache`.
Eviction's `release_slot` rebuilt `batched_cache` from this STALE
dict, silently reverting the surviving request's cache to its state
at admission time instead of its actual current advanced state --
confirmed via `git stash` A/B that the new test fails identically on
the pre-fix code and passes on the fix. **The in-process test suite
from step 6 never caught this** -- sharing `StepMessage` as a live
Python object happened to never exercise the buggy bookkeeping path,
exactly the class of bug this step was built to surface. Fixed by
removing the separate dict entirely: `RankOneMirrorSession` now always
extracts current per-slot state from the live `batched_cache` on
demand (mirrors `BatchedDecodeSession`'s own established
`_extract_all_current_slot_caches` pattern) -- one source of truth,
never a second copy that can drift. Verified: real-wire test passes,
all in-process tests still pass, stable across 3 repeated runs,
basedpyright/ruff clean, full suite 223 passing with the same 1
pre-existing unrelated failure.

**Phase 1, step 9 (real 2-process harness, closes Risk #1) — DONE,
2026-08-05.** Built `test_pp_batched_decode_subprocess.py` +
`_pp_subprocess_worker.py`: a genuine 2-OS-process correctness
harness using MLX's real `ring` distributed backend over localhost
TCP (no cluster/RDMA hardware needed). This closes design doc
Section 8 Risk #1's own required "dedicated correctness test... at
temp=0" for real DSv4-Flash -- a real (small) DSv4-Flash model,
including the exact `HyperConnection` multi-output `@mx.compile`d
function that crashed this session's in-process 2-thread harness
(Section 8 Risk #1's earlier update), runs a real batched-PP forward
pass cleanly across two genuinely separate processes with zero
errors. Also serves the integration-risk purpose a `consult` review
flagged when scoping the next production-wiring step: a real process
boundary is the one thing an in-process/threaded test suite
structurally cannot exercise. Two real bugs were found and fixed
while BUILDING the harness itself (layer-installation ordering;
`subprocess.Popen` silently dropping its `env=` dict) -- not in the
already-verified session/protocol code, which passed on the first
correctly-configured run. Both tests pass, stable across 3 repeated
runs, basedpyright/ruff clean, marked `@pytest.mark.slow` (excluded
from the default fast suite, matching this project's convention).

Still NOT wired into `mlx_generate`/`stream_generate`'s real request-
admission path (`ExoBatchGenerator`'s async task queue) -- that is a
separate, larger integration surface (touching request routing/
concurrency control at the API-server boundary, not just the MLX
forward-pass mechanics this session's modules own) and is the
concrete next step before a real cluster A/B of the BATCHED path
specifically becomes possible (Phase 0.5's transport-only A/B didn't
need this since it stayed at concurrency=1 throughout). Per the same
`consult` review: this wiring should land behind an opt-in flag
(default off, zero change to the existing single-request path),
proven via runner-level tests targeting the actual call sites (not
re-verifying the session, which this step and step 8 already did),
with any real-cluster behavior validation explicitly deferred to the
separate cluster A/B step.

**`ExoBatchGenerator` feature-interaction matrix (2026-08-05, per a
`consult` review before attempting the wiring) -- required reading
before touching `ExoBatchGenerator`'s ~3500 lines of hardened
production code.** `ExoBatchGenerator` has accumulated substantial
correctness hardening this session's batched-decode session has NOT
been tested against (all of this session's testing exercises the
session/protocol/transport in isolation or paired with a plain model,
never alongside these production features). Per-feature status:

- **MTP / speculative decode (DSv4 self-speculative, DSpark) —
  KNOWN INCOMPATIBLE, not yet redesigned for.** MTP/DSpark accept a
  VARIABLE number of draft tokens per real step (γ-token batched
  verify, not always advancing by exactly 1). This session's
  `BatchedDecodeSession`/scheduler protocol assumes every active slot
  advances by EXACTLY 1 token per step (`TokenGeneratedEvent`'s
  `request_ids` tuple all treated identically, `n_tokens=1` baked into
  `BatchEntry` construction throughout). Running MTP/DSpark under
  batched decode as built would desynchronize the two slots' cache
  positions the very first time their accepted-token counts diverge.
  Design doc Section 6.2 item 7 and Phase 4 (above) already scope
  DSpark to concurrency=1-only for exactly this reason -- this is a
  confirmed, already-anticipated limitation, not a newly discovered
  gap, but worth restating explicitly here since it's the single
  most likely thing an integrator reaches for first (MTP is the
  default speculative path when `EXO_SPECULATIVE=1`).
- **Cancellation mid-batch (one slot cancelled, other continues) —
  PARTIALLY VERIFIED.** The eviction protocol (`evict_request`/
  `on_evict_ack`, DRAINING state) is real and tested end-to-end,
  INCLUDING the asymmetric case (`test_full_lifecycle_over_real_wire_matches_serial_plain_forwards`:
  A evicted, B continues solo with a different remaining length) --
  this is the mechanism a real cancellation would drive. NOT verified:
  a cancellation arriving from `ExoBatchGenerator`'s existing
  degeneration-kill-switch / stop-sequence / client-disconnect paths,
  which are today's SINGLE-REQUEST cancellation surfaces and have
  never been exercised against this session's eviction protocol.
- **Per-slot streaming/detokenization state (partial-UTF-8 buffers,
  stop-sequence match windows, tool-call parser state) — NOT
  VERIFIED, believed compatible by construction.** This session's
  session classes only produce raw token ids per request per step
  (`finish_step`'s `{request_id: (new_token, is_done)}`); they never
  touch detokenization, stop-sequence matching, or tool parsing.
  `ExoBatchGenerator`'s existing per-task `GeneratorQueue`/
  `output_generator` plumbing (already keyed by uid, matching this
  session's `request_id` keying) SHOULD compose cleanly since neither
  side shares mutable state with the other -- but this has never
  actually been run together.
- **Prefix-cache reuse (`KVPrefixCache`) — NOT VERIFIED, likely
  requires new logic.** `KVPrefixCache` assumes a single serial
  per-request cache lifecycle (snapshot/restore keyed by trie
  position); `BatchedCacheRouter`'s slot-indexed batched cache has a
  different lifecycle (merge-into-batch, extract-on-evict). No
  analysis has been done on whether/how a prefix-cache HIT could feed
  into `BatchedDecodeSession.admit_request`'s `prefilled_cache`
  parameter, or whether the batched cache's slot-reuse (DRAINING ->
  FREE -> reassigned) is compatible with the prefix-cache's own
  snapshot bookkeeping.
- **Vision / tool calling / reasoning-budget-limiter / loop-detection
  kill-switches — NOT VERIFIED, believed independent.** These operate
  on already-detokenized text/response objects downstream of raw
  token generation and don't touch cache/scheduling state directly,
  so they're LIKELY compatible by construction (same reasoning as
  streaming state above) -- but, again, never actually run together.

**Recommended integration shape, unblocked by the above:** gate on
BOTH the opt-in flag AND per-request feature detection -- any request
using vision, tools, MTP/DSpark, or hitting a prefix-cache HIT falls
back to today's existing single-request path automatically (not a
routing decision the integrator has to invent case-by-case); only
DECODE-ONLY, no-speculation, no-vision, no-tools, cold-cache requests
are eligible for the batched path. This matches Phase 1's own already-
confirmed scope (decode-only, no speculative decode) and turns "does
this interact badly" into "was this request eligible," which is a
cheap, loud, testable gate rather than a silent correctness risk.

**Building blocks for this integration completed, 2026-08-05** (per
the above shape, `consult`-reviewed):

- `pp_batched_decode_eligibility.py` -- `is_eligible_for_batched_decode`,
  the pure gate function itself. 13 tests, basedpyright/ruff clean.
- `pp_batched_decode_adapter.py` -- `BatchedDecodeResponseAdapter`,
  translating `BatchedDecodeSession`'s raw `{request_id: (token,
  is_done)}` step output into `finish_reason` classification
  (None/stop/length), mirroring `_step_pp_spec`'s own EOS-membership-
  test/max_tokens logic (not reimplementing it independently -- a
  second `consult` review's explicit requirement, to prevent silent
  drift between the two paths). 7 tests using real Llama forward
  passes, basedpyright/ruff clean.

**Actual `ExoBatchGenerator.submit()`/`step()` dispatch wiring —
DELIBERATELY NOT DONE, 2026-08-05, after a third `consult` review.**
This is the one piece of Phase 1's remaining scope this session
concluded should NOT be attempted blind. The critical fact that
changed the risk calculus while investigating this: **the actual
concurrency admission gate is `EXO_MAX_CONCURRENT_REQUESTS=1`,
hardcoded for Pipeline mode in `start_cluster.sh` as a documented
CORRECTNESS fix** ("PP's shared per-rank decode-loop state cannot
survive >1 concurrent request without data corruption/wedging" --
`start_cluster.sh`'s own 2026-07-19 comment, a real production
incident). Wiring `BatchedDecodeSession` into `ExoBatchGenerator`
alone does NOT enable N=2 concurrency; that separate gate would ALSO
need to change, which is a bigger, safety-relevant edit this session
never scoped or attempted.

But the deeper reason to stop here, surfaced by the third `consult`
review directly: **`submit()`/`step()` are the same functions EVERY
request already goes through today, including today's normal
batch=1/serial traffic.** Even a flag-gated new branch inside those
functions changes their control flow, and this session has zero
ability to execute `ExoBatchGenerator` end-to-end (no loaded model,
no real runner plumbing, no cluster this stretch) to verify the
existing single-request path is unperturbed by the edit. Per that
review: "stopping is defensible only after (a) a real-path batch-
size-1 smoke test and (b) a mocked two-request interleaving test" --
neither of which this session could produce without either cluster
access or building a much larger mocked-runner harness, which was
judged out of scope for this stretch. The two completed building
blocks above (eligibility gate, response adapter) are the genuinely
safe, fully independently-testable pieces; the actual `submit()`/
`step()` edit is where "component-tested" stops being sufficient
justification and real end-to-end verification becomes mandatory.

**For whoever picks this up next:** the remaining wiring is
mechanical (mirror `_submit_pp_spec`/`_step_pp_spec`/
`_close_pp_spec_gen`'s existing three-method shape exactly, dispatch
via a new `self._batched_decode_active` flag alongside the existing
`self._pp_spec_active` check in `submit()`/`step()`, construct
`GenerationBatch.Response(uid=..., token=..., logprobs=mx.zeros(1),
finish_reason=..., prompt_cache=None, all_tokens=None)` from the
adapter's classification -- matching `_step_pp_spec`'s own Response
construction exactly). What is NOT yet done and must happen before or
during that edit: (1) the `EXO_MAX_CONCURRENT_REQUESTS=1` gate in
`start_cluster.sh` needs its own explicit, separately-reviewed change
to ever let a 2nd request reach this code; (2) a real-path batch=1
smoke test proving the flag-off (or flag-on-but-single-request) case
is byte-identical to today's existing behavior; (3) ideally a mocked
2-request interleaving test at the `ExoBatchGenerator` level itself,
not just at the `BatchedDecodeSession` level below it.

**Update, 2026-08-05 (later the same day, after explicit user go-ahead
to attempt the dispatch wiring anyway):** Built the two remaining
prerequisite pieces the earlier attempt didn't yet have --
`install_batched_decode_pipeline_layers` (a real, load-time,
`EXO_PP_METAFRAME`-pattern-mirroring function that installs
`BatchedMetaFramedPipelineFirstLayer`/`LastLayer` onto an already-
sharded model -- closes a real gap: without this, the batched layers
only ever existed in test harnesses, never on a real loaded model) and
`get_batched_pipeline_info` (mirrors `pp_speculation.get_pipeline_info`'s
exact contract, detecting the BATCHED layers specifically so a caller
doesn't conflate them with Phase 0.5's single-request metaframe layers
or the legacy PP layers). Both gated behind a new `EXO_PP_BATCHED_DECODE`
flag (default off, requires `EXO_PP_METAFRAME=1` as a prerequisite),
both fully tested (12 tests total across two files, real Llama models,
zero mocks except for the two `mx.distributed.*` passthrough patches a
single-rank test setup genuinely needs), basedpyright/ruff clean, zero
new regressions.

Then attempted the actual `submit()`/`step()` dispatch and stopped
again -- this time for a DIFFERENT, deeper reason than the admission-
gate/unverified-hot-path concern above (both still fully apply too). A
fourth `consult` review, asked specifically about the mechanics of
admitting a SECOND request mid-decode under real PP lockstep,
surfaced a genuine design gap this session had not yet worked through:
**rank 0 and rank 1 both run the identical `ExoBatchGenerator` class,
symmetrically** (already true today, per the existing PP-spec code's
own `self.group.rank()`-driven branching) -- but `BatchedDecodeSession`
(rank 0, decides) and `RankOneMirrorSession` (rank 1, mirrors) must
NEVER independently decide admission from their own local view. If
each rank's `submit()` call independently evaluated the eligibility
gate and decided to admit, a race between the two ranks' local
decisions (e.g. one rank's request arrives fractionally before the
other's, or the gate evaluates a race-prone condition like a prefix-
cache hit that could differ transiently between ranks) could produce
MISMATCHED batch composition between rank 0 and rank 1's own driver
state -- exactly the class of bug this whole session's wire protocol
(`pp_scheduler_wire.py`) and `StepMessage.entries`-as-single-source-of-
truth design were built to prevent for the DECODE step, but which
`submit()`'s admission path has no equivalent mechanism for yet.
Rank 0 must decide admission and broadcast that decision (a REAL wire
message, not implicit agreement from both ranks running the same
code on the same inputs) for rank 1 to replay identically; nothing in
this session's design covers what that broadcast looks like at
`submit()` time (as opposed to `prepare_step()`/`finish_step()` time,
which the wire protocol already covers correctly). Designing that
admission-broadcast mechanism, and verifying it doesn't reintroduce
the exact "two ranks independently reconfiguring shared wire-link
state" bug class that `PPSpecAlreadyActiveError`'s own docstring
describes as a real, already-fixed 2026-07-20 incident for the
speculative-decode path, is real, unstarted design work -- not
mechanical wiring.

**CORRECTION, 2026-08-05 (same day, later still):** the above
"need a new admission-broadcast mechanism" conclusion was WRONG --
re-tracing this session's OWN already-built code (not new
investigation) found the mechanism already exists and was simply not
recalled at the point of the third `consult` review:
`BatchedDecodeDriver.admit_request`/`RankOneMirrorDriver` (built and
tested earlier this session, `pp_batched_decode_driver.py`) already
implement exactly "rank 0 decides admission, encodes it into a real
`StepMessage` (a cache slot transitioning from FREE to occupied,
detected reactively by `RankOneMirror.validate_step`), rank 1 mirrors
it via the real wire (`pp_scheduler_wire.py`)" -- the SAME mechanism
already proven correct for per-step batch composition changes
(admission is not a structurally different operation from a slot
gaining/losing an entry between steps, which this session's tests
already cover). No new protocol design is needed for the wire-level
admission decision itself.

The other real concern from the earlier analysis -- whether
`is_eligible_for_batched_decode`'s inputs are guaranteed
cross-rank-identical -- also resolves cleanly: `submit()` ALREADY has
a real, production-proven cross-rank agreement mechanism for the one
genuinely rank-local input in the gate (`is_prefix_cache_hit`, since
each rank's own `KVPrefixCache` is a local, independently-populated
structure) -- `pipeline_agree_prefix_hit_length()`
(`utils_mlx.py`, used today under `EXO_PP_NO_COORD_COLLECTIVE=1`, the
standard PP launch config) runs BEFORE the eligibility check would
need it, agreeing every rank to the SAME hit-length via the real p2p
wire protocol ("unanimous or cold": any mismatch collapses to
hit_length=0 on every rank identically). Every other gate input
(`has_images`, `has_tools`, `uses_speculative_decode`,
`sharding_is_pipeline`, `batched_decode_enabled`) is derived from
`task_params`/config/env, which every rank already receives
identically via the master's own event-sourced broadcast
(`GLOBAL_EVENTS`, per `AGENTS.md`'s architecture: "Master indexes events and
broadcasts; workers apply indexed events" -- confirmed via
`worker/main.py`'s `_event_applier`/`_start_runner_task`, both ranks'
runners receive the SAME `TextGeneration` task object from the SAME
ordered event log). So the eligibility gate, evaluated identically and
independently on both ranks against inputs that are ALREADY guaranteed
identical by existing infrastructure, produces the same admission
verdict on both ranks without needing its own separate broadcast --
this satisfies the "input determinism" requirement a `consult` review
flagged as the correct test (as opposed to a NEW broadcast of the
verdict itself, which would be redundant given the wire-level
admission mechanism above already exists).

Net effect of this correction: prerequisite item 2 below is CLOSED --
no new design work needed, only the same mechanical wiring the
original (pre-correction) plan already described. The real remaining
blockers are items 1, 3, and 4 below (admission-gate change,
real-path verification, interleaving test) -- none of which this
session has cluster access or a mocked-runner harness to close safely
without an explicit new push in that direction.

**Full standing list of prerequisites for whoever attempts the
`ExoBatchGenerator` dispatch wiring next**, updated:
1. `EXO_MAX_CONCURRENT_REQUESTS=1` (`start_cluster.sh`) needs its own
   explicit, separately-reviewed change -- the batched path is
   unreachable dead code without it.
2. ~~A rank-0-decides/rank-1-replays admission-broadcast mechanism~~
   CLOSED (see correction above) -- `BatchedDecodeDriver`/
   `RankOneMirrorDriver`/`pp_scheduler_wire.py` already provide this;
   the eligibility gate's inputs are already cross-rank-identical via
   existing infrastructure (`pipeline_agree_prefix_hit_length` +
   the master's event-sourced task broadcast). No new design needed.
3. A real-path batch=1 smoke test proving the flag-off (or flag-on-
   but-single-request) case is byte-identical to today's existing
   behavior.
4. ~~A mocked (or, better, real 2-process...) 2-request interleaving
   test at the `ExoBatchGenerator` level~~ CLOSED, 2026-08-05 (same
   day, later still) -- see below.

**Item 4 CLOSED: `pp_batched_decode_glue.py`
(`Rank0BatchedDecodeGlue`/`Rank1BatchedDecodeGlue`) + a real 2-process
lifecycle test.** Attempting the actual mechanical `submit()`/`step()`
edit surfaced one MORE genuine design requirement beyond the
already-closed admission-broadcast question: `submit()` must NEVER
perform synchronous cross-rank wire I/O directly. Per a `consult`
review: doing so would create a SECOND independent writer racing the
existing decode-step loop's own wire traffic -- the classic
multi-writer collective-ordering hazard, and since `submit()` is
shared by every request (not just batched ones), a hang here would
wedge the whole rank, not just the new path. The review's recommended
fix, now built: a strict single-writer PIGGYBACK pattern.
`enqueue_admission()` (the only thing `submit()` ever calls) is pure
in-memory queueing with zero wire I/O -- cannot hang, cannot race.
`tick()` (the only thing `step()` ever calls, from the exact same
single call site `_step_pp_spec` already uses) is the ONLY place that
ever touches the wire for this session, and does at most ONE of
{admit one pending request, run one decode step} per call -- the same
single-writer discipline `PPSpecAlreadyActiveError`'s own entry guard
already enforces for its shared wire-link state, applied here.

Rank 1's admission detection reuses `RankOneMirrorDriver`'s
ALREADY-BUILT reactive mechanism (a `cache_slot` transitioning
FREE-to-occupied within a normal `StepMessage`) rather than a new
message kind -- slot-reuse ambiguity is structurally impossible per
`SchedulerCore`'s own DRAINING-until-ack invariant (verified earlier
this session), so "newly occupied" can only ever mean "genuinely new
request." Rank 1's own prefilled cache for a to-be-admitted request
is staged LOCALLY (`stage_local_cache`) and never crosses the wire --
only rank 0's admission decision does.

Verified with a real 2-PROCESS test
(`test_pp_batched_decode_glue_subprocess.py`, extending this
session's established subprocess harness): a full `submit()`/
`step()`-SHAPED lifecycle across two genuine OS processes -- enqueue
request A upfront, tick to admit, tick to decode solo, enqueue
request B MID-STREAM (while A is already decoding), tick to admit B
alongside A's ongoing decode, tick both together, `complete_request`
to evict A via a REAL `EvictMessage`/`EvictAckMessage` round-trip,
tick B solo to completion -- matching two independent serial
plain-forward golden references exactly. 1/1 passing, stable across 3
repeated runs, basedpyright/ruff clean, full worker suite 254 passing
with the same 1 pre-existing unrelated failure.

**Item 4's harness itself doubles as the "mocked 2-request
interleaving test" this prerequisite originally called for** -- it is
real (not mocked) at every layer except the runner/`ExoBatchGenerator`
wrapper itself, which is exactly the piece the next step (below) adds.

Item 1 (`EXO_MAX_CONCURRENT_REQUESTS=1`) and item 3 (batch=1 smoke
test against the real `ExoBatchGenerator` wrapper) remain the only
open prerequisites.

**Item 3 CLOSED -- `ExoBatchGenerator.submit()`/`step()`
wiring itself DONE, 2026-08-05 (same day, later still).** The actual
mechanical edit landed: `submit()` gains one new branch point
(mirrors `_pp_spec_active`'s own single branch) running the
eligibility gate and, if eligible, calling `_submit_batched_decode`
(pure `enqueue_admission` + the SAME `_active_tasks`/`_EngineTask`
bookkeeping every other path uses); `step()` gains one new branch
(`_step_batched_decode`) calling `tick()` and translating the result
into real `GenerationBatch.Response` objects -- the exact same
contract `_step_pp_spec` already returns. Construction happens once
in `__post_init__`, detecting rank via `get_batched_pipeline_info()`
and building the rank-appropriate glue -- mutually exclusive with
PP-spec by construction (different layer classes).

Verified inert when the flag is off (the only state this can ship
in right now) via a REAL git-worktree A/B run: the pre-edit commit
and the edited tree, run side-by-side against the identical
2-request sequential-submit scenario through a real
`ExoBatchGenerator` instance, produced byte-for-byte IDENTICAL
tokens for every single generated token across both requests -- the
only diff in the JSON trace was the new `_batched_decode_active`
attribute existing at all (expected, doesn't exist pre-edit). This
is the strongest form of the "prove flag-off changes nothing"
guarantee this session's `consult` reviews called for: not a
hand-reconstructed independent reference (two earlier drafts of the
committed regression test got mlx-lm's internal
`insert()`/`prefill()` token-consumption accounting wrong trying
exactly that), but a genuine differential comparison against the
real, unedited code.

`basedpyright`/`ruff` both matched origin/main's pre-edit baseline
error count exactly (0 new errors either tool). Full worker suite:
255 passing (+1 new committed regression test), same 1 pre-existing
unrelated failure, 0 regressions.

**Remaining before this path is reachable/safe to actually enable in
production:**
1. ~~`EXO_MAX_CONCURRENT_REQUESTS=1` (`start_cluster.sh`) needs its own
   explicit, separately-reviewed change~~ DONE, 2026-08-05 (user
   go-ahead given) -- `start_cluster.sh` now forwards
   `EXO_PP_BATCHED_DECODE` and relaxes the cap to 2 ONLY when that
   flag is genuinely active; every other Pipeline-mode path keeps the
   original cap=1 enforcement unchanged. See commit
   `ccf780f90`.
2. ~~A real 2-node cluster A/B~~ ATTEMPTED, 2026-08-05 (user go-ahead
   given) -- see Section 15 below for the full campaign: three real
   wiring/environment bugs found and fixed via iterative real-cluster
   testing, culminating in a genuinely successful single-request
   result. **N=2 concurrent admission is NOT yet safe** -- a real,
   previously-undiscovered architectural gap (not a wiring bug) was
   found: nothing today guarantees all ranks agree on the exact tick
   boundary where a second request gets admitted mid-stream, so
   request B's prefill traffic on one rank can race request A's
   decode-step traffic on the other over the same physical wire link.
   This produced a real jaccl deadline/deadlock under a genuine
   2-concurrent-request test. Section 15 has the full analysis and
   the concrete, unstarted design work needed before N=2 is safe.

**Phase 2 — Extend to prefill batching + chunked interleaving:** Add
batched/chunked prefill through the new scheduler, reusing
`prefill_batched`'s padding/masking logic adapted for the PP split.
Validate correctness (Phase 0-style diff) for BOTH the prefill and
decode halves together, at 2 concurrent requests with different context
lengths.

**Phase 2 — concrete design (2026-08-06, scoping session, no code
written yet).** Picks up from the N=2 admission-race campaign's own
handoff (`docs/batched-decode-n2-admission-handoff-2026-08-05.md`'s
"2026-08-06 handoff: starting point for Phase 2/3/4" section) now that
campaign is closed. Traced the actual current code path first
(`ExoBatchGenerator._submit_batched_decode_deferred` →
`_DeferredPrefill.run_prefill()` → `generate.py`'s `prefill()`): today
that closure runs a request's ENTIRE multi-chunk prefill (every
`EXO_PREFILL_STEP_SIZE`-sized chunk, e.g. dozens of chunks for a
100K+-token prompt) synchronously inside a single `Rank0BatchedDecodeGlue
.tick()` call, with no yield point back to the scheduler for another
request's decode steps in between — this is the concrete mechanism
behind this section's "long prefill doesn't starve concurrent decode
users" requirement; it is not yet true today.

**Interleaving model — consult-reviewed, 2026-08-06: separate
alternating steps, NOT a mixed per-step tensor.** Considered joining a
prefill-chunk's rows and a decode batch's rows into one padded
`(B_step, L_chunk, hidden)` forward pass per step (mirroring item 1's
"ONE batched tensor covering all of them" wording literally). Rejected:
padding decode's 1-token rows up to a 4096-token prefill chunk's length
wastes ~4000x compute on every decode row every step it co-occurs with
a chunk, and mixed-phase rows need per-row causal masks/RoPE
offsets/cache-write boundaries in the SAME forward pass — exactly the
subtle shape/masking bug class the closed admission-race campaign spent
7 real hardware bugs eliminating, reintroduced for a net throughput
loss. A ragged/packed block-diagonal-mask alternative (Sarathi-Serve/
vLLM-style chunked-prefill continuous batching) was also considered and
explicitly rejected for this design: uncertain MLX varlen-attention
kernel support, and at the confirmed N=2 hard ceiling (Section 13.3) it
buys negligible benefit over plain alternation to justify the added
correctness surface. Recorded here so a future pass doesn't
"rediscover" and re-litigate either rejected option.

Chosen mechanism, in terms of the ALREADY-EXISTING wire protocol (no
new message kind):
1. **Make `run_prefill()` resumable, not call-to-completion.** Refactor
   `prefill()`'s internal `while offset < max_length` chunk loop
   (`generate.py`) so a caller can drive exactly ONE chunk and get back
   a resumable cursor, instead of looping every chunk in one call.
   `_DeferredPrefill` becomes a per-chunk step function, invoked once
   per prefill-chunk tick rather than once per admission.
2. **Reuse `BatchEntry.phase=PREFILL` on `StepMessage`, not a new
   message kind.** `StepMessage.entries` already carries
   `phase`/`expected_cache_len`/`n_tokens` per entry (Phase 1 built
   `Phase.PREFILL` into the enum specifically as this forward-compatible
   placeholder — see `pp_scheduler_protocol.py`'s module docstring
   Scope note). A prefill-chunk step is simply a `StepMessage` with one
   `phase=PREFILL` entry; rank 1 dispatches on `.phase` the same way it
   already dispatches on message kind. `PrefillMessage`/
   `PrefillReadyMessage` stay scoped to their proven one-shot job (the
   admission-time "start this new request's prefill" announcement +
   readiness handshake) — not stretched into a per-chunk protocol.
3. **`Rank0BatchedDecodeGlue.tick()` gets one new rung**, between the
   existing rung 2 (prefill-grant announcement) and rung 3 (decode
   step): if a request is mid-chunked-prefill, emit its next chunk as a
   `phase=PREFILL` `StepMessage`. Per this section's own "one prefill
   chunk ... then a batch of pending decode steps ... alternating"
   wording, this rung ALTERNATES with decode rather than unconditionally
   outranking it the way the one-shot prefill-grant rung does today —
   Decision (below) fixes the ratio at 1:1 for the first cut.
4. No new ack/NACK handshake per chunk — the one-shot
   `PrefillReadyMessage` before chunk 1 already establishes both ranks
   are in lockstep on this request; subsequent chunks are ordinary
   `StepMessage`s on the already-proven single-writer channel.

**Two scope decisions locked in 2026-08-06 (best-judgment default,
simplest-first, per this campaign's own established discipline of
correctness before throughput and revisiting only if data shows it's
insufficient — not re-litigated without new evidence):**
- **Alternation ratio: fixed 1 prefill-chunk-tick : 1 decode-tick,
  always.** No adaptive `EXO_PREFILL_STEP_SIZE` shrinking based on
  decode load in the first cut. Ship this, measure real per-chunk
  latency and decode-stall impact on real hardware, revisit only if
  Phase 3/4 throughput data shows fixed alternation isn't enough — not
  a speculative optimization up front.
- **At most ONE request mid-prefill at a time.** At the confirmed N=2
  hard ceiling, Phase 2's scope is "one request's prefill chunks
  timesliced against decode steps of already-active OTHER request(s)" —
  NOT simultaneous multi-request prefill batching. This means Phase 2
  does NOT need `prefill_batched`'s real N-stream padded-batch/masking
  machinery for the PP split; it only needs the SAME single-stream
  `prefill()` chunk loop made resumable (item 1 above). If a second
  request arrives while one is already mid-prefill, it queues behind it
  in `Rank0BatchedDecodeGlue`'s existing `_pending_prefill` list exactly
  as today — no new queueing logic needed, this already exists.

Not yet started: the actual `prefill()` resumable-chunk refactor, the
new `tick()` rung, and the Phase 0-style correctness diff at 2
concurrent requests with different context lengths this section
originally called for. Every real-cluster step still needs the user's
own fresh explicit go-ahead per standing rules.

**Phase 2 design — `consult` review (2026-08-06), 8 findings, code-
verified: the design above is NOT ready for implementation as written.**
Requested a full independent critique of the design above before
writing any code (matching this campaign's own established practice of
a `consult` review before every real design decision, not just this
one). The two most severe findings were spot-checked against the ACTUAL
`pp_scheduler_protocol.py` code, not taken on faith — both confirmed
true and more severe than the review itself estimated:

1. **The prefill->decode KV cache handoff is completely unaddressed,
   and may be the REAL starvation source Phase 2 exists to fix.** This
   is a hybrid PP-prefill/TP-decode design -- the KV cache built during
   PP-layout prefill must be resharded into decode's TP layout at the
   transition. Chunking the COMPUTE (item 1 above) says nothing about
   this handoff. For a 100K+-token prompt this could be a large,
   blocking cache-resharding transfer at the last chunk -- potentially
   seconds of decode starvation for the concurrent user, which is
   exactly the requirement this phase exists to satisfy. Needs an
   explicit answer (chunk/overlap it, quantify why a one-shot handoff
   is acceptable, or scope it as a separate sub-problem) before code.
2. **No latency budget was computed, and fixed 1:1 alternation at the
   DEFAULT `EXO_PREFILL_STEP_SIZE=4096` plausibly fails outright.** A
   4096-token chunk is plausibly 1-4+ seconds of wall time on this
   hardware for a large model; 1:1 alternation means the concurrent
   decode user gets one token per chunk-duration -- sub-1-tok/s decode
   for the ENTIRE prefill, which does not satisfy "long prefill doesn't
   starve concurrent decode users." Chunk size (not adaptive shrinking,
   which was correctly scoped out) is the actual load-bearing knob left
   unaddressed. **Action before code: measure real per-chunk wall time
   on real hardware at a few `EXO_PREFILL_STEP_SIZE` values, pick a
   target decode inter-token-latency ceiling during prefill, and derive
   a static chunk size from that measurement -- write both the
   measurement and the derived value down.** `tick()` running one chunk
   synchronously also bounds scheduler responsiveness to
   eviction/admission events by the same chunk duration.
3. **CODE-VERIFIED, more severe than estimated: the state-machine work
   is completely unbuilt, not just "extend it."** Direct read of
   `pp_scheduler_protocol.py` confirms: `SchedulerCore._handle_new_request`
   hardcodes `phase=Phase.DECODE` for EVERY new request (Phase 1's own
   scope note: "every request modeled by this module begins
   already-prefilled, in DECODE") -- there is currently NO code path
   anywhere that ever constructs a record with `phase=PREFILL`.
   `RankOneMirror` never reads or branches on `.phase` at all -- the
   ONLY reference to `.phase` in the entire module (line ~554) WRITES
   it onto the outgoing wire `BatchEntry`, it is never validated on
   receipt. So "reuse `phase=PREFILL`, no new message kind" was
   accurate about the WIRE shape but seriously understated the real
   work: this phase needs (a) a new `SchedulerCore` event/handler for
   "advance this request's prefill by one chunk" (mirroring
   `_handle_new_request`'s shape but for chunk progression), (b) an
   explicit PREFILL->DECODE transition handler (not implicit), and (c)
   a genuinely new `RankOneMirror` validation branch for prefill-chunk
   entries -- none of which exist in any form today. Budget this as
   real, from-scratch state-machine work, not a small extension.
4. **Underspecified, needs explicit answers before code (per-item, not
   yet decided):**
   - Mirror validation per PREFILL chunk: `expected_cache_len`
     progression per chunk, INCLUDING the final partial chunk
     (`prompt_len mod chunk_size`) and the existing off-by-one
     convention (`prefill()` is called with `prompt_tokens[:-1]`,
     caller-side "drop the last token" -- the resumable-chunk version
     must preserve this exactly, not re-derive it per chunk and risk
     drift).
   - Who samples the first generated token and on which step -- final
     PREFILL chunk or first DECODE step? Left implicit is a guaranteed
     off-by-one/divergence site (same failure class as bug #7's rank-1
     admission-gate omission from the closed campaign).
   - Enforce "no mixed-phase entries in one `StepMessage`" as a
     MIRROR-VALIDATED invariant (loud `ProtocolViolationError` on
     violation), not just an implementation convention nobody checks.
   - Rank 1's actual handler for a PREFILL `StepMessage`: which cache
     object, which model call path, and how the per-chunk activation
     transfer sequences against the single-writer control channel.
     Alternation means prefill and decode traffic never overlap in
     time on the wire -- state this explicitly as a load-bearing
     invariant the design relies on, not just an implied consequence.
   - Eviction/cancellation MID-chunked-prefill: resumable-cursor
     teardown, partial-KV-cache freeing on both ranks in the PP layout,
     mirror state cleanup, ordering of `EvictMessage` vs an in-flight
     PREFILL `StepMessage` (the single-writer channel gives ordering
     for free -- say so explicitly), and a client-timeout story for a
     second request queued behind a multi-minute prefill in
     `_pending_prefill`.
   - Alternation-state edge cases: decode request finishes (EOS)
     mid-prefill (chunks then run back-to-back, presumably -- confirm
     and make it fall out of the state machine, not a special case); no
     decode work present at admission time; interaction with rung 2
     (prefill-GRANT) firing for a second request while a first request
     is already mid-chunk. Whose-turn state must be DERIVABLE from the
     state machine, never a separately-mutable flag that can desync.
5. **MLX-specific risks not yet addressed:** (a) laziness -- the
   resumable cursor must force `mx.eval`/`async_eval` at each yield
   point or an unbounded lazy graph accumulates silently across ticks
   (same failure class, different call site, as the design doc's
   already-fixed v3 metaframe bug -- Section 9, "SECOND REAL CLUSTER
   ATTEMPT" -- where an unevaled lazy send node was silently discarded);
   (b) peak memory during interleaving = a growing 100K+-token prefill
   cache + both decode requests' caches + both working sets held
   simultaneously -- needs a real back-of-envelope number for the
   largest supported model before committing, same discipline as the
   Pre-Phase-0 KV-memory check already did for Phase 1.

**Verdict: items 1-3 above must be resolved with real answers (not
best-judgment defaults) before any code is written.** Items 4-5 can
remain open questions carried into implementation IF they are written
down explicitly first (which this entry now does) -- the review's own
framing, endorsed here. The "no per-chunk ack" decision (mechanism item
4, above) remains sound AS LONG AS finding 3's real mirror-validation
work is actually built -- it is what makes the no-ack decision safe,
and it does not exist yet.

**Follow-up code/data audit (2026-08-06, same session) -- finding 1
RETRACTED, finding 2 substantially revised with real hardware numbers,
design PIVOTS from chunk-level alternation to intra-chunk layer-boundary
yield points:**

**Finding 1 (prefill->decode cache reshard) is RETRACTED -- false
premise, confirmed by code + doc cross-check.** This design doc's own
title is a naming leftover: Section 6.1 explicitly states the CHOSEN
architecture "keep[s] PP's layer-split topology ... for BOTH prefill
and decode -- do not disaggregate PP vs TP by phase." The PP-prefill/
TP-decode phase-disaggregation alternative (a genuinely different
design that WOULD have needed a layout reshard) was evaluated and
explicitly REJECTED/PULLED in Section 7 (insufficient memory headroom
for dual weight-layout residency, ~2.7GB/node). Confirmed directly in
code: `_run_deferred_prefill_for_grant` (`batch_generate.py`) returns
`prefilled_cache: KVCacheType` from `run_prefill()` and passes it
UNCHANGED into `enqueue_admission`/`stage_local_cache` -- the exact
same cache type the decode session already operates on, no conversion
step anywhere. `pp_batched_cache_router.py`'s `merge_request_caches`/
`extract_request_cache` have zero TP/reshard-related code. There is no
cache-layout handoff in this architecture because there is only ever
one layout (PP). This finding should not have been raised without
first re-reading Section 6.1 -- noted as a process lesson: check the
doc's own stated architecture before speculating about a problem a
plausible-sounding doc TITLE implies.

**Finding 2 (latency budget) revised with REAL measured hardware
numbers -- the original estimate was based on the WRONG chunk size and
the real numbers are worse, structurally, not just numerically.**
Corrected input: the deployed production chunk size is
`EXO_PREFILL_STEP_SIZE=2048` (`start_cluster.sh`, quality- and
perf-validated), NOT the code default of 4096 the original finding
assumed. Real measured GPU time per 2048-token chunk (`mx.metal
.gpu_time_ns` + `mx.synchronize`, single-request PP prefill, prior
session's profiling work): 2K context=4.78s/chunk, 64K=4.85s,
129K=5.06s, 227K=5.38s, 358K=5.92s, 489K=6.46s, 522K=6.49s (93-95%
GPU-utilized throughout, not idle-bound). **Fixed ratio-based
alternation (any N decode-ticks-per-chunk) cannot fix this**, per a
second `consult` review: changing N changes the duty cycle, not the
INTERRUPTION GRANULARITY -- the worst-case inter-token gap for the
concurrent decode user is still one full chunk (4.8-6.5s) regardless of
N. Getting a tolerable AVERAGE decode rate via more decode-ticks per
chunk roughly doubles total prefill wall time (a 500K prefill is
already ~256 chunks x ~5.7s = ~24 minutes; doubling it fails any
plausible prefill-latency budget). This is not a tuning problem, it is
a granularity problem -- the mechanism itself (item 3, "one new tick()
rung, alternates chunk-vs-decode") is the wrong shape, not just
mistuned.

**Chunk size cannot be shrunk as the fix either -- real, measured
quality hazard, not just a throughput one.** A prior chunk-size sweep
at 500K context found chunk sizes OTHER than 2048 produce mode-collapse
garbage output (chunk=256/384/1024 all produced incoherent repeated-
token garbage; only chunk=2048 was coherent) -- an unexplained,
never-root-caused boundary-indexing-shaped bug, previously deprioritized
because it wasn't on the throughput-critical path. This means "shrink
`EXO_PREFILL_STEP_SIZE` while a decode request is active" (the obvious-
looking adaptive-chunk-size fix, previously scoped OUT for being a
premature optimization, not for being unsafe) is now flagged as a
POSSIBLE CORRECTNESS HAZARD, not merely a deferred optimization --
downgrading chunk size to interleave more finely could silently corrupt
the interleaved request's prefill instead of just slowing it down. Do
not revisit "adaptive chunk shrinking" as a fix without first
root-causing this bug.

**Revised direction (consult-reviewed): intra-chunk layer-boundary
yield points, not chunk-level alternation.** Split ONE 2048-token
chunk's forward pass into segments of K consecutive transformer layers
(both nodes' own PP layer range independently, e.g. ~2-3 layers per
segment against this model's ~60 total layers, ~90ms/layer at the
measured chunk times above), with an eval/async_eval boundary and a
"decode work pending" check between segments -- pause the prefill
between segments, run one decode tick, resume the SAME in-flight chunk
where it left off. This is safe against the chunk-size quality hazard
above because it is ORTHOGONAL to it: the chunk stays 2048 tokens
through every layer, unchanged; only the SCHEDULING of layers within
that unchanged forward pass gets interrupted between segments. The
existing PP rank-0/rank-1 boundary is already a free yield point (2
segments); this adds sync points WITHIN each rank's own layer range.
Rough math: segment size of 2-3 layers implies a worst-case decode gap
in the low hundreds of ms (segment time + one decode-tick time) instead
of 4.8-6.5s -- a qualitatively different result, not a smaller version
of the same one.

**Real measurements needed before this can be locked in (none taken
yet, real-cluster time, needs the user's own fresh explicit go-ahead
per standing rules):**
1. Real decode-tick wall time on this hardware (not yet measured
   anywhere in this doc's history) -- required to size segment
   granularity against a concrete "worst-case decode gap" target, not
   a guess.
2. A per-layer-segment-size sweep (candidates: 1, 2, 4, 8 layers) to
   measure real eval-boundary overhead against the current 93-95%
   GPU-utilization baseline -- losing kernel fusion / adding sync
   points at each boundary has a real, unmeasured cost that could erode
   the chunk-level throughput number materially. If overhead is bad,
   coarser segments (larger decode gap, less overhead) may be the
   forced compromise -- write down the real trade curve, don't assume
   a segment size.
3. **Standing risk carried forward, not yet resolved:** the un-root-
   caused <2048-chunk-size quality bug is a signal of general fragility
   somewhere in the prefill path's cache-offset/boundary-indexing logic.
   Layer-level segmentation SHOULD be orthogonal to it (chunk shape is
   unchanged) but "should" is not proof for a bug that was never
   root-caused. Before trusting layer-interleaved prefill in production,
   re-run the SAME 500K coherence check (needle-in-haystack or
   equivalent) with layer-interleaved prefill + concurrent decode
   active, not just single-request layer-interleaved prefill alone.

**Updated verdict:** finding 1 is closed (retracted, no action needed).
Finding 2's chunk-alternation mechanism (Phase 2 design mechanism item
3, above) is SUPERSEDED by the intra-chunk layer-boundary approach --
the "one new `tick()` rung, alternates 1:1" mechanism as originally
written should not be implemented; it structurally cannot meet the
"doesn't starve concurrent decode" requirement regardless of tuning.
Finding 3 (state-machine work) gets STRICTLY BIGGER under this revision
-- instead of a per-chunk phase transition, the state machine now needs
a per-LAYER-SEGMENT resumption point within a single logical chunk, a
real budgeting/measurement pass (items 1-2 above) before any code, and
the same real mirror-validation build-out finding 3 already called for,
now scoped to segment boundaries instead of chunk boundaries.

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

## 15. Real 2-node cluster campaign (2026-08-05) — three wiring/env
    bugs found and fixed, single-request path VERIFIED WORKING, N=2
    concurrent admission found UNSAFE (new architectural gap)

**STATUS (2026-08-06, final): SEVEN real bugs found via real N=2
hardware testing. ALL SEVEN are FIXED AND HARDWARE-VERIFIED. N=2
genuinely concurrent requests now work cleanly on real 2-node
hardware.** See `docs/batched-decode-n2-admission-handoff-2026-08-05.md`'s
"2026-08-06 fix: in-band PrefillMessage admission signal", "2026-08-06
follow-up: 4 real bugs in eviction+slot-reuse", "2026-08-06 fix:
prefill forward-pass race (PrefillReadyMessage)", "2026-08-06 fix:
eliminate cross-rank eligibility divergence (drop is_prefix_cache_hit)",
and "2026-08-06 fix: bug #7's third root cause (rank 1 admission-gate
never drains) + hardware verification" sections for the full
implementation and verification writeups. Bug #7 (the last one) had
THREE separate root causes found iteratively across three real
hardware attempts: `EXO_DSV4_BATCHED_PREFILL` bypassing the
single-writer channel, a NACK/timeout conflation in the retry guard,
and rank 1's `runner.py`-level admission gate never draining while
batched-decode was active (the real blocker). A 4th hardware attempt
after all three fixes ran 8 genuinely concurrent requests across 4
rounds with zero crashes, zero 500s, zero wire errors. Single-request
PP remains unaffected throughout. `EXO_PP_BATCHED_DECODE=1` stays OFF
by `start_cluster.sh`'s default (intentionally opt-in) but is now a
genuinely verified, working code path. The single_request_fallback
(ineligible-request) path remains unfixed, explicitly out of scope for
this campaign.

Following the user's explicit go-ahead for a real cluster A/B (Section
9's remaining item 2), this section records the full campaign: four
real cluster launch attempts, three genuine bugs found and fixed via
the standard cycle (crash → restore to known-good → diagnose locally →
fix → redeploy → retry), and one real, previously-undiscovered
architectural gap that blocks N=2 concurrency specifically.

**Standing discipline followed throughout:** every crash was followed
IMMEDIATELY by killing the crash-looping launcher and relaunching with
`EXO_PP_METAFRAME=1` only (no `EXO_PP_BATCHED_DECODE`) to restore the
cluster to its last-known-good state, verified via a real
`curl`-issued "capital of France" → "Paris" `finish_reason=stop`
inference each time before any further diagnosis — never left the
cluster in a crashed/unverified state while investigating.

### Attempt 1: batched layers crash outside `batch_step_scope`

**Symptom:** both runners `RunnerFailed` immediately at warmup, before
any real request:
```
RuntimeError: BatchedMetaFramedPipelineFirstLayer/LastLayer called
outside an active batch_step_scope(...) block -- this is a caller bug
(forgot to wrap the model(...) call), not a data-driven condition;
refusing to guess at a request set
```

**Root cause:** installing `BatchedMetaFramedPipelineFirstLayer`/
`LastLayer` at model-load time means EVERY forward pass through the
model goes through them — not just `BatchedDecodeSession`'s own
`tick()`-driven decode steps. Runner warmup's `prefill()` call
(existing, unmodified code, no `batch_step_scope` wrapper — it was
never designed to need one) crashed immediately on the fail-loud
`_require_batch_step_context()` guard. The guard's fail-loud behavior
was exactly correct for a genuine caller bug; the actual bug was that
these classes never handled the legitimate "no batch context active"
case at all.

**Fix (commit `95ae867db`):** `BatchedMetaFramedPipelineFirstLayer`/
`LastLayer` now SUBCLASS `MetaFramedPipelineFirstLayer`/
`MetaFramedPipelineLastLayer` (Phase 0.5's already-cluster-verified
single-request classes) instead of `CustomMlxLayer` directly, and fall
back to `super().__call__()` whenever no `BatchStepContext` is active
— reusing Phase 0.5's proven prefill/queue_sends/decode-gather logic
UNCHANGED rather than reimplementing it. Per a `consult` review,
subclassing (not composition-with-setter-forwarding) was the correct
shape here specifically because both layer kinds construct with the
exact same arguments as their base class, so there's no divergent
state to reconcile — and it comes with a real bonus: `auto_parallel.py`'s
`set_pipeline_prefill`/`set_pipeline_queue_sends` already
`isinstance`-check for `MetaFramedPipelineLastLayer`, so the subclass
picks up correct `is_prefill`/`queue_sends` behavior for the fallback
path for free, zero additional wiring.

New regression test
(`test_batched_layers_fall_back_to_single_request_outside_batch_step_scope`)
verified via `git stash` A/B to fail against the pre-fix code with the
EXACT error message the cluster hit, and pass against the fix.

### Attempt 2: rank 1's `submit()` never called `stage_local_cache`

**Symptom:** first real single request after the Attempt-1 fix crashed
on rank 1:
```
GlueError: tick(): rank 1 received an admission for request_id=0
cache_slot=0 but has no staged local prefilled cache for it --
stage_local_cache was never called for this request_id.
```

**Root cause:** `submit()`'s dispatch gate only checked
`self._batched_decode_rank0_glue is not None` — always `False` on
rank 1 (that glue lives on `_batched_decode_rank1_glue` instead), so
every one of rank 1's requests silently fell through to the old
serial `_mlx_gen.insert()` path and never staged its locally-prefilled
KV cache. When rank 0's admission for that request arrived over the
real wire, rank 1's glue had nothing staged to bind it to.

**Fix (commit `ba02ba9c8`):** two changes. (1) `submit()`'s dispatch
gate now checks EITHER glue being set, not just rank 0's. (2)
`_submit_batched_decode()` now branches on which glue is present: rank
0 calls `Rank0BatchedDecodeGlue.enqueue_admission` (unchanged), rank 1
calls `Rank1BatchedDecodeGlue.stage_local_cache` instead. Both
branches derive `uid`/`cache_slot` identically from the SAME symmetric
counter state (`self._uid_counter` / `len(self._active_tasks)`) —
correct because both ranks process the identical, globally-ordered
stream of eligible submissions per this fork's own event-sourcing
architecture, so no new cross-rank message is needed to keep them in
sync.

New regression test
(`test_rank1_submit_dispatches_to_stage_local_cache_not_serial_insert`)
constructs a real `ExoBatchGenerator` with a real `Rank1BatchedDecodeGlue`
attached and proves `submit()` calls `stage_local_cache`, not the old
serial path — verified via `git stash` A/B to fail against the pre-fix
code and pass against the fix.

### Attempt 3: `GenerationBatch.Response` kwargs didn't match the real
    mlx-lm submodule — a LOCAL DEV-ENVIRONMENT bug, not a cluster bug

**Symptom:** first real single request after the Attempt-2 fix crashed
on rank 0:
```
TypeError: GenerationBatch.Response.__init__() got an unexpected
keyword argument 'current_state'
```

**Root cause, once diagnosed: NOT a cluster/deployment problem.** The
local dev Mac's `~/repos/exo/.venv` had silently drifted from the
vendored `./mlx-lm` submodule after a plain `uv sync` earlier in the
session. `uv.lock` pins `mlx-lm` by an exact git SHA, but the package
version string never changes between mlx-lm commits, so `uv sync`
alone reports "already satisfied" and skips reinstalling even when the
submodule gitlink has moved. `start_cluster.sh`'s own comment (~line
1156) documents this exact trap for the CLUSTER nodes and fixes it via
`uv pip install --no-deps --force-reinstall ./mlx-lm` immediately after
`uv sync --extra mlx --all-packages` — but the same fix needs running
LOCALLY too after any submodule-adjacent work, and hadn't been. The
drifted local venv had a NEWER `GenerationBatch.Response` with
`current_state`/`match_sequence` fields that don't exist in the real,
currently-vendored submodule (or on either cluster node) — so code
written and unit-tested locally against the stale, newer signature
crashed immediately against the real deployed mlx-lm. Compounding
factor: a stale committed type stub (`.typings/mlx_lm/generate.pyi`)
had the SAME phantom fields, so `basedpyright` agreed with the broken
code instead of catching it.

**Fix (commit `69a4f21e9`):** (1) rebuilt the local venv correctly
(`uv sync --extra mlx --all-packages` + `uv pip install --no-deps
--force-reinstall ./mlx-lm` + Rust bindings rebuild), matching
`start_cluster.sh`'s own node-sync sequence exactly; (2) removed the
two non-existent kwargs from the one call site that had them; (3)
fixed the stale type stub to match the real submodule.

**Significant side discovery:** the full worker suite's "1 pre-existing
unrelated failure" reported consistently throughout this ENTIRE
session (both this stretch and prior ones) was ALSO caused by this
same venv drift, not a real codebase issue — confirmed by re-running
that specific test 3× against the corrected venv, all passing. Saved
to warm memory (fact 1216) as a durable environment-hygiene lesson.

### Attempt 4: single request SUCCEEDS — first genuinely successful
    real-cluster result on this design's decode path

With all three fixes deployed, a real single request through the
batched-decode path returned a clean, correct result:
```json
{"finish_reason": "stop", "content": "Paris", ...}
```
Both ranks' logs confirmed the batched-decode session actually
engaged (`"Phase 1 batched-decode ENABLED (rank 0, admission+decode
glue constructed)"` / `"...(rank 1, mirror glue constructed)"`), not a
silent fallback to the serial path. This is a real milestone: every
piece of Phase 1's admission → decode → response machinery, built and
unit-tested across this and prior sessions, now genuinely works
end-to-end on real hardware for the single-request case.

### The N=2 concurrent-admission race — a real, NEW architectural gap

Sending 2 genuinely concurrent HTTP requests (via a Python
`ThreadPoolExecutor`, not sequential `curl` calls) to the same runner
produced an HTTP 500, with the runner log showing:
```
[jaccl-v2] DEADLINE rank=0 call_id=604 all_recv=0/1 chunks_posted=1 small=1 peer_in_call=0
[mlx scheduler] captured St13runtime_error in task: [jaccl] reliable_all_reduce_v2 deadline — clean re-place
jaccl transport fault in generator.step(): [jaccl] reliable_all_reduce_v2 deadline — clean re-place.
jaccl reconnect failed (RuntimeError('[jaccl] Recv failed: peer closed connection (EOF) fd=52 remaining=4'))
Runner crashed with critical exception [jaccl] reliable_all_reduce_v2 deadline — clean re-place
```
(The exact per-rank collective trace at the moment of deadlock was
NOT captured — the log rotated before it could be pulled during the
subsequent restore-to-known-good. The finding below is the
architectural analysis of WHY this class of failure is expected given
the current design, not a byte-for-byte forensic trace of this
specific occurrence. A follow-up attempt with `JACCL_TRACE_STEP=1` or
similar per-step tracing enabled would be needed to capture the exact
trace if bulletproof confirmation is wanted before starting the fix
below.)

**Root cause (per a `consult` review, restated more precisely than the
first-pass framing):** the bug is NOT "prefill's wire traffic and
decode's wire traffic are not mutually exclusive" — within one rank,
the single synchronous `handle_generation_tasks()` while-loop in
`runner.py` DOES serialize `submit()`/prefill and `step()`/decode; they
cannot literally overlap on a given rank. The actual flaw is
**unsynchronized admission decisions across ranks**: nothing today
guarantees every rank decides to admit a given second request (call it
B) at the same tick boundary. Each rank's runner independently pulls
work off its own local `_work_queue` (populated by the master's
event-sourced broadcast, which IS globally ordered — see Section 9's
existing "why no new broadcast-admission protocol is needed" analysis
for `submit()`/`_submit_batched_decode`'s own uid/cache_slot
symmetry) and independently decides, each loop iteration, whether to
call `submit()` (issuing PREFILL's own p2p send/recv collectives,
using the single-request metaframe layers via this session's Attempt-1
fallback) or `step()` (issuing DECODE's collectives via the batched
glue's `tick()`). If rank 0 reaches request B's `submit()` in its own
loop iteration before rank 1 has processed the same event off its own
queue — plausible, since nothing synchronizes the two ranks' local
polling cadence — rank 0 can start issuing prefill's collective
pattern while rank 1 is still mid-`tick()` issuing decode's pattern.
Two ranks running MISMATCHED collective operations on jaccl is exactly
the deadline/deadlock signature observed.

**Why this didn't surface in Attempt 4 or in this session's earlier
2-process subprocess tests:** the subprocess harness
(`test_pp_batched_decode_glue_subprocess.py`) drives BOTH ranks'
`glue.tick()`/`glue.enqueue_admission()`/`glue.stage_local_cache()`
calls explicitly, in lockstep, from a single test driver — it never
exercises the REAL runner's independent per-rank event-loop polling at
all, so it structurally cannot reproduce a race that only exists
because two independent `runner.py` processes decide independently
when to advance. This is a real, high-value catch for future
regression-test design: a "real 2-process test" that still drives both
sides from one script is not equivalent to two genuinely independent
runner event loops.

**What does NOT need to change:** the existing decode-step protocol
itself (`pp_scheduler_wire.py`, `Rank0BatchedDecodeGlue`/
`Rank1BatchedDecodeGlue`, `SchedulerCore`'s DRAINING-until-ack
invariant) is not implicated — it was built single-writer from the
start specifically to avoid exactly this class of hazard WITHIN the
decode-step loop (see Section 9 item 4's "piggyback" design). The gap
is specifically at the SEAM between `submit()`'s prefill dispatch and
`step()`'s decode dispatch — the point where a runner decides which of
the two to call next, independently per rank.

**Concrete, UNSTARTED design work needed before N=2 admission is
safe** (per the consult review's stated fix direction — this is a
recommendation, not yet designed or built):

1. **All ranks must agree, deterministically, on the exact tick
   boundary where a new request is admitted** — not merely "both ranks
   eventually see the event," but "both ranks switch from decode-mode
   collectives to prefill-mode collectives (or vice versa) at the same
   logical step." Per the consult review, the cheapest correct
   mechanism is likely IN-BAND: fold the admission signal into the
   EXISTING decode-step wire traffic (e.g. a flag in the batched
   metaframe header, or piggybacked onto the next `StepMessage`) so
   every rank naturally switches at the same well-defined point in the
   traffic it's already synchronized on — rather than an
   out-of-band/independent per-rank queue-polling decision, which is
   exactly what creates the race today.
2. **Only the driver/head rank (rank 0) should independently decide
   "start prefill for request B now"** — rank 1 should never make that
   decision from its own local queue state; it should react to a
   signal from rank 0, mirroring the same "rank 0 decides, rank 1
   reacts reactively" pattern the existing decode-admission protocol
   already uses successfully (`RankOneMirrorDriver`'s reactive
   cache_slot-transition detection).
3. A genuinely new regression test that exercises two REAL, independent
   `runner.py`-equivalent event loops (not a single test driver calling
   both sides' glue methods directly) — since this is the exact gap
   that let the race through undetected in this session's existing
   subprocess test.
4. Once (1)-(3) exist, re-attempt the real 2-node N=2 concurrent test
   — with per-step jaccl tracing enabled this time
   (`JACCL_TRACE_STEP=1` or equivalent) to capture the exact collective
   trace and confirm the fix, not just the absence of a crash.

**Cluster state at end of session:** restored to known-good
(`EXO_PP_METAFRAME=1` only, `EXO_PP_BATCHED_DECODE` unset/off,
`deepseek-ai/DeepSeek-V4-Flash-0731`, `RunnerReady` 2/2), verified via
a real inference. All three fixes from this campaign are committed and
pushed to `main` (commits `95ae867db`, `ba02ba9c8`, `69a4f21e9`,
`ccf780f90`). The single-request batched-decode path is genuinely
proven working on real hardware; N=2 concurrent admission remains
correctly gated OFF pending the design work above — `EXO_MAX_CONCURRENT_REQUESTS`
is still forced to 1 unless `EXO_PP_BATCHED_DECODE=1` is explicitly
set (opt-in only), so this campaign's findings do not put any
production traffic at risk.



