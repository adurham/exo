# Hybrid PP-Prefill / TP-Decode Sharding for DeepSeek-V4-Flash on the exo Cluster

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

3. **Decode throughput: ≥37.5 tok/s aggregate at c=2, MTP on.**
   STATUS: this exact number is ALREADY CONFIRMED ACHIEVED under TP —
   warm memory fact 745 (2026-06-26, `concurrent_bench.py`, MTP on,
   seq-split on, "verified config"): 37.5 tok/s aggregate (18.7 tok/s
   per-request × 2 concurrent), stable (tail_ratio 1.02, zero errors),
   beating the team's own 30 tok/s c=2 target by 25%. CRITICAL SCOPING
   CAVEAT that must not get lost: fact 745's benchmark used a SHORT
   prompt (120 words) with `max_tokens=512` — i.e., this number is
   confirmed at LOW context depth, not at 100K/500K. No measurement of
   c=2 MTP-on decode throughput at real (100K+) context depth exists in
   this fork's history as of this doc. Decode throughput typically
   degrades with context depth (KV-cache read cost grows), so 37.5 tok/s
   holding at 500K is NOT something to assume — it is something Phase 3's
   throughput validation MUST explicitly test at multiple context depths
   (short, 100K, 500K), not just reproduce the short-prompt number and
   declare victory. Also note: MTP, not DSpark, is the mechanism behind
   this number — DSpark has its own separate, unresolved reliability bug
   (Section 8 item 3) and is explicitly NOT part of this throughput
   requirement's validated baseline.

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
  would push per-node expert-weight residency from ~50% to ~75%. NOT
  adopted as the primary direction for this doc (see Section 7) but kept
  as a fallback if Section 6's batched-PP design hits an unexpected wall.

## 5. Gate-check result: the "can DSv4 batch with heterogeneous lengths"
   question is ALREADY ANSWERED YES by this fork's own TP code

This is the most important finding of this design doc's research phase,
and it changes the effort estimate significantly. The consult's stated
biggest unknown — whether DSv4's MLA/sparse-attention/indexer code can
handle a real batch dimension with per-request-different sequence
lengths, or whether it's written assuming batch=1 — is not actually
unknown. It's already built, tested, and running in production TP mode
today:

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

CONSEQUENCE for the hybrid design: instead of writing new batched-attention
code for DSv4 from scratch (a multi-week, high-risk undertaking touching
the model's numerics), the batched-PP design in Section 6 can most likely
reuse `prefill_batched`'s existing padding/masking/cache-prepare machinery
almost directly — the NEW work is primarily in the PP SCHEDULING/transport
layer (rank-0-as-scheduler, metadata-framed send/recv, micro-batch
interleaving for decode), not in the model's attention math. This does
NOT mean zero risk (see Section 8), but it means the single biggest named
unknown from the external consult is resolved in this design's favor
before writing a line of new code.

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

## 7. Alternative considered and set aside: phase disaggregation
   (PP-prefill → TP-decode handoff)

The consult also raised, as a possibly-cheaper alternative: use PP only
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
  weights viewed two ways — so the union would need real memory headroom
  (the consult's rough estimate: per-node expert-weight residency rising
  from ~50% to ~75% of the full expert set). This needs to be checked
  against DSv4-Flash-8bit's actual per-node footprint before it can be
  ruled in OR out; not yet done as of this doc.
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

This alternative is kept as a documented fallback: if Section 6's
batched-PP scheduler design hits a wall during Phase 1 (see Section 9)
that makes it structurally infeasible, phase disaggregation is the
next direction to prototype, and the memory-residency math above should
be done first before investing further engineering time in it.

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

## 9. Proposed phased plan (no code written yet — this is the plan to
   review, not a commitment to start immediately)

**Phase 0 — Correctness baseline (before any new scheduler code):**
Write a standalone, offline test harness that feeds 2+ different-length
prompts through TP's EXISTING `prefill_batched` path and through
PP's existing single-request path (serially, one at a time, as a
reference), and diffs the resulting logits/KV-cache state at matching
positions. This validates that TP's batching machinery genuinely
produces request-A-identical-to-serial-A / request-B-identical-to-
serial-B output (no cross-contamination) BEFORE building anything new on
top of it — establishes ground truth for what "correct" looks like once
the PP version exists to compare against.

**Phase 1 — Rank-0 scheduler skeleton, decode-only, 2 concurrent
requests, NO speculative decode:** Build the metadata-frame protocol and
the rank-0 scheduler for the SIMPLEST case first — 2 concurrent plain
(no DSpark) decode-only requests (both already prefilled via today's
existing serial PP prefill, just testing the NEW concurrent decode
path). Validate byte-for-byte correctness against 2 serial single-
request PP runs before touching throughput at all.

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

**Phase 4 — N > 2 concurrent requests, cancellation, DSpark gating:**
Extend the scheduler to handle more than 2 in-flight requests, wire up
cancellation (item 6), and add the DSpark concurrency=1-only gate
(item 7). Full correctness + throughput validation at realistic
concurrency levels (mirroring actual Hermes usage patterns — the
existing `/tmp/hermes_stress_test.py` corpus-replay harness built for
the DSpark-off stability soak test is a natural fit to reuse/extend here
for a mixed prefill+decode concurrent-load validation).

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

## 11. Open questions for review before implementation starts

1. Does the phased plan's ordering make sense, or should Phase 0's
   correctness-first framing be even MORE front-loaded (e.g. should
   Phase 0 also include an adversarial fuzzing pass — many random
   concurrent request combinations — before Phase 1 begins, rather than
   after)?
2. Is concurrency=1-only DSpark gating (risk #2) an acceptable permanent
   scope boundary for v1, or should batched-DSpark be pulled INTO this
   design's scope rather than deferred?
3. Should Section 7's phase-disaggregation alternative get its memory-
   residency math checked now (cheap, ~30 min per the consult) even
   though it's not the primary direction, just to have it fully ruled
   in/out rather than left as an open fallback?
4. What's the actual timeline expectation — is this a "next available
   multi-day focused session" project, or should it be scoped even
   larger (e.g. spread across several sessions with explicit checkpoint
   reviews after each phase)?
