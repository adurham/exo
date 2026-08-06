# Batched-decode Phase 1: N=2 concurrent admission — handoff (2026-08-05)

**STATUS UPDATE (2026-08-06, later still): bug #6's fix does NOT
close N=2 on real hardware -- a SEVENTH real bug (or bugs) found.**
Bug #6's own fix (dropping `is_prefix_cache_hit`, see below) is
correct for what it touches and did not regress anything -- but the
first real N=2 hardware run after deploying it crashed again, with
the EXACT SAME error signature bug #5 was supposed to have already
closed (`SchedulerWireProtocolError: recv_header: version mismatch --
received 3, this rank expects 1`), via a THIRD call path neither bug
#5 nor bug #6 touched. A follow-up attempt with
`EXO_DSV4_BATCHED_PREFILL=0` (an older, unrelated, always-on-by-default
rendezvous-batching mechanism suspected as the interacting factor)
did NOT fix it either -- it crashed again, faster, with a DIFFERENT
raw transport error (`[jaccl] Recv failed with errno=54`, not a
version-mismatch parse error). See "2026-08-06 finding: N=2 still
crashes after bug #6's fix (bug #7, root cause NOT yet identified)"
at the bottom of this file for the full repro, both crash signatures,
and what's ruled out so far. **Bugs #1-5 remain hardware-verified**
-- see "2026-08-06 fix: in-band PrefillMessage admission signal",
"2026-08-06 follow-up: 4 real bugs in eviction+slot-reuse", and
"2026-08-06 fix: prefill forward-pass race (PrefillReadyMessage)".
Single-request PP is unaffected and repeatedly verified working on
real hardware, including immediately after BOTH of today's crashes
(self-healed cleanly each time). `EXO_PP_BATCHED_DECODE=1` remains
UNSAFE for production -- the cluster must stay in the safe known-good
config (single-request PP, `EXO_PP_BATCHED_DECODE` unset) until bug
#7 is actually root-caused and closed.

Design doc: `docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md`
(Section 15 has the full campaign writeup this doc summarizes into an
actionable next-session starting point).

Branch: `main` · exo `669cd9f3e`

## TL;DR

Phase 1's batched-decode path (2 concurrent PP requests sharing one
decode-step batch) is **fully wired and VERIFIED WORKING for a single
request on the real 2-node cluster** — first genuine success on this
design's decode path. **N=2 genuinely concurrent requests deadlock the
cluster** (`[jaccl] reliable_all_reduce_v2 deadline`) due to a real,
previously-undiscovered architectural gap: nothing today guarantees
both ranks agree on the exact tick boundary where a second request
gets admitted mid-stream. This doc is the concrete starting point for
closing that gap.

`EXO_MAX_CONCURRENT_REQUESTS` stays forced to 1 unless
`EXO_PP_BATCHED_DECODE=1` is explicitly set (opt-in only) — production
traffic is unaffected by any of this.

## How to reproduce the deadlock

```bash
DSV4_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-0731 \
DSV4_SHARDING=Pipeline \
EXO_PP_METAFRAME=1 \
EXO_PP_BATCHED_DECODE=1 \
EXO_SPECULATIVE=0 EXO_DSV4_DSPARK=0 EXO_DSV4_MTP=0 \
./start_cluster.sh
```

Wait for `RunnerReady (2/2)`. A single sequential request works fine:

```bash
curl -s http://adams-mac-studio-m4-1.local:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash-0731","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":50,"temperature":0}'
# -> clean {"finish_reason": "stop", "content": "Paris", ...}
```

Two **genuinely concurrent** requests (must actually overlap in wall
time — sequential curl calls won't reproduce it) deadlock:

```python
import concurrent.futures, json, urllib.request

def make_request(prompt):
    data = json.dumps({"model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50, "temperature": 0}).encode()
    req = urllib.request.Request(
        "http://adams-mac-studio-m4-1.local:52415/v1/chat/completions",
        data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read())

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    futures = [ex.submit(make_request, p) for p in
               ["Count from 1 to 5.", "Name 3 colors."]]
    results = [f.result() for f in futures]  # -> HTTPError 500
```

Runner log on crash:
```
[jaccl-v2] DEADLINE rank=0 call_id=604 all_recv=0/1 chunks_posted=1 small=1 peer_in_call=0
[mlx scheduler] captured St13runtime_error: [jaccl] reliable_all_reduce_v2 deadline — clean re-place
jaccl reconnect failed (RuntimeError('[jaccl] Recv failed: peer closed connection (EOF) fd=52 remaining=4'))
Runner crashed with critical exception [jaccl] reliable_all_reduce_v2 deadline — clean re-place
```

**Known gap:** the exact per-rank collective at the moment of deadlock
was never captured (log rotated before it could be pulled). First step
of the next session should be re-reproducing this WITH
`JACCL_TRACE_STEP=1` (or equivalent per-step jaccl tracing) enabled so
the fix can be verified against a real trace, not just "no crash."

## Root cause (analysis, not yet a captured trace — see gap above)

Per a `consult` review during the original investigation:

**NOT** "prefill's wire traffic and decode's wire traffic overlap" —
within a single rank, `runner.py`'s `handle_generation_tasks()` is one
synchronous while-loop; `submit()`/prefill and `step()`/decode cannot
literally run concurrently on the same rank.

**IS** "unsynchronized admission decisions across ranks." Each rank's
runner independently pulls work off its own local `_work_queue` and
independently decides, per loop iteration, whether to call `submit()`
(prefill's own p2p collectives, via the Attempt-1-fixed single-request
metaframe layers) or `step()` (decode's collectives, via
`Rank0BatchedDecodeGlue`/`Rank1BatchedDecodeGlue.tick()`). Nothing
guarantees rank 0 and rank 1 reach the SAME decision at the SAME
logical moment. If rank 0 starts request B's `submit()`/prefill while
rank 1 is still mid-`tick()` on decode, the two ranks issue mismatched
collective operations on jaccl — deadlock.

**Why the existing subprocess test didn't catch this:**
`test_pp_batched_decode_glue_subprocess.py` drives BOTH ranks'
`glue.tick()`/`enqueue_admission()`/`stage_local_cache()` calls
explicitly, in lockstep, from ONE test driver script. It never
exercises two genuinely independent `runner.py`-equivalent event
loops polling their own queues on their own schedule — so it
structurally cannot reproduce a race that only exists because of that
independence. This is the concrete gap in the regression-test design,
not just the production code.

**What's NOT implicated:** the decode-step protocol itself
(`pp_scheduler_wire.py`, the glue classes, `SchedulerCore`'s
DRAINING-until-ack invariant) — it was already built single-writer
specifically to avoid this class of hazard WITHIN the decode loop. The
gap is at the SEAM between `submit()`'s prefill dispatch and `step()`'s
decode dispatch, where each rank decides independently which to call
next.

## Concrete next steps (none started)

1. **Design the rank-0-decides / rank-1-reacts admission signal.**
   Likely shape (per the consult review, not yet designed in detail):
   fold "admit request B now" into the EXISTING decode-step wire
   traffic (a flag in the batched metaframe header, or piggybacked
   onto the next `StepMessage` rank 0 already sends every decode
   step) rather than a new out-of-band message. This mirrors the
   already-proven pattern `RankOneMirrorDriver` uses for the existing
   decode-step admission detection (a `cache_slot` transitioning
   FREE→occupied within a normal `StepMessage`) — the same idea, one
   level up, at the prefill/decode SEAM instead of within decode
   alone.
   - Concretely: rank 1 should NEVER independently decide "start
     prefill for a newly-visible request" from its own local queue
     state. It should stay in decode-mode collectives until it
     observes rank 0's admission signal, then switch.
2. **Build a genuinely independent 2-process regression test** — two
   real OS processes each running their own polling loop against
   their own local queue (mirroring `runner.py`'s actual structure,
   not the existing lockstep-driven subprocess test) — that can
   actually reproduce the race before the fix, and prove it's closed
   after.
3. **Re-attempt the real cluster test** with `JACCL_TRACE_STEP=1`
   enabled to capture the exact collective trace this time.
4. Once N=2 is verified stable, resume Section 9's Phase 2/3/4 plan
   (prefill batching, micro-batch interleaving, cancellation +
   DSpark-gating + realistic-load validation) — all still unstarted,
   correctly blocked on this.

## Standing reminders for whoever picks this up

- **Cluster relaunch/live-cluster testing needs the user's own
  explicit separate go-ahead** — not implied by "keep going" on
  design/implementation work. This was explicitly granted for the
  campaign that produced this handoff; get a fresh explicit go-ahead
  before the next real cluster attempt.
- **Local dev venv drift is real and already bit this exact code
  once** (see design doc Section 15, Attempt 3). Before trusting local
  pytest/basedpyright results after ANY `./mlx-lm` or `./mlx` submodule
  bump, run:
  ```bash
  cd ~/repos/exo && uv sync --extra mlx --all-packages && \
    uv pip install --no-deps --force-reinstall ./mlx && \
    uv pip install --no-deps --force-reinstall ./mlx-lm && \
    uv pip install maturin && \
    uv run maturin develop --release -m rust/exo_rs/Cargo.toml
  ```
  Also spot-check `.typings/mlx_lm/*.pyi` for drift against the real
  submodule if `basedpyright` results look suspicious (agrees with
  code that then fails against the real, deployed mlx-lm).
- Every crash → restore to known-good IMMEDIATELY
  (`EXO_PP_METAFRAME=1` only, no `EXO_PP_BATCHED_DECODE`), verify with
  a real inference (`"capital of France"` → `"Paris"`,
  `finish_reason=stop`) before further diagnosis. Never leave the
  cluster in a crashed/unverified state while investigating.
- `EXO_MAX_CONCURRENT_REQUESTS` cap-relaxation-to-2
  (`start_cluster.sh`, commit `ccf780f90`) is ALREADY correctly gated
  on `EXO_PP_BATCHED_DECODE=1` — do not touch that gate as part of
  fixing the admission race; the gate itself is fine, the race is a
  separate, deeper problem underneath it.

---

## 2026-08-06 fix: in-band PrefillMessage admission signal

Implements the design this handoff's "Concrete next steps" section 1
called for: an in-band signal, folded into the existing single-writer
`tick()` wire channel, so both ranks agree on the exact tick boundary
where a request's prefill begins -- rank 1 never independently decides
to prefill.

### What shipped

1. **New wire control message, `MSG_KIND_PREFILL`**
   (`pp_scheduler_wire.py`) + `PrefillMessage` frozen dataclass
   (`pp_scheduler_protocol.py`, fields: `step_id`, `request_id`,
   `cache_slot`, `n_prompt_tokens`, `flags` -- bit 0
   `PREFILL_FLAG_SINGLE_REQUEST_FALLBACK` for the ineligible-request
   case). Wire layout mirrors `MSG_KIND_STEP`'s established
   header-then-body pattern exactly. 7 new unit tests, all passing;
   0 basedpyright/ruff errors.

2. **`pp_batched_decode_glue.py` extended** (both classes gain a new
   `PrefillGrant` return value from `tick()`):
   - `Rank0BatchedDecodeGlue.enqueue_prefill(...)` -- pure in-memory
     queueing (mirrors `enqueue_admission`'s zero-wire-I/O guarantee),
     called from `submit()` INSTEAD OF running prefill inline.
   - `tick()`'s priority ladder reordered to: (1) admit an
     already-prefilled pending request if its slot is free, (2)
     announce a queued prefill via a real `PrefillMessage` send + a
     `PrefillGrant` return IF its target slot is neither occupied nor
     already reserved, (3) run one decode step, (4) idle. **Order 2
     before 3 is load-bearing** -- an earlier draft put decode before
     the new-prefill-grant check, which meant `has_active_requests()`
     staying `True` for a request's entire generation would starve
     ALL future admissions forever, defeating N=2 concurrency
     entirely. Caught via a `consult` review before shipping.
   - `_reserved_slots: set[int]` closes a second race a `consult`
     review caught: without reserving a slot the instant its
     `PrefillMessage` is sent, a SECOND `tick()` call (before the
     first grant's `enqueue_admission` arrives) could grant a
     different request onto the same physical slot.
   - `Rank1BatchedDecodeGlue.tick()` gains a matching `MSG_KIND_PREFILL`
     branch: reactively decodes the `PrefillMessage` and returns its
     own `PrefillGrant`. Rank 1 has no other path to ever decide
     "prefill this now" -- it only ever learns this from a grant
     `tick()` itself produced.

3. **`batch_generate.py` (`ExoBatchGenerator`) integration:**
   - The batched-decode eligibility check moved BEFORE prefill runs
     (was after). An eligible request's prefill now runs inside a
     deferred closure (`_DeferredPrefill`, `_submit_batched_decode_deferred`)
     registered in `_deferred_prefill_by_uid`, NOT executed inline in
     `submit()`.
   - `_step_batched_decode` now handles a non-None `PrefillGrant` from
     either rank's `tick()`: runs the deferred closure synchronously,
     still inside the same `step()` call, then folds the result in via
     `enqueue_admission`/`stage_local_cache`.
   - **Rank-1 grant-parking** (`_parked_prefill_grants`): closes a
     real, expected race where rank 0's `tick()` can produce a grant
     for a `request_id` rank 1's own `submit()` hasn't registered yet
     (two independent per-rank event loops consuming the SAME
     globally-ordered broadcast at different speeds -- not a bug, a
     genuine timing window). Resolved via a real state-machine
     completion (whichever side -- the grant or the registration --
     arrives second services the request), NOT a retry/sleep loop,
     per the standing root-cause-only rule.
   - Scope: only the ELIGIBLE/batched-decode path is fixed. The
     INELIGIBLE fallback (vision/tools/speculative-decode requests,
     or non-Pipeline sharding) still runs prefill inline, unchanged --
     explicitly scoped out, documented inline at the eligibility-check
     call site. A mixed eligible+ineligible concurrent-admission race
     is a known, real follow-up, not yet closed.

### Verification (real, not simulated)

- **The regression gate itself flipped from XFAIL to a genuine PASS**:
  `test_pp_admission_race_subprocess.py`'s
  `test_independent_per_rank_event_loops_do_not_desynchronize_the_wire`
  -- two real OS processes, real MLX ring transport, genuinely
  independent per-rank event loops with random per-rank jitter (not a
  lockstep test driver) -- confirmed XPASS against the fix (`1 xpassed
  in 15.27s`), then re-verified as a plain `PASSED` (`1 passed in
  15.35s`) after removing the now-unnecessary `xfail` marker. Run
  across 5 independent seeds every time; all clean.
- Full worker test suite: **252 fast tests + 14 slow (real-subprocess)
  tests, all passing**, including every pre-existing 2-process
  correctness test for this subsystem
  (`test_pp_batched_decode_glue_subprocess.py`,
  `test_pp_batched_decode_subprocess.py`,
  `test_pp_batched_correctness_harness.py`, `test_pp_metaframe.py`).
- `basedpyright`/`ruff` on every touched file: zero NEW errors
  introduced (diffed against a clean pre-change baseline; the
  pre-existing ~305 file-scope / ~9350 whole-repo basedpyright errors
  and 9 ruff findings are unrelated, untouched code and were confirmed
  identical before and after).
- `test_batch_generate_rank1_batched_decode_dispatch.py` rewritten
  (its old assertion -- "stage_local_cache called synchronously inside
  submit()" -- directly contradicted the new, correct architecture) to
  a composed two-phase test that drives the REAL grant-servicing
  method (`_run_deferred_prefill_for_grant`, the same one
  `_step_batched_decode` calls), so a miswired grant-to-staging path
  still fails it the way the original 2026-08-05 bug would have.

### Not yet verified (still needs the user's explicit go-ahead)

Nothing in this fix has run on the real 2-node cluster yet. Per this
doc's own standing reminder above, a real cluster relaunch/test needs
its own separate explicit go-ahead -- not implied by this write-up.
Before that attempt: re-enable `JACCL_TRACE_STEP=1` (or equivalent) so
a real trace can confirm the fix on real hardware, not just "no
crash," and follow the same crash → restore-to-known-good → diagnose
discipline used throughout the original campaign.

**UPDATE (2026-08-06, same day, later): this DID subsequently run on
real 2-node hardware** with explicit user go-ahead. Single-request PP
(the smoke-test baseline) confirmed clean every time
(`finish_reason=stop`, "Paris"). The admission-decision race itself
(this section's own subject) is genuinely closed. But the FIRST real
N=2 concurrent-request attempt surfaced FOUR MORE real bugs in the
adjacent eviction/slot-reuse code, all invisible until this fix made
N=2 slot-lifecycle reachable for the first time -- see the next
section for the full writeup, then "2026-08-06 finding: prefill
forward-pass race (NOT FIXED)" after that for a FIFTH, still-open bug
that blocks safely enabling `EXO_PP_BATCHED_DECODE=1` in production.

---

## 2026-08-06 follow-up: 4 real bugs in eviction+slot-reuse under N=2

All four found via the SAME real 2-process test (`_pp_glue_subprocess_worker.py`,
extended with a third request C admitted into a freshly-evicted slot)
during one focused debugging chain, immediately after the admission
race fix above first ran on real hardware and hit an eviction+reuse
scenario for the first time ever. Commit `ab68565a6`.

**Common root cause for all four**: `EXO_MAX_CONCURRENT_REQUESTS` was
ALWAYS forced to 1 under Pipeline sharding before this session's
admission-race work relaxed it to 2 for the batched-decode path. That
meant real eviction-then-slot-reuse (request A occupies slot 0, A
finishes and evicts, request C is admitted into slot 0 -- A's now-freed
slot) was structurally UNREACHABLE until this session. All four bugs
below are real, pre-existing latent defects that simply had zero
opportunity to manifest before N=2 concurrency became possible.

1. **Stuck-DRAINING mirror bug.** `RankOneMirror.build_evict_ack()`
   (`pp_scheduler_protocol.py`) existed, was fully correct, and had
   its own passing unit tests -- but nothing in PRODUCTION ever called
   it. `Rank1BatchedDecodeGlue.tick()`'s `MSG_KIND_EVICT` branch
   hand-constructed an `EvictAckMessage` directly instead of routing
   through `build_evict_ack`, so rank 1's own `RankOneMirror._slot_state`
   never transitioned back to FREE after an eviction -- it stayed
   permanently DRAINING. Invisible under
   `EXO_MAX_CONCURRENT_REQUESTS=1` (a slot could never be reused).
   Real symptom: `ProtocolViolationError: StepMessage ... schedules
   request_id=N on cache_slot=M which is DRAINING` the instant a real
   N=2 test tried to reuse a freed slot -- the module's own "#1
   corruption vector" fail-stop guard working exactly as designed.
   Fixed by routing both ranks' eviction-ack construction through
   `build_evict_ack` (added to `RankOneMirrorDriver` and
   `RankOneMirrorSession` as thin wrappers).

2. **Slot-number vs physical-cache-row-position bug.**
   `extract_request_cache` (via mlx-lm's `BatchKVCache.extract(idx)`)
   treats `idx` as a raw PHYSICAL array row -- but
   `merge_request_caches` builds the batched cache by enumerating
   occupied SLOTS in ascending slot-NUMBER order. Physical row
   position only equals slot number by coincidence when slot 0 is
   always occupied and no lower slot is ever evicted while a higher
   one survives -- exactly the precondition N=1 always satisfied
   trivially. Fixed via a new `_physical_position_for_slot` helper
   (both `BatchedDecodeSession` and `RankOneMirrorSession`) that maps
   a logical slot number to its actual enumeration position in
   `cache_router.occupied_slots()`, used everywhere `extract_request_cache`
   is called on a single slot.

3. **Extraction-vs-mutation ordering bug (2 instances).**
   `admit_request` and `on_evict_ack` extracted existing slot caches
   AFTER mutating the cache_router's occupancy (`assign_slot`/
   `release_slot`), so the CORRECT extraction logic from bug #2 read
   against occupancy that no longer matched what `batched_cache`
   physically held. Fixed by snapshotting existing caches BEFORE the
   router mutation in both methods (both `BatchedDecodeSession` and
   `RankOneMirrorSession`'s `admit_request`).

4. **Wire-protocol vs physical-cache ordering divergence (the most
   dangerous variant -- silently WRONG output, not a crash).**
   `SchedulerCore._active_batch_entries()` sorted `StepMessage.entries`
   by REQUEST_ID (`sorted(self._requests.items())`, dict key =
   request_id). But the physical `batched_cache` row order AND
   `BatchedDecodeSession.prepare_step()`'s own token-tensor row order
   are BOTH cache_slot-ordered -- a real physical constraint (fixed by
   `merge_request_caches`'s construction), not a convention that could
   be changed on that side instead. Under real slot reuse (a NEW,
   HIGHER request_id admitted into a LOWER, freshly-freed cache_slot)
   these two orders diverge. First caught by `prepare_step()`'s own
   defensive check (`BatchStepContext order (2, 3) does not match ...
   (3, 2)`); after fixing that ordering source, the SAME divergence
   resurfaced as silently WRONG decoded tokens for the surviving
   request (caught only by diffing against an independent golden
   serial-forward reference, since nothing crashed). Fixed by changing
   `_active_batch_entries()`'s sort key to cache_slot -- confirmed safe
   for every consumer of `StepMessage.entries`'s order (`RankOneMirror
   .validate_step`'s duplicate-slot check and `RankOneMirrorDriver
   .on_step_message`'s assign/advance loop both key off each entry's
   own fields, never positional index).

**Verification**: extended `test_pp_batched_decode_glue_subprocess.py`
(genuine 2-process test, not a mock) with request C admitted into A's
freshly-evicted slot, checked against an independent golden
serial-forward reference for ALL THREE requests -- this is now the
permanent regression gate for all four bugs. 252 fast + 14 slow tests
pass; zero new basedpyright/ruff errors vs baseline.

---

## 2026-08-06 finding: prefill forward-pass race (root cause + design)

**STATUS (2026-08-06, later): FIXED at the code level -- see the new
"2026-08-06 fix: prefill forward-pass race (PrefillReadyMessage)"
section at the bottom of this file for the full implementation.** The
diagnosis below is left unmodified as the historical record of how
this was found. This was the reason `EXO_PP_BATCHED_DECODE=1` stayed
UNSAFE for production despite everything above being fixed and
verified. Found
on the SECOND real N=2 hardware attempt (after bugs 1-4 above were
fixed, committed as `ab68565a6`, and re-deployed) -- two fresh,
cold-cache concurrent requests crashed the cluster again, with a NEW
error signature not seen before:

- Rank 0 (the node running its own real prefill forward pass via
  `_run_deferred_prefill_for_grant` → `deferred.run_prefill()`):
  crashed mid-`send_metaframe` with `[jaccl] reliable_all_reduce_v2
  deadline`.
- Rank 1 (the peer): crashed with `SchedulerWireProtocolError:
  recv_header: version mismatch -- received 3, this rank expects 1`
  -- it called `pp_scheduler_wire.recv_header()` (expecting the
  scheduler-wire protocol, version 1) and instead read raw
  `pp_metaframe` bytes (`METAFRAME_PROTOCOL_VERSION=3`) off the wire.

**Root cause (confirmed via reading `_run_deferred_prefill_for_grant`,
`pp_batched_decode_glue.py`'s `tick()`, and both node logs -- not
speculation):**

The admission-race fix (section above) correctly single-writer-gates
the DECISION to prefill (`PrefillMessage`/`PrefillGrant`, via
`tick()`). But it does NOT gate the ACTUAL PREFILL FORWARD PASS that
runs immediately afterward. `_run_deferred_prefill_for_grant` calls
`deferred.run_prefill()` synchronously right after receiving a grant
-- and that forward pass internally uses `pp_metaframe.py`'s
`send_metaframe`/`recv_metaframe`, a COMPLETELY SEPARATE wire
transport from the scheduler-wire protocol, for the per-layer
hidden-state handoff (Phase 1's batched-decode layers only wrap the
DECODE step; prefill still runs through the legacy single-request
metaframe layer code path, unbatched).

Both ranks are SUPPOSED to run this forward pass in lockstep (this is
the SAME mechanism that already works correctly for single-request
prefill) -- but there's a structural gap in HOW rank 1 gets there:
`_run_deferred_prefill_for_grant`'s own rank-1 "grant PARKING"
mechanism (`_parked_prefill_grants`, built specifically to handle the
real, expected timing skew between two independently-scheduled OS
processes' `submit()`/`tick()` calls) lets rank 1 receive a
`PrefillGrant` and DEFER running its own matching forward pass --
without any way to tell rank 0 it did so. Rank 0, having decided to
grant the prefill, proceeds UNCONDITIONALLY and immediately into its
own real forward pass, sending real metaframe activation bytes onto
the wire. If rank 1 parked instead of running its matching side (a
real, not-rare timing window), rank 1's NEXT wire read is a
`scheduler-wire recv_header()` call from its own next `tick()`
iteration -- which reads rank 0's metaframe bytes and misparses them
as a `SchedulerWireProtocolError`.

**Consult-reviewed fix direction (not yet implemented):** treat
`PrefillMessage` as an in-band CHANNEL-MODE SWITCH, not merely a
notification. Once a rank consumes a `PrefillMessage` off the wire,
that rank's protocol state machine must commit to running its
matching metaframe-mode forward pass to completion before issuing any
further `recv_header()` call on that channel -- i.e. the parking
mechanism as currently designed (park now, run later, out of band) is
itself the structural defect; parking a `PrefillGrant` without
immediately running its metaframe side breaks the lockstep invariant
the transport requires. Since the channel is a single ordered FIFO
stream per rank pair, rank 0 does not need proof rank 1 is ready
before it starts sending metaframe bytes -- they simply queue in the
transport; rank 1's state machine, once it eventually processes the
`PrefillMessage` (whenever its own event loop gets there), must treat
that as an unconditional "run the matching forward pass now, do not
return to scheduler-wire in between" commitment, never a decision it
can defer to later. This likely means `_run_deferred_prefill_for_grant`
on rank 1 needs to become a BLOCKING call inline within the SAME
`tick()`/wire-consuming call that received the `PrefillMessage` --
never park-and-return -- which may require restructuring how
`Rank1BatchedDecodeGlue.tick()` and `_run_deferred_prefill_for_grant`
compose (currently two separate calls from `_step_batched_decode`,
with the parking dict bridging a gap between them).

**Explicitly NOT yet done:**
- No fix implemented -- this section is a diagnosis + fix-direction
  writeup only, not a patch.
- No new regression test exists for this specific race (the existing
  2-process glue test predates the parking mechanism being exercised
  by a genuine cross-process timing race -- it always runs both ranks'
  admissions in a controlled, non-racy order).
- Real hardware re-verification after any fix attempt needs its own
  fresh explicit go-ahead, same as every other real-cluster step in
  this campaign.

**Practical guidance for whoever picks this up next:** `EXO_PP_BATCHED_DECODE=1`
must stay OFF (the `start_cluster.sh` default) in any deployment until
this is closed. `EXO_PP_METAFRAME=1` alone (single-request PP,
metaframe transport, no batched-decode) is unaffected by this bug and
has been repeatedly verified clean on real hardware throughout this
session.

---

## 2026-08-06 fix: prefill forward-pass race (PrefillReadyMessage)

Implements the "channel-mode switch" direction diagnosed above, in a
concrete NACK+retry shape (a `consult` review confirmed the
alternative -- rank 1 doing a bounded, blocking POLL for its own
local registration -- would structurally deadlock: rank 1's task
dispatch pipeline and its `tick()` call both run on the SAME
main-thread runner loop, so a wait for registration inside `tick()`
can never be satisfied by that SAME thread's own later work). Commit
`f4e6972a9`.

### What shipped

1. **New wire message, `PrefillReadyMessage`
   (`MSG_KIND_PREFILL_READY=5`, `pp_scheduler_wire.py`/
   `pp_scheduler_protocol.py`)** -- rank 1's reply to every
   `PrefillMessage`, ALWAYS sent synchronously and unconditionally
   from inside `tick()`, never withheld: `ready=True` if
   `mark_prefill_registered(request_id)` was already called (rank 1's
   own local `_DeferredPrefill` genuinely exists and is about to run),
   `ready=False` (a real, expected NACK -- not an error) otherwise.

2. **`Rank1BatchedDecodeGlue` extended**: new `mark_prefill_registered`
   (pure local set-membership update, zero wire I/O, called from
   `submit()` the instant registration happens -- same "just data,
   caller does the real work" shape as every other `*_registered`/
   `*_staged` method in this glue layer) and `_registered_request_ids`
   tracking set. `tick()`'s `MSG_KIND_PREFILL` branch now ALWAYS
   replies before returning -- either a real `PrefillGrant` (on ready)
   or `None` (on NACK, no grant this tick).

3. **`Rank0BatchedDecodeGlue` extended**: after sending a
   `PrefillMessage`, `tick()` now BLOCKS on `recv_prefill_ready_message`
   before returning a `PrefillGrant` to the caller -- this is what
   makes it safe for the caller to immediately follow a grant with
   real metaframe wire traffic; rank 1 has JUST confirmed readiness on
   the SAME wire. On NACK: does NOT retry synchronously (the caller
   gets an empty tick, exactly as if nothing happened this cycle) --
   the SAME pending request stays at the front of `_pending_prefill`
   (not popped) with its slot still reserved, so a LATER real `tick()`
   call retries it, giving rank 1's own event loop a genuine,
   unblocked window to process its pending `submit()` work in between
   attempts. Bounded at `_PREFILL_READY_MAX_RETRIES=50` consecutive
   NACKs before failing loud (a real stall on rank 1 -- crashed, hung,
   or a genuine registration bug -- rather than an expected timing
   race).

4. **Removed**: rank 1's `_parked_prefill_grants` mechanism
   (`ExoBatchGenerator`) -- the structural defect this whole fix
   closes. `_run_deferred_prefill_for_grant`'s missing-`_DeferredPrefill`
   case is now a hard `GlueError` on EITHER rank (previously only
   rank 0 failed loud there; rank 1 silently parked) -- since the
   ack/NACK handshake means rank 1 now NEVER returns a grant for an
   unregistered request in the first place, hitting this path is
   structurally impossible on both ranks, not just rank 0.

### Verification (real 2-process tests, not simulated)

- `test_pp_admission_race_subprocess.py` (the SAME genuinely-
  independent-per-rank-event-loop regression test that gates the
  original admission-race fix) required the identical fix on its own
  worker script: rank 1 now calls `mark_prefill_registered` on its OWN
  independent random schedule (mirroring rank 0's own independently-
  randomized `enqueue_prefill` timing, deliberately NOT synchronized
  with it) -- exercising the SAME kind of real timing race the
  ack/NACK handshake exists to resolve, rather than trivially always
  registering before rank 0's grant attempt. Confirmed via a captured
  real trace that NACK-then-retry-then-success genuinely occurred:
  seed 41's trace shows `PrefillMessage` sent at `it=3` (implicitly
  NACK'd -- no `run_grant_prefill_b` followed), then
  `mark_prefill_registered_b` at `it=4`, then a successful grant+run at
  the SAME `it=4` tick. Passes consistently across repeated runs (5
  seeds each).
- 252 fast + 14 slow tests pass; zero new basedpyright/ruff errors vs
  the established baseline (305 file-scope / 9348 whole-repo
  basedpyright, 9 ruff -- all pre-existing, confirmed identical
  before/after via `git stash` diff).

### Not yet verified (needs its own fresh explicit go-ahead)

This fix has NOT run on real 2-node hardware yet. The prior two N=2
hardware attempts each surfaced a genuinely new bug this session
hadn't seen in local testing -- there is real reason to expect a
**UPDATE (2026-08-06, same day, later): this DID run on real N=2
hardware, with explicit user go-ahead.** Result: the PrefillReadyMessage
fix worked EXACTLY as designed -- no crash, no deadlock, no
`SchedulerWireProtocolError`. The fail-stop guard fired cleanly
instead (`GlueError: rank 1 NACK'd PrefillMessage ... 50 consecutive
times ... Refusing to retry forever`), confirming the ack/NACK
handshake behaves correctly under real RDMA/jaccl transport. That
clean failure led directly to discovering a SIXTH, architecturally
distinct bug -- see "2026-08-06 finding: cross-rank eligibility
divergence (is_prefix_cache_hit)" below.

---

## 2026-08-06 finding: cross-rank eligibility divergence (is_prefix_cache_hit)

**Status: found on real N=2 hardware, NOT yet fixed.** Unlike bugs
1-5, this one did NOT crash or deadlock the cluster -- the
PrefillReadyMessage fail-stop guard (bug #5's own fix) caught it
cleanly and loudly. This is a genuinely different bug class from
everything above: not a wire-ordering/timing race, but a
**cross-rank disagreement about which requests are even eligible**
for the batched-decode path in the first place.

### Real repro (exact scenario from the hardware test)

Two concurrent, cold-start, DIFFERENT-topic requests: "Count from 1
to 5." (request A) and "Name 3 colors." (request B), both using the
same chat template. A admits and completes its prefill first (no
concurrency issue there). B's `submit()` then runs, on BOTH ranks,
independently.

**Confirmed via real node logs from both ranks**: after A's prefill,
B's KV-prefix-cache lookup found a 2-token shared prefix with A (the
`<|begin_of_sentence|><|User|>` chat-template boilerplate both
prompts share) -- `[shared_prefix=2 tok (16%)]`, logged identically
on BOTH ranks. That means `local_hit_length=2 > 0` on both ranks,
which (per `is_eligible_for_batched_decode`'s own documented rule)
makes `is_prefix_cache_hit=True` -- and a prefix-cache-hit request is
explicitly, deliberately INELIGIBLE for the batched-decode path
(`pp_batched_decode_eligibility.py`'s own comment: "KVPrefixCache's
serial snapshot/restore lifecycle vs BatchedCacheRouter's slot-based
lifecycle has not been analyzed for compatibility").

Rank 0's `tick()` nonetheless sent a real `PrefillMessage` for
request B and retried it 50 times before failing loud -- meaning rank
0 believed B was ELIGIBLE (`enqueue_prefill` was called, i.e.
`eligibility.eligible` was `True` on rank 0's side) while rank 1
never called `mark_prefill_registered` for it (consistent with rank
1 believing B was INELIGIBLE and routing it through the old serial
fallback instead). **The two ranks reached different eligibility
verdicts for the SAME logical request.**

### Root cause (partially confirmed, one open question)

The `is_prefix_cache_hit` eligibility input is computed from each
rank's own SEPARATE `KVPrefixCache` object -- a real, physically
distinct per-process Python data structure (each rank owns a
disjoint slice of the model's LAYERS, but each maintains its OWN
prefix-cache trie tracking the same logical token sequences). The
existing `pipeline_agree_prefix_hit_length()` function's own
docstring already assumes this can diverge in the general case
("Each rank's independently-maintained KVPrefixCache SHOULD reach an
identical local hit-length ... in the common case ... This function
verifies that with a cheap min+max reduce rather than assuming it")
-- but the 2026-08-06 admission-race fix's own follow-up
optimization (commit `f705c9abe`, "skip cross-rank prefix-hit
agreement wire call when local hit_length=0") means the WIRE
AGREEMENT that would normally catch and reconcile this divergence is
ONLY skipped when a rank's own `local_hit_length` is exactly 0 --
i.e. it fires correctly for genuinely-cold requests, but does
nothing to help two ranks whose local `KVPrefixCache` states have
ALREADY diverged (e.g. one rank folded A's KV state into its trie
before evaluating B's eligibility, the other hadn't yet) BEFORE the
eligibility check runs, since eligibility is computed from
`local_hit_length` directly, not from the (already fixed, real) wire
agreement's result.

There is ALSO a pre-existing, still-open diagnostic comment in
`cache.py`'s `get_kv_cache()` (added 2026-07-23, gated on
`EXO_PREFIX_CACHE_DIAG=1`, never removed): *"Diagnostic: root-causing
why rank 1 (PP mode) never reports a local hit-length while rank 0
does, on an exact-repeat prompt."* This confirms cross-rank
KVPrefixCache divergence is a REAL, PRE-EXISTING, independently
already-suspected issue in this codebase -- not something newly
introduced by this session's admission-race work. This session's
work is what first made it possible for that divergence to actually
manifest as an observable failure (N=2 concurrent admission was
structurally unreachable before `EXO_MAX_CONCURRENT_REQUESTS` was
relaxed from 1 to 2), the same pattern as bugs #1-4.

**Open question, NOT yet resolved**: is the observed log line
(`[shared_prefix=2 tok (16%)]`, logged IDENTICALLY on both ranks) the
lookup that FED eligibility (`get_kv_cache()`'s own `_longest_prefix_match`
call, BEFORE prefill), or the WRITE-BACK that happens AFTER prefill
completes (`add_kv_cache()`'s own diagnostic `_longest_prefix_match`
call, logged for a different purpose -- cross-session dedup
visibility, per that function's own comment)? The log line's format
(`shared_prefix=N tok`) appears in both call sites and this session's
log archaeology could not conclusively disambiguate which one fired
inside the eligibility-determining window versus which one is
post-hoc write-back telemetry, given both ranks' `.log.prev` files
only capture INFO-level output (no `EXO_PREFIX_CACHE_DIAG=1` was set
for this run). If it's the write-back line, the actual pre-eligibility
`local_hit_length` values that diverged are NOT directly visible in
the captured logs and this write-up's mechanism, while consistent
with all observed evidence, is a well-supported HYPOTHESIS, not a
byte-for-byte confirmed trace.

### Candidate fix direction (not attempted -- needs a fresh session)

Per a `consult` review: this needs the SAME class of fix as the
original admission-decision race (bug #1) -- fold the ELIGIBILITY
decision itself into the single-writer `tick()`-gated channel, so
rank 0 decides eligibility once (from its own local state) and TELLS
rank 1, rather than each rank independently computing eligibility
from local state that can genuinely diverge. This is real,
non-trivial protocol work (comparable in scope to bug #1's fix, not
a quick patch), with a real open question the consult flagged: if
rank 0 tells rank 1 "this request IS eligible" but rank 1's own local
KVPrefixCache genuinely has a real prefix hit for it, what does rank
1 do with that real local state when forced down the batched-decode
path against its own cache's evidence? That interaction with the
EXISTING `pipeline_agree_prefix_hit_length()` wire-agreement
mechanism (and this session's own `f705c9abe` skip-when-local-0
optimization) needs to be traced through carefully -- not attempted
this session, deliberately, to avoid starting fresh protocol-layer
work at the end of an already-long, multi-fix session.

### Explicitly NOT yet done

- No fix implemented -- this section is a diagnosis + fix-direction
  writeup only, matching how bug #5 (prefill forward-pass race) was
  first documented before being fixed in a later pass.
- The open question above (which `shared_prefix` log line was
  captured) is NOT resolved -- whoever picks this up next should
  re-run with `EXO_PREFIX_CACHE_DIAG=1` on both nodes to get an
  unambiguous, rank-tagged trace of `local_hit_length` at the ACTUAL
  eligibility-check call site, before assuming this write-up's
  mechanism is 100% correct.
- No new regression test exists for this specific divergence (the
  2-process test suite's requests are all either genuinely disjoint
  cold-cache prompts with no shared chat-template ambiguity, or the
  scenario doesn't naturally arise in the smaller synthetic prompts
  those tests use).
- The cluster was restored to known-good (single-request PP,
  verified with a real "Paris" completion) immediately after this
  finding -- no corrupted or crashed state was left running.

### Practical guidance

`EXO_PP_BATCHED_DECODE=1` must stay OFF (the `start_cluster.sh`
default) in any production deployment. It is a REAL improvement over
the pre-session state (the original hard deadlock is closed, and this
bug fails loud/clean instead of corrupting or hanging), but N=2
concurrent requests sharing even a trivial chat-template prefix (which
is the COMMON case for any two requests using the same model/template)
can still hit a clean but user-visible failure. Single-request PP
(`EXO_PP_METAFRAME=1` alone, `EXO_PP_BATCHED_DECODE` unset) remains
fully unaffected and repeatedly verified working throughout this
entire multi-session campaign.

---

## 2026-08-06 archaeology: which `shared_prefix` log line was captured

Resolves the "open question, NOT yet resolved" from the section above,
via read-only code archaeology (no hardware rerun needed).

**Definitive finding: the captured `[shared_prefix=2 tok (16%)]` line
came from `add_kv_cache()`'s post-prefill write-back diagnostic
(`cache.py:872`), NOT from `get_kv_cache()`'s pre-eligibility lookup.**

- `add_kv_cache()` (defined `cache.py:812`) emits, unconditionally at
  `logger.info` level (no env-var gate), a line of the exact form
  `[shared_prefix={shared_depth} tok ({share_pct}%), unique=... tok,
  trie_leaves=..., trie_bytes=...]` at `cache.py:872-876`. A
  repo-wide grep confirms this is the ONLY call site anywhere in
  source that can produce the `shared_prefix=` token.
- `get_kv_cache()` (defined `cache.py:1165`) has its own diagnostic
  lines (`[PREFIX_DIAG rank=...] ...`, `cache.py:1186-1189` and
  `1202-1205`) but they use a DIFFERENT format (no `shared_prefix=`
  substring) and are gated behind `EXO_PREFIX_CACHE_DIAG=1`
  (`cache.py:1183`, `if _diag:` at 1185/1201) -- silent in the
  captured run, since no diag flag was set.
- Call ordering in `batch_generate.py` confirms `add_kv_cache()` runs
  strictly AFTER the eligibility-determining lookup: `get_kv_cache()`
  at line ~2018 feeds `is_eligible_for_batched_decode()` at line
  ~2188, prefill runs, THEN `add_kv_cache()` runs at line ~4230 as a
  post-hoc cross-session-dedup write-back.

**Conclusion**: the write-up's hypothesis in the section above was
correct -- the pre-eligibility `local_hit_length` values that actually
diverged across ranks are NOT recoverable from the existing INFO-only
log capture. A fresh `EXO_PREFIX_CACHE_DIAG=1` hardware rerun would
have been the only way to see them directly. **This is now moot**: the
fix below (2026-08-06) eliminates `is_prefix_cache_hit` from the
eligibility computation entirely, so there is no longer any
per-rank-divergent value to trace in the first place.

---

## 2026-08-06 fix: eliminate cross-rank eligibility divergence (drop is_prefix_cache_hit)

**Status: FIXED at the code level, NOT yet hardware-verified.** Closes
bug #6, the last of the six real bugs found this campaign.

### Design note: this is NOT the wire-message fix originally scoped

The "candidate fix direction" section above (and the original
next-session handoff) proposed folding the eligibility DECISION into
the single-writer `tick()`-gated wire channel -- rank 0 decides once
and tells rank 1, mirroring the bug #1 (`PrefillMessage`) pattern. Two
`consult` reviews during this session's implementation concluded a
SIMPLER, strictly-better fix exists: **eliminate the divergent input
entirely rather than building a channel to coordinate over it.**
Reasoning:

- Wire-communicating rank 0's verdict to rank 1 introduces an ordering
  dependency (rank 1 must block or defer until the verdict arrives) --
  exactly the class of complexity that produced bugs #1 and #5 in the
  first place.
- `is_prefix_cache_hit` was the ONLY eligibility input that could ever
  diverge across ranks. Every other input
  (`has_images`/`has_tools`/`uses_speculative_decode`/
  `sharding_is_pipeline`/`batched_decode_enabled`) is either derived
  from the request itself (both ranks receive the identical request
  via the existing broadcast/queue) or static, cluster-wide-identical
  startup config (`EXO_PP_BATCHED_DECODE` env, static shard topology).
  Removing the one divergence-capable parameter makes both ranks
  provably compute the IDENTICAL verdict with ZERO wire coordination
  -- strictly simpler than adding a message to reconcile a value that
  didn't need to exist as an eligibility input at all.
- The existing design already treated ANY non-zero prefix hit as
  making a request INELIGIBLE for batched-decode (the
  `BatchedCacheRouter` vs `KVPrefixCache` lifecycle-incompatibility
  concern documented in the original eligibility module). So dropping
  the input doesn't change behavior for genuinely cache-benefiting
  requests being pulled INTO batched-decode -- it only removes the
  possibility of DIVERGENCE, at the cost of batched-eligible requests
  never getting even a trivial chat-template-prefix cache hit while
  `EXO_PP_BATCHED_DECODE=1` is active (an intentional, accepted
  tradeoff -- see the code comment at the fix site for the full
  reasoning).

### What shipped

1. **`pp_batched_decode_eligibility.py`**: `is_prefix_cache_hit`
   removed from `is_eligible_for_batched_decode()`'s signature
   entirely, along with its corresponding ineligibility branch/reason.
   A new docstring paragraph states the invariant explicitly:
   batched-decode eligibility is a PURE function of (request payload,
   static startup config) -- NEVER of per-rank mutable state like
   `KVPrefixCache`. This is now enforced structurally by the function
   signature itself, not by a runtime flag check.

2. **`batch_generate.py` (`ExoBatchGenerator.submit()`)**: the
   batched-decode ROUTING decision moved to run BEFORE any
   `KVPrefixCache` lookup (previously the eligibility check ran after
   `get_kv_cache()`, consuming its `local_hit_length` result). Seed
   and sampler construction were hoisted above the routing decision so
   both the batched and serial paths can share one construction
   (avoids duplicating that block). If a request is batched-eligible,
   it is dispatched to the deferred-prefill batched path with a fresh
   cache -- `get_kv_cache()` is never called, so the trie is never
   read OR mutated for that request. If a request is shape-ineligible
   (or batched-decode is inactive), it falls through UNCHANGED to the
   existing serial path, which is now the ONLY place
   `get_kv_cache()`/`pipeline_agree_prefix_hit_length()` run --
   preserving 100% of existing serial-path prefix-cache behavior for
   genuinely-ineligible requests (vision/tools/speculative-decode/
   non-Pipeline-sharding).

3. **`add_kv_cache()` fold-on-completion**: left UNCHANGED. Completed
   batched-decode requests still fold their tokens into
   `KVPrefixCache` on completion, so a LATER shape-ineligible request
   (routed to the serial path) can still benefit from cache reuse
   against a request that happened to go through batched-decode.
   Since nothing reads the cache for batched-eligible requests
   anymore, there is no divergence risk from keeping this fold --
   serial-path lookups already tolerate/reconcile via the existing
   `pipeline_agree_prefix_hit_length()` wire-agreement mechanism,
   unchanged by this fix.

4. **`test_pp_batched_decode_eligibility.py`**: two new regression
   tests. `test_signature_has_no_per_rank_mutable_state_input` asserts
   via `inspect.signature()` that the function's parameter set is
   EXACTLY the request-derived/static-config allowlist -- forbidding
   `is_prefix_cache_hit` and lookalike per-rank-state parameter names
   from ever being re-added without this test failing first.
   `test_two_simulated_ranks_compute_identical_verdicts_regardless_of_per_rank_state`
   proves by construction that two simulated ranks calling the
   function with identical request/config inputs get identical
   verdicts, across every eligibility branch. Pre-existing tests
   updated to drop the now-removed `is_prefix_cache_hit` kwarg and
   disqualifying-condition case.

### Verification (real, not simulated)

- `uv run basedpyright` on all touched files: 305 errors (matches the
  established whole-repo baseline exactly, confirmed via `git stash`
  diff -- zero new errors introduced).
- `uv run ruff check` on all touched files: 9 errors (matches
  baseline exactly; `git stash` diff confirms the only differences
  are line-number shifts from the code insertion, not new findings).
- `uv run pytest -m ""` across the full relevant subsystem test set --
  `test_pp_admission_race_subprocess.py`,
  `test_pp_batched_decode_subprocess.py`,
  `test_pp_batched_decode_glue_subprocess.py`, `test_pp_metaframe.py`,
  `test_pp_batched_correctness_harness.py`,
  `test_batch_generate_rank1_batched_decode_dispatch.py`,
  `test_batch_generate_batched_decode_flag_off_smoke.py`, and
  `test_pp_batched_decode_eligibility.py` (including both new tests)
  -- **40 passed, 0 failed**. This includes the genuine 2-OS-process,
  independent-event-loop subprocess tests (`test_pp_admission_race_
  subprocess.py` and friends), not just in-process unit tests.
- Broader `uv run pytest src/` run (excluding known-pre-existing
  unrelated collection errors in `src/exo/download/tests` and
  `test_routing_concurrency.py`, confirmed pre-existing via the same
  `git stash` diff technique) showed no new failures attributable to
  this change.

### Not yet verified (needs its own fresh explicit go-ahead)

This fix has NOT run on real 2-node hardware yet. Per this doc's own
standing reminder, a real cluster relaunch/test needs its own separate
explicit go-ahead -- not implied by this write-up. `EXO_PP_BATCHED_
DECODE=1` must stay OFF in any deployment (the cluster should remain
in its current safe known-good config: single-request PP,
`EXO_PP_BATCHED_DECODE` unset) until a real N=2 hardware run confirms
this fix behaves as designed -- specifically, that two genuinely
concurrent, cold-start, DIFFERENT-topic requests sharing a
chat-template prefix (the exact bug #6 repro scenario) now both reach
the IDENTICAL eligibility verdict and no longer trigger the
PrefillReadyMessage NACK-retry fail-stop guard.

**UPDATE (2026-08-06, later still): this DID run on real N=2
hardware, with explicit user go-ahead -- and it crashed. Bug #6's own
fix worked as designed (no eligibility-divergence symptom observed),
but N=2 is still NOT safe.** See the next section for the full
writeup.

---

## 2026-08-06 finding: N=2 still crashes after bug #6's fix (bug #7, root cause NOT yet identified)

**Status: found on real N=2 hardware, root cause NOT yet identified.**
Bug #6's fix (dropping `is_prefix_cache_hit` from the eligibility
computation) is verified correct for what it touches -- the specific
divergence symptom it was designed to close (rank 0 believing a
request eligible while rank 1 believes it ineligible, producing a
PrefillReadyMessage 50x-NACK fail-stop) did NOT occur on this run.
But the underlying goal -- N=2 genuinely concurrent requests working
cleanly -- is still not achieved. Two DIFFERENT crash signatures were
observed across two back-to-back attempts, and neither points at
bug #6's own code.

### Repro (identical to bug #6's own repro scenario)

Cluster launched with the exact bug #6 hardware-test config:
```bash
DSV4_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-0731 DSV4_SHARDING=Pipeline \
EXO_PP_METAFRAME=1 EXO_PP_BATCHED_DECODE=1 \
EXO_SPECULATIVE=0 EXO_DSV4_DSPARK=0 EXO_DSV4_MTP=0 \
./start_cluster.sh
```
Both nodes confirmed on commit `5588d674a` (includes bug #6's fix).
Single-request smoke test clean (`finish_reason=stop`, "Paris") before
each concurrent attempt, per standing discipline. Two genuinely
concurrent, cold-start, DIFFERENT-topic requests fired via a
`ThreadPoolExecutor(max_workers=2)` against `/v1/chat/completions`:
`"Count from 1 to 5."` and `"Name 3 colors."` -- the identical repro
from bug #6's own finding.

### Attempt 1: `EXO_DSV4_BATCHED_PREFILL=1` (default) -- crashed after ~18s

Both requests returned HTTP 500. Rank 0 (node2) log shows
`Rendezvous batched 2 concurrent tasks (window=200ms)` followed by
`Starting batched prefill: B=2 max_L=10 lengths=[10, 7]` --
`EXO_DSV4_BATCHED_PREFILL` (a separate, older, always-on-by-default
mechanism in `runner.py`'s `handle_generation_tasks` that rendezvous-
batches concurrent requests into a single `prefill_batched()` call
BEFORE the PP batched-decode admission machinery ever sees them --
unrelated to and pre-dating bugs #1-#6's entire scope) took the two
concurrent requests down its own batched-prefill path over the
metaframe transport.

Rank 1 (node1) crashed with the **exact bug #5 error signature**:
```
SchedulerWireProtocolError: recv_header: version mismatch -- received 3,
this rank expects 1. Both ranks must run identical exo builds; refusing
to guess at a compatible decoding.
```
via traceback: `runner.py handle_generation_tasks` -> `batch_generate.py
step()` -> `_step_batched_decode()` -> `self._batched_decode_rank1_glue
.tick(self.model)` -> `pp_scheduler_wire.py recv_header()` raises.

Rank 0 (node2) simultaneously deadlined on its NEXT decode collective:
`[jaccl-v2] DEADLINE rank=0 call_id=533 all_recv=0/1 chunks_posted=1
small=1 peer_in_call=0` -> `[jaccl] reliable_all_reduce_v2 deadline --
clean re-place` -> in-place reconnect attempted -> reconnect failed
(`[jaccl] Recv failed: peer closed connection (EOF) fd=52
remaining=4`) -> propagated as a re-place.

`peer_in_call=0` on rank 0's deadline confirms rank 1 never entered
that collective -- rank 1's crash is causally FIRST, rank 0's deadline
and EOF are downstream symptoms of rank 1 dying, not an independent
failure.

**Working hypothesis (NOT confirmed by trace, no `JACCL_TRACE_STEP`
enabled)**: rank 0 took `EXO_DSV4_BATCHED_PREFILL`'s rendezvous path
for the two concurrent requests -- a genuinely batched prefill (B=2)
over the metaframe transport, entirely outside the PP batched-decode
`tick()`-gated channel bugs #1-#6 were scoped around -- while rank 1
was still polling that scheduler-wire channel via `tick()`. Rank 1's
`recv_header()` on that channel then ate rank 0's batched-prefill
metaframe bytes and misparsed them as a scheduler-wire header. This
would be the SAME class of decode/prefill wire-framing collision bug
#5 closed for the `PrefillMessage`/`PrefillReadyMessage` path
specifically -- but via a THIRD call path (`EXO_DSV4_BATCHED_PREFILL`'s
rendezvous mechanism) that bug #5's fix never touched and was never in
scope for.

### Attempt 2: `EXO_DSV4_BATCHED_PREFILL=0` -- STILL crashed, differently, after ~2.3s

Relaunched with `EXO_DSV4_BATCHED_PREFILL=0` added on top of the same
config, specifically to test the Attempt-1 hypothesis. Confirmed via
`ps` on the running process that the env var took
(`EXO_DSV4_BATCHED_PREFILL=0` present, `EXO_PP_BATCHED_DECODE=1` still
active). Single-request smoke test clean again before the retry.

**The hypothesis was WRONG, or at best incomplete**: the identical
concurrent repro STILL failed -- both requests HTTP 500 again -- but
in ~2.3s this time (vs ~18s in Attempt 1), and with a DIFFERENT error
signature. Rank 1 (node1) crashed inside the SAME call path
(`_step_batched_decode()` -> `Rank1BatchedDecodeGlue.tick()` ->
`recv_header()`) but this time at the `mx.eval(header)` line itself,
with a raw transport-level error, not a parsed-header version
mismatch:
```
RuntimeError: [jaccl] Recv failed with errno=54 n=-1 fd=53 remaining=16
flags=0x2 nonblock=0
```
Rank 0 (node2) hit the identical raw error
(`jaccl transport fault in generator.step(): [jaccl] Recv failed with
errno=54 ...`) at essentially the same moment, attempted an in-place
reconnect, and the runner was re-placed. Both nodes self-healed
cleanly (new runner spun up automatically); single-request smoke test
after this crash was clean (`finish_reason=stop`, "Paris") with no
manual intervention needed beyond the automatic re-place.

errno=54 is `ECONNRESET` on macOS/BSD -- a raw connection reset, not
a framing/parse-level mismatch. This is NOT the same failure mode as
Attempt 1's `SchedulerWireProtocolError`, which strongly suggests
`EXO_DSV4_BATCHED_PREFILL` was NOT the (sole) interacting factor --
disabling it changed the race's timing/outcome but did not close it.

### What this rules out

- **NOT a regression from bug #6's own diff.** Bug #6's fix touches
  only `pp_batched_decode_eligibility.py` and the routing/ordering in
  `batch_generate.py`'s `submit()` -- neither crash traceback passes
  through that code. Both crashes originate inside
  `Rank1BatchedDecodeGlue.tick()`'s `recv_header()` call, in code bug
  #6 never touched.
- **NOT (solely) `EXO_DSV4_BATCHED_PREFILL`.** Disabling it changed
  the crash's timing and error signature but did not prevent a crash.
  Either it's a secondary contributing factor (removing it removes
  ONE source of extra wire traffic but not the root race), or it was
  never the real cause and Attempt 1's specific error signature was
  coincidental.
- **NOT an eligibility-divergence symptom.** Neither crash produced
  bug #6's specific NACK-storm fail-stop (`GlueError: rank 1 NACK'd
  PrefillMessage ... 50 consecutive times`). Bug #6's fix appears to
  be doing its job; the cluster is failing for a DIFFERENT reason
  underneath it.

### What's still open

- **Root cause NOT identified.** Both crashes point at the same
  general SEAM (decode-step `tick()` polling on `pp_scheduler_wire`
  vs SOME other traffic hitting the same rank-pair transport at the
  same time) that bugs #1, #5, and now this finding all share -- but
  WHAT the other traffic is in Attempt 2 (with `EXO_DSV4_BATCHED_
  PREFILL=0` ruled out as sole cause) is not yet identified.
- **No `JACCL_TRACE_STEP=1` or `EXO_PREFIX_CACHE_DIAG=1` was enabled**
  for either attempt -- both this finding's root cause AND the
  still-unresolved bug #6 log-line question from the earlier
  archaeology section would benefit from a fresh hardware run with
  full tracing enabled.
- **Two visibly different crash signatures from what should be the
  identical repro** (version-mismatch parse error vs raw ECONNRESET)
  suggests either a genuine timing-dependent race with multiple
  possible failure surfaces, or two distinct bugs both reachable from
  the same N=2 concurrent-admission scenario. Not yet distinguished.
- Cluster was restored to safe known-good (single-request PP,
  `EXO_PP_BATCHED_DECODE` unset) after each crash, verified clean with
  a real "Paris" completion each time -- no corrupted or crashed state
  left running at any point.

### Practical guidance

`EXO_PP_BATCHED_DECODE=1` must stay OFF (the `start_cluster.sh`
default) in any deployment. Bug #6's fix, while itself correct, has
NOT resolved N=2 concurrency -- there is at least one more real,
unidentified bug in this path. Single-request PP
(`EXO_PP_METAFRAME=1` alone, `EXO_PP_BATCHED_DECODE` unset) remains
fully unaffected and repeatedly verified working, including
immediately after both of today's crashes.

