# Batched-decode Phase 1: N=2 concurrent admission — handoff (2026-08-05)

**STATUS UPDATE (2026-08-06): FIXED.** The admission race documented
below is closed. See the new section at the bottom of this file,
"2026-08-06 fix: in-band PrefillMessage admission signal", for the
full implementation summary, verification evidence, and what remains
explicitly out of scope. The original handoff content is left
unmodified below for the historical record of the investigation.

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

