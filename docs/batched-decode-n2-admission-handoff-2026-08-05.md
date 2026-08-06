# Batched-decode Phase 1: N=2 concurrent admission — handoff (2026-08-05)

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
