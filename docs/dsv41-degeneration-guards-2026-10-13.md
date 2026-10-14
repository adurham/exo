# DSv4.1 degeneration (repetition-loop) guards — ported into the dsv41 decode path (2026-10-13)

## Verdict (SHIPPED on `main`, unit-verified; NO live-cluster contact)

The two degeneration detectors the GENERIC MLX path already shipped (exact-token cycle +
long-period block-hash cycle) are now wired into the custom `dsv41` engine's decode path, which
deliberately bypasses `mlx_generate`/`ExoBatchGenerator` and so had **no guard at all** (its only
stop conditions were EOS, stop sequences and `MAX_TOKENS`). The incident that motivated this port
was an interactive agentic session that streamed **~3m41s of unbroken repetition** with nothing able
to stop it until the human client cancelled — a greedy decode collapse is an absorbing state, so a
detector that kills the turn is the only deterministic guarantee of termination.

Repo-only work: no cluster operation, no node edits, no relaunch. Unit tests only.

## Provenance / refs

- Base: `adurham/exo` `main` @ `ba4c3763f` (fetched over HTTPS; the shared checkout
  `~/repos/exo` was on `deploy/next23b-qsplit`, which is AHEAD of `main` and does NOT represent
   what is being changed here). All work in a dedicated worktree `/private/tmp/dsv41-guards`.
- Guards reused **verbatim** from `src/exo/worker/engines/mlx/generator/batch_generate.py`
  (`_detect_token_loop`, `_detect_long_period_loop` + every `EXO_LOOP_DETECT*` knob),
  so the two engines keep ONE source of truth and a threshold change can never apply to only one.
- Mirrors `src/exo/worker/tests/unittests/test_runner/test_loop_detection.py` for the detector
  semantics and the real-incident regression case.

## The hook point (one choke-point, every committed token)

`dsv41/engine.py::_generate` detokenizes one committed token at a time from the
`committed_tokens()` closure (the single place each ROUND's committed ids are appended to
`produced`). The guard is fed **every** committed token there, in order:

* never a **draft** token (an unconfirmed draft never reaches the client and must not shape the
  verdict), and
* never only a **round boundary** (a speculative round commits a whole batch at once, while the
  detectors' cycle alignment is defined over the raw committed stream).

Termination routes through the engine's EXISTING terminal path: a `_final_response` with
`finish_reason` (which `map_responses_to_chunks` turns into an `ErrorChunk` for `"error"`), the
stream closes with the normal `FinishedResponse`, and the turn is **rolled back** rather than
committed. No new collective, no new wire type, no change to `rounds.py`, `session.py` or the park
machinery.

**Rank agreement is by construction.** The verdict is a pure function of the committed ids, and both
TP ranks commit byte-identical tokens (the DSpark round only ever emits target-confirmed tokens), so
both ranks stop together. A loop is never confirmed on one rank only.

## Finish-reason decision (deliberate): default `"error"` — and why

**Decision: reuse `EXO_LOOP_DETECT_ACTION` with the SAME default as the generic path (`"error"`).**

Checked how the client actually behaves, because the choice is only safe if a retry cannot multiply
the very runaway the guard exists to cut. Verified in the source on this machine
(`~/repos/hermes-agent`, also byte-identical to `~/.hermes/hermes-agent` @ `ca134f69e26`):

- Hermes **DOES auto-retry** a provider stream that ends `finish_reason="error"`: the same provider
  call is re-issued up to **3 times** (`agent/conversation_loop.py:1747-1776`
  `_run_api_retry_loop`; `api_max_retries` default 3 at `hermes_cli/config_defaults.py:133`;
  jittered 2 s→4 s→8 s backoff at `agent/turn_recovery.py:1445`), and an exo error finish is
  classified retryable (`agent/error_classifier.py:86` — a status-less `ProviderStreamError` lands
  on `_V_UNKNOWN`, whose `retryable` defaults True).
- So on a GREEDY engine a retry replays the identical loop and its **success probability is exactly
  0**: up to 4 traversals in total (the original attempt + 3 retries), not 3.
- **Why that is nonetheless the right default.** The waste is bounded BY THE DETECTOR, not by the
  runaway: the guard fires at the loop's ONSET (~150–900 committed tokens ≈ 5–30 s at this
  cluster's ~30 tok/s, plus ~14 s of backoff) ⇒ roughly **1–2 minutes to a guaranteed clean
  failure**, versus the 3 m 41 s the incident burned before a human intervened. And the alternative,
  `"stop"`, terminates once but hands the agent loop a degenerate turn as a **successful** reply —
  and, if the loop started inside a tool-call argument, a truncated malformed tool call on a turn the
  client believes completed. A loop of the pre-port kind is already past the point where its output is
  usable, which is exactly why the generic path chose the loud failure too.
- Reversibility: `EXO_LOOP_DETECT_ACTION=stop` (terminate, surface the already-streamed partial) and
  `=warn` (log only, the pre-port behaviour) remain available; an unrecognized value falls back to
  the *terminating* default rather than silently disarming the guard.
- Rejected alternatives: `"content_filter"` (lies about the cause; would still spend a deterministic
  retry) and `"length"` (agent clients commonly auto-continue on length, which re-enters the loop).
- **Noted, NOT implemented (out of scope):** an engine-side deterministic-retry short-circuit
  (hash of the prompt/conversation state → "looped at token N", TTL) that would make the 3 retries
  free. It needs zero client changes and is a small, self-contained follow-up.

## Session / park consistency on guard-fire

A guard-fire **must not** checkpoint a degenerate conversation, so the engine takes the *failure* path,
not the completion path: `_abort_turn` → `Conversation.cancel()` (rewind to the turn's `_base`,
restore the draft window to the same row, prune the orphaned draft/anchor snapshots the decode-side
ladder took above it) plus `drop` for a one-off conversation. Consequences, all deliberate:

- the collapsed turn contributes **no rows** to what the cache holds, so a parked/evicted conversation
  can never carry a corrupt generation into the next turn;
- the client's view and the session's history diverge only in that the (already-streamed) reply is
  re-fed on the next turn — the normal behaviour of the engine's existing failure path;
- `turn.committed` stays False and `finish()` is not called, so no completion checkpoint is published.

## Observability (added in the same commit series)

Per request, one line per event, all greppable by the `[DSV41] obs` / dump markers:

- `[DSV41] obs first-token turn=<id> latency=<s>s (prefill <s>s, <n> prompt tokens, reuse=<n>)`
  — time from request start to the FIRST committed token.
- `[DSV41] obs progress turn=<id> committed=<n> elapsed=<s>s tok_s=<r>` every ~10 s of generation.
- `[DSV41] obs turn-end turn=<id> committed=<n> emitted=<n> elapsed=<s>s tok_s=<r> reason=<r>`
  — one summary per request.
- `[DSV41] DEGENERATION DETECTED turn=<id> at committed=<n> emitted=<n>: <detector> period=<p>
  repeated>=<r>x action=<a> guard_env_digest=<hex> prompt_tokens~<n> cycle_token_ids=[…]
  cycle_text=<repr>` — logged on every rank (the digest is identical across ranks iff the knobs are,
  which IS the cross-rank knob check).
- On a **guard-fire OR a client cancel** (rank 0 only): the last ≤ **2048** committed token ids are
  written as JSON to exo's log dir (`EXO_LOG_DIR`, override `EXO_DSV41_DEGENERATION_DUMP_DIR`), so
  the next incident's actual loop text is on disk — this one's was never captured. Filename
  `dsv41-degeneration-<reason>-<stamp>-<pid>.json`, carrying the reason, task, generated count,
  detector/period/repeats, and the guard env. Bounded and best-effort: a write failure logs and
  returns `None` rather than disturbing serving.
- Startup, once per rank: `[DSV41] loop-guard env (rank <r>/<w>, digest=<hex>): <every knob>=<value>`
  — the EFFECTIVE values, the repo's established "checkable deployed config" convention.

## Files changed

| file | change |
| --- | --- |
| `src/exo/worker/engines/mlx/dsv41/degeneration.py` | **new** — the guard driver, action policy, env report/digest, bounded token dump |
| `src/exo/worker/engines/mlx/dsv41/engine.py` | guard feed + observability in `_generate`; `_abort_turn`, `_record_degeneration`, `_dump_short_turn`; startup env line |
| `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_degeneration.py` | **new** — 45 tests (see below) |

## Test evidence

Command (this machine, CPU-only, no checkpoint, no GPU):

```bash
cd /private/tmp/dsv41-guards   # the main worktree
EXO_DASHBOARD_DIR=/Users/adam.durham/repos/exo/dashboard/build \
PYTHONPATH=/private/tmp/dsv41-guards/src:/private/tmp/dsv41-guards/mlx-lm \
/Users/adam.durham/repos/exo/.venv/bin/python -m pytest \
  src/exo/worker/engines/mlx/dsv41/tests \
  src/exo/worker/tests/unittests/test_runner/test_loop_detection.py \
  -q -p no:cacheprovider
```

- **245 passed** (222 dsv41 + 23 generic loop-detection), reproduced **3×** consecutively; baseline
  before the change was 178 dsv41 / 201 total, so the change adds 44 tests and removes none.
- Gates: `ruff check` → clean; `ruff format --check` → clean. `basedpyright` on `engine.py`:
  **113 errors both BEFORE and AFTER** (zero new; the file's pre-existing count); the two new files
  carry 4 and 90 errors, all of the repo's inherited `loguru`-untyped / untyped-test-harness
  classes that sibling modules (`rounds.py` 104, `session.py` 177) share. `nix fmt` was NOT run
  (not installed on this machine).

What the new suite covers beyond the generic one:

- detector threshold parity (period-1 spam, min-repeats boundary, max-period boundary, the **real
  confirmed 14-token cycle** regression, varied/broken cycles, long-period block cycles, exact-match
  re-verify), **plus** that the guard is disabled by the env knob;
- **the round-batched feed** (the dsv41-specific requirement): a mixed stream fed in round-sized
  batches produces the same verdict as a token-at-a-time feed; a real engine with a DSpark head whose
  rounds commit ≥2 tokens fires where the guard predicts (asserted against the dump) and a LYING draft
  never shifts it (proved by the round really accepting drafts: `_spec_accepted > 0`);
- termination through the real engine + parser: `ErrorChunk` with the diagnostic for the `"error"`
  action, a clean `"stop"` terminal for `"stop"`, and no termination at all for `"warn"`;
- session consistency: the guard-fired turn is rolled back (offset 0 / no history / not inflight) and
  the conversation still serves the next turn; a NORMAL turn still checkpoints (negative control);
- observability: the detection line, the first-token / progress / turn-end lines, the client-cancel
  dump, the rank-0-only gate, the 2048-id bound, and that a dump write failure is swallowed;
- **offline replay of REAL recorded text** (see below).

## Offline replay (real recorded text, this machine)

`~/.hermes/state.db` (`read-only`), re-tokenized with the checkpoint family's tokenizer
(`~/.exo/models/deepseek-ai--DeepSeek-V4-Flash/tokenizer.json`), at the SHIPPED defaults:

- `20261013_115838_90d257` (successful long agentic run): 88 text fields, **43 892 tokens, 0
  detections**.
- `20261013_184432_180a8c` (the session that thrashed; its final loop text was never persisted, so
  its stored reasoning is the false-positive material): 63 fields, **32 588 tokens, 0 detections**.

Both at zero detections is the result that matters: the guard does not fire on genuine output. The test
is `skipif`-guarded on the tokenizer and the DB existing, so CI without them skips it loudly rather
than silently.

## Residual risks / not verified

- **No live-cluster validation** (by instruction): no relaunch, no cluster traffic. The port is
  unit-verified only; the guard's in-situ behaviour on TP2 (both ranks terminating together) is argued
  from purity + byte-identical committed tokens, not measured live.
- **The already-streamed degenerate tail cannot be retracted.** The engine streams tokens as they commit,
  so by the time any detector can fire, the repeated tail HAS been sent. Neither `"error"` nor `"stop"`
  can remove it; the choice only controls what the client does with the turn. (Mitigation: the token that
  *completes* the cycle is not detokenized, so nothing new is added.)
- **Cross-rank env diff is log-based, not exchanged.** A dedicated rank-agreement collective was
  considered and deliberately NOT added: the engine has no startup-synchronised program point (`warmup`
  is only reached on the reconnect path, and the request-path collectives pair across the task/cancel
  channels in a way that is not trivially symmetric), so a new collective there would add a whole new
  hang class for a diagnostic. Instead the effective values are logged at startup per rank AND carried as
  a digest on every guard-fire line, so the two ranks' lines for the SAME request can be diffed.
- **`main` does not carry the §G1–G4 park hardening** (`coord.py` / `fingerprint.py` / `shutdown.py`
  live on the p20/restart branches). The base park layer (`dsv41/park.py`) IS on `main`, so the
  "never park a degenerate conversation" requirement is implemented and tested here; the G1+ idle
  checkpoint path is not part of this tree.
- The detectors are shared, so this port inherits their tuning: min_repeats=6 (tight) / 3 (long),
  max_period 24 / 12 blocks. A loop with paraphrase drift (period 60–400+, no exact repeat) is still
  NOT caught — that is the (out-of-scope, phase-2) reasoning-budget limiter's job.
