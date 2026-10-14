# DSv4.1 degeneration (repetition-loop) guards — ported into the dsv41 decode path

2026-10-13 (ported, commit `12548e373`) · **fix-up pass 2026-10-13b, after an adversarial audit**

## Verdict (SHIPPED on `main`, unit-verified; NO live-cluster contact)

The two degeneration detectors the GENERIC MLX path already shipped (exact-token cycle +
long-period block-hash cycle) are wired into the custom `dsv41` engine's decode path, which
deliberately bypasses `mlx_generate`/`ExoBatchGenerator` and so had **no guard at all** (its only
stop conditions were EOS, stop sequences and `MAX_TOKENS`). The incident that motivated this port
was an interactive agentic session that streamed **~3m41s of unbroken repetition** with nothing able
to stop it until the human client cancelled — a greedy decode collapse is an absorbing state, so a
detector that kills the turn is the only deterministic guarantee of termination.

Repo-only work: no cluster operation, no node edits, no relaunch. Unit tests only.

The audit's verdict was NO-SHIP as-is with four required items; the core (verbatim detector import,
every committed token fed in order at the single choke-point, termination via the existing plumbing,
rollback via `_abort_turn`→`session.cancel()`, real test coverage, replays reproduced) was found
sound. This pass fixes the four items and the minors — see **Fix-up pass (2026-10-13b)** at the end.

## Provenance / refs

- Base: `adurham/exo` `main` @ `ba4c3763f` (fetched over HTTPS; the shared checkout `~/repos/exo` was
  on `deploy/next23b-qsplit`, which is AHEAD of `main` and does NOT represent what is being changed
  here). Ported work in a worktree of `main`; this fix-up is a FF commit series on top of `12548e373`.
- Guards reused **verbatim** from `src/exo/worker/engines/mlx/generator/batch_generate.py`
  (`_detect_token_loop`, `_detect_long_period_loop` + every `EXO_LOOP_DETECT*` knob), so the two
  engines keep ONE source of truth and a threshold change can never apply to only one.
- Mirrors `src/exo/worker/tests/unittests/test_runner/test_loop_detection.py` for the detector
  semantics and the real-incident regression case.

## The hook point (one choke-point, every committed token)

`dsv41/engine.py::_generate` detokenizes one committed token at a time from the `committed_tokens()`
closure (the single place each ROUND's committed ids are appended to `produced`). The guard is fed
**every** committed token there, in order:

- never a **draft** token (an unconfirmed draft never reaches the client and must not shape the
  verdict), and
- never only a **round boundary** (a speculative round commits a whole batch at once, while the
  detectors' cycle alignment is defined over the raw committed stream).

**Feed alignment with the generic path (audit item 4a — resolved as *no offset*, not a change).**
`produced = [anchor] + rounds`, and the guard is fed each yielded id including the anchor. The
generic path feeds every *generated* token from the first one; the engine's `anchor` **is** that
first generated token — `rounds._one_round` emits only the token PREDICTED from the anchor, never the
anchor itself (so nothing is fed twice). `produced = [anchor] + rounds` is therefore exactly "first
generated token onward", byte-for-byte the same stream the generic path feeds. Pinned by
`test_guard_stream_starts_at_the_first_generated_token` (the anchor is the dump's first committed id
and the engine fires on the same flat token a raw feed does). The earlier "1-token offset" concern was
a miscount of `_feed`'s 0-based index vs `returned_tokens`' 1-based count.

The **terminating EOS id is no longer fed** (audit nit): `guard.feed(tid)` now runs *after* the
`tid in eos_ids` check, matching the generic path, which never feeds the token carrying the
terminating `finish_reason`. (Behaviourally inert — a single EOS cannot form a 6-cycle — it just keeps
the fed stream identical to the generic path's.)

Termination routes through the engine's EXISTING terminal path: a `_final_response` with
`finish_reason` (which `map_responses_to_chunks` turns into an `ErrorChunk` for `"error"`), the
stream closes with the normal `FinishedResponse`, and the turn is **rolled back** rather than
committed. No new wire type, no change to `rounds.py`, `session.py` or the park machinery.

## Rank agreement — argued AND now enforced (audit item 2)

The verdict is a pure function of the committed ids, and both TP ranks commit byte-identical tokens
(the DSpark round only ever emits target-confirmed tokens), so both ranks compute the same verdict
and would stop together. The ONE way the ranks can disagree is a **knob** that differs between the
two nodes: with, say, a different `EXO_LOOP_DETECT_MIN_REPEATS`, one rank terminates a degenerate
turn while its peer keeps decoding — and the peer then wedges forever in the model's own TP
`all_sum`. That is a hang class, not cosmetic drift.

- **Startup fail-fast (new):** `degeneration.assert_guard_env_agrees_across_ranks(coord_group)` is
  called from `Dsv41Engine.__init__` right where `get_coord_group(self.group)` is already invoked.
  That call is the engine's existing "same code path on both ranks at the same point" lockstep
  (its own docstring's requirement), so the check reuses an existing startup sync point rather than
  adding a new one. It reduces each rank's `guard_env_fingerprint()` with `mx_min_int` (the existing
  all-min helper) BOTH ways — min and max — and raises `RuntimeError` on disagreement, so a mismatch
  **fails the boot visibly** instead of surviving as a hung decode. `group is None` (single rank,
  PP mode, the unit tests) is a no-op.
- The effective values are ALSO logged once per rank at construction (`guard_env_report`) and the
  guard-fire WARNING carries the env digest (`guard_env_digest`), so the two ranks' lines for the
  same request remain directly diffable in the runner logs — the cross-rank knob check by eye, kept
  as a second, independent signal.

Residual (documented, not fixed — see Residual risks): the check rides the coord group at
construction, so it inherits that group's existing lockstep contract; and PP mode has no coord
channel, so it is not covered there.

## Finish-reason decision (audit item 3 — REWRITTEN; the earlier x3-retry rationale was WRONG)

**Decision: reuse `EXO_LOOP_DETECT_ACTION` with the SAME default as the generic path (`"error"`).**
The *reasoning* an earlier revision of this note gave for that default was factually wrong for this
client and is corrected here from the source on this machine (`~/.hermes/hermes-agent` @ `ca134f69e2`).

What ACTUALLY happens on a guard-fire, traced end to end:

1. **exo side.** `map_responses_to_chunks` turns the guard's `finish_reason="error"`
   `GenerationResponse` into an `ErrorChunk`, and the OpenAI-compatible adapter frames an ErrorChunk
   as an SSE **`data: {"error": {...}}`** event (`src/exo/api/adapters/chat_completions.py:304`), **not**
   a `finish_reason="error"` delta (never a normal `choices[].finish_reason`).
2. **Hermes side — no x3 API retry after partial delivery.** By the time the guard fires the turn has
   already streamed its prefix, so Hermes takes the *partial-delivery* branch: it does **NOT** re-issue
   the provider call. `agent/chat_completion_helpers.py:4011-4027`: "Streaming failed after partial
   delivery, not retrying" and it returns `False` — the only exceptions are a tool call in flight with a
   transient error, or the upstream toolcall clean-fail classification. A plain guard error is neither.
3. **Hermes side — partial stub + bounded auto-continue.** At `chat_completion_helpers.py:4384-4387`
   a partial-delivery error returns `_partial_stream_stub()`, a stub tagged
   `FINISH_REASON_LENGTH` ("the loop enters its continuation/retry path instead of accepting truncated
   output as a complete turn"). `agent/turn_response_check.py:173` routes `finish_reason == "length"`
   into `recover_from_truncation(...)`, which auto-continues the turn — bounded.

So the client-visible outcome of a guard-fire is: **bounded auto-continuations, then a truncation
banner** — i.e. the same shape as a length-truncated turn. There is **no x3 API retry** and **no
"1–2 min to a guaranteed clean failure"** as the earlier note claimed. The old note's premise (a
status-less `ProviderStreamError` retried 3× with jittered 2–8 s backoff) describes the *undelivered*
error path, not this one.

**Why `"error"` is still the right default, with the tradeoff stated.** The choice is about what the
guard *signals*, not about retries:
- `"error"` is parity with the generic path (one knob, one default) and it keeps the failure **loud
  and honest**: the degenerate partial is never surfaced as the agent's successful reply, and a loop
  inside a tool-call argument never becomes a truncated, malformed tool call on a turn the client
  believes completed. The client then does its bounded continue/truncation thing — the degenerate
  turn is not silently accepted.
- **The tradeoff, explicitly:** because Hermes auto-continues on the `"length"` stub, an `"error"`
  guard-fire costs a bounded number of extra continuations of an already-degenerate generation before
  the turn is finally surfaced as truncated. `"stop"` would surface the degenerate text immediately
  as a *successful* turn, which is worse (broken output shown, no signal at all). The auto-continue
  is bounded, so it is a bounded waste, and it is the only price of keeping the honest failure signal.
- `"stop"` (terminate, surface the partial) and `"warn"` (log only) remain available through the same
  `EXO_LOOP_DETECT_ACTION` knob; an unrecognized value falls back to the terminating default rather
  than silently disarming the guard.
- Rejected: `"content_filter"` (lies about the cause) and `"length"` (would *natively* drive Hermes's
  auto-continue, re-entering the loop with no error signal at all).

## Deploy staging recommendation

Bring the guard up **`warn`-first on the live client**, then arm it:

1. Stage 1: `EXO_LOOP_DETECT_ACTION=warn` on both ranks. The guard logs every detection (and dumps
   the committed tail) but never terminates. This measures the **real fire rate and fire content** on
   live traffic — exactly the evidence the unit tests and the offline replay cannot give — and costs
   nothing but log volume.
2. Stage 2: flip to `EXO_LOOP_DETECT_ACTION=error` (the default) once the observed fire rate is
   understood and no legitimate output is being flagged.
3. Watch `git diff`-style: keep both ranks' startup `loop-guard env` lines (`guard_env_digest`)
   identical; a mismatch now fails the boot (see Rank agreement).

## Integration note — restart branches (`deploy/next21-restart`, `deploy/next22-rankfix`)

Those branches **rewrite `session.py` / `park.py`**. The guard's guard-fire path ends the turn via
`_abort_turn` → `session.cancel()` and **bypasses `_end_turn` / `finish(checkpoint=True)`** — i.e. a
turn can now end *without* going through the completion/park machinery. Before merging the guard with
either restart branch, their **per-turn park/checkpoint invariants must be audited to tolerate a turn
ending via `_abort_turn`** (an aborted turn must not be parked, must not publish a checkpoint, and must
leave the cache / draft window / history at the pre-turn point those branches' invariants assume).
On `main` today the base park layer (`dsv41/park.py`) is present and this is implemented + tested
here; the restart branches' additional idle-checkpoint paths are not part of this tree.

## Separator false-positive fix (audit item 1) — in the SHARED detector

`_detect_token_loop` fires on the SHORTEST period; a run of one separator glyph is a period-1 cycle at
`min_repeats=6`, so **any** run of ≥6 identical separator tokens — a markdown horizontal rule
(`----------`), an `=====` banner, a long punctuation run, a dotted ellipsis, a code-fence/ASCII-art
row, a table separator row — killed a legitimate turn. This is inherited from the shared detector but
newly exposed on the agentic dsv41 path, so it is fixed **in the shared detector**, not at the call
site:

- `_detect_token_loop(..., token_text=None)` gained an optional decoder. When supplied, a tail whose
  last `max_period` tokens decode to **no word/number content** (`_is_separator_only_text`: empty
  whitespace/punctuation only) is **exempt** — one decode per call, and only on the (rare) first
  match, so the hot path is unaffected. Every candidate cycle is a suffix of that window, so a single
  decode decides the whole call.
- Word and whitespace+word cycles (a looping `" the"`) still fire **exactly as before**, and an EMPTY
  decode (an unknown id, e.g. a special token) is deliberately NOT treated as a separator — the
  conservative behaviour is kept.
- Termination is not lost: a **sustained** single-glyph spam is still caught, later, by the block-hash
  long-period detector (72+ identical tokens = 3 identical 24-token blocks). The exemption removes the
  false positive; it does not remove the guarantee.
- Both engines pass their own tokenizer decode: the generic path passes `self.tokenizer.decode`, the
  dsv41 guard is constructed with `token_text=tokenizer.decode`. One source of truth, one behaviour.

Tests: `TestSeparatorExemption` in the generic suite (markdown rule, `=` banner, ellipsis,
code-fence/ASCII-art, table row, indentation whitespace → **no fire**; period-1 word loop and a
word+separator cycle → **still fire**; empty decode → **not** exempted) and, through the real engine
in the dsv41 suite, a 40-dash rule serves cleanly with no guard-fire, while an 80-dash spam is still
terminated by the long-period detector.

## Session / park consistency on guard-fire

A guard-fire **must not** checkpoint a degenerate conversation, so the engine takes the *failure*
path, not the completion path: `_abort_turn` → `Conversation.cancel()` (rewind to the turn's `_base`,
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
  — one summary per request. `committed` is `len(produced)` on a normal turn (every id the cache
  holds, including a final round's overshoot past a stop point) and, on the guard/cancel paths, the
  count the guard actually judged (`guard_consumed`), since the aborted round's tail is rolled back
  and never committed.
- `[DSV41] DEGENERATION DETECTED turn=<id> at committed=<n> emitted=<n>: <detector> period=<p>
  repeated>=<r>x action=<a> guard_env_digest=<hex> prompt_tokens~<n> cycle_token_ids=[…]
  cycle_text=<repr>` — logged on every rank (the digest is identical across ranks iff the knobs are,
  which IS the cross-rank knob check).
- On a **guard-fire OR a client cancel** (rank 0 only): the last ≤ **2048** committed token ids are
  written as JSON to exo's log dir (`EXO_LOG_DIR`, override `EXO_DSV41_DEGENERATION_DUMP_DIR`), so the
  next incident's actual loop text is on disk — this one's was never captured. Filename
  `dsv41-degeneration-<reason>-<stamp>-<pid>.json`, carrying the reason, task, generated count,
  detector/period/repeats, and the guard env. Bounded and best-effort: a write failure logs and
  returns `None` rather than disturbing serving. **A client-cancel now also emits its `obs turn-end`
  line** (audit item 4c — made consistent); a `GeneratorExit` (an abandoned/GC'd generator, not a
  client cancel) still does **not** dump or log a turn-end, deliberately: nothing was cut short.
- Startup, once per rank: `[DSV41] loop-guard env (rank <r>/<w>, digest=<hex>): <every knob>=<value>`
  — the EFFECTIVE values, the repo's established "checkable deployed config" convention; plus the
  boot-fatal cross-rank equality check (see Rank agreement).

## Files changed

| file | change |
| --- | --- |
| `src/exo/worker/engines/mlx/dsv41/degeneration.py` | **new** — the guard driver, action policy, env report/digest + **fingerprint / cross-rank equality check**, bounded token dump, optional `token_text` |
| `src/exo/worker/engines/mlx/dsv41/engine.py` | guard feed + observability in `_generate`; `_abort_turn`, `_record_degeneration`, `_dump_short_turn`; startup env line **+ boot-fatal rank check**; EOS not fed; turn-end on cancel; `committed=` on the guard/cancel paths |
| `src/exo/worker/engines/mlx/generator/batch_generate.py` | **shared detector** `_detect_token_loop` gains the optional separator exemption (`_is_separator_only_text`); the generic call site passes `token_text=self.tokenizer.decode` |
| `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_degeneration.py` | **new** — guard tests (see below) |
| `src/exo/worker/engines/mlx/dsv41/tests/conftest.py` | `FakeTokenizer.decode` now decodes with its own `text_of` (it previously hardcoded `DEFAULT_CHUNKS`, so a custom-`text_of` fake decoded unknown ids to `""` — silently disabling the separator exemption in any test that exercised it) |
| `src/exo/worker/tests/unittests/test_runner/test_loop_detection.py` | `TestSeparatorExemption` — the shared detector's negative/positive separator cases |

## Test evidence

Command (this machine, CPU-only, no checkpoint, no GPU):

```bash
cd <worktree>
EXO_DASHBOARD_DIR=/Users/adam.durham/repos/exo/dashboard/build \
PYTHONPATH=<worktree>/src:<worktree>/mlx-lm \
/Users/adam.durham/repos/exo/.venv/bin/python -m pytest \
  src/exo/worker/engines/mlx/dsv41/tests \
  src/exo/worker/tests/unittests/test_runner/test_loop_detection.py \
  -q -p no:cacheprovider
```

- **263 passed** (231 dsv41 + 32 generic loop-detection) after this fix-up.
- Counts, corrected (audit item 4b — the previous note's split/arithmetic were wrong):
  - at the ORIGINAL port `12548e373`: **223 dsv41 + 22 generic = 245** (not 222/23; the totals matched
    only by accident of the same off-by-one).
  - true baseline before the port (`ba4c3763f`): **200** (178 dsv41 + 22 generic), **not 201**; the
    port's delta was **+45**, not +44.
  - after this fix-up: 231 dsv41 + 32 generic = **263** ⇒ +18 over `12548e373`, +63 over baseline.
- Gates: `ruff check` → clean; `ruff format --check` → clean **on every file this change touches**
  (the fix-up reflowed 3 sites — audit nit). The wider `dsv41/` tree has 10 files that `ruff format
  --check` would reformat; that drift is PRE-EXISTING at `ba4c3763f` and `12548e373` (verified
  identical against a clean worktree of the base) and is untouched by this change.
- `nix fmt` was NOT run (not installed on this machine); `basedpyright` was not re-run (not part of the
  acceptance gates; the two new symbols keep the module's existing inherited error classes).

What the new suite covers beyond the generic one:

- detector threshold parity (period-1 spam, min-repeats boundary, max-period boundary, the **real
  confirmed 14-token cycle** regression, varied/broken cycles, long-period block cycles, exact-match
  re-verify), **plus** that the guard is disabled by the env knob;
- **the round-batched feed** (the dsv41-specific requirement): a mixed stream fed in round-sized
  batches produces the same verdict as a token-at-a-time feed; a real engine with a DSpark head whose
  rounds commit ≥2 tokens fires where the guard predicts (asserted against the dump) and a LYING draft
  never shifts it (proved by the round really accepting drafts: `_spec_accepted > 0`);
- **feed alignment**: the guard's first fed id is the anchor / first generated token, no offset;
- **the separator exemption** (markdown rule, `=` banner, ellipsis, code-fence/ASCII-art, table row,
  whitespace) fires nothing, while a sustained separator spam is still terminated by the long detector;
- **the cross-rank env check**: no-op without a group, passes when the reduce agrees, raises loudly on
  disagreement;
- termination through the real engine + parser: `ErrorChunk` with the diagnostic for the `"error"`
  action, a clean `"stop"` terminal for `"stop"`, and no termination at all for `"warn"`;
- session consistency: the guard-fired turn is rolled back (offset 0 / no history / not inflight) and
  the conversation still serves the next turn; a NORMAL turn still checkpoints (negative control);
- observability: the detection line, the first-token / progress / turn-end lines, the client-cancel
  dump, the rank-0-only gate, the 2048-id bound, and that a dump write failure is swallowed;
- **offline replay of REAL recorded text** (see below).

## Offline replay (real recorded text, this machine)

`~/.hermes/state.db` (`read-only`), re-tokenized with the checkpoint family's tokenizer
(`~/.exo/models/deepseek-ai--DeepSeek-V4-Flash/tokenizer.json`), at the SHIPPED defaults, re-run after
the separator fix:

- `20261013_115838_90d257` (successful long agentic run): 88 text fields, **43 892 tokens, 0
  detections**.
- `20261013_184432_180a8c` (the session that thrashed; its final loop text was never persisted, so its
  stored reasoning is the false-positive material): 63 fields, **32 588 tokens, 0 detections**.

Both at zero detections is the result that matters: the guard does not fire on genuine output. (The
separator fix could only ever *reduce* fires, so the counts are unchanged.) The test is `skipif`-guarded
on the tokenizer and the DB existing, so CI without them skips it loudly rather than silently.

## Residual risks / not verified

- **No live-cluster validation** (by instruction): no relaunch, no cluster traffic. The port is
  unit-verified only; the guard's in-situ behaviour on TP2 (both ranks terminating together) is argued
  from purity + byte-identical committed tokens, and now also enforced by the boot-time knob-equality
  check — but not measured live.
- **The already-streamed degenerate tail cannot be retracted.** The engine streams tokens as they
  commit, so by the time any detector can fire, the repeated tail HAS been sent. Neither `"error"` nor
  `"stop"` can remove it; the choice only controls what the client does with the turn. (Mitigation: the
  token that *completes* the cycle is not detokenized, so nothing new is added.)
- **The cross-rank check rides the coord group at engine construction.** It is placed where
  `get_coord_group` is already called in matching order and inherits that call's existing lockstep
  contract; it adds no new collective *primitive* (it reuses `mx_min_int`) but does add two matched
  reduces at that point. **PP mode is not covered** (`get_coord_group` returns `None` there — no coord
  channel); PP mode already runs one request at a time in lockstep across ranks, so the guard's
  termination is the only cross-rank decision, and a knob mismatch there is not detected at boot.
- **`main` does not carry the §G1–G4 park hardening** (`coord.py` / `fingerprint.py` / `shutdown.py`
  live on the p20/restart branches). The base park layer (`dsv41/park.py`) IS on `main`, so the "never
  park a degenerate conversation" requirement is implemented and tested here; see the Integration note
  for what the restart branches must audit before merging the guard.
- The detectors are shared, so this port inherits their tuning: min_repeats=6 (tight) / 3 (long),
  max_period 24 / 12 blocks. A loop with paraphrase drift (period 60–400+, no exact repeat) is still
  NOT caught — that is the (out-of-scope, phase-2) reasoning-budget limiter's job.
- The separator exemption is text-driven, so it needs a tokenizer at the call site. Both serve paths
  pass one; a bare detector (`token_text=None`) keeps the pre-fix behaviour, which is what the
  threshold-parity tests pin.

## Fix-up pass (2026-10-13b) — per-item disposition

1. **MAJOR period-1 separator false positive — FIXED** in the shared detector (see above).
2. **MAJOR rank-config equality — FIXED (enforced)** via the boot-fatal min/max reduce at the existing
   construction-point coord collective (see Rank agreement).
3. **MAJOR finish-reason rationale — FIXED (rewritten)** from the source-traced client path; staging
   recommendation + integration note added.
4. **MINORS:**
   - (a) anchor feed — investigated; **no offset** (the anchor IS the first generated token). Documented
     + pinned by a test. No code change.
   - (b) buildnote counts — **corrected** (223/22/245, baseline 200, +45; and the new 231/32/263).
   - (c) `GeneratorExit` close skips the dump (deliberate) and a cancelled turn emitted no `obs
     turn-end` — **the cancel path now logs its turn-end** (made consistent); the `GeneratorExit` no-dump
     is documented as deliberate.
   - (d) nits — `log_turn_end` now takes the guard-judged count on the guard/cancel paths (aligned with
     the dump's `produced[:guard_consumed]`); the "ruff reflowed 2 lines" claim corrected to **3 sites**;
     `guard.feed` now runs **after** the EOS check so the terminating EOS is not fed (parity with the
     generic path).
