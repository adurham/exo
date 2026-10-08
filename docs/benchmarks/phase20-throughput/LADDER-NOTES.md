# LADDER-NOTES — Phase 0c delta-prefill ladder (`bench/phase20_delta_ladder.py`)

Owner: BRIEF L (branch `p20/ladder`). The PM runs the live chunks; this worker
built and proved the tooling **offline** against `bench/phase20_tests/mock_exo_server.py`
(which replays real exo.log line shapes). **No generation request was ever sent to
the real cluster.**

## What the tool does (plan 0c, verbatim)

1. **Ctx ladder**: prefill to ctx in {20K, 50K, 110K}, then a **2048-row delta**;
   `rows/s` from the engine log.
2. **Delta-size sweep at 50K**: deltas of {256, 1024, 4096, 8192} rows; 3 reps/cell.
3. **Fresh-feed 100K reference** once (expect 265–275 rows/s; ~6.5 min cold).

Subcommands: `pilot | chunk1 | chunk2 | chunk2a | chunk2b | chunk3 | fresh100k | summarize`
(`--api`, `--reps`, `--chars-per-token`, `--out-dir`, `--dry-run`, `--max-wall`,
`--base-salt`, `--log-source node=path`). `summarize` rebuilds Table A + Table B
(markdown + CSV) and prints the three 0c decision verdicts.

## Mechanics that had to be right (all from this repo's history)

1. **Row truth is the engine log, never an estimate.**
   `[DSV41] turn reuse: prompt=N prefill=M reuse=R cache=C rewind=W UNCOMMITTED`
   is emitted by `session.prefill` **only when reuse>0**; `delta_rows = M`.
   `prefill_s = t(turn reuse) − t(prefill controls)` (fallback: the `API request:
   POST /v1/chat/completions` line, for small deltas where the controls line is
   not emitted). Both `rows_per_s_log = M/prefill_s_log` **and** the client-side
   cross-check `rows_per_s_ttft = M/ttft` are recorded. A **cold** feed has no
   turn-reuseline → `rows/s = usage.prompt_tokens / TTFT` (prior method:
   100000/373.9 = 267.5).
   Log windows are read through a snapshot-before / `tail -c +<size+1>`-after seam
   (`LogReader`/`SSHNodeLog`); offline it is `LocalFileLog` over the mock's file.
2. **A delta exists only if the prompt LCP reaches a checkpoint.** `SessionCache.plan()`
   rewinds to the newest checkpoint at/below the LCP (turn/prompt ends + a 1024-row
   ladder + a ~512-row margin rung). Appending to the *same* user message changes the
   trailing template tokens and collapses to a rung. So the rep shape is the real
   agentic **multi-turn branching** form:
   `messages = [user: SALT+filler(ctx), assistant: <reply text>, user: <new delta text>]`
   whose rendered prefix contains the previous prompt exactly. Each rep branches from
   the base's prompt-end checkpoint, so **reps branch from the same base depth**.
   A rep with `reuse < 0.9 × base_rows` is `collapsed` → excluded from the table and
   listed (the `reuse undershoot` WARNING also flags it).
3. **Unique salt ONCE at the head of every base** (`SALT-<hex>`, `secrets.token_hex(8)`
   equivalent via `os.urandom(8).hex()`). park/restore persists conversations across
   relaunches, so a reused filler can match a restored session (observed fake
   1914 tok/s). The salt is never re-inserted inside the filler (that re-tokenizes).
   Every feed believed fresh is verified to have NO `turn reuse:` line (or reuse≈0).
4. **Never put a big prompt on a command line** (E2BIG): HTTP bodies originate in
   Python only.
5. Request shape: OpenAI-compatible `POST {API}/v1/chat/completions`, `max_tokens=1`,
   `temperature=0`, `stream=true`; the `: generation_stats` SSE **COMMENT** frame is
   parsed (it does **not** start with `data:`); `reasoning_effort` omitted; 3600 s
   timeout, the guard wall cap governs.
6. **Chunk discipline.** Before each step predict its wall (`rows/200*1.3 + 15 s`); if
   it would exceed the chunk's remaining wall, STOP before starting it, mark
   `NOT_RUN_WALL_CAP`. `guard.register_own_request()` immediately before every HTTP
   request; `guard.check()` between requests; on `ChunkAborted` stop cleanly, write
   what you have, exit **75**. A canary runs before each chunk.

## Delta size labels are NOMINAL

Chars→tokens is a calibration: `--chars-per-token` (default 5.111, the battery
filler's measured value; phase19 sentences were 5.59). The table carries the
**actual** `prefill=M` from the log; `--chars-per-token` only sizes the text.
A `pilot` run (4K base + {256,1024} deltas) validates shapes before real chunks.

## Output schema (raw JSONL per request)

`docs/benchmarks/phase20-throughput/raw/delta_ladder.<chunk>.jsonl`, one record:
`chunk, cell, rep, kind (fresh|delta), ctx_nominal, ctx_label, delta_nominal,
prompt_tokens, completion_tokens, ttft_s, wall_s, log_prefill(=M), reuse(=R),
rewind(=W), prefill_controls_rows, prefill_s_log, has_turn_reuse, log_node, t_start,
chars_per_token, rows_per_s_log, rows_per_s_ttft, collapsed, invalid, notes`.

`summarize` finds every `delta_ladder.*.jsonl`, builds:
* **Table A** — ctx ladder (rows/s median/min/max per ctx at the 2048 delta + the fresh reference)
* **Table B** — delta-size sweep at 50K (rows/s median/min/max, actual rows, vs-4096 ratio)

and evaluates the decision rules; it writes `delta_ladder.summary.md` + `.csv`.

## Decision rules (PREREG 0c)

* `rows/s` degrading **>15%** 20K→110K at the 2048 delta ⇒ **ctx-depth cost**.
* 256-row `rows/s` **< 50%** of the 4096-row at the same ctx ⇒ **fixed per-call
  overhead** ⇒ Phase 4 lever.
* **flat** (<10% spread) across delta sizes ⇒ **Phase 4 skipped**, slope recorded.
* **Falsifier** — fresh-feed reference **< 255 rows/s** ⇒ cluster degraded: re-canary,
  emit `DEGRADED_REFERENCE`, mark every cell of that chunk invalid, do **not** record.
  `fresh100k` writes `raw/fresh_ref.json`; every other chunk refuses to record
  (exit 3) when that reference is below 255.

## Chunk walls (dry-run, 2026-10-07)

Wall model `rows/200*1.3 + 15 s`; +90 s canary estimate. `--max-wall 900`.

| chunk | steps | pred total | + canary | fits 900 s? |
|---|---|---|---|---|
| chunk1 | 20K base + 3×d2048, 50K base + 3×d2048 | 654.9 s (10.9 min) | 744.9 s (12.4 min) | yes |
| **chunk2** | 110K base + 3×d2048 | 814.9 s (13.6 min) | **904.9 s (15.1 min)** | **no** (5 s over the 900 s cap with canary) |
| chunk3 | 50K base + 3×(256,1024,4096,8192) | 784.6 s (13.1 min) | 874.6 s (14.6 min) | yes |
| fresh100k | 1×100K cold feed | 665.0 s (11.1 min) | 755.0 s (12.6 min) | yes |
| pilot | 4K base + {256,1024} | 79.3 s | 169.3 s | yes |

**chunk2 does not fit** (110K base 730 s + 3×28.3 s deltas + canary ≈ 905 s > 900 s).
**Split it**: `chunk2a` = the 110K cold feed alone (730 s + canary = 820 s, fits);
`chunk2b` = the 3×2048 delta reps against the already-resident 110K base
(84.9 s + canary = 174.9 s, fits). The base persists across the two runs (park/restore),
so `chunk2b` branches from the base depth established by `chunk2a` — same salt, so the
checkpoint is the same. (If the resident conversation is evicted between runs, the
`chunk2b` reps will show `collapsed`/no-reuse and must be re-run after re-establishing
the base — the tool flags this rather than silently recording it.)

## Offline proof

`bench/phase20_tests/mock_exo_server.py` — a stdlib `http.server` that serves the SSE
stream (`: generation_stats` comment + `usage`) and appends real-shaped log lines
(`prefill controls` / `turn reuse` / `reuse undershoot`) to a temp file. It models the
rewind: branching (multi-turn) → `reuse = base_rows, prefill = delta_rows`;
single-message-append → collapses to a 1024-row rung. Timing is derived
(`prefill_s_log = rows / rows_per_s`) so the math is exactly checkable.

Test incantation:
```
cd /private/tmp/p20-ladder && PYTHONPATH=bench \
  /Users/adam.durham/repos/exo/.venv/bin/python -m pytest --noconftest \
  bench/phase20_tests/test_phase20_delta_ladder.py -q -p no:cacheprovider
```
Covers: salt-once, size calibration, log extraction on REAL lines (copied verbatim
from `m41_0901_1400.log.zst` + a live re-derive if the zst is present), delta-vs-collapse
classification, rows/s math (log + ttft), wall-cap pre-check, guard-abort path, all
three decision rules + DEGRADED_REFERENCE on synthetic JSONL, summarize md/csv, the
mock end-to-end, and compatibility with the REAL sibling `phase20_guard.py` classes.
Two tests are **sabotage-proven** (`*`↔`/` in rows/s; inverted collapse threshold both
fail the mutated source).

## NOT verified (offline build only)

* **Real engine behaviour of the branching reps** — that a multi-turn
  `[base, assistant, delta]` actually rewinds to the base's prompt-end checkpoint and
  yields `reuse ≈ base_rows` on the live cluster. Verified only by the mock + the log
  shape; the PM's live `pilot` is the real test.
* **`SessionCache.plan()` rung geometry** (1024/512) — taken from the brief, not
  re-derived from engine source.
* **Calibration constant** 5.111 chars/token — a prior measurement; `pilot` +
  `--chars-per-token` must re-calibrate against a real `usage.prompt_tokens`.
* **SSH log transport** (`SSHNodeLog.size`/`read_from`, `tail -c +N`) — never exercised
  against a node here; only `LocalFileLog` was.
* **Wall model** `rows/200*1.3+15` — a PREREG formula, not measured; real per-step
  walls may differ, which is why the runtime wall pre-check charges actual wall.

## FIX 1 — persistent own-request registry (survive aborted chunks)

**Symptom.** `ChunkGuard.__enter__` runs the entry idle-check, which refuses to start
if any generation POST is on a node's log newer than `min_idle_s` (600 s) that is *not*
one of our own registered requests (within ±2 s). The tool only ever registered its own
requests **in memory**, so a chunk that was aborted or killed mid-run left its POSTs in
the node logs with no surviving registration. The *next* run's entry check then treated
those POSTs as foreign and refused to start for up to 10 minutes — and every subsequent
chunk would stall the same way.

**Fix.** `run_chunk` now loads a **persistent** own-request registry before entering
`ChunkGuard` (and before the standalone pre-chunk canary) and passes **both**
`own_requests=<loaded list>` and `registry_path=<registry file>` into the constructor.
`ChunkGuard.register_own_request()` already appends a `{"label":…, "t":…}` JSONL line to
`registry_path`, so the tool now remembers its own registrations **across runs and across
chunks**.

* New helper `load_own_requests(path)` reads that JSONL and returns the `t` epochs as
  floats; a missing file (first run) or a malformed/blank line is tolerated and skipped
  (never fails a chunk). Registrations are made *immediately before* each POST, so an
  epoch read back can only correspond to a POST already on a node.
* Default registry path: `<out-dir>/raw/own_requests.jsonl` — beside the chunk JSONL and
  the `<label>.guard.json` the guard already writes.
* New CLI flag `--own-registry PATH` overrides it.
* No request/measurement logic, JSONL record schema, or decision rule was changed.

**Tests** (`bench/phase20_tests/test_phase20_delta_ladder.py`, 36 → 39):
`test_load_own_requests_roundtrips_guard_registry` (round-trips what the real
`ChunkGuard.register_own_request` writes, plus missing/malformed tolerance);
`test_registry_survives_restart_entry_idle_check_passes` (two sequential chunk-like
passes sharing one registry path over the fake-ssh/fake-clock seam: pass 1 registers and
is killed; pass 2 builds a **fresh** guard that *loads* the registry → entry idle-check
`ok True`; a genuinely non-own POST still refuses);
`test_run_chunk_loads_and_persists_default_registry` (run_chunk wires the default path
and persists 3 registrations). All three FAIL without the change and pass with it.

## FIX 2 — rep uniqueness, wall-cap accounting, ladder-aware collapse (live-repro bugs)

Three bugs reproduced live on the 2-node cluster (chunk1, 2026-10-07). Ground truth:

| step | prompt | log_prefill | reuse | rewind | wall | note |
|---|---|---|---|---|---|---|
| ctx20k base (fresh) | 18390 | — | — | — | 63.5 s | cold |
| ctx20k d2048 rep1 | 20254 | 3870 | 16384 | 18390 | 14.5 s | real delta |
| ctx20k d2048 rep2 | 20254 | **0** | 20254 | — | 0.3 s | full cache hit |
| ctx20k d2048 rep3 | 20254 | **0** | 20254 | — | 0.3 s | full cache hit |
| ctx50k d2048 rep1 | 47561 | 2505 | 45056 | 45705 | 10.3 s | real delta |
| ctx50k d2048 rep2 | 47561 | **0** | 47561 | — | 0.4 s | full cache hit |

**(a) Reps not unique → reps 2..N are full cache hits that measure nothing.**
`build_delta(delta_tokens, seed=…)` was deterministic in `seed + delta_tokens`, so
every rep of the same nominal size built a BYTE-IDENTICAL payload; the engine then
served the previous rep's exact prompt from cache (`prefill=0`, `reuse == prompt`,
wall ≈ 0.3 s) and the tool recorded it as a valid 0-row delta. **Fix:** `build_delta`
gains a `nonce` (the rep index) folded into the fill seed (`+ 1000003*nonce`), and the
ladder passes `nonce=st["rep"]`. Each rep is now `[base, reply, distinct-delta-N]`,
the engine rewinds to the base checkpoint, and a real prefill is measured. Salt-free
and `build_base` untouched. Additionally, a delta with `prefill=0` / `reuse == prompt`
is now classified `cache_hit=True` and **excluded** from Table A/B (`summarize` gains
a `cache_hits` list, surfaced in the summary md); it is never silently averaged in as
a 0-row measurement.

**(b) Wall pre-check double-counted predicted time.** The pre-check computed
`elapsed = monotonic() − chunk_start` (REAL wall) **and** added `pred_cum` (a running
sum of *predicted* walls), so it blocked when `elapsed + pred_cum + pred > cap`,
double-counting time already spent — the live chunk1 hit `NOT_RUN_WALL_CAP: ctx50k_d2048
pred 28s + used 892s > cap 900s` although the true elapsed wall was far lower.
**Fix:** the gate is now `elapsed + pred > cap` (true elapsed only). `pred_cum` is
retained purely as a predicted-elapsed **report**, never a gate. The record schema is
unchanged.

**(c) `collapsed` threshold mis-calibrated to a 0.9×base rule.** The engine keeps
checkpoints on a ~2048-row ladder, so a base whose end is not a rung legitimately
rewinds to the largest rung ≤ the base end (base 18390 → checkpoint 16384; 45705 →
45056). `reuse < 0.9*base_rows` flagged these real deltas as collapsed
(16384/18390 = 0.89). **Fix:** `phase20_common.is_collapsed(reuse, base_rows, rung=2048)`
flags collapsed only when `reuse < base_rows − 2048` (more than one ladder rung below
the base's **ACTUAL** row count, taken from the base request's `usage.prompt_tokens`
where available). `LADDER_RUNG = 2048` is the documented rung. Live check: reuse 16384
vs base 18390 → OK; reuse 2048 vs base 20000 → collapsed.

**Tests** (39 → 47): rep-delta uniqueness (`test_rep_delta_texts_unique_per_rep`,
`test_ladder_build_messages_unique_delta_per_rep`); full-cache-hit classification +
exclusion (`test_mock_cache_hit_rep_measured_nothing_and_excluded`,
`test_record_classifies_prefill_zero_as_cache_hit`); wall gate
(`test_wall_precheck_ignores_predicted_cumulative` — a mock run must NOT emit
`NOT_RUN_WALL_CAP`; `test_wall_precheck_blocks_truly_over_step_not_earlier` — the
corrected gate blocks exactly the first truly-over step, whereas the old gate blocks
one step early); ladder-aware collapse (`test_collapsed_rule_ladder_rung`,
`test_record_collapse_uses_base_actual_depth`). **All 8 FAIL on the pre-fix sources
(7 assert, 1 via the two-run gate divergence) and pass after**; the 39 pre-existing
tests stay green. The mock now models an exact-prompt full-cache hit (`prefill=0`,
`reuse == prompt`) and uses a **2048**-row rung.

**Live re-run** (idle cluster, scratch out-dir): pilot base 3775 rows cold (13.2 s);
delta reps `log_prefill=2000` (reuse 2048) and `log_prefill=2722` (reuse 2048) — both
real deltas, neither a cache hit (the pilot has one rep per size, so same-size rep
uniqueness is proven by the unit + mock tests rather than by the pilot).


