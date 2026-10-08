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
