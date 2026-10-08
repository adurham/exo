# BRIEF Z — Phase 0a: per-call decomposition of the real turn (`turn_decomp.csv`)

Tier: mid-coder. Worktree: `/private/tmp/phase20-campaign` (branch `deploy/phase20-campaign`). NO cluster access beyond read-only file reads of local copies.
You own ONLY: `bench/phase20_turn_decomp.py`, `bench/phase20_tests/test_phase20_turn_decomp.py`, `docs/benchmarks/phase20-throughput/turn_decomp.csv`,
`.../turn_decomp_summary.json`, `.../turn_decomp.md`. Read `CHILD-COMMON.md` and `PREREG.md` (0a gate + cache-hit rule) first.
Write partial results to disk as you go (a findings file survives a timeout; a summary that was never sent does not). Tool-call budget: ~60. Hard stop: 45 min.

## The question
For the real 42-call Hermes turn (session `20261007_092009_9a2ed7`, `provider='custom'`, model `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`), split each call's `latency_seconds`
into **measured** prefill time and decode time from exo.log markers (NOT from the old `output/20.66` estimate in `phase19-latency/raw/phase1-accounting.md` - that is exactly what we are replacing),
and decide cache hit/miss per call.

## Sources
* Ledger: state.db `api_calls` for the session (42 rows: `call_seq, started_at, ended_at, latency_seconds, input_tokens, cache_read_tokens, output_tokens, reasoning_tokens, prompt_tokens_total`).
* Log: m4-1 boot window saved at `/Users/adam.durham/.hermes/cache/scratch/phase20/m41_0901_1400.log.zst` (`zstdcat`; the window 2026-10-07 09:20:00-09:52:00 contains the whole turn; it also contains ~47 POSTs vs 42 calls because of other activity - match carefully).
  Cross-check the same markers on m4-2 if you can obtain its log for that window (`ssh studio2 'ls ~/.exo/exo_log/'`; the 09:01-14:00 boot is `exo.2026-10-07_09-01-57_725076.log.zst` there; `scp` it to the scratch dir first, read-only). Rank/node: m4-1 = studio1 = rank 1 (API/master node); m4-2 = studio2 = rank 0.
* Prior art (re-derive, do not copy blindly): `/private/tmp/next14-gamma/docs/benchmarks/phase19-latency/raw/A1-real-turn-split.md`, `phase19-latency/raw/phase2-delta-audit.md`.

## Marker semantics (verify each on real lines, document in turn_decomp.md)
* `API request: POST /v1/chat/completions` - client arrival (t_post).
* `Starting task TextGeneration(` / `received chat request` / `runner running` - worker/runner picked it up.
* `[DSV41] session reuse: this N-token prompt matches a resident conversation on M rows` and `[DSV41] prefill controls: ... (rows=R, base=2048)` - near the START of prefill (+0.2-0.4 s after POST). NOT emitted for every call (check which calls lack `prefill controls`).
* `[DSV41] turn reuse: prompt=N prefill=M reuse=R cache=C rewind=W UNCOMMITTED` - emitted when `session.prefill` RETURNS (prefill-completion instant), only when reuse>0. Call 1 is cold (no such line).
* `runner idle: reclaimed MLX allocator pool` / `runner ready` and `Executing command: TaskFinished(...)` - request end.
Define and document: `t_prefill_start` (prefer prefill-controls/session-reuse line else t_post), `t_prefill_end` (turn-reuse line; for the cold call 1 look for ANY real end-of-prefill marker, e.g. the first `hier geometry`-free gap, first decode-side line, or the stderr `[DSV41]` lines; if none exists, mark call 1 prefill/decode split as UNSPLIT - do not estimate), `t_end` (TaskFinished / runner ready; compare with the ledger `ended_at`).
`prefill_s = t_prefill_end - t_prefill_start`, `decode_s = t_end - t_prefill_end`, plus `pre_s = t_prefill_start - t_post` and `post_s = ledger ended_at - t_end` (should be tiny). Report the per-call residual `latency_s - (pre_s + prefill_s + decode_s + post_s)`.
Clock note: state.db epochs are UTC; exo.log local CDT (UTC-5 today); verify alignment by matching each call's POST line to `started_at` (expect offsets < 1 s); report the measured laptop<->node skew implied.

## CSV (exactly these columns, one row per call, 42 rows, plus extra columns after them if useful)
`call_idx, started_at, ended_at, latency_s, ctx_tokens_at_call, delta_rows_prefilled, prefill_s, output_tokens, decode_s, gap_to_next_call_s, cache_hit`
(`started_at/ended_at` as ISO CDT strings AND keep epoch columns after the required ones; `ctx_tokens_at_call = prompt_tokens_total`; `delta_rows_prefilled` = `prefill=` from the turn-reuse line, or `rows=` for cold call 1; `gap_to_next_call_s = next.started_at - this.ended_at`, blank for the last;
`cache_hit` per the plan rule: `delta_rows <= 1.2 x (ctx_at_call - ctx_at_prev_call)` -> hit; call 1 = cold start, `cache_hit=false` but EXCLUDED from the miss count and labelled cold. If `ctx_at_call - ctx_at_prev_call <= 0` the rule degenerates - flag the row and show what you did.)
Extra analysis columns (after the required ones): `reuse_rows, rewind, decode_tok_s = output_tokens/decode_s, prefill_rows_per_s = delta_rows/prefill_s, pre_s, post_s, residual_s, split_status`.

## Summary json + md
Totals: `sum(prefill_s)`, `sum(decode_s)`, `sum(gap_to_next_call_s)` (expect 283.86), miss count, `sum(latency)` (expect 1530.59), and the **gate**: `sum(prefill_s + decode_s)` within +-10% of 1530.59 (report the residual as "model-side unaccounted"). Time-weighted shares (decode vs prefill of model time; model time vs wall span 1814.45 s). Per-call decode tok/s distribution (min/median/max, weighted mean = sum(output)/sum(decode_s)) and prefill rows/s distribution (excluding calls <500 rows where fixed overhead dominates; say so). Also assert for all calls with a turn-reuse line that `prompt == reuse + prefill` (this establishes rows == tokens for prefill; report any exception).
Falsifier (plan): if per-call components cannot be reconstructed from logs, stop after 30 minutes and report exactly which fields are missing - do not estimate.

## Tests
A pytest that runs the extractor against a compact REAL fixture (a few hundred log lines for 3 calls: the cold call, a mid-size delta, a tiny delta - extract with zstdcat+grep into `bench/phase20_tests/fixtures/turn_decomp/`, <100 KB) and asserts parsed values; a CSV schema test; a cache-hit-rule unit test incl. the degenerate case.
`cd /private/tmp/phase20-campaign && PYTHONPATH=bench /Users/adam.durham/repos/exo/.venv/bin/python -m pytest bench/phase20_tests/test_phase20_turn_decomp.py -q -p no:cacheprovider`

## Deliverable
Commit (exact-path adds, do not push). Report: SHA, status, pytest line, the three totals + miss count + gate verdict, the marker-semantics findings (esp. which calls lack which markers and why), and a NOT-verified list. Paste the first 5 and last 3 CSV rows.
