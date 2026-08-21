# MoE gate+up fusion: validated decode win, prefill neutral — 2026-08-21 (session 2, part 4)

## Lever tested

`EXO_DSV4_MOE_FUSED_GATE_UP=1` — installs `FusedSwitchGLU`
(`src/exo/worker/engines/mlx/auto_parallel.py::_install_fused_gate_up`),
which pre-concatenates `gate_proj` + `up_proj` weights so MoE dispatches
ONE `gather_qmm` for both instead of two (plus `down_proj` unchanged =
2 total dispatches instead of 3). Bit-equivalent math: same weights,
just concatenated along the output axis and split back after. This was
added 2026-06-26 (commit `7bdda3955`), wired through `start_cluster.sh`,
off by default, and **never A/B tested** until tonight — no follow-up
commit or doc existed. The commit message itself flagged it needed
testing for quality (needle) AND decode t/s before trust, especially at
B>1 due to a historical (different, already-removed) fusion's B>1
degeneration bug.

## Method

1. Relaunched cluster with `EXO_DSV4_MOE_FUSED_GATE_UP=1`. Verified live
   on both nodes via `ps aux`; confirmed no "fusion failed" warning in
   either node's log (clean install).
2. Ran the standard 100K/300K/500K depth ladder — all three needle checks
   passed, throughput at parity with the known-good baseline (362.8/
   348.5/331.1 tok/s prefill vs baseline 368.7/351.9/333.2 — within
   normal run-to-run noise).
3. `bench/exo_bench.py` requires the `exo_tools` workspace package which
   isn't importable from the plain `.venv` (separate uv workspace member
   per AGENTS.md) — wrote a minimal standalone decode-focused probe
   (`bench/decode_probe.py`, new file, checked into the repo) instead:
   small prompt (~512 tok), `bench: true` (bans EOS, forces the full
   requested length so decode gets a clean long sample instead of
   natural-EOS-truncated short completions), `max_tokens=300`.
4. Ran 8 repetitions with fusion ON, then reverted to a clean baseline
   relaunch (confirmed `EXO_DSV4_MOE_FUSED_GATE_UP` absent via `ps aux`)
   and ran 8 repetitions with fusion OFF, same probe, same prompt.
5. Verified real generated text quality on both configs (not just tok/s
   per standing rule) — coherent, correct, on-topic completions in both.

## Result

| Config | n | mean decode tok/s | stdev |
|---|---|---|---|
| Fusion ON | 8 | 18.879 | 0.158 |
| Fusion OFF (baseline) | 8 | 18.328 | 0.173 |

**+3.01% decode throughput**, clean separation (means differ by ~3.2×
combined stdev — not noise). Prefill throughput unaffected (within noise
at all three depths) — expected, since MoE dispatch-overhead savings are
a fixed per-layer cost that matters proportionally more when each
forward pass does very little compute (single-token decode steps) than
when each forward pass does a large chunk of prefill compute.

Quality: verified coherent, correct, on-topic real generated text on the
fusion-ON config (CAP theorem explanation, factually correct, well-formed
sentences) — not just a token-count metric.

## Concurrency (B>1) note — deliberately not chased tonight

An unrelated crash was hit while probing concurrency=2 during this
investigation: `IndexError: list assignment index out of range` in
`PoolingCache.store_overlap_carry` (`mlx-lm/mlx_lm/models/cache.py:2020`,
`self._overlap_carry_valid[i] = True` — a batch-size mismatch between
`_overlap_carry_valid`'s allocated length and the `produced` list it's
indexed against). Full traceback confirms this is in the DSv4
`SparseCompressedAttention` → `compressor` → `pool_cache.store_overlap_carry`
path — nowhere near `SwitchGLU`/`gather_qmm`/the fusion code under test.
This is very likely a pre-existing multi-stream batching bug independent
of `EXO_DSV4_MOE_FUSED_GATE_UP`, not something introduced by tonight's
lever. Per explicit user direction, concurrency correctness was NOT in
scope for this investigation — noted here for the record, not chased.
The runner self-healed cleanly (restarted, cluster returned to healthy,
c=1 requests continued working normally) — no reboot was needed.

## Conclusion

**Real, validated, currently-untapped +3% decode throughput lever.**
Recommend enabling `EXO_DSV4_MOE_FUSED_GATE_UP=1` as a new default in
`start_cluster.sh` for c=1 production use, pending the user's explicit
sign-off (not done automatically — this changes default runtime
behavior, distinct from a diagnostic-only relaunch). B>1 concurrency
correctness remains unvalidated for this flag specifically (blocked on
the unrelated pool_cache crash above, which prevents testing B>1 at
all right now, fusion on or off) and should not be assumed safe at
concurrency>1 until that's separately verified.

## New file added

`bench/decode_probe.py` — minimal decode-throughput A/B probe, no
external workspace-package dependency beyond `httpx` (already in
`.venv`). Reusable for any future MoE/attention micro-optimization
A/B test that needs a clean long-decode sample rather than the
depth-ladder script's short natural-EOS completions.
