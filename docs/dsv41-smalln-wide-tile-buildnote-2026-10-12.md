# DSv4.1 — small-n wide tile (R-B): the 128 MiB route cliff — deploy buildnote (2026-10-12)

**Branch:** exo `deploy/next24-routing` = live tip `60008939` + this fix. **mlx-lm pin:**
`6a5c78f4` → **`a0573e3`** (`perf/indexer-route-v2`; gitlink + `uv.lock` advanced together).
**Status at write time:** constructed + validated OFFLINE; **not deployed** (0 boots).

## The defect (in situ)

Verify forwards (γ3 → n=4) and decode (n=1) take the indexer's full-width fallback
(`n ≤ _FENCE_MIN_ROWS = 16`). Above a context-dependent crossing the 128 MiB budget sends
them down the **tiled** branch at `min(_TILE=512, cap)` columns per tile — and every tile
ends in a blocking host fence (~135–175 µs) over ~10 µs of GPU work, so the call is
fence-bound (586 fences at 300K, ~90 % GPU idle). Live, current build (idxprof6):
steady γ3 decode rounds **103.5 ms at 245K vs 628 ms at 300K vs 978 ms at 500K** — a
6–9× cliff; the 245K control proves it is the rule, not the depth. At γ3: 5 of 8 indexer
calls per round are hit above 254.2K; all 8 above 508.4K.

## The fix (R-B)

`mlx_lm/models/deepseek_v41/indexer.py` — in `tile_width()` only, when
`n ≤ _FENCE_MIN_ROWS`: the width becomes the **aligned budget cap** instead of
`min(_TILE, cap)` (at n=4: 254,200 columns; 2–4 fences per call instead of hundreds).
Unchanged: the tiled/untiled decision, the budget, the L2-full fp32 row, branch/top-k,
and the routing key on the full chunk's row count (D3). Deterministic, static, and
rank-identical. **Switch `DSV41_INDEXER_SMALLN_WIDE_TILE`** (read at import): default `1`
(new rule); `0` restores today's rule **byte-for-byte** (verified 67/67) — forwarded by
the launcher (line added next to `L2_FULL`), so rollback can be an env flip at relaunch.

## Evidence

- Offline, real patched code (idxprof7; idxprof5 grid): (4,300K) **81.5 → 4.7 ms (17×)**;
  (4,500K) **136.4 → 5.9 ms (23×)**; n=8/12/16 cells 9–12×; decode-like rounds at
  300K/600K/1M flat at **4.1 / 7.0 / 11.1 ms** vs today's 80 / 160 / 265 ms; one compiled
  body, no recompile spikes. Peak ≈ 268 MB at n=4, independent of nb.
- Expected live effect (DERIVED from idxprof6): **~515 ms saved per round at 300K**
  (~5.8×, ~3.5 → ~20 tok/s), **~870 ms at 500K** (~9×). No change below 254K at γ3.
- Exactness: index sets + candidate masks identical across new/old/untiled in 67/67
  harness cases (bench + production-invariant env) and 87 unit cases incl. injected exact
  ties and remainder edges. The fp32 row may differ at the ulp level in ≤ a few elements
  at some shapes (already true today for 512-vs-untiled at 1M) → **the live token-identity
  battery is the gate.** Fallback if it diverges: `=0` / R-C (bitwise-identical alternative).
- Tests: new suite **9/9**; dsv41 suites **347/10/2** patched vs **338/10/2** baseline —
  the 10 failures are pre-existing at `6a5c78f4` (triage separately).

## Deploy gates (all must pass, in order)

0. **Provenance.** venv `mlx_lm/models/deepseek_v41/indexer.py` md5 == signed tree, on
   both nodes; runner mlx-lm sha reads `a0573e3`.
1. **Token-identity battery.** Fixed greedy prompts, temp 0, cold
   (`use_prefix_cache=false`): standard short/mid set + **≥1 decode at L > 254.2K**
   (idxprof6 p300k, sha256 f1c941f3…) **+ ≥1 at L > 508.4K**. Pass: byte-identical
   outputs vs pre-deploy captures. On any mismatch: do not ship; flip
   `DSV41_INDEXER_SMALLN_WIDE_TILE=0` (or R-C) and report.
2. **Quality battery, quick**, incl. a ≥350K needle case; no regression vs the last
   recorded baseline.
3. **Deep decode tok/s, before vs after.** idxprof6 method (master `ChunkGenerated`
   clusters, 20 ms rule, steady rounds, first round excluded). At p300k expect
   ~100–110 ms/round (was 628); at p500k ~105–115 ms (was 978); 245K control unchanged
   (~103.5). Pass: 300K/500K rounds ≤ 1.2× the 245K control. Record ioreg util (~93 %)
   and runner CPU (~1.6 cores) if convenient.
4. **Hygiene.** No ERROR/CRITICAL/SIGKILL/OOM in either node log; both runners
   `RunnerReady`; GPU locks free.

## Rollback

- **Env flip (no code change):** relaunch with `DSV41_INDEXER_SMALLN_WIDE_TILE=0`
  (forward line now present on this branch) — reproduces today's widths/outputs
  byte-for-byte (67/67 offline).
- **Full:** reset both nodes to `60008939` and relaunch (previous pin `6a5c78f4`).

## Out of scope / known

- Banded prefill tails with n_full 17–32 still tile at 512 (~0.01–0.03 % occupancy).
- The first decode round after a long prefill (5.0 s at 500K) is a separate defect; exclude
  it from the tok/s comparison.
- 10 pre-existing red dsv41 tests at `6a5c78f4` (incl. `test_dsv41_indexer_smallm_hier.py`).
- `bench/indexer_split.build_indexer` never installed real `wq_b`/`wk` (random-init
  `nn.Linear`); idxprof7 re-ran with real projections (conclusions hold). Shared harness
  fix still pending.
