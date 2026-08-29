# P01a — all_sum arrival-skew at depth, measured directly inside the collective (2026-08-29)

## 1. Question and why it was still open

The 2026-08-24 additivity doc (`docs/p3-followup-allsum-wait-at-depth-2026-08-24.md`)
ruled the +1.67..+2.52 ms/tok depth residual "NOT collective" from GPU
occupancy — but flagged two honest holes: (§4.3) its arrival-skew figure
(+0.070 ms/tok growth) was *derived* from occupancy arithmetic, not
measured; and (§5.3) a wait *hidden inside* command-buffer "busy" would be
invisible to the occupancy method, with the "decisive experiment" being a
direct CPU-side timer on the collective. This work runs that experiment:
per-call, per-rank, cross-rank-matched timing **inside** the jaccl
collective itself, at both depths, on the production build+config.

## 2. Method

- **Instrumentation:** `JACCL_TRACE_CALLS=1 JACCL_TRACE_TIMING=1` relaunch
  (relaunch #1) on both nodes — full production verbon3 env plus ONLY the
  two trace vars (env diff verified: the trace launch scripts differ from
  `verbon3_launch.sh` by exactly `JACCL_TRACE_CALLS=1 JACCL_TRACE_TIMING=1`).
  The timer is `steady_clock` inside C++ `reliable_all_reduce_v2`
  (`mesh.cpp`), so each rank's `transport_us` INCLUDES its wait for the
  peer's data: cross-rank per-call difference is the arrival-skew signal,
  immune to MLX graph laziness and to the occupancy method's §5.3 hole.
- **Probes:** two serialized runs of `bench/p3_depth_anchor_probe.py`
  (nonce-fronted non-degenerate filler, `use_prefix_cache=false`,
  EOS-banning `/bench` route), executed one at a time with pre-launch
  pgrep checks and call_id watermarks recorded between probes.
  - Probe 1: **real `usage.prompt_tokens=100,022`**, 409 completion tok,
    decode 36.42 tok/s (27.46 ms/tok), TTFT 305.8 s.
  - Probe 2: **real `usage.prompt_tokens=352,645`**, 320 completion tok,
    decode 28.06 tok/s (35.64 ms/tok), TTFT 1022.5 s.
  - (Both returned `finish_reason=stop` despite the bench route's EOS ban
    — see Limitations.)
- **Matching:** per-rank trace files matched by `call_id`. Probe 2:
  22,556/22,556 calls matched (100%). Probe 1: analyzed up to its
  recorded watermark (call_id ≤ 13,285), 100% match within it.
- **Segmentation:** decode = everything after the last 16 MB
  prefill-chunk collective; verified against expected cadence (~16-17
  verify-class collectives per token under batched verify).
- Raw traces + analyzer + JSON: `tmp/p01a-20260829/` (traces_probe1/,
  traces_probe2/, `analyze_skew.py`, `skew_analysis_results.json`,
  probe JSONs, watermark files).

## 3. Results

### 3.1 Decode verify-class collectives (32 KB / 24 KB / 16 KB — the per-token cadence)

| depth | calls/tok | r0 transport ms/tok | r1 transport ms/tok | Σ\|skew\| ms/tok | med \|skew\|/call |
|---|---|---|---|---|---|
| 100K | 16.1 | 0.883 | 0.785 | **0.494** | 4.8–6.2 µs |
| 352.6K | 17.1 | 0.964 | 0.858 | **0.573** | 4.7–7.3 µs |
| **Δ** | | **+0.081** | **+0.073** | **+0.079** | ~0 |

- **Arrival-skew growth with depth: +0.079 ms/tok** — 3–5% of the
  +1.67..+2.52 residual band. Independently confirms the 2026-08-24
  occupancy-derived +0.070 ms/tok with a completely different method.
- **Total in-collective time (transport + peer-wait) is also depth-flat**:
  each rank's summed verify-class time grows < +0.09 ms/tok. Even
  charging every microsecond inside the collective to the residual can't
  close it — this directly closes §5.3's decisive-experiment ask for the
  in-collective component.
- Median per-call transport is depth-FLAT (36.9–39.9 µs @100K vs
  37.4–40.1 µs @352.6K), extending the 2026-08-21 "transport is fast"
  result from ~512-token context to 352.6K.
- **No straggler rank:** r0-slower = 44–50% of calls in every class at
  both depths (coin flip). Severe (>1 ms) skew events: 0.03–0.34% of
  verify-class calls, no consistent direction. The 2026-08-22 build's
  4.2× rank0-straggler tail asymmetry does NOT reproduce on this build.

### 3.2 Per-request (NOT per-token) classes — where skew IS large

Two decode-segment populations carry big skew but occur a fixed number
of times per REQUEST, so they amortize toward zero over long
generations and cannot own a steady-state per-token residual:

| class | n/request | total \|skew\| @100K | @352.6K |
|---|---|---|---|
| 8192 B (sequential-path/warm cycles) | 215 | 176.7 ms | 357.5 ms |
| multi-MB transition/tail collectives | 85 / 44 | 74.8 ms | 297.2 ms |

Naively dividing these by completion tokens is what makes "all-decode
skew" look like it grows +1.51 ms/tok — an artifact of the short (320–409
tok) windows. Flagged so nobody re-derives a phantom regression from it.

### 3.3 Prefill (recorded for completeness; not the residual's domain)

16 MB chunk collectives: mean |skew| ≈ 1.3–1.4 ms/call, 13.5–14.6% of
chunks with >1 ms skew, straggler split 49–50% (symmetric). Total
prefill skew 5.8 s @100K → 18.9 s @352.6K (~1.9% of the 1022 s TTFT).
Real but a prefill-wall-clock topic; scales with chunk count, per-chunk
skew depth-flat.

## 4. Verdict

**Phase 1(a) CLOSED: arrival skew at depth is ruled out as the residual's
owner.** Directly measured inside the collective (both ranks,
call-matched, production build), per-token arrival skew grows only
+0.079 ms/tok from 100K→352.6K — 3–5% of the band — and total
in-collective time grows < +0.09 ms/tok/rank. Combined with 2026-08-24's
occupancy result (idle shrinks with depth) the collective story is now
closed from BOTH sides of the §5.3 divide: not in idle, not inside the
collective call. Residual remains unattributed on-GPU busy work;
surviving candidates: MoE-at-depth interplay, allocator/~90 GB-resident
regime.

## 5. Limitations

- **Cadence normalization:** these probes ran the PRODUCTION spec-on
  config (DSpark/MTP + batched verify), where verify-class collective
  cadence is ~16-17 calls per emitted token, not spec-off's 43. Per-CALL
  skew (the fundamental signal) is depth-flat: mean |skew|/call 30.7 µs
  @100K → 33.5 µs @352.6K. Rescaled to spec-off's 43-calls/tok cadence
  (the regime the +1.67..+2.52 band was derived in): 1.32 → 1.44 ms/tok,
  growth **+0.12 ms/tok ≈ 5-7% of the band** — conclusion unchanged in
  both regimes.
- n=1 probe per depth (skew stats are over 5.5–6.6K matched calls per
  depth, but one request each).
- `/bench` route returned `finish_reason=stop` at 409/320 tokens instead
  of running to `max_tokens=2000` with `length` — shorter decode windows
  than intended (still ≥5.4K verify-class collectives per depth; medians
  and totals are stable). Worth a separate look at why the EOS ban
  didn't hold on this build; does not affect skew arithmetic, which is
  per-call.
- Probe 1's trailing calls (a client-timeout-killed first probe-2
  attempt, cancelled server-side via `PREFILL_CANCELLED_PATH`) were
  excluded via the recorded call_id watermark (13,285); probe 2 ran on a
  fresh JIT-cycled instance with its own trace files (100% match).
- The CPU-side timer bounds the in-collective wait only; a GPU-side
  fence wait after the collective is not measured here (it was bounded
  by the 2026-08-24 occupancy work and 1(b)'s chain harness).

## 6. Cluster state after

Production restored (relaunch #2): SIGTERM'd trace-on runners, verified
zero exo processes/screens, relaunched byte-identical
`/tmp/verbon3_launch.sh` on both nodes. Verified: `ps eww` shows ZERO
`JACCL_TRACE_*` vars (and all production flags: MTP=1, SPECULATIVE=1,
DSPARK=1, VERIFY_BATCH=1, MIN_CTX=8192); no new jaccl trace files
created by the new runners (behavioral proof); smoke probe @2K clean —
coherent output, 35.15 tok/s decode, finish_reason=length.
