# P3 worker C2 — live two-rank GPU busy/idle capture at 100K context; deep (352.6K) point BLOCKED by a runner GPU-timeout that the capture itself probably triggered — 2026-08-23

**One-line result**: at ~100K real context the live cluster's GPU is
**82.98% (rank0) / 83.06% (rank1)** busy over a 50s decode-interior window, i.e.
**idle ≈ 6.1–6.5 ms/token per rank** — but the planned 352.6K counterpart was
never captured, because the 100K capture's stop/finalize phase is the prime
suspect in a `[METAL] GPU Timeout Error` that killed rank1's runner and took the
model instance down. **The pre-registered two-depth decision rule is therefore
UNRESOLVED**, with a clearly-labelled secondary cross-methodology comparison
below that leans (weakly) toward "idle is flat with depth".

**Status of the cluster at the end of this work**: `instances: []`, runner
`6ac91846` (m4-2 / rank1) `RunnerFailed`, runner `f85456ee` (m4-1 / rank0)
`RunnerRunning`. **I did not relaunch it** — the task's hard rules forbid
cluster relaunch / runner restart / model load-unload, and I held that line even
though it cost me the second half of my own measurement. A human or a
differently-scoped worker must restore the instance.

---

## 0. What was asked vs what exists

| deliverable | status |
|---|---|
| busy/idle ms/token, rank × depth, at ~100K | **DONE** (both ranks, n=1) |
| busy/idle ms/token, rank × depth, at ~352.6K | **NOT DONE** — cluster down before it could run |
| decision-rule verdict (idle flat vs idle grows) | **UNRESOLVED** — needs both depths by construction |
| T5 occupancy comparison / 9s-window methodology gap | **PARTLY CLOSED** — a 50s window now exists, at 100K (T5's was 9s at 300K) |
| powermetrics GPU clock | **SKIPPED** — `sudo -n true` returns "a password is required" on both studios; per the task's rule, not attempted |

---

## 1. Method

Identical read-only attach methodology to T2
(`docs/gpu-occupancy-clock-gap-postfix-2026-08-22.md`) and T5
(`docs/long-context-gpu-occupancy-2026-08-22.md`).

**Runner PIDs** (live, found via `ps -axo pid,ppid,command`, the
`multiprocessing.spawn` children of `python -m exo`, each ~24.9 GB RSS):

```
m4-1 (rank0, API node) : 46718   (parent 43509 = .venv/bin/python -m exo -v)
m4-2 (rank1)           : 45206   (parent 42153 = .venv/bin/python -m exo -v)
```

**Load generator**: `/tmp/c2_probe.py`, a thin wrapper that
`importlib`-loads B1's `bench/p3_depth_anchor_probe.py` and reuses its
`build_prompt()` **verbatim** — same UUID nonce at the prompt front, same
8-topic non-degenerate filler, same tokenizer-targeted binary search, same
`use_prefix_cache: False`, same **EOS-banning `/bench/chat/completions`
route**, same `max_tokens=2000`, same depth cap of 355,000. The only additions:
(a) it writes a marker file the instant the first streamed event arrives, and
(b) it records absolute `time.time()` per streamed event so the client-side
per-token stream can be restricted to exactly the trace window afterwards.
Decode-window math is B1's unchanged: window = last stamp − first stamp, so
TTFT/prefill is outside it by construction.

**Capture orchestration**: `/tmp/c2_capture.py` blocks on the marker file, waits
a further 6s, then fires both ranks' `xctrace` in parallel over ssh:

```
xcrun xctrace record --template 'Metal System Trace' --attach <pid> \
    --output /tmp/c2_<tag>_<rank>.trace --time-limit 50s
```

**Prefill is never traced.** Per
`docs/p2-xctrace-prefill-collective-wedge-2026-08-23.md`, attaching during a live
deep prefill wedged the collective 3/3 and is a documented HAZARD. The capture
starts 6s *after* the first decode token and runs 50s, entirely inside a 71.2s
decode window.

**Export + parse**: `xctrace export --xpath
'/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'` (the
established workaround; `--toc` still fails on attach-mode traces), parsed by
`/tmp/c2_occ_parse.py` — a streaming `ElementTree.iterparse` script that
resolves Instruments' id/ref back-reference compression, keeps **only rows whose
`process` field contains our runner PID**, and computes **interval-union** busy
time (merged overlaps, never a naive sum, which would exceed wall on concurrent
command buffers).

**Pipeline validation before the real run**: a 15s smoke capture on rank1 during
a short 400-token request exported and parsed cleanly (51,714 own-process rows,
75.59% whole-span / 82.06% trimmed occupancy), confirming PID selection, the
xpath, and the union math all work end to end.

---

## 2. Timeline of the 100K run (real, from logs)

```
13:45:03.777  PROMPT_READY (544,306 chars, locally predicted 100,021 tokens)
13:50:07.701  DECODE_START  ttft=303.92s          <- prefill ends, decode begins
13:50:13.723  CAPTURE_LAUNCH (decode_start + 6.02s)
                rank0 adams-mac-studio-m4-1: xctrace pid 57262, launched at +0.40s
                rank1 adams-mac-studio-m4-2: xctrace pid 55719, launched at +0.40s
13:51:03.72   capture --time-limit 50s expires; both "Reached specified time limit"
13:51:10.2    (decode_start + 62.5s) FIRST large inter-token stall: 880.7 ms
              ... cascade of 200ms-2.6s stalls, 12 of them, over the next 6.1s
13:51:18.921  last streamed event; stream ends WITHOUT a usage block
              rank1 runner dies: [METAL] Command buffer execution failed:
                                 Caused GPU Timeout Error (00000002:***)
```

Both ranks were attached **within 0.40s of each other** and the capture window
(`decode_start+6.0s → +56.0s`) sits strictly inside the decode window
(`0 → 71.2s`). Clock alignment was verified explicitly, not assumed: the trace
spans are 50.554s (rank0) and 50.455s (rank1) against a commanded 50s limit, and
the probe's own wall stamps put 1330 streamed events inside the aligned window.

---

## 3. Raw parse output (verbatim)

### rank0 — m4-1, pid 46718

```json
{
 "label": "100k_rank0",
 "pid": 46718,
 "total_rows_in_export": 250435,
 "own_process_rows": 214182,
 "own_process_rows_depth0": 156582,
 "channel_counts": {"Compute": 214136, "Fragment": 34, "Vertex": 12},
 "top_processes": [["python3.13 (46718)", 214182], ["WindowServer (516)", 36155],
                   ["SecurityAgent (659)", 70], ["", 28]],
 "span_lo_ns": 401104541,
 "span_hi_ns": 50954841083,
 "whole_span": {"window_s": 50.553736542, "busy_s": 41.949511788,
                "busy_pct": 82.98004194635224, "idle_pct": 17.01995805364776,
                "n_merged_runs": 156408, "n_gaps": 156407},
 "trimmed_2s": {"window_s": 46.553736542, "busy_s": 38.648062308,
                "busy_pct": 83.01817464884343, "idle_pct": 16.98182535115658,
                "n_merged_runs": 144067, "n_gaps": 144066},
 "trimmed_2s_depth0_only": {"window_s": 46.553736542, "busy_s": 37.70735412,
                "busy_pct": 80.99748145023989, "idle_pct": 19.002518549760108},
 "burst_isolated": {"n_bursts": 1, "window_s": 50.553736542,
                "busy_s": 41.949511788, "busy_pct": 82.98004194635224,
                "idle_pct": 17.01995805364776, "burst_spans_s": [50.554]},
 "gap_stats_us": {"n": 144066, "median": 0.917, "mean": 54.875364305248986,
                  "p95": 164.458, "p99": 239.916, "max": 131472.5,
                  "total_ms": 7905.674234, "gaps_gt_1ms": 80,
                  "time_in_gaps_gt_1ms_ms": 510.891375}
}
```

### rank1 — m4-2, pid 45206

```json
{
 "label": "100k_rank1",
 "pid": 45206,
 "total_rows_in_export": 249760,
 "own_process_rows": 213816,
 "own_process_rows_depth0": 156494,
 "channel_counts": {"Compute": 213772, "Fragment": 36, "Vertex": 8},
 "top_processes": [["python3.13 (45206)", 213816], ["WindowServer (501)", 35827],
                   ["SecurityAgent (637)", 70], ["", 47]],
 "span_lo_ns": 390699250,
 "span_hi_ns": 50845851083,
 "whole_span": {"window_s": 50.455151833, "busy_s": 41.907108484,
                "busy_pct": 83.05813571368705, "idle_pct": 16.94186428631295,
                "n_merged_runs": 156155, "n_gaps": 156154},
 "trimmed_2s": {"window_s": 46.455151833, "busy_s": 38.59300166,
                "busy_pct": 83.07582719509051, "idle_pct": 16.924172804909492,
                "n_merged_runs": 143821, "n_gaps": 143820},
 "trimmed_2s_depth0_only": {"window_s": 46.455151833, "busy_s": 37.643583054,
                "busy_pct": 81.0320956205753, "idle_pct": 18.9679043794247},
 "burst_isolated": {"n_bursts": 1, "window_s": 50.455151833,
                "busy_s": 41.907108484, "busy_pct": 83.05813571368705,
                "idle_pct": 16.94186428631295, "burst_spans_s": [50.455]},
 "gap_stats_us": {"n": 143820, "median": 0.916, "mean": 54.6659331525518,
                  "p95": 165.125, "p99": 265.208, "max": 131520.875,
                  "total_ms": 7862.054506, "gaps_gt_1ms": 75,
                  "time_in_gaps_gt_1ms_ms": 486.486675}
}
```

### Client-side per-token stream, restricted to the aligned trace window

```
capture launch wall 1787511013.723  decode_start 1787511007.701  (= +6.02s)
rank0 trace span 50.554s  busy% 82.98
rank1 trace span 50.455s  busy% 83.06

EVENTS in aligned window: 1330 over 50.532s
  event-based ms/token (window) = 38.023
  gaps n=1329 median=35.79 mean=38.02 p90=54.15 max=164.51
```

---

## 4. The 2×2 table — one row measured, one row missing

**Measured occupancy (%)** — the only primary quantity this capture produces:

| depth | rank0 busy % | rank1 busy % | window |
|---|---|---|---|
| **~100K (this run)** | **82.98** | **83.06** | **50.55 s, decode-interior** |
| ~352.6K | *NOT CAPTURED* | *NOT CAPTURED* | — |

**Measured decode ms/token** (client-side, same run, same window):
**38.023 ms/token** in-window (median inter-token gap 35.79 ms).
B1's untraced 100K anchor for the same prompt construction: **35.79 ms/token**.

**DERIVED busy/idle ms/token** — this is arithmetic on the two measured
quantities above, kept explicitly separate from them:

| basis | rank | busy ms/tok | idle ms/tok | idle ÷ 43 all_sums |
|---|---|---|---|---|
| traced in-window (38.023 ms/tok) | rank0 | 31.55 | **6.47** | 0.150 ms |
| traced in-window (38.023 ms/tok) | rank1 | 31.58 | **6.44** | 0.150 ms |
| B1 untraced anchor (35.79 ms/tok) | rank0 | 29.70 | **6.09** | 0.142 ms |
| B1 untraced anchor (35.79 ms/tok) | rank1 | 29.73 | **6.06** | 0.141 ms |

The traced-basis row is what the trace itself saw; the untraced-anchor row
removes the ~6% tracing overhead by applying the same occupancy fraction to
B1's clean anchor. Both are given because neither is unambiguously "the"
answer — tracing inflates wall time, but occupancy is a ratio and is the more
transferable quantity.

**Per-rank symmetry is the notable single-depth finding.** rank0 and rank1
agree to **0.08 percentage points** of occupancy (82.98 vs 83.06) and to
**0.03 ms/token** of derived idle. At 100K, the two ranks are not waiting on
each other asymmetrically in any measurable way — there is no detectable
arrival skew between them at this depth. That is a real, if partial, constraint
on the all_sum-skew hypothesis: whatever collective wait exists at 100K, it is
**symmetric**, and it amounts to at most ~0.15 ms per all_sum call if *all*
idle were charged to the collective (which it certainly is not — ordinary
per-kernel CPU-dispatch latency lives in there too).

---

## 5. Decision rule — stated against, and honestly unresolved

The pre-registered rule was:

> if idle ms/token at 352.6K ≈ idle ms/token at 100K (within ~±1 ms) while busy
> ms/token grows by roughly the full observed wall delta, then
> GROWING-COLLECTIVE-WAIT IS RULED OUT and the residual is on-GPU busy work.

**I have one of the two required depths. The rule cannot be evaluated.** It is a
two-depth comparison; one point does not resolve it, and I am not going to
dress up a single measurement as a verdict.

### Secondary, cross-methodology comparison — SUGGESTIVE ONLY

Prior captures on this same cluster with this same methodology give two other
occupancy points. Combining each with the matching B1/T1 decode anchor yields
an idle ms/token series:

| depth | occupancy (source) | anchor ms/tok | derived busy | derived **idle** |
|---|---|---|---|---|
| short ctx (~520) | 78.64 / 78.86% (T2, 30s window) | 33.75 (B1) | 26.54 / 26.62 | **7.21 / 7.13** |
| **~100K** | **82.98 / 83.06% (this run, 50s)** | 35.79 (B1) | 29.70 / 29.73 | **6.09 / 6.06** |
| ~300K | 82.43 / 82.70% (T5, 9s window) | 42.59 (B1 @352.6K) | 35.11 / 35.22 | **7.48 / 7.37** |

Read narrowly: derived idle sits in a **6.1–7.5 ms/token** band across a 680×
range of context depth, with **no monotone growth** — it is *lower* at 100K
than at short context, and the 300K value is barely above short-context. Busy
time, by contrast, climbs monotonically and substantially (26.5 → 29.7 → 35.1
ms/token). That is the *direction* the decision rule's "ruled out" branch
predicts.

**Why this is NOT the verdict**, and must not be reported as one:

- **Three different capture windows** (30s, 50s, 9s) on three different runs.
  Occupancy is demonstrably window-length sensitive: my own smoke capture read
  75.59% whole-span vs 82.06% trimmed on the *same* trace.
- **T5's 300K point used a non-EOS-banned 9s window** — the very methodology
  gap this worker was sent to close. Its anchor pairing is also mismatched
  (T5 measured 300K; the 42.59 ms/tok anchor is B1's at 352.6K).
- **The depths differ** (300K vs the 352.6K the rule specifies).
- **n=1 everywhere.**

So: the residual's source remains **formally undetermined**. The honest summary
is that nothing in the available data supports *growing* collective wait, and
the weak cross-run evidence points at busy-work growth — but the clean,
same-methodology, two-depth experiment that would settle it did not happen.

### moe.all_sum verdict for P3

Downgraded from what the task hoped for. What can be said:

- payload is **L-independent** (code-verified by worker A: fixed `(1,1,4096)`,
  43×/token) — unchanged;
- **wait growth is not empirically bounded** — my one depth cannot bound a
  *growth*;
- at 100K, total per-rank idle is **6.06–6.09 ms/token** (untraced basis), an
  **upper bound** on all_sum wait at that depth (≤0.142 ms per call), and it is
  **symmetric between ranks to within 0.03 ms/token**.

---

## 6. The incident: the capture probably killed the runner

This is the second substantive finding, and arguably the more operationally
important one.

**What happened**: the traced 50s were metronomic (median inter-token gap
35.79 ms, max 164.51 ms, no stall above 200 ms anywhere inside the window).
**6.5 s after the capture ended**, a cascade of 12 stalls of 200 ms–2.6 s hit
in 6.1 s, the stream ended without a usage block, and rank1's runner died in
`mx.async_eval(y)` inside the MoE ffn with:

```
RuntimeError: [METAL] Command buffer execution failed: Caused GPU Timeout Error (00000002:***)
```

preceded by `[wait_for_one] slow: elapsed=3.0s n_active=6 (polling; self-abort at 20000ms)`.

**Attribution — probable, not proven.** The signature matches
`docs/p2-xctrace-prefill-collective-wedge-2026-08-23.md`'s incidents 1 and 2
closely: there, too, the runner ran healthily *during* the capture and the
fatal silence began **2–3 s after the tracer had already detached**. Here it is
6.5 s. That doc's incidents were all during prefill; mine was decode-only.

**Suspected mechanism (hypothesis, not measured)**: the *stop/finalize* phase,
not the recording. While recording, each trace directory was **10 GB**; it
finalized down to **1.7 GB**, and finalization took ~25 minutes of wall time
per rank. That is a large burst of I/O and memory pressure landing on a node
holding an ~85 GB-resident model — precisely when the stalls begin. This is
consistent with, but not established by, the evidence I have.

**This falsifies, or at least sharply narrows, P2's safety claim.** That doc
states plainly: *"Decode-window and idle-window captures remain fine (many
prior successes, §2.7/T2)."* That claim rests on short-context captures and on
T5's 9s/300K window. **It does not extend to a 50s capture at 100K depth.** The
risk scaler looks like **trace size/duration**, not prefill-vs-decode: T5's
300K capture was deep but only 9s and survived; mine was shallower but 50s and
did not.

**Recommended protocol for whoever retries this** (after restoring the cluster):

1. Keep the capture window in the **12–15 s** envelope that has repeatedly
   succeeded, not 50 s. Occupancy is a ratio; a 15 s interior window over
   ~400 decode tokens is statistically ample (my smoke run had 51,714 intervals
   in 15 s).
2. Decode-interior only, both ranks simultaneously — that part worked.
3. Expect resource pressure for **tens of seconds after** `--time-limit`
   expires; do not assume the run is safe the moment the tracer says "Reached
   specified time limit".
4. Budget ~25 min per rank for finalization before `export` will work.
5. Do the deep (352.6K) point **first** next time — it costs ~18 min of prefill
   and is the point that is actually missing.

---

## 7. T5 comparison — the 9s methodology gap, partially closed

T5 flagged that its 300K occupancy figure came from a ~9s window and asked for
a longer confirmatory capture. This run delivers a **50.5s** window — 5.6×
longer — with EOS **genuinely** banned via the `/bench` route (2000 max_tokens,
71.2s decode window), at 100K rather than 300K.

| capture | window | depth | rank0 | rank1 |
|---|---|---|---|---|
| T2 | ~30 s | short (~512) | 78.64% | 78.86% |
| **C2 (this)** | **50.55 s** | **~100K** | **82.98%** | **83.06%** |
| T5 | ~9 s | ~300K | 82.43% | 82.70% |

**T5's central qualitative claim survives the longer window**: occupancy at
depth is meaningfully HIGHER than at short context (~83% vs ~78.6–78.9%), and
both ranks agree. My 100K figure is, if anything, slightly *above* T5's 300K
figure, which mildly undercuts the idea that occupancy keeps climbing with
depth — it looks more like a step up from short-context to deep-context and
then a plateau. **The gap is closed at 100K, not at 300K+**; a long-window deep
capture remains outstanding.

Gap-shape also reproduces T5: **median gap 0.92 µs** (rank0) / **0.92 µs**
(rank1) with mean ~54.8 µs — T5 saw median ~0.96–1.00 µs at 300K vs ~89–95 µs
at short context. So the "many sub-microsecond back-to-back dispatch gaps"
structure is confirmed on a 5.6× longer window.

---

## 8. Limitations — read before citing anything above

- **n=1 at one depth.** The planned second depth does not exist.
- **The decision rule is unresolved.** §5's three-point comparison is
  cross-run, cross-window, cross-methodology, and must not be cited as the
  verdict.
- **No `usage` block for this run.** The stream died before the final usage
  chunk, so `usage.prompt_tokens` was never returned and **depth is inferred**:
  the same builder, same target (100,000), same tokenizer produced a locally
  predicted 100,021 tokens here, and landed at a real 100,026 on B1's run with
  an identical target. That is strong but indirect. `finish_reason` was also
  never delivered (`None`) — normally the EOS-ban check.
- **The traced decode ran ~6% slower than untraced** (38.02 vs 35.79 ms/token).
  Instruments captures its own dispatch cost; this is the expected, documented
  overhead. Reassuringly the in-window **median** gap (35.79 ms) matches B1's
  untraced mean almost exactly, which is evidence the in-window stream was not
  materially perturbed — the overhead shows up in the mean/tail, not the
  median.
- **Per-kernel attribution is still impossible from this template.**
  `channel_counts` is 214,136/214,182 = **99.98% "Compute"** on rank0 (same on
  rank1) — the same limitation T2/T3/T5 all hit. Nothing here isolates
  `moe.all_sum` from any other kernel; the "idle ÷ 43" figures are ceilings
  derived by division, not measurements of the collective.
- **Idle here is whole-process GPU idle**, which includes ordinary per-kernel
  CPU-dispatch latency, not just collective wait. Attributing all of it to
  all_sum would be wrong.
- **powermetrics not captured** — passwordless sudo unavailable on both
  studios; per the task's explicit instruction, not prompted for.
- **The runner-death attribution is a hypothesis** supported by timing
  coincidence with two prior P2 incidents, not a root cause.
- **The cluster was left down.** Deliberate, per the no-relaunch rule.
- **Nothing was committed to git.** No file under `~/repos/exo` on either
  studio was created, edited, or deleted; all studio artifacts went to `/tmp`
  and the multi-GB trace/XML files were removed afterwards (~24 GB reclaimed;
  m4-1 back to 137 Gi free, m4-2 to 161 Gi free).

---

## 9. Files

Local (this Mac):
- `/tmp/c2_probe.py` — probe wrapper reusing B1's `build_prompt` + `/bench` route.
- `/tmp/c2_capture.py` — marker-triggered simultaneous dual-rank capture launcher.
- `/tmp/c2_occ_parse.py` — streaming xctrace XML → interval-union occupancy parser.
- `/tmp/c2_100k.json` — full probe result incl. every per-token wall stamp.
- `/tmp/c2_100k.log` — raw probe stdout.
- `/tmp/c2_100k_capture_meta.json` — capture launch wall time, decode start, PIDs.
- `/tmp/c2_100k_rank0_occ.json`, `/tmp/c2_100k_rank1_occ.json` — parse output (§3).

On the studios (`/tmp` only, trace + XML files deleted):
- `/tmp/c2_100k_rank{0,1}_occ.json`, `/tmp/c2_100k_rank{0,1}_rec.log`,
  `/tmp/c2_occ_parse.py`.
