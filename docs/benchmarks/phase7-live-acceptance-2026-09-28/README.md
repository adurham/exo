# Phase 7 — Live DSpark acceptance on production (no relaunch)

Date: 2026-09-28
Hardware: 2x Mac Studio M4 Max (macstudio-m4-1 / macstudio-m4-2), 128 GB each
Model: DeepSeek-V4-Flash-Vision-Exp (production)
Status: **complete** — acceptance measured live; production never restarted

## Bottom line

The acceptance rate that prices V4.1's speculative decode was measured for the
first time — **without a relaunch**. The planned method (relaunch production
with `EXO_DSV4_MTP_LOG_INTERVAL` on) turned out to be unnecessary: the master
computes Prometheus counters from worker `GenerationStats` regardless of that
flag, so the number was already being recorded. The flag gates only the log
lines.

**Measured (production, gamma=3):**

| sample | condition | cycles | accepted | accepted/cycle | % of drafts |
|---|---|---:|---:|---:|---:|
| 1 | 320 tok, temp 0 | 143 | 177 | 1.238 | 41.3% |
| 2 | 2x300 tok, temp 0 | 260 | 338 | 1.300 | 43.3% |
| 3 | 3x600 tok, temp 0.7 | 885 | 915 | 1.034 | 34.5% |
| 4 | 3x600 tok, temp 0.7 (rerun) | 892 | 908 | 1.018 | 33.9% |
| all-time | since worker start | — | — | 1.058 | 35.3% |

Instrument identity check: cycles + accepted == completion tokens **exactly**
on the 1,800-token batch (892 + 908 = 1800). The counter pair is the decode
loop's own bookkeeping, not an estimate.

## Method

Counters: `exo_mtp_cycles_total`, `exo_mtp_accepted_drafts_total`
(`model_id="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"`) from the elected
master's `/metrics` (`http://100.91.246.26:52415/metrics`). Delta across
controlled completions driven through the OpenAI-compatible API at
`/v1/chat/completions`, then divided.

Scripts: `scripts/p10_live_accept.sh` (320 tok, temp 0),
`scripts/p10_greedy_accept.sh` (2x300 tok, temp 0),
`scripts/p10_batch_accept.sh` (3x600 tok, temp 0.7 — run twice),
`scripts/p10c_greedy_hist.sh`. Raw outputs in `raw/`.

**No relaunch:** an earlier relaunch attempt for this purpose aborted at
Thunderbolt discovery (stale node ssh aliases — fail-closed, before any
shutdown) with production verified untouched (pid elapsed time unchanged, API
serving throughout). By then the metric path was found, so the relaunch was
dropped entirely. Repair + evidence: `docs/launcher-address-drift-repair-2026-09-28.md`.

## What it means for V4.1

Sibling-model evidence — Vision-Exp is V4-Flash + vision, the same DSpark
design lineage but a different checkpoint and head — says: **DeepSeek DSpark
heads on this stack accept 34-43% of drafts** (greedy higher, stochastic
lower), not the 60% the optimistic row assumed.

Pricing the 40-layer port projection from phase 3 (gamma=5 table,
164.4 ms/round):

| drafts accepted | tokens/cycle | ms/token | tok/s |
|---|---:|---:|---:|
| 34% | 2.70 | 60.9 | 16.4 |
| 40% | 3.00 | 54.8 | 18.25 |
| 43% | 3.15 | 52.2 | 19.2 |
| 60% | 4.00 | 41.1 | 24.33 |

Break-even is 12.4% — the measured band clears it by ~3x, so speculation is
comfortably net-positive on either build path.

**Caveats.** (a) This is the sibling's number, not V4.1's own — V4.1's head
needs a full-body runtime to measure, and that is one of the two candidate
builds. (b) Production runs gamma=3; the port's DSpark block is 5.
Acceptance as a fraction-of-drafts is the portable quantity. (c) The table
above assumes the port's current (affine) kernels; the EXL3-kernel question is
phase 8.

## Artifacts

- `scripts/` — the four sample scripts
- `raw/samples_2026-09-28.txt` — greedy histogram + first sample
- `raw/batch_accept.txt` — 1,800-token batch, run twice
- `raw/metrics_family.txt` — full MTP metric family snapshot
