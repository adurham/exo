# DSv4.1 — small-n wide tile (R-B): LIVE ROUND result — gate FAIL → ROLLED BACK (2026-10-13)

**Verdict:** the fix is real, exact and large (4.4–5.1× on deep decode rounds), but it missed its
pre-registered decode gate (≤ 1.2× the 245K control). Per protocol the treatment was rolled back with
the reserve boot; production is back on `60008939` / `6a5c78f4` — verified independently: HEAD, mlx-lm
sha, `indexer.py` md5 `2879c87e…`, runner env, baseline identity 5/5 byte-identical, API 200.
**Boots: 2 of 2 used** (treatment + rollback).

**Artifact:** exo `deploy/next24-routing @ 9fe1d593` = live tip + mlx-lm pin `a0573e3` (the fix) +
buildnote + kill-switch forward line. Mechanical gate **PASSED** on both nodes (HEAD 9fe1d593,
submodule a0573e3, venv md5 `bfcf067b…`, env `EXO_DSV41_CKPT=1` + `DSV41_INDEXER_QSPLIT=1`, READY 2/2,
helper regenerated). Downtime ~4.5 min.

**Token identity: PASS 7/7** — 5 rebuilt identity prompts (300 tok; originals no longer exist, rebuilt
deterministically, shas in `idxprof8/prompts/identity_meta.json`) + p300k/p500k (128 tok) vs the idxprof6
captures; all cold, temp 0, byte-identical.

**Deep decode (the gate): FAIL**

| probe | before ms/round | after ms/round | speed-up | × 245K control |
|---|---:|---:|---:|---:|
| 245K control | 103.5 | 103.0 | 1.00× | 1.00 |
| 300K | 628 | **143** | **4.4×** | **1.39 ✘** (bar ≤1.2) |
| 500K | 978 | **190** | **5.1×** | **1.84 ✘** |

- 128-token decode wall: **36.2 → 8.2 s @300K**; **62.8 → 15.4 s @500K**; prefill unchanged
  (267.7→273.6 / 234.2→237.0 rows/s).
- Runner CPU recovered to ~1.5–1.7 cores (matches control), but **GPU utilisation stays 65–72 % vs 93 %
  at the control** → a second, depth-proportional cost remains above the 254.2K crossing:
  **~+40 ms/round @300K, ~+87 ms/round @500K (~0.2 ms per 1K tokens)**, unexplained by the offline model
  (the harness attributed only ~4–7 ms/round to the tile width). Candidates: the remaining 2–4 fences
  × 8 calls/round, the 268 MB/call transient, other depth-proportional host work.
- The slow first round after a long prefill persists (4.16 s @500K) — separate known defect.

**Quality battery:** NOT RUN (killed at the gate-3 fail while its 350K build was still prefilling; no
verdict exists — the treatment was never shipped, so none is owed).

**Hygiene:** clean on both boots (0 ERROR/CRITICAL/SIGKILL/OOM; the Tracebacks are WARNING-level model-card
and file-fetch messages, present before).

**End state:** production = `deploy/next23b-qsplit @ 60008939` + mlx-lm `6a5c78f4`, flags
`EXO_DSV41_CKPT=1` + `DSV41_INDEXER_QSPLIT=1`; runners m4-1 90300 / m4-2 11226; API 200.
`deploy/next24-routing` remains staged and redeployable as-is. Side effect: a local
`deploy/next24-routing` ref in both node checkouts (harmless).

**Owner options:** (1) redeploy as-is with a revised bar — bank the 4.4–5.1× (exact; one relaunch);
(2) investigate the residual first — offline-first, then likely one live round.

Artifacts: `scratch/idxprof8/` (REPORT.md, raw/, out/, pre|post|rb/, battery partial).
