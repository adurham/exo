# DSv4.1 — small-n wide tile (R-B): LIVE ROUND result — gate FAIL → rolled back; then **BANKED** on owner decision (2026-10-13)

**Verdict (round 1):** the fix is real, exact and large (4.4–5.1× on deep decode rounds), but it missed its
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

**Round-1 end state (superseded — see the banking update below):** production = `deploy/next23b-qsplit @ 60008939` + mlx-lm `6a5c78f4`, flags
`EXO_DSV41_CKPT=1` + `DSV41_INDEXER_QSPLIT=1`; runners m4-1 90300 / m4-2 11226; API 200.
`deploy/next24-routing` remains staged and redeployable as-is. Side effect: a local
`deploy/next24-routing` ref in both node checkouts (harmless).

**Owner options:** (1) redeploy as-is with a revised bar — bank the 4.4–5.1× (exact; one relaunch);
(2) investigate the residual first — offline-first, then likely one live round.

Artifacts: `scratch/idxprof8/` (REPORT.md, raw/, out/, pre|post|rb/, battery partial).

---

## UPDATE — banking redeploy (2026-10-13 ~04:30Z): **BANKED — the fix is LIVE**

Owner decision: "bank now" (accept the revised bar). Redeploy of the same artifact
(`deploy/next24-routing @ 9fe1d593`, mlx-lm `a0573e3`); revised acceptance = abort only on identity
mismatch / battery dirty / errors / >250 ms steady rounds.

- **Redeploy:** 1 boot, ~3.6 min downtime, CLEAN_EXIT both nodes, READY 2/2.
- **Mechanical: PASS** on both nodes (HEAD, submodule, venv md5 `bfcf067b…`, env, helper regenerated).
- **Identity: PASS** — 5/5 byte-identical vs the `idxprof8/pre` captures; deep p300k and p245k also
  identical to the idxprof6 captures.
- **Quality battery quick (120K, `--skip-park`): CLEAN** — needles 6/6, tools 10/10, prose 0 dirty /
  0 review of 20 (= qsplitB reference; no regression). The gate that never ran in round 1 is closed.
- **p300k deep decode: PASS** — steady median **148 ms** (mean 146.8, p10–p90 146–150) vs 628 ms
  pre-fix → **4.24×**; **14.7 tok/s**; 128-token decode 8.6 s (was 36.2 s); first round 229 ms.
  245K control 98 ms. GPU ~70 % during decode (the known residual, unchanged).
- p500k not re-run (190 ms measured on this same build in round 1); the ≥350K needle variant not run.
- **Orchestrator verification (independent):** both nodes `9fe1d593` / `a0573e3f`; runner env
  `DSV41_INDEXER_QSPLIT=1` + `EXO_DSV41_CKPT=1`, `DSV41_INDEXER_SMALLN_WIDE_TILE` unset (default ON);
  runners m4-1 97629 / m4-2 18287; API smoke OK.
- **End state: LIVE.** Rollback if ever needed: `DSV41_INDEXER_SMALLN_WIDE_TILE=0` (relaunch) or
  idxprof8 §5 (switch back to `deploy/next23b-qsplit` + submodule update + relaunch).
- Artifacts: round 1 `scratch/idxprof8/`; banking round `scratch/idxprof9/`.
