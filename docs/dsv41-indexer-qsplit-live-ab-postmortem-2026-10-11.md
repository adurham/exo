# DSv4.1 indexer-qsplit (`DSV41_INDEXER_QSPLIT`) — LIVE A/B post-mortem — 2026-10-11

**Verdict: ABORT — token-identity divergence.** The treatment was taken live,
A/B'd, and **reverted** to the known-good baseline
(`deploy/next22-rankfix @ ade481853`, `EXO_DSV41_CKPT=1`).

Artifact under test: `deploy/next23-qsplit @ 9cfc7935` (= `deploy/next22-rankfix`
+ 4 commits; mlx-lm `perf/dsv41-indexer-qsplit @ e0340628`). The split bands the
replicated DSv4.1 attention-indexer's **query** rows across the 2 TP ranks and
reconstructs the int32 `cidx` selection with a zero-padded `all_sum`; the design
claims **byte-exact** output (per-row top-k is row-independent). Flag default OFF.
Run log: `~/.hermes/cache/scratch/dsv41_live3/runlog.md`.

## 1. What passed

- **Construct suites** (own worktree @ 9cfc7935 / e0340628): exo dsv41 **367
  passed**, mlx-lm qsplit **21 passed** (PM-reproduced).
- **Mechanical gate:** both nodes `9cfc7935d` / mlx-lm `e034062`; runner env has
  `DSV41_INDEXER_QSPLIT=1` **and** `EXO_DSV41_CKPT=1`; READY 2/2; DSv4 pinned (jit=false).
- **Engagement (real, not a silent no-op):** each node's log shows exactly one
  `[DSV41] indexer qsplit: DSV41_INDEXER_QSPLIT=1 (effective at import)` +
  `engine built … prefill chunk 2048`; the band engages for w=2 2048-row chunks.
- **Prefill A/B (cold feeds, flag OFF → ON):**

  | depth | OFF rows/s | ON rows/s | Δ rows/s | modeled (bench actual) |
  |---|---|---|---|---|
  | ~50K  | 275.2 | 298.6 | **+8.5 %**  | +10.4 % |
  | ~105K | 261.6 | 292.5 | **+11.8 %** | +14.5 % |

  A real, positive prefill win at both depths, in the modeled band (~2–3 pp below
  the offline bench actual — live e2e amortization).
- **Battery Phase A** (repo copy, quick depth 120000, `--skip-park`): **CLEAN**
  (needles 6/6, tools 10/10, prose 0 DIRTY / 0 REVIEW).

## 2. Why it aborted — token identity

Pre-registered gate: identical greedy prompts (temp 0) must give identical tokens;
**any divergence = ABORT**. 5 fixed byte-identical prompts, 300 greedy tokens each:

- 4/5 prompts: `top1_agreement = 1.0` (300/300).
- `identity_3` (2126 tok): **diverges deterministically at captured index 4**
  (`top1 = 0.04`). Both continuations are plausible:
  - OFF: `Field cryptography is the practice of using the physical world itself, …`
  - ON : `Field cryptography is the branch of practical secrecy concerned with how messages …`

Divergence is real, not a measurement artifact:
- same bytes (prompt_tokens 2126 both; same file sha);
- **both sides cold prefill** (`park.restore cold reason=rank-empty` + distinct
  `prefill controls … (rows=N, base=2048)`; no warm `session reuse`);
- treatment deterministic (two independent flag-ON runs agree 1.0); baseline
  stable (two flag-ON re-runs both diverge at index 4; control re-runs of id0/id2
  re-agree 1.0).

## 3. Root-cause correlation (actionable)

Only the one prompt whose **second prefill chunk** engages the band diverges:

| prompt | tokens | chunks | chunk-2 rows | band on chunk-2 | result |
|---|---|---|---|---|---|
| id0 | 1869 | 1 | —      | n/a   | agree |
| id1 | 1947 | 1 | —      | n/a   | agree |
| id2 | 2051 | 2 | 3      | odd → OFF | agree |
| id3 | 2126 | 2 | 78     | **even, 39>16 → ON** | **DIVERGE @4** |
| id4 | 2177 | 2 | 129    | odd → OFF | agree |

⇒ The exactness failure is on a **banded NON-FIRST prefill chunk** — a
position/causal-mask offset read from the band start instead of `chunk_base +
band_offset`, or a shape-dependent score GEMM (M=39/1024 vs 78/2048) whose bf16
rounding flips a margin-0 tie at the k=512 top-k boundary. Single-chunk unit tests
miss both. This is the same class as the campaign's documented indexer bf16
top-k tie-break (2026-10-09, "output-affecting change"). Note the single-chunk
prompts all agree, so the failure is specifically the chunk-2 band path.

## 4. Revert

`git checkout --detach ade481853` + `git submodule update --init mlx-lm` (→
689e4eafd); launcher `EXO_TARGET_BRANCH=deploy/next22-rankfix EXO_DSV41_CKPT=1`
→ EXIT 0; both nodes `ade481853` / `689e4ea`, `EXO_DSV41_CKPT=1`, no
`DSV41_INDEXER_QSPLIT`, READY 2/2, jit=false. **Boot count: 2** (1 treatment + 1
reserve/revert). No node-side edits; launcher-only relaunches; no force-push.

## 5. Note / deviation

The round's launch instruction passed `EXO_DSV41_INDEXER_QSPLIT=1`, but the
launcher's forwarder (`start_cluster.sh:2723`) reads **`DSV41_INDEXER_QSPLIT`**
(no `EXO_` prefix — unlike the `EXO_DSV41_CKPT` family). The runner is started
over `ssh … zsh -l -c` and inherits only the launcher's `EXO_ENV`, so the
`EXO_`-prefixed name is a **silent no-op** (the split would never engage). Ran
with the correct `DSV41_INDEXER_QSPLIT=1`; engagement independently confirmed in
the node logs. Fix the brief/launcher naming before any re-run.

## 6. To re-open

Fix the band's non-first-chunk handling (offset/mask frame and/or band-row-count-
stable score path), then add unit coverage at the real chunk sizes with a nonzero
chunk base (reconstruct the **score** tensor, not just the selection) before any
redeploy.
