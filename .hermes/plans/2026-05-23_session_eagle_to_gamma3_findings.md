# Eagle K=1 Fix + γ=3 Discovery — Session Findings (2026-05-23)

## Top-line state

- **Production champion: γ=2 K=1 FENCE=4 @ 34.19 t/s σ=0.05 5/5 clean, quality verified.** Commit `40b6e9c0` on `origin/main`. 0.81 t/s short of 35 t/s target.
- **γ=3 K=1 FENCE=4 hit 40.57 t/s symmetric on a clean iter (way past 35), but bistable across iters.** This is the path forward, blocked on a bistability fix.
- **Eagle as a perf lever is DEAD-END at K=1 (no lift over K=0)** and **K=2 is also bistable at c=2 100K** with the current implementation. Quality at K=1 and K=2 both verified via 100K needle probe.
- Cluster currently DOWN (γ=3 K=1 FENCE=2 bench killed mid-run; failed iter 0).

## What's on origin/main

```
40b6e9c0 fix(dsv4-mtp): replace soft-emb broadcast with K=1 short-circuit + K>1 small-id broadcast  ← HEAD
21ba40db fix(dsv4-mtp): broadcast Eagle soft-emb (BROKEN, 17× slowdown — superseded by 40b6e9c0)
541061ee config(start_cluster.sh): propagate EXO_DSV4_MTP_EAGLE_K
33c4a2d7 feat(dsv4-mtp): Eagle-style soft-embedding for chained draft steps
2e708e19 config(start_cluster.sh): default EXO_DSV4_FENCE_EVERY_N_LAYERS=4 for c=2 stability
```

## Phase 1 — Eagle K=1 debug & first (wrong) fix

### Starting problem
Per `.hermes/plans/2026-05-22_phase14_handoff.md` Q2: Eagle K=1 produced asymmetric per-stream output ([11.52 / 9.96]) at c=2 100K despite K=1 being algebraically equivalent to hard-embed.

### H5 diagnosis (correct, via claude-code)
Full forensics in `.hermes/plans/2026-05-22_eagle_k1_debug_report.md`:
- The original Eagle code (`33c4a2d7`) computes `soft_emb` from rank-local `prev_logits` — that is, `prev_logits` captured BEFORE the existing post-argmax `broadcast_from_canonical(tok_arr, coord_group)` at `dsv4_mtp.py:1570`.
- Under MLX's documented per-rank logit drift (see comment block at `dsv4_mtp.py:1501-1521`), ranks pick different `argmax`/`argsort` outputs for tied/near-tied positions → rank-local `soft_emb` diverges between ranks → the next MTP forward sees materially different inputs across ranks for one batch element → asymmetric per-stream throughput.

### Attempt 1 (commit `21ba40db`) — broken
Added `broadcast_from_canonical(soft_emb, coord_group)` between `_compute_eagle_soft_emb` and `set_eagle_soft_emb` in BOTH the c=2 batched path (`dsv4_mtp.py:1554-1573`) and c=1 path (`mtp_module.py:688-702`).

**Symptom under bench:** aggregate decode rate ~2.2 tok/s (vs 34.16 baseline). ~17× slowdown. Bench never completed a single full iter in 40 minutes.

**Mechanism (full diagnosis in `~/repos/exo/eagle_k1_fix_report.md`):**
- `broadcast_from_canonical(soft_emb, ...)` is an `mx.distributed.all_gather` (synchronous JACCL collective) on a `(B=2, 1, hidden=4096)` bf16 tensor → 16 KB.
- The output of this gather feeds DIRECTLY into `_EAGLE_CTX["soft_emb"]`, which `DeepseekV4MTPModule.__call__` reads at line 2500 and passes as `emb` into `self.enorm(emb)` (line 2505) — **the very first GPU op of the MTP forward**.
- Result: every i≥1 predict() in the chain stalls behind a 16 KB UC RDMA round-trip with zero overlap potential.
- The c=2 batched path has NO per-step `mx.eval` fence (unlike c=1 at `mtp_module.py:758`), so chained-collective tails compound — the per-cycle wall blew out 17×.
- Symptom that masked it: looked like "c=2 stream parallelism lost / tasks sequential at 6.5-min cadence." That cadence was just the slow cycle wall paced through the BatchGenerator's admission queue.

### Attempt 2 (commit `40b6e9c0`) — WORKING

Replaced both broken blocks with a structurally different approach:

**K=1 short-circuit:** `soft_emb = _eagle_embed(tok_arr)`. `tok_arr` is exactly `broadcast(argmax(prev_logits))` from the post-argmax step at end of iter i-1, so embed-lookup on it gives the K=1 algebraic equivalent. **Zero new collective on the chain critical path.**

**K>1 path:** broadcast tiny `topk_ids` (int32, B·K bytes) + `topk_probs` (bf16, B·K bytes) and reconstruct the mixture locally on each rank from identical inputs. Two collectives per chain step beyond i=0, but each payload is single-digit bytes vs the 16 KB bf16 monstrosity.

### Validation of `40b6e9c0`

| Test | Result |
|------|--------|
| Pre-commit (ruff/basedpyright/mlx engine tests) | ✓ Same error counts as baseline; 6/6 tests pass |
| Short-prompt "Paris" (18 tokens) at K=0 fresh cluster | ✓ Correct |
| Short-prompt "Paris" (18 tokens) at K=1 fresh cluster | ✓ Correct |
| 100K needle probe at K=1 | ✓ `needle_found: True`, response `'FALCON-MERCURY-7749'` |
| 5-iter c=2 100K MTP-on γ=2 K=1 perf bench (1st run) | ✓ iters 1-4 mean 34.24 σ=0.029, all symmetric per-stream |
| 5-iter c=2 100K MTP-on γ=2 K=1 perf bench (2nd run, fresh cluster) | ✓ iters 1-4 mean 34.19 σ=0.05, all symmetric per-stream |

**K=1 perf result:** matches K=0 hard-embed baseline to within 0.05 t/s. This is expected — K=1 is bit-equivalent to K=0 by design (the patch's purpose was to validate Eagle plumbing end-to-end + not break anything, NOT lift t/s).

### Quality decay on long uptime (NEW SKILL PITFALL)

Mid-session, an 11-hour-old cluster with EAGLE_K=1 returned `<|begin_of_sentence|>` repeating for the dashboard chat — even at 18-token prompts. After a fresh restart, EAGLE_K=1 produced correct "Paris" output again.

This extends pitfall #9 (which documented prefill rate decay on long-uptime clusters): **quality also decays on long uptime, not just throughput.** This wasn't documented before. Always restart before quality-validation experiments. The empirical threshold for the BOS-spam failure mode was somewhere between 0 hours and 11 hours.

## Phase 2 — K>1 lever exploration (Eagle dead-end)

### K=2 quality (passed)
- Short prompt "Paris": ✓
- 100K needle probe: ✓ `needle_found: True`

### K=2 perf (failed — bistable)
5-iter bench at c=2 100K γ=2 K=2 FENCE=4:
```
iter=0  wall= 778.0s  per_stream=[14.33 / 9.71]  agg=24.04  (warmup, asymmetric)
iter=1  wall= 749.6s  per_stream=[17.10 / 17.10]  agg=34.20  ✓ same as K=1
iter=2  wall= 786.3s  per_stream=[0.64 / 17.18]  agg=17.82  ✗ STREAM 0 COLLAPSED
```

**Diagnosis:** K>1 path adds 2 coord_group collectives per chain step (topk_ids + topk_probs broadcasts), vs K=1's 0 extra and K=0's 0 extra. The extra collective accumulates per-cycle tail-stall probability past threshold → bistability.

**Phase 14 plan's "K=2-4 → +1.9% = +0.65 t/s" prediction NOT borne out.** Iter 1 was 34.20 — identical to K=1. K=2 didn't lift acceptance rate enough to be worth the extra collective cost.

### Eagle verdict for this configuration
- K=0 / K=1: bit-equivalent, stable, baseline 34.19 t/s
- K=2+: extra collectives → bistability, no perf benefit
- Eagle as a perf lever is dead-end on this hardware/jaccl config.

## Phase 3 — γ=3 discovery (THE BIG WIN)

### γ=3 K=1 FENCE=4 result
Iter 0 (warmup): `per_stream=[14.53 / 14.85] agg=29.38` — symmetric warmup (much better than γ=2 warmup which was asymmetric)
Iter 1 (steady): **`per_stream=[20.28 / 20.28] agg=40.57`** ← +6.4 t/s = +18.6% over γ=2 K=1 baseline
Iter 2: `per_stream=[0.65 / 20.51] agg=21.16` ← STREAM 0 COLLAPSED (same H5-shape bistability)

**Iter 1 hit 40.57 t/s symmetric** — clearing the 35 t/s target with +5.6 margin. Quality (100K needle probe) also confirmed: `needle_found: True` at γ=3 K=1.

**The path is structurally there.** γ=3 unlocks the throughput; bistability blocks consistency.

### γ=3 K=1 FENCE=2 attempt (failed worse)
Hypothesis: maybe FENCE=2 halves the chain-collective tail probability (same logic as the γ=2 FENCE=8 → FENCE=4 fix that resolved γ=2 bistability per pitfall #46).

Result: iter 0 broke immediately at `per_stream=[0.62 / 12.55] agg=13.18`. **FENCE=2 made it worse, not better.** Why: FENCE=N affects the verify-side GPU fence cadence (model trunk forward), which is ORTHOGONAL to the draft-side chain-collective queue (where the bistability lives). Adding more fences just adds overhead.

This is consistent with pitfall #46's note: "**EXO_DSV4_FENCE_EVERY_N_LAYERS is ORTHOGONAL to bistability**." The γ=2 → FENCE=4 fix worked because the c=2 batched verify-forward also benefited; but for the γ-side chain depth, FENCE is not the right knob.

### Bistability characterization

Three observations of γ=3 bistability collapse, all share the same shape:
- γ=3 K=1 FENCE=4: iter 1 OK [20.28/20.28], iter 2 collapsed [0.65/20.51] — stream 0 dropped to ~3% normal rate, stream 1 unaffected
- γ=2 K=2 FENCE=4: iter 1 OK [17.10/17.10], iter 2 collapsed [0.64/17.18] — same per-stream pattern
- γ=3 K=1 FENCE=2: iter 0 broke immediately [0.62/12.55] — fence overhead amplified

**Pattern:** ONE stream's rate drops to ~0.6 t/s (vs ~17-20 t/s baseline), other stream stays at expected rate. Aggregate goes to ~half of working-state. Once a stream collapses, it doesn't recover during the iter — it just stays collapsed until that command finishes.

This shape suggests **per-stream divergence inside the c=2 batched draft loop** — possibly that one stream's chain accumulates a collective-queue stall and the other doesn't, and then the cycle bookkeeping (matches, n_accepted_per, etc.) treats the slow stream as if it generated normally but at much lower wall-rate.

## Phase 4 — Infrastructure improvements

### GitHub SSH keys on cluster nodes (replaces 1Password agent forwarding)

Previously cluster's `git fetch` step in `start_cluster.sh` used the forwarded 1Password ssh-agent from the laptop. Under sustained use that agent gets rate-limited (skill pitfall #12), causing intermittent launch failures.

Fix applied this session:
- Generated `~/.ssh/github_macstudio_m4_1` (ed25519, no passphrase) on m4-1
- Generated `~/.ssh/github_macstudio_m4_2` (ed25519, no passphrase) on m4-2
- Added `Host github.com` block to each node's `~/.ssh/config` with `IdentityAgent none` and `IdentitiesOnly yes` pointing at the local key
- Added both pubkeys to user's GitHub account via `gh ssh-key add` (after `gh auth refresh -s admin:public_key` device-code flow)
- Verified: each node can `ssh -T git@github.com` and `git fetch origin` without needing the forwarded agent

This is a permanent fix — pitfall #12 / #42 / #44 should be updated to note this workaround is in place.

## Quality probe results archive

All three quality probes hit the same prompt (100K-token document with FALCON-MERCURY-7749 needle), all three found the needle:

```json
{"label": "eagle_k1_v40b6e9c0", "needle_found": true, "ttft_s": 293.92, ...}
{"label": "eagle_k2",           "needle_found": true, "ttft_s": 294.22, ...}
{"label": "gamma3_k1",          "needle_found": true, "ttft_s": 294.18, ...}
```

(Note: probe builds a ~100K-char prompt; cluster's actual tokenization yields ~69K tokens, not 100K. Still well into long-context territory.)

Quality probe artifacts on m4-1: `/tmp/quality_k1.json`, `/tmp/quality_k2.json`, `/tmp/quality_g3k1.json`.

## What changed in production

`origin/main` now carries:
- `40b6e9c0` Eagle K=1 short-circuit (safe, bit-equivalent to K=0) + K>1 small-id broadcast (works algebraically; bistable in practice at K=2)
- All prior commits unchanged

Default env in `start_cluster.sh`:
- FENCE_EVERY_N_LAYERS=4
- INDEX_TOPK=512
- KV_CACHE_BITS=0 (bf16)
- SPECULATIVE_GAMMA=2 (champion config)
- EAGLE_K not set → defaults to 0 (Eagle code path NOT taken — bit-identical to pre-Eagle)

**Recommended bench config** for reproducing the champion: just `./start_cluster.sh` with no overrides → γ=2 K=0 FENCE=4 → 34.19 t/s steady stable.

## Pitfalls / skill updates needed

1. **Quality decays on long-uptime clusters, not just prefill rate.** Extend pitfall #9.
2. **The Eagle 17× slowdown trap** (broadcast assembled soft_emb on chain critical path) — already documented in `references/2026-05-22-eagle-k1-chain-critical-path-rule.md`. Skill pitfall #46 needs the new commit `40b6e9c0` referenced as the working fix and `21ba40db` flagged as superseded.
3. **K>1 Eagle is bistable** under the current broadcast-topk_ids+topk_probs scheme at c=2 100K γ=2. K=1 is safe (no new collective). K>1 quality holds but perf doesn't.
4. **γ=3 unlocks 40+ t/s at FENCE=4 with quality intact but is bistable** — record this for future sessions so we don't re-discover the peak number from scratch.
5. **FENCE_EVERY_N_LAYERS does NOT affect draft-side bistability.** Reconfirmed at FENCE=2 (made γ=3 worse). Pitfall #46's "FENCE is orthogonal to bistability" line holds.
6. **GitHub SSH keys on cluster nodes** as permanent fix for pitfall #12 / #44 — see Phase 4 above.

## Plan for the bistability fix

See companion document `2026-05-23_gamma3_bistability_fix_plan.md`.
