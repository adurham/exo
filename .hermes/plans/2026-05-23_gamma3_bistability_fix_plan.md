# γ=3 c=2 Bistability Fix — Plan (2026-05-23)

## Goal

Stabilize γ=3 c=2 100K MTP-on K=1 so iter 1+ all match the symmetric 40.57 t/s observed in the 2026-05-23 session. Production target: 35 t/s sustained ≥5 iters σ<0.5 with quality (100K needle) intact.

## What we already know about the mechanism

### Confirmed (from this session)
- Iter 1 of γ=3 K=1 FENCE=4 hit **40.57 t/s symmetric [20.28/20.28]** — the structural capability is there.
- Iter 2 same config: **collapsed to [0.65 / 20.51] = 21.16 t/s** — stream 0 dropped to ~3% of normal rate, stream 1 stayed at expected ~20 t/s.
- The shape is per-stream divergence inside the c=2 batched chain, NOT a cluster-wide stall.
- FENCE=2 made it WORSE, not better → bistability is NOT a verify-side fence issue.

### Confirmed (from prior sessions, skill pitfall #46 + the May 17 RESOLUTION reference)
- γ≥2 c=2 bistability mechanism: **chain-collective queue depth accumulates per-cycle peer-CQE arrival tail probability**.
- The c=1 path's fix (commit `ce61e46b`) was a per-step `mx.eval(tok_arr)` at `mtp_module.py:786` that forces the GPU/comm command buffer to drain between chain steps. This is what makes c=1 γ=2 stable.
- The γ=2 c=2 fix (commit `2e708e19`) sidestepped this by halving chain depth via FENCE=8 → FENCE=4. Worked because γ=2's 2-step chain at FENCE=4 keeps the queue shallow enough that peer-CQE tail probability stays below threshold.

### The new (γ=3) observation that points at the fix
γ=3 means **3 chain steps**, vs γ=2's 2. That's 50% more chained-collective queue depth per cycle. FENCE=4 was tuned to keep γ=2's queue below threshold; γ=3 blows past it.

### The CRITICAL code-path asymmetry

**c=1 path** (`mtp_module.py::draft_tokens`, lines 682-786): per-step `mx.eval(tok_arr)` fence at line 786:
```python
# Per-step fence — see comment above the loop.
if i + 1 < gamma:
    mx.eval(tok_arr)
```
This DOES drain the chain-collective queue between every chain step. c=1 γ=2 is stable BECAUSE of this fence.

**c=2 batched path** (`dsv4_mtp.py::_draft_tokens_batched`, lines 1486-1665): **NO equivalent per-step fence.** The loop at lines 1549-1665+ runs all γ chained `predict()` calls back-to-back with no GPU/comm sync between them. At γ=2 this was bistable, and the FENCE=4 default flip fixed it by reducing chain depth. At γ=3 the chain is one step deeper and bistability returns.

### So the fix is

**Port the per-step `mx.eval(tok_arr)` fence from `mtp_module.py:786` into the c=2 batched draft loop at `dsv4_mtp.py:1665ish` (end of for-loop body).**

This is the c=2 analog of the c=1 fix that resolved γ=2 c=1 bistability. It was simply never added to the c=2 path because the FENCE=4 default flip masked the problem at γ=2. With γ=3 we've outrun the mask.

## Step-by-step plan

### Step 0 — Confirm baseline (~12 min)
1. Restart cluster with default env (γ=2 K=0 FENCE=4 — the safe production champion):
   ```bash
   cd ~/repos/exo && ./start_cluster.sh
   ```
2. Verify cluster reaches READY 2/2 nodes.
3. Run 5-iter c=2 bench to confirm 34.16-ish baseline holds post-this-session.
4. Quality smoke: `"What is the capital of France?"` → expect "Paris".

This step protects against any cluster-state drift from today's many restarts.

### Step 1 — Read the EXACT code path before editing (~5 min)
Re-read these to make sure the fence placement is correct:
- `src/exo/worker/engines/mlx/speculative/mtp_module.py:682-790` (c=1 chain with the working fence) — the canonical implementation
- `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:1486-1665` (c=2 batched chain — needs the fence) — the patch target
- The comments at `mtp_module.py:669-681` ("Break the chained-collective dependency between successive MTP draft steps...") — the rationale doc

Specifically confirm:
- `tok_arr` exists at the right scope inside the c=2 batched loop's body — it does (assigned at lines 1627 / 1649 in both temp=0 and temp>0 branches).
- The fence `if i + 1 < gamma: mx.eval(tok_arr)` only runs when there IS a next iter, matching c=1.
- The fence does NOT need to be inside the eagle install/clear block — it's loop-level.

### Step 2 — Patch (~5 min)
Add to `dsv4_mtp.py::_draft_tokens_batched` immediately AFTER the temp=0/temp>0 if-else block in the for loop body (around current line 1665, after `draft_probs.append(...)` in both branches). Should look like:

```python
for i in range(gamma):
    # ... existing Eagle install + predict + clear ...
    # ... existing temp=0 / temp>0 argmax/categorical + tok_arr broadcast ...

    # NEW: per-step fence to drain the chained-collective queue between
    # MTP draft steps. Without this, γ≥2 chained predicts queue lazy
    # all_sums in the GPU/comm command buffer, accumulating peer-CQE tail
    # probability. Identical pattern to mtp_module.py::draft_tokens:786
    # which fixed γ=2 c=1 bistability. The c=2 batched path was missing
    # this fence — FENCE=4 default masked the issue at γ=2, but γ=3's
    # 50% deeper chain blew past the mask. Cost: one extra GPU sync per
    # chain step (microseconds). Benefit: eliminates the iter-N+1 stream
    # collapse that produced [0.65/20.51] iter-2 readings at γ=3.
    if i + 1 < gamma:
        mx.eval(tok_arr)
```

Make sure the fence is INSIDE the `for i in range(gamma)` loop but OUTSIDE the temp=0/temp>0 if-else. Both branches set `tok_arr` so it's defined either way.

### Step 3 — Validate locally (~3 min)
- `git diff` — confirm the diff is just the new fence block + comment.
- `uv run ruff check src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` — 0 new errors above baseline.
- `uv run basedpyright src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` — 0 new errors above baseline (`mx.eval(...)` is already used elsewhere in the file).
- `uv run pytest src/exo/worker/engines/mlx/tests -x -q` — should be 6/6 like the prior fixes.

### Step 4 — Commit + push (~2 min)
Commit message draft:
```
fix(dsv4-mtp): per-step mx.eval fence in c=2 batched draft chain

Mirrors the c=1 path's per-step fence at mtp_module.py:786 that resolved
γ=2 c=1 bistability. The c=2 batched draft loop at
dsv4_mtp.py::_draft_tokens_batched was missing this fence because the
γ=2 c=2 bistability was sidestepped by FENCE=4 default reducing chain
depth (commit 2e708e19) rather than by fencing each chain step. At γ=3
the chain is 50% deeper and FENCE=4 is no longer sufficient — peer-CQE
arrival tail probability accumulates past threshold mid-cycle and one
stream collapses.

Empirical proof: γ=3 K=1 c=2 100K iter 1 = [20.28/20.28] agg=40.57 ✓,
iter 2 = [0.65/20.51] agg=21.16 ✗ — one-stream collapse, same shape
as the γ=2 c=2 bistability the FENCE=4 default fixed.

Cost: one extra GPU sync per chain step (~µs). Benefit: enables γ=3 c=2
at 40+ t/s sustained, clearing the 35 t/s target with margin.

Forensics: .hermes/plans/2026-05-23_session_eagle_to_gamma3_findings.md
           .hermes/plans/2026-05-23_gamma3_bistability_fix_plan.md
```
Push to `origin/main` (cluster will hard-reset to this commit on next launch).

### Step 5 — Validate on cluster (~45 min total)

5a. Restart cluster with the same γ=3 K=1 FENCE=4 config that exposed the bug (~5 min):
```bash
cd ~/repos/exo && EXO_DSV4_MTP_EAGLE_K=1 EXO_DSV4_MTP=1 EXO_SPECULATIVE_GAMMA=3 ./start_cluster.sh
```

5b. **Quality probe first** (~6 min) — confirm γ=3 K=1 still produces correct 100K-context output with the new fence:
```bash
ssh m4-1 'cd ~/repos/exo && .venv/bin/python -u bench/quality_probe_dsv4.py --base-url http://192.168.86.201:52415 --target-tokens 100000 --max-tokens 256 --label gamma3_k1_fenced --out /tmp/quality_g3k1_fenced.json'
```
Acceptance: `needle_found: True`.

5c. **5-iter perf bench** (~30 min) — the real test:
```bash
ssh m4-1 "screen -dmS g3k1_fenced zsh -l -c 'cd ~/repos/exo && .venv/bin/python -u /tmp/bench_c2_temp0_5iter_phase14a.py > /tmp/g3k1_fenced_5iter.log 2>&1'"
```
Acceptance (per skill pitfall #41 γ=2 protocol, applied here to γ=3):
- ≥5 iters all >35 t/s (since target is 35)
- σ < 0.5
- All iters per-stream symmetric (or within 1 t/s — iter-0 warmup gets some asymmetry leeway)
- NO bistability collapse (no iter where one stream drops below 10 t/s while the other is normal)

5d. If 5/5 clean → run a 10-iter bench at γ=3 K=1 FENCE=4 to upgrade the acceptance signal (the γ=2 protocol asked for 10/10 historically and the new champion claim should match). Same script with `n_iters=10` (or just run two 5-iter benches back-to-back on the same cluster).

### Step 6 — Update production defaults (if validated)
If γ=3 stably clears 35 t/s:
1. Flip `EXO_SPECULATIVE_GAMMA` default in `start_cluster.sh` from 2 → 3.
2. Document the new champion config: `γ=3 K=0 (or K=1, identical) FENCE=4 TOPK=512`.
3. Commit + push.
4. Update skill pitfall #46 with the new champion number and config.

If NOT validated (per-step fence insufficient at γ=3), fall back to Step 7.

### Step 7 — Plan B if Step 5 fails

If the per-step fence doesn't fully stabilize γ=3:

**Hypothesis 7a:** The fence drains the comm queue but the mx.eval is itself a sync that adds latency, and γ=3 adds enough latency that prefill+decode timing edges into a regime where some other tail (e.g. RDMA driver coalescing, GPU clock governor) kicks in. Diagnostic: enable `JACCL_POLL_INSTRUMENT=1 JACCL_POLL_INSTRUMENT_THRESHOLD_US=5000` and compare iter-1 vs iter-2 traces.

**Hypothesis 7b:** The fence needs to be on more than just `tok_arr` — the actual graph dependency that's queueing might also include `prev_logits` (used by Eagle path) or `h` (the hidden state). Try `mx.eval(tok_arr, h, prev_logits)` if simple `mx.eval(tok_arr)` doesn't suffice.

**Hypothesis 7c:** The bistability is c=2-specific because of per-stream cache state divergence inside `PerStreamBatchRotatingKVCache`. The c=1 fence works on c=1 (single stream); the c=2 analog might need a per-stream fence rather than a global `mx.eval`. Diagnostic: read `mtp_module.py:541` (per-stream mask creation in c=2) and check whether per-stream cache state could drift independently.

**Hypothesis 7d (longer):** Generalize the fence pattern beyond just MTP draft — there may be other chained-collective points in the c=2 verify-forward or accept path that also benefit from per-step drain. Lower-priority; only pursue if 7a-c don't pan out.

## Risks

1. **Per-step fence adds latency.** The c=1 fence cost was "microseconds" per chain step at γ=2. At γ=3 c=2 with B=2, we add 2 extra fence points per cycle (chain steps 0→1, 1→2). If per-fence cost on this hardware is more than ~100µs, the +6.4 t/s gain from γ=3 could erode. Empirical: c=1 γ=2 with the fence runs at 30.5 t/s vs ~29 t/s without (pitfall #46 history), so fence cost is <5% wall. At γ=3 we expect similar overhead — should still leave us well above 35.

2. **`mx.eval(tok_arr)` only drains the GPU compute stream up through tok_arr's dependency.** If the bistability mechanism involves comm collectives that tok_arr does NOT transitively depend on, the fence won't fully drain them. The c=1 result suggests this isn't the case (the fence works for c=1 γ=2), but c=2's batched-prefill / per-stream cache mechanics may introduce additional collectives the fence misses.

3. **Bistability might not be 100% chain-depth driven.** There may be other contributing factors (thermal state, cluster uptime per pitfall #9, jaccl QP state). Bench discipline: always restart cluster fresh before each γ=3 bench iteration.

4. **Quality changes are POSSIBLE if the fence alters lazy-graph evaluation timing in a way that affects RNG state.** Greedy (temp=0) decode should be deterministic regardless; temp>0 path could see subtle changes. Acceptance: quality probe must still pass at γ=3 post-fence — we already confirmed γ=3 quality passes at FENCE=4, the fence change shouldn't alter that.

## Don't re-attempt

From pitfall #46 and this session's findings:
- mlx eager-commit (`4d21baa2`) — broke γ=1
- Event::signal mlx commits — 2× SIGKILL
- Mach RT class — falsified
- `mx.async_eval` swap of the per-step fence (`823f9fb9`) — broke
- Broadcasting the assembled Eagle soft_emb (`21ba40db`) — 17× slowdown
- FENCE=2 to fix γ=3 bistability — made it WORSE (this session)
- `EXO_RUNNER_QOS=user_interactive` alone — Python QoS pin doesn't reach C++ stream worker threads

## Expected outcome

If the per-step `mx.eval(tok_arr)` fence in the c=2 batched chain works as predicted (matching c=1's working fence), γ=3 K=1 c=2 100K should produce:
- 5/5 iters at 38-42 t/s (steady state)
- σ < 0.3
- All iters per-stream symmetric
- 100K needle probe still passes

That puts us **at 38-42 t/s with quality preserved — well past the 35 t/s target.**

If this works, the champion config becomes:
```
EXO_SPECULATIVE_GAMMA=3
EXO_DSV4_FENCE_EVERY_N_LAYERS=4
EXO_DSV4_INDEX_TOPK=512
EXO_DSV4_MTP=1
EXO_KV_CACHE_BITS=0
EXO_DSV4_MTP_EAGLE_K=0  (or 1, identical)
```

## Estimated wall-clock to complete

- Steps 0-4 (code change): ~30 min
- Step 5 (validation): ~60 min (cluster restart + quality probe + 5-iter bench + 10-iter bench)
- **Total: ~90 min from start to validated champion update.**

If Step 5 reveals Hypothesis 7a/b/c is needed, add 60-90 min for diagnostics + retry.
