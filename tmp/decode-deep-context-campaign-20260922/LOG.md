# Overnight session log — deep-context decode (2026-09-22/23)

## Authorization
User granted full free reign overnight incl. relaunches, then went to bed:
*"both lol ... you have full free reign overnight to do what is needed on the
cluster"*. Targets: (T1) 500K must not collapse; (T2) 250K >= 30 t/s.

## Phase 0 — state + canary: PASSED
- HEAD `1b4c126a3`, tree clean.
- Raw fp16 GEMM canary: **14.86 / 14.86 TFLOPS on both nodes** (healthy
  baseline is 14.8-15.2). No reboot needed; GPU power state is good.

## Phase 1 — measured findings

### Finding 1: tok/s is acceptance-driven, and acceptance is DEPTH-driven
Controlled ladder, identical probe + identical prompt, only depth varies
(this is a real control — `long_decode_probe.py` builds the same prompt at
every depth: same filler, same needle, same 900-word essay ask):

| depth | decode tps | acc/cycle | tok/cycle | ms/cycle | pageins | needle |
|---|---|---|---|---|---|---|
| 27,850 | **46.30** | **2.077** | 3.077 | 66.5 | 587 | ✓ |
| 127,929 | **29.72** | **0.814** | 1.818 | 61.2 | 400 | ✓ |

- **Acceptance falls 2.55x with depth** (2.077 -> 0.814).
- **Cycle cost is flat** (66.5 -> 61.2 ms).
- **Page-ins are LOW and do not rise** (587 -> 400).

So at <=128K the degradation is **draft-quality**, not compute and not memory.
This also supersedes my earlier "content, not depth" hypothesis
(`overnight-corrections-...md` Correction B): that was wrong, because the
probe's prompt is identical across rungs. The 46.3 t/s rung and the 33 t/s
rungs in the earlier ladder used *different scripts* — the earlier ladder's
rungs were `depth_sweep.py` (different prompt). Within ONE script, depth is
the only variable and acceptance clearly falls.

**Correction to Correction B: depth DOES degrade acceptance. Content was a
confound between scripts, not within this ladder.**

### Finding 2: the DSpark draft head's context window is 128 tokens
Source-read (`mlx-lm/models/deepseek_v4.py`):
- `DeepseekV4DSparkStage.attn` is a `DSparkLocalAttention` whose cache is
  `RotatingKVCache(max_size=config.sliding_window)` = **128 tokens**.
- Context conditioning arrives via `append_ctx()`, which pushes
  **target-model hiddens from layers `dspark_target_layer_ids` = [40, 41, 42]**
  (`h.mean(dim=2)`) into that 128-slot ring each round.
- `draft_block()` attends the draft block over `[ctx window; block]`.

So the drafter conditions on a **128-token local window** of the target's own
late-layer hidden states, regardless of total sequence depth.

**Candidate mechanism for Finding 1:** as depth grows, the target's
next-token distribution shifts (long-context attention), but the drafter's
view remains a fixed 128-token local slice — so its drafts track the target
less well: acceptance falls. This is an architecture/training property, not
a config knob.

**Alternative to rule out:** something in `append_ctx` / ring-trim /
snapshot-restore at depth feeding the drafter stale or wrong context.
Distinguishable by instrumenting the ctx ring contents at depth.

## Phase 2 — DSpark sizing: the planned lever is DEAD
Measured actual head component sizes (shape-level, from the served config):

| component | GB (full shape) | share |
|---|---|---|
| stage FFN (`switch_mlp`+`shared_experts`) x3 | 77.613 | **97.8%** |
| all non-FFN (attn x3, main_proj, markov, norms) | 1.770 | 2.2% |

Scaled to the quantized head (~10.876 GB on disk): FFN 10.633 GB,
non-FFN **0.243 GB**. Sharding all non-FFN recovers **~0.12 GB/rank**.
Against the 4.9 GB needed at 565K, that is ~2% of the fix.

**Phase 4 ("shard non-expert projections") CANCELLED — measured 50x too small.**
The head is already ~50% sharded (FFN split by the live
`EXO_DSV4_DSPARK_TP_SHARD=1`).

## Phase 3 — remaining levers, re-scoped

For **T1 (500K collapse)**, the refault mechanism stands (32,469 page-ins
measured at 42K depth under tighter memory), and it needs multi-GB footprint
reduction. Options after Correction A:
1. **`DSV4_WIRED_LIMIT_MB` raise** — launcher currently pins 115000 (lowered
   from 124000 on 2026-06-29 to prevent a Metal-allocator wedge). Raising it
   is a direct, reversible headroom test, BUT trades against a documented
   hard-wedge failure mode (prefill ~5x drop needing reboot). **Needs care:
   test at one node, small step, with the wedge signature monitored.**
2. **DSpark head quantization below mxfp4** — quality-gated (drives acceptance).
3. **Spill/page DSpark stages** — large change; not a night.

For **T2 (250K >= 30 t/s)**, Finding 1 says the limiter is draft acceptance at
depth. Levers:
- `EXO_DSV4_DSPARK_CONF_TAU` (0.5 default) — pruning threshold; tau sweep on
  a prior baseline showed 0.5 already at plateau (0.4=33.7, 0.5=36.1, 0.6=36.2),
  so likely little left.
- `EXO_DSV4_MTP_DRAFT_LMHEAD_BITS=4` — measured 2026-07-06 as +4% at 4K but
  **-8.7% at 586K** (sign flips with depth) -> would HURT here.
- The 128-token ctx window itself is a model/architecture constant.

## Status at checkpoint (23:22)
- Ladder still running: 250K and 500K rungs pending.
- No config changes made yet. Cluster untouched apart from measurement load.
- Committed: plan, corrections doc, this log.

## Next
1. Finish ladder (250K, 500K) -> get acceptance + pageins at the two target depths.
2. If 500K shows pageins dominating (as at 42K): test footprint/headroom levers.
3. If acceptance is the limiter at 500K too: T1 and T2 share a root cause and
   the honest answer may be "acceptance degrades with depth" — a property, not
   a bug — with the refault being the fixable part.
4. Leave cluster healthy; write final summary.
