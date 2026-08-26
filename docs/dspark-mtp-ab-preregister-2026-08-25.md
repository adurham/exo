# DSpark Native + MTP Enablement — Pre-Registered A/B & Static Audit (2026-08-25)

**Status:** PHASE 0 STATIC AUDIT COMPLETE. Phase 1 pre-registration written.
**Blocked on:** live cluster (currently DOWN — unrelated debugger investigating
"no nodes available" placement failure for DeepSeek-V4-Flash-0731). No relaunch,
no flag flip, no baseline capture until cluster is back AND user explicitly
approves. This doc is pre-registered BEFORE any data collection per house rule.

**Audit by:** GLM-5.2 (Ollama Cloud), acting as PM.
**Repo HEAD at audit:** `61efad499802bc766eeb1015558ad92537f8ae91` (main, clean).
**Plans cross-referenced:** `/tmp/glm_plan.md` (GLM-5.2, authoritative),
`/tmp/dspark_plan.md` (DSv4-Pro), `/tmp/kimi_reasoning.md` (Kimi-K3).

---

## ⚠️ CRITICAL FINDING — DSpark head is REPLICATED, not SHARDED

The single most important static-audit result, and the biggest deviation from
the plan's assumptions:

**`DeepseekV4ShardingStrategy.shard_model` does NOT shard `model.model.dspark`.**

The DSpark draft head attaches as `model.model.dspark`
(`utils_mlx.py:866`, `:1016`) **before** tensor sharding runs, and the overlay
comment at `utils_mlx.py:370-372` claims it is "Attached BEFORE tensor sharding
so its DeepseekV4MoE ffns shard exactly like the native mtp head's." **This
claim is FALSE.** The sharding strategy was never updated to recurse into
`dspark`:

- `src/exo/worker/engines/mlx/auto_parallel.py:1031` —
  `class DeepseekV4ShardingStrategy(TensorParallelShardingStrategy)`.
- `shard_model` at lines `1049-1180` iterates ONLY:
  - `layers = model.model.layers` (line 1054, loop 1062-1133) — sets
    `layer.ffn.sharding_group = self.group` (1087) + shards `shared_experts`
    and `switch_mlp` in place.
  - `mtp_blocks = list(getattr(model.model, "mtp", []) or [])` (line 1059,
    loop 1153-1178) — sets `mtp.ffn.sharding_group = self.group` (1156) +
    shards each MTP block's ffn identically.
- **Definitive grep: zero `dspark` references in `auto_parallel.py`'s sharding
  code** (the only `dspark` mentions are comments about a *PP* tap-capture fix
  at lines 622-633, unrelated to TP sharding).
- **No `super().shard_model()` call** in the method (base class
  `TensorParallelShardingStrategy.shard_model` is `@abstractmethod`, line 869-873,
  no generic walk).
- **No generic `model.modules()`/`children()` walk** in `shard_model`.

The dspark head's stages DO contain `DeepseekV4MoE` ffns that WOULD shard if
reached — `mlx-lm/mlx_lm/models/deepseek_v4.py:6305` `class
DeepseekV4DSparkStage`, line 6320 `self.ffn = DeepseekV4MoE(config,
body_layer_idx)` (the same `DeepseekV4MoE` class used by main layers and MTP
heads, which shards via `sum_gradients` inside `__call__` once
`.sharding_group` is set, per the strategy docstring at auto_parallel.py:1044-1046).
But because the sharding loop never sets `dspark.stages[j].ffn.sharding_group`,
the cross-rank reduction never fires.

**Consequence:** the DSpark head runs **REPLICATED full-size on EVERY rank**
(~10 GB/node, ~20 GB total across the 2-node cluster), NOT sharded (~5 GB/node).
The plan's memory budget — which assumed the head shards to ~5 GB/node — is
**optimistic by a factor of 2× on the per-node cost**. This finding was
confirmed via a second-opinion consult (the reviewer agreed the conclusion
holds after verifying there is no `super().shard_model()` call and no generic
walk).

**Implication for the enablement decision:** the memory feasibility question
becomes *harder*, not easier. With 128 GB nodes documented at ~125 GB
co-resident weights, adding a *replicated* ~10 GB head per node leaves ~−7 GB
headroom per node (i.e. it does NOT fit without memory recovery). The two-stage
staging idea (Kimi-K3: relaunch #1 head-load-only with SPECULATIVE=0 to
validate memory, relaunch #2 full spec) is now **more strongly indicated** —
but ONLY after the memory audit (Phase 0.1) confirms ≥12 GB free per node after
recovery actions, AND a code fix to shard the dspark head is evaluated. See
"Proposed flag set" below.

---

## Phase 0 Static Audit — Checklist with file:line evidence

### 0.1 DSpark load path (`utils_mlx.py:358-470`)

- **Native head path reads `mtp.0/1/2.*` from the checkpoint** — confirmed.
  `_overlay_dsv4_dspark_native` (`utils_mlx.py:879`) reads the
  currently-loading checkpoint's own `mtp.*` safetensors shards via
  `Model.sanitize()`'s generic transform pipeline with `n_mtp_override=3`
  (docstring, `utils_mlx.py:894-908`). The selection is gated on
  `EXO_DSV4_DSPARK_NATIVE=1` (`utils_mlx.py:441`).
- **Head attaches BEFORE tensor sharding** — confirmed.
  `inner.dspark = mod` at `utils_mlx.py:866` (local overlay) and `:1016`
  (native overlay); `shard_and_load` dispatches to `tensor_auto_parallel` at
  `utils_mlx.py:500`, which runs AFTER the overlay block (lines 359-470).
- **`set_dspark_taps` armed** at `utils_mlx.py:867` and `:1017` with
  `mod.target_layer_ids` (i.e. `[40,41,42]` per config).

### 0.2 Sharding of the dspark submodule — **BUG / NOT SHARDED**

See the CRITICAL FINDING above. Verdict: **REPLICATED (~10 GB/node)**, not
sharded. Evidence:
- `auto_parallel.py:1049-1180` `DeepseekV4ShardingStrategy.shard_model` — no
  `dspark` reference, no generic walk, no `super()` call.
- `deepseek_v4.py:6320` `DeepseekV4DSparkStage.ffn = DeepseekV4MoE(...)` —
  would shard if reached, but isn't reached.
- Confirmed via consult (no missed generic path).

This is the highest-risk code path flagged in `/tmp/glm_plan.md` Phase 1.1
step 2. **A sharding code fix is required before a sharded trial can run.**
Until then, any trial runs the head replicated at ~10 GB/node.

### 0.3 MTP head double-load question — NO double-load

- `mlx-lm/mlx_lm/models/deepseek_v4.py:6500-6507`: the native MTP head is
  `self.mtp = [DeepseekV4MTPModule(config, i) for i in
  range(config.num_nextn_predict_layers)]`, gated on
  `config.num_nextn_predict_layers > 0` AND `EXO_DSV4_MTP=1`.
- The DSpark head is a SEPARATE structure: `model.model.dspark` (a
  `DeepseekV4DSparkModule`, `deepseek_v4.py:6340`), attached by the overlay,
  NOT the same object as `self.mtp`.
- **In native mode (EXO_DSV4_MTP_DEDICATED unset), setting both
  EXO_DSV4_MTP=1 and EXO_DSV4_DSPARK=1 loads BOTH `self.mtp` (the native MTP
  head) AND `model.model.dspark` (the DSpark head).** This is TWO heads.
  However, the consumer path (`dsv4_mtp.py`) uses the DSpark head for drafting
  when DSPARK is attached, and the native MTP head is the verifier's own
  structure — they are not redundant weights. The "double-load = ~20 GB →
  guaranteed OOM" concern in the plan is **partially valid**: you do load two
  distinct modules, but they serve different roles (the MTP head is the
  checkpoint's own next-token-predict structure that the main model forward
  may reference; the DSpark head is the draft generator). The memory cost is
  the SUM of both heads' resident sizes. The native MTP head's size depends
  on `num_nextn_predict_layers=1` (one module) — small relative to the DSpark
  3-stage head. **This needs empirical measurement at load time** (blocked on
  cluster).

### 0.4 `EXO_DSV4_MTP_DEDICATED` default — UNSET (correct)

- `utils_mlx.py:358-362`: the dedicated-MTP overlay is gated on
  `EXO_DSV4_MTP=1 AND EXO_DSV4_MTP_DEDICATED=1`. Default is `"0"` (falsy) →
  the native MTP head stays the default. **Confirmed: unset by default.**
  We want native MTP head, so we leave `EXO_DSV4_MTP_DEDICATED` unset.

### 0.5 TP consumer double-gate — CONFIRMED (both flags required)

- `utils_mlx.py:421`: `_tp_consumer = _spec_on and
  os.environ.get("EXO_DSV4_MTP", "0") == "1"` — the TP DSpark draft branch
  is reachable ONLY when `EXO_SPECULATIVE=1` AND `EXO_DSV4_MTP=1` (both, not
  either). This matches the plan's requirement.
- `dsv4_mtp.py:370-371`: the DSpark TP cycle runs under
  `EXO_SPECULATIVE=1 EXO_DSV4_MTP=1 EXO_DSV4_DSPARK=1` (all three; MTP=1 is a
  hard prerequisite even for DSpark).
- **No PP fallthrough in tensor-sharding mode** — `_pp_consumer`
  (`utils_mlx.py:422-426`) additionally requires
  `isinstance(shard_metadata, PipelineShardMetadata)`, which is False under
  Tensor sharding. So in TP mode, the PP decode loop is unreachable. No
  accidental fallthrough.

### 0.6 Head-load gate — single env var (confirmed)

- `utils_mlx.py:418`: `_dspark_env_on = os.environ.get("EXO_DSV4_DSPARK",
  "0") == "1"` — single master gate.
- The gate then checks consumer reachability (`_dspark_usable`, line 427 =
  `_tp_consumer or _pp_consumer or _dspark_force`). If
  `EXO_DSV4_DSPARK=1` but no consumer is reachable, the head load is SKIPPED
  with a warning (~10 GB/node reclaimed) — lines 428-438.
- `EXO_DSV4_DSPARK_FORCE_LOAD=1` (`utils_mlx.py:420`) overrides the gate (for
  measuring the head's own load/memory cost deliberately). This is a
  *measurement* override, NOT a second required key.
- **Confirmed: single env var (`EXO_DSV4_DSPARK=1`) opens the gate, with a
  consumer-reachability check — NOT a two-key system.** The plan's concern
  about a "second gate" is resolved: there is no `EXO_DSV4_DSPARK_LOAD`
  second key.

### 0.7 Checkpoint `mtp.0/1/2.*` + config in HF cache — **BLOCKER: weights absent**

- `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/`
  contains ONLY `config.json` (1888 B), `tokenizer_config.json` (801 B),
  `tokenizer.json` (6.1 MB). Total cache size: **6.1 MB**.
- **NO `.safetensors` weight files** — not `mtp.0.*`, not `mtp.1.*`, not
  `mtp.2.*`, not even `model.safetensors`. The 155 GB model weights are NOT
  downloaded locally on this node. (This is consistent with the cluster
  state: model unloaded, placement failing.)
- `config.json` DOES declare the DSpark params correctly:
  - `dspark_block_size = 5` ✅
  - `dspark_target_layer_ids = [40, 41, 42]` ✅
  - `dspark_markov_rank = 256` ✅
  - `num_nextn_predict_layers = 1` ✅
- **BLOCKER:** before `EXO_DSV4_DSPARK_NATIVE=1` can be flipped, the native
  head tensors (`mtp.0/1/2.*` safetensors) must be present on BOTH nodes.
  Current state: absent on this node; status on node 1 unknown (cluster
  down). This must be verified once the cluster is back. The plan's Phase
  1.5 checklist item "Native head safetensors exist in local model cache"
  is **NOT met**.

### 0.8 Degeneration test harness — INTACT

- `bench/spec_degen_capture.py` — imports cleanly, `--help` runs. Contains
  the STABLE trigger set: 6 system+user two-message prompts
  (`sys_primary_colors`, `sys_capital_france`, `sys_count_to_five`,
  `sys_long_essay`, `sys_long_steps`, `sys_long_list`) + 1 single-user
  negative control (`control_user_only`). The system+user two-message
  format that triggered the Eagle K=8 collapse IS present (lines 37-89).
- `bench/spec_degen_diff.py` — imports cleanly, `--help` runs. Takes
  `--trace` (dsv4_spec_trace jsonl), optional `--ground-truth`
  (spec_degen_capture JSON), `--max-period`, `--min-repeats`.
- `bench/mtp_eagle_microbench.py` and `bench/mtp_longctx_probe.py` — both
  import cleanly, `--help` runs.
- **Regression gate is ready.** The trigger set is embedded directly in
  `spec_degen_capture.py` (no external dependency). Quality-check FIRST
  before looking at t/s, per house rule.

---

## Phase 1 — Pre-Registered A/B Design

### Hypothesis

- **H1:** DSpark native + MTP with `EXO_SPECULATIVE_GAMMA=2` increases decode
  throughput (tokens/sec) by **≥15% over the spec-off baseline at 100K
  context**, with draft acceptance rate **≥60%**.
- **H1b:** At 352.6K context, decode t/s improvement **≥10%**.
- **H0 (null):** No throughput improvement, OR acceptance rate <40%, OR any
  quality regression → revert.

### Acceptance threshold (strict, from reconciled plan deltas)

- ≥15% decode t/s improvement at 100K ctx (p50, n≥3 runs per condition).
- ≥10% decode t/s improvement at 352.6K ctx.
- Draft acceptance rate ≥60%.
- TTFT no worse than +10% vs baseline.
- Zero quality regression on `spec_degen_capture.py` trigger set (no BOS-spam,
  no short loops on ANY system+user prompt).
- Peak RSS < 126 GB per node (2 GB safety margin under 128 GB), 0 swap.
- 50 consecutive decode steps without desync or OOM.

### Baseline (Treatment A — current spec-off, NO relaunch needed)

```
EXO_SPECULATIVE=0
EXO_DSV4_MTP=0
EXO_DSV4_DSPARK=0   (head-load gate skips; ~10 GB/node reclaimed)
EXO_DSV4_HC_COLLAPSE_KERNEL=1   (keep existing prefill opt)
# EXO_DSV4_POOL_GROW_STEP=256 is OPT-IN and UNSET — first trial isolates
# DSpark+MTP from it (reconciled plan delta (a)). Do NOT enable in trial #1.
```
Metric: decode t/s at 100K and 352.6K ctx, single-stream, greedy (temp=0).
TTFT. Peak RSS per node. Output text for quality diff.

### Treatment (Treatment B — requires relaunch, APPROVAL GATE)

See "Proposed flag set" below. Two-stage staging (Kimi-K3 idea) is recommended
given the REPLICATED-head finding.

### Fixed prompt set (committed before any run)

From `bench/spec_degen_capture.py` PROMPTS list (lines 37-89), run all 7:
- 3 short system+user: `sys_primary_colors`, `sys_capital_france`,
  `sys_count_to_five`
- 3 long system+user (the degeneration "starts correct then collapses
  mid-stream" class): `sys_long_essay`, `sys_long_steps`, `sys_long_list`
- 1 single-user negative control: `control_user_only`
Plus, for the batched-verify landmine (skill
`exo-speculative-decode-correctness` RESOLVED 2026-08-01/02): one
self-verification math prompt (e.g. `math_digit_sum` from `bench/hard_eval.py`
if present) run spec-ON vs spec-OFF as a control.

### Measurement protocol

- Same prompt set for A and B (fixed, committed above).
- Context pre-filled to 100K (and 352.6K for the second depth) with identical
  prefix.
- 5 prompts × 256 output tokens each, per treatment, per depth.
- t/s = wall-clock `time.monotonic` first-to-last token.
- Acceptance rate from exo debug logs (grep `accepted|rejected|draft`).
- Quality: raw output text saved to files, diffed with `spec_degen_diff.py`.
- **Quality check FIRST, before t/s** (house rule — prevents motivated
  reasoning overriding the quality gate).

### Failure criteria (ANY → immediate revert)

- OOM kill on either node.
- Swap pressure > 0 bytes (`vm_stat`).
- BOS-spam or loop output on ANY trigger prompt.
- Desync between nodes (shard mismatch, tensor shape error).
- Throughput DECREASE vs baseline.
- Head-load failure (dspark weights not found, fallback to overlay, crash).
- TTFT worse than +10% vs baseline.

---

## Proposed flag set for first live trial

Given the REPLICATED-head finding (~10 GB/node, not ~5 GB/node), the memory
budget is tighter than the plan assumed. **Two-stage staging is
recommended** (Kimi-K3 idea, reconciled plan delta (c)):

### Stage 1 — Head-load validation (SPECULATIVE=0, zero decode risk)

```
EXO_DSV4_DSPARK=1
EXO_DSV4_DSPARK_NATIVE=1
EXO_DSV4_MTP=1
EXO_SPECULATIVE=0        # head loads but no speculative drafting
# Unset:
#   EXO_DSV4_DSPARK_DIR       (native, not local overlay)
#   EXO_DSV4_MTP_DEDICATED    (native MTP head)
#   EXO_PP_DRAFT_MODEL        (TP, not PP)
#   EXO_DSV4_POOL_GROW_STEP   (isolate from it — delta (a))
# Keep:
EXO_DSV4_HC_COLLAPSE_KERNEL=1
```
**Gate:** head loads without OOM, `mx.metal.get_active_memory()` increases by
~10 GB (replicated, per node), config values match
(`dspark_block_size=5`, `target_layer_ids=[40,41,42]`, `markov_rank=256`),
decode still works (non-speculative), `spec_degen_capture.py` shows no diff
vs baseline. **If OOM here → STOP, the head does not fit replicated.**

### Stage 2 — Full speculative (only if Stage 1 passes)

```
EXO_SPECULATIVE=1
EXO_DSV4_MTP=1
EXO_DSV4_DSPARK=1
EXO_DSV4_DSPARK_NATIVE=1
EXO_DSV4_HC_COLLAPSE_KERNEL=1
EXO_SPECULATIVE_GAMMA=2     # conservative — Eagle K=8 champion degenerated
EXO_SPECULATIVE_TEMP=0.0    # eliminate draft randomness
EXO_SPECULATIVE_ALPHA=1.0
# Unset (same as Stage 1):
#   EXO_DSV4_DSPARK_DIR, EXO_DSV4_MTP_DEDICATED, EXO_PP_DRAFT_MODEL,
#   EXO_DSV4_POOL_GROW_STEP
```
**Rationale for gamma=2:** the Eagle K=8 champion degenerated on system+user
prompts. Start conservative. gamma=2 = at most 2 draft tokens/step, minimal
runaway-loop risk, easy to validate quality. Can increase to 3-4 later if
quality holds.

**NOTE on the REPLICATED head:** because the head is NOT sharded (Phase 0.2
finding), Stage 1's memory check is the gate that determines whether a
full-spec trial is even feasible without a code fix. If Stage 1 OOMs, the
path forward is EITHER (i) a code fix to shard `model.model.dspark` in
`DeepseekV4ShardingStrategy.shard_model`, OR (ii) memory recovery (Phase 0.2
options A-D in `/tmp/glm_plan.md`) to free ≥12 GB/node. A code fix is the
cleaner long-term answer (it halves the per-node head cost) but requires its
own audit + A/B. **Flag-flip approval should NOT be requested until Stage 1
passes.**

---

## Rollback runbook (pre-written, copy-paste ready)

If ANY failure criterion fires during Stage 1 or Stage 2, revert to the
spec-off baseline. **All relaunches require explicit user approval (session
runs through the cluster; a relaunch kills the conversation).**

### Revert flags to spec-off baseline (Treatment A)

On BOTH nodes, set the env to exactly:
```bash
export EXO_SPECULATIVE=0
export EXO_DSV4_MTP=0
export EXO_DSV4_DSPARK=0
export EXO_DSV4_DSPARK_NATIVE=0
export EXO_DSV4_HC_COLLAPSE_KERNEL=1
unset EXO_DSV4_DSPARK_DIR EXO_DSV4_MTP_DEDICATED EXO_PP_DRAFT_MODEL
unset EXO_SPECULATIVE_GAMMA EXO_SPECULATIVE_TEMP EXO_SPECULATIVE_ALPHA
unset EXO_DSV4_POOL_GROW_STEP   # do not enable in trial #1
# Verify:
env | grep -E 'EXO_SPECULATIVE|EXO_DSV4' | sort
```
Expected output:
```
EXO_DSV4_DSPARK=0
EXO_DSV4_DSPARK_NATIVE=0
EXO_DSV4_HC_COLLAPSE_KERNEL=1
EXO_DSV4_MTP=0
EXO_SPECULATIVE=0
```

### Relaunch (APPROVAL GATE — user must explicitly approve)

```bash
# Coordinate a maintenance window; warn that the session will drop.
# start_cluster.sh is NOT edited by this audit (house rule). The user
# handles the relaunch command separately.
# After relaunch, verify head is NOT loaded (gate skipped):
ssh <node> "grep 'DSpark head load SKIPPED' ~/.exo/exo_log/exo.log | tail -1"
# Expected: "DSpark head load SKIPPED (~10 GB/node reclaimed): ..."
# Then verify decode works:
# (run spec_degen_capture.py against the baseline)
```

### If the head must be force-unloaded (emergency)

`EXO_DSV4_DSPARK=0` is sufficient — the head-load gate skips the overlay
entirely when the flag is off (or when no consumer is reachable). No separate
unload command exists; a relaunch with `EXO_DSV4_DSPARK=0` is the revert.

---

## Approval gate log

| Gate | Status | Approver | When | Understanding of session-loss risk |
|---|---|---|---|---|
| Phase 0 static audit | COMPLETE (this doc) | GLM-5.2 (PM) | 2026-08-25 | N/A — no relaunch |
| Stage 1 head-load relaunch | PENDING | user | — | relaunch kills session |
| Stage 2 full-spec relaunch | PENDING | user | — | relaunch kills session |

**No relaunch is requested by this audit.** All flag flips are pre-registered
only. The cluster is currently DOWN (unrelated debugger investigating); even
if it were up, the user must explicitly approve each relaunch.

---

## What is blocked on the live cluster

- **Phase 0.1 memory audit** — measure peak RSS per node at steady-state
  decode with loaded context. Cannot run (cluster down, model unloaded).
- **Phase 0.7 checkpoint weight verification on node 1** — confirm
  `mtp.0/1/2.*` safetensors present on BOTH nodes (node 0 confirmed ABSENT).
- **Phase 1 baseline capture (Treatment A)** — 5 prompts × 256 tokens at 100K
  and 352.6K ctx, spec-off. No relaunch needed but needs cluster up.
- **Phase 2.2 single-node dry run** — load just the dspark head, print
  `mx.metal.get_active_memory()` delta. Needs cluster.
- **Stage 1 / Stage 2 relaunch + validation** — needs cluster + user approval.

## What is verified (no cluster needed) — summary

| Item | Verdict | Evidence |
|---|---|---|
| Native head path reads `mtp.0/1/2.*` | ✅ | `utils_mlx.py:879,441` |
| Head attaches BEFORE sharding | ✅ | `utils_mlx.py:866,1016`, dispatch at `:500` |
| DSpark head sharded across TP | ❌ **REPLICATED (~10GB/node)** | `auto_parallel.py:1049-1180` no dspark ref; `deepseek_v4.py:6320` has the MoE; consult-confirmed |
| MTP head double-load | partial — two distinct modules, sum costs | `deepseek_v4.py:6500-6507` (mtp) vs `utils_mlx.py:866` (dspark) |
| `EXO_DSV4_MTP_DEDICATED` default unset | ✅ | `utils_mlx.py:358-362` |
| TP consumer double-gate (SPEC + MTP) | ✅ | `utils_mlx.py:421`, `dsv4_mtp.py:370-371` |
| No PP fallthrough in TP mode | ✅ | `utils_mlx.py:422-426` requires PipelineShardMetadata |
| Head-load gate single var | ✅ | `utils_mlx.py:418` (FORCE_LOAD is measurement override, not 2nd key) |
| `spec_degen_capture.py` intact + triggers | ✅ | `bench/spec_degen_capture.py:37-89`, `--help` runs |
| Config dspark params correct | ✅ | `config.json`: block_size=5, target=[40,41,42], markov=256 |
| Native head weights on disk | ❌ **ABSENT (blocker)** | HF cache 6.1MB, no safetensors |

---

## Next approval-gate status

**No approval requested.** This audit completes all static-work items. The
next step that needs the user is: (1) cluster back up (debugger's job), (2)
verify `mtp.0/1/2.*` weights present on both nodes, (3) Phase 0.1 memory
audit, (4) if memory passes, explicit approval for Stage 1 head-load
relaunch. The REPLICATED-head finding may prompt the user to instead
prioritize a sharding code fix before any flag flip — that decision is the
user's, not this audit's.