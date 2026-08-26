# DSpark Native + MTP Enablement — Pre-Registered A/B & Static Audit (2026-08-25)

**Status:** PHASE 0 STATIC AUDIT COMPLETE. Post-reboot TB link + checkpoint
verification COMPLETE (2026-08-25 21:39 CDT). Stage-1 launch command written
below — PENDING USER APPROVAL GATE. No relaunch, no flag flip, no baseline
capture until the user explicitly approves and launches. This doc is
pre-registered BEFORE any data collection per house rule.

**Audit by:** GLM-5.2 (Ollama Cloud), acting as PM.
**Repo HEAD at audit:** `61efad499802bc766eeb1015558ad92537f8ae91` (Phase 0
commit `07906d8b0`, main, clean). Post-reboot verification appended 2026-08-25
21:40+ CDT on the same HEAD.
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

### 0.4 `EXO_DSV4_MTP_DEDICATED` default — CORRECTED: defaults to 1 on the launch path (NOT unset)

- `utils_mlx.py:358-362`: the dedicated-MTP overlay is gated on
  `EXO_DSV4_MTP=1 AND EXO_DSV4_MTP_DEDICATED==1`. The Python reads env
  default `"0"` (`utils_mlx.py:361`).
- **BUT `start_cluster.sh:468` has `: "${EXO_DSV4_MTP_DEDICATED:=1}"`
  inside the `if [ "${DSV4_ENABLED}" = "1" ]` block (line 354).** With
  `DSV4_ENABLED=1` (the production default), if you launch with
  `EXO_DSV4_MTP=1` but leave `EXO_DSV4_MTP_DEDICATED` unset, the shell
  script defaults it to `1` and forwards it to the runner via the EXO_ENV
  allowlist line at `start_cluster.sh:1921`
  (`[ -n "${EXO_DSV4_MTP_DEDICATED:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_DEDICATED=$EXO_DSV4_MTP_DEDICATED"`).
  The live process therefore sees `MTP_DEDICATED=1` and runs
  `_overlay_dsv4_dedicated_mtp(model, model_path)` — overlaying the
  external `mlx-community/DeepSeek-V4-Flash-MTP-bf16` head onto
  `model.model.mtp[0]` BEFORE the DSpark native overlay runs
  (`utils_mlx.py:364`), which conflicts with `EXO_DSV4_DSPARK_NATIVE=1`'s
  intent of using the checkpoint's OWN `mtp.0/1/2.*` weights.
- **CORRECTION to the Phase 0 audit:** the earlier finding (0.4) "unset
  by default → native MTP head stays the default" was reading the Python
  env default, not the launch-path default. **Stage 1 MUST explicitly set
  `EXO_DSV4_MTP_DEDICATED=0`** to keep the native checkpoint-bundled MTP
  head that `EXO_DSV4_DSPARK_NATIVE=1` expects. This is a launch-critical
  correction.

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

### 0.7 Checkpoint `mtp.0/1/2.*` + config — RESOLVED: weights PRESENT on both nodes (2026-08-25 post-reboot verification)

- **Earlier "BLOCKER: weights absent" finding was based on inspecting the
  WRONG machine's HuggingFace cache** (`~/.cache/huggingface/hub/...` on a
  non-cluster host). The exo cluster loads from `~/.exo/models/`, not the
  HF hub cache. Post-reboot verification on BOTH cluster nodes confirms
  the native head weights ARE present.
- **Node1** (`adams-mac-studio-m4-1.local`):
  `~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731/` = **155 GB**,
  48 safetensors shards (`model-00001-of-00048` … `model-00048-of-00048`),
  `model.safetensors.index.json` (hash `810e55576e2d29570d6b9a0ffaa8202f7cec1ea2`).
- **Node2** (`adams-mac-studio-m4-2.local`):
  `~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731/` = **165 GB**,
  48 safetensors shards (identical filenames), `model.safetensors.index.json`
  (identical hash `810e55576e2d29570d6b9a0ffaa8202f7cec1ea2`).
- **The 155G vs 165G size difference is APFS sparse/clone accounting, NOT
  a content difference**: both nodes have identical file counts (57 entries),
  identical 48 shard names, and a byte-identical index.json (same SHA-1).
  Node2's larger `du` reflects fewer sparse-block references in its APFS
  container, not extra weight tensors.
- **`mtp.0/1/2.*` weights are packed INSIDE the 48 `model-*` shards, not
  as separate files** — confirmed via `model.safetensors.index.json`'s
  `weight_map` on node2: **4705 `mtp.*` keys**, spanning `mtp.0` through
  `mtp.2`, resident in shards `model-00046-of-00048` through
  `model-00048-of-00048` (e.g. `mtp.0.attn.wq_a.weight ->
  model-00046-of-00048.safetensors`). The Phase 0 audit's `ls | grep mtp`
  returned 0 because it looked for standalone `mtp.*` files; the weights
  live inside the unified shards.
- `config.json` declares the DSpark params correctly:
  - `dspark_block_size = 5` ✅
  - `dspark_target_layer_ids = [40, 41, 42]` ✅
  - `dspark_markov_rank = 256` ✅
  - `num_nextn_predict_layers = 1` ✅
- **The local overlay head is also present on both nodes:**
  `~/.exo/models/local--DeepSeek-V4-Flash-DSpark-MTP/` (the separately-
  converted head used when `EXO_DSV4_DSPARK_NATIVE` is unset and
  `EXO_DSV4_DSPARK_DIR` points at it). Stage 1 uses NATIVE, not this.
- **BLOCKER RESOLVED.** `EXO_DSV4_DSPARK_NATIVE=1` can be flipped; the
  native head tensors are present on both nodes. The plan's Phase 1.5
  checklist item "Native head safetensors exist in local model cache" IS
  met.

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

### Stage 1 — Head-load validation (SPECULATIVE=0 + FORCE_LOAD, zero decode risk)

**CORRECTION to the original Stage-1 plan (verified 2026-08-25 post-reboot
via consult + code re-read):** with `EXO_SPECULATIVE=0` alone, the DSpark
head-load gate at `utils_mlx.py:427`
(`_dspark_usable = _tp_consumer or _pp_consumer or _dspark_force`) is **False**
— `_tp_consumer` needs `SPECULATIVE=1`, `_pp_consumer` needs Pipeline
sharding + `EXO_PP_DRAFT_MODEL`. The head would be **SKIPPED** with the
"~10 GB/node reclaimed" warning, not loaded. The original Stage-1 text "head
loads but no speculative drafting" was **wrong for this code path** — it
described the skip path, not the load path.

To genuinely validate the native head LOAD + memory cost without engaging
speculative decode, Stage 1 must add `EXO_DSV4_DSPARK_FORCE_LOAD=1`
(`utils_mlx.py:420`), which makes `_dspark_force=True` → `_dspark_usable=True`
→ head attaches via `_overlay_dsv4_dspark_native` (line 444). The draft
cycle itself is gated separately at `batch_generate.py:813`
(`use_speculative = os.environ.get("EXO_SPECULATIVE", "0") == "1"`) and is
only constructed inside `if use_speculative:` (line 822), so `SPECULATIVE=0`
guarantees no drafting even with the head loaded. This is the load-only
override the consult confirmed.

```
EXO_DSV4_DSPARK=1
EXO_DSV4_DSPARK_NATIVE=1
EXO_DSV4_DSPARK_FORCE_LOAD=1   # NEW: bypass consumer-reachability gate so head LOADS under SPECULATIVE=0
EXO_DSV4_MTP=1
EXO_DSV4_MTP_DEDICATED=0       # NEW: explicitly 0 — start_cluster.sh:468 defaults this to 1 when DSV4_ENABLED=1
EXO_SPECULATIVE=0              # head loads but draft cycle NOT constructed (batch_generate.py:813,822)
EXO_DSV4_HC_COLLAPSE_KERNEL=1  # keep existing prefill opt
# Explicitly unset/empty (do NOT export):
#   EXO_DSV4_DSPARK_DIR       (native, not local overlay)
#   EXO_PP_DRAFT_MODEL        (TP, not PP; also avoids tokenizer-mismatch gibberish)
#   EXO_DSV4_POOL_GROW_STEP   (isolate from it — reconciled plan delta (a))
```

#### EXACT approved Stage-1 launch command (user runs this in tmux)

Mirrors the established background-launch pattern from
`docs/hc-expand-kernel-ab-2026-08-24.md:538` (`tmux new-session -d -s <name>
'... ./start_cluster.sh 2>&1 | tee /tmp/<name>.log'`), with the Stage-1 env
vars set inline before `./start_cluster.sh` (the script's `${VAR:-default}`
and `[ -n "${VAR:-}" ]` allowlist lines pick them up):

```bash
tmux new-session -d -s dspark_s1 \
  'cd ~/repos/exo && \
   EXO_DSV4_DSPARK=1 \
   EXO_DSV4_DSPARK_NATIVE=1 \
   EXO_DSV4_DSPARK_FORCE_LOAD=1 \
   EXO_DSV4_MTP=1 \
   EXO_DSV4_MTP_DEDICATED=0 \
   EXO_SPECULATIVE=0 \
   EXO_DSV4_HC_COLLAPSE_KERNEL=1 \
   ./start_cluster.sh 2>&1 | tee /tmp/dspark_s1.log'
```

**Env-forwarding verification (why each inline var reaches the runner):**
- `EXO_DSV4_DSPARK` → `start_cluster.sh:1784` allowlist → `EXO_ENV`.
- `EXO_DSV4_DSPARK_NATIVE` → `start_cluster.sh:1819` allowlist → `EXO_ENV`.
- `EXO_DSV4_DSPARK_FORCE_LOAD` → `start_cluster.sh:1787` allowlist → `EXO_ENV`.
- `EXO_DSV4_MTP` → `start_cluster.sh:1728` allowlist → `EXO_ENV`.
- `EXO_DSV4_MTP_DEDICATED` → `start_cluster.sh:1921` allowlist → `EXO_ENV`.
- `EXO_SPECULATIVE` → `start_cluster.sh:1631` (unconditional) → `EXO_ENV`.
- `EXO_DSV4_HC_COLLAPSE_KERNEL` → `start_cluster.sh:2202` allowlist → `EXO_ENV`.
- `EXO_DSV4_DSPARK_DIR`, `EXO_PP_DRAFT_MODEL`, `EXO_DSV4_POOL_GROW_STEP` →
  not exported, so their `[ -n "${VAR:-}" ]` allowlist lines are skipped
  (the runner sees them unset = the code's `os.environ.get(..., "0")` default).

**Stage-1 gate (what "passes" means):**
1. Head loads without OOM on either node.
2. `mx.metal.get_active_memory()` increases by ~10 GB/node (replicated head,
   per the Phase 0.2 finding) — measured post-load.
3. Config values match: `dspark_block_size=5`, `target_layer_ids=[40,41,42]`,
   `markov_rank=256`.
4. Decode still works (non-speculative) — a short smoke completion returns
   coherent text.
5. `spec_degen_capture.py` shows no diff vs baseline (quality gate FIRST).
6. Peak RSS < 126 GB/node, 0 swap.

**If OOM here → STOP, the head does not fit replicated.** Path forward is
either (i) a code fix to shard `model.model.dspark` in
`DeepseekV4ShardingStrategy.shard_model` (halves per-node cost), or (ii)
memory recovery to free ≥12 GB/node. A code fix is the cleaner long-term
answer but requires its own audit + A/B. **Flag-flip approval for Stage 2
should NOT be requested until Stage 1 passes.**

### Stage 2 — Full speculative (only if Stage 1 passes)

```
EXO_SPECULATIVE=1
EXO_DSV4_MTP=1
EXO_DSV4_MTP_DEDICATED=0       # explicit 0 (same correction as Stage 1)
EXO_DSV4_DSPARK=1
EXO_DSV4_DSPARK_NATIVE=1
EXO_DSV4_HC_COLLAPSE_KERNEL=1
EXO_SPECULATIVE_GAMMA=2     # conservative — Eagle K=8 champion degenerated
EXO_SPECULATIVE_TEMP=0.0    # eliminate draft randomness
EXO_SPECULATIVE_ALPHA=1.0
# Explicitly unset/empty (do NOT export):
#   EXO_DSV4_DSPARK_DIR       (native, not local overlay)
#   EXO_DSV4_DSPARK_FORCE_LOAD  (NOT needed — SPECULATIVE=1 makes _tp_consumer=True)
#   EXO_PP_DRAFT_MODEL        (TP, not PP)
#   EXO_DSV4_POOL_GROW_STEP   (isolate from it — delta (a))
```

#### EXACT approved Stage-2 launch command (user runs this in tmux, only after Stage 1 passes)

```bash
tmux new-session -d -s dspark_s2 \
  'cd ~/repos/exo && \
   EXO_SPECULATIVE=1 \
   EXO_DSV4_MTP=1 \
   EXO_DSV4_MTP_DEDICATED=0 \
   EXO_DSV4_DSPARK=1 \
   EXO_DSV4_DSPARK_NATIVE=1 \
   EXO_DSV4_HC_COLLAPSE_KERNEL=1 \
   EXO_SPECULATIVE_GAMMA=2 \
   EXO_SPECULATIVE_TEMP=0.0 \
   EXO_SPECULATIVE_ALPHA=1.0 \
   ./start_cluster.sh 2>&1 | tee /tmp/dspark_s2.log'
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
| Post-reboot TB link + checkpoint verification | COMPLETE (this doc, §"Post-reboot verification") | GLM-5.2 (PM) | 2026-08-25 21:39 CDT | N/A — read-only SSH, no relaunch |
| Stage 1 head-load relaunch | PENDING — exact command written (§Stage 1) | user | — | relaunch kills session |
| Stage 2 full-spec relaunch | PENDING — exact command written (§Stage 2) | user | — | relaunch kills session |

**No relaunch is requested by this audit.** All flag flips are pre-registered
only. The cluster is back up post-reboot (TB link healthy, exo NOT yet
started); the user must explicitly approve and launch each stage themselves
in tmux (their established pattern — a separate dispatch will handle
post-launch verification once they confirm).

---

## Post-reboot verification (2026-08-25 21:39 CDT, read-only)

Context: the TB/RDMA wedge was diagnosed earlier (AppleThunderboltRDMA
teardown at 20:23:22 after a runner was SIGKILLed; TB link dropped; all ports
"No device connected"). The user rebooted both Studios. This section records
the read-only verification that the link is fully healthy on BOTH nodes before
any launch is approved.

### TB/RDMA link — HEALTHY on both nodes

**Node1** (`adams-mac-studio-m4-1.local`), uptime `4 mins` at check:
- `ifconfig`: `inet 192.168.200.1 netmask 0xffffff00 broadcast 192.168.200.255`
  on the TB bridge interface, `status: active`, `media: autoselect <full-duplex>`.
- `system_profiler SPThunderboltDataType -detailLevel basic`: two ports
  reporting `Status: Device connected` / `Link Status: 0x2` (peer = Mac16,9,
  Device ID 0xA, Vendor ID 0x0A27 = Apple). No "No device connected" entries.
- `pgrep -fl "python -m exo"`: `NO_EXO_PROC` (exo not started, as expected).
- `ping -c 5 192.168.200.2` (node1 → node2 across TB): `5 packets transmitted,
  5 packets received, 0.0% packet loss`, `round-trip min/avg/max/stddev =
  0.667/0.849/0.996/0.113 ms`.

**Node2** (`adams-mac-studio-m4-2.local`), uptime `4 mins` at check:
- `ifconfig`: `inet 192.168.200.2 netmask 0xffffff00 broadcast 192.168.200.255`
  on the TB bridge interface, `status: active`, `media: autoselect <full-duplex>`.
- `system_profiler SPThunderboltDataType -detailLevel basic`: two ports
  reporting `Status: Device connected` / `Link Status: 0x2` (peer = Mac16,9,
  same Device/Vendor ID). No "No device connected" entries.
- `pgrep -fl "python -m exo"`: `NO_EXO_PROC` (exo not started, as expected).
- `ping -c 5 192.168.200.1` (node2 → node1 across TB): `5 packets transmitted,
  5 packets received, 0.0% packet loss`, `round-trip min/avg/max/stddev =
  0.477/0.608/0.667/0.071 ms`.

**Verdict: TB/RDMA link fully healthy on both nodes, both directions, 0% loss,
sub-ms latency, both ports "Device connected", no stale exo processes.** The
post-reboot wedge fix is confirmed. RDMA port states will be re-verified by
`start_cluster.sh` itself during the launch (it prints `rdma_enN(...)=PORT_ACTIVE`).

### Checkpoint + DSpark weights — PRESENT on both nodes

See Phase 0.7 above for the full evidence. Summary:
- Node1 `~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731/` = 155 GB, 48
  shards, index hash `810e55576e2d29570d6b9a0ffaa8202f7cec1ea2`.
- Node2 `~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731/` = 165 GB, 48
  shards, identical index hash. Size delta = APFS sparse accounting, not content.
- 4705 `mtp.*` keys (mtp.0/1/2) packed inside shards 46-48 on both nodes.
- Local overlay head `local--DeepSeek-V4-Flash-DSpark-MTP/` present on both
  (not used by Stage 1's NATIVE path).

---

## Post-launch verification checklist (exact commands, run AFTER the user confirms the Stage-1 launch)

These are read-only verification commands for the post-launch dispatch to run
once the user has launched Stage 1 in tmux and confirmed the cluster is up.
They are NOT run by this PM session. **Quality gate (spec_degen) runs FIRST,
before any throughput measurement.**

### 1. Cluster up + runners ready (wait for the launcher log)

```bash
# On the laptop (where tmux is), tail the launch log until READY:
tail -f /tmp/dspark_s1.log
# Wait for: "READY (2/2)", "HEALTHY! (Nodes: 2, Identities: 2)",
#           "rdma_en3(...)=PORT_ACTIVE" / "rdma_en4(...)=PORT_ACTIVE" on both nodes.
# Abort if: "CRITICAL ERROR: Cluster out of sync", "RunnerFailed",
#           "placement not ready", or no READY within ~6 min.
```

### 2. Env-var propagation audit (both nodes — catch the allowlist trap)

```bash
for n in adams-mac-studio-m4-1.local adams-mac-studio-m4-2.local; do
  echo "=== $n ==="
  ssh adam.durham@$n "ps aux | grep 'python -m exo -v' | grep -v grep | head -1 | awk '{print \$2}' | xargs -I{} ps eww {} | grep -oE 'EXO_DSV4_DSPARK=[^ ]*|EXO_DSV4_DSPARK_NATIVE=[^ ]*|EXO_DSV4_DSPARK_FORCE_LOAD=[^ ]*|EXO_DSV4_MTP=[^ ]*|EXO_DSV4_MTP_DEDICATED=[^ ]*|EXO_SPECULATIVE=[^ ]*|EXO_DSV4_HC_COLLAPSE_KERNEL=[^ ]*|EXO_DSV4_DSPARK_DIR=[^ ]*|EXO_PP_DRAFT_MODEL=[^ ]*|EXO_DSV4_POOL_GROW_STEP=[^ ]*' | sort"
done
```
**Expected on BOTH nodes** (identical — rank consistency):
```
EXO_DSV4_DSPARK=1
EXO_DSV4_DSPARK_NATIVE=1
EXO_DSV4_DSPARK_FORCE_LOAD=1
EXO_DSV4_MTP=1
EXO_DSV4_MTP_DEDICATED=0
EXO_SPECULATIVE=0
EXO_DSV4_HC_COLLAPSE_KERNEL=1
```
(`EXO_DSV4_DSPARK_DIR`, `EXO_PP_DRAFT_MODEL`, `EXO_DSV4_POOL_GROW_STEP` must
NOT appear — confirming they were not forwarded.)

### 3. Head-load log greps (dspark loaded, native head, no fallback)

```bash
for n in adams-mac-studio-m4-1.local adams-mac-studio-m4-2.local; do
  echo "=== $n ==="
  ssh adam.durham@$n "grep -E 'DSpark draft head attached|DSpark head load SKIPPED|DSv4 DSpark overlay failed|DSv4 dedicated MTP overlay failed|NATIVE checkpoint-bundled' ~/.exo/exo_log/exo.log ~/.exo/exo.log.prev 2>/dev/null | tail -5"
done
```
**Expected on BOTH nodes:** `DSpark draft head attached ... (NATIVE
checkpoint-bundled head, N tensors, 3 stages, ...)`. **Must NOT see:**
`DSpark head load SKIPPED` (means FORCE_LOAD didn't propagate), `DSv4 DSpark
overlay failed` (head load error → fallback), or `DSv4 dedicated MTP overlay
failed` (only fires if MTP_DEDICATED=1 leaked through).

### 4. Memory audit (RSS + footprint + metal active memory, per node)

```bash
for n in adams-mac-studio-m4-1.local adams-mac-studio-m4-2.local; do
  echo "=== $n ==="
  ssh adam.durham@$n "
    PID=\$(ps aux | grep 'python -m exo -v' | grep -v grep | head -1 | awk '{print \$2}');
    echo '--- RSS (ps) ---'; ps -o rss,vsz,pid,command -p \$PID 2>/dev/null;
    echo '--- footprint (real unified memory, NOT RSS) ---'; footprint \$PID 2>/dev/null | head -5;
    echo '--- system memory_pressure ---'; memory_pressure 2>/dev/null | head -3;
    echo '--- vm_stat (swap) ---'; vm_stat | grep -iE 'swap|free|wired';
    echo '--- top 3 RSS python procs ---'; ps aux | grep python | grep -v grep | sort -k5 -rn | head -3 | awk '{print \$5/1024/1024 \" GB RSS - \" \$11,\$12,\$13}'
  "
done
```
**Gate:** peak footprint < 126 GB/node, 0 swap (swap used = 0), `memory_pressure`
not in critical/warn. If the runner pid can't be found, the cluster didn't come
up — check `/tmp/dspark_s1.log` for RunnerFailed. For the truest per-process
number on Apple Silicon, prefer `footprint <pid>` over `ps RSS` (RSS excludes
Metal/GPU unified memory — see `exo-cluster-debugging` skill pitfall).

### 5. Decode smoke (non-speculative — SPECULATIVE=0 so this is plain decode)

```bash
curl -s http://adams-mac-studio-m4-1.local:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash-0731","messages":[{"role":"user","content":"Say exactly: hello world"}],"max_tokens":20,"temperature":0}' \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); print("CONTENT:", repr(r["choices"][0]["message"]["content"])); print("FINISH:", r["choices"][0]["finish_reason"]); print("PROMPT_TOK:", r["usage"]["prompt_tokens"], "COMPLETION_TOK:", r["usage"]["completion_tokens"])'
```
**Expected:** coherent text containing "hello world" (case-insensitive),
`finish_reason=stop`, non-zero completion tokens. **Abort if:** empty content,
BOS-spam (`<｜` repetition), `finish_reason=length` with garbage, or HTTP 5xx.

### 6. spec_degen baseline + diff (QUALITY GATE — runs FIRST, before any t/s)

```bash
# Capture the Stage-1 (head-loaded, spec-off) baseline on the trigger set:
cd ~/repos/exo
uv run python bench/spec_degen_capture.py \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --out /tmp/dspark_s1_specdegen.json \
  --max-tokens 256 2>&1 | tee /tmp/dspark_s1_specdegen.log

# If a spec-off-without-head baseline was captured pre-launch, diff against it:
uv run python bench/spec_degen_diff.py \
  --trace /tmp/dspark_s1_specdegen.json \
  --ground-truth /tmp/spec_off_no_head_baseline.json \
  --max-period 6 --min-repeats 3 2>&1 | tee /tmp/dspark_s1_specdegen_diff.log
```
**Gate:** zero BOS-spam, zero period-≥3 loops on ANY of the 6 system+user
trigger prompts, `control_user_only` clean. If ANY trigger degenerates → STOP,
do not proceed to throughput or Stage 2. (See `exo-dsv4-degeneration-sampler`
skill for the full trigger-set semantics.)

### 7. 50 clean decode steps (stability gate)

```bash
# Run a single longer generation and confirm it completes without desync/OOM:
curl -s http://adams-mac-studio-m4-1.local:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash-0731","messages":[{"role":"user","content":"Count from 1 to 50, one number per line."}],"max_tokens":300,"temperature":0}' \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); c=r["choices"][0]["message"]["content"]; print("LINES:", c.count(chr(10))+1); print("FINISH:", r["choices"][0]["finish_reason"]); print(c[:200])'
# Then check both node logs for desync/OOM in the window:
for n in adams-mac-studio-m4-1.local adams-mac-studio-m4-2.local; do
  ssh adam.durham@$n "grep -iE 'desync|oom|killed|shard mismatch|tensor shape' ~/.exo/exo_log/exo.log | tail -3"
done
```
**Gate:** ≥50 clean decode steps, no desync/OOM/shard-mismatch log entries.

---

## What is blocked on the live cluster

- **Phase 0.1 memory audit** — measure peak footprint per node at steady-state
  with the head loaded. NOW RUNNABLE (cluster back up) but requires the
  Stage-1 launch first (the head must be loaded to measure it).
- **Phase 1 baseline capture (Treatment A, spec-off without head)** — 5
  prompts × 256 tokens at 100K and 352.6K ctx, spec-off, NO head. Runnable
  with a separate spec-off launch (not this doc's Stage 1). Optional: the
  Stage-1 launch itself (head loaded, spec-off) can serve as the "head
  loaded, spec-off" reference for the spec_degen diff.
- **Stage 1 head-load relaunch + validation** — needs user launch approval
  (command written above). Post-launch verification = the checklist above.
- **Stage 2 full-spec relaunch + validation** — needs Stage 1 to pass first.

## What is verified (no cluster needed) — summary

| Item | Verdict | Evidence |
|---|---|---|
| Native head path reads `mtp.0/1/2.*` | ✅ | `utils_mlx.py:879,441` |
| Head attaches BEFORE sharding | ✅ | `utils_mlx.py:866,1016`, dispatch at `:500` |
| DSpark head sharded across TP | ❌ **REPLICATED (~10GB/node)** | `auto_parallel.py:1049-1180` no dspark ref; `deepseek_v4.py:6320` has the MoE; consult-confirmed |
| MTP head double-load | partial — two distinct modules, sum costs | `deepseek_v4.py:6500-6507` (mtp) vs `utils_mlx.py:866` (dspark) |
| `EXO_DSV4_MTP_DEDICATED` launch-path default | ⚠️ **defaults to 1 in start_cluster.sh:468** (NOT unset) — Stage 1 must explicitly set =0 | `start_cluster.sh:468,1921`, `utils_mlx.py:361` |
| TP consumer double-gate (SPEC + MTP) | ✅ | `utils_mlx.py:421`, `dsv4_mtp.py:370-371` |
| No PP fallthrough in TP mode | ✅ | `utils_mlx.py:422-426` requires PipelineShardMetadata |
| Head-load gate single var + FORCE_LOAD override | ✅ | `utils_mlx.py:418,420,427` (FORCE_LOAD bypasses consumer gate; draft cycle gated separately at `batch_generate.py:813,822`) |
| `spec_degen_capture.py` intact + triggers | ✅ | `bench/spec_degen_capture.py:37-89`, `--help` runs |
| Config dspark params correct | ✅ | `config.json`: block_size=5, target=[40,41,42], markov=256 |
| Native head weights on disk | ✅ **PRESENT (blocker resolved)** | both nodes 48 shards, 4705 mtp.* keys in index, hash `810e5557...` identical |
| Post-reboot TB link (node1) | ✅ HEALTHY | `192.168.200.1` active, 2 ports "Device connected", ping node2 0% loss 0.85ms avg, no exo proc |
| Post-reboot TB link (node2) | ✅ HEALTHY | `192.168.200.2` active, 2 ports "Device connected", ping node1 0% loss 0.61ms avg, no exo proc |
| Checkpoint consistency across nodes | ✅ identical | same 57 files, 48 shard names, index.json SHA-1 `810e5557...` on both; 155G/165G = APFS sparse accounting |
| Stage-1 launch command env-forwarding | ✅ verified | each inline var maps to a `start_cluster.sh` EXO_ENV allowlist line (cited per-var above) |

---

## Next approval-gate status

**No relaunch is requested by this audit.** All static-work + read-only
verification items are complete. The cluster is back up, TB link healthy,
weights present on both nodes, and the exact Stage-1 launch command is written
above. **What I'm waiting on:** the user to explicitly approve and run the
Stage-1 `tmux new-session -d -s dspark_s1 ...` command themselves (their
established pattern — a relaunch kills this session). A separate post-launch
dispatch will run the verification checklist above once the user confirms the
cluster is up. The REPLICATED-head finding may prompt the user to instead
prioritize a sharding code fix before any flag flip — that decision is the
user's, not this audit's.

---

# Stage 2b — "DSpark native at 3 tokens" A/B (pre-registered 2026-08-26)

## Background / why a Stage 2b

Stage 2 (spec-ON, `EXO_SPECULATIVE_GAMMA=2`) FAILED throughput at both depths
(−3.1% @100K, −11.7% @352.6K vs the 28.46/25.07 spec-off baseline) and was
REVERTED. **Post-mortem finding (verified by supervisor, 2026-08-26):** the
DSpark draft branch at `dsv4_mtp.py:3673` did `gamma = _dspark.block_size`
UNCONDITIONALLY — it SILENTLY IGNORED `EXO_SPECULATIVE_GAMMA` on the DSpark
path. So Stage 2 did NOT test γ=2; it tested `block_size=5` (anchor + 4 draft
positions) with confidence pruning (`EXO_DSV4_DSPARK_CONF_TAU=0.5`). The
DSpark native head is a 3-stage block (`n_stages = n_mtp_layers = 3`,
`deepseek_v4.py`) trained for **width-3** draft/verify — not 5, not 2.

**User hypothesis:** "DSpark should be native at 3 tokens I think" — the head
should run natively at width 3 (draft + verify = 3 positions), matching its
3-stage training. This is a root-cause fix + re-test, not a parameter sweep.

## Code fix shipped (commit `433dce6c1`, main)

`src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` DSpark branch:
1. After `gamma = _dspark.block_size`, an explicitly-set
   `EXO_SPECULATIVE_GAMMA` (>0) now CAPS gamma to `min(env, block_size)`.
   Default (env unset) stays `block_size` — zero behavior change for
   existing configs.
2. `width=gamma` passed to `_dspark.draft(...)` so draft compute AND verify
   length both truncate to the same width (matching `draft()`'s width
   contract at `deepseek_v4.py:6393-6424`: "truncates the draft to fewer than
   block_size positions ... callers MUST trim their caches by the SAME width
   afterwards").
3. DSpark cache trim changed from `_dspark.block_size` to `gamma` (the
   physical KV written by `draft(width=gamma)` is exactly `gamma` positions
   per stage; trimming by `block_size` would over-trim into the next cycle).

`start_cluster.sh` already forwards `EXO_SPECULATIVE_GAMMA` (line 1632,
unconditional) and `EXO_DSV4_MTP_LOG_INTERVAL` (line 1939, opt-in) to
runners — no forwarding change needed. Gates: basedpyright 726→726 (zero
NEW), ruff 20→20 (zero NEW), `test_pp_speculation_spec_tag.py` 45/45 pass.

## Hypothesis

Width-3 draft/verify aligns with the DSpark head's 3-stage training depth.
Stage 2 ran width-5 (block_size, the env was silently ignored) → the verify
forward processed 2 extra positions the head was never trained to draft,
plausibly tanking acceptance and adding per-row verify cost. At width 3,
acceptance should rise (drafts within the trained distribution) and the
per-cycle verify cost falls (3 rows not 5), so the net decode rate should
beat BOTH the spec-off baseline (28.46/25.07) AND the Stage-2 width-5
result (27.57/22.13).

## Flags (Stage 2b)

```
EXO_SPECULATIVE=1
EXO_DSV4_MTP=1
EXO_DSV4_DSPARK=1
EXO_DSV4_DSPARK_NATIVE=1
EXO_DSV4_DSPARK_FORCE_LOAD=1
EXO_DSV4_MTP_DEDICATED=0
EXO_SPECULATIVE_GAMMA=3          # <-- THE FIX: now honored, caps width to 3
EXO_SPECULATIVE_TEMP=0.0
EXO_SPECULATIVE_ALPHA=1.0
EXO_DSV4_HC_COLLAPSE_KERNEL=1
EXO_DSV4_MTP_LOG_INTERVAL=1      # <-- per-cycle acceptance stats (was 0 in Stage 2)
# EXO_DSV4_DSPARK_CONF_TAU stays default 0.5 (confidence pruning still active)
```

`EXO_DSV4_MTP_LOG_INTERVAL=1` is added because Stage 2 had
`accepted_prediction_tokens=0` on the OpenAI usage path (the TP code path
does not populate those API counters). The internal per-cycle logger at
`dsv4_mtp.py:1969-1998` (`_record_acceptance`, gated by `_LOG_INTERVAL` =
`EXO_DSV4_MTP_LOG_INTERVAL`) emits `[MTP] cycles=N mean_accept=X/γ hist=...`
every cycle — this is the acceptance evidence channel this run.

## Acceptance bars (must hit ALL to PROMOTE)

| metric | bar | baseline (spec-off) | Stage 2 (width-5) |
|---|---|---|---|
| tok/s @100K | ≥ +15% vs 28.46 → ≥ 32.73 | 28.46 | 27.57 (−3.1%) |
| tok/s @352.6K | ≥ +10% vs 25.07 → ≥ 27.58 | 25.07 | 22.13 (−11.7%) |
| mean_accept/γ (from LOG_INTERVAL logs) | ≥ 0.5 (i.e. ≥1.5/3 accepted/cycle) | n/a | unmeasured |
| spec_degen 7/7 trigger set | 0/7 diff vs baseline (no BOS-spam/loop/collapse) | clean | clean |
| math_digit_sum control | answer correct (115), finish=stop, no loop | clean | clean |
| early-stop anomaly | NONE — completion reaches max_tokens (2000), finish=length like baseline | 2000/length | **760/stop, 340/stop (ANOMALY)** |

**Rollback:** any throughput DECREASE vs baseline at either depth, OR any
quality fail (BOS-spam/loop/collapse, math loop, early-stop anomaly
recurs), OR memory blow → REVERT to spec-off (`dspark_revert` command:
`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 EXO_DSV4_HC_COLLAPSE_KERNEL=1`).

## Procedure

1. SIGTERM old tmux sessions (dspark_revert) — SIGTERM only, NEVER kill -9.
2. `tmux new-session -d -s dspark_s2b 'cd ~/repos/exo && <flags above>
   ./start_cluster.sh 2>&1 | tee /tmp/dspark_s2b.log'`, wait READY (2/2).
3. Env-audit both nodes: `ps eww $(pgrep -f spawn_main | head -1) | tr ' '
   '\n' | grep EXO_SPECULATIVE_GAMMA` — CONFIRM `=3` reached runners this
   time (Stage 2 could not confirm because the env was ignored in code;
   now it must be honored AND present in the process env).
4. Head-load greps: `DSpark draft head attached` (tensors, stages,
   block_size, taps) on the rank that owns it.
5. Memory audit: `footprint -p $(pgrep -f spawn_main | head -1)` both
   nodes; expect ~89GB (Stage 1/2 level), 0 swap.
6. QUALITY FIRST:
   - `bench/spec_degen_capture.py` 7 trigger prompts vs baseline →
     `spec_degen_diff.py`, expect 0/7 diff.
   - `bench/hard_eval.py --tasks math_digit_sum --max-tokens 8000
     --temperature 0` → expect answer=115, finish=stop, no loop.
   - ANY BOS-spam/loop/collapse = hard fail, revert immediately.
7. `bench/p3_depth_anchor_probe.py --host <api-node> --port 52415 --model
   deepseek-ai/DeepSeek-V4-Flash-0731 --target-tokens 100000 --max-tokens
   2000` → record tok/s (events AND usage), TTFT, completion_tokens +
   finish_reason (anomaly check), memory. ONE AT A TIME.
8. Repeat at `--target-tokens 352600`. One at a time.
9. Grep runner logs for `[MTP] cycles=... mean_accept=.../3 hist=...` (the
   LOG_INTERVAL output) — extract acceptance stats.
10. DECIDE vs bars: PASS → keep Stage 2b flags running (cluster STAYS on
    them); FAIL → revert to spec-off (`dspark_revert`).
11. `docs/PERFORMANCE_HISTORY.md` entry with real numbers + 2-3
    generated-text samples. Commit + push everything.

---

# Stage 2c — "EOS-ban fix + width-3" A/B (pre-registered 2026-08-26)

## Background / why a Stage 2c

Stage 2b (width-3, commit `433dce6c1`) measured **+13.3% tok/s @100K
(32.25 vs 28.46 baseline)** — the first real DSpark win — but the
**early-stop anomaly RECURRED**: completion 1148/2000, `finish=stop`.
Quality was 7/7 PASS, math_digit_sum PASS (115), acceptance 1.585/3 =
52.8% (from `EXO_DSV4_MTP_LOG_INTERVAL=1` `[MTP]` log lines). The
throughput number is real but INVALIDATED by the early stop — the
probe didn't run to 2000 tokens, so the tok/s is measured over a
shorter decode window (inflated).

**Root cause (confirmed by independent audit with file:line):**
`dsv4_speculative_forward` (`dsv4_mtp.py:1420`) calls the model RAW —
no logits_processors applied. At temp=0 the bonus token is
`mx.argmax(verify_logits[0])` at the post-acceptance position, staged
as `_next_tokens` and yielded UNCONDITIONALLY. If the target's raw
argmax there is EOS, EOS enters the committed stream and the
`StopSequenceMatcher` fires `finish=stop`. The non-spec baseline is
immune because `generate.py`'s `_step` applies the `ban_token_ids`
logits_processor (`ban_token_ids` at `generate.py:1718`, wired in
`batch_generate.py:2658-2660`, `eos_ids_from_tokenizer` at
`generate.py:1915`) before sampling. This is independent of width —
Stage 2b's recurrence at 1148 tokens proves it.

## Code fix shipped (commit `ebe272fd7`, main)

`src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`:
1. New import: `from exo.worker.engines.mlx.generator.generate import
   ban_token_ids` (the same function the baseline uses).
2. `DSv4MTPBatchGenerator.__init__` gains an `eos_ids: list[int] |
   None = None` kwarg, stored as `self._spec_eos_ids`. Passed from
   `batch_generate.py:853` as `eos_ids=list(stop_tokens)` (where
   `stop_tokens = set(eos_ids_from_tokenizer(self.tokenizer))` is
   already computed at `batch_generate.py:814`).
3. In all three verify paths (`_speculative_next` linear at ~3904,
   `_speculative_next_batch` at ~2527, `_speculative_next_tree` at
   ~5167), apply `ban_token_ids(self._spec_eos_ids)` to
   `verify_logits` BEFORE any argmax, logsumexp, or tie-break derives
   from them. The ban sets `logits[..., eos_id] = -1e9` across ALL
   verify positions, so neither the per-row accepted-draft argmax
   (`target_tokens`) NOR the bonus argmax (`all_next`) can select EOS.
4. Env-gated: `EXO_DSV4_SPEC_EOS_BAN` default `"1"` (ON for spec
   path); `"0"` opts out for debug. Zero non-spec behavior change:
   the ban only runs inside the speculative verify cycle
   (`EXO_SPECULATIVE=1`).

Gates: basedpyright 726→725 (zero NEW), ruff zero NEW (import
ordering auto-fixed by `ruff --fix`), `test_pp_speculation_spec_tag`
45/45 pass, `test_reasoning_budget_limiter` 27/27 pass.

## Hypothesis

The EOS ban fixes the early-stop anomaly (completion will reach
2000, `finish=length` like baseline). With early-stop gone, width-3
acceptance (~52.8% per Stage 2b) + full 2000-token decode window →
net decode win over the 28.46/25.07 spec-off baseline. The Stage 2b
32.25 tok/s was inflated by the short window; the real number will be
lower but must still clear the +15%/+10% bars to PROMOTE.

## Flags (Stage 2c)

```
EXO_SPECULATIVE=1
EXO_DSV4_MTP=1
EXO_DSV4_DSPARK=1
EXO_DSV4_DSPARK_NATIVE=1
EXO_DSV4_DSPARK_FORCE_LOAD=1
EXO_DSV4_MTP_DEDICATED=0
EXO_SPECULATIVE_GAMMA=3          # width-3 (Stage 2b fix, honored)
EXO_SPECULATIVE_TEMP=0.0
EXO_SPECULATIVE_ALPHA=1.0
EXO_DSV4_HC_COLLAPSE_KERNEL=1
EXO_DSV4_MTP_LOG_INTERVAL=1      # per-cycle acceptance stats
# EXO_DSV4_SPEC_EOS_BAN defaults to 1 (ON) — no need to set explicitly
```

Identical to Stage 2b flags; the EOS-ban fix is in the code
(commit `ebe272fd7`), not a flag. `start_cluster.sh` already forwards
all the above (verified in the Stage 2b doc).

## Acceptance bars (must hit ALL to PROMOTE)

| metric | bar | baseline (spec-off) | Stage 2b (width-3, EOS bug) |
|---|---|---|---|
| tok/s @100K | ≥ +15% vs 28.46 → ≥ 32.73 | 28.46 | 32.25 (INFLATED — early stop) |
| tok/s @352.6K | ≥ +10% vs 25.07 → ≥ 27.58 | 25.07 | not measured (early stop) |
| mean_accept/γ (from LOG_INTERVAL logs) | ≥ 0.5 (i.e. ≥1.5/3) | n/a | 1.585/3 = 52.8% ✅ |
| spec_degen 7/7 trigger set | 0/7 diff (no BOS-spam/loop/collapse) | clean | clean ✅ |
| math_digit_sum control | answer correct (115), finish=stop, no loop | clean | clean ✅ |
| early-stop anomaly | **NONE — completion 2000, finish=length REQUIRED** | 2000/length | **1148/stop (ANOMALY)** |
| memory per node | < 126 GB, 0 swap | 91/90 GB | ~89 GB ✅ |

**The early-stop anomaly MUST be GONE.** If completion < 2000 or
`finish=stop` at 100K/352.6K probes → hard fail, revert immediately
(spec-off relaunch), capture the verify-audit log
(`EXO_DSV4_MTP_VERIFY_AUDIT` if set) and report.

**Rollback:** any throughput DECREASE vs baseline at either depth,
OR any quality fail (BOS-spam/loop/collapse, math loop, early-stop
anomaly recurs), OR memory blow → REVERT to spec-off (`dspark_revert`:
`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 EXO_DSV4_HC_COLLAPSE_KERNEL=1`).

## Procedure

1. SIGTERM old tmux session (`tmux kill-session -t dspark_s2b`) —
   SIGTERM only, NEVER `kill -9` (SIGKILLed runner wedges the TB/RDMA
   link — documented). Verify no exo runners remain on BOTH studios
   via `ssh <node> pgrep -fl 'python -m exo'` before relaunching.
2. `tmux new-session -d -s dspark_s2c 'cd ~/repos/exo && <flags above>
   ./start_cluster.sh 2>&1 | tee /tmp/dspark_s2c.log'`, wait READY
   (2/2).
3. Env-audit both nodes: `ps eww $(pgrep -f spawn_main | head -1) |
   tr ' ' '\n' | grep EXO_SPECULATIVE_GAMMA` — CONFIRM `=3` reached
   runners. Also confirm `EXO_DSV4_MTP_LOG_INTERVAL=1`.
4. Head-load greps: `DSpark draft head attached` (tensors, stages,
   block_size, taps) on the rank that owns it.
5. Memory audit: `footprint -p $(pgrep -f spawn_main | head -1)` both
   nodes; expect ~89 GB, 0 swap.
6. QUALITY FIRST:
   - `bench/spec_degen_capture.py` 7 trigger prompts vs baseline →
     `spec_degen_diff.py`, expect 0/7 diff.
   - `bench/hard_eval.py --tasks math_digit_sum --max-tokens 8000
     --temperature 0` → expect answer=115, finish=stop, no loop.
   - ANY BOS-spam/loop/collapse = hard fail, revert immediately.
7. `bench/p3_depth_anchor_probe.py --host <api-node> --port 52415
   --model deepseek-ai/DeepSeek-V4-Flash-0731 --target-tokens 100000
   --max-tokens 2000` → record tok/s (events AND usage), TTFT,
   completion_tokens + finish_reason (MUST be 2000/length — if it
   stops early, capture verify-audit log + report), memory. ONE AT
   A TIME.
8. Repeat at `--target-tokens 352600`. One at a time.
9. Grep runner logs for `[MTP] cycles=... mean_accept=.../3 hist=...`
   (the LOG_INTERVAL output) — extract acceptance stats.
10. DECIDE vs bars: PASS → keep Stage 2c flags running (cluster STAYS
    on them); FAIL → revert to spec-off (`dspark_revert`).
11. `docs/PERFORMANCE_HISTORY.md` entry with real numbers + 2-3
    generated-text samples. Commit + push everything.

---

## Stage 3 — Pre-Registration (2b config + EOS_BAN default-off fix) — 2026-08-26

**Auditor:** GLM-5.2 (Ollama Cloud), acting as PM.
**Repo HEAD:** `5076824d5` (fix(dspark): make verify-audit bonus_val extraction
fully recursive) + `d8c671501` (fix(dspark): default EXO_DSV4_SPEC_EOS_BAN to 0).
**Rationale:** Stage 2c (ban ON) confirmed the natural-end theory — the ban
breaks losslessness by forcing the committed-stream argmax from EOS (token 1,
in stop trie) to 128822 (`<|end|>` think_end, NOT in stop trie) at natural-end
positions → spec commits 128822 → no stop match → continues → model repeats
→ `.<|end|>Paris` period-3 loop (2/7 spec_degen prompts degenerated). Stage 2b
(ban OFF, width-3) was 7/7 clean, byte-identical to baseline on short prompts,
+13.3% throughput @100K. The fix flips the ban DEFAULT to OFF (env still
overridable to ON for experiments). This Stage 3 validates the fix as the
promotion candidate.

**Config (Stage 3 = Stage 2b + fix):** width-3 (`EXO_SPECULATIVE_GAMMA=3`),
NO EOS ban (`EXO_DSV4_SPEC_EOS_BAN` unset → code default "0" OFF),
`EXO_DSV4_MTP_LOG_INTERVAL=1` (acceptance stats), audit-off (no
AUDIT_ALL/AUDIT path for the perf run). All other flags identical to Stage 2b
(see launch env in Stage 2b section above).

**PRIMARY bar (losslessness — MUST PASS):** committed stream identical to
baseline on all 7 spec_degen prompts INCLUDING finish/stop behavior:
- Where baseline runs to 2000/length, spec must run to 2000/length.
- Where baseline stops naturally (finish=stop), spec must stop at the same
  point (natural-end equality: token 1 enters committed stream → stop trie
  matches → finish=stop, identical to baseline).
- No BOS-spam, no loop, no collapse, no early-stop anomaly.
- `bench/spec_degen_diff.py` against baseline ground-truth: 0/7 diff.
- `math_digit_sum` (temp=0, 8000 tokens): answer=115, finish=stop, no loop.

**SECONDARY bars (throughput — target, not hard gate):**
- tok/s @100K > baseline 28.46 (target >=+10% → >=31.3).
- tok/s @352.6K > baseline 25.07 (target >=+5% → >=26.3).
- acceptance >=50% (from `[MTP]` LOG_INTERVAL lines).
- memory <126 GB per node, 0 swap.

**Rollback:** ANY divergence from baseline on the 7 prompts (including
finish/stop behavior), OR any quality fail (loop/collapse/early-stop),
OR throughput DECREASE vs baseline → REVERT to spec-off
(`EXO_SPECULATIVE=0`). The cluster does NOT stay on Stage 3 config if the
primary bar fails.

**Procedure:**
1. Cluster is SIGTERM'd and clean (verified 2026-08-26 12:30 CDT).
2. Relaunch 2b+fix config (BAN unset=default off, GAMMA=3, LOG_INTERVAL=1)
   on both nodes via screen, wait for DSpark head attach + model load.
3. Env-audit both nodes: CONFIRM `EXO_DSV4_SPEC_EOS_BAN` is UNSET (not in
   env) or `=0` — the ban must be OFF. Confirm `EXO_SPECULATIVE_GAMMA=3`,
   `EXO_DSV4_MTP_LOG_INTERVAL=1`.
4. QUALITY FIRST (one at a time, temp=0):
   - `bench/spec_degen_capture.py` 7 prompts → diff vs baseline → 0/7.
   - `bench/hard_eval.py --tasks math_digit_sum --max-tokens 8000
     --temperature 0` → answer=115, finish=stop, no loop.
5. p3 probes ONE AT A TIME:
   - `bench/p3_depth_anchor_probe.py --target-tokens 100000 --max-tokens 2000`
     → tok/s (events+usage), TTFT, completion/finish (expect 2000/length),
     memory.
   - `bench/p3_depth_anchor_probe.py --target-tokens 352600 --max-tokens 2000`
     → same.
6. Grep `[MTP] cycles=... mean_accept=.../3` for acceptance stats.
7. DECIDE vs bars: PROMOTE (cluster STAYS Stage 3) or REVERT (spec-off).
8. `docs/PERFORMANCE_HISTORY.md` entry + report. Commit + push.

---

## Stage 3 — RESULT (2026-08-26, GLM-5.2 PM): PRIMARY FAIL → REVERT

**Status: REVERT.** The Stage 3 validation ran against the pre-registered bars
and tripped the PRIMARY hard gate at BOTH probe depths. Cluster does NOT stay
on Stage 3.

### PRIMARY (losslessness, hard gate) — FAIL

Depth probes (`p3_depth_anchor_probe`, `/bench/chat/completions`, temp=0):

| depth | baseline | stage3 | verdict |
|---|---|---|---|
| 100K | completion=2000, finish=length | **completion=369, finish=stop** | **FAIL** (early-stop anomaly) |
| 352.6K | completion=2000, finish=length | **completion=462, finish=stop** | **FAIL** (early-stop anomaly) |

Per the pre-registered bar: "If completion < 2000 or finish=stop at
100K/352.6K probes → hard fail, revert immediately." Both depths tripped it.
Additionally, the spec-ON committed stream diverges from the spec-OFF baseline
**from token 1** (common-prefix length 0) — a deeper losslessness/correctness
divergence, not merely a different stop point.

**7/7 spec_degen short-prompt parity: PASS** (sys_capital_france→"Paris"/stop
matched baseline exactly; sys_count_to_five→"One, two, three, four,
five."/stop matched). **math_digit_sum: PASS** (3/3, answer=115, finish=stop).
Both held — but they do not cover the deep-probe regime where the anomaly
recurred.

### SECONDARY (throughput, target not hard gate) — FAIL at 352K (below baseline)

| depth | baseline tok/s | stage3 tok/s | delta | bar | met? |
|---|---|---|---|---|---|
| 100K | 28.46 | 29.54 | +3.8% | ≥+10% | NO (target miss) |
| 352.6K | 25.07 | 24.30 | −3.1% | ≥+5% | **NO — below baseline (hard rollback)** |

The 352K throughput DECREASE vs baseline is an independent hard rollback
criterion ("throughput DECREASE vs baseline → REVERT"). The 100K +3.8% is
above baseline but below the +10% target, and is measured over a truncated
12.46s decode window (369 tokens, not 2000) — not a clean comparison.

### Acceptance + memory (both PASS, do not override PRIMARY)

- Acceptance: mean_accept=2.134/3 = **71.1%** (exceeds 50% bar; hist
  0:277, 1:407, 2:318, 3:1264, cycles=2266). Irrelevant to the finish gate.
- Memory: m4-2=95 GB, m4-1=93 GB footprint, 0 swap (under 126 GB bar). PASS.

### Determinism check — nondeterministic at temp=0 (root-cause clue)

A repeat 100K probe (same config/prompt/temp=0) produced completion=504 vs
the first run's 369 — a 37% difference with the same text opening. Greedy
decode should be deterministic; the spec verify path is not. This also
explains the Stage 2b (1148) vs Stage 3 (369) discrepancy: it is NOT the ban
code's presence (the ban gate is a verified no-op when `EXO_DSV4_SPEC_EOS_BAN`
is unset), it is nondeterminism in the spec verify path at 100K+ depth.

### Protocol-calibration note (does NOT excuse the fail)

The `/bench` endpoint bans EOS in the committed stream, so the spec-OFF
baseline's "2000/length" is an artifact — the baseline literally cannot
emit EOS. Stage 3's spec path (ban OFF) CAN emit EOS, so its stop may be the
model's true natural greedy end. The discriminator (does spec stream 1..N
match an UNBANNED greedy stream?) cannot be answered from the `/bench`
endpoint. Next-protocol gap: a `/v1/chat/completions` (no bench flag) probe
at 100K to capture the model's true natural-stop point. Per the
pre-registered bars, this calibration issue does not excuse the fail — the
bars were written against the `/bench` baseline and Stage 3 diverged.

### Decision

**REVERT to spec-off.** The user must run the `dspark_revert` relaunch
(SIGTERM-only; the PM session does not relaunch). Full numbers, samples, and
analysis in `docs/PERFORMANCE_HISTORY.md` Stage 3 entry. Artifacts in
`/tmp/ab/s3_*`. Cluster is still running Stage 3 config pending the user's
revert relaunch.