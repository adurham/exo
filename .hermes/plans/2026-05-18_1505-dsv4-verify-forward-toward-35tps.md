# DSv4 Verify-Forward Optimization Plan — Path from 30 t/s to 35 t/s

> **For Hermes:** This is a perf-engineering plan with bench gates between tasks, NOT a TDD-style greenfield plan. Each task ends in a real measurement on the live cluster. Subagent delegation OK for code-reading tasks; cluster bench tasks should stay inline so the operator (Hermes orchestrator + user) can review iter-by-iter.

**Goal:** Achieve >=35 t/s at gamma=2 100K c=1 on DSv4-Flash-8bit with model-correct quality settings (`index_topk=512`). Current baseline: ~30 t/s.

**Architecture:** Verify forward = 91% of cycle (56 ms / 62 ms total). Three identified levers in `mlx-lm/mlx_lm/models/deepseek_v4.py` can together yield 5-7 ms savings if implemented carefully. ACK/jaccl-side optimizations were proven counterproductive -- DO NOT touch them.

**Tech Stack:** exo + mlx-lm + mlx (forks of Apple's). All Python changes deploy via uv git source pins in pyproject.toml; mlx-side changes require careful binary verification (see Task 0).

---

## Critical Context (Read First)

### The False Champion Lesson (2026-05-18)

Multiple "champions" claimed yesterday were **measurement artifacts on the same unfixed binary**. The mlx submodule has TWO independent pinnings and they're not the same thing:

1. **Git submodule** at `mlx/` -- used for reading code, NOT what gets executed.
2. **`tool.uv.sources` in pyproject.toml** -- what `uv sync` installs into the venv. This is what executes.

Yesterday morning every "mlx fix" went into the submodule but `pyproject.toml` pointed at `adurham/mlx@main` (not the fix branch), so the binary was unchanged. When the pyproject WAS finally updated to the fix branch, the deployed code regressed performance catastrophically (4.3 t/s, 10-min READY check).

**Required deployment-verification protocol** (Task 0 below) for every mlx-side change.

### Three Things That DON'T Work

Do NOT propose, attempt, or revisit:

1. **ACK barrier optimization** (`ack_sync_pre` wiring, dedicated ACK QP for top-level, fastskip). Proven 2026-05-18 to cause severe perf regression when deployed. The 50 us/RT cost estimate was wrong.
2. **Removing the `mtp_module.py:654` `mx.eval(tok_arr)` per-step fence.** Reintroduces gamma=2 iter-1 stall. Confirmed twice.
3. **`mx.async_eval` swap of the per-step fence.** Iter-1 stall. Confirmed.

### What DOES Work (Quality vs Speed Reality)

- `EXO_DSV4_INDEX_TOPK=160`: 32.35 t/s but quality "unvalidated" per skill doc.
- `EXO_DSV4_INDEX_TOPK=512` (model default, correct quality): **~30 t/s**. <- This is the new baseline.
- `start_cluster.sh` default updated to 512 (commit `59df6258`).
- Target 35 t/s = **+17% lift from quality-correct baseline**.

### Profile Breakdown (verified 2026-05-18 on production cluster)

```
draft     4.52 ms   (7%)
verify   56.44 ms   (91%)   <- The bottleneck
accept    0.87 ms
rollback  0.19 ms
total    62.01 ms
```

Verify = 86 MoE layers + 86 attn layers. Sparse-pooled attention layers dominate (most layers are sparse-compressed). alpha_2 ~= 0.42 (gamma=2 acceptance rate, capped by model's `num_nextn_predict_layers=1`).

---

## Task 0: Verify Deployment Plumbing (REQUIRED before any mlx-side change)

**Objective:** Confirm changes actually land in the runtime binary on both nodes. This is non-negotiable -- yesterday burned an afternoon on changes that never deployed.

### Step 1: After ANY mlx/mlx-lm push + cluster relaunch, verify the binary

For mlx-lm changes (Python file): grep for a unique string from your patch in
`~/repos/exo/.venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py`.
Must return >=1. If 0, the patch did not deploy.

For mlx changes (C++ in `libmlx.dylib`): use
`strings ~/repos/exo/.venv/lib/python3.13/site-packages/mlx/lib/libmlx.dylib | grep <unique-string>`.

Note the path is `lib/libmlx.dylib` -- NOT `core.cpython-313-darwin.so` which is just the Python binding shim (~1.5 MB) and won't have the implementation strings.

### Step 2: Run a small inference probe BEFORE benching at 100K

```
curl -s --max-time 30 -X POST http://192.168.86.201:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/DeepSeek-V4-Flash-8bit","messages":[{"role":"user","content":"What is 2+2? Reply briefly."}],"max_tokens":30,"temperature":0}'
```

Expected: response contains `"4"`. If the response is empty, garbage, or 500, the patch is broken -- revert immediately, don't bench.

### Step 3: Run a SMALL-PREFILL bench before the 100K bench

Two iters at `--prompt-words 1000` complete in ~1 min. If this fails OR throughput is wildly off, the patch is broken at small L. Don't bench 100K.

**Verification gate:** Both small-prefill probe AND inference probe must pass before promoting to 100K bench.

---

## Task 1: Re-establish Quality-Correct Baseline (10-iter measurement)

**Objective:** Anchor the real starting t/s with statistical confidence at `index_topk=512`. Earlier numbers (29.9 warmup, 30.1 iter-1) are too few samples.

### Step 1: Verify cluster env

Confirm `EXO_SPECULATIVE_GAMMA=2`, `EXO_DSV4_FENCE_EVERY_N_LAYERS=43`, `EXO_DSV4_INDEX_TOPK=512`.

If FENCE is 8 (the start_cluster.sh default for this var), relaunch with `EXO_DSV4_FENCE_EVERY_N_LAYERS=43 ./start_cluster.sh`.

### Step 2: Launch 10-iter bench in screen on m4-1

```
ssh ... "screen -dmS bench_base zsh -l -c '
cd ~/repos/exo
uv run python3 bench/concurrent_bench.py \
  --host localhost --port 52415 \
  --model mlx-community/DeepSeek-V4-Flash-8bit \
  --concurrency 1 --iterations 10 --warmup 1 --max-tokens 128 \
  --prompt-words 75000 --timeout 3600 \
  --json-out ~/bench_baseline_topk512_\$(date +%s).json \
  --label baseline-topk512-g2-10iter \
  2>&1 | tee ~/bench_baseline_topk512.log
'"
```

### Step 3: Active poll (every 30s)

Use a state-change poll: only report when iter count grows OR process dies.
Pseudocode:
```
LAST=0
loop:
  cnt = ssh "if ! pgrep -f concurrent_bench >/dev/null; then echo DEAD; else grep -cE 'iter=' ~/bench_baseline_topk512.log; fi"
  if cnt == DEAD: break
  if cnt > LAST:
    LAST = cnt
    fetch and report last iter line
  sleep 30
```

Expected wall: ~70 min total (11 iters x ~6.25 min).

### Step 4: Compute statistics

After bench completes, parse JSON for per-iter `agg_tps`. Required for "baseline" claim:
- >=10 scored iters
- All >= 29 t/s
- sigma < 0.5
- 0 errors

### Step 5: Tag the baseline

```
cd /Users/adam.durham/repos/exo
git tag -a baseline-2026-05-19-mtp-g2-topk512-XX.X -m "Quality-correct baseline ..."
git push origin baseline-2026-05-19-mtp-g2-topk512-XX.X
```

**Gate:** Do not proceed to Task 2 without >=10/10 clean iters. If bistability appears (any iter <29 t/s), investigate jaccl mesh.cpp on origin/main BEFORE touching anything else -- it means today's deployment state has a bug we don't know about.

---

## Task 2: Lever 1 (Take 2) -- Fused SDPA for L_q>1 with Prefill-Safe Mask Reshape

**Objective:** Refactor `_sparse_pooled_attention` L_q>1 path to use `mx.fast.scaled_dot_product_attention` via batch-axis fold. The first attempt (commit `4aa77635`, reverted as `8aa2f131`) failed at prefill with a concatenate dim-mismatch -- the mask reshape didn't handle prefill's shapes.

**Files:**
- Modify: `mlx-lm/mlx_lm/models/deepseek_v4.py:680-836` (the `_sparse_pooled_attention` function).
- Reference: same file, lines 756-820 (the working L_q==1 fast path).

### Step 1: Read the prefill mask shapes (do not skip)

Add temporary debug instrumentation at the top of `_sparse_pooled_attention`:

```python
import os as _os_dbg
if _os_dbg.environ.get("EXO_DSV4_DBG_SPATTN") == "1":
    import sys as _sys_dbg
    lm_shape = local_mask.shape if local_mask is not None else None
    pm_shape = pooled_mask.shape if pooled_mask is not None else None
    lm_dim = local_mask.ndim if local_mask is not None else None
    pm_dim = pooled_mask.ndim if pooled_mask is not None else None
    print(f"[SPATTN] L={L} local_mask={lm_shape} (ndim={lm_dim}) pooled_mask={pm_shape} (ndim={pm_dim})", file=_sys_dbg.stderr, flush=True)
```

Commit this DEBUG ONLY patch. Push. Relaunch cluster with `EXO_DSV4_DBG_SPATTN=1`. Run a single small inference (the curl in Task 0 Step 2).

Capture: `ssh ... "grep '\[SPATTN\]' ~/exo.log | head -30"`

This tells you the exact mask shapes at small prefill (L=~30-200) AND at decode (L=3). Find the SMALLEST L where local_mask is None or has unexpected rank -- that's where the previous patch broke.

### Step 2: Diagnose the dim-mismatch

The error was:
```
[concatenate] All the input arrays must have the same number of dimensions. However, got arrays with dimensions 2 and 4.
```

A 2D mask suggests one of:
- A scalar/causal mask that's still in (L_q, L_kv) form
- A mask dtype branch that returned a non-broadcasted tensor
- An MLX-side broadcast that materialized differently in prefill

Look at the L_q==1 path (lines 760-800) and identify where it assumes both masks become 4D `(B, H, L, *)`. The previous patch assumed the same for L>1 but didn't verify.

### Step 3: Write the prefill-safe refactor

Key insight: the mask normalizer needs to handle:
- Both masks None -> no mask passed to SDPA
- One present, one None -> synthesize all-pass for the missing side
- 2D causal mask (L_q, L_kv) form -> reshape EXPLICITLY to (B, H, L, K)
- 4D with shape[1]==1 -> broadcast head axis if H>1

Pseudocode:
```python
def _to_4d_mask(m, B, H, L, K, target_dtype):
    if m is None:
        if target_dtype == mx.bool_:
            return mx.ones((B, H, L, K), dtype=mx.bool_)
        return mx.zeros((B, H, L, K), dtype=target_dtype)
    if m.ndim == 2:
        m = m[None, None, :, :]
        m = mx.broadcast_to(m, (B, H, L, K))
    elif m.ndim == 4:
        if m.shape[1] == 1 and H > 1:
            m = mx.broadcast_to(m, (B, H, L, K))
    else:
        raise ValueError(f"unexpected mask ndim={m.ndim} shape={m.shape}")
    if m.dtype != target_dtype:
        m = m.astype(target_dtype)
    return m

# In the L_q>1 path, after gather + reshape:
combined_mask = None
if local_mask is not None or pooled_mask is not None:
    target = local_mask.dtype if local_mask is not None else pooled_mask.dtype
    lm = _to_4d_mask(local_mask, B, H, L, sw, target)
    pm = _to_4d_mask(pooled_mask, B, H, L, k, target)
    combined_4d = mx.concatenate([lm, pm], axis=-1)  # (B, H, L, sw+k)
    combined_mask = combined_4d.transpose(0, 2, 1, 3).reshape(B * L, H, 1, sw + k)
```

### Step 4: Apply the patch (do NOT commit yet)

Update `_sparse_pooled_attention` with:
- The `_to_4d_mask` helper at module scope just above `_sparse_pooled_attention`.
- The new L_q>1 path using `_to_4d_mask` for both `local_mask` and `pooled_mask`.
- Keep the L_q==1 fast path unchanged (don't touch what works).
- Remove or refactor `_sparse_pooled_attention_inner` once the new path replaces it. Or keep it as a dead-code fallback gated by an env var for emergency rollback.

### Step 5: Local syntax check

```
cd /Users/adam.durham/repos/exo/mlx-lm
python3 -c "import py_compile; py_compile.compile('mlx_lm/models/deepseek_v4.py', doraise=True); print('OK')"
```

### Step 6: Commit + push mlx-lm

```
cd /Users/adam.durham/repos/exo/mlx-lm
git add mlx_lm/models/deepseek_v4.py
git commit -m "perf(dsv4-attn): fold L_q into batch for fused SDPA (mask-shape-safe)"
git push origin main
```

### Step 7: Bump exo submodule + uv.lock

```
cd /Users/adam.durham/repos/exo
uv lock --upgrade-package mlx-lm
git add mlx-lm uv.lock
git commit -m "deps: bump mlx-lm to <sha> (lever 1 take 2)"
git push origin main
```

### Step 8: Restart cluster + Task 0 verification

Tear down both nodes, launch, active-poll for READY, then run the Task 0 protocol IN FULL:
1. Patch-deployed strings-grep.
2. Inference probe.
3. Small-prefill 2-iter bench.

**Gate:** If small-prefill bench fails or quality probe returns garbage:
```
cd /Users/adam.durham/repos/exo/mlx-lm && git revert HEAD --no-edit && git push origin main
cd /Users/adam.durham/repos/exo && uv lock --upgrade-package mlx-lm && git add mlx-lm uv.lock && git commit -m "revert: lever 1 take 2 broke X" && git push origin main
# Then tear down + relaunch cluster.
```

### Step 9: 100K 10-iter bench (only if Task 0 gates pass)

Same launch + active-poll as Task 1. Compute mean, sigma. Compare to Task 1 baseline.

**Success criteria:**
- 10/10 iters complete with 0 errors.
- All >= 29 t/s.
- Mean >= baseline + 1.5 t/s (projected 3-4 ms verify saving = ~+1.5-2 t/s at the cycle).
- sigma < 0.5.

**Failure response:** Revert, capture the failure mode, write a follow-up plan for Lever 2 only.

---

## Task 3: Lever 2 -- Collapse Per-Layer mx.compile Into Fewer Boundaries

**Only run if Task 2 lands and t/s < 35.**

**Objective:** Reduce per-layer Python<->MLX boundary crossings by fusing `_raw_post_attn` and `_raw_ffn_pre` into a single `_raw_attn_to_ffn` compile.

**Files:**
- Modify: `mlx-lm/mlx_lm/models/deepseek_v4.py:2030-2095` (`DeepseekV4Block.install_compiled_forward` and the four `_raw_*` helpers).
- Reference: same file, the `__call__` body that dispatches `self._compiled_post_attn(...)` -> `self._compiled_ffn_pre(...)`.

### Pitfall warning

Pitfall #16: `@mx.compile` on a function whose body calls another `@mx.compile`'d function is net-NEGATIVE. The proposed `_raw_attn_to_ffn` calls `hc_expand` which IS `@mx.compile`'d in `hyper_connection.py:259`. Two options:

(a) **Inline `_hc_expand_op`'s body** into `_raw_attn_to_ffn` (eliminates the nested compile boundary). Cleanest.

(b) **Keep `hc_expand` as is but don't wrap `_raw_attn_to_ffn` in mx.compile** -- call it as a plain Python fn. Loses some optimization but avoids the pitfall.

Recommend (a).

### Step 1: Read `_hc_expand_op` body

```
grep -n "_hc_expand_op\|^def hc_expand" /Users/adam.durham/repos/exo/mlx-lm/mlx_lm/models/hyper_connection.py | head
```

Capture the body. It's a small fn (see Lever 3 in the code-analyzer report for exact code). Inline as `_hc_expand_inline` in deepseek_v4.py module scope.

### Step 2: Write the fused helper

```python
def _hc_expand_inline(x, residual, post, comb):
    y = post[..., None] * x[:, :, None, :].astype(mx.float32)
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
    return y.astype(x.dtype)

# In DeepseekV4Block:
def _raw_attn_to_ffn(self, attn_out, residual_a, post_a, comb_a):
    h = _hc_expand_inline(attn_out, residual_a, post_a, comb_a)
    x, post_f, comb_f = self.ffn_hc(h)
    normed = self.ffn_norm(x)
    return normed, h, post_f, comb_f

def install_compiled_forward(self):
    if self._compiled_attn_pre is not None:
        return
    self._compiled_attn_pre    = mx.compile(self._raw_attn_pre)
    self._compiled_attn_to_ffn = mx.compile(self._raw_attn_to_ffn)
    self._compiled_post_ffn    = mx.compile(self._raw_post_ffn)
```

### Step 3: Update `__call__`

Replace the two-call sequence with one `_compiled_attn_to_ffn`. Also delete the BUILD_PROBE timing blocks that reference the removed compile sections OR update them to match the new structure.

### Step 4: Deploy + verify cycle (same as Task 2 Steps 5-9)

py_compile check -> commit mlx-lm -> bump exo submodule + lock -> push -> restart cluster -> Task 0 deployment verify -> small-prefill probe -> 100K 10-iter bench.

**Success criteria:** Mean t/s >= Task 2 result + 1 t/s, sigma < 0.5, 10/10 clean.

---

## Task 4: Lever 3 -- Inline `_hc_expand_op` Across Both Callers

**Only run if Task 3 doesn't land us at 35 and we want the extra ~1 ms.**

**Objective:** If Task 3 (a) was taken (which inlines hc_expand into one path), do the same inline for `_raw_post_ffn` to eliminate the SECOND compile-cache lookup per layer.

**Files:** `mlx-lm/mlx_lm/models/deepseek_v4.py:2080+` (`_raw_post_ffn`).

Subtask is identical to Task 3's Step 2 applied to the post-FFN side. Same deploy+verify cycle.

**Success criteria:** Mean t/s >= Task 3 result + 0.5 t/s.

---

## Task 5: If Still Short -- Re-Profile to Find the Next Bottleneck

**Trigger:** Task 4 lands but mean < 35.

**Objective:** Re-run `EXO_DSV4_MTP_PROFILE=20` to get an UPDATED phase breakdown after Levers 1-3. The verify time should have dropped from 56 ms to ~49-51 ms. If still >=45 ms, there's more meat there.

### Step 1: Restart cluster with profile env

```
cd /Users/adam.durham/repos/exo && \
  EXO_DSV4_FENCE_EVERY_N_LAYERS=43 EXO_DSV4_MTP_PROFILE=20 EXO_DSV4_MTP_LOG=1 EXO_DSV4_MTP_LOG_INTERVAL=50 \
  ./start_cluster.sh
```

### Step 2: Run 2-iter bench (~12 min)

Same script as Task 2 but `--iterations 2 --warmup 1`.

### Step 3: Grep profile dumps

```
ssh ... "grep '\[MTP-PROF\]' ~/exo.log | tail -30"
ssh ... "grep '\[MTP\] cycles=' ~/exo.log | tail -5"
```

### Step 4: Decide next lever based on data

If verify is now < 45 ms but we're still short, the slack is in `draft` or `accept`. Read `draft_tokens` and `_speculative_next` accept block.

If verify is still ~50 ms, the `_sparse_pooled_attention` change didn't bite as hard as projected -- investigate why. Subagent code-analyzer pass at this point would be appropriate.

---

## Files Likely To Change

- `mlx-lm/mlx_lm/models/deepseek_v4.py` (primary)
- `pyproject.toml` (only if changing mlx-lm git source -- unlikely after today's fix)
- `uv.lock` (auto-regenerated)
- `mlx-lm` submodule pointer in exo

**Do NOT modify:**
- `mlx/` submodule contents (any mlx-side change requires the Task 0 plumbing protocol AND has shown to regress perf)
- `start_cluster.sh` env defaults beyond what's already set
- `src/exo/worker/engines/mlx/speculative/*.py` (especially `mtp_module.py:654` -- that fence is load-bearing)

---

## Tests / Validation

### Per-task validation (Task 0 protocol)

For each mlx-lm change:
1. Local `py_compile` clean.
2. Deployment strings-grep returns >=1.
3. Inference probe at small context returns sensible output.
4. Small-prefill 2-iter bench completes with 0 errors.
5. THEN 100K 10-iter bench, all iters >= 29 t/s, sigma < 0.5.

### Quality gate

After each successful Task, run the canonical quality probe:

```
# scripts/quality_probe_dsv4.py from skill -- copy to m4-1:~/repos/exo/bench/ first per pitfall #13
ssh ... "cd ~/repos/exo && uv run python3 bench/quality_probe_dsv4.py \
  --host localhost --port 52415 \
  --model mlx-community/DeepSeek-V4-Flash-8bit \
  --target-tokens 100000 --output ~/quality_<TASK>.json"
```

Quality must hold: needle-found = True, response coherent.

---

## Risks, Tradeoffs, and Open Questions

### Known risks

1. **Lever 1 take 2 might still break prefill.** The mask shape diagnostic in Task 2 Step 1 is the protection. If shapes don't match the assumptions, the helper needs adjustment.

2. **The fused SDPA's `sinks` semantics at B*L large batch.** The L_q==1 path passes `sinks` directly; same code path will exercise this at B*L scale. Should work (sinks is shape (H,) and broadcast across batch), but the microbench at L=3 in Task 0 Step 3 should catch any anomaly.

3. **Task 3 nested-compile pitfall.** Inlining `_hc_expand_op` mitigates it. If we go option (b) instead, the gain shrinks to maybe 0.5 t/s.

4. **The 4.3 t/s mlx-fix regression mystery.** I never figured out WHY deploying the jaccl ACK QP / ack_sync_pre changes caused such severe slowdown. The fix branch is still on origin (`fix/jaccl-ack-qp-top-level`) for future investigation but DON'T re-deploy without understanding the regression mechanism first.

### Tradeoffs

- **Patch verification overhead (Task 0).** Adds 5-10 min per task. Non-negotiable after yesterday's deployment fiasco.
- **mlx-lm rebuilds.** uv builds wheel on cluster nodes on each new mlx-lm git pin. ~30-60 sec per node. Acceptable.

### Open questions

1. **Does `mx.fast.scaled_dot_product_attention` accept B*L=3 with H=128 cleanly on macOS Metal?** Likely yes (the L_q==1 path exercises this at B=1, H=128). Task 0 Step 3 small-prefill bench will confirm.

2. **What's the actual alpha_2 vs index_topk relationship?** When we ran at topk=160 we saw alpha_2=0.42. At topk=512 the acceptance might be slightly higher (more accurate scoring), partially closing the gap on its own. Worth noting from the profile-pass acceptance log in Task 1 / Task 5.

3. **Is the cluster's ~30 t/s at topk=512 baseline actually stable, or does it have its own bistability?** Task 1's 10-iter run answers this. If we see any iter <29, that's a Day 0 problem before we can lever anything.

---

## Quick Reference: Cluster Operations

### Launch cluster (default = topk=512 quality-correct)
```
cd /Users/adam.durham/repos/exo && EXO_DSV4_FENCE_EVERY_N_LAYERS=43 ./start_cluster.sh
```

### Tear down both nodes
```
SSH_OPTS="-i ~/.ssh/exo_cluster -o IdentitiesOnly=yes"
for host in 192.168.86.201 192.168.86.202; do
  ssh $SSH_OPTS adam.durham@$host '
    screen -S exorun -X quit 2>/dev/null
    sleep 1
    pkill -9 -f "python -m exo" 2>/dev/null
    pkill -9 -f "from multiprocessing" 2>/dev/null
    sleep 2'
done
```

### Bench launch template
```
SSH_OPTS="-i ~/.ssh/exo_cluster -o IdentitiesOnly=yes"
ssh $SSH_OPTS adam.durham@192.168.86.201 "
  screen -dmS bench_<LABEL> zsh -l -c '
cd ~/repos/exo
uv run python3 bench/concurrent_bench.py \
  --host localhost --port 52415 \
  --model mlx-community/DeepSeek-V4-Flash-8bit \
  --concurrency 1 --iterations 10 --warmup 1 --max-tokens 128 \
  --prompt-words 75000 --timeout 3600 \
  --json-out ~/bench_<LABEL>_\$(date +%s).json \
  --label <LABEL> \
  2>&1 | tee ~/bench_<LABEL>.log
'"
```

### Active iter poll (use INSTEAD of dumb sleeps)
```
LAST=0
while true; do
  cnt=$(ssh ... "if ! pgrep -f concurrent_bench >/dev/null; then echo DEAD; exit; fi; grep -cE 'iter=' ~/bench_<LABEL>.log")
  [[ "$cnt" == "DEAD" ]] && break
  if (( cnt > LAST )); then
    LAST=$cnt
    ssh ... "grep -E 'iter=' ~/bench_<LABEL>.log | tail -1"
  fi
  sleep 30
done
```

---

## DO NOT (Hard Constraints)

- **DO NOT modify mlx submodule code** without going through Task 0 protocol AND understanding why yesterday's "fixes" regressed perf. Skip mlx-side changes entirely unless absolutely necessary.
- **DO NOT remove the mtp_module.py:654 mx.eval fence.** Confirmed twice it causes gamma=2 iter-1 stall.
- **DO NOT swap mx.eval for mx.async_eval** in the per-step fence. Same stall.
- **DO NOT touch ack_sync_pre / ack_sync_post / ACK QP code** -- proven counterproductive.
- **DO NOT lower index_topk below 512** without an explicit quality regression budget from the user.
- **DO NOT bench at 100K before validating at small prefill.** Yesterday's broken Lever 1 attempt was caught only after spending 4 min in a doomed 70-min bench because I skipped this.
- **DO NOT use long `sleep N && check`** patterns. Use active state-change polling.
- **DO NOT claim a champion** without >=10 iters at the production config, all >=29 t/s, sigma<0.5, 0 errors.

## Remember

- A small-prefill (~1K word) bench takes ~1 minute and catches 90% of bugs.
- The mlx-lm submodule update path has been verified working: edit -> commit -> push origin/main -> `uv lock --upgrade-package mlx-lm` -> commit lockfile -> relaunch cluster -> strings-grep verify on `lib/libmlx.dylib` (or for mlx-lm, just grep the Python file directly).
- Don't make me wrong about deployment again. Verify before benching.
