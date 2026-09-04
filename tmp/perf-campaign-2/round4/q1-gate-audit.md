# Q1 GATE AUDIT — is the c=1 async fence ARMED at decode steady state?

**Date:** 2026-09-03 · **Mode:** read-only static audit, no code changes, nothing committed.
**Repo:** `/Users/adam.durham/repos/exo` (branch `main`, working tree = deploy vehicle via `start_cluster.sh` rsync).

**Config under audit (LIVE, both nodes):**
`MLX_JACCL_SHARDING_MODE=Tensor`, `EXO_SPECULATIVE=1`, `EXO_DSV4_MTP=1`, `EXO_DSV4_DSPARK=1`,
`EXO_DSV4_FENCE_ASYNC=1`, `EXO_DSV4_FENCE_ASYNC_C2=0`, `EXO_DSV4_FENCE_EVERY_N_LAYERS=4`.

Model-side file audited: `mlx-lm/mlx_lm/models/deepseek_v4.py`. Verified byte-identical to the
installed copy `.venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py` for both the
registry header block (lines 110–175) and the gate block (lines 3110–3155) — `diff` clean.
Both files are 343983 bytes.

---

## VERDICT (short)

**ARMED at c=1 decode steady state.** Every fence site reached during a normal c=1 DSpark
verify/decode cycle passes all five gate conjuncts.

**BUT** there are exactly **two** decode-time (not prefill, not transition) windows that fall
back to blocking `mx.eval(y)`:

1. **The regime-b rollback commit-forward** — `dsv4_mtp.py:5035`, a full 43-layer
   `self.model(commit_input, cache=...)` executed between an explicit
   `_set_fence_async(False)` at `dsv4_mtp.py:4982` and the re-arm at `dsv4_mtp.py:5050`.
   Fires on rejection cycles where the cache-level exact undo is not applicable.
2. **The entire c≥2 window** (by design, `FENCE_ASYNC_C2=0`) — all three gate legs
   (engine, cache, `B<=MAX_B`) go False simultaneously.

Plus two findings that materially change how this knob should be read:

- **`EXO_DSV4_FENCE_EVERY_N_LAYERS=4` IS A DEAD KNOB.** `self._fence_every_n` is computed at
  `deepseek_v4.py:2911` and **never read anywhere** — the only other occurrence in the file is a
  historical comment at `deepseek_v4.py:3088`. Grep across `mlx-lm/`, `.venv/.../mlx_lm/` and
  `src/` confirms zero read sites. The fence runs at **every** MoE layer regardless of this value.
- **The DSpark draft forward never touches the fence site at all** under today's config, because
  the DSpark head is NOT TP-sharded (see §4.2). So the fence cost is entirely the main model's
  43 MoE layers.

---

## 1. THE GATE

`mlx-lm/mlx_lm/models/deepseek_v4.py:3115-3121` (inside `DeepseekV4MoE.__call__`):

```
3115|                     elif (
3116|                         _FENCE_ASYNC
3117|                         and _fence_key_ok("engine")
3118|                         and _fence_key_ok("cache")
3119|                         and y.shape[0] <= _FENCE_ASYNC_MAX_B
3120|                         and y.shape[1] <= 8
3121|                     ):
```
- async branch: `deepseek_v4.py:3130` → `mx.async_eval(y)`
- blocking branch: `deepseek_v4.py:3150` → `mx.eval(y)`
- probe branch takes precedence: `deepseek_v4.py:3095` `if _ALLSUM_PROBE_ENABLED:` →
  unconditional blocking `mx.eval(y)` at `deepseek_v4.py:3098`. `EXO_DSV4_ALLSUM_PROBE` is unset
  in prod (`start_cluster.sh:1743`, only exported when non-empty), so this branch is inert.

Guarding scope: the whole block is inside `if self.sharding_group is not None:` at
`deepseek_v4.py:3074`, immediately after `mx.distributed.all_sum` at `deepseek_v4.py:3076`.
**Only MoE modules whose `sharding_group` was assigned reach the fence.**

Env inputs:
- `_FENCE_ASYNC` — `deepseek_v4.py:96`, from `EXO_DSV4_FENCE_ASYNC` (=1) → **True**.
- `_FENCE_ASYNC_MAX_B` — `deepseek_v4.py:103`,
  `max(1, int(EXO_DSV4_FENCE_ASYNC_C2 or 0) or 1)` → with C2=0 → **1**.
- `y.shape[1] <= 8` — verify L. Under DSpark, `gamma = min(EXO_SPECULATIVE_GAMMA, block_size)`
  (`dsv4_mtp.py:3947,3955`), `EXO_SPECULATIVE_GAMMA=3` (`start_cluster.sh:176`), confidence
  pruning at `dsv4_mtp.py:4031` can only shrink it (`gamma = max(1, min(_kept, gamma))`), so
  verify L = gamma+1 ∈ [2,4] (`_verify_len = gamma + 1`, `dsv4_mtp.py:4102`). Plain decode L=1.
  **Always ≤ 8.** ✅

Post-fence: `deepseek_v4.py:3151` `y = finalize(y)`. `finalize` is
`mlx-lm/mlx_lm/profiler.py:109-113` — a no-op when no profiler hook is registered, but
`mx.eval(x)` when one is. **Gotcha: registering `SpanProfilerHook` silently converts every
async fence back into a blocking eval one line later.** Any measurement of "the fence" taken with
the span profiler on is measuring the blocking path.

There is exactly **one** async/blocking fence gate site in the file (grep `mx.async_eval` →
`deepseek_v4.py:3130` only; the other `mx.async_eval` uses in the repo are in `dsv4_mtp.py`
and unrelated to this gate). Other `all_sum` sites (attention tails at `deepseek_v4.py:4317-4322`,
`5099-5104`, and the seq-split `all_gather` at `5090-5098`) have **no** fence at all — they never
call `mx.eval`/`mx.async_eval`. So the fence question is scoped exactly to the MoE `all_sum`.

---

## 2. OWNER REGISTRY — every set site

State:
- `deepseek_v4.py:117` — `_FENCE_ASYNC_CTX = {"engine": False, "cache": False}` (arming values)
- `deepseek_v4.py:131` — `_FENCE_ASYNC_REGISTERED = {"engine": True, "cache": False}` (registration)
- `deepseek_v4.py:134-139` — `_register_fence_async_owner(key)` → sets REGISTERED[key]=True
- `deepseek_v4.py:142-147` — `_fence_key_ok(key)`: `if not REGISTERED.get(key): return True`
  else `return CTX.get(key, False)`
- `deepseek_v4.py:156-158` — `_set_fence_async_ok(ok, key="engine")` → `CTX[key] = bool(ok)`

### 2.1 "engine" — sole owner `batch_generate.py`

Setter wrapper: `src/exo/worker/engines/mlx/generator/batch_generate.py:2141-2157`
(`_set_fence_async_engine`), which on disarm ALSO calls `mx.synchronize()` at
`batch_generate.py:2155` (drain before mutating shared state).

**engine := True** — only via `batch_generate.py:2157`, reached only from
`_update_fence_arming` (`batch_generate.py:2159-2165`) with the predicate at
`batch_generate.py:2163-2164`:
`1 <= len(self._active_tasks) <= limit and not self._pp_spec_active`,
`limit = max(1, int(EXO_DSV4_FENCE_ASYNC_C2 or 0) or 1)` = **1**.

`_update_fence_arming` call sites (all of them):
- `batch_generate.py:1260` (submit-batched-decode-deferred completion)
- `batch_generate.py:1557` (deferred-prefill submit completion)
- `batch_generate.py:2785` (end of `submit()`)
- `batch_generate.py:3234` (per-uid end of `submit_batched()`)
- `batch_generate.py:3528` (end of `_submit_pp_spec`)
- `batch_generate.py:4645` (a request finished, `is_done` branch of `_step`)
- `batch_generate.py:5068` (`remove()` path)
- `batch_generate.py:5147` (`reset_after_reconnect`)

**engine := False** — `batch_generate.py:2154` (+ `mx.synchronize()` at 2155), reached from:
- `_update_fence_arming` when the predicate is False (0 or ≥2 active tasks, or PP-spec active)
- `batch_generate.py:2192` — unconditional at the top of `submit()`
- `batch_generate.py:2825` — unconditional at the top of `submit_batched()`

`_pp_spec_active` (`batch_generate.py:699` default False, set True only at
`batch_generate.py:974`) requires `get_pipeline_info(model) is not None`
(`pp_speculation.py:129-136`, which scans for a `PipelineLastLayer`). Under Tensor sharding no
such layer exists → **`_pp_spec_active` is False**, so the engine predicate reduces to
`len(_active_tasks) == 1`.

### 2.2 "cache" — sole owner `dsv4_mtp.py` (`DSv4MTPPredictor`)

Setter wrapper: `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:1229-1244`
(`_set_fence_async`), disarm also `mx.synchronize()` (`dsv4_mtp.py:1242` region;
`_set_fence_async_ok(False, key="cache")` at 1241, sync immediately after).

**cache := True** — `dsv4_mtp.py:1244`, reached from:
- `dsv4_mtp.py:1269` — end of `snapshot_for_uid()` (post-prefill join)
- `dsv4_mtp.py:1304` — `activate_for_uids()` **no-transition fast path**
  (`self._set_fence_async(len(uids_t) <= _FENCE_ASYNC_MAX_STREAMS)`) ← **this is the steady-state
  re-arm, fired on every spec cycle**
- `dsv4_mtp.py:1349` — `activate_for_uids()` after a real BS transition
- `dsv4_mtp.py:3260` — end of `_speculative_next_batch` rollback (`N <= MAX_STREAMS`)
- `dsv4_mtp.py:5050` — end of the B=1 `_SPEC_STATE_RESTORE` rollback

`_FENCE_ASYNC_MAX_STREAMS` = `dsv4_mtp.py:774-776`,
`max(1, int(EXO_DSV4_FENCE_ASYNC_C2 or 0) or 1)` → **1**. Mirrors the model-side `_FENCE_ASYNC_MAX_B`.

**cache := False** — `dsv4_mtp.py:1241` (+ sync), reached from:
- `dsv4_mtp.py:1204` — top of `reset_cache()` — ⚠ **no re-arm inside this method**
- `dsv4_mtp.py:1265` — top of `snapshot_for_uid()` (re-armed at 1269)
- `dsv4_mtp.py:1310` — `activate_for_uids()` transition path (re-armed at 1349)
- `dsv4_mtp.py:3106` — `_speculative_next_batch` pre-rollback drain (re-armed at 3260)
- `dsv4_mtp.py:4982` — `_speculative_next` (B=1) pre-rollback drain (re-armed at 5050)
- also the `len(uids_t) <= MAX_STREAMS` expressions at 1304/1349/3260 evaluate to False at N≥2

### 2.3 Fail-closed asymmetry

`_fence_key_ok` is fail-CLOSED for registered owners and fail-OPEN for unregistered ones
(`deepseek_v4.py:145-147`). "engine" is registered unconditionally at module import
(`deepseek_v4.py:131`), so it is ALWAYS required. "cache" starts unregistered and only becomes
required when `DSv4MTPPredictor.__init__` completes.

---

## 3. REGISTRATION — who calls `_register_fence_async_owner("cache")`, and is it gated?

Single call site: `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:1169-1171`, at the very end
of `DSv4MTPPredictor.__init__` (guarded only by `try/except ImportError`, `dsv4_mtp.py:1168,1172-1173`).

Construction sites of `DSv4MTPPredictor` (grep, whole repo):
- `batch_generate.py:838` — the TP path, inside `__post_init__`
- `batch_generate.py:1010` — the PP path (`_pp_mtp`), unreachable under Tensor sharding

The TP path at `batch_generate.py:838` is gated by, in order:
1. `use_speculative = os.environ.get("EXO_SPECULATIVE","0") == "1"` — `batch_generate.py:805`. **True.**
2. `is_dsv4_with_mtp` — `batch_generate.py:824-830`: `type(inner).__name__ == "DeepseekV4Model"`
   **and** `hasattr(inner, "mtp")` **and** `len(inner.mtp) > 0`.
   `inner.mtp` exists only when `config.num_nextn_predict_layers > 0 and EXO_DSV4_MTP == "1"`
   (`deepseek_v4.py:6792-6799`). **True** under MTP=1 on a checkpoint with MTP layers.

**Answer to the gating question:** the constructor runs **unconditionally at generator
construction** — it is NOT behind a DSpark-context-pool check, NOT behind a context-length check,
and NOT behind a c=1-vs-c=2 check. `__post_init__` runs once per `BatchGenerator` instance, at
model-load time, before any request exists. Notably, the comment at `batch_generate.py:856-861`
records that `__post_init__` "runs unconditionally at generator construction, before PP vs non-PP
is even decided".

So under MTP=1 + DSPARK=1 + Tensor: **`_FENCE_ASYNC_REGISTERED["cache"] = True` from load time
onward**, and `_fence_key_ok("cache")` is therefore the strict `CTX["cache"]` read for the whole
process lifetime. This is exactly the regime the 08-22 doc
(`docs/async-fence-cache-owner-dead-code-root-cause-2026-08-22.md`) said was the *dangerous* one —
the fix's fail-open escape hatch does **not** apply here.

**Corollary:** the 08-22 root cause (cache registered-and-permanently-False) is NOT the situation
today, because DSpark's presence does not bypass the MTP predictor — the DSpark head is only
loaded *because* MTP=1 makes a consumer reachable (`utils_mlx.py:422` `_tp_consumer = _spec_on and
EXO_DSV4_MTP == "1"`, gate at `utils_mlx.py:429`), and that same MTP=1 is what constructs the
predictor that arms "cache". The two are coupled in the safe direction.

---

## 4. DECODE-TIME PATH ENUMERATION

Preconditions established above and used throughout: `_FENCE_ASYNC=True`,
`_FENCE_ASYNC_MAX_B=1`, `_FENCE_ASYNC_MAX_STREAMS=1`, engine limit=1, `_pp_spec_active=False`,
`REGISTERED = {"engine": True, "cache": True}`.

Also relevant prod env (from `start_cluster.sh`): `EXO_DSV4_SPEC_STATE_RESTORE=1` (`:246`),
`EXO_DSV4_SPEC_CACHE_ROLLBACK=1` (`:247`), `EXO_DSV4_POOL_RESTORE_AFTER_TRIM=1` (`:238`),
`EXO_DSV4_MTP_TIE_REVERIFY=0` (`:321`), `EXO_DSV4_MTP_MAX_CTX=0` (`:320`),
`EXO_DSV4_MTP_C2_MAX_CTX=1` (`:1994`), `EXO_DSV4_VERIFY_ROWSEQ=1` (`:322`),
`EXO_DSV4_ROWSEQ_FULLBLOCK=1` (`:302`), `EXO_DSV4_MOE_PARTS_ROWSEQ=shared` (`:304`).
`EXO_DSV4_DRAFT_EPILOGUE`, `EXO_DSV4_TREE_DRAFT`, `EXO_DSV4_SPEC_SHADOW`,
`EXO_DSV4_DSPARK_TP_SHARD`, `EXO_DSV4_ALLSUM_PROBE`, `EXO_DSV4_FENCE_GATE_DIAG` are all
unset/default-off (`start_cluster.sh` exports them only when non-empty; `DSPARK_TP_SHARD` does not
appear in the script at all).

### 4.1 Main verify forward (the dominant decode path) — **ARMED** ✅
Dispatch: `dsv4_mtp.py:2444` `if spec_eligible:` → `2459-2460` `activate_for_uids(uids)` →
`2462` `return [], self._speculative_next(uids[0])` for `len(uids)==1`.
- `activate_for_uids` with `uids_t == self._active_uids` takes the fast path at
  `dsv4_mtp.py:1301-1305`, which calls `_set_fence_async(1 <= 1)` → **cache=True** on *every*
  cycle. This is the load-bearing steady-state re-arm.
- engine was set True by the last `_update_fence_arming` (`batch_generate.py:2785`) and nothing
  in the decode loop touches it → **engine=True**.
- Verify forward at `dsv4_mtp.py:4145-4150` (`dsv4_speculative_forward`) → 43× `DeepseekV4MoE`
  with `sharding_group` set (`auto_parallel.py:1110`) → fence site with B=1, L=gamma+1 ≤ 4.
- All five conjuncts pass → `deepseek_v4.py:3130` `mx.async_eval(y)`. **ARMED.**

### 4.2 DSpark draft forward — **NO FENCE SITE REACHED** (neither armed nor blocking)
`dsv4_mtp.py:3987` / `4663` call `_dspark.draft(...)` → `deepseek_v4.py:6725-6737` →
`DeepseekV4DSparkStage.__call__` (`deepseek_v4.py:6618-6629`) → `self.ffn(...)` at
`deepseek_v4.py:6628`, which is a `DeepseekV4MoE`.
Its `sharding_group` is assigned **only** inside the `EXO_DSV4_DSPARK_TP_SHARD=1` branch
(`auto_parallel.py:1239` gate, assignment at `auto_parallel.py:1267`; it is the sole
`sharding_group` assignment onto a DSpark stage anywhere — grep confirms no other). That env var
is **not set anywhere in `start_cluster.sh`** (grep count = 0) → default `"0"`
(`auto_parallel.py:1239`) → `_dspark` stays `None` at `auto_parallel.py:1240-1242` → **the DSpark
stages' MoE `sharding_group` is never set** → `deepseek_v4.py:3074` is False → the whole fence
block (3075-3151) is skipped.
**Consequence: the DSpark draft forward performs no MoE all_sum and no fence. The entire fence
cost per cycle belongs to the main model's verify (and any commit) forward.**

⚠ **ENV-CONDITIONAL — verify on the live nodes before relying on this.** This flag is not in the
task's stated live env list, and it is *not* part of `start_cluster.sh`'s canonical export set, but
it HAS been set manually in past campaign launches: `tmp/p01a-20260829/*_launch_node*.sh` (all four,
dated 2026-08-29) carry `EXO_DSV4_DSPARK_TP_SHARD=1`, and `bench/phase_c_cycle_ab.sh:52`
fingerprints the live process env for exactly that string. Every launch script newer than that
(`tmp/p02c-20260829` = 0, and all of `tmp/p05..p08`, `tmp/verify-decomposition-20260901`,
`tmp/hardening-round7-20260903` = unset) leaves it off.
**If the live runners were started with `DSPARK_TP_SHARD=1`, invert this section:** the DSpark
head's 3 stages each get a sharded MoE, adding 3 more fence sites per draft. Those sites would
also be **ARMED** (same B=1, and the draft block width `bs = width = gamma ≤ 3` so
`y.shape[1] ≤ 8` still holds, `deepseek_v4.py:6719,6731`), so the ARMED/BLOCKING verdict is
unchanged — only the *count* of fence sites per cycle changes (43 → 46). Confirm with
`ps eww -p <runner-pid> | tr ' ' '\n' | grep DSPARK_TP_SHARD` on each node.

### 4.3 First decode step after prefill (`_first_step_and_capture_batch`) — **ARMED** ✅
`dsv4_mtp.py:2447-2449`: any uid not in `_mtp_prefilled` routes to
`_first_step_and_capture_batch` (`dsv4_mtp.py:2472`), which calls
`super(MTPBatchGenerator, self)._next()` at `dsv4_mtp.py:2488` — an ordinary B=1, L=1 decode
forward. At this point engine=True (`batch_generate.py:2785`) and cache=True
(`dsv4_mtp.py:1269`, set at the end of `snapshot_for_uid` during `submit()`). Gate passes.
**ARMED.** Note this path does NOT call `activate_for_uids`, so it relies on `snapshot_for_uid`'s
arming — which is why the submit→snapshot ordering matters.

### 4.4 Accept path (full acceptance, `rollback == 0`) — **ARMED** ✅
`dsv4_mtp.py:4954` `rollback = gamma - n_accepted`. When 0, neither the
`if rollback > 0 and _SPEC_STATE_RESTORE` branch (`dsv4_mtp.py:4955`) nor the `elif rollback > 0`
branch (`dsv4_mtp.py:5054`) is entered, so **no `_set_fence_async(False)` runs at all**. The DSpark
ctx feed (`dsv4_mtp.py:4633-4638` `append_ctx`) touches only unsharded DSpark caches. cache stays
True through the whole cycle. **ARMED.**

### 4.5 Rollback path, cache-level exact undo (`_cache_level == True`) — **ARMED-with-a-drain**
`dsv4_mtp.py:4970-4980` computes `_cache_level`; at `dsv4_mtp.py:4982`
`self.mtp._set_fence_async(False)` → cache=False **and a full `mx.synchronize()` drain**.
The `if _cache_level:` branch (`dsv4_mtp.py:4989-5026`) runs **no model forward** —
`rollback_spec_write` (`cache.py:865-880`) and `spec_rollback` are pure cache ops. Re-arm at
`dsv4_mtp.py:5050`. So no fence site executes while cache is False.
**Verdict: no blocking fence, but a real per-rejection `mx.synchronize()` stall at
`dsv4_mtp.py:1242`.** That stall is a genuine decode-time cost of FENCE_ASYNC, not a fence
fallback. (The code documents this explicitly at `dsv4_mtp.py:4983-4987`: "`_set_fence_async(False)`
ends in `mx.synchronize()`: this bracket is the drain of everything still in flight from the
verify forward".)

### 4.6 Rollback path, commit-forward fallback (`_cache_level == False`) — **FALLBACK-BLOCKING** ❌
`dsv4_mtp.py:5027` `else:` → `5029` `restore_spec_state`, `5031-5033` `restore_meta`, then
**`dsv4_mtp.py:5035`: `_commit_logits = self.model(commit_input, cache=gen_batch.prompt_cache)`**
— a full 43-layer main-model forward, executed with `CTX["cache"] == False` (set at
`dsv4_mtp.py:4982`, not restored until `dsv4_mtp.py:5050`).
Every one of that forward's 43 MoE fence sites takes `deepseek_v4.py:3150` `mx.eval(y)`.
**This is a genuine decode-steady-state blocking window.**
`_cache_level` is False whenever any ring's `spec_pushed_rows() != _verify_len` or any pool's
`spec_can_rollback` returns False (`cache.py:1771-1789` — mixed flush attribution with a
non-single-row push). Frequency is workload-dependent (pool `ratio`, gamma vs ratio), not zero.

### 4.7 Rollback path, non-`_SPEC_STATE_RESTORE` branch — **NOT REACHED**
`dsv4_mtp.py:5054` `elif rollback > 0:` is only reachable when `_SPEC_STATE_RESTORE` is False.
`EXO_DSV4_SPEC_STATE_RESTORE=1` (`start_cluster.sh:246`), so this branch — including its own
commit-forward at `dsv4_mtp.py:5105` — is dead in prod. (Had it been live it would have been
worse: it runs the commit-forward **without** any `_set_fence_async(False)` drain at all.)

### 4.8 Tie re-verify — **NOT REACHED**
`dsv4_mtp.py:5159` `if _tie_reverify and temp == 0:`; `_tie_reverify` can only become 1 at
`dsv4_mtp.py:4573`, guarded by `dsv4_mtp.py:4566` `EXO_DSV4_MTP_TIE_REVERIFY == "1"`.
Default 0 (`start_cluster.sh:321`). Dead.

### 4.9 Prefill → decode transition — **BLOCKING BY DESIGN (correct)**
`submit()` disarms engine unconditionally at `batch_generate.py:2192` (with `mx.synchronize()`,
`batch_generate.py:2155`) before any forward for the new request; `submit_batched()` does the
same at `batch_generate.py:2825`. The MTP prefill sequence then does
`reset_cache()` (`batch_generate.py:2710` → `dsv4_mtp.py:1204` cache=False) →
`predict()` (`batch_generate.py:2721`) → `snapshot_for_uid()` (`batch_generate.py:2728` →
`dsv4_mtp.py:1265` False, `1269` True). engine is restored at `batch_generate.py:2785`.
So the whole prefill runs blocking, both keys re-arm at the end. **Correct, and out of scope of
"decode steady state".**

### 4.10 Batched-prefill boundary (`submit_batched`) — **BLOCKING BY DESIGN**
`batch_generate.py:2825` disarms engine for the whole batched-prefill; per-uid MTP prefill at
`batch_generate.py:3149` (`reset_cache`) / `3179` (`snapshot_for_uid`) toggles cache the same way;
`batch_generate.py:3234` re-arms engine per uid — but with ≥2 tasks admitted the predicate at
`batch_generate.py:2163-2164` is False, so engine stays **False** for the resulting c≥2 session.
Consistent with 4.11.

### 4.11 c=2 boundary with `FENCE_ASYNC_C2=0` — **FALLBACK-BLOCKING (by design)** ❌
All three legs disarm independently:
- engine: `batch_generate.py:2163-2164` → `1 <= 2 <= 1` is False → `_set_fence_async_engine(False)`
- cache: `dsv4_mtp.py:1304` / `1349` / `3260` all evaluate `N <= 1` → False
- shape: `deepseek_v4.py:3119` `y.shape[0]=2 <= _FENCE_ASYNC_MAX_B=1` → False

Recovery to c=1: engine re-arms at `batch_generate.py:4645` (is_done) or `5068` (remove);
cache re-arms at the next spec cycle's `activate_for_uids` (`dsv4_mtp.py:1304` if the surviving
uid set is unchanged from `_active_uids`, else `dsv4_mtp.py:1349` after the rebuild).
No stuck-False path found for the c=2→c=1 return.

### 4.12 `EXO_DSV4_MTP_C2_MAX_CTX=1` interaction — **no fence impact; c≥2 is blocking anyway**
`dsv4_mtp.py:2377-2400`: at `len(gen_batch) >= 2` with `_c2_max = 1`, any cache offset > 1 sets
`spec_eligible = False` → dispatch falls to `dsv4_mtp.py:2467` `super()._next()` (plain batched
decode). Because that path never calls `activate_for_uids`, **`CTX["cache"]` is left at whatever
value it last held and `_active_uids` goes stale**. This is benign for the fence because engine is
already False at c≥2 and `y.shape[0]=2 > MAX_B=1`.
On return to c=1 the next spec cycle calls `activate_for_uids` and re-arms (equality fast path at
`dsv4_mtp.py:1301-1305`, or the rebuild path at `1310`→`1349` if the surviving uid differs from the
stale `_active_uids`). **No stuck-False leak.**

### 4.13 Cache trim / rebuild — covered
- MTP cache trims at `dsv4_mtp.py:5043-5048` (B=1) and `3248-3255` (B=N) happen inside the
  disarmed bracket. Correct.
- Target prompt-cache trims at `dsv4_mtp.py:5074-5079` are in the dead 4.7 branch.
- `drop_uid` (`dsv4_mtp.py:1412-1420`) touches no fence key and no cache — it only pops the
  snapshot. Safe.

### 4.14 `predict()`'s implicit `reset_cache` — a latent hole, NOT live today ⚠
`dsv4_mtp.py:1467-1480`: on a B mismatch or cache-class mismatch, `predict()` sets
`self._cache = None` then calls `self.reset_cache(batch_size=B)` at `dsv4_mtp.py:1480`.
`reset_cache` disarms cache at `dsv4_mtp.py:1204` and **never re-arms** (the method has no
`_set_fence_async(True)`). If this fired mid-decode, cache would sit False until the next
`activate_for_uids` / `snapshot_for_uid`.
Under today's config it is **not reachable on the c=1 path**, because with `_dspark is not None`
(`dsv4_mtp.py:3946`) the draft goes through `_dspark.draft()` and the `self.mtp.predict(...)`
call at `dsv4_mtp.py:4057` (`draft_tokens(self.mtp, ...)`) is the `else` branch only.
`predict()` is otherwise called only from `_draft_tokens_batched` (`dsv4_mtp.py:3626`), which is
c≥2-only. **If the DSpark overlay ever fails on both ranks** (`utils_mlx.py:446-449`, rank-consistency
detach at `utils_mlx.py:452+`), the fall-back MTP-1 draft path becomes live and this hole opens.
Flagging as a conditional risk, not a current fallback.

### 4.15 `warmup_speculative` — dead code
`batch_generate.py:2044` defines it; `batch_generate.py:2062` calls `mtp.reset_cache()` (cache→False,
no re-arm). Grep finds **no callers** anywhere in `src/`. Inert.

---

## 5. `_fence_key_ok` SEMANTICS — is there ANY path where "cache" registers but sits False at decode steady state?

Semantics confirmed at `deepseek_v4.py:142-147`: unregistered ⇒ pass; registered ⇒ must be True
(fail-closed). Under today's config "cache" IS registered (§3), so every False window is a real
blocking window.

Steady-state answer: **No permanent stuck-False path exists**, because
`activate_for_uids`'s equality fast path (`dsv4_mtp.py:1301-1305`) re-arms `cache` on **every**
spec cycle, before every verify forward. That single line is what makes the current config
structurally different from the 08-22 dead-code regime — the arming is per-cycle idempotent, not
one-shot at join time.

But there are **bounded** registered-but-False windows during decode:
1. `dsv4_mtp.py:4982` → `5050` on every rejection cycle (B=1). If `_cache_level` is False, a real
   43-layer forward runs inside it (§4.6). If True, only a `mx.synchronize()` stall (§4.5).
2. `dsv4_mtp.py:3106` → `3260` on every c≥2 rejection cycle (moot — c≥2 is blocking anyway).
3. `dsv4_mtp.py:1310` → `1349` across any real BS transition.
4. `dsv4_mtp.py:1204` in `reset_cache()` with no re-arm — only reachable mid-decode via
   `predict()` (§4.14), which is dead under DSpark.

---

## 6. BOTTOM LINE

**The c=1 async fence IS armed at decode steady state under today's config.** The 08-22 dead-code
condition does not apply: `DSv4MTPPredictor.__init__` runs unconditionally at generator
construction under MTP=1 (`batch_generate.py:838`, gate at `batch_generate.py:805,824-830`),
registering "cache" (`dsv4_mtp.py:1169-1171`), and `activate_for_uids`'s no-transition fast path
(`dsv4_mtp.py:1304`) re-arms it before every single verify forward. engine is True whenever exactly
one request is active (`batch_generate.py:2163-2164`). Shape legs pass (B=1; L=gamma+1≤4).
Gate `deepseek_v4.py:3115-3121` → `deepseek_v4.py:3130` `mx.async_eval(y)`.

**Decode-time windows that DO fall back to blocking `mx.eval(y)`:**

| # | Window | Cite | Verdict |
|---|--------|------|---------|
| A | Rejection cycle, cache-level undo not applicable → full commit-forward with cache=False | disarm `dsv4_mtp.py:4982`; forward `dsv4_mtp.py:5035`; re-arm `dsv4_mtp.py:5050`; blocking at `deepseek_v4.py:3150` | **FALLBACK-BLOCKING** |
| B | Entire c≥2 session | `batch_generate.py:2163-2164`, `dsv4_mtp.py:1304/1349/3260`, `deepseek_v4.py:3119` | **FALLBACK-BLOCKING (by design, C2=0)** |
| C | Rejection cycle with cache-level undo (no forward) | `dsv4_mtp.py:4982` + `mx.synchronize()` at `dsv4_mtp.py:1242` | not a fence fallback, but a real per-rejection GPU drain |
| D | `predict()`-triggered `reset_cache` leaving cache False with no re-arm | `dsv4_mtp.py:1480` → `dsv4_mtp.py:1204` | latent; unreachable today (DSpark branch), opens if the DSpark overlay detaches |

**Two corrections to the config's stated intent:**
1. `EXO_DSV4_FENCE_EVERY_N_LAYERS=4` does nothing — `_fence_every_n` (`deepseek_v4.py:2911`)
   has no read site anywhere in the codebase. Fencing is per-layer, always.
2. The DSpark draft forward contributes zero fence sites under the canonical `start_cluster.sh`
   env, because `EXO_DSV4_DSPARK_TP_SHARD` is absent from it → the DSpark stage MoEs get no
   `sharding_group` (`auto_parallel.py:1239-1242,1267`) → `deepseek_v4.py:3074` short-circuits.
   **Caveat: past campaign launches set this flag manually (`tmp/p01a-20260829/*.sh`); if it is
   =1 on the live runners the count is 46 fence sites/cycle instead of 43, all still ARMED.
   Verify with `ps eww` on the nodes — see §4.2.**

**Open questions / assumptions stated explicitly:**
- I could not read the live checkpoint's `config.json` (`~/.exo/models/*V4*/config.json` not
  present on this laptop), so `dspark_block_size` is assumed to be the code default 5
  (`deepseek_v4.py:6650`). It does not change the L≤8 conclusion because gamma is capped by
  `EXO_SPECULATIVE_GAMMA=3` (`dsv4_mtp.py:3955`).
- The *frequency* of window A (`_cache_level == False`) is not determined by static analysis. It
  depends on `PoolingCache.spec_can_rollback` (`cache.py:1771-1789`) at runtime. Measuring it
  needs `EXO_DSV4_RB_PROFILE=1` + `EXO_DSV4_MTP_PROFILE>0` (the `rb_pool_restores` /
  `rb_commitfwd` series, `dsv4_mtp.py:5015-5024`, `5044-5049`) or `_CYCLE_STATS`'s `cache_rb`
  counter (`dsv4_mtp.py:5119-5136`).
- Any measurement of the fence taken with a `SpanProfilerHook` registered is invalid: `finalize`
  (`deepseek_v4.py:3151` → `profiler.py:109-113`) turns `mx.async_eval` into `mx.eval` one line later.
