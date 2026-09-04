# Gamma Resolution / DEDICATED / EAGLE_K Audit

Repo: `/Users/adam.durham/repos/exo`, HEAD `ccc692ff382934b82c4054908df2f6c5547b91c0` (branch main).
Cluster nodes: `macstudio-m4-1`, `macstudio-m4-2` (read-only ssh checks only).

## VERDICT (one line each)

- **Q1 GAMMA:** Base value is `self.gamma = gamma` from the constructor arg (`mtp_batch_generator.py:44`), which `batch_generate.py:835` sets to `int(os.environ.get("EXO_SPECULATIVE_GAMMA", "2"))` on the DSv4-MTP construction path — but on the **DSpark branch inside `_speculative_next`** (BS=1 only), a *local* `gamma` variable is unconditionally re-bound to `_dspark.block_size` (5) and then capped to `min(EXO_SPECULATIVE_GAMMA, block_size)` (`dsv4_mtp.py:3946-3955`); `self.gamma` itself is never mutated.
- **Q2 MISMATCH:** The `/3` denominator in the histogram is `self.gamma` (`dsv4_mtp.py:2175`), which is fixed at construction time from `EXO_SPECULATIVE_GAMMA` (or default `2`) and is **never reassigned** anywhere in the file — so a run logging `mean_accept=X/3` was constructed with `EXO_SPECULATIVE_GAMMA` resolving to `3` (or unset, defaulting via a different launcher path), **not** `2`; the `von3shard0.log` run's own construction-time INFO line reads `γ=3`, directly contradicting the `relaunch_exo_v2st.sh`(-family) script's `EXO_SPECULATIVE_GAMMA=2` — this is UNRESOLVED as "why 3 not 2" from static reading alone (see Q2 detail).
- **Q3 DEDICATED:** `EXO_DSV4_MTP_DEDICATED=1` gates a one-time weight **overlay** (`_overlay_dsv4_dedicated_mtp`, `utils_mlx.py:365`/`651`) that replaces `model.model.mtp[0]`'s weights with `mlx-community/DeepSeek-V4-Flash-MTP-bf16` (re-quantized affine/mxfp) — it does **not** touch `self.gamma`, chain depth, DSpark, or accept/verify logic, and (critically) it only affects the **classic MTP head** (`model.model.mtp[0]`), which is a *completely different draft module* from `model.model.dspark` — DSpark ignores DEDICATED entirely.
- **Q4 EAGLE_K:** LIVE, not dead — `self.mtp.eagle_k` (set from `EXO_DSV4_MTP_EAGLE_K` at `dsv4_mtp.py:1108`) is read inside `_draft_tokens_batched` (`dsv4_mtp.py:3545`), which is called from `_speculative_next_batch` (`dsv4_mtp.py:2639`), reachable whenever `len(gen_batch) >= 2` (`dsv4_mtp.py:2461`) regardless of DSpark/DEDICATED settings — start_cluster.sh's own "DORMANT" comment (line 192-199) is **wrong on the current code path** (it only reasoned about the BS=1 chained path being replaced by DSpark, missing the still-live BS≥2 batched path).
- **Q5 COST:** Cheap — `mlx-community/DeepSeek-V4-Flash-MTP-bf16` is already fully cached on **both** `macstudio-m4-1` and `macstudio-m4-2` (3.4G each, resolved symlink present), so flipping `EXO_DSV4_MTP_DEDICATED=1` on the current checkpoint requires no download/conversion and would work immediately (subject to it affecting only the classic-MTP path, per Q3 — it has zero effect on DSpark, which is what production actually drafts with today).

---

## Q1 — Gamma resolution chain (full detail)

### Constructor default and base assignment

`MTPBatchGenerator.__init__` (base class used by `DSv4MTPBatchGenerator`):

```
src/exo/worker/engines/mlx/speculative/mtp_batch_generator.py:37:        gamma: int = 2,
src/exo/worker/engines/mlx/speculative/mtp_batch_generator.py:44:        self.gamma: int = gamma
```

`self.gamma` is set exactly once, at construction, and is **never reassigned** anywhere else in `dsv4_mtp.py` or `mtp_batch_generator.py` (confirmed via `grep -n "self\.gamma\s*="` across both files — the only assignment hit is line 44 above). All other occurrences (`dsv4_mtp.py:1638, 2166, 2175, 2361, 2437, 2582, 3865, 5488`) are *reads* of `self.gamma` or local-variable copies (`gamma = self.gamma`).

### The env var that feeds the constructor

`batch_generate.py` (`ExoBatchGenerator.__post_init__`), DSv4-MTP branch:

```
src/exo/worker/engines/mlx/generator/batch_generate.py:835:                    gamma = int(os.environ.get("EXO_SPECULATIVE_GAMMA", "2"))
...
src/exo/worker/engines/mlx/generator/batch_generate.py:839:                    self._mlx_gen = DSv4MTPBatchGenerator(
src/exo/worker/engines/mlx/generator/batch_generate.py:842:                        gamma=gamma,
```

This is the **only** call site that constructs `DSv4MTPBatchGenerator` in the whole repo (`grep -rn "DSv4MTPBatchGenerator("` finds exactly this one instantiation plus the class definition). So for the DSv4 path, `self.gamma` = `int(EXO_SPECULATIVE_GAMMA)` if set (parses as int, no clamp at this site), else default `2`.

The Qwen3.5 path is a **separate, independent** env var and constructor call:

```
src/exo/worker/engines/mlx/generator/batch_generate.py:884-890:
                    # Per-model gamma for the Qwen3.5-style MTP path. Qwen3.6's
                    # dedicated head is trained with block_size=3, so it
                    # sustains a deeper draft chain than DSv4's depth-1 head.
                    # Default γ=3 here, set ONLY by EXO_QWEN_SPECULATIVE_GAMMA
                    # — independent of the DSv4 EXO_SPECULATIVE_GAMMA so the two
                    # models can run different chain depths concurrently.
                    gamma = int(os.environ.get("EXO_QWEN_SPECULATIVE_GAMMA", "3"))
```

`EXO_QWEN_SPECULATIVE_GAMMA` never touches `DSv4MTPBatchGenerator`/`self.gamma` on the DSv4 path — confirmed irrelevant to the DSv4 (production) model.

### The DSpark local re-bind (does NOT touch `self.gamma`)

Inside `_speculative_next` (the **BS=1-only** single-stream draft/verify cycle), there is a *local* variable `gamma` initialized from `self.gamma` and then conditionally overwritten when the model has a `dspark` submodule attached:

```
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:3865:        gamma = self.gamma
...
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:3913-3925:
        # Env-gated width cap (2026-08-26): the DSpark head is a 3-stage block
        # (n_stages = n_mtp_layers = 3, deepseek_v4.py) trained for width-3
        # draft/verify. block_size is 5 (anchor + 4). Previously this branch
        # unconditionally re-bound gamma to block_size, SILENTLY IGNORING
        # EXO_SPECULATIVE_GAMMA on the DSpark path — so a launch that set
        # EXO_SPECULATIVE_GAMMA=2 actually ran width-5 drafts with confidence
        # pruning. Now an explicitly-set EXO_SPECULATIVE_GAMMA (>0) CAPS gamma
        # to min(env, block_size), and width=gamma is passed to draft() so the
        # draft compute AND the verify length both truncate to the same width
        ...
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:3926:        _dspark = getattr(getattr(self.model, "model", None), "dspark", None)
...
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:3946-3955:
        if _dspark is not None:
            gamma = _dspark.block_size
            _env_gamma = os.environ.get("EXO_SPECULATIVE_GAMMA", "")
            if _env_gamma:
                try:
                    _env_gamma_int = int(_env_gamma)
                except ValueError:
                    _env_gamma_int = 0
                if _env_gamma_int > 0:
                    gamma = min(_env_gamma_int, _dspark.block_size)
```

`_dspark.block_size` for the currently-attached native head is `5` — confirmed by the load-time log line on macstudio-m4-1 (`~/exo_von3shard0.log:2656`):

```
DSpark draft head attached from ... (NATIVE checkpoint-bundled head, 115 tensors, 3 stages, block_size=5, taps=[40, 41, 42]).
```

So under DSpark at BS=1: effective drafting width = `min(EXO_SPECULATIVE_GAMMA, 5)` if `EXO_SPECULATIVE_GAMMA` is set and positive, else `5`. **This local `gamma` is used for the draft-width computation and the epilogue, but the acceptance-histogram code (`_record_acceptance`, line 2166/2175) reads `self.gamma`, which this branch never touches** — see Q2.

### Precedence order (explicit)

For the DSv4-MTP construction (which governs both the classic-MTP-only path and the DSpark path, since DSpark is dispatched from inside the same `DSv4MTPBatchGenerator._speculative_next`):

1. `self.gamma` (histogram denominator, BS≥2 batched-path draft width, `_speculative_next_batch`'s `gamma = self.gamma` at line 2582) = `int(EXO_SPECULATIVE_GAMMA)` if set at process launch, else `2`. Fixed for the generator's lifetime; **no runtime clamp against any model-config value applies to `self.gamma` itself.**
2. BS=1 **DSpark-active** draft width (a local variable, separate from `self.gamma`) = `min(int(EXO_SPECULATIVE_GAMMA), _dspark.block_size=5)` if `EXO_SPECULATIVE_GAMMA` set & >0, else `_dspark.block_size=5`.
3. BS=1 **non-DSpark** (classic MTP head only, `_dspark is None`) draft width = `self.gamma` unmodified (line 3865, never re-bound because the `if _dspark is not None:` block at 3946 is skipped).
4. `EXO_QWEN_SPECULATIVE_GAMMA` is entirely independent — governs only the Qwen3.5-style `MTPBatchGenerator` construction (`batch_generate.py:890/899`), never reachable for a DSv4 checkpoint.

There is **no model-config-derived clamp** (no `num_nextn_predict_layers`/`block_size`-from-checkpoint-JSON path) feeding `self.gamma` directly — the only checkpoint-derived quantity involved is `_dspark.block_size`, and it only participates in the **local** BS=1 draft-width variable, never in `self.gamma`.

---

## Q2 — The mismatch: is `/3` = `self.gamma`, and how does GAMMA=2 produce `/3`?

**Denominator identity (definitive):** the f-string at the emit site is

```
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:2173-2177:
            logger.warning(
                f"[MTP] cycles={self._spec_cycles} "
                f"mean_accept={mean:.3f}/{self.gamma} "
                f"hist={hist}"
            )
```

The denominator is unambiguously `self.gamma` — a hardcoded attribute read, not the local DSpark-branch `gamma` variable (that local variable is scoped inside `_speculative_next`/`_draft_tokens_batched` and is not accessible here; `_record_acceptance` only ever sees `self.gamma`). This also matches Q1: `self.gamma` is fixed at construction and can only be `3` if `EXO_SPECULATIVE_GAMMA` resolved to `3` (or was unset with some other default — but the base-class default is `2`, not `3`, so unset does not explain `/3` either).

**Is there a clamp that raises 2→3?** No. Grep for every mutation of `self.gamma` (`grep -n "self\.gamma\s*="` across `dsv4_mtp.py` and `mtp_batch_generator.py`) returns exactly one hit: `mtp_batch_generator.py:44` (`self.gamma: int = gamma`), the constructor assignment. There is no post-construction clamp, min/max, or model-config override applied to `self.gamma` anywhere in the file set that implements `DSv4MTPBatchGenerator`.

**Runtime evidence from the actual log (not the launch script) settles this differently than the working hypothesis:** the `von3shard0.log` run's own construction-time log line reads:

```
macstudio-m4-1:~/exo_von3shard0.log:4851:
[ 2026-08-29 01:12:48.773 | INFO | exo.worker.engines.mlx.generator.batch_generate:__post_init__:857 ] DSv4 MTP speculative decoding enabled (γ=3, T=0.0)
```

This INFO line is emitted immediately after `gamma = int(os.environ.get("EXO_SPECULATIVE_GAMMA", "2"))` and the `DSv4MTPBatchGenerator(...)` construction (`batch_generate.py:835-850`), i.e. it reports the **actual resolved value passed into `self.gamma`** for that process. It says `γ=3`, not `γ=2`.

This directly means: **for the specific process that produced this log file, `EXO_SPECULATIVE_GAMMA` resolved to `3` at process-launch time, not `2`.** The `relaunch_exo_v2st.sh` (and its near-identical siblings `relaunch_exo_refcheck.sh`, `relaunch_exo_final.sh`, etc. — all still present on the node and all export `EXO_SPECULATIVE_GAMMA=2 ... EXO_DSV4_MTP_DEDICATED=1 ... EXO_DSV4_MTP_EAGLE_K=8`) does NOT match this construction-time value.

**UNRESOLVED — exactly which launcher (env) actually produced `exo_von3shard0.log`.** I could not find, via read-only inspection of `~/*.sh`, `~/.zsh_history` (empty/inaccessible for the relevant window), or the log itself (no argv/env dump line was found in the log — `EXO_SPECULATIVE_GAMMA=` never appears verbatim in the log text), a script on the node whose exported `EXO_SPECULATIVE_GAMMA` is `3` (or unset) *and* `EXO_DSV4_MTP_DEDICATED=1` *and* `EXO_DSV4_DSPARK=1` simultaneously — the scripts matching the "GAMMA=2 + DEDICATED=1" combination given in the task's background (`relaunch_exo_v2st.sh` and its 11 siblings) all say GAMMA=2, which **contradicts** the log's own `γ=3` self-report. Either (a) the log was produced by a *different*, no-longer-present launcher (e.g. `~/relaunch_exo.sh` was overwritten since — its current content on disk, dated Sep 3, shows GAMMA=3 but DEDICATED=0, i.e. today's production config, not the Aug 29 DSpark+DEDICATED=1 config), or (b) the shell process's actual exported env at fork time differed from any script currently readable on disk (e.g. edited inline before running, or a since-deleted script).

**What runtime observation would disambiguate:** re-run (do not do this now — read-only audit) a DSpark+DEDICATED=1 launch with `EXO_DSV4_MTP_LOG_INTERVAL` set low and immediately grep the fresh log for the `DSv4 MTP speculative decoding enabled (γ=N...)` INFO line (emitted once, at `batch_generate.py:850`, right after construction) — that line is authoritative for `self.gamma` (and hence the histogram denominator) for that exact process, sidestepping any ambiguity about which script/env was actually exported. Given the evidence in hand, the working hypothesis "EXO_SPECULATIVE_GAMMA=2 did not govern the MTP path's chain depth in that run" is **plausible but not proven** — the more direct reading of the log's own self-reported `γ=3` is simply that the effective `EXO_SPECULATIVE_GAMMA` at launch was `3`, full stop; no code-level mechanism was found that would silently promote a genuinely-exported `GAMMA=2` to `self.gamma=3`.

---

## Q3 — DEDICATED (load-bearing question)

### Read site

```
src/exo/worker/engines/mlx/utils_mlx.py:355-368:
    # Optionally overlay the DEDICATED mlx-community DSv4 MTP head onto the
    # native (checkpoint-bundled) mtp[0], BEFORE tensor sharding. The dedicated
    # head (mlx-community/DeepSeek-V4-Flash-MTP-bf16) is the same trained MTP
    # weights re-packaged (fused switch_mlp, decoder.* prefix, affine-8bit).
    # We overlay here while the module is still unsharded so the subsequent
    # tensor_auto_parallel shards it identically to the native head. Gated by
    # EXO_DSV4_MTP_DEDICATED=1 so the proven native path stays the default.
    if (
        os.environ.get("EXO_DSV4_MTP", "0") == "1"
        and os.environ.get("EXO_DSV4_MTP_DEDICATED", "0") == "1"
    ):
        try:
            _overlay_dsv4_dedicated_mtp(model, model_path)
        except Exception as e:
            logger.warning(
                f"DSv4 dedicated MTP overlay failed ({e}); keeping native MTP head."
            )
```

This is the **only** read site of `EXO_DSV4_MTP_DEDICATED` in Python code (`grep -rn "EXO_DSV4_MTP_DEDICATED" --include="*.py" .` returns exactly this occurrence plus a docstring reference at line 359 and a `fingerprint.py` registry entry that just labels it for provenance tracking — not a functional consumer).

### What it switches

`_overlay_dsv4_dedicated_mtp` (`utils_mlx.py:651-768`) downloads `mlx-community/DeepSeek-V4-Flash-MTP-bf16`'s `model.safetensors`, strips the `decoder.` prefix, re-quantizes to match the head's on-disk packing, and calls `mtp0.load_weights(...)` where `mtp0 = model.model.mtp[0]`:

```
src/exo/worker/engines/mlx/utils_mlx.py:670-677:
    inner = getattr(model, "model", None)
    mtp_list = getattr(inner, "mtp", None) if inner is not None else None
    if not mtp_list:
        raise RuntimeError("model has no model.mtp[] to overlay")
    mtp0 = mtp_list[0]

    repo = "mlx-community/DeepSeek-V4-Flash-MTP-bf16"
    sf = hf_hub_download(repo, "model.safetensors")
```
```
src/exo/worker/engines/mlx/utils_mlx.py:763:
    mtp0.load_weights(list(remap.items()), strict=False)
```

**This is `model.model.mtp[0]` — the classic (single-stage, checkpoint-bundled) MTP head, consumed by `DSv4MTPPredictor` (`dsv4_mtp.py:1037-1071`, `inner.mtp[mtp_idx]`).** It is a strictly different module from `model.model.dspark` (`DeepseekV4DSparkModule`, attached separately by `_overlay_dsv4_dspark`/`_overlay_dsv4_dspark_native`, gated by `EXO_DSV4_DSPARK`, `utils_mlx.py:419-470` region). `DEDICATED` has **zero read sites and zero code paths touching `dspark`** — confirmed by `grep -n "EXO_DSV4_MTP_DEDICATED"` across the whole repo showing only the `utils_mlx.py` gate and its own docstring/registry mentions, none inside the DSpark overlay functions.

### Does DEDICATED change chain depth / MTP-layer count / accept-verify logic?

No. `EXO_DSV4_MTP_DEDICATED` does not appear anywhere in `dsv4_mtp.py` (confirmed: zero matches for `EXO_DSV4_MTP_DEDICATED` in that file). `self.gamma`, the DSpark `block_size`/local `gamma` re-bind, and the acceptance-histogram/accept-verify code (`_speculative_next`, `_speculative_next_batch`, `_record_acceptance`) are entirely unaware of this flag. It only affects **which weights sit inside `mtp[0]`** at load time, before generation even begins.

### Practical consequence for the Aug 29 run's 85.4% acceptance

Since the Aug 29 run used `EXO_DSV4_DSPARK=1` (native head, per the load-time log line quoted in Q1) **and** the DSpark draft branch in `_speculative_next` is what actually ran (the `if _dspark is not None:` block, `dsv4_mtp.py:3946`), the classic `mtp[0]` head (which is what `DEDICATED=1` would have overlaid) was **not the module doing the drafting** in that run at all — DSpark's own native 3-stage head was. So even if `EXO_DSV4_MTP_DEDICATED=1` was genuinely exported for that run, it is very unlikely to be the mechanism behind the 85.4% acceptance, because DEDICATED's target module (`mtp[0]`) is not the module DSpark uses to draft. **This directly weakens the task's working hypothesis** ("DEDICATED=1 selects a dedicated/trained draft head with much better acceptance") — DEDICATED cannot explain a DSpark-path acceptance number, because DEDICATED's overlay target is orthogonal to DSpark's draft head. The much more likely explanatory variable for the 85.4%-vs-47% gap, per this code reading, is **DSpark itself being active** (native 3-stage head, `block_size=5`, width capped to `min(GAMMA,5)`) versus classic single-stage MTP chaining — not the DEDICATED flag. This is a code-only inference; **UNRESOLVED** whether DSpark alone (independent of DEDICATED) explains the acceptance gap — the disambiguating experiment is a same-day, same-checkpoint A/B: `EXO_DSV4_DSPARK=1 EXO_DSV4_MTP_DEDICATED=0` vs `EXO_DSV4_DSPARK=1 EXO_DSV4_MTP_DEDICATED=1` at matched `EXO_SPECULATIVE_GAMMA`, comparing histograms — held out of scope here per the read-only/no-relaunch constraint.

---

## Q4 — EAGLE_K: live or dead?

### Read sites

```
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:1108:
        self.eagle_k = int(os.environ.get("EXO_DSV4_MTP_EAGLE_K", "0"))
```

This is on `DSv4MTPPredictor.__init__` (the classic-MTP-head wrapper, instantiated once per generator at `batch_generate.py:838`: `mtp_pred = DSv4MTPPredictor(self.model, mtp_idx=0)`).

### Consumers

```
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:3545:
        _eagle_k = int(getattr(self.mtp, "eagle_k", 0) or 0)
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:3546:
        _eagle_embed = getattr(self.mtp, "embed_tokens", None) if _eagle_k > 0 else None
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:3547:
        _eagle_active = _eagle_k > 0 and _eagle_embed is not None
```

These three lines are inside `_draft_tokens_batched`, whose def is:

```
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:3464:
def _draft_tokens_batched(
```

`_draft_tokens_batched` is called from exactly one place:

```
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:2639-2641:
        draft_ids_list, draft_probs_list = self._draft_tokens_batched(
            stacked_pre_norm, next_tokens_arr, gamma, _tvec, all_greedy
        )
```

— inside `_speculative_next_batch`, the **BS≥2 batched spec-decode path** (`dsv4_mtp.py:2567: def _speculative_next_batch`). This is reachable from `_next`'s dispatch logic:

```
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:2452-2461:
        if spec_eligible:
            ...
            if len(uids) == 1:
                return [], self._speculative_next(uids[0])
            return [], self._speculative_next_batch(uids)
```

i.e. `_speculative_next_batch`/`_draft_tokens_batched`/EAGLE_K fires whenever `spec_eligible` is true (requires `self.gamma > 0`, no pending prompts/unprocessed sequences — `dsv4_mtp.py:2361-2366`) AND there are `≥2` concurrent uids in the generation batch (`len(gen_batch) >= 2` at line 2461, mirrored by `spec_eligible and len(gen_batch) >= 2` at line 2461's guard region). This is a normal, expected runtime condition (any time ≥2 requests are decoding concurrently), not a corner case.

### Reachability under today's production config (`EXO_DSV4_DSPARK=1`, `EXO_DSV4_MTP_DEDICATED=0`)

`_speculative_next_batch`/`_draft_tokens_batched` contains **no DSpark branch at all** — `grep -n "_dspark"` restricted to lines 2560-3463 (the full body of `_speculative_next_batch` and everything up to `_draft_tokens_batched`'s definition) returns zero hits. DSpark's `_dspark is not None` branch only exists inside `_speculative_next` (the BS=1 path, `dsv4_mtp.py:3946`), which is a **separate function** from `_speculative_next_batch`. This means: **at BS≥2, DSpark is never consulted at all — the batched path always uses the classic chained-MTP `_draft_tokens_batched`/`self.mtp.predict()` route, which is exactly where EAGLE_K is read.** So EAGLE_K is reachable under `EXO_DSV4_DSPARK=1 EXO_DSV4_MTP_DEDICATED=0` whenever the batch has ≥2 concurrent streams — this is **not gated out** by DSpark being enabled, because DSpark only intercepts the BS=1 code path.

(At BS=1 with DSpark active, the DSpark branch runs instead, and EAGLE_K's `_draft_tokens_batched` is not reached for that cycle — so EAGLE_K's *effective* reach is BS≥2 only, regardless of DSpark. It is not dead, but it is conditionally dormant per-cycle depending on concurrency, not per-cluster-config.)

### start_cluster.sh's own claim

```
start_cluster.sh:192-199:
# DORMANT IN PRODUCTION (confirmed 2026-08-22, see
# docs/roofline-sanity-check-inputs-confirmed-2026-08-22.md): this tunes
# the classic single-MTP-head chained draft path, which is separate from
# DSpark (see EXO_DSV4_DSPARK above). Live production config runs
# EXO_DSV4_MTP=0 (classic MTP disabled -- DSpark was meant to fully
# replace it, per the comment above EXO_DSV4_DSPARK), so this K value has
# no effect on current decode. Left at its tuned default in case classic
# MTP is ever re-enabled as a DSpark fallback.
: "${EXO_DSV4_MTP_EAGLE_K:=8}"
```

**The comment's own premise is stale and does not match today's `start_cluster.sh` defaults.** The comment asserts "Live production config runs `EXO_DSV4_MTP=0`" — but `start_cluster.sh`'s own default, a few hundred lines later, is:

```
start_cluster.sh:496:
  : "${EXO_DSV4_MTP:=1}"
```

and the currently-active production launcher on macstudio-m4-1 (`~/relaunch_exo.sh`, dated Sep 3, exports `EXO_DSV4_MTP=1 EXO_DSV4_DSPARK=1 ... EXO_DSV4_MTP_DEDICATED=0 ... EXO_DSV4_MTP_EAGLE_K=8`) confirms `EXO_DSV4_MTP=1` is what's actually running, not `0`. With `EXO_DSV4_MTP=1`, `is_dsv4_with_mtp` is true (`batch_generate.py:822-827`), `DSv4MTPBatchGenerator` is constructed, `DSv4MTPPredictor` is built with `self.eagle_k` from `EXO_DSV4_MTP_EAGLE_K`, and — per the BS≥2 reachability analysis above — `_draft_tokens_batched` and its EAGLE_K logic run any time ≥2 streams are concurrently decoding, DSpark or not.

**Verdict: NOT a dead knob, and the start_cluster.sh comment's stated reason (`EXO_DSV4_MTP=0` in production) is factually false on today's code/config** — `EXO_DSV4_MTP=1` is both the script's own default and the value in the actual production launcher. EAGLE_K is live at BS≥2 under exactly the production config given (`EXO_DSV4_DSPARK=1 EXO_DSV4_MTP_DEDICATED=0`). This is a **different class of finding** from `EXO_DSV4_FENCE_EVERY_N_LAYERS` (an assigned-but-never-read var) — EAGLE_K genuinely has a live, reachable consumer; the codebase's own inline documentation about it is simply outdated/incorrect.

---

## Q5 — Cost of flipping DEDICATED

### Load path / existence requirement

`_overlay_dsv4_dedicated_mtp` uses `huggingface_hub.hf_hub_download(repo, "model.safetensors")` (`utils_mlx.py:668-677`) — this is a **content-addressed HF cache lookup**, not a `EXO_DSV4_DSPARK_DIR`-style local-path requirement. If the file is already in the local HF cache, `hf_hub_download` returns immediately without any network I/O; if absent, it downloads.

### Error/fallback if weights are missing

```
src/exo/worker/engines/mlx/utils_mlx.py:363-368:
        try:
            _overlay_dsv4_dedicated_mtp(model, model_path)
        except Exception as e:
            logger.warning(
                f"DSv4 dedicated MTP overlay failed ({e}); keeping native MTP head."
            )
```

A missing/failed download does **not** crash the run — it logs a warning and silently keeps the native (checkpoint-bundled) `mtp[0]` head. So even in the worst case, flipping the flag is non-destructive.

### Convert script

`scripts/convert_dsv4_mtp.sh` exists in the repo (confirmed via file search: `./scripts/convert_dsv4_mtp.sh`), consistent with the task's claim — not inspected further as Q5 only requires confirming the artifact's *presence*, which was verified directly (below), making the convert script moot for this specific flip.

### Cluster artifact check (read-only, both nodes)

```
macstudio-m4-1:
$ find ~ -maxdepth 6 -iname '*DeepSeek-V4-Flash-MTP*'
/Users/adam.durham/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-MTP-bf16

$ du -sh ~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-MTP-bf16
3.4G	/Users/adam.durham/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-MTP-bf16

$ find .../models--mlx-community--DeepSeek-V4-Flash-MTP-bf16 -name '*.safetensors'
.../snapshots/0171b72cf11b5bba695aef01b0de237bef0a6640/model.safetensors -> ../../blobs/fa0216c37d82724918730893df28f85997dedef340db263bf5a8b5cf8e3dae43

macstudio-m4-2:  (identical results)
3.4G	.../models--mlx-community--DeepSeek-V4-Flash-MTP-bf16
.../snapshots/0171b72cf11b5bba695aef01b0de237bef0a6640/model.safetensors -> ../../blobs/fa0216c37d82724918730893df28f85997dedef340db263bf5a8b5cf8e3dae43
```

Both nodes have the **identical resolved snapshot** (same commit hash `0171b72cf11b5bba695aef01b0de237bef0a6640`, same blob), and the symlink resolves (not a broken/partial download marker) — `hf_hub_download` for this repo will be a cache hit with zero network I/O on both nodes.

### Verdict

**Cheap experiment.** Flipping `EXO_DSV4_MTP_DEDICATED=1` on the current checkpoint requires no download, no conversion step, and has a safe try/except fallback if anything goes wrong. However — per Q3 — this flag only affects `model.model.mtp[0]` (the classic MTP head), which is **not the module DSpark drafts with** under today's production config (`EXO_DSV4_DSPARK=1`). At BS=1 with DSpark active, flipping DEDICATED would have **no observable effect on the drafting path actually in use** (DSpark's own native head is separate and unaffected). It would only matter for: (a) BS=1 runs with `EXO_DSV4_DSPARK=0` (classic-MTP-only drafting), or (b) indirectly, nothing at BS≥2 either, since `_speculative_next_batch`/`_draft_tokens_batched` also just calls `self.mtp.predict()` against whatever weights sit in `mtp[0]` — so DEDICATED *would* matter for the BS≥2 batched path too, independent of DSpark (DSpark has no BS≥2 branch, confirmed in Q4). Net: DEDICATED is cheap to flip, but its effect is confined to the classic-MTP head's drafting quality, not DSpark's.
