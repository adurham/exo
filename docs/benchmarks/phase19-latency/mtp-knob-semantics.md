# DSv4.1 MTP / speculative-decode env knobs — semantics and liveness

Scope: read-only source trace of the `EXO_DSV4_*` / `EXO_SPECULATIVE_*` speculative-decode
knobs named in the campaign brief, plus the draft/verify call sites and the sampling /
`repetition_penalty` handling. Every claim carries `file:line`.

Repo under test: exo fork `/Users/adam.durham/repos/exo`, branch `deploy/next13`
@ `f0840af1c` (working tree clean except an untracked `tmp/` dir). mlx-lm submodule
`/Users/adam.durham/repos/exo/mlx-lm`.

---

## 0. THE DECISIVE FINDING — these knobs are on a DIFFERENT engine than the one deployed

The deployed checkpoint is `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`, and the
brief itself states `engine='dsv41'`. That is a **card-level** dispatch that bypasses the
entire `dsv4_mtp.py` speculative stack:

- The model card sets `engine = "dsv41"` explicitly:
  `resources/inference_model_cards/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw.toml:56`
  (added by `e1695d803`, 2026-10-01; the launcher default `DSV4_MODEL_ID` was switched to this
  checkpoint the same day, `start_cluster.sh:439`, commit `09f2097eb`).
- Dispatch reads that field and returns a `Dsv41Builder`, NOT `MlxBuilder`:
  `src/exo/worker/engines/mlx/dsv41/dispatch.py:27-29` (predicate), `:51-58` (builder choice),
  called from `src/exo/worker/runner/bootstrap.py:332-339`.
- `Dsv41Builder.build` constructs `Dsv41Engine`
  (`src/exo/worker/engines/mlx/dsv41/builder.py:89-108`), which serves one request at a time
  through `dsv41/rounds.py::_one_round` and **never imports `batch_generate.py`,
  `ExoBatchGenerator`, `KVPrefixCache`, `tensor_auto_parallel`, or `dsv4_mtp.py`**
  (`dsv41/engine.py:7-30` documents exactly this; grep for `dsv4_mtp` in `dsv41/` returns
  nothing).
- Production's own phase-22 capture confirms the served path:
  `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/README.md:33` — *"The served model's
  prefill is exo's `dsv41/session.py::engine_prefill`"*.

**Consequence.** Every knob in the brief's levers list that is named `EXO_DSV4_MTP_*`,
`EXO_DSV4_BS_MIN_ACCEPT`, `EXO_DSV4_SPEC_*`, `EXO_DSV4_MTP_C2_MAX_CTX`, `EXO_DSV4_MTP_MAX_CTX`,
`EXO_DSV4_MTP_DEDICATED`, `EXO_DSV4_MTP_ACCEPT_LOGPROBS`, `EXO_DSV4_MTP_TIEBREAK_*`, and
`EXO_DSV4_MTP_EAGLE_K` is read **only** by the legacy `dsv4_mtp.py` path (the
`mlx_lm.models.deepseek_v4.DeepseekV4Model` engine). It is not read by the live `dsv41`
engine. A grep of the whole tree confirms each symbol is defined/consumed in `dsv4_mtp.py`
(and its mlx-lm companions `deepseek_v4.py`, `cache.py`) and nowhere in `dsv41/`:

```
EXO_DSV4_MTP_ACCEPT_LOGPROBS   -> dsv4_mtp.py
EXO_DSV4_MTP_TIEBREAK_FIX/EPS  -> dsv4_mtp.py
EXO_DSV4_MTP_EAGLE_K           -> dsv4_mtp.py, mtp_module.py, mlx-lm/deepseek_v4.py
EXO_DSV4_MTP_C2_MAX_CTX        -> dsv4_mtp.py
EXO_DSV4_MTP_MAX_CTX           -> dsv4_mtp.py
EXO_DSV4_DSPARK (+NATIVE/TP_SHARD) -> dsv4_mtp.py, utils_mlx.py, auto_parallel.py, ...
EXO_SPECULATIVE_GAMMA          -> batch_generate.py, dsv4_mtp.py
EXO_DSV4_MTP_DEDICATED         -> utils_mlx.py
EXO_DSV4_SPEC_CACHE_ROLLBACK   -> dsv4_mtp.py, mlx-lm/cache.py
EXO_DSV4_SPEC_STATE_RESTORE    -> dsv4_mtp.py
EXO_DSV4_BS_MIN_ACCEPT         -> dsv4_mtp.py
EXO_DSV4_MTP_TIE_REVERIFY      -> dsv4_mtp.py
```

`EXO_SPECULATIVE_GAMMA` is the single exception that touches both engines' *files*, but even
there the value reaches only the legacy generator (`batch_generate.py:845`) and the DSpark
branch of `dsv4_mtp.py:3969`; the live `dsv41` engine takes gamma from its own dataclass
default and adaptive policy (`dsv41/engine.py:280-281, 983-986`), never from the env.

**What the live `dsv41` engine's spec loop actually is.** A single-round DSpark draft + chunk
verify driven by `mlx_lm.models.deepseek_v41.spec` / `.mtp` / `.sampling`, with its own knob
namespace `EXO_DSV41_*` and `DSV41_*` (e.g. `EXO_DSV41_SPECULATIVE`,
`dsv41/builder.py:43-47`; `EXO_DSV41_PARK`, `dsv41/session.py:77`; the `DSV41_*` family in
`mlx_lm/models/deepseek_v41/{spec,prefill,mtp,exo_*}.py`). None of those are in the brief's
list.

The per-knob sections below therefore give **both** readings: (A) what the code does on the
legacy `dsv4_mtp.py` path the knob belongs to, and (B) whether that path is reachable for the
deployed `dsv41` checkpoint. Sections 11-14 (call sites, sampling, `repetition_penalty`) cover
the two engines explicitly.

Config assumed (the brief's levers, confirmed against
`docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/production-launch-env-m4-{1,2}.txt`):
`EXO_DSV4_MTP=1`, `EXO_DSV4_DSPARK=1 (+NATIVE, +TP_SHARD)`, `EXO_SPECULATIVE_GAMMA=3`,
`EXO_DSV4_MTP_EAGLE_K=8`, `EXO_DSV4_MTP_ACCEPT_LOGPROBS=1`, `EXO_DSV4_MTP_TIEBREAK_FIX=0`,
`EXO_DSV4_MTP_C2_MAX_CTX=1`, `EXO_DSV4_MTP_MAX_CTX=0`, `EXO_DSV4_SPEC_CACHE_ROLLBACK=1`,
`EXO_DSV4_SPEC_STATE_RESTORE=1`, `EXO_DSV4_MTP_DEDICATED=0`, `EXO_DSV4_BS_MIN_ACCEPT=1`,
`EXO_DSV4_MTP_TIE_REVERIFY=0`.

---

## 1. EXO_DSV4_MTP_ACCEPT_LOGPROBS  (value: `1`)

- **(a) read site.** `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:340`
  (`_ACCEPT_LOGPROBS = os.environ.get("EXO_DSV4_MTP_ACCEPT_LOGPROBS", "0") == "1"`).
  Consumed at `dsv4_mtp.py:4243-4247` (B=1 temp=0 accept/bonus argmax), `:2733-2735` (B>1),
  `:5595-5598` (tree path).
- **(b) import vs per-call.** **Module import** (line 340 is at column 0, module scope). Toggling
  requires a runner restart.
- **(c) default when unset.** `"0"` → OFF (raw-logits argmax).
- **(d) mechanism.** At temp=0 the legacy verify computes `logprobs_all = verify_logits[0] -
  logsumexp(...)` (bf16). With the knob ON, both the accepted-draft argmax and the bonus argmax
  are taken over that *normalized* tensor (`:4246` `target_tokens = argmax(logprobs_all[:gamma])`,
  `:4247` `all_next = argmax(logprobs_all)`) instead of over raw `verify_logits` (`:4249`). This
  makes MTP-on token selection rule-identical to the MTP-off generator, whose `_step` also
  argmaxes over logsumexp-normalized logits in bf16 — closing the documented first-index argmax
  flip on near-tied logits (`:321-340`).
- **(e) live?** **No.** The knob is read only inside the `DeepseekV4Model` MTP generator; the
  deployed checkpoint routes to `Dsv41Engine` (`dispatch.py:51-58`), which never loads
  `dsv4_mtp.py`. Dormant.
- **(f) risk class.** **Changes output distribution / numerics** — it selects a different argmax
  rule, so it can change emitted tokens (byte-identity gate class). On the live engine it is
  inert.

## 2. EXO_DSV4_MTP_TIEBREAK_FIX (value `0`) and EXO_DSV4_MTP_TIEBREAK_EPS (value `0.5`)

- **(a) read sites.** `TIEBREAK_FIX`: `dsv4_mtp.py:4275` (inside the temp==0 branch).
  `TIEBREAK_EPS`: `dsv4_mtp.py:4276`.
- **(b) import vs per-call.** **Per call** — both are `os.environ.get(...)` at the accept step,
  so they can be toggled without a restart (unlike the module-level flags). The default string is
  `"1"` (`:4275`), i.e. the code default is ON.
- **(c) default when unset.** `TIEBREAK_FIX` default `"1"` (ON); `TIEBREAK_EPS` default `"0.5"`.
  (Note the code default is the opposite of the deployed value `0`.)
- **(d) mechanism.** At temp=0, among tokens within `eps` logits of the per-position max it picks
  the **lowest token id** via a masked `argmin` over an id array (`:4277-4287`), so a ~1-ulp
  batched-vs-sequential tie resolves identically on both and cannot cascade the generation onto a
  degenerate trajectory. It is applied to `all_next` (the **bonus** selection) only, never to
  accepted drafts (`:4268-4270`), because drafts are already in the KV cache. `TIEBREAK_EPS`
  is the tie window; a larger eps masks more near-ties toward the lowest id.
- **(e) live?** **No** — same reason as §1 (legacy `dsv4_mtp.py` path only). Dormant. (The
  header comment at `:337-339` records that this mechanism is superseded by
  `EXO_DSV4_MTP_ACCEPT_LOGPROBS` even within the legacy path.)
- **(f) risk class.** **Changes output distribution / numerics** (a deterministic token-selection
  rule). Inert on the live engine.

## 3. EXO_DSV4_MTP_EAGLE_K (value `8`)

- **(a) read sites.** Predictor construction: `dsv4_mtp.py:1108`
  (`self.eagle_k = int(os.environ.get("EXO_DSV4_MTP_EAGLE_K", "0"))`).
  Consumed per draft call in `mtp_module.py:789` (`_eagle_k = getattr(mtp_pred, "eagle_k", 0)`) and
  per batched draft step in `dsv4_mtp.py:3566`.
- **(b) import vs per-call.** **Instance construction** (once per `DSv4MTPPredictor`, i.e. once per
  generator build in `batch_generate.py:848`); consumed per call from the stored attribute.
- **(c) default when unset.** `"0"` → OFF (hard-argmax embedding for every chained draft step).
- **(d) mechanism.** With K>0, every chained MTP `predict()` beyond the first replaces its input
  embedding with a probability-weighted mixture of the previous step's top-K vocab embeddings
  (`mtp_module.py:781-871`; c=2 variant `dsv4_mtp.py:3560-3636`). The mixture is built from the
  previous step's logits; `EXO_DSV4_MTP_EAGLE_T` (default 1.0) sharpens it. Purpose: raise step-1
  P(top-1) so the γ≥2 chained drafts accept more often.
- **(e) live? Is the EAGLE/tree path reachable with DSPARK on?** **No — dormant by construction.**
  Two independent proofs:
  1. The knob only affects the **chained-MTP draft** (`draft_tokens` / `_draft_tokens_batched`),
     which is the `else` branch of the DSpark gate. In the B=1 cycle the draft is either the
     DSpark block draft (`dsv4_mtp.py:4008-4016`, taken when `_dspark is not None`, set at
     `:3947`) or the chained MTP path (`:4078-4086`, the `else`). The DSpark draft is "one
     parallel block forward + Markov sequential sampling" (`:3928-3932`) and never reads
     `eagle_k`. With `EXO_DSV4_DSPARK=1` + a native head attached (`utils_mlx.py:461-468`), the
     `_dspark is not None` branch always wins, so the EAGLE chain never runs.
  2. Even the file owner of the knob (`mlx_lm/models/deepseek_v4.py:683-686`) only **honors** a
     side-channel soft-emb (`_EAGLE_CTX`) if a caller populates it; the DSpark path populates
     `_DSPARK_CTX` (`deepseek_v4.py:694-705`, `set_dspark_taps`), not `_EAGLE_CTX`.
  Additionally the whole engine is the legacy one, so it is dormant for the deployed checkpoint
  even before the DSpark gate.
- **(f) risk class.** **Changes output distribution** (it changes the *draft proposal*
  distribution → changes acceptance and can change committed tokens). Inert on the live engine.

## 4. EXO_DSV4_MTP_C2_MAX_CTX (value `1`) and EXO_DSV4_MTP_MAX_CTX (value `0`)

- **(a) read sites.** `C2_MAX_CTX`: `dsv4_mtp.py:2399` (inside the `len(gen_batch) >= 2` branch of
  the `_next` dispatch). `MAX_CTX`: `dsv4_mtp.py:2435` (guarded by `if spec_eligible:`).
- **(b) import vs per-call.** **Per call** — both are `os.environ.get(...)` on every `_next`
  dispatch, so a restart is not required (launcher changes only).
- **(c) defaults.** `C2_MAX_CTX` default `"0"`; `MAX_CTX` default `"0"`. For both, `0` means "no
  gate".
- **(d) mechanism.**
  - `C2_MAX_CTX`: at BS≥2, if the value is nonzero it is used as a **context-length threshold**;
    the code scans each cache's `offset` and sets `spec_eligible = False` once the max offset
    exceeds the threshold (`:2406-2421`). **A value of `1` therefore disables spec at c≥2 for
    every real generation** (any context > 1), which is exactly the deployed intent — the header
    at `:2366-2381` explicitly corrects the old "gate removed" comment and says `=1` is live and
    disabling c≥2 spec. `0` would *arm* c≥2 speculation.
  - `MAX_CTX`: a context cap at **any** concurrency; with a positive value it sets
    `spec_eligible = False` once the max cache offset exceeds it (`:2436-2452`). `=0` disables the
    cap.
- **(e) live?** **No** — legacy `dsv4_mtp.py` path only. Dormant for the deployed `dsv41` engine.
  (Within the legacy path they *would* be live and `C2_MAX_CTX=1` would force spec-off at c≥2.)
- **(f) risk class.** **Pure scheduling / gating.** They only choose whether the spec cycle runs;
  they do not alter the token-selection rule. Inert on the live engine.

## 5. EXO_DSV4_DSPARK (+ EXO_DSV4_DSPARK_NATIVE, EXO_DSV4_DSPARK_TP_SHARD)

- **(a) read sites.**
  - Head-load gate: `src/exo/worker/engines/mlx/utils_mlx.py:440`
    (`_dspark_env_on = os.environ.get("EXO_DSV4_DSPARK", "0") == "1"`), consumed at
    `:450-473` (skip/attach decision).
  - Native-vs-local selection: `utils_mlx.py:463`
    (`_use_native = os.environ.get("EXO_DSV4_DSPARK_NATIVE", "0") == "1"`), used at `:465-468`.
  - TP-shard of the head: `src/exo/worker/engines/mlx/auto_parallel.py:1239`
    (`_dspark_tp_shard = os.environ.get("EXO_DSV4_DSPARK_TP_SHARD", "0") == "1"`), used at
    `:1240-1243`.
  - Runtime consumer: the side channel `mlx_lm/models/deepseek_v4.py:695`
    (`"enabled": os.environ.get("EXO_DSV4_DSPARK", "0") == "1"`) and `set_dspark_taps`
    `deepseek_v4.py:701-705`.
  - The `pp_speculation.py` path also reads `EXO_DSV4_DSPARK` (PP consumer; not taken here —
    `MLX_JACCL_SHARDING_MODE=Tensor`, production env line 120).
- **(b) import vs per-call.** Head-load flags read once at model load. The `deepseek_v4.py:695`
  module-level dict value is evaluated at import; `set_dspark_taps` re-derives `enabled` at call
  time from the same env.
- **(c) defaults.** `DSPARK` `"0"`, `DSPARK_NATIVE` `"0"` (i.e. local head), `DSPARK_TP_SHARD` `"0"`.
- **(d) mechanism.** `EXO_DSV4_DSPARK=1` attaches the 3-stage DSpark draft head
  (`mtp.{0,1,2}`) to the model and enables the taps side channel. `_NATIVE` selects the
  checkpoint's own trained `mtp.*` weights instead of the separately converted local head
  (`utils_mlx.py:397-404, 463-468`). `_TP_SHARD` shards the head's MoE FFNs across ranks in
  `auto_parallel.py:1243-1309`, so the draft forward issues paired `all_sum` collectives.
- **(e) live?** **Dormant on the live engine for the same routing reason** — `dsv41` load uses
  `exl3_build.build_model`/`build_mtp` (`dsv41/load.py:132, 156, 217-230`) and its own
  `head.draft` (`dsv41/rounds.py:408`), never `utils_mlx._overlay_dsv4_dspark` or
  `deepseek_v4._DSPARK_CTX`. On the *legacy* path these would be live and load-bearing; the live
  `dsv41` engine has its own equivalent (its draft head is attached in `load.py:68-69, 207-243`).
- **(f) risk class.** Head-load/attachment and TP-shard geometry — **scheduling / memory**; it
  does not by itself change the accept rule (though a *different* head changes the draft
  distribution, which is the accelerator's whole point). Inert on the live engine.

## 6. EXO_SPECULATIVE_GAMMA (value `3`)

- **(a) read sites.** `src/exo/worker/engines/mlx/generator/batch_generate.py:845`
  (`gamma = int(os.environ.get("EXO_SPECULATIVE_GAMMA", "2"))`, legacy generator construction);
  and on the legacy DSpark branch `dsv4_mtp.py:3969-3976`, where it **caps** gamma to
  `min(env, block_size)`.
- **(b) import vs per-call.** `batch_generate.py:845` is read at generator construction;
  `dsv4_mtp.py:3969` is read per cycle on the DSpark branch.
- **(c) default when unset.** `"2"` in `batch_generate.py:845`; `dsv41` engine default is its own
  dataclass field `gamma: int = 3` (`dsv41/engine.py:280`).
- **(d) mechanism.** On the legacy path, `gamma` is the number of draft tokens the chained MTP
  predicts and the verify forward rows (`verify_input = [anchor, d_0..d_{γ-1}]`, length γ+1). On
  the DSpark branch it is re-bound to `block_size` unless the env explicitly caps it
  (`dsv4_mtp.py:3967-3976`). On the live engine the *adaptive* `GammaPolicy` starts at
  `self.gamma=3` and then picks γ∈{1,2,3,4} per round from observed acceptance
  (`dsv41/rounds.py:291-295, 402`; `mlx_lm/models/deepseek_v41/spec.py:80-125`).
- **(e) live?** **Not from the env.** The live `dsv41` engine never reads `EXO_SPECULATIVE_GAMMA`
  (grep: the symbol appears only in `batch_generate.py` and `dsv4_mtp.py`). It happens to start at
  γ=3 by its own default, but the value comes from the dataclass field, not the env, and is then
  adaptive.
- **(f) risk class.** **Scheduling** (draft length / rows per verify). Does not change the
  accept rule; changes only how many candidates are proposed per round.

## 7. EXO_DSV4_MTP_DEDICATED (value `0`)

- **(a) read site.** `src/exo/worker/engines/mlx/utils_mlx.py:383`
  (`and os.environ.get("EXO_DSV4_MTP_DEDICATED", "0") == "1"`), gate at `:381-390`.
- **(b) import vs per-call.** Read once at model load (inside `shard_and_load`).
- **(c) default when unset.** `"0"` → keep the checkpoint-native MTP head.
- **(d) mechanism.** With `=1` (and `EXO_DSV4_MTP=1`), overlays the separately-converted dedicated
  head `mlx-community/DeepSeek-V4-Flash-MTP-bf16` over the checkpoint's bundled `mtp[0]` before
  tensor sharding; `=0` keeps the native head. Different head weights ⇒ different draft
  distribution.
- **(e) live?** **No** — legacy `dsv4_mtp`/`DeepseekV4Model` load path only (`utils_mlx.py` is not
  on the `dsv41` load path; `dsv41/load.py:6-11` states the generic post-load machinery does not
  apply). Dormant. Also note it is gated behind `EXO_DSV4_MTP=1`, a legacy-path flag.
- **(f) risk class.** Would change the draft proposal distribution (**numerics via weights**);
  inert on the live engine.

## 8. EXO_DSV4_SPEC_CACHE_ROLLBACK (value `1`), EXO_DSV4_SPEC_STATE_RESTORE (value `1`)

- **(a) read sites.** `SPEC_STATE_RESTORE`: `dsv4_mtp.py:705`
  (`_SPEC_STATE_RESTORE = os.environ.get("EXO_DSV4_SPEC_STATE_RESTORE", "0") == "1"`).
  `SPEC_CACHE_ROLLBACK`: `dsv4_mtp.py:719`
  (`_SPEC_CACHE_ROLLBACK = os.environ.get("EXO_DSV4_SPEC_CACHE_ROLLBACK", "0") == "1"`).
  Consumed at `:4131-4149` (snapshot/arm), `:4215-4222` (disarm), `:4974-5011` (rollback).
  The mlx-lm cache side (`arm/restore_spec_state`) is defined at
  `mlx-lm/mlx_lm/models/cache.py:844, 1706` (docstrings name the same env).
- **(b) import vs per-call.** Both are **module-import** reads (column 0, `dsv4_mtp.py`). Restart
  required to toggle.
- **(c) defaults.** Both `"0"`.
- **(d) mechanism.** `SPEC_STATE_RESTORE=1` replaces the legacy per-`trim()` rollback: it
  snapshots every ring + pool before the verify forward, and on any rejection restores them
  wholesale, then re-commits `[y]+accepted` with one small forward (`dsv4_mtp.py:686-705`,
  `:4974-4990`). `SPEC_CACHE_ROLLBACK=1` (requires `SPEC_STATE_RESTORE`) replaces that
  commit-forward with a cache-level exact undo: stash the rows the verify pushed
  (`arm_spec_stash`) and re-push only committed rows on rejection, avoiding the −41% decode cost
  of the full re-forward (`dsv4_mtp.py:707-719`).
- **(e) live?** **No** — legacy `dsv4_mtp.py` path only. The live `dsv41` engine has its own,
  independent rollback (`dsv41/rounds.py:422-453` `snap`/`stashes`/`rollback` from
  `mlx_lm/models/deepseek_v41/spec.py:20-73`). Dormant.
- **(f) risk class.** **Scheduling + correctness-mechanism** (cache rollback exactness). They
  affect bit-exactness *within the legacy path* but are inert on the live engine.

## 9. EXO_DSV4_BS_MIN_ACCEPT (value `1`)

- **(a) read site.** `dsv4_mtp.py:284`
  (`_BS_MIN_ACCEPT = os.environ.get("EXO_DSV4_BS_MIN_ACCEPT", "1") != "0"`). Consumed at
  `:2828` (B>1 greedy clamp) and `:3052` (B>1 sampling clamp).
- **(b) import vs per-call.** **Module import** (column 0).
- **(c) default when unset.** Non-`"0"` → ON.
- **(d) mechanism.** At BS>1 it clamps every stream's acceptance to `n_min = min(n_accepted)`
  across streams, keeping all batch-uniform pool/indexer caches in lockstep. Without it,
  per-stream acceptance diverges from the batch-uniform pool rollback and bleeds committed tokens
  from higher-accepting streams' pools → a repetition attractor within ~300 tok at temp>0
  (`:262-284`, `:303-318` logs a loud warning when OFF).
- **(e) live?** **No** — legacy batched `dsv4_mtp` path (the `dsv41` engine is single-request /
  batch-size-1 by design: `dsv41/engine.py:45, 264`). Dormant.
- **(f) risk class.** **Scheduling + correctness** (which committed tokens are kept). Inert on the
  live engine.

## 10. EXO_DSV4_MTP_TIE_REVERIFY (value `0`)

- **(a) read site.** `dsv4_mtp.py:4587`
  (`if temp == 0 and os.environ.get("EXO_DSV4_MTP_TIE_REVERIFY", "0") == "1":`), body
  `:4588-4597`. (`EXO_DSV4_MTP_TIE_REVERIFY_EPS` read at `:4588`.) A related log gate at `:5204`.
- **(b) import vs per-call.** **Per call** (inside the accept step, temp=0 only).
- **(c) default when unset.** `"0"` → OFF.
- **(d) mechanism.** On the rows that decide the cycle's commits (0..n_accepted), it scans the
  top-2 logit gap; at the first row whose gap < `eps` (default `1.0`) it truncates acceptance
  there, marks the cycle, and after rollback emits the *clean single-token forward's* argmax at
  that position — ground-truth sequential decode rather than a heuristic tie-break
  (`:4568-4597`). Cost: one extra 1-token forward on tie cycles.
- **(e) live?** **No** — legacy `dsv4_mtp.py` path only; also the code comments mark it
  "RETIRED FROM PROD 2026-07-10" (`:447-448`). Dormant.
- **(f) risk class.** **Changes output distribution** (it substitutes a different argmax). Inert
  on the live engine.

---

## 11. Draft-forward and verify-forward call sites and shapes

### 11a. LIVE path — `Dsv41Engine` (what this checkpoint actually runs)

- **Draft forward.** `src/exo/worker/engines/mlx/dsv41/rounds.py:408-414`:
  ```
  drafted = head.draft(
      anchor.reshape(-1),          # anchor ids [b], b=1
      getattr(model, "embed", None),
      getattr(model, "head", None),
      draft_state,                 # the DSpark ctx window (head.make_cache(1))
      width=gamma,                 # gamma from the adaptive policy, start 3
  )
  ```
  `head` is the DSpark head (`mlx_lm/models/deepseek_v41/mtp.py:290-339`). Its `draft` runs
  **one parallel block forward** over `(b, bs)` ids (`mtp.py:303-315`) followed by a sequential
  first-order Markov sample loop (`mtp.py:320-332`), returning `(draft_tokens [b, bs],
  confidence [b, bs])` — i.e. **γ candidate tokens per round** with `bs = width = gamma`.
  (`rounds.py:415-418` unwraps the tuple and takes `drafted[0]`.)

- **Verify forward.** `rounds.py:419-427`:
  ```
  verify_in = mx.concatenate([anchor.reshape(1, 1), drafted.reshape(1, gamma)], axis=1)  # (1, gamma+1)
  ...
  logits, taps = model(verify_in, cache, return_taps=True, argmax=True)                 # (1, gamma+1)
  ```
  So the batched verify is **anchor + γ drafts = γ+1 rows** in ONE body forward. With
  `argmax=True` the head returns *token ids* (`mlx_lm/models/deepseek_v41/model.py:334-337`;
  `rounds.py:431` `target = [int(v) for v in logits[0]]`). Preconditions are grown to
  `1 + gamma` rows at the eval-clean boundary (`rounds.py:406`), and the rejected suffix is
  rolled back by `spec.rollback` (`rounds.py:430-453`).

  In the deployed config the brief's `EXO_SPECULATIVE_GAMMA=3` does **not** set γ; the engine's
  γ is `GammaPolicy(start=3)` then adaptive (`dsv41/engine.py:983-986`, `rounds.py:402`).

### 11b. Legacy path — `DSv4MTPBatchGenerator` (the knobs' owner; NOT deployed)

- **B=1 draft.** `dsv4_mtp.py:3917-4086`. Two sub-branches keyed on `_dspark`
  (`_dspark = getattr(self.model.model, "dspark", None)`, `:3947`):
  - DSpark branch (`:3967-4076`): `_dspark.draft(y.reshape(1), embed, lm_head, _dsc,
    temperature=temp, sample_fn=..., width=gamma)` returns `(_toks, _corrected, _dspark_conf)`;
    `gamma` was re-bound to `block_size` (capped by env) at `:3968-3976`.
  - Chained-MTP branch (`:4077-4086`): `draft_tokens(self.mtp, pre_norm, next_token_arr, gamma,
    temp, ...)` returns γ chained drafts (`mtp_module.py:815-942`).
- **B=1 verify.** `:4093-4100` builds `draft_concat = concatenate([d.reshape(1,1) for d in
  draft_ids], axis=1)` → `(1, γ)`, then `verify_input = concatenate([next_token_arr, draft_concat],
  axis=1)` → **`(1, γ+1)`**; the forward is `dsv4_speculative_forward(self.model, verify_input,
  gen_batch.prompt_cache, self._captured)` at `:4166` (helper defined `:1531-1583`, returns
  `(pre_norm, logits)`).
- **B>1 draft.** `dsv4_mtp.py:2660-2662` `_draft_tokens_batched(stacked_pre_norm (N,1,hidden),
  next_tokens_arr (N,1), gamma, _tvec (N,1), all_greedy)`; returns `draft_ids_list[i]` shape
  `(N,)` and `draft_probs_list[i]` `(N, vocab)` at temp>0 (`:3493-3521`).
- **B>1 verify.** `:2673-2678` `draft_concat (N, γ)` then `verify_input (N, γ+1)`; forward at
  `:2698`.
- **Tree verify.** `:5566` (`dsv4_speculative_forward` on the tree rows) — reachable only when
  `EXO_DSV4_TREE_DRAFT=1` (`:125`), which is not in the config.

Both engines draft γ candidates and verify `γ+1` rows in one forward; the live engine simply
does it through a different module pair.

---

## 12. Sampling: GPU `mx` ops vs CPU/numpy, and the temp>0 vs greedy path

### 12a. LEGACY path (`dsv4_mtp.py` / `mtp_module.py`) — GPU `mx` ops

Sampling is done **on device with `mx` ops**, not pulled to numpy:

- Draft sampling (temp>0): `mtp_module.py:925-927`
  (`q = mx.softmax(logits / temp); tok_arr = mx.random.categorical(logits * (1.0/temp))`); c=2
  variant `dsv4_mtp.py:3697-3699` (`q = mx.softmax(logits/tvec)`, `mx.random.categorical(logits/tvec)`).
- Verify accept/bonus (temp>0): `dsv4_mtp.py:4337-4343` (`p = mx.softmax(verify_logits[0,i]/temp)`,
  ratio), `:4403` (`mx.random.uniform`), `:4413-4425` (`mx.random.categorical` on the residual /
  bonus logits, routed through `_mtp_filter_logits`).
- Greedy (temp=0): `mx.argmax` (`:4230`, `:4246-4249`); **no RNG consumed**.
- The only host returns are scalar `.item()` for committed tokens and small `.tolist()` dumps for
  traces/penalties. **No full-vocab tensor is pulled to CPU/numpy for sampling.**

**Materially different code path for temp>0?** **Yes.** The branch is explicit:
`if temp == 0:` at `dsv4_mtp.py:4239` (greedy argmax) vs `else:` at `:4335` (rejection-sampling
per `mtp_batch_generator.py:237-253` shape). Greedy consumes no RNG; sampling draws uniforms and
categoricals and uses a per-stream temperature (`:2613-2632`).

### 12b. LIVE path (`dsv41` engine) — greedy only; the sampling module is NOT wired

- `dsv41/rounds.py::_one_round` is **always greedy**: `mx.argmax` for the plain round
  (`rounds.py:367-368`, `:390`) and `argmax=True` on the verify forward (`:424-427`). The
  committed token is the target's own argmax (`rounds.py:441-447`). There is no temperature
  term anywhere in `_one_round`.
- The engine never consults temperature: `_refuse_unsupported` checks only images
  (`rounds.py:68-90`), and `_generate` (engine.py `:689-...)` reads `params.temperature`
  **nowhere**. The one function that would read it, `Dsv41Engine._resolve_sampler`
  (`dsv41/engine.py:1040-1051`), **has no caller in the tree** (grep for `_resolve_sampler`
  returns only its own definition and the docstring at `:29`).
- Consequence: a `temperature > 0` request is **not actually refused** — it is **silently served
  greedy**. The card documents the intended refusal (`...toml:65-73`, "a non-zero temperature is
  refused at request time") and the engine docstring repeats it (`engine.py:27-29`), but the
  refusal code (`engine.py:1044-1050`) is unreachable. Flagging this as a real behavior/contract
  gap worth a follow-up.
- The device-side sampling *implementation* exists but is only reachable through
  `mlx_lm.models.deepseek_v41.spec.generate(temperature>0)` (`spec.py:229-242`), which the engine
  never calls. That module (`deepseek_v41/sampling.py`) does the accept/reject arithmetic
  **on the host in fp32 numpy** (`sampling.py:38`, `_Device.host` at `:210-211`, `_as_int` at
  `:178-180`), pulling only the small per-row summary over the wire (`sampling.py:18-25`), while
  the *draws* stay lazy in the verify graph (`sampling.py:35-38`).
- **Materially different code path for temp>0?** **No.** On the live engine there is no
  temp-dependent branch at all; temp=0 and temp>0 both execute the identical greedy
  `_one_round`. (On the legacy engine, yes — §12a.)

---

## 13. `repetition_penalty`

- **Generic MLX path (not the live one).** Resolved in `batch_generate.py:2689-2704` and
  `generate.py:2432-2441`; `1.0` collapses to `None` so mlx-lm skips the processor
  (`batch_generate.py:2689-2695`). Launcher default is `DSV4_REPETITION_PENALTY:=1.0`
  (`start_cluster.sh:895`), with the 1.1→23.3% incident and its revert documented at
  `start_cluster.sh:879-895`.
- **Legacy MTP path.** A *separate* knob `EXO_DSV4_MTP_REP_PEN` (default **1.3**) feeds
  `_mtp_filter_logits` (`dsv4_mtp.py:1689`, applied at `:1737-1750`, called at `:4416, 4424,
  3026-3032`). This is **not** the request `repetition_penalty`; it is an MTP-only sampling-parity
  penalty. (Default 1.3, so it is *not* a no-op on that path — but that path is dormant.)
- **LIVE `dsv41` engine.** Grep of `src/exo/worker/engines/mlx/dsv41/` finds **no**
  `repetition_penalty` / `presence_penalty` / `frequency_penalty` handling at all (only an
  unrelated prose mention in `dsml.py:137`). The engine is pure greedy argmax, applies no
  sampling penalty, and therefore satisfies the "1.0 / no-op" requirement **trivially: it applies
  no repetition penalty of any kind.** The prior 1.1→23.3% corruption class cannot arise on this
  engine because there is no penalty term to mis-set.

---

## 14. Risk-class summary (legacy path semantics)

| knob | value | changes output/numerics? | pure scheduling? |
|---|---|---|---|
| `EXO_DSV4_MTP_ACCEPT_LOGPROBS` | 1 | **yes** (argmax rule) | no |
| `EXO_DSV4_MTP_TIEBREAK_FIX` / `_EPS` | 0 / 0.5 | **yes** (tie selection) | no |
| `EXO_DSV4_MTP_EAGLE_K` | 8 | **yes** (draft distribution) | no |
| `EXO_DSV4_MTP_C2_MAX_CTX` | 1 | no | **yes** (c≥2 gate) |
| `EXO_DSV4_MTP_MAX_CTX` | 0 | no | **yes** (ctx gate) |
| `EXO_DSV4_DSPARK` (+NATIVE/TP_SHARD) | 1 | head weights (draft dist.) | attachment/shard |
| `EXO_SPECULATIVE_GAMMA` | 3 | no | **yes** (draft length) |
| `EXO_DSV4_MTP_DEDICATED` | 0 | draft weights (draft dist.) | head selection |
| `EXO_DSV4_SPEC_CACHE_ROLLBACK` | 1 | rollback exactness | partial |
| `EXO_DSV4_SPEC_STATE_RESTORE` | 1 | rollback exactness | partial |
| `EXO_DSV4_BS_MIN_ACCEPT` | 1 | no (which tokens kept) | partial |
| `EXO_DSV4_MTP_TIE_REVERIFY` | 0 | **yes** (argmax substitution) | no |

All of the above are **inert on the deployed `dsv41` engine** (§0).

---

## DORMANT / DEAD KNOBS

Knobs whose code path is NOT reachable for the deployed checkpoint
(`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`, `engine = "dsv41"`). The gate in every
case is the card→engine dispatch, proven at
`resources/inference_model_cards/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw.toml:56`
+ `src/exo/worker/engines/mlx/dsv41/dispatch.py:51-58` + `src/exo/worker/runner/bootstrap.py:332-339`;
`Dsv41Builder`/`Dsv41Engine` (`dsv41/builder.py:89-108`) never import `dsv4_mtp.py`
(`dsv41/engine.py:7-30`). Each knob below is read only inside the legacy
`dsv4_mtp.py`/`DeepseekV4Model` stack:

| knob | proving read-site (legacy path) | extra dormant-gate proof |
|---|---|---|
| `EXO_DSV4_MTP_ACCEPT_LOGPROBS` | `dsv4_mtp.py:340` | only consumer `:4243,2733,5595` in `dsv4_mtp.py` |
| `EXO_DSV4_MTP_TIEBREAK_FIX` | `dsv4_mtp.py:4275` | only consumer `:4277-4287` |
| `EXO_DSV4_MTP_TIEBREAK_EPS` | `dsv4_mtp.py:4276` | only consumer `:4280` |
| `EXO_DSV4_MTP_EAGLE_K` | `dsv4_mtp.py:1108` | consumed `mtp_module.py:789`, `dsv4_mtp.py:3566`; and the chained-MTP branch is dead when the DSpark branch wins (`dsv4_mtp.py:3947,4008 vs 4078`) |
| `EXO_DSV4_MTP_C2_MAX_CTX` | `dsv4_mtp.py:2399` | batch-only branch `len(gen_batch)>=2` (`:2398`); live engine is BS=1 (`dsv41/engine.py:45,264`) |
| `EXO_DSV4_MTP_MAX_CTX` | `dsv4_mtp.py:2435` | only consumer `:2436-2452` |
| `EXO_DSV4_DSPARK` | `utils_mlx.py:440`; `deepseek_v4.py:695` | `dsv41` loads via `exl3_build` (`dsv41/load.py:132,156`) |
| `EXO_DSV4_DSPARK_NATIVE` | `utils_mlx.py:463` | inside `_overlay_dsv4_dspark*`, never called by `dsv41` |
| `EXO_DSV4_DSPARK_TP_SHARD` | `auto_parallel.py:1239` | `tensor_auto_parallel` not used by `dsv41` (`dsv41/load.py:6-11`) |
| `EXO_SPECULATIVE_GAMMA` | `batch_generate.py:845`; `dsv4_mtp.py:3969` | live gamma from `dsv41/engine.py:280-281,983-986` + `rounds.py:402` |
| `EXO_DSV4_MTP_DEDICATED` | `utils_mlx.py:383` | legacy-model load only (`utils_mlx.py:381-390`) |
| `EXO_DSV4_SPEC_CACHE_ROLLBACK` | `dsv4_mtp.py:719` | consumed `:4143,4215,4992`; `dsv41` has own rollback (`rounds.py:422-453`) |
| `EXO_DSV4_SPEC_STATE_RESTORE` | `dsv4_mtp.py:705` | consumed `:4131,4215,4974` |
| `EXO_DSV4_BS_MIN_ACCEPT` | `dsv4_mtp.py:284` | consumed `:2828,3052` (B>1 only) |
| `EXO_DSV4_MTP_TIE_REVERIFY` | `dsv4_mtp.py:4587` | consumed `:4588-4597` |

Also dead for the deployed engine:
- **`Dsv41Engine._resolve_sampler`** (`dsv41/engine.py:1040-1051`) — no caller in the tree; the
  documented temperature>0 refusal never fires. Flagged in §12b.
- **DSpark branch's `EXO_SPECULATIVE_GAMMA` cap** (`dsv4_mtp.py:3969-3976`) — legacy DSpark path,
  not the live `dsv41` DSpark.
