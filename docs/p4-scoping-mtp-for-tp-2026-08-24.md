# P4 scoping — MTP-for-TP feasibility + break-even arithmetic (2026-08-24)

> # ⚠️ SUPERSEDED IN PART — READ THIS BANNER FIRST (P4v2, 2026-08-24)
>
> **The ordering in the original memo below is WRONG and is retracted.**
> It scoped *classic single-head MTP* first and DSpark second. Per user
> direction and confirmed by code, that is backwards:
>
> - **DSpark is the DeepSeek-V4-native MTP variant and this cluster's
>   intended sole speculation mechanism** (`start_cluster.sh:176-178`,
>   `:452-458`, `2d696ff60`'s commit body: *"DSpark is this cluster's
>   intended SOLE MTP mechanism"*).
> - Production env already carries `EXO_DSV4_DSPARK=1` / `EXO_DSV4_MTP=0`
>   (verified live via `ps eww` on both nodes, 2026-08-24).
> - **The crux question the original memo never asked has now been
>   answered from code: the TP draft+verify cycle DOES have a first-class
>   DSpark branch.** `_speculative_next` checks for the attached DSpark
>   module *before* the classic path and, when present, drafts with the
>   3-stage head (`speculative/dsv4_mtp.py:3481-3486`). DSpark drafting at
>   TP is **not** PP-only. §A below is the code-truth writeup.
>
> **What the original memo still gets right and is NOT retracted:** the
> cost model `C(k)=(k+1)·A+N+k·d` and its inputs (§3.1-3.3), the finding
> that the verify floor at TP=2 is row-sequential attention rather than
> the collective (§3.3), the risk register (§4), and the measurement
> discipline (§5). Those carry forward into the P4v2 plan unchanged.
>
> **What is retracted:** the ordering ("classic MTP first"), the
> "DSpark second: NO-GO for now" verdict line in the VERDICT table (it
> rested on conflating the DSpark *head* with the FULLBLOCK *verify
> mode* — see §C), and the "What IS missing is a `-0731`-matched classic
> MTP head" framing of the blocker (§B shows the real blocker is
> different and worse).
>
> **New sections, all machine-verified against exo `6dcd61e4a` /
> mlx-lm `0854b39` with `file:line` for every claim:**
>
> | § | Contents |
> |---|---|
> | **§A** | Code-truth: every `EXO_DSV4_DSPARK` read site; what prod runs today; the full TP DSpark draft path; the silent-partial-load hazard |
> | **§B** | The 2026-07-26 disable bug (`2d696ff60`) — status: **FIXED**, with the caveat that re-enable was never completed |
> | **§C** | DSpark-specific economics: the `block_size=5` draft geometry, the corrected break-even curve, and the head-vs-mode distinction |
> | **§D** | The P4v2 primary plan: first milestone, fence-owner control, silent-partial-load guard, pre-registered gate |
>
> Everything from `## VERDICT` down to `## 6. Summary for the PM` is the
> **original P4 (v1) memo, preserved verbatim** for audit. Read §A-§D
> first; treat v1 as supporting material.

---

## §A. CODE-TRUTH: what `EXO_DSV4_DSPARK` actually does at TP=2

### A.1 Every site that reads `EXO_DSV4_DSPARK`

Exhaustive `grep` across `src/` and the `mlx-lm` submodule (excluding
`__pycache__`, `.venv`, and `build/` duplicates):

| # | Site | What it does |
|---|---|---|
| 1 | `src/exo/worker/engines/mlx/utils_mlx.py:383` | **Load-time attach gate.** `if os.environ.get("EXO_DSV4_DSPARK","0")=="1":` → overlays the DSpark head onto `model.model.dspark` |
| 2 | `utils_mlx.py:385` | `EXO_DSV4_DSPARK_NATIVE` sub-selector: native checkpoint-bundled head vs. locally-converted head |
| 3 | `utils_mlx.py:739` | `EXO_DSV4_DSPARK_DIR` — path of the locally-converted head (default `~/.exo/models/local--DeepSeek-V4-Flash-DSpark-MTP`) |
| 4 | `mlx-lm/mlx_lm/models/deepseek_v4.py:608` | `_DSPARK_CTX["enabled"]` module-level init from the env |
| 5 | `deepseek_v4.py:617` | `set_dspark_taps()` re-evaluates the same env when arming taps |
| 6 | `speculative/dsv4_mtp.py:3527` | `EXO_DSV4_DSPARK_CONF_TAU` (default `0.5`) — confidence-based block pruning |
| 7 | `pp_speculation.py:1627,1711,1760` | PP-path comments + the "requires `EXO_DSV4_DSPARK=1`" error message |
| 8 | `generator/batch_generate.py:3416-3420` | Comment documenting that attach-at-load is the ONLY control surface |
| 9 | `start_cluster.sh:1715` | `: "${EXO_DSV4_DSPARK:=1}"` — the production default |

**Note there is no separate "use DSpark for TP drafting" switch.** The
runtime consumers all test for the *presence of the attached module*
(`getattr(inner, "dspark", None)`), not the env var. That is deliberate
(`batch_generate.py:3416-3430`) so the rank-consistency guard
(`utils_mlx.py:397-417`) can detach group-wide and every consumer falls
back correctly.

### A.2 Production today (`DSPARK=1`, `MTP=0`) — what actually executes

Live `ps eww` on both nodes, 2026-08-24 (read-only):

```
EXO_SPECULATIVE=0          EXO_DSV4_MTP=0        EXO_DSV4_DSPARK=1
EXO_DSV4_DSPARK_NATIVE     (unset → local head)  MLX_JACCL_SHARDING_MODE=Tensor
EXO_DSV4_ROWSEQ_FULLBLOCK=1                      EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=0
```

Live log, both nodes:
`DSpark draft head attached from ~/.exo/models/local--DeepSeek-V4-Flash-DSpark-MTP (115 tensors, 3 stages, block_size=5, taps=[40,41,42])`,
2 × `DSpark ctx warmed`, **0** × `DSv4 MTP speculative decoding enabled`,
**0** × `PP speculation using ...`.

**Correction to the task premise:** the generator is not skipped merely
because `is_dsv4_with_mtp` is false — it is skipped one level earlier.
`__post_init__` wraps the whole block in
`if use_speculative:` where `use_speculative = os.environ.get("EXO_SPECULATIVE","0")=="1"`
(`batch_generate.py:813, 822`). With `EXO_SPECULATIVE=0` the branch is
never entered at all. **Both gates are independently closed today**
(`EXO_SPECULATIVE=0` *and* `EXO_DSV4_MTP=0`); either one alone suffices to
prevent construction.

So, with the generator never constructed, `EXO_DSV4_DSPARK=1` today buys
exactly three things, all of whose outputs are discarded:

| Work | Site | Frequency | Cost verdict |
|---|---|---|---|
| **(a) Head attach** — load + `nn.quantize` + `load_weights` of a 3-stage draft head | `utils_mlx.py:383-390` → `_overlay_dsv4_dspark` (`:714-816`) | **Once per model load** | **~10 GB of unified memory held permanently** (`model.safetensors` = 10,876,789,654 B on disk, measured). Also seconds of load time. This is the ONLY material cost, and it is a *memory* cost, not a throughput cost. On 128 GB nodes already at ~125 GB for weights this is not negligible. |
| **(b) hc-mean tap capture** at global layers 40/41/42 | `deepseek_v4.py:6705-6707, 6722-6724` | **Every target forward** — every prefill chunk AND every decode step | `h.mean(axis=2)` over `(B,L,hc_mult,hidden)` → `(B,L,4096)`, 3× per forward, plus a dict rebuild. At decode `L=1` this is ~3 tiny reductions per token — **negligible but non-zero, and it is on the decode critical path.** At prefill with large `L` it is 3 reductions over the full chunk — still small relative to 43 layers of MoE, but it scales with prefill chunk size. **Not measured; do not claim a number.** |
| **(c) `append_ctx` warm** — builds fresh rotating caches + one projection through `main_proj`/`main_norm` + `mx.eval` | `batch_generate.py:2748-2760` | **Per request** (inside `submit()`) | Per-request, not per-token. One small projection + a forced `mx.eval` on the cache keys. Negligible at request granularity. |

**Wasted-work verdict:** the per-token and per-request costs (b)+(c) are
**too small to be worth an emergency patch**, and both sit on paths that
would be needed the moment speculation is re-enabled. The cost that IS
worth acting on is **(a): ~10 GB of resident memory for a head that
cannot currently be used.** On a cluster whose documented constraint is
"~125.4 GB of 128 GB" for co-resident weights, 10 GB of dead draft head
is the single largest recoverable resource here. **Recommendation:
either re-enable speculation (making the 10 GB productive) or set
`EXO_DSV4_DSPARK=0` (reclaiming it) — the current state is the worst of
both.** Note this contradicts `start_cluster.sh:1712`'s claim that DSpark
"costs nothing extra to warm under TP"; that comment accounts for the
compute but not the residency.

### A.3 THE CRUX: which head drafts in the TP cycle?

**Answer: the DSpark 3-stage head drafts, and it takes priority over the
classic MTP module.** This *confirms* the user's expectation and
*reverses* the v1 memo's ordering.

The TP speculative cycle is `DSv4MTPBatchGenerator._speculative_next`
(`dsv4_mtp.py:3415`). At its drafting step it does:

```python
# dsv4_mtp.py:3481-3486
_dspark = getattr(getattr(self.model, "model", None), "dspark", None)
if _dspark is not None:
    gamma = _dspark.block_size          # γ re-bound to 5 for the WHOLE cycle
    if getattr(self, "_dspark_caches", None) is None:
        self._dspark_caches = _dspark.make_cache()
    _dsc = self._dspark_caches
    ...
    _toks, _corrected, _dspark_conf = _dspark.draft(...)   # :3502-3509
else:
    draft_ids, draft_probs = draft_tokens(self.mtp, ...)   # :3549  ← classic
```

So the classic single-head path at `:3549` is the **`else` branch**. If
the DSpark module is attached, the classic MTP module never drafts.

**Full traced draft path at TP (all in `_speculative_next`):**

1. **Draft** — `_dspark.draft(y, embed_tokens, lm_head, caches, temperature, sample_fn)`
   (`dsv4_mtp.py:3502-3509`). Inside `DeepseekV4DSparkModule.draft`
   (`deepseek_v4.py:6393-6475`): builds `[anchor, noise×(bs-1)]`, runs ONE
   parallel forward through all 3 stages, then a sequential rank-256
   Markov loop samples `bs` tokens left-to-right.
2. **k (draft length)** — `gamma = _dspark.block_size` = **5**
   (`deepseek_v4.py:6358`, confirmed live: `block_size=5`). **The 3 stages
   are NOT 3 draft tokens** — the stages are depth (a 3-layer draft
   network), the block is width. One `draft()` call returns **5** draft
   tokens, not 1/2/3. Then confidence pruning
   (`EXO_DSV4_DSPARK_CONF_TAU`, default `0.5`, `dsv4_mtp.py:3526-3540`)
   truncates to the leading prefix clearing τ, floored at 1:
   `gamma = max(1, min(_kept, gamma))`. **So k is dynamic in [1,5], not
   fixed** — a fact the economics in §C must handle and v1's fixed-k
   model did not.
3. **Cross-rank sync** — `_dspark_sample` broadcasts each sampled token
   through the isolated coord subgroup (`dsv4_mtp.py:3491-3500`),
   preserving the draft-determinism discipline that the Eagle-K1 incident
   established.
4. **Cache hygiene** — every stage cache is trimmed by `block_size`
   immediately after drafting (`:3512-3513`); block KV must not persist.
5. **Verify** — unchanged, shared with classic MTP. `verify_input =
   [y, d0..d_{γ-1}]` (`:3562-3567`) → `dsv4_speculative_forward`
   (`:3628`). **The verify path is k-agnostic**: it sizes off `gamma`,
   which was re-bound to the DSpark block. Multi-token drafts at TP are
   handled, subject to the rowseq gate (below).
6. **Ctx feed** — after acceptance, the hc-means the verify forward just
   captured at the tap layers are appended for the committed rows only:
   `_dspark.append_ctx(_ctx_cat[:, :n_accepted+1], self._dspark_caches)`
   (`:3984-3991`). **This is the same side channel the warm-up at
   `batch_generate.py:2748-2760` primes** — so yes, the warmed ctx does
   feed the draft, and the P4-worker's read of `dsv4_mtp.py:3484-3486`
   was correct.

**Does `EXO_DSV4_MTP=1` alone construct the classic module?** Yes —
`DeepseekV4Model.__init__` creates `self.mtp = [DeepseekV4MTPModule(...)]`
gated on `num_nextn_predict_layers > 0 and EXO_DSV4_MTP == "1"`
(`deepseek_v4.py:6500-6507`). **And that construction is mandatory for
the DSpark path too**, which is the trap:

> **`EXO_DSV4_MTP=1` is a hard prerequisite for DSpark drafting at TP,
> even though DSpark then bypasses the module it creates.**
>
> Chain: `is_dsv4_with_mtp` (`batch_generate.py:830-834`) requires
> `hasattr(inner,"mtp") and len(inner.mtp)>0`. That attribute only exists
> when `EXO_DSV4_MTP=1` (`deepseek_v4.py:6500-6507`). If it is false, the
> `DSv4MTPBatchGenerator` is never constructed (`:837-847`) — and
> `_speculative_next`, including its DSpark branch, is unreachable. There
> is **no** DSpark-only construction path under TP.

So the correct TP DSpark env is **all three**:
`EXO_SPECULATIVE=1` + `EXO_DSV4_MTP=1` + `EXO_DSV4_DSPARK=1`. The classic
module gets built and then not used for drafting (it is still used for
`DSv4MTPPredictor`'s pre-norm bookkeeping and the fence-owner
registration). This is un-obvious, undocumented, and is exactly the class
of thing that produces a silently-wrong A/B.

**No new wiring is required for DSpark drafting at TP.** The missing
pieces are configuration and observability, not code:

| Missing piece | Size |
|---|---|
| Env: set the three flags together | 0 (config) |
| A log line when the TP DSpark branch takes the draft (there is none — the only DSpark log lines are PP's and the load-time attach) | ~3 lines, `dsv4_mtp.py:3482` |
| Acceptance/`k`-histogram logging derived from already-materialized tensors | ~15 lines |
| Param-tree assertion after overlay (see §A.4) | ~10 lines |

### A.4 The silent-partial-load hazard — corrected, and it is real but NOT where v1 said

v1 warned that a mismatched head loads silently-partial because
`strict=False`. **Checked: for DSpark specifically, that is wrong — both
DSpark overlays use `strict=True`** (`utils_mlx.py:807` for the local
head, `:953` for the native head). A key/shape mismatch there raises.

The `strict=False` at `utils_mlx.py:349` is on the **main model load**
(`load_model(model_path, lazy=True, strict=False)`), and `:706` is the
**dedicated classic-MTP overlay**. Those are different code paths.

**But a real, live silent-partial-load hazard exists anyway — a different
one.** Production runs `EXO_DSV4_DSPARK_NATIVE` **unset**, i.e. the
`_overlay_dsv4_dspark` local-head path (`utils_mlx.py:390`). That head
lives at `~/.exo/models/local--DeepSeek-V4-Flash-DSpark-MTP` and was
**converted from the PREVIEW checkpoint's mtp shards**
(`utils_mlx.py:718-722`, `start_cluster.sh:1718-1726`), while the cluster
serves `deepseek-ai/DeepSeek-V4-Flash-0731` (12 log hits). `strict=True`
guarantees the *shapes* match — it cannot detect that the *weights were
trained against a different target model's hidden-state distribution*.

That is a **numerically-silent** mismatch: it loads cleanly, drafts
confidently, and simply has depressed acceptance. The `-0731` checkpoint
ships its own bundled 3-stage head (measured: **4,705 `mtp.*` keys**
across `mtp.0/1/2` — 1568/1565/1572 — in
`model.safetensors.index.json`), reachable via
`EXO_DSV4_DSPARK_NATIVE=1` (`utils_mlx.py:385-388` →
`_overlay_dsv4_dspark_native`, `:817-960`), and **it has never been
A/B tested live** (§5.3 "DSpark native head (INCONCLUSIVE)").

> **Note on the task brief's "81-param mtp.* tree":** not reproduced.
> The `-0731` index lists 4,705 `mtp.*` tensor keys. The 81 figure is
> likely a post-`sanitize()` *module-level* param count (sanitize stacks
> per-expert weights into single tensors), not the on-disk key count.
> Whatever the right number, **the guard must assert it rather than
> assume it** — see §D.3.

---

## §B. The 2026-07-26 disable bug — STATUS: FIXED (but re-enable never completed)

**`2d696ff60`** *"fix(dsv4): flip speculative-decode defaults OFF pending
DSpark bugfix"* (Sun Jul 26 2026). The bug: a confirmed A/B/A-verified
correctness failure — a specific prompt reliably looped into a
**self-doubt reasoning spiral**, burning the whole 8192-token budget
without producing an answer, with DSpark active; clean and correct with
speculation genuinely off. It flipped `EXO_SPECULATIVE=0`,
`EXO_DSV4_MTP=0`, `EXO_DSV4_DSPARK=0` together.

**It was subsequently root-caused and fixed.** Per `start_cluster.sh:355-381`
(the `EXO_SPECULATIVE` comment block, dated **2026-08-02**):

- Root cause was **not** DSpark-specific. It was DeepseekV4's known
  *"L>1 batched verify ≠ L sequential decode steps"* numerics-drift class
  — the same mechanism previously root-caused for MTP tool-call
  corruption. The same loop reproduced under classic draft-model
  speculation AND plain chained MTP.
- Ruled out along the way: cache-wraparound/BOS-spam (zero wrap events
  across a full failing run) and numerics near-ties (live logit-margin
  capture showed mean ~9 logits at every self-doubt transition — nowhere
  near a tie).
- **Fix:** `EXO_DSV4_VERIFY_ROWSEQ` (attention) + `EXO_DSV4_ROWSEQ_FULLBLOCK`
  (hc-adjacent ops and norms) + originally `..._FULLBLOCK_MOE`, making the
  verify forward bitwise-equivalent to sequential decode.
- **Validated:** 2× identical temp=0 reruns of the original failing prompt
  (`math_digit_sum`, 8000-token budget) → clean convergence both times
  (`finish_reason=stop`, correct boxed answer) vs. the prior 18-24×
  never-terminating repetition. `EXO_SPECULATIVE` default was flipped back
  to `1` (`start_cluster.sh:382`).

**So the blocker is cleared in code — with three caveats that belong in
the plan, not in a footnote:**

1. **The re-enable was never completed.** `start_cluster.sh:382` sets
   `EXO_SPECULATIVE:=1`, yet the live process shows `EXO_SPECULATIVE=0`
   on both nodes. Something downstream (a launch-time override or an
   operator export) is still forcing it off, and `EXO_DSV4_MTP:=0`
   (`:459`) was never flipped back either — so even with
   `EXO_SPECULATIVE=1` the generator would still not construct (§A.3).
   **Reconciling the intended default against the live process env is
   milestone 0 of any revival.**
2. **The fix is not bitwise-exact in the shipped config.**
   `EXO_DSV4_ROWSEQ_FULLBLOCK_MOE` was later turned OFF (`:284`) in
   favour of `EXO_DSV4_MOE_PARTS_ROWSEQ=shared` (`:285`), which carries a
   documented **0.023% residual divergence**, not zero. That is the first
   thing to re-audit if loop symptoms return.
3. **The `exo-speculative-decode-correctness` skill's RESOLVED section
   still says the opposite** — it concludes the root cause is
   "block-batched verify itself," present in all three mechanisms, and
   recommends `EXO_SPECULATIVE=0` as a *permanent* default for
   reasoning-heavy work. That was written 2026-08-01/02, i.e. concurrent
   with the fix, and does not incorporate the 2026-08-02 rowseq/FULLBLOCK
   validation. **The skill is stale on this point and should be patched.**
   Until it is, treat the correctness question as *"fixed, with a
   validated repro, but re-validation required before trusting it"* — a
   fresh A/B/A on `math_digit_sum` is cheap and is explicitly what
   `2d696ff60`'s own commit body demands before re-trusting.

---

## §C. DSpark-specific economics

### C.1 Head vs. decode-loop mode — the distinction that changes the verdict

The v1 "DSpark second: NO-GO" line rested on `§5.3`'s **15.9× collapse**.
That verdict does **not** attach to the DSpark head:

- **DSpark the head** — `DeepseekV4DSparkModule` (`deepseek_v4.py:6340`).
  Produces drafts. The collapse measurement recorded **94% draft
  acceptance** during the very cycle that took `r1_verify_fwd=1455.8 ms`.
  The head was drafting *excellently* while verify collapsed.
- **FULLBLOCK the verify mode** — `EXO_DSV4_ROWSEQ_FULLBLOCK=1`
  (`deepseek_v4.py:1591`, default-on `start_cluster.sh:283`). Runs the
  entire non-MoE block **once per drafted row**. Cost ≈ `k+1` × per-row
  attention, and per-row attention includes DSv4's Indexer top-k over the
  compressed KV cache, which grows with context. The collapse is that
  product — a `k`-multiplier problem exposed at `k=4` (block_size 5).

**The re-scoped plan avoids FULLBLOCK-at-large-k, not FULLBLOCK.** It
cannot drop FULLBLOCK: FULLBLOCK is half the §B correctness fix.
What replaces the collapsing configuration is **confidence pruning at a
high τ to force small k** — `EXO_DSV4_DSPARK_CONF_TAU`
(`dsv4_mtp.py:3526-3540`) already truncates the block to the leading
high-confidence prefix, floored at 1. Setting τ high (e.g. 0.9) drives
the effective k toward 1-2, which is exactly the regime the arithmetic
below says is survivable — **and it does so using draft-side information
only, preserving the non-anticipating property, so target-distribution
exactness is retained** (`dsv4_mtp.py:3517-3525`).

There is also a hard structural ceiling worth noting: rowseq requires
`B*L ≤ 8` (`deepseek_v4.py:5102`). At B=1, DSpark's full block gives
`L = k+1 = 6` — inside the gate, but at c=2 (B=2) `B*L = 12` **exceeds
it**, silently dropping the verify back to the classic batched path,
i.e. back to the pre-fix numerics that caused the §B bug. **DSpark at
full block_size is incompatible with concurrent serving under the current
rowseq gate.** With τ-pruning to k≤2 (`L≤3`), c=2 gives `B*L=6` and stays
inside. This is an independent argument for small k.

### C.2 Cost model, restated for DSpark

Same model as v1 §3.2, `C(k) = (k+1)·A + N + D`, with one structural
change: **DSpark's draft cost `D` is flat per cycle, not `k·d`.** The
block forward through the 3 stages is ONE parallel pass regardless of
block width (`deepseek_v4.py:6444-6448`); only the lightweight Markov
sampling loop (`:6453-6465`) is per-position, and it is an embedding
lookup + a rank-256 matmul per step. This is DSpark's actual structural
advantage over chained MTP and the v1 model could not express it.

**Assumptions (all inherited from v1 §3.2, restated so the reviewer can
attack them):** `A` = 43-layer attention path, multiplied by `(k+1)`
because rowseq runs exactly that many single-token-equivalent passes;
`N = T − A` treated **flat in k** (generous to speculation, since
FULLBLOCK also per-rows the hc ops living inside `N`); tokens per cycle
= `1 + accepted`; break-even where `C(k)/T = 1 + a`.

`D` for DSpark is **not measured anywhere.** It is a 3-stage transformer
block (each stage has its own attention + MoE ffn, `deepseek_v4.py:6365`)
plus a full `lm_head` projection over the block — structurally **more**
expensive per cycle than classic MTP's ~2.7 ms/draft. Swept `D = 4-12 ms`
with **8 ms as the central placeholder**. *This is the weakest input in
the whole analysis and the first thing the live prototype must measure.*

Inputs (unchanged, v1 §3.1): `T@100K = 34.55 ms`, `A@100K = 16.57`,
`N@100K = 17.98`; `T@352.6K = 39.49`, `A@352.6K = 19.13`, `N = 20.36`.

### C.3 Break-even: required mean-accepted tokens `a*` per cycle

@100K:

| k | D=4ms | D=6ms | **D=8ms** | D=10ms | D=12ms |
|---|---|---|---|---|---|
| **1** | 0.595 | 0.653 | **0.711** | 0.769 | 0.827 |
| **2** | 1.075 | 1.133 | **1.191** | 1.249 | 1.307 |
| 3 | 1.555 | 1.612 | **1.670** | 1.728 | 1.786 |
| 4 | 2.034 | 2.092 | **2.150** | 2.208 | 2.266 |
| **5 (full block)** | 2.514 | 2.572 | **2.630** | 2.687 | 2.745 |

@352.6K:

| k | D=4ms | D=6ms | **D=8ms** | D=10ms | D=12ms |
|---|---|---|---|---|---|
| **1** | 0.586 | 0.636 | **0.687** | 0.738 | 0.788 |
| **2** | 1.070 | 1.121 | **1.171** | 1.222 | 1.273 |
| 3 | 1.555 | 1.605 | **1.656** | 1.707 | 1.757 |
| 4 | 2.039 | 2.090 | **2.140** | 2.191 | 2.242 |
| **5 (full block)** | 2.523 | 2.574 | **2.625** | 2.675 | 2.726 |

### C.4 Gain surface at D=8ms, @100K

| k | a=0.66 | a=1.04 | a=1.5 | a=2.0 | a=2.5 | a=3.0 |
|---|---|---|---|---|---|---|
| **1** | **−3.0%** | — | — | — | — | — |
| **2** | −24.2% | **−6.9%** | **+14.1%** | +36.9% | — | — |
| 3 | −37.8% | −23.6% | −6.4% | +12.3% | +31.1% | +49.8% |
| 4 | −47.3% | −35.2% | −20.6% | −4.8% | +11.1% | +27.0% |
| **5** | −54.3% | −43.8% | −31.1% | −17.3% | −3.6% | +10.2% |

Equivalent per-draft acceptance under a geometric model (`a = Σ pⁱ`), D=8ms @100K:

| k | a* | required per-draft p* |
|---|---|---|
| 1 | 0.711 | **0.711** |
| 2 | 1.191 | **0.700** |
| 3 | 1.670 | **0.735** |
| 4 | 2.150 | **0.767** |
| 5 | 2.630 | **0.793** |

### C.5 Reading — GO/NO-GO

**The DSpark verdict is materially better than v1's classic-MTP verdict
at k=2, and materially worse at k=1.** Because `D` is flat, the required
per-draft acceptance is essentially *flat in k* (0.70-0.79) rather than
rising — so DSpark does not carry v1's "second draft can't pay for a
third attention pass" penalty in the same way. The penalty is instead
front-loaded: a flat 8 ms draft cost is expensive to amortize over a
single token, which is why **k=1 is the worst configuration for DSpark**
(−3.0% at a=0.66) — the exact inverse of classic MTP.

- **k=2 is the target regime.** Break-even `a* ≈ 1.19` (p* ≈ 0.70).
  Historical DSpark acceptance is genuinely encouraging: **94% draft
  acceptance** was recorded in the FULLBLOCK-collapse investigation, and
  a checkpoint-*mismatched* DSpark head measured **~0.64** on `-0731`.
  Those two numbers are wildly inconsistent and were measured under
  different definitions — **neither is usable as a decision input.**
  At `a=1.5` (p≈0.75 geometric) k=2 returns **+14.1%**; at `a=1.04`
  (classic MTP's measured γ=2 rate) it **loses 6.9%**.
- **k≥3 needs `a≥2.0`** — that is p≈0.77+ sustained, which nothing in
  this repo's history supports at depth.
- **Full block (k=5) is a NO-GO** and always was: `a*=2.63` and it also
  breaks the `B*L≤8` rowseq gate at c=2 (§C.1).

**VERDICT: conditional GO for DSpark at TP with k pinned to 2 via
confidence pruning — contingent on a live measurement of `D` and `a`,
because the decision swings from −6.9% to +14.1% across an acceptance
range this repo has no trustworthy number for.** That range is wider than
v1's classic-MTP band, which cuts both ways: more upside, and a genuinely
undetermined sign. This is a *measure-first* GO, not a *ship-it* GO.

> **REVIEWER FLAG (carried forward from P4v1):** the consult-review of
> this cost model **timed out in P4v1 and was never completed.** The
> model has not been independently reviewed by a second opinion. Its
> load-bearing assumptions are (i) `N` flat in `k`, (ii) `(k+1)·A` as the
> rowseq multiplier, (iii) `D` flat in `k` for DSpark, and (iv) a
> `D=8 ms` placeholder with **zero** empirical anchor. **A reviewer should
> attack (iv) first — it is a guess, and the whole surface moves with it.**

---

## §D. THE P4v2 PLAN (primary; classic MTP demoted to fallback footnote)

**Objective: measure `D`, `a`, and the effective-`k` distribution for the
DSpark head at TP=2 under production depth — before one speculative token
reaches a user.**

### D.0 Milestone 0 — reconcile config (no code)

`start_cluster.sh` intends `EXO_SPECULATIVE=1` (`:382`) but the live
process has `EXO_SPECULATIVE=0` on both nodes. Find the override. Also
note `EXO_DSV4_MTP:=0` (`:459`) must become `1` — **it is a hard
prerequisite for the DSpark TP path** (§A.3), which is not documented
anywhere and is the most likely way this project silently measures
nothing. Deliverable: a `ps eww`-verified three-flag config
(`EXO_SPECULATIVE=1 EXO_DSV4_MTP=1 EXO_DSV4_DSPARK=1`) on both nodes.
**Size: 0 code, 0.25 session.**

### D.1 Milestone 1 — the smallest live increment

**Does the code support shadow/suppressed speculation today? Checked:
NO.** There is no env gate that runs draft+verify and discards the
result. `_speculative_next` commits accepted tokens unconditionally.
`EXO_DSV4_MTP_VERIFY_AUDIT` (`dsv4_mtp.py:3994+`) is a *diagnostic on the
committed path*, not a suppressor. So a minimal env-gated patch is
required:

```
EXO_DSV4_SPEC_SHADOW=1   (new, default 0)
```

Behaviour: run the DSpark draft and the verify forward exactly as today;
log the metrics below; then **commit only the single token the anchor
position's verify logits produce** (`verify_logits[:,0,:]` argmax — which
IS the sequential-decode token by rowseq's own bitwise-equivalence
guarantee) and roll the caches back as if `n_accepted = 0`. The rollback
machinery for exactly this already exists (`dsv4_mtp.py:3596-3617` pool
snapshots + ring `save_spec_state`).

Log per cycle, **derived only from tensors the real forward already
materialized** (the skill's hard rule — no extra `model()` calls, no
extra sync, budget against jaccl's hardcoded 15 s drain deadline):
`k_drafted` (post-τ), `k_confidence_pruned_from` (always 5),
`n_would_accept`, per-position accept mask, `draft_ms`, `verify_ms`,
`L_q`, and the fence-arming state.

**Hard self-identifying gate:** with `SHADOW=1`, output must be
**byte-identical** to a `EXO_SPECULATIVE=0` run on a fixed temp=0 prompt.
If it is not, the measurement is invalid and says so on its own.
**Size: ~60-80 lines in `dsv4_mtp.py`, 0.5-1 session.**

### D.2 The fence-owner confound CONTROL

`DSv4MTPPredictor.__init__` calls `_register_fence_async_owner("cache")`
(`dsv4_mtp.py:849-852`), flipping `cache` from *defaulted-satisfied* to
*genuinely required* process-wide. Enabling MTP therefore toggles a
fence-arming regime worth **+23-67% decode** — completely swamping the
speculation effect.

> **CONTROL: never A/B `MTP=0` vs `MTP=1`.**
> The only valid comparison is
> **`MTP=1,DSPARK=1,SHADOW=1` (spec suppressed)** vs
> **`MTP=1,DSPARK=1,SHADOW=0` (spec active)**.
> Both arms construct the predictor, so both arms carry the identical
> fence-owner regime; the only difference is whether accepted tokens
> commit. Confirm real arming state per arm via
> `EXO_DSV4_FENCE_GATE_DIAG=1` (`deepseek_v4.py:3068-3080`) — do not
> assume it.

The suppressed arm doubles as the honest baseline: it is a
speculation-free decode that pays every fence, tap, and draft cost, so
`(active − suppressed)` isolates the speculative win exactly.

### D.3 The silent-partial-load GUARD

Before trusting **any** acceptance number, assert the head that loaded is
the head you think loaded. `strict=True` (`utils_mlx.py:807`, `:953`)
already catches key/shape mismatch — the uncaught failure is a
**shape-compatible, distribution-mismatched** head (§A.4). Exact check,
to run immediately after overlay and to log unconditionally:

1. **Provenance assert** — record which overlay ran
   (`_overlay_dsv4_dspark` vs `_overlay_dsv4_dspark_native`), the
   resolved source path, and — for the native path — that the source
   `model_path` is **byte-identical to the serving checkpoint path**.
   Fail loudly if the local-head path is used while serving `-0731`
   (today's state) unless explicitly acknowledged via an env override.
2. **Param-tree assert** — `tree_flatten(mod.parameters())`; assert
   `len(loaded_keys) == len(expected_keys)` **and** `set` equality
   against `DeepseekV4DSparkModule(inner.args)`'s own freshly-constructed
   tree, `0 missing / 0 extra` (the standalone-validation pattern from
   `dsv4-0731-dspark-native-head-plan-2026-08-03.md:41-46`). **Assert
   the count; do not hardcode a remembered constant** — the "81 params"
   figure in circulation did not reproduce against the `-0731` index
   (4,705 on-disk `mtp.*` keys, §A.4).
3. **Config assert** — `block_size`, `markov_rank`, `target_layer_ids`,
   `n_mtp_layers` from the head's config must match what
   `set_dspark_taps` armed and what `_speculative_next` reads. A tap-id
   mismatch silently yields `get_dspark_ctx() → None` (`deepseek_v4.py:621-628`),
   which **disables ctx conditioning without any error** — drafts still
   flow, acceptance is just quietly worse. This is the highest-value
   assert of the three.
4. **Non-degenerate-draft assert** — over the first 100 shadow cycles,
   assert the drafted-token distribution is not constant and that
   `noise_token_id` (128799) never appears in a draft. A
   silently-mis-conditioned head degenerates this way.

**Size: ~30-40 lines, `utils_mlx.py` + `dsv4_mtp.py`. 0.25 session.**

### D.4 Pre-registered decision gate

v1's gate was written for k=1 chained MTP, where `p` and mean-accept
coincide. DSpark's k∈[1,5] geometry breaks that identity, so the gate is
restated on **mean accepted tokens per cycle `a`**, measured at the
**τ-pruned effective k**, at **both** 100K and 352.6K, ≥3 runs each.
Thresholds are read off §C.3 at the **measured** `D` (not the 8 ms
placeholder) — if measured `D` lands outside the 4-12 ms sweep, the
surface is recomputed before the gate is applied.

Let `a*` = break-even from §C.3 at measured `D` and the observed modal `k`.

| Condition | Decision |
|---|---|
| `a ≥ a* + 0.30` at **both** depths | **PROCEED** to wire speculation into the stream, behind a needle-in-haystack correctness gate + a fresh A/B/A on `math_digit_sum` (`2d696ff60`'s own re-trust condition) |
| `a* ≤ a < a* + 0.30` | **HOLD.** Positive but inside this fork's demonstrated reproducibility risk (two "champion" claims that failed to reproduce, one cratering 32.3 → 4.3 tok/s). Park, or fund the rowseq-multiplier work instead |
| `a < a*` at either depth | **NO-GO** at that `k`. Retry once at a higher τ (smaller k); if the smallest k still fails, **NO-GO permanently for DSpark-at-TP on this checkpoint** — record it in §5.3 so it is never re-litigated |

For orientation against v1's p-based gate: at D=8ms, k=2, the
`a ≥ a*+0.30` proceed threshold is `a ≥ 1.49`, i.e. geometric
**p ≈ 0.75** — deliberately stricter than v1's `p ≥ 0.70`, because
DSpark's `D` is unmeasured and its gain surface is steeper.

**Falsification, stated up front:** if measured verify cost exceeds
`(k+1)·A + N` by more than ~15% at either depth, the §C.2 cost model is
wrong and every number above must be re-derived before any decision is
taken.

**Explicitly NOT in this phase:** running at full `block_size=5`
(arithmetic rejects it, §C.5, and it breaks the c=2 rowseq gate);
concurrent (c=2) speculation; any rowseq/FULLBLOCK relaxation.

### D.5 Sizing

| Milestone | Size |
|---|---|
| D.0 config reconciliation (`ps eww`-verified 3-flag env) | 0.25 session |
| D.1 `EXO_DSV4_SPEC_SHADOW` + metrics (~60-80 lines) | 0.5-1 session |
| D.3 load guard (~30-40 lines) | 0.25 session |
| D.2 live depth matrix, both arms, ≥3 runs, 2 depths | 0.5 session |
| **Total to a decision** | **1.5-2 sessions** |

### D.6 Classic single-head MTP — demoted to fallback (footnote)

Retained only as a contingency. Take it up **only if** DSpark fails the
D.4 gate at every `k` *and* the failure is attributable to the head
(depressed acceptance) rather than to the verify floor (which classic MTP
shares and cannot escape). v1's arithmetic still applies to it unchanged:
k=1 break-even p\* ≈ 0.566/0.560, measured ~0.66 → +6.0/+6.4%; k=2 loses.
Note its blocker is unchanged and unresolved: **no `-0731`-matched
classic MTP head exists** — only the preview-vintage dedicated repo — so
it inherits the same checkpoint-mismatch problem as today's local DSpark
head, without DSpark's escape hatch (`EXO_DSV4_DSPARK_NATIVE=1`, which
points at a head genuinely bundled with `-0731`).

---
---

# ORIGINAL P4 (v1) MEMO — PRESERVED VERBATIM BELOW

*Superseded in ordering and in its DSpark verdict by §A-§D above. Its
cost model, verify-floor finding, risk register, and measurement
discipline remain valid and are carried forward.*

**Scope:** READ-ONLY feasibility scoping of speculative decode (classic
single-head MTP first, DSpark second) for the **TP=2** sharding mode — the
standing "unrealized throughput upside" gap flagged in
`PERFORMANCE_HISTORY.md` §1 (lines 86-94).

**Method constraints:** No cluster access, no generations, no `.py` edits.
A concurrent worker owned the live cluster for xctrace captures during this
work. Every number below is cited from an existing doc or computed from
cited numbers; **zero new measurements were taken.** Every wiring claim
carries a `file:line`.

**Build under analysis:** exo `4ac0dbd7a`, mlx-lm submodule `0854b39`,
model `deepseek-ai/DeepSeek-V4-Flash-0731` fp8, 43 layers, TP=2 across
2× Mac Studio M4 Max over TB5 RDMA (jaccl).

---

## VERDICT (read this first)

**GO — but a narrowly-scoped, measurement-only prototype, and NOT for the
reason the §1 gap statement implies.** Two corrections to the premise, then
the arithmetic.

| | Finding |
|---|---|
| **§1's premise is half wrong** | The TP draft+verify path is **already fully wired and live-reachable**. `DSv4MTPBatchGenerator._speculative_next` (`dsv4_mtp.py:3415`) is the TP-mode speculative cycle; it is constructed at `batch_generate.py:847` and dispatched every step via `self._mlx_gen.next()` at `batch_generate.py:4228`. **Nothing needs to be built.** The gap is two flipped-off env vars and one absent weights file. |
| **The verify floor is NOT the collective** | `all_sum` traffic for a k-draft batched verify is **identical-or-better** than k+1 sequential decodes (43 calls either way vs (k+1)×43; see §3.3). The verify floor at TP=2 is **row-sequential attention**, a correctness requirement, not a transport cost. |
| **k=1 clears break-even; k=2 does not** | k=1: break-even p\* = **0.566** @100K / **0.560** @352.6K; measured acceptance **0.66** → **+6.0% / +6.4%**. k=2: break-even mean_accept\* = **1.133**; measured **1.04** → **−4.4% / −3.8%**, a LOSS. |
| **The margin is thin and rides on a checkpoint-mismatched number** | 0.66 vs 0.566 is a **0.094-wide** acceptance band. The 0.66 was measured in June 2026 on the **original** Flash checkpoint, not `-0731`. On `-0731`, a checkpoint-mismatched head measured **~0.64** acceptance. A 0.10 acceptance miss puts the whole thing at zero. |
| **DSpark second: NO-GO for now** | DSpark's block_size=5 lands it in the k≥2 regime that the arithmetic already rejects, and it carries the unresolved 15.9x context-depth collapse (§5.3). Do not schedule DSpark until k=1 MTP has produced live acceptance data. |

**Engineering size:** ~0 for the code; **1-2 sessions** for the weights
question + the measurement harness + a live acceptance/verify-cost capture.
The expensive part is not implementation, it is *earning the right to trust
the acceptance number*.

**Recommended increment:** draft-and-verify with **acceptance logging and
speculation output DISABLED** — measure live acceptance rate and real verify
cost at 100K and 352.6K before a single token of speculation reaches the
stream. Defined in §5.

---

## 1. What §1 says vs. what the code says

§1 (lines 86-94) states MTP/DSpark is "NOT wired up for this cluster's TP
sharding mode," citing zero `"PP speculation using DSpark"` log lines against
12 `"DSpark ctx warmed"` lines.

**Both observations are correct. The inference drawn from them is not.**

### 1.1 The log-line evidence is about the PP path only

`"PP speculation using DSpark"` is emitted at
`batch_generate.py:3466`, inside `_start_pp_spec_gen`, which dispatches
`pp_dspark_decode_loop`. That whole block is gated on
`get_pipeline_info(self.model) is not None` (`batch_generate.py:980`), and
`get_pipeline_info` returns non-None **only** when it finds a
`PipelineLastLayer` instance in `model.layers`
(`pp_speculation.py:133-140`):

```python
def get_pipeline_info(model: nn.Module) -> tuple[int, int, mx.distributed.Group] | None:
    for layer in model.layers:
        if isinstance(layer, PipelineLastLayer):
            return (layer.r, layer.s, layer.group)
    return None
```

Under `DSV4_SHARDING=Tensor` (the default since 2026-08-16,
`start_cluster.sh:346`), no `PipelineLastLayer` is ever installed — the load
path takes `case TensorShardMetadata()` at `utils_mlx.py:442`, not
`case PipelineShardMetadata()` at `:445`. So `pp_dspark_decode_loop` is
correctly, by design, unreachable at TP. **Zero such log lines is the
expected, correct signal, not evidence of a missing capability.**

This is exactly the §12/perf-hypothesis-discipline check #1 — *grep where
the guard is SET, not just where it is read.*

### 1.2 The "DSpark ctx warmed" lines are NOT stray module warmup

The 12 `"DSpark ctx warmed"` lines come from `batch_generate.py:2760`,
inside `submit()` — the **TP request path**, not a load-time warmup. That
block builds a real DSpark ctx-KV cache and appends real projected hidden
states:

```python
_dspark_mod = getattr(getattr(self._mlx_gen, "model", None), "model", None)
_dspark_mod = getattr(_dspark_mod, "dspark", None)          # :2748-2749
if _dspark_mod is not None:
    self._mlx_gen._dspark_caches = _dspark_mod.make_cache()  # :2753
    _ds_ctx = get_dspark_ctx(_dspark_mod.target_layer_ids)   # :2754
    ...
        _dspark_mod.append_ctx(_ds_ctx, self._mlx_gen._dspark_caches)  # :2758
        logger.info(f"DSpark ctx warmed ({_ds_ctx.shape[1]} positions)") # :2760
```

Those warmed caches are consumed by `_dspark_caches` in
`dsv4_mtp.py:3484-3486` — inside `_speculative_next`, the **TP** speculative
cycle. So the TP-side DSpark plumbing is live end-to-end; it simply never
fires because `_speculative_next` itself is never reached (see §1.3).

### 1.3 The actual gate — a plain env var, nothing structural

The TP decode loop is `ExoBatchGenerator.step()` → `self._mlx_gen.next()`
(`batch_generate.py:4228`), in the `else` branch after `_batched_decode_active`
and `was_pp_spec_step` are both false — the ordinary TP case.

Which generator `self._mlx_gen` *is* gets decided once, in
`__post_init__`:

| Location | Condition | Result |
|---|---|---|
| `batch_generate.py:813` | `use_speculative = EXO_SPECULATIVE == "1"` | master switch |
| `batch_generate.py:830-834` | `is_dsv4_with_mtp` = inner is `DeepseekV4Model` **and** `hasattr(inner, "mtp")` **and** `len(inner.mtp) > 0` | the real gate |
| `batch_generate.py:847` | if true → `self._mlx_gen = DSv4MTPBatchGenerator(...)` | **speculative TP generator** |
| `batch_generate.py:876` | elif DSv4 → `MlxBatchGenerator(...)` | plain sequential decode |

`inner.mtp` exists only if `deepseek_v4.py:6500-6507` created it:

```python
if (
    config.num_nextn_predict_layers > 0
    and os.environ.get("EXO_DSV4_MTP", "0") == "1"
):
    self.mtp = [DeepseekV4MTPModule(config, i) for i in ...]
```

Production runs `EXO_DSV4_MTP=0` (`start_cluster.sh:459`). So `inner.mtp`
never exists → `is_dsv4_with_mtp` is False → `MlxBatchGenerator` →
`_speculative_next` never runs.

**`EXO_DSV4_MTP=1` is the entire switch.** With it set, `DSv4MTPBatchGenerator`
is constructed, `_speculative_next` runs every step, and DSpark (already
attached at `utils_mlx.py:810`, gated by the already-set
`EXO_DSV4_DSPARK=1` at `utils_mlx.py:383`) takes over the draft inside it via
`dsv4_mtp.py:3481-3483`.

> **Correction for §1 of `PERFORMANCE_HISTORY.md`:** the sentence "DSpark/MTP
> speculative decode ... not yet attempted for TP" should read "not currently
> *enabled* for TP; the machinery is wired and reachable via `EXO_DSV4_MTP=1`,
> and was the production default until 2026-07-26 (`2d696ff60`)." The
> capability was *turned off for a correctness bug*, never absent.

### 1.4 The one thing that IS genuinely missing: the weights

This is the real blocker, and it is not mentioned in §1.

`-0731`'s bundled `mtp.*` tensors are a **3-stage DSpark head**, not a
classic 1-head MTP head — 81 params across 3 stages
(`dsv4-0731-dspark-native-head-plan-2026-08-03.md:43-44`). But
`deepseek_v4.py:6504` builds `num_nextn_predict_layers` (= 1, per `-0731`'s
`config.json`) `DeepseekV4MTPModule`s, and `sanitize()` keeps `mtp.0.*`
under that same count (`:7005-7011`). `DeepseekV4MTPModule` and
`DeepseekV4DSparkStage` have **different parameter trees** —
`DeepseekV4MTPModule` owns `enorm/hnorm/e_proj/h_proj`
(`deepseek_v4.py:5196-5200`) which the DSpark stage does not
(`:6314-6324`). Loading is `strict=False` (`utils_mlx.py:349`), so a
mismatch loads **partially and silently** — the exact failure class that
produced confidently-wrong output in the `hc_attn`/`attn_hc` incident
(§11).

The escape hatch already exists and is default-on:
`EXO_DSV4_MTP_DEDICATED=1` (`start_cluster.sh:468`) overlays
`mlx-community/DeepSeek-V4-Flash-MTP-bf16` onto `mtp[0]` before sharding
(`utils_mlx.py:359-367`, `:594-619`). **But that head is preview-vintage,
trained on the original Flash checkpoint's hidden-state distribution, not
`-0731`'s** — precisely the mismatch the DSpark native-head plan blames for
depressed acceptance (`dsv4-0731-dspark-native-head-plan-2026-08-03.md:63-68`).

Neither repo is present in the local HF cache (checked: no `*MTP*` or
`*DSpark*` directory under `~/.cache/huggingface/hub/`), so the first live
run will attempt an `hf_hub_download` at load time (`utils_mlx.py:620`).

**This is the BLOCKED-ON component of an otherwise-GO verdict**, and it is
the single largest threat to the acceptance number the whole economics
rides on. Step 0 of §5 resolves it.

---

## 2. Code map

| Component | Location | Notes |
|---|---|---|
| **TP decode loop** | `batch_generate.py:4228` (`self._mlx_gen.next()`) | the ordinary TP path; `step()` at `:4203-4230` |
| **TP speculative cycle** | `dsv4_mtp.py:3415` `_speculative_next(uid)` | draft → verify → accept → rollback, TP flavor |
| **TP spec dispatch** | `mtp_batch_generator.py:117-127` | `spec_eligible` requires BS=1, no pending prefill, no unprocessed seqs |
| **TP batched (BS>1) spec** | `dsv4_mtp.py:2211` `_speculative_next_batch` | min-acceptance strategy across uids |
| **Generator selection** | `batch_generate.py:830-847` | `is_dsv4_with_mtp` gate |
| **TP sharding group lookup** | `dsv4_mtp.py:1761-1791` | reads `layer0.ffn.sharding_group` — TP-specific, no PP analogue |
| **PP-only DSpark loop** | `pp_speculation.py:1697` `pp_dspark_decode_loop` | unreachable at TP; see §1.1 |
| **PP-only gate** | `pp_speculation.py:133-140` + `batch_generate.py:980` | `isinstance(layer, PipelineLastLayer)` |
| **Classic MTP head module** | `deepseek_v4.py:5166` `DeepseekV4MTPModule` | one decoder block + e_proj/h_proj; shares embed/lm_head with target |
| **MTP head construction** | `deepseek_v4.py:6500-6507` | gated `EXO_DSV4_MTP=1` |
| **MTP weight retention** | `deepseek_v4.py:7005-7011` (`sanitize`) | same gate; strips `mtp.*` when off |
| **Dedicated-head overlay** | `utils_mlx.py:359-367`, `:594` | pre-sharding, so TP shards it like the native head |
| **DSpark head attach** | `utils_mlx.py:383`, `:810` (`inner.dspark = mod`) | `EXO_DSV4_DSPARK=1`, already set in production |
| **DSpark draft, TP path** | `dsv4_mtp.py:3481-3541` | rebinds `gamma = _dspark.block_size` for the whole cycle |
| **Row-seq verify (the floor)** | `deepseek_v4.py:5095-5107` | the correctness constraint driving all the economics |
| **`moe.all_sum`** | `deepseek_v4.py:3005-3007` | the complete decode-time collective inventory |
| **Async fence gate** | `deepseek_v4.py:3046-3061` | `L <= 8` — verify at k≤2 stays inside it (§4.3) |
| **Fence "cache" owner** | `dsv4_mtp.py:850-852`, `:909` | registered by `DSv4MTPPredictor.__init__` |

### 2.1 What the verify step looks like under TP=2

Both nodes participate in every verify — it is a whole-model forward.
Concretely, per verify:

1. Draft: `k` chained MTP `predict()` calls (`dsv4_mtp.py:3465-3475`), each
   one decoder block + one `lm_head`. Cross-rank draft sync rides the
   **coord subgroup**, not the model TP group (`dsv4_mtp.py:3443`,
   `:3496-3499`) — an isolated `next_call_id_` space, the 2026-05-07 c=2
   race fix.
2. Verify: ONE forward at `L_q = k+1` over all 43 layers. Under
   `EXO_DSV4_VERIFY_ROWSEQ=1` + `EXO_DSV4_ROWSEQ_FULLBLOCK=1` (both
   default-on, `start_cluster.sh:303`/`:283`), the attention block **and**
   the hc ops/norms run **per row** as `(B,1)` decode-path calls with
   per-row cache updates; only the MoE ffn stays batched
   (`deepseek_v4.py:5107-5117`, `:1578-1585`).
3. Accept/rollback: `cache.trim(rollback)` per prompt-cache entry
   (`dsv4_mtp.py:3421-3423`).

**The rowseq requirement is what makes this expensive, and it is not
optional.** It is the shipped fix for two reproduced production
correctness bugs (§5.3, §11) — batched L>1 verify is *not* numerically
equal to L sequential decode steps on this model.

---

## 3. Economics

All inputs cited. No new measurements.

### 3.1 Inputs

| Quantity | Value | Source |
|---|---|---|
| Decode @100K | 28.94 tok/s = **34.55 ms/tok** | `PERFORMANCE_HISTORY.md:2881` |
| Decode @352.6K | 25.32 tok/s = **39.49 ms/tok** | `PERFORMANCE_HISTORY.md:2880` |
| 43-layer attention path @100K | **16.57 ms/tok** | P3 worker C, `PERFORMANCE_HISTORY.md:2469` |
| 43-layer attention path @352.6K | **19.13 ms/tok** | same |
| `moe.all_sum` | 43 calls/tok, 8192 B, 36.1/36.0 µs median | `PERFORMANCE_HISTORY.md:512`, `:1946` |
| `all_sum` total | **1.55-2.85 ms/tok** | `decode-time-budget-synthesis-2026-08-22.md` |
| Straggler idle (P6) | **~1.7%** of decode wall, upper bound | `PERFORMANCE_HISTORY.md:1946` |
| MTP acceptance, γ=2 | hist 0:34% 1:28% 2:38%, **mean 1.04/2** | `deepseek-v4-mtp-performance.md:97` |
| Prefill | 359.7 / 348.2 / 324.1 tok/s @100K/300K/500K | `PERFORMANCE_HISTORY.md:71-73` |

Attention is **48.0%** of the per-token budget @100K and **48.4%** @352.6K —
consistent with worker C's own "45-46% of B1's live per-token budget at both
depths" (`PERFORMANCE_HISTORY.md:2479`). Two independent instruments agree,
so `A` is a trustworthy input.

### 3.2 Cost model

For a k-draft cycle under row-sequential verify:

```
C(k) = (k+1)·A  +  N  +  k·d
```

- `A` — 43-layer attention path (per-token, B=1, L_q=1). Multiplied by
  `(k+1)` because rowseq runs exactly `(k+1)` such passes. This is
  defensible *by rowseq's own design goal*: each verify row is constructed
  to be bitwise-identical to a real single-token decode step
  (`deepseek_v4.py:5108-5112`), which is precisely what worker C measured.
- `N = T − A` — the remainder, treated **flat in k**. MoE stays batched
  (`deepseek_v4.py:1580-1582`), and MoE is the dominant term in `N`. This
  is **generous to MTP**: `FULLBLOCK=1` also per-rows the hc ops and norms,
  which live inside `N`. Real `C(k)` is therefore ≥ the model's.
- `d` — per-draft MTP head cost. Anchored from the June-2026 profile
  (verify = 93.4% of an 81.7 ms γ=2 cycle,
  `PERFORMANCE_HISTORY.md:1185`) → draft+accept+rollback ≈ 5.4 ms for 2
  drafts ≈ **2.7 ms/draft**. Swept `d = 2-5 ms`, central 3.
- Tokens per cycle = `1 + accepted`. Break-even: `p* = C(k)/T − 1`.

**Model self-check** (perf-hypothesis-discipline #2 — *do the arithmetic
you are implying*): the codebase claims rowseq costs "~1.6x vs classic
batched verify at short ctx" (`deepseek_v4.py:1560-1562`). At short ctx
(T≈33, A=12.88, `PERFORMANCE_HISTORY.md:2469`), the model gives rowseq/classic
= **1.36x at k=1, 1.66x at k=2**. It reproduces the documented ratio. The
model is not fabricating its central multiplier.

### 3.3 The collective is not the floor

Per verify, `y` has shape `(B, L_q, 4096)` bf16 → 8192 B at `L_q=1` (matching
the measured 8192-byte call, `PERFORMANCE_HISTORY.md:512`), 16384 at
`L_q=2`, 24576 at `L_q=3`.

| | k=1 | k=2 |
|---|---|---|
| Sequential (k+1 decodes) | 86 calls | 129 calls |
| One batched verify | **43 calls** | **43 calls** |
| Ratio | 2× fewer | 3× fewer |

Even assuming payload cost scales **fully linearly** with `L_q` — the worst
case for verify, since transport at 8-24 KB is latency-dominated (36 µs
median for 8 KB) — total collective time is **identical**: 5.70 ms either
way at k=1, 8.55 ms at k=2. Any sublinearity makes verify strictly cheaper.

**Answering the brief's question directly: a k-draft verify adds ZERO extra
`all_sum` traffic per verified token versus k sequential decodes, and
plausibly reduces it.** The P6 ~1.7% straggler bound is a subset of a
collective share that is itself only 3.9-8.3% of the post-fence-fix budget
(recomputed against 34.55/39.49 ms, **not** the stale 53.48 ms denominator —
§13 flagged that recomputation).

**The verify floor at TP=2 is row-sequential attention, not the wire.**
This is the single most important structural finding in this memo: the
token-tree verify-floor argument (§5.3 — "~30 ms floor from KV attention
over long context ... the FLOOR, not marginal cost, dominates") transfers
to TP, but its mechanism here is the `(k+1)×A` rowseq multiplier, and that
multiplier is a *correctness* cost, not a topology cost. Buying TP-specific
transport improvements would not move it.

### 3.4 Break-even

**k=1 (single MTP head, one draft):**

| d | p\* @100K | p\* @352.6K |
|---|---|---|
| 2.0 ms | 0.537 | 0.535 |
| **3.0 ms** | **0.566** | **0.560** |
| 4.0 ms | 0.595 | 0.586 |
| 5.0 ms | 0.624 | 0.611 |

Measured P(first draft accepted) = 0.28 + 0.38 = **0.66**
(`deepseek-v4-mtp-performance.md:97`).

**k=2:**

| d | mean_accept\* @100K | mean_accept\* @352.6K |
|---|---|---|
| 2.0 ms | 1.075 | 1.070 |
| **3.0 ms** | **1.133** | **1.121** |
| 4.0 ms | 1.191 | 1.171 |
| 5.0 ms | 1.249 | 1.222 |

Measured mean_accept = **1.04**.

### 3.5 Result

At measured acceptance:

| d | k=1 @100K | k=1 @352.6K | k=2 @100K | k=2 @352.6K |
|---|---|---|---|---|
| 2.0 ms | +8.0% | +8.1% | −1.7% | −1.5% |
| **3.0 ms** | **+6.0%** | **+6.4%** | **−4.4%** | **−3.8%** |
| 4.0 ms | +4.1% | +4.7% | −6.9% | −6.1% |
| 5.0 ms | +2.2% | +3.0% | −9.3% | −8.2% |

**A 1-head MTP draft-verify at TP=2 clears the verify floor at k=1 and
fails it at k=2.** k=2 fails for exactly the reason token-tree failed
(§5.3): the second draft's marginal acceptance (1.04 − 0.66 = 0.38) does
not pay for a third rowseq attention pass.

Note this **inverts the γ sweet spot** from the PP-mode finding (§5.4: γ=2
best, γ=1 −6%, γ=3 −18%). Under rowseq per-row verify the marginal cost of
a draft is a full attention pass rather than one extra row of a batched
one, so the optimum moves down to k=1. Anyone carrying γ=2 forward from the
PP-era docs would be importing a stale conclusion.

### 3.6 Honest statement of the margin

**0.66 vs 0.566 is a 0.094-wide acceptance band. That band is the entire
thesis.**

Gain surface (cell = % tok/s change @100K / @352.6K):

| d\p | 0.50 | 0.55 | 0.60 | **0.66** | 0.70 | 0.75 |
|---|---|---|---|---|---|---|
| 2.0 ms | −2.4/−2.3 | +0.8/+1.0 | +4.1/+4.2 | **+8.0/+8.1** | +10.6/+10.7 | +13.8/+14.0 |
| **3.0 ms** | −4.2/−3.9 | −1.0/−0.7 | +2.1/+2.5 | **+6.0/+6.4** | +8.5/+8.9 | +11.7/+12.2 |
| 4.0 ms | −6.0/−5.4 | −2.8/−2.3 | +0.3/+0.9 | **+4.1/+4.7** | +6.6/+7.2 | +9.7/+10.4 |
| 5.0 ms | −7.7/−6.9 | −4.6/−3.8 | −1.5/−0.7 | **+2.2/+3.0** | +4.7/+5.5 | +7.7/+8.6 |

Four reasons to distrust 0.66 as a forward-looking input:

1. **Wrong checkpoint.** Measured June 2026 on the original Flash
   checkpoint. Production is `-0731`, re-post-trained for agentic tasks
   with plausibly shifted internal representations
   (`dsv4-0731-dspark-native-head-plan-2026-08-03.md:63-68`).
2. **Wrong head.** The only classic-MTP head available for `-0731` is the
   preview-vintage dedicated repo (§1.4). On `-0731`, a
   checkpoint-mismatched DSpark head measured **~0.64** acceptance
   (`dsv4-0731-dspark-native-head-plan-2026-08-03.md:50`). **At p=0.64,
   d=3 ms: +4.7% / +5.1%** — still positive, but thinner.
3. **Wrong depth.** Acceptance was never characterized at 100K/352.6K on
   this fork. §5.3's DSpark cliff is the standing proof that
   short-context speculative results do not transfer to depth.
4. **Wrong config.** Measured before `ROWSEQ_FULLBLOCK` was default-on.

**A 0.10 acceptance miss takes the entire win to zero.** For calibration,
`EXO_DSV4_POOL_GROW_STEP=256` — one env var — already delivered **+9.79%
@352.6K** (`PERFORMANCE_HISTORY.md:2762`), i.e. *more* than this whole
project's central case, for vastly less risk. That is the honest
opportunity-cost framing for the PM.

This is why the verdict is a **measurement-only** prototype and not "flip
the flag and benchmark." The one number that decides everything has never
been measured under production conditions, and this repo's history
(`PERFORMANCE_HISTORY.md:996-1014`: two "champion" claims that failed to
reproduce, one cratering 32.3 → 4.3 tok/s) is the reason to measure it
before wiring it.

### 3.7 What would make this comfortable

The `(k+1)×A` multiplier is the whole problem. Illustrative sensitivity —
**not a proposal**, and any such change would need its own losslessness
gate:

| attention multiplier at k=1 | break-even p\* | gain at p=0.66 |
|---|---|---|
| **2.00× (today, FULLBLOCK=1)** | **0.566** | **+6.0%** |
| 1.75× | 0.447 | +14.8% |
| 1.50× | 0.327 | +25.1% |

The leverage is enormous — but §5.3's MoE-side bisect precedent
(`EXO_DSV4_MOE_PARTS_ROWSEQ=shared`) is a cautionary tale: it looked like
+12% at 300-500 token prompts and turned out catastrophic at depth. Any
attention-side rowseq relaxation must be depth-tested before it is
believed. Flagged as a possible *follow-on*, explicitly out of scope here.

---

## 4. Risks

### 4.1 Correctness landmines that apply to TP

| Risk | Source | Applies to TP? |
|---|---|---|
| **Batched-verify ≠ sequential numerics** | skill RESOLVED section; §5.3 | **YES — the primary one.** All three spec mechanisms destabilized on self-verification prompts; root cause is block-batched verify itself. Mitigated (not eliminated) by rowseq+FULLBLOCK, which is *why* the economics are tight. |
| **`EXO_DSV4_MOE_PARTS_ROWSEQ=shared` residual** | `start_cluster.sh:285` | **YES.** Production default is 0.023% divergence (1/4300), **not** bitwise-zero. The skill names this the first thing to re-audit if verify-drift symptoms reappear. |
| **MTP-boundary EOS/special-token emission** | §5.4; `exo-dsv4-degeneration-sampler` | **YES.** A draft token accepted past an EOS boundary can emit a special token the sequential path would not. Verify with `EXO_PP_SPEC_FINISH_LOG=1` against real sampled token IDs — **never** by string-matching JSON output (skill: ruled-out-theories). |
| **Batch-invariance env requirements** | `start_cluster.sh:230`, `:241` | **YES, TP-specific.** `MLX_GEMV_BATCH_INVARIANT=1` and `MLX_STEEL_BATCH_INVARIANT=1` are both default-on and load-bearing. Note the rowseq gate requires `B*L ≤ 8` (`deepseek_v4.py:5102`) because qmv/qmm invariance is proven only for M=1..8 — at BS=1, k≤2 is safely inside. |
| **Cross-rank draft divergence** | §5.3 Eagle K1 | **YES, TP-specific and severe.** MLX produces ~1ulp per-rank logit drift that flips argmax on near-ties; any tensor computed from rank-local logits without the cross-rank broadcast reintroduces divergence → differing `n_accepted` → differing all_sum message sizes → **jaccl LEN_ERR wedge**. Already handled (`dsv4_mtp.py:3443`, `:3496-3499`, `:1854-1875`), but it is the thing to break by accident. |
| **Degeneration kill-switch blindness** | skill Pitfalls | **YES.** The detector needs a short-period *literal* repeat; DSpark's semantic loop varied wording and was never caught. "Kill-switch silent" ≠ "healthy." |
| **DSpark FULLBLOCK depth collapse** | §5.3 | **YES if DSpark is enabled.** 15.9× collapse, worse than no speculation at depth. Unresolved. |
| **`PPSpecAlreadyActiveError`** | skill Pitfalls | **NO.** PP-path-only (`batch_generate.py:3275-3286`). |
| **jaccl 15s RDMA drain deadline** | skill | **PARTIAL.** A hardcoded C++ constant with no env override. Less acute at TP (no PP handoff), but any added per-cycle latency on one rank still risks the other's next wait. Budget instrumentation against 15s for the whole cycle. |

### 4.2 Instrumentation is itself a hazard

The skill's central methodology finding: **do not build an inline
synchronous numerics audit.** Three attempts in one session escalated
fault → different fault → unrecovered hang. The validated safe pattern is
**derive the signal from a tensor the real forward already materialized** —
`EXO_PP_DSPARK_VERIFY_MARGIN_LOG` style, zero extra `model()` calls. §5's
harness follows that rule strictly.

### 4.3 Async-fence interaction — checked, and it holds

Worth confirming because it could have silently erased the entire baseline.
The async fence (worth +23-67% decode, §2.8) requires `y.shape[1] <= 8`
(`deepseek_v4.py:3051`). Under rowseq, the MoE ffn sees `L_q = k+1` ≤ 3, and
per-row attention calls are `(B,1)`. **k ≤ 2 stays inside the gate.**

But the `"cache"` owner key matters: `DSv4MTPPredictor.__init__` calls
`_register_fence_async_owner("cache")` (`dsv4_mtp.py:850-852`), which flips
`cache` from *defaulted-satisfied* to *genuinely required*
(`deepseek_v4.py:125-141`). From then on the fence engages only when the
predictor actually arms it. `_set_fence_async(False)` is called at
`dsv4_mtp.py:884` (init), `:945`, `:988`, `:2697`, `:4227` — each
disarm ending in `mx.synchronize()`.

**Consequence:** enabling `EXO_DSV4_MTP=1` changes async-fence arming
behavior for the whole process, including any window where speculation is
not producing tokens. **A naive A/B of `EXO_DSV4_MTP=0` vs `=1` therefore
does not isolate speculation — it also toggles a fence-arming requirement
worth up to +23-67%.** Any measurement must confirm real arming state via
`EXO_DSV4_FENCE_GATE_DIAG=1` (`deepseek_v4.py:3068-3080`) rather than
assuming it. This is precisely the §2.8 class of bug — a flag that looks
live and silently is not — and it is the most likely way this project
produces a confidently wrong number.

---

## 5. Recommended increment

**Smallest testable increment: measure acceptance and verify cost live,
with speculation output DISABLED.** Not one token of speculative output
reaches the stream until the numbers justify it.

**Step 0 — resolve the weights question (do this first; it is BLOCKED-ON).**
Offline, no cluster. Determine whether a classic 1-head MTP head exists for
`-0731` at all, or whether only the preview-vintage dedicated head is
available. Verify key/shape match against `DeepseekV4MTPModule`'s parameter
tree **before** any load, since `strict=False` (`utils_mlx.py:349`) will
otherwise load a mismatch silently. Follow the standalone-validation pattern
from `dsv4-0731-dspark-native-head-plan-2026-08-03.md:41-46` (0 missing / 0
extra of N params). **If only the mismatched head exists, the expected
acceptance is ~0.64, not 0.66, and the memo's central case drops to
+4.7/+5.1%** — the PM should know that before authorizing cluster time.

**Step 1 — acceptance + verify-cost logging, speculation output OFF.**
Add a diagnostic-only path that drafts, verifies, logs, and then **discards
the speculative result**, committing the token the sequential path would
have produced. Derive every logged quantity from tensors the real forward
already materialized (§4.2). Log per cycle: `n_accepted`, the accept
histogram, real draft ms, real verify ms, `L_q`, and the fence-arming state.
Success criterion: byte-identical output vs `EXO_DSV4_MTP=0` on a fixed
temp=0 prompt — if speculation is discarded, output **must** be unchanged,
and that is a hard gate that makes an invalid run self-identifying
(§12: *build the gate before the measurement*).

**Step 2 — measure at real depth.** 100K **and** 352.6K, ≥3 runs each,
EOS genuinely banned via `/bench/chat/completions` (the
`bench=True`-routing quirk, `PERFORMANCE_HISTORY.md:2445-2452`), on
`bench/p3_depth_anchor_probe.py`. Confirm the env actually reached the
runner via `ps eww` on both nodes — the §2 check that silently made two
A/B arms identical. Depth-testing is non-negotiable: §5.3's cliff is the
standing proof that short-context speculative results do not transfer.

**Pre-registered decision gate — write this down before running:**

- Live p ≥ **0.70** at both depths → proceed to wire speculation into the
  token stream, with a needle-in-haystack correctness gate.
- 0.60 ≤ p < 0.70 → the win is ≤ ~+6% and inside this fork's demonstrated
  reproducibility risk. **Do not ship.** Either park it, or fund the §3.7
  rowseq-relaxation investigation first — that is where the real leverage
  is.
- p < 0.60 → **NO-GO, permanently, for the k=1 MTP-for-TP lever on this
  checkpoint.** Record it in §5.3 alongside token-tree so it is never
  re-litigated.

**Falsification stated up front:** if measured verify cost exceeds
`2·A + N` by more than ~15% at either depth, the cost model in §3.2 is
wrong and every number in this memo must be re-derived before any decision.

**Do NOT run k=2 or DSpark in this phase.** The arithmetic already rejects
k≥2 (§3.5); running it would only re-measure a known loss while adding
DSpark's unresolved depth cliff.

### Engineering estimate

| Task | Size |
|---|---|
| Step 0: weights resolution + offline key/shape validation | 0.5 session |
| Step 1: acceptance/verify-cost logging, output disabled | 0.5-1 session |
| Step 2: live depth-matrix measurement + write-up | 0.5 session |
| **Total to a decision** | **1.5-2 sessions** |
| *If gate passes*: wire into token stream + needle gates | +1-2 sessions |

Small — **because the code already exists.** The cost is measurement
discipline, not implementation.

---

## 6. Summary for the PM

1. **The §1 "gap" is mostly a flipped-off env var, not missing code.** The
   TP speculative path is wired and reachable (`dsv4_mtp.py:3415`,
   dispatched from `batch_generate.py:847`); `EXO_DSV4_MTP=1` is the switch.
   It was the production default until 2026-07-26 and was turned off for a
   correctness bug (`2d696ff60`), never absent.
2. **What IS missing is a `-0731`-matched classic MTP head.** That is the
   real blocker and the main threat to the acceptance assumption.
3. **The verify floor at TP=2 is row-sequential attention, not the
   collective.** A k-draft verify adds zero extra `all_sum` traffic — it
   uses 2-3× *fewer* calls. Transport optimization cannot help here.
4. **k=1 clears break-even; k=2 does not.** +6.0/+6.4% at measured
   acceptance; k=2 loses 3.8-4.4%. The PP-era γ=2 sweet spot does **not**
   carry over.
5. **The margin is 0.094 acceptance points wide**, and the 0.66 it rides on
   was measured on a different checkpoint, a different head, a different
   depth, and a different rowseq config. A one-line env change already
   delivered +9.79% on this cluster for far less risk.
6. **GO for a measurement-only prototype with a pre-registered decision
   gate.** 1.5-2 sessions to a defensible answer.

---

## Appendix: corrections proposed to `PERFORMANCE_HISTORY.md`

Not applied by this memo (docs-only commit, and §1 is the consolidated
reference — proposing rather than editing):

- **§1, lines 86-94.** "NOT wired up for this cluster's TP sharding mode"
  → the TP machinery IS wired (`dsv4_mtp.py:3415`, `batch_generate.py:847`);
  it is disabled by `EXO_DSV4_MTP=0` (`start_cluster.sh:459`) and blocked
  by the absence of a `-0731`-matched classic MTP head. The zero
  `"PP speculation using DSpark"` lines are the *expected* signal at TP
  (that log line is PP-path-only, `batch_generate.py:3466`), not evidence
  of a missing capability.
- **§1, same block.** The 12 `"DSpark ctx warmed"` lines are emitted from
  the **TP request path** (`batch_generate.py:2760`, inside `submit()`),
  not a load-time warmup — TP-side DSpark plumbing is live end-to-end.
- **§5.4.** The "γ=2 sweet spot" is a PP-mode/batched-verify finding. Under
  TP with rowseq per-row verify, the arithmetic in §3.5 puts the optimum at
  **k=1**; γ=2 is a net loss. Worth a pointer so the stale conclusion is
  not carried forward.
