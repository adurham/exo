# TOKEN63 — LEAD-B0 findings + B1 single-boot differential probe design

Status: **offline design + B0 recon only.** No cluster access, no boot, no live/installed
code patched. Repo read: `mlx_lm/models/deepseek_v41/` in `/private/tmp/next18-lever2/`
(mlx-lm `deploy/next18-lever2 @ 16830e1`, the shipped commit). Harness written to
`/private/tmp/p20-b-probe/` (scratch, NOT deployed, NOT committed).

Scope: close the token-63 **understanding** loop and harden the harness. The lever stays
shipped (owner ruling: R8a battery governs). This doc does not re-litigate the ship.

---

## §B0 — offline findings

### B0.1 Margin analysis — VERDICT: **NOT POSSIBLE from existing data**

The brief asks to extract the top-1 minus top-2 logit margin at token 63 from the pre-ship
vs lever streams, to test the "benign near-tie" hypothesis offline. **No legA artifact
contains logit/logprob data**, so this extraction cannot be done; the margin must come from
the boot probe (§B1). Confirmed by reading every legA file the brief named:

| file | top-level keys | logprobs/logits? |
|---|---|---|
| `/private/tmp/p5r1/legA_base.json` (pre-ship ref) | `label, prompt_sha, prompt_tokens, max_tokens, runs, comparisons, all_text_identical, all_ids_identical` | **no** |
| `/private/tmp/p5r1/legA_prod_control.json` (prod⋈prod control) | same | **no** |
| `/private/tmp/p5r1/legA_restore.json` | same | **no** |
| `/private/tmp/p5r1/legA_prompt.json` | `prompt, depth, task, salt` (the prompt itself) | **no** |
| `/Users/.../scratch/p5/r1b/legA_lever.json` (lever) | same run schema | **no** |
| `/Users/.../scratch/p5/r1b/legA_r1b_restore.json` (restored prod) | same | **no** |
| `/private/tmp/p5r1/legA_probe_tiny.json` | `wall_s, deltas, finish, stats` (timing only) | **no** |
| `/Users/.../scratch/p5/r1/legA_*.json` | **does not exist** (no `r1/` legA_ files) | — |

Each `runs[i]` carries only: `i, wall_s, finish, sha_think, sha_content, think_chars,
content_chars, ndelta_thinking, ndelta_content, generation_tokens, prompt_tokens, ids[],
n_ids, thinking, content`. The only "score-like" fields are sha hashes of the produced text.
`ids[]` is 300 ints — token identities, no probabilities attached.

**Honest statement:** the "benign near-tie" hypothesis (that token 63 was a near-tie the
fp32 row broke differently from bf16) **cannot be tested offline** — there is no logit row
to compute a margin from. `token_63` itself is known: id `23393` (production) vs `5789`
(lever), a *large* id-distance, both continuations coherent ("Need verify." vs "Need avoid
mistakes."). A large id-distance is *consistent with* a near-tie (an argmax flip can land on
any id), but it is not evidence either way. The margin is a boot-probe deliverable.

### B0.2 Small-n guard shape check — which `n` the indexer actually sees

`_FENCE_MIN_ROWS` (default 16, `_gates.py`) gates both levers via `n` == the forward's query-row
count. The L2-full lever engages `_row_dtype(n) = fp32` iff `_L2_FULL and n <= 16`
(`indexer.py:152`). What `n` values occur:

| path | `n` | indexer called? | in guard scope | capture NS=1,4 covered? |
|---|---|---|---|---|
| **decode** (`spec.py:311-317`, `plain_step` `:292`) | **1** (anchor row) | yes | yes (`1<=16`) | **yes** |
| **verify** (`spec.py:314` `vin=[anchor;draft]`) | **g+1** where g∈{1,2,3,4} → **2,3,4,5**; warmup uses g=3 → **4** | yes | yes (`<=16`) | **only n=4** (NS=1,4); n=2,3,5 NOT sampled |
| **prefill** (`prefill.py`, `DSV41_PREFILL_CHUNK≈512`; 20073-tok prompt) | **512** (last chunk 97) | yes | **no** (`>16` → HIER/bf16 row) | no (large-n not captured) |
| **draft / MTP head** (`mtp.py`) | n/a | **NO** | out of scope | n/a |

Key B0.2 findings:

1. **The draft/MTP path never routes through `Indexer.__call__`.** `DraftAttention._kv`
   (`mtp.py:122-129`) and `DraftAttention.draft_block` (`mtp.py:136-170`) do dense local
   sliding-window attention only; the module docstring states draft attention is
   "LOCAL ONLY … no compressed-KV / indexer / candidate machinery"
   (`compress_ratios[40:43] == [0,0,0]`). So the draft head is **out of scope** of the guard
   and of the token-63 mechanism (it can still be the *conduit* of a diverged anchor via the
   verify round, but it is not a lever effect site).

2. **The m=4 verify rows (i) are in scope and were covered (as n=4).** Spec-decode verifies
   `[anchor] + draft` rows in one body forward (`spec.py:314`), so the verify indexer call is
   n = g+1, not n=4 always. **The capture only sampled n=1 and n=4**, so verify rounds with
   g=1,2,4 (n=2,3,5) were never captured. That is a real sampling gap: if token 63 landed in a
   g=2/g=3/g=4 round the n=5 (or n=3) indexer call carried it, and the NS=1,4 capture missed it.

3. **Which n held token 63 depends on the adaptive gamma, which is NOT recoverable from legA.**
   The greedy spec loop picks `g` from `GammaPolicy` (`spec.py:87`, warmup g=3, then adaptive
   over {1,2,3,4}); the committed-token stream is recorded but the per-round `g`/`n` is not in
   the artifact. So B0 cannot pin token 63's round shape — **this is exactly why the B1 probe
   records a per-call ROUTING census with the round boundaries** (see §B1).

4. **The guard's true engagement region is n∈{1,2,3,4,5}**, all `<=16`, i.e. every decode and
   verify call is inside the guard. Large-n prefill (`n≈512`) is `>16` and takes the
   hierarchical/bf16-row path — unchanged by the lever. So the lever's entire surface is
   **small-n decode+verify**, and the capture sampled only 2 of the 5 shapes in that surface.

### B0.3 The routing correction that matters (pre-ship vs lever)

The R1B post-mortem (`PHASE5-R1B-RESULTS.md` §8) settled the mechanism: the lever changes the
small-n **stored row dtype** (bf16 → fp32). The correct pre-ship comparator is **not**
`_HIER=0`: pre-ship production ran with `_HIER=1` (the default), so at small n its `Indexer.__call__`
fell through the `if _HIER and n > _FENCE_MIN_ROWS` guard to the tiled/untiled fallback storing
a **bf16** row. The lever's only change is that this same fallback stores an **fp32** row.
Therefore the sharpest single-variable differential is:

> **flip ONLY `indexer._L2_FULL`, hold `_HIER`(on) and `_FENCE_MIN_ROWS`(16) fixed.**

Both arms then run the identical code path; the only difference is `_row_dtype(n)` for `n<=16`.
That is what the B1 probe's `set_lever` does (`/private/tmp/p20-b-probe/token63_probe.py`).
(`_HIER=0` remains available as a coarse confirmation arm via `set_hier`, but it is a
multi-variable change and is not the primary differential.)

### B0.4 — DIRECT lever differential from the R1b capture (offline)

Script: `/private/tmp/p20-b-probe/token63_direct_diff.py` (raw output
`/private/tmp/p20-b-probe/direct_diff.out`). Loads the EXISTING R1b capture
`/Users/adam.durham/.hermes/cache/scratch/p5/r1b/next18_91k.m4-{1,2}.npz` and, for every
REPLAYABLE record, computes the direct comparator **`bf16-row (pre-ship fallback) vs fp32-row
(L2-full)`** — NOT the replay's `fp32-row vs hier`. `row_scores` is reused from
`bench/next18_replay_capture.py` (verified equal to `indexer._score_body_eager` + the final
`.astype(dtype)`); `topk_from_row` semantics (argpartition then **position-sort**) confirmed
from `indexer.py:303`. No boot, no ssh, no installed-code patch.

| quantity | m4-1 | m4-2 |
|---|---|---|
| records | 128 | 128 |
| replayable (source/plain) | 64 | 64 |
| **consumer (un-replayable)** | **64 (50.0 %)** | **64 (50.0 %)** |
| **`L2full_vs_bf16row_ndiff`** (positional, replay comparator) | **39,917** | **42,802** |
| `L2full_vs_hier_ndiff` | 0 | 0 |
| `bf16row_vs_hier_ndiff` | 39,917 | 42,802 |
| true column-**set** symmetric difference | **822** | **844** |
| order-sensitive: differing slots in **score-descending rank order** | 60,902 | 76,690 |
| records with a ranking-order difference | 64 / 64 | 64 / 64 |
| set-equal-but-order-different records | 0 | 9 |
| diff-type: **tie-break** records | 64 | 55 |
| diff-type: genuine ranking records | **0** | **0** |
| max bf16 gap among the swapped columns | 0.0 | 0.0 |
| slots | 131,072 | 131,072 |

**Findings.**

1. **The direct differential reproduces ~40K exactly and the triangle identity holds.**
   `L2full_vs_hier = 0` on both nodes, so `bf16row_vs_hier` ≡ `L2full_vs_bf16row` by
   construction: 39,917 (m4-1) / 42,802 (m4-2). This CONFIRMS the brief's prediction and
   shows the R1b §8 conclusion — "`L2full_vs_hier = 0` → no offline lever effect" — was
   drawn from the **wrong comparator**. The comparator production actually differs on at
   small n is `bf16-row vs fp32-row`, and it is non-zero.

2. **Magnitude, honestly.** The ~40K is the replay's comparator: `topk_from_row` runs
   `argpartition` then **sorts the k indices back into position order**, so a single
   boundary column-swap shifts every sorted element after it. The TRUE committed-selection
   change (symmetric difference of the two selected column **sets**) is only **822 / 844 of
   131,072 slots (≈0.63 %)** — i.e. the positional count is inflated ~48–50× by the sort, and
   9/64 (m4-2) records differ ONLY in order (identical column sets). The honest headline is
   therefore "~830 columns actually change membership", not "40K".

3. **Difference TYPE = pure tie-break, zero genuine re-ranking.** For every replayable
   record, the columns the two picks disagree on are scored EXACTLY EQUAL in bf16 (max bf16
   gap among swapped columns = 0.0; 0 ranking rows on either node). Mechanism: bf16 rounding
   collapses distinct fp32 near-ties at the k-th boundary into exact bf16 ties; `argpartition`
   then breaks each tie differently for the bf16 row vs the fp32 row. This is exactly the
   behaviour the `_ROW_DTYPE` docstring (`indexer.py:56-65`) predicts ("scores that round to
   the same bf16 value can flip which of them lands in the top-k"). It is NOT a
   score-ordering error — no well-separated score is ever mis-ranked.

4. **Order-sensitive result.** Ranked score-descending (rather than position-sorted), **64/64
   replayable records on BOTH nodes show a ranking-order difference** (60,902 / 76,690
   differing ranked slots). So the lever's effect is visible in the ranking sequence of
   every replayable record — the R1b replay's set/position comparator understated it.

**CONSUMER-LAYER GAP (honest).** 64/128 records on each node — **50.0 %** — are consumer
layers 24/28/32/36 (`uses_candidates=True`). The capture stores `q/w/lens/index_k/out` but
**not `shared.candidates`**, so those records cannot be replayed and the direct differential
**cannot be computed for them** (`next18_replay_capture.py:61` skips them for the same
reason). Layer 20 (the candidate source) IS replayable, so the mask that the consumers compose
over is produced-but-not-captured; without it the consumer top-k differential is unavailable.
Combined with the fact that the differential is measured on the indexer alone (no attention /
logits propagation), the offline capture cannot quantify the effect on token 63 — only that
the effect EXISTS.

**VERDICT.** The existing capture **does** show the lever (bf16→fp32 small-n row) changes
committed index selections on real production-H tensors: a tie-break-class column-set change
of ≈830 columns per node and a ranking-order difference in 100 % of replayable records. That
is **mechanistically sufficient** to drive a greedy token divergence and **falsifies the R1b
§8 "no offline lever effect" reading** (which used `fp32-vs-hier`, the wrong comparator). But
it does **NOT** offline the token-63 mechanism end-to-end: (a) the consumer half of the
records is un-replayable, so the full layer stack is not covered; (b) the raw set magnitude is
small (≈0.63 % of slots) and purely tie-break, confined to bf16 boundary ties; (c) the
indexer diff is not propagated through attention to the final logits/token. The capture
establishes **presence and type** of the lever effect offline; pinning it to token 63 still
requires the B1 single-boot differential (which also makes the consumer layers replayable).

---

## §B1 — preferred probe: single-boot runtime-toggle differential

### B1.0 Feasibility (the linchpin) — VERIFIED from source

A runtime monkeypatch of the lever globals takes effect without a reboot, because the reads
are at **call time**, not import time:

| symbol | defined | read at | evidence |
|---|---|---|---|
| `indexer._L2_FULL` | `indexer.py:139` (module global) | `indexer.py:152` inside `_row_dtype(n)`, **at call time** | `if _L2_FULL and n <= _FENCE_MIN_ROWS:` — a global-name lookup on every call |
| `indexer._HIER` | `indexer.py:178` | `indexer.py:591` inside `Indexer.__call__`, **at call time** | `if _HIER and n > _FENCE_MIN_ROWS:` |
| `indexer._SMALLN_ROW_BF16` | `indexer.py:140` | `indexer.py:153` inside `_row_dtype`, **at call time** | `return _ROW_DTYPE if _SMALLN_ROW_BF16 else mx.float32` |
| `indexer._FENCE_MIN_ROWS` | imported by-value `indexer.py:83` (`from ._gates import _FENCE_MIN_ROWS`) | `indexer.py:152/591` | must be patched as the **`indexer` attribute**, not `_gates` |
| `sparse_attention._FENCE_MIN_ROWS` | imported by-value `sparse_attention.py:74` | `sparse_attention.py:425/449` | patch the `sparse_attention` attribute too (lever-1 shares the threshold) |

`_L2_FULL`, `_HIER`, `_SMALLN_ROW_BF16` are true module globals → `IX._L2_FULL = False` is
seen by the next call. `_FENCE_MIN_ROWS` is a *by-value* import → the capture harness already
handles this (`next18_capture.py:329`: `IX._FENCE_MIN_ROWS = fence`); the B1 probe does the same.

**Verdict: runtime-toggle IS feasible — B1 preferred approach confirmed.** (Locally proven: the
unit test asserts `indexer._L2_FULL` flip changes `_row_dtype(1)` from `bfloat16` to `float32`
with no reboot — 29/29 assertions pass.)

### B1.1 What the capture boot must do (exact)

Install `token63_probe` (module in `/private/tmp/p20-b-probe/`, mirror of the R1b launcher; the
hook is the ONLY live-side addition and is inert without `DSV41_TOKEN63_CAPTURE`):

```
DSV41_TOKEN63_CAPTURE=/tmp/token63.npz
DSV41_TOKEN63_WINDOW=55:63
DSV41_TOKEN63_MAXCALLS=256
DSV41_TOKEN63_STORE_K=1
# bootstrap import: import token63_probe  (env auto-installs; or install() explicitly)
```

One process, one fixed prompt (`RM.build_prompt(20000,'r1fix-tokenid','count')`, sha
`deec4f8d1a3c71fa`), **no reboot between arms**. Driver sequence (also in
`/private/tmp/p20-b-probe/README.md`):

1. **MONITOR** — as shipped (`set_lever(True)`, `_HIER` default on). Decode to the window;
   the hook records the ROUTING census (n, path, row_dtype, is_candidate_source, layer_id)
   for every indexer call from the start, so token 63's round shape is captured.
2. **ARM off** — `cap.set_lever(False)`; **rebuild the prefix cache from scratch** (identical
   prompt/tokenization, fresh `model.make_cache`); decode the same window; flush.
3. **ARM on** — `cap.set_lever(True)`; rebuild again; decode the same window; flush.
4. **DETERMINISM replicate** — `set_lever(True)`; rebuild; decode a **third** time; the
   §determinism precondition (below) requires arm-on#2 ≡ arm-on#1 bit-for-bit.
5. **Second boot** repeats the same driver → cross-boot determinism.

Why one process: removes reboot/prompt/prefix-cache confounds; and **the consumer layers
(24/28/32/36) become exactly replayable** because `shared.candidates` is produced and consumed
in the same process — the gap that invalidated 64/128 of the R1b capture
(`next18_capture.py` never stores `shared.candidates`; `next18_replay_capture.py:61` skips any
`uses_candidates` record, and top-k is compared as a SET not as a ranking).

### B1.2 Recorded per indexer call (bounded window)

Routing/identity is recorded for **every** indexer call (`n, b, nb, k, ratio, layer_id,
start_pos, offset, is_candidate_source, uses_candidates, owns_k, L2_FULL, HIER, FENCE, path,
row_dtype, lever_on, ts`). The **tensor** payload is recorded only inside the window
`lo<=start_pos<=hi` (`DSV41_TOKEN63_WINDOW`, default `55:63`), to keep the capture bounded:

- **indexer top-k indices IN ORDER** — `topk_row_ordered`, value-descending (NOT `topk_from_row`'s
  position-sorted set, and NOT a set union). A near-tie flip shows up as a permutation; a set
  comparison hides it. (This is the specific fix for the R1b replay's set-vs-order flaw.)
- **sha256 of the stored score ROW** — `row_hash`, taken on the fp32 canonical byte view so an
  fp32 row and its bf16 sibling hash differently exactly when their values differ (the lever's
  exact surface). Plus the top-`k+4` (value, index) pairs so a boundary tie is inspectable.
- **sha256 of the attention output** — the `sparse_attn` output for the layer (requires a
  small `sparse_attn`/`Attention.__call__` hook site; specified here, to be added at capture time).
- **final logits top-5 with margins** — reuse the existing exact path
  `model(..., argmax=True, logprobs=k)` → `head.topk_logprobs` → `logprobs.combine`
  (`exl3_build.py:334`, `logprobs.py:30`). Record `top_ids[1:6]`, `top_logprobs[1:6]` (already
  log-softmax'd) and derive margins as `top_lp[0]-top_lp[1]`, `top_lp[0]-top_lp[2]`. This yields
  the token-63 margin the offline artifacts lack (§B0.1).
- **ROUND BOUNDARIES** — a lightweight wrapper on `spec.generate` records each round's
  `(start_pos, g, committed_ids, accepted_count)` (the quantities `spec.py:331-342` computes but
  does not persist), so token 63's round and its verify shape `n=g+1` are recoverable.

### B1.3 Memory / flush discipline (bounded, off-thread)

- **Off-thread flush.** `flush()` snapshots the ring on the calling thread (list copy) and hands
  it to a **daemon writer thread**; `np.savez_compressed` + `os.replace` happen there. Single-slot
  coalescing (newest snapshot superset wins). This is the R1b fix, reproduced — the R1 SIGKILL was
  an inline ~197 MB deflate starving the runner's event channel → 45 s hang-watchdog.
- **Bounded queue + drop logging.** Ring `deque(maxlen=MAXCALLS)` (default 256); overflow
  increments `dropped` and is reported in the flush log + `meta_dropped`. Ring holds **routing
  records only** (no tensors), so it never grows with context.
- **Window bound = the anti-watchdog fix.** The tensor payload is only captured for
  `start_pos ∈ [lo, hi]` (default 9 positions), NOT all 300 decode/verify calls. A full-run
  capture is what tripped the 45 s watchdog on a runner (`next18_capture.py` docstring "MEMORY"
  + R1B §2) — do NOT repeat it. Window records are ~9 positions × ≤8 index layers.
- **Stated memory budget vs the 115 GiB wired guardrail.** Per-tensor bytes: `q` n·h·d·4 ≈ n·131 KB,
  `w` n·h·4 ≈ n·0.5 KB, `lens` n·4, `index_k` nb·d·2 (bf16, `nb≈512`@token63 ≈ 131 KB),
  `topk` n·(k+4)·8 ≈ n·4 KB, one logits row 5·8 B, attn-out hashes 32 B. Per layer per position
  ≈ **265 KB** → 9 positions × 8 index layers ≈ 19 MB. **Ring budget 512 MiB**
  (`DSV41_TOKEN63_BUDGET_MB`) ⇒ the window capture is ~4% of budget and **<0.5% of the 115 GiB
  guardrail**; the writer's peak is one ring snapshot. No `mx.empty`/transient is held by the hook
  (all conversions are `.astype(...)` copies that die after `np.asarray`).
- **Bounded sidecar.** The human-readable `*.meta.jsonl` writes scalars/hashes only (R1b's was
  ~8 GB because array values were serialized to JSON — never that again).
- **Gate sentinel** (`<path>.gate`): hook is an immediate `os.path.exists` no-op while it exists,
  so the probe can never run during a timing arm.

### B1.4 Harness edits (exact)

- **NEW** `/private/tmp/p20-b-probe/token63_probe.py` — the hook (written, lint-clean, unit-tested).
- **NEW** `/private/tmp/p20-b-probe/test_token63_probe.py` — local unit test (written, 29/29 pass).
- **At capture time (not offline):** add the two small hook sites the module documents —
  `sparse_attn` output hash, and the `spec.generate` round-boundary wrapper. These touch a loaded
  model's call graph and are deliberately *specified, not written blind*.

---

## §Fidelity gate + determinism precondition (PRE-REGISTERED, verbatim)

> **FIDELITY GATE (pre-registered).** Replaying build X against X's OWN captures must reproduce
> X's captured outputs **bit-exactly, consumer layers (24/28/32/36) included**. The replay must
> compare each layer's top-k indices **in score/ranking order** (not as a set) and must carry the
> captured `shared.candidates` mask into the consumer layers so no record is skipped. If the
> harness cannot self-reproduce X's own captured outputs, the ONLY permitted report is
> **"harness invalid"** — the token-63 differential is then NOT evaluable and may NOT be reported
> as "0 diffs" or any diff count. A non-zero self-reproduction diff invalidates the harness, not
> the build.

> **DETERMINISM PRECONDITION (pre-registered).** Before any difference between the lever arm and
> the pre-ship arm is attributed to the lever, confirm, on the SAME fixed prompt:
> (a) the **same** build produces **bit-identical** output twice **within one boot** (arm-on#2 ≡
> arm-on#1, ids and sha_think identical); and
> (b) the same build produces bit-identical output **across two boots**. Both arms must also run
> the **same prompt, the same tokenization, and the same prefix-cache state** (each arm rebuilds
> its prefix cache from scratch, in-process, from the identical prompt). If determinism fails, the
> run is **inconclusive** — no attribution is made.

---

## Evidence / provenance

- Divergence re-confirmed offline: base `ids[63]=23393` vs lever `5789` (first divergence index 63;
  `legA_base.json` sha_think `e65639ce0dd6ef41` vs `legA_lever.json` `681b28c3fe24a5e3`).
- Source lines cited are from `/private/tmp/next18-lever2/` at HEAD `16830e1`
  (`indexer.py`, `_gates.py`, `mtp.py`, `spec.py`, `prefill.py`, `attention.py`, `model.py`,
  `exl3_build.py`, `logprobs.py`, `bench/next18_capture.py`, `bench/next18_replay_capture.py`).
- Local linchpin proof: `/private/tmp/p20-b-probe/test_token63_probe.py` → `RESULT: ALL PASS`.
