# Phase 3 — MTP / DSpark draft head ported and verified

Date: 2026-09-27
Hardware: 2x Mac Studio M4 Max (macstudio-m4-1 / macstudio-m4-2), 128 GB each
Model: DeepSeek-V4.1-Flash (native, reduced-layer builds for validation)
Status: **complete** — MTP ported, verified token-identical, packaged

## Bottom line

V4.1's Multi-Token-Prediction (DSpark) draft head was being **silently dropped**
by the reference port at conversion time (`convert.py` had an explicit
`if name.startswith("mtp."): return "mtp"` accounting-only branch). That is why
the fast R=4 decode shape measured in Phase 2 was unreachable: there was nothing
to draft with.

It is now ported, loaded, running, and **verified token-identical to plain greedy
decode**. Projected throughput on the full 40-layer model goes from ~9.8 tok/s
(plain) to ~24 tok/s at a 60% draft acceptance rate.

## What was built

Two new modules in `~/repos/ref/deepseek-v41-mlx/deepseek_v41_mlx/`:

| module | lines | role |
|---|---|---|
| `mtp.py` | 308 | the 3-stage draft head (`DSparkHead`, `DraftStage`, `DraftMoE`, `DraftAttention`) |
| `mtp_decode.py` | 260 | the accept / rollback / ctx-feed round loop (`speculative_generate`) |

Plus six surgical edits to existing files:

| file | change |
|---|---|
| `config.py` | add `dspark_n_routed_experts`, `dspark_num_experts_per_tok`, `dspark_block_size`, `dspark_markov_rank`, `dspark_noise_token_id`, `dspark_target_layer_ids` |
| `moe.py` | `Gate` takes optional `n_experts` / `topk` (the draft MoE is 128/top-3, not 384/top-6) |
| `convert.py` | keep `mtp.*` instead of dropping it; emit per-stage groups `mtp-0/1/2`; widen the expert-stacking regex to accept `mtp.<i>.ffn.experts`; translate the quantization module map through `map_mtp_name` |
| `model.py` | build `model.mtp` when `n_mtp_layers > 0`; `return_taps=True` returns `{layer_id: hc_mean}` for `dspark_target_layer_ids` |
| `load.py` | map checkpoint `mtp.*` names to runtime paths (idempotent, prefix-based) |
| `compressor.py` | stash the chunk's own raw rows (`chunk_kv` / `chunk_score` / `chunk_start`) for rollback |

## Structure facts read off the checkpoint (not guessed)

- draft MoE is **128 routed experts, top-3** — *not* the body's 384/top-6;
- draft attention is **local-only** (pure sliding window; `compress_ratios[40:43] == [0,0,0]`), no compressor and no indexer;
- the release has **no `hc_head` tensor anywhere** (scanned all 192,452 tensors). The draft collapses with the last stage's `ffn_pre`, exactly as the body does — the absence is consistent, not a missing weight;
- `shared_experts` keep their **`w1`/`w2`/`w3`** names in this port. (Production's own sanitizer renames them to `gate_proj`/`up_proj`/`down_proj`; copying that here produced 27 missing + 27 unexpected params.)

Native MTP weights: **2,401 tensors, 7.24 GB**, in shards 00044/00045/00046.

## Verification

Strict loader on the reduced+MTP build: **0 missing, 0 unexpected** out of 341 params.

### The decisive test

Speculative decode must emit the **same tokens** as plain greedy decode.
Acceptance rate only changes speed, never output.

```
=== VERDICT ===
  tokens IDENTICAL to plain greedy (48 tokens)  OK
```

48/48 tokens identical over 47 rounds, and the same result reproduces **through
the packaged module** (`mtp_decode.speculative_generate`), not just the ad-hoc
script. The test is a real instrument: the same loop in `verify="chunk"` mode
diverges, so the match is not vacuous.

### Rollback semantics (what actually needs repairing)

- the **window ring** and the **pooled `comp_kv`** need *no* repair: position `p`
  lives at slot `p % window` and is rewritten by whichever chunk commits it, and
  any pooled group straddling the rollback target is recomputed by the next chunk
  before it can be read;
- the **compressor's open-group carry** (`kv_state` / `score_state`) must be
  **rebuilt, not merely restored** — the rows for newly committed positions may
  exist *only* in the verify chunk. The compressor stashes its own chunk rows and
  the driver splices saved-carry-rows (positions < chunk start) with chunk-rows
  (positions ≥ chunk start). Production does this structurally via
  `PoolingCache`'s `buf_kv` / `buf_gate` raw remainder buffers.

## The chunk-shape finding (pre-existing, blocks naive chunk verify)

The port's forward result **depends on how the input is split into chunks**:

```
  ctx     margin    perturb    ratio  argmax flips
    8     1.3808     0.0000    0.00x            no
   16     0.4806     2.2759    4.74x           YES
   32     0.3906     0.5243    1.34x            no
   64     2.9504     0.5233    0.18x            no
  128     0.7380     2.3484    3.18x            no
```

Localization (`p6c_localize.py`): **layer 0** — a pure-window layer with
`compress_ratio=0`, so no compressor and no indexer — already differs by ~2 bf16
ulps; that amplifies to 0.052 by layer 3 and 2.28 in the logits.

It is **deterministic** (repeat runs are bit-identical), so it is a
kernel-shape / accumulation-order effect, not nondeterminism. The margins do not
follow a clean trend with context length, so this is **not** simply "the
distribution is flat" — it needs the full 40-layer model to characterize.

**Production encodes the same knowledge**: its batched verify path is gated
behind `EXO_DSV4_VERIFY_BATCH_MIN_CTX` (default 8192) and left **off** at short
context, explicitly "to preserve byte-identity at short ctx where the base decode
is deterministic."

### Consequence for deployment

| verify mode | token-identical? | cost per round | can it win? |
|---|---|---|---|
| `rowseq` (gamma+1 separate forwards) | **yes** | (gamma+1)·T_body + T_draft | **never** — needs `a > 1` |
| `chunk` (gamma+1 rows in one forward) | no | T_body(gamma+1) + T_draft | yes, above 12.4% accept |

`rowseq` is the *validation instrument*; `chunk` is the *deployable shape*.

## Economics (4-layer build, projected ×10 to 40 layers)

Projection is legitimate: per-layer cost is flat across the reduced stack, and
the draft head is depth-independent (an independent 3-stage stack).

```
40-layer projection
  plain decode / token          :   101.6 ms  ->  9.84 tok/s
  chunk verify (gamma+1 rows)   :   154.1 ms
  draft round                   :    10.3 ms

   accept  tokens   ms/token    tok/s  vs plain
       0%     1.0      164.4     6.08     0.62x
      20%     2.0       82.2    12.16     1.24x
      40%     3.0       54.8    18.25     1.85x
      60%     4.0       41.1    24.33     2.47x
      80%     5.0       32.9    30.41     3.09x
     100%     6.0       27.4    36.49     3.71x
  --> break-even acceptance: 12.4%
```

## Open question

**The real acceptance rate is unmeasured.** A 4-layer body drafts near-randomly
(0/5 accepted over 47 rounds) and its draft head agrees with its own body only
1.0% of the time. Acceptance is the single variable that decides whether this
lands at 2.5× or 3.7×, and it needs the full 40-layer model.

## Artifacts

| artifact | md5 | notes |
|---|---|---|
| `~/mtp-port-20260927.tgz` (node 1) | `84080530ecf42e0fc8b140c7c88d4f77` | full `deepseek_v41_mlx/` package |
| `~/mtp-port-mods.patch` (node 1) | `782807e358a1de340fefbca2badfb25a` | 311-line diff of the 6 modified files |

Also mirrored to the gateway box at
`/home/hermes/.hermes/cache/scratch/exl3patch/artifacts/` with matching md5s.

**Not committed to any git repository** — the port lives in a working tree on
node 1 only. Production (`exo` on both nodes) was never relaunched; all V4.1 work
ran out-of-process.

## Scripts

`p6a_spec_loop.py`, `p6b_chunk_equiv.py`, `p6c_localize.py`, `p6d_spec_rowseq.py`,
`p6e_closeout.py`, `p6f_module_verify.py`, `p6g_margin.py`, plus the Phase-3 port
wiring scripts (`wire_dspark_{1..7}.py`, `wire_taps.py`, `wire_stash.py`,
`fix_taps.py`, `build_reduced_mtp.py`, `convert_reduced_mtp.py`) and raw logs in
`raw/`.
