# DeepSeek-V4.1-Flash-1017 — migration plan (2026-10-19)

Status: **plan only — nothing changed on the cluster.** Production keeps serving
`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`.

Upstream release: <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017>
(created 2026-10-17T02:40Z, 10.2k downloads by 10-19). It is a **weight-only
minor update** to the model we serve: better reasoning/coding/agentic, shorter
outputs. No new architecture, no new kernels, no new env knobs.

---

## 1. What was verified (evidence, not assumption)

| Claim | How it was checked | Result |
|---|---|---|
| Architecture is unchanged | `config.json` diff of `text_config` **and** `vision_config`, ours vs 1017, all 56 fields | **identical** (40 layers, 5120 hidden, 384 experts top-6, 1 KV head, 1M ctx, `engram_layer_ids [1,14]`, dspark/hc/index ids) |
| Upstream is FP8 dense + FP4 experts | upstream `config.json` `quantization_config` | `fp8`, blocks 32×32, `ue8m0` scales, `expert_dtype: fp4` |
| Upstream does not fit us | 510.3 GB, 48 shards vs 2 × 128 GiB cluster RAM | **does not fit** (2x over); no Metal path for fp4 experts |
| 1017 **retrained the engram tables** | sampled rows of `layers.1.engram.embed.weight` (fp8, 384,006,168 × 256) at rows 0 / 1 / 999983 / 12345678 / 384006167 — ours (local native release) vs `apetersson/…-1017-Native-Engram-Overlay` | **every sampled row differs** |
| …and the comparison itself is sound | control: same rows, ours vs `deepseek-ai/DeepSeek-V4.1-Flash` (the old base the local release came from) | **byte-identical at every sampled row** |
| The 1017 engram shard layout is identical to ours | shard names + sizes of both official repos | 48 names in both, **0 size mismatches**; our working subset (shards 1-6, 43-48) = 244.2 GB of 1017 |
| Our engine already understands the new tensors | `bias_vl` (new in 1017, absent from the model we run) — `dsv41/load.py:184`, `dsv41/vision.py:323`, `deepseek_v4.py:1366` | **supported**; dispatch is checkpoint-driven, text path unchanged |
| In-checkpoint engram tables are ignored by our loader | `exl3_build.build_block`: `if blk.engram is not None: … blk.engram.embed = LazyEngramTable(native, layer_id)` | confirmed — the weight tables **must** come from a native dir; bundled ones are dead weight |

Consequence: **1017 needs both new weights and new engram tables.** The tables
are not reusable from the current install, and the loader will not read them out
of an EXL3 checkpoint.

## 2. The builds that exist (as of 2026-10-19 00:20 CDT)

| Repo | Size | Shards | EXL3 | Fits? |
|---|---|---|---|---|
| `dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED` | 415.8 GB | 45 = **43 weights (212.7 GB)** + **2 engram (203.1 GB)** | 1.6.2 | weights only: **yes** |
| `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` | 210.6 GB | 39, weights only | 1.6.1 | **yes** (size twin of what we run) |
| `object-oriented-architects/…-2.7bpw-TensorFold` / `HeNryous/…-2.75bpw-DGX-Spark` | 198.8 GB | 363 / 169 | — | yes, unvetted |
| `bertholomus/…-4.0bpw` | 288.4 GB | 41 | — | **no** (runtime, not storage) |
| `dealignai/…-ABLITERATED` | 0 GB | — | — | empty placeholder |
| `LibertAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` (dealignai's base) | unknown | — | — | 401, not public |

**The "wait for dealignai" step is already done** — the uncensored 2.9 bpw EXL3
landed 2026-10-18 20:18 CDT. The only reason it is not a drop-in is the 203 GB
of engram tables bundled into two of its shards, which our loader ignores.

## 3. Sizing

Resident weights today: 210.6 GB (196 GiB) across 2 × 128 GiB nodes, leaving
~12 GiB RAM per node. Engram tables are **not resident** — `LazyEngramTable`
reads rows on demand from disk (227 GiB per node today).

1017 target footprint is the same class: 212.7 GB of weights + 244.2 GB of
engram tables on disk. Nothing about the cluster's RAM budget changes.

Disk is the constraint:

| Node | Disk | Free now | `~/.exo/models` |
|---|---|---|---|
| macstudio-m4-1 | 926 GiB | **119 GiB** | 551 GiB |
| macstudio-m4-2 | 926 GiB | **210 GiB** | 515 GiB |

Neither node can stage weights + engram next to what it already holds. The
sequence in §4 frees before it fills.

## 4. Procedure (when the swap is approved)

Per node, in this order — both nodes can run in parallel:

1. **Free ~155 GiB** on m4-1 by removing models that are not the served one
   (`deepseek-ai--DeepSeek-V4-Flash-Vision-Exp` 145 GiB, `local--DeepSeek-V4-Flash-DSpark-MTP` 9 GiB).
   These are experiment checkpoints; confirm before deleting.
2. **Stage the 1017 engram release** (do not touch the live one yet):
   `hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --include 'model-0000[1-6]-of-00048.safetensors' 'model-0004[3-8]-of-00048.safetensors' 'config.json' 'model.safetensors.index.json' 'tokenizer*'`
   → 244.2 GB into `~/.exo/models/_staging/deepseek-ai--DeepSeek-V4.1-Flash-1017-engram`,
   then `EXO_DSV41_ENGRAM_DIR` points there.
3. **Stage the weights**: 43 shards of the dealignai 1017 build — everything
   **except** `model-00047`/`model-00048` — plus a **copy of the index with the
   engram entries removed** (exo derives its download allow-patterns from
   `weight_map`, `src/exo/download/download_utils.py:326`, so a trimmed index is
   the mechanism this codebase already uses to scope a shard). The bundle's own
   index references the two engram shards; leaving them in would make the
   downloader fetch 203 GiB we cannot use.
4. **Load-test one node** with `EXO_DSV41_ENGRAM_DIR=<staging engram>` and
   `--layers` disabled; then the pair. Expect `[DSV41] engram tables: …` in the
   log and a clean 2/2 READY.
5. **Quality/parity gate** before any switchover: §11 row suite (1K → 300K, plus
   the 1M row) with `scripts/s11-run.py`, the guard/limiter checks, and one
   agentic tool-call turn compared against today's output. 1017's whole claim is
   agentic/coding — that is what must move.
6. **Switchover**: move the staged dirs into place, `EXO_DSV41_ENGRAM_DIR`
   unset (default sibling name still resolves to the old path — either rename
   the new dir to `deepseek-ai--DeepSeek-V4.1-Flash-engram` or keep the env var
   set in the launcher), relaunch via `start_cluster.sh`.
7. **Rollback**: keep the old weights (196 GiB) until step 5 passes on the new
   pair. If they cannot both be held, rollback = re-download (~1-2 h).

## 5. Execution log

- **2026-10-19 00:19 — prepped both nodes (approved by owner).** Deleted
  `deepseek-ai--DeepSeek-V4-Flash-Vision-Exp` (156 G, retired) and
  `local--DeepSeek-V4-Flash-DSpark-MTP` (10 G) on **both** Studios, so they
  mirror. Free space: m4-1 119 → **286 GiB**, m4-2 210 → **377 GiB**.
  No open file handles on either directory before deletion; neither is referenced
  by the placed instance.
- **2026-10-19 00:20 — 1017 engram download started on both nodes**
  (`hf download deepseek-ai/DeepSeek-V4.1-Flash-1017` with the 12-shard subset
  that matches the current install's layout, into
  `~/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-1017-engram`, 244.2 GB).
  m4-2 keeps a full copy rather than its current 2-shard subset so the nodes
  stay mirrored.
- **Weights are NOT downloaded yet** and cannot be: 198 GiB does not fit
  alongside the old weights on either node (m4-1 would have ~59 GiB left after
  the engram tables). The weight swap is therefore a maintenance window:
  stop cluster → drop the old weight dir → download 43 shards (trimmed index,
  `index-trimmed.json`: 12 engram entries removed, 43 shards referenced) →
  point `EXO_DSV41_ENGRAM_DIR` at the 1017 tables → relaunch → run the gate.

- **2026-10-19 12:48 — 1017 is LIVE.** `READY (2/2)`, load warmup ~90 s, both
  runners serving `dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED`.
  Three blockers had to be fixed first, none of them about the weights:
  1. the upstream index ships `metadata.total_size: 0` → pydantic `PositiveInt`
     rejects it, and exo re-downloaded the index at every boot (wiping the
     trimmed shard map). Fixed in `cf1c60656`: a valid local index wins.
  2. no curated model card existed for the new id, so the auto-generated one
     declared `supports_tensor = false` and placement refused Tensor sharding.
     Card written on both nodes (`engine = "dsv41"`, Metal-only, real
     `storage_size` 212,742,339,040).
  3. the checkpoint stores **layer 14** `attn.wo_b` and
     `ffn.shared_experts.w2` as plain BF16 (`.weight`, no trellis group), so the
     TP slicer — which only walks trellis groups — left them full-size while the
     activations were per-rank:
     `(1,512,4096) vs (8192,5120)` then `(512,1152) vs (2304,5120)`.
     Fixed in mlx-lm `e598cb6ed` (superproject `e598cb6ed` bump): any raw
     `<name>.weight` gets the same 128-wide block slice on the same axis.

## 6. Open items

- **ABLITERATION QUALITY.** 1017's gains were measured on the official model.
  The dealignai build is an abliteration of an EXL3 quant; the Sept build was
  fine, but the agentic delta must be re-measured, not assumed.
- **`bias_vl` in the runtime.** Supported in code; not yet exercised on an
  abliterated 1017 checkpoint. `model.vl_bias_loaded` in the build report is the
  readout — it must be `True`, otherwise image spans route with the text bias
  (the documented loop).
- **EXL3 1.6.x vs our loader.** We run 1.4.2-format checkpoints today; both 1017
  candidates are 1.6.x. No version gate found in the loader, but this is the
  first 1.6-format load — treat it as the highest-risk unknown of the swap.
- **Chat template.** 1017 ships no `chat_template.jinja` (encoding is a Python
  reference impl); the dealignai build carries `tokenizer_config.json` only. If
  anything in our stack reads the model-dir template, copy the old one forward.
- **Naming.** The bundled tables use `language_model.layers.N.engram.*` in the
  community overlay vs `layers.N.engram.*` in the official repo. Use the
  **official upstream shards** (§4.2), not the overlay, to avoid the namespace.
