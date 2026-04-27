# Fork Notes

Tracks the divergences between `adurham/mlx` / `adurham/mlx-lm` `main` and
their ml-explore upstream, plus the exo-side workarounds. Update this file
whenever the fork gains/drops a patch against upstream.

File map:
| Topic | Where |
|---|---|
| Dependency pins | `uv.lock` (`mlx` and `mlx-lm` have `git = "…?branch=main#<sha>"`) |
| Fork sources | `pyproject.toml` `[tool.uv.sources]` |
| KV cache config → runtime | [kv-cache-architecture.md](./kv-cache-architecture.md) |
| Upstream PR/issue tracker | [upstream-prs.md](./upstream-prs.md) |

## adurham/mlx-lm main

Pinned (as of exo `65655ced`): `6f2a29680ac8db93f3eca62a0efd2ae631d241a5`.

### Why we're on main, not a snapshot

Main carries upstream ml-explore's BatchGenerator rewrite (`PromptProcessingBatch`
+ `GenerationBatch`, `Response` moved out of `BatchGenerator`, `next()` returns
`(prompt_responses, generation_responses)`, `next_generated()` returns only
the generation list). Exo has been ported to this API — see the migration
commit chain `519024e4`, `2b1b4bf9`, `97bd2126`.

### What works

- `KVCache`, `QuantizedKVCache`, `RotatingKVCache`, `ArraysCache` all behave
  as before.
- `QuantizedKVCache.merge` (upstream added) lets 4-bit KV flow through
  `_merge_caches` without the stale
  `does not yet support batching with history` error.

### What was ported — **MTP speculative decoding** ✓

`MTPBatchGenerator` ported to the new mlx-lm API (commit `5abedbf`). Now
subclasses `BatchGenerator` correctly, uses `_generation_batch` /
`_unprocessed_sequences` / `SequenceStateMachine`, and `_next()` returns
`(prompt_resps, gen_resps)`. Deque token buffer drains one per call; GDN
cache rolled back for rejected drafts via `c.offset` / `c.rollback()`.

**QWEN35_ENABLED still 0** — flip to 1 in `start_cluster.sh` after confirming
the live smoke test shows "MTP speculative decoding enabled" in the log and no
regressions on Huihui.

### What needs a port — **opt_batch_gen fast-next patch**

`src/exo/worker/engines/mlx/patches/opt_batch_gen.py` monkey-patches
`BatchGenerator.next` with a hand-rolled fast path that overlaps GPU
forward, async-eval, and detokenisation. It uses `self.active_batch` / the
old `Response` class. The patch is already gated off when
`EXO_SPECULATIVE=1` (start_cluster.sh sets that), so on the live cluster
it's not loaded. If it's ever re-enabled, the internal references need the
same rename (`active_batch → _generation_batch`, Response → GenerationBatch.Response).

### DeepSeek-V4 prefill chunk cap

DSv4's `compress_ratio == 4` attention layers (`mlx_lm/models/deepseek_v4.py:1439-1448`)
flatten the per-query top-`k` indexer gather to a dense KV of length `L × k`,
then pass it to `scaled_dot_product_attention`. The score tensor is therefore
`(B, n_heads, L, L × k)` — cubic in `L` until `k` saturates at
`index_topk=512` (i.e. `L ≥ 2048`), then quadratic. Single allocation must fit
under Apple Silicon's per-buffer cap of ~80 GiB (`86,586,540,032` bytes on
M4 Max).

Hard ceiling on prefill chunk size — observed and projected:

| L (chunk) | k = min(512, L/4) | per-layer score buf (bf16) |
|---|---|---|
| 1024 | 256 | 34 GB |
| 1280 | 320 | 67 GB |
| 1500 | 375 | **108 GB** ← crashes |
| 1904 | 476 | **221 GB** ← observed 2026-04-26 (227 GB Metal rejection) |
| 2048 | 512 | 275 GB |

**Set `EXO_PREFILL_STEP_SIZE ≤ 1024` (we use 512) for DSv4.** The default 2048
or `start_cluster.sh`'s 4096 will OOM on any multi-turn chat where the prior
assistant turn pushes total prompt past ~1.2K tokens. Reported on Blaizzy
[#1192](https://github.com/ml-explore/mlx-lm/pull/1192#issuecomment-4323191525);
real fix is upstream (sparse query-grouped SDPA that never materializes the
dense `L × k` view).

## adurham/mlx main

Pinned (as of exo `65655ced`): `1cfcb5b6fb769d9cbe312860646f13fb788090b3`.

### What we're carrying against upstream

Two revert commits on top of `e64e280d` (the upstream merge):

1. `1a176363 Revert "Fix jaccl init bug (#3418)"`
2. `1cfcb5b6 Revert "Jaccl refactor (#3412)"`

### Why

Upstream's #3412 (jaccl refactor, author: Angelos Katharopoulos, April 14)
splits the jaccl implementation into a `lib/jaccl/` tree with new Config
objects and a ring/mesh dispatch. On our 2-rank RDMA setup it crashes at
`mx.distributed.init(backend="jaccl", strict=True)` with:

```
ValueError: vector
```

thrown from inside the nanobind layer (libc++'s `std::vector::at` default
`what()` on out-of-range). The symptom on rank 1 is a cascading
`RuntimeError: [jaccl] Send failed with errno=22` when the rank-0 coordinator
dies mid-handshake.

We haven't rooted the underlying bug yet. Reverting #3412 + its follow-up
#3418 restores the pre-refactor jaccl (single-file `jaccl.cpp` + top-level
`mesh.cpp`/`ring.cpp`/`utils.cpp`) which has been shipping on our cluster
for months. The reverts are purely mechanical — no conflicts — so pulling
future upstream main merges shouldn't re-apply them automatically.

### Work to re-enable the refactor

The pre-refactor jaccl worked fine on our 2-rank Thunderbolt RDMA, so
`#3412` is strictly a regression for this topology — not a pre-existing
bug. Upstream is unaware because the intersection of (exo users × jaccl
backend × 2 ranks × Apple-Silicon-over-Thunderbolt-Bridge × post-#3412
release) is effectively just this cluster. A proper fix + upstream PR
is the right long-term path rather than carrying reverts indefinitely.

**Phase 1 — minimal repro (no exo in the loop).** ~20-line Python
harness that calls `mx.distributed.init(backend="jaccl", strict=True)`
with the same `MLX_IBV_DEVICES` + `MLX_JACCL_COORDINATOR` env that
`start_cluster.sh` sets. Run against **upstream mlx main** (NOT our
reverted fork) on both Studios. Capture the rank-0 traceback and the
rank-1 `RuntimeError: [jaccl] Send failed with errno=22` cascade.
~30 minutes. Temporarily takes MiniMax down during the test window —
run in a maintenance window, not while the prediction-bot needs it.

**Phase 2 — root cause.** Three candidates in probability order:

1. *Config parsing* — devices parsed as `string` instead of `[string]`
   in the new 3D tensor. Quickest check: log the parsed `Config`
   object at the top of `jaccl::init` before `MeshGroup::MeshGroup`.
2. *Empty-diagonal short-circuit in `create_connections`* — at N=2
   the 3D mesh is degenerate; some `std::vector::at` call likely
   indexes a missing row. `grep -n '\.at(' lib/jaccl/` — probably
   only a handful of sites.
3. *Bind/listen ordering in `SideChannel::SideChannel`* — timing log
   on both ranks. Less likely (both ranks fail symmetrically, which
   suggests a config/shape bug rather than a race).

Variance here: config-parsing = ~1 hour, QP ordering = ~1 day.

**Phase 3 — patch + local verify.** Fork upstream mlx into a working
copy, apply the fix, `uv sync` against a local editable path
(`pyproject.toml` has the commented-out lines for this at `[tool.uv.
sources]` already). Run `start_cluster.sh` *without* the two revert
commits. MiniMax READY at 2-rank = fix confirmed. ~1 hour + rebuild
time.

**Phase 4 — upstream PR.** Ship against `ml-explore/mlx` with a
regression test (ideally a pure config-parsing unit test that doesn't
need RDMA hardware). Reference `#3412` + `#3418`. Once merged, drop
`1a176363` + `1cfcb5b6` from this fork and bump the pin.

> **Upstream tracking:** classified "won't upstream the revert" in
> [upstream-prs.md](./upstream-prs.md) — an issue + reproducer is the
> right artifact, not a revert PR.

## adurham/mlx main — quantized SDPA kernel + dispatch knob

Pinned (as of exo `6ae331fe`): `1f6eb6bd` (`sdpa: port unquant 2-pass
perf improvements to quant kernel`).

### Custom primitives carried on main

- `mx.fast.scaled_dot_product_attention_quant(q, k_packed, k_scales,
  k_biases, v_packed, v_scales, v_biases, *, scale, group_size, bits,
  do_causal, sinks)` — Metal kernel that fuses the quant dequantize
  into the 2-pass FlashDecoding inner loop. Decode-only gate
  (`q.shape(2) == 1`); prefill / unsupported configs fall through to
  `dequantize() + scaled_dot_product_attention()`. Used by MiniMax
  decode via `mlx_lm/models/base.py:117`.
- Kernel body lives in `mlx/backend/metal/kernels/sdpa_vector_quant.h`.
  Carries 4 optimizations ported from the unquant `sdpa_vector_2pass_1`
  path (Q hoist + pre-scale, V memory-read hoist, bits=5 half-pack
  uint64 load, contiguous-chunk access). Local M4 Max microbench
  −26–29 % vs unoptimized; **cluster M4 Ultra neutral** (see
  `docs/minimax-quantized-sdpa-design.md` + `memory/phase2_real_goal.md`).

### MLX_SDPA_BLOCKS env var

> **Upstream tracking:** opened as PR #3455 (see
> [upstream-prs.md](./upstream-prs.md)) — applies to upstream's existing
> 2-pass kernel without needing the broader Phase-2 quant work.


`scaled_dot_product_attention.cpp` carries an env-var override for the
2-pass blocks heuristic: `sdpa_2pass_blocks_override()` reads
`MLX_SDPA_BLOCKS`, and if set to a positive int, overrides both the
bf16 and quantized dispatches. No-op when unset.

**Empirical sweet-spot on the 2-rank M4 Ultra cluster for MiniMax-
M2.7 at 50K context: `MLX_SDPA_BLOCKS=88`** gives +6.5 % decode
(26.14 → 27.86 tok/s). Sharp cliff at `blocks=92`. Reason: matches
~320 concurrent simdgroup slots on M4 Ultra (4 kv_heads × 88 = 352
TGs ≈ 1.1 dispatch rounds). See `memory/minimax_sdpa_blocks88.md` for
the full sweep + cluster config context.

Forwarded through `start_cluster.sh` as of exo `6ae331fe`. Not baked
in as default because optimum is workload-specific.

## Open optimization projects

### MiniMax fused MoE kernels

`src/exo/worker/engines/mlx/patches/` only contains `qwen3_5/` and
`qwen3_5_moe/` fused batched-MoE kernels. `mlx-community/MiniMax-M2.7-4bit-mxfp4`
(256 experts, top-8, 62 layers) runs the vanilla MLX MoE path: unfused
router → Python-level top-k → per-expert kernel launches → gather → TP
all-reduce, repeated for every layer for every decode token.

Symptom: at ~40K context under TP=2, MiniMax decode collapses to ~16 tok/s
even though memory, KV cache, and attention all look healthy. The per-token
MoE dispatch + 62 cross-rank all-reduces per step are the ceiling — context
length makes it feel worse only because attention's absolute cost is also
growing in parallel.

What to build: a `patches/minimax/` module paralleling `patches/qwen3_5_moe/`
— batched MoE dispatch (256-expert scatter/gather), shared-expert fusion if
MiniMax has one, and whatever Q/K/V + RMSNorm hot block MiniMax attention
has. Re-tune the Qwen3.5 kernel shapes for MiniMax's `hidden_size=3072`,
`intermediate_size=1536`, `num_attention_heads=48`, `num_key_value_heads=8`.

Reference:
- `src/exo/worker/engines/mlx/patches/qwen3_5_moe/apply.py` — entry point
- `src/exo/worker/engines/mlx/patches/__init__.py::maybe_apply_patches` — dispatch on `model_type`
- `~/.claude/projects/-Users-adam-durham-repos-exo/memory/minimax_moe_decode_bottleneck.md` — the full diagnosis + numbers

### MTPBatchGenerator port (see above) and jaccl refactor debug (see above)

## Updating these pins

```bash
# Bump mlx-lm alone
uv lock --upgrade-package mlx-lm

# Bump mlx alone
uv lock --upgrade-package mlx

# Both
uv lock --upgrade-package mlx --upgrade-package mlx-lm
```

After bumping, smoke-test with:

```bash
uv run pytest src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py tests/test_vision_cache.py
bash ./start_cluster.sh   # redeploys both Studios
python3 /tmp/stream_probe.py   # quick TTFT + streaming check (see docs/)
```

If MiniMax fails to come up, check the first `RunnerFailed` traceback in
`~/.exo/exo_log/exo.log` — most API-drift failures manifest there with a
clear `AttributeError`. Record the delta in this file when resolving.
