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

### What needs a port — **MTP speculative decoding**

`src/exo/worker/engines/mlx/speculative/mtp_batch_generator.py` subclasses
mlx-lm's `BatchGenerator` and uses:

- `self.active_batch` — gone. Replace with `self._generation_batch` (non-empty when `len(self._generation_batch) > 0`).
- `self.unprocessed_prompts` — gone. Replace with `self._unprocessed_sequences` (deque).
- `self.stop_tokens` instance attr — gone. Stop logic lives in `SequenceStateMachine` now; use `self._default_state_machine`.
- `self._next()` returning `List[Response]` — now returns `(prompt_resps, gen_resps)`. Fast-path consumers should call `self._generation_batch.next()` directly (returns `List[GenerationBatch.Response]`) and skip the prompt side.

Exo detects the new API at runtime (`hasattr(MlxBatchGenerator, "active_batch")`)
and falls back to plain `BatchGenerator` with the warning:

```
EXO_SPECULATIVE=1 but this mlx-lm version split BatchGenerator into
_prompt_batch/_generation_batch — MTPBatchGenerator is not ported yet.
Running without MTP speculative decoding.
```

Huihui (prediction-bot scout) still runs. The 40% MTP throughput win is the
cost of the fallback. Qwen3.5-397B-A17B (the other MTP user) is currently
disabled in `start_cluster.sh` (`QWEN35_ENABLED=0`); re-enabling it requires
this port.

### What needs a port — **opt_batch_gen fast-next patch**

`src/exo/worker/engines/mlx/patches/opt_batch_gen.py` monkey-patches
`BatchGenerator.next` with a hand-rolled fast path that overlaps GPU
forward, async-eval, and detokenisation. It uses `self.active_batch` / the
old `Response` class. The patch is already gated off when
`EXO_SPECULATIVE=1` (start_cluster.sh sets that), so on the live cluster
it's not loaded. If it's ever re-enabled, the internal references need the
same rename (`active_batch → _generation_batch`, Response → GenerationBatch.Response).

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
