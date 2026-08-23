"""P3 Worker C3 — IN-SITU test of the PoolingCache buffer-donation hypothesis.

WHAT THIS IS
------------
Worker C's §4.3 pool-write probe measured "up to +6.35 ms/token" of compressor
pool-write cost over 100K -> 352.6K, and explicitly labelled it an UPPER BOUND
because the probe itself held a reference to the pool storage, which is exactly
the thing that defeats MLX's slice-update buffer donation
(mlx-lm/mlx_lm/models/cache.py:1537-1556).

This harness does NOT re-run an isolated probe. It drives the **REAL production
decode loop** -- `mlx_lm.generate.BatchGenerator` -> `GenerationBatch._step()`
(mlx-lm/mlx_lm/generate.py:1564-1712), the exact function exo's
`ExoBatchGenerator.step()` calls via `self._mlx_gen.next()` -- over the REAL
DeepSeek-V4 attention stack (43 real `v4_attention_factory` blocks inside the
REAL `DeepseekV4Model` / `Model` classes) with REAL `PoolingCache`s pre-filled
to depth L, a real sampler and the real bench-mode `ban_token_ids` logits
processor, and asks: does donation actually fire, or not?

CALL CHAIN THIS REPRODUCES (production, DSV4_SHARDING=Tensor, EXO_SPECULATIVE=0
path or MTP-disabled path -- see the findings doc for the citation trail):
  exo ExoBatchGenerator.submit()   src/exo/worker/engines/mlx/generator/batch_generate.py:2678
    -> mlx_lm BatchGenerator.insert()                    mlx-lm/mlx_lm/generate.py:1915
  exo ExoBatchGenerator.step()     batch_generate.py:4131
    -> mlx_lm BatchGenerator.next()/._next()             mlx-lm/mlx_lm/generate.py:2097
      -> GenerationBatch.next()                          mlx-lm/mlx_lm/generate.py:1739
        -> GenerationBatch._step()                       mlx-lm/mlx_lm/generate.py:1564
          -> Model.__call__ -> DeepseekV4Model._forward_steps -> DeepseekV4Block.__call__
             -> SparseCompressedAttention/CompressedAttention -> Compressor
                -> PoolingCache.update_and_fetch_deferred  cache.py:1466-1556

WHAT IS SIMPLIFIED (and why the comparison survives)
----------------------------------------------------
`DeepseekV4MoE` is replaced by a depth-INDEPENDENT stub MLP (a 256-expert
top-6 MoE at real width is 397B params; it does not fit and it is explicitly
out of scope). Every other part of the loop is the real production object.
Because the stub's cost does not depend on L, it cancels exactly in every
depth DELTA reported here. Absolute ms/token is therefore NOT a production
per-token number; only the deltas and the (a)/(b)/(c) comparison are.

THE THREE CONFIGURATIONS
------------------------
  (a) PRODUCTION  : EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES=8388608 (start_cluster.sh
                    default). At both depths the pool is far above 8 MiB, so
                    `update_and_fetch_deferred` takes the *donation* branch:
                    pre_write is None, no reference is held, and
                    `mx.async_eval(self._pool_storage)` (cache.py:1551) is
                    issued. Loop pipelining is production's (async_eval + one
                    blocking eval on the PREVIOUS step's tokens).
  (b) DEFEATED    : EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES=1<<40. Now
                    `storage_bytes <= _POOL_DEFER_COPY_MAX_BYTES` is true, so
                    the code captures `pre_write = self._pool_storage[...]` and
                    HOLDS it across the slice-assign -> donation cannot fire ->
                    the write degrades to a full O(P*D) copy. This is the exact
                    failure mode cache.py:1537-1553 documents, reached through
                    a production env var (so it is also a live A/B knob).
  (c) MAX-ENABLED : production env, plus a full `mx.synchronize()` after every
                    step. A drained pipeline means no in-flight prior-step
                    graph can be holding a stale view of the pool buffer at the
                    moment the slice-update evaluates -- donation is
                    structurally guaranteed. This is the loop-level analogue of
                    Worker C's per-step async_eval headline pattern.

  (a) ~ (c)  => production does NOT hit the donation failure (negative result).
  (a) ~ (b)  => production DOES hit it.

ALLOCATOR EVIDENCE
------------------
Per step we record mx.get_active_memory / get_peak_memory / get_cache_memory.
A failing donation must allocate a fresh pool buffer on every pooled-flush step
(~90 MB at 352.6K for a ratio-4 layer x 21 layers = ~1.9 GB of transient
allocation per 4-step cycle). Working donation writes in place: ~0.

Run (studio):
  .venv/bin/python /tmp/p3_donation_insitu_harness.py \
      --depths 100026,352599 --steps 96 --warmup 24 --json /tmp/p3c3.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# ── production env (start_cluster.sh defaults) — MUST precede mlx import ──
_PROD_ENV = {
    "EXO_DSV4_INDEX_TOPK": "512",
    "EXO_KV_CACHE_BITS": "0",
    "EXO_COMPUTE_DTYPE": "bf16",
    "EXO_DSV4_SPARSE_SDPA_TILE": "128",
    "EXO_DSV4_SEQ_SPLIT": "1",
    "EXO_DSV4_EXACT_TOPK": "1",
    "EXO_DSV4_TOPK_FUSED": "0",
    "EXO_DSV4_SPARSE_FUSED_SDPA": "0",
    "EXO_DSV4_ATTN_ALLSUM": "0",
    "EXO_DSV4_SINGLE_GATHER": "1",
    "EXO_DSV4_FENCE_ASYNC": "1",           # start_cluster.sh:1626
    "EXO_DSV4_FENCE_EVERY_N_LAYERS": "4",  # start_cluster.sh:437
    "EXO_DSV4_PREFILL_ARGPARTITION": "1",
    "EXO_DSV4_ARGPARTITION_MIN_P": "8192",
}
for _k, _v in _PROD_ENV.items():
    os.environ.setdefault(_k, _v)
for _k in ("EXO_DSV4_INDEXER_PBLOCK", "EXO_DSV4_QA_KV_FUSED", "EXO_DSV4_FP32_ACT",
           "EXO_DSV4_MTP", "EXO_PROFILER", "EXO_DSV4_SECTION_TIME"):
    os.environ.pop(_k, None)

_ARGV_POOL = None
for _i, _a in enumerate(sys.argv):
    if _a == "--pool-max-bytes":
        _ARGV_POOL = sys.argv[_i + 1]
if _ARGV_POOL is None:
    _ARGV_POOL = "8388608"          # start_cluster.sh production default
os.environ["EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES"] = _ARGV_POOL

import mlx.core as mx        # noqa: E402
import mlx.nn as nn          # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
_MLXLM = _REPO / "mlx-lm"
if not _MLXLM.is_dir():
    _MLXLM = Path.home() / "repos" / "exo" / "mlx-lm"
sys.path.insert(0, str(_MLXLM))

from mlx_lm.models import deepseek_v4 as dv4                      # noqa: E402
from mlx_lm.models.cache import CacheList, PoolingCache, RotatingKVCache  # noqa: E402


# ───────────────────────── production config ─────────────────────────
COMPRESS_RATIOS = [0, 0] + [4, 128] * 20 + [4]
assert len(COMPRESS_RATIOS) == 43 and COMPRESS_RATIOS.count(4) == 21

CFG = dict(
    model_type="deepseek_v4", vocab_size=129280, hidden_size=4096,
    intermediate_size=18432, moe_intermediate_size=2048, num_hidden_layers=43,
    num_attention_heads=64, num_key_value_heads=1, n_shared_experts=1,
    n_routed_experts=256, num_experts_per_tok=6, head_dim=512,
    index_head_dim=128, index_n_heads=64, index_topk=512, o_groups=8,
    o_lora_rank=1024, q_lora_rank=1024, qk_rope_head_dim=64,
    sliding_window=128, max_position_embeddings=1048576, rms_norm_eps=1e-6,
    rope_theta=10000, compress_rope_theta=160000,
    rope_scaling=dict(beta_fast=32, beta_slow=1, factor=16,
                      original_max_position_embeddings=65536, type="yarn"),
    routed_scaling_factor=1.5, scoring_func="sqrtsoftplus",
    topk_method="noaux_tc", norm_topk_prob=True, attention_bias=False,
    compress_ratios=COMPRESS_RATIOS, num_nextn_predict_layers=1,
    hidden_act="silu", swiglu_limit=10.0, tie_word_embeddings=False,
)

DTYPE = mx.bfloat16
HEAD_DIM, INDEX_DIM, SW = 512, 128, 128


# ─────────────── MoE stub (the ONLY non-production component) ───────────────
class StubMoE(nn.Module):
    """Depth-independent stand-in for DeepseekV4MoE.

    Same constructor signature and the same `__call__(x, input_ids)` contract
    the real DeepseekV4Block invokes, plus the `.layer_idx` / `.sharding_group`
    attributes the block body touches. Cost is O(hidden^2) and INDEPENDENT of
    context depth L, so it cancels exactly in every depth delta.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.sharding_group = None
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        return self.proj(x)


def _quant_predicate(path, module):
    """Replicates make_quantization_config (deepseek_v4.py:899-931)."""
    if not hasattr(module, "to_quantized"):
        return False
    if ".attn.w" in path or ".attn.indexer.wq" in path:
        return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    if "embed_tokens" in path or "lm_head" in path:
        return {"group_size": 64, "bits": 8, "mode": "affine"}
    if ".attn." in path or path.endswith(".attn"):
        return {"group_size": 64, "bits": 8, "mode": "affine"}
    return False        # leave the stub MLP / norms unquantized


def build_model():
    dv4.DeepseekV4MoE = StubMoE          # patch BEFORE constructing blocks
    args = dv4.ModelArgs.from_dict(CFG)
    model = dv4.Model(args)
    model.set_dtype(DTYPE)
    nn.quantize(model, class_predicate=_quant_predicate)
    mx.eval(model.parameters())
    return model


# ─────────────────── synthetic cache pre-fill at depth L ───────────────────
def _fill_rotating(rc, L, B=1, D=HEAD_DIM):
    rc.keys = mx.random.normal((B, 1, rc.max_size, D)).astype(DTYPE)
    rc.values = mx.zeros((B, 1, rc.max_size, 0), dtype=DTYPE)
    rc.offset = L
    rc._idx = rc.max_size
    mx.eval(rc.keys, rc.values)


def _fill_pool(pc, L, dim, B=1):
    P = L // pc.ratio
    alloc = max(pc.step, ((P + 1 + pc.step - 1) // pc.step) * pc.step)
    pc._pool_storage = mx.random.normal((B, alloc, dim)).astype(DTYPE)
    pc._pool_offset = P
    pc._pending_offset_bump = 0
    rem = L % pc.ratio
    out_dim = dim * (2 if pc.ratio == 4 else 1)
    if rem:
        pc.buf_kv = mx.random.normal((B, pc.ratio, out_dim)).astype(DTYPE)
        pc.buf_gate = mx.random.normal((B, pc.ratio, out_dim)).astype(DTYPE)
        pc.remainder = rem
    if pc.ratio == 4:
        half = out_dim // 2
        pc._overlap_kv_carry = mx.random.normal((B, 1, pc.ratio, half)).astype(DTYPE)
        pc._overlap_gate_carry = mx.random.normal((B, 1, pc.ratio, half)).astype(DTYPE)
    mx.eval(pc._pool_storage)
    if pc.buf_kv is not None:
        mx.eval(pc.buf_kv, pc.buf_gate)
    if pc._overlap_kv_carry is not None:
        mx.eval(pc._overlap_kv_carry, pc._overlap_gate_carry)
    return P


def prefill_cache(model, L, pool_slack=0):
    """Build the REAL Model.make_cache() (deepseek_v4.py:6956-6979) and put it
    into the exact state a depth-L prefill would have left it in.

    ``pool_slack`` over-allocates the pool storage by that many entries. This
    is config (d): BatchPoolingCache.update_and_fetch_deferred grows its pool
    with `mx.concatenate([pooled, pad])` sized to EXACTLY max_pool
    (cache.py:1898-1902) -- i.e. +1 entry per flush, with NO step-chunked
    headroom, so the concat fires on EVERY flush and copies the whole pool.
    A concat is a fresh allocation; donation cannot apply to it. Pre-padding
    the pool makes `pooled.shape[1] >= max_pool` hold for the whole run, so
    the concat never fires and only the slice-assign (the donatable write)
    remains. NOTE: merge() re-packs the pool to exactly max(pool_sizes)
    (cache.py:2688-2695), so the slack must be applied to _pool_storage AND
    survive the merge -- we apply it post-merge instead (see run_real_loop).
    """
    caches = model.make_cache()
    stats = {"P_comp4": 0, "P_idx": 0, "P_comp128": 0}
    for ci, ratio in zip(caches, COMPRESS_RATIOS):
        if ratio == 0:
            _fill_rotating(ci, L)
            continue
        subs = ci.caches
        _fill_rotating(subs[0], L)
        stats_key = "P_comp4" if ratio == 4 else "P_comp128"
        stats[stats_key] = _fill_pool(subs[1], L, HEAD_DIM)
        if ratio == 4:
            stats["P_idx"] = _fill_pool(subs[2], L, INDEX_DIM)
    return caches, stats


def pool_storage_bytes(caches):
    tot = 0
    for c in caches:
        subs = getattr(c, "caches", [c])
        for s in subs:
            st = getattr(s, "_pool_storage", None)
            if st is not None:
                tot += st.size * st.dtype.size
    return tot


# ──────────────────────── the REAL production loop ────────────────────────
def run_real_loop(model, caches, steps, warmup, sync_each_step=False, eos_id=1,
                  pool_slack=0):
    """Drive mlx_lm's REAL BatchGenerator exactly as exo's ExoBatchGenerator
    does (batch_generate.py:2678 insert / :4131 step), timing every step and
    sampling MLX allocator counters around it.

    `sync_each_step` adds a full mx.synchronize() after each step == config (c).

    NOTE (verified in-situ): BatchGenerator does NOT keep the PoolingCache
    objects passed to insert(). PromptProcessingBatch.__init__ runs
    `self.prompt_cache = _merge_caches(caches)` (generate.py:1261), and
    PoolingCache.merge returns a **BatchPoolingCache** (cache.py:1826). So the
    class the production decode loop actually exercises is BatchPoolingCache,
    whose donation branch is cache.py:1900-1920 -- the direct twin of
    PoolingCache's at cache.py:1537-1556, gated by the same
    _POOL_DEFER_COPY_MAX_BYTES. This harness therefore tests the REAL
    production cache class, not the single-stream one. The returned
    `live_caches` handle lets the caller inspect post-run pool state.
    """
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    # production sampler resolution: greedy for bench (temp 0) -> argmax path,
    # but exo always builds a real sampler object (batch_generate.py:2270).
    sampler = make_sampler(temp=0.0)

    # production bench mode prepends ban_token_ids (batch_generate.py:2658-2660)
    def ban(_history, logits):
        logits[..., eos_id] = -1e9
        return logits

    gen = BatchGenerator(
        model,
        max_tokens=steps + warmup + 8,
        stop_tokens=[[eos_id]],
        prefill_step_size=4096,
    )
    # exo passes last_tokens = prompt_tokens[-2:] (batch_generate.py:2612) plus
    # the already-prefilled cache -- reproduce exactly.
    gen.insert(
        prompts=[[17, 23]],
        max_tokens=[steps + warmup + 8],
        caches=[list(caches)],
        samplers=[sampler],
        logits_processors=[[ban]],
    )

    per_step, alloc = [], []
    n = 0
    # Drain prompt-processing passes until the caches have actually migrated
    # into the GENERATION batch. _next() splits a sequence out of
    # PromptProcessingBatch into GenerationBatch only once its last 1-token
    # segment is reached (generate.py:2127-2147), so the live BatchPoolingCache
    # objects do not exist on _generation_batch until then.
    for _ in range(8):
        gen.next()
        if len(gen._generation_batch) > 0:
            break
    mx.synchronize()

    # ── config (d): suppress the per-flush mx.concatenate pool growth ──
    # BatchPoolingCache grows by EXACTLY +1 entry per flush
    # (cache.py:1899-1903), so the concat fires every flush and copies the
    # whole O(P*D) pool into a fresh buffer -- a cost donation cannot touch.
    # Pre-pad the LIVE (post-merge) pool so pooled.shape[1] stays ahead of
    # max_pool for the entire run; only the donatable slice-assign remains.
    if pool_slack:
        _padded = []
        for c in gen._generation_batch.prompt_cache:
            for s in getattr(c, "caches", [c]):
                if type(s).__name__ == "BatchPoolingCache" and s.pooled is not None:
                    B, P, D = s.pooled.shape
                    pad = mx.zeros((B, pool_slack, D), dtype=s.pooled.dtype)
                    s.pooled = mx.concatenate([s.pooled, pad], axis=1)
                    s._visible_width = s.pooled.shape[1]
                    _padded.append(s.pooled)
        mx.eval(_padded)
        mx.synchronize()
        print(f"  (d) pre-padded {len(_padded)} BatchPoolingCache pools by "
              f"+{pool_slack} entries -> per-flush mx.concatenate suppressed")
        if not _padded:
            raise SystemExit("config (d) FAILED: no live BatchPoolingCache found")

    while n < warmup + steps:
        a0 = mx.get_active_memory()
        c0 = mx.get_cache_memory()
        p0 = mx.get_peak_memory()
        t0 = time.perf_counter()
        gen.next()
        if sync_each_step:
            mx.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        if n >= warmup:
            per_step.append(dt)
            alloc.append((mx.get_active_memory() - a0,
                          mx.get_cache_memory() - c0,
                          mx.get_peak_memory() - p0))
        n += 1
    mx.synchronize()
    return per_step, alloc, gen


def summarize(xs):
    xs = sorted(xs)
    n = len(xs)
    return dict(
        n=n, mean=statistics.mean(xs), median=statistics.median(xs),
        p10=xs[int(0.10 * n)], p90=xs[int(0.90 * n)], min=xs[0], max=xs[-1],
        stdev=statistics.stdev(xs) if n > 1 else 0.0,
    )


def phase_split(per_step):
    """A ratio-4 PoolingCache flushes (writes a pooled entry) on 1 step in 4.

    Split the per-step series by (index mod 4) and identify the FLUSH phase as
    the one with the largest median. `flush - nonflush` is the whole cost of
    the 21 sparse layers' pool write in one step; /4 gives its amortized
    per-token contribution. THIS is the donation signal: a donated (in-place)
    write is ~free, a defeated one is an O(P*D) copy that scales with pool
    size and therefore with depth.
    """
    ph = {k: [per_step[i] for i in range(len(per_step)) if i % 4 == k]
          for k in range(4)}
    meds = {k: statistics.median(v) for k, v in ph.items() if v}
    flush_k = max(meds, key=meds.get)
    nonflush = [v for k, vs in ph.items() if k != flush_k for v in vs]
    return dict(
        phase_medians={k: round(v, 3) for k, v in meds.items()},
        flush_phase=flush_k,
        flush_median=meds[flush_k],
        nonflush_median=statistics.median(nonflush),
        flush_excess_ms=meds[flush_k] - statistics.median(nonflush),
        flush_excess_amortized_ms=(meds[flush_k] - statistics.median(nonflush)) / 4.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="100026,352599")
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--warmup", type=int, default=24)
    ap.add_argument("--config", default="a", choices=["a", "b", "c", "d"])
    ap.add_argument("--pool-max-bytes", default="8388608")
    ap.add_argument("--json", default="")
    ap.add_argument("--mem-guard-gb", type=float, default=20.0)
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    print(f"MLX {mx.__version__}  device={mx.default_device()}")
    print(f"host={os.uname().nodename}")
    print(f"CONFIG = ({args.config})   "
          f"EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES={os.environ['EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES']}"
          f"  (cache.py:_POOL_DEFER_COPY_MAX_BYTES={dv4_pool_max()})")
    print(f"steps={args.steps} warmup={args.warmup}  sync_every_step={args.config == 'c'}")
    print("building REAL DeepseekV4 Model (43 real attention blocks, stub MoE)...")
    t0 = time.perf_counter()
    model = build_model()
    print(f"  built in {time.perf_counter() - t0:.1f}s   "
          f"active={mx.get_active_memory() / 1e9:.2f} GB")

    out = {"config": args.config, "host": os.uname().nodename,
           "pool_max_bytes": os.environ["EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES"],
           "steps": args.steps, "warmup": args.warmup, "depths": {}}

    for L in depths:
        caches, st = prefill_cache(model, L)
        pb = pool_storage_bytes(caches)
        act = mx.get_active_memory() / 1e9
        print(f"\n=== L = {L:,} ===")
        print(f"  pool storage total {pb / 1e6:.1f} MB  "
              f"(P_comp4={st['P_comp4']:,} P_idx={st['P_idx']:,} P_comp128={st['P_comp128']:,})")
        print(f"  active GPU mem after prefill: {act:.2f} GB")
        if act > args.mem_guard_gb:
            raise SystemExit(f"MEM GUARD: {act:.2f} GB > {args.mem_guard_gb} GB")

        per_step, alloc, gen = run_real_loop(
            model, caches, args.steps, args.warmup,
            sync_each_step=(args.config == "c"),
            pool_slack=(args.steps + args.warmup + 16) if args.config == "d" else 0,
        )
        # VERIFY the pool actually advanced INSIDE the real loop. Read the
        # LIVE cache the generator owns (a BatchPoolingCache produced by
        # _merge_caches), not the PoolingCache we handed in -- insert() does
        # not keep ours.
        _live = gen._generation_batch.prompt_cache
        _r4 = [c for c, r in zip(_live, COMPRESS_RATIOS) if r == 4][0]
        _pcache = _r4.caches[1]
        _off_after = (_pcache._pool_lengths[0] if hasattr(_pcache, "_pool_lengths")
                      else _pcache._pool_offset)
        _n_ran = args.warmup + args.steps + 1
        print(f"  LIVE CACHE CLASS: {type(_pcache).__name__}  (ratio-4 compressor pool)")
        print(f"  POOL ADVANCE CHECK: pool length {st['P_comp4']:,} -> {_off_after:,} "
              f"(+{_off_after - st['P_comp4']} over {_n_ran} steps, expect ~{_n_ran // 4})")
        s = summarize(per_step)
        ps = phase_split(per_step)
        d_active = [a[0] for a in alloc]
        d_peak = [a[2] for a in alloc]
        print(f"  per-step ms: median {s['median']:.3f}  mean {s['mean']:.3f}  "
              f"p10 {s['p10']:.3f}  p90 {s['p90']:.3f}  min {s['min']:.3f}  max {s['max']:.3f}")
        print(f"  PHASE SPLIT (mod 4): {ps['phase_medians']}")
        print(f"    flush phase={ps['flush_phase']}  flush median {ps['flush_median']:.3f}  "
              f"non-flush median {ps['nonflush_median']:.3f}")
        print(f"    FLUSH EXCESS = {ps['flush_excess_ms']:+.3f} ms/flush-step  "
              f"= {ps['flush_excess_amortized_ms']:+.3f} ms/token amortized")
        print(f"  per-step ACTIVE-mem delta (MB): median {statistics.median(d_active) / 1e6:.2f}  "
              f"max {max(d_active) / 1e6:.2f}")
        print(f"  per-step PEAK-mem delta  (MB): median {statistics.median(d_peak) / 1e6:.2f}  "
              f"max {max(d_peak) / 1e6:.2f}")
        print(f"  peak GPU mem so far: {mx.get_peak_memory() / 1e9:.2f} GB")
        print(f"  raw first 24 step ms: {[round(x, 3) for x in per_step[:24]]}")
        out["depths"][str(L)] = dict(
            L=L, pool_bytes=pb, stats=s, phase=ps, per_step_ms=per_step,
            d_active_bytes=d_active, d_peak_bytes=d_peak,
            peak_gb=mx.get_peak_memory() / 1e9,
        )
        del caches
        mx.clear_cache()
        mx.reset_peak_memory()

    if len(depths) == 2:
        a, b = out["depths"][str(depths[0])], out["depths"][str(depths[1])]
        print(f"\n=== DEPTH DELTA ({depths[0]:,} -> {depths[1]:,}) ===")
        print(f"  median ms/step: {a['stats']['median']:.3f} -> {b['stats']['median']:.3f}"
              f"   delta {b['stats']['median'] - a['stats']['median']:+.3f} ms")
        out["delta_median_ms"] = b["stats"]["median"] - a["stats"]["median"]

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")


def dv4_pool_max():
    from mlx_lm.models import cache as _c
    return _c._POOL_DEFER_COPY_MAX_BYTES


if __name__ == "__main__":
    main()
