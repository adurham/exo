# Phase-1 design input — dsv41 decode-round MECHANICS (BRIEF R1, Q1–Q6)

**Scope.** Read-only source forensics, no GPU/cluster contact. Every cost statement below is
source-derived or explicitly labelled **INFERRED**. Citations are `path:line` against two
read-only trees: the engine at `/private/tmp/next16-instr/src/exo/worker/engines/mlx/dsv41/`
(branch `deploy/next16-instr`) and the authoritative pinned mlx-lm at
`/Users/adam.durham/repos/exo/mlx-lm/mlx_lm/models/deepseek_v41/` (the worktree's `mlx-lm/` is an
empty submodule; the shared checkout is the read source, pin `6cc9c1e`).

## HEADLINE (changes the timer design; read before Q6)

The premise "one `mx.eval(logits)` per round, the draft/verify graph fused into it" is **true
only syntactically**. The body forward of every **compressing** layer contains an **ungated
host round-trip** at default env: `sparse_attention._column_boundary(...)` calls
`int(mx.min(...))` and `bool(mx.all(...).item())` for **every layer with `kv2 is not None` in
every body forward, including the 4-row verify**, and the gate that would disable it lives in a
*different module with the opposite default*. This means the decode round is host-serialized
per-layer, not once per round — which is exactly what the wall model has been unable to
explain. Details in Q3; the timer consequence is in Q6.

---

## Q1 — draft path (`head.draft(...)`, `rounds.py:414`)

**Class.** `head` is `mlx_lm.models.deepseek_v41.mtp.DSparkHead` (`mtp.py:255`), attached by
`load.build_draft_head` → `exl3_build.build_mtp` (`exl3_build.py:667-720`).

**How many forwards?** **ONE parallel forward over the whole block**, not `gamma` sequential
steps:

> `mtp.py:26-31` — "The draft is ONE parallel forward over ``[anchor, noise x (block-1)]``
> producing base logits for every block position, then a rank-``markov_rank`` first-order
> transition bias injects intra-block dependency during a sequential sampling loop."

```python
# mtp.py:303-315
block_ids = mx.concatenate([anchor_tokens[:, None],
                            mx.full((b, bs - 1), self.noise_token_id, ...)], axis=1)
x = embed(block_ids)
...
for stage, c in zip(self.stages, caches):
    x, pre_mix = stage(x, pre_mix, c)
x = hc_pre(x, pre_mix)
```

The sequential part is a host-free Markov sampling loop (`mtp.py:322-333`, `for k in range(bs)`)
— `markov_embed` → add to `base_logits[:,k,:]` → `argmax` — all `mx` ops, **no `.item()`**.

**Layer count / dims.** `n_stages = args.n_mtp_layers or 3` (`mtp.py:261`) → **3 stages**; block
`dspark_block_size` default 5 but the call passes `width=gamma=3` → `bs=3`. Each stage reuses the
body args: `dim=5120, n_heads=64, head_dim=512, o_groups=8` (`config.py:33-43`). Draft MoE is
**128 experts, top-3** (`mtp.py:174-184`, `dspark_n_experts`/`dspark_topk`), vs the body's 384/top-6.

**TP collectives?** Yes — the draft is **partly sharded on each rank, not replicated**:
- draft routed experts are width-sliced and `stage.ffn.group = group` is set
  (`exl3_build.py:678-681`); `DraftMoE.__call__` does `all_sum` when `group` is set
  (`mtp.py:193-195`) → **one `all_sum` per stage = 3 per draft**.
- the markov head is vocab-sharded (`exl3_build.py:713-718`, `head.vocab_sharded=True`), so the
  draft's token pick goes through `head.combine_argmax` (`mtp.py:317,327`) → **one more `all_sum`**.
- draft attention / `main_proj` are replicated.

So `draft()` runs **4 cross-rank `all_sum`s** per call (3 MoE + 1 argmax), wrapped in
`_coll.warm_guard` (`mtp.py:292`) — host-synced for the first 2 calls with each key, off after.

**Hidden host syncs in `draft()` / `append_ctx` / `make_cache`?** **None.** `grep` of `mtp.py`
shows no `.item()`, `int(` of an array, `np.asarray`, `.tolist()`, or `mx.eval`. `make_cache`
(`mtp.py:280-282`) allocates `DraftWindow`s lazily (`mx.zeros`, `mtp.py:67`). `append_ctx`
(`mtp.py:284-288`) is `main_norm(main_proj(hidden))` + per-stage `DraftAttention.append_ctx`
(`mtp.py:131-134`) → a ring write; no sync.

**Return / dtype.** `_draft` returns `(draft_tokens [b, bs], confidence [b, bs])` (`mtp.py:334-338`);
`draft_tokens = mx.stack(toks, axis=1)` where `toks` are `argmax`/`combine_argmax` int32 values.
`rounds.py` takes `drafted = drafted[0]` if a tuple (`rounds.py:422-423`) and then
`drafted = drafted.astype(mx.int32)` (`rounds.py:424`) — **lazy** (an MLX op, no eval).

**State `append_ctx` mutates.** Per stage: `DraftWindow.win_kv` (a `[b, window, head_dim]` fp32
ring, `mtp.py:67`) written in `DraftWindow.append` via `self.win_kv[:, slots] = kv[...]`
(`mtp.py:75-76`), and `n_ctx` (`mtp.py:77`). `main_proj`/`main_norm` are consumed, not stored.
Per round the engine appends `accepted+1` rows (`rounds.py:460`:
`head.append_ctx(tapcat(taps)[:, : accepted + 1], draft_state)`). This is an **O(1)-ring,
O(k·head_dim) write** (k = accepted+1 ≤ 4) plus an O(n_tap·dim·dim) `main_proj` GEMM — **not
O(ctx)**: the ring keeps only the last `window` rows regardless of context. The GEMM dominates
and is context-independent.

---

## Q2 — snapshot / stash / rollback (`spec.py`)

All three live in `spec.py` and contain **no `mx.eval` / `mx.synchronize` / `.item()`** — they
are pure lazy graph-build + host bookkeeping.

**`snap(cache, start_pos)` (`spec.py:20-30`).** Per layer with a `comp_state`, saves the open-group
carry rows needed to rebuild:
```python
# spec.py:26-30
m = start_pos % lc.ratio
st.append((mx.array(cs.kv_state[:, :m]), mx.array(cs.score_state[:, :m]), m)
          if m else (None, None, 0))
return start_pos, st
```
`mx.array(...)` is a **lazy copy** (no sync). Shape is `[b, m, head_dim]`, `m = start_pos % ratio`
≤ `ratio` (≤2). **Cost: O(ratio·head_dim) per layer = O(1)** — independent of context depth.

**`stashes(cache)` (`spec.py:33-35`).** Returns **references** to the compressor's stashed raw
rows: `(lc.comp_state.chunk_kv, lc.comp_state.chunk_score)`. O(1), no copy. (`chunk_kv`/`chunk_score`
are stored by the compressor each forward, `compressor.py:72-75`.)

**`rollback(cache, sn, target, st)` (`spec.py:61-73`).**
```python
# spec.py:62-73
chunk_start, saved = sn
cache.offset = target                      # host int, O(1)
for lc, sv, stash in zip(cache.layers, saved, st):
    if lc.comp_state is None:
        continue
    if stash[0] is None:
        continue
    _rebuild(lc.comp_state, lc.ratio, chunk_start, target, sv, stash)
```
`_rebuild` (`spec.py:38-58`) concatenates the saved carry slice + the stashed chunk rows and
slice-assigns them back: `cs.kv_state[:, :need] = nk` (`spec.py:57-58`). Shape `[b, need, head_dim]`,
`need = target - ratio*(target//ratio)` ≤ ratio. **All lazy, no sync. Cost O(ratio·head_dim) = O(1),
independent of context depth** (this is the important answer for 100K: rollback is NOT O(ctx)).

**Cache components touched per layer:** **only the compressor open-group carry**
(`comp_state.kv_state` / `comp_state.score_state`, `spec.py:57-58`). **Not** touched: the window
ring `win_kv` (position-addressed, rewritten before it is read again — module docstring
`spec.py:7-10`), `comp_kv`, `index_k`, engram history. `cache.offset` is set on the host
(`spec.py:63`). At 100K ctx the compressor carry is still only `[b, ≤2, 512]`.

**Force expressions for a Mode-2 timer.**
- For `spec.rollback(...)`: evaluate each layer's carry destination, which forces the
  concatenate+slice-assign graph:
  `mx.eval([lc.comp_state.kv_state for lc in cache.layers if lc.comp_state is not None])`.
- For `head.append_ctx(...)`: evaluate each draft window's ring, which forces the
  `main_proj`/`main_norm` GEMM + the `win_kv` slice-assign:
  `mx.eval([w.win_kv for w in draft_state])`.

Both are the **existing idiom already in-tree**: `session._checkpoint` does exactly the draft
form — `saved = [(w.win_kv + 0, int(w.n_ctx)) for w in self.draft_state]; mx.eval([kv for kv, _ in saved])`
(`session.py:805-807`) — and `SessionCache.snapshot` does the ring form —
`rings = [lc.ring_snapshot() for lc in self.cache.layers]; mx.eval([r for r in rings if r is not None])`
(`session_cache.py:371-372`). `spec.generate`'s harness path simply lets the next round's
`mx.eval(am, d)` drain them (`spec.py:318`). **In the engine, the drain is `rounds.py:435`.**

---

## Q3 — verify forward (`model(verify_in, cache, return_taps=True, argmax=True)`, M=4)

### Q3a — syncs inside `Model.__call__` for small M

**The `_fence_every` / "≤16-row skip".** `Model.__call__` (`model.py:192-215`) chooses the
collective guard by row count:
```python
# model.py:210-211
guard = (_coll.sync_collectives() if int(input_ids.shape[1]) > 16
         else _coll.warm_guard(key))
```
For the 4-row verify, `sync_collectives()` is **not** entered; `warm_guard(key)` host-syncs
collectives only for the first `DSV41_SYNC_WARM_CALLS=2` calls with that shape key, then returns
`contextlib.nullcontext()` (`collective.py:52-62`). So after warmup the per-call collective guard
is a no-op. Likewise the per-layer fence: `fence = getattr(self, "_fence_every", 0) if n > 1 else 0`
(`model.py:271`) → **0 for a 4-row verify**, so the `mx.eval(h, pre_mix)` at `model.py:298-299`
never runs. `_fence_every` is only set during prefill and restored in `finally` (`prefill.py:153-154,
203-204`). **No `.item()`/`int()` anywhere in `_forward`.**

**BUT the verify is NOT sync-free** — `sparse_attention._column_boundary` fires on every
compressing layer (see Q3c). This is the finding that contradicts the "one sync" premise.

### Q3b — `argmax=True`, `return_taps=True`

```python
# model.py:334-337
elif argmax and hasattr(self.head, "argmax"):
    logits = self.head.argmax(h.astype(mx.float32))   # token ids [b, n]
elif argmax:
    logits = mx.argmax(self.head(h.astype(mx.float32)), axis=-1).astype(mx.int32)
```
In production `self.head` is `ShardedHead` (`exl3_build.py:635`), whose `argmax` is
`combine_argmax(local(h))` (`exl3_build.py:331-332`): each rank takes the vocab-slice
`mx.argmax`, packs `(max, index)`, pads and does **one `all_sum`**, then a device-side argmax over
the `[world, rows, 2]` buffer with ties to the lowest id (`exl3_build.py:317-329`). **Return dtype:
int32 `[b, n]`** — no full-vocab logits tensor is ever materialized. `return_taps=True` allocates
`taps[layer_id] = h.mean(axis=2)` for each `layer in args.dspark_target_layer_ids`
(`model.py:264,283-284`) — 3 tensors `[1, 4, 5120]` (the tapcat concat is 3·5120 = 15360 wide,
`rounds.py:382-383`).

### Q3c — data-dependent host decisions in the decode body (the load-bearing part)

**`sparse_attention._column_boundary` runs a host round-trip on every compressing layer, in
decode, at default env.** Two independent module-level gates disagree:

```python
# attention.py:57   (default OFF)
_COLSPLIT = os.environ.get("DSV41_SPARSE_COLSPLIT", "0") == "1"
# attention.py:214
extra = {"colsplit": n_window} if _COLSPLIT else {}
o = sparse_attn(q, kv_all, sink, idxs, self.softmax_scale,
                chunk=_PREFILL_CHUNK, kv2=kv2, split=offset, **extra)
```
```python
# sparse_attention.py:137  (default ON — opposite)
_COLSPLIT = os.environ.get("DSV41_SPARSE_COLSPLIT", "1") == "1"
# sparse_attention.py:412-416   (no row-count gate; runs for m=4 too)
colsplit_i: int = -1
if kv2 is not None and _COLSPLIT:
    w = _column_boundary(topk_idxs, split_i, -1 if colsplit is None else int(colsplit))
    if w is not None:
        colsplit_i = w
```
Attention passes `colsplit=None` (its own gate is OFF), so `_column_boundary` receives `-1` and
takes the derivation branch, which is a **`int()` host sync** as its first statement:
```python
# sparse_attention.py:294-302
def _leading_window_columns(icb, split):
    lead = (icb < split).astype(mx.int32)
    run = mx.cumprod(lead, axis=-1)
    return int(mx.min(mx.sum(run, axis=-1)))      # <-- host sync
```
and a **second** host round-trip at
```python
# sparse_attention.py:289
if not bool(mx.all(checks).item()):               # <-- host sync
    return None
```
`kv2` is non-None on every **compressing** layer at decode (set at `attention.py:209`,
`kv2 = src.comp_kv[:bsz, :compress_len]`, inside `if compress_len:` under `if self.ratio`), i.e.
all `ratio>0` layers, which the port docstring (`indexer.py:5-9`) puts at kv-source 2/8/14/20 plus
their consumers — most of layers 2..39. The production launch env (verified snapshot,
`.../phase20-throughput/raw/prod-env-pre-relaunch1-m4-1-ps-eww.txt`) does **not** set
`DSV41_SPARSE_COLSPLIT`, so `sparse_attention`'s default `"1"` applies. **INFERRED** cost:
the engine author's own adjacent comment measures a per-layer host sync in this exact call as
*"~124 vs ~110 ms per spec round"* i.e. **~14 ms/round** (`sparse_attention.py:399-403`), so a
per-compressing-layer host round-trip is a first-order term in the unexplained ~55 ms.

**Indexer top-k.** Untiled/tiled score selection is device-side `mx.argpartition`
(`indexer.py:622`, `indexer.py:250`). The untiled decode fast-path has no eval. **However**
`DSV41_INDEXER_HIER` defaults ON (`indexer.py:122`) and its entry point
`hierarchical_topk_prod` runs unconditionally regardless of row count (`indexer.py:531`), and it
`mx.eval`s per strip: `mx.eval(s)` (`indexer_hierarchical.py:253`), `mx.eval(block_maxima)`
(`:263`), `mx.eval(part)`/`mx.eval(ids)` (`:369,372`), `mx.eval(bm)` (`:392`),
`mx.eval(best_v, best_i)` (`:510`), plus a `.item()` at `max_cand = int(blk.sum(axis=-1).max().item())`
(`indexer_hierarchical.py:347`). **INFERRED (not confirmed for the live engine):** if decode
takes the hierarchical path, each of the 8 index-source layers drains many strips per round.
This is a second, independent candidate for the missing ~55 ms and is cheap to falsify
(`DSV41_INDEXER_HIER=0` A/B). See the NOT-verified list.

**`cache.ensure_capacity`** (`rounds.py:412` → `cache.py:188-231`) is **pure host** int math +
`mx.clear_cache()`; it never syncs the GPU and never calls `.item()`. But `mx.clear_cache()`
(`cache.py:231`) frees the pool, so when growth fires it perturbs every subsequent bracket.

**Recompile.** `mx.compile` caches are keyed per shape (`sparse_attention.py:342`,
`indexer.py:100-105`). Decode n=1 and draft width=3 are stable shapes, so no steady-state
recompile; a gamma change under the adaptive policy would recompile the verify/draft shapes once.

---

## Q4 — host work around the round

**The decode loop** — `Dsv41Engine._rounds` (`engine.py:1012-1047`):
```python
# engine.py:1012-1047 (abridged)
while n < max_tokens:
    active = self._active
    self._check_cancel(...)                         # 1014, host
    ...
    committed, _round_ms, _accepted, _gamma = _one_round(...)   # 1016-1026
    if head is not None:                            # spec counters, host ints
        self._spec_rounds += 1; self._spec_accepted += int(_accepted); ...
    batch = [int(t) for t in committed]             # 1036, host
    n += len(batch); token = batch[-1]
    session.maybe_checkpoint()                      # 1044  (cadence-gated)
    yield batch, lps                               # 1045  ← generator suspends here
    if session.eos_id in batch: return
```

**Consumers of the yield (all on the same thread).** `_generate` pulls round-batches through
`committed_tokens()` (`engine.py:801-809`), then per token: `detokenizer.add_token(tid)` +
`last_segment` (`engine.py:841-842`), `_stop_index` (`845`), `_mid_response(...)` construction
(`868`). That stream is wrapped by `dsv41_output_parser` (thinking split + DSML parse +
`count_reasoning_tokens` + `map_responses_to_chunks`, `output.py:98-108`), pulled by
`Dsv41Engine.step` (`engine.py:427-437`), returned as a list, then `runner.py:653`
`results = self.generator.step()` sends each via `send_chunk` (`runner.py:729,877-884`). All of
this is on the **same thread that runs `_one_round`** (`runner.py:649-730`), so it serializes
against the GPU eval loop.

**Per-round host operations outside `_one_round`:** (1) `_check_cancel` → `MpReceiver.collect()`
queue drain (`engine.py:1051-1055`, host-only); (2) spec counters + `batch` list build
(`engine.py:1029-1038`); (3) `session.maybe_checkpoint()` (`engine.py:1044`); (4) the entire
detokenize/parse/SSE chain (above); (5) no per-round `logger.` on the dsv41 path (only
`rounds.py:401` once per request).

**INFERRED >1 ms per round:** the detokenize+parser+chunk-construction chain per emitted token
(≈1+accepted per round) is the prime candidate for a few ms; `maybe_checkpoint` when it fires
(default every 1024 rows, `session.py:112-113`) does a real `mx.eval(rings)` +
`mx.eval(win_kv)` (`session_cache.py:372`, `session.py:807`) — a several-ms spike roughly once per
~340 rounds; `_check_cancel` is normally sub-ms.

**Where `emit_ms` should bracket.** From just before the `yield` (`engine.py:1045`) to just before
the next `_one_round(...)` (`engine.py:1016`). That window absorbs the consumer's drain of this
round's whole batch (detokenize → parse → send) while the generator is suspended, plus the next
iteration's `_check_cancel`. Note `maybe_checkpoint` (`1044`) sits **before** the yield, so it
lands in the *round* side, not `emit_ms`, unless the bracket is started after `_one_round`
returns — decide and document either way; the C2 closure (D5) cares that no ms is double-counted
or dropped.

---

## Q5 — existing instrumentation

- **`phase_marks`** (`src/exo/worker/engines/mlx/phase_marks.py`, gated by `EXO_PHASE_MARKS`,
  read once at import, `:40`): a request-lifecycle mark recorder. `grep` shows **no `mark(` in
  `dsv41/`**; the only runner marks are `template_rendered_ms` (`batch_generator.py:1165`) and
  `trie_matched_ms` (`cache.py:1262`). **Dormant for dsv41** — nothing per-round to reuse.
- **`mlx_lm.profiler.span` / `SpanProfilerHook`** (`profiler.py:96-113,197-232`): **present and
  wired into the dsv41 forward** — `model.py:36` imports `span`, and `Block.__call__` /
  `_fused_call` open `span("attn")` / `span("ffn")` (`model.py:123,132,141,147`), with
  `attention.py` spans like `attn.proj_qkv`, `attn.kv_cache`, `attn.indexer`, `attn.sdpa`,
  `attn.all_sum`, and `moe.all_sum`. But the hook only registers when `EXO_PROFILER` is set
  (`bootstrap.py:115-158`); **production does not set it** (absent from the launch-env snapshot)
  → `span()` returns `_NULL_SPAN` (no-op, `profiler.py:101-106`). **Perturbation:** with
  `EXO_PROFILER=spans` and the default `EXO_PROFILER_SYNC_SPANS` unset, spans measure
  **graph-build only** (meaningless for GPU attribution); with `EXO_PROFILER_SYNC_SPANS=1` each
  span entry/exit does `mx.synchronize()` at both boundaries (`profiler.py:223-231`), i.e. **every
  attn/ffn span per layer is serialized** — accurate shares, large throughput loss.
- **`_PhaseTimer`** (`speculative/dsv4_mtp.py:815`): the generic-path MTP profiler. The dsv41
  package never imports `dsv4_mtp`; **dormant for dsv41** (confirmed: `grep` of `dsv41/` for
  `EXO_PHASE_MARKS|EXO_PROFILER|_PhaseTimer` returns 0).
- **`VERIFY_MS`** (`spec.py:77`): a module literal dict, not a timer.

Net: **nothing per-round is reusable unmodified**; `span` is the closest instrument but its
accurate mode is perturbing and it has no round-level bracket.

---

## Q6 — timer design verdict

### Q6a — PROF=1 (host wallclock only at natural sync points; non-perturbing)

Only `time.perf_counter()` readings; no added `mx.eval`. Natural host-blocking points are the
existing syncs (`rounds.py:435`, and the per-layer syncs of Q3c, which PROF=1 cannot see).

| bracket | open (before) | close (after) | measures |
|---|---|---|---|
| `round_total` | `rounds.py:368` | after `return` `rounds.py:461` | whole round wall (already present as `round_ms`) |
| `draft_build` | `rounds.py:414` | `rounds.py:427` (after `verify_in` concat) | **host graph-build of the draft only** (~sub-ms) |
| `verify_block` | `rounds.py:430`/`433` | `rounds.py:435` `mx.eval(logits)` | **the only honest blocking bracket**: drains the draft + verify + all per-layer Q3c syncs + the *previous round's* deferred rollback/append_ctx |
| `tail_bookkeep` | `rounds.py:436` | `rounds.py:465` | accept loop + `snap`/`rollback`/`append_ctx` graph-build (host, deferred) |
| `emit` | `engine.py:1045` (before `yield`) | `engine.py:1016` (before next `_one_round`) | consumer drain of the batch (detokenize/parse/send) |

**Which bracket absorbs the previous round's deferred tail — the `verify_block` bracket.** The
rollback (`rounds.py:459`) and `append_ctx` (`rounds.py:460`) queue lazy work with no eval
(`spec.py:61-73` and `mtp.py:284-288` contain no `mx.eval`), so their GPU work is only forced by
the **next** round's `mx.eval(logits)` (`rounds.py:435`) — i.e. it lands inside round N+1's
`verify_block`, not in `tail_bookkeep`. Consequently `tail_bookkeep` in PROF=1 measures host
graph-build and looks near-free while its compute is billed to the next `verify_block`.

### Q6b — PROF=2 (`mx.eval` at bracket ends; serializing)

Add an eval at each bracket close so each bracket drains its own graph.

| bracket | open | close + force expression | measures |
|---|---|---|---|
| `draft` | `rounds.py:414` | `mx.eval(drafted, [w.win_kv for w in draft_state])` | real GPU draft: parallel block fwd + markov loop + 3 MoE `all_sum` + 1 argmax `all_sum` |
| `verify` | `rounds.py:430`/`433` | `rounds.py:435` `mx.eval(logits)` (already) | the verify forward **only** (draft now drained) |
| `rollback` | `rounds.py:459` | `mx.eval([lc.comp_state.kv_state for lc in cache.layers if lc.comp_state is not None])` | the O(1) carry rebuild |
| `append_ctx` | `rounds.py:460` | `mx.eval([w.win_kv for w in draft_state])` | the `main_proj` GEMM + ring write |
| `emit` | as PROF=1 | as PROF=1 (host only) | consumer drain |

### Traps that make a bracket lie

1. **The "one sync" model is false (Q3c).** `verify_block` absorbs draft + verify + *all* per-layer
   `_column_boundary` host syncs + the previous round's rollback/append tail. Blaming all of it on
   "verify" is the central mis-attribution risk. PROF=2 is **mandatory** to split it.
2. **`draft_build` in PROF=1 measures graph-build, not the draft's ~11 ms GPU cost** (phase-17:
   draft round 11.1 ms, `README §Numbers`); the draft's compute is billed to `verify_block` because
   `verify_in` consumes `drafted` (`rounds.py:425-427`), so the fused verify graph reaches it.
3. **An `mx.eval` that closes a bracket mid-round changes fusion** and can reorder accumulation —
   R8a risk (PREREG D4). Keep evals only at round boundaries, never between `snap` (`428`) and the
   verify, and treat any R8a diff as a first-divergence investigation, not an auto-revert.
4. **PROF=2 makes the round longer** (serializes; author measured this failure mode at ~124 vs
   ~110 ms, `sparse_attention.py:399-403`). Report PROF=2 vs PROF=1 vs unset (gate G1.4) — never
   quote a PROF=2 round time as the production round time.
5. **`maybe_checkpoint` spikes.** When the checkpoint cadence fires (`engine.py:1044`) it evals the
   rings + draft window (`session_cache.py:372`, `session.py:807`). If it lands inside `emit`,
   the *median* is fine but the *max* lies. Exclude the checkpoint round from `emit` stats.
6. **`ensure_capacity` / `mx.clear_cache()`** (`cache.py:231`, reached from `rounds.py:412`) frees
   the pool when the cache grows; a bracket that spans a growth event is not comparable to one
   that does not. Report growth rounds separately.
7. **`_column_boundary` fallback.** On an ambiguous layout it returns `None` and the call silently
   takes the exact where-select path (`sparse_attention.py:412-416`, `252-291`); the sync count per
   layer is then data-*and* layout-dependent, so a per-layer average is a distribution, not a
   constant — report medians and IQR, not a single mean.

### Is `draft_1 / draft_2 / draft_3` (per-MTP-step) meaningful?

**No.** The DSpark draft is **one parallel forward** over the whole block (`mtp.py:303-315`); the
3 positions are computed together, and the sequential part (`mtp.py:322-333`) is a host-free
`mx`-op loop with no per-step sync — so host wall-clock per "step" measures nothing distinct. The
only meaningful per-stage split is **`draft_stage0/1/2`** (the `for stage, c in zip(self.stages, ...)`
loop, `mtp.py:313-314`), and it is meaningful *only under PROF=2* with an `mx.eval` after each
stage. Recommend bracket names: `draft` (whole), optional `draft_stage{0,1,2}` under PROF=2.

---

## Phase-3 lever candidates (code-grounded, ≤3)

All ranges **INFERRED** (source + the author's own adjacent measurements; nothing here was run on
the cluster).

1. **Kill the per-compressing-layer host sync in `sparse_attention._column_boundary`.**
   `sparse_attention.py:137`'s gate defaults ON while `attention.py:57`'s defaults OFF, so
   `int()` (`:302`) + `.item()` (`:289`) fire on every `ratio>0` layer at decode for nothing.
   Cheapest lever: align the two defaults (or pass a static `colsplit` hint) so the derivation
   branch never runs; `DSV41_SPARSE_COLSPLIT=0` on both modules is the immediate A/B.
   **INFERRED 5–15 ms/round** (anchored on the author's own 124-vs-110 ms per-layer-sync figure).
   *Confirm:* PROF=1/2 `round_ms` median with `DSV41_SPARSE_COLSPLIT` unset vs `=0`, ≥5 reps.

2. **Audit the indexer hierarchical per-strip `mx.eval`s on the decode path.**
   `hierarchical_topk_prod` (`indexer.py:531`→`indexer_hierarchical.py:545-622`) evals per strip
   (`:253,263,369,372,392,510`) and a `.item()` (`:347`) with no row-count gate, on the 8
   index-source layers. **INFERRED 3–8 ms/round** if decode takes this path.
   *Confirm:* `DSV41_INDEXER_HIER=0` A/B on `round_ms` at 100K ctx (and a PROF=1 per-layer count).

3. **Keep lazy rollback/append bookkeeping folded + ≤1 host sync.** `spec.rollback` is already
   O(1) and eval-free; the lever is to *not* add evals and to ensure the tail drains in the same
   natural sync (no extra round-trip), plus trim the per-round Python host chain (Q4).
   **INFERRED 0–2 ms/round** — small; do NOT expect this alone to close the gap.

**Arithmetic reminder (from PREREG):** 30 t/s needs *both* adaptive gamma *and* ≥15 ms/round of
Phase-3 savings; lever 1 alone is plausibly the bulk of that 15 ms.

---

## NOT verified (no GPU/cluster touched — every cost statement is source-derived)

- No `mx.eval`/`.item()` cost was measured; all ms ranges are **INFERRED** from source structure
  and the author's adjacent comments/README (phase-17 draft 11.1 ms; 124-vs-110 ms per-layer sync).
- **Unconfirmed:** whether the live decode actually enters `sparse_attention._column_boundary`
  with `kv2 is not None` on every compressing layer (it requires `kv2` set at `attention.py:209`
  for that layer *this* round). The code path and the default-on gate are confirmed; the live
  firing rate is not.
- **Unconfirmed:** whether decode takes the `_HIER` indexer path (the gate is default-on and has no
  row-count guard, but I did not trace a live call); the lever-2 range is correspondingly soft.
- **Unconfirmed:** the *number* of compressing layers that actually pass `kv2` per round (the
  `kv2`/`comp_len` condition is context-dependent; only the ≥1-per-compressing-layer fact is
  established).
- No `_one_round`, `spec.py`, `mtp.py`, `sparse_attention.py`, or `indexer_hierarchical.py` code
  was executed; the "one sync" premise was not re-tested against a running process.
- The exact PROF=2 force expressions were not executed; they are read off the destination arrays
  written by `spec.rollback` (`spec.py:57-58`) and `DraftWindow.append` (`mtp.py:75-76`).
