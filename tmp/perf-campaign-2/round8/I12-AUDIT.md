# I12 — SERIAL-vs-BATCHED PREFILL PARITY AUDIT (campaign 2, round 8)

**Type:** read-only code audit. No source file under `src/` was modified. No cluster contact.
**Question:** at concurrency=1 (production), does the SERIAL prefill driver actually reach each
of six shipped campaign-1 prefill optimizations, or do they live only under `prefill_batched`?

Line numbers are as of the working tree at audit time. `mlx-lm/` is a git submodule
(`adurham/mlx-lm` @ `7f14654`); its files are byte-identical to the installed
`.venv/lib/python3.13/site-packages/mlx_lm/` copies (verified with `diff -q` for both
`models/deepseek_v4.py` and `generate.py`), so a cite into `mlx-lm/mlx_lm/...` is a cite
into the code the live runner imports.

---

## 0. THE DRIVER FORK — established once, reused by every item below

The batched driver is selected in the runner's step loop, and only there:

- `src/exo/worker/runner/llm_inference/batch_generator.py:757`
  `if agreed_slots > 1 and agreed_queue_len >= 2:` → `_batched_start_task` (`:765`)
  → `ExoBatchGenerator.submit_batched` (`batch_generate.py:2789`) → `prefill_batched`
  (`batch_generate.py:3057` → `generate.py:1302`).
- At concurrency=1 `agreed_queue_len` is 1, the branch is False, and the request goes through
  `_start_task` → `ExoBatchGenerator.submit()` (`batch_generate.py:2179`).
- `submit()` calls the SERIAL driver at `src/exo/worker/engines/mlx/generator/batch_generate.py:2559`
  → `prefill()` at `src/exo/worker/engines/mlx/generator/generate.py:741`.
- Belt-and-braces: even if the batched branch were entered with a single task,
  `batch_generate.py:2826` (`if len(tasks) <= 1: return [self.submit(*t) for t in tasks]`)
  routes back to the serial `submit()`.

**Inside `prefill()` the serial driver forks again** — and this second fork is the one that
matters for the whole audit:

- `generate.py:901` `is_pipeline = _has_pipeline_communication_layer(model)`
  (`generate.py:422`) is True only if a `PipelineFirstLayer`/`PipelineLastLayer`/
  `MetaFramedPipeline*Layer` is installed (`generate.py:454-464`). Those are installed **only**
  by `pipeline_auto_parallel` (`src/exo/worker/engines/mlx/auto_parallel.py:559`), which runs
  **only** on the `PipelineShardMetadata` branch (`src/exo/worker/engines/mlx/utils_mlx.py:504`).
- Production is `MLX_JACCL_SHARDING_MODE=Tensor` → `TensorShardMetadata` →
  `tensor_auto_parallel` (`utils_mlx.py:501`), which installs no pipeline layer.
  **So `is_pipeline` is False in production.**
- Therefore `generate.py:907` (`if is_pipeline and num_tokens >= prefill_step_size`) is False and
  the `else` at `generate.py:923` runs: `stream_generate(...)` at `generate.py:936`.

**The production serial prefill chunk loop is mlx-lm's `generate_step`, not exo's PP loop.**
This is the single most important structural fact in this audit, and every "SHARED" verdict
below rests on it:

```
mlx-lm/mlx_lm/generate.py:833   stream_generate -> generate_step(prompt, model, **kwargs)
mlx-lm/mlx_lm/generate.py:468   while total_prompt_tokens - prompt_processed_tokens > 1:   # THE chunk loop
mlx-lm/mlx_lm/generate.py:473     _chunk = prefill_step_size
mlx-lm/mlx_lm/generate.py:481     _model_call(input_tokens=prompt[:n_to_process][None], ...)
mlx-lm/mlx_lm/generate.py:388       -> model(input_tokens, cache=prompt_cache)
```

and from that single `model(...)` call the whole DSv4 forward is entered, identically for both
drivers:

```
mlx-lm/mlx_lm/models/deepseek_v4.py:7280  Model.__call__
mlx-lm/mlx_lm/models/deepseek_v4.py:7285    h = self.model(inputs, cache)
mlx-lm/mlx_lm/models/deepseek_v4.py:7256  DeepseekV4Model.__call__
mlx-lm/mlx_lm/models/deepseek_v4.py:7268    *_, (_kind, _idx, out) = self._forward_steps(inputs, cache)
mlx-lm/mlx_lm/models/deepseek_v4.py:6862  DeepseekV4Model._forward_steps
mlx-lm/mlx_lm/models/deepseek_v4.py:7105    h = layer(h, mask, layer_cache, inputs)     # per-layer loop
mlx-lm/mlx_lm/models/deepseek_v4.py:5193  DeepseekV4Block.__call__
mlx-lm/mlx_lm/models/deepseek_v4.py:5494    x = self.attn(normed, mask=mask, cache=cache)
```

`prefill_batched`'s own chunk loop reaches the *same* `model(...)` call at
`generate.py:1489`. **Everything below the `model(...)` boundary is therefore shared by both
drivers by construction** — the two drivers differ only in the outer loop and in the
tensor's batch dimension (serial B=1, batched B=N).

Live layer census, from the deployed checkpoint's `config.json`
(`~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash/.../config.json`):
`num_hidden_layers=43`, `compress_ratios` = {`4`: 21, `128`: 20, `0`: 3}, `sliding_window=128`,
`index_topk=512`. Per `v4_attention_factory` (`deepseek_v4.py:5168-5175`) that is
**20 `CompressedAttention` layers (item 1's home) and 21 `SparseCompressedAttention` layers
(items 2/3's home) in every forward pass** — neither class is a rare or conditional layer.

---

## VERDICT TABLE

| # | Item | Verdict | Decisive file:line | One-line reasoning |
|---|------|---------|--------------------|--------------------|
| 1 | Tiled compressed SDPA (`EXO_DSV4_QUERY_TILED_SDPA`, P09) | **SHARED** | `mlx-lm/mlx_lm/models/deepseek_v4.py:4540`; reached via `src/exo/.../generate.py:936` → `mlx-lm/mlx_lm/generate.py:481` → `deepseek_v4.py:5494` | Lives inside `CompressedAttention.__call__`, below the shared `model(...)` boundary; its `q.shape[0] != 1` gate (`:3628`) is *satisfied* by serial B=1 and would be *violated* by the batched driver's B≥2 — reachability runs the opposite way to a bypass. |
| 2 | Exact fused top-k prefill (`EXO_DSV4_EXACT_TOPK_PREFILL`, P08) | **SHARED** | `mlx-lm/mlx_lm/models/deepseek_v4.py:4182-4188`; reached via `deepseek_v4.py:4905` → `:4023` | Lives inside `Indexer.__call__`, called unconditionally by all 21 `SparseCompressedAttention` layers; the only gate on the flag is `scores.shape[1] <= 16 or _EXACT_TOPK_PREFILL`, which has no driver/queue-length term. |
| 3 | Indexer path (`EXO_DSV4_INDEX_TOPK`, `EXO_DSV4_PREFILL_ARGPARTITION`, `EXO_DSV4_ARGPARTITION_MIN_P`) | **SHARED** | `mlx-lm/mlx_lm/models/deepseek_v4.py:4905` (call), `:3987` (INDEX_TOPK), `:4203-4207` (ARGPARTITION + MIN_P) | Same `Indexer.__call__` body as item 2; `index_topk` is read in `Indexer.__init__` at model-construction time, and the argpartition branch is gated on `scores.shape[1] > 1` (true for any prefill chunk), not on the driver. |
| 4 | Prefix-cache keying / snapshot insertion (`EXO_LEAF_SNAPSHOT_RETENTION`) | **SHARED** — and *serial-exclusive* | `src/exo/.../generate.py:859` (snapshot capture), `src/exo/.../batch_generate.py:2591` → `:5256` → `src/exo/.../cache.py:963` | The serial path is the **only** path that captures snapshots and writes the prefix cache; `prefill_batched` returns empty snapshots (`generate.py:1635`) and `submit_batched` never calls `_save_prefix_cache`. Serial reaches strictly more, not less. |
| 5 | `EXO_PREFILL_STEP_SIZE` chunking (prod 2048) | **SHARED** | `src/exo/.../generate.py:904` (env read on the serial path), `mlx-lm/mlx_lm/generate.py:473` (chunk applied) | `self.prefill_step_size` is never populated by the builder (`src/exo/worker/engines/mlx/builder.py:179-190` passes no such kwarg), so it is `None` and `prefill()` reads the env var itself at `:904`, then hands it to `stream_generate` at `:943`. |
| 6 | Clear-cache cadence (`EXO_PREFILL_CLEAR_CACHE_INTERVAL`, prod 1) | **SHARED** — and *serial-exclusive* | `mlx-lm/mlx_lm/generate.py:544-547` | The interval knob exists **only** in mlx-lm's `generate_step` chunk loop, i.e. exactly the loop the serial driver runs; `prefill_batched` ignores it and hard-calls `mx.clear_cache()` every chunk (`src/exo/.../generate.py:1563`). |

**BYPASSED: 0. AMBIGUOUS: 0. SHARED: 6.**

---

## TRACED CALL CHAINS

### Item 1 — Tiled compressed SDPA (`EXO_DSV4_QUERY_TILED_SDPA` / `_B`) — SHARED

```
src/exo/worker/runner/llm_inference/batch_generator.py:757   (agreed_queue_len>=2 == False at c=1) -> serial branch
src/exo/worker/engines/mlx/generator/batch_generate.py:2179  ExoBatchGenerator.submit
src/exo/worker/engines/mlx/generator/batch_generate.py:2559    prefill(...)                       # SERIAL ENTRY
src/exo/worker/engines/mlx/generator/generate.py:741         prefill
src/exo/worker/engines/mlx/generator/generate.py:901           is_pipeline = False   (TP; see §0)
src/exo/worker/engines/mlx/generator/generate.py:923           else-branch taken
src/exo/worker/engines/mlx/generator/generate.py:936             stream_generate(...)
mlx-lm/mlx_lm/generate.py:833                                 generate_step(...)
mlx-lm/mlx_lm/generate.py:468                                   prefill chunk loop
mlx-lm/mlx_lm/generate.py:481                                     _model_call(...)
mlx-lm/mlx_lm/generate.py:388                                       model(input_tokens, cache=prompt_cache)
mlx-lm/mlx_lm/models/deepseek_v4.py:7285                          h = self.model(inputs, cache)
mlx-lm/mlx_lm/models/deepseek_v4.py:7268                            self._forward_steps(inputs, cache)
mlx-lm/mlx_lm/models/deepseek_v4.py:7105                              h = layer(h, mask, layer_cache, inputs)
mlx-lm/mlx_lm/models/deepseek_v4.py:5494                                x = self.attn(normed, mask=mask, cache=cache)
mlx-lm/mlx_lm/models/deepseek_v4.py:4450                              CompressedAttention.__call__   (20 of 43 layers)
mlx-lm/mlx_lm/models/deepseek_v4.py:4540                                elif _QUERY_TILED_SDPA and _query_tiled_ok(...)   <-- GATED CODE
mlx-lm/mlx_lm/models/deepseek_v4.py:3617                                  _query_tiled_ok
```

Shape gate audited condition by condition against a real serial B=1 chunk
(`_query_tiled_ok`, `deepseek_v4.py:3628-3643`):

- `:3628` `isinstance(mask, mx.array) and q.shape[0] != 1 -> False`. The mask is a real array:
  built at `deepseek_v4.py:6980` with `return_array=True`, then widened to `[local | pooled]`
  by `_extend_mask` (`:4512` → `:1408`). **`q.shape[0] == 1` holds on the serial driver
  (B=1) and would FAIL on the batched driver (B≥2).** This gate is the reverse of a bypass:
  the optimization is serial-only-ish, not batched-only.
- `:3631` `n_q >= 2*_QUERY_TILED_B` (= 128 at the production `_B=64`) and `mask.shape[-2] == n_q`.
  Production `EXO_PREFILL_STEP_SIZE=2048`; under TP seq-split (`:4521-4526`,
  `EXO_DSV4_SEQ_SPLIT=1` default, `start_cluster.sh:124`) q and mask are sliced to the same
  band (`:4533`, `:4535`), so `n_q` is 1024 per rank and the row counts stay equal. Both far
  above 128.
- `:3634` `kv.shape[2] > pool`: `kv` is `concatenate([local_kv, pooled])` (`:4511`), so this is
  "at least one local key", true for any real chunk.
- `:3637` `local_cache.offset >= local_len`: `local_cache` is a `RotatingKVCache(max_size=128)`
  (`deepseek_v4.py:7399-7412`). Before the window fills, `update_and_fetch` returns
  `keys[..., :offset, :]` (`cache.py:711`) so `local_len == offset`; once full it returns the
  whole 128-wide buffer (`cache.py:712`) while `offset` keeps climbing, so `offset >= 128 == local_len`.
  Holds in both regimes.
- `:3642` tail-block visibility: `(tail or 64) - 1 + sliding_window(128) >= 1`, trivially true.

**Note on shared-ness:** the batched driver reaches the *same line* `:4540` via
`generate.py:1489`, but its `q.shape[0]` is N≥2, so `_query_tiled_ok` returns False at `:3628`
and the batched driver falls back to the plain fused SDPA. The optimization is reached by the
SERIAL driver and, in practice, only by it.

### Item 2 — Exact fused top-k in prefill (`EXO_DSV4_EXACT_TOPK_PREFILL`) — SHARED

```
   ... identical prefix through mlx-lm/mlx_lm/models/deepseek_v4.py:5494 ...
mlx-lm/mlx_lm/models/deepseek_v4.py:4774   SparseCompressedAttention.__call__   (21 of 43 layers)
mlx-lm/mlx_lm/models/deepseek_v4.py:4905     self.indexer(x, q_residual, self.rope, idx_cache, offset, seq_band=_seq_band)
mlx-lm/mlx_lm/models/deepseek_v4.py:4023   Indexer.__call__
mlx-lm/mlx_lm/models/deepseek_v4.py:4146     k = min(self.index_topk, pooled.shape[1])
mlx-lm/mlx_lm/models/deepseek_v4.py:4160     with span("indexer.topk"):
mlx-lm/mlx_lm/models/deepseek_v4.py:4162-4165   approximate-fused branch requires scores.shape[1]==1 (:4163) -> skipped in prefill
mlx-lm/mlx_lm/models/deepseek_v4.py:4182-4188 if (_topk_result is None and _EXACT_TOPK
                                                 and (scores.shape[1] <= 16 or _EXACT_TOPK_PREFILL)
                                                 and "exact_topk_off" not in _topk_targets):   <-- GATED CODE
mlx-lm/mlx_lm/models/deepseek_v4.py:4186          exact = _exact_topk(scores, k)
mlx-lm/mlx_lm/models/deepseek_v4.py:3812        _exact_topk
```

`_EXACT_TOPK` defaults to `"1"` (`:3554`); `_EXACT_TOPK_PREFILL` is the audited env
(`:3562`, set to `1` by `start_cluster.sh:40`). The `scores.shape[1] <= 16 or _EXACT_TOPK_PREFILL`
disjunction is the *only* thing standing between a prefill chunk and the kernel, and it carries
no driver, batch-size, or queue-length term. The `self.indexer(...)` call at `:4905` is inside
the unconditional `else` of the nop-target check at `:4888`, so it fires on every
`SparseCompressedAttention` layer of every forward.

The seq-split band (`seq_band=_seq_band`) narrows `scores`' row count from 2048 to 1024
(`:4058-4061`) but never to ≤16, so the disjunction genuinely depends on the flag, exactly as
the P08 A/B assumed.

### Item 3 — Indexer path (`EXO_DSV4_INDEX_TOPK`, `EXO_DSV4_PREFILL_ARGPARTITION`, `EXO_DSV4_ARGPARTITION_MIN_P`) — SHARED

```
   ... identical prefix through mlx-lm/mlx_lm/models/deepseek_v4.py:4905 -> :4023 (Indexer.__call__) ...
mlx-lm/mlx_lm/models/deepseek_v4.py:3987   Indexer.__init__: _topk_env = os.environ.get("EXO_DSV4_INDEX_TOPK")
mlx-lm/mlx_lm/models/deepseek_v4.py:3988     self.index_topk = int(_topk_env) if _topk_env else config.index_topk
mlx-lm/mlx_lm/models/deepseek_v4.py:4146   k = min(self.index_topk, pooled.shape[1])
mlx-lm/mlx_lm/models/deepseek_v4.py:4203-4207 if (_topk_result is None and scores.shape[1] > 1
                                                 and EXO_DSV4_PREFILL_ARGPARTITION == "1"
                                                 and pooled.shape[1] >= EXO_DSV4_ARGPARTITION_MIN_P):   <-- GATED CODE
mlx-lm/mlx_lm/models/deepseek_v4.py:4207        _topk_result = mx.argpartition(-scores, kth=k-1, axis=-1)[..., :k]
```

`EXO_DSV4_INDEX_TOPK` is consumed at **model construction**, not per-request, so it is
driver-independent by definition. The argpartition branch's gate is `scores.shape[1] > 1`
(any prefill chunk, serial or batched) plus a pool-depth threshold — no driver term.

Ordering note, not a reachability finding: with `EXO_DSV4_EXACT_TOPK_PREFILL=1` (production),
item 2's branch at `:4182` sets `_topk_result` first, so item 3's argpartition branch at `:4203`
is skipped by its own `_topk_result is None` precondition. That is intra-item precedence between
two flags on the *same* serial path, not a serial-vs-batched bypass — both branches are equally
reachable from the serial driver and their relative order is identical under both drivers.

### Item 4 — Prefix-cache keying / snapshot insertion (`EXO_LEAF_SNAPSHOT_RETENTION`) — SHARED (serial-exclusive)

Capture side (inside the serial driver only):

```
src/exo/worker/engines/mlx/generator/generate.py:741    prefill
src/exo/worker/engines/mlx/generator/generate.py:828      has_ssm = has_non_kv_caches(cache)
src/exo/worker/engines/mlx/cache.py:499                     has_non_kv_caches -> is_non_trimmable_cache_entry
src/exo/worker/engines/mlx/cache.py:496                       isinstance(c, (ArraysCache, RotatingKVCache, CacheList)) -> True for DSv4
src/exo/worker/engines/mlx/generator/generate.py:839      progress_callback  (passed to stream_generate at :946)
src/exo/worker/engines/mlx/generator/generate.py:859        snapshots.append(snapshot_ssm_states(cache, snapshot_offset + processed))
src/exo/worker/engines/mlx/generator/generate.py:867-868    trim to _SNAPSHOT_RETENTION (= 2, generate.py:344)
mlx-lm/mlx_lm/generate.py:528                             prompt_progress_callback(...) fires it once per chunk
src/exo/worker/engines/mlx/generator/generate.py:1003     _final_snapshots = snapshots[:-1]
```

`has_ssm` is True for DSv4 because `is_non_trimmable_cache_entry` treats `CacheList`
structurally (`cache.py:496`), and every compressed/sparse DSv4 layer is a `CacheList`
(`deepseek_v4.py:7399`, `:7408`).

Keying / insertion side:

```
src/exo/worker/engines/mlx/generator/batch_generate.py:2559  prefill(..., snapshot_offset=prefix_hit_length)   # :2569
src/exo/worker/engines/mlx/generator/batch_generate.py:2586    with T("submit.save_prefix_cache")
src/exo/worker/engines/mlx/generator/batch_generate.py:2591      self._save_prefix_cache(all_prompt_tokens, cache, cache_snapshots, prefix_hit_length, matched_index, ...)
src/exo/worker/engines/mlx/generator/batch_generate.py:5256    _save_prefix_cache
src/exo/worker/engines/mlx/generator/batch_generate.py:5289      kv_prefix_cache.update_kv_cache(...)     (matched leaf + hit-ratio path)
src/exo/worker/engines/mlx/generator/batch_generate.py:5300      kv_prefix_cache.add_kv_cache(...)        (new-leaf path)
src/exo/worker/engines/mlx/cache.py:941-945                     merged = old snapshots <= restore_pos, then extend
src/exo/worker/engines/mlx/cache.py:963-964                     if len(merged) > _LEAF_SNAPSHOT_RETENTION: _select_spaced_snapshots(...)
src/exo/worker/engines/mlx/cache.py:81                          _LEAF_SNAPSHOT_RETENTION = int(os.environ.get("EXO_LEAF_SNAPSHOT_RETENTION", "4"))
```

The lookup that produces the key (`prefix_hit_length`, `matched_index`) is likewise on the serial
path: `batch_generate.py:2386-2389` (`kv_prefix_cache.get_kv_cache(...)`).

**The batched driver reaches strictly LESS of this than the serial one**, which is the opposite of
a bypass and worth stating explicitly:
- `prefill_batched` never snapshots — it returns a list of empty lists
  (`src/exo/.../generate.py:1635`, with the comment "Empty SSM snapshots — DSv4 doesn't use
  ArraysCache").
- `submit_batched` (`batch_generate.py:2789-3238`) contains **no** `_save_prefix_cache` /
  `add_kv_cache` / `update_kv_cache` call at all; its per-task caches are built fresh with the
  prefix cache deliberately bypassed (`batch_generate.py:2972-2977`).

### Item 5 — `EXO_PREFILL_STEP_SIZE` chunking — SHARED

```
src/exo/worker/engines/mlx/builder.py:179    BatchGenerator(...)  -- no prefill_step_size kwarg passed
src/exo/worker/runner/llm_inference/batch_generator.py:413   prefill_step_size: int | None = None   (stays None)
src/exo/worker/runner/llm_inference/batch_generator.py:470     -> ExoBatchGenerator(prefill_step_size=None)
src/exo/worker/engines/mlx/generator/batch_generate.py:2568   prefill(..., prefill_step_size=self.prefill_step_size)  # None
src/exo/worker/engines/mlx/generator/generate.py:903-904      if prefill_step_size is None:
                                                                prefill_step_size = int(os.environ.get("EXO_PREFILL_STEP_SIZE", "4096"))   <-- ENV READ, SERIAL PATH
src/exo/worker/engines/mlx/generator/generate.py:943          stream_generate(..., prefill_step_size=prefill_step_size)
mlx-lm/mlx_lm/generate.py:473                                 _chunk = prefill_step_size
mlx-lm/mlx_lm/generate.py:474                                 n_to_process = min(_chunk, remaining)
mlx-lm/mlx_lm/generate.py:481                                 _model_call(input_tokens=prompt[:n_to_process][None], ...)   <-- CHUNK APPLIED
```

`prefill_batched` has its own, separate copy of the same env read at `generate.py:1412`.
Both drivers honour the knob; the serial one demonstrably does so via the chain above.

One real, already-documented asymmetry, recorded for accuracy: the `// min(4, group.size())`
per-rank halving at `src/exo/.../generate.py:522` lives in `_pipeline_parallel_prefill_steps`,
i.e. the **PP** loop only. Production (TP, `is_pipeline=False`) never executes line 522, so the
serial chunk fed to `_model_call` is the full 2048. The 1024 figure that appears in TP traces
comes from the *seq-split* band (`deepseek_v4.py:4521-4533`), not from line 522. This does not
change the verdict; it is noted so a follow-up does not mis-attribute the halving.

### Item 6 — Clear-cache cadence (`EXO_PREFILL_CLEAR_CACHE_INTERVAL`) — SHARED (serial-exclusive)

```
src/exo/worker/engines/mlx/generator/generate.py:936   stream_generate(...)
mlx-lm/mlx_lm/generate.py:833                          generate_step(...)
mlx-lm/mlx_lm/generate.py:468                          prefill chunk loop
mlx-lm/mlx_lm/generate.py:544                            _clear_interval = int(os.environ.get("EXO_PREFILL_CLEAR_CACHE_INTERVAL", "1"))   <-- GATED CODE
mlx-lm/mlx_lm/generate.py:545                            _chunk_idx += 1
mlx-lm/mlx_lm/generate.py:546-547                        if _clear_interval > 0 and _chunk_idx % _clear_interval == 0: mx.clear_cache()
```

This knob exists **only** in mlx-lm's `generate_step` loop — which is precisely the loop the
production serial driver runs. `prefill_batched` does not read it and calls `mx.clear_cache()`
unconditionally every chunk (`src/exo/.../generate.py:1563`); exo's PP loop is likewise not the
production path. At the production value `1`, the cadence is identical to the unconditional
behaviour, so the knob is reached and is currently a no-op *by value*, not by unreachability.

---

## WHAT WOULD CHANGE IF BYPASSED

**Nothing — no item is BYPASSED and no item is AMBIGUOUS, so no follow-up measurement is
triggered under the round-8 pre-registration.** Per PRE-REGISTRATION.md: *"If all SHARED: close
I12, audit IS the deliverable."*

Recorded for completeness, so a future reader can see what the trigger *would* have been and
what the pre-registered expectations were:

- **Item 1 (tiled compressed SDPA), pre-registered expectation had it been BYPASSED: +7.2%**
  (original campaign-1 A/B delta). The function that would have needed to call what:
  `CompressedAttention.__call__` (`mlx-lm/mlx_lm/models/deepseek_v4.py:4450`) would have had to
  reach the tiled branch at `:4540`. It does — via `DeepseekV4Block.__call__:5494` on the shared
  `model(...)` path entered from `stream_generate`. No change required.
- **Item 2 (exact fused top-k prefill), pre-registered expectation had it been BYPASSED: +1.6%**
  (original campaign-1 A/B delta). The function that would have needed to call what:
  `Indexer.__call__` (`:4023`) would have had to reach `_exact_topk` (`:3812`) via the
  `_EXACT_TOPK_PREFILL` disjunction at `:4184`. It does — via
  `SparseCompressedAttention.__call__:4905`. No change required.

**Residual risk this audit does NOT cover (stated rather than silently assumed).** This is a
static reachability audit. It proves the serial call path *reaches* each gated site; it does not
prove each gate's runtime *predicate* evaluates true on live production tensors. The one item
where that distinction has real teeth is item 1, whose `_query_tiled_ok` shape gate
(`deepseek_v4.py:3617-3643`) is genuinely data-dependent. I audited all five of its conditions
against the deployed config (B=1, `n_q`=1024/2048 ≥ 128, `sliding_window`=128,
`RotatingKVCache(max_size=128)`) and each holds — but "holds under my reading of the shapes" is a
weaker claim than "observed true in a live run". If cheap runtime confirmation is ever wanted, the
zero-relaunch way to get it is a counter or one-line log inside the `:4540` branch on the next
relaunch that happens for another reason; it does not justify a relaunch of its own, and the
round-8 pre-registration does not require one.

## Assumptions and open questions

1. **Assumed:** production runs `MLX_JACCL_SHARDING_MODE=Tensor` (stated in
   PRE-REGISTRATION.md's "Environment observed at round start"). The entire audit's second fork
   (`is_pipeline=False` → `stream_generate`) depends on this. Under PP the serial driver would
   instead run `pipeline_parallel_prefill` (`generate.py:911`) and items 5/6 would need re-tracing,
   since exo's PP chunk loop has its own step-size halving (`:522`) and does not read
   `EXO_PREFILL_CLEAR_CACHE_INTERVAL` at all.
2. **Assumed:** the deployed 6-bit checkpoint shares the audited `compress_ratios` layer census.
   I read `mlx-community/DeepSeek-V4-Flash` and `deepseek-ai/DeepSeek-V4-Flash-0731` configs from
   the local HF cache; both give 20 `CompressedAttention` + 21 `SparseCompressedAttention` layers.
   I did not verify which snapshot directory the live runner loaded (that would need cluster
   contact, which is out of scope).
3. **Verified, not assumed:** the submodule `mlx-lm/` working tree is byte-identical to the
   installed `site-packages/mlx_lm` for both audited files, so the cites point at executing code.
