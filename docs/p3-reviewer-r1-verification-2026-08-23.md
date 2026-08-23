# P3 Reviewer R1 — independent verification of Worker A and Worker B1 (2026-08-23)

**Scope**: read-only re-derivation of the central claims in
`docs/p3-worker-a-kv-read-inventory-2026-08-23.md` and
`docs/p3-worker-b1-live-depth-anchors-2026-08-23.md` from the primary sources
(code, config, and the workers' own pasted raw output). No cluster contact, no
bench scripts run, no code edited. The only file modified is the one
`PERFORMANCE_HISTORY.md` append at the bottom of this doc's deliverable list.

**Environment**: `~/repos/exo` @ `6bc843bfc`; submodule `mlx-lm` @ `1fea494`
(both matching what the workers reported). `git status --porcelain` before my
append showed only ` M docs/PERFORMANCE_HISTORY.md`, `?? bench/p3_depth_anchor_probe.py`,
and the two worker docs — consistent with "nothing committed".

**Independent config source**: rather than re-`ssh` the studio, I read the
checkpoint config from the local HF cache, which is a genuinely independent copy
of the same file Worker A `cat`'d on m4-1:
`~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json`.

---

## Verdict table

| # | Claim | Verdict |
|---|---|---|
| A1 | TP shards MoE only; attention REPLICATED; exo never calls mlx-lm `Model.shard()` | **CONFIRMED** (and the older doc is wrong) |
| A2 | KV cache is bf16 unquantized in production (`EXO_KV_CACHE_BITS:=0`, 0-sentinel honored) | **CONFIRMED** |
| A3 | Sparse core attention reads a top-k=512 subset, not all L | **CONFIRMED** |
| A4 | Indexer scans an L/4 pool × 128 dims × bf16 over 21 sparse layers → 1344 B/ctx-token | **CONFIRMED** |
| A5 | TP all_sum decode payload shapes are L-independent; 43 calls/token | **CONFIRMED** |
| A6 | `bytes_per_rank(L) = 5,297,553,408 + 1930.25·L`; ~5.978 GB @352.6K; +1.64 ms @297 GB/s | **CONFIRMED** (formula/table fully self-consistent) |
| A6b | §4 sanity aside: "~74× smaller than a naive dense-MLA estimate" | **REFUTED** — the real ratio is **6.39×** |
| B1a | `decode_probe.py`'s `{"bench": true}` on `/v1` is silently dropped; EOS ban only on `/bench` route | **CONFIRMED** |
| B1b | Anchor tok/s, ms/token, deltas; decode-only; 2000/2000 @ `finish_reason=length` | **CONFIRMED** |
| B1c | Uniform rightward shift, dispersion falls, no tail fattening | **CONFIRMED** |
| — | Both PERFORMANCE_HISTORY entries exist, correctly placed, no drift vs source docs | **CONFIRMED** (one stale-baseline note, see below) |

Summary: **A — 6/6 central claims CONFIRMED, 1 incidental sub-claim REFUTED
(the "74×" aside in §4). B1 — probe bug CONFIRMED, anchors arithmetically
sound.**

---

## Worker A

### A1 — MoE-only sharding, attention replicated — **CONFIRMED**

```
$ grep -rn '\.shard(' src/exo/
src/exo/shared/types/worker/instances.py:88:        shard = self.instance.shard(self.bound_runner_id)
```
Exactly one hit, and it is unrelated (an `Instance.shard(runner_id)` accessor,
not `nn.Module.shard`). Worker A's characterization is exact.

```
$ sed -n '1031,1034p' src/exo/worker/engines/mlx/auto_parallel.py
class DeepseekV4ShardingStrategy(TensorParallelShardingStrategy):
    """Sharding for DeepSeek V4 Flash / Pro — MoE-only.

    Replicates attention on every rank; shards only the MoE block.
```
The docstring is present at the cited lines. I did **not** stop at the docstring
(the skill's Error-3 lesson: comments are claims, not facts) — I checked the
loop body:

- `auto_parallel.py:1087` — `layer.ffn.sharding_group = self.group` (MoE only).
- `auto_parallel.py:1077` — `if os.environ.get("EXO_DSV4_QA_KV_FUSED", "0") == "1":`
  → the only conditional attention mutation, default **off**.
- `grep -n 'n_heads' src/exo/worker/engines/mlx/auto_parallel.py` → the only
  `n_heads //= self.N` in the file is at **line 890**, inside the *generic*
  `TensorParallelShardingStrategy.shard_model`, not the DSv4 subclass.
- Dispatch confirmed: `auto_parallel.py:766` —
  `elif isinstance(model, DeepseekV4Model): ... DeepseekV4ShardingStrategy(...)`.

The mlx-lm head-halving does exist but is unreachable from exo:
```
$ grep -n 'n_heads //=\|def shard' mlx-lm/mlx_lm/models/deepseek_v4.py
7156:    def shard(self, group: Optional[mx.distributed.Group] = None):
7170:            layer.attn.n_heads //= N
```
Its only caller is `mlx-lm/mlx_lm/utils.py:759` (`model.shard(tensor_group)`),
which lives inside `sharded_load` (defined at `utils.py:678`). exo imports
`load_model` (`src/exo/worker/engines/mlx/utils_mlx.py:41`) and `load`
(`pp_speculation.py:3504`) — **never `sharded_load`** — and routes TP through
`tensor_auto_parallel` at `utils_mlx.py:444`. So line 7170 never executes on
this cluster.

**The contradiction is resolved in Worker A's favour.**
`docs/dsv4-attention-kernel-efficiency-2026-08-18.md:38` (`64 -> 32 per TP
rank`) and `:52-55` ("plus `n_heads//=N`") describe mlx-lm's upstream
`shard()`, which exo's TP path does not call. That older doc is **wrong on this
specific point**. Note it is also internally muddled — line 52 asserts
"Attention is replicated, not sharded" and then line 53 lists head-halving as
part of that replication, which is self-contradictory. Depth-scaling verdicts
are unaffected either way; only the absolute constant term moves.

### A2 — bf16 unquantized KV — **CONFIRMED**

```
$ sed -n '148,151p' start_cluster.sh
if [ -n "$EXO_TURBOQUANT" ]; then
    EXO_KV_CACHE_BITS=""
else
    : "${EXO_KV_CACHE_BITS:=0}"
```
Cite is exact (line 151), and `start_cluster.sh:138` `: "${EXO_TURBOQUANT:=}"`
confirms the empty default, so the `else` branch is the live one. Exported at
`start_cluster.sh:1584`.

```
$ sed -n '2391p' src/exo/worker/engines/mlx/cache.py
    return KV_CACHE_BITS or None
```
Worker A cited `cache.py:~2393`; the actual return is **line 2391** (the guard
comment spans 2386-2390). Off by two — the `~` in the doc makes this acceptable,
and the mechanism is exactly as described. `constants.py:16` shows
`KV_CACHE_BITS: int | None = int(os.environ["EXO_KV_CACHE_BITS"]) if os.environ.get("EXO_KV_CACHE_BITS") else None`
→ with the env var set to the string `"0"`, `KV_CACHE_BITS == 0`, and
`0 or None` → `None`. The install branch is `cache.py:2443` `elif bits is not
None:` → **not taken**. `QuantizedKVCache` never installs. Confirmed.

Third leg also holds: `deepseek_v4.py:4078`, `:4241`, `:4544` all read
`kv = kv.astype(mx.bfloat16)  # keep KV cache bf16 (batch-invariant)`.

This independently re-confirms the finding already recorded in the
`exo-cluster-debugging` skill (Error 2): KV quantization is a no-op here.

### A3 — sparse attention reads top-k=512, not all L — **CONFIRMED**

Config (independent copy, read via `json.load`):
`index_topk = 512`, `index_head_dim = 128`, `index_n_heads = 64`,
`head_dim = 512`, `sliding_window = 128`, `num_hidden_layers = 43`,
`num_key_value_heads = 1`. Every dimension Worker A quoted in §0 matches
byte-for-byte.

Not overridden downward: `start_cluster.sh:33` `: "${EXO_DSV4_INDEX_TOPK:=512}"`,
exported at `:1681`; read at `deepseek_v4.py:3760`
(`self.index_topk = int(_topk_env) if _topk_env else config.index_topk`) — so
env and config agree on 512 regardless of which wins.

Gather is bounded by k, verified at `deepseek_v4.py:2534-2545`:
```python
        P_dim = pooled.shape[1]
        k_dim = topk.shape[2]
        with span("attn.gather"):
            pooled_flat = pooled.reshape(B * P_dim, D)
            offset = (mx.arange(B) * P_dim).reshape(B, 1, 1)
            topk_flat = (topk + offset).reshape(-1)
            pooled_kv = pooled_flat[topk_flat].reshape(B, L, k_dim, D)
```
`topk_flat` has exactly `B·L·k_dim` elements, so the gather touches `k_dim`
rows per query irrespective of `P_dim`. `k = min(self.index_topk, pooled.shape[1])`
at `deepseek_v4.py:3888` (doc said 3885 — off by three, mechanism identical).
The OPT-10 comment at `:2529-2533` states "does NOT scale with P" and the code
backs it. Combined width `SW + K = 128 + 512 = 640` rows: confirmed via
`combined_kv = mx.concatenate([local_kv, pooled_kv], axis=2)` at `:2549`.

Dense-fallback guard confirmed at **`deepseek_v4.py:4614`**
(`elif pooled.shape[1] <= self.indexer.index_topk:`) — Worker A cited
`4448-4449`, which is a **wrong line number** (4448 is `return self.wq_a(x),
self.wkv(x)`). The guard exists and behaves as described; only the cite is
stale. Non-material.

### A4 — indexer scans the L/4 pool over 21 sparse layers — **CONFIRMED**

Layer census, recomputed from the real config rather than trusting the doc's
table:
```python
>>> collections.Counter(config['compress_ratios'][:43])
Counter({4: 21, 128: 20, 0: 2})
```
`len(compress_ratios) == 46` raw, truncated to 43 at
`mlx-lm/mlx_lm/models/deepseek_v4.py:888`
(`self.compress_ratios = list(self.compress_ratios[: self.num_hidden_layers])`)
— cite exact. **21 sparse / 20 compressed / 2 local confirmed independently.**
Class mapping confirmed at `deepseek_v4.py:4844-4851` (`v4_attention_factory`),
cite exact.

L/4 pooling confirmed structurally: `Model.make_cache` (`deepseek_v4.py:6956-6979`,
cite exact) builds `CacheList(RotatingKVCache(sliding_window), PoolingCache(ratio),
PoolingCache(ratio))` for `SparseCompressedAttention`, and `PoolingCache`
(`mlx-lm/mlx_lm/models/cache.py:1270`) emits one pooled entry per `ratio`
positions (`accumulate_windows`, `usable = (total // self.ratio) * self.ratio`,
`cache.py:1382+`). With `ratio == 4`, `P = L/4`. Indexer pool width is
`index_head_dim = 128` (`deepseek_v4.py:3753`, `self.head_dim = config.index_head_dim`).

Full-pool read confirmed: `_indexer_score` at `deepseek_v4.py:3680` is
`return q_weighted @ pooled.swapaxes(-1, -2)  # (B, L, P)` — a GEMM against the
**entire** P-length pool, with the 64 heads folded into the query beforehand
(`:3676-3679`), so the pool is read once, not 64×. Untiled path is live:
`_INDEXER_PBLOCK = int(os.environ.get("EXO_DSV4_INDEXER_PBLOCK", "0"))`
(`deepseek_v4.py:327`), gate `if _INDEXER_PBLOCK > 0:` at `:3823`, and
`start_cluster.sh:1958` exports it only conditionally (`[ -n ... ] &&`), so it
stays 0.

Slope arithmetic re-derived independently: `21 · (1/4) · 128 · 2 = 1344.0`
B/context-token. ✔ Matches. Share of the depth slope: `1344/1930.25 = 69.63%`
— the doc's "70%" is fair.

4-pass topk kernel confirmed: `_EXACT_TOPK = ... "1"` at `deepseek_v4.py:3426`
(default on), decode gate `scores.shape[1] <= 16` at `:3921`, and four separate
`for (uint i = ...; i < P; ...)` / `for (uint i = lo; i < hi; i++)` loops over
`scores[row + i]` in the Metal source at the cited offsets. `EXO_DSV4_TOPK_FUSED`
defaults `"0"` (`:3899`) and is only conditionally exported
(`start_cluster.sh:1680`), so the approximate kernel is off. ✔

### A5 — all_sum payloads are L-independent — **CONFIRMED**

My own grep reproduces Worker A's, with one addition the doc's §3 grep block
omitted:
```
$ grep -n 'mx.distributed.all_sum\|mx.distributed.all_gather' mlx-lm/mlx_lm/models/deepseek_v4.py
507, 508, 528, 536   (wrapper capture / monkey-patch install)
3007, 4117, 4360, 4369, 4379, 4818, 4827, 4837, 5416, 6758
```
**`4379` is missing from Worker A's pasted grep** (it lists 4360/4369 but not
4379). I checked it: `deepseek_v4.py:4375` `elif self.sharding_group is not
None and _ATTN_ALLSUM:` → it is the `CompressedAttention` tail all_sum, gated by
the same dead `_ATTN_ALLSUM` flag as 4117/4837, and its payload is `out`
(B, L, hidden). **Its omission does not change the verdict** — it is dead for
the same reason and its shape is L_q-shaped, not cache-length-shaped. Flagged
as an incompleteness in the doc's evidence block, not an error in its
conclusion.

Gating verified:
- `_ATTN_ALLSUM = os.environ.get("EXO_DSV4_ATTN_ALLSUM", "1") == "1"`
  (`deepseek_v4.py:1626`, cite exact), but production sets
  `: "${EXO_DSV4_ATTN_ALLSUM:=0}"` (`start_cluster.sh:1755`, cite exact,
  exported `:1756`). All four tail sites (4113, 4375, 4833, 5414) are gated on
  it → dead.
- `_SEQ_SPLIT_MIN_L = int(os.environ.get("EXO_DSV4_SEQ_SPLIT_MIN_L", "16"))`
  (`deepseek_v4.py:225`, cite exact); gates at `:4262` and `:4478` require
  `L >= _SEQ_SPLIT_MIN_L`, and decode has L=1 → seq-split gathers dead.
- Surviving site `deepseek_v4.py:3007` `y = mx.distributed.all_sum(y,
  group=self.sharding_group)` inside `span("moe.all_sum")`, cite exact.
  `y` is the post-combine MoE output, `(B, L_q, hidden) = (1, 1, 4096)`.

The 43 calls/token figure follows from `auto_parallel.py:1087` setting
`layer.ffn.sharding_group` on every layer, with `num_hidden_layers = 43` from
config. The `auto_parallel.py:1141-1152` correction comment is present and says
exactly what the doc quotes — and unlike the stale comment that caused the
skill's Error 3, this one is the *corrected* version and matches
`_all_to_sharded`'s `max(weight.ndim - 2, 0)` axis selector.

**No dimension of any decode-reachable collective is a function of cache
length.** Confirmed.

### A6 — arithmetic — **CONFIRMED**, with one refuted aside

Recomputed every component from the doc's own expressions (not its results):

Constants: `A1 = 43·128·512·2 = 5,636,096`; `A2 = 21·512·512·2 = 11,010,048`;
`A3 = 21·(128+512)·512·2·2 = 27,525,120`; plus the doc's `A4 = 5,253,382,144`.
Sum = **5,297,553,408** — **exactly** the doc's C total.

Depth slope: B1 1344.00, B2 10.50, B3 42.00, B4 480.00, B5 42.00, B6 10.50,
B7 1.25 → **1930.25** exactly. Every row reproduces from its stated expression;
no rounding drift.

Evaluations: `bytes_per_rank(100000) = 5.4906 GB`, `(352595) = 5.9781 GB`,
`(500000) = 6.2627 GB` — all three match the doc's table to 4 decimals.
At the task's stated L = 352,599: **5.97816 GB ≈ 5.978 GB** ✔.
Delta 100K→352.6K = 487,571,499 B = **+487.6 MB** ✔.

At 297 GB/s: 18.487 / 20.128 / 21.086 ms, **Δ = +1.6417 ms** (100K→352.6K) and
+0.958 ms (352.6K→500K) — matches the doc's +1.64 / +0.96. The other three
bandwidth rows (327.6, 409.5, 546 GB/s) also reproduce to the doc's stated
2 decimals, including the deltas. **No inconsistency between the component
table and the headline formula.**

§4 resident-cache table also reproduces exactly (0.694 / 2.431 / 3.446 GB;
6936 / 6896 / 6891 B/token).

**REFUTED sub-claim (A6b).** `docs/p3-worker-a-kv-read-inventory-2026-08-23.md:417-419`
says the ~6.9 KB/token resident figure "is ~74× smaller than a naive dense-MLA
estimate would give (43 layers × 512 dims × 2 B = 44 KB/token)". The doc states
its own inputs; `44032 / 6891 = **6.39×**, not 74×`. The 44 KB/token figure and
the 6.9 KB/token figure are both correct — only the stated ratio between them
is wrong, by ~11.6×. This is a rhetorical sanity-check aside in §4 and touches
**no** formula, table, or depth verdict; nothing downstream depends on it. It
should not be repeated in the synthesis.

**Non-blocking note on §5's framing** (not a claim I was asked to adjudicate,
but it feeds the synthesis): §5 expresses +1.64 ms as "≈3% of one token's wall
time" against a "~54.6 ms/token" baseline taken from
`docs/decode-roofline-dispatch-bound-2026-08-21.md:32-33`. Worker B1's fresh
measurement at that exact depth is **42.59 ms/token**, against which the same
1.64 ms is **3.85%**. B1 already flagged this; I confirm the recomputation.

---

## Worker B1

### B1a — the `bench` field is silently dropped on `/v1` — **CONFIRMED**

```
$ grep -n 'class ChatCompletionRequest\|class BenchChatCompletionRequest' src/exo/api/types/api.py
243:class ChatCompletionRequest(BaseModel):
286:class BenchChatCompletionRequest(ChatCompletionRequest):
```
Cite (`api.py:243`) is exact. I read the full field list at `243-284`: `model`,
`frequency_penalty`, `messages`, `logit_bias`, `logprobs`, `top_logprobs`,
`max_tokens`, `n`, `presence_penalty`, `response_format`, `seed`, `stop`,
`stream`, `stream_options`, `temperature`, `top_p`, `top_k`, `tools`,
`reasoning_effort`, `enable_thinking`, `min_p`, `repetition_penalty`,
`repetition_context_size`, `tool_choice`, `parallel_tool_calls`, `user`,
`service_tier`, `correlation_id`. **There is no `bench` field.** It is a plain
`BaseModel` with no `model_config`/`ConfigDict` anywhere in the file
(`grep -n 'model_config\|ConfigDict\|extra=' src/exo/api/types/api.py` → no
hits), so pydantic v2's default `extra='ignore'` applies and the key is dropped
without error. `BenchChatCompletionRequest` (`:286`) adds only
`use_prefix_cache: bool = False` — it does **not** add `bench` either; the flag
is force-set server-side.

Route binding confirms which model each endpoint parses:
```
$ sed -n '451,456p' src/exo/api/main.py
        self.app.post("/v1/chat/completions", response_model=None)(
            self.chat_completions
        )
        self.app.post("/bench/chat/completions", response_model=None)(
            self.bench_chat_completions
        )
```
`chat_completions(self, payload: ChatCompletionRequest)` at `main.py:1129-1130`;
`bench_chat_completions(self, payload: BenchChatCompletionRequest)` at
`main.py:1183-1184`. The `/bench` handler force-sets the flag at
`main.py:1192-1197`:
```python
        task_params = task_params.model_copy(
            update={
                "stream": False,
                "bench": True,
                "use_prefix_cache": payload.use_prefix_cache,
            }
        )
```
`chat_request_to_text_generation` (`src/exo/api/adapters/chat_completions.py:62`)
contains **zero** occurrences of `bench` (`grep -n 'bench'` → no hits), so the
`/v1` path can never set it; `TextGenerationTaskParams.bench` defaults to
`False` (`src/exo/shared/types/text_generation.py:137`).

EOS-ban site confirmed at the cited line:
```
$ sed -n '2658,2660p' src/exo/worker/engines/mlx/generator/batch_generate.py
            if is_bench:
                eos_ids = eos_ids_from_tokenizer(self.tokenizer)
                logits_processors = [ban_token_ids(eos_ids)] + logits_processors
```
with `is_bench = task_params.bench` at `batch_generate.py:2233`. Cite exact.

And the probe really does hit the wrong route:
`bench/decode_probe.py:23` `"bench": True,` in the body, posted at `:30` to
`f"{base_url}/v1/chat/completions"`, while its docstring at `:3-4` claims the
EOS ban. **Bug confirmed end-to-end from code alone**; I did not re-run B1's
live A/B, which was not needed.

### B1b — anchor arithmetic — **CONFIRMED**

Recomputed from the raw pasted `DECODE WINDOW` and `completion_tokens` in B1's
§2, not from its headline table:

| depth | window | tokens | my tok/s | my ms/tok | doc |
|---|---|---|---|---|---|
| 520 | 67.47 s | 2000 | 29.64 | 33.73 | 29.63 / 33.75 |
| 100,026 | 71.55 s | 2000 | 27.95 | 35.77 | 27.94 / 35.79 |
| 352,599 | 85.13 s | 2000 | 23.49 | 42.56 | 23.48 / 42.59 |

All within ±0.03 — pure display-rounding of the 2-dp window. The event-based
numbers also reproduce (1844/67.47 = 27.33 vs 27.32; 1988/71.55 = 27.78 vs
27.77; 1987/85.13 = 23.34 vs 23.33).

Deltas from my own ms/token: 520→100,026 = **+2.04 ms** (doc: +2.04);
100,026→352,599 = **+6.79 ms** (doc: +6.80); total = **+8.83 ms** (doc: +8.84),
tok/s change −6.15 = **−20.7%** (doc: −20.8%). Per-100K normalization:
2.04/0.995 = **+2.05 ms/100K** and 6.79/2.526 = **+2.69 ms/100K** — both exactly
as claimed, so the "mildly super-linear, not saturating" reading is sound.

The task brief's "+6.80 ms 100K→352.6K" matches the doc's segment table; note
this is the **520→deep total minus the first span**, i.e. the same +6.79/+6.80
number, not an independent third quantity.

**Decode-only accounting confirmed** against the doc's stated methodology
(§1.2: "Decode window = last streamed event − first streamed event. TTFT/prefill
is outside the window by construction"). Cross-check: at 352,599 the total wall
clock is 1143.75 s and TTFT 1058.62 s; `1143.75 − 1058.62 = 85.13 s` = exactly
the reported decode window. Same identity holds at 520 (70.84 − 3.37 = 67.47)
and at 100,026 (345.26 − 273.71 = 71.55). Prefill is genuinely excluded.

**2000/2000 with `finish_reason=length` confirmed at all three points** from
the raw blocks: each shows `completion_tokens: 2000`, `finish_reason: length`,
and `'completion_tokens': 2000` inside the full usage dict, with
`'cached_tokens': 0` — so no prefix-cache shortcut turned a deep run shallow.
Depths are read from `usage.prompt_tokens` (520 / 100026 / 352599), matching the
headline table.

T1 comparison arithmetic also checks: (27.94−26.91)/26.91 = **+3.83%** (doc
+3.8%); (23.48−21.51)/21.51 = **+9.16%** (doc +9.2%); 1000/21.51 = 46.49 ms and
1000/26.91 = 37.16 ms, matching the T1 ms/tok column.

### B1c — uniform shift, no tail fattening — **CONFIRMED as internally consistent**

Recomputed the doc's "deep − short" column from its own p-values:
p10 +8.41, p50 +7.31, p90 +0.83, p99 +16.37, max −1913.81, mean +6.25,
stdev −35.93 — **every cell matches** the doc's §3 table.

The three sub-claims are consistent with those numbers:
1. *Whole distribution translates rightward*: p10 18.02→22.77→26.43,
   p50 31.85→34.28→39.16, mean 36.61→36.01→42.86. p10 and p50 are strictly
   monotone in depth and move by comparable amounts (+8.41 vs +7.31) — the fast
   decile slows as much as the median, which is the signature of a uniform
   shift rather than a burst. Sound. (Caveat the doc itself does not state: the
   *mean* is **not** monotone — 36.61 at short vs 36.01 at 100K — because the
   short run's two multi-second spikes inflate its mean. The doc's own text
   handles this correctly by attributing the spikes to short context, but a
   reader skimming the mean row could be misled. Minor presentational nit, not
   an error.)
2. *Tail does not fatten*: outlier fraction 1.25% (short) vs 0.55% (100K) and
   0.76% (deep); max 2111.30 (short) vs 138.23 / 197.49 (deep). Both metrics
   move the opposite way from "tail fattening". Sound.
3. *Dispersion falls*: stdev 55.55 → 16.28 → 19.62. Sound, though note the drop
   is short→100K; 100K→deep is a mild *rise* (16.28→19.62). The doc's phrasing
   "dispersion does not grow with depth" is defensible only because the deep
   stdev stays far below the short one; strictly, between the two deep points
   dispersion rises ~20% while the mean rises ~19% — i.e. the coefficient of
   variation is roughly flat. That is still fully consistent with "uniform
   multiplicative-ish shift, no new tail", so the conclusion stands.
4. p50-implied throughput: 1000/31.85 = 31.40, 1000/34.28 = 29.17,
   1000/39.16 = 25.54 — matches the doc's 31.39 / 29.17 / 25.54, monotone in
   depth. Sound.

Prefill aside (§5) also reproduces: 100026/273.71 = 365.4 tok/s and
352599/1058.62 = 333.1 tok/s, as stated.

---

## PERFORMANCE_HISTORY.md entry check — **both present, correctly placed, no drift**

```
$ grep -n '^\*\*NEW (' docs/PERFORMANCE_HISTORY.md | tail -3
756:**NEW (2026-08-23, P4): TP=2 width-sharding does NOT create skinny-GEMM
2343:**NEW (2026-08-23, P3 worker A): attention-path read bandwidth is NOT the
2385:**NEW (2026-08-23, P3 worker B1): fresh live depth anchors with a REAL
```
Both exist, both in the same trailing region as the 2026-08-22 session-4
`NEW (...)` entries (T2–T10 at lines 1781–2054), immediately before the
"Quick-reference: closed levers" appendix. Placement is correct and consistent.

Drift check against the source docs: I read both entries in full (lines
2343-2384 and 2385-2429) and compared every quantitative claim to the underlying
doc. **No drift found.** Specifically verified: the formula (5.298 GB + 1930 B·L)
and all three evaluations; "1344 of the 1930 B/token (70%)"; the
`EXO_KV_CACHE_BITS:=0` / `cache.py:2393` pair; `+1.64` / `+0.96` ms; the
`ATTN_ALLSUM:=0` and `L >= 16` gating; the `Model.shard()` correction and its
"depth verdicts unaffected" hedge; and on the B1 side all three anchors,
2000/2000 + `finish_reason=length` + `cached_tokens=0`, the p10/p50/mean/stdev
/outlier figures, +8.84 ms / −20.8%, +2.05 and +2.69 per 100K, and the +3.8% /
+9.2% T1 comparisons with the "NOT claimed as an improvement" hedge intact.

Two observations, neither a drift:
- The worker-A entry carries the "~3% of the ~54.6 ms/token baseline" figure.
  B1's entry (appended later) explicitly corrects the denominator to the
  measured 42.59 ms/token → ~3.9%. The history file therefore contains both;
  a reader taking only the A entry gets a slightly understated share. Worth the
  synthesis stating the corrected 3.9% once.
- The A entry's "74×" aside was **not** carried into PERFORMANCE_HISTORY — only
  into the source doc §4. So the refuted number is not in the durable record.

---

## Method notes / what I did not do

- Read-only throughout. No bench script executed, no `curl`, no `ssh`, no
  cluster contact. `start_cluster.sh` was read with `sed -n`, never run.
- I did not re-run B1's live probes; B1a is fully decidable from code and B1b/c
  are arithmetic over B1's own pasted raw output, which is the correct
  independent check (recomputing from the raw fields rather than trusting the
  summary table).
- I did not attempt to validate A4's byte-count *modelling assumptions* (whether
  MLX materializes `mx.concatenate`, whether SDPA re-reads per head-group).
  Worker A flags both as open in its §6 and the depth verdict is insensitive to
  both, as its own sensitivity analysis (m → 1610.25, prediction → +1.37 ms)
  shows. Those remain the right things for a measurement worker to close, not a
  code reviewer.
- Line-number drift found and recorded in three places (`cache.py:2393`→2391,
  `deepseek_v4.py:3885`→3888, `deepseek_v4.py:4448-4449`→4614). All cosmetic;
  every cited mechanism exists.
