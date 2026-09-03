# VERIFICATION — independent audit of 5 high-stakes round-1 claims

Auditor: independent verification subagent, 2026-09-03. **READ-ONLY** — no patches, no
commits, no cluster changes, no inference requests. Cluster touched only with `ssh` +
`ps`/`ls`/`cat`/`md5`/`grep` and two read-only HTTP GETs (`/v1/models`, `/state`).
Microbenchmarks run on **this MacBook** (Apple M4 Max, 36 GB) against the live venv
(`/Users/adam.durham/repos/exo/.venv`), not on the Studios.

## HEADLINE

| # | Claim | Verdict |
|---|---|---|
| 1 | Per-forward collective count at decode is 43, not 86 | **VERIFIED** |
| 2 | Attention is REPLICATED, not head-sharded; `ATTN_ALLSUM=0` live | **VERIFIED** |
| 3 | Routed experts deployed at mxfp4 b4 g32 | **VERIFIED** (with two additions I3 omitted) |
| 4 | I3 achieved 284.9 GB/s = 52.2% of peak at M=4 | **REFUTED** — wrong quantization mode benched, wrong bytes constant. Corrected: **341.9 GB/s = 62.6%**, which **changes the band that fired** |
| 5 | I4's AMBIGUOUS band call | **REFUTED** — the band's primary clause was NOT met. Correct verdict is **BLOCKER STANDS**. No partial hit was ever served |

🚩 **TWO REFUTATIONS. Both change a decision.** Claim 4 moves I3 out of the
"<60% → kernel work is FUNDED" band into the "60–80% → decision waits on I1" band.
Claim 5 inverts I4's re-open recommendation.

Also corrected in passing: the live-build provenance claim (`repos/exo/mlx` @ `e40a416b2`)
is **VERIFIED**, and I6's "the venv `mlx` on this MacBook is the same" is **wrong** —
see the note under Claim 1.

---

# CLAIM 1 — "the per-forward collective count at decode is 43, not 86"

## VERDICT: **VERIFIED**

### (a) `sum_gradients`' forward really is identity — CONFIRMED

Live `mlx` build provenance first, because the claim hinges on reading the right file.
On **192.168.86.201** (`~/repos/exo/.venv/.../mlx-0.32.1.dev20260822+e40a416b2.dist-info/direct_url.json`):

```json
{"url":"file:///Users/adam.durham/repos/exo/mlx","dir_info":{}}
```

`~/repos/exo/mlx` HEAD = `e40a416b20851d118b061b3a57d8cab70f5756de`. Matches the version
string. **The prior agent's provenance claim is VERIFIED.**

> ⚠️ **Correction to I6, and a trap for anyone reproducing this.** The venv on **this
> MacBook** is a *different* install: its dist-info is
> `mlx-0.32.0.dev20260804+ac73d0c9` with
> `direct_url.json = {"url":"https://github.com/adurham/mlx.git", "commit_id":"ac73d0c9e..."}`
> — i.e. the MacBook venv's mlx came from **`~/repos/mlx`** (HEAD `ac73d0c9e`), the
> WRONG tree. I confirmed the file I read is nonetheless the right one by md5:
> `repos/exo/mlx/python/mlx/nn/layers/distributed.py` = `4132533cc67594a3a79c09926377b7e2`
> = the Studio's installed `mlx/nn/layers/distributed.py`. So my cite is against live
> code, but **I6's line 50-52 statement that "`~/repos/exo/mlx` on this machine is
> byte-identical to the Studio" is true of the source tree and NOT of what this
> MacBook's venv imports.** Anyone re-running an mlx-behavioural bench locally is
> testing `ac73d0c9e`, not `e40a416b2`.

`repos/exo/mlx/python/mlx/nn/layers/distributed.py:14-27`, verbatim:

```python
@lru_cache
def sum_gradients(group):
    if group.size() == 1:
        return lambda x: x

    @mx.custom_function
    def f(x):
        return x                                        # :21  FORWARD = IDENTITY

    @f.vjp
    def f(x, dx, _):
        return mx.distributed.all_sum(dx, group=group)  # :25  BACKWARD ONLY

    return f
```

Line 21 is the entire forward. The only `mx.distributed` call in the function is inside
the `@f.vjp` at :25, which MLX invokes only when computing a vector-Jacobian product.
Inference never calls it. **(a) CONFIRMED — zero collectives at decode.**

Call site: `mlx-lm/mlx_lm/models/deepseek_v4.py:2958-2959`
(`if self.sharding_group is not None: x = sum_gradients(self.sharding_group)(x)`) — inside
`DeepseekV4MoE.__call__`, matching the cited line.

### (b) Is it really what the "2 per layer" comment referred to? — **PLAUSIBLE, NOT PROVABLE**

I flag this as the weakest link in the agent's argument and I am not going to paper over it.

`start_cluster.sh:440-442` (blamed to `cb68628d89`, but the text originates at
`719283194`, 2026-05-17, per `git log -S "86-deep"`):

```
#   fence=43 (one fence per forward) on gamma>=2 builds up an 86-deep
#   chained-collective dependency in the GPU/comm-stream command buffer
#   (43 layers x 2 all_sums per layer).
```

I checked the tree **as of that commit** to see whether a second all_sum ever existed:

- `git show 719283194:src/exo/worker/engines/mlx/auto_parallel.py` — the
  `DeepseekV4ShardingStrategy` loop at that revision set **only** `layer.ffn.sharding_group`
  and sharded only the six ffn linears. It did **not** set `layer.attn.sharding_group`
  (that was added later, by `4e5bd29f0`, the OPT-3 seq-split commit).
- `mlx-lm` @ `a16a5f2` (the newest mlx-lm commit ≤ 2026-05-18) already had the same
  `sum_gradients`-on-input / `all_sum`-on-output structure.

So on 2026-05-17 the per-layer collective count was **also 1**, and the second one the
comment names did not exist then either. `sum_gradients` sitting textually adjacent to
`all_sum` in `DeepseekV4MoE.__call__` is by far the most likely origin of the miscount,
but **I cannot prove authorial intent from the repo.** It does not matter for the number:
whatever the author meant, there is only one collective per layer, then and now.

### (c) Full enumeration of collectives — I did this myself, independently

Complete inventory of `mx.distributed.*` in the live `deepseek_v4.py` (7559 lines):
lines **3076, 4320, 4625, 4634, 4644, 5083, 5092, 5102, 5708, 6941, 7080, 7089**.
(I derived this list by grep, not by reading I6's; it agrees.)

| # | line | site | fires at M=4 decode? | why |
|---|---|---|---|---|
| 1 | 3076 | `moe.all_sum` — `all_sum(y, group=self.sharding_group)` | ✅ **YES, ×43** | `layer.ffn.sharding_group` set for every layer at `auto_parallel.py:1110` |
| 2 | 4320 | `LocalAttention` tail all_sum | ❌ NO | guarded `sharding_group is not None and _ATTN_ALLSUM`. `LocalAttention.sharding_group` is **never set** (`auto_parallel.py:1122-1126` gates on class name ∈ {Sparse,Compressed}); AND `_ATTN_ALLSUM=False` |
| 3 | 4625 | `CompressedAttention` seq-split pad+all_sum | ❌ NO | requires `_seq`, which requires `L >= _SEQ_SPLIT_MIN_L=16` (`:231`, checked `:4462-4465`). L=4 |
| 4 | 4634 | `CompressedAttention` seq-split all_gather | ❌ NO | same `_seq` gate |
| 5 | 4644 | `CompressedAttention` tail all_sum | ❌ NO | `_ATTN_ALLSUM=False` |
| 6 | 5083 | `SparseCompressedAttention` seq-split all_sum | ❌ NO | `_seq` gate (`:4740-4743`) |
| 7 | 5092 | `SparseCompressedAttention` seq-split all_gather | ❌ NO | `_seq` gate |
| 8 | 5102 | `SparseCompressedAttention` tail all_sum | ❌ NO | `_ATTN_ALLSUM=False` |
| 9 | 5708 | `_rowsdpa_sharding_allsum` (vec verify path) | ❌ NO | `if _sg is not None and _ATTN_ALLSUM:` (`:5705`). **`_ATTN_ALLSUM=False`** ← *this one is on the live M=4 vec-verify path and would fire ×43 if the env var were 1.* |
| 10 | 6941 | `model.recv` | ❌ NO | `pipeline_size==1` under `MLX_JACCL_SHARDING_MODE=Tensor` |
| 11 | 7080 | `model.send` | ❌ NO | same |
| 12 | 7089 | `model.all_gather` | ❌ NO | same |

`_ATTN_ALLSUM = os.environ.get("EXO_DSV4_ATTN_ALLSUM", "1") == "1"` — `deepseek_v4.py:1695`.
Live value `0` on **both** nodes (Claim 2 evidence). Note the **default is 1**: if the env
var were ever dropped from the launch line, sites 2/5/8/9 arm and the count becomes 86.

**Site #9 (`:5708`) is one I6's table does not enumerate individually.** It is the
`ROWSDPA=3` vec-verify tail, and `EXO_DSV4_VERIFY_ROWSEQ_VEC=1` /
`..._ROWSDPA=3` are both live — so this is the code path M=4 verify actually takes, and it
is the *most* likely of the attention sites to fire. It is off for the same single reason
(`_ATTN_ALLSUM`). I6's conclusion survives; its enumeration is one line short.

Now the **exo TP sharding layer**, which the task asked me to grep separately and which I6
does not enumerate at all:

- `auto_parallel.py:1428` — `all_sum(combined_ss)` in `_fused_sharded_qk_norm`. Called only
  from `:1602`, inside `WrappedMiniMaxAttention.__call__`. **MiniMax only.** DSv4 never
  constructs it. ❌ not on the DSv4 path.
- `auto_parallel.py:1044-1049` — `ShardedMoE.moe.all_sum`. Wrapper class; DSv4 explicitly
  does **not** use it (docstring `:1067-1069`: "no ShardedMoE wrapper is needed"). ❌
- `auto_parallel.py:2175` — `WrappedGemma4Experts`. Gemma only. ❌
- `auto_parallel.py:143/187/315/361` — pipeline `send`/`recv`. PP only. ❌
- **The sharded-linear wrappers are the one place a hidden collective could have lurked, and
  they do not apply.** `mlx/nn/layers/distributed.py:333` (`ShardedToAllLinear.__call__`)
  and `:585` (`QuantizedShardedToAllLinear.__call__`) each contain a forward `all_sum`.
  If DSv4 used `shard_linear` for `switch_mlp.down_proj`, that would be a **second real
  per-layer collective** and Claim 1 would be dead. It does not: `DeepseekV4ShardingStrategy`
  uses `sharded_to_all_linear_in_place` (`auto_parallel.py:1131`), which is
  `partial(shard_inplace, ...)` (`:765-769`). `shard_inplace` only rewrites the parameter
  dict (`mlx/nn/layers/distributed.py:118-156`, terminal line `module.update(_shard(...))`)
  and its own docstring says *"The module doesn't change so in order for distributed
  communication to happen the module needs to natively support it"* — the module stays a
  plain `QuantizedSwitchLinear`, no collective. **This is the check that could have broken
  the claim, and it holds.** ✅

**TOTAL AT M=4 DECODE: 43.** Claim 1 VERIFIED. Payload: `y` is `(B,L,hidden) = (1,4,4096)`
bf16 = **32 KiB** per collective; I re-derived this from `_moe_post_combine`
(`deepseek_v4.py:1156`) collapsing the top-k axis, and confirm I6's figure.

Adjacent, not part of the 43: the MTP block's ffn is also sharded
(`auto_parallel.py:1179`) → +1 all_sum per **draft** forward. DSpark is unsharded
(`EXO_DSV4_DSPARK_TP_SHARD` absent → default `"0"`, `auto_parallel.py:1239`) → 0.

---

# CLAIM 2 — "attention is REPLICATED, not head-sharded, in production"

## VERDICT: **VERIFIED** (both halves, independently)

### Code path

`Model.shard` at `deepseek_v4.py:7531-7559` does contain real head-sharding —
`layer.attn.wq_b = shard_linear(...)` (`:7537`), `shard_inplace(layer.attn.wo_a, ...)`
(`:7543`), `layer.attn.attn_sink = mx.split(...)` (`:7544`), and
**`layer.attn.n_heads //= N` (`:7545`)**. The cited line 7545 is correct.

I then chased every caller of `.shard(` in both repos:

```
$ grep -rn "\.shard(" --include=*.py src/ mlx-lm/mlx_lm/ | grep -v shard_inplace|shard_linear|test
mlx-lm/mlx_lm/utils.py:849:        model.shard(tensor_group)          <- the ONLY call
src/exo/shared/types/worker/instances.py:88                (unrelated: instance.shard(runner_id))
mlx-lm/mlx_lm/models/{longcat_flash_ngram,kimi_k25}.py     (other model classes)
```

`utils.py:849` is inside mlx-lm's own `load()` convenience function. exo does **not** use
it — exo calls `load_model()` directly (`utils_mlx.py:221`, `:350`) and then routes to
`tensor_auto_parallel` → `DeepseekV4ShardingStrategy` (dispatched
`auto_parallel.py:787-794`). The `DeepseekV4ShardingStrategy.shard_model` loop
(`auto_parallel.py:1084-1155`) touches exactly six modules per layer — all `ffn.*`
(`:1127-1132`) — and never assigns any attention weight. `layer.attn.sharding_group` is set
only under `_DSV4_SEQ_SPLIT` and only for `SparseCompressedAttention`/`CompressedAttention`
(`:1122-1126`), which is an activation-partition for prefill, not a weight shard, and is
length-gated off at L=4.

**`deepseek_v4.py:7545` is dead code in this deployment. CONFIRMED.**

> Note for the campaign: **I3's shape table (I3-KERNEL-MICROBENCH.md line 38) cites
> `num_attention_heads` as "32/rank under TP=2, via `shard()`'s `n_heads //= N`".**
> That is the exact dead-code trap, and I3 fell into it. I3's own line 186-188 later
> half-catches it. This did not affect I3's headline MoE number (which used correct
> per-rank MoE shapes), but its `attn.wq_b` kernel-sum row is benched at a shape
> production does not use.

### Live env var — verified on the RUNNING processes, both nodes

```
$ ps eww <runner-pid>   # runner = the multiprocessing-fork child, not the -m exo parent
192.168.86.201  pid 53377 : EXO_DSV4_ATTN_ALLSUM=0
192.168.86.202  pid 57665 : EXO_DSV4_ATTN_ALLSUM=0
```

Also confirmed on both: `MLX_JACCL_SHARDING_MODE=Tensor`, `EXO_DSV4_SEQ_SPLIT=1`,
`EXO_SPECULATIVE_GAMMA=3`, `EXO_PREFILL_STEP_SIZE=2048`, `EXO_DSV4_MOE_PARTS_ROWSEQ=shared`,
`EXO_DSV4_LMHEAD_MXFP8=1`, `EXO_DSV4_VERIFY_ROWSEQ_VEC=1`. `EXO_DSV4_DSPARK_TP_SHARD`
absent on both. I read the **child runner** env, not just the launch line, so this is the
value the model code actually sees.

Consistency with Claim 1 holds: attention is replicated → no second per-layer collective.

---

# CLAIM 3 — routed experts are mxfp4 (bits=4, group_size=32)

## VERDICT: **VERIFIED**, with **two deployed quantizations I3 states incorrectly or omits**

`make_quantization_config` — `deepseek_v4.py:905-935` (I3 said "~907–935"; close enough):

```python
mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}          # :906
mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}          # :907
experts        = {k: mxfp4 for k,_ in flat_modules
                  if ".ffn.switch_mlp." in k and k.endswith("_proj")}   # :910-914
shared_experts = {k: mxfp8 ... if ".ffn.shared_experts." in k}          # :915
attn           = {k: mxfp8 ... if ".attn.w" in k or ".attn.indexer.wq" in k}  # :916-918
mtp_proj       = {k: mxfp8 ... e_proj/h_proj}                           # :923-927
return {"group_size": 64, "bits": 8, "mode": "affine", **experts, **shared_experts, **attn, **mtp_proj}
```

**Is this reached for the deployed checkpoint?** Yes, and I traced it rather than assuming.
`~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731/config.json` (read on .201) has **no
top-level `"quantization"` key**, only `"quantization_config": {"quant_method":"fp8",
"fmt":"e4m3","scale_fmt":"ue8m0","weight_block_size":[128,128]}`. So `utils.py:522`
(`if (quantization := config.get("quantization")) is not None`) is skipped and control
reaches the `elif quantization_config := ...` chain, landing at:

```python
utils.py:548-554
elif quant_method == "fp8" and config.get("model_type") == "deepseek_v4":
    quantization = make_quantization_config(model)
    config["quantization"] = quantization
    _quantize(quantization)
```

The per-key `_is_mxfp_override` scale-dtype guard (`utils.py:451-461`) does **not** filter
anything here — it runs inside `_quantize` via `config["quantization"].setdefault(...)`, and
`config["quantization"]` is already the full dict, so every setdefault is a no-op. **The
mxfp4 override is applied unconditionally on this checkpoint.** ✅

Cross-check against the actual on-disk tensors (safetensors header, node .201,
`model-00003-of-00048.safetensors`):

| tensor | on-disk dtype | on-disk scale |
|---|---|---|
| `layers.1.ffn.experts.0.w1.weight` | **I8** `[2048,2048]` | `F8_E8M0 [2048,128]` |
| `layers.1.ffn.shared_experts.w1.weight` | F8_E4M3 `[2048,4096]` | `F8_E8M0 [16,32]` |
| `layers.1.attn.wq_b.weight` | F8_E4M3 `[32768,1024]` | `F8_E8M0 [256,8]` |
| `layers.1.ffn.gate.weight` | **BF16** `[256,4096]` | *(none)* |
| `head.weight` | **BF16** `[129280,4096]` | *(none — `head.scale` absent)* |

The expert branch of `sanitize` (`deepseek_v4.py:7436-7442`) fires exactly on the I8 case
(`v.shape[-1]*16 == weight.shape[-1]` → `128*16 == 2048` ✓), renaming `.scale`→`.scales` and
`view(uint32)`ing the weight. Group size checks out arithmetically: uint32 `[2048,512]` →
`in_dim = 512*32/4 = 4096`; `4096 / 128 groups = 32`. **Consistent with mxfp4 g=32 b=4.**
The upstream `fp8/e4m3/[128,128]` block in `config.json` is indeed only the storage format,
never the runtime one. **I3's core finding is CORRECT.**

## Deployed bits — the answer to the question as asked

| weight class | mode | bits | group_size | evidence |
|---|---|---|---|---|
| **Routed experts** (`ffn.switch_mlp.{gate,up,down}_proj`) | **mxfp4** | **4** | **32** | `deepseek_v4.py:910-914` |
| **Shared experts** (`ffn.shared_experts.*`) | mxfp8 | 8 | 32 | `:915` |
| **Attention projections** (`attn.w*`, `attn.indexer.wq`) | mxfp8 | 8 | 32 | `:916-918` |
| **`lm_head`** | **mxfp8** | **8** | **32** | ⚠️ **not** the affine-8/g64 fallback — see below |
| MoE router (`ffn.gate`) | **unquantized bf16** | 16 | — | BF16 on disk, no `.scales`, not in the override list → `class_predicate` returns False (`utils.py:501-511`) |
| MTP `e_proj`/`h_proj` | mxfp8 | 8 | 32 | `:923-927` |

⚠️ **Correction to I3.** I3-KERNEL-MICROBENCH.md line 55 says *"Everything else (default
fallback): affine, group_size=64, bits=8"*. For `lm_head` that is **wrong in production**.
`head.weight` is BF16 with no `.scales`, so `class_predicate` (`utils.py:501-511`) declines
to quantize it at load. It is then quantized **in place** by
`utils.py:631-660` under `EXO_DSV4_LMHEAD_MXFP8`:

```python
if os.environ.get("EXO_DSV4_LMHEAD_MXFP8","0") == "1" and model_type == "deepseek_v4":
    qmod = mod.to_quantized(group_size=32, bits=8, mode="mxfp8")     # utils.py:645
```

`EXO_DSV4_LMHEAD_MXFP8=1` is **live on both nodes** (`ps eww`), and the runner log confirms
it executed:

```
2026-09-03 01:37:06.632 | Runner stderr: [LMHEAD_MXFP8] quantized lm_head to mxfp8 (group=32, bits=8)
```

So `lm_head` is **mxfp8 b8 g32, ~0.53 GB/rank replicated**, not the 1.059 GB BF16 the P05
comment describes as the pre-fix state, and not affine-g64. Byte accounting must use the
quantized figure. The MoE router being bf16 is also worth carrying — I3's *script* gets this
right (`i3_microbench.py:180-181`) but the *report* does not say it.

---

# CLAIM 4 — the I3 bytes-read formula and the 52.2% figure

## VERDICT: **REFUTED**

Two independent defects. The first is a straight quantization-mode error I reproduced on
hardware; the second is a formula/semantics question that the script's own choice of
indices happens to make *nearly* moot — the opposite of the double-count the task
hypothesised.

## (a1) 🚩 THE SCRIPT BENCHED THE WRONG QUANTIZATION MODE — and its bytes constant is wrong

`i3_microbench.py:106-108`:

```python
switch.gate_proj = switch.gate_proj.to_quantized(group_size=group_size, bits=bits, mode="affine")
switch.up_proj   = ... mode="affine"
switch.down_proj = ... mode="affine"
```

with `EXPERT_BITS=4, EXPERT_GROUP_SIZE=32` (`:50-51`). The report (line 86) calls this
*"the exact call `make_quantization_config` implies"*. **It is not.**
`make_quantization_config` specifies `mode="mxfp4"` (`deepseek_v4.py:906`). `affine` and
`mxfp4` are different formats **and different Metal kernels** (`affine_gather_qmv_fast` vs
`fp_gather_qmv_fast`, selected by the `mode` string at
`repos/exo/mlx/mlx/backend/metal/quantized.cpp:1087`).

The byte consequence is large, and it is in the script's constant. `i3_microbench.py:131`:

```python
scale_bias_bytes = n_gathered_rows * n_groups_per_expert_per_proj * n_projs * 2 * 2
```

— i.e. *"affine mode stores fp16 scale and fp16 bias per group"* (comment, `:130`).
**Both halves of that are wrong.** I measured the actual arrays on the live venv:

```
affine b4 g32 : weight uint32 (256,1024,512)  scales float32 (256,1024,128)  biases float32 (256,1024,128)
mxfp4 b4 g32  : weight uint32 (256,1024,512)  scales uint8   (256,1024,128)  biases = None
```

- affine scales/biases are **fp32 (4 B), not fp16 (2 B)** → the script undercounts affine
  scale+bias by exactly 2×.
- **mxfp4 has no biases at all** and its scales are 1-byte E8M0.

Per-expert-per-projection bytes:

| | weight | scales | biases | **total** |
|---|---|---|---|---|
| script's assumption | 2,097,152 | 262,144 | 262,144 | **2,621,440** |
| **affine b4 g32 (what it ran)** | 2,097,152 | 524,288 | 524,288 | **3,145,728** |
| **mxfp4 b4 g32 (what production runs)** | 2,097,152 | 131,072 | 0 | **2,228,224** |

At M=4, 24 pairs, 3 projections: script used **188,743,680 B**; the affine run really moved
**226,492,416 B** (+20.0%); production mxfp4 would be **160,432,128 B** (−15.0%).

**Corrected achieved bandwidth for the run I3 actually performed** (its own timings,
0.6625 ms mean / 0.6271 ms min):

| | I3 reported | **corrected (affine, as-run)** |
|---|---|---|
| M=4 mean | 284.9 GB/s = **52.2%** | **341.9 GB/s = 62.6%** |
| M=4 best | 301.0 GB/s = 55.1% | **361.2 GB/s = 66.2%** |
| M=1 mean | 145.5 GB/s = 26.7% | **174.7 GB/s = 32.0%** |
| M=1 best | 157.2 GB/s = 28.8% | **188.7 GB/s = 34.6%** |

I independently re-ran the same shapes on this M4 Max under both modes
(`SwitchGLU(4096, 1024, 256)`, 200 iters, 20 warmup, identical timing discipline):

```
affine b4 g32  M=4, 24 pairs/23 distinct : 0.8464 ms  -> 267.6 GB/s   (48.9% of 546)
mxfp4  b4 g32  M=4, 24 pairs/23 distinct : 0.6501 ms  -> 246.8 GB/s   (45.2% of 546)
mxfp4  b4 g32  M=1,  6 pairs/ 6 distinct : 0.3558 ms  -> 112.7 GB/s   (20.6%)
```

My absolute numbers are lower than the Studio's (36 GB MacBook vs 128 GB Studio; my
measured streaming-read proxy was only ~69 GB/s on a 512 MB bf16 reduction, so this
machine's memory subsystem is materially weaker and my absolutes are **not** transferable).
The *ratios* are what matter, and they confirm the direction: mxfp4 is meaningfully faster
per call and moves ~29% fewer bytes than affine at the same bits.

### 🚩 THE BAND THAT FIRED IS WRONG

I3's pre-registered bands: `≥80%` → kernels fine · `<60%` → **kernel work FUNDED** ·
`60–80%` → **decision waits on I1**.

I3 reported 52.2% and declared *"BAND FIRED: <60% → MLX KERNEL WORK IS FUNDED … 52.2% is
well inside the <60% band, not a near-miss."* The corrected figure for the run it performed
is **62.6% mean / 66.2% best**, which lands in the **60–80% band → DECISION WAITS ON I1**.

**This is a decision-changing error.** It is not a rounding quibble: the whole point of the
band was to fund kernel work without waiting on I1, and the corrected number reverses that.
I3's own "sanity check passed" reasoning (report line 136-139) also fails — it argues M=4
takes ~2× M=1's time "despite reading ~4× the bytes", presented as evidence against a timing
bug; but the bytes ratio is exactly 4.00× under its own formula either way, so that check
never had discriminating power.

**Caveat I want stated plainly:** the corrected 62.6% describes the *affine* kernel, which
production does not run. The honest statement is *"the deployed mxfp4 path was never
benched."* A clean rerun at `mode="mxfp4"` with mxfp4 byte accounting is required before
any band is declared. My local mxfp4-vs-affine ratio suggests mxfp4 will land in a similar
or slightly lower percentage, i.e. plausibly straddling the 60% line — which is exactly why
guessing is not acceptable here.

## (a2) The dedup / double-count question — **NO INCONSISTENCY BETWEEN I3 AND I6**

The task hypothesised that I3's 24-pair denominator might double-count against I6's measured
2.37× cache-reuse multiplier, inflating the reported GB/s by ~2.37×. **It does not, and the
reason is what indices the script passed.**

`i3_microbench.py:111`:

```python
indices = mx.random.randint(0, n_routed_experts, shape=(M, top_k))   # M=4, top_k=6, over 256 experts
```

24 draws uniform over 256 experts. Expected distinct = `256·(1−(255/256)^24) ≈ 22.9`.
I reproduced the script's exact draw on the live venv: **23 distinct out of 24 pairs.**
There is essentially **nothing to dedup** — the synthetic case is the *disjoint* corner.
So for the case the script measured, `n_gathered_rows = M*top_k = 24` is the right count,
and the per-pair and per-distinct denominators differ by only 24/23 = 4.3%. **The bytes
formula is structurally correct for what the script ran.** No double-count. The two reports
do not contradict each other; I3's 24-pair assumption and I6's "no dedup" finding are the
same claim.

I verified the no-dedup mechanism myself at the dispatch level rather than trusting I6.
`GatherQMM::eval_gpu` (`repos/exo/mlx/mlx/backend/metal/quantized.cpp:1840`) with
M=1, K=4096, N=1024, B=24, E=256:

- the run-length `gather_qmv_rhs` path (`:1889`) requires **`right_sorted_ == true`** AND
  `B >= 16` AND `B/E >= 2`. `sorted_indices=False` is passed from `switch_layers.py:191`
  because `do_sort = indices.size >= 64` is false at 24 (`switch_layers.py:182`); and
  `B/E = 24/256 = 0`. **Doubly excluded.**
- the steel-tile `gather_qmm` (`:1962`) needs `M >= vector_limit`. I read
  `get_qmv_batch_limit` (`quantized.cpp:86-127`): for a non-`d` arch with `D,O <= 4096` it
  returns **12** (or 10 on arch_gen 13/14). M=1. **Excluded.**
- → falls through to `gather_qmv` (`:1984`), grid `(M, ceil(N/bn), B)` — **B, the pair count,
  is `tid.z`**, one independent threadgroup per (row,expert) pair.

So the kernel really does issue 24 independent tile streams. ✅

**But the crux question the task asked — "did the hardware actually move 24 pairs' worth of
DRAM bytes?" — is UNDETERMINED, and it matters differently than the task supposed.**

- For **I3's synthetic case (23 distinct)**, per-pair and per-distinct bytes coincide, so
  the achieved-GB/s figure is well-posed regardless of cache behaviour. The ~52%→62.6%
  correction above stands on its own and is not touched by this.
- For **production**, the real distinct-expert count per M=4 verify is unmeasured. If real
  routing shows meaningful row overlap, then production reads *fewer* distinct bytes than
  24 pairs implies, and a per-pair byte model would overstate the ideal DRAM traffic.
- My own reuse discriminator on this machine (mxfp4, same 24 pairs, 6 distinct vs 23
  distinct): 0.5193 ms vs 0.6501 ms — only **1.25×**, where full dedup predicts 1.00× and
  zero dedup predicts 3.83×. So there **is** partial reuse, but it is much closer to the
  no-dedup end. I6's 2.37× "same-6-experts, 6→24 pairs" ratio is the same phenomenon
  measured on a different axis and is not in conflict.

**What would settle it:** the `EXO_DSV4_ROUTE_HIST=1` + `EXO_DSV4_ROUTE_HIST_DECODE_ONLY=1`
hook (already present, `deepseek_v4.py:2944-2955`) to get the real distinct-expert
distribution per verify, cross-referenced with a Metal counter capture of DRAM bytes on the
`moe.switch_mlp` span. Until then, quote achieved bandwidth **with the index distribution
stated**, never bare.

## (b) `mx.eval` discipline — **CORRECT**

`i3_microbench.py:75-81`:

```python
for _ in range(n_iters):
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)        # inside the timed region
    sync()              # mx.synchronize(), also inside
    t1 = time.perf_counter()
```

`eval` **and** `synchronize` both land before `t1`, with a 20-iteration warmup that also
evals (`:70-73`). This is right, and it is the thing that most often invalidates MLX
benchmarks. No lazy-graph artifact. ✅ The discipline being correct is precisely why the
timings are reusable for my corrected-bytes recomputation above.

## (c) 0.6625 ms × 43 = 28.5 ms vs the 56 ms bracket — arithmetic right, framing misleading

0.6625 × 43 = **28.49 ms**, = **50.9%** of the 56 ms verify bracket. Arithmetic ✅.

But note what that 28.5 ms represents: it is 43 **separately-dispatched, separately-eval'd,
separately-synchronized** `SwitchGLU` calls. Production builds one lazy graph per verify and
evals it once, so 28.5 ms includes 43 command-buffer round trips production does not pay.
I3's own report is admirably candid about this for the kernel-sum check (report lines
162-183, where forced per-op eval produced a nonsensical 123.71 ms > 56 ms bracket) — but
the same caveat applies, at lower magnitude, to the ×43 MoE figure, and the report does not
carry it there.

For scale: I6 measured 43 **chained** `switch_mlp` calls in ONE graph at 11.58–18.62 ms
(I6 report line 217-220) — i.e. **~1.5–2.5× less** than 28.5 ms for nominally the same work.
That gap is the dispatch overhead the per-call method bakes in. Using the fused-graph
number, the routed-expert path is more like **21–33%** of the 56 ms bracket, not 51%.
Both are "plausible shares"; the 51% figure should not be quoted without the caveat.

---

# CLAIM 5 — the I4 band application

## VERDICT: **REFUTED.** The correct verbatim verdict is **BLOCKER STANDS**, not AMBIGUOUS

### What the band literally requires

> *"Partial hits CONFIRMED at ≥3 chunks (diverging variants show **cached_tokens** ≥ ~6144
> AND the early-divergence control shows ~0) → round-4 blocker INVALID, Fix B RE-OPENED.
> **Diverging variants ~0 even at ≥3 chunks → blocker STANDS.** Anything ambiguous →
> AMBIGUOUS."*

The band names one metric: **`cached_tokens`**. Measured (`audit_results.json`):

| req | prompt_tokens | **cached_tokens** | runner-log `shared_prefix` |
|---|---|---|---|
| A baseline | 7524 | 0 | 0 |
| B exact repeat | 7524 | **7522** | *(no insert event)* |
| C/D/E diverge @6491 | 8292 | **0** | 6491 (78%) |
| F control diverge @1503 | 12704 | **0** | 1503 (11%) |

**Diverging variants show `cached_tokens = 0` at ≥3 chunks.** That is verbatim the
BLOCKER-STANDS clause. The agent's own table (I4 report line 101-104) records it, then the
VERDICT section (line 128-133) **substitutes a different metric** — the runner-log
`shared_prefix` — declares the primary clause met on that substitute, and downgrades to
AMBIGUOUS on the control. Swapping the metric is not "applying the band verbatim," and the
report says it is applying it verbatim (line 128-130).

### 🚩 AND THE SUBSTITUTE METRIC DOES NOT MEAN WHAT I4 SAYS IT MEANS

This is the load-bearing error, and it is not a technicality. I4 concludes (lines 121-124):

> *"the HTTP API's `cached_tokens` field under-reports partial hits … but the actual
> prefix-cache engine (the runner-log-visible trie) did serve the expected large partial
> hit"*

**That is false. `shared_prefix` is not a cache hit. No partial hit was served.**

`shared_prefix` is emitted by **`add_kv_cache`** — the *insertion* path — at
`cache.py:864-868`, from:

```python
cache.py:823-825
# Measure how much of this new session already exists in the trie so we
# can see cross-session dedup in the logs.
_, shared_depth = self._longest_prefix_match(prompt_tokens, [])
```

It is a **descriptive statistic about trie overlap at insert time**, computed on a session
being stored. It says nothing about whether that session's prefill reused anything. Its own
code comment says so.

`cached_tokens`, by contrast, **is** the hit: `cached_tokens = state.prefix_hit_length`
(`batch_generate.py:4619-4621`), and `prefix_hit_length = local_hit_length =
len(all_prompt_tokens) - len(remaining_tokens)` returned by `get_kv_cache`
(`batch_generate.py:2388-2393`). It is the direct, load-bearing number.

Three independent lines of evidence confirm C/D/E/F were **real misses**:

1. **`add_kv_cache` ran, not `update_kv_cache`.** The save path
   (`batch_generate.py:5284-5305`) calls `update_kv_cache` when
   `matched_index is not None and prefix_hit_length >= min_prefix_hit_length and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE`,
   else `add_kv_cache`. The live log shows `add_kv_cache` for C, D, E **and** F — so the
   hit was rejected or nonexistent for every one of them.

2. **The code names this exact failure for this exact model.** `get_kv_cache` computes
   `has_non_sliceable` from the donor's layer mask; `_sliceable_layer_mask` treats every
   `CacheList` as non-sliceable (`cache.py:496`, `is_non_trimmable_cache_entry`), and DSv4
   layers are `CacheList`s. But `snapshot_ssm_states` stores `None` for **trimmable**
   CacheLists (`cache.py:356-358`: `elif isinstance(c, CacheList) and not
   c.is_trimmable(): ... else: states.append(None)`). So `has_non_sliceable=True` while
   every snapshot state is `None` → `_materialize_cache_to_depth(strict_snapshot=True)`
   returns `None` → **`get_kv_cache` STRICT-MISS** (`cache.py:1320-1331`) → full prefill,
   `hit_length = 0`. The docstring states it outright at **`cache.py:1725-1726`**:

   > *"DSv4 hits this on EVERY partial hit: `snapshot_ssm_states` stores None for trimmable
   > CacheLists, which is what DSv4 layers use."*

   The exact-match path (B) survives because `is_exact and not has_non_sliceable` is not the
   only route — B restored via a leaf whose full state is materializable, which is why B and
   only B shows a nonzero `cached_tokens`.

3. **Latency proves it end-to-end.** From `audit_results.json`:
   - A (7524 tok, cold): 20.17 s → 2.68 ms/tok
   - **C (8292 tok, "78% hit"): 26.40 s → 3.18 ms/tok**
   - B (7524 tok, exact hit): **1.41 s**

   A genuine 78% reuse would leave ~1801 tokens to prefill, ≈5–6 s. C took **26.4 s and was
   slower per token than the cold baseline.** There is no reading of this in which a partial
   hit was served.

### Was AMBIGUOUS an under- or over-call?

**Over-call.** The agent under-called the negative result: it had a clean BLOCKER-STANDS
signal on the band's own metric and softened it to AMBIGUOUS by substituting a metric that
measures something else. The control clause is then moot — you never reach it, because the
primary clause already resolved to the STANDS branch. (The agent's reading of the control as
"failed toward *more* caching" is likewise built on `shared_prefix`; F's `cached_tokens` was
0, i.e. the control behaved exactly as pre-registered.)

To be fair to the agent: it **did** flag the discrepancy loudly (report lines 106-124) and
did not bury it. The failure is one of interpretation — treating an insert-time trie
statistic as a hit metric — not of candour.

### On the task's hypothesis: is `cached_tokens=0` an observability gap? — **NO**

The task asked whether the round-4 audit might have been "measuring an observability gap
rather than a cache miss." I checked the wiring specifically because that would have been
exculpatory. **It is not a gap.** `cached_tokens` is wired directly to `prefix_hit_length`
(`batch_generate.py:4619-4621`; identical in the non-batched path,
`generate.py:2506-2507`), which is the same variable that decides whether prefill is skipped.
There is no second, richer number being hidden from HTTP. `cached_tokens=0` **is** the cache
miss, faithfully reported. Round-4's use of it was methodologically sound.

### 🔎 MY OWN ANALYSIS (explicitly NOT the band): is the round-4 blocker invalid anyway?

Separately from the verdict, I agree with I4's *narrow* structural point and disagree with
what it implies.

**The narrow point is correct.** With `EXO_PREFILL_STEP_SIZE=2048` (verified live on both
nodes), a 381-token prompt never reaches a chunk boundary, so no snapshot is ever taken
(`generate.py:859`, snapshot appended once per chunk iteration) and the partial-hit path is
structurally unreachable. A 381-token test cannot distinguish "partial hits broken" from
"partial hits untested." As a *test design*, round-4's was inadequate.

**But that does not re-open Fix B, because I4's own experiment supplies the properly-sized
test and it fails too.** At 8292 tokens with divergence at ~6491 — four chunks in, exactly
the regime round-4 could not reach — `cached_tokens` is still 0, latency shows no reuse, and
the code explains why in a docstring written for this model. So:

- round-4's *test* was invalid ✅ (I4 is right)
- round-4's *conclusion* — that partial prefix-cache hits do not work on DSv4 — is
  **independently CONFIRMED by I4's own better-designed experiment** ❌ (I4 is wrong to
  treat this as a re-open)

The real finding hiding in I4's data is more useful than the one it reported: **DSv4 partial
prefix-cache hits are broken by a specific, identified, single-line-diagnosable cause** —
the `is_non_trimmable_cache_entry` / `snapshot_ssm_states` disagreement about trimmable
`CacheList`s (`cache.py:496` vs `cache.py:356`). One treats them as non-sliceable (so a
snapshot is *required*), the other declines to snapshot them (so none is ever *available*).
That is a concrete fix target, not an ambiguity. I did not attempt the fix — read-only.

---

# WHAT I CHECKED AND COULD NOT SETTLE

- **Claim 1(b), authorial intent of the "2 all_sums per layer" comment.** UNVERIFIABLE from
  the repo. I checked the tree at the comment's originating commit (`719283194`, 2026-05-17)
  and confirmed only one per-layer collective existed then too. `sum_gradients` adjacency is
  the likely cause; the count is 43 either way.
- **Production's real distinct-expert count per M=4 verify.** Not measured by anyone. Needed
  before any per-pair byte model is used for production ideal-time. Hook exists
  (`deepseek_v4.py:2944-2955`).
- **The deployed mxfp4 kernel's achieved bandwidth at production shapes on Studio hardware.**
  Never benched — I3 benched `affine`. My mxfp4 numbers are from a 36 GB MacBook whose
  streaming-read proxy measured only ~69 GB/s, so my absolutes do not transfer to the
  Studios. **This is the single highest-value rerun.**
- **Whether the mxfp4 rerun lands above or below the 60% band line.** I deliberately did not
  guess.

## Provenance of everything I cite

- `mlx-lm` live: `repos/exo/mlx-lm` @ `37260bbd6` — `deepseek_v4.py` md5
  `0c09ff466f0454493fc8c74d546d077d`, identical to the Studio's installed copy. Cites valid
  against running code.
- `mlx` live: `repos/exo/mlx` @ `e40a416b2` — `nn/layers/distributed.py` md5
  `4132533cc67594a3a79c09926377b7e2`, identical to the Studio's installed copy.
- `exo` live: `repos/exo` @ `808fea7d3` on the Studio.
- ⚠️ `~/repos/mlx` (@ `ac73d0c9e`) is **not** the live build. This MacBook's venv installs
  from it — do not bench mlx behaviour locally without accounting for that.
- Env values read from the **runner child processes** (pid 53377 / 57665) via `ps eww`, not
  from `start_cluster.sh` defaults.
