# I6 + COLLECTIVE COUNT — code read, round 1

Read-only archaeology. Every claim below carries a `path:line` cite. Measurements
labelled MEASURED were taken on **this machine** (also an Apple M4 Max,
`applegpu_g16s`) against the **byte-identical** kernel source the Studios run —
see "WHICH MLX_LM COPY IS LIVE". No cluster state was touched.

---

## WHICH MLX_LM COPY IS LIVE

**LOUD WARNING #1 — the mlx-lm venv copy is a real directory, not a symlink, but
it is byte-identical to the fork source, so reading the fork is safe.**

- The runner process on 192.168.86.201 is `~/repos/exo/.venv/bin/python -m exo -v`
  (PID 53377, `ps aux`), so the live interpreter is `~/repos/exo/.venv`.
- `~/repos/exo/.venv/bin/python -c 'import mlx_lm; print(mlx_lm.__file__)'` on the
  Studio →
  `/Users/adam.durham/repos/exo/.venv/lib/python3.13/site-packages/mlx_lm/__init__.py`.
  It is a **real directory**, NOT an egg-link/.pth to the fork. (`site-packages`
  contains only `exo.pth`, `_editable_impl_exo_bench.pth`,
  `_editable_impl_exo_tools.pth`, `_virtualenv.pth`, `distutils-precedence.pth`
  — none point at mlx-lm.)
- BUT: on the Studio, `md5 -q` of
  `.venv/.../site-packages/mlx_lm/models/deepseek_v4.py` and
  `~/repos/exo/mlx-lm/mlx_lm/models/deepseek_v4.py` both = `0c09ff466f0454493fc8c74d546d077d`,
  and the local `~/repos/exo/mlx-lm/mlx_lm/models/deepseek_v4.py` on this machine
  has the **same** md5. **The fork source I read IS what executes.** Line numbers
  in this document are valid against the running code.
- `~/repos/exo/mlx-lm/build/lib/mlx_lm/` (stale build) is NOT on the import path
  and was not read.

**LOUD WARNING #2 — THE REAL VENV TRAP IS IN `mlx`, NOT `mlx_lm`. The task brief
pointed at `/Users/adam.durham/repos/mlx`. That is the WRONG mlx. It is not what
runs.**

- Installed mlx version on the Studio: `0.32.1.dev20260822+e40a416b2`
  (`mlx-0.32.1.dev20260822+e40a416b2.dist-info`).
- `direct_url.json` in that dist-info: `{"url":"file:///Users/adam.durham/repos/exo/mlx","dir_info":{}}`
  → **the live mlx was built from `~/repos/exo/mlx`, a SECOND mlx checkout.**
- `~/repos/exo/mlx` HEAD = `e40a416b2 fix(jaccl): move JACCL_TRACE_TIMING to RingGroup`
  — matches the installed version string exactly. Working tree clean.
- `~/repos/mlx` HEAD = `ac73d0c9e` (local) / `1fe020ed3` (Studio), and
  `git cat-file -t e40a416b2` in `~/repos/mlx` → **`fatal: Not a valid object name`**.
  The commit that built the live mlx *does not exist* in `~/repos/mlx`.
- Source files differ: `mlx/backend/metal/quantized.cpp` md5 is
  `17cccac008e4f4764da5ec787196fb86` in `~/repos/mlx` vs
  `fb3b76561a61168b6e6b6e59082c64a3` in `~/repos/exo/mlx` (the live one).
  `~/repos/mlx` also has an uncommitted `M mlx/backend/metal/matmul.cpp` on the Studio.
- `~/repos/exo/mlx` on **this machine** is byte-identical to `~/repos/exo/mlx` on
  the Studio (`quantized.cpp` = `fb3b7656…`, `kernels/quantized.h` = `3fc5a1f2…`
  on both).

**All mlx cites below are `repos/exo/mlx/...`.** I re-did the kernel trace against
that tree after catching this. For the specific functions at issue
(`GatherQMM::eval_gpu`, `gather_qmv`) the two trees happen to be textually
identical (verified by `diff`), so the dispatch conclusion would not have changed
— but the byte-level divergence elsewhere means `~/repos/mlx` must not be cited
as the running code.

Live env confirmed off the running process (`ps eww 53377`):
`EXO_DSV4_MOE_PARTS_ROWSEQ=shared`, `EXO_DSV4_ATTN_ALLSUM=0`,
`EXO_DSV4_SEQ_SPLIT=1`, `EXO_DSV4_VERIFY_ROWSEQ=1`,
`EXO_DSV4_ROWSEQ_FULLBLOCK=1`, `EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=0`,
`EXO_DSV4_MTP=1`, `EXO_DSV4_DSPARK=1`, `EXO_SPECULATIVE_GAMMA=3`,
`MLX_JACCL_SHARDING_MODE=Tensor`, `EXO_DSV4_DSPARK_TP_SHARD` **absent** (=0).

---

## I6 VERDICT

**PER-(row,expert) PAIR AT DISPATCH — the kernel performs NO dedup. But DRAM
bytes land between the two extremes because of GPU cache reuse.**

Verdict in one line: **not once-per-verify, and not a clean 4x either — the
routed-expert bytes multiplier at M=4 vs M=1 is 1.4x–2.4x depending on routing
overlap, MEASURED 2.37x in the realistic (low-overlap) case.**

| quantity | value |
|---|---|
| dispatch semantics | one weight-tile load per **(row, expert) pair**; no grouping, no sorting, no dedup |
| distinct experts, M=4, k=6 | 6 (total overlap) … 24 (disjoint); real routing is near the disjoint end |
| **bytes multiplier vs M=1** | **2.37x** MEASURED at 6 distinct / 24 pairs; **3.81x** at 24 distinct / 24 pairs |
| naive "perfectly shared" ideal | 1.0x (if 4 rows hit the same 6 experts) |
| naive "full duplication" ceiling | 4.0x |

**This does NOT account for the whole 1.75–2.16x gap, but it is a large piece of
it.** Critically, the *ideal* M=4 verify is not 1x — the union of 4 rows'
top-6 sets is genuinely more experts than one row needs, so some of the 2.37x is
irreducible work, not waste. The recoverable slice is the difference between
"stream each distinct expert once" and "stream each pair once", MEASURED below.

---

## I6 EVIDENCE

### (a) The MoE forward and the exact MLX op

Routed experts: `DeepseekV4MoE.switch_mlp` is a `SwitchGLU`
(`mlx-lm/mlx_lm/models/deepseek_v4.py:2916-2921`), constructed with
`n_routed_experts` (`:2919`). Config for the deployed checkpoint
(`~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731/config.json`, read on the
Studio): `hidden_size=4096`, `num_hidden_layers=43`, `n_routed_experts=256`,
`num_experts_per_tok=6`, `moe_intermediate_size=2048`, `n_shared_experts=1`.

Call site at M=4 verify — because `EXO_DSV4_MOE_PARTS_ROWSEQ=shared` does NOT
contain `switch`, the routed path takes the **batched** else-branch:

- `mlx-lm/mlx_lm/models/deepseek_v4.py:3026` — `if "switch" in _prs:` → false
- `mlx-lm/mlx_lm/models/deepseek_v4.py:3039` — `y = finalize(self.switch_mlp(x, inds))`
  ← ONE call with all 4 rows

`SwitchGLU.__call__` (`mlx-lm/mlx_lm/models/switch_layers.py:177-203`):
- `:178` `x = mx.expand_dims(x, (-2, -3))` → x becomes `(1, 4, 1, 1, 4096)`
- `:182` `do_sort = indices.size >= 64` → at M=4, `indices.size = 1*4*6 = 24`
  → **`do_sort = False`. THE SORT DOES NOT RUN AT VERIFY.** (It also does not run
  at M=1, size 6.) `_gather_sort` (`:13-18`) is skipped entirely.
- `:191,193,197` three `QuantizedSwitchLinear` calls with `sorted_indices=False`.

`QuantizedSwitchLinear.__call__` (`mlx-lm/mlx_lm/models/switch_layers.py:76-91`)
→ `mx.gather_qmm(x, weight, scales, biases, rhs_indices=indices, transpose=True,
group_size, bits, mode, sorted_indices=False)` (`:77-88`).

**MEASURED shapes** (reproduced with the live venv python + live mlx, TP=2
per-rank shard `IS = 2048/2 = 1024`, mxfp4 gs=32 per
`make_quantization_config` `deepseek_v4.py:911-914` which assigns
`mxfp4 {group_size:32, bits:4}` to every `.ffn.switch_mlp.*_proj`):

```
L=1: GatherQMM  M=1 K=4096 N=1024 B=6   E=256   out_shape=[1,1,6,1,1024]
L=4: GatherQMM  M=1 K=4096 N=1024 B=24  E=256   out_shape=[1,4,6,1,1024]
```

Note `M=1` in **both** cases. The 4 verify rows do not become a matmul M
dimension — `expand_dims` at `switch_layers.py:178` pushes them into the **batch**
dimension B. **B is exactly the (row, expert) pair count: 4 rows × 6 experts = 24.**

### (b) Kernel indexing — how (row, expert) maps to a weight-tile load

`GatherQMM::eval_gpu` — `repos/exo/mlx/mlx/backend/metal/quantized.cpp:1840`.
Shape math at `:1856-1860`: `K=x.shape(-1)`, `M=x.shape(-2)`, `N=out.shape(-1)`,
`B=out.size()/M/N`, `E=w.size()/w.shape(-1)/w.shape(-2)`.

Dispatch branches, in order, evaluated for our M=1 K=4096 N=1024 B=24 E=256:

1. `:1889` `gather_qmv_rhs` (the run-length "stream expert weights once per
   same-expert run" path — the ONLY dedup-capable branch) requires
   `right_sorted_ == true`. **`right_sorted_` is false** because
   `switch_layers.py:191` passes `sorted_indices=False` (do_sort is false at
   L≤10). `ops.cpp:5285` `gather_qmm` forwards it into the primitive at
   `sorted_indices && !lhs_indices_` / `sorted_indices && !rhs_indices_`
   (`repos/exo/mlx/mlx/ops.cpp`, GatherQMM ctor args). **BRANCH NOT TAKEN.**
   It additionally requires `B/E >= 2`; here `B/E = 24/256 = 0`. Doubly excluded.
2. `:1914` `gather_qmm_rhs` — requires `right_sorted_ == true` and `B/E >= 4`.
   **NOT TAKEN** (same two reasons).
3. `:1940` `gather_qmm_rhs_lhs` (OPT-9 sorted prefill) — requires
   `right_sorted_ == true && M >= 16`. **NOT TAKEN** (M=1).
4. `:1962` `gather_qmm` steel tile — requires `M >= vector_limit`.
   `vector_limit = get_qmv_batch_limit(K=4096, N=1024, d)` (`:1861`). On
   `applegpu_g16s` (arch_size `s`, not `d`) with `D<=4096 && O<=4096` → returns
   **12** (`quantized.cpp:86-128`, the `default:` arm). M=1 < 12. **NOT TAKEN.**
5. `:1983` `transpose_` is true (`switch_layers.py:83`) → **`gather_qmv` IS THE
   KERNEL THAT RUNS.**

`gather_qmv` — `repos/exo/mlx/mlx/backend/metal/quantized.cpp:962`. Grid setup:

```
:980   MTL::Size grid_dims(M, (N + bn - 1) / bn, B);      // bn = 8
```

**`B` — the (row,expert) pair count, 24 — is the `tid.z` grid dimension.** Each
`tid.z` slice is an independent threadgroup batch.

Per-threadgroup addressing, `adjust_matrix_offsets`
(`repos/exo/mlx/mlx/backend/metal/kernels/quantized.h:1648`, the
lhs/rhs-indices overload):

```
x_idx = lhs_indices[tid.z * lhs_strides[0]];
w_idx = rhs_indices[tid.z * rhs_strides[0]];
...
w += w_idx * w_strides[0];  scales += w_idx * s_strides[0];
```

Then `fp_qmv_fast_impl` / `affine_gather_qmv_fast`
(`repos/exo/mlx/mlx/backend/metal/kernels/fp_quantized.h:1469`, impl at `:325`)
walks the full `in_vec_size` with
`for (int k = 0; k < in_vec_size; k += block_size) { ... qdot(wl, x_thread, s); }`
(`fp_quantized.h:361-372`).

**Answer to (b): each (row, expert) pair independently re-derives its own base
weight pointer from `rhs_indices[tid.z]` and independently streams the full
expert tile. There is NO grouping, NO sorting, NO run-length encoding, NO dedup
on this path. Two threadgroups whose `rhs_indices` happen to name the same expert
issue two fully independent, uncoordinated reads of the same weight bytes.**

### (c) BYTES-READ verdict — MEASURED, not inferred

The kernel relies **entirely on the GPU cache**, not on explicit dedup. I say
that as a code fact (see (b)). The follow-on question — does a tile realistically
stay resident across rows — I did **not** have to infer, because this machine is
the same GPU (`applegpu_g16s`, M4 Max) running byte-identical kernel source. So I
measured it.

Per-distinct-expert bytes, TP=2 per-rank, mxfp4 gs=32, 3 projections:
`3 × (4096×1024×0.5 + 4096×1024/32) = 6.68 MB`.
Full 256-expert table per layer per rank = **1.71 GB** — orders of magnitude
beyond any Apple GPU cache, so a "fits in L2" story is a non-starter at the table
level; the only question is intra-dispatch reuse across the 24 concurrent pairs.

**MEASURED — 43 chained `switch_mlp` calls in ONE graph** (matching production's
single-graph forward, so per-call command-buffer overhead is not double-counted),
fresh random expert sets per rep so nothing is pre-warmed:

| case | total ms | per-layer ms | distinct GB | BW if once-per-distinct | BW if per-pair |
|---|---|---|---|---|---|
| M=1 draft (6 pairs, 6 distinct) | 4.89 | 0.114 | 1.72 | 352.9 GB/s | 352.9 GB/s |
| M=4 best (24 pairs, **6** distinct) | 11.58 | 0.269 | 1.72 | 148.9 GB/s | **595.5 GB/s** |
| M=4 mid (24 pairs, 12 distinct) | 14.00 | 0.326 | 3.45 | 246.4 GB/s | 492.9 GB/s |
| M=4 worst (24 pairs, 24 distinct) | 18.62 | 0.433 | 6.90 | 370.5 GB/s | 370.5 GB/s |

Discriminators:

- **A) same 6 experts, 6 pairs → 24 pairs: 2.37x.** If the cache deduped
  perfectly this would be **1.00x** (identical 1.72 GB of distinct weights either
  way). If DRAM streamed every pair this would be **4.00x**. It is 2.37x —
  **partial cache reuse, closer to the duplication end.**
- **B) 24 pairs, 6 distinct → 24 distinct: 1.61x.** If bytes tracked distinct
  experts this would be 4.00x; if bytes tracked pairs, 1.00x. 1.61x again lands
  in between.
- **C) M=4 worst vs M=1: 3.81x.**

Sanity check on the numbers: the "M=4 best, 6 distinct" row implies **595 GB/s**
if you insist every pair hit DRAM — that exceeds the ~546 GB/s M4 Max ceiling, so
some reuse provably occurred. Equally, "once-per-distinct" would imply only
148.9 GB/s, far under what the same kernel achieves at 24 distinct (370.5 GB/s),
so full dedup provably did NOT occur. **Both extremes are ruled out by the
measurement itself.** The truth is partial reuse.

**BYTES VERDICT: at M=4 with the deployed config, routed-expert weight bytes are
streamed from DRAM ~2.4x per verify relative to the M=1 case when rows share
experts, and ~3.8x when they don't. The realistic 43-layer verify cost of the
routed path measured 11.6–18.6 ms per rank** — against a "read the union of
activated experts once" ideal of roughly 1.72–6.90 GB / 546 GB/s = 3.1–12.6 ms.
So there is real headroom here, on the order of **~4–8 ms per verify**, but it is
NOT the full 56 ms and NOT the entire 1.75–2.16x gap.

Attribution caveat, stated plainly: my harness measures the MoE routed path in
isolation on an idle GPU with no TP collectives, no attention, and no memory
pressure from a 95 GB resident model. On the live cluster, cache reuse will be
**worse** (attention/KV traffic evicts expert tiles between dispatches), so 2.37x
is a **lower bound** on the live multiplier. I did not measure the live number.

**The single highest-leverage fix implied by this trace:** `do_sort` is gated at
`indices.size >= 64` (`switch_layers.py:182`) and verify has only 24 indices, so
the **sorted / run-length `gather_qmv_rhs` path — which exists precisely to
"stream expert weights once per same-expert run" (`quantized.cpp:1878-1888`) —
is never reachable at decode or verify.** Its own gate `B/E >= 2`
(`quantized.cpp:1894`) also fails at B=24, E=256, so simply lowering the sort
threshold is necessary but not sufficient.

### (d) SHARED experts vs ROUTED experts — the answer DIFFERS

`shared_experts` is a dense `DeepseekV4MLP` (`deepseek_v4.py:2922-2925`), three
plain `nn.Linear` (`:2880-2882`), quantized **mxfp8** not mxfp4
(`deepseek_v4.py:915`). It has no gather and no routing — every row uses the same
weights, so there is no (row, expert) pair concept at all.

But `EXO_DSV4_MOE_PARTS_ROWSEQ=shared` **is** the deployed default, and it
**does** send shared experts down a per-row loop:

- `deepseek_v4.py:3047` `if "shared" in _prs:` → **TRUE** in production
- `deepseek_v4.py:3048-3054` — `mx.concatenate([self.shared_experts(x[:, _j:_j+1]) for _j in range(_prs_L)], axis=1)` — **4 separate M=1 calls**
- vs `deepseek_v4.py:3056` `shared_out = self.shared_experts(x)` — the single batched call

Gate: `_prs` is cleared unless `x.shape[0] == 1 and 2 <= _prs_L <= 8`
(`deepseek_v4.py:2972-2977`), which M=4 c=1 verify satisfies.

**MEASURED** (43 layers, mxfp8 gs=32, per-rank shared intermediate 1024,
12.98 MB/layer/rank, 0.56 GB total):

| | total ms | eff BW |
|---|---|---|
| shared BATCHED (1 call, M=4) | 3.98 | 140.1 GB/s |
| shared PER-ROW (4 calls, M=1) | 5.09 | 109.7 GB/s |
| **ratio** | **1.28x** | **+1.10 ms per verify** |

So: shared experts are **not** re-read 4x from DRAM (they are only 0.56 GB and
partly cache-resident), but the per-row split does cost a **measured +1.1 ms per
verify** in dispatch overhead and lost batching — a small, cheap, *bounded* win
if `MOE_PARTS_ROWSEQ` can be dropped. Note the comment at
`deepseek_v4.py:2966-2969` explicitly acknowledges this tradeoff ("gate/combine/
shared are cheap per-row; switch (the expert gather) is the expensive one").
That comment is accurate — my measurement confirms it.

Also worth noting for the campaign: `EXO_DSV4_ROWSEQ_FULLBLOCK=1` is live, and
`EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=0`, so the MoE ffn is called **once per layer**
with all 4 rows concatenated (`deepseek_v4.py:5308-5310`), not per row. Attention
inside FULLBLOCK, however, IS a per-row loop (`deepseek_v4.py:5255-5272`).

---

## COLLECTIVE COUNT VERDICT

**The `start_cluster.sh` comment claiming "43 layers × 2 all_sums per layer = 86
per forward" is WRONG. The correct count is 43 — exactly half.**

Per verify forward at decode, TP=2:

| site | count | cite |
|---|---|---|
| MoE `all_sum` (1 per decoder layer) | **43** | `deepseek_v4.py:3076` |
| post-attention `all_sum` | **0** | disabled, see below |
| attention seq-split `all_sum`/`all_gather` | **0** | length-gated off at decode |
| `sum_gradients` on MoE input | **0** | forward is identity, see below |
| embedding / lm_head / final norm | **0** | no collective in TP mode |
| pipeline `send`/`recv`/`all_gather` | **0** | `pipeline_size == 1` in TP mode |
| **TOTAL, per-layer loop + outside it** | **43** | |

Outside the per-layer loop, **within the model forward: zero.** The only
collectives outside the layer loop live in the *speculative driver*, not the
forward, and fire per **cycle**, not per forward (see COLLECTIVE EVIDENCE §5).

### (e) Payload bytes per collective at M=4 decode

The MoE `all_sum` reduces `y`, shape `(B, L, hidden)` — confirmed by the NOP-probe
contract at `deepseek_v4.py:469` ("`moe` -> `DeepseekV4MoE.__call__` returns zeros
(shape `(B, L, hidden)`)") and by `_moe_post_combine` returning
`(y*scores).sum(-2) + shared_out` (`deepseek_v4.py:1156`), which collapses the
top-k axis.

- shape = `(1, 4, 4096)`, dtype **bf16** (2 B)
- **payload = 1 × 4 × 4096 × 2 = 32,768 bytes = 32 KiB per collective**
- **per forward = 43 × 32 KiB = 1.375 MiB**

dtype note: `EXO_COMPUTE_DTYPE=bf16` is set live, and even if an fp32 activation
reached the collective, `_collective_fp32_safe`
(`deepseek_v4.py:520-531`, installed at `:534-542`) downcasts fp32→bf16 before
the transfer and upcasts after — so **the wire payload is bf16 unconditionally**.
32 KiB is the number to use for I1.

At 43 × 32 KiB = 1.375 MiB per forward, this is a **latency-bound, not
bandwidth-bound** collective load — 43 round trips of 32 KiB each. For I1, the
per-collective *fixed* cost (RDMA round trip + fence) is what matters, not the
byte volume.

---

## IS ATTENTION REPLICATED

**YES — attention is REPLICATED on both ranks, not head-sharded. Confirmed from
code, three independent ways. The "further head-sharding beyond FFN" campaign
item is correctly scoped: it is genuinely not done yet.**

1. **The sharding strategy says so and explains why.**
   `src/exo/worker/engines/mlx/auto_parallel.py:1054-1070`, the
   `DeepseekV4ShardingStrategy` docstring: *"Sharding for DeepSeek V4 Flash / Pro —
   MoE-only. Replicates attention on every rank; shards only the MoE block."*
   With the reason at `:1058-1062`: sharding `wq_b` across heads breaks
   `_grouped_output_projection`'s manual reshape of `wo_a.weight/.scales/.biases`
   and crashes inside `mx.quantized_matmul`.

2. **The shard loop only touches ffn.** `auto_parallel.py:1127-1132` shards
   exactly six things per layer — `ffn.shared_experts.{gate,down,up}_proj` and
   `ffn.switch_mlp.{gate,down,up}_proj`. **No attention weight is sharded.**
   Contrast the (unused) in-model `Model.shard` at
   `deepseek_v4.py:7531-7559`, which DOES shard `attn.wq_b` (`:7537`), shard
   `attn.wo_a` (`:7543`), split `attn_sink` (`:7544`) and do
   `layer.attn.n_heads //= N` (`:7545`) — **that method is dead code in this
   deployment.** Production goes through
   `utils_mlx.py:501` → `tensor_auto_parallel` → `DeepseekV4ShardingStrategy`
   (dispatched at `auto_parallel.py:788`), never `Model.shard`. This is the trap
   that would make a casual grep conclude "head-sharded": **`deepseek_v4.py:7545`
   `n_heads //= N` is real code that never runs.**

3. **`attn.sharding_group` is set only for the seq-split prefill path, and that
   path is length-gated off at decode.** `auto_parallel.py:1122-1126` sets
   `layer.attn.sharding_group` only when `_DSV4_SEQ_SPLIT` and the class is
   `SparseCompressedAttention` or `CompressedAttention`. `LocalAttention` never
   gets one (`auto_parallel.py:1121`).

**And the post-attention all_sum is doubly dead:**

- The tails at `deepseek_v4.py:4316`, `:4640`, `:5102` are all guarded by
  `self.sharding_group is not None and _ATTN_ALLSUM`.
  `_ATTN_ALLSUM = os.environ.get("EXO_DSV4_ATTN_ALLSUM", "1") == "1"`
  (`deepseek_v4.py:1695`). **Live env has `EXO_DSV4_ATTN_ALLSUM=0`** (read off
  PID 53377). → **every attention all_sum is OFF.**
- Even with it on, the comment at `deepseek_v4.py:1685-1694` states the situation
  outright: *"DSv4 REPLICATES attention on every rank (MoE-only sharding), yet the
  seq-split strategy sets sharding_group on the compressed/sparse classes, so the
  legacy tail all_sum SUMS TWO (near-)identical replicas"* — i.e. that all_sum was
  never a real reduction, it was a bug that doubled the attention branch.
  `EXO_DSV4_ATTN_ALLSUM=0` is the fix, and it is deployed.
- The seq-split `all_sum`/`all_gather` (`deepseek_v4.py:4625`, `:4634`, `:5083`,
  `:5092`) requires `L >= _SEQ_SPLIT_MIN_L` where `_SEQ_SPLIT_MIN_L = 16`
  (`deepseek_v4.py:231`), checked at `:4464-4465` and `:4742-4743`. **At M=4
  verify, L=4 < 16 → seq-split is off.** It is a prefill-only path.

**So the true count is 43, not 86 — and the missing 43 are the attention all_sums,
which are disabled by env AND would be semantically wrong if enabled AND are
gated off by sequence length at decode anyway.** Three independent reasons, any
one of which is sufficient.

---

## COLLECTIVE EVIDENCE

### 1. The MoE all_sum — the only per-layer collective

`mlx-lm/mlx_lm/models/deepseek_v4.py:3074-3076`:

```python
if self.sharding_group is not None:
    with span("moe.all_sum"):
        y = mx.distributed.all_sum(y, group=self.sharding_group)
```

`sharding_group` is set for every layer at `auto_parallel.py:1110`
(`layer.ffn.sharding_group = self.group`). 43 layers → **43 all_sums**.

It is a *real* reduction: `all_to_sharded` slices axis `max(ndim-2, 0)`
(`auto_parallel.py:744-748`) and `SwitchLinear.weight` is
`(num_experts, output_dims, input_dims)` (`switch_layers.py:100-104`) → axis 1,
the **intermediate width**. Both ranks hold **all 256 experts at half width**;
experts are never partitioned by identity. This is stated explicitly, with a
correction notice, at `auto_parallel.py:1164-1175` — and my measured shapes
confirm it (`E=256` on the per-rank shard, `N=1024 = 2048/2`).

Consequence worth flagging for the campaign: **each rank reads the weights of
every activated expert, at half width.** There is no expert-to-rank affinity to
exploit — "co-locate an expert on a node" has nothing to bind to
(`auto_parallel.py:1173-1175`).

### 2. `sum_gradients` is NOT a forward collective

`deepseek_v4.py:2958-2959` calls `sum_gradients(self.sharding_group)(x)` on the
MoE input. Definition,
`.venv/lib/python3.13/site-packages/mlx/nn/layers/distributed.py:15-27`:

```python
@mx.custom_function
def f(x):
    return x            # forward = identity, NO collective
@f.vjp
def f(x, dx, _):
    return mx.distributed.all_sum(dx, group=group)   # backward only
```

**Forward is identity. At inference this fires zero collectives.** Anyone counting
`sum_gradients` as "the second all_sum per layer" would land on 86 — I believe
this is precisely the error in the `start_cluster.sh` comment.

### 3. Per-layer count is 1 even under the rowseq/FULLBLOCK paths

With `EXO_DSV4_ROWSEQ_FULLBLOCK=1` and `EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=0`, the
per-row loop covers attention/norms only (`deepseek_v4.py:5255-5272`) and the ffn
is invoked **once** on the concatenated rows (`deepseek_v4.py:5307-5310`):

```python
else:
    _fb_ffn = self.ffn(
        mx.concatenate([r[0] for r in _fb_rows], axis=1), input_ids
    )
```

So the MoE all_sum count is **1 per layer, not L per layer**. Had
`FULLBLOCK_MOE=1` been set (`deepseek_v4.py:5296-5306`), it would be 4 per layer
= 172 per forward — **worth flagging as a live footgun for the campaign, since
that env var exists and is currently 0.**

Likewise the `"switch"`/`"combine"` members of `MOE_PARTS_ROWSEQ` would multiply
work but not collectives (the all_sum sits outside those branches at `:3074`).

### 4. No collectives outside the layer loop in the model forward

- Embedding: `deepseek_v4.py:6890-6896` — reshape/contiguous only.
- Mask build: `deepseek_v4.py:6908-6929` — local.
- `model.recv` (`deepseek_v4.py:6941`) is gated on
  `pipeline_rank < pipeline_size - 1`; `model.send` (`:7080`) on
  `pipeline_rank != 0`; `model.all_gather` (`:7089`) on `pipeline_size > 1`.
  `PipelineMixin.__init__` sets `pipeline_rank = 0, pipeline_size = 1`
  (`mlx-lm/mlx_lm/models/pipeline.py:9-10`) and `.pipeline(group)` — the only
  thing that changes them (`pipeline.py:18-22`) — is called **only** from
  `pipeline_auto_parallel` (`utils_mlx.py:504`), which is the
  `PipelineShardMetadata` arm of the match at `utils_mlx.py:499-504`. Production
  is `MLX_JACCL_SHARDING_MODE=Tensor` → `TensorShardMetadata` arm
  (`utils_mlx.py:499-501`). **All three pipeline collectives are dead in TP mode.**
- lm_head / final norm: no `mx.distributed` call. The complete inventory of
  `mx.distributed.*` in `deepseek_v4.py` is lines
  3076, 4320, 4625, 4634, 4644, 5083, 5092, 5102, 5708, 6941, 7080, 7089 — and
  every one is accounted for above.

### 5. Coordination collectives — per CYCLE, not per forward

These are outside the model forward entirely, in the speculative driver, and are
routed onto a **separate TCP-backed coord group** (`get_coord_group`,
`utils_mlx.py:1869-1899`) precisely so they don't collide with the model TP
call-id space (race documented at `dsv4_mtp.py:2236-2243`).

- `dsv4_mtp.py:2254` — `all_sum(presence_arr, group=coord_group)`, uid
  intersection, `int32[1024]` = **4096 B**, once per `_next()` call.
- `dsv4_mtp.py:2293` — `all_sum` of `gen_batch._num_tokens`.
- `broadcast_from_canonical` (`mtp_module.py:680-708`, an `all_gather` + slice at
  `:708`) — called at `dsv4_mtp.py:2793, 3015, 3609, 3612, 3653, 3681, 3943,
  4616, 5173, 5602` and `mtp_module.py:865, 868, 898`. These scale with **γ ×
  cycles** (noted at `dsv4_mtp.py:3507-3510`), i.e. the draft phase, on tiny
  int32 token arrays.

**These are NOT part of the 43.** For I1, the honest framing is: **43 model-TP
all_sums of 32 KiB per verify forward, plus a γ-scaled tail of small int32 coord
collectives on a separate transport per speculative cycle.** I did not attempt an
exact coord-collective count per cycle — it is control-flow-dependent (batch
size, acceptance, tree-verify on/off) and would need a live trace, not a read.

### 6. Draft-phase MoE all_sums (adjacent, flagged for completeness)

The MTP block's ffn is also a sharded `DeepseekV4MoE`
(`auto_parallel.py:1179-1185`), so each MTP forward adds **1** MoE all_sum.
`num_nextn_predict_layers=1` in the deployed config → 1 MTP block. DSpark stages
(`dspark_target_layer_ids=[40,41,42]`) are **not** sharded — `EXO_DSV4_DSPARK_TP_SHARD`
is absent from the live env (default `"0"`, `auto_parallel.py:1239`), so
`model.model.dspark` is left alone and its `ffn.sharding_group` stays `None` →
**0 collectives from DSpark.** These are draft-phase, not verify-forward, so they
do not change the 43.

---

## UNDETERMINED FROM CODE

1. **The live cluster's actual routed-expert cache-reuse multiplier.** I measured
   2.37x on an idle same-model GPU with byte-identical kernels. On the live node
   — 95 GB resident, concurrent attention/KV traffic, TP collectives interleaving
   — reuse will be worse and the multiplier higher. Bounded below by 2.37x,
   above by 4.0x. **What would settle it:** an `EXO_DSV4_ROUTE_HIST=1` +
   `EXO_DSV4_ROUTE_HIST_DECODE_ONLY=1` capture (the hook already exists at
   `deepseek_v4.py:2944-2955`) to get the real distinct-expert-per-verify
   distribution, cross-referenced against a Metal counter capture of actual DRAM
   bytes on the `moe.switch_mlp` span.

2. **Exact coord-collective count per speculative cycle.** Control-flow dependent
   (batch size, acceptance length, tree-verify path). Would need `JACCL_TRACE_HASH=1`
   or a live counter, not a code read. The 43 model-TP collectives per verify
   forward ARE determined; the coord tail is not.

3. **How much of the residual 56 ms verify is collective latency vs kernel
   inefficiency.** This read supplies the input (43 × 32 KiB) but cannot divide
   the remainder — that is I1's job.

## What I did NOT verify

- I did not run anything on the cluster; both Studios were touched only with
  `ps`, `ls`, `cat`, `md5`, `git log/status`, and read-only `python -c` imports.
- I did not read `~/repos/exo/mlx-lm/build/lib/` (confirmed not on the import path).
- Benchmarks in this document ran on **this** machine (M4 Max, same GPU arch,
  byte-identical kernel source), NOT on the Studios.
