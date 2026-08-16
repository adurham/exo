# Section 110: Closing the last live candidate — is `MLX_JACCL_ACK_RETRANSMIT_US` reachable from PP decode?

Scope: pure static code reading of `src/`, `mlx/`. No cluster runs (read-only ssh log greps only,
none needed — the answer resolved from code structure alone). Builds on Section 100 (`MLX_JACCL_ACK_RETRANSMIT_US`
rank-1 candidate, reachability left open) and Sections 85/86/89/96/100 refuted-hypotheses list.

## TL;DR verdict

**`MLX_JACCL_ACK_RETRANSMIT_US` is NOT reachable from the PP batched-decode per-token path.
CONFIRMED by code structure, not inference.** The last live candidate is dead. The 550ms decode
stall's mechanism is still unexplained after six refuted/killed hypotheses (five from before this
report, plus this one).

**The mechanism, whatever it is, is PP-specific in its blocking form** — it lives in
`send()`/`recv()`'s p2p-retry machinery, which TP never calls. But the *ACK-retransmit code path*
itself (the thing this section was sent to check) is real infrastructure that TP's decode
collectives DO route through — as a rare, loss-triggered fallback, not a per-token cost. That is a
separate, secondary risk worth flagging for requirement 3, not the same bug as this stall.

---

## 1. The call chain from rank0's last-layer `mx.eval(output)` — CONFIRMED

`BatchedMetaFramedPipelineLastLayer.__call__` (`pp_batched_decode_layers.py:296-341`) does:

```python
output: mx.array = self.original_layer(x, *args, **kwargs)   # build: graph only
mx.eval(output)                                                # eval: 550-686ms measured
```

`self.original_layer` is DeepSeek-V4's per-layer `forward` in `mlx-lm/mlx_lm/models/deepseek_v4.py`.
Traced every `mx.distributed.*` call site in that file (lines 2837, 3870, 4100-4119, 4545-4564,
5143, 6476, 6485) and their gating:

- **MoE `all_sum`** (`deepseek_v4.py:2837`) and **attention-tail `all_sum`** (`:3870`, `:4119`,
  `:4564`, `:5143`) are ALL gated `if self.sharding_group is not None:`. `sharding_group` is set
  only by `TensorParallelShardingStrategy` (Section 89, re-confirmed here: not re-litigated,
  code still reads the same way). Under `DSV4_SHARDING=Pipeline` these blocks never execute —
  **already-settled, carried forward.**
- **`mx.distributed.send`** (`:6476`, model-level, not the per-layer wrapper) and
  **`mx.distributed.all_gather`** (`:6485`) at the very end of the model's `forward` are gated
  `if not _nop_pipeline and pipeline_rank != 0` / `pipeline_size > 1` — these ARE reachable under
  PP, but they are **outside** the per-layer body; per-layer, the last decoder layer itself makes
  no `mx.distributed.*` call. The wrapper class's own explicit `mx.distributed.send` (line 368-ish
  in `pp_batched_decode_layers.py`, `last_layer_send`) already has its own separate timer and
  reads 0.1-0.3ms (Section 100, re-confirmed, not redone).

**Conclusion: the ONLY cross-rank primitive nested inside `self.original_layer(x, ...)`'s graph
for one decode token under Pipeline sharding is nothing at all** — no `all_sum`/`all_gather`/
`send`/`recv` fires inside the per-layer forward when `sharding_group is None`. The 550ms is spent
inside `mx.eval(output)` evaluating a purely-local compute graph (attention + MoE math on this
rank's shard of layers) — NOT waiting on any collective.

This reframes the investigation: **the eval is not blocked on a nested distributed call at all.**
Whatever it's waiting on must be inside plain MLX compute/scheduling, not jaccl.

---

## 2. Is `jaccl_ack_retransmit_us` reachable via `send()`/`recv()` (the ops PP DOES call)? — CONFIRMED NO

Read `send()` (mesh_impl.h:1967-2420) and `recv()` (mesh_impl.h:2422-2530+) in full. Neither
function calls `ack_sync_pre`, `ack_sync_post`, or `drain_acks` anywhere in their bodies. Their
only quiet/retransmit timer is `jaccl_p2p_drain_quiet_for(num_chunks)` (mesh_impl.h:2108, 2543),
which reads `jaccl_p2p_drain_quiet_us()` — a **different knob** (`MLX_JACCL_P2P_DRAIN_QUIET_US`,
default 0 = adaptive: 100ms floor + 1ms/chunk, capped 500ms), split out specifically so it would
NOT reuse `jaccl_ack_retransmit_us`. `jaccl_ack_retransmit_us()` itself (mesh_impl.h:174) is only
read at two sites: `reliable_all_reduce_v2` (line 725) and `reliable_all_reduce` (line 1251,
comment "reuse knob"). Both are collective all-reduce paths, never called from `send()`/`recv()`.

**Structural confirmation, not just call-graph absence:** `mesh.cpp:15-58` (the HARDWARE QP BUDGET
comment, dated 2026-08-10) states explicitly and unconditionally:

> PP (Pipeline): `connections_` + `p2p_retry_connections_` + `ack_connections_` — send()/recv()'s
> retry protocol needs `p2p_retry_connections_` on every p2p handoff. `ack_connections_` is needed
> for **the ONE warmup-time collective** (pp_batched_decode_glue.py's layer-count all_sum).
> ... TP (Tensor, and the DEFAULT): `connections_` + `ack_connections_` + `pool_connections_` —
> both are jaccl-v2's reliable-ARQ machinery, used throughout TP's collectives. **TP never calls
> the raw send()/recv() p2p path**, so `p2p_retry_connections_` is NOT built.

This is the mesh's own connection-topology invariant, enforced at construction (`mesh.cpp:182,
204`: `pool_connections_` is skipped entirely in PP mode; `p2p_retry_connections_` is skipped
entirely in TP mode). It is not merely "this code path doesn't happen to call it" — **the ACK/pool
QP infrastructure that `ack_sync_pre`/`drain_acks`/`jaccl_ack_retransmit_us` depend on is only
fully wired for the collective (`all_reduce`/`all_gather`) call path.** `send()`/`recv()` use a
disjoint QP (`p2p_retry_connections_`) and a disjoint retry mechanism (`p2p_retry_exchange`,
mesh_impl.h:3300) with its own quiet timer (`jaccl_p2p_drain_quiet_for`).

The ONE `ack_connections_` usage reachable under PP — `pp_batched_decode_glue.py`'s scatter+`all_sum`
handshake (lines 217-288) — is explicitly documented (line 217) as "call-once-at-warmup discipline...
NEVER per-request." It cannot produce a per-token, every-single-decode-step cost.

**Verdict: `MLX_JACCL_ACK_RETRANSMIT_US` cannot fire on the PP batched-decode per-token path,
under any prompt length, chunked or plain.** It has no code path into `send()`/`recv()`, and its
one PP call site fires once at process warmup, not once per token. The `first_layer_recv=0.1-0.3ms`,
`last_layer_send=0.1-0.3ms`, `gather_send=0.1ms` measurements from Section 100 are consistent with
this: those ARE the `send()`/`recv()` calls, and they're fast because they never touch the ACK
machinery. Rank 1 from Section 100 is now **CLOSED, refuted.**

---

## 3. The step function (chunked vs plain prefill) — NOT explained by this candidate; two candidate carriers identified but UNPROVEN

Since the ACK-retransmit timer is unreachable from decode at all, it cannot explain why chunked
prefill (>=2048 tokens) vs plain prefill (<2048) produces a step-function difference in per-token
decode cost. The carrier must be something else that differs structurally between the two prefill
paths and then persists into decode. Two candidates surfaced during this reading; **neither is
confirmed** — flagging as INFERRED / for future instrumentation only:

1. **`ForwardStepInfo.queue_sends`** (`pp_metaframe.py:145-303`). This ambient per-call flag
   changes whether the underlying `mx.distributed.send` fires immediately or is queued/deferred
   (`pp_metaframe.py:214-216` docstring: "queue_sends=False: send immediately; queue_sends=True,
   ..."). `pp_metaframe.py:165-166` notes the existing chunked-prefill call path
   (`pipeline_parallel_prefill`) "sets `is_prefill=True` but NEVER sets `queue_sends=True`" —
   i.e. there is already a known, documented inconsistency in how prefill phases set this
   context-local state, and `BatchedMetaFramedPipelineLastLayer` (the batched-decode class in
   scope for the 550ms) is a *different* code path (`pp_batched_decode_layers.py`) that does NOT
   read `ForwardStepInfo`/`queue_sends` at all — it hard-codes its own logic. This makes
   `queue_sends` an unlikely direct carrier into the batched-decode last-layer path specifically,
   but the surrounding ambient-state machinery (`ContextVar`-based, same general class of bug
   the module docstrings call out repeatedly as historically fragile) has not been fully ruled
   out as leaving OTHER stale context (e.g. `defer_header`, `phase`) that some non-batched
   fallback branch inside `original_layer` or the cache consults.
2. **Cache object identity/shape left by chunked prefill.** Not traced in this pass — the chunked
   prefill path (`pp_prefill_session.py`, `pp_scheduler_protocol.py::_handle_new_chunked_prefill_request`/
   `_handle_prefill_chunk_advanced`) drives the model forward in ~2048-token segments via a
   resumable generator (`interruptible=True`, the `yield ("layer", _ap_i, h)` point at
   `deepseek_v4.py:6465` noted in code as "the ONLY yield point... reached after EVERY layer when
   interruptible=True"). Whether this segmented/resumed execution leaves the KV cache object (or
   MLX's lazy-graph dependency chain via `cache_item.keys = mx.depends(cache_item.keys, h)` at
   `deepseek_v4.py:6480`) in a different structural state than a single eager plain-prefill call —
   e.g. a longer unevaluated dependency chain, more numerous cache segments, or an extra
   `mx.depends` linkage that forces the *next* decode step's `mx.eval` to walk/resolve a larger
   graph — was NOT traced end-to-end in this pass. This is the most promising unexplored carrier
   given Section 100's own finding that `last_layer_eval` is "a single lump covering whatever
   `mx.eval(output)` actually waits on" with no narrower Python timer, combined with this
   section's finding that eval waits on **pure local compute**, not a distributed call: a bloated
   or fragmented cache/dependency-graph structure from chunked prefill is now the leading
   candidate for making that local eval slow, but this is INFERRED, not measured.

**What would resolve this:** instrument cache object state (chunk count, `mx.depends` chain
length/depth, buffer layout) immediately after chunked vs. plain prefill completes, before the
first post-prefill decode step's `mx.eval(output)` — a static read cannot settle this; it needs a
before/after property comparison on the actual cache object at prefill-boundary.

---

## 4. Does this mechanism exist on the TP decode path? — MIXED, IMPORTANT FOR REQUIREMENT 3

Two separable questions:

**(a) Would the SAME 550ms flat-per-token blocking mechanism hit TP decode?**
**Cannot say yes or an unqualified no — INFERRED, leaning no as currently understood, but the
underlying compute-bound-eval mechanism (§1-§3, not yet identified) has not been shown to be
PP-specific in its root cause.** What §1 established is structural: for PP, the per-layer forward
makes zero collective calls (sharding_group is None), so the 550ms is pure local compute/eval
cost on rank0's shard of layers. TP decode's per-layer forward is different code
(`sharding_group is not None`, MoE `all_sum` and attention `all_sum` DO fire per layer, per
Section 89's own citation). So TP decode's per-token cost profile is not directly comparable to
PP's — TP adds real collective calls PP doesn't have, on top of whatever local-eval cost this
investigation hasn't yet identified. If the still-unknown root cause of the 550ms is something
generic to MLX's lazy-eval/Metal scheduling on this hardware (a real possibility now that §1 has
ruled out a distributed-call explanation), it could affect TP's local eval time too — but this
is speculation, not evidence. **Needs a direct TP decode timing run to answer, not further static
reading.**

**(b) Is `MLX_JACCL_ACK_RETRANSMIT_US`/the ACK-retransmit code path reachable on TP decode?**
**YES — CONFIRMED, but only as a loss-triggered fallback, not a per-token cost.** Per §2's
mesh.cpp citation, TP mode builds `ack_connections_` + `pool_connections_` and "TP never calls the
raw send()/recv() p2p path" — all TP cross-rank traffic goes through
`all_reduce`/`all_gather`/`reliable_all_reduce_v2`. The mesh_impl.h comment at line 1461-1463
states `ack_sync_pre`/`ack_sync_post` are "on the hot path of every reliable_all_reduce_v2
collective in TP mode." `jaccl_ack_retransmit_us()` is read inside `reliable_all_reduce_v2`
(line 725, `quiet_us`) and `reliable_all_reduce` (line 1251, `drain_quiet_us`) — both are on TP's
collective call path (MoE `all_sum`, attention `all_sum`, `all_gather` all route through
`all_reduce`/`all_gather` in mesh_impl.h, confirmed at §1's citation of `jaccl.cpp:105-152`
dispatching `all_sum`/`all_gather`/`send`/`recv` each to their own `group_->` method).

So on the topology actually being shipped (TP for both phases), **every decode-step MoE `all_sum`
and attention `all_sum` genuinely executes inside the code that owns the 500ms retransmit
constant** — but per the comments throughout mesh_impl.h ("it only fires on genuine loss," "the
600 -> 40 retransmit rounds" framing, "quiet_us... so it only fires on genuine loss"), this is a
**stall-detection/retransmit fallback that requires zero forward progress to trigger**, not a
per-call cost paid on every healthy collective. It should NOT reproduce a flat always-on 550ms/
token unless TP decode is independently dropping/losing frames on every single token, which would
be a much more visible and different symptom (intermittent multi-second spikes matching lost-frame
retry-count arithmetic, not a rock-steady 550-686ms band).

**Bottom line for requirement 3:** the *transport code path* that carries `jaccl_ack_retransmit_us`
is real infrastructure inside TP's collective hot path and deserves its own health-check
instrumentation before the TP ship (a live packet-loss/retransmit event on this timer would show
up as periodic multi-hundred-ms-to-second spikes, worth grepping runner logs for
`"serving retransmit"` / `"exceeded max retransmit rounds"` during any TP decode soak test) — but
it is NOT the same mechanism as the still-unexplained flat 550ms PP stall, and there is no
evidence it fires under healthy conditions on either topology.

---

## 5. What to instrument next (proposed, not executed)

Given §1's finding that the last-layer eval touches **no distributed call** under PP, the
investigation should pivot away from jaccl/transport entirely and toward local MLX compute/graph
structure:

1. **Cache/dependency-graph shape probe** (the leading unproven lead from §3): dump cache object
   metadata (segment/chunk count, any `mx.depends` chain length reachable from the cache's
   graph node, buffer dtype/shape) immediately at the prefill→decode boundary for both a plain
   (<2048 tok) and chunked (>=2048 tok) run, and diff them. This is the most direct test of the
   step-function carrier hypothesis and requires no new timers, just a one-off cache inspection
   script at that boundary — read-only, no cluster relaunch needed if it can hook an existing
   checkpoint/debug dump path.
2. **Split `last_layer_eval` further**: since §1 proves the per-layer forward is pure local
   compute under PP, the next narrower question is which sub-operation inside
   `self.original_layer` dominates the eval — attention vs MoE vs norm — via `span()`-gated
   sub-timers already present in `deepseek_v4.py` (`with span("moe.all_sum")` etc. show the
   pattern exists; equivalent spans around non-distributed sub-blocks would isolate the compute).
3. **TP decode timing run** (§4a): the only way to answer whether TP shares the unknown-root-cause
   compute-bound eval cost is to actually run TP decode at depth and compare wall-clock per-token
   cost against PP's plain/chunked step function. Static reading cannot resolve this.

Do NOT re-promote `MLX_JACCL_ACK_RETRANSMIT_US`, `MLX_JACCL_P2P_DRAIN_QUIET_US`, or any other
jaccl retransmit/quiet timer as an explanation for the PP stall going forward — §1 and §2 close
that avenue on code-structure grounds, independent of any further live A/B testing.

---

## Files read (for traceability)
- `bench/section100_timeout_constant_hunt.md` (full, prior findings carried forward as cited)
- `src/exo/worker/engines/mlx/pp_batched_decode_layers.py` (full class bodies, lines ~200-450)
- `src/exo/worker/engines/mlx/pp_metaframe.py` (`ForwardStepInfo`/`queue_sends` mechanism, lines
  ~7-310, ~890-1033)
- `src/exo/worker/engines/mlx/pp_batched_decode_glue.py` (warmup all_sum handshake, lines 217-288)
- `src/exo/worker/engines/mlx/pp_prefill_session.py`, `pp_scheduler_protocol.py`,
  `pp_scheduler_wire.py` (chunked-prefill call sites, targeted grep + context, not full read)
- `mlx-lm/mlx_lm/models/deepseek_v4.py` (all `mx.distributed.*` call sites and their gating,
  lines 2662-2900, 3790-3900, 4090-4570, 5100-5150, 6370-6490)
- `mlx/mlx/distributed/jaccl/lib/jaccl/mesh_impl.h` (targeted: 26-330, 1100-1130, 1380-1650,
  1857-2530, 3300-3520 — `send`/`recv`/`all_gather`/`all_reduce`/`drain_acks`/`ack_sync_pre`/
  `ack_sync_post`/`p2p_retry_exchange` bodies and the quiet/retransmit constant definitions)
- `mlx/mlx/distributed/jaccl/lib/jaccl/mesh.cpp` (`jaccl_pipeline_mode_enabled()` and the QP
  budget comment, lines 15-70, plus connection-construction gating at 154-345)
- `mlx/mlx/distributed/jaccl/jaccl.cpp` (`all_sum`/`all_gather`/`send`/`recv` dispatch to
  `group_->` methods, lines 90-190)
