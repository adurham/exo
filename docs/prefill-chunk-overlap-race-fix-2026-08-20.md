# EXO_PREFILL_CHUNK_OVERLAP — cross-stream ordering race: root cause, fix, and campaign closure

**Date:** 2026-08-20
**Follow-up to:** `docs/prefill-chunk-overlap-live-test-2026-08-20.md`
**Status:** correctness bug FIXED; throughput lever **DEAD / CLOSED**.

---

## 1. Background

`EXO_PREFILL_CHUNK_OVERLAP` (introduced in `b7aa41920`, env-forwarded in
`76f030e62`, default OFF) double-buffers the per-chunk KV-cache sync in
`prefill_batched()` so one chunk's collectives overlap the next chunk's
compute.

Its single live deployment (2026-08-20, 2 trials) produced:

- **1 of 2 trials: a real correctness anomaly.** Wrong secret-recall
  output — non-crashing, non-truncated, genuinely different logits (not a
  byte-shift / detokenizer artifact).
- **FLAT throughput.** No measurable gain.

## 2. Root cause

In the `prefill_batched()` chunk loop, the per-chunk sequence was:

1. Build this chunk's forward-pass graph, including its internal TP
   `all_sum` collectives, on `generation_stream`.
2. `mx_barrier(group)` — which issues its **own** 1-element `all_sum` on a
   **separate CPU stream** (`utils_mlx.py:mx_barrier`), with **zero data
   dependency** on the compute graph.
3. **Only then** `mx.eval(_prev_cache_sync)` — draining the **previous**
   chunk's collectives.
4. `mx.async_eval(...)` — non-blocking dispatch of this chunk's own
   collectives.

Because the barrier (step 2) was posted **before** the previous chunk's
collectives were drained (step 3), and the barrier lives on a causally
disconnected stream, the two TP ranks had **no structural guarantee** of
posting their barrier-vs-chunk-collective sequence in the same relative
order onto the distributed transport run-to-run.

If the transport matches collectives by **wire-arrival order** rather than
explicit tags, one rank can match its 1-element barrier `all_sum` against
the **other rank's real attention/MoE reduction tensor** — silent numeric
corruption. That signature is an exact match for the observed anomaly:
wrong logits, no crash, no hang, no truncation.

This is a genuine happens-before violation, not a narrow timing window.

## 3. Fix applied

**File:** `src/exo/worker/engines/mlx/generator/generate.py`, `prefill_batched()`
chunk loop (~L1462–1510).

**Change: swap `[barrier, then drain]` → `[drain, then barrier]`.**

The depth-1 double-buffer's block-on-previous step was moved from *just
before* this chunk's `async_eval` to the **top of the loop iteration**,
ahead of the barrier:

```python
# drain PREVIOUS chunk's collectives FIRST
if _overlap and _prev_cache_sync is not None:
    mx.eval(_prev_cache_sync)
    _prev_cache_sync = None
    request_trace.record(f"prefill_batched.chunk{chunk_idx}.drain_prev", _t_drain)

# ONLY THEN the barrier
mx_barrier(group)

...
if _overlap:
    _cache_state = [c.state for c in batched_cache]
    mx.async_eval(_cache_state)
    _prev_cache_sync = _cache_state
```

This preserves both properties the original design needed — in-flight
depth is still bounded at 1 (long prefills don't OOM) and the compute
overlap still happens — while restoring a **strict happens-before**: no
barrier collective is ever posted while a real chunk collective from a
prior chunk is still in flight, on either rank.

The `mx.eval(_prev_cache_sync)` tail-drain after the loop (for the final
chunk) is unchanged and still required.

### 3a. Alternative fix considered and REJECTED (with evidence)

The other candidate was to move `mx_barrier`'s `all_sum` onto
`generation_stream`, letting MLX's same-stream program-order guarantee
serialize it. **This is not possible.** Implemented and tested locally —
the ring backend rejects a GPU-stream collective outright:

```
File ".../src/exo/worker/engines/mlx/utils_mlx.py", line 1992, in mx_barrier
    mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group, stream=barrier_stream))
File ".../mlx_lm/models/deepseek_v4.py", line 476, in wrapped
    return fn(x.astype(mx.bfloat16), *args, **kwargs).astype(mx.float32)
RuntimeError: bad_variant_access
```

`utils_mlx.py` was therefore reverted to its original form and is
**unmodified** by this fix. This is a useful standing constraint: exo's
distributed collectives must stay on the CPU stream under the ring
backend. A note recording this is left inline at the barrier call site.

## 4. Diagnostic logging added (rank-desync visibility)

`_overlap` was previously never logged, so a future per-rank flag desync
(e.g. a manual single-node relaunch bypassing `start_cluster.sh`'s env
fanout) would have been **completely silent** — one rank running the
overlapped loop and the other the serial one, diverging their collective
sequences with no log trace.

Added `_log_chunk_overlap_config()` (`generate.py`, ~L1240), fired once per
process at the top of `prefill_batched()`:

```
[PREFILL_CHUNK_OVERLAP] host=adams-macbook-pro-m4 rank=0/2 \
    EXO_PREFILL_CHUNK_OVERLAP='0' resolved_overlap=False
[PREFILL_CHUNK_OVERLAP] host=adams-macbook-pro-m4 rank=1/2 \
    EXO_PREFILL_CHUNK_OVERLAP='0' resolved_overlap=False
```

(Real captured output from the 2-rank loopback run.) Logs hostname, rank,
world size, the raw env string, and the resolved boolean — a desync now
shows up as two non-matching lines.

## 5. Local verification

**Harness:** `bench/local_repro_chunk_overlap.py` — new. Drives the **real**
`prefill_batched()` against a **real 2-rank `mx.distributed` group** on this
one machine (loopback ring, **zero live-cluster involvement**), using a tiny
synthetic DeepSeek-V4 model whose MoE/attention layers issue **genuine
`all_sum` collectives**. RNG is seeded identically before weight init on
both ranks, so cross-rank comparison is meaningful. No tokenizer/HF
download (token ids from `mx.random.randint`).

Two prompt lengths, both crossing multiple chunk boundaries at
`prefill_step_size=32`:

| prompt_len | processed len | kind | chunks |
|---|---|---|---|
| 129 | 128 | exact multiple of 32 | 4 |
| 100 | 99  | remainder (3×32 + 3), odd last chunk | 4 |

Run command:

```
.venv/bin/mlx.launch -n 2 --backend ring \
    .venv/bin/python bench/local_repro_chunk_overlap.py
```

### (a) BEFORE the fix — did the race reproduce?

**No — not reproduced on loopback.** 3 independent 2-rank runs against the
pristine `HEAD` version of `generate.py`, all PASS (overlap=0 vs overlap=1
bit-exact, both prompt lengths, both ranks). Sample:

```
[rank 0/2] === prompt_len=129 (exact-multiple, step=32, 4 chunks) ===
[rank 0/2] cache_state_identical      = True
[rank 0/2] decode_argmax_identical    = True
[rank 0/2] decode_logits_max_abs_diff = 0.000e+00
[rank 0/2] decode_logits_bit_exact    = True
[rank 0/2] RESULT prompt_len=129: PASS
[rank 0/2] OVERALL: PASS
```

**This is expected and is NOT evidence the race is unreal.** Loopback ring
on a single machine with a tiny model gives both ranks near-identical
scheduling and a trivially fast transport — precisely the conditions under
which the two ranks' post ordering stays incidentally aligned. The live
anomaly appeared in 1 of 2 trials on real hardware at 500K context with
real inter-node TB5/RDMA latency. **The race remains a code-consistent,
statically-argued defect; it is not dynamically proven, and this document
does not claim otherwise.** The fix is justified by the ordering argument,
not by a repro.

### (b) AFTER the fix — acceptance bar

3 independent 2-rank runs, all PASS. Full output of one run:

```
[rank 1/2] === prompt_len=129 (exact-multiple, step=32, 4 chunks) ===
[rank 1/2] cache_state_identical      = True
[rank 1/2] decode_argmax_identical    = True
[rank 1/2] decode_logits_max_abs_diff = 0.000e+00
[rank 1/2] decode_logits_bit_exact    = True
[rank 1/2] XRANK overlap0_digest=e31271d812738840|ff0d73a0366adbd1+faab6a2996170067|797212b801f3150e+ad3890d02bec423a+4e491a793c40b06f|a8b268577304a63e+26a6ab83f2837971|0727afc6f12f51f9+a2c84c1fed0f8cc0+5133ed80228a826a|61ef35dd6cfcd9bf
[rank 1/2] XRANK overlap0_argmax=[1203, 1504, 98, 622]
[rank 1/2] RESULT prompt_len=129: PASS
[rank 0/2] cache_state_identical      = True
[rank 0/2] decode_argmax_identical    = True
[rank 0/2] decode_logits_max_abs_diff = 0.000e+00
[rank 0/2] decode_logits_bit_exact    = True
[rank 0/2] XRANK overlap0_digest=e31271d812738840|ff0d73a0366adbd1+faab6a2996170067|797212b801f3150e+ad3890d02bec423a+4e491a793c40b06f|a8b268577304a63e+26a6ab83f2837971|0727afc6f12f51f9+a2c84c1fed0f8cc0+5133ed80228a826a|61ef35dd6cfcd9bf
[rank 0/2] XRANK overlap0_argmax=[1203, 1504, 98, 622]
[rank 0/2] RESULT prompt_len=129: PASS

[rank 1/2] === prompt_len=100 (remainder, step=32, 4 chunks) ===
[rank 1/2] cache_state_identical      = True
[rank 1/2] decode_argmax_identical    = True
[rank 1/2] decode_logits_max_abs_diff = 0.000e+00
[rank 1/2] decode_logits_bit_exact    = True
[rank 1/2] XRANK overlap0_argmax=[1726, 1726, 1726, 1726]
[rank 1/2] RESULT prompt_len=100: PASS
[rank 1/2] OVERALL: PASS
[rank 0/2] cache_state_identical      = True
[rank 0/2] decode_argmax_identical    = True
[rank 0/2] decode_logits_max_abs_diff = 0.000e+00
[rank 0/2] decode_logits_bit_exact    = True
[rank 0/2] XRANK overlap0_argmax=[1726, 1726, 1726, 1726]
[rank 0/2] RESULT prompt_len=100: PASS
[rank 0/2] OVERALL: PASS
```

Three things hold simultaneously:

1. **overlap=0 ≡ overlap=1**, bit-exact (`max_abs_diff = 0.000e+00`), on
   cache state and decode logits, on both chunk-boundary cases.
2. **rank 0 ≡ rank 1** — the `XRANK` cache digests and argmaxes are
   character-for-character identical across ranks, i.e. both ranks reduced
   the same values (no mis-matched collective).
3. Reproducible across 3 runs.

Single-process sanity mode (`--single`, `group=None`) also PASSes both
cases, confirming the harness isn't vacuously passing.

### Harness caveat

`bench/local_repro_chunk_overlap.py` installs a small class-attribute shim
giving `mlx_lm.models.cache.BatchPoolingCache` a `_overlap_kv_carry` /
`_overlap_gate_carry` default of `None`. The installed `mlx_lm` defines
these on `PoolingCache` but not its batched sibling, so the batched DSv4
compressor path `AttributeError`s on chunk 0 without it. That is a
**pre-existing gap in the vendored `mlx_lm`, orthogonal to this race**, and
is shimmed only inside the bench script — no production code was changed
for it. It is worth fixing upstream separately.

## 6. Gate results

| Gate | Result |
|---|---|
| `uv run ruff check` on `generate.py` | **10 errors before, 10 after — zero new.** All 10 pre-exist at `HEAD` (import sorting, `E702` semicolons, `SIM105`) and are in code untouched by this change. |
| `uv run ruff check bench/local_repro_chunk_overlap.py` | `All checks passed!` |
| `uv run basedpyright` on `generate.py` | **131 errors before → 123 after.** Zero new; 8 fewer. |
| `uv run pytest src/exo/worker/engines/mlx/tests/test_prefill_batched.py -m ""` | `3 passed in 5.97s` |
| `nix fmt` | **NOT RUN** — `nix` is not installed on this machine (`which nix` → not found). |

## 7. Campaign verdict — THIS AVENUE IS CLOSED

Two separate conclusions, deliberately kept distinct:

**Correctness (this document's fix): KEEP.** The ordering defect was real
and code-consistent. The fix is a root-cause happens-before restoration —
no sleeps, retries, or timeouts. `EXO_PREFILL_CHUNK_OVERLAP` is now
*believed* correctness-safe. Worth keeping in the tree for repo hygiene so
the flag isn't a live landmine for anyone who flips it.

**Throughput: DEAD. This avenue is CLOSED.**

- Estimated maximum possible gain from this lever is **~1–3%, likely near
  0%**.
- The cache-sync cost it targets is **not even a separately itemized cost**
  in the wall-time profile at long context.
- MLX's existing pipelining already hides most comms cost past ~300K
  context (confirmed by a prior NOP-ablation on collectives).
- The one live deployment measured **FLAT throughput**.

Against the DSv4-Flash prefill campaign (baseline 339 tok/s @ 500K, target
350), this was **lever 12 of 12 — all dead or noise.**

### Standing decisions

- `EXO_PREFILL_CHUNK_OVERLAP` **remains OFF by default, permanently.**
- **Do not re-litigate this lever for throughput.** Do not attempt to make
  the overlap faster, deeper (depth >1), or wider. The ceiling has been
  measured and it is not there.
- Do not treat the correctness fix as a reason to re-enable the flag in
  production. It is a hygiene fix, not an unblock.
- The "prefill throughput via compute/comm overlap" avenue is **closed**.
  Any future prefill work should target the measured compute-bound ceiling
  (see `docs/dsv4-attention-kernel-efficiency-2026-08-18.md` and
  `docs/gpu-utilization-confirmed-saturated-2026-08-18.md`), not comms.

## 8. NOT verified

- **The live cluster was never touched during this work** — no ssh, no
  curl, no relaunch. 100% local repo work plus loopback `mlx.launch -n 2
  --backend ring` on this machine.
- The fix is **not** validated on real 2-node TB5/RDMA hardware, at real
  500K context, or against the real DSv4-Flash checkpoint.
- The original race was **never dynamically reproduced** — see §5(a). The
  fix rests on the static ordering argument.
- Loopback ring is not the jaccl backend. The wire-arrival-ordering
  property of jaccl specifically was not tested.
- `nix fmt` was not run (not installed).
