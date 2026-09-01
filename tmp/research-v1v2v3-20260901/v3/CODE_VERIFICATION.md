# CODE_VERIFICATION.md — DeepSeek-V4-Flash MTP per-cycle state-snapshot profiling path

**Date:** 2026-09-01
**Task:** Verify prior researcher's claims + answer Q1–Q6 against LIVE code in `/Users/adam.durham/repos/exo` (branch main).
**Mode:** READ-ONLY. No files modified except this report.

---

## Files read (absolute paths)

| File | Path |
|---|---|
| MTP spec orchestrator | `/Users/adam.durham/repos/exo/src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` |
| MLX-LM cache classes | `/Users/adam.durham/repos/exo/mlx-lm/mlx_lm/models/cache.py` |
| Cluster launch script | `/Users/adam.durham/repos/exo/start_cluster.sh` |
| DSv4 model (cache construction) | `/Users/adam.durham/repos/exo/mlx-lm/mlx_lm/models/deepseek_v4.py` |
| MTP batch generator (gamma) | `/Users/adam.durham/repos/exo/src/exo/worker/engines/mlx/speculative/mtp_batch_generator.py` |
| Residency analysis doc | `/Users/adam.durham/repos/exo/docs/dspark-352k-residency-analysis-2026-08-27.md` |
| Memory regression doc | `/Users/adam.durham/repos/exo/docs/dspark-352k-memory-regression-2026-08-27.md` |
| KV-read inventory doc | `/Users/adam.durham/repos/exo/docs/p3-worker-a-kv-read-inventory-2026-08-23.md` |

---

## CLAIM VERDICTS

### CLAIM 1 — "An 'rb_snap' profiling bracket exists at dsv4_mtp.py:4111-4145."
**Verdict: CONFIRMED** (current lines 4111–4145, unchanged).

The bracket opens at 4111 and closes at 4145. Verbatim:

```python
# dsv4_mtp.py:4111-4115
        _rbp = _RB_PROFILE and prof is not None
        _t_rb_snap0 = 0.0
        if _rbp:
            mx.synchronize()
            _t_rb_snap0 = time.perf_counter()
```
```python
# dsv4_mtp.py:4141-4145
        if _rbp and prof is not None:
            mx.synchronize()
            prof.record(
                "rb_snap", (time.perf_counter() - _t_rb_snap0) * 1000.0
            )
```

---

### CLAIM 2 — "dsv4_mtp.py:4118-4128 unconditionally calls save_meta() on all ~41 pool caches plus every ring snapshot, on EVERY verify cycle."
**Verdict: DRIFTED** — mechanism CONFIRMED, but the pool count "~41" is **WRONG** (actual = **62**), and "unconditionally" is conditional on `_SPEC_STATE_RESTORE` (which is the production default ON).

The block at 4118–4128 (verbatim):

```python
# dsv4_mtp.py:4118-4128
        if _SPEC_STATE_RESTORE:
            # Unified rollback: snapshot EVERYTHING the verify can mutate
            # (all pools — every pool's remainder grows on every row — and
            # every ring, by O(1) reference).
            _pool_snaps = [pc.save_meta() for pc in _pool_caches]
            for c in gen_batch.prompt_cache:
                subs = c.caches if hasattr(c, "caches") else [c]
                for sub in subs:
                    if hasattr(sub, "save_spec_state"):
                        _ring_caches.append(sub)
                        _ring_snaps.append(sub.save_spec_state())
```

- **"on EVERY verify cycle"** — CONFIRMED. This block sits inside `_speculative_next` (dsv4_mtp.py:3829), which runs once per verify/accept cycle.
- **"unconditionally"** — DRIFTED. It is gated by `if _SPEC_STATE_RESTORE:` (line 4118). `_SPEC_STATE_RESTORE` defaults to ON in the shipped baseline (`start_cluster.sh:246`), so in production it IS effectively unconditional — but the code path is conditional.
- **"all ~41 pool caches"** — **WRONG number.** The actual pool count is **62** (see Q6). The "41" is a stale code comment at dsv4_mtp.py:4107 ("This avoids 41 × mx.array() sync copies..."), which does not match the code that determines the count. This is a stale-comment-as-fact trap.
- **"plus every ring snapshot"** — CONFIRMED. `save_spec_state()` is called on every sub-cache that has it (line 4128).

---

### CLAIM 3 — "BatchPoolingCache.save_meta (cache.py:2533-2568) materializes buf_kv/buf_gate via synchronous mx.array() copies."
**Verdict: CONFIRMED** (current lines 2533–2568, unchanged), with one precision note: `mx.array()` is *lazy graph construction*, not itself a synchronous device copy — the copies are materialized (forced) later by the caller's `mx.synchronize()`/`mx.eval`. The claim's substance (materialized copies) is correct.

Verbatim (cache.py:2533-2550):

```python
    def save_meta(self):
        """Snapshot mutable pooling state for speculative rollback.
        ...
        buf_kv/buf_gate are materialized (copied) because accumulate_windows
        slice-assigns into them in place. pooled is NOT copied: restore_meta
        only rewinds _pool_lengths, and the next real flush overwrites the
        rejected pooled slots via the slice-assign at update_and_fetch.
        """
        buf_kv = None if self.buf_kv is None else mx.array(self.buf_kv)
        buf_gate = None if self.buf_gate is None else mx.array(self.buf_gate)
```

The single-stream `PoolingCache.save_meta` (cache.py:1676-1709) does the same `mx.array()` copies (lines 1701-1702), gated on `remainder > 0`.

---

### CLAIM 4 — "A flush-predicting filter exists only in a legacy branch (dsv4_mtp.py:4137-4140) and is useless at the production config gamma=3 / verify_len=4 / ratio=4 because the pool flushes every cycle."
**Verdict: CONFIRMED** (current lines 4137–4140, unchanged), with a strengthening note: in production the legacy branch is not even reached, because `_SPEC_STATE_RESTORE=1` routes to the 4118–4135 branch instead.

Verbatim (dsv4_mtp.py:4136-4140):

```python
        else:
            _pool_snaps = [
                pc.save_meta() if _pool_may_flush(pc, _verify_len) else None
                for pc in _pool_caches
            ]
```

The predicate (dsv4_mtp.py:758-765):

```python
def _pool_may_flush(pc: Any, verify_len: int) -> bool:
    """Snapshot predicate: could the next ``verify_len``-token forward flush
    this pool? (Snapshotting is the expensive part — buf copies — so only
    pools that can flush are snapshotted.)"""
    rem = pc.remainder
    if isinstance(rem, list):  # BatchPoolingCache: per-stream
        rem = max(rem) if rem else 0
    return rem + verify_len >= pc.ratio
```

With `gamma=3` → `_verify_len = gamma + 1 = 4` (dsv4_mtp.py:4110), and `ratio=4` pools: `rem + 4 >= 4` is **always true** (rem ≥ 0), so `_pool_may_flush` returns True for every ratio-4 pool on every cycle — the filter is a no-op for them. For ratio-128 pools it only fires when `rem >= 124`. So the claim is correct: the filter is useless for ratio-4 pools at verify_len=4. **Strengthening:** because `_SPEC_STATE_RESTORE=1` is the production default, the `else` branch (4137–4140) is dead in production — the 4118–4135 branch snapshots ALL pools unconditionally regardless of the filter.

---

### CLAIM 5 — "start_cluster.sh:246 defaults EXO_DSV4_SPEC_STATE_RESTORE to ON."
**Verdict: CONFIRMED** (current line 246, unchanged).

Verbatim (start_cluster.sh:245-248):

```bash
: "${EXO_DSV4_ROWSEQ_ROWMASK:=1}"
: "${EXO_DSV4_SPEC_STATE_RESTORE:=1}"
: "${EXO_DSV4_SPEC_CACHE_ROLLBACK:=1}"
: "${EXO_DSV4_SPEC_CACHE_ROLLBACK_C2:=1}"
```

`:=1` sets the default to ON when unset. Note the code-level default in dsv4_mtp.py:702 is `"0"` (OFF); the script overrides it to ON at launch. So the *shipped baseline* runs with `_SPEC_STATE_RESTORE=True`.

---

### CLAIM 6 — "The ring path already takes O(1) references at dsv4_mtp.py:692-694."
**Verdict: WRONG.** The cited lines are a **code comment** claiming O(1) references, but the actual implementation **copies** the ring via `mx.array()`. This is a stale-comment-as-fact trap — the exact class of error the audit warned about.

The comment (dsv4_mtp.py:692-694):

```python
# with junk drafts). With EXO_DSV4_SPEC_STATE_RESTORE=1 the B=1 path takes
# an O(1) reference snapshot of every ring (slice-assign rebinds
# keys/values, so pre-verify refs preserve pre-verify contents) plus a
```

The actual implementation (cache.py:777-784, 832-837):

```python
    def save_spec_state(self):
        """Materialized snapshot of the full ring for speculative rollback.
        ...
        NOTE: mx ``__setitem__`` mutates IN PLACE (aliased) — a bare
        reference does NOT preserve pre-write contents — so keys/values are
        copied here (``mx.array``). Small: one local ring per layer.
        ...
        return (
            None if self.keys is None else mx.array(self.keys),
            None if self.values is None else mx.array(self.values),
            self.offset,
            self._idx,
        )
```

**The ring path does NOT take O(1) references — it copies keys/values via `mx.array()`.** The docstring explicitly states a bare reference would NOT preserve pre-write contents because `mx.__setitem__` mutates in place. Per the residency doc (dspark-352k-residency-analysis-2026-08-27.md:248), this is 43 rings × 128 KB = **5.5 MB copied per verify cycle**.

**Critical implication for the V3 hypothesis:** the proposed fix ("copy-on-write / O(1)-reference pool snapshots, mirroring what the ring path already does") is premised on the ring path being O(1). It is not — the ring path copies too. The premise is false and must be re-examined before the fix is designed.

---

## QUESTION ANSWERS

### Q1. Exact env var names, parse, default, effect

**`EXO_DSV4_MTP_PROFILE`**
- Read at: `dsv4_mtp.py:242`
- Parse: `int(os.environ.get("EXO_DSV4_MTP_PROFILE", "0"))` → integer
- Default when unset: `0`
- Effect when >0: creates the `_phase_timer` (`dsv4_mtp.py:825`), which brackets draft/verify/accept phases with `mx.eval` + `perf_counter` and dumps aggregated stats every `_PROFILE_INTERVAL` cycles. `_PROFILE_INTERVAL` IS the value of the env var (e.g. `=50` dumps every 50 cycles).

```python
# dsv4_mtp.py:242
_PROFILE_INTERVAL = int(os.environ.get("EXO_DSV4_MTP_PROFILE", "0"))
# dsv4_mtp.py:825
_phase_timer = _PhaseTimer() if _PROFILE_INTERVAL > 0 else None
```

**`EXO_DSV4_RB_PROFILE`**
- Read at: `dsv4_mtp.py:258`
- Parse: `os.environ.get("EXO_DSV4_RB_PROFILE", "0") == "1"` → boolean (exactly the string `"1"`)
- Default when unset: `"0"` → False
- Effect when `"1"`: enables rollback sub-phase attribution brackets (`rb_snap`/`rb_gate`/`rb_drain`/`rb_ring`/`rb_pool`/`rb_pool_restores`/`rb_commitfwd`/`rb_tail`) with `mx.synchronize()` sub-boundaries. **Requires `EXO_DSV4_MTP_PROFILE>0`** (the `prof is not None` guard in `_rbp = _RB_PROFILE and prof is not None`, dsv4_mtp.py:4111).

```python
# dsv4_mtp.py:258
_RB_PROFILE = os.environ.get("EXO_DSV4_RB_PROFILE", "0") == "1"
```

**Both must be set to get `rb_snap` output.** `EXO_DSV4_MTP_PROFILE` alone gives the base draft/verify/accept/rollback/total phases; `EXO_DSV4_RB_PROFILE=1` adds the rb_* sub-phases on top.

---

### Q2. What rb_snap measures, what it emits, format, stream, frequency, units

**Measures:** the pre-verify snapshot/arm block — from the opening `mx.synchronize()` (dsv4_mtp.py:4114) through the `save_meta()`/`save_spec_state()` calls (4118–4128) to the closing `mx.synchronize()` (4142). This is the wall time of materializing all pool + ring snapshots, including the forced device sync.

**Emit:** `prof.record("rb_snap", (time.perf_counter() - _t_rb_snap0) * 1000.0)` (dsv4_mtp.py:4143-4144). The value is in **milliseconds** (`* 1000.0`).

**Output line format** (from `_PhaseTimer.dump`, dsv4_mtp.py:806-822):

```python
logger.warning(f"[MTP-PROF] cycles={self.cycles} {bs_summary}")
...
logger.warning(
    f"[MTP-PROF]   B={b} {phase:10s} mean={mean:6.2f}ms "
    f"min={min(xs):6.2f}ms max={max(xs):6.2f}ms n={len(xs)}"
)
```

So the rb_snap line looks like:
```
[MTP-PROF]   B=1 rb_snap    mean=  2.34ms min=  1.10ms max=  5.67ms n=50
```
(plus a header `[MTP-PROF] cycles=50 B=1:50`).

**Stream:** `logger.warning(...)` → Python logging. The runner is launched with `>> ~/exo.log 2>&1` (start_cluster.sh:2771), so it lands in **`~/exo.log` on the node** (not stdout of the launch shell, not a separate runner log). It is emitted by whichever rank runs `_speculative_next` — in PP mode that is the rank executing the MTP loop.

**Frequency:** aggregated and dumped every `_PROFILE_INTERVAL` cycles (dsv4_mtp.py:803: `if _PROFILE_INTERVAL > 0 and self.cycles % _PROFILE_INTERVAL == 0: self.dump()`). The dump reports mean/min/max/n over the interval, not per-cycle lines.

**Units:** milliseconds (ms).

---

### Q3. Other profiling output enabled by EXO_DSV4_MTP_PROFILE

The `_PhaseTimer` emits these phases (dsv4_mtp.py:811-814). The "known" set is `draft, verify, accept, commit, rollback, total`; any extra keys recorded (the rb_* sub-phases) are appended in sorted order:

```python
known = ("draft", "verify", "accept", "commit", "rollback", "total")
for b in sorted(self.samples.keys()):
    extras = tuple(sorted(k for k in self.samples[b] if k not in known))
    for phase in known + extras:
```

Emitted metric names (all in the same `[MTP-PROF]   B={b} {phase} mean=... min=... max=... n=...` format, ms):
- `draft` (dsv4_mtp.py:2640, 4078, 5509)
- `verify` (dsv4_mtp.py:2699, 4218, 5568)
- `accept` (dsv4_mtp.py:3066, 4918, 5658)
- `commit` (dsv4_mtp.py:5745)
- `rollback` (dsv4_mtp.py:3384)
- `total` (dsv4_mtp.py:3387)
- `rb_snap` (dsv4_mtp.py:4143)
- `rb_gate` (dsv4_mtp.py:4978)
- `rb_drain` (dsv4_mtp.py:4985)
- `rb_ring` (dsv4_mtp.py:4998)
- `rb_pool` (dsv4_mtp.py:5022)
- `rb_pool_restores` (dsv4_mtp.py:5023) — a count, not ms
- `rb_commitfwd` (dsv4_mtp.py:5041)
- `rb_tail` (dsv4_mtp.py:5054)

The rb_* phases only appear when `EXO_DSV4_RB_PROFILE=1`. The `rb_pool_restores` value is a float count of pools taking the restore+re-accumulate branch that cycle (not a time).

---

### Q4. Side effects beyond measurement — does the profiler perturb what it measures?

**YES — it serializes the pipeline and forces device syncs.** This is documented in the code comments themselves.

The `EXO_DSV4_MTP_PROFILE` comment (dsv4_mtp.py:238-241):

```python
# Per-cycle phase timing. When EXO_DSV4_MTP_PROFILE > 0, brackets the
# draft / verify / accept phases with mx.eval + perf_counter, summarising
# every N cycles. Inserts evals at phase boundaries which serialises
# pipelining — measurements are upper bounds on real production walls.
```

The `EXO_DSV4_RB_PROFILE` comment (dsv4_mtp.py:251-253):

```python
# This gate splits them with mx.synchronize() sub-boundaries (serialises
# the pipeline — SHARES are trustworthy, absolute totals are upper
# bounds, same caveat as EXO_DSV4_SECTION_TIME) and also brackets the
# pre-verify snapshot/arm block (rb_snap), whose mx.array copies
# otherwise hide inside the "verify" phase.
```

Concrete perturbations:
- `mx.eval` at phase boundaries (e.g. dsv4_mtp.py:2638 `mx.eval(*draft_ids_list)`, 4162 `mx.eval(verify_pre_norm, verify_logits)`) forces lazy graph construction to materialize, serializing the pipelined draft/verify/accept.
- `mx.synchronize()` at the rb_snap boundaries (4114, 4142) forces the snapshot `mx.array()` copies to actually complete on device.

**Implication for the plan:** `rb_snap` is an **upper bound** on the real production snapshot cost (it includes the forced sync and the serialization penalty). The *shares* (rb_snap as a fraction of the total cycle) are the trustworthy signal; the absolute ms is inflated. This is the correct conservative direction for the pre-registered gate (rb_snap ≥ 3-5% of cycle time) — if rb_snap is small even under this serialized, upper-bound measurement, the hypothesis is genuinely weak.

---

### Q5. How start_cluster.sh propagates env vars; exact invocation; baseline env

**Mechanism:** env vars are assembled into a single space-separated string `EXO_ENV` of `VAR=VAL` pairs, then interpolated as inline `VAR=VAL` prefixes to the python command inside a `screen` session launched over `ssh`.

The two profiling vars are appended to `EXO_ENV` only if set (start_cluster.sh:2018, 2022):

```bash
[ -n "${EXO_DSV4_MTP_PROFILE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_PROFILE=$EXO_DSV4_MTP_PROFILE"
...
[ -n "${EXO_DSV4_RB_PROFILE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_RB_PROFILE=$EXO_DSV4_RB_PROFILE"
```

The launch command (start_cluster.sh:2771):

```bash
LAUNCH_CMD="cd ~/repos/exo && $EXO_ENV EXO_DISCOVERY_PEERS=$NODE_PEERS .venv/bin/python -m exo -v >> ~/exo.log 2>&1 $LAUNCH_TAIL"
```

Executed via (start_cluster.sh:2823):

```bash
ssh "$NODE" "screen -dmS exorun zsh -l -c '$LAUNCH_CMD'"
```

So the env vars are passed as literal `VAR=VAL` prefixes on the `python -m exo` command line inside a detached `screen` session on each node. The script reads them from the **launching shell's environment** (via `${EXO_DSV4_MTP_PROFILE:-}`), so the user sets them in the shell before running the script.

**Exact invocation to launch the cluster with both profiling vars set:**

```bash
EXO_DSV4_MTP_PROFILE=50 EXO_DSV4_RB_PROFILE=1 ./start_cluster.sh
```
(or `export EXO_DSV4_MTP_PROFILE=50 EXO_DSV4_RB_PROFILE=1` first, then `./start_cluster.sh`). The `=50` sets the dump interval to every 50 cycles; any positive integer works. `EXO_DSV4_RB_PROFILE=1` is required for the rb_snap line.

**Baseline env vars (campaign baseline vs. shipped defaults):**

The campaign baseline names are abbreviations; the real env var names carry the `EXO_DSV4_` prefix. Verified against the script:

| Campaign name | Real env var | Default | Line |
|---|---|---|---|
| LMHEAD_MXFP8 | `EXO_DSV4_LMHEAD_MXFP8` | `1` | start_cluster.sh:599 |
| EXACT_TOPK_PREFILL | `EXO_DSV4_EXACT_TOPK_PREFILL` | `1` | start_cluster.sh:40 |
| QUERY_TILED_SDPA | `EXO_DSV4_QUERY_TILED_SDPA` | `1` | start_cluster.sh:48 |
| VERIFY_BATCH | `EXO_DSV4_VERIFY_BATCH` | `1` | start_cluster.sh:335 |
| (depth gate) | `EXO_DSV4_VERIFY_BATCH_MIN_CTX` | `8192` | start_cluster.sh:341 |

Other shipped defaults relevant to the snapshot path: `EXO_DSV4_SPEC_STATE_RESTORE=1` (line 246), `EXO_DSV4_SPEC_CACHE_ROLLBACK=1` (line 247), `EXO_DSV4_SPEC_CACHE_ROLLBACK_C2=1` (line 248), `EXO_SPECULATIVE_GAMMA=3` (line 176), `EXO_DSV4_ROWSEQ_ROWMASK=1` (line 245).

---

### Q6. How many pool caches, and where is that number determined?

**Actual count: 62 pools** (not ~41). Determined by the model config's `compress_ratios` + `make_cache` in `deepseek_v4.py`.

The count is derived from:
- `num_hidden_layers = 43` (deepseek_v4.py:839)
- `compress_ratios` truncated to 43 entries (deepseek_v4.py:894)
- `make_cache` (deepseek_v4.py:7331-7354): each `SparseCompressedAttention` layer (ratio=4) gets **2** pools (compressor + indexer), each `CompressedAttention` layer (ratio=128) gets **1** pool, each `LocalAttention` layer (ratio=0) gets **0** pools.

```python
# deepseek_v4.py:7331-7353
    def make_cache(self):
        caches = []
        for layer in self.layers:
            ratio = layer.attn.compress_ratio
            if ratio == 0:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            elif isinstance(layer.attn, SparseCompressedAttention):
                # local + compressor pool + indexer pool
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=self.args.sliding_window),
                        PoolingCache(ratio),
                        PoolingCache(ratio),
                    )
                )
            else:
                # local + compressor pool
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=self.args.sliding_window),
                        PoolingCache(ratio),
                    )
                )
```

Layer census (p3-worker-a-kv-read-inventory-2026-08-23.md:44-48): ratio 0 = 2 layers, ratio 4 = 21 layers, ratio 128 = 20 layers. (The dspark-352k-residency doc at line 50-52 says "3 local layers" — the two docs disagree on the local-layer count; the p3 inventory's 2+21+20=43 is self-consistent.)

**Pool count = 21 sparse × 2 + 20 compressed × 1 = 62 pools.**

The "~41" in the claim (and in the code comment at dsv4_mtp.py:4107) is **stale/wrong**. The residency doc independently confirms 62: "21 layers × 2 pools = 336 KB" + "20 layers = 5.12 MB" (dspark-352k-residency-analysis-2026-08-27.md:255-256), and "4 rows × (21+21+20) pool caches" (line 226).

---

## ACCEPTANCE ASSERTIONS

1. **Every claim verdict is one of CONFIRMED / DRIFTED / WRONG / NOT FOUND** — yes. C1 CONFIRMED, C2 DRIFTED, C3 CONFIRMED, C4 CONFIRMED, C5 CONFIRMED, C6 WRONG. No hedging.
2. **Every verdict carries a verbatim code quote + current file:line** — yes (see above).
3. **Cited line ranges that don't contain the claim are flagged with what IS there** — C6: lines 692-694 contain a *comment* claiming O(1) references, but the implementation copies; C2: the "~41" is a stale comment at 4107, actual count is 62.
4. **Absolute path of every file read** — reported in the table above.
5. **No guessing at env var behavior from naming** — every env var behavior is cited to the line that reads it (Q1).

---

## Key findings for the V3 measurement plan

1. **To get `rb_snap`, you MUST set BOTH `EXO_DSV4_MTP_PROFILE=N` (N>0) AND `EXO_DSV4_RB_PROFILE=1`.** Setting only one produces no rb_snap line. Launch: `EXO_DSV4_MTP_PROFILE=50 EXO_DSV4_RB_PROFILE=1 ./start_cluster.sh`.
2. **Output goes to `~/exo.log` on the node** (via `logger.warning`, runner launched with `>> ~/exo.log 2>&1`), aggregated every N cycles as `[MTP-PROF]   B=1 rb_snap    mean=...ms min=...ms max=...ms n=...`. Units ms.
3. **The profiler perturbs what it measures** — it inserts `mx.eval`/`mx.synchronize()` that serialize the pipeline. `rb_snap` is an **upper bound**; the *share* (rb_snap/total) is the trustworthy signal. This is the correct conservative direction for the pre-registered gate.
4. **CRITICAL — the V3 fix premise is FALSE.** The ring path does NOT take O(1) references (C6 WRONG). `save_spec_state` copies keys/values via `mx.array()` (43 rings × 128 KB = 5.5 MB/cycle). The proposed "mirror the ring path's O(1) references" fix has no existing O(1) reference implementation to mirror — the ring path copies too. The fix design must be re-derived.
5. **Pool count is 62, not ~41** — the "41" is a stale comment. Per-cycle pool snapshot cost is ~5.5 MB (buf copies) + 5.5 MB (rings) ≈ 11 MB/cycle, consistent with the V3 hypothesis's ~11 MB figure.
6. **The flush-predicting filter (C4) is dead in production** — `_SPEC_STATE_RESTORE=1` routes to the unconditional-snapshot branch, so the legacy `_pool_may_flush` filter is never reached. The unconditional per-cycle snapshot is real and active in the shipped baseline.
