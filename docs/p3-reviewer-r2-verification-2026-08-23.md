# P3 Reviewer R2 — independent verification of workers C, C3, D + PERFORMANCE_HISTORY integrity (2026-08-23)

**Scope**: second reviewer. R1 already verified workers A and B1 — NOT re-verified here.
This pass covers **C3** (the new `BatchPoolingCache`-concat mechanism, the load-bearing
claim of the coming synthesis), **C** (kernel-delta spot-check), **D** (crash-forensics
spot-check), and the integrity of the seven P3 entries in `docs/PERFORMANCE_HISTORY.md`.

**Method**: read-only. Every verdict below was re-derived from the primary source (the
code at the cited file:line, or the worker's own pasted raw output), not from the
worker's prose. Arithmetic recomputed in Python. No cluster contact, no bench runs, no
ssh. One write: the PERFORMANCE_HISTORY append.

**Headline**: **C3 4 of 6 CONFIRMED, 1 CONFIRMED-WITH-CORRECTION, 1 REFUTED.**
The REFUTED one is **C3-4, the correctness precondition of the proposed live A/B** —
the patch is (I believe) safe, but **for a different reason than the doc gives**, and
the doc's stated justification is wrong in both of its halves. Since the PM is about to
forward that A/B spec, this needs correcting before it ships. C3-6's "43 layers" is
also wrong (should be 21, and the per-step figure is 4× too large) — cosmetic, changes
no verdict. C's numbers are internally consistent; **C-2's overshoot is real and
C's doc does NOT address it** — the PM must carry it as a caveat. D's evidence chain is
sound and does not overclaim. History file is intact, all seven entries present, no
drift found.

---

## C3-1 — production decode routes caches through `_merge_caches` → `BatchPoolingCache` — **CONFIRMED**

```
$ grep -n "_merge_caches" mlx-lm/mlx_lm/generate.py
1002:def _merge_caches(caches):
1261:        self.prompt_cache = _merge_caches(caches)

$ sed -n '1259,1262p' mlx-lm/mlx_lm/generate.py
        self.uids = uids
        self.prompt_cache = _merge_caches(caches)          # in PromptProcessingBatch.__init__

$ awk 'NR>=1002 && NR<=1015' mlx-lm/mlx_lm/generate.py
def _merge_caches(caches):
    ...
        if hasattr(caches[0][i], "merge"):
            batch_cache.append(caches[0][i].merge([c[i] for c in caches]))

$ grep -n "class PoolingCache\|return BatchPoolingCache.merge\|class BatchPoolingCache" mlx-lm/mlx_lm/models/cache.py
1270:class PoolingCache(_BaseCache):
1823:        return BatchPoolingCache.merge(caches)      # PoolingCache.merge, decorated @classmethod at 1821-1822
1826:class BatchPoolingCache(_BaseCache):
```

`PromptProcessingBatch.__init__` (generate.py:1229/1261) unconditionally converts, and
`PoolingCache.merge` returns a `BatchPoolingCache` (cache.py:1823, body at 2666-2701).
The exo entry points check out: `batch_generate.py:2680` calls `self._mlx_gen.insert(...)`
(C3 cites 2678 — that is the closing paren of the preceding call; **±2 line drift, immaterial**),
and `batch_generate.py:4228` is `_prompt_responses, responses = self._mlx_gen.next()`, exactly
as cited. C3's runtime print (`LIVE CACHE CLASS: BatchPoolingCache`) is consistent with the code.

Cite-accuracy note: C3 gives `cache.py:1822-1823` and `generate.py:1261`; actual are 1823
and 1261. Within tolerance.

---

## C3-2 — the growth asymmetry (concat-to-exact-max_pool vs 256-chunk) — **CONFIRMED**

`BatchPoolingCache.update_and_fetch_deferred`, **cache.py:1899-1903 — exactly as cited**:

```
1885|        max_pool = max(self._pool_lengths) + max_new
1898|        else:
1899|            if self.pooled.shape[1] < max_pool:
1900|                pad = mx.zeros(
1901|                    (B, max_pool - self.pooled.shape[1], D), dtype=px.dtype
1902|                )
1903|                self.pooled = mx.concatenate([self.pooled, pad], axis=1)
```

`max_pool = max(_pool_lengths) + max_new`, so at ratio-4 decode `max_new == 1` and the pad
is **exactly one entry**. There is no chunking, no slack, no `step` term anywhere in this
branch. `mx.concatenate` allocates a fresh `(B, max_pool, D)` and copies — unconditional
O(P·D) **on every flush**. Confirmed.

`PoolingCache.update_and_fetch_deferred`, **cache.py:1517-1528 — as cited (1522-1528)**:

```
1517|        elif new_offset > self._pool_storage.shape[1]:
1522|            current_size = self._pool_storage.shape[1]
1523|            grow_by = max(self.step, new_offset - current_size + 1)
1524|            new_size = current_size + grow_by
1527|            self._pool_storage = mx.zeros((B, new_size, D), dtype=px.dtype)
1528|            self._pool_storage[:, : self._pool_offset] = old[:, : self._pool_offset]

$ grep -n "step: int = 256" mlx-lm/mlx_lm/models/cache.py
1290:    step: int = 256        # PoolingCache
```

`grow_by = max(256, 1)` = 256, and the `elif` only fires when `new_offset` exceeds
storage — so 1 growth in 256 flushes. The asymmetry is real and exactly as described.

Two structural confirmations C3 did not state but which make the claim stronger:
- `PoolingCache` keeps a **separate storage/view split** (`_pool_storage` + `_pool_offset`,
  cache.py:1360-1364 `pooled` property returns `self._pool_storage[:, :self._pool_offset]`).
  `BatchPoolingCache` has no such split — `self.pooled` IS the storage
  (`size()` = `self.pooled.shape[1]`, cache.py:2532-2533). That is *why* it cannot carry
  slack the way `PoolingCache` does: its length and its capacity are the same number.
- Reaches production decode: `SparseCompressedAttention` (21 layers, ratio 4) drives
  `Compressor` → `<pool>.update_and_fetch_deferred` every step, flushing 1-in-4.
  Census independently re-derived by R1 (21×r4 / 20×r128 / 2×r0) and matches.

---

## C3-3 — no cross-step pool-reference holder on the live path — **CONFIRMED (all four cites)**

| cite | verified |
|---|---|
| `mx.eval(inputs, self._current_logprobs)` @ generate.py:**1639** | **exact hit.** Line 1639 verbatim. Comment at 1636-1638 confirms it fences the PREVIOUS step's outputs. |
| `eager_detach_caches(self.prompt_cache)` @ generate.py:**1650** | **1-line drift**: the import is at 1650, the call at **1651**. Immaterial. |
| `mx.async_eval(self._next_tokens, ...)` @ generate.py:1632 | exact hit. |
| `_pp_spec_active` gate @ batch_generate.py:**971-981** | **exact hit.** `if (use_speculative and draft_path and self.group is not None and self.group.size() > 1)` (971-976) then `if get_pipeline_info(self.model) is not None: self._pp_spec_active = True` (980-981). In Tensor mode there is no pipeline split → returns None → stays False. |

Production env cross-checked against `start_cluster.sh` defaults — every value C3 lists is
right: `DSV4_SHARDING:=Tensor` (:346), `EXO_DSV4_MTP:=0` (:459), `EXO_DSV4_DSPARK:=1` (:1715),
`EXO_KV_CACHE_BITS:=0` (:151), `EXO_DSV4_INDEX_TOPK:=512` (:33), `EXO_DSV4_ATTN_ALLSUM:=0`
(:1755), `EXO_DSV4_FENCE_ASYNC:=1` (:1626), `EXO_DSV4_FENCE_EVERY_N_LAYERS:=4` (:437),
`EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES:=8388608` (:2069).

Also verified C3's supporting note that `eager_detach_caches` cannot reach `pooled`:
`_CACHE_DETACH_ATTRS = ("keys","values")` — correct, so it is neutral w.r.t. the pool, exactly
as C3 says. And `mx.async_eval(self.pooled)` at cache.py:**1918** (C3 said 1918 — exact hit),
inside the `visible is None` branch that production takes (pool ≫ 8 MiB).

---

## C3-4 — the A/B's correctness precondition — **REFUTED as stated** (the conclusion is probably right; the reasoning is not)

This is the claim I was asked to weigh most heavily, because the PM will forward the A/B
spec. **C3 §8.1's correctness note is wrong in both of its halves.** The proposed patch
still looks *safe*, but nobody should ship it believing C3's stated reason.

C3 §8.1 says:

> growing the pool *larger* than `max_pool` is already safe by construction — `make_mask`
> clamps the mask width to `self._visible_width` (`cache.py:2174-2176`), and `_visible_width`
> is set from the returned tensor at `cache.py:1920`, so extra trailing capacity is masked
> out exactly as the existing deferred-slot slack is. This is the same invariant
> `PoolingCache` relies on for its own 256-chunked storage.

**Half 1 — the `_visible_width` clamp does NOT mask the pad. It is a no-op for this case.**

```
1904|            pool_bytes = self.pooled.size * self.pooled.dtype.size
1905|            visible = (
1906|                None if pool_bytes > _POOL_DEFER_COPY_MAX_BYTES else self.pooled
1907|            )
...
1916|        if visible is None:
1918|            mx.async_eval(self.pooled)
1919|            visible = self.pooled
1920|        self._visible_width = visible.shape[1]
1921|        return visible

2170|        B, P, _ = self.pooled.shape
2175|        if self._visible_width is not None:
2176|            P = min(P, self._visible_width)
```

In **both** branches `visible` is `self.pooled` **in full** — including any trailing pad.
So `_visible_width == self.pooled.shape[1] == P`, and `min(P, self._visible_width)` returns
`P` unchanged. The clamp cannot remove a pad, because the pad is inside the tensor whose
width defines `_visible_width`. (The clamp's real job, per its own comment at 2171-2174 and
the rollback paths at 2442-2444 / 2528-2530, is the case where storage was *sliced back*
below the returned width — a different scenario.)

What actually masks the pad is the **length mask**, two lines further on:

```
2177|        pool_lengths = mx.array(self._pool_lengths)
2180|        pool_idx = mx.arange(P)[None, None, :]
2181|        valid = pool_idx < pool_lengths[:, None, None]
2185|            if all(pl == P for pl in self._pool_lengths):
2186|                return None
2187|            return valid
```

Padded columns have `pool_idx >= pool_lengths` → `valid == False` → they are masked. So the
**invariant that makes the A/B safe is `_pool_lengths`, not `_visible_width`.** Same
end state, completely different mechanism — and the difference matters, because it changes
which code you must not break when writing the patch.

**Half 2 — `PoolingCache` does NOT rely on this invariant at all.** It cannot: it has no
`_visible_width` (grep: every one of the 11 occurrences — 1839, 1920, 1944, 2163, 2175,
2176, 2440, 2444, 2479, 2481, 2530 — is inside `BatchPoolingCache`, lines 1826-2729;
zero inside `PoolingCache`, 1270-1825). `PoolingCache` hides its slack **structurally**:
its `pooled` property (1360-1364) returns `self._pool_storage[:, :self._pool_offset]`, so
the 256-chunk slack is *never visible to any consumer in the first place* and needs no
masking. That is the opposite of the C3 patch's shape, where the pad IS handed to the model.

**Does the patch still work?** I believe yes, but by a route C3 never checked — and it has
one live consequence C3 does not mention:

- Today at steady state, `all(pl == P)` holds and `make_mask` returns **`None`** (2185-2186).
  With the pool padded, `pl != P`, so it returns the **`valid` array** instead. Decode
  therefore switches from a no-mask path to a masked path on every step.
- Traced to the consumer: `Indexer.__call__` takes `deepseek_v4.py:3840 if pmask is not None`,
  and since this `valid` is 3-D, `_tail_ok` (which requires `pmask.ndim == 2`, :3858) is
  **False** — so it takes the full `mx.where` at :3883. At decode `L_q=1` that where() is
  over a `(1,1,P)` tensor, so the cost is O(P) not O(L·P) — small, but **not zero, and not
  today's path**.
- Selection stays correct: masked scores get `mx.finfo.min` (:3886), and `k = min(512, P)`
  (:3888) with ~25K-88K real entries ≫ 512, so pads can never enter the top-k.
- The dense-concat branch (`deepseek_v4.py:4614`, `pooled.shape[1] <= index_topk`) is not
  reachable at these depths, so a padded pool is never concatenated wholesale into SDPA.

**Verdict: REFUTED as stated / conclusion salvageable.** The A/B is not a correctness bug
on my reading, but C3's justification must be replaced before the PM forwards it, and the
mask-path switch (None → `valid`) should be called out as a small confound in the A/B
itself: arm B changes *two* things (no concat, plus a masked indexer path), not one.
**Recommended precondition for the A/B: assert `k < min(_pool_lengths)` holds at both
depths, and note that a slice-off-the-pad-before-return variant would avoid the mask-path
change entirely.**

---

## C3-5 — arithmetic — **CONFIRMED** (with one internal inconsistency worth flagging)

All recomputed from C3's own tables:

| check | C3 says | recomputed | verdict |
|---|---|---|---|
| concat cost @100K = (a)−(d) | +0.538 | 0.557 − 0.019 = **0.538** | exact |
| concat cost @352.6K = (a)−(d) | +2.449 | 2.504 − 0.055 = **2.449** | exact |
| depth delta | +1.911 | 2.449 − 0.538 = **1.911** | exact |
| median-of-3 cross-form | +1.911 | 1.947 − 0.036 = **1.911** | exact (agrees to 0.000, not 0.04 as the doc modestly claims) |
| C 2.56 + C3 1.911 | +4.47 | **4.471** | exact |
| C 3.34 + C3 1.911 | (range top) | **5.251** | matches the task's +5.25 |
| explained vs live 6.80 | ~66% | **65.8%** (4.471/6.80) … **77.2%** (5.251/6.80) | confirmed |
| residual | ~1.5-2.3 | **2.329** … **1.549** | confirmed |
| (a)≈(c) @100K | within 0.011 | \|0.557−0.553\| = **0.004** | confirmed |
| (a)≈(c) @352.6K | within 0.011 | \|2.504−2.493\| = **0.011** | confirmed |

**Internal inconsistency (cosmetic, no verdict changes)**: C3 uses **two different values
of the same quantity in two different sections**. §5.2's table uses **+1.95** (the
median-of-3 flush-excess delta 1.947) giving sum **+4.51** and residual **+2.29**; §6's
verdict uses **+1.91** (the (a)−(d) delta) giving **+4.47**. Both are defensible, they
differ by 0.04, but the doc never says it switched. The PERFORMANCE_HISTORY entry uses
+1.91 / +4.47 — the tighter and better-derived pair. **PM should quote +1.91 / +4.47 / 66%
and drop the +4.51 figure**, or state the two derivations explicitly.

**mod-4 periodicity, independently checked from C3's own raw series** (§3.1):
`[41.73, 41.607, 41.612, 51.106, 42.059, 41.563, 41.74, 51.352, ...]` — indices 3, 7, 11,
15, 19, 23 are the six spikes in 24 steps; every one is ≡3 mod 4 and every non-spike is
41.5-42.3. Perfectly clean, matches the stated `flush phase=3` and the +10.014 excess
(51.684 − 41.670 = **10.014**, exact). **Consistent.**

**Allocator-stat check** (§3.7): claimed compressor pool 88,149 × 512 × 2 B = **90.27 MB**;
indexer 88,149 × 128 × 2 B = **22.57 MB**; total **112.8 MB** vs 107.09 MB observed.
Arithmetic exact, and the doc's explanation (allocator reuses part of the freed block) is
a reasonable account of the 5% shortfall. The (d) drop to 10.06 MB is a **10.6×** reduction
(107.09/10.06 = 10.65). **Consistent.** The negative control C3 flags — (b) shows 107.1 MB,
*not more* than (a) — is a genuine and well-reasoned consistency check.

---

## C3-6 — the "C would have measured ~+0.48 ms/step higher / ~+10 ms/token over 43 layers" sanity check — **REFUTED (both numbers wrong); verdict unaffected**

C3 §5.1 writes:

> C's r=4 layer at 352.6K was 0.5403 ms/step … This worker measures the flush excess alone
> at +10.0 ms/flush-step (+2.50 amortized). If C's bench had been paying the production
> concat every flush, C's r=4 number would have been ~+0.48 ms/step higher — a ~+10 ms/token
> difference in the 43-layer total

Recomputed. C3's +10.014 ms/flush-step is the **whole-step** excess, i.e. **all 21 sparse
layers together** — C3's own §2.2 says exactly that ("isolates the entire pool-write cost
of all 21 sparse layers in one step").

- per sparse layer per flush-step: 10.014 / 21 = **0.4769 ms** ← this is where "+0.48" came from
- but C's 0.5403 is a **per-step average over 256 steps**, of which only 64 are flushes.
  The correct like-for-like adder is 0.4769 × 64/256 = **+0.1192 ms/step**, not +0.48.
- **"43 layers" is the wrong multiplier.** The concat only exists on layers that carry a
  pool flush at ratio 4 — the **21** `SparseCompressedAttention` layers (r=128 layers flush
  1-in-128 and were separately measured flat). C3 itself uses 21 everywhere else (§2.2, §5.1
  first sentence "11.35 ms/token across 21 layers"). The 43 is a slip.
- Correct per-token figure: 0.1192 × 21 = **+2.503 ms/token** — which, reassuringly, is
  *exactly* C3's own independently-derived +2.449/+2.504 concat cost at that depth. The
  doc's "~+10 ms/token" is **4.1× too large**, arising from combining the un-amortized
  per-flush figure with the 43-layer count.

**Does this change any verdict? No — it strengthens the disjointness argument.** The claim
being supported is "C did not pay the production concat, because if it had, C's number
would have been visibly bigger." The correct magnitude is +2.50 ms/token on C's 43-layer
total of 19.13 — a **13% inflation**, far outside C's ±2% fit residuals and its ≤0.02
run-to-run spread. So C demonstrably did not pay it, and **C3's disjointness conclusion
holds on corrected arithmetic**. The +0.48/+10 numbers must not be quoted.

Cross-check of the same claim from the other direction (independent, and it holds): C's
`PoolingCache` amortizes 1 growth per 256 flushes; C3's ~1/64 estimate of the leakage is
0.04 ms/token at 352.6K, negligible vs +1.91. Both routes agree C's number is concat-free.

---

## C-1 — C's headline table — **CONFIRMED**

Every figure in the task's C-1 checklist matches C's §0 and its pasted raw output at §3
(the doc contains its own bench stdout, so the table and the raw are cross-checkable):

- 16.568 @100,026 / 19.130 @352,599 / 21.520 @500,000 / 12.876 @520 — table §0 == raw §3 == §5 SCALING block. **Consistent, three places.**
- Δ = 19.130 − 16.568 = **+2.562** ✓ (doc says +2.56)
- alternate fencing runs +2.962 and +3.344 ✓ (15.178→18.140; 15.039→18.383 — both recompute exactly)
- linear fit above 100K `15.2182 + 1.2139/100K` with residuals **+0.82% / −1.92% / +1.08%** — all within ±2% ✓, and the (+,−,+) sign pattern the doc reads as faint concavity is really there.
- component ×layer scalings spot-checked: indexer.score 0.0473 × 21 = 0.993 ✓ (doc 0.993); sdpa compress 0.0776 × 20 = 1.552 ✓; Δ indexer (0.0473−0.0280)×21 = **+0.405** ✓.
- 43-layer totals recompute from the per-class medians: 0.2715×2 + 0.5403×21 + 0.3620×20 = 0.543 + 11.346 + 7.240 = **19.129** ✓ (doc 19.130).

---

## C-2 — the 520→100K overshoot — **CONFIRMED as a real gap, and C's doc does NOT address it. FLAG FOR THE PM.**

- C's kernel delta over 520→100,026: 16.568 − 12.876 = **+3.692 ms** (C's own table, "+3.710 per 100K").
- B1's **live end-to-end** delta over the same span: 35.79 − 33.75 = **+2.04 ms** (B1 §3, "+2.05 ms/100K").

**A component of the budget grows +3.69 ms while the whole budget grows +2.04 ms.** That is
structurally impossible unless something else *shrinks* by ≥1.65 ms over the same span, or
the bench's absolute deltas are inflated on that segment.

Searched C's doc for any acknowledgement:

```
$ grep -n "exceeds the live\|larger than the live\|more than the live\|3.71\|3.710\|+2.04\|2.05 ms" \
    docs/p3-worker-c-attn-kernel-walltime-2026-08-23.md
40:| 100,026 | **16.568** | +3.692 | +3.710 |
282:   100,026       16.568          3.692        3.710
439:| 520 → 100,026 | **+3.710** |
444:+2.05 ms/100K over the first 100K and +2.69 ms/100K over the next 253K — costs
```

Line 443-451 is the only place both numbers appear, and it compares them **only in shape**
("the attention kernels do the reverse … a large fixed step from 520→100K"), attributing the
step to fixed kernel-launch/first-pool-allocation effects and the ≈2048 dense→sparse branch
switch. **It never notices that its own +3.69 exceeds the live +2.04 on that span**, and its
§6 sanity check deliberately looks only at the 100K/352.6K points, where the share (45-46%)
is comfortable. §7's limitations do not mention it either.

**Why the PM must carry it**: C's absolute deltas are validated at the *level* (12.88 ms vs
A's 13.1 ms constant term, ~2%) but this shows at least one *segment delta* overshoots
reality by ~1.65 ms. That bounds how literally C's +2.56 on the deep span can be read —
plausibly it too carries a fixed-overhead component that production overlaps away. It does
**not** invalidate the C+C3 additivity argument (which is a ratio/disjointness argument,
not an absolute one), but "explained ~66%" should be quoted as an estimate with a
one-sided error bar, not a measurement. C3's own §5.2 caveat about C's absolute total being
understated is arguing the *opposite* direction; the PM should present both.

---

## D-1 — the rank-label correction is evidence-backed — **CONFIRMED**

D §0 quotes each node's own init line, which is the right primary source:

```
m4-1 exo.log 11:11:00.778  ... mlx_distributed_init:143 ] Starting initialization for rank 1
m4-2 exo.log 11:11:00.759  ... mlx_distributed_init:143 ] Starting initialization for rank 0
```

Plus three independent corroborations, all named and each independently discriminating:
jaccl coordinator roles (rank 0 **binds** `0.0.0.0:57547` on m4-2; rank 1 **dials**
`192.168.200.2:57547` from m4-1), the `[jaccl] tcp coord group rank=N` markers in each
node's runner stderr, and the PID carrying the Metal error (46718, and the §1 traceback is
quoted verbatim from m4-1's log). D also correctly propagates the consequence (C2's
`100k_rank0`/`100k_rank1` occupancy blocks are swapped) **and** correctly bounds it (they
agree to 0.08 pp, so no C2 conclusion changes). This is exactly the right shape for a
correction. **Confirmed — evidence quoted, not asserted.**

**One unresolved cross-doc conflict the PM must settle** (not D's error): worker **C** says
its bench ran on "`adams-mac-studio-m4-2.local` (**rank1**, production silicon)" (C §0
line 17), and **C3** likewise reads env "off the running **rank-1** process command line on
`adams-mac-studio-m4-2`" (C3 §1.1). D proves that during the 11:10:58 launch **m4-2 was
rank 0**. Since C, C2, C3 and D all ran against that same launch, **C's and C3's "rank1 on
m4-2" labels are wrong the same way C2's were** — m4-2 was rank0. This is label-only: both
ranks replicate attention (worker A / R1), C3 explicitly notes single-rank/B=1, and neither
doc's numbers depend on which rank the silicon was. **No number changes; the PM should fix
the labels in the synthesis rather than let a third doc carry the swap forward.**

---

## D-2 — the headline timeline is internally consistent — **CONFIRMED**

Every element cross-checks against D's own quoted excerpts, and the ordering is coherent:

| claim | D's own evidence | consistent? |
|---|---|---|
| 13:51:43.416 Metal error | §3.1 `13:51:43.4158 python3.13[46718] (Metal) … Caused GPU Timeout Error`; §1 traceback logged 13:51:43.489 (73 ms later — exception surfaces after the driver error) | yes |
| 13:51:48 kernel "2 GPURestarts in 398 submissions" | §3.1 `13:51:48.0435 kernel[0] (IOGPUFamily) … Deny submissions/ignore app[] with 2 GPURestarts in 398 submissions` | yes, verbatim |
| 13:52:04 reaped | §2 `13:52:04.34 runner exits signal 15 after "7 attempts :)"`, and §1's note that the error repeats 269× explains why 7 attempts were needed | yes, and self-explaining |
| 13:52:06 peer SIGKILL by hang-watchdog | §2 `13:52:06.17 m4-2 hang-watchdog: rank0 runner no event for 47s (>45s), SIGKILL`; §3.5 corroborates m4-2 blocked in `wait_for_one` | yes |

The 47 s watchdog interval is itself consistent: m4-2's last healthy step was 13:51:13.975
(the 85.10 ms one), and 13:52:06.17 − 13:51:13.975 ≈ **52 s**, comfortably past the 45 s
threshold with the event gap D reports. The memory chain (jetsam 13:51:12.35 → first
swapfile 13:51:15 → 26 swapfiles ending 13:51:43, "the last one in the same second as the
GPU timeout") is monotone and tightly ordered. **No internal contradiction found.**

The swapfile-per-minute baseline (§3.3: `03:16 ×24 | 03:17 ×27 | … | 13:51 ×27`, every burst
coinciding with a tracer window) is the strongest single piece of evidence in the doc and it
is presented as such. Good.

---

## D-3 — does D's verdict overclaim? — **CONFIRMED: it does not. It separates "best-supported" from "proven" explicitly.**

D §4 ranks five mechanisms with an explicit confidence label on each: #1 *"best supported"*,
#2 *"plausible contributor, **not separable**"*, #3 *"refuted"*, #4 *"weak / not the driver"*,
#5 *"ruled out"*. §4's "Honest residual" states the limit in the doc's own words —
*"Insufficient telemetry to distinguish 1-final-step vs 2"* — and refuses to pick
(*"I am not going to pick between these"*).

More importantly, §5 does the harder thing: having established a tracing-risk verdict, it
argues **against its own convenient conclusion**, keeping the production-relevant caveat
alive (90.3 GB resident / 115.3 GB peak of 137 GB; *"nothing in this incident proves a
500K-context untraced decode can't reach the same jetsam/swap regime on its own"*) and
giving the PM an explicit bottom-line phrasing that forbids the overclaim in **both**
directions. §8 lists n=1, retrospective-only, depth-inferred, no-reproduction, and the fact
that the 90.3/115.3 GB numbers are **m4-2's** (the peer), not the crash node's — a limitation
that cuts against its own headline and which a less careful doc would have buried.

**No overclaim found. This is the most epistemically disciplined of the three docs reviewed.**

---

## FILE INTEGRITY — `docs/PERFORMANCE_HISTORY.md`

```
$ grep -c "<<<<<<<\|>>>>>>>" docs/PERFORMANCE_HISTORY.md      -> 0
$ wc -l docs/PERFORMANCE_HISTORY.md                            -> 2699
$ git status --porcelain                                       -> " M docs/PERFORMANCE_HISTORY.md" (uncommitted, as intended)
```

**All seven P3 entries present, in landing order, contiguous, none duplicated or truncated:**

| # | entry | line range | ends with source-doc ref? |
|---|---|---|---|
| 1 | P3 worker A | **2343-2383** | yes, `p3-worker-a-kv-read-inventory-2026-08-23.md` |
| 2 | P3 worker B1 | **2385-2424** | yes, `p3-worker-b1-live-depth-anchors-2026-08-23.md` |
| 3 | P3 reviewer R1 | **2426-2456** | yes, `p3-reviewer-r1-verification-2026-08-23.md` |
| 4 | P3 worker C | **2458-2506** | yes, `p3-worker-c-attn-kernel-walltime-2026-08-23.md` |
| 5 | P3 worker C2 | **2508-2553** | yes, `p3-worker-c2-depth-busy-idle-capture-2026-08-23.md` |
| 6 | P3 worker D | **2555-2609** | yes, `p3-worker-d-metal-timeout-crash-forensics-2026-08-23.md` |
| 7 | P3 worker C3 | **2611-2666** | yes, `p3-worker-c3-donation-failure-insitu-2026-08-23.md` |

Order is chronological by landing (A, B1, R1, C, C2, D, C3) and matches the docs' mtimes
(13:01 / 13:09 / 13:22 / 13:30 / 14:31 / 15:02 / 15:32). Each entry is a single `**NEW
(2026-08-23, P3 …)**` block separated by one blank line; no entry is nested inside another;
the region terminates cleanly at the `---` on 2668 followed by the "Quick-reference" table.
The `xctrace` C2 hazard row was correctly added to that closing table (line 2699).

**Drift check — every headline number in each entry re-read against its source doc:**

| entry | numbers checked | drift |
|---|---|---|
| C (2458-2506) | 12.88/16.57/19.13/21.52; +2.56 (+2.96/+3.34); 38-49%; ~3.5-4.2; fit 15.22+1.21/100K ±2%; +3.71→+1.01→+1.62; +0.405/+0.088/+0.261; 477/558 vs 405 GB/s; 94% indexer; 0.197 ms floor; 0.461→0.494→0.555; +6.35 upper bound | **none** |
| C2 (2508-2553) | 82.98/83.06%; 6.09/6.06 idle; ≤0.142/all_sum; 38.02 traced vs 35.79; 50.55 s; 214,182/213,816 | none. **Note: entry preserves C2's original rank labels**, which D §0 corrects. Entry does not carry D's correction, but D's own entry (2562-2563) states it explicitly and says no C2 conclusion changes — acceptable, though a one-line pointer in C2's entry would be cleaner. |
| D (2555-2609) | 46718/m4-1/rank1; 13:51:43.416; 2 GPURestarts in 398 submissions; ≥25 daemons; 23.6 GB compressor; 26 swapfiles/18 s; 7-day clean; 90.3/115.3 of 137 GB; 85.10 ms step; 133.5 of 137.4 GB free | **none** |
| C3 (2611-2666) | +0.557/+2.504 (a); +0.553/+2.493 (c); +1.850/+3.437 (b); +0.019/+0.055 (d); +1.00/+1.07 (e); 0.011; 98%; p90 51.72→42.11; 107.1→10.1 MB (10.6×); +0.538/+2.449/+1.91; +4.47; 66%; ~1.5-2.3; 42.59→~40.14; 23.48→~24.91 (+6.1%); +6.80→~+4.89 | **none vs the source doc.** The entry correctly uses the tighter +1.91/+4.47 pair rather than §5.2's +1.95/+4.51 (see C3-5). |

**However — two claims that are drift-free but inherit errors from their source docs, and
therefore need correcting in the file as well as in the synthesis:**

1. Line **2648-2649**: *"had C paid it, C's r=4 layer would have been ~+0.48 ms/step higher,
   which C did not observe"* — inherits C3-6's error. Correct value **+0.12 ms/step**
   (+2.50 ms/token over 21 layers, a 13% inflation of C's total). The argument survives; the
   number does not.
2. Line **2656-2657**: *"Safe by the existing `_visible_width` mask clamp
   (`cache.py:2174-2176`)"* — inherits C3-4. The clamp is a no-op for a trailing pad; safety
   comes from the `_pool_lengths` length mask at `cache.py:2177-2181`. **This is the one I'd
   fix first**, because it is a correctness rationale attached to a patch someone will write.

Per this task's read-only rule I have **not** edited those lines — reporting only.

---

## Summary table

| claim | verdict |
|---|---|
| C3-1 production routes caches through `_merge_caches` → `BatchPoolingCache` | **CONFIRMED** |
| C3-2 concat-to-exact-max_pool every flush vs 256-chunk 1-in-256 | **CONFIRMED** |
| C3-3 no cross-step pool-reference holder (4 cites) | **CONFIRMED** |
| C3-4 A/B correctness precondition (`_visible_width` clamp) | **REFUTED as stated** — clamp is a no-op; real invariant is `_pool_lengths`; `PoolingCache` uses a structural view, not this invariant. Patch still looks safe; rationale must be rewritten; mask-path switch is an unflagged A/B confound. |
| C3-5 arithmetic (+0.538/+2.449/+1.91; 66%; (a)≈(c)) | **CONFIRMED** — plus an unflagged +1.95-vs-+1.91 switch between §5.2 and §6 |
| C3-6 "+0.48 ms/step / ~+10 ms/token over 43 layers" | **REFUTED** — should be +0.12 ms/step, ×**21** layers, +2.50 ms/token. Conclusion unaffected. |
| C-1 headline table, deltas, fit | **CONFIRMED** (recomputed from C's own raw output in three places) |
| C-2 520→100K overshoot (+3.69 kernel vs +2.04 live) | **CONFIRMED as real; UNADDRESSED by C's doc — PM must carry it as a caveat** |
| D-1 rank-label correction supported by quoted logs | **CONFIRMED** (+ C and C3 carry the same swap — label-only) |
| D-2 timeline internally consistent | **CONFIRMED** |
| D-3 verdict does not overclaim | **CONFIRMED** |
| PERFORMANCE_HISTORY integrity (7 entries, order, no markers, no drift) | **CONFIRMED** (2 inherited errors flagged above) |

## What the PM must change in the synthesis

1. **Do not repeat C3 §8.1's correctness note.** Replace with: the pad is masked by
   `_pool_lengths` (`cache.py:2177-2181`), not by the `_visible_width` clamp; `PoolingCache`
   hides its slack structurally via the `pooled` property (1360-1364) and relies on no such
   invariant. Add the A/B confound: padding flips `make_mask` from `None` to a `valid` array,
   changing the indexer's mask path (`deepseek_v4.py:3840/3883`) — so arm B changes two things.
2. **Drop "+0.48 ms/step" and "~+10 ms/token over 43 layers."** Use +0.12 ms/step / **21**
   layers / +2.50 ms/token (13% inflation of C's 19.13). The disjointness conclusion stands.
3. **Carry the 520→100K overshoot as an explicit caveat** on C's absolute deltas; quote
   "explained ~66%" as an estimate with a one-sided error bar, not a measurement.
4. **Quote +1.91 / +4.47 / 66%**, not §5.2's +1.95 / +4.51 / residual +2.29, or state both.
5. **Fix the rank labels**: m4-2 was **rank 0** this launch (D §0). C and C3 both say "rank1
   on m4-2." Numbers unaffected; don't propagate the swap a third time.

**Scope**: read-only except the PERFORMANCE_HISTORY append. Nothing fixed, nothing committed,
no cluster contact.
