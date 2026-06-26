# B=2 Concurrent Quality + MTP — Resolution

**Date:** 2026-06-24
**Status:** RESOLVED. Two root-cause bugs fixed, both verified across a
B=2 100K→500K needle sweep with MTP on and the c≥2 gate removed.

---

## TL;DR

B=2 concurrent inference on DSv4-Flash had two independent quality bugs,
both surfacing only at B≥2 (concurrent streams) and high context:

1. **Seq-split all_gather reconstruction** scrambled which stream's
   query-row bands landed in which stream → one stream decoded garbage
   (`' CAP'`) during prefill. Fixed.
2. **MTP bootstrap offset rebase** rebased the absolute position
   (138696) down to the ring cursor (~131) at the first MTP-on decode
   step → RoPE/pool position mismatch → degenerate output (newline /
   `' the'` loop) from the first decode step. Fixed.

Both are the same bug class — **batch-unsafe code that worked at B=1 but
broke at B≥2** — joining the already-removed `FUSED_MOE`/`COMPILE_FFN`/
`COMPILE_LAYER` opts (see `auto_parallel.py:107`).

After both fixes, B=2 concurrent needle-in-haystack passes clean at
100K/200K/300K/500K with MTP on and the c≥2 gate off. MTP-on is now the
default at c≥2 high context.

---

## Bug 1: Seq-split all_gather (prefill)

**Root cause:** `CompressedAttention` and `SparseCompressedAttention`
(deepseek_v4.py:2299 and :2674) split the query axis across TP ranks
(OPT-3 seq-split) and reconstruct via:
```python
out = mx.distributed.all_gather(out, group=_sg).reshape(_B, L, _H)
```
`all_gather` of a `(B, band, H)` tensor produces `(N*B, band, H)` in
**rank-major** memory order: `[r0s0, r0s1, r1s0, r1s1, ...]`. The
`reshape(B, L, H)` is **row-major** and interprets that as
`[s0_band0, s0_band1, s1_band0, ...]`. At B>1 these orderings don't
match → stream 0 gets rank 0's partial of *both* streams, never rank 1's
contribution → garbage. At B=1 they coincide (the comment "rank == row
order" was only true at B=1).

**Fix** (mlx-lm `8a9cdee`):
```python
_g = mx.distributed.all_gather(out, group=_sg)      # (N*B, band, H) rank-major
out = (_g.reshape(_N, _B, _band, _H)                 # (N, B, band, H)
        .transpose(1, 0, 2, 3)                        # (B, N, band, H)
        .reshape(_B, L, _H))                          # (B, L, H)
```
Rebuilds each stream's L axis from its own per-rank bands. Bit-exact at
B=1 (N=1 → transpose/reshape is identity).

**Verified:** B=2 100K/200K/300K/500K all `all_needles=True`,
367/353/340/318 t/s aggregate prefill (beats the bugged baseline's
298-317 at 353K).

---

## Bug 2: MTP bootstrap offset rebase (decode)

**Root cause:** `PerStreamBatchRotatingKVCache._bootstrap_per_stream_ring`
(cache.py:2681) read:
```python
abs_off = int(self._offset)   # ring write cursor (~131)
```
but `self._offset` is the **ring write cursor** (capped at ~max_size),
not the **logical position** (138696, in the `self.offset` tensor). The
bootstrap then set `self.offset = mx.full((B,), abs_off)` = 131,
**discarding the true 138696 position**. RoPE and the rotating mask used
position 131 while the (unrebased) `BatchPoolingCache` held entries at
the 138696 scale → position mismatch → degenerate output (newline /
`' the'` period-1 loop) from the **first decode step** after the MTP-on
cache swap.

The bootstrap comment even said "Keep absolute offsets — RoPE needs the
true position" — violated by reading `self._offset`.

**Found via ground-truth diag** (`EXO_DSV4_SWAP_DIAG`): before swap
`offset=[138696,138696] _offset=131`; after the old bootstrap
`offset=[131,131]`. The rebase was directly visible.

**Fix** (mlx-lm `48a4a3c`):
```python
abs_off = int(self.offset[0])   # logical position, not ring cursor
```

**Verified:** MTP-on, c≥2 gate off, B=2 200K/300K/500K all
`all_needles=True`, no degeneration (was: both streams empty, 0 decoded
tokens, `DEGENERATION DETECTED` at token 8).

---

## Gate removal

The `EXO_DSV4_MTP_C2_MAX_CTX` gate (dsv4_mtp.py:~1620) that disabled
spec at c≥2 high context was a mitigation for Bug 2 before the root
cause was found. With Bug 2 fixed, the gate is no longer needed.
**Default flipped:** `EXO_DSV4_MTP_C2_MAX_CTX` default is now `0` =
"no gate" (was `0` = "always disable spec for c≥2"). The threshold
mechanism is kept as an opt-in safety net: set a real threshold (e.g.
`200000`) to re-enable the gate if a future regression surfaces, without
a code change.

MTP-on is now the default at c≥2 high context.

---

## Commit map (all on adurham fork mains)

### adurham/mlx-lm
| Commit | What |
|--------|------|
| `8a9cdee` | Bug 1: seq-split all_gather batch-safe at B>1 |
| `48a4a3c` | Bug 2: bootstrap abs_off from logical offset, not ring cursor |

### adurham/exo
| Commit | What |
|--------|------|
| `8317ba84` | Bug 1 gitlink bump |
| `3fdb6ac0` | Bug 2 gitlink bump |
| `47fdf32a` | Cleanup: remove diag instrumentation + disable c≥2 gate |

### Diagnostic commits (since reverted by the cleanup)
| Commit | What |
|--------|------|
| `16ce4acd` | diag: verify-logit dump in spec trace (EXO_DSV4_SPEC_LOGIT_DUMP) |
| `37263e68` | diag: forward EXO_DSV4_SPEC_LOGIT_DUMP to runner env |
| `c1949249` | diag: swap-diag dump (EXO_DSV4_SWAP_DIAG) — found Bug 2 |

---

## Lessons (process, for next time)

Two failed fixes were shipped before the ground-truth diag found Bug 2:
- **Pool rollback theory** (facts 715-717): theorized the all-or-nothing
  `BatchPoolingCache.restore_meta` wiped accepted tokens. Disproven by
  A/B (and by the trace showing the corruption was in the *forward*, not
  the rollback).
- **Bootstrap slice-direction theory** (`096515f`): theorized the buffer
  was ~255 wide with newest at the tail and `[:valid]` grabbed the
  oldest. The diag showed the buffer was exactly 128 wide (head==tail),
  so the fix was a literal no-op. Disproven by A/B.

Both were plausible-sounding theories committed without direct evidence.
The `EXO_DSV4_SWAP_DIAG` dump is what actually pinned Bug 2 — it showed
the offset rebase (138696→131) directly. **Lesson: instrument for
ground truth before theorizing.** A dump of the actual cache state
beats reasoning about buffer layouts you can't see.

Also: the `bistab=True` / per-stream-tps asymmetry in the probe output
is a **measurement artifact** of `decode_tps = generation_tokens /
(total_s - ttft)` with `--max-tokens 256` over a huge prefill TTFT —
not a real compute stall. Don't quote those decode t/s numbers as real
throughput.