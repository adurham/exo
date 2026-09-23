# Ratchet is DEPTH-specific, not leaf-related — and the prefix cache is working perfectly (2026-09-23)

## Discriminator result

Two arms at 80-90K depth, watching the `[MEM] before prefill` baseline:

**ARM 1 — cache MISS (fresh uuid → new leaf each request):**
| run | ctx | cached | before-prefill active |
|---|---|---|---|
| miss 0 | 88,746 | 0 | 102.12 GB |
| miss 1 | 87,848 | 0 | 100.69 GB |
| miss 2 | 88,746 | 0 | 98.98 GB |

**ARM 2 — cache HIT (identical prompt → same leaf):**
| run | ctx | cached | before-prefill active |
|---|---|---|---|
| hit 0 | 78,878 | 0 | 96.08 GB |
| hit 1 | 78,879 | **78,877** | 93.40 GB |
| hit 2 | 78,879 | **78,877** | 93.92 GB |

## Conclusions

### 1. The ratchet is NOT caused by cache-busting / leaf accumulation

Neither arm ratchets at 80-90K. Both decline slightly and settle (~93-102 GB).
So generating a new leaf per request does **not** by itself produce the climb.
My hypothesis that tonight's probes overstated the problem via cache-busting is
**not supported at this depth**.

The ratchet therefore appears to be **depth-specific**: it was observed only
across ~275K+ requests (90.67 → 105.26 GB, +15 GB over six runs), and does not
occur at 80-90K.

### 2. The prefix cache is working excellently

`cached = 78,877 of 78,879` tokens (99.997%). A repeated long prompt becomes a
**1-token** prefill instead of a 79K prefill. That is the difference between a
multi-minute prefill and a ~second-scale one.

Practical consequence: for a **continuing conversation** (the user's real
workload), each turn re-uses the prior context rather than re-prefilling it.
The expensive cold-prefill path is the exception, not the rule.

### 3. What remains unexplained

- **Why the ratchet only bites at ~275K+.** Candidates:
  (a) the per-leaf snapshot set (`EXO_LEAF_SNAPSHOT_RETENTION`) — but measured
      `[SNAPMEM]` at only 0.17 GB, so unlikely alone;
  (b) the "prefill working-set" leak documented as real-and-unfixed in fact 778
      (activation/intermediate buffers not freed between successive large
      prefills on a growing leaf);
  (c) KV growth itself at 275K+ (~5.5 GB/request at the measured
      ~0.02 MB/token) combined with the Metal allocator not promptly
      reclaiming evicted leaves (fact 772).
- The +3-4.5 GB *per deep request* deltas are larger than KV growth alone,
  which is why (b)/(c) are the leading candidates.

## Honest status of the night's two targets

- **T2 (250K ≥ 30 t/s):** measured mean 26.18 (n=6) with 1 collapse in 6.
  Unchanged conclusion: typically 26-29, with an intermittent ~19.
- **T1 (500K no collapse):** real, stochastic (13.36 vs 24.15).
- **Both are the same phenomenon**, and it is now localised to **~275K+**.
- Below ~120K the system is stable AND the prefix cache makes repeat turns
  cheap — so the user's routine workflow is in better shape than the raw
  depth numbers suggest.

## What I did NOT do
- No config changes. No relaunch. No wired-limit change.
- Rationale unchanged: every cheap footprint lever measured tonight is ~17x
  too small (0.29 GB vs ~5 GB needed), and the remaining levers (wired-limit
  raise, head quantization) carry documented risk that shouldn't be taken
  unilaterally overnight.

## Next (for the user, with numbers to decide on)
1. Decide whether ≥250K context is a real production requirement. At ≤120K the
   cluster is stable, at target throughput, with a working prefix cache.
2. If ≥250K matters: the ratchet needs a code-level fix in the prefill
   working-set / allocator-reclaim path (fact 778 territory), not a knob.
   That is a scoped engineering task, not a config change.
3. Establish the collapse RATE (N≥10 at 275K+) so any fix can be shown to work.
