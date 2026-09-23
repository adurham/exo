# VERIFICATION at 565-619K: the accounting fix is correct but NOT sufficient (2026-09-23)

## The measurement, fix live

`verify_565k.py`, 565,000 requested depth, byte-cap fix deployed (`89ebdbff0`)
and confirmed live (`maxPrefixBytes: 12884901888` in `/state`).

| metric | pre-fix reference | run 0 WITH fix |
|---|---|---|
| decode tps | 13.36 - 24.15 (stochastic) | **13.12** |
| page-ins in request | 6,599 | **66,003** |
| peak memory | 117.24 GB | **117.88 GB** |
| before-prefill active | — | 107.32 GB |
| wired limit | 115.4 GB | 115.4 GB |

**The collapse still happens.** 13.12 t/s is inside the pre-fix collapse band,
and refaulting is *worse* (66,003 page-ins vs 6,599). Peak (117.88 GB) still
exceeds the wired limit (115.4 GB) by ~2.5 GB.

## Why the byte cap did not fire (and why that is CORRECT, not a bug)

Resident leaves before the 619K add: 383,838 / 364,787 / 368,591 / 619,462 tok.

`_evict_if_needed` order is (1) memory pressure, (2) session cap, (3) byte cap —
and it runs **before** the new leaf is inserted. So:

- session cap (`reserve_slot`) evicts 4 -> 3 first
- the 3 kept leaves happen to be ~370K: 1,117,216 tok = **10.70 GiB**
- byte cap (12 GiB): 10.70 < 12.00 -> **silent, correctly**

The cap stayed silent because the co-resident leaves were ~370K, not because the
accounting is broken. Verified directly: exo's accounting now matches `.nbytes`
exactly per layer on real DSv4 shapes.

**Consequence: the "fires" direction of the fix remains UNVALIDATED on hardware.**
It requires three >=565K leaves resident simultaneously (3 x 5.41 = 16.23 GiB >
12 GiB). Every deep sweep so far has mixed ~370K and one deep leaf.

## What this means for the user's ask (a): stop the 500K collapse

**Not solved.** Honest decomposition:

- Retained leaves at this moment: 4 leaves = 16.63 GiB. The byte cap would have
  trimmed SOME of that if 3+ leaves were each deep, but even removing a full
  5.4 GiB leaf leaves the peak near 112 GB — still pressed against a 115.4 GB
  limit with ~104 GB of fixed footprint.
- The dominant term is NOT the prefix cache at all. The ~104 GB baseline is
  weights (~77.5 GB/node measured from safetensors headers) + MLX runtime + MTP
  head + activation buffers. At 565K the prefill working set pushes total to
  117.88 GB.
- So the collapse is fundamentally **a total-footprint-vs-wired-limit problem**,
  and the prefix cache is a ~16 GiB contributor to a ~118 GB total.

## The two real levers this implies (neither a mitigation)

1. **Raise the wired limit** from 115000 MB toward 124000 MB. The 124000 value
   was deliberately lowered on 2026-06-29 to prevent a Metal-allocator wedge, but
   that wedge's trigger was specifically **DSv4 + Qwen3.6 CO-HOSTING** ("124000
   on a 137 GB node co-hosting DSv4 (~79 GB wired steady) + Qwen3.6 left only
   ~13 GB for OS + transient prefill scratch"). The deep-context runs tonight
   were **DSv4 solo** — no Qwen co-resident (verified: no qwen processes, one
   instance). So the documented wedge trigger is absent. This is the highest-value
   experiment and it is reversible via `DSV4_WIRED_LIMIT_MB`.
2. **Reduce the fixed footprint**, i.e. the ~104 GB baseline. That is the only
   way to make 565K comfortable under a fixed limit. Not attempted tonight.

## Not attempted / why

The wired-limit experiment needs its own relaunch and a pre-registered band, and
it carries a documented (if trigger-absent) wedge risk that can require a reboot
to clear — which the user's standing constraint says not to do without say-so.
Left as a decision for the user rather than taken unilaterally at the end of an
overnight session.
