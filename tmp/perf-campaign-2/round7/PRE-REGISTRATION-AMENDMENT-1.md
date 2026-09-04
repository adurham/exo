# ROUND 7 — PRE-REGISTRATION AMENDMENT 1

**Committed BEFORE the arm-Z relaunch and before any steel-BI arm.** Nothing in this file is
written with knowledge of a comparison arm; only arm P (the production baseline, 200ms) has been
measured at the time of writing.

---

## The problem this amendment fixes: the pre-registered TTFT design cannot resolve its own effect

Arm P came back (n=5, cache-cold ~2K, `prefill_s`):

| rep | prefill_s (ms) | prompt_tokens | prefix_cache_hit |
|---|---|---|---|
| 1 | 7800 | 2285 | none |
| 2 | 7660 | 2216 | none |
| 3 | 8030 | 2331 | none |
| 4 | 7940 | 2285 | none |
| 5 | 7900 | 2285 | none |

**median 7900 ms, range [7660, 8030], width 370 ms.**

The effect we are trying to detect is **200 ms**. The within-arm range on a single arm is
**370 ms — 1.85x the effect.** At n=5 the median's sampling error is on the order of ±90 ms and the
difference-of-medians on the order of ±125 ms, against a band, `[−250, −150]`, that is only 100 ms
wide.

**This design can produce a number that lands in the band by luck, or misses it by luck, with
roughly similar probability.** Shipping on it would be exactly the "believable garbage" failure
this campaign has already hit twice (round 5's harness, round 6's acceptance at n=3). The band is
not wrong — it is the brief's band and it still governs — but the *instrument* underneath it is
too coarse, and that is visible now, before the arm runs, rather than after.

The reason is structural: at ~2,300 prompt tokens, prefill itself is ~7.7 s and carries all the
variance, while the rendezvous sleep is a fixed ~200 ms sitting on top of it. We are measuring a
constant by differencing two large, noisy numbers.

## The fix: measure the constant where it is not swamped

The rendezvous window is **prompt-length-independent**. It is a `queue.get(timeout=remaining)`
drain loop that runs *before* `generator.step()` and therefore before any prefill work
(`runner.py:571-604`, gate at `:580`). Its cost is the same 200 ms on a 20-token prompt as on a
2,300-token prompt.

So: measure it on a prompt small enough that prefill is not the dominant term.

### Amendment — added instrument (supplementary, does NOT replace the pre-registered one)

**A2 — short-prompt TTFT, n=10 per arm.** A ~20-token prompt (`long_decode_probe.py 20
--max-tokens 16`), fresh uuid salt per rep so every rep stays cache-cold, `prefix_cache_hit` must
read `none`. n=10 rather than n=5 because the reps are seconds not minutes, so the extra precision
is nearly free.

Decision statistic, stated now: `median(A2 | Z) − median(A2 | P)`, in ms, reported with both arms'
full ranges.

### What each instrument is for — fixed now, so neither can be cherry-picked later

| instrument | role | governs? |
|---|---|---|
| **A1** — 2K, n=5, `prefill_s` (the pre-registered one) | the brief's stated measurement; kept verbatim, reported in full | **YES — the `[−250, −150]` ship band applies to A1's delta, exactly as pre-registered** |
| **A2** — 20-token, n=10, `prefill_s` (this amendment) | isolates the constant from prefill variance; corroborates that any A1 delta is the rendezvous and not drift | **NO — diagnostic only** |

**Both are reported whatever they say.** A1 keeps the ship decision because that is what was
pre-registered and the brief is explicit about the band. A2 exists so the round can state whether
A1's number is *trustworthy*, which A1 alone cannot answer.

### Pre-registered handling of disagreement (written before either arm Z number exists)

- **A1 in band AND A2 ≈ −200 ms** → SHIP, and the corroboration is stated.
- **A1 in band BUT A2 nowhere near −200 ms** → **HOLD.** A1 landing in a 100 ms-wide band while the
  high-resolution instrument disagrees means A1 hit the band by noise. Do not ship on it.
- **A1 outside band BUT A2 ≈ −200 ms cleanly** → **HOLD, per the brief** ("Delta outside that band
  → report, do not ship"). Report that the effect is real and prompt-length-independent, that the
  A1 design was underpowered, and recommend the follow-up. **This round does not ship on an
  instrument it added to itself mid-round.** The brief's band wins.
- Arm P is NOT re-measured after seeing arm Z. Its five reps above are frozen.

### What is NOT changing
No new harness. `--run-id` is additive and unused by any timing path. Arm P's A1 numbers stand as
recorded. The clean-logs requirement (20 sequential requests, zero rank disagreement) is unchanged
and remains an independent veto on Task A regardless of any timing number.

---

## Note on arm P's absolute TTFT (recorded, not acted on)
7.9 s to first token on a ~2,300-token prompt is ~290 tok/s effective, against the ~416 tok/s
prefill rate round 6 measured at 88K. Small-prompt prefill carries fixed per-request overhead that
amortizes away at depth, so this is not anomalous — but it is also the reason the 200 ms is only
~2.5% of A1's signal. Recorded here so the ratio is not rediscovered as a surprise later.
