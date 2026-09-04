# ROUND 6 — PRE-REGISTRATION AMENDMENT 1

**Written and committed BEFORE any sweep arm was launched.** Phase 0 is complete; no arm data
exists yet. This amendment changes one instrument choice and fixes the sweep depth. It does not
touch the band in PRE-REGISTRATION.md §3, which stays exactly as originally written.

---

## A1. Phase 0 PASSED — and it validated the measurement path, not just the level

| probe | depth arg | **achieved `prompt_tokens`** | **server `stats.generation_tps`** | pre-registered band | verdict |
|---|---|---|---|---|---|
| short | 2000 | 2,262 | **29.058 t/s** | 20.6 – 32.6 | **PASS** |
| deep | 128000 | 143,964 | **32.816 t/s** | 24.0 – 40.0 | **PASS** |

Both reps `decode_sample_trustworthy = true` (1500 and 1200 completion tokens, floor is 400),
`finish_reason = "length"` (non-null).

**Every one of round 5's four failure signatures was checked for and is absent:**

| round-5 failure | its tell | round-6 Phase-0 observation |
|---|---|---|
| burst-timed stream (~14x impossible) | client rate 10–20x the server's | client 29.15 vs server 29.06 (**0.3%**); client 32.88 vs server 32.82 (**0.2%**) |
| chunk rate mislabelled as token rate | rate independent of token count | server-measured, computed inside the generator; not derived from chunks at all |
| 3 s TTFT misread as fast prefill | prefill far too fast to be real | deep prefill **346.4 s** for 143,964 tok (~415 tok/s) — a real prefill |
| silent prefix-cache hit | warm KV reuse | **`prefix_cache_hit: "none"`** on both probes |

Env on the real runner PIDs at Phase 0 (31608 on .201, 39813 on .202): `EXO_SPECULATIVE_GAMMA=3`,
`EXO_DSV4_MTP_LOG_INTERVAL` **unset**, `EXO_DSV4_MTP_PROFILE` **unset**. API `/v1/models` HTTP 200.

`stats.generation_tps` is produced at `batch_generate.py:4568-4576` and packaged into
`GenerationStats` at `:4599`. On the streaming chat path it is emitted as an **SSE comment line**
(`: generation_stats {...}`, `chat_completions.py:293-296`), not inside a `data:` chunk — which is
why the probe never surfaced it before. The permitted edit captures that line, nothing else.

---

## A2. AMENDMENT: drop `EXO_DSV4_MTP_LOG_INTERVAL`; take acceptance from the stats counters

### What the task brief asked for
> "Also set `EXO_DSV4_MTP_LOG_INTERVAL` on every arm (import-time; it rides the relaunch) so the
> counted acceptance is recorded alongside the throughput for each γ — this is free and it is what
> makes derived vs measured cross-checkable."

The stated PURPOSE is: counted acceptance recorded per arm, alongside throughput, cross-checkable.
The env var was the assumed means to that end. **The Phase-0 data shows the end is reachable
without the means** — and the means is not free after all.

### Why the means is not free
The operations record carries an explicit contrary warning, pre-registered by me in §4 as a
confound BEFORE any data was taken:

> `EXO_DSV4_MTP_LOG_INTERVAL=50` appears to re-trigger JACCL stalls — diagnostic-only env, do NOT
> use during a reported-tok/s bench (May 15 2026: champion env + log_interval gave clean warmup
> 31.76 t/s but iter1 10.57 / iter2 6.28).

That is not a constant offset that would cancel in an A/B. It is **instability** (31.76 → 10.57 →
6.28 across successive iterations of one arm). An unstable arm has no meaningful median, and
throughput is the one thing this round exists to measure. Round 5 already measured acceptance;
it did not measure throughput. Trading a sound primary metric for a redundant secondary one is
the wrong trade.

### The free replacement, and why I trust it
The server's `stats` object already carries `mtp_cycles_cumulative` and
`mtp_accepted_drafts_cumulative`. They are **cumulative over the runner process lifetime**, so
they must be **delta'd between consecutive requests**. Phase 0's two back-to-back probes:

| probe | `mtp_cycles_cumulative` | `mtp_accepted_drafts_cumulative` | generation_tokens |
|---|---|---|---|
| 1 (2K) | 667 | 853 | 1500 |
| 2 (144K) | 1176 | 1543 | 1200 |

Probe-2 deltas: `Δcycles = 509`, `Δaccepted = 690`.

**Self-validating identity.** Each cycle commits exactly one anchor token plus its accepted
drafts, so `tokens_per_cycle` must equal `1 + Δaccepted/Δcycles`:

- measured `1200 / 509 = 2.358`
- predicted `1 + 690/509 = 2.355`
- **agreement 0.1%**

Under the rival (per-request) reading the same numbers self-contradict: `1200/1176 = 1.02`
tokens/cycle while `accepted (1543) > cycles (1176)`, which is impossible. So the cumulative
reading is not a preference — it is the only one consistent with the arithmetic.

### The rule, fixed now
1. Sweep runs **WITHOUT** `EXO_DSV4_MTP_LOG_INTERVAL` (and without `EXO_DSV4_MTP_PROFILE`). The
   swept arms therefore differ from the Phase-0 production boot in **γ only** — a cleaner A/B
   than the brief's own recipe would have produced.
2. Acceptance per arm comes from counter deltas across that arm's consecutive reps.
3. **The identity `tokens/cycle == 1 + Δaccepted/Δcycles` is checked on EVERY rep.** It is
   free and self-validating.
4. **Fallback, fixed in advance:** if the identity fails on any rep by more than 2%, the
   counter-delta acceptance is declared unreliable, and acceptance for the remaining arms is
   obtained by setting `EXO_DSV4_MTP_LOG_INTERVAL` on those arms — with the §4 stability checks
   (C1/C2) applied and any resulting instability reported, not absorbed.

Because the env var is now absent on all arms, pre-registered confound check **C1 is moot**.
**C2 (stability: each arm's 3-rep range width ≤ 8 t/s) still applies** — it is a general
bistability guard, and this cluster has a documented history of bistable γ≥2 behaviour.

---

## A3. AMENDMENT: fix the sweep depth to land at ~89K, not 144K

Phase 0's deep probe used argument `128000` and achieved **143,964** `prompt_tokens`. That clears
the `>= 85K` floor, but the floor exists to catch **undershoot** (round 5 asked 89K and got 62K);
it is not licence to run 62% deeper than specified. The task says 89K. Reasons to correct it:

1. **Regime fidelity.** The record's 30–34 t/s reference band and this campaign's prior results
   are at ~89–100K. The record separately notes a model cliff beyond 100K. A sweep at 144K may
   not be comparable to the campaign it is supposed to extend.
2. **Memory headroom.** Phase 0 peaked at **100.06 GB** at 144K. γ=4 adds another draft/verify
   row. The record documents stream interrupts from peak-memory reclaim stalls at high context —
   an avoidable way to lose an arm.
3. **Time box.** Prefill dominates: ~346 s at 144K vs ~215 s at 89K. Across 5 arms × 4 reps that
   is roughly **45 minutes** of pure prefill saved, inside a 4-hour box.

**Calibration.** The probe's prompt is linear in the depth argument (fixed-size salted filler
blocks), so the observed ratio `143,964 / 128,000 = 1.1247` extrapolates:
`89,000 / 1.1247 ≈ 79,100` → **use argument `79000`**, predicted ~89.0K achieved.

**Binding rules:** the warmup rep confirms achieved depth; the argument is adjusted if needed to
land in **85,000–95,000**; and once fixed, **the identical argument is used for every arm** —
depth equality across arms matters more than hitting 89K exactly.

---

## A4. Unchanged

The band (§3), the arm order (§1), the quality gate (§5), ship/hold (§6), the degrade path (§7)
and all hard constraints (§8) are **unchanged**. The C2 stability check survives from §4. The
predictions in §9 stand as written and will be scored honestly.
