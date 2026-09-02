# PREFILL ROUND 2 — PM report (feedback-loop round 2)

**Date:** 2026-09-02 · **Repo:** exo @ `80db9a855` (main) · **PM:** delegation subagent
**Mode:** All items zero-cluster-cost. Items 1–3 executed (local CPU + read-only). Item 4 prepared only — **no cluster run, no code changes, no commits.**
**Inputs:** fable's round-2 directives; round-1 report (`tmp/prefill-round1-20260902/REPORT.md`); dive report (`tmp/prefix-cache-dive-20260902/REPORT.md`); `tmp/real-usage-capture-20260902/phase1/requests.jsonl`; `~/.hermes/state.db` (read-only).

---

## HEADLINE

**The LCP probe FAILS fable's pre-registered bands. Fix B is NO-GO as currently scoped.**

The failure is not re-tokenization instability. Re-tokenization is near-perfect: **47 of 54 turn pairs reproduce the decode output byte-identically**, and the production template path — including thinking-marker stripping — was measured to be a **no-op on 54/54 turns**. The FAIL comes entirely from the **p10 gate**, tripped by 7 turns that score exactly 0.000.

Those 7 zeros have a single, fully-identified cause: **the client injects a one-character space pad into `reasoning_content` on turns where the model emitted no reasoning.** That pad lands at the first position of the re-fed region, so the longest-common-prefix is 0 even though the entire remaining output is byte-identical.

**This is not a probe artifact — it is real production behavior of the real client.** Those 7 turns would be genuine silent no-op turns in deployment: on ~13% of turns the trie would forfeit everything from position 0. That is precisely the failure mode fable's p10 bar was registered to catch. **The correct reading is not "a trivial blocker" — it is that Fix B has an unstated hard dependency on client-side field serialization, and the current client violates it.** NO-GO is substantively correct, not merely procedurally correct.

| # | Item | Status | Result |
|---|---|---|---|
| 1 | LCP-coverage probe | **DONE** | **FAIL** (p10 = 0.0% < 20%) — bands applied verbatim |
| 2 | Survivorship check | **DONE** | 21.9% reproduces exactly; but it is **length-dependent**, not a fleet constant |
| 3 | SDPA attribution reopen | **DONE** | **Closure is invalid** — per-call vs per-token units conflation. Reopened. |
| 4 | Decode discriminator | **PREPARED** | Direct-timestamp design delivered; ablation demoted (2 of 4 arms hang the cluster) |

---

## 1. LCP-COVERAGE PROBE — the verdict, applied verbatim

### Fable's pre-registered bands (quoted exactly)
- **PASS:** median LCP_coverage ≥ 90% AND p10 ≥ 70%
- **FAIL:** median < 60% **OR** p10 < 20%
- **INDETERMINATE** (60–90% median, or p10 20–70%): score as PRACTICAL-FAIL for round 2

### Measured distribution (n = 54 pairs)

| statistic | LCP_coverage |
|---|---:|
| min | 0.0000 |
| **p10** | **0.0000** |
| p25 | 1.0000 |
| **median** | **1.0000** |
| mean | 0.8704 |
| max | 1.0000 |

**Distribution is strictly bimodal: 47 pairs at exactly 1.000, 7 pairs at exactly 0.000, zero pairs in between.**
Token-weighted coverage of the B bucket: **96.2%** (variant A) / **88.4%** (variant B lower bound).

### VERDICT: **FAIL**

Median 1.000 clears the PASS median bar. **p10 = 0.0000 < 20% trips the FAIL criterion.** FAIL is an OR condition, so it fires. Not INDETERMINATE — the FAIL threshold is met outright.

**Fix B is NO-GO as currently scoped.**

### PM verification (I did not take the worker's word)
The probe subagent hit its iteration cap and never self-reported, so every load-bearing number was re-derived independently before the gate was applied:

- **Distribution recomputed** by me from the raw per-pair array: median 1.000, mean 0.8704, 7 zeros / 47 ones / 0 in between — matches the worker's summary exactly.
- **p10 is robust to method**: 0.0000 under linear interpolation, nearest-rank, and inclusive quantiles. The gate does not hinge on a percentile convention.
- **Reconstruction sanity**: proxy decode tokens sum to 41,413 vs 41,414 recorded `completion_tokens` (Δ = −1 across 54 turns; single pair off by one token). Independently re-derived bucket B = 41,414 straight from `requests.jsonl`. The probe measured the right turns at the right sizes.
- **Causal claim verified against `~/.hermes/state.db` directly** (session `20260901_123854_bb65ee`, 93 assistant rows, starts 12:39 — matching `call_seq=33` at 12:38:40): 13 rows carry `reasoning_content=' '`, and in **all 13** the underlying `reasoning` is empty. The `' '` pad pattern is real and widespread (1,408 occurrences across the DB).

### The natural experiment (why the cause is established, not asserted)
9 pairs had no reasoning content. **7 got the space pad → all 7 scored 0.000. The other 2 had `reasoning_content = null` → both scored 1.000.** No pair that *had* reasoning scored 0.

Pad presence perfectly predicts zero coverage; absence-of-thinking does not. **The pad is the cause.**

### Risk #1 — answered head-on
Fable's pre-registered risk #1 was that thinking-marker stripping shifts RoPE positions and caps reusable KV near zero. **Measured: it does not.** `_strip_v4_thinking_markers` fires on `content`, which never contains markers (exo's parser splits reasoning into `reasoning_content` upstream), so the strip is a **no-op on 54/54 turns**. Separately, exo's encoder is tool-conditional: with tools present `drop_thinking` is disabled, so prior-turn reasoning is re-fed in place, verbatim. 45/54 turns open with real thinking and still score 1.000.

**Zero occurrences** of: thinking-strip position shift, DSML tool-call re-serialization divergence, boundary BPE merge, or system-prompt mutation.

### Honesty ledger (carried up, not buried)
- **The decode side is a PROXY.** Raw decode token ids are not persisted anywhere (searched `requests.jsonl`, the exo event-log archive, and `state.db`). Decode text was reconstructed from stored verbatim `reasoning`/`content`/tool_calls and re-tokenized with the real DSv4 tokenizer.
- **Direction of bias: OPTIMISTIC.** Canonical re-tokenization cannot reproduce decode-time BPE segmentation quirks, and the tool-call region matches by construction (both sides render from the same parsed arguments). The true figure can only be equal or worse.
- **8 multi-invoke pairs are bounded, not pinned** (variant A 1.000 vs variant B 0.755–0.984) because the wire shape of multi-block tool calls is not persisted.
- A proxy this optimistic still FAILS the gate. That strengthens the NO-GO rather than weakening it.

---

## 2. SURVIVORSHIP CHECK — 21.9% is real, but it is a length figure, not a fleet figure

**Fable's question:** does 21.9% overstate Fix B's aggregate value?

**Reproduction (I re-derived this from raw `requests.jsonl`, independent of both prior reports):**

```
total uncached prefill : 189,010
A cold start           :  92,594  (49.0%)
B own completions      :  41,414  (21.91%)   <- the figure under review
C genuinely new        :  55,002  (29.1%)
```

**The pooled 21.9% is arithmetically correct and reproduces exactly.** It is not an accounting overstatement.

**But it is strongly length-dependent** (measured, my own recompute):

| first N turns | uncached | B | B-share |
|---:|---:|---:|---:|
| 2 | 94,619 | 1,982 | **2.09%** |
| 6 | 98,461 | 4,169 | 4.23% |
| 21 | 135,849 | 10,235 | 7.53% |
| 31 | 147,717 | 18,093 | 12.25% |
| 51 | 184,657 | 39,855 | 21.58% |
| **55** | **189,010** | **41,414** | **21.91%** |

Mechanism: **bucket B is a per-turn cost that accumulates; bucket A is a one-time per-session fixed cost.** A session needs ~50 warm turns before B-share reaches 20%. Warm-only (excluding cold start), B is **42.95%** of remaining uncached.

**Verdict: the survivorship concern is directionally VALID, but via session LENGTH rather than cross-session pooling.** This capture is **one** contiguous session with one relaunch, so it cannot establish a cross-session distribution — and no such distribution should be manufactured from it. If typical sessions are much shorter than ~50 turns, quoting 21.9% as Fix B's fleet value **overstates it materially** (a 6-turn session sees ~4%). If sessions are comparably long, it is representative.

**Status: UNDETERMINED from n=1 session.** What would settle it: N ≥ 20–30 sessions with the existing per-request fields, reported both unweighted and uncached-token-weighted.

**Flagged:** bucket A (49%) is the *more* survivorship-laden bucket — its entire value is set by relaunch cadence, still unmeasured.

---

## 3. SDPA ATTRIBUTION — REOPENED. The "fully closed" claim does not survive.

### What the original actually measured
A live 2-node A/B at **12,068 tokens**, `EXO_PREFILL_STEP_SIZE` 2048 vs 4096, sync-mode spans. Measured `attn.sdpa` 0.4153 → 0.8428 **ms/token** = **2.029x**. It was declared to match `bench/sdpa_subtile_microbench.py`'s linear prediction (~1.86–2.00x) "almost exactly," and the thread was closed as "SDPA scales linearly, full stop."

### The crack: a per-call vs per-token units conflation
**The two 2.0x numbers are in different units.** The microbenchmark's ~2.0x is **per-call** (one 2048-row call vs one 1024-row call). The cluster's 2.029x is **per-token**. With SEQ_SPLIT, doubling step size **halves the call count** while doubling tokens per call:

```
ms/token ratio = (calls_4096 / calls_2048) x per-call ratio = 0.5 x per-call ratio
```

**So linear per-call scaling predicts a per-token ratio of ~1.0x — not 2.0x.**

**PM independent verification** (I recomputed this from the doc's own raw numbers):

```
per-call 2048-arm : 0.4153 x 12,068 / 126 calls =  39.78 ms
per-call 4096-arm : 0.8428 x 12,068 /  63 calls = 161.44 ms
per-call ratio    : 4.059x        <- superlinear
identity check    : per-call ratio == 2 x ms/token ratio  -> TRUE
```

The doc compared a per-token ratio against a per-call prediction and concluded the mechanism was confirmed. **It is not.** The cluster's real per-call scaling is ~4.06x for a 2x row doubling.

**The original "3.15x mystery" was never resolved — only relabeled.** The non-sync data independently shows 3.15x per-call *directly measured*. Sync and non-sync agree the cluster is superlinear per-call (~3.1–4.1x) while the isolated fixed-KV bench says 2.0x. The 2.029x ms/token **is that same superlinear signal seen through a halved call count**.

**Sync-mode bias runs the wrong way to rescue the claim:** per-call sync overhead is a larger fraction of a *smaller* call, which **dilutes** the per-call ratio. Bias is opposite to the observed direction.

Ruled out or bounded: ragged remainder chunks (≲10–15%), `mx.compile` recompiles (sparse inner kernel is `shapeless=True`), LocalAttention no-split, depth confound (~5%), KV-length mismatch.

### Are the SDPA closure and the depth degradation the same phenomenon?
**No — and this refutes fable's round-2 hypothesis.** Reported as a clean negative rather than forced into a connection.

The depth slope (426.0 → 418.6 → 406.6 tok/s across 89K/150K/250K; 5.8 ms/chunk per 2048-token step) is **~86% explained by known, designed-in O(P) terms**: compressed-SDPA pooled-prefix attention (~3.1 ms) + indexer score GEMM (~1.8 ms) ≈ 4.9–5.0 of 5.8. Sparse SDPA (fixed top-k=512) and collectives contribute ~0.

That same O(P) mechanism **cannot** have produced the 2.029x at 12K: at that depth the O(P) term is only ~2.2% of the measured `attn.sdpa` per-token cost. Even doubling it entirely moves the ratio ~2%, not 102%.

**Two separate phenomena:** (1) depth degradation = expected O(P) attention behavior, mostly accounted for; (2) a **per-call superlinearity in SDPA at doubled per-rank query rows, mechanism genuinely unknown** — unexplained by O(P), unexplained by sync mode, contradicted by the isolated benchmark. The closure conflated (2)'s units and declared (1)'s territory clean in the same stroke.

### Pre-registered re-measurement at 250K (design only, NOT requested this round)
Primary probe is **`EXO_DSV4_SEQ_SPLIT=0` at STEP_SIZE=2048** — doubles per-rank rows at *identical chunk count, boundaries, and depths*, removing the depth-alignment confound that muddied the original.

| arm | STEP_SIZE | SEQ_SPLIT | purpose | chunks | rows/rank |
|---|---|---|---|---|---|
| A control | 2048 | 1 | standing baseline | 123 | 1024 |
| B primary | 2048 | **0** | same depths, rows doubled | 123 | 2048 |
| C continuity | 4096 | 1 | replicates 12K geometry at 250K | 62 | 2048 |

Mandatory gotchas baked in: `EXO_PROFILER_SYNC_SPANS=1` **paired with `EXO_RUNNER_HANG_TIMEOUT_SECONDS=600`** (45 s default SIGKILLs the runner mid-request — a documented prior failure); measure the **batched** `prefill_batched` path (`generate.py:1269`, called `batch_generate.py:3068`), never the eager fallback, verified via `Starting batched prefill:` log lines; **per-call recorded as primary, ms/token as derived secondary** (the exact lesson of this reopening); no SIGUSR1 (a mistimed signal crashed a rank once).

**Pre-registered bands, fixed in advance** (R = per-call ratio at 2048 vs 1024 rows): **REFUTES** R ∈ [1.8, 2.2]; **CONFIRMS superlinear** R ≥ 2.7; **INDETERMINATE** 2.2 < R < 2.7. Reductio required before quoting any per-call figure: calls × per-call ≤ measured wall.

**Cluster cost: ~75–90 min.** Not requested this round.

---

## 4. DECODE-SIDE DISCRIMINATOR — prepared, not run

Both candidate line ranges **verified at HEAD `80db9a855`** (no drift): A = `dsv4_mtp.py:2259-2310` (fenced coord collectives), B = `batch_generator.py:678-720` (`agree_on_tasks` / `agree_on_cancellations_fast`).

**Recommendation: direct per-collective `perf_counter` deltas, NOT sampled ablation** — matching fable's stated preference, and here the safety analysis makes it decisive.

**The independent-ablation design was built as instructed, but 2 of its 4 arms are UNSAFE and would hang the cluster:**
- `agree_on_tasks` is the **only** path that fills `_queue` — skipping it means new requests are never admitted (**hang**).
- `agree_on_cancellations_fast` is load-bearing — skipping it reintroduces the measured 133.7 s cancellation-latency bug.
- Candidate A's fenced collectives guard against rank drift before TP forwards — NOP-ing can wedge on batch-size transitions.

So B-only-off and both-off are hang-risk arms. This is a strong independent reason to prefer direct timing, which changes no behavior.

**Instrumentation reuses existing machinery** (no new plumbing): A → the `_mtp_trace_log` JSONL path (`dsv4_mtp.py:1876`, gated `EXO_DSV4_MTP_TRANSITION_TRACE=1`); B → the `[PROF]` logger idiom already in that class (`batch_generator.py:532-534`). Overhead ~40–100 ns against a ~1.2 ms/call signal = 0.003–0.008%.

**Depth-stratified** (per fable, since acceptance decays 1.411 → 1.312 → 1.226): S1 0–4K, S2 4–16K, S3 16–32K, S4 32K+. Decision on **S3–S4 means**, no post-hoc threshold tuning.

**Pre-registered bands:** A costs if `(all_sum + all_max) ≥ 600 µs/call` AND ≥10% of decode-step wall in S3–S4; inconclusive if strata straddle a boundary ±20%.

**Cost: ~1.5–2 h** direct timing (vs 4–6 h for the unsafe ablation).

**Worker-flagged, PM-verified:** `batch_generator.py` has **duplicated** `agree_*` methods — `SequentialGenerator` (:159, :208) and `BatchGenerator` (:507, :561). **Instrument only the BatchGenerator copies.** Confirmed by direct grep.

---

## 5. RECOMMENDATION

### Fix B: **NO-GO as currently scoped.**

Applied verbatim, the probe FAILS on p10. I am not softening it, and I am explicitly *not* re-scoring the distribution with the pad excluded — that would be laundering the gate.

The substantive reason NO-GO is right (not just the procedural one): **Fix B has an unstated hard dependency on client-side field serialization.** A single injected byte from a separate codebase zeroes the optimization for an entire turn. Shipping Fix B without addressing that dependency ships a feature that is silently a no-op on ~13% of turns — exactly the outcome the p10 bar exists to prevent.

### What round 2 genuinely established
1. **Re-tokenization stability is not the blocker** — 47/54 byte-identical; the strip path is a measured no-op. Fable's pre-registered risk #1 is retired on evidence.
2. **The blocker is a named, located, client-side serialization behavior** with a clean natural experiment behind it (7/7 vs 2/2).
3. **The 21.9% is length-dependent**, so Fix B's fleet value is unestablished regardless of the probe outcome.
4. **The SDPA closure is invalid** — a real reopened lever, independent of Fix B.

### Round-3 agenda (in priority order)
1. **Client-side change: stop emitting the `" "` pad** (emit empty, or omit the field). Separate codebase, separate change — *not* a modification to Fix B and not a loosening of its bands.
2. **Re-run the LCP probe against the SAME pre-registered bands.** Fix B stays NO-GO until a fresh measurement passes on its own. I am deliberately **not** pre-announcing the expected result — the counterfactual is plausible (strong natural experiment, no mid-distribution mass) but it is a hypothesis, and this project's documented failure mode is exactly conclusions outrunning evidence. Note also that n=9 no-reasoning pairs is small, and the probe has shown the pad is the only divergence mechanism *that occurred in 54 pairs* — not that it is the only one possible.
3. **Session-length distribution** (N ≥ 20–30 sessions) to convert 21.9% into a real fleet number. Cheap, read-only, and it gates the value of the entire workstream.
4. **SDPA per-call re-measurement at 250K** — designed and pre-registered above, ~75–90 min. The named next lever now that the probe has failed.

---

## 6. THE SINGLE APPROVAL ASK

**Authorize the one-line client change (stop padding `reasoning_content` with `" "`) plus a re-run of the LCP probe under the identical pre-registered bands.**

Zero cluster GPU time. Zero cost to the exo repo. Fix B remains NO-GO until that re-probe passes on its own merits.

*(Not requested this round, listed for sequencing: the 250K SDPA re-measurement at ~75–90 min, and the decode direct-timing instrumentation at ~1.5–2 h. Both are designed, pre-registered, and ready — neither is being asked for yet.)*

---

## Appendix — provenance and constraints

- **Artifacts:** `findings/lcp-probe.md`, `findings/lcp_probe.json` (54 pairs, variant A/B bounds, divergence snippets), `findings/survivorship.md`, `findings/survivorship.json`, `findings/sdpa-reopen.md`, `findings/decode-discriminator.md`, plus the probe script `probe_lcp.py`.
- **PM verification performed** (not delegated): independent recompute of the LCP distribution and p10 under three percentile methods; independent re-derivation of buckets A/B/C and the length curve from raw `requests.jsonl`; direct read-only query of `~/.hermes/state.db` confirming the space-pad natural experiment; independent recompute of the SDPA per-call arithmetic; direct `git`/`sed`/`grep` verification of every cited `file:line` in the decode design.
- **Constraints honored:** no cluster contact, no ssh, no inference, no benchmarks, no code changes to `src/`, no commits, no pushes. All writes confined to `tmp/prefill-round2-20260902/`.
- **Discrepancy noted:** `findings/sdpa-reopen.md` cites repo HEAD `cb1f91903` (carried from the round-1 report); actual HEAD at analysis time is `80db9a855`. The cited `file:line` references in that doc were spot-checked and are current; the header is stale, the content is not.
- **Known limitation:** the LCP probe's decode side is an optimistic proxy (raw decode token ids are not persisted anywhere). A FAIL under an optimistic proxy is a robust FAIL; a future PASS under the same proxy would carry the proxy's optimism and should be read accordingly.
