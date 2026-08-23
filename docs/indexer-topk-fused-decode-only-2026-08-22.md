# T10 continued: attn.indexer's fused top-k lever is structurally decode-only — closes cheaply via code reading, no live test needed — 2026-08-22 (session 4)

## Why this check

Per a Fable consult's advice after the HyperConnection false lead: cheap,
high-value follow-up before wrapping T10 for the session. `attn.indexer`
(4.0% of prefill wall time) has an existing opt-in flag
`EXO_DSV4_TOPK_FUSED` (live-toggleable via `/tmp/dsv4_nop_targets`, no
relaunch needed) whose docstring claims "~5x speedup at the pipelined
chain level." Prior testing of this flag (`docs/fork-notes.md`, an
older May-era MTP session) found only a ~3% latent win below the noise
floor — but that test was decode-specific, and the flag was never
explicitly re-checked for the current TP prefill regime.

## Real finding — settled by code reading, no live test required

Read `Indexer.__call__`'s real top-k dispatch logic
(`deepseek_v4.py:3888-3949`). The fused-kernel gate:

```python
if (_topk_enabled
        and scores.shape[1] == 1
        and pmask is None
        and k <= 1024):
    fused = _fused_topk(scores, k)
```

**`scores.shape[1] == 1` is an explicit, structural decode-only
condition** — `scores.shape[1]` is the query-row dimension, which
equals `L` (the number of tokens being processed in this call). Real
prefill chunks run at `L=2048` (`EXO_PREFILL_STEP_SIZE`, the standing
default) — `scores.shape[1]` is always 2048 during prefill, never 1.

**This means `topk_fused` structurally cannot engage during prefill at
all, regardless of the flag's value.** No live cluster test (NOP-toggle
or relaunch) is needed to settle this — the shape gate alone proves the
code path is unreachable in the prefill regime. Confirmed via direct
production check (`ps eww` on the live runner PID) that neither
`EXO_DSV4_TOPK_FUSED` nor the `topk_fused` NOP-target file were active
before this check (clean baseline, `/tmp/dsv4_nop_targets` didn't
exist on either node) — not that it would have mattered given the
structural gate.

Real prefill top-k always takes one of two OTHER paths: `argpartition`
(gated behind the separate, already-tested `EXO_DSV4_PREFILL_ARGPARTITION`
lever — see §3.1/§13, already closed with real numbers) or the
fallback `argsort+slice`. Neither is `topk_fused`.

## Conclusion

**This sub-lead is CLOSED cheaply, without a live cluster test.** The
prior fork-notes A/B (28.77 vs 29.7 tok/s decode, ~3% below noise) was
testing the ONLY regime this specific flag can ever affect — there is
no separate, untested "prefill regime" for `EXO_DSV4_TOPK_FUSED`
specifically, because the flag's own code gate structurally excludes
prefill by shape. This upgrades the prior finding from "possibly stale
for the current architecture" to "confirmed still fully applicable and
closed" — a real, useful clarification even though it didn't uncover a
new win.

## T10 status at end of session 4 — explicit stopping point per Fable's recommended decision criterion

Two real sub-investigations this session, both settled with NULL
results (no hidden bug found):
1. **HyperConnection training-gate** (attn_hc/ffn_hc, 4.6% of wall
   time) — investigated in depth (standalone microbench, Fable consult,
   live production checkpoint verification). Fast kernel already fires
   correctly in production. NOT a bug.
2. **attn.indexer's fused top-k** (part of the indexer's 4.0% of wall
   time) — closed cheaply via code reading. Structurally decode-only,
   not applicable to prefill, already correctly tested in its only
   applicable regime.

**Genuinely still unexplored** (not investigated this session, flagged
explicitly for a future continuation, per Fable's advice not to start
fresh sub-investigations this late in an already-long session):
- `layer.attn_residual`/`ffn_residual` (`hc_expand`, 4.4% of wall time
  combined) — the microbench in the HyperConnection investigation
  measured this at 2361µs/call but did not decompose it further (no
  training-gate applies to this specific function, so the same class
  of lead doesn't apply — would need its own investigation from
  scratch).
- `moe.gate`/`moe.post_combine` (5.1% combined) — not read this
  session at all.

**Explicit decision criterion for the next session** (per Fable's
advice, stated now so it isn't re-litigated from scratch): two real,
independently-investigated NULL results (HyperConnection, indexer
fused-topk) is a mild signal that prefill's 28.8% non-GEMM remainder
may be predominantly legitimate small-op overhead rather than
containing one single dramatic hidden bug like the async-fence case.
**One more genuine NULL result** from investigating `hc_expand` or
`moe.gate`/`post_combine` with the same rigor **would justify
demoting T10** from "actively hunt for a hidden bug" to "accepted
overhead, documented and closed" — at which point the 1.40x
theoretical headroom figure (T7) should be reframed as a genuine
architectural/dispatch-count ceiling rather than a to-be-found bug.
Conversely, if either remaining span DOES turn up a real fixable issue,
T10 continues to be the highest-priority item per T6's standing
recommendation.
