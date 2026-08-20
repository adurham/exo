EXO_PREFILL_CHUNK_OVERLAP live test: real, unresolved correctness anomaly found (2026-08-20)
==================================================================================================

Summary
----------

Deployed the real TP-native compute/comm overlap mechanism
(`EXO_PREFILL_CHUNK_OVERLAP=1`, commit `b7aa41920`, built into
`prefill_batched()`'s actual per-chunk loop in
`generate.py`, not a synthetic script) to the live 2-node cluster for
a real correctness+throughput test.

Result: no crash, no hang, no HTTP errors -- but a genuine correctness
anomaly on one of two trials.

Trials (fresh 38K-token prompts, `cached_tokens: 0` confirmed each time,
flag=1 on both ranks confirmed matching)
-----------------------------------------------------------------------------

- **Trial 1**: expected `OVLP-9471-CHK-3433`, got
  `SECLP-9473-CHK-3433` -- WRONG on 2 of ~15 characters, both the
  prefix and one digit group. `finish_reason: stop` (not truncated),
  only 37 reasoning tokens before answering (thin confidence signal).
  Cumulative throughput ~167-171 tok/s, essentially unchanged from
  baseline.
- **Trial 2**: expected `OVLP2-7665-CHK-7732`, got
  `SECRET CODE: OVLP2-7665-CH` -- correct, just truncated by
  `max_tokens=50` (`finish_reason: length`).

Interpretation
------------------

This is NOT a clean pass and NOT a clean, reproducible failure --
it's a real, ambiguous signal that needs proper investigation, not a
declaration either way. Two live trials is not enough to establish
whether this is:
(a) genuine model imprecision on a hard long-context recall task
    (plausible baseline miss rate not established this session --
    no matched flag=0 control was run on the identical prompts before
    reverting),
(b) a REAL correctness bug from the overlap change -- most concerning
    interpretation, and exactly the class of subtle "fast but wrong"
    bug Fable warned about when reviewing the design
    (a stream that escapes the collective's dependency chain can look
    like it works while silently reading stale/wrong state).

Action taken
---------------

Reverted to standing baseline (`EXO_PREFILL_STEP_SIZE=2048`, flag
unset) immediately, before running a third trial or attempting to
diagnose live. Cluster relaunch hit a real, SEPARATE, unrelated
hang (a runner hung mid-load, self-healed via the standing
`HANG_TIMEOUT_SECONDS` watchdog + automatic re-placement, ~14 minutes
to fully reload the 144GB checkpoint from a cold cache) -- confirmed
via direct process RSS growth (16.9GB and climbing) that this was
genuine slow loading, not a second hang, before waiting it out.
Cluster now confirmed fully healthy and responding, standing config
verified on both ranks.

What must happen before this flag is ever re-enabled
----------------------------------------------------------

1. **A proper, controlled A/B is required**, not two live spot-checks:
   the SAME set of prompts run through BOTH flag=0 and flag=1,
   multiple trials each, to establish whether trial 1's miss is within
   the model's normal baseline imprecision rate or specific to the
   overlap path.
2. **Logit-level comparison, not just token-level.** A greedy-argmax
   match (or near-match) can hide real numerical divergence -- compare
   full logit distributions or at minimum log-prob of the correct
   continuation, flag=0 vs flag=1, on identical prompts.
3. **Re-check the double-buffer boundary specifically.** The most
   likely bug class here, per Fable's original design review: does
   the depth-1 `mx.eval(_prev_cache_sync)` genuinely block until the
   PREVIOUS chunk's cache write is fully materialized before the
   current chunk's forward pass reads that cache? If there's any path
   where a chunk's forward reads cache state that hasn't finished
   materializing, that produces exactly this failure signature --
   correct most of the time (cache usually finishes in time), wrong
   occasionally (a race, exacerbated at whatever real system load was
   present during trial 1).
4. Do NOT re-enable on production traffic until 1-3 are done and pass
   cleanly, per this session's own standing correctness-gate discipline
   applied consistently to every other lever tonight.

Status
---------

Code remains committed on `main`, default OFF
(`EXO_PREFILL_CHUNK_OVERLAP=0`), so standing production behavior is
completely unaffected. This is real, unfinished work with a real open
correctness question -- not abandoned, not shipped, genuinely paused
pending the controlled A/B above.
