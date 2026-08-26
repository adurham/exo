# DSpark/MTP spec-decode verdict — Tier 1 byte-identity (2026-08-26)

**Step 3 of the corrected spec-decode verdict protocol. Captured the 7-prompt
degen set (bench/spec_degen_capture.py, /v1/chat/completions, temp=0) under
spec-OFF and spec-ON at 512 max_tokens (enough for the 3 short prompts to
reach finish=stop), plus a 3× determinism check per arm on the divergent
prompt.**

## Tier 1 verdict: PARTIAL — 2/3 short prompts byte-identical, 1/3 differs

| prompt | spec-OFF | spec-ON | byte-identical? |
|---|---|---|---|
| sys_capital_france | content="Paris" (5c), reasoning 151c, finish=stop | content="Paris" (5c), reasoning 151c, finish=stop | **YES** |
| sys_count_to_five | content="One, two, three, four, five." (28c), reasoning 141c, finish=stop | identical | **YES** |
| sys_primary_colors | content 238c, reasoning 1275c, finish=stop | content 262c, reasoning 1228c, finish=stop | **NO** |
| sys_long_essay | length-truncated | length-truncated | n/a (truncated) |
| sys_long_steps | length-truncated | length-truncated | n/a (truncated) |
| sys_long_list | length-truncated | length-truncated | n/a (truncated) |
| control_user_only | length-truncated | length-truncated | n/a (truncated) |

**2 of the 3 short prompts that reached finish=stop are byte-identical
(content AND reasoning_content) across spec-OFF vs spec-ON.** The third
(sys_primary_colors) differs.

## Determinism control: BOTH arms are internally deterministic on the divergent prompt

Ran sys_primary_colors 3× against each arm (same process, back-to-back):

| arm | run0 | run1 | run2 | internally deterministic? |
|---|---|---|---|---|
| spec-OFF | 238c/1275c | 238c/1275c | 238c/1275c | YES (3/3 identical) |
| spec-ON | 262c/1228c | 262c/1228c | 262c/1228c | YES (3/3 identical) |

Both arms are byte-deterministic run-to-run on this prompt, but they produce
**different** outputs from each other. This is NOT base-model nondeterminism
(the spec-OFF arm is perfectly stable 3/3). It is a **consistent, deterministic
divergence introduced by the spec-decode path** — exactly the batched-verify
drift documented in `exo-speculative-decode-correctness` and the M1 shadow-gate
byte-identity gate.

## Root cause: the known rowseq/MoE 0.023%/row residual (not a new bug)

The divergence mechanism is documented and understood (M1 shadow-gate doc,
`docs/p4v2-m1-shadow-gate-results-and-recovery-2026-08-24.md` §"Byte-identity
gate: FAILS under the shipped config"):

- The shipped config runs `EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=0` +
  `EXO_DSV4_MTP_ACCEPT_LOGPROBS=1` + `EXO_DSV4_MOE_PARTS_ROWSEQ=shared`, which
  leaves a **~0.023%/row residual** vs bitwise-exact in the MoE shared experts.
- One early low-margin argmax flip (a counting/decision position in the
  reasoning) cascades the whole trajectory: spec-ON's reasoning commits to a
  different phrasing ("In additive color theory...") than spec-OFF's ("In the
  additive color model..."), which propagates to different content.
- The bitwise-exact path requires `EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=1` (run the
  MoE shared-experts per-row too), which the live config does NOT set because
  it carries a perf cost (the FULLBLOCK context-scaling cliff,
  `docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md`).

This is the same finding the M1 shadow gate already measured: shadow output
was deterministic across reruns (byte-identical) but diverged from the
production build on the identical temp=0 prompt (the model even counts the
corpus differently: "46 sections/8 topics" prod vs "45 sections/9 topics"
shadow). Tier 1 reproduces that at the short-prompt scale: the spec path is
self-consistent but not bit-equivalent to the serial path under the shipped
MoE config.

## What this means for the verdict

- **Tier 1 does NOT cleanly pass** (the pre-registered bar was "byte-identical
  on all 7"). 1 of 3 comparable short prompts diverges, deterministically, due
  to the shipped MoE residual.
- This is NOT a spec acceptance bug (Gate A is clean — acceptance is strict
  argmax). It is a numerics residual in the verify forward's MoE path that
  flips a near-tie early and cascades.
- **The divergence is deterministic and bounded** — it does not grow or
  randomize across runs. Spec-ON is a fixed (if different) trajectory, not a
  noisy one.
- The consults' Tier 1 criterion ("hard gate, any mismatch = spec bug, blocks
  promotion") would call this a FAIL. But the consults also assumed short
  prompts are bit-deterministic across arms; the M1 shadow gate already showed
  they are NOT under the shipped MoE config (the residual is ~0.023%/row,
  enough to flip a near-tie). So Tier 1's "hard gate" is really testing the
  MoE-rowseq residual, not spec acceptance correctness.
- **For the verdict, this is a neutral-to-negative signal, not a blocking
  one.** The PROMOTE bar (median delta ≥ +10% AND lower CI ≥ +5%) is already
  unreachable on the C_s arithmetic (step 1), so the Tier 1 result does not
  change the decision direction — it confirms the mechanism is not
  production-clean, which reinforces REVERT.

## Artifacts

- `/tmp/ab/tier1/degen_specoff_512.json` — spec-OFF @512 (TRUE baseline, 7 prompts)
- `/tmp/ab/tier1/degen_specoff_2.json` — spec-ON @512 (mislabeled filename; 7 prompts)
- `/tmp/ab/tier1/degen_specon.json` — spec-ON @200 (7 prompts)
- `/tmp/ab/tier1/degen_specoff.json` — spec-OFF @200 (7 prompts, first capture)
- `/tmp/ab/tier1/degen_diff_proper.py` — comparison script
- `/tmp/ab/tier1/determinism_check.py` — 3× determinism probe
- `bench/spec_degen_capture.py` — capture tool (7-prompt degen set)