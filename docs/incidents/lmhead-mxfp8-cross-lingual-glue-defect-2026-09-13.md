# `EXO_DSV4_LMHEAD_MXFP8=1` causes a live, user-facing text-quality defect (fixed)

**Found:** 2026-09-13, from a live user report against the production
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` deployment (post DSpark-native-head
fix + mlx/mlx-lm upstream merge, both landed 2026-09-12).

**Severity:** high, user-facing, deterministic. Not a crash and not visible to
throughput or needle-in-haystack benchmarks — the cluster looked completely
healthy on every metric that was being watched. Only reading the actual
generated text surfaces it.

**Status:** FIXED. `EXO_DSV4_LMHEAD_MXFP8` default flipped 1→0 in
`start_cluster.sh`, commit `1da54ee192203194a35bddbb53625f7cb11799d9`
(2026-09-14). Confirmed 0/10 defect rate live post-fix, multiple independent
verifications. Live production env var confirmed `EXO_DSV4_LMHEAD_MXFP8=0`
on both nodes as of this writing.

---

## 1. Symptom

Correct English words get a garbled nonsense suffix spliced on with no space
and no punctuation, sometimes in a different script than the surrounding
text:

- "...back to the camerauden." (should be "...camera.")
- "...top of the thighsuden."
- "...textured pattern火热 and ribbed cuffs." (Chinese 火热 glued onto "pattern")
- "...against a dark backgroundфабрика." (Russian "фабрика" = factory)
- "...grass or pavement他身上." (Chinese 他身上 = "on his body")
- "angleсь" (Cyrillic сь glued onto "angle")

Deterministic repro: 10/10 defect rate at temp=0 pre-fix, 0/10 post-fix
(10-trial battery, same image, same prompt). Vision prompts triggered it far
more reliably than text-only prompts (~100% vs ~33%), but it is not
vision-specific — the underlying bug is in the shared lm_head projection used
by every decode call, vision-conditioned generation (longer prompts,
different KV layout) just triggers the low-margin decode positions more
often.

## 2. Root cause

`EXO_DSV4_LMHEAD_MXFP8=1` (shipped ON by default since commit `80ec8ec03`,
2026-08-30) quantizes the DeepSeek-V4 lm_head (129280×4096, 1.059 GB/rank
unquantized BF16) to mxfp8 (group=32, bits=8) in place at load time, for a
measured +6.0% decode throughput win. The quantization noise this introduces
is large enough, at specific low-margin decode positions, to flip which
token wins the top-1 argmax — not a rounding error in a probability that
still picks the same winner, a **genuine change of which token gets selected**.

Root-caused via raw logprobs capture (`/v1/chat/completions` with
`logprobs=True, top_logprobs=10, temp=0`, 273 sampled tokens): at the defect
token, the model's own top-10 is flat cross-lingual garbage (top1 Cyrillic
сь, then 利/火热/hing/ž) with the semantically-correct token demoted to
rank 2 by roughly 1 nat. `delta == top1` for every one of the 273 sampled
tokens in the capture — the detokenizer is exonerated, it renders exactly
whatever the model actually selected. Per-row logit error at the defect
positions: mean 0.53 / rms 0.68, against a logit std of 11.3.

## 3. Why the ship-decision eval didn't catch this

The original ship-decision eval (commit `80ec8ec03`, 2026-08-30) was a
15-task battery of high-margin exact-match/executed-code tasks (arithmetic,
factual recall, code that gets executed and asserted). It scored 15/15
byte-identical between the mxfp8-on and mxfp8-off arms. That eval design is
structurally incapable of detecting this defect: high-margin tasks have a
large gap between the correct token and its nearest competitor, so
quantization noise doesn't have room to flip the argmax. The failure mode
only appears in **low-margin free-form prose** — exactly the regime
open-ended vision image-descriptions live in — which the 15-task battery
never exercised.

The pre-existing loader comment in `mlx-lm/mlx_lm/utils.py` had already
flagged a ~11.5% *estimated* top-1 flip rate at ship time and reasoned "free
prose does visibly diverge... while the substance stays correct." That
reasoning is exactly what this incident disproves: low-margin flips are not
benign rewording, they are a genuine wrong-token selection that reads as
corrupted text to the end user.

## 4. The ~11.5% figure — what it actually is (frequently mis-cited, correct it if seen elsewhere)

The "~11.5% flip rate" is **not a directly observed all-token flip rate**.
It's a derived estimate, the product of two separate measurements:

- (a) a ~13% top-1 flip rate measured on **synthetic** inputs, concentrated
  100% in the margin<3.6-nat near-tie band (0% flips above that margin), and
- (b) a real-generation margin distribution (n=3999 committed tokens across
  mixed context lengths, temp=0) showing 42.7% of real tokens fall below
  that 3.6 margin threshold.

~11.5% ≈ (b) × (per-band flip rate in (a)). It is explicitly flagged as "an
ESTIMATE, not a direct measurement" in the `mlx_lm/utils.py` loader comment
and in the commit message. If this figure gets re-quoted anywhere in the
future, carry that caveat with it — it was never a measured flip rate over
all decode tokens.

## 5. Fix and cost

`start_cluster.sh`: `EXO_DSV4_LMHEAD_MXFP8` default flipped `1`→`0` (full
BF16 lm_head). Two independent 100K-context decode measurements post-fix:
37.05 and 37.65 tok/s, vs a 38.62 tok/s mxfp8-on baseline — roughly a
2.5-4% decode cost, reported as a range because the ~2 tok/s spread between
the two post-fix runs is normal run-to-run measurement noise on this
benchmark, not a discrepancy to reconcile. (A later, unrelated remeasurement
of the same bf16-only config in a different session came back at 39.06
tok/s — further evidence this benchmark simply has ~2-3 tok/s of natural
run-to-run variance; don't treat any single throughput number here as an
exact point value.)

Full technical detail on why no cheaper fix was viable — three follow-up
investigations, all negative results, real engineering effort, do not
re-attempt without reading first — is in
`docs/lmhead-mxfp8-defect-and-fallback-investigations-2026-09-14.md`.

## 6. Why this matters for validation discipline

Every other fix landed in the 2026-09-12–14 session was validated by
decode throughput (tok/s) and needle-in-haystack task-completion — both of
which score this defect as "healthy." A benchmark focused on speed and
task-completion needle-finding will **never** catch a subword-splicing
quality bug; the needle is still found even when the prose around it is
garbled. When validating any decode-path change (quantization, DSpark,
detokenizer, speculative decoding), also eyeball the actual generated text
for garbled fragments — or better, capture raw logprobs
(`logprobs=True, top_logprobs=10, temp=0`) on a free-form-prose prompt and
check the top-10 for cross-script/nonsense tokens at low-margin positions,
not just whether the sampled token matches the streamed detokenizer output.

## 7. Cross-reference

- Fix commit: `1da54ee192203194a35bddbb53625f7cb11799d9` — commit message is
  the single most detailed, already-reviewed technical narrative; read it
  directly (`git show 1da54ee192203194a35bddbb53625f7cb11799d9`) rather than
  re-deriving.
- Full investigation detail (alternative quantization schemes, the
  conditional-fallback implementation attempt, the `mx.where` structural
  finding): `docs/lmhead-mxfp8-defect-and-fallback-investigations-2026-09-14.md`.
- Predecessor investigation (the original P05 ship-decision numerics,
  2026-08-30): `tmp/p05-lmhead-mxfp8-20260830/`, `tmp/p05-review-20260830/`.
- Original ship-decision commit: `80ec8ec03399ec52a6e4e129e669054ac1fc65e6`
  ("perf(dsv4): ship lm_head mxfp8 by default (+6.0% decode @100K, zero
  quality cost)") — the "zero quality cost" claim in that subject line is
  what this incident disproves for low-margin free-form prose specifically;
  the commit is not reverted (the mxfp8 quantization code path itself is
  still present and functional, just no longer the default), only the
  default changed.
