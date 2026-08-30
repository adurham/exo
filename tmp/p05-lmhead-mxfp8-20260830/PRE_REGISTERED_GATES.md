# P05 Phase A live A/B — PRE-REGISTERED GATES (written BEFORE any relaunch)

# Decision framework: the offline numerics (lmhead_numerics_v2.json + real
# margins from live logprobs) established:
#   - logit error: mean 0.53, rms 0.68, vs logit std 11.3 (SNR ~16.6x)
#   - top-1 flips: 0% for margin > 3.6; 13.3% synthetic overall;
#     expected real-token flip rate ~16% (near-ties only)
#   - studio microbench: lm_head 1.64x-1.84x at M=1/3/4 ->
#     -2.69 ms/cycle ~ -0.84 ms/token projected (~+18% decode tok/s if the
#     whole bucket scales; P03 measured bucket at 5.38 ms/token total,
#     cycle 17.2 ms at 3.2 tok/cycle -> baseline ~ 60-65 tok/s at 100K,
#     lm_head family is 27% of the bucket)
#
# GATES (fixed now, before the run):
#   G1 QUALITY (needle): ab_probe_tier1.py at 5K and 100K, 3 runs each
#      per arm (A=mxfp8 knob ON, B=production BF16).
#      PASS: needle_hit in >= 5/6 runs with the knob ON, zero BOS spam,
#      coherent output eyeball on every run.
#      FAIL -> revert immediately, document closed-negative.
#   G2 THROUGHPUT: decode tok/s at 100K, 3 runs per arm.
#      PASS: mxfp8 arm median >= BF16 arm median * 1.03 (>= +3%) AND
#      per-run mxfp8 > BF16 in >= 4/6 pairwise comparisons.
#      Report the raw deltas either way.
#   G3 SAFETY: both runners up with 5/5 verbon3 flags + EXO_DSV4_LMHEAD_MXFP8=1
#      (+ EXO_DSV4_PRENORM_H_DUMP knob) present via ps eww; memory < 125GB
#      resident on either node; no runner crash during the test window.
#      Behavioral knob-verification: "[LMHEAD_MXFP8] quantized lm_head" in
#      the runner log at model-load time on both nodes.
#   G4 DUMP: >= 12 prenorm_h_*.bin captures land in the dump dir for the
#      offline real-h replay (Phase A follow-up + Phase B real-x check).
#   G5 (added pre-run, still before any measurement) SAME-PROMPT DIVERGENCE:
#      one fixed prompt at temp=0 max_tokens=200 per arm; report the
#      byte-diff rate between arms (expected nonzero given the ~16%
#      near-tie flip rate — this quantifies the visible quality cost).
#
# PROTOCOL AMENDMENT (fixed BEFORE the baseline run, replacing the earlier
# "alternating arms" wording): arm B (BF16 production baseline) is measured
# on the CURRENT live production cluster (16h-warm — bias AGAINST arm A
# winning, i.e. conservative for a positive claim), then ONE relaunch with
# the knob ON measures arm A after the same warmup procedure. Full
# alternation would cost 6 relaunches; the pre-registered statistical bar
# stays as written. Both arms use the identical probe sequence: warmup(2),
# needle 5K x3, needle 100K x3 (decode_tps from these = the throughput
# sample), same-prompt x1.
#
# Relaunch plan: 1 relaunch with production env + EXO_DSV4_LMHEAD_MXFP8=1
# + EXO_DSV4_PRENORM_H_DUMP=<dir>. Then 1 relaunch to restore production.
# Total: 2 of the 8-relaunch budget.
GATES = "recorded"