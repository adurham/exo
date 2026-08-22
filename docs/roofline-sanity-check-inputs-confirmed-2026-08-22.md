# Roofline sanity check: inputs confirmed correct, one dormant-flag finding — 2026-08-22 (session 3, offline + read-only)

## Why this check

Per Fable's ranked next-steps from earlier tonight's review: sanity-check
the roofline ceiling calculation (`docs/decode-roofline-dispatch-bound-2026-08-21.md`)
used real active-expert bytes at the actual production decode
configuration (real gamma, real speculative-decode state), not total
model parameters or an assumed/default configuration — before trusting
the "~12% of theoretical peak" headroom claim further. Pure read-only
code inspection and arithmetic; zero cluster risk, no relaunch.

## Check #1: does production actually run speculative decode (gamma>1)?

Confirmed live: `EXO_DSV4_DSPARK=1`, `EXO_DSV4_MTP_EAGLE_K=8`,
`EXO_DSV4_MTP=0`. At first glance this looked like it might mean the
roofline's implicit "1 real forward pass = 1 generated token"
assumption was wrong — DSpark speculative decode, if active, would mean
multiple draft/verify forward passes per accepted token, changing the
real compute-per-token accounting.

**Traced the actual code path and found DSpark's decode loop
(`pp_dspark_decode_loop` in `src/exo/worker/engines/mlx/pp_speculation.py`)
is PP-only** — it's dispatched from inside a code block that imports
`pp_rank`/`pp_world_size`/`pp_group` from `get_pipeline_info()`, and the
surrounding comment in `batch_generate.py` explicitly states DSpark "is
dispatched entirely separately at generate-time via
`pp_dspark_decode_loop`... this `__post_init__` runs unconditionally...
before PP vs non-PP is even decided, so it never sees DSpark" for the
non-PP (TP) construction path. No TP-specific DSpark decode loop exists
anywhere in the codebase.

Confirmed via live log: `grep`ing `~/exo.log` for the actual PP-DSpark
usage log line (`"PP speculation using DSpark"`) found **zero matches**
across the session, while the harmless module-attachment log line
(`"DSpark ctx warmed"`) fires on every request — confirming the module
loads and warms its context buffers but its actual speculative decode
loop is never invoked under TP.

**Conclusion: decode is genuinely plain autoregressive (1 real forward
pass per generated token) in production tonight.** The roofline's core
assumption was already correct — no adjustment needed. Real observed
decode throughput variance (stdev 0.052 tok/s across 3 clean-baseline
runs) is also consistent with simple autoregressive decode, not
speculative accept/reject variance, as a secondary corroborating signal.

**Real, separate finding worth flagging**: `EXO_DSV4_DSPARK=1` and
`EXO_DSV4_MTP_EAGLE_K=8` are live in production's `start_cluster.sh`
default config but have zero effect under the TP topology actually in
use — they are dormant flags for this deployment. Not a bug (the code
correctly no-ops rather than crashing or silently doing something
wrong), but worth knowing: these flags are NOT what's providing any
speculative-decode benefit tonight, if anyone reading the env var list
assumed otherwise.

## Check #2: is the bytes/active-param ratio correct?

The cluster's own `/state` endpoint model card reports
`"quantization": "fp8"` — a coarse, single-word tag. Taken literally,
this would suggest 1.0 bytes/param (pure FP8), which would change the
roofline's active-weight-bytes-per-token calculation materially.

**Checked against the real on-disk model size, also from `/state`**:
`storageSize.inBytes = 166,878,536,440` for the full 284B-parameter
model. Real bytes/param: `166,878,536,440 / 284,000,000,000 = 0.5876`
— between pure FP4 (0.5) and pure FP8 (1.0), confirming a genuinely
MIXED-precision scheme despite the coarse `"fp8"` label (which is very
likely a simplified/dominant-dtype tag applied by the model-card
metadata system, not a literal claim that every tensor is FP8).

**This is exactly the ratio the original roofline calculation already
used** (0.588 bytes/param, stated explicitly in
`docs/decode-roofline-dispatch-bound-2026-08-21.md`'s own math) — the
original calculation was already pulling this from the real measured
on-disk size, not guessing or assuming a label. No correction needed.

## Conclusion

**Both roofline inputs checked and confirmed correct.** The "~12% of
theoretical bandwidth-bound peak" headroom finding from earlier tonight
stands as previously stated — no adjustment warranted. This closes out
the queued sanity-check with a real, verified answer (not just "looks
fine") rather than leaving it as an unconfirmed assumption.

One incidental, real, and separate finding surfaced during this check:
`EXO_DSV4_DSPARK`/`EXO_DSV4_MTP_EAGLE_K` are dormant under the TP
topology tonight's production runs. Flagging for anyone who assumes
those flags are providing decode-time acceleration — they currently are
not.
