# EXO_DSV4_FUSED_SOFTMAX: negative result, real correctness break — 2026-08-21 (session 2, part 5)

## Lever tested

`EXO_DSV4_FUSED_SOFTMAX=1` (`mlx-lm/mlx_lm/models/deepseek_v4.py`, added
"OPT-12" 2026-07-14). Replaces the DSv4 sparse/pooled attention's unfused
logsumexp+logaddexp+exp softmax chain with one custom Metal kernel
(`_get_fused_softmax_kernel` / `_fused_softmax_inner`) that computes
attention weights directly over the concatenated local+pooled score
space, eliminating intermediate materialization. Code comment explicitly
flags: "default 0 — needs A/B validation". Never tested before tonight.

## Method

1. Relaunched with `EXO_DSV4_MOE_FUSED_GATE_UP=1 EXO_DSV4_FUSED_SOFTMAX=1`
   (gate+up fusion kept ON as an already-validated baseline; softmax
   fusion is the new variable under test). Verified both flags live via
   `ps aux` on both nodes.
2. Confirmed the fused kernel was actually dispatching (not silently
   falling back) via the code's own built-in file-toggle diagnostics:
   `touch /tmp/dsv4_fused_dispatch` (per-dispatch counter) and
   `touch /tmp/dsv4_fused_debug` (per-call condition log). Both confirmed
   the fused path firing at L=128 chunks during a 100K-context prefill
   (`/tmp/dsv4_fused_dispatch_count` reached 5712 lines).
3. Ran the standard 100K-context needle-in-haystack correctness test.
   Result: **needle FAIL** — model produced a confused, repetitive,
   degenerate response fixating on the wrong topic ("garbage collection"
   instead of the real needle "project Nightingale" / secret code),
   never found the actual embedded secret, output looped the same
   sentence 3+ times before hitting `finish_reason: length`.
4. Isolated the variable: reran the IDENTICAL needle prompt with
   `EXO_DSV4_FUSED_SOFTMAX` reverted to default (0) and
   `EXO_DSV4_MOE_FUSED_GATE_UP=1` held constant. Result: **needle PASS**
   — clean, correct retrieval (`FALCON-MERCURY-7749`), coherent
   single-pass reasoning, normal `finish_reason: stop`.

## Result: confirmed real correctness regression, isolated to this flag

The same prompt, same context depth, same gate+up fusion state — only
`EXO_DSV4_FUSED_SOFTMAX` differed between the failing and passing runs.
This rules out noise, an unrelated bug, or interaction with the gate+up
fusion (which was constant across both runs and independently validated
clean in the prior test). The fused softmax kernel produces a genuine
quality regression at real production context depth (100K), not just a
narrow edge case — the model's attention output is wrong enough to
derail multi-step reasoning entirely.

## Conclusion

**Do not enable `EXO_DSV4_FUSED_SOFTMAX`.** The "needs A/B validation"
warning in the code comment was correct to be there, and the validation
now says no. This is consistent with the neighboring
`EXO_DSV4_SPARSE_FUSED_SDPA` gate's OWN code comment (default OFF since
2026-07-07: "the model-level equivalence gate measured worst |dlogit|
0.141 with 5/60 argmax flips") — that flag documented its own quality
regression and stayed off; this test shows `EXO_DSV4_FUSED_SOFTMAX`
belongs in the same category. Both touch the same sparse/pooled
attention softmax internals; a Metal kernel reimplementation of a
multi-step softmax reduction (local scores + pooled scores + sink value,
merged into one SIMD-group reduction) is a plausible place for a subtle
numerical or masking bug to hide that a naive dispatch-count/latency
check would never catch — only an actual needle/quality check surfaces
it, which is exactly why the standing rule (never quote t/s without
showing real generated text) matters.

Reverted to `EXO_DSV4_FUSED_SOFTMAX` unset (default OFF) as part of this
same test cycle — confirmed via the passing needle rerun above.
`EXO_DSV4_MOE_FUSED_GATE_UP=1` remains the only validated-safe new lever
from tonight's session.

## Note: not yet cleaned up

The diagnostic toggle files (`/tmp/dsv4_fused_dispatch`,
`/tmp/dsv4_fused_debug`) were created on both nodes during this test.
They are local-filesystem file-existence toggles (not env vars, not
persisted across reboot) and default to no-ops when absent — harmless
if left, but should be removed with `rm -f` on both nodes before
considering this investigation fully closed out, since they add a small
per-call `os.path.exists()` check that's pure overhead once diagnosis is
done.
