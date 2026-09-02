# A3 — LCP re-probe, pad removed (round 3)

**Date:** 2026-09-03 · **Repo:** exo · **Runner:** mid-tier subagent (this file)
**Scope per pre-registration:** same 54 turn pairs, same tokenizer, same variant A/B
bounding, same LCP definition, same percentile conventions — the ONLY change is the
treatment of the `reasoning_content` space-pad in the re-fed assistant message.

## 0. The one-paragraph answer

With the pad removed, **54 of 54 turn pairs reproduce the decode output
byte-identically** (LCP_coverage = 1.000). Distribution collapses from round-2's
strictly-bimodal 47×1.000 / 7×0.000 to a single mass at 1.000 with **zero pairs at
0.000 and zero in between**. The 7 previously-zeroing no-reasoning turns (which all
carried the client's single-space pad) go to 1.000 under both post-fix shapes
(omitted field AND empty string). The two round-2 null no-reasoning pairs stay 1.000.
**Under the pre-registered bands this re-probe PASSES: median 1.000 ≥ 0.90 and
p10 1.000 ≥ 0.70.** Read with the registered caveat: the decode side is an optimistic
proxy, so this is best expressed as **"FAIL ruled out under optimistic reconstruction"**
— the verdict framing is the PM's call.

## 1. Band evaluation (quoted verbatim from round 2)

- **PASS:** median LCP_coverage ≥ 90% AND p10 ≥ 70%
- **FAIL:** median < 60% OR p10 < 20%
- **INDETERMINATE** (60–90% median, or p10 20–70%): scored as PRACTICAL-FAIL

**Result:** median 1.000 ≥ 0.90 ✓ and p10 1.000 ≥ 0.70 ✓ → **PASS bands fire.**

## 2. Full distribution (n = 54)

| statistic | value |
|---|---:|
| min | 1.0000 |
| **p10** (linear interpolation) | **1.0000** |
| p10 (nearest-rank) | 1.0000 |
| p10 (inclusive) | 1.0000 |
| p25 | 1.0000 |
| median | 1.0000 |
| mean | 1.0000 |
| max | 1.0000 |

Counts: **0 pairs at 0.000, 54 pairs at 1.000, 0 pairs in between.**

## 3. Token-weighted coverage of bucket B

| run | variant A | variant B (lower bound) |
|---|---:|---:|
| round-2 original | 0.9623 | 0.8839 |
| round-3 PAD (replica of round-2) | 0.9623 | 0.8839 |
| **round-3 OMIT (primary)** | **1.0000** | **0.9215** |
| round-3 EMPTY (sensitivity) | 1.0000 | 0.9215 |

## 4. Paired before/after — the 9 no-reasoning pairs (7 padded + 2 null)

| pair | decomp_tok | R2 cov | R2 cause | R3 OMIT cov | R3 EMPTY cov |
|---|---:|---:|---|---|---|
| 34->35 | 1003 | 0.000 | reasoning-echo-space-pad | 1.000 | 1.000 |
| 38->39 (null) | 64 | 1.000 | full-match | 1.000 | 1.000 |
| 48->49 | 98 | 0.000 | reasoning-echo-space-pad | 1.000 | 1.000 |
| 50->51 | 118 | 0.000 | reasoning-echo-space-pad | 1.000 | 1.000 |
| 51->52 | 99 | 0.000 | reasoning-echo-space-pad | 1.000 | 1.000 |
| 53->54 | 99 | 0.000 | reasoning-echo-space-pad | 1.000 | 1.000 |
| 56->57 | 99 | 0.000 | reasoning-echo-space-pad | 1.000 | 1.000 |
| 63->64 (null) | 43 | 1.000 | full-match | 1.000 | 1.000 |
| 84->85 | 46 | 0.000 | reasoning-echo-space-pad | 1.000 | 1.000 |

All 7 padded → 1.000; both null rows stay 1.000. Decode tokens are unchanged
(recorded `completion_tokens`), so the pad removal affects only the re-feed side.

## 5. Sanity checks (round 2's checks, re-run)

- **Proxy decode tokens:** sum = 41,413 vs 41,414 recorded `completion_tokens` (Δ=−1
  total; max per-pair error 0.02%). ✓ matches round 2.
- **Bucket B re-derivation:** unchanged from round 2 (21.91% of uncached; 41,414
  tokens from `requests.jsonl`). ✓
- **Position invariant:** `len(prompt(n)) − common_prefix == 0` on all 54 pairs. ✓
- **Baseline replica:** running the SAME script with `PAD_HANDLING='pad'` reproduces
  round-2's exact distribution (min 0.000, p10 0.000, median 1.000, mean 0.8704,
  47×1.000 / 7×0.000) — methodology lock confirmed, no unstated change crept in.

## 6. The only change made

The round-3 script is a byte-for-byte copy of round-2 `probe_lcp.py` with (a) the
`reasoning_content` pad handling replaced in `wire_messages_for_call` and (b) the
output filename changed. No other logic changed. Verbatim diff:

```diff
--- /Users/adam.durham/repos/exo/tmp/prefill-round2-20260902/probe_lcp.py	2026-09-02 16:48:44
+++ /Users/adam.durham/repos/exo/tmp/prefill-round3-20260902/a3_probe_lcp.py	2026-09-02 17:23:23
@@ -35,8 +35,16 @@
 SNAP = ('/Users/adam.durham/.cache/huggingface/hub/models--deepseek-ai--'
         'DeepSeek-V4-Flash-0731/snapshots/'
         '7872f01b1d1fe23eabc4c98b48bffcef5a386062')
-OUT_JSON = ('/Users/adam.durham/repos/exo/tmp/prefill-round2-20260902/'
-            'findings/lcp_probe.json')
+
+# ----------------------------------------------------------- round-3 pad fix
+# THE ONLY PERMITTED CHANGE (methodology lock). See wire_messages_for_call.
+#   'omit'  (primary)   : field OMITTED on no-reasoning turns (post-fix client)
+#   'empty' (sensitivity): field present as "" (empty string)
+#   'pad'   (baseline)  : reproduces round-2 exactly (verify delta == 0)
+PAD_HANDLING = 'omit'
+
+OUT_JSON = ('/Users/adam.durham/repos/exo/tmp/prefill-round3-20260902/'
+            f'findings/lcp_probe_round3_{PAD_HANDLING}.json')
 SESSION = '20260901_120301_93ad7b'
 
 BOS = '<｜begin▁of▁sentence｜>'
@@ -148,6 +156,16 @@
     slices[q] = kept
 
 # ------------------------------------------------------ wire message builder
+# ROUND 3 - THE ONLY PERMITTED CHANGE (methodology lock):
+# Round 2 stored the client's space-pad `reasoning_content=' '` on no-reasoning
+# turns and re-fed it verbatim, which put one spurious token at the head of the
+# re-fed region and zeroed LCP on 7 pairs. The post-fix client OMITS the field
+# when the model emitted no reasoning (round 2's own natural experiment: the 2
+# null rows scored 1.000). We model that fix here.
+#   PAD_HANDLING='omit'  (primary)  : field OMITTED on no-reasoning turns
+#   PAD_HANDLING='empty' (sensitivity): field present as "" (empty string)
+#   PAD_HANDLING='pad'   (baseline) : reproduces round-2 exactly (re-verify delta=0)
+# (This block is documentation; the effective value is set at the top of the file.)
 def wire_messages_for_call(seq):
     """Cumulative OpenAI-format message list Hermes sent for call `seq`."""
     wm = [{'role': 'system', 'content': SYSTEM_PROMPT}]
@@ -166,8 +184,20 @@
                                 'function': {'name': fn.get('name'),
                                              'arguments': fn.get('arguments')}})
                 a = {'role': 'assistant', 'content': m['content'] or ''}
-                if m['reasoning_content'] is not None:
-                    a['reasoning_content'] = m['reasoning_content']
+                model_reasoned = bool(m['reasoning'] and str(m['reasoning']).strip())
+                rc = m['reasoning_content']
+                if rc is not None and model_reasoned:
+                    # model actually reasoned -> echo verbatim (unchanged from round 2)
+                    a['reasoning_content'] = rc
+                elif rc == ' ' and not model_reasoned:
+                    # client space-pad on a no-reasoning turn -> the only change
+                    if PAD_HANDLING == 'empty':
+                        a['reasoning_content'] = ''
+                    elif PAD_HANDLING == 'omit':
+                        pass  # field omitted entirely (post-fix client)
+                    else:  # 'pad' baseline reproduces round-2 exactly
+                        a['reasoning_content'] = rc
+                # rc is None on no-reasoning turns (the 2 null rows): omit, as round 2 did
                 if tcs:
                     a['tool_calls'] = tcs
                 wm.append(a)

```

## 7. Honesty ledger

- **MEASURED:** all 54 per-pair ids, the distribution, the 9-pair table, token-weighted
  coverage under both variants, the proxy-sum and bucket-B sanity numbers.
- **PROXY (optimistic):** raw decode token ids are not persisted anywhere; decode text
  is reconstructed from stored reasoning/content/tool_calls and re-tokenized with the
  real DSv4 tokenizer. Canonical re-tokenization cannot reproduce decode-time BPE
  segmentation quirks, and the tool-call region matches the re-feed by construction
  (both render from the same parsed arguments). Bias direction is optimistic — a pass
  here does NOT prove the mechanism works in production. **This is best read as "FAIL
  ruled out under optimistic reconstruction", not as proof of the fix.**
- **NOT APPLIED:** the ship/no-ship verdict. Per pre-registration, the re-probe result
  maps to CONDITIONAL-GO (conditions C1 de-optimize-the-proxy and C2 fleet-value) and
  the verdict framing is the PM's call.

## 8. Files

- `a3_probe_lcp.py` — modified probe (primary mode is `PAD_HANDLING='omit'`)
- `findings/lcp_probe_round3.json` — canonical per-pair array (primary OMIT run, 54 pairs)
- `findings/lcp_probe_round3_omit.json` / `_empty.json` / `_pad.json` — the three runs
