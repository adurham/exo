[BLOCK] evidence-live:L939 "Attempt 1, which had only ~98 K cached, is the closer proxy: it reached 802,816 of 1,000,003 tokens"
SRC: evidence-live:L701 "prefill reached **802,816 / 1,000,003 tokens** (80.3%)"
FIX: Drop "~98 K cached": with ~98 K cached the figures are 802,816/~902 K (89%) and ~99 K remaining, not 80.3% and ~197 K.

[BLOCK] evidence-live:L704 "matches a resident conversation on 299946 rows"
SRC: evidence-live:L859 "`prompt_tokens_details.cached_tokens` | **299,008**"
FIX: Change 299946 to 299,008 so it matches the API field and the server line's reuse=299008.

[BLOCK] evidence.md:L242 "batched-ON median 36.63 vs spec-OFF 27.15 (+36.71%), CI [+28.26,+51.02]%, 12/12 pairs"
SRC: evidence.md:L48 "36.63 tok/s median, fixed window ... vs 27.15 spec-OFF = +36.71%, 12/12 pairs"
FIX: 36.63/27.15 = +34.92%. Show +34.9% in both rows, or label +36.71% as a paired-difference/ratio statistic.

[NOTE] evidence-live:L133 "= **6.54 s** for 200 tokens = **30.4 tok/s**"
SRC: evidence-live:L168 "| d1k | 1003 | 0 | 82 | `stop` | 3.93 | 255.2 | 2.35 | **34.5** |" (34.5 = 81/2.35, i.e. n-1)
FIX: 200/6.54 = 30.6 tok/s; either display 30.6 or say "199 decode tokens" (table convention) at L133 and L186.

[NOTE] evidence.md:L243 "ON median 28.30 (IQR 27.44-29.90) vs OFF median 27.49 (+1.87%, CI [-0.82,+9.45] -> FAIL)"
SRC: evidence.md:L248 "median 28.44 vs stripped-OFF 24.19 = +17.57%" (same section: ratio of medians)
FIX: 28.30/27.49 = +2.95%, not +1.87%; correct the percentage or show the paired statistic it came from.

[NOTE] evidence.md:L292 "random letter-strings 28.66 (-12.6%)"
SRC: evidence.md:L292 "natural high-entropy prose 33.58 = pooled repetitive 33.58"
FIX: 28.66 vs 33.58 is -14.7%, not -12.6%; correct the delta.

[NOTE] evidence.md:L185 "PP hides 55.8 s drain outside the loop"
SRC: evidence.md:L183 "PP 277.0 (TTFT 255.06 s) | 70,656 real"
FIX: 255.06 - 70,656/372 = ~65 s, and the TTFT gap vs TP is 62.6 s; either figure, not 55.8 s.

CHECKED: 34
END P3A
