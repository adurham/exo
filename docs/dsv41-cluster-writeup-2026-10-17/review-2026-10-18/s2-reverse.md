[BLOCK] WRITEUP.md:842 "agree to within 0.3% on every deep row on the unrounded values"
SRC: evidence-live.md:169 "| d10k | 10004 | 0 | 152 | `stop` | 36.37 | 275.1 | 4.80 | **31.4** | 4.83 | **31.3** | 41.34 |"
FIX: "agree to within 0.3% on the 100K-300K rows; 10K implies 0.6% (>=0.02 s gap on 4.8 s) and 50K 0.4%"
[FIX] WRITEUP.md:885 "its 6.54 s decode window (30.4 tok/s) is server-measured"
SRC: evidence-live.md:129 "elapsed=667.9s tok_s=0.3 reason=length" (200 / (667.9-661.36) = 30.6)
FIX: "its 6.54 s decode window (30.6 tok/s = 200/6.54, not 30.4) is server-measured"
[NOTE] WRITEUP.md:902 "decode arrives in speculative rounds of ~3 tokens at every"
SRC: evidence-live.md:864 "| client decode window | **33.87 s** |" (the 1M run was streamed; no round data exists)
FIX: "decode arrives in speculative rounds of ~3 tokens at every depth measured for it (1K-300K; the 1M runs have no round data)"
[NOTE] WRITEUP.md:916 "per-round cost of unidentified origin (residency / spill is a named candidate, not proven),"
SRC: evidence-live.md:211 "A constant ~3 tokens/round with the round cost rising 5.3x is a per-round memory-side cost (KV read / residency)"
FIX: "per-round memory-side cost whose owner is unidentified (KV read vs residency is a named candidate, not proven),"
[FIX] WRITEUP.md:941 "Two caveats keep"
SRC: WRITEUP.md:944 "bands differ (~68 tok/s vs 98.1 at the deepest, a cross-attempt swing of the size the repo's"
FIX: "Three caveats keep"
[FIX] WRITEUP.md:958 "`reuse=299008` -- those rows are attempt 1's partial prefill of the same prompt, left resident"
SRC: WRITEUP.md:943 "that run carried a partial cache of its own (~98K, §9.4)"
FIX: "`reuse=299008` -- those rows are a resident prefix of the prompt left by attempt 1, whose own cache is unreconciled"
[FIX] WRITEUP.md:708 "The last deltas on the wire, verbatim, in order:"
SRC: evidence-live.md:302 'delta.reasoning_content = ":\n\n"' (six earlier deltas precede the quoted seven)
FIX: "The last deltas on the wire, verbatim, in order (the six before them, up to the newline delta, are elided):"
[NOTE] WRITEUP.md:958 "`reuse=299008` -- those rows are attempt 1's partial prefill"
SRC: evidence-live.md:704 "matches a resident conversation on 299946 rows"
FIX: "`reuse=299008` -- those rows are a resident prefix (the prefill line reports a 299,946-row match)"
CHECKED: 88
END S2
