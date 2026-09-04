# Round 7 — Arm `MLX_STEEL_BATCH_INVARIANT=0` (bi0) — Byte-Identity Gate Report

## Relaunch

Teardown of both nodes confirmed 0 exo processes before relaunch. Relaunched via:
```
tmux new-session -d -s r7bi0 'cd ~/repos/exo && DSV4_KV_CACHE_BITS=0 EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0 MLX_STEEL_BATCH_INVARIANT=0 ./start_cluster.sh 2>&1 | tee /tmp/r7bi0_launch.log'
```
Prompted with "local HEAD not on origin/main" (local `7e68ecbc6` vs `origin/main` `b5fd90c67`) — answered `y` (informational only, no push, per instructions).

Launch reached:
```
Waiting for cluster to stabilize...... HEALTHY! (Nodes: 2, Identities: 2)
...
Waiting for 2 DeepSeek V4 runner(s) to become Ready......................... READY (2/2)
```
Both nodes synchronized on commit `7e68ecbc6`.

## STEP 2 — Arm verification (mandatory, both nodes PASSED)

Runner PIDs: node .201 (macstudio-m4-1) pid **347**; node .202 (macstudio-m4-2) pid **10834** — both `.venv/bin/python -m exo -v`.

`ps eww <pid> | tr ' ' '\n' | grep -E 'MLX_STEEL_BATCH_INVARIANT|EXO_BATCHED_PREFILL_RENDEZVOUS_MS|EXO_DSV4_VERIFY_ROWSEQ_VEC|EXO_DSV4_VERIFY_BATCH|EXO_SPECULATIVE_GAMMA'`

**Node .201 (pid 347):**
```
EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0
EXO_SPECULATIVE_GAMMA=3
EXO_DSV4_VERIFY_ROWSEQ_VEC=1
EXO_DSV4_VERIFY_ROWSEQ_VEC_ROWSDPA=3
MLX_STEEL_BATCH_INVARIANT=0
EXO_DSV4_VERIFY_BATCH=1
EXO_DSV4_VERIFY_BATCH_MIN_CTX=8192
```

**Node .202 (pid 10834):**
```
EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0
EXO_SPECULATIVE_GAMMA=3
EXO_DSV4_VERIFY_ROWSEQ_VEC=1
EXO_DSV4_VERIFY_ROWSEQ_VEC_ROWSDPA=3
MLX_STEEL_BATCH_INVARIANT=0
EXO_DSV4_VERIFY_BATCH=1
EXO_DSV4_VERIFY_BATCH_MIN_CTX=8192
```

Required values confirmed on BOTH nodes: `MLX_STEEL_BATCH_INVARIANT=0` ✓, `EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0` ✓, `EXO_SPECULATIVE_GAMMA=3` ✓.
Recorded (expected 1/1): `EXO_DSV4_VERIFY_ROWSEQ_VEC=1` ✓, `EXO_DSV4_VERIFY_BATCH=1` ✓.

## STEP 3 — Capture (bi0 side)

- `identity_gate.py --capture --tag bi0` → `tmp/perf-campaign-2/round7/results/identity_bi0.json` (5 prompts captured, 0 errors).
- `long_decode_probe.py 4000 --max-tokens 300 --run-id r7fixed4k --tag bi0_id4k` → `bi0_id4k.json`, `prompt_tokens=3811` (matches reference), `finish_reason=length`, `needle_hit=true`.
- `long_decode_probe.py 79000 --max-tokens 300 --run-id r7fixed89k --tag bi0_id89k` → `bi0_id89k.json`, `prompt_tokens=74455` (matches reference), `finish_reason=length`, `needle_hit=true`.

## STEP 4 — THE BYTE-IDENTITY GATE

### (i) `identity_gate.py --compare identity_biA.json identity_bi0.json` — full output:

```
[sys_primary_colors] DIFFERS at byte offset 13 (A=2652B, B=2582B)
...A=b'{"content": "", "finish_reason": "length", "reasoning'...
...B=b'{"content": "The three primary colors", "finish_reaso'...
[sys_capital_france] BYTE-IDENTICAL (606 bytes)
[sys_count_to_five] DIFFERS at byte offset 13 (A=468B, B=616B)
...A=b'{"content": "1, 2, 3, 4, 5", "finish_reason": "stop",'...
...B=b'{"content": "One, two, three, four, five.", "finish_r'...
[tool_git_status] BYTE-IDENTICAL (504 bytes)
[tool_read_config] BYTE-IDENTICAL (264 bytes)

FAIL: one or more of 5 prompts differ between biA and bi0
```
**Result: FAIL.** 2 of 5 prompts differ (`sys_primary_colors`, `sys_count_to_five`); 3 of 5 byte-identical.

### (ii) 4K comparison (`biA_id4k.json` vs `bi0_id4k.json`)

- `prompt_tokens`: A=3811, B=3811, expected=3811 — **MATCH** ✓
- `run_id`: A=`r7fixed4k`, B=`r7fixed4k` — **MATCH** ✓
- Byte-for-byte comparison of `reasoning_content + content` (exact string equality, no normalization):
  - **NOT IDENTICAL**
  - First differing character index: **8**
  - len(A)=1469, len(B)=1331
  - A context (from idx 8): `'We need to answer two tasks. First, state the secret authorization code for project Nightingale exactly as i'`
  - B context (from idx 8): `'We need answer user. Need parse document. Need state secret code exactly: "FALCON-MERCURY-7749" appears in o'`
  - The two reasoning traces diverge almost immediately (character 8) and stay divergent through materially different content/length.

### (iii) 89K comparison (`biA_id89k.json` vs `bi0_id89k.json`)

- `prompt_tokens`: A=74455, B=74455, expected=74455 — **MATCH** ✓
- `run_id`: A=`r7fixed89k`, B=`r7fixed89k` — **MATCH** ✓
- Byte-for-byte comparison of `reasoning_content + content`:
  - **NOT IDENTICAL**
  - First differing character index: **442**
  - len(A)=1298, len(B)=1260
  - A context (~100 chars before/after idx 442): `'tify O(N) observers, attach/detach O(1) if list, space O(N). B-trees: search/insert/delete O(log n) average/worst, space O(n), node branching factor, disk I/O. Need essay. Need ensure no extra? We can'`
  - B context (~100 chars before/after idx 442): `'tify O(N) observers, attach/detach O(1) if list, space O(N). B-trees: search/insert/delete O(log n) time, space O(n), disk I/O optimized, high branching factor, etc. Need essay 900+ words. Need not me'`
  - Text matches through the shared prefix up to idx 442, then diverges in wording/phrasing for the remainder.

## HARD STOPPING RULE — applied

All three comparisons show a **MISMATCH**:
1. `identity_gate.py --compare` → FAIL (2/5 prompts differ)
2. 4K byte comparison → NOT IDENTICAL (diverges at char 8)
3. 89K byte comparison → NOT IDENTICAL (diverges at char 442)

Per the pre-registered stopping rule: **STOPPED. STEP 5 (decode benchmarking) was NOT run.** The knob failed its correctness gate; running decode throughput measurements on a configuration that already failed byte-identity would have wasted ~30 minutes decorating a decision already made.

## Summary

1. **Identity comparisons**: ALL THREE FAILED.
   - (i) identity_gate compare: FAIL — 2/5 prompts differ (`sys_primary_colors`, `sys_count_to_five`)
   - (ii) 4K (below 8192 threshold): NOT byte-identical — first diff at char 8
   - (iii) 89K (above 8192 threshold): NOT byte-identical — first diff at char 442
2. **STEP 5 not run** — the hard stopping rule triggered because all three identity comparisons failed. No decode benchmarks were collected for this arm.
3. **Decode median/range**: N/A — not measured.

Cluster left running as-is (READY 2/2, HEALTHY, bi0 config still live) per instructions; PM owns the final restore.
