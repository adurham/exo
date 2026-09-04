# Round 7 — Arm Z (EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0) + biA reference capture

Boot: fresh relaunch via `tmux new-session -d -s r7z ... DSV4_KV_CACHE_BITS=0 EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0 ./start_cluster.sh`.
Launch log: `/tmp/r7z_launch.log`. Reached `HEALTHY! (Nodes: 2, Identities: 2)` then `READY (2/2)`.
Commit synced on both nodes: `7e68ecbc6` (local HEAD, confirmed via "Nodes synchronized on commit 7e68ecbc6.").

## Step 1 — Relaunch

Pre-relaunch teardown confirmed ZERO exo processes on both nodes (empty `ps` grep, non-error exit).

Relaunch completed cleanly through dashboard build, dependency sync, Rust rebuild, and RDMA/Thunderbolt verification on both nodes, ending with:

```
Waiting for cluster to stabilize...... HEALTHY! (Nodes: 2, Identities: 2)
...
Auto-placing DeepSeek V4 Flash (deepseek-ai/DeepSeek-V4-Flash-0731) across both Studios via RDMA...
Waiting for 2 DeepSeek V4 runner(s) to become Ready......................... READY (2/2)
```

## Step 2 — Verify the arm (mandatory gate)

Real runner pids found via `ps -eo pid,command | grep '.venv/bin/python -m exo -v'`:
- 192.168.86.201: pid **82663**
- 192.168.86.202: pid **92765**

`ps eww <pid> | tr ' ' '\n' | grep -E 'EXO_BATCHED_PREFILL_RENDEZVOUS_MS|EXO_DSV4_BATCHED_PREFILL|MLX_STEEL_BATCH_INVARIANT|EXO_SPECULATIVE_GAMMA|EXO_DSV4_MTP='`

**201 (pid 82663):**
```
EXO_DSV4_BATCHED_PREFILL=1
EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0
EXO_SPECULATIVE_GAMMA=3
EXO_DSV4_MTP=1
MLX_STEEL_BATCH_INVARIANT=1
```

**202 (pid 92765):**
```
EXO_DSV4_BATCHED_PREFILL=1
EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0
EXO_SPECULATIVE_GAMMA=3
EXO_DSV4_MTP=1
MLX_STEEL_BATCH_INVARIANT=1
```

**GATE PASSED** — both nodes match required arm exactly (`EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0`, `EXO_DSV4_BATCHED_PREFILL=1`, `MLX_STEEL_BATCH_INVARIANT=1`, `EXO_SPECULATIVE_GAMMA=3`).

## Step 3 — TTFT, two instruments

### Set 1: `Z_ttft` — 2000-token prompt, max-tokens 200, n=5

| tag | prefill_s (ms) | prompt_tokens | prefix_cache_hit |
|---|---|---|---|
| Z_ttft_r1 | 8470.0 | 2377 | none |
| Z_ttft_r2 | 6670.0 | 2285 | none |
| Z_ttft_r3 | 7720.0 | 2285 | none |
| Z_ttft_r4 | 8330.0 | 2354 | none |
| Z_ttft_r5 | 8020.0 | 2239 | none |

**Median: 8020.0 ms — Range: 6670.0–8470.0 ms**

(`decode_sample_trustworthy` is `false` for all — expected, TTFT-only measurement.)

### Set 2: `Z_short` — 20-token prompt, max-tokens 16, n=10

| tag | prefill_s (ms) | prompt_tokens | prefix_cache_hit |
|---|---|---|---|
| Z_short_r1 | 1760.0 | 234 | none |
| Z_short_r2 | 1460.0 | 230 | none |
| Z_short_r3 | 1410.0 | 226 | none |
| Z_short_r4 | 1430.0 | 226 | none |
| Z_short_r5 | 1580.0 | 234 | none |
| Z_short_r6 | 1710.0 | 232 | none |
| Z_short_r7 | 1420.0 | 226 | none |
| Z_short_r8 | 1620.0 | 232 | none |
| Z_short_r9 | 1560.0 | 230 | none |
| Z_short_r10 | 1450.0 | 222 | none |

**Median: 1510.0 ms — Range: 1410.0–1760.0 ms**

(`decode_sample_trustworthy` is `false` for all — expected.)

## Step 4 — Arm A decode (steel-BI opening bracket), 79K-token prompt, max-tokens 1200

1 warmup (discarded) + 3 measured. Each rep took several minutes (prefill ~200s alone). All 3 measured reps passed validity on first attempt (no re-runs needed):
`decode_sample_trustworthy==true`, `prompt_tokens>=85000`, `finish_reason=='length'`, `prefix_cache_hit=='none'`.

Warmup (discarded): `A_warmup` — prompt_tokens 84199, server_generation_tps 34.70 (not counted in stats).

| rep | server_generation_tps | prompt_tokens | completion_tokens | trustworthy | finish_reason | needle_hit | prefix_cache_hit |
|---|---|---|---|---|---|---|---|
| A_r1 | 29.7292 | 90402 | 1200 | true | length | true | none |
| A_r2 | 31.6076 | 86857 | 1200 | true | length | true | none |
| A_r3 | 30.7791 | 85086 | 1200 | true | length | true | none |

**Median server_generation_tps: 30.7791 — Range: 29.7292–31.6076**

## Step 5 — Byte-identity REFERENCE capture (biA)

```
python3 tmp/perf-campaign-2/round5/identity_gate.py --capture --tag biA --out tmp/perf-campaign-2/round7/results/identity_biA.json
python3 bench/long_decode_probe.py 4000 --max-tokens 300 --run-id r7fixed4k --tag biA_id4k --out tmp/perf-campaign-2/round7/results/biA_id4k.json
python3 bench/long_decode_probe.py 79000 --max-tokens 300 --run-id r7fixed89k --tag biA_id89k --out tmp/perf-campaign-2/round7/results/biA_id89k.json
```

- `identity_biA.json`: captured **5 prompts**, 0 errors (`sys_primary_colors`, `sys_capital_france`, `sys_count_to_five`, `tool_git_status`, `tool_read_config`). File confirmed written (11421 bytes).
- `biA_id4k.json` (run-id `r7fixed4k`, below 8192 verify-batch threshold): prompt_tokens 3811, content_chars **717**, needle_hit **true**. File confirmed written (2305 bytes).
- `biA_id89k.json` (run-id `r7fixed89k`, above 8192 verify-batch threshold): prompt_tokens 74455, content_chars **0** (all output in reasoning_chars=1298), needle_hit **true**. File confirmed written (2120 bytes).

All three deliverable files verified present via `ls -la`.

## Step 6 — Clean-logs check

Live log location discovered per node:
```
ssh 192.168.86.201 "ls -t ~/repos/exo/*.log 2>/dev/null | head; ls -t ~/exo.log 2>/dev/null"
ssh 192.168.86.202 "ls -t ~/repos/exo/*.log 2>/dev/null | head; ls -t ~/exo.log 2>/dev/null"
```
→ no logs at `~/repos/exo/*.log`; live log is `~/exo.log` on both nodes.

Grep executed:
```
ssh 192.168.86.201 "tail -400 ~/exo.log | grep -iE 'error|traceback|disagree|mismatch|task set|rank'"
ssh 192.168.86.202 "tail -400 ~/exo.log | grep -iE 'error|traceback|disagree|mismatch|task set|rank'"
grep -iE 'error|traceback|disagree|mismatch|task set|rank' /tmp/r7z_launch.log
```

**Results:**
- 201 `~/exo.log` (tail -400): **no matches** (grep exit 1).
- 202 `~/exo.log` (tail -400): **no matches** (grep exit 1).
- `/tmp/r7z_launch.log`: 2 matches, both are dashboard build artifact filenames (`.svelte-kit/output/server/entries/fallbacks/error.svelte.js`), not actual errors — no genuine error/traceback/disagreement/mismatch/rank-conflict lines found anywhere.

**Logs clean.**

---

## Final Summary

- **TTFT (2000-tok) prefill**: median **8020.0 ms**, range **6670.0–8470.0 ms** (n=5)
- **TTFT (20-tok, short) prefill**: median **1510.0 ms**, range **1410.0–1760.0 ms** (n=10)
- **Arm A decode (~85-90K prompt, 1200-tok completion) server_generation_tps**: median **30.7791**, range **29.7292–31.6076** (n=3, all valid on first attempt)
- **Logs**: clean on both nodes and in the launch log; no errors, tracebacks, disagreements, mismatches, or rank conflicts found across 20+ requests this boot.

No deltas vs. baseline computed and no ship/hold judgment made, per instructions — that is the PM's call.
