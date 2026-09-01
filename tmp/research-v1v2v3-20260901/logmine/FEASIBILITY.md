# V2 FEASIBILITY — server-side per-cycle MTP acceptance counters for the Aug 31 2026 P11–P15 boots

**Verdict: PARTIAL — the raw acceptance data EXISTS but is UNTIMESTAMPED and UNATTRIBUTABLE to specific boots, and the existing parser (`bench/mtp_cycle_time.py`) matches ZERO lines in it.**

This is a **NOT-FEASIBLE** verdict for the V2 workstream *as specified* (per-cycle acceptance joined to per-boot/per-rep decode), because the counters cannot be attributed to the Aug 31 boots. The data is not absent — it is present but unusable for the stated purpose without a timestamp/attribution mechanism that does not exist in the retained logs.

---

## 1. The parser format (`bench/mtp_cycle_time.py`)

The parser reads ONE line per speculative cycle with a **millisecond timestamp** and a **monotonically increasing cycle counter**, then differences consecutive lines for wall-time-per-cycle. The exact regex (verbatim, lines 32–34):

```python
LINE = re.compile(
    r"(\d\d):(\d\d):(\d\d)\.(\d+).*?\[MTP\] cycles=(\d+)\s+mean_accept=([\d.]+)"
)
```

The regex **requires** a `HH:MM:SS.mmm` timestamp prefix before `[MTP] cycles=N mean_accept=X`. The `.*?` between the timestamp and `[MTP]` is non-greedy but the timestamp is mandatory. The parser also requires `c1 - c0 == 1` (consecutive cycle counters) and `0.0 < dt < 500.0` ms to build deltas (lines 70–75).

**The intended log line format** (from the skill `exo-cluster-operations`, and confirmed present in the runner stderr) is:
```
[MTP] cycles=50 mean_accept=0.640/2 hist=0:22,1:24,2:4
```
This line has **NO timestamp prefix** — it does not match the parser regex.

---

## 2. Log inventory (verbatim `ls -la`, both nodes)

### adams-mac-studio-m4-1.local — `~/.exo/exo_log/`
```
total 173616
drwxr-xr-x   9 adam.durham  staff       288 Sep  1 12:37 .
drwxr-xr-x@ 16 adam.durham  staff       512 Aug 31 01:22 ..
-rw-r--r--   1 adam.durham  staff    539437 Aug 31 21:25 exo.2026-08-31_20-42-00_939681.log.zst
-rw-r--r--   1 adam.durham  staff    537809 Aug 31 22:04 exo.2026-08-31_21-25-50_093140.log.zst
-rw-r--r--   1 adam.durham  staff    617649 Aug 31 22:54 exo.2026-08-31_22-04-39_248306.log.zst
-rw-r--r--   1 adam.durham  staff    904294 Sep  1 00:09 exo.2026-08-31_22-54-04_689174.log.zst
-rw-r--r--   1 adam.durham  staff   2380564 Sep  1 12:37 exo.2026-09-01_00-09-25_523979.log.zst
-rw-r--r--   1 adam.durham  staff  76254543 Sep  1 15:17 exo.log
drwxr-xr-x   4 adam.durham  staff       128 May 27 09:23 runner_log
```

### adams-mac-studio-m4-2.local — `~/.exo/exo_log/`
```
total 179624
drwxr-xr-x   9 adam.durham  staff       288 Sep  1 12:37 .
drwxr-xr-x  16 adam.durham  staff       512 Aug 31 01:22 ..
-rw-r--r--   1 adam.durham  staff    202952 Aug 31 21:25 exo.2026-08-31_20-42-02_407486.log.zst
-rw-r--r--   1 adam.durham  staff    203369 Aug 31 22:04 exo.2026-08-31_21-25-51_457048.log.zst
-rw-r--r--   1 adam.durham  staff    233678 Aug 31 22:54 exo.2026-08-31_22-04-40_562493.log.zst
-rw-r--r--   1 adam.durham  staff    378345 Sep  1 00:09 exo.2026-08-31_22-54-05_862579.log.zst
-rw-r--r--   1 adam.durham  staff   6853832 Sep  1 12:37 exo.2026-09-01_00-09-26_828044.log.zst
-rw-r--r--   1 adam.durham  staff  67416100 Sep  1 15:17 exo.log
drwxr-xr-x   4 adam.durham  staff       128 May 27 09:23 runner_log
```

### `runner_log/` (both nodes)
```
total 2636224   (m4-1) / total 2713600 (m4-2)
drwxr-xr-x  4 adam.durham  staff         128 May 27 09:23 .
drwxr-xr-x  9 adam.durham  staff         288 Sep  1 12:37 ..
-rw-r--r--  1 adam.durham  staff  1337122726 Sep  1 12:38 stderr.log   (m4-1)
-rw-r--r--  1 adam.durham  staff  1376719767 Sep  1 12:38 stderr.log   (m4-2)
-rw-r--r--  1 adam.durham  staff      269138 Aug 30 11:50 stdout.log  (m4-1)
-rw-r--r--  1 adam.durham  staff      281744 Jul 18 20:59 stdout.log  (m4-2)
```

**Decompression tooling on the nodes:** `zstdcat`/`zstd` binaries NOT present; python `zstandard`/`zstd` modules NOT present (verified `ModuleNotFoundError` on both). Decompression was done **locally** (laptop has `/opt/homebrew/bin/zstd`) after read-only `scp` of the `.zst` files. This is stated explicitly per the task's requirement.

---

## 3. Log retention coverage

**Rotated `.zst` logs (decompressed) cover Aug 31 2026 20:42 → Sep 1 12:35, continuously, on both nodes.** Timestamp coverage per file (first→last timestamped line):

| file | first_ts | last_ts |
|---|---|---|
| node1 `...20-42-00` | 2026-08-31 20:42:00 | 2026-08-31 21:17:12 |
| node1 `...21-25-50` | 2026-08-31 21:25:49 | 2026-08-31 22:01:47 |
| node1 `...22-04-39` | 2026-08-31 22:04:38 | 2026-08-31 22:51:18 |
| node1 `...22-54-04` | 2026-08-31 22:54:04 | 2026-09-01 00:06:09 |
| node1 `...00-09-25` | 2026-09-01 00:09:24 | 2026-09-01 12:34:27 |
| node2 `...20-42-02` | 2026-08-31 20:42:01 | 2026-08-31 21:18:37 |
| node2 `...21-25-51` | 2026-08-31 21:25:50 | 2026-08-31 22:03:11 |
| node2 `...22-04-40` | 2026-08-31 22:04:40 | 2026-08-31 22:52:41 |
| node2 `...22-54-05` | 2026-08-31 22:54:05 | 2026-09-01 00:07:39 |
| node2 `...00-09-26` | 2026-09-01 00:09:26 | 2026-09-01 12:35:52 |

**The live `exo.log` covers Sep 1 12:37 → 15:18 (both nodes), NOT Aug 31.**

**Boot-to-log mapping (from `docs/PERFORMANCE_HISTORY.md` P13/P14/P15 pre-flight PIDs):**
- **P13 boot** (m4-1 pid 7728 / m4-2 pid 7776) → `...20-42-00` / `...20-42-02` (20:42)
- **P14 Arm A boot** (m4-1 33861 / m4-2 34716) → `...21-25-50` / `...21-25-51` (21:25)
- **P14 Arm B boot** (m4-1 55062 / m4-2 56386) → `...22-04-39` / `...22-04-40` (22:04)
- **P15 boot** (m4-1 81596 / m4-2 83149) → `...22-54-04` / `...22-54-05` (22:54)
- **P11/P12** predate the retained window (P12 relaunch was 17:46, P11 earlier); their logs are NOT retained.

**So the retained rotated logs cover P13, P14A, P14B, and P15 (Aug 31 20:42 → Sep 1 00:06). P11 and P12 are NOT covered by any retained log.**

---

## 4. MTP acceptance counter search — EXACT counts

### 4a. Rotated exo.log files (the timestamped, boot-attributable logs)

**`[MTP]` count = 0 in ALL 10 rotated logs** (both nodes, all 5 files each). Command: `grep -c "\[MTP\]" <file>`.

**`mean_accept` count = 0 in ALL 10 rotated logs.**

The only `accept`/`acceptance`/`accepted` hits in the Aug 31 rotated logs are:
- `[DSPARK-GUARD]` load-guard WARNING lines (1–2 per boot, unrelated to MTP acceptance)
- Sep 1 log: Hermes campaign prompt text (the word "acceptance" in task descriptions)

**None of these are MTP per-cycle acceptance counters.**

### 4b. Runner `stderr.log` (the 1.3 GB runner capture)

**`[MTP] cycles=` count = 59,582 on BOTH nodes** (identical). These ARE the per-cycle acceptance counters. Sample (verbatim, node1):
```
[MTP] cycles=50 mean_accept=0.640/2 hist=0:22,1:24,2:4
[MTP] cycles=100 mean_accept=1.200/2 hist=0:24,1:32,2:44
```
Last line (verbatim, node1):
```
[MTP] cycles=92 mean_accept=1.630/3 hist=0:18,1:26,2:20,3:28
```

**BUT: these lines have NO timestamp prefix.** Verified:
- `grep -cE "^\[?[0-9]{2}:[0-9]{2}:[0-9]{2}"` on stderr = **0** (no HH:MM:SS-prefixed lines)
- `grep -c "2026-08-31"` on stderr = **0**; `grep -c "2026-09-01"` = **0** (no date anywhere)
- `grep "\[MTP\]" stderr | grep -cE "[0-9]{2}:[0-9]{2}:[0-9]{2}"` = **0** (no MTP line carries a timestamp on the same line)

**The parser regex matches ZERO lines in stderr** (verified by running the exact `LINE` regex over the full 14M-line file on both nodes: `parser-regex matches in stderr: 0`).

### 4c. Timestamped `[MTP ACCEPT]` lines (different format, NOT parser-compatible)

The stderr also contains 3,217 lines of a **different, timestamped** format:
```
2026-06-08 13:42:12.893 | INFO | exo.worker.engines.mlx.speculative.mtp_batch_generator:_speculative_next:303 - [MTP ACCEPT] gamma=3 cycles=64 accept_rate=0.750 tokens/cycle=3.25 decode_tok/s=5.9
```
These are all dated **June 8 – Aug 21, 2026** (last one 2026-08-21 14:19). They do NOT cover Aug 31, and their format (`[MTP ACCEPT] ... accept_rate=... tokens/cycle=...`) does not match the parser regex either.

### 4d. Attribution of the untimestamped MTP lines

The 59,582 untimestamped `[MTP] cycles=` lines span the entire stderr (line 817 → 14,041,962). Distribution:
- 492 lines in the first 3.09M lines (the June–July timestamped region)
- **59,090 lines in the tail region AFTER the last timestamped line (13,956,833 = 2026-08-21 14:19)** — i.e. the Aug 21 → Sep 1 window, which INCLUDES Aug 31

The tail-region MTP lines are **gamma=3** (`mean_accept=X/3`), matching the P13–P15 config (`GAMMA=3`). The earlier gamma=2 lines (`mean_accept=X/2`, 485 total) are in the June–July region.

**However, because these lines carry no timestamps, they CANNOT be attributed to specific boots or to the P13/P14/P15 reps.** The stderr is a single append-only file with no boot-boundary markers (no `main_inner:345` pid lines, no date stamps in the tail). The file mtime is Sep 1 12:38, and the last timestamped content is Aug 21, so the tail MTP lines span Aug 21→Sep 1 — but there is no way to say which lines belong to which boot or which rep.

---

## 5. VERDICT

**PARTIAL — NOT FEASIBLE for the V2 workstream as specified.**

- **The per-cycle MTP acceptance counters DO exist** (59,582 lines in runner `stderr.log`, both nodes, identical), and the tail region (59,090 lines) falls in the Aug 21→Sep 1 window that includes Aug 31.
- **BUT they are untimestamped and unattributable to specific boots or reps.** The existing parser (`mtp_cycle_time.py`) matches **0** of them because it requires a `HH:MM:SS.mmm` timestamp prefix that these lines do not carry.
- **The timestamped, boot-attributable rotated exo.log files contain ZERO `[MTP]` lines** — the counters were never written to the main log during the Aug 31 boots.
- **P11 and P12 are not covered by any retained log** (their boots predate the 20:42 retention start).

**Consequence:** The V2 hypothesis (per-cycle acceptance as a boot-level state variable, joined to per-boot/per-rep decode) **cannot be tested from the retained logs**. The acceptance data exists but cannot be joined to the decode measurements because there is no timestamp bridge. A workstream that requires per-boot acceptance attribution is infeasible on this evidence.

**What WOULD be feasible (if the parent wants it):** the 59,090 tail-region MTP lines are a continuous, ordered per-cycle acceptance stream for the Aug 21→Sep 1 window. If the parent can supply a *separate* timestamped anchor (e.g. the probe-side rep start/end times from `tmp/p15-replication-20260831/`), the MTP lines could in principle be aligned by cycle-count continuity — but this is fragile and was NOT part of the specified task. The clean answer is: **the counters exist but are not boot-attributable; V2 as specified is NOT FEASIBLE.**

---

## Appendix — raw evidence files

- `raw/ls_exo_log_node1.txt`, `raw/ls_exo_log_node2.txt` — verbatim `ls -la` of `~/.exo/exo_log/`
- `raw/ls_runner_log_node1.txt`, `raw/ls_runner_log_node2.txt` — verbatim `ls -la` of `runner_log/`
- `raw/acceptance_raw_node1.txt`, `raw/acceptance_raw_node2.txt` — all 59,582 `[MTP] cycles=` lines from each node's stderr (untimestamped; header notes this)
- `raw/metal_iogpu_aug31.txt` — Metal/IOGPU grep counts (all 0) on Aug 31 logs
- `raw/memory_jetsam_aug31.txt` — memory/jetsam grep counts + wired-limit samples
- `raw/p15_crash_node1.txt`, `raw/p15_crash_node2.txt` — P15 BatchPoolingCache crash + restart verbatim
