# MTRACE-NOTES — `bench/phase20_mtrace.py` (Metal System Trace for gate 0d / Phase-1 cross-check)

Status: **built, live-validated on the laptop AND on node studio2 with a
throw-away toy process, and hardened** (FIX 1, §8). The PM runs the live attach
on a node during a decode window. Nothing here was run against the exo runner.

This is the plan's **Fallback-B** tool for gate 0d (used when powermetrics
`GPU HW active residency` lands within 5 points of idle → wrong GPU/process)
and the **independent cross-check** of the Phase-1 host timer: it reads the
GPU's *own* command-buffer timeline from **outside** the process — no code
change, no relaunch, no graph edit.

---

## 1. Environment / permissions (discovered, verified)

| thing | finding |
|---|---|
| xctrace | `xcrun xctrace version` → **27.0 (27A266a)** on laptop, `studio1`, `studio2` |
| template | `Metal System Trace` present in `xctrace list templates` on all three |
| laptop attach | `xcrun xctrace record --attach <pid> …` works **with no sudo** (attached to a self-launched MLX toy on the M4 Max) |
| node attach | works **with no sudo** on **studio2** (see §4) — Developer mode enabled, user is in `_developer` |
| recording-mode | `Deferred` (trace is flushed at stop; file lands after the run ends) |
| hard timeout | **macOS has NO coreutils `timeout`** (`zsh: command not found: timeout` on studio2). The `record` subcommand wraps the remote command with `/usr/bin/perl -e 'alarm shift; exec @ARGV' <secs+45> …` (perl ships with macOS). |

## 2. Data model (xctrace 27.0)

* A `<table schema="…">` holds ordered `<col><mnemonic>…` children; every
  `<row>` has children **in the same order as the cols**.
* Row children carry inline text, or an `id` (first sighting) / `ref` pointing
  at a prior `id`. **Resolve refs positionally** — that is what the parser does.
* `start-time` / `duration` engineering-types are integer **nanoseconds**
  (`00:00.782.830` ↔ `782830041`).
* **GPU busy timeline** = `metal-gpu-state-intervals` rows with `state=Active`;
  `Idle` rows partition the complement. A row may be split per GPU channel, so
  **union (merge) before summing** — summing over-counts.
* **Per-encoder GPU exec intervals** = `metal-gpu-intervals` (channel-name
  Compute/Render, event-label, cmdbuffer-id, encoder-id, gpu-submission-id,
  start-latency = CPU→GPU latency ns).
* Command-buffer submissions = `metal-application-command-buffer-submissions`.
* Perf states = `gpu-performance-state-intervals`.

### Verbatim `xcrun xctrace export --input x.trace --toc` excerpt (metal tables)
```
<table schema="metal-shader-profiler-intervals" target-pid="SINGLE" documentation="Denotes Shader Timeline intervals"/>
<table schema="metal-residency-set-usage-event" documentation="Denotes a MTLResidencySet usage"/>
<table schema="metal-application-encoders-list" target-pid="SINGLE" documentation="Denotes an a list of encoders created by the tracing Metal application."/>
<table schema="metal-driver-event-per-thread-intervals" target-pid="SINGLE" documentation="Denotes an interesting period of memory activity in the Metal driver."/>
<table schema="metal-gpu-execution-points" documentation="Denotes an interesting points of activity in the GPU."/>
<table schema="metal-gpu-info" documentation="Marks GPU info"/>
<table schema="metal-driver-intervals" target-pid="SINGLE" documentation="Denotes an interesting period of activity in the Metal driver."/>
<table schema="metal-application-command-buffer-submissions" target-pid="SINGLE" documentation="Marks Metal command buffer submissions"/>
<table schema="metal-command-buffer-completed" documentation="Marks when Metal Command Buffer are completed"/>
<table schema="metal-application-intervals" target-pid="SINGLE" documentation="Denotes an interesting period of activity in the Metal applications."/>
<table requested-consistent-state="0" schema="gpu-performance-state-intervals" documentation="Denotes the current and induced performance state of the device."/>
<table schema="metal-gpu-submission-to-command-buffer-id" documentation="Maps between Metal command buffer ID and accelerator ID"/>
<table schema="metal-gpu-intervals" target-pid="SINGLE" documentation="Denotes an interesting period of activity in the GPU."/>
<table schema="metal-command-buffer-to-accelerator-id" documentation="Maps between Metal command buffer ID and accelerator ID"/>
<table schema="metal-gpu-state-intervals" documentation="Denotes GPU state (on/off) periods"/>
```
(The candidate `metal-gpu-encoder-intervals` schema **does not exist** in 27.0;
per-encoder execution intervals live in `metal-gpu-intervals`.)

### Column findings (verbatim mnemonics)
* `metal-gpu-state-intervals`: `start, duration, state, label, color, num-events, gpu`
  — `state` ∈ {Active, Idle}.
* `metal-gpu-intervals`: `start, duration, channel-name, frame-number,
  start-latency, event-depth, event-label, state, connection-UUID, color,
  process, gpu, channel-subtitle, iosurface-accesses, bytes, cmdbuffer-id,
  encoder-id, gpu-submission-id`.
* `metal-application-command-buffer-submissions`: `start, duration, event-type
  (=CommandBufferSubmission), gpu, track-label, num-encoders, encoder-time,
  process, thread, event-icon, event-label, encoder-time-label, frame-number, …`.
* `gpu-performance-state-intervals`: `start, duration, gpu-performance-state
  (Minimum|…), track-label, is-induced, narrative, event-type`.

### Sizes / timings actually measured (200×10×1024 matmul toy, 8 s limit)
* `.trace` bundle: **~43–57 MB** on disk for an 8 s recording
  (~30–90 MB/s of trace duration; use as the record-size estimate).
* `--toc`: ~30 s export, 34 KB XML.
* one table export (`metal-gpu-state-intervals`): ~2 s, ~1 MB for 429 rows;
  `metal-gpu-intervals` ~238 KB / 215 rows. Export is fast; the *record*
  (deferred flush) dominates.
* A real decode trace (many rounds/s over 30 s) will be **larger** — the parser
  is `iterparse`-streaming (`element.clear()` per row) precisely for that.

## 3. Laptop ground-truth validation (the required ±10 % check)

Workload: `mx.eval` of a 10-deep matmul+softmax burst, then `time.sleep(0.02)`
every batch — a KNOWN duty cycle and KNOWN gap cadence, all timestamps printed
host-side to `truth.json` (never trusting the trace for ground truth).

```
== CANONICAL VALIDATION (single Metal System Trace, laptop M4 Max) ==
trace window ms            6864.124
recovered busy fraction    0.1942
host known duty cycle      0.2192   (busy 1.511 s / wall 6.894 s)
absolute delta             0.025    target <= 0.10  ->  True
recovered median gap ms    27.586
host injected gap ms       27.362   (sleep 20.0 ms + eval 7.49 ms)
rounds @ gap>=3ms          205
n active bursts (merged)   211
```

* **Busy fraction recovered within ±10 % absolute** (0.1942 vs 0.2192, Δ 0.025).
  The ~0.02 residual ≈ Metal launch/serialisation overhead not visible to the
  host timer — expected.
* **Gaps found at the right cadence**: recovered median gap 27.586 ms vs host
  injected 27.362 ms; 205 rounds for 200 batches (the extra split is the tiny
  first command buffer at trace start).
* **Do NOT fold in `metal-gpu-intervals` durations** to compute *busy* — they
  are per-*encoder* intervals that overlap each other; only the
  `metal-gpu-state-intervals` Active union is the busy timeline. (If you point
  the analyzer at a directory holding several trace files, it unions all of
  them; for a decode run keep it to ONE trace's own export dir.)

## 4. Node attach smoke (the ONE permitted node action)

On **studio2** (rank 0, IDLE at the time), a throw-away `python -c` MLX matmul
loop for ~40 s via `~/repos/exo/.venv/bin/python`, written to `/tmp/p20_toy.py`:

```
$ ssh studio2 '… xcrun xctrace record --template "Metal System Trace" \
    --attach <toy pid> --time-limit 8s --no-prompt --output /tmp/p20_toy.trace'
Starting recording with the Metal System Trace template. Attaching to: python (24094). Time limit: 8.0 s
Reached specified time limit, ending recording...
Output file saved as: p20_toy.trace            # 43 MB
$ ssh studio2 '… xctrace export … metal-gpu-state-intervals …'
Finished export to the file: /tmp/p20_toy_state.xml   # 967 KB
$ scp studio2:/tmp/p20_toy_state.xml node_p20_toy_state.xml
```
* **Permissions: none required — no sudo, no password, no interactive prompt.**
  `DevToolsSecurity -status` → *Developer mode is currently enabled*; the user
  is in `_developer`. The `--no-prompt` flag suppresses the privacy warning.
* `sudo` was **not** needed for record or export. (The toy write to `/tmp` was
  the only node write; nothing under `~/repos/exo` was touched.)
* The toy process self-exited; `/tmp/p20_toy.trace` was left on the node as a
  throw-away (the brief allows `/tmp` scratch for this smoke).

## 5. How the PM runs it LIVE on a node during a decode window

The runner pid per node is the `multiprocessing.spawn … spawn_main` child:
```
ssh studio2 "pgrep -f 'multiprocessing.spawn import spawn_main'"   # rank 0
ssh studio1 "pgrep -f 'multiprocessing.spawn import spawn_main'"   # rank 1
```
Then, **during** the decode window (start after the first SSE content token):

```bash
# 0) sanity: dry-run the exact command first
PYTHONPATH=bench .venv/bin/python bench/phase20_mtrace.py record \
    --node studio2 --pid <RUNNER_PID> --secs 30 --out /tmp/p20_decode.trace --dry-run

# 1) record (30 s of decode). PM ONLY. This ATTACHES to the exo runner.
PYTHONPATH=bench .venv/bin/python bench/phase20_mtrace.py record \
    --node studio2 --pid <RUNNER_PID> --secs 30 --out /tmp/p20_decode.trace

# 2) export the GPU tables (over ssh, scp back) into a local dir
mkdir -p /tmp/p20_decode_export
PYTHONPATH=bench .venv/bin/python bench/phase20_mtrace.py export \
    --node studio2 --trace /tmp/p20_decode.trace --out /tmp/p20_decode_export
#   (add --toc to dump only the table of contents)

# 3) analyze, restricted to the decode window if desired (ns since trace start)
PYTHONPATH=bench .venv/bin/python bench/phase20_mtrace.py analyze \
    --dir /tmp/p20_decode_export \
    --window-start-ns <A> --window-end-ns <B> \
    --round-gap-ms 3 --json out.json --md out.md
```

* **Aligning the window with an SSE token**: the trace's `start-time`s are ns
  relative to the recording start (the `<info><summary><start-date>`). Take the
  wall-clock epoch of the first SSE content token, subtract the recording
  start-date epoch → `--window-start-ns` (×1e9). The same works for the last
  token to close the window. Without a window, `analyze` uses
  [`first Active`, `last Active`] automatically.
* **Two ranks**: a single `xctrace` attach records ONE process. To compare
  ranks, run the record on studio1 and studio2 (separately) and diff the two
  analyses; the runner is a different pid on each node. Do the two recordings
  back-to-back on the same request, or accept they are different requests.
* **Expected output**: `gpu_busy.busy_fraction`, idle-gap histogram (>50 µs),
  command-buffer count & rate, per-encoder interval stats, a heuristic round
  segmentation (default `--round-gap-ms 3` — a decode round's inter-round bubble
  is a few ms, so gaps ≥3 ms split rounds), and the top-10 longest GPU
  intervals. The `>10 ms` histogram bucket + round `gap_ms` are where the
  per-layer TP2 comm bubbles surface.

## 6. Known limitations / what NOT to conclude

* **Overhead is UNMEASURED on the exo runner.** A Metal System Trace attach adds
  instrumentation. Quantify it by comparing decode tok/s with the trace NOT
  running vs WITH it (same prompt, same window). Until then, treat a traced run's
  tok/s as perturbed. The 0d/Phase-1 conclusions must rest on the *relative*
  busy/idle shape, not on absolute tok/s from a traced run.
* **GPU-idle-in-trace ≠ GPU unused by comm.** An idle gap inside the decode
  round is consistent with a host/collective wait (TP2 per-layer allreduce) —
  it does NOT by itself prove the GPU is starved by comm vs. by launch overhead.
  Correlate with the Phase-1 `PROF` brackets (collective time) before claiming.
* **Deferred recording** buffers the trace; `--time-limit` ends the recording but
  the file is written afterwards (~seconds). Don't expect the file mid-window.
* **The attach is process-scoped and can perturb the target** — the exo
  watchdog already knows an "xctrace attach" false-positive hang class. Only ONE
  rank at a time; never attach to both simultaneously.
* **`metal-gpu-intervals` overlaps** (see §3) — do not sum them for busy.

## 7. Reproduce the offline validation

```bash
# workload (ground truth) + record + export, on the laptop
python workload.py --batches 200 --k 10 --size 1024 --gap 0.02 --json truth.json
xcrun xctrace record --template 'Metal System Trace' --output x.trace \
      --time-limit 8s --no-prompt --launch -- .venv/bin/python workload.py …
xcrun xctrace export --input x.trace \
      --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-state-intervals"]' \
      --output state.xml
python bench/phase20_mtrace.py analyze --dir <dir-with-state.xml>

# pure-parse tests (no xctrace, no cluster)
PYTHONPATH=bench .venv/bin/python -m pytest --noconftest \
  bench/phase20_tests/test_phase20_mtrace.py -q -p no:cacheprovider   # 26 passed
```

## 8. FIX 1 — live end-to-end validation + the real hazard found

Dispatched against two *reported* defects. Both were checked against the live
CLI and **neither reproduces on xctrace 27.0 (27A266a)**; the real defect was a
third one, fixed in this commit.

### 8.1 Reported defect (1) — `--xpath` on `xctrace export` is NOT a defect

`xcrun xctrace help export` (verbatim, 27.0) documents it *and* gives an example
in the exact form the tool uses:

```
usage: xctrace export [<trace>] [<options>] [--toc | --xpath expression]
options:
	--toc                         Present entities to export in the table of contents form
	--xpath <expression>          Choose elements to export using specified XPath expression
examples:
	xctrace export --input input.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="my-table-schema"]'
```

Run against a real 76 MB laptop `.trace`: **rc=0**, wrote a real 1.12 MB XML
with `name="metal-gpu-state-intervals"` and **4697 rows**. The `run[@number="1"]`
form is the documented one. (Likely origin of the report: an older Xcode/CLT
whose `export` predated `--xpath`; not reproducible here.)

### 8.2 Reported defect (2) — `--attach` is NOT a defect

`xcrun xctrace help record` (verbatim, 27.0): `--attach <pid|name>  Attach and
record process with the given name or pid` (usage line `[--attach |
--all-processes | --launch -- command]`). A PID works. Attached to a live MLX
toy on the **laptop** and, remotely, on **studio2** via the tool's own `record`
subcommand — both produced real traces (59 MB / 38 MB). No sudo, no prompt.

### 8.3 THE REAL DEFECT — the export fails silently and the tool called it success

When `--xpath` matches nothing (wrong schema name, or run index ≠ 1), `xctrace
export` **exits 0 and writes a 65-byte `<trace-query-result/>`**:

```
$ xcrun xctrace export --input laptop2.trace \
    --xpath '/trace-toc/run[@number="1"]/data/table[@schema="does-not-exist-xyz"]' \
    --output missing.xml
Finished export to the file: missing.xml      # rc=0
$ wc -c missing.xml
      65  missing.xml      # <?xml ...><trace-query-result/></trace-query-result>
```

The old `cmd_export` reported `(schema, path, bytes)` and returned 0 regardless
— so a no-op export was indistinguishable from a good one. **This is consistent
with the prior run's "live analysis UNFINISHED"** (a bad/renamed schema would
have produced an empty table with a success return code), though I could not
reproduce that specific session. Fix: `cmd_export` now **parses every file back**
via `validate_export()` (ok / missing / empty / unreadable), prints per-schema
rows+bytes+status, lists the *available* schemas from `--toc` on failure, and
**exits 2** if any schema matched nothing. `toc` also returns nonzero on failure.

Verification (real traces, exit codes are the tool's own):

```
$ python bench/phase20_mtrace.py export --trace laptop2.trace --out /tmp/ok
  ok         metal-gpu-state-intervals   rows=2603  bytes=619657  rc=0
  ...                                                                    # -> exit 0
$ python bench/phase20_mtrace.py export --trace laptop2.trace --out /tmp/bad \
      --schemas does-not-exist-xyz
  missing    does-not-exist-xyz  rows=0  bytes=65  rc=0  [no <table> matched the xpath]
  # available schemas (17) from --toc: metal-gpu-state-intervals ...
  # export: 0/1 ok, 0 empty (warn), 1 FAILED                              # -> exit 2
```

### 8.4 Live toy validation — laptop AND studio2 (the required ±0.10 check)

Workload: uniform rounds of a 100-iter 2048² fp16 matmul+softmax burst then a
known `sleep`; host-side per-burst timestamps → `truth.json` (host timer is the
ground truth, never the trace). Traced via `record --attach <pid>`, exported,
analyzed.

| box | host known duty | recovered busy frac | Δ abs | host median gap | recovered inter-round gap | rounds |
|---|---|---|---|---|---|---|
| laptop M4 Max | 0.2272 | **0.1936** | 0.034 ✔ | 30.01 ms | 31.06 ms | 219 |
| studio2 M4-2  | 0.2038 | **0.1643** | 0.040 ✔ | 30.76 ms | 30.60 ms | 263 |

Both within the ≤0.10 target; gaps recovered at the injected cadence (Δ ≤ 1 ms).
The residual ≈0.03 ≈ the per-round kernel/launch overhead the sleep loop hides
from the host timer — expected, same direction as §3.

```
# laptop, analyze output (analyze --dir laptop2_export)
# Metal System Trace analysis
- Window: 258262666 .. 8399575916 ns (8141.313 ms, auto)
- **GPU-busy fraction: 19.36%** (busy 1576.334 ms / idle 6564.979 ms)
- Active intervals: 1300
## Idle gaps (>50 us)
- gaps: 1299, >50us: 219, median 0.000875 ms, max 33.192583 ms
  - 1-3ms: 1   - 3-10ms: 2   - >10ms: 216
## Encoder intervals
- n=1302 total 1576.348 ms; mean 1.2107 / median 1.1952 / p90 1.3879 / max 1.7862 ms
## Command buffers
- distinct ids 1296, rate 159.19 /s; submissions 1284 (157.71 /s)
## Rounds (gap>=3.0 ms)
- count 219, busy median 7.105 ms, gap median 0.004 ms   (6 bursts/round)
```

### 8.5 Working commands (verbatim — copy these for the live run)

```bash
# record (attach to a pid; laptop or via ssh on a node)
xcrun xctrace record --template 'Metal System Trace' --attach <PID> \
      --time-limit 8s --no-prompt --output /tmp/x.trace

# list the schemas actually in the trace
xcrun xctrace export --input /tmp/x.trace --toc --output /tmp/toc.xml

# export one table (VERIFIED form in 27.0)
xcrun xctrace export --input /tmp/x.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-state-intervals"]' \
  --output /tmp/state.xml
```

The tool wraps all three (`record`, `export`, `export --toc`) and validates each
exported file, so prefer the tool over the raw commands.

### 8.6 Schema list + permissions (re-confirmed on this run)

`--toc` on a real Metal System Trace lists 40+ tables; the ones this tool uses
(all present): `metal-gpu-state-intervals`, `metal-gpu-intervals`,
`metal-application-command-buffer-submissions`, `gpu-performance-state-intervals`,
`metal-command-buffer-completed`. Permissions: **no sudo, no password, no
interactive prompt** on the laptop or studio2 for record *or* export; `--no
-prompt` suppresses the privacy warning. macOS still has **no coreutils
`timeout`** — `record` uses `/usr/bin/perl -e 'alarm shift; exec @ARGV'`. On
studio2 everything (toy, trace, exports) was written to `/tmp` and removed
after; nothing under `~/repos/exo` was touched.
