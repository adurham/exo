# Phase 0 — Speculative-Decode Acceptance Histogram Audit (READ-ONLY)

Date: 2026-09-03 (local machine clock; cluster nodes report the same day)
HEAD checked: `ccc692ff3` (local repo), `232d1f6b7` (both cluster nodes — cluster
is NOT on HEAD, it's several commits behind; noted for the record, not touched)
Scope: read-only. No cluster relaunch, no runner restart, no git push performed.

---

## A) Is `EXO_DSV4_MTP_LOG_INTERVAL` currently SET on the LIVE runner processes?

**NO — not set on either node, on any of the 4 runner-family PIDs per node.**

macstudio-m4-1 runner PIDs (`pgrep -f 'python -m exo'`): `79254 79255 79256 79266`
macstudio-m4-2 runner PIDs: `86058 86059 86060 86069`

`ps eww -p <pid> | tr ' ' '\n' | grep -E 'EXO_DSV4_MTP|EXO_SPECULATIVE'` on every
one of those 8 PIDs returned the identical env block, with **no
`EXO_DSV4_MTP_LOG_INTERVAL` key present at all** (absent, not zero):

```
EXO_SPECULATIVE=1
EXO_SPECULATIVE_GAMMA=3
EXO_DSV4_MTP=1
EXO_DSV4_MTP_C2_MAX_CTX=1
EXO_DSV4_MTP_DEDICATED=0
EXO_DSV4_MTP_EAGLE_K=8
EXO_DSV4_MTP_TIEBREAK_FIX=0
EXO_DSV4_MTP_TIEBREAK_EPS=0.5
EXO_DSV4_MTP_ACCEPT_LOGPROBS=1
EXO_DSV4_MTP_MAX_CTX=0
EXO_DSV4_MTP_TIE_REVERIFY=0
```
(identical on all 4 PIDs × 2 nodes = 8/8 identical results)

Confirms current production gamma is `EXO_SPECULATIVE_GAMMA=3`, matching the
campaign's stated ground truth.

**`start_cluster.sh` grep** (local repo, `/Users/adam.durham/repos/exo/start_cluster.sh:2042`):

```
2042:  [ -n "${EXO_DSV4_MTP_LOG_INTERVAL:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_LOG_INTERVAL=$EXO_DSV4_MTP_LOG_INTERVAL"
```

This is a **conditional pass-through, not a default** — it only forwards
`EXO_DSV4_MTP_LOG_INTERVAL` into the launched runner's env if the *launcher's own
shell* already has it set (`-n` = non-empty check); there is no `EXO_DSV4_MTP_LOG_INTERVAL:-<default>` assignment anywhere else in the file that would give it an
implicit non-zero value. Combined with (A)'s live `ps eww` result, the variable
was simply never exported before the current production cluster was launched —
this is consistent, not contradictory, with the module-level default of `"0"`
seen in (B).

---

## B) READ-SITE AUDIT — import-time or lazy?

File used in production: `/Users/adam.durham/repos/exo/src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`
(this is the fork's OWN speculative-decode module, not the mlx-lm submodule —
`mlx_lm/models/dsv4_mtp.py` does not exist under this checkout; searched
`mlx-lm/mlx_lm/models/` for any dsv4-named file and found none. The task
description's expected path is stale/wrong; the real path is the one above.)

Local repo, line 118:
```
118:_LOG_INTERVAL = int(os.environ.get("EXO_DSV4_MTP_LOG_INTERVAL", "0"))
```

This is a bare module-level statement (no enclosing `def`/`class` at that
indentation) — it executes exactly once, at Python `import` time, and is cached
into the module-global `_LOG_INTERVAL`. Every later read of `_LOG_INTERVAL`
(e.g. `dsv4_mtp.py:2168`, `if _LOG_INTERVAL <= 0: return`) reads that cached
int, NOT `os.environ` again.

**Verdict: IMPORT-TIME.** Setting/changing `EXO_DSV4_MTP_LOG_INTERVAL` on a
live, already-running runner process (`os.environ` mutation via some external
hook, `/proc`-style tricks, etc.) would NOT be picked up — it requires a full
process restart (module re-import) to take effect. There is no per-call or
lazy re-read anywhere in the file for this variable.

**Installed copy on the cluster nodes:** checked directly (not via a
long-running find that could time out) — both nodes run the SAME repo checkout
path (`/Users/adam.durham/repos/exo/...`, no separate venv-installed copy of
this file exists; it's imported straight from the source tree, not from a
site-packages install). Confirmed identical line 118 content on both nodes:

```
macstudio-m4-1: _LOG_INTERVAL = int(os.environ.get("EXO_DSV4_MTP_LOG_INTERVAL", "0"))
macstudio-m4-2: _LOG_INTERVAL = int(os.environ.get("EXO_DSV4_MTP_LOG_INTERVAL", "0"))
```

Both nodes are on git SHA `232d1f6b7` — several commits behind the local repo's
`ccc692ff3` HEAD, but line 118 itself is identical text in both, so the
import-time verdict applies to what's actually running.

---

## C) HISTOGRAM SEARCH (last 48h)

**NOT EMITTED in the last 48h — LOG_INTERVAL was unset on the live runners for
that entire window.**

Live/current logs on both nodes (`~/exo.log`, `~/.exo/exo_log/exo.log`, dated
today) were grepped with:
```
grep -h -E 'accept_hist=|mean_accept=|k_hist|bypos' ~/exo.log ~/.exo/exo_log/exo.log
```
Result: **0 matches on macstudio-m4-1, 0 matches on macstudio-m4-2.** No
`/tmp/exo*.log` files exist on either node currently. No screen-scrollback logs
were found beyond the file-based ones already checked.

Local machine: `tmp/perf-campaign-2/round1` through `round4` and the rest of
`tmp/` were grepped for the same pattern — **0 matches anywhere in the local
repo's campaign artifacts.** No files dated 2026-09-03 (today) contain this
pattern locally either.

**However — older, HISTORICAL matches DO exist on both nodes**, from a prior
manual-instrumentation campaign around 2026-08-27 through 2026-08-31 (well
outside the 48h window; nodes' log directories retain many hand-named log
files from that campaign, e.g. `exo_verbon3.log`, `exo_von3shard0.log`,
`exo_g0rowseq.log`). These are NOT current/production evidence — they predate
today by 4-7 days and were produced under manually-launched
`relaunch_exo_*.sh` scripts (screen sessions with `EXO_DSV4_MTP_LOG_INTERVAL=50`
explicitly set), not the current production launch. Reported in (D) below for
completeness since they are the only histogram data that exists anywhere on
disk, but they do NOT reflect the current γ=3 production config being swept.

Match counts per node (old files only):
```
macstudio-m4-1: exo_verbon3.log:23618 exo_verbon2.log:6199 exo_verbon3epi.log:5576
                exo_von3cc0.log:2141 exo_von3shard0.log:2120 exo_von3rowseq.log:551
                exo_rowseq.log:195 exo_g0rowseq.log:114 exo_verbon.log:112
                (+ two /tmp files: exo_1108.log:1267, exo_1753.log:1638 — same
                vintage, from the memory-leak investigation, not this campaign)
macstudio-m4-2: identical filenames/counts (both nodes log independently but
                the file set/counts line up — these were dual-node captures
                from the same historical runs)
```
No `bypos=` (the per-position curve) or `accept_hist=` (the DSpark-shadow
histogram) lines were found in ANY file, old or new, on either node — only the
`mean_accept=`/`hist=` MTP-path line (dsv4_mtp.py:2172-2176) ever fired. The
`bypos`/`accept_hist`/`k_hist` triple (dsv4_mtp.py:637-648, the
`_ShadowStats.summary()` DSpark-shadow path) requires `EXO_DSV4_SPEC_SHADOW=1`,
which was never set in any of the found relaunch scripts.

---

## D) Histogram dump (from the historical, non-48h data — since C found nothing live)

Most recent MTP acceptance-histogram lines found (from
`macstudio-m4-1:~/exo_von3shard0.log`, timestamped 2026-08-29 02:09 —
**5 days before this audit, not within the 48h window**):

```
[ 2026-08-29 02:09:30.449 | WARNING | ... ] Runner stderr: [MTP] cycles=2081 mean_accept=2.559/3 hist=0:118,1:181,2:201,3:1581
[ 2026-08-29 02:09:30.558 | WARNING | ... ] Runner stderr: [MTP] cycles=2082 mean_accept=2.560/3 hist=0:118,1:181,2:201,3:1582
[ 2026-08-29 02:09:30.657 | WARNING | ... ] Runner stderr: [MTP] cycles=2083 mean_accept=2.559/3 hist=0:118,1:182,2:201,3:1582
[ 2026-08-29 02:09:34.128 | WARNING | ... ] Runner stderr: [MTP] cycles=2116 mean_accept=2.560/3 hist=0:119,1:185,2:203,3:1609
[ 2026-08-29 02:09:34.228 | WARNING | ... ] Runner stderr: [MTP] cycles=2117 mean_accept=2.561/3 hist=0:119,1:185,2:203,3:1610
[ 2026-08-29 02:09:34.334 | WARNING | ... ] Runner stderr: [MTP] cycles=2118 mean_accept=2.561/3 hist=0:119,1:185,2:203,3:1611
[ 2026-08-29 02:09:34.435 | WARNING | ... ] Runner stderr: [MTP] cycles=2119 mean_accept=2.561/3 hist=0:119,1:185,2:203,3:1612
[ 2026-08-29 02:09:34.536 | WARNING | ... ] Runner stderr: [MTP] cycles=2120 mean_accept=2.561/3 hist=0:119,1:185,2:203,3:1613
```

**Final-cycle cumulative histogram (cycles=2120):**
`hist=0:119,1:185,2:203,3:1613` → total samples = 119+185+203+1613 = 2120.
- P(accept=0) = 119/2120 = 5.6%
- P(accept=1) = 185/2120 = 8.7%
- P(accept=2) = 203/2120 = 9.6%
- P(accept=3) = 1613/2120 = 76.1%  ← accept-ceiling hit >3 in 4 cycles

`mean_accept=2.561/3` — note the counter's denominator (`self.gamma`) is 3,
even though the launching `relaunch_exo_v2st.sh` script (which produced this
file) exported `EXO_SPECULATIVE_GAMMA=2` at the top-level `screen` invocation.
This is a real discrepancy worth flagging for triage, not resolved here — the
MTP path's effective `self.gamma` clearly resolved to 3 regardless of that
outer env var (this script also set `EXO_DSV4_MTP_DEDICATED=1` and
`EXO_DSV4_MTP_LOG_INTERVAL=50`; some other MTP-specific gamma source, not
`EXO_SPECULATIVE_GAMMA`, likely governs the dedicated-MTP chain depth). Do not
assume `EXO_SPECULATIVE_GAMMA` is the sole gamma knob for the MTP path without
checking `dsv4_mtp.py`'s own gamma-resolution logic (line ~3949 has an
`_env_gamma` reference not traced further here — out of this task's 25-minute
box).

**No `bypos=` (per-position acceptance curve) line was ever found in any log,
old or new, on either node.** Per-position data is emitted only by the
DSpark-shadow path (`_ShadowStats.summary()`), gated by `EXO_DSV4_SPEC_SHADOW=1`
— that flag does not appear in any relaunch script found on either node. So
there is **no bypos/per-position cliff data to report** — the gamma sweep
cannot currently be pre-registered against a real per-position acceptance
curve; only the cumulative 0/1/2/3 histogram above exists, and only from a
5-day-old, off-window, possibly gamma-mismatched capture.

**Bottom line for the gamma-sweep pre-registration:** the aggregate P(accept=3)
≈76% from the one historical sample suggests a lot of mass IS piling up at the
γ=3 ceiling, which would argue "extending gamma might help" — but given (1)
this data is 5 days stale, (2) the apparent gamma/denominator mismatch above is
unexplained, and (3) no per-position curve exists at all, this is **weak,
unverified evidence, not a basis for predicting the sweep's direction.** The
clean way to get real data is to set `EXO_DSV4_MTP_LOG_INTERVAL` (see E) on the
NEXT relaunch anyway (zero extra cost, it's needed for the sweep regardless) —
this phase should not be used to skip that step.

---

## E) True-acceptance counter locations

**Counters (module: `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`):**

MTP path (the one actually live in production — plain MTP, `EXO_DSV4_MTP=1`,
DSpark shadow OFF):
- `self._spec_cycles` (int) — total verify cycles counted. Init line 1637-ish
  (near `self._spec_accept_hist` init at **line 1638**); incremented at
  **line 2164**: `self._spec_cycles += 1`
- `self._spec_total_accepted` (int) — running sum of accepted-token counts.
  Incremented at **line 2165**: `self._spec_total_accepted += n_accepted`
- `self._spec_accept_hist` (list[int], size `gamma+1`) — the `hist=` histogram
  bins. Initialized **line 1638**: `self._spec_accept_hist: list[int] = [0] * (self.gamma + 1)`.
  Incremented at **line 2166-2167**:
  ```python
  if 0 <= n_accepted <= self.gamma:
      self._spec_accept_hist[n_accepted] += 1
  ```
- **Increment site (all three, one call per verify cycle):**
  `_record_acceptance(self, n_accepted: int)`, **lines 2150-2167**.
- **Log-emit site:** **lines 2168-2177** — same method, gated by
  `if _LOG_INTERVAL <= 0: return` then `if self._spec_cycles % _LOG_INTERVAL == 0:`,
  emits via `logger.warning(f"[MTP] cycles=... mean_accept=... hist=...")`.

DSpark-shadow path (NOT live in production currently — needs
`EXO_DSV4_SPEC_SHADOW=1`, separate from plain MTP):
- Counters live in the `_ShadowStats` class, **lines 552-654**:
  `self.cycles`, `self.total_would_accept`, `self.total_drafted`,
  `self.accept_hist` (dict), `self.k_hist` (dict), `self.pos_reached` /
  `self.pos_hit` (dicts — these two are what produce `bypos=`).
- **Increment site:** `_ShadowStats.record(...)`, **lines 582-618** (called
  once per shadow-mode verify cycle from elsewhere in the file; not traced to
  its call site in this pass — out of scope/time-box).
- **Log-emit site:** `_ShadowStats.summary()`, **lines 631-653**, called from
  **line 4521**: `logger.warning(_sh_stats.summary())`, gated by the cycle
  count crossing `_SPEC_SHADOW_GUARD_CYCLES` (**lines 4505-4521**).

**Exact shell command to extract `mean_accept` and `hist` from a runner log
after a run** (MTP path, the one actually relevant to the production γ sweep):

```bash
grep -oE '\[MTP\] cycles=[0-9]+ mean_accept=[0-9.]+/[0-9]+ hist=[0-9:,]+' ~/exo.log | tail -1
```

For the DSpark-shadow path (if `EXO_DSV4_SPEC_SHADOW=1` is ever set), to get
the fuller line including `bypos`:
```bash
grep -oE '\[DSPARK-SHADOW\].*' ~/exo.log | tail -1
```

---

## Summary / action items for the gamma sweep

1. **`EXO_DSV4_MTP_LOG_INTERVAL` is unset on the live production runners on
   both nodes right now** — confirmed via `ps eww` on all 8 runner PIDs.
2. **It's a load-time constant (line 118)** — any relaunch for the γ-sweep
   should include `EXO_DSV4_MTP_LOG_INTERVAL=<N>` in the launch env from the
   start; there's no way to turn it on for an already-running process.
3. **No usable current or 48h-fresh histogram data exists on either node** —
   confirmed via direct grep of both nodes' live logs (0 matches) and the
   local repo's round1-4 campaign artifacts (0 matches).
4. **One stale (5-day-old) MTP histogram sample exists**, showing
   `hist=0:119,1:185,2:203,3:1613` (P(accept=3)≈76%) at what its own log line
   claims is gamma=3 — but the launching script's `EXO_SPECULATIVE_GAMMA=2`
   env var doesn't match that denominator, an unexplained discrepancy flagged
   for triage, not resolved. Treat this as weak/unreliable evidence only.
5. **No `bypos` (per-position) data exists anywhere** — the DSpark-shadow
   path that emits it has never been enabled in any surviving log on either
   node. The requested "does the curve cliff after position 1" question
   currently has **zero evidence either way**.
6. **Recommendation:** since a relaunch is already required to sweep gamma,
   set `EXO_DSV4_MTP_LOG_INTERVAL` to a reasonable interval (e.g. 50-100, per
   the historical scripts) on the FIRST sweep arm so at least aggregate
   accept-histogram data comes back fresh and current; consider also setting
   `EXO_DSV4_SPEC_SHADOW=1` if a true per-position `bypos` curve is wanted
   before committing to the full 4-arm sweep, since none has ever been
   captured.
