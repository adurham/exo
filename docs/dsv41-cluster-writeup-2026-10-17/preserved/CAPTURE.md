# CAPTURE.md — how raw evidence is preserved (and why it is not in git)

Written 2026-10-18, after the 2026-10-17 writeup lost its raw evidence to log rotation.

## What went wrong

The writeup's raw sources — the 10-16/10-17 node log, the `prompts/` request bodies
(incl. the 3.39 MB 1M-token body), and every `.sse` capture — were referenced in
`evidence.md` but had no durable home. They lived in a rotating log and a scratch
directory on the controller laptop. By review time:

- `mdfind` = 0 hits for `run_160k.sse.timed.txt`, `prompts_160k.json`, `prompts_1m.json`
- no APFS local snapshots, no Time Machine destination on any of the three hosts
- the capture-day archive `exo.2026-10-16_18-38-04_871911.log.zst` is **unrecoverable**

exo retains roughly **5 rotated archives per node**. Each relaunch rotates one out.
The 10-17 corroborating lines were already at the edge of that window; one more
rotation would have taken them too. That is the whole failure: evidence lifetime was
shorter than writeup lifetime, and nothing was watching.

## The two mechanisms

### 1. `scripts/capture-evidence.sh <run-id>` — freeze at capture time

Run it **while the run is still on the node**, not at writeup time.

```
./scripts/capture-evidence.sh dsv41-cluster-writeup-2026-10-17
```

It pulls, into `docs/<writeup-dir>/preserved/`:

| source | what lands there |
|---|---|
| both nodes, `~/.exo/exo_log/` | rotated `exo.*.log.zst` archives + `dsv41-degeneration-*.json` verdicts |
| both nodes, live files | `live/exo.log.<stamp>`, `live/runner_log.<stamp>/tail.txt` (bounded 5 MB tail) |
| both nodes | `INVENTORY-<stamp>.txt` — hostname, OS, `du`, `ls -la` at capture (retention proof) |
| controller | same, under `controller/` |
| client run workspace | `prompts/`, `*.sse`, `*.sse.*` under `client/` (optional; see below) |

and emits, alongside them:

- `<run-id>-preserved-<stamp>.tar.zst` — portable bundle of everything captured
- `SHA256SUMS` — sha256 of every captured file plus the bundle
- `CAPTURE-RECEIPT-<stamp>.txt` — sources, counts, bytes, gaps, verify command
- `CAPTURE-RUNS.tsv` — append-only ledger across captures
- `ABSENT-client-artifacts-<stamp>.txt` when no `prompts/`/`.sse` source exists

**The window defaults to the run's own date** (parsed from `<run-id>`, override with
`--since`), so a capture carries that run's archives, not months of history.

**Read-only on the Mac Studios.** Every remote operation is an rsync *pull*, or
`ssh ls/du/hostname/find/tail`. No file is created, modified or deleted on a node;
nothing is restarted. Verified by inspection: the only writes are local.

`prompts/` and `.sse` captures are client-side; they are not on the nodes and they die
with the run workspace. The script probes six known locations and takes
`--run-workspace DIR` for the rest. **If a run's request bodies matter, pass
`--run-workspace` at capture time** — that is exactly the class of artifact that was
lost, and no server-side job can recover it after the fact.

Verify a capture:

```
cd docs/dsv41-cluster-writeup-2026-10-17/preserved
shasum -a 256 -c SHA256SUMS
zstd -t *.tar.zst
```

### 2. `com.adam.exo-archive-pull` — hourly append-only mirror

A per-user launchd agent (`scripts/launchd/com.adam.exo-archive-pull.plist`) runs
`scripts/exo-archive-pull.sh` every hour and on load. launchd rather than cron because
it fires a missed tick on wake and survives reboot on a machine that sleeps constantly.

It rsyncs `~/.exo/exo_log/` from `macstudio-m4-1`, `macstudio-m4-2` and the controller
into:

```
~/exo-archives/<node>/     # mirrors, append-only
~/exo-archives/MANIFEST.txt
~/exo-archives/.sha256-index.tsv
~/exo-archives/rsync.log
```

Properties that matter:

- **`--ignore-existing`** — a rotated archive is immutable once written. The mirror
  never overwrites, never re-transfers, never deletes. Append-only by construction, so
  a bad hour cannot corrupt a good archive.
- **One sha256 pass per file, ever** — the index is incremental; an hour with nothing
  new hashes nothing (measured: 8 s idle run, 0 files hashed).
- **Single instance** — a lock directory stops a stuck ssh from stacking up.
- **Bounded** — the job halts mirroring at 40 GB rather than filling the disk; it never
  deletes, because pruning archives is a human decision.
- **`runner_log/` is excluded by default.** `stderr.log` is ~1.3 GB per node and
  appends forever; a `--ignore-existing` mirror would freeze it at the first hour for
  2.7 GB. `capture-evidence.sh` keeps a bounded tail instead. Set
  `EXO_ARCHIVE_INCLUDE_RUNNER_LOG=1` if you want the full files.

Install / reinstall (no sudo, idempotent):

```
./scripts/install-exo-archive-pull.sh
```

**CAVEAT — the mirror is only as continuous as the laptop.** The controller is a
MacBook. Lid closed, asleep, or off the tailnet means the hourly tick does not happen;
launchd coalesces one missed interval into a single run on wake, so the schedule
recovers, but any archive rotated **while the laptop was asleep** is gone. This job
closes the "nobody was watching" gap, not the physics gap. For a long unattended run,
still run `capture-evidence.sh` yourself at the end.

## Where the archives live, and why not here

`~/exo-archives/` on the controller laptop — **outside the repo**:

```
~/exo-archives/controller/        360 KB
~/exo-archives/macstudio-m4-1/    ~1.4 GB (of which runner_log 1.4 GB)
~/exo-archives/macstudio-m4-2/    ~1.5 GB
total                             2.9 GB, 61 files, 16 rotated archives
```

**Not committed, for three reasons:**

1. **Size.** 2.9 GB of node logs in a repo whose entire `.git` is 2.8 GB. Checking it
   in would roughly double the repository for data that is append-only and regenerable
   from the mirror.
2. **It is not source.** These are compressed runtime logs. Git stores every byte
   forever; there is no path to removing them later without a history rewrite.
3. **git-lfs costs money.** LFS bandwidth and storage are billed; this project runs at
   $0. A 2.9 GB LFS payload would be a recurring charge for artifacts that a filesystem
   mirror already serves.

What *is* committed from `preserved/` is the part with lasting value: the manifests,
the receipts, the ledger, the inventories, the absent-artifact declarations, and any
hand-extracted lines (see `MANIFEST.txt` for the 10-17 salvage). `preserved/.gitignore`
is written by the capture script and keeps the bulk out:

```
*.tar.zst
*.log.zst
*/live/
*/runner_log*/
```

So `git status` shows the evidence that a capture happened and what it contained,
while the bytes stay on disk at `~/exo-archives/` (mirror) and
`docs/<writeup>/preserved/` (this run's window, git-ignored).

## Standing rules

1. Capture at the end of the run, on the same day. Rotation does not wait for review.
2. Pass `--run-workspace` if the run produced `prompts/` or `.sse` files; nothing
   server-side can reconstruct them.
3. Never point the mirror at a `--delete` — archives are immutable.
4. Treat `CAPTURE-RUNS.tsv` as the audit trail. A writeup with no capture row has no
   provenance and should say so, as `evidence.md` now does for the 10-17 gaps.
