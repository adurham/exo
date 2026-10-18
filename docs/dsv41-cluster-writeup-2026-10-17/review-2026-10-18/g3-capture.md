# G3 — capture script + hourly launchd pull job

Started 2026-10-18 (laptop: adams-macbook-pro-m4, controller role).

## Facts established by inspection (read-only)
- Nodes: macstudio-m4-1 (Adams-Mac-Studio-M4-1), macstudio-m4-2 (Adams-Mac-Studio-M4-2), both reachable over Tailscale with key auth (`~/.ssh/exo_cluster`, no 1Password agent needed).
- Log dir on all three: `~/.exo/exo_log/` — `exo.log` (live) + rotated `exo.<ts>.log.zst`, `runner_log/{stdout,stderr}.log`, plus `dsv41-degeneration-*.json` verdict tokens.
- Sizes: m4-1 1.4 GB / m4-2 1.5 GB / controller 360 KB. Whole cluster ~2.9 GB.
- NO `prompts/` dir and NO `*.sse*` files exist on any of the three right now (`find` + `mdfind` = 0) — they were client-side run-workspace artifacts that died with the scratch dir. Script must therefore treat them as OPTIONAL inputs: absent = recorded as absent, not an error.
- Retention is ~5 archives per node; further rotation will drop the 10-17 corroborating lines.

## Delivered

1. `scripts/capture-evidence.sh <run-id>` (exec, 15 KB)
   - pulls per-node `~/.exo/exo_log/`: rotated `exo.*.log.zst`, `dsv41-degeneration-*.json`, live `exo.log` snapshot, bounded runner_log tail (5 MB; full 1.3 GB/node only with `--with-runner-log`), plus an INVENTORY with `du`/`ls -la` as retention proof.
   - controller handled as a third source (local rsync, same layout).
   - `prompts/` + `*.sse*` from the client run workspace: 6 auto-probe paths + `--run-workspace`; when absent it writes `ABSENT-client-artifacts-<stamp>.txt` instead of failing silently — exactly the class that was lost.
   - emits `.tar.zst` bundle, `SHA256SUMS`, `CAPTURE-RECEIPT-<stamp>.txt`, append-only `CAPTURE-RUNS.tsv`, and `preserved/.gitignore`.
   - window defaults to the run-id's date (`--since` overrides) so a capture carries that run, not months of history.
   - read-only on nodes: rsync pulls + `ssh ls/du/hostname/find/tail` only.

2. `scripts/exo-archive-pull.sh` + `scripts/launchd/com.adam.exo-archive-pull.plist` + `scripts/install-exo-archive-pull.sh`
   - hourly (`StartInterval 3600`, `RunAtLoad`) rsync `--ignore-existing` from both nodes + controller into `~/exo-archives/<node>/`; single-instance lock; incremental sha256 index (only new files hashed); 40 GB halt; never deletes.
   - `runner_log/` excluded by default (1.3 GB/node, would freeze at first hour for 2.7 GB); opt-in via EXO_ARCHIVE_INCLUDE_RUNNER_LOG=1.

3. `docs/dsv41-cluster-writeup-2026-10-17/preserved/CAPTURE.md` — mechanism, off-repo location, why not committed (2.9 GB / git-lfs costs money / not source), laptop-asleep caveat, standing rules.

## Real output

- Full capture run: `51 files (10 archives, 83,845,120 bytes)` in 8 s, exit 0. `shasum -c SHA256SUMS` → all OK; `zstd -t` → OK; tar lists both node dirs.
- Empty-window self-test (`--since 2026-10-19`, scratch target): exit 0, 12 files, 0 archives, absent-client note written, scratch dir removed.
- Pull job foreground: `new=1 files_total=13 transferred=1,442,612` (m4-1, the 14-47 archive) → hashed 61 → `DONE new=1 hashed=61 index=61 mirror=2.9GB rc=0`, 8 s.
- Second run under launchd-equivalent env (`env -i HOME=... PATH=...`, cwd=/): exit 0, `new=0 hashed=0`, proving idempotence and that no shell rc / ssh-agent is required.
- `~/exo-archives/MANIFEST.txt` verifies: `shasum -a 256 -c <(grep -v '^#' MANIFEST.txt)` → all 61 OK. 16 rotated archives, 2.9 GB.

## BLOCKED: launchd install denied

`./scripts/install-exo-archive-pull.sh` was **denied by the user** before running. Not retried, not rephrased.
State: plist is `plutil -lint` OK, installer `bash -n` OK, but
`launchctl print gui/501/com.adam.exo-archive-pull` → "Could not find service" and
`~/Library/LaunchAgents/com.adam.exo-archive-pull.plist` does not exist. **The hourly job is NOT installed.**
One command installs it: `./scripts/install-exo-archive-pull.sh` (copies plist, bootstraps, kickstarts, prints last exit status).
Substitute proof obtained instead: the job's program runs clean with launchd's exact environment (empty env, minimal PATH, cwd=/), exit 0, idempotent.

## Notes

- `~` in an rsync remote path is NOT expanded when `--files-from` is used (rsync chdir()s on the sender): first attempt failed with `chdir: No such file or directory`. Fixed by pinning REMOTE_LOG=/Users/adam.durham/.exo/exo_log (same account on all three hosts).
- First attempt also copied 2.9 GB of runner_log into preserved/; caught, replaced with a bounded tail, bulk removed.
- Untouched by me: `docs/.../WRITEUP.md` (modified by another session), `scripts/gen-prompts.py`, `scripts/check-s11.sh` (untracked, other work).

END G3
