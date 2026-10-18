#!/bin/bash
# exo-archive-pull.sh — hourly, append-only mirror of exo's rotating log dirs.
#
# Installed as launchd agent com.adam.exo-archive-pull (hourly, RunAtLoad).
# See docs/dsv41-cluster-writeup-2026-10-17/preserved/CAPTURE.md.
#
# WHY: exo keeps ~5 rotated archives per node. The 2026-10-16 capture-day archive
# was gone before the writeup was reviewed. This mirrors each node's ~/.exo/exo_log
# to ~/exo-archives/<node>/ once an hour, so rotation on the node can never be the
# reason a raw log disappears again.
#
# Rules:
#   - --ignore-existing: a rotated archive is immutable once written. Never overwrite,
#     never re-hash, never delete. The mirror is append-only by construction.
#   - READ-ONLY on the nodes: rsync pulls. No remote writes, no --remove-source-files.
#   - Bounded: whole dir is ~3 GB today; a bounded tail is not used here because the
#     point is byte-exact retention. If it grows past MAX_GB the job warns and stops
#     adding (no deletion — that is a human decision).
#   - Incremental sha256 index: only new files are hashed, so an hourly run touches
#     ~nothing when there is nothing new.
#
# CAVEAT: the controller is a laptop. Asleep / off-tailnet / closed lid = the hourly
# tick is missed. launchd coalesces one missed StartInterval into a single run on wake,
# so the gap closes on the next wake — but archives rotated *while asleep* are gone.
# Nothing here can fix that; keep long runs' capture-evidence.sh run at the end.
set -uo pipefail

export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin

DEST="$HOME/exo-archives"
NODES="${EXO_ARCHIVE_NODES:-macstudio-m4-1 macstudio-m4-2}"
LOG="$DEST/rsync.log"
INDEX="$DEST/.sha256-index.tsv"
LOCK="$DEST/.pull.lock"
MAX_GB=40
# runner_log/stderr.log is ~1.3 GB per node and appends forever; a --ignore-existing
# mirror would freeze it at the first hour and cost 2.7 GB. Excluded by default —
# capture-evidence.sh takes a bounded tail at capture time instead. Opt in if needed.
RUNNER_LOG_ARG=(--exclude=runner_log/)
[[ ${EXO_ARCHIVE_INCLUDE_RUNNER_LOG:-0} == 1 ]] && RUNNER_LOG_ARG=()
RSYNC=/opt/homebrew/bin/rsync
[[ -x $RSYNC ]] || RSYNC=/usr/bin/rsync
LOG_DIR=/Users/adam.durham/.exo/exo_log   # absolute: rsync chdir()s, ~ stays literal
SSH_CMD=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3)

mkdir -p "$DEST"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$LOG"; }

# single instance: a stuck ssh must not stack up hourly
if ! mkdir "$LOCK" 2>/dev/null; then
  log "SKIP another pull is running (lock $LOCK held since $(stat -f %Sm "$LOCK" 2>/dev/null))"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

size_gb=$(du -sm "$DEST" 2>/dev/null | awk '{printf "%.1f", $1/1024}')
if [[ ${size_gb%%.*} -ge $MAX_GB ]]; then
  log "HALT mirror is ${size_gb} GB >= ${MAX_GB} GB cap; not pulling. Prune by hand (nothing here deletes)."
  exit 0
fi

rc=0
pulled=0
log "RUN start mirror=${size_gb}GB nodes=[$NODES]"

for node in $NODES; do
  mkdir -p "$DEST/$node"
  before=$(find "$DEST/$node" -type f 2>/dev/null | wc -l | tr -d ' ')
  out=$("$RSYNC" -a --ignore-existing --timeout=300 --stats "${RUNNER_LOG_ARG[@]}" \
        -e "${SSH_CMD[*]}" "$node:$LOG_DIR/" "$DEST/$node/" 2>&1)
  st=$?
  after=$(find "$DEST/$node" -type f 2>/dev/null | wc -l | tr -d ' ')
  new=$((after - before))
  pulled=$((pulled + new))
  if [[ $st -ne 0 ]]; then
    rc=1
    log "FAIL $node rsync rc=$st: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-300)"
  else
    bytes=$(printf '%s\n' "$out" | awk -F': *' '/Total transferred file size/{print $2; exit}')
    log "OK   $node new=$new files_total=$after transferred=${bytes:-0}"
  fi
done

# controller's own log dir (cheap, local)
mkdir -p "$DEST/controller"
before=$(find "$DEST/controller" -type f 2>/dev/null | wc -l | tr -d ' ')
"$RSYNC" -a --ignore-existing "${RUNNER_LOG_ARG[@]}" "$HOME/.exo/exo_log/" "$DEST/controller/" 2>/dev/null
after=$(find "$DEST/controller" -type f 2>/dev/null | wc -l | tr -d ' ')
pulled=$((pulled + after - before))
log "OK   controller new=$((after - before)) files_total=$after"

# ---- incremental sha256 index + manifest (only new files are hashed)
[[ -f $INDEX ]] || : >"$INDEX"
cd "$DEST" || exit 1
added=0
while IFS= read -r -d '' f; do
  rel=${f#./}
  case "$rel" in .sha256-index.tsv|MANIFEST.txt|rsync.log|.pull* ) continue ;; esac
  if ! awk -F'\t' -v p="$rel" '$1==p{found=1; exit} END{exit !found}' "$INDEX"; then
    h=$(shasum -a 256 "$f" | awk '{print $1}')
    printf '%s\t%s\n' "$rel" "$h" >>"$INDEX"
    added=$((added + 1))
  fi
done < <(find . -type f -print0)

if [[ $added -gt 0 || ! -f MANIFEST.txt ]]; then
  { printf '# exo-archives manifest — sha256  relpath, regenerated %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '# append-only mirror of ~/.exo/exo_log on each node; verify with: cd %s && shasum -a 256 -c <(grep -v "^#" MANIFEST.txt)\n' "$DEST"
    awk -F'\t' '{printf "%s  %s\n", $2, $1}' "$INDEX" | sort -k2
  } >"$DEST/MANIFEST.txt.tmp"
  mv "$DEST/MANIFEST.txt.tmp" "$DEST/MANIFEST.txt"
fi

files=$(wc -l <"$INDEX" | tr -d ' ')
size_gb=$(du -sm "$DEST" 2>/dev/null | awk '{printf "%.1f", $1/1024}')
log "DONE new=$pulled hashed=$added index=$files mirror=${size_gb}GB rc=$rc"
exit $rc
