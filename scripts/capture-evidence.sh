#!/bin/bash
# capture-evidence.sh — freeze every raw artifact behind a writeup, before rotation eats it.
#
# WHY THIS EXISTS: the 2026-10-17 writeup's raw evidence (the 10-16/10-17 node log,
# the prompts/ bodies, the .sse captures) lived only in a rotating log and a scratch
# dir. Both were gone by review time; the 10-16 capture-day archive is unrecoverable.
# Run this AT CAPTURE TIME, while the node still holds the archive.
#
# READ-ONLY ON THE NODES. Every remote command is rsync-from-remote, ssh ls/du/uname.
# No remote writes, no remote deletes, no --remove-source-files anywhere.
#
# Usage:
#   scripts/capture-evidence.sh <run-id> [options]
#
#   <run-id>              writeup/run identifier, e.g. dsv41-cluster-writeup-2026-10-17
#                         or 2026-10-17 (unique substring match against docs/*)
#
# Options:
#   --writeup-dir DIR     explicit target docs dir (skips run-id resolution)
#   --run-workspace DIR   dir that holds prompts/ and *.sse* for this run
#                         (default: auto-probe, see RUN_WS_CANDIDATES)
#   --nodes "A B"         nodes to pull from (default: macstudio-m4-1 macstudio-m4-2)
#   --no-controller       skip the local controller log dir
#   --no-tarball          skip the .tar.zst bundle
#   --since YYYY-MM-DD    only archives whose mtime is >= this date
#   --dry-run             print what would be pulled, write nothing
#   -h | --help
#
# Output (all under docs/<writeup-dir>/preserved/):
#   <node>/exo.*.log.zst, <node>/dsv41-*.json     immutable rotated archives + verdicts
#   <node>/live/<name>.<stamp>                    mutable exo.log / runner_log snapshot
#   <node>/INVENTORY-<stamp>.txt                  what the node held (retention proof)
#   client/prompts/, client/*.sse*                request bodies and stream captures
#   SHA256SUMS                                    sha256 of everything captured this run
#   CAPTURE-RECEIPT-<stamp>.txt                   provenance receipt (sources, counts, hashes)
#   CAPTURE-RUNS.tsv                              append-only ledger of every capture
#   <run-id>-preserved-<stamp>.tar.zst            the bundle
#
# Exit: 0 all sources captured | 2 partial (a node or input was unreachable/absent) | 1 fatal
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

RSYNC=/opt/homebrew/bin/rsync
[[ -x $RSYNC ]] || RSYNC=/usr/bin/rsync
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3)
export RSYNC_RSH="ssh ${SSH_OPTS[*]}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOCAL_HOST=$(hostname -s)

NODES="macstudio-m4-1 macstudio-m4-2"
DO_CONTROLLER=1
DO_TARBALL=1
DRY_RUN=0
SINCE=""
WRITEUP_DIR_OPT=""
RUN_WS=""
# All three hosts log to the same absolute path (same account on each: adam.durham).
# Absolute, not ~, because rsync --files-from chdir()s on the sender where ~ is literal.
REMOTE_LOG=/Users/adam.durham/.exo/exo_log
TAIL_BYTES=5000000      # bounded runner_log tail (stderr.log is ~1.3 GB/node)
WITH_RUNNER_LOG=0

die() { printf 'capture-evidence: %s\n' "$*" >&2; exit 1; }
say() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }

usage() { sed -n '3,48p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# ---------------------------------------------------------------- args
RUN_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --writeup-dir)   WRITEUP_DIR_OPT=${2:?}; shift 2 ;;
    --run-workspace) RUN_WS=${2:?}; shift 2 ;;
    --nodes)         NODES=${2:?}; shift 2 ;;
    --since)         SINCE=${2:?}; shift 2 ;;
    --remote-log)    REMOTE_LOG=${2:?}; shift 2 ;;
    --with-runner-log) WITH_RUNNER_LOG=1; shift ;;
    --no-controller) DO_CONTROLLER=0; shift ;;
    --no-tarball)    DO_TARBALL=0; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage 0 ;;
    -*)              die "unknown option: $1" ;;
    *)               [[ -z $RUN_ID ]] || die "unexpected argument: $1"; RUN_ID=$1; shift ;;
  esac
done
[[ -n $RUN_ID || -n $WRITEUP_DIR_OPT ]] || usage 1

# ---------------------------------------------------------------- resolve target
if [[ -n $WRITEUP_DIR_OPT ]]; then
  OUT_DOCS=$WRITEUP_DIR_OPT
else
  shopt -s nullglob
  matches=()
  for d in "$REPO_ROOT"/docs/*"$RUN_ID"*; do
    [[ -d $d ]] && matches+=("$d")
  done
  shopt -u nullglob
  if [[ -d $REPO_ROOT/docs/$RUN_ID ]]; then
    OUT_DOCS=$REPO_ROOT/docs/$RUN_ID
  elif [[ ${#matches[@]} -eq 1 ]]; then
    OUT_DOCS=${matches[0]}
  elif [[ ${#matches[@]} -gt 1 ]]; then
    printf 'capture-evidence: run-id "%s" is ambiguous:\n' "$RUN_ID" >&2
    printf '  %s\n' "${matches[@]}" >&2
    die "pass --writeup-dir explicitly"
  else
    OUT_DOCS=$REPO_ROOT/docs/$RUN_ID
  fi
fi
OUT=$OUT_DOCS/preserved
[[ -n $RUN_ID ]] || RUN_ID=$(basename -- "$OUT_DOCS")

say "run-id=$RUN_ID"
say "target=$OUT"

# ---------------------------------------------------------------- gather inputs
captured=()     # top-level items placed under $OUT (for the tarball)
failures=()
notes=()

rsync_common=(-a --ignore-existing --timeout=180)

# Default the window to the run's own date: capturing a run should not drag in
# months of unrelated history (that lives in ~/exo-archives, off-repo).
if [[ -z $SINCE && -n $RUN_ID ]] && [[ $RUN_ID =~ (20[0-9]{2}-[0-9]{2}-[0-9]{2}) ]]; then
  SINCE=${BASH_REMATCH[1]}
  say "window: archives with mtime >= $SINCE (derived from run-id; override with --since)"
fi

# Build the remote file list (read-only `find`), then pull exactly those paths.
# Used instead of rsync filters because rsync has no mtime predicate.
remote_list() { # $1=node  -> prints newline-separated relative paths
  local node=$1 pred=""
  [[ -n $SINCE ]] && pred="-newermt '$SINCE'"
  ssh "${SSH_OPTS[@]}" "$node" \
    "cd $REMOTE_LOG 2>/dev/null || exit 3; find . -maxdepth 1 -type f \( -name 'exo.*.log.zst' -o -name 'dsv41-*.json' \) $pred -print 2>/dev/null | sed 's|^\./||' | sort" 2>/dev/null
}

local_list() { # controller
  local pred=()
  [[ -n $SINCE ]] && pred=(-newermt "$SINCE")
  ( cd "$HOME/.exo/exo_log" 2>/dev/null || exit 3
    find . -maxdepth 1 -type f \( -name 'exo.*.log.zst' -o -name 'dsv41-*.json' \) ${pred[@]:-} -print 2>/dev/null |
      sed 's|^\./||' | sort )
}

pull_node() {
  local node=$1 dest=$OUT/$1
  say "node $node: probing ~/.exo/exo_log"
  if ! ssh "${SSH_OPTS[@]}" "$node" 'du -sh ~/.exo/exo_log 2>/dev/null; ls -1 ~/.exo/exo_log 2>/dev/null | wc -l' >/tmp/.ce_probe.$$ 2>/tmp/.ce_err.$$; then
    failures+=("$node unreachable: $(tr '\n' ' ' </tmp/.ce_err.$$)")
    notes+=("$node: MISSING (ssh failed)")
    rm -f /tmp/.ce_probe.$$ /tmp/.ce_err.$$
    return
  fi
  local du n
  du=$(awk '{print $1}' /tmp/.ce_probe.$$ | sed -n 1p); n=$(sed -n 2p /tmp/.ce_probe.$$ | tr -d ' ')
  rm -f /tmp/.ce_probe.$$ /tmp/.ce_err.$$
  say "node $node: $du, $n entries"

  if [[ $DRY_RUN -eq 1 ]]; then
    remote_list "$node" | sed "s|^|  would pull $node/|" >&2
    return
  fi
  mkdir -p "$dest/live"

  # immutable: rotated archives + degeneracy verdict JSONs, exactly those listed
  local list=$OUT/.list.$node.txt
  if ! remote_list "$node" >"$list"; then
    failures+=("$node file listing failed"); notes+=("$node: archives incomplete")
  elif [[ -s $list ]]; then
    # absolute REMOTE_LOG: rsync --files-from chdir()s on the sender, so a literal
    # `~` is not shell-expanded there.
    "$RSYNC" "${rsync_common[@]}" --files-from="$list" \
      "$node:$REMOTE_LOG/" "$dest/" 2>>"$OUT/.rsync.err" \
      || { failures+=("$node rsync(archives) failed"); notes+=("$node: archives incomplete"); }
  else
    notes+=("$node: no archives matching window $SINCE")
  fi
  rm -f "$list"

  # mutable: live log snapshotted by name so nothing silently freezes
  "$RSYNC" -a --timeout=180 "$node:$REMOTE_LOG/exo.log" \
      "$dest/live/exo.log.$STAMP" 2>>"$OUT/.rsync.err" || notes+=("$node: no live exo.log")

  # runner_log/stderr.log is ~1.3 GB per node. Default to a bounded tail; the full
  # files are already mirrored hourly by the launchd job into ~/exo-archives/.
  if [[ $WITH_RUNNER_LOG -eq 1 ]]; then
    "$RSYNC" -a --timeout=600 "$node:$REMOTE_LOG/runner_log/" \
        "$dest/live/runner_log.$STAMP/" 2>>"$OUT/.rsync.err" || notes+=("$node: runner_log copy failed")
  else
    mkdir -p "$dest/live/runner_log.$STAMP"
    ssh "${SSH_OPTS[@]}" "$node" \
      "cd $REMOTE_LOG/runner_log 2>/dev/null || exit 0
       for f in *.log; do
         [ -f \"\$f\" ] || continue
         echo \"# \$f size=\$(stat -f %z \"\$f\") bytes, last $TAIL_BYTES bytes captured $STAMP\"
         tail -c $TAIL_BYTES \"\$f\"
       done" >"$dest/live/runner_log.$STAMP/tail.txt" 2>>"$OUT/.rsync.err" \
      || notes+=("$node: runner_log tail failed")
  fi

  # retention proof: what the node held at capture time
  {
    echo "# node inventory at capture, $STAMP (UTC)"
    ssh "${SSH_OPTS[@]}" "$node" 'hostname; sw_vers -productVersion; echo; du -sh ~/.exo/exo_log; echo; ls -la ~/.exo/exo_log' 2>&1
  } >"$dest/INVENTORY-$STAMP.txt" 2>/dev/null || notes+=("$node: inventory failed")

  captured+=("$node")
  printf '%s\t%s\t%s\t%s\n' "$STAMP" "$node" "$du" "$n"
}

# controller (this laptop) — same layout, local rsync
pull_controller() {
  local src=$HOME/.exo/exo_log dest=$OUT/controller
  say "controller ($LOCAL_HOST): $src"
  if [[ ! -d $src ]]; then
    notes+=("controller: no $src"); return
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    local_list | sed 's|^|  would pull controller/|' >&2
    return
  fi
  mkdir -p "$dest/live"
  local list=$OUT/.list.controller.txt
  if local_list >"$list" && [[ -s $list ]]; then
    "$RSYNC" "${rsync_common[@]}" --files-from="$list" "$src/" "$dest/" 2>>"$OUT/.rsync.err" \
      || failures+=("controller rsync failed")
  else
    notes+=("controller: no archives matching window $SINCE")
  fi
  rm -f "$list"
  "$RSYNC" -a "$src/exo.log" "$dest/live/exo.log.$STAMP" 2>/dev/null || true
  "$RSYNC" -a "$src/runner_log/" "$dest/live/runner_log.$STAMP/" 2>/dev/null || true
  {
    echo "# controller inventory at capture, $STAMP (UTC)"
    echo "host=$LOCAL_HOST"; echo; du -sh "$src"; echo; ls -la "$src"
  } >"$dest/INVENTORY-$STAMP.txt"
  captured+=("controller")
}

# client-side run workspace: prompts/ + *.sse*. Optional — usually already gone.
RUN_WS_CANDIDATES=(
  "$HOME/exo-runs/$RUN_ID"
  "$HOME/exo-runs/latest"
  "$HOME/.hermes/cache/scratch/$RUN_ID"
  "$HOME/.hermes/cache/scratch/writeup"
  "$REPO_ROOT/runs/$RUN_ID"
  "$OUT_DOCS/run-workspace"
)
pull_client() {
  local ws=""
  if [[ -n $RUN_WS ]]; then
    ws=$RUN_WS
  else
    local c
    for c in "${RUN_WS_CANDIDATES[@]}"; do
      if [[ -d $c ]] && { [[ -d $c/prompts ]] || compgen -G "$c/*.sse*" >/dev/null; }; then ws=$c; break; fi
    done
  fi
  if [[ -z $ws ]]; then
    notes+=("client: no prompts/ or .sse source found (probed ${#RUN_WS_CANDIDATES[@]} paths + --run-workspace)")
    if [[ $DRY_RUN -eq 0 ]]; then
      {
        echo "run-id:  $RUN_ID"
        echo "stamp:   $STAMP"
        echo "status:  NO CLIENT ARTIFACTS FOUND AT CAPTURE TIME"
        echo
        echo "prompts/ request bodies and .sse stream captures were not present. Probed:"
        printf '  %s\n' "${RUN_WS_CANDIDATES[@]}"
        echo
        echo "These are not on the nodes (server writes only exo_log); they live in the"
        echo "client run workspace and die with it. Pass --run-workspace DIR at capture time."
      } >"$OUT/ABSENT-client-artifacts-$STAMP.txt"
      captured+=("ABSENT-client-artifacts-$STAMP.txt")
    fi
    return
  fi
  say "client workspace: $ws"
  [[ $DRY_RUN -eq 1 ]] && { echo "  would pull prompts/ + *.sse* from $ws" >&2; return; }
  mkdir -p "$OUT/client"
  [[ -d $ws/prompts ]] && "$RSYNC" -a --ignore-existing "$ws/prompts/" "$OUT/client/prompts/" 2>/dev/null
  # .sse captures anywhere in the workspace (bounded depth, no follow)
  find "$ws" -maxdepth 3 -type f \( -name '*.sse' -o -name '*.sse.*' \) -print0 2>/dev/null |
    while IFS= read -r -d '' f; do
      rel=${f#"$ws"/}
      mkdir -p "$OUT/client/$(dirname "$rel")"
      cp -p "$f" "$OUT/client/$rel"
    done
  if [[ -n $(find "$OUT/client" -mindepth 1 -type f 2>/dev/null | head -1) ]]; then
    captured+=("client")
  else
    notes+=("client: workspace $ws held nothing to copy")
    rmdir "$OUT/client" 2>/dev/null
  fi
}

# ---------------------------------------------------------------- run
if [[ $DRY_RUN -eq 0 ]]; then
  mkdir -p "$OUT"
  # Bulk never goes into git (2.9 GB of archives; no LFS budget). Only manifests,
  # receipts and hand-extracted lines are committed. Written every run so a new
  # writeup's preserved/ is protected without anyone remembering to do it.
  cat >"$OUT/.gitignore" <<'GITIGNORE'
# Raw capture bulk — lives off-repo at ~/exo-archives/. Never commit.
*.tar.zst
*.log.zst
*/live/
*/runner_log*/
GITIGNORE
fi
RETENTION_TSV=$(mktemp)
: >"$OUT/.rsync.err" 2>/dev/null || true

for node in $NODES; do pull_node "$node" >>"$RETENTION_TSV"; done
[[ $DO_CONTROLLER -eq 1 ]] && pull_controller
pull_client

if [[ $DRY_RUN -eq 1 ]]; then
  rm -f "$RETENTION_TSV" "$OUT/.rsync.err" 2>/dev/null
  say "dry run complete"; exit 0
fi

# ---------------------------------------------------------------- manifest
say "hashing"
( cd "$OUT" && find . -type f \
    ! -name 'SHA256SUMS' ! -name 'CAPTURE-RECEIPT-*' ! -name '*.tar.zst' ! -name '.rsync.err' \
    ! -name 'CAPTURE-RUNS.tsv' -print0 | sort -z | xargs -0 shasum -a 256 ) >"$OUT/SHA256SUMS"

file_count=$(wc -l <"$OUT/SHA256SUMS" | tr -d ' ')
total_bytes=$(du -sk "$OUT" | awk '{print $1*1024}')
archive_count=$(find "$OUT" -name 'exo.*.log.zst' | wc -l | tr -d ' ')

# ---------------------------------------------------------------- tarball
bundle=""
if [[ $DO_TARBALL -eq 1 && ${#captured[@]} -gt 0 ]]; then
  say "bundling"
  bundle=$OUT/${RUN_ID}-preserved-${STAMP}.tar.zst
  tar -cf - -C "$OUT" "${captured[@]}" 2>/dev/null | zstd -q -19 -T0 -o "$bundle" \
    || { failures+=("tarball failed"); bundle=""; }
  [[ -n $bundle ]] && shasum -a 256 "$bundle" | awk '{print $1"  '"$(basename "$bundle")"'"}' >>"$OUT/SHA256SUMS"
fi

# ---------------------------------------------------------------- receipt
receipt=$OUT/CAPTURE-RECEIPT-$STAMP.txt
{
  echo "# Evidence capture receipt"
  echo
  echo "run-id:        $RUN_ID"
  echo "captured_at:   $STAMP (UTC)"
  echo "captured_from: $LOCAL_HOST (controller role, $REPO_ROOT)"
  echo "target:        $OUT"
  echo "tool:          scripts/capture-evidence.sh"
  echo
  echo "## Access model"
  echo "READ-ONLY on the Mac Studios. Remote operations were rsync reads from"
  echo "~/.exo/exo_log/ plus ssh ls/du/hostname. No file on any node was created,"
  echo "modified, or deleted; no service was restarted."
  echo
  echo "## Sources"
  echo "- nodes: ${NODES}"
  echo "- controller: ${DO_CONTROLLER}"
  echo "- client workspace: ${RUN_WS:-auto-probe}"
  echo
  echo "## Result"
  echo "- files captured: $file_count"
  echo "- rotated archives (exo.*.log.zst): $archive_count"
  echo "- bytes on disk: $total_bytes"
  [[ -n $bundle ]] && echo "- bundle: $(basename "$bundle")  sha256=$(shasum -a 256 "$bundle" | awk '{print $1}')"
  echo "- manifest: SHA256SUMS ($file_count entries)"
  echo
  if [[ ${#notes[@]} -gt 0 ]]; then
    echo "## Notes / gaps"
    printf -- '- %s\n' "${notes[@]}"
    echo
  fi
  if [[ ${#failures[@]} -gt 0 ]]; then
    echo "## FAILURES (partial capture)"
    printf -- '- %s\n' "${failures[@]}"
    echo
  fi
  echo "## Node retention at capture"
  if [[ -s $RETENTION_TSV ]]; then
    echo '```'
    printf 'stamp\tnode\tsize\tentries\n'
    cat "$RETENTION_TSV"
    echo '```'
  else
    echo "(no node data)"
  fi
  echo
  echo "## Verify"
  echo '```'
  echo "cd $OUT && shasum -a 256 -c SHA256SUMS"
  [[ -n $bundle ]] && echo "zstd -t $(basename "$bundle")"
  echo '```'
} >"$receipt"

# append-only ledger
if [[ ! -f $OUT/CAPTURE-RUNS.tsv ]]; then
  printf 'stamp\trun_id\thost\tfiles\tarchives\tbytes\tbundle_sha256\tfailures\n' >"$OUT/CAPTURE-RUNS.tsv"
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$STAMP" "$RUN_ID" "$LOCAL_HOST" "$file_count" \
  "$archive_count" "$total_bytes" "$( [[ -n $bundle ]] && shasum -a 256 "$bundle" | awk '{print $1}' || echo - )" \
  "${#failures[@]}" >>"$OUT/CAPTURE-RUNS.tsv"

[[ -f $OUT/.rsync.err ]] && [[ -s $OUT/.rsync.err ]] && mv "$OUT/.rsync.err" "$OUT/rsync-errors-$STAMP.log" || rm -f "$OUT/.rsync.err"
rm -f "$RETENTION_TSV"

say "captured $file_count files ($archive_count archives, $total_bytes bytes) -> $OUT"
[[ -n $bundle ]] && say "bundle: $bundle"
[[ ${#failures[@]} -gt 0 ]] && { printf 'capture-evidence: partial capture\n' >&2; printf '  - %s\n' "${failures[@]}" >&2; exit 2; }
exit 0
