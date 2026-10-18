#!/bin/bash
# install-exo-archive-pull.sh — install/reinstall the hourly archive-pull agent.
# Idempotent. No sudo: it is a per-user LaunchAgent.
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PLIST_SRC=$REPO/scripts/launchd/com.adam.exo-archive-pull.plist
LABEL=com.adam.exo-archive-pull
DEST=$HOME/Library/LaunchAgents/$LABEL.plist
UID_N=$(id -u)

[[ -f $PLIST_SRC ]] || { echo "missing $PLIST_SRC" >&2; exit 1; }
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/exo-archives"

install -m 0644 "$PLIST_SRC" "$DEST"
echo "installed $DEST"

# unload any previous definition; ignore "not loaded"
launchctl bootout "gui/$UID_N/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID_N" "$DEST"
echo "bootstrapped gui/$UID_N/$LABEL"

# prove it runs now, not in an hour
launchctl kickstart -k "gui/$UID_N/$LABEL"
echo "kickstarted; last exit status:"
launchctl print "gui/$UID_N/$LABEL" | grep -E 'state|last exit|runs|program' | sed 's/^/  /'
