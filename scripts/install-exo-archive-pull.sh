#!/bin/bash
# install-exo-archive-pull.sh — install/reinstall the hourly archive-pull agent.
# Idempotent. No sudo: it is a per-user LaunchAgent.
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PLIST_SRC=$REPO/scripts/launchd/com.adam.exo-archive-pull.plist
LABEL=com.adam.exo-archive-pull
DEST=$HOME/Library/LaunchAgents/$LABEL.plist
UID_N=$(id -u)

# The agent must NOT run the script out of the git working tree. A checkout of an
# older SHA (A/B runs, bisects) removes scripts/exo-archive-pull.sh, and launchd then
# fails every tick with exit 127 until the tree is back — silently, into a log nobody
# reads. Deploy a copy to a path that checkout cannot move, and point the plist there.
DEPLOY_DIR=$HOME/.local/share/exo-archive-pull
DEPLOY=$DEPLOY_DIR/exo-archive-pull.sh

[[ -f $PLIST_SRC ]] || { echo "missing $PLIST_SRC" >&2; exit 1; }
[[ -f $REPO/scripts/exo-archive-pull.sh ]] || { echo "missing $REPO/scripts/exo-archive-pull.sh" >&2; exit 1; }
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/exo-archives" "$DEPLOY_DIR"

install -m 0755 "$REPO/scripts/exo-archive-pull.sh" "$DEPLOY"
echo "deployed $DEPLOY"

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
