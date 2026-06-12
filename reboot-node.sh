#!/bin/bash
# reboot-node.sh — reboot an exo Mac Studio with FileVault auto-unlock, fully
# unattended. Pulls the FileVault password from 1Password ON THE CONTROLLER
# (Touch-ID / op CLI) and feeds it to the node's `fdesetup authrestart` over
# ssh stdin. The password NEVER lands on disk or on the node — it lives only
# in the inputplist piped through the ssh pipe for the one command.
#
# Prereqs:
#   - `op` CLI signed in on the controller (op account list)
#   - 1Password item "Mac Studio filevault" (Private vault): username + password
#   - On each node, a sudoers drop-in allowing NOPASSWD for ONLY this command:
#       adam.durham ALL=(root) NOPASSWD: /usr/bin/fdesetup authrestart*
#     Install once with: sudo visudo -f /etc/sudoers.d/fdesetup-authrestart
#
# Usage:
#   ./reboot-node.sh macstudio-m4-1
#   ./reboot-node.sh macstudio-m4-1 macstudio-m4-2     # both, sequentially
#   ./reboot-node.sh --check macstudio-m4-1            # verify creds+sudo, no reboot
set -euo pipefail

OP_USER_REF="op://sxghbs2czadjbz76hh4jkssvb4/Mac Studio filevault/username"
OP_PW_REF="op://sxghbs2czadjbz76hh4jkssvb4/Mac Studio filevault/password"

CHECK=0
if [ "${1:-}" = "--check" ]; then CHECK=1; shift; fi
if [ $# -eq 0 ]; then echo "usage: $0 [--check] <node> [node...]" >&2; exit 1; fi

# Resolve creds once from 1Password (prompts Touch ID if needed).
FV_USER="$(op read "$OP_USER_REF")"
FV_PW="$(op read "$OP_PW_REF")"
if [ -z "$FV_USER" ] || [ -z "$FV_PW" ]; then
  echo "ERROR: could not read FileVault creds from 1Password ($OP_PW_REF)." >&2
  exit 1
fi

make_plist() {
  cat <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Username</key><string>${FV_USER}</string>
  <key>Password</key><string>${FV_PW}</string>
</dict>
</plist>
PL
}

for NODE in "$@"; do
  echo "=== $NODE ==="
  if ! ssh -o ConnectTimeout=8 "$NODE" 'fdesetup supportsauthrestart' >/dev/null 2>&1; then
    echo "  ERROR: $NODE unreachable or authrestart unsupported; skipping." >&2
    continue
  fi
  if [ "$CHECK" = "1" ]; then
    # Dry run: confirm NOPASSWD sudo for fdesetup is wired (no reboot).
    if ssh -o ConnectTimeout=8 "$NODE" 'sudo -n fdesetup status' >/dev/null 2>&1; then
      echo "  OK: reachable, authrestart supported, NOPASSWD sudo wired. (dry run)"
    else
      echo "  WARN: NOPASSWD sudo for fdesetup not wired — authrestart will prompt for sudo pw." >&2
    fi
    continue
  fi
  echo "  issuing authrestart (reboots + auto-unlocks on next boot)..."
  # Pipe the plist over ssh stdin straight into fdesetup; nothing written to disk.
  make_plist | ssh -o ConnectTimeout=8 "$NODE" 'sudo -n /usr/bin/fdesetup authrestart -inputplist' \
    || echo "  NOTE: connection dropped (expected — the box is rebooting)."
done
echo "done. nodes will return ssh-able + unlocked in ~2-3 min."
