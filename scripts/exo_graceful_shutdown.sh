#!/bin/bash
# exo_graceful_shutdown.sh -- self-match-immune, pattern-complete graceful
# shutdown of exo on ONE node.
#
# Runs ON the target node. start_cluster.sh delivers it over
# `ssh "$NODE" bash -s` STDIN -- never as a command argument.
#
# ---------------------------------------------------------------------------
# WHY THIS FILE EXISTS
# ---------------------------------------------------------------------------
# See docs/incidents/pkill-self-match-forces-sigkill-2026-09-10.md.
#
# start_cluster.sh used to inline the shutdown as
#
#     ssh "$NODE" "pkill -TERM -f 'python.*exo'; ..."
#     ssh "$NODE" "pgrep -f 'python.*exo'"          # <-- liveness probe
#
# The ssh CLIENT's own argv contains the literal text `python.*exo`. When
# $NODE is the node the deploy is being driven FROM (the coordinator), that
# ssh client is a local process and is therefore visible to the pgrep running
# on the far end of its own connection. The ERE `python.*exo` matches that
# literal argv text (`python`, `.*` eats `.*`, `exo`), so the probe MATCHES
# ITSELF and reports exo alive even when exo is stone dead.
#
# Consequences, in order:
#   1. The graceful-exit wait can never succeed on the coordinator.
#   2. So EVERY deploy driven from a Studio escalated to `pkill -9`.
#   3. SIGKILL skips the C++ static-duration destructors that call
#      destroy_qp/dealloc_pd/close_device, leaking live RDMA queue pairs on
#      the Thunderbolt NIC. Those accumulate and wedge the Apple TB stack to
#      "No device connected", whose only documented escape hatch is a
#      physical reboot -- which an agent session cannot perform (FileVault on,
#      `sudo -n reboot` needs a password). So the bug quietly rolled the dice
#      on an unrecoverable state on every single deploy.
#   4. The `pkill -9 -f 'python.*exo'` then killed the SIBLING ssh clients
#      that were themselves carrying the pattern in their argv -- the
#      `Killed: 9 ssh "$NODE" ...` lines in the deploy logs.
#
# Reproduced 4/4 on the coordinator and 0/4 on the secondary across the
# 2026-09-09 logs, and again live on 2026-09-10 from m4-1:
#
#   $ ssh macstudio-m4-1 "pgrep -lf 'python.*ZZSELFMATCHPROBE9271'"
#   39377 ssh -o BatchMode=yes macstudio-m4-1 pgrep -lf 'python.*ZZSELFMATCHPROBE9271'
#   -> exit 0 (match) with no such process anywhere on the box
#
# ---------------------------------------------------------------------------
# THE FIX -- three independent layers
# ---------------------------------------------------------------------------
# 1. DELIVERY. The whole checker arrives over ssh STDIN, so NO pattern text
#    appears in ANY argv anywhere: not the ssh client's, not sshd's, not the
#    remote shell's (`bash -s`). pgrep literally cannot see it. This layer
#    alone is sufficient; the other two are defence in depth for the day
#    someone reintroduces an inline `ssh $NODE "pgrep ..."`.
# 2. BRACKET TRICK. Every pattern is written `[p]ython.*exo`. The regex
#    `[p]ython` matches the string "python" but NOT the literal source text
#    "[p]ython", so a pattern that does leak into an argv still cannot match
#    itself.
# 3. PID EXCLUSION. $$ and $PPID are filtered out of every match list.
#
# PATTERN COMPLETENESS. The old code kill-matched THREE different things
# (python.*exo, exo.main, and the 52415/52416 port holders) but liveness-
# probed only the FIRST. A survivor visible only to the other two was
# invisible to the wait loop, so "gone" was never actually proven. All three
# are checked here, exactly as the hand-run gate used for the 2026-09-10
# deploy did (~/exo_alive_check.sh, ~/graceful_term.sh -- both reported
# CLEAN_EXIT after 1s on both nodes).
#
# NOTE on `exo.main`: as of 2026-09-10 the live invocation is
# `.venv/bin/python -m exo -v`, so `[e]xo.main` matches nothing real
# (verified on both Studios: pgrep -f '[e]xo.main' -> empty). It is retained
# for older/other invocations and is now harmless, because it can no longer
# match itself either. Before this fix it was a pure self-match liability:
# `ssh macstudio-m4-1 "pgrep -lf 'ZZSELFMATCHPROBE9271.main'"` self-matched
# identically in the same 2026-09-10 reproduction.
#
# EXIT CODE is always 0 -- this script reports, it does not gate. The caller
# decides what to do with EXO_SHUTDOWN_VERDICT=<...> on the last line.

set -u

# Grace period in whole seconds before escalating to SIGKILL. Passed as $1
# (a bare integer -- no pattern text), defaulted and validated here.
GRACE=${1:-15}
case "$GRACE" in
  '' | *[!0-9]*) GRACE=15 ;;
esac

# Pattern overrides exist ONLY so scripts/test_shutdown_self_match.sh can
# exercise THIS EXACT FILE against a synthetic marker instead of the real
# exo process. The test sets them from a prelude prepended to this script's
# STDIN -- the same argv-invisible channel production uses, never a command
# line. Unset in production, which is the case start_cluster.sh always hits.
P_PROC=${_EXO_SHUTDOWN_P1:-'[p]ython.*exo'}
P_MAIN=${_EXO_SHUTDOWN_P2:-'[e]xo.main'}
P_PORTS=${_EXO_SHUTDOWN_PORTS:-'52415,52416'}

SELF=$$
PARENT=${PPID:-0}
HOST=$(hostname -s 2>/dev/null || uname -n)

# Drop our own pid and our parent's from any match list. pgrep already
# excludes itself, but a subshell or the shell running this script could in
# principle match a future, looser pattern.
_not_self() { grep -v -e "^${SELF}\$" -e "^${PARENT}\$"; }

# Processes the SIGTERM/SIGKILL sweep targets by name.
_kill_targets() {
  {
    pgrep -f "$P_PROC" 2>/dev/null
    pgrep -f "$P_MAIN" 2>/dev/null
  } | _not_self | sort -u
}

# Processes still holding exo's API/runner ports. Cannot be spoofed by a
# command line, which makes this the most trustworthy of the three signals.
_port_holders() { lsof -ti:"$P_PORTS" 2>/dev/null | _not_self | sort -u; }

# Full liveness set = union of all three signals. "Dead" means all three are
# empty, not just the first one.
_live() { { _kill_targets; _port_holders; } | sort -u; }

echo "  [$HOST] shutdown gate: proc='$P_PROC' main='$P_MAIN' ports='$P_PORTS' grace=${GRACE}s"

_now=$(_live | tr '\n' ' ')
if [ -z "${_now// /}" ]; then
  echo "  [$HOST] nothing matching is running"
  echo "EXO_SHUTDOWN_VERDICT=ALREADY_DEAD"
  exit 0
fi
echo "  [$HOST] live: ${_now}"

# GRACEFUL FIRST. exo runners hold live RoCE/RDMA queue pairs (jaccl TP).
# A normal interpreter exit runs the C++ static destructors that free them;
# SIGKILL does not. (root cause: warm-mem fact 526; 2026-06-08)
_targets=$(_kill_targets | tr '\n' ' ')
echo "  [$HOST] SIGTERM -> ${_targets:-<none by name; port holders only>}"
for _p in $_targets; do
  kill -TERM "$_p" 2>/dev/null || true
done

_i=0
while [ "$_i" -lt "$GRACE" ]; do
  sleep 1
  _i=$((_i + 1))
  if [ -z "$(_live | tr -d '\n ')" ]; then
    echo "  [$HOST] CLEAN_EXIT after ${_i}s (NO SIGKILL — destructors ran, RDMA QPs released)"
    echo "EXO_SHUTDOWN_VERDICT=CLEAN_EXIT"
    exit 0
  fi
done

# Last resort only, and now only when something is GENUINELY still alive --
# which before this fix was never actually established on the coordinator.
_survivors=$(_live | tr '\n' ' ')
echo "  [$HOST] WARNING: still alive after ${GRACE}s: ${_survivors}"
echo "  [$HOST] escalating to SIGKILL (may leak RDMA QPs; reboot this node if the TB stack wedges)"
for _p in $_survivors; do
  kill -9 "$_p" 2>/dev/null || true
done
sleep 1

_after=$(_live | tr '\n' ' ')
if [ -z "${_after// /}" ]; then
  echo "EXO_SHUTDOWN_VERDICT=SIGKILLED"
else
  echo "  [$HOST] ERROR: survived SIGKILL: ${_after}"
  echo "EXO_SHUTDOWN_VERDICT=SURVIVED_SIGKILL"
fi
exit 0
