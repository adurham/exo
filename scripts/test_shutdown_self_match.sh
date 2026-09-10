#!/bin/bash
# test_shutdown_self_match.sh -- proves scripts/exo_graceful_shutdown.sh does
# NOT self-match, using the same synthetic-marker method that reproduced the
# bug in the first place.
#
# Regression test for
# docs/incidents/pkill-self-match-forces-sigkill-2026-09-10.md.
#
# WHAT IS ACTUALLY UNDER TEST
#   The real scripts/exo_graceful_shutdown.sh file that start_cluster.sh
#   ships, delivered the same way start_cluster.sh delivers it (over ssh
#   STDIN as `bash -s`). Only the PATTERNS are swapped for a synthetic marker
#   that matches no real process, via _EXO_SHUTDOWN_P1/_P2/_PORTS, so the
#   test can never touch a live exo. The pattern overrides themselves travel
#   on STDIN too (prepended to the helper), never on a command line -- if
#   they were passed as ssh arguments the test would reintroduce the very
#   argv exposure it is checking for.
#
# WHY A SYNTHETIC MARKER IS SUFFICIENT
#   The bug is purely about a pattern matching the TEXT OF ITSELF in an argv.
#   That is a property of the pattern and the delivery channel, not of the
#   process being searched for. A marker in the same shape as the real
#   patterns exercises the identical code path with zero risk to the cluster.
#
# USAGE
#   scripts/test_shutdown_self_match.sh [coordinator-node] [remote-node]
#   Defaults: macstudio-m4-1 macstudio-m4-2
#
#   MUST be run FROM the coordinator node for cases 1-3 to be meaningful:
#   the bug only manifests when the ssh client is local to the node being
#   probed. Run from anywhere else and those cases pass trivially. The
#   script detects this and says so.
#
# SAFETY
#   Read-only with respect to exo. The only process it ever signals is a
#   `sleep` it starts itself under a unique marker and reaps itself.

set -u

COORD=${1:-macstudio-m4-1}
REMOTE=${2:-macstudio-m4-2}
HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/exo_graceful_shutdown.sh"

if [ ! -r "$HELPER" ]; then
  echo "FATAL: helper not readable at $HELPER" >&2
  exit 1
fi

MARK="ZZSHUTDOWNSELFTEST$$"
PASS=0
FAIL=0

_ok() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
_no() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "======================================================================"
echo "self-match regression test for $HELPER"
echo "marker=$MARK  coordinator=$COORD  remote=$REMOTE"
echo "driving from: $(hostname -s 2>/dev/null || uname -n)"
echo "======================================================================"
echo

# Run the REAL helper against synthetic patterns. Overrides are prepended to
# the helper on STDIN so nothing pattern-bearing ever reaches an argv.
_run_helper() {
  local node=$1 grace=$2
  {
    printf "_EXO_SHUTDOWN_P1='[%s]%s'\n" "${MARK:0:1}" "${MARK:1}"
    printf "_EXO_SHUTDOWN_P2='[%s]%s.main'\n" "${MARK:0:1}" "${MARK:1}"
    printf "_EXO_SHUTDOWN_PORTS='65432,65433'\n"
    cat "$HELPER"
  } | ssh "$node" "bash -s $grace" 2>&1
}

_verdict() { sed -n 's/^EXO_SHUTDOWN_VERDICT=//p' <<<"$1" | tail -1; }

# ---------------------------------------------------------------------------
echo "[1] BASELINE — reproduce the ORIGINAL bug on the coordinator."
echo "    Old form: pattern text inside the ssh command argument."
_old_self=$(ssh "$COORD" "pgrep -lf 'python.*$MARK'" 2>&1)
_old_self_rc=$?
echo "    ssh $COORD \"pgrep -lf 'python.*\$MARK'\" -> rc=$_old_self_rc"
[ -n "$_old_self" ] && echo "      matched: $_old_self"
if [ $_old_self_rc -eq 0 ]; then
  _ok "old inline form self-matches on the coordinator (bug reproduced; nothing named $MARK exists)"
  BUG_REPRODUCIBLE=1
else
  echo "  NOTE: old form did not self-match — this test is not being driven"
  echo "        from $COORD, so cases 1-3 cannot demonstrate anything. Re-run"
  echo "        this script ON $COORD for a meaningful result."
  BUG_REPRODUCIBLE=0
fi
echo

# ---------------------------------------------------------------------------
echo "[2] CONTROL — same old form against the REMOTE node."
_old_rem=$(ssh "$REMOTE" "pgrep -lf 'python.*$MARK'" 2>&1)
_old_rem_rc=$?
echo "    ssh $REMOTE \"pgrep -lf 'python.*\$MARK'\" -> rc=$_old_rem_rc"
if [ $_old_rem_rc -ne 0 ]; then
  _ok "old form does NOT match on the remote node (asymmetry confirmed: it is self-match, not a stray process)"
else
  _no "old form matched on the remote node too — something named $MARK actually exists; test invalid"
fi
echo

# ---------------------------------------------------------------------------
echo "[3] THE FIX — real helper, coordinator, nothing alive."
_out=$(_run_helper "$COORD" 3)
echo "$_out" | sed 's/^/    /'
_v=$(_verdict "$_out")
if [ "$_v" = "ALREADY_DEAD" ]; then
  _ok "helper reports ALREADY_DEAD on the coordinator (no self-match)"
else
  _no "expected ALREADY_DEAD, got '${_v:-<none>}' — helper is self-matching"
fi
echo

# ---------------------------------------------------------------------------
echo "[4] THE FIX — real helper, remote node, nothing alive."
_out=$(_run_helper "$REMOTE" 3)
_v=$(_verdict "$_out")
echo "    verdict=$_v"
if [ "$_v" = "ALREADY_DEAD" ]; then
  _ok "helper reports ALREADY_DEAD on the remote node"
else
  _no "expected ALREADY_DEAD, got '${_v:-<none>}'"
fi
echo

# ---------------------------------------------------------------------------
echo "[5] TRUE POSITIVE — a real process matching the marker MUST be found,"
echo "    SIGTERMed, and confirmed gone. Proves [3]/[4] are not vacuous."
echo "    (dummy 'sleep' started and reaped by this test; exo is untouched)"
_dummy_rc=0
_dummy=$(ssh "$COORD" "nohup /bin/sh -c 'exec -a $MARK sleep 300' >/dev/null 2>&1 & echo \$!" 2>&1) || _dummy_rc=$?
echo "    started dummy pid=$_dummy on $COORD"
sleep 1
_seen=$(ssh "$COORD" "pgrep -f '[${MARK:0:1}]${MARK:1}' | tr '\n' ' '" 2>&1)
echo "    visible to bracket-trick pgrep: [${_seen}]"
if [ -n "${_seen// /}" ]; then
  _ok "dummy is visible to the helper's own matching logic"
  _out=$(_run_helper "$COORD" 10)
  echo "$_out" | sed 's/^/    /'
  _v=$(_verdict "$_out")
  if [ "$_v" = "CLEAN_EXIT" ]; then
    _ok "helper SIGTERMed the dummy and confirmed CLEAN_EXIT (no SIGKILL)"
  else
    _no "expected CLEAN_EXIT, got '${_v:-<none>}'"
  fi
  _left=$(ssh "$COORD" "pgrep -f '[${MARK:0:1}]${MARK:1}' | tr '\n' ' '" 2>&1)
  if [ -z "${_left// /}" ]; then
    _ok "dummy is gone afterwards"
  else
    _no "dummy survived: $_left"
    ssh "$COORD" "pkill -9 -f '[${MARK:0:1}]${MARK:1}'" >/dev/null 2>&1
  fi
else
  _no "could not start/see the dummy process — case 5 inconclusive"
  ssh "$COORD" "pkill -9 -f '[${MARK:0:1}]${MARK:1}'" >/dev/null 2>&1
fi
echo

# ---------------------------------------------------------------------------
echo "[6] ARGV HYGIENE — the helper's pattern text must appear in NO argv"
echo "    on the node while it runs."
echo "    The checker greps with the BRACKET form, so the checker's own argv"
echo "    (which necessarily carries the bracketed text) cannot match itself;"
echo "    only a genuine PLAINTEXT leak by the helper can. Expected count: 0."
_bg_out=$(mktemp)
( _run_helper "$COORD" 6 >"$_bg_out" 2>&1 ) &
_bgpid=$!
sleep 2
_argv=$(ssh "$COORD" "ps -Ao pid,command | grep -cE '[${MARK:0:1}]${MARK:1}' || true" 2>&1)
echo "    argvs containing plaintext '\$MARK' while the helper runs: $_argv"
if [ "${_argv:-1}" -eq 0 ]; then
  _ok "helper leaks no pattern text into any argv (0 = clean; the old inline form would show >=1)"
else
  _no "pattern text visible in $_argv argv(s) — delivery channel is leaking"
  ssh "$COORD" "ps -Ao pid,command | grep -E '[${MARK:0:1}]${MARK:1}'" 2>&1 | sed 's/^/      /'
fi
wait $_bgpid 2>/dev/null
rm -f "$_bg_out"
echo

# ---------------------------------------------------------------------------
echo "[7] ARGV HYGIENE, NEGATIVE CONTROL — the same check MUST catch the OLD"
echo "    inline form's leak, or case [6] proves nothing."
ssh "$COORD" "pgrep -f 'python.*$MARK' >/dev/null 2>&1; sleep 4" &
_leakpid=$!
sleep 2
_argv_old=$(ssh "$COORD" "ps -Ao pid,command | grep -cE '[${MARK:0:1}]${MARK:1}' || true" 2>&1)
echo "    argvs containing plaintext '\$MARK' during an OLD-form ssh: $_argv_old"
if [ "${_argv_old:-0}" -ge 1 ]; then
  _ok "check detects the old form's argv leak (so case [6]'s 0 is meaningful)"
else
  _no "check did not detect the old form's leak — case [6] is not a valid test"
fi
wait $_leakpid 2>/dev/null
echo

# ---------------------------------------------------------------------------
echo "======================================================================"
echo "PASS=$PASS  FAIL=$FAIL"
if [ "$BUG_REPRODUCIBLE" -eq 0 ]; then
  echo "NOTE: not driven from $COORD — cases 1-3 were not a real test of the bug."
fi
echo "======================================================================"
[ "$FAIL" -eq 0 ]
