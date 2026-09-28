#!/usr/bin/env bash
# Harvest real MTP/DSpark acceptance data from production's log.
# The exo log is DEBUG-level and huge; extract only the acceptance lines.
LOG="$HOME/exo.log"
echo "=== log ==="
ls -la "$LOG"
echo
echo "=== line count ==="
wc -l < "$LOG"

echo
echo "=== MTP / DSpark / acceptance mentions (counts) ==="
for pat in accept MTP DSpark dspark spec_accept mean_accept n_accepted; do
  printf '  %-14s %s\n' "$pat" "$(grep -c -F "$pat" "$LOG" 2>/dev/null || echo 0)"
done

echo
echo "=== acceptance-shaped lines (last 20) ==="
grep -F -e 'mean_accept' -e 'n_accepted' -e 'accept_hist' -e 'spec_accept' "$LOG" 2>/dev/null | tail -20

echo
echo "=== any DSpark overlay fallback / failure lines? ==="
grep -i -F -e 'dspark' "$LOG" 2>/dev/null | grep -i -e 'fail' -e 'fallback' -e 'missing' | tail -12

echo
echo "=== DSpark / MTP startup lines (did the overlay load?) ==="
grep -i -F -e 'dspark' "$LOG" 2>/dev/null | head -12

echo
echo "=== EXO_DSV4_MTP / DSPARK env actually in the process ==="
PID=$(pgrep -f '\.venv/bin/python -m exo' | head -1)
echo "  pid=$PID"
if [ -n "$PID" ]; then
  ps eww -p "$PID" 2>/dev/null | tr ' ' '\n' | grep -E '^EXO_(DSV4_MTP|DSV4_DSPARK|SPECULATIVE|SPECULATIVE_GAMMA)' | sort
fi
echo
echo "DONE"
