#!/usr/bin/env bash
# Hunt for the REAL acceptance telemetry from production's live log.
LOG="$HOME/exo.log"
ls -la "$LOG" 2>&1

echo
echo "=== accept_rate / tokens-per-cycle telemetry (production's own) ==="
grep -aoE 'accept_rate=[0-9.]+ tokens/cycle=[0-9.]+[^"]{0,60}' "$LOG" 2>/dev/null | tail -20
echo "  (count: $(grep -ac 'accept_rate=' "$LOG" 2>/dev/null))"

echo
echo "=== the telemetry line's full shape (one sample) ==="
grep -a 'accept_rate=' "$LOG" 2>/dev/null | tail -3

echo
echo "=== DSpark overlay status lines ==="
grep -aiE 'dspark' "$LOG" 2>/dev/null | grep -aiE 'overlay|fallback|falling back|loaded|guard' | tail -12

echo
echo "=== MTP / speculative mode lines ==="
grep -aiE 'MTP-[0-9]|speculative' "$LOG" 2>/dev/null | tail -12

echo
echo "DONE"
