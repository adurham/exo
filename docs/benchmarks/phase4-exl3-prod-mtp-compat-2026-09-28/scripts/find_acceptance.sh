#!/usr/bin/env bash
# Find production's DSpark/MTP acceptance accounting.
# exo logs at DEBUG; extract only acceptance-related lines.
LOG="$HOME/exo.log"
echo "=== log size ==="
ls -la "$LOG" 2>&1
echo
echo "=== acceptance-ish line shapes (deduped templates) ==="
grep -aoE '\[[A-Za-z0-9_-]*(MTP|DSPARK|SPEC)[A-Za-z0-9_-]*\][^0-9]{0,40}' "$LOG" 2>/dev/null \
  | sort | uniq -c | sort -rn | head -30
echo
echo "=== any 'accept' / 'acceptance' lines (sample) ==="
grep -aiE 'accept' "$LOG" 2>/dev/null | tail -25
echo
echo "=== EXO_DSV4_MTP_LOG / env knobs referenced in code ==="
cd "$HOME/repos/exo" 2>/dev/null || exit 0
grep -rn "EXO_DSV4" --include=*.py src/ mlx-lm/ 2>/dev/null | sed 's/:.*EXO_DSV4/ -> EXO_DSV4/' | sort -u | head -40
echo
echo "DONE"
