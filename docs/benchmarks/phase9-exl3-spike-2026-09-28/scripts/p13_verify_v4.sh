#!/bin/sh
cd "$HOME" || exit 1
PY="$HOME/phase1-exl3/.venv/bin/python"
echo "=== arm patched-defaults ==="
BENCH_EXPERTS=384 "$PY" "$HOME/p2_final_check.py" > "$HOME/p13-verify-v4-default.log" 2>&1
echo "rc=$? default"
echo "=== arm v4-gates-off ==="
EXL3_DECODE_SWAR=0 EXL3_XDIRECT=0 BENCH_EXPERTS=384 "$PY" "$HOME/p2_final_check.py" > "$HOME/p13-verify-v4-off.log" 2>&1
echo "rc=$? off"
echo "P13_VERIFY_ALL_DONE $(date '+%T')"
