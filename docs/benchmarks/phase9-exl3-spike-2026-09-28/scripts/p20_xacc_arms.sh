#!/bin/sh
cd "$HOME" || exit 1
PY="$HOME/phase1-exl3/.venv/bin/python"
echo "=== xacc arm xdir ==="
EXL3_XACC=xdir BENCH_EXPERTS=384 "$PY" "$HOME/p20_xaccess_probe.py" > "$HOME/p20-xacc-xdir.log" 2>&1
echo "rc=$? xdir"
echo "=== xacc arm imm ==="
EXL3_XDIRECT=0 EXL3_XACC=imm BENCH_EXPERTS=384 "$PY" "$HOME/p20_xaccess_probe.py" > "$HOME/p20-xacc-imm.log" 2>&1
echo "rc=$? imm"
echo "P20_XACC_DONE $(date '+%T')"
