#!/bin/sh
cd "$HOME" || exit 1
PY="$HOME/phase1-exl3/.venv/bin/python"
for arm in control cheap; do
  echo "=== chain2 arm $arm $(date '+%T') ==="
  EXL3_CHAIN2=$arm BENCH_EXPERTS=384 "$PY" "$HOME/p19_chain2_probe.py" > "$HOME/p19-chain2-$arm.log" 2>&1
  echo "rc=$? arm=$arm"
done
echo "P19_CHAIN2_DONE $(date '+%T')"
