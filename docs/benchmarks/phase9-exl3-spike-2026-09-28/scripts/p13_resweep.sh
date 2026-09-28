#!/bin/sh
# p13_resweep.sh -- re-sweep geometry on the PATCHED library (SWAR+XDIRECT
# defaults ON). The ALU balance changed, so the old optimum may have moved.
PY="$HOME/phase1-exl3/.venv/bin/python"
S="$HOME/p13-sweep2-20260928"
mkdir -p "$S"
run() { # T C label
  OUT="$S/$3.log"
  EXL3_MOE_A2_TILES=$1 EXL3_MOE_CHUNK=$2 BENCH_EXPERTS=384 \
    "$PY" "$HOME/p2h_sweep.py" 60 > "$OUT" 2>&1
  echo "DONE $3 rc=$? $(date '+%T') :: $(grep 'R=1:' "$OUT" | head -1)"
}
run 1 0    T1-C0
run 1 384  T1-C384
run 1 1152 T1-C1152
run 2 0    T2-C0
run 2 1152 T2-C1152
echo "=== FUSED=0 on patched lib ==="
EXL3_MOE_FUSED=0 BENCH_EXPERTS=384 "$PY" "$HOME/p2_final_check.py" > "$S/fused0.log" 2>&1
echo "DONE fused0 rc=$? $(date '+%T') :: $(grep 'decode R=1' "$S/fused0.log")"
echo "RESWEEP_COMPLETE $(date '+%F %T')"
