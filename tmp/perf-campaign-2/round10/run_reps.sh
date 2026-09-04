#!/bin/bash
# Round-10 rep runner. Usage: run_reps.sh <LABEL e.g. A|Z1|B|Z2>
# 5x 2K reps then 25x short reps, sequential, JSON under round10/results/
set -u
L="$1"
R10=/Users/adam.durham/repos/exo/tmp/perf-campaign-2/round10
RES="$R10/results"
mkdir -p "$RES"
cd /Users/adam.durham/repos/exo || exit 1
S=/tmp/r10_reps_${L}_status.txt
echo "REPS_START $L $(date -u +%FT%TZ)" > "$S"
for i in 1 2 3 4 5; do
  python3 bench/long_decode_probe.py 2000 --max-tokens 16 --tag ${L}_2k_r$i --out "$RES/${L}_2k_r$i.json" >> /tmp/r10_reps_${L}.log 2>&1
  echo "2k_r$i rc=$? $(date -u +%FT%TZ)" >> "$S"
done
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do
  python3 bench/long_decode_probe.py 20 --max-tokens 16 --tag ${L}_short_r$i --out "$RES/${L}_short_r$i.json" >> /tmp/r10_reps_${L}.log 2>&1
  echo "short_r$i rc=$? $(date -u +%FT%TZ)" >> "$S"
done
echo "REPS_DONE $L $(date -u +%FT%TZ)" >> "$S"
