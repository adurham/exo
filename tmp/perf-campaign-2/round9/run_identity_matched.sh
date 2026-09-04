#!/bin/bash
# Round-9 byte-identity capture at budgets MATCHED to boot1's RV=200 capture
# (short 64, 2k 64, 89k 200) so the diff is apples-to-apples.
# Usage: run_identity_matched.sh <LABEL e.g. RV0b4>
set -u
A="$1"
R9=/Users/adam.durham/repos/exo/tmp/perf-campaign-2/round9
RES="$R9/results"
mkdir -p "$RES"
cd /Users/adam.durham/repos/exo || exit 1
S=/tmp/r9_identm_${A}_status.txt
echo "IDENTM_START $A $(date -u +%FT%TZ)" > "$S"
run() {
  python3 bench/long_decode_probe.py "$2" --max-tokens "$3" --run-id r9id \
    --tag identm_${A}_$1 --out "$RES/identitym_${A}_$1.json" >> /tmp/r9_identm_${A}.log 2>&1
  echo "$1 rc=$? $(date -u +%FT%TZ)" >> "$S"
}
run short 20 64
run 2k 2000 64
run 89k 89000 200
echo "IDENTM_DONE $A $(date -u +%FT%TZ)" >> "$S"
