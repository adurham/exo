#!/bin/bash
# Round-9 byte-identity capture. Usage: run_identity.sh <ARM e.g. RV200|RV0>
# Three deterministic prompts (fixed --run-id r9id), temp=0, content saved to
# results/identity_<ARM>_{short,2k,89k}.txt
set -u
A="$1"
R9=/Users/adam.durham/repos/exo/tmp/perf-campaign-2/round9
RES="$R9/results"
mkdir -p "$RES"
cd /Users/adam.durham/repos/exo || exit 1
S=/tmp/r9_ident_${A}_status.txt
echo "IDENT_START $A $(date -u +%FT%TZ)" > "$S"

run() {  # name target maxtok
  python3 bench/long_decode_probe.py "$2" --max-tokens "$3" --run-id r9id \
    --tag ident_${A}_$1 --out "$RES/identity_${A}_$1.json" >> /tmp/r9_ident_${A}.log 2>&1
  echo "$1 rc=$? $(date -u +%FT%TZ)" >> "$S"
  python3 -c "
import json,sys
d=json.load(open('$RES/identity_${A}_$1.json'))
open('$RES/identity_${A}_$1.txt','w').write(d['content'])
open('$RES/identity_${A}_$1.reasoning.txt','w').write(d['reasoning_content'])
print('$1', 'ptok', d['prompt_tokens'], 'ctok', d['completion_tokens'],
      'content_chars', d['content_chars'], 'run_id', d['run_id'])
" >> "$S" 2>&1
}

run short 20 400
run 2k 2000 400
run 89k 89000 200
echo "IDENT_DONE $A $(date -u +%FT%TZ)" >> "$S"
