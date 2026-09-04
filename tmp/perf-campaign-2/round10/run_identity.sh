#!/bin/bash
# Round-10 byte-identity capture. Usage: run_identity.sh <ARM e.g. RV200|RV0> <CAPNUM>
# Three deterministic prompts (fixed --run-id r10id), temp=0, content saved to
# results/identity_<ARM>_c<CAPNUM>_{short,2k,89k}.json/.txt/.reasoning.txt
#
# Token budgets are FIXED and must be identical across arms/captures:
#   short: target_tokens 20,   --max-tokens 64
#   2k:    target_tokens 2000, --max-tokens 64
#   89k:   target_tokens 89000,--max-tokens 200
#
# The capture-number arg lets the same arm be captured twice for a
# within-arm self-control (byte-identity gate).
set -u
# Pin interpreter: bare python3 can resolve to Homebrew python (no httpx);
# this must match the interpreter used for the already-collected arm data.
PY=/usr/bin/python3
A="$1"
C="$2"
R10=/Users/adam.durham/repos/exo/tmp/perf-campaign-2/round10
RES="$R10/results"
mkdir -p "$RES"
cd /Users/adam.durham/repos/exo || exit 1
S=/tmp/r10_ident_${A}_c${C}_status.txt
echo "IDENT_START $A c$C $(date -u +%FT%TZ)" > "$S"

run() {  # name target maxtok
  "$PY" bench/long_decode_probe.py "$2" --max-tokens "$3" --run-id r10id \
    --tag ident_${A}_c${C}_$1 --out "$RES/identity_${A}_c${C}_$1.json" >> /tmp/r10_ident_${A}_c${C}.log 2>&1
  echo "$1 rc=$? $(date -u +%FT%TZ)" >> "$S"
  "$PY" -c "
import json,sys
d=json.load(open('$RES/identity_${A}_c${C}_$1.json'))
open('$RES/identity_${A}_c${C}_$1.txt','w').write(d['content'])
open('$RES/identity_${A}_c${C}_$1.reasoning.txt','w').write(d['reasoning_content'])
ss = d.get('server_stats') or {}
print('$1', 'ptok', d['prompt_tokens'], 'ctok', d['completion_tokens'],
      'prefix_cache_hit', ss.get('prefix_cache_hit', 'MISSING'),
      'content_chars', d['content_chars'], 'run_id', d['run_id'])
" >> "$S" 2>&1
}

run short 20 64
run 2k 2000 64
run 89k 89000 200
echo "IDENT_DONE $A c$C $(date -u +%FT%TZ)" >> "$S"
