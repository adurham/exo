#!/bin/bash
# Round-10 89K same-arm self-control (determinism check). Usage:
#   run_89k_selfcontrol.sh <ARM>
# Runs the 89K prompt THREE times at fixed --run-id r10id, --max-tokens 200,
# target_tokens 89000, identical every time. Output to round10/results/ as
# steelbi_<ARM>_89k_{1,2,3}.json plus .txt extractions of reasoning_content
# and content (concatenated, reasoning first then content).
set -u
A="$1"
R10=/Users/adam.durham/repos/exo/tmp/perf-campaign-2/round10
RES="$R10/results"
mkdir -p "$RES"
cd /Users/adam.durham/repos/exo || exit 1
S=/tmp/r10_89kself_${A}_status.txt
echo "89K_SELFCONTROL_START $A $(date -u +%FT%TZ)" > "$S"

for i in 1 2 3; do
  python3 bench/long_decode_probe.py 89000 --max-tokens 200 --run-id r10id \
    --tag steelbi_${A}_89k_$i --out "$RES/steelbi_${A}_89k_$i.json" >> /tmp/r10_89kself_${A}.log 2>&1
  echo "89k_$i rc=$? $(date -u +%FT%TZ)" >> "$S"
  python3 -c "
import json,sys
d=json.load(open('$RES/steelbi_${A}_89k_$i.json'))
open('$RES/steelbi_${A}_89k_$i.txt','w').write(d['reasoning_content'] + d['content'])
ss = d.get('server_stats') or {}
print('89k_$i', 'ptok', d['prompt_tokens'], 'ctok', d['completion_tokens'],
      'prefix_cache_hit', ss.get('prefix_cache_hit', 'MISSING'),
      'content_chars', d['content_chars'], 'run_id', d['run_id'])
" >> "$S" 2>&1
done
echo "89K_SELFCONTROL_DONE $A $(date -u +%FT%TZ)" >> "$S"
