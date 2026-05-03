#!/usr/bin/env bash
# After-GPQA-bench runbook: swap from DSv4-6bit to DSv4-8bit and re-run
# only the questions that failed at 6bit. Compare per-question results.
#
# Reads the 6bit final results JSON to determine which question indices
# to retry, redeploys the cluster on 8bit via DSV4_MODEL_ID, then runs
# the targeted bench at 8bit and diffs against 6bit.

set -euo pipefail

EXO_HOST="${EXO_HOST:-adams-mac-studio-m4-1.local}"
RESULTS_DIR_6BIT="eval_results/mlx-community_DeepSeek-V4-Flash-6bit/gpqa_diamond"
RESULTS_DIR_8BIT="eval_results/mlx-community_DeepSeek-V4-Flash-8bit/gpqa_diamond"
MODEL_8BIT="mlx-community/DeepSeek-V4-Flash-8bit"

# ── 1. Compute the indices that failed on 6bit ─────────────────────────────
RESULTS_FILE_6BIT=$(ls -t "$RESULTS_DIR_6BIT"/c1_*.json 2>/dev/null | head -1)
if [[ -z "$RESULTS_FILE_6BIT" ]]; then
  echo "ERROR: 6bit results file not found in $RESULTS_DIR_6BIT"
  exit 1
fi
echo "[swap_8bit] reading 6bit results from $RESULTS_FILE_6BIT"

FAILED_INDICES=$(python3 -c "
import json
with open('$RESULTS_FILE_6BIT') as f:
    d = json.load(f)
fails = [r['question_id'] for r in d['results'] if not r.get('correct')]
print(','.join(map(str, sorted(fails))))
")
echo "[swap_8bit] failed indices on 6bit: ${FAILED_INDICES}"
if [[ -z "$FAILED_INDICES" ]]; then
  echo "[swap_8bit] no failures to re-run; exiting"
  exit 0
fi

# ── 2. Redeploy cluster on 8bit (start_cluster.sh handles delete + place) ──
echo "[swap_8bit] redeploying cluster with DSV4_MODEL_ID=${MODEL_8BIT}..."
DSV4_MODEL_ID="$MODEL_8BIT" ./start_cluster.sh

# ── 3. Run targeted re-bench at 8bit on failed indices ─────────────────────
mkdir -p "$RESULTS_DIR_8BIT"
echo "[swap_8bit] running targeted re-bench at 8bit on indices: ${FAILED_INDICES}"
EXO_HOST="$EXO_HOST" uv run python bench/exo_eval.py \
  --tasks gpqa_diamond \
  --indices "$FAILED_INDICES" \
  --num-concurrent 1 \
  --reuse-instance \
  --model "$MODEL_8BIT"

# ── 4. Per-question diff ───────────────────────────────────────────────────
RESULTS_FILE_8BIT=$(ls -t "$RESULTS_DIR_8BIT"/c1_*.json 2>/dev/null | head -1)
echo
echo "[swap_8bit] comparing 6bit vs 8bit on retried indices..."
python3 - "$RESULTS_FILE_6BIT" "${RESULTS_FILE_8BIT:-}" "$FAILED_INDICES" << 'PY'
import json, sys
six_path, eight_path, failed_csv = sys.argv[1], sys.argv[2], sys.argv[3]
fails = [int(x) for x in failed_csv.split(',')]
with open(six_path) as f:
    six_bit = {r['question_id']: r for r in json.load(f)['results']}
if not eight_path:
    print("8bit results JSON not found")
    raise SystemExit(0)
with open(eight_path) as f:
    eight_bit = {r['question_id']: r for r in json.load(f)['results']}
print(f"\n{'qid':>3}  {'6bit':>6}  {'8bit':>6}  {'gold':>4}  recovered")
print('-' * 50)
recovered = 0
for q in sorted(fails):
    s = six_bit.get(q, {})
    e = eight_bit.get(q, {})
    sx = s.get('extracted_answer','—') or '<none>'
    ex = e.get('extracted_answer','—') or '<none>'
    g  = s.get('gold_answer','—')
    fixed = '✅' if e.get('correct') and not s.get('correct') else ''
    if fixed: recovered += 1
    print(f"Q{q:<3}  {sx:>6}  {ex:>6}  {g:>4}  {fixed}")
print('-' * 50)
print(f"recovered: {recovered}/{len(fails)} = {100*recovered/len(fails):.1f}%")
PY
