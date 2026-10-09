#!/bin/bash
# RESTORE production for the Q1 dense-quant eval round.
# Target: exo fb4f9290b + mlx-lm 16830e1, BOTH nodes, DSV41_DENSE unset.
set -uo pipefail
cd ~/repos/exo || exit 1
echo "== idle guard =="
~/repos/exo/.venv/bin/python /private/tmp/phase20-campaign/bench/phase20_guard.py wait-idle --max-wait 1800 --poll 30 || \
  ~/repos/exo/.venv/bin/python /private/tmp/phase20-campaign/bench/phase20_guard.py idle
echo "== checking out production trees =="
git checkout --detach fb4f9290b
git -C mlx-lm checkout 16830e1
echo "exo=$(git rev-parse --short HEAD) mlx-lm=$(git -C mlx-lm rev-parse --short HEAD)"
grep -c "AffineProj.from_weight" mlx-lm/mlx_lm/models/deepseek_v41/exl3_build.py || true   # expect 0 on prod
unset DSV41_DENSE DSV41_DENSE_TP
export EXO_TARGET_BRANCH=deploy/next18-identity   # fb4f9290b lives here (not on origin/main)
mkdir -p /tmp/q1quant
./start_cluster.sh > /tmp/q1quant/restore_prod.log 2>&1
grep -aE "Nodes synchronized|READY|HEALTHY" /tmp/q1quant/restore_prod.log | tail
~/repos/exo/.venv/bin/python /private/tmp/phase20-campaign/bench/phase20_guard.py canary
for h in studio1 studio2; do
  ssh -o BatchMode=yes "$h" 'printf "%s exo=%s mlx=%s\n" "$(hostname -s)" "$(git -C ~/repos/exo rev-parse --short HEAD)" "$(git -C ~/repos/exo/mlx-lm rev-parse --short HEAD)"'
done
