#!/usr/bin/env bash
# Runner for the 3-node distributed hybrid test.
# Usage: bash tests/run_distributed_hybrid.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_distributed_hybrid.py"

export PYENV_VERSION=3.12.12
PYTHON="$(python3 -c 'import sys; print(sys.executable)')"

echo "=== Running 3-node hybrid distributed test ==="
echo "Python: $PYTHON"
echo "Test:   $TEST_SCRIPT"
echo ""

python3 -c "
import os, sys, subprocess, tempfile, json

python = '$PYTHON'
script = '$TEST_SCRIPT'

ring_hosts = [['127.0.0.1:32323'], ['127.0.0.1:32324'], ['127.0.0.1:32325']]
hostfile = json.dumps(ring_hosts)

procs = []
for rank in range(3):
    env = os.environ.copy()
    env['MLX_RANK'] = str(rank)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        f.write(hostfile)
        env['MLX_HOSTFILE'] = f.name
    p = subprocess.Popen(
        [python, script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    procs.append(p)

failed = False
for rank, p in enumerate(procs):
    out, _ = p.communicate(timeout=120)
    print(f'--- RANK {rank} (exit={p.returncode}) ---')
    print(out.decode())
    if p.returncode != 0:
        failed = True

if failed:
    print('❌ TEST FAILED')
    sys.exit(1)
else:
    print('✅ ALL RANKS PASSED')
"
