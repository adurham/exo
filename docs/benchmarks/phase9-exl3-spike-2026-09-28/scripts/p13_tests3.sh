#!/bin/sh
cd ~/repos/ref/PonyExl3 || exit 1
PY=~/phase1-exl3/.venv/bin/python
UV=$(command -v uv || echo ~/.local/bin/uv)
echo "uv=$UV"
"$UV" pip install --python "$PY" -q pytest 2>&1 | tail -3
echo "=== moe_activation ==="
$PY -m pytest tests/test_moe_activation.py -x -q 2>&1 | tail -5
echo "=== mlx_gemv ==="
$PY -m pytest tests/test_mlx_gemv.py -x -q 2>&1 | tail -8
echo "=== codebook ==="
$PY -m pytest tests/test_codebook.py -x -q 2>&1 | tail -5
echo "=== gemma4_model ==="
$PY -m pytest tests/test_gemma4_model.py -x -q 2>&1 | tail -5
echo "PONYEXL3_TESTS3_DONE $(date '+%T')"
