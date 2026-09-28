#!/bin/bash
# p45 -> p44-rerun chain (node1). p44 re-run doubles the serving-geometry C arm
# so the key 0.430/1.040/1.459 numbers are reproducible, not a single sample.
cd "$HOME" || exit 1
~/phase1-exl3/.venv/bin/python ~/p45_mxfp4_geometry.py > ~/p45-serving.out 2>&1
echo "P45_EXIT=$?"
~/phase1-exl3/.venv/bin/python ~/p44_full_layer.py > ~/p44-serving-rerun.out 2>&1
echo "P44_EXIT=$?"
echo CHAIN_DONE
