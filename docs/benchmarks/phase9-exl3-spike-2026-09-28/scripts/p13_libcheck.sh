#!/bin/sh
# Runs after the verify driver (PID $1): measure the ADOPTED library state.
# No env vars -> library defaults (SWAR+XDIRECT ON) are what's exercised.
# p16's own script-level patches stay OFF (EXL3_DIRECT/EXL3_SWAR unset != its
# gate names EXL3_DIRECT... wait, they ARE its gate names -- but unset means
# its patching is skipped; the LIBRARY gates are EXL3_DECODE_SWAR/EXL3_XDIRECT,
# also unset -> library defaults ON). md5s must match the pre-change REF.
V1="$1"
while kill -0 "$V1" 2>/dev/null; do sleep 15; done
cd "$HOME" || exit 1
PY="$HOME/phase1-exl3/.venv/bin/python"
echo "=== library-adopted p16 (clean env, defaults ON) $(date '+%T') ==="
BENCH_EXPERTS=384 "$PY" "$HOME/p16_direct_probe.py" > "$HOME/p13-libcheck-adopted.log" 2>&1
echo "rc=$? adopted"
echo "=== library gates OFF via env (proves override works) $(date '+%T') ==="
EXL3_DECODE_SWAR=0 EXL3_XDIRECT=0 BENCH_EXPERTS=384 "$PY" "$HOME/p16_direct_probe.py" > "$HOME/p13-libcheck-off.log" 2>&1
echo "rc=$? off"
echo "P13_LIBCHECK_DONE $(date '+%T')"
