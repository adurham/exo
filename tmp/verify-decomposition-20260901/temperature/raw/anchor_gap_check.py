#!/usr/bin/env python3
# Distance from the arm window start (b) to the last MTP-PROF line strictly
# before it, per node. Emitted so the analysis dispatch knows the anchor gap.
import subprocess

WINDOWS = {"macstudio-m4-1": 141198005, "macstudio-m4-2": 166599375}

SCRIPT = r'''
import sys
b = int(sys.argv[1])
path = "/Users/adam.durham/exo.log"
CHUNK = 2_000_000
with open(path, "rb") as f:
    start = max(0, b - CHUNK)
    f.seek(start)
    head = f.read(b - start)
idx = head.rfind(b"MTP-PROF")
if idx < 0:
    print(f"NOT_FOUND_within_{CHUNK}_bytes")
else:
    gap = len(head) - idx
    print(f"gap_bytes={gap} (last MTP-PROF {gap} bytes before window start; searched {len(head)} bytes back)")
'''

for node, offset in WINDOWS.items():
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", node, "python3", "-", str(offset)],
        input=SCRIPT, capture_output=True, text=True, timeout=120,
    )
    print(node, "->", r.stdout.strip() or r.stderr.strip()[-200:])