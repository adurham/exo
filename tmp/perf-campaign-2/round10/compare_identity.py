#!/usr/bin/env python3
"""Round-10 identity comparator. Usage: compare_identity.py FILE1.json FILE2.json ...

For each JSON result file: concatenates reasoning_content + content, prints
char count and the first 12 hex chars of the sha256, then reports which
files are byte-identical to which. Used for both the byte-identity gate
(cross-arm identity capture) and the 89K self-control (same-arm determinism).
"""
import hashlib
import json
import sys
from pathlib import Path


def load_concat(path):
    d = json.loads(Path(path).read_text())
    reasoning = d.get("reasoning_content") or ""
    content = d.get("content") or ""
    return reasoning + content


def main(argv):
    if not argv:
        print("usage: compare_identity.py FILE1.json FILE2.json ...", file=sys.stderr)
        return 2

    info = []
    for path in argv:
        text = load_concat(path)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        info.append({"path": path, "chars": len(text), "sha256_12": digest})
        print(f"{path}\tchars={len(text)}\tsha256_12={digest}")

    print()
    print("=== identity groups (byte-identical concatenations) ===")
    seen = []
    for i, a in enumerate(info):
        if any(a["path"] in g for g in seen):
            continue
        group = [a["path"]]
        for b in info[i + 1:]:
            if b["sha256_12"] == a["sha256_12"]:
                group.append(b["path"])
        seen.append(group)
        if len(group) > 1:
            print(f"IDENTICAL: {group}")
        else:
            print(f"UNIQUE: {group[0]}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
