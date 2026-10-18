#!/usr/bin/env python3
"""A/B report for the 300K decode question, plus the cold-1M row if present.

Reference points
  * WRITEUP §9.2 300K row (measured at 5a18b81/991cf11): cached=0, prefill 1535.25 s,
    195.4 tok/s, client decode 5.9, server decode 6.0, 189 completion tokens.
  * REPRO-RUN-2026-10-18 300K row (measured at b88d65dd): cached=159,744, prefill 680.98 s,
    206.0 tok/s, client 21.86, server 22.12.
"""
import json
import os
import sys

OUT = os.path.dirname(os.path.abspath(__file__)) + "/out"

CLAIM_300K = dict(cached=0, prefill_s=1535.25, prefill_tps=195.4, client=5.9, server=6.0,
                  completion=189)
PRIOR_300K_B88 = dict(cached=159744, prefill_s=680.98, prefill_tps=206.0, client=21.86,
                      server=22.12, completion=200)


def load(label):
    p = os.path.join(OUT, label + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def show(label, row):
    if row is None:
        print(f"  {label}: (no artifact)")
        return
    print(f"  {label}: prompt={row.get('prompt_tokens')} cached={row.get('cached_tokens')} "
          f"completion={row.get('completion_tokens')} finish={row.get('finish_reason')} "
          f"prefill={row.get('prefill_s')}s ({row.get('prefill_tok_s_uncached')} tok/s) "
          f"client_dec={row.get('client_decode_tok_s')} server_dec={row.get('server_decode_tok_s')} "
          f"wall={row.get('wall_s')}s errs={len(row.get('error_events') or [])}")


print("300K A/B (cold rows, park store emptied before each boot)")
print("-" * 78)
rows = {lbl: load(lbl) for lbl in ("ab-5a18b81-d300k", "ab-b88d65dd-d300k")}
for lbl, row in rows.items():
    show(lbl, row)
print("\nreferences:")
print(f"  writeup §9.2 (5a18b81/991cf11): cached={CLAIM_300K['cached']} "
      f"prefill={CLAIM_300K['prefill_s']}s ({CLAIM_300K['prefill_tps']} tok/s) "
      f"client_dec={CLAIM_300K['client']} server_dec={CLAIM_300K['server']} "
      f"completion={CLAIM_300K['completion']}")
print(f"  2026-10-18 repro (b88d65dd, warm prefix): cached={PRIOR_300K_B88['cached']} "
      f"prefill={PRIOR_300K_B88['prefill_s']}s ({PRIOR_300K_B88['prefill_tps']} tok/s) "
      f"client_dec={PRIOR_300K_B88['client']} server_dec={PRIOR_300K_B88['server']}")

a, b = rows["ab-5a18b81-d300k"], rows["ab-b88d65dd-d300k"]
if a and b and a.get("server_decode_tok_s") and b.get("server_decode_tok_s"):
    ratio = b["server_decode_tok_s"] / a["server_decode_tok_s"]
    cold_equal = (a.get("cached_tokens") == 0 and b.get("cached_tokens") == 0)
    print(f"\nserver decode ratio b88d65dd / 5a18b81 = {ratio:.2f}x "
          f"(both cold: {cold_equal})")
    print("  ratio ~1  -> the 4x was NOT a code change between the SHAs; §9's 6 t/s "
          "figure came from different conditions than a cold re-run today.")
    print("  ratio ~4  -> the gain is in the 5a18b81..b88d65dd diff.")

for lbl in ("cold-1m-b88d65dd",):
    row = load(lbl)
    if row:
        print(f"\nCold 1M row:\n")
        show(lbl, row)
        for ev in (row.get("error_events") or [])[:4]:
            print(f"    error_event: {json.dumps(ev)[:300]}")
