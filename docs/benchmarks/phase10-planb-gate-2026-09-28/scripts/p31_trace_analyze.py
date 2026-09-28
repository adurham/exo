#!/usr/bin/env python3
"""Day-2 gate analysis: full 40-layer routing trace -> concentration + LRU residency -> I/O requirement."""
import json, os, collections, statistics

SCR = os.path.dirname(os.path.abspath(__file__))
TR = json.load(open(os.path.join(SCR, "p30-logs", "p30_exl3_trace.json")))
recs = TR["records"]
print("[chk] n_steps=%s prompt_tokens=%s records=%d" % (TR.get("n_steps"), TR.get("prompt_tokens"), len(recs)))
print("[chk] gate_meta:", json.dumps(TR.get("gate_meta", {}))) 

per = collections.defaultdict(list)
for layer, idx in recs:
    per[layer].append(tuple(idx))
layers = sorted(per)
print("[chk] layers=%d  rows@first/mid/last: %s" % (len(layers), [len(per[l]) for l in (layers[0], layers[20], layers[-1])]))
lens = collections.Counter(len(t) for l in layers for t in per[l])
print("[chk] idx tuple lengths:", dict(lens))
ST2 = [(130,206,317,251,197,72),(130,206,197,238,323,237)]
print("[chk] L0 first-2 == selftest:", list(per[layers[0]][:2]) == ST2)

cov = {l: len({e for t in per[l] for e in t}) for l in layers}
print("[stat] distinct experts touched per layer: min=%d med=%.0f max=%d of 384" % (min(cov.values()), statistics.median(cov.values()), max(cov.values())))
sh10 = []
for l in layers:
    c = collections.Counter(e for t in per[l] for e in t)
    sh10.append(sum(v for _, v in c.most_common(10)) / sum(c.values()))
print("[stat] top-10 share per layer: med=%.3f min=%.3f max=%.3f" % (statistics.median(sh10), min(sh10), max(sh10)))

def lru_hits(seq, cap):
    od = collections.OrderedDict(); h = 0
    for t in seq:
        for e in t:
            if e in od:
                h += 1; od.move_to_end(e)
            else:
                od[e] = 1
                if len(od) > cap: od.popitem(last=False)
    return h

def hit_rate(cap, skip=0):
    H = N = 0
    for l in layers:
        seq = per[l][skip:]
        H += lru_hits(seq, cap); N += 6 * len(seq)
    return H / N

print("[lru] cap sweep (hit overall / steady tokens>100):")
for cap in (128, 192, 256, 320):
    print("      cap=%3d  all=%.4f  steady=%.4f" % (cap, hit_rate(cap, 0), hit_rate(cap, 100)))

CAP = 256
ph = {l: lru_hits(per[l][100:], CAP) / (6 * len(per[l][100:])) for l in layers}
print("[lru] per-layer steady hit @cap256:", " ".join("%02d:%.2f" % (l, ph[l]) for l in layers))

hitss = hit_rate(CAP, 100)
miss_layer = (1 - hitss) * 6
print("[io] steady hit @cap256 = %.4f -> misses/token/layer = %.4f -> per-rank (20 layers) = %.3f fetches/token" % (hitss, miss_layer, miss_layer * 20))

NATIVE_B = 7.219e9 / 384
print("[io] native expert fetch = %.2f MB" % (NATIVE_B / 1e6))
try:
    LB = json.load(open(os.path.join(SCR, "p30-logs", "p31_layer_bytes.json")))
except Exception:
    LB = None
def report(name, btok):
    print("[io] %s: %.1f MB/token/rank" % (name, btok / 1e6))
    for S in (25, 32):
        print("       required at %d tok/s = %.2f GB/s    (measured 6.20)" % (S, btok * S / 1e9))
    print("       ceiling @6.2 GB/s = %.1f tok/s" % (6.2e9 / btok))
report("NATIVE flat 18.80MB x 20 layers", miss_layer * 20 * NATIVE_B)
if LB:
    for half, nm in ((layers[:20], "layers 00-19"), (layers[20:], "layers 20-39")):
        b = sum(miss_layer * LB[str(l)] for l in half)
        report("EXL3 bytes, %s" % nm, b)

print("ANALYSIS_DONE")
