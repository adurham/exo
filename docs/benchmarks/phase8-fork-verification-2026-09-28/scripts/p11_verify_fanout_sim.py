#!/usr/bin/env python3
"""p11: verify-amplified cold-read sim for Plan B (streamed experts).

Input: p8c_trace.json -- real gate decisions recorded from the port
(gates tagged [-1(mtp), 0, 1, 2, 3]; 490 prompt + 256 decoded = 746 positions;
6 picks/position/layer for the 4 body layers present).

Model (mirrors p8e_esidency_sim2: per-layer LRU caches; rank = 20 layers):
  plain    : 1 row/step, advance 1  -> union = 1 row's picks
  spec a=X : 5 rows/step (anchor+4 drafts verified every chunk),
             advance = 1+4a (tokens committed per chunk)
  Cold bytes per committed token = |union - resident| * expert_bytes / advance.
  Steady state: skip first 85 positions (p8e convention).

Question (from the design consult): does chunk verify's 5-row fan-out collapse
Plan B's ~40 tok/s SSD ceiling?
"""
import json
from collections import defaultdict

TRACE = '/home/hermes/.hermes/cache/scratch/exl3patch/artifacts/p8c_trace.json'
GB = 1e9
SSD = 6.5 * GB          # sustained cold read, measured on node1
NLAY = 20               # layers per rank under TP=2 pipeline sharding
EB = {'native': 7.219 * GB / 384, 'exl3': 5.113 * GB / 384}

d = json.load(open(TRACE))
per = defaultdict(list)
for layer, picks in d['records']:
    if layer >= 0:
        per[layer].append(picks)
body = sorted(per)
NPOS = len(per[body[0]])
assert all(len(per[l]) == NPOS for l in body), 'ragged trace'
NL = len(body)
print(f"trace: layers={body} positions={NPOS} picks/layer={sum(len(p) for p in per[body[0]])}")

class LRU:
    __slots__ = ('cap', 'd', 't')
    def __init__(self, cap): self.cap = cap; self.d = {}; self.t = 0
    def hit(self, x): return x in self.d
    def acc(self, x): self.t += 1; self.d[x] = self.t
    def trim(self):
        while len(self.d) > self.cap:
            k = min(self.d, key=self.d.get); del self.d[k]

def sim(cap, rows, adv, skip):
    caches = {l: LRU(cap) for l in body}
    i = 0; cold = 0; tok = 0
    while i + rows - 1 < NPOS:
        cc = 0
        for l in body:
            need = set()
            for j in range(i, i + rows):
                need.update(per[l][j])
            c = caches[l]
            cc += sum(1 for e in need if not c.hit(e))
            for e in need: c.acc(e)
            c.trim()
        if i >= skip:
            cold += cc; tok += min(adv, NPOS - i)
        i += adv
    return cold, tok

print(f"expert bytes: native {EB['native']/1e6:.2f} MB | EXL3 {EB['exl3']/1e6:.2f} MB | SSD {SSD/1e9:.1f} GB/s")

# calibration vs p8e (plain, cap 256, steady): expect ~8.7 cold experts/token, ~162.7 MB, ~40 tok/s
c0, t0 = sim(256, 1, 1, 85)
cf, tf = sim(256, 1, 1, 0)
print(f"\nCALIBRATION plain cap=256: steady {c0*(NLAY/NL)/t0:.2f} cold/tok ({c0*(NLAY/NL)/t0*EB['native']/1e6:.1f} MB/tok, {SSD/(c0*(NLAY/NL)/t0*EB['native']):.1f} tok/s) | from-empty {cf*(NLAY/NL)/tf:.2f} cold/tok")
print("p8e reference:              steady 8.70 cold/tok (162.7 MB/tok, 39.9 tok/s) | from-empty 11.70")

modes = [('plain', 1, 1), ('spec a=0%', 5, 1), ('spec a=25%', 5, 2), ('spec a=50%', 5, 3), ('spec a=100%', 5, 5)]
for cap in (64, 128, 192, 256, 320, 384):
    bud_n = cap * EB['native'] * NLAY / GB
    print(f"\n=== cap={cap} experts/layer  (rank budget: native {bud_n:.1f} GB / EXL3 {cap*EB['exl3']*NLAY/GB:.1f} GB) ===")
    print(f"  {'mode':<13}{'cold/tok':>9}{'nat MB/tok':>12}{'nat tok/s':>11}{'exl MB/tok':>12}{'exl tok/s':>11}")
    for name, rows, adv in modes:
        cold, tok = sim(cap, rows, adv, 85)
        if tok == 0: continue
        ct = cold * (NLAY / NL) / tok
        nm = ct * EB['native'] / 1e6; em = ct * EB['exl3'] / 1e6
        print(f"  {name:<13}{ct:>9.2f}{nm:>12.1f}{SSD/(ct*EB['native']):>11.1f}{em:>12.1f}{SSD/(ct*EB['exl3']):>11.1f}")
print("\nDONE")
