#!/usr/bin/env python3
"""p23_ssd_soak2.py -- Plan B device soak, take 2.

Take 1 (p23) showed the process-level rates are CACHE-INFLATED (F_NOCACHE does
not truly bypass on APFS; the node1 native pattern re-read a 28.9 GB set from
the 128 GB unified cache at "26 GB/s").  Ground truth = iostat (device-level),
sampled by the driver concurrently.

Design: exl3 real per-expert ranges only (204 GB working set per pass -> cache
can hold only a fraction), depth 8 (= designed reader count), shuffle order,
1 Hz process series retained as an upper bound.

Usage: p23_ssd_soak2.py SECONDS [DEPTH]
"""
import fcntl
import json
import os
import random
import struct
import sys
import threading
import time

F_NOCACHE = 48
HOME = os.path.expanduser("~")
EXL3 = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
DT = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
      "I8": 1, "U8": 1, "U16": 2, "U32": 4, "U64": 8, "BOOL": 1,
      "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}
SUF = ["w1.trellis", "w1.suh", "w1.svh",
       "w3.trellis", "w3.suh", "w3.svh",
       "w2.trellis", "w2.suh", "w2.svh"]

def rd_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n

def build(model_dir):
    idx = json.loads(open(os.path.join(model_dir,
                       "model.safetensors.index.json")).read())["weight_map"]
    hdr = {}
    for sh in sorted(set(idx.values())):
        p = os.path.join(model_dir, sh)
        if os.path.exists(p):
            h, hlen = rd_header(p)
            hdr[sh] = (p, hlen, h)
    layers = sorted({int(k.split(".")[1]) for k in idx
                     if k.startswith("layers.") and ".ffn.experts." in k})
    out = []
    for L in layers:
        exps = sorted({int(k.split("experts.")[1].split(".")[0]) for k in idx
                       if k.startswith("layers.%d.ffn.experts." % L)})
        for e in exps:
            rng = []
            for suf in SUF:
                k = "layers.%d.ffn.experts.%d.%s" % (L, e, suf)
                sh = idx.get(k); ent = hdr.get(sh, (None, None, {}))[2].get(k) if sh in hdr else None
                if not ent:
                    rng = None; break
                p, hlen, _ = hdr[sh]
                o0, o1 = ent["data_offsets"]
                rng.append([p, hlen + o0, o1 - o0])
            if rng:
                out.append(rng)
    per = sum(r[2] for r in out[0])
    tot = sum(sum(r[2] for r in f) for f in out)
    print("[build] %d expert fetches, %.2f MB/fetch, working set %.1f GB"
          % (len(out), per / 1e6, tot / 1e9))
    return out

def open_fds(paths):
    fds = {}
    for p in sorted(paths):
        fd = os.open(p, os.O_RDONLY)
        try:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        except Exception:
            pass
        fds[p] = fd
    return fds

def worker(ff, counter, idx, stop, bufsize):
    buf = bytearray(bufsize); mv = memoryview(buf)
    rnd = random.Random(7000 + idx)
    n = len(ff); c = 0
    while not stop.is_set():
        f = ff[rnd.randrange(n)]
        for fd, off, ln in f:
            os.preadv(fd, [mv[:ln]], off)
            c += ln
        counter[idx] = c

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 8
print("=== p23b soak2 begin %s node=%s seconds=%.0f depth=%d"
      % (time.strftime("%F %T"), os.uname().nodename, SECONDS, DEPTH))
sys.stdout.flush()

fe = build(EXL3)
fds = open_fds({r[0] for f in fe for r in f})
print("[fds] %d" % len(fds)); sys.stdout.flush()
ff = [[(fds[p], o, l) for p, o, l in f] for f in fe]
bufsize = 1 + max(r[2] for f in ff for r in f)

STOP = threading.Event()
counter = [0] * DEPTH
ts = [threading.Thread(target=worker, args=(ff, counter, i, STOP, bufsize),
                       daemon=True) for i in range(DEPTH)]
t0 = time.perf_counter()
for t in ts:
    t.start()
rows = []
nxt = t0 + 1.0
prev = 0
print("[SOAKBEGIN %.3f]" % time.time()); sys.stdout.flush()
while True:
    time.sleep(0.05)
    now = time.perf_counter()
    if now >= nxt:
        tot = sum(counter)
        rows.append((now - t0, (tot - prev) / 1e6, tot / 1e6))
        prev = tot
        nxt += 1.0
        if len(rows) % 60 == 0:
            r = rows[-1]
            print("[soak] t=%5.0fs rate=%6.2f GB/s cum=%9.1f GB"
                  % (r[0], r[1] * 1000, r[2]))
            sys.stdout.flush()
    if now - t0 >= SECONDS:
        break
print("[SOAKEND %.3f]" % time.time()); sys.stdout.flush()
STOP.set()
for t in ts:
    t.join(timeout=10)
el = time.perf_counter() - t0
tot = sum(counter)
print("[total] %.1f s, %.1f GB, process-level %.2f GB/s (UPPER BOUND - "
      "iostat is truth)" % (el, tot / 1e9, tot / 1e9 / el))
rr = sorted(r[1] * 1000 for r in rows if r[0] > 30)
n = len(rr)
if n:
    print("[series] p10=%.2f p50=%.2f p90=%.2f GB/s (n=%d)"
          % (rr[int(n * 0.10)], rr[n // 2], rr[int(n * 0.90)], n))
print("=== p23b soak2 done %s ===" % time.strftime("%F %T"))
