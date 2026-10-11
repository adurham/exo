# Task D — GPU clock / active-residency check (~5–10 min)

Passwordless `sudo -n /usr/bin/powermetrics` **confirmed** on both nodes.

Sustained load: `p20_pmp_taskD_load.py` (dense bf16 4096×8192×8192 matmul loop,
**1240 matmuls in 45.0 s → 15.14 TF sustained**).  Sampled with
`powermetrics -i 1000 --samplers gpu_power`.

## Results (min / max / mean)

| run | samples | GPU clock MHz (min/max/mean) | active residency % (min/max/mean) | GPU power mW (mean) |
|---|---|---|---|---|
| **studio2 idle** (baseline) | 20 | 766 / 771 / **768.7** | 3.86 / 4.46 / **3.96** | 15.1 |
| **studio2 LOAD** (~40 s) | 40 | 1578 / 1578 / **1578.0** | 100.0 / 100.0 / **100.0** | **54785.8** |
| studio1 idle (ref, in parallel) | 20 | 703 / 774 / 764.1 | 3.43 / 9.14 / 3.99 | 15.6 |
| studio1 during studio2 load (ref) | 45 | 764 / 771 / 768.2 | 3.39 / 3.91 / 3.51 | 12.2 |

## Verdict: **NO clock sag — confirmed.**

- Under sustained load studio2 pins the **maximum** GPU clock (**1578 MHz**,
  min==max==mean, zero variance across all 40 samples) at **100 % active
  residency** for the whole ~40 s window.
- Idle clock is ~769 MHz at ~4 % residency; load clock (1578 MHz) is *above* idle
  (768.7 MHz) — the GPU boosts and holds, it does not throttle/sag.
- studio1 remained fully idle throughout (residency ~3.5 %, clock ~768 MHz) — the
  bench load did not spill onto the peer node (TP2/RDMA not engaged by this offline
  single-node load), and confirms the studio1 idle reference.

Raw: `raw/D/pm_idle_studio2.txt`, `pm_load_studio2.txt`, `pm_idle_studio1.txt`,
`pm_load_studio1.txt`, `load.log`, `taskD_summary.json`.
