# DEFINITIVE: 565-638K collapse is NOT fixed by the accounting fix (2026-09-23)

## Both verification runs, fix live

| | run 0 | run 1 | pre-fix reference |
|---|---|---|---|
| requested depth | 565K | 565K | 565K |
| actual depth | 619,462 | 638,449 | — |
| decode tps | **13.12** | **12.04** | 13.36 - 24.15 (stochastic) |
| page-ins | **66,003** | **39,930** | 6,599 |
| peak memory | **117.88 GB** | **123.80 GB** | 117.24 GB |
| before-prefill active | 107.32 GB | **114.67 GB** | — |
| wired limit | 115.4 GB | 115.4 GB | 115.4 GB |

**Verdict: the accounting fix is correct but NOT sufficient.** Both runs are in
the collapse band. Peak exceeds the wired limit by 2.5-8.4 GB in both.

## The three retention mechanisms and why none prevents this

1. **Session cap (4)** — count-based; with `reserve_slot` it trims to 3 leaves.
   Fires reliably.
2. **Byte cap (12 GiB)** — now correctly measured (the 89ebdbff0 fix), but it
   only fires if the **3 retained leaves** exceed 12 GiB, i.e. each >=~420K.
   Tonight's mix was ~370K leaves, so 3 x ~370K = 10.70 GiB -> correctly silent.
3. **Memory pressure (`0.85 x total`)** — ~116.8 GB of SYSTEM memory on a 128 GiB
   node. The post-prefill measurement sits right around that line, so it fires
   inconsistently.

Critically, `_evict_if_needed` runs **before** the new leaf is inserted, so any
cap always lags by one leaf: the 619K leaf that causes the overshoot is inserted
after the check that would have prevented it.

## Where the memory actually is (reconciled)

- **Fixed baseline: ~104 GB.** Weights ~77.5 GB/node (measured from safetensors
  headers, per the authoritative forensics reference) + MLX Metal runtime + MTP
  head + activation buffers. This is not negotiable without a smaller/quantized
  model.
- **Retained leaves: 16.63 GiB** at the moment of the 619K add.
- **Prefill transient: ~9 GB** (run 1: 114.67 GB before -> 123.80 GB peak).

So at 638K: 104 + 16.6 + ~9 = ~130 GB against a 115.4 GB limit. Even removing ALL
retained leaves (16.6 GB) would leave ~113 GB — barely under, with zero margin.

## The safe lever this identifies

The **prefill transient** (~9 GB) is the one term that is neither fixed nor
required. And there is existing, disabled machinery aimed exactly at it:

`mlx-lm/mlx_lm/generate.py` (~L450-465) implements context-adaptive chunk
sizing, off because the launcher ships the knobs empty:
```
: "${EXO_PREFILL_STEP_SIZE_HIGH_CTX:=}"
: "${EXO_PREFILL_STEP_SIZE_CROSSOVER:=}"
```
Its own comment: *"at HIGH context the indexer scores transient (B, H=64, L, P)
scales with BOTH chunk size L and pooled context P, so larger chunks hit memory
bandwidth pressure. Measured: 256-chunk is +39% at 100K but -30% at 380K vs
128-chunk. So start large and shrink past a crossover."* Defaults named:
HIGH_CTX=128, CROSSOVER=200000.

We are currently running **2048-token chunks at 638K** — 8x larger than the 256
the comment already measures as -30% in this regime. The transient scales with L.

This is a safe, reversible test (two env vars, no wedge risk) and it targets the
transient directly. It is the next experiment.

## What is NOT being tested, and why

Raising `iogpu.wired_limit_mb` 115000 -> 124000 would give ~9 GB of slack and
would directly address the overshoot. **Not tested unilaterally.** Its own
root-cause commit (`0a94e5443`) states the wedge is *"NOT recoverable by relaunch
(wired memory pinned; only a full reboot clears it)"*. A failed experiment
therefore leaves the cluster wedged until the user reboots — which their standing
constraint forbids me from doing unasked. This one needs an explicit decision.
