# Known-good prefill/decode baseline — 2026-08-21

Tag: `known-good-prefill-20260821-165048` (exo repo + mlx submodule, both
tagged at the exact commits this benchmark ran against — see "Exact
commits" below).

This is a durable reference point: real, live, end-to-end measured
throughput on the 2-node Mac Studio M4 Max cluster (TP=2, DeepSeek-V4-
Flash-671B, jaccl RDMA-over-Thunderbolt transport), taken immediately after
a full session of jaccl transport hardening (9 bug fixes) and a dual-cable
network topology split (see `docs/dual-cable-topology-and-qp-budget-2026-08-21.md`
for the full technical writeup of that work). Use this as the baseline to
beat for any future prefill/decode optimization work, and as the reference
state to diff against if something regresses later.

## Results

Measured via `bench/phase3_precheck_depth_throughput.py` (tokenizer-ground-
truth token counts, real live API calls against the running cluster — not
synthetic/estimated), with a needle-in-haystack correctness check on every
run so no throughput number is reported without confirming the output was
actually coherent.

| Context | Prefill throughput | Decode throughput | Needle check |
|---|---|---|---|
| 100,000 tokens | 366.6 tok/s | 17.48 tok/s | PASS |
| 300,000 tokens | 351.5 tok/s | 18.60 tok/s | PASS |
| 500,000 tokens | 331.6 tok/s | 17.26 tok/s | PASS |

Prefill throughput decreasing gently with context depth is expected (more
KV-cache pressure at longer context); the shape and magnitude here is
consistent across 3 independent measurement runs taken during this session
(333.9 tok/s, 334.0 tok/s, and 331.6 tok/s at 500K across three separate
runs — tight, repeatable, no variance spikes).

This is at parity with the pre-session baseline (339 tok/s @ 500K context,
from an earlier multi-week prefill-throughput campaign). **Tonight's
transport hardening did not regress prefill throughput.** No optimization
work was attempted this session — these numbers are a stability/regression
checkpoint, not the result of a speed-focused effort. Making prefill faster
than this is legitimate, separate, still-open future work.

## Verification performed alongside the throughput numbers

- Zero jaccl fault signatures (`all_gather STALLED`, `reliable_all_reduce_v2
  deadline`, segfault, `unordered_map::at`, silent-fallback warnings,
  link-local coordinator warnings) across the entire stress-test log,
  covering all three context depths run back-to-back with no reboot needed
  in between.
- Both runners `RunnerReady` before and after the full run.
- GPU power draw symmetric across both nodes during compute (~45-52W each,
  confirmed via `sudo powermetrics --samplers gpu_power`) — the earlier
  false-alarm asymmetric-power signature (this session, see section below)
  did not recur.
- Correct model output confirmed via the needle-in-haystack check on all
  three runs, not just throughput numbers in isolation.

## Known operational constraint (not a bug, a real hardware behavior)

The Apple Thunderbolt RDMA transport degrades under repeated rapid
teardown/reconnect cycles — observed this session after ~10 rapid restart-
and-test cycles during iterative debugging: prefill throughput crashed to
~130-165 tok/s with asymmetric GPU power (~7W vs ~20W across the two
nodes). **A full reboot of both Mac Studios cleared it completely** — this
is not something the code fixes or should try to work around; it's a
documented recurring pattern (see warm memory: "Thunderbolt RDMA link
wedges reliably on exo teardown... leaked RDMA QPs suspected... full
reboot required to recover").

**Practical implication for future benchmarking:** don't trust a throughput
number gathered after many rapid restart cycles without first checking GPU
power symmetry as a cheap canary, or just reboot before establishing a
fresh baseline. The numbers in this doc were taken on a freshly-rebooted
cluster specifically to avoid this contamination.

## Exact commits (what this baseline was measured against)

- `~/repos/exo` (main): `2eada90c3`
- `~/repos/exo/mlx` submodule (fork, `adurham/mlx` main): `1c591e105`
- `~/repos/exo/mlx-lm` submodule: unchanged this session, pinned at
  `5e88545a33ad94527de9861d75253dd6bcc1e7e5`

Both the exo repo and the mlx submodule are tagged
`known-good-prefill-20260821-165048` at these exact commits.

## Reproduction

```
cd ~/repos/exo
.venv/bin/python bench/phase3_precheck_depth_throughput.py \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --targets 100000,300000,500000 \
  --json-out /tmp/stress_test.json
```

Requires the cluster already `READY` (both runners) via `start_cluster.sh`.
If throughput looks anomalously low or GPU power is asymmetric between
nodes, reboot both Mac Studios first (see "Known operational constraint"
above) before trusting the numbers.
