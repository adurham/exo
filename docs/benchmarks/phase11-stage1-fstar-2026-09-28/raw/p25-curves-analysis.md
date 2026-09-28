# p25 soak (take 3) -- full-run curve analysis
2026-09-28, gateway. Sources: node1|node2 `p25-iostat-*.txt` (per-second device truth),
`p25-soak-*.log`, `p25-env-*.log`. First/last idle samples dropped.

node1 (macstudio-m4-1), n=2327 busy seconds
  run median 6076 / p10 2768 / p90 6218 MB/s; last-300s mean 5769
  60s buckets: [6020, 6210, 6219, 6211, 6209, 6217, 6206, 6074, 6019, 6145, 6082, 5650, 6029, 5833, 5428, 5483, 5578, 5857, 5204, 5789, 5868, 5055, 4094, 4771, 5593, 3788, 5500, 5883, 5529, 5602, 5442, 5136, 6099, 5996, 6063, 5338, 5436, 6081]
  dip groups (60s-mean<5200 sustained >=150s): [(1258, 1548, 3512)]
  single-second low in t1260-1700 window: 2523 MB/s; KB/t plateau 413 vs dip 414
node2 (macstudio-m4-2), n=2329 busy seconds
  run median 6243 / p10 2975 / p90 6292 MB/s; last-300s mean 6269
  60s buckets: [5901, 6056, 6273, 6272, 6276, 6214, 6136, 6011, 6042, 6201, 5996, 6273, 6272, 6269, 6269, 6272, 6271, 6225, 5712, 6189, 6165, 4914, 4936, 4107, 4611, 3834, 4532, 4132, 5230, 5252, 5883, 5637, 6279, 6275, 6278, 6276, 6276, 6277]
  dip groups: [(1250, 1703, 3469)]
  single-second low in t1260-1700 window: 2765 MB/s; KB/t plateau 405 vs dip 404

Interpretation
  - The mid-run dip RECURS (both p24 - cut mid-dip - and p25, full 39 min) on BOTH
    nodes, and RECOVERS on both (n2 exact: 6275-6279 by t~1920; n1 near-full).
  - Node2 is NOT slow: run median n2 6243 >= n1 6076 MB/s.
    The p24 "node2 half speed" tail reading was a cut-mid-dip artifact. Retracted.
  - Request size ~constant through the dip (KB/t plateau vs dip above): the device
    serves fewer requests, not smaller ones.
  - Env: free pages pinned near the floor from early in both runs; swap flat
    (n1) / +26MB (n2); no thermal warnings. No environment correlate found.
    Candidates left: SSD-internal (thermal cycles / media housekeeping) or macOS
    memory-pressure stalls. Matters only to a streamed (Plan B) design.
