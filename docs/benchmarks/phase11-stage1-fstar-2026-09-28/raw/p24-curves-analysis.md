# p24 soak - full per-second iostat curve analysis
Parsed 2026-09-28 (gateway). Source: p24-logs/node1|node2/p24-iostat-*.txt (archived copies of node files).
Method: parse every sample row (KB/t, tps, MB/s); drop first 3; 50s buckets + rolling-60s means.

node1 (Adams-Mac-Studio-M4-1), n=1501 samples
  plateau t=3..1160s: mean 6175, median 6210 MB/s (min 1849, max 6345)
  buckets (50s, MB/s): 6227..6202 flat through ~t1150, then 6162 5794 5817 6110 5483 5125 4795 4232 4006
  min rolling-60s: 3512 @ t~1378s | last-300s mean: 4704 | terminal samples ~2690
  late mix: 62% >=5500, 33% <3000 | first sustained 60s-mean<5000: t~1231s = 00:02:36 local
node2 (Adams-Mac-Studio-M4-2), n=1517
  plateau t=3..1160s: mean 6253, median 6289 MB/s (min 2853, max 6354)
  buckets: 6294..6291 flat through ~t1150, then 6225 5516 6254 6252 4471 6180 4697 3662 2912 2923
  min rolling-60s: 2895 @ t~1456s | last-300s mean: 3991 | terminal samples ~2900
  late mix: 41% >=5500, 48% <3000 | first sustained sag (<0.7x median): t~1312s; 60s-mean<5000 from t~1189s = 00:01:58 local

Common
  onset ~00:02 local on both; run killed at t~1510s (2160s driver planned -> cut by operator to release the 40-layer trace gate)
  iostat = device ground truth; soak-internal rate counters are known-bad (ignore)
  env at 01:05: swap used node1 36494/36864 MB, node2 13438/14336 MB; free pages node1 ~26 GB, node2 ~0.3 GB
  no local event found at onset (exo log churn flat/low 23:55-00:04); sustained-load device behavior left standing
  follow-up: p25 (started 01:16:55 / 01:17:09 local, 2400s driver, 2340s soak) + per-minute swap/vm_stat/therm sampler
