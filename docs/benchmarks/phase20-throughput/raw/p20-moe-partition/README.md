# p20-moe-partition raw artifacts (see docs/dsv41-moe-partition-buildnote-2026-10-10.md)
- p20_moe_partition_bitexact_studio2.json / bitexact_studio2.log — AUTHORITATIVE bit-exact (153/153 0-diff)
- bitexact_studio2_attempt1_cbassert.log — aborted first launch (harness assumed cb=0; ckpt is MUL1)
- p20_moe_partition_bitexact_laptop.json / bitexact_laptop.log — laptop pre-check (75/75 0-diff, final harness: cb=2 on all 13 mixes + cb=0/cb=1 on the real trace)
- p20_moe_partition_bench.json / bench_studio2.log — studio2 timing v19d vs v20 (interleaved)
- guard_{pre,post}_{bitexact,bench}_{idle,canary}.json — phase20_guard before/after each GPU region
- node_md5_after.txt — node site-packages / src-tree / scratch gemv_metal.py md5 after the runs
- laptop_probe_and_ablation.txt — exploratory laptop numbers (v20b, v21 rejected; ablation)
- v21_256thread_NEGATIVE_source.py.txt — rejected v21 kernel source (not in the patch)
Scratch on studio2: /tmp/p20moe_partition_pkg (copied node mlx_lm + patched gemv_metal.py), /tmp/p20moe_partition.
