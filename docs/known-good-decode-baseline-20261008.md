# Known-good decode baseline — next17-levers (2026-10-08)

State: **SHIPPED** 2026-10-08 16:07 CDT. `deploy/next13 @ 576e9d279` (exo) + mlx-lm `3bf8316`;
both nodes serving, gates unset (defaults), canary healthy 14.85/14.86, READY (2/2).

Numbers (same-session symmetric arms; benign = 20K ctx, g3, 800 tok; agentic = 91K real-session replay, g3,
800 tok; median ms/round):

| arm | benign | agentic |
|---|---|---|
| production f4bb14746 (before) | 145.75 ms / 25.5 t/s | 157.10 ms / 19.5 t/s |
| next17 defaults (lever-1 code) | **118.56 ms / 31.5 t/s** | **130.07 ms / 23.7 t/s** |
| next17 + `DSV41_INDEXER_HIER=0` | — | **101.06 ms / 30.2 t/s** |

Split (same-build, additive): lever-1 code = −27.03 ms/round; lever-2 (indexer HIER) = −29.01 ms/round;
total −56.04 ms/round. mean_accepted unchanged across arms.

Verification: R8a battery CLEAN (needles 6/6, tools 10/10, prose 0 dirty, park PASS); guard present in the
installed venv module on both nodes; post-boot canary healthy; launcher "Nodes synchronized on commit
576e9d279" + READY 2/2; post-ship parity smoke (benign 3×20K) — log `/tmp/p3b/ship_smoke.log`.

Reproduction: driver + scripts `~/.hermes/cache/scratch/p3b/{p3b_driver.py, ship_next17.sh}`; harness and
guard `/private/tmp/phase20-campaign/bench/`; artifacts `docs/benchmarks/phase20-throughput/raw/p3b/`
(campaign branch `deploy/phase20-campaign @ 0b10977ab`); protocol `PHASE3B-SHIP-VALIDATION.md` there.

Tags: `known-good-decode-next17-20261008-160746` on exo `576e9d279` and mlx-lm `3bf8316` (both adurham forks).

Next: lever-2 code guard (spec §7 of PHASE3B-SHIP-VALIDATION.md) with index-identity proof → target ≈101 ms
agentic ≈ 30 t/s.
