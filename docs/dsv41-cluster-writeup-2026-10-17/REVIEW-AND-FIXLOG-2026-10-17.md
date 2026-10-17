# FIXES-APPLIED.md — WRITEUP.md consolidated fix application

Applied 2026-10-17 to `the writeup working copy` (pre-edit 1,247 lines -> 1,263; content-only growth from added provenance/hedge clauses). One entry per item ID. Line numbers below are pre-edit.

---

**A1 — 160K row decode basis (§0.1 L42, §9.2 L830).** VERDICT: FIXED.
- §0.1 row basis note: old "client decode window, spec-ON" -> new "client decode window, spec-ON; the 160K point is server-side only (that run used `stream:false` -- client decode not separable)".
- §9.2 row: old client decode cells "6.54 / (server only) 30.4" -> new "n/a / n/a", server cell "30.4 (server decode window 6.54 s, server-side only)". Kept in row, 6.54 s and 30.4 tok/s preserved in server context.
- Source: evidence-live.md:118-120 ("client-side decode t/s: NOT SEPARABLE on this attempt"), :133 (server decode window 6.54 s = 30.4 tok/s, "marked server-side only").

**A2 — "72 GB model resident" (§4.1 L434).** VERDICT: FIXED. Old "resumes serving with the 72 GB model **resident**" -> "resumes serving with the 108.76 GB model **resident**". Source: evidence.md D4.1:553 (per-rank weights 108.76 GB measured); D4.3:624-625 shows 72 GB is the older exl3-plan era pair.

**A3 — +36.71% statistic (§0 L30, §0.1 L45, §6.1 L583, §9.5 L902).** VERDICT: FIXED.
- Read evidence.md:410-430: the packs pair 36.63/27.15 (fixed-window **medians**) with "+36.71%" and never name the statistic behind that percentage (evidence.md:48, :242, :420; the §9-§10 notes :419 only say the level 36.63 is a fixed-window median).
- Read the cited doc itself: `docs/dspark-mtp-production-baseline-2026-08-27.md:16-21` — its own table row literally reads "median fixed-window tok/s @100K 36.63 / 27.15" with delta "**+36.71%**". The doc is the origin of the pairing; it names no statistic for the %.
- Fix applied in all four locations: the +36.71% is now attributed as "the baseline doc's own stated delta" with the added clause "(their direct ratio is +34.9%)". No stat for the % is asserted beyond what the doc itself states. NOTE: the same claim also appears in §9.5 (not in the parent's list); the identical verified fix was applied there too for consistency — logged here, flagged as an adjacent instance.
- Arithmetic: 36.63/27.15 = +34.9% (computed).

**A4 — component-sum vs du (~0.2%) (§1.3 L109-112, §5.4 L551).** VERDICT: FIXED.
- §1.3: appended "(The component sum -- 2x each width-sharded part plus the replicated 6.50 -- is 211.0 GB, ~0.2% apart from the 210.68 GB on disk; du vs component-sum.)"
- §5.4: appended "(components sum to 211.0 GB per pair vs 210.68 GB on disk, ~0.2% apart -- du vs component-sum)".
- Arithmetic verified: 2x(98.02+3.84+0.40)+6.50 = 211.02; 0.34/210.68 = 0.16% (~0.2%). Components per evidence.md D4.1:557-559; 210.68 GB per D4.1:551.

**A5 — round-cost table implied tok/s (§9.2 L847-853).** VERDICT: FIXED (recomputed).
- The source round-cost table (evidence-live.md:203-209) carries the exact same inconsistent derived column as the writeup, so the fix recomputes implied tok/s = tokens/round ÷ gap: 1K 33.6 -> **34.1**; 10K 34.1 -> **33.7**; 50K 31.9 -> **31.5**; 100K 32.6 and 300K 6.0 consistent, unchanged (3.0/0.092 = 32.6; 2.9/0.486 = 5.97 ≈ 6.0). No gap values changed.

**A6 — 34x span-vs-transport gap (§10.4 L958).** VERDICT: FIXED (number kept, denominator corrected).
- The cited doc `docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md:18-51` contains the 4094 µs / 36.1 µs / 66.3 µs pairs but contains **no 34x**. 34x originates at `docs/PERFORMANCE_HISTORY.md:502`: "~120µs (isolated microbenchmark floor), vs in-model sync-span average of ~4094µs — a **34x gap**" (also :3015-3017, the exact line the writeup cites).
- New text: "**Which sub-cause dominates the 34x all_sum span-vs-transport gap** (34x = the ~4094 us sync-span average vs the ~120 us isolated-microbenchmark floor; eval-fence vs dispatch vs Python scheduling)". 4094/120 = 34.1 ✓; vs medians/means the ratios would be 113x/62x, now unambiguous.

**A7 — "the old 32168 default" (§12.3 L1181).** VERDICT: NO-CHANGE-NEEDED (refuted by audit, logged only). The writeup's text already reflects the code comment (`EXO_DSV41_DEFAULT_MAX_OUTPUT_TOKENS` 8192; 0 = old 32168 default; ceiling 32768) — matches PERFORMANCE_HISTORY.md:12182 verbatim.

**A8 — kill cadence (§12.1 L1144-1146).** VERDICT: FIXED (measured average stated).
- Source: PERFORMANCE_HISTORY.md:12178 states "~every 7.5 min — 22 kills" but its own window 04:41-07:54 = 193 min; 193/22 = 8.8 min. The source's cadence conflicts with its own window, so per instruction the measured average is stated.
- Old: "roughly every 7.5 minutes -- 22 kills, 12 on rank0 and 10 on rank1"; New: "roughly every 8.8 minutes -- 22 kills over that 193-minute window, 12 on rank0 and 10 on rank1". "~25 kill/re-place cycles" kept.

**A9 — killer-case rate (§12.4 L1196-1197).** VERDICT: FIXED (corrected to delta/window).
- Source PERFORMANCE_HISTORY.md:12184 states "190+ s ... committed advanced 2,263 -> 5,863 (~28 tok/s)". ~28 tok/s is not presented as the runner's logged rate and is not the delta over the stated window: (5,863-2,263)/190 = 18.9 tok/s. Treated as delta/window -> corrected to ~19 tok/s, window kept explicit: "a 190+ s window with zero client output in which committed advanced 2,263 -> 5,863 (~19 tok/s)".

**A10 — 44 vs 43 layers (§4.2 L452).** VERDICT: FIXED (defined; residual AMBIGUOUS).
- Source check: `docs/dsv4-memory-leaks-2026-06-27.md:41-43,58` — "DSv4 uses `RotatingKVCache` for all 44 layers ... all 44 caches". The 43 elsewhere (§3.1 L270, §6.2 L626) traces to `docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md:21-23` — "43 layers ... exactly 43 `moe.all_sum` calls per token, one per layer". The two docs assert different layer counts for (apparently) the same DSv4-era model; they cannot be reconciled from available sources without inventing.
- Fix: first row cell now reads "all 44 non-sliceable layer caches (44 is the leak doc's layer-cache count; the 43 used in §3.1 and §6.2 is the `moe.all_sum` calls-per-token count, sourced separately)". AMBIGUOUS residue: the underlying 44-vs-43 layer-count discrepancy across corpus docs is unresolved by the packs.

**A11 — capture corpus clause (§11.5 L1102).** VERDICT: FIXED. Old "the corpus is the repo's own `docs/**.md` concatenated in sorted order" -> "the corpus is the repo's own `docs/**.md` plus the repo's top-level handoff docs, concatenated in sorted order". Source: evidence-live.md:86.

**A12 — §9.1 needle verbatim locations (§9.1 L818-819).** VERDICT: FIXED. Old "Several are reproduced verbatim in §5.4, §6.1 and §7" -> "Several are reproduced verbatim in §5.4 and §6.1". (§7 reproduces deltas/error lines and limiter reasoning, not a needle answer.)

**A13 — "bootstrap 95% CI" (§9.5 L902).** VERDICT: NO-CHANGE-NEEDED. The pack row (evidence.md:242) abbreviates, but the cited doc states it explicitly: `docs/dspark-mtp-production-baseline-2026-08-27.md:19` — "95% bootstrap CI (10K resamples) [+28.26%, +51.02%]". Writeup's "bootstrap 95% CI [+28.26%, +51.02]%" matches. No change.

**A14 — §10.1 "proximate fix is in" (L925-926).** VERDICT: FIXED (softened to pack status).
- evidence.md:851 area (List B, B1): the pack's status wording is "gc_limit-threshold model is best standing explanation but NOT live-confirmed"; no shipped/in-place cliff fix is recorded in the packs themselves (the fix lives only in the writeup's §5.3 citation chain, e.g. `docs/b2-quality-handoff-2026-06-24.md:27`, exo `463ac5d`).
- Old: "The proximate fix is in; the mechanism underneath is not." -> New: "The mitigation is in and measured (§5.3); the mechanism underneath is not live-confirmed."

**A15 — max_pd (§2.2 L179-180, §11.1 L1029).** VERDICT: NO-CHANGE-NEEDED. The pack row (evidence.md:694) truncates with "..." but the cited doc gives the full probe: `docs/dual-cable-topology-and-qp-budget-2026-08-21.md:61-63` — "`max_qp:3, max_cq:6, max_pd:6`". max_pd:6 is supported. Kept.

**A16 — prefill-cliff hedge (§5.3 L534-537).** VERDICT: FIXED. Appended after the mechanism sentence: "This is the best standing explanation, not live-confirmed (`PERFORMANCE_HISTORY.md:2943-2953`; §10.1)." Matches pack hedge evidence.md:851 (B1) / :18.

**A17 — §8 provenance sentence (§8 opening, L740).** VERDICT: FIXED. Added: "This section draws on live reads of the nodes' config and scripts (cited inline below: `reboot-node.sh`, the live `smctl` config, `docs/xcode-removal-launcher-clt-fallback-2026-08-25.md`, commit `70e0423`), not on the two evidence packs."
- Existence verified: `reboot-node.sh` at exo repo root (`git ls-files`); `docs/xcode-removal-launcher-clt-fallback-2026-08-25.md` present in docs/; smctl config is live-read per the prose itself (§8.1 already says "read live from `/etc/smctl/config.toml` ... not from the evidence pack"); commit 70e0423 previously verified in the citation audit.

**A18 — §12 provenance sentence (§12 opening, L1139-1140).** VERDICT: FIXED. Added to the opening paragraph: "...and it draws on the incident captures behind the 2026-10-17 PERFORMANCE_HISTORY entry -- the supervisor log and the detector report -- plus the deploy's own validation logs, with commits cited inline."

**A19 — header provenance claim (L3-5).** VERDICT: FIXED. Old "Every measured number traces to `evidence.md` or `evidence-live.md`" -> "Measured numbers trace to the two evidence packs (`evidence.md`, `evidence-live.md`); §3, §4, §5.1, §5.3 and §6.2 additionally cite the fork's own dated docs and commits (named inline); §8 and §12 draw on the live captures they name. Private identifiers are scrubbed to neutral placeholders."

**A20 — session ID scrub (§12.1 L1144; §12.4 L1201).** VERDICT: FIXED.
- §12.1: old "a session on the cluster model (`20261016_225559_dc7b77`)" -> "a session on the cluster model" (the "overnight" qualifier was already carried by the sentence's opening; no count removed — the 22-kill counts untouched at :1146-1148).
- §12.4: old "a cluster-hosted Hermes session (`20261017_101155_863649`, provider exo)" -> "a cluster-hosted Hermes session (provider exo)". Verified: neither ID string remains anywhere in the file.

**A21 — title em-dash (L1).** VERDICT: FIXED. "M4 Max — how" -> "M4 Max -- how". Verified the em-dash was the only non-ASCII besides § (0xa7), and the post-edit file still contains exactly one non-ASCII codepoint class: §. All § symbols kept.

**A22 — "~7 GB of headroom" (§12.5 L1230).** VERDICT: AMBIGUOUS (resolved via explicit derivation).
- The 2026-10-17 PERFORMANCE_HISTORY entry (docs/PERFORMANCE_HISTORY.md:12176-12187) does not state the ~7 GB figure or any KV-headroom context, so the number cannot be quoted as sourced. It IS derivable from pack figures: 120 GB (wired limit, evidence.md D4.4:634, :645) - 108.76 GB (weights, D4.1:553) - 4.13 GiB = 4.43 GB (1M KV, D4.2:594) = 6.81 GB ≈ ~7 GB.
- Fix: "(~7 GB ... (120 GB wired limit - 108.76 GB weights - 4.13 GiB 1M KV = ~6.8 GB)" — context depth made explicit; the arithmetic is stated so the reader sees the derivation.

**A23 — "~4.4 KB per generated token" (§12.5 L1229).** VERDICT: FIXED. Old "=> ~4.4 KB per generated token" -> "=> ~4.2 KiB per generated token". Arithmetic: 4.13 GiB / 1,048,576 tokens = 4,229.6 B ≈ 4.2 KiB (also exact: 4.13 GiB/2^20 = 4.13 KiB/token ≈ 4.2). The old figure double-counted a KB/GiB conversion.

**A24 — copy-edit nits.** VERDICT: FIXED (all sub-items).
- L59: "Two identical Apple Mac Studio," -> "Two identical Apple Mac Studios,".
- L221: "Two Mac Studio M4 Max," -> "Two Mac Studios M4 Max," (per the instruction's "Mac Studios").
- L502/L505-506/L511: SHAs backticked — `9ea86f9`, `6391efc`, `bd1bfd1`. Also the same sentence's trailing bare SHA on L512 ("We pin mlx-lm at 676f803.") was backticked -> `676f803` (adjacent instance of the same item, logged).
- L564: "wall 6760.5 s" -> "wall 6,760.5 s".
- §6.1 verbatim quote (L597): appended " [...]" after "corrected those percentages downward to a real ~2.9-" (the quote stops mid-sentence; per evidence-live.md:256 the source text continues).
- L717 vs L732: L732 now reads "96 tokens of reasoning (the 322 characters quoted above) instead of the default run's 1,996 characters, a clean `stop`" — clarifies 1,996 chars = the default (600-token) run's reasoning length, 322 chars = the limited run quoted above (evidence-live.md:349-352).
- US spelling: busy-but-serialised -> serialized (L815); parallelised -> parallelized (L880); behaviour -> behavior (L1130). One additional instance of the same class found and fixed: "the limiter honours" -> "honors" (§11.7). No other British forms found (grep for honou/recognis/analys/optimis/colour/defence/serialis/parallelis/behaviour etc.).

**A25 — "-- -8.2 GB" sign collision (§0 L27).** VERDICT: FIXED. Old "does not close -- -8.2 GB best case" -> "does not close -- from -8.2 GB best case". (§5.4/L560's "-8.2 GB (best) to -15.3 GB (worst)" needed no change.)

**A26 — EXL3 build numbers (§5.1 L500-512, §2.4 L215).** VERDICT: NO-CHANGE-NEEDED (all numbers verified against cited repo sources; logged as checked).
- 1.909x / 1.979x: `docs/benchmarks/phase9-exl3-spike-2026-09-28/README.md:25-26` ("decode R=1 ... 0.733 ms (1.909x)", "verify R=4 ... 2.086 ms (1.979x)"; column header "ratio vs prod mxfp4" supports "vs production MXFP4"). Also repeated in phase11 README:39.
- RMS 150.5 vs 34.6: `docs/benchmarks/phase10-planb-gate-2026-09-28/README.md:111-115` — "the layer rms jumps 4.21 -> **150.45** (absmax 20401)" and "Post-fix L39 rms **34.58**" — the writeup's 150.5/34.6 are direct rounds of these.
- cos 0.9999999: `docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md:75,93` (rank-slice partials sum to full module, cos 0.9999999 both) + mlx-lm repo's `exl3/README.md:71`.
- per-layer cos >= 0.99983 with `bd1bfd1`: `docs/benchmarks/phase13-dsv41-model-file-2026-09-28/README.md:17,44` and `docs/deepseek-v41-exl3-plan.md:27`.
- All cited docs exist; no cite name or number required fixing.

**A27 — §6.2 jaccl-timing / arithmetic-check numbers.** VERDICT: NO-CHANGE-NEEDED (spot-checked against cited docs; all match; logged).
- median 36.1 µs / mean 66.3 µs / 8192-byte payload / 45,666 calls per rank: `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md` (n row "45,666" at both ranks; mean 66.3 / median 36.1 are the **rank0** column values — rank1 is 58.9/36.0. Writeup quotes the rank0 pair; noted, no change).
- "Only 0.04% of calls exceeded the earlier sync-span figure" ✓ (18/45,666 = 0.04%, doc ~:114).
- ~4094 µs/call -> 43 calls = 176.0 ms/token vs 53.48 ms/token -> 3.29x; 43 x 36.1 µs = 1.55 ms/token = 2.9-5.3% of wall: `docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md:18-51` ✓ all values verbatim.
- Note: the parent's A6 referenced "docs/allsum-sync-span-artifact-2026-08-22.md (lines 18-51)" — the file's actual name is `allsum-sync-span-artifact-arithmetic-check-2026-08-22.md`, which is what the writeup already cites; no cite fix needed. 34x does not appear in that doc (see A6).

**A28 — spot-checks (read the cited ranges).** Four RESOLVES, one range fixed:
- `docs/incidents/pkill-self-match-forces-sigkill-2026-09-10.md:105-107` — RESOLVES: "CLEAN_EXIT after 1s (NO SIGKILL USED)", ports `PORT_ACTIVE`, 0% loss, "wired+compressor drained to 3.6 / 3.8 GB" — matches §3.3's cite content.
- `docs/PERFORMANCE_HISTORY.md:4110-4112` — RESOLVES: AppleThunderboltRDMA teardown 20:23:22 after a runner SIGKILLed, TB link dropped, all ports "No device connected", fixed by rebooting both Studios — matches §3.2.
- `docs/dsv41-next27-ringfix-construct-2026-10-14.md:227-230` — RESOLVES: 2536 rejected speculative rounds rank-0 vs rank-1, 0 cross-rank mismatches, VERDICT PASS — matches §4.3.
- `PREFILL_CLIFF_HANDOFF.md:48-51,78-82` — RESOLVES: ~270 t/s @100K declining to ~168 @340K, sharp cliff to ~40-48 t/s @340-380K, periodic 8-32 s stalls on 128-token chunks; attn.indexer 4532->5362 µs (+18%) with max ~22 ms — matches §5.3. (File lives at the exo repo ROOT, not docs/; the writeup cites it without the docs/ prefix, which is accurate — no change.)
- `docs/deep-context-refault-mechanism-confirmed-2026-09-22.md` — file has 98 lines; §10.1's ":95-99" fixed to ":95-98". Lines 95-98 content RESOLVES: "Mechanism: CONFIRMED (page-ins measured, swap/compressor ruled out). Fix: NOT yet attempted."

**B1 — 32168 default.** VERDICT: REFUTED (no change; logged). The writeup's §12.3 matches PERFORMANCE_HISTORY.md:12182's code-comment description exactly.

**B2 — `mlx/jaccl/mesh.cpp` mention (~L159).** VERDICT: REFUTED (no change; logged). §2.1's parenthetical explicitly notes the top-level-path lookup returns 0 commits — the low-signal mention is deliberate, and evidence.md:494-504 confirms the real path is `mlx/distributed/jaccl/lib/jaccl/mesh.cpp`.

**B3 — § symbols.** VERDICT: REFUTED (no change; logged). § typography kept by decision; post-edit file non-ASCII census is exactly {§}.

---

## Verdict counts

- FIXED: 22 (A1-A6, A8-A12, A14, A16-A21, A23-A25)
- NO-CHANGE-NEEDED: 6 (A7, A13, A15, A26, A27, plus A28's four RESOLVES)
- REFUTED / logged-only: 4 (A7 per audit + B1, B2, B3)
- AMBIGUOUS: 2 (A10 — 44-vs-43 layer-count discrepancy across corpus docs, resolved by defining the two counts; A22 — "~7 GB" not stated in source, resolved by making the derivation explicit from pack-sourced figures)

## Notable flags for the parent

1. A3: the % statistic is unresolvable in the sources — the baseline doc itself pairs medians 36.63/27.15 with "+36.71%" and names no statistic; the fix attributes the number to the doc and states the medians' direct ratio (+34.9%) alongside. §9.5 (not in the parent's list) received the identical fix.
2. A5: the source round-cost table in evidence-live.md carries the same internally inconsistent "implied tok/s" column; fixed by recomputation, gaps unmodified.
3. A8: the source's own "~every 7.5 min" contradicts its own 04:41-07:54 window (193/22 = 8.8 min); the measured average now stands in the text.
4. A9: source's "(~28 tok/s)" is neither labeled as a logged rate nor the delta/window; corrected to ~19 tok/s with the window explicit.
5. The pack round-cost table and PERFORMANCE_HISTORY 2026-10-17 entry are untouched (repo/pack files read-only, as required).