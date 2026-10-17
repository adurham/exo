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

---

# Round 2 — post-Fable review (2026-10-17, same day)

Fable (claude-fable-5) reviewed the document; its back-half context was truncated by the consult
backend at ~§5.4, so this round resolves its front-half findings and a second pass covers §5.4–§12
separately. Items below were each verified against the repo/config/live state before fixing.

**R2-1 — wired-limit units; headroom recomputed (§0, §0.1, §1.1, §5.4, §12.5).** The evidence pack's
D4.4 formula reads `iogpu.wired_limit_mb=120000` as 120.0 GB; the corpus's own convention is
120000 MiB = 125.83 GB = 117.19 GiB (`docs/PERFORMANCE_HISTORY.md:11924,11979` — "W = 125.829 GB
(120000 MiB)"). FIXED: §1.1 now states the unit; headroom recomputed 125.83 − 108.76 − [3.4..4.5] −
[16..22] = **-2.3 to -9.4 GB** (was -8.2/-15.3); §12.5's cap-headroom ~6.8 → ~12.6 GB; §5.4's
"reproduces the gap column" reworded to the correction doc's actual 8.8-14.8 GB gap column
(`docs/deepseek-v41-memory-budget-correction-2026-09-27.md:88-95`).

**R2-2 — `maxPrefixSessions` vs the 4→1 leak fix (§4.2).** Verified live: /state reports
`maxPrefixSessions=4`; git history: `9182190` set 4→1, then `4d09fcd` restored 1→4 ("live-session
eviction"), current launcher default 4 with the ~12 GB 100K worst case re-documented. FIXED: the
row now carries the restore.

**R2-3 — 44 vs 43 layer counts (§4.2 cell, §3.1, §6.2).** Resolved by era, from the live configs:
June leak doc = V4-Flash era (43 hidden layers; 44 cache entries); the Aug arithmetic-check's 43
`moe.all_sum` calls/token is that same engine; this writeup's V4.1 checkpoint = 40 hidden layers
(`num_hidden_layers = 40`, read from the node's config; /state `nLayers: 40`). The in-table
editorial parenthetical was removed; one explanatory note now sits after the §4.2 table and short
tags at §3.1/§6.2.

**R2-4 — watchdog timing (§3.4).** Replaced "(45 s probe + 20 s grace)"/"45-180 s" with the code
truth: `HANG_TIMEOUT_SECONDS=45`, then `HANG_PROBE_INTERVAL_SECONDS=20` growth checks — a genuinely
wedged round dies at ~65 s (`src/exo/worker/runner/supervisor.py`).

**R2-5 — §5.3 corrections.** `device.cpp` is mlx — the citation now reads "read in mlx
`device.cpp:768-773`; landed by exo `463ac5d`" (exo commit touches only start_cluster.sh, verified);
the June fix vs the August mechanism-explanation timeline is now stated; "already gone" is scoped
to the **340K** cliff (1M deep-context degradation is a separate regime, §9.3); the 340K overlap in
the ledge sentence fixed; §5.3's cross-ref trimmed to §5.4/§6.1.

**R2-6 — §5.1 wording.** four pieces (kernels, loader, checkpoint reader, layer builder); "nothing
**upstream** understood the format"; "agreeing to cos 0.9999999"; the 1.909x/1.979x baseline stated
(EXL3 vs production MXFP4 kernels).

**R2-7 — §1.3/§5.2.** "no `kv_lora_rank` key in the config" (matches the pack's own wording); bf16
rationale corrected from "products" to "values" (the code comment's own claim, cache.py:26-29); the
comment's ~3.2 GiB noted as a rounding of the 3.31 GiB component sum.

**R2-8 — copy items.** §4 intro "two"→"three" failure modes; §4.3 retitled to the spec-rollback fix;
§3.2's TODO marker converted to prose; "(3.3, 3.4)"/"Section 8"/"Section 4" normalized to §-style;
port 52416 glossed (the file server); §0 leads with the medians' +34.9% (baseline doc's +36.71%
kept alongside); "127 GB of 128 GiB" → "~127 GB of the 137 GB (128 GiB)".

**Refuted/kept, with sources (no change):** `maxPrefixSessions`/`maxPrefixBytes` ARE in /state
(a first regex missed them); the `cache.py:29` comment exists and does say ~3.2 GiB; `32168` stays.


---

# Round 3 — Fable tail review (2026-10-17, same day; sections 5.4–12)

Fable's back-half review arrived as six complete-coverage chunks (the consult backend truncates long
contexts, so the document was delivered piecewise). Its verdict: no sign-off, with a specific list.
Every item was verified against the repo/config/live state, then fixed or resolved. ~45 edits.

**Corrections with sources.** §6.2: regime confusion fixed (62% is a *prefill* share at 12K/38K; the
2.9-5.3% correction is *decode*-side; the 265-vs-178 ms/call figure is `docs/moe-allsum-sharedscale-
CORRECTED-final-2026-08-19.md`'s real-prefill measurement, a different regime from the 36 us decode
calls; the quant lever is closed as an engineering dead end, not a math impossibility). §6.1: the
10.13 GB DSpark-head / 99.4 GB residency figures labeled as the 2026-08-27 V4-Flash+DSpark build
(era mismatch vs §5.4 resolved by labeling, not by mixing). §9.2: the 1K row and its client/server
divergence footnoted; "doubling-and-a-half" corrected (the falloff is from 160K to 300K); the
"not gradually" claim narrowed to what five ~110-119K points can support; the 300K row's ~33.5 s
client wall vs server elapsed (1600.8 vs 1567.3 s) explained; the per-round mechanism restated as
origin-unidentified with the non-linearity argument (5.3x cost for 1.9x context; 1M decode only 2x
slower still). §9.3: attempt-1 band rate labeled a progress-position figure with its own ~98K
partial cache; cross-attempt comparison caveated; "irreducibly" scoped to this implementation; band
table's 299,008/303,104 offset explained (one 4,096-row chunk). §9.4: 111.8 tok/s labeled progress-
based vs attempt 2's uncached-based 104.2; the 299,008-of-802,816 reuse bound marked not
established; 2.9 h -> 2.8 h (recomputed). §9.5: the +36.71% statistic named as unnameable from the
sources (paired; the medians give +34.9%); CI brackets fixed; the ~16% gap vs the §9.2 100K row
explained (different instrument, different boot; the repo's own cross-boot swings are 6+ tok/s);
"gamma=3 signature" softened to consistent-with (~2/3 acceptance). §10.x: "reproducible" -> two
context sizes; refault-confirmed vs refault-as-cause separated; k_eff glossed as bounded-2-5 in the
corpus's note; the 8-32 s stall labeled the June V4-Flash-era observation + absence-under-
mitigation; the §10.3 probe-vs-36.1-us instrument distinction; the 500K 81% figure named; the 9.5%
slice labeled prefill-side; the 34x bullet reconciled with §6.2's forced-sync artifact and the
36.1-vs-120 us harness difference; the 1.7% straggler bound marked pre-correction instrument; the
43/token count glossed; c=2 notation unified; the DSpark frame picked (not reproduced per the
source correction; the p4-scoping entry stale); hc_expand conflict dissolved to the real open item;
"cached rows" -> tokens.

**§11 (repro section).** Attempt-1 2 h cap vs attempt 2's 6,760.5 s reconciled; the `-w CURLMETA`
line now explicitly *not* the timing harness (the epoch stamps come from a line-stamping filter);
"or later" dropped from the tested macOS build; checkpoint size convention stated (du-allocated
210.68 GB vs `storageSize=210713013432` B ~= 210.71 GB SI); wired-limit reapply noted; Tailscale/
MagicDNS + node discovery added to requirements; exo-row SHA clarified as *post*-fold (contains
the §12 fix); `EXO_TARGET_BRANCH` explained and shown inline; mlx branch-column caveat; wheels
wording fixed; "all changes go through git" corrected to the rsync-of-working-tree truth (commit
and push first -- the push-check enforces it); placement step added to §11.4 (automatic at launch,
~2-3 min/node); storageSize separators dropped; prompt generator/scratch provenance stated;
~600 KB body-size note; which-node log note; API-on-every-node note; cable count clarified; banana
repro's expected outcome + designed-false-positive shape stated; limiter needs enable_thinking.

**§12.** Intro re-garbled sentence fixed (the *fix* is why the on-cluster session could write the
lead sections); 22-vs-25 count reconciled (extra cycles dropped); 210 GB reload corrected to
~109 GB/rank; the kill-cluster arithmetic now closes (45-67 s ~= 1,500-2,000 tokens at ~30 tok/s);
guard-silence reconciled with the preamble loop (line-level near-repeat, below the exact-cycle
trigger); 2,048-id suffix glossed; DECODE claim narrowed to *independent of `step()` returning*;
Fix-B/10-05 references dated; the decode beat's relation to the hang probe stated; cap resolver
precedence stated (client > default > clamped by ceiling) and the caps labeled a blast-radius
bound, not the root-cause fix; 11,608-token validation reconciled with the 8192 default (harness
sent its own max_tokens); step-2's ~19 tok/s labeled a server-side window; step-3 window cross-ref;
the detector sweep's W/M/J defined and the negatives' tripping stated; 4.2 KiB -> ~4.1 KiB
(4,229 B; the earlier figure had used the 1,000,003-token denominator); breaker counts stated;
"damage" sentence re-scoped; extra_body audit moved to §12.6; escape-ladder n defined and its
scope distinguished from the declined detector; "when a CONFIRMED loop does fire" reworded.

**Verified no-change (logged).** 32168 is correct (engine MAX_TOKENS constant, per the code
comment -- the 32768 two words later is the ceiling); `32168` vs `32768` confusion is Fable's,
not the document's.


---

# Round 4 — final pre-publication verification (Fable, chunked re-read of the finished text)

An 11-chunk verification pass over the final document (chunks of ~9 KB; the backend truncates
longer contexts). Verdict per chunk: mostly clean with a final set of precision items, all
applied. ~30 edits.

**Applied:** "108.76 GB measured" -> "computed from shard headers" (the basis was always stated;
the verb was wrong); the §2 opening narrowed to the supportable claim (upstream MLX ships jaccl
-- the fork follows its `lib/jaccl` layout -- but as best-effort UC; what is fork-local is the
*reliability* layer; the old "no jaccl upstream" wording is gone); §3.2's mechanism sentence
softened to "the only mechanism on record" + the broader-teardown note; §3.3/§3.4 SIGKILL
"ban" scoped to the deploy path (the watchdog's own kill is separate); §3.4's ~65 s vs ~90 s
reconciled as kill latency vs detect-and-re-place; §4 intro rewritten to match what §4.1-4.3
actually claim (recoverable vs re-place split, leaks fixed to a plateau, ring fixed); §4.1
"confirmed" softened (the precursor evidence does not separate the two SIGABRT sources); §5.2
adds the per-source-layer cache rule (four kv-source layers, consumers reference) and fixes
"top of the range" to the full-count wording; §5.1's ratio direction fixed (kernels faster than
production MXFP4; 1.909x decode R=1 / 1.979x verify R=4); the §9.3 band note now says the band
times sum to attempt 2's prefill (the earlier question resolved: the first ~299K rows were
cache); §5.4's dichotomy replaced with a three-branch reading (conservative estimate / pages
outside the wired cap / allocator reuse); §6.2's closing line no longer presents a decode
measurement as the correction of a prefill share (62% rests on the disowned instrument; the
decode-side share is 2.9-5.3%); §9.5's "~16%" corrected to "~20% above (that row ~16% below)";
§8.1's governor claim attributed, not asserted; §7.2's token/character comparison anchored
(96 tokens = the 322 characters quoted; the default is 600 tokens = 1,996 chars); §9.2's
"5.3x for 1.9x" fixed to "5.3x from 100K to 300K while the context triples"; §10.8's ~2.9 h ->
~2.8 h (matches §9.4); §9.4 gains the counter-mismatch caveat (attempt 1's 802,816 vs ~98K
cached; the ~197K remainder is an upper bound); §9.3's "position 4K" -> "near the prompt's
start"; §9.1's 0.3% claim qualified as unrounded-values; §9.2's heading softened ("not a
simple artifact"); §9 intro's "first 160K attempt" -> "the 160K attempt"; §11.5's 1M-window
paragraph now explains why attempt 2 finished where attempt 1 hit the cap (cache reuse left
~701K rows) and the stream:true sentence qualified (client-side vs server-side); §11.2's
b88d65dd marked explicitly post-capture; §12.5's ~12.6 GB labelled pre-activation with §5.4's
net figure cited; §12.1's "1,500-2,000" -> "1,350-2,000"; §12.2's draft-prose claim scoped to
the four decoded streams; §12.4 step 3 cites §3.4 for the probe timing.

**Refuted, logged:** 32168 is correct (the engine's MAX_TOKENS constant; 32768 is the ceiling);
the chunk-5 band/wall "arithmetic failure" dissolves once the 299,008-row reuse is applied (the
bands cover exactly the rows attempt 2 had to prefill).
