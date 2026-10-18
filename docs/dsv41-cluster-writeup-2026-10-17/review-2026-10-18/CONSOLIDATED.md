# Consolidated findings — pre-verified, awaiting Fable ruling

Source: 4 DSv4.1-Flash subagents (deleg_712f14d2), 22 raw findings, deduped to 21.
Every item below was re-verified by the parent against the cited source before being
listed. Verdicts: CONFIRMED / RE-SCOPED / REFUTED. Nothing has been applied yet.

## CONFIRMED — BLOCK (contradicts source)

R01 [S1-3] WRITEUP.md:529 — "`gu_svh`/`dn_suh` replicated"
SRC: mlx-lm `6391efc:mlx_lm/models/exl3/loader.py:11` "dn_svh replicated"; `:203,207` slices gu_svh and dn_suh.
The writeup names the SLICED pair as replicated — the two pairs are swapped.
FIX: "...matching down K-tiles, `gu_suh`/`dn_svh` replicated"

R02 [S1-2] WRITEUP.md:405 — "`b4871e8` gave it a real subgroup teardown/rebuild path instead of silently degrading to the QP-only reset"
SRC: mlx `git log -1 b4871e8` -> "fix(jaccl): reconnect() must reset the v2 pool QP — TP post-reconnect segfault"; body: reconnect() reset connections_/ack_/p2p_ but skipped pool_connections_.
The commit ADDED the pool-QP reset to the existing path; it did not replace the QP-only reset.
FIX: "`b4871e8` made that QP-only fallback also reset and re-arm the v2 pool QP, which it had silently skipped"

R03 [S1-1] WRITEUP.md:236 — "provides reliability over UC in software for the TP"
SRC: WRITEUP.md:682 "all_gather path, which has no reliability layer at all"; mesh_impl.h:1935 "all_gather was left on the bare".
Same transport, two stories; L236 overstates the scope.
FIX: "...for the all_reduce family; all_gather stays on raw UC posts with a StallWatch only (§6.2)"

R04 [S4-1] WRITEUP.md:1201 — "cd prompts"
SRC: `git ls-tree b88d65dd --name-only | grep -c '^prompts'` -> 0. §11.5 itself says the prompt files live outside the repo.
The only copy-paste test command fails on its first line.
FIX: "mkdir -p prompts first; the capture session's prompt files and their generator are not in this repo (§11.5)"

R05 [S4-2] WRITEUP.md:1123 — "the launcher discovers the nodes over the tailnet"
SRC: `git show b88d65dd:start_cluster.sh | sed -n '1025,1028p'` -> M4_1_IP="192.168.86.48", M4_1_PEER_ID="12D3KooWDGQK...", M4_2_IP, M4_2_PEER_ID all hardcoded (plus NODES at :1113).
A third party cannot launch without editing constants the section never mentions.
FIX: "the launcher hardcodes both nodes' IPs, libp2p peer IDs and NODES (`start_cluster.sh:1025-1028`, `:1113`); edit those before the first launch"

## CONFIRMED — FIX (internal inconsistency)

R06 [S3-1] WRITEUP.md:1105 — "The guard's `action` knob (`warn` vs `error`) was not toggled"
SRC: WRITEUP.md:1389 "CHANGED from `action=\"error\"` to `action=\"truncate\"`".
Open-vs-shipped defect.
FIX: "The guard's `action` knob now defaults to `truncate` (§12.6); `warn` vs `truncate` was not A/B-toggled"

R07 [S3-2] WRITEUP.md:207-208 — "SIGKILL was banned on the data path"
SRC: WRITEUP.md:329 "the reason SIGKILL is banned on the deploy path -- the watchdog's own SIGKILL is a separate".
Label drift left in an un-passed lead section.
FIX: "...banned on the deploy path (the watchdog's own SIGKILL is separate)"

R08 [S3-3] WRITEUP.md:68 — "Every doc in the corpus pins 115000 MB (120.6 GB)"
SRC: WRITEUP.md:583 "The evidence pack's D4.4 formula reads the same knob as 120.0 GB".
FIX: "Most corpus docs pin 115000 MB (120.6 GB) and the pack's D4.4 formula reads 120.0 GB; we use 125.83 GB"

R09 [S3-4] WRITEUP.md:54 — basis row "guard / limiter | live fire at committed=269, period=1"
SRC: WRITEUP.md:1386 "the period-1 banana repro no longer guard-killed".
The basis row is stale against the shipped escape.
FIX: append "(pre-§12.6 rule; the contiguous-period escape ships default ON and no longer kills this repro)"

R10 [S2-5] WRITEUP.md:941 — "Two caveats keep"
SRC: WRITEUP.md:942-945 lists THREE (progress-position rate; partial cache; band swing).
FIX: "Three caveats keep"

R11 [S2-2] WRITEUP.md:885 — "its 6.54 s decode window (30.4 tok/s)"
SRC: evidence-live.md:129 "elapsed=667.9s"; 200 / (667.9-661.36) = 30.58.
The writeup reproduces the pack's own arithmetic slip.
FIX: "(30.6 tok/s = 200/6.54, not 30.4)"

R12 [S2-6] WRITEUP.md:958 — "`reuse=299008` -- those rows are attempt 1's partial prefill of the same prompt, left resident"
SRC: WRITEUP.md:943 "that run carried a partial cache of its own (~98K)"; evidence-live.md:704 "matches a resident conversation on 299946 rows".
Attempt 1's cache was ~98K, not 299K.
FIX: "`reuse=299008` -- those rows are a resident prefix of the prompt left by attempt 1, whose own cache is unreconciled"

R13 [S2-7] WRITEUP.md:708 — "The last deltas on the wire, verbatim, in order:"
SRC: evidence-live.md:296-303 — six earlier deltas precede the quoted seven.
FIX: "...in order (the six before them, up to the newline delta, are elided):"

R14 [S4-3] WRITEUP.md:1169 — "git checkout main"
SRC: `git rev-parse main` -> 1f91afe3a; the table at :1135 pins b88d65dd.
The documented command does not reproduce the documented build.
FIX: "`git checkout b88d65dd330478e8e1b8a565848af3c863be7a06` (the SHA in the table above), then `git submodule update --init --recursive`"

R15 [S4-4] WRITEUP.md:1173 — "Shutdown is SIGTERM-only via `scripts/exo_graceful_shutdown.sh`"
SRC: `git show b88d65dd:scripts/exo_graceful_shutdown.sh:5-6` "Runs ON the target node. start_cluster.sh delivers it over `ssh \"$NODE\" bash -s` STDIN".
Implies the controller runs it.
FIX: "...runs ON each node (the launcher pipes it over `ssh node bash -s`), not on the controller"

## CONFIRMED — NOTE (unsupported / imprecise, not contradicted)

R16 [S1-4] WRITEUP.md:564 — "the limit is read in mlx `device.cpp:768-773`"
SRC: `grep -n 'max_mb_per_buffer_ = env' mlx/mlx/backend/metal/device.cpp` -> 807.
FIX: "device.cpp:807"

R17 [S2-3] WRITEUP.md:902 — "at every" depth
SRC: no round-size data exists for the 1M runs (they were streamed; evidence-live.md:687,864).
FIX: "...at every depth measured for it (1K-300K; the 1M runs have no round data)"

R18 [S2-4] WRITEUP.md:916 — "per-round cost of unidentified origin"
SRC: evidence-live.md:211 "a per-round memory-side cost (KV read / residency)".
The writeup is weaker than its own pack.
FIX: "per-round memory-side cost whose owner is unidentified (KV read vs residency is a named candidate, not proven)"

R19 [S4-5] WRITEUP.md:1249 — "over its `0.75 x max_tokens` default (capped at 16384)"
SRC: `git show b88d65dd:...reasoning_budget.py:91-92` returns request_value verbatim, uncapped; the ceiling bounds only the default.
FIX: "specify the 16384 ceiling bounds the default only; a positive per-request value is returned verbatim, uncapped"

R20 [S4-6] WRITEUP.md:944 — "a cross-attempt swing of the size the repo's own boot-variance notes describe"
SRC: evidence.md:291 "between-boot spread 6.02 tok/s" on ~32 tok/s decode medians (~19%); this swing is 68.3 -> 98.1 = +44%.
The claim understates.
FIX: "a 44% same-position swing (68.3 to 98.1 tok/s), larger than the 6.02 tok/s between-boot spread the corpus documents"

## RE-SCOPED

R21 [S2-1] WRITEUP.md:842 — "agree to within 0.3% on every deep row on the unrounded values"
SRC: evidence-live.md rows — d100k 6.49/6.50 = 0.154%, d300k 31.67/31.66 = 0.032% (both under 0.3%); d10k 0.625%, d50k 0.363%.
The claim is TRUE as scoped to deep rows; the agent tagged BLOCK but it is a clarification.
FIX: "...on every deep row (100K-300K) on the unrounded values; the shallower 10K/50K rows imply 0.6%/0.4%"

## REFUTED

None. All 21 survived re-verification.
