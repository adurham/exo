HANDOFF: exo chunked-prefill campaign — Section 22 deployed, real STALL found in Section 23, root cause not yet identified

Repo: ~/repos/exo, branch main, tree clean (after this commit), HEAD will be this commit.
mlx submodule pinned at b1e1ae09b (unchanged this session — jaccl stale-message-seq fix, deployed+validated Sections 18/20).
mlx-lm submodule pinned at bd5d6764 (unchanged this session).

CLUSTER STATE: torn down cleanly at end of session (zero exo processes, zero screen
sessions, verified via pgrep+screen -ls on both nodes) at explicit user request
("leave it as-is and write up findings only"). NOT relaunched into ANY config,
including the previously-safe default. Next session needs its own explicit
relaunch go-ahead, same standing rule as always.

=== What's proven and safe (unchanged from before this session) ===

Section 18/20's jaccl stale-message-seq fix (mlx bdf78e752 + dead-kernel removal
b1e1ae09b) — deployed, validated on real hardware, two consecutive ~72K-token
chunked-prefill runs past the crash-prone chunk 51-60 depth, zero desyncs, zero
crashes. This is the cluster's known-good baseline: relaunch with
EXO_PP_METAFRAME=0 EXO_PP_BATCHED_DECODE=0 (both flags' script defaults — a bare
`./start_cluster.sh` with no overrides) to get back to this proven state.

=== What THIS session did ===

1. Deployed Section 22's fix (ee7fae663, bounded-blocking-ack chunk-boundary
   race fix) to both studios via clean git reset — succeeded, pure Python change,
   no rebuild needed. Verified commit cbad76dc0 on both nodes.

2. Attempted real-hardware validation per the PRIOR handoff's instructions — hit
   TWO false starts before reaching a real test, each one a genuine methodology
   bug, not a cluster problem:
   a. Bare `./start_cluster.sh` (no env overrides) → EXO_PP_BATCHED_DECODE
      defaults to 0 in the script. Two "clean" 72K-token runs completed with
      zero errors, but log inspection proved ZERO chunk-drive activity —
      the traffic silently took the old synchronous-prefill fallback path the
      entire time. This was NOT a validation of anything Section-22-specific.
   b. EXO_PP_BATCHED_DECODE=1 alone → still zero chunk-drive activity.
      EXO_PP_BATCHED_DECODE=1 requires EXO_PP_METAFRAME=1 as a hard
      prerequisite (install_batched_decode_pipeline_layers only runs inside
      the METAFRAME=1 branch in utils_mlx.py) — undocumented as a launch
      requirement in the prior handoff, which had asserted both flags were
      "active in production."
   LESSON: a clean/successful-looking generation is NOT evidence a specific
   code path executed. Always grep the runner log for that path's own marker
   lines (PREFILL_REGISTER / chunk_index / PrefillChunkDoneAck) before trusting
   a green run as validation of anything path-specific.

3. Third attempt, BOTH flags set (EXO_PP_METAFRAME=1 EXO_PP_BATCHED_DECODE=1):
   REAL chunk-drive activity confirmed (matched PREFILL_ADVANCE_SEND/RECV,
   seq 1-11, both ranks). start_cluster.sh itself printed a warning that this
   was "the FIRST real cluster run of this path" — this exact combination had
   never been exercised on real 2-node hardware before tonight.

   RESULT: hard stall. Chunk 0's 11-advance sequence completed cleanly on both
   ranks, then BOTH immediately hit `[jaccl] recv() deadline in drain`, did a
   clean `reconnect_fresh` (device contexts rebuilt, completed without error on
   either side), and then produced ZERO further activity for 8+ minutes (test
   killed at that point — the self-abort ceiling is 30 minutes,
   MLX_EVENT_WAIT_TIMEOUT_MS=1800000, never reached). GPU confirmed idle via
   powermetrics during the stall (23mW, ~6% active residency) — genuinely
   stuck, not slow-but-working.

   Full design-doc writeup with both competing hypotheses (Section-22-specific
   gap in the post-reconnect recv path, vs. a pre-existing jaccl transport
   flakiness that Section 22's new blocking wait merely exposes for the first
   time): docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md, Section 23.

   Evidence preserved: docs/incidents/2026-08-08-section23-stall-{m4-1,m4-2}.log
   (both nodes' runner stderr covering all three launch attempts).

   Validation script: bench/section22_chunk_drive_validate.py (self-written —
   the existing bench/context_stress.py is hardcoded to a DIFFERENT model
   mlx-community/Qwen3.5-397B-A17B-4bit not loaded on this cluster's current
   placement, 404s against DeepSeek-V4-Flash; worth fixing/parameterizing if
   this class of test gets run again).

=== Section 22's fix status: NOT validated, NOT safe to leave default-on ===

The prior handoff's confidence that this fix was ready modulo real-hardware
validation was premature — the validation itself surfaced a new, real, blocking
failure mode. Section 22's bounded-blocking-ack design is not proven broken
either — the stall may be entirely a pre-existing jaccl transport issue that
the old fire-and-forget rank-0-never-waits behavior simply never surfaced
(nothing on the old path ever blocked long enough to notice a mid-stream jaccl
reconnect). Both hypotheses are live; neither is confirmed.

=== Next session's concrete priorities, in order ===

1. Do NOT relaunch with EXO_PP_METAFRAME=1 EXO_PP_BATCHED_DECODE=1 again without
   first reading design doc Section 23 in full and deciding which hypothesis to
   chase first.

2. Reproduce the stall ONE more time deliberately, with either extra jaccl-level
   tracing or a much shorter MLX_EVENT_WAIT_TIMEOUT_MS override (e.g. 60000 = 1
   minute instead of 30) on a throwaway diagnostic run, to determine: does
   Rank0BatchedDecodeGlue's bounded recv ever resume after reconnect_fresh
   completes (just very slowly), or does it never observe the reconnect at all
   (structural gap needing a real fix)?

3. Depending on which hypothesis wins:
   - If Section 22's own gap: fix is likely in the
     recv_prefill_chunk_done_ack_message call in pp_batched_decode_glue.py
     (~line 1150) — needs either a retry-after-reconnect path, or the jaccl
     "clean re-place" language needs to actually propagate as a Python-catchable
     exception through mx.distributed.recv_like's binding rather than hanging.
   - If pre-existing jaccl flakiness: this becomes an mlx-submodule (C++)
     investigation into what happens to an in-flight recv when reconnect_fresh
     fires mid-wait — not an exo-side fix.

4. Section 22 should NOT be considered validated/safe-default until this stall
   is root-caused and fixed. Falling back to Section 20's already-proven state
   (both flags OFF, the script default) is not a regression — it's simply the
   currently-proven config.

5. Section 17's memory-headroom check and real cancel/abort test remain
   deferred — now for a fourth session running, each displaced by a genuinely
   new real discovery rather than skipped. Worth flagging the streak.

Standing rules for this repo, unchanged: working tree must stay clean, commit +
push immediately after every verified change. Any cluster restart needs its own
fresh explicit go-ahead per turn — approving code is not approval to deploy or
relaunch. Full incident details: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md,
Section 23 (newest entry, at the bottom).
