HANDOFF: Section 43 investigation continued (RDMA p2p_retry_exchange
STALLED bug). TWO real bugs found and FIXED this session, confirmed on
real hardware. The ORIGINAL bug (p2p_retry_exchange STALLED, 300s
zero-progress) is STILL NOT REPRODUCED under the new fixes -- every
run so far has hit a DIFFERENT bug first. A THIRD real symptom just
surfaced in the most recent run and is NOT YET DIAGNOSED -- this is
the stopping point.

Repo: ~/repos/exo, branch main, tree clean, HEAD 7050300c5.
mlx submodule pinned at ba84c66e5. Full prior context:
docs/handoff-2026-08-10-section43.md (first half of this
investigation, same session's earlier state) and design doc Section 43
(docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md).

CLUSTER STATE: as of last check this session, both runners were
RunnerConnecting (mid-reload after the run3b test). Check current
state first thing next session -- may need nothing, may need a fresh
`JACCL_TRACE_PROGRESS=1 ./start_cluster.sh` relaunch if stuck.

=== This session's arc ===

1. Added diagnostic-only instrumentation to p2p_retry_exchange
   (mlx@88291c1f0, exo@4b9afd266): EXCHANGE_ENTER logs every call's
   (peer, data_src_rank, seq, round); EXCHANGE_REJECT logs every
   frame p2p_retry_process_completion silently discards on a magic/
   seq/data_src_rank mismatch. Gated on JACCL_TRACE_PROGRESS=1 --
   MUST be set when running start_cluster.sh or the diagnostic is
   silently inert (bit us once this session -- first deploy forgot
   this and produced zero trace output).

2. Root-caused the handoff's "call_id=27 recurrence" lead as a RED
   HERRING: that "call_id" in the STALLED message is actually
   stall.tick(metric, "p2p_retry_exchange", rank_, seq) -- the
   per-(peer,direction) `seq` counter (send_seq_[dst]++/
   recv_seq_[src]++, uint16_t, 2-element array), NOT the outer
   collective's own call_id. reconnect_fresh() constructs a fresh
   MeshImpl each time, re-zeroing seq to 0. Small numbers repeating
   after a reconnect carry no signal. Confirmed by pulling both
   nodes' full seq sequences for one run: 184 distinct transfers on
   both ranks, ending at the same seq=88 -- no permanent desync in
   that run's data.

3. BUG 1 FOUND + FIXED (mlx@ba84c66e5): p2p_retry_send_bitmask's
   per-slot send-completion wait loop used jaccl_stall_timeout_us()
   (generic default, 8s) instead of jaccl_p2p_retry_stall_timeout_us()
   (this exchange's own dedicated default, 300s) -- directly
   contradicting the adjacent comment, which explicitly says the loop
   is "bounded by the same generous peer-liveness timeout... reuses
   the same backstop for simplicity." CONFIRMED on real hardware:
   this wrong constant threw a premature fatal "NIC/QP fault" error
   after only 8s of a merely-late (not lost) completion under real
   load, crashing a live generation request at ~18s wall-clock --
   well before the outer p2p_retry_exchange loop could ever reach ITS
   own correctly-configured 300s window. This bug was MASKING the
   original Section 43 bug (crashing the exchange before 300s of
   real zero-progress could ever elapse). Fix: use the right
   constant. VERIFIED: re-run after this fix alone reached 27 tokens
   / clean 200 cancel / CPU converged to idle -- but then hit a
   SECOND crash (below), and after fixing that, a THIRD unexplained
   symptom (also below) -- so this fix is necessary but the original
   bug is still unreached.

4. BUG 2 FOUND + FIXED (exo@7050300c5): ExoBatchGenerator.cancel(uids)
   handled three lifecycle states a uid can be in when cancelled
   (chunk-drive prefill, batched-decode queued, batched-decode
   admitted) but had ZERO handling for a FOURTH: PP speculative-decode
   mode (self._pp_spec_gen_by_uid[uid], populated by _submit_pp_spec's
   entry guard). CONFIRMED on real hardware: cancelling mid PP-spec
   decode returned HTTP 200 successfully but left
   _pp_spec_gen_by_uid[uid] populated forever; the NEXT submitted
   request hit _submit_pp_spec's guard and raised
   PPSpecAlreadyActiveError -- which, despite that exception's own
   docstring claiming it "surfaces as a clean task failure the caller
   can retry, not an uncaught crash," was observed to be EXACTLY that:
   full runner process exit (exitcode=0), BOTH ranks independently,
   within ~1ms of each other. Since DSpark speculation is ON by
   default in this cluster (EXO_SPECULATIVE=1), essentially every
   real decode goes through this path. Fix: call the existing,
   already-tested self._close_pp_spec_gen(uid) from cancel() too --
   verified safe as LOCAL-ONLY cleanup (no new cross-rank wire
   handshake needed) because every PP-spec decode loop's `finally:`
   block only resets local layer mode-flags (_configure_layers), zero
   wire I/O -- confirmed by reading pp_dspark_decode_loop's finally:
   block directly. A `consult` review agreed with this reasoning
   after two rounds of verification (checked for rank-gating in the
   call path -- none found, both ranks run cancel() symmetrically
   same as the other 3 branches; checked the finally: block for wire
   ops -- none found). basedpyright showed 305 pre-existing errors,
   ZERO new ones from this edit.

5. VERIFICATION RUN with both fixes deployed (run3b,
   /tmp/section43_run3b.log): NO CRASH this time -- real progress:
   25 tokens seen, cancel POST returned 200 in 5.29s. BUT a NEW,
   UNDIAGNOSED symptom: rank0 (macstudio-m4-1, pid 27104)'s CPU TIME
   kept climbing for the full 90s post-cancel polling window ("STILL
   GROWING at the last sample -- never converged to idle"), while
   rank1 (macstudio-m4-2) converged to idle normally. The post-cancel
   health check (a trivial follow-up request) FAILED with HTTP 500.
   This pattern (one rank busy-looping post-cancel) is structurally
   the SAME SHAPE as the original pre-Section-27 bug the test was
   built to catch ("runner staying pinned at ~100% CPU forever after
   cancellation, since the batched-decode session never learned about
   it") -- but that bug was supposedly fixed by the batched-decode/
   chunk-drive-abort work already in cancel(). Whether this is: (a) a
   NEW gap similar in shape but a different missed lifecycle state,
   (b) a side effect specific to PP-spec generators specifically
   (closing an active generator's underlying MLX ops mid-forward-pass
   without proper drain?), or (c) something else entirely -- NOT YET
   INVESTIGATED. This is the session's stopping point.

=== Next session's concrete priorities, in order ===

1. Check cluster state first (`curl .../state` for runner states).
   If stuck in RunnerConnecting/RunnerLoading, relaunch with
   `JACCL_TRACE_PROGRESS=1 ./start_cluster.sh` and poll to Ready
   before doing anything else (30-90s typical).

2. Pull exo.log from macstudio-m4-1 (rank0, the one whose CPU kept
   growing) for the FULL window from run3b's cancel_issued_at
   (t=546363.5, wall clock ~2026-08-10 14:39:23 CDT -- convert from
   the log's own timestamps) through the end of the 90s poll window.
   Look for: what task/loop is actually running that's consuming
   growing CPU time (is it still inside _step_pp_spec/the PP-spec
   generator body? stuck in a retry loop? something in the
   MetaFrame/batched-decode path?). The [jaccl-p2p]/[jaccl-prog]
   trace lines (still gated on JACCL_TRACE_PROGRESS=1, still live in
   this build) should show exactly where rank0 was looping.

3. Get the actual HTTP 500 error body/traceback from the post-cancel
   health-check request -- exo.log should have it near the end of
   the run3b window. This is likely explained by the same root cause
   as item 2 (runner still busy/wedged when the health-check request
   landed), but confirm rather than assume.

4. Once root-caused: is this the fault of the Bug-2 fix itself
   (self._close_pp_spec_gen(uid) called from cancel() while the
   generator was still actively yielding inside a wire op that
   isn't as side-effect-free as believed -- re-examine
   pp_speculative_decode_loop/pp_chained_decode_loop too, not just
   pp_dspark_decode_loop, since which decode loop is active depends
   on model config) or a pre-existing, separate gap this session's
   fixes simply unmasked (most likely, given the pattern of each
   fix reaching one bug further before hitting the next -- same as
   Bug 1 unmasking Bug 2). Root-cause fix, not a retry/timeout
   mitigation, per this campaign's standing discipline.

5. Once THAT is fixed and verified: re-run the FULL 5x
   section27_cancel_abort_test.py pass. If it clears cleanly for the
   first time this session, that's real progress -- but the ORIGINAL
   Section 43 target (p2p_retry_exchange STALLED, metric=0, 300s) has
   STILL never been reproduced under the diagnostic build. Keep
   running the 5x pass (or longer continuous-generation soak tests)
   until either (a) it reproduces and the EXCHANGE_ENTER/
   EXCHANGE_REJECT trace lines around that window can finally be
   read, or (b) enough clean runs accumulate to reconsider whether
   Bugs 1+2 (both of which crashed/wedged well before 300s could
   elapse) were ACTUALLY masking a distinct third bug all along, or
   were themselves sufficient to explain the originally-reported
   300s stalls (i.e. the "300s" in the original report may have
   been Bug 1's 8s timeout accumulating across many
   reconnect_fresh() cycles, not one single 300s stall -- worth
   cross-checking against the ORIGINAL session's raw log timestamps
   if that log is still on disk, to see if the gap between events
   was actually one continuous 300s window or several shorter
   crash-reconnect-crash cycles that just LOOKED like 300s in
   aggregate).

6. Diagnostic instrumentation (EXCHANGE_ENTER/EXCHANGE_REJECT) is
   still live and low-cost (gated). Leave it in place until the
   original bug is either confirmed root-caused or confirmed to no
   longer reproduce after enough clean runs -- do not strip it
   prematurely.

7. The Section 39/40 mutual-deadlock watchdog cron
   (exo-section39-deadlock-watch) and the Section 41
   CPU/thread-scheduling-contention investigation both remain open
   and untouched -- independent tracks, not blocked on the above.
