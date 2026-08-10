HANDOFF: Section 37 Phase 1 (RDMA p2p-retry migration) deployed to real
2-node hardware. TWO real bugs found and root-cause-fixed, confirmed
working. A THIRD real bug found inside p2p_retry_exchange itself --
NOT fixed. This is the actual stopping point for this session.

Repo: ~/repos/exo, branch main, tree clean, HEAD 7e7232445.
mlx submodule pinned at c8369ccf1 (bumped TWICE this session -- see
below). Full technical detail: docs/hybrid-pp-prefill-tp-decode-
design-2026-08-04.md, Section 43.

CLUSTER STATE: LEFT RUNNING (self-healed after the last p2p_retry_exchange
stall+reconnect cycle), both runners RunnerRunning, PP mode.
macstudio-m4-1 PID 12916, macstudio-m4-2 PID 10662. Launched with
DSV4_SHARDING=Pipeline EXO_PP_METAFRAME=1 EXO_PP_BATCHED_DECODE=1
JACCL_TRACE_PROGRESS=1 -- same env as every prior session. Cluster is
"healthy" by health-check but CANNOT currently complete a single
end-to-end generation request -- every real request stalls and gets
reconnect-recovered, losing the in-flight work every time.

=== This session's arc: deploy Section 42's already-written code,
    find and fix two bugs, find a third ===

Section 42 (prior session) un-shelved Section 37 Phase 1 and left it
already implemented and committed (exo@e3a39694a / mlx@42cb74fc1) but
NEVER DEPLOYED/TESTED on real hardware. This session's job was purely
deploy + verify.

1. Relaunched the cluster. Build completed fine (both nodes' uv sync /
   mlx build succeeded, no code issue there), but runners looped forever
   in PREPARING. Root-caused via `ibv_devinfo -v` on both nodes: the
   Thunderbolt RDMA HCA reports `max_qp=3`, a real hardware/driver
   ceiling. jaccl's MeshGroup ctor was building 4 QP types per peer
   (connections_/ack_connections_/pool_connections_ from before, plus
   Section 37's new p2p_retry_connections_). The 4th ibv_create_qp
   always failed EBUSY. The EXISTING backoff-retry code
   (_init_jaccl_with_backoff, utils_mlx.py) was written for a genuinely
   DIFFERENT transient cause (leaked QPs from a crashed runner) and
   retried forever without ever succeeding, because this cause was
   structural. Confirmed via `consult` before implementing: mode-
   conditional QP construction (only build the QPs the active mode
   needs) was the correct root-cause fix, not a deeper retry/backoff.

2. FIX 1 delegated + implemented + reviewed + deployed: mlx@49b316d5d,
   exo@bc6383cd5. New MLX_JACCL_SHARDING_MODE env var (mirrors
   DSV4_SHARDING, exported by start_cluster.sh). PP mode builds
   connections_ + p2p_retry_connections_ + ack_connections_ (skips
   pool_connections_). TP mode (default) builds connections_ +
   ack_connections_ + pool_connections_ (skips p2p_retry_connections_).
   Both = exactly 3 QPs. The implementing subagent caught a real gap
   the orchestrator's brief missed: reconnect_fresh() also unconditionally
   rebuilt all 4 vectors and needed the same gating, or every hard-
   recovery cycle would re-hit the same EBUSY. Verified via `git diff`
   review before deploy: polarity correct at all 4 gate sites, no stray
   edits, subgroup ctor (TP-only, unaffected) reread and confirmed still
   correct.
   VERIFIED ON HARDWARE: relaunch showed zero "Couldn't create queue
   pair" errors; live process env confirmed
   MLX_JACCL_SHARDING_MODE=Pipeline; jaccl init succeeded past the point
   that used to loop forever.

3. FIX 1 immediately exposed FIX 2's bug on the very next relaunch: PP
   mode's one warmup-time collective (exchange_prefill_peer_layer_count
   / handshake_metaframe_protocol, both call mx.distributed.all_sum once
   at model-load time) now had pool_connections_ empty, so all_reduce()'s
   dispatch fell through to reliable_all_reduce (non-v2) -- which posts
   on connections_[peer], the SAME QP PP's raw send()/recv() MetaFrame
   pipeline traffic uses. Confirmed on real hardware: deterministic
   20/20 crash, "MetaFrame protocol version mismatch: received 16256".
   16256 = 0x3F80 hex = the high half of IEEE-754 1.0f -- literal
   all_reduce payload landing in a MetaFrame header buffer, not noise.
   Exact same two-protocols-on-one-QP bug class this codebase already
   split ack/pool/p2p_retry into dedicated QPs to avoid (twice,
   documented in its own comments).

4. FIX 2 delegated + implemented + reviewed + deployed: mlx@c8369ccf1,
   exo@7e7232445. New ack_all_reduce_small() runs PP's tiny warmup
   collectives over the otherwise-idle ack_connections_ QP instead.
   Non-trivial: post_ack_recvs(0) already pre-posts 64 recv WRs on that
   QP at ctor time (before Python even sees the ctor return) -- a naive
   fresh post_recv would queue BEHIND those 64 and still read the wrong
   slot (same corruption class in a new shape). Fix reuses ack_sync_pre's
   posting pattern + a forked drain_acks_exchange() that reads the
   landed payload out of ack_recv_buffers_ BEFORE the existing
   replenish-path memsets it. Falls back to reliable_all_reduce for
   anything it can't service (>2 ranks, payload > one FRAME_SIZE=4096B
   ack buffer). Verified via manual trace of every symbol referenced
   (StallWatch, cached_ack_recvs_, wr_id_work_type/peer/call_id,
   make_wr_id, ACK_RECV_WR/ACK_SEND_WR, poll()) confirming none were
   invented, and confirmed all_sum is collective_mutex_-guarded so no
   race with other collectives.
   VERIFIED ON HARDWARE: relaunch reached READY (2/2), both runners
   RunnerReady, zero crash/mismatch errors -- FIRST time all day the
   cluster came up clean end-to-end.

5. Ran the standard 5x section27_cancel_abort_test.py verification pass
   (required before calling Section 37 Phase 1 done, per this campaign's
   own established discipline). Every run failed identically: zero
   tokens seen, command_id=None, ~305s elapsed, no client-visible error.
   Root cause found in runner stderr on BOTH nodes:

     [jaccl] p2p_retry_exchange STALLED rank=1 call_id=27 metric=0
     (no forward progress for >300000ms; UC completion lost — throwing
     for clean re-place)

   This is INSIDE p2p_retry_exchange itself -- the actual protocol
   Section 37 Phase 1 built to replace the old TCP p2p_retry_barrier,
   NOT either of today's two QP-allocation fixes (both of which are
   confirmed working correctly up to this point and should NOT be
   reverted while investigating this). metric=0 means the STALLED
   throw's own liveness metric (a popcount over peer_frame_seen) never
   moved for the full 300s -- i.e. literally zero frames were ever
   confirmed received, in either direction, the whole time.

   Hit 3 times across the 5x run (call_id=192 once, call_id=27 twice).
   The call_id=27 RECURRENCE across what should be logically distinct
   exchanges is itself suspicious and UNEXPLAINED -- flagged as the
   most promising lead, not yet investigated.

   Each time, MLX_JACCL_RECONNECT_FRESH=1's soft-recovery successfully
   rebuilds the QPs in-place ("Attempting in-place reconnect (both
   ranks) to avoid a re-place" -> reconnect_fresh rank=1 ENTER -> benign
   IOConnectUnmapMemory kr=0xe00002c2 noise -> cluster back to
   RunnerRunning) but the in-flight REQUEST is always lost. No request
   has completed successfully in PP mode with today's fixes deployed.

=== User's decision ===

Given three options (keep digging now / revert Section 37 Phase 1
entirely and stop / get full details first and decide after), user
chose: STOP HERE, write this handoff for a fresh session. No
investigation into Bug 3 beyond locating+confirming the STALLED throw
site was done this session.

=== Next session's concrete priorities, in order ===

1. Read p2p_retry_exchange's actual implementation in
   mlx/distributed/jaccl/lib/jaccl/mesh_impl.h (search
   "p2p_retry_exchange" and "P2P_RETRY_STALL_TIMEOUT" -- there is a
   class doc comment near its definition describing the intended
   protocol design from Section 37/39's prior-session implementation).
2. Investigate the call_id=27 recurrence first -- it's the most
   concrete, most suspicious lead. Confirm whether call_id reuse across
   distinct p2p exchanges is by-design (some other scoping mechanism
   makes it safe) or itself the bug (e.g. causes the receiver to
   misclassify/discard real incoming frames against a stale call_id,
   which would directly explain metric=0 / "no frame ever seen").
3. Cross-reference against the two correctness requirements the
   `consult` review flagged for Phase 1 BEFORE the prior session
   implemented it (Section 42): (a) epoch/round tag validated BEFORE
   merge on a stable copy-then-repost buffer, not the raw CQE-notified
   one; (b) a received-frames BITMAP indexed by chunk_index, not a
   need_recv-style counter. A "genuinely zero progress, ever" stall
   (as opposed to slow-but-progressing) is consistent with one of these
   two requirements not actually being met by the as-written
   implementation -- check this before assuming a subtler timing bug.
4. Once root-caused and fixed: redeploy, then re-run the SAME 5x
   section27_cancel_abort_test.py pass that caught this bug (do not
   skip straight to declaring success -- Bugs 1 and 2 both looked fine
   until this specific test ran). Artifacts from today's failed runs:
   /tmp/section27_run1.log, /tmp/section27_run2.log (runs 3-5 were
   killed once the failure pattern was confirmed non-random, not worth
   re-reading).
5. The CPU/thread-scheduling-contention investigation from Section 41
   remains open and untouched -- independent track, not blocked on or
   blocking item 1-4 above.
6. The Section 39/40 mutual-deadlock watchdog cron
   (exo-section39-deadlock-watch, ~/.hermes/watch/exo_deadlock_hit.log)
   is STILL ACTIVE and STILL EMPTY -- but it watches for a DIFFERENT
   signature (a BARRIER line's elapsed_us climbing) than today's bug
   (p2p_retry_exchange's own internal STALLED throw, which already logs
   loudly and unprompted to exo.log). The watchdog likely does not need
   a new pattern for this -- the existing throw is already sufficiently
   loud -- but worth a 30-second sanity check before assuming so.
