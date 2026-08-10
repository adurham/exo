HANDOFF: Section 39's deadlock investigation produced one disproven
hypothesis (not a fix); a live cluster check corrected Section 36's
TCP-interface finding, which triggered (then reversed) a call to
shelve Section 37's RDMA migration. Net result: RDMA migration stands
as the plan, resume implementation next session. Priority: Section 37
Phase 1 implementation, deadlock watch continues passively via cron.

Repo: ~/repos/exo, branch main, tree clean, HEAD 4abf1fcc7.
mlx submodule pinned at 67994264f, mlx-lm at bd5d6764 -- UNCHANGED
from the prior session's handoff (no submodule work this session,
pure src/ diagnostics + docs).

CLUSTER STATE: LEFT RUNNING (not torn down), healthy, both runner
PIDs alive (macstudio-m4-1 PID 52092, macstudio-m4-2 PID 51685),
idle/CPU-normal. Launched with DSV4_SHARDING=Pipeline
EXO_PP_METAFRAME=1 EXO_PP_BATCHED_DECODE=1 JACCL_TRACE_PROGRESS=1.
Confirmed jaccl coordinator on this run: en0/Ethernet
(192.168.86.201<->192.168.86.202), NOT Thunderbolt. Real PP split
confirmed via /state: 22 layers rank0 (0-22), 21 layers rank1 (22-43).
Next session can pick this instance up directly (no relaunch needed)
OR relaunch fresh -- either way needs the user's own explicit
relaunch go-ahead if tearing down first, same standing rule as always.

A background cron watchdog is ALSO active:
`exo-section39-deadlock-watch` (job_id 779a344366ec), runs every 15
min via `~/.hermes/scripts/exo_deadlock_watch.sh` (no_agent=true,
zero LLM cost), checks both nodes' runner_log/stderr.log for a
BARRIER line with elapsed_us > 2min AND the same call_id repeating
>20 times in the last 500 log lines -- the actual Section 39/40
deadlock signature. Silent unless it fires; hits are appended to
`~/.hermes/watch/exo_deadlock_hit.log` (empty as of this handoff).
Check that file first thing next session before assuming anything.

=== What's proven and safe (carried forward from prior handoffs) ===

Section 30/31/32/38/39 (prior sessions): all deployed, holding, no
changes this session. See prior handoff (docs/handoff-2026-08-09-
section39.md) for full detail if needed -- not re-summarized here to
keep this handoff focused on what's NEW.

=== This session's actual arc (Sections 40-42) ===

1. Loaded Section 39's handoff: one concrete unresolved bug (a real
   mutual deadlock at call_id=411, both ranks recv()ing from each
   other simultaneously, neither sending, 21+ min silent hang,
   p2p_retry_barrier succeeding every round throughout). Formed ONE
   hypothesis from static analysis: rank0's ceil(peer_prefill_layer_
   count/max_layers) advance-budget arithmetic might under-count what
   rank1's real advance() calls need to reach done=True.

2. Added 4 TEMP DIAGNOSTIC log lines to pp_batched_decode_glue.py
   (LAYER_COUNT_EXCHANGE, RANK0_LOCAL_ADVANCE, HANDOFF_BUDGET,
   PREFILL_ADVANCE_APPLIED) to test this live -- logging only, zero
   runtime effect. Committed+pushed as exo@625d0f32b (this commit is
   STILL LIVE and useful -- do not revert it, it's the instrumentation
   needed to catch the deadlock if it recurs).

3. Deployed, relaunched, ran section27_cancel_abort_test.py 10x over
   ~35 min of real sustained chunk-drive traffic. Result: the
   hypothesis is DISPROVEN by direct measurement (budgeted==applied
   advances exactly, every chunk, every run) but the original deadlock
   also did NOT reproduce (9/10 PASS, 1 FAIL was an unrelated, correctly-
   self-healing genuine jaccl transport fault -- barrier's own 300s
   deadline fired as designed, clean crash+re-place, NOT the silent-
   hang signature).

4. USER CORRECTED FRAMING MID-SESSION: I initially summarized the 9/10
   clean passes as a checkable win. User: "not safe to assume we have
   it fixed?" -- correct call. Non-reproduction in 35 min against a bug
   that took 21+ min of EXTENDED runtime to surface once, tested via
   only one narrow load pattern, is weak evidence, not a clean bill of
   health. Wrote Section 40 with the honest framing: hypothesis
   disproven, bug UNFIXED, do not treat as resolved.

5. User: "move on in the design doc but still watch for the stall."
   Set up the cron watchdog described above (passive, silent, real
   signal detection -- not a blind "still running?" ping). Moved to
   Section 37 (RDMA migration for jaccl's p2p_retry_barrier got-bitmask
   exchange, designed but never implemented) as the next open thread.

6. Before writing any implementation code, sanity-checked the actual
   live interface jaccl's TCP control-plane uses (via `lsof -i -P -n
   -p <runner_pid>`). Found it on en0/Ethernet, NOT en3/Thunderbolt --
   contradicting Section 36's finding from the prior night. Re-verified
   via 2 full teardown+relaunch cycles: 3/3 total this session landed
   on Ethernet, deterministically. Confirmed the interface-selection
   code (find_ip_prioritised) hasn't changed since Dec 2025 -- not a
   regression. Most likely explanation: EXO_DISCOVERY_PEERS (zenoh's
   OWN separate discovery-bootstrap env var, genuinely uses a
   Thunderbolt IP by start_cluster.sh's own design, lives in the same
   log stream) was plausibly what Section 36 actually grepped, conflated
   with jaccl's own MeshGroup coordinator address.

7. Got a `consult` review of Section 37's OR-merge got-bitmask design
   BEFORE finding the interface discrepancy -- independently valuable,
   flagged two real correctness risks: (a) a stale-epoch frame merged
   into the current epoch before an epoch check runs is silent DATA
   CORRUPTION (chunk falsely marked delivered, sender omits it from
   retransmission) -- not just a stall; (b) "reuse drain_acks's retry
   loop verbatim" is wrong -- drain_acks uses a need_recv COUNTER
   (fine for 1 fixed completion), but N variable-length bitmask frames
   need a BITMAP (a counter double-counts duplicate CQEs and can
   falsely report "all received" with a missing chunk_index).

8. Wrote Section 41: correction to Section 36, retracts Section 37's
   stated justification ("TCP starves under RDMA link contention" --
   not supported, the two paths are on separate physical NICs) and
   (WRONGLY, corrected in #9) concluded the whole migration should be
   shelved.

9. USER PUSHBACK: "don't you think full RDMA migration is a good
   idea?? cause I disagree personally if so." Correct catch --
   disproving ONE justification isn't disproving the design decision.
   Wrote Section 42: UN-SHELVES the migration. It stands on independent
   merits (removes blocking-TCP-recv as a stall vector regardless of
   root cause -- including the new leading CPU/thread-scheduling-
   contention hypothesis; architectural consistency; reuses proven
   drain_acks infra; en0 selection is a runtime classification, not a
   permanent guarantee). Standing decision: PROCEED with Section 37
   Phase 1 next session. The consult review's two correctness flags
   from step 7 are now explicit, non-negotiable design REQUIREMENTS
   for the implementation, not reasons to avoid building it.

10. User asked to pause for the night after the Section 42 correction.
    No implementation code was written or attempted. Both doc commits
    (Sections 40-42) pushed clean.

=== Concrete unresolved bug, still open (unchanged from Section 39,
    now with passive monitoring instead of active chasing) ===

The mutual deadlock at the scheduler-protocol layer
(pp_scheduler_wire.py / pp_batched_decode_glue.py) is STILL COMPLETELY
UNFIXED. This session ruled out ONE plausible mechanism (layer-count/
advance-budget mismatch) with real measurement, and failed to
reproduce it in ~35 min of testing (weak evidence either way -- see
Section 40's own caveats). The cron watchdog is the only thing keeping
an eye on it now; it will NOT deliver to this session's chat (local-
only save per Hermes' cron rules) -- check
`~/.hermes/watch/exo_deadlock_hit.log` directly at the start of next
session before doing anything else.

=== Next session's concrete priorities, in order ===

1. Check `~/.hermes/watch/exo_deadlock_hit.log` first. If it has a
   hit, that takes priority over everything else below -- it means
   the deadlock reproduced passively and the diagnostic tracing from
   this session (LAYER_COUNT_EXCHANGE/RANK0_LOCAL_ADVANCE/
   HANDOFF_BUDGET/PREFILL_ADVANCE_APPLIED, all still live) should show
   exactly what state the two ranks were in when it happened.
2. If no hit: proceed with Section 37 Phase 1 implementation --
   migrate `p2p_retry_barrier`'s got-bitmask exchange from TCP
   (mesh_impl.h/rdma.h in the mlx submodule) to a chunked, self-healing
   UC exchange modeled on `drain_acks()`/`ack_connections_`. Concrete
   requirements per Section 42:
   - Epoch/round tag validated BEFORE merge, on a stable
     (copy-then-repost) buffer, not the raw CQE-notified one.
   - A received-frames BITMAP indexed by chunk_index, NOT a
     need_recv-style counter.
   - Explicit termination/release-signal handling (self-terminating
     mutual-bitmask-echo, with the tail case handled: a rank
     satisfying its own exit condition while the peer is still
     retransmitting into a now-stale epoch).
   This is real C++ work in the mlx submodule -- follow the standing
   git-coherent-deploy discipline (edit local mlx checkout -> commit
   +push to adurham/mlx fork main -> bump exo's submodule pointer +
   uv.lock -> git reset --hard on both studios -> start_cluster.sh
   rebuilds from clean submodule). Given the scope (~2-4 new
   functions, real RDMA queue-pair semantics, genuine correctness
   risk if the epoch/bitmap requirements are skipped), this is a good
   candidate for delegating the implementation to `claude -p` with a
   tightly-scoped brief citing Section 37's design + Section 42's
   three explicit requirements -- but the orchestrator (this session)
   should verify the real end-to-end path (a live cancel-test PASS
   with the migration deployed) before trusting a subagent's "tests
   pass" self-report, per the claude-code skill's own standing
   discipline.
3. The CPU/thread-scheduling-contention investigation (checking
   runner-process thread state/QoS during a real stall, using EXO_
   RUNNER_QOS and thread-priority APIs) remains open and is ADDITIVE
   to item 2, not a prerequisite -- can run in parallel or be picked
   up independently if item 2 stalls for any reason.
4. Once Section 37 Phase 1 is implemented and deployed, the standard
   5x-clean cancel-test verification pass applies before calling it
   done, per this campaign's own established discipline.

Full technical detail for everything above: design doc
docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md, Sections 40-42.
