HANDOFF: exo chunked-prefill campaign — Section 22 FULLY VALIDATED (worker + client level), fix deployed. Priority: Section 17 next.

Repo: ~/repos/exo, branch main, tree clean, HEAD will be this commit.
mlx submodule pinned at b1e1ae09b (unchanged — jaccl stale-message-seq fix, deployed+validated Sections 18/20).
mlx-lm submodule pinned at bd5d6764 (unchanged).

CLUSTER STATE: torn down cleanly at end of session (zero exo processes, zero
screen sessions, verified via pgrep+screen -ls on both nodes). Next session
needs its own explicit relaunch go-ahead, same standing rule as always.

=== What's proven and safe ===

Section 18/20's jaccl stale-message-seq fix — deployed, validated, real hardware.
Section 22's bounded-blocking-ack chunk-boundary fix — deployed, validated at
BOTH the worker level (Section 23) AND the client/API level (Section 24/25),
on real 2-node hardware, tonight. This is CLOSED. Full details: design doc
Sections 21-25.

=== Summary of tonight's full arc ===

1. Deployed Section 22 (ee7fae663 -> committed as cbad76dc0). Two false-start
   validation attempts (missing EXO_PP_BATCHED_DECODE=1, then missing its
   EXO_PP_METAFRAME=1 prerequisite) caught before being trusted -- both looked
   like clean passes but never actually touched the chunk-drive code path
   (verified via runner-log marker-line greps, not just "request succeeded").

2. Third attempt, BOTH flags set correctly: real chunk-drive activity, but a
   hard 8+ minute client-visible hang. Root-caused via faulthandler Python-
   stack dumps (project's own pre-built SIGUSR1 hook, /tmp/exo_faulthandler_enabled
   marker) on both nodes at the exact moment of the stall: the WORKER was
   completely healthy and idle -- it had already caught the jaccl fault,
   called group.reconnect() (succeeded), called reset_after_reconnect() (drops
   in-flight state), marked the task TaskStatus.Failed, and returned to serving.
   Confirmed via /state too. The bug was NOT in Section 22's own logic at all.

3. Real root cause: API._apply_state()'s event loop (src/exo/api/main.py) only
   ever reacted to ChunkGenerated/NodeGatheredInfo/InstanceDeleted/TracesMerged
   to feed the HTTP client's streaming queue -- it never reacted to
   TaskStatusUpdated. A worker-side TaskStatus.Failed transition (from ANY
   cause, not just jaccl -- runner.py has several send_task_status(...,
   Failed) call sites sharing this exact defect) had zero path to ever reach
   the waiting client. Fixed: new API._fail_stream_for_task() method, wired
   into the event loop on TaskStatus.Failed, sends an ErrorChunk + closes the
   client's queue. 5 new unit tests. Committed as 7a945e5d, pushed.

4. Deployed the fix, then validated it end-to-end on real hardware TWICE:
   - Six consecutive clean 72K-token chunk-drive runs on a warm cluster
     (transport self-healed via jaccl's soft-RC retransmit several times,
     never needed a full reconnect -- also a good confirmation that
     Section 22's chunk-drive path handles ordinary transient hiccups fine).
   - The actual target scenario: fresh cold-start relaunch, first request,
     reproduced the EXACT hard-reconnect signature from Section 23
     (chunk 0, 11 advances, recv() deadline, reconnect_fresh COMPLETE) --
     and this time the client got a real HTTP 200 + SSE error event
     ("Task failed") in 19.7 seconds instead of an 8+ minute hang. Confirmed
     via raw httpx client reading wire-level response lines, not just the
     higher-level test script.

Section 22's validation loop is now closed: chunk-drive works correctly under
normal operation AND under real transport faults, and the client-visible
outcome of a worker-level failure/recovery is now correct too.

=== Incidental finding, NOT chased tonight, flagged for later ===

A throwaway diagnostic script accidentally built a ~459K-token prompt (6x
larger than intended -- a copy-paste loop-count bug in the script, NOT a real
traffic pattern) and hit a DIFFERENT, genuine bug in Section 22's own code:
`GlueError: tick(): reached RANK1_DRAINING with
_prefill_rank1_advances_remaining=0 -- the chunk-drive state machine has a
real bug, refusing to send a meaningless advance`. The glue's own fail-loud
guard caught it correctly (not a hang), and tonight's API fix ALSO correctly
surfaced this one to the client. Worth its own investigation: is this a real
chunk-count/advance-count mismatch bug at very large context, or purely an
artifact of the malformed oversized test prompt? See design doc Section 25.

=== Next session's concrete priorities, in order ===

1. Section 17's memory-headroom check (2 concurrent deep-context KV caches)
   and the real cancel/abort test on real hardware -- now deferred for a
   FIFTH session running. Every deferral so far has been for a genuinely new,
   real, higher-priority discovery (not skipped) -- but if a sixth session
   opens without doing this first, that itself is worth a hard look. START
   HERE unless something breaks loudly enough to demand otherwise.

2. Optional/low-priority: give _fail_stream_for_task's ErrorChunk a more
   diagnostic message on the jaccl-reconnect path specifically (currently a
   generic "Task failed" default, since TaskStatusUpdated carries no message
   text on this fault path).

3. Optional/separate: investigate the RANK1_DRAINING/
   _prefill_rank1_advances_remaining=0 guard found incidentally above.

4. Decide (with the user) whether EXO_PP_METAFRAME=1 EXO_PP_BATCHED_DECODE=1
   should become the new production default now that both the worker-level
   and client-level recovery paths are proven correct, or stays opt-in
   pending more real-world soak time. Not decided tonight.

Standing rules for this repo, unchanged: working tree must stay clean, commit
+ push immediately after every verified change. Any cluster restart needs its
own fresh explicit go-ahead per turn -- approving code is not approval to
deploy or relaunch. Full incident details, in order: design doc Sections
21 (found), 22 (fix), 23 (worker-level validation + stall found), 24 (root
cause + API fix), 25 (client-level validation, closes the loop).
