HANDOFF: jaccl TCP/RDMA reliability campaign — Section 39's non-fatal
retransmit fix deployed + verified (zero crashes across real testing), but
it unmasked a genuine SECOND bug: a real send/recv mutual deadlock at the
scheduler-protocol layer, no longer force-cleared by jaccl's old crude
timeout. Priority: root-cause that deadlock next.

Repo: ~/repos/exo, branch main, tree clean, HEAD d58fc80f3.
mlx submodule pinned at 67994264f (Section 38's p2p_channel_ deadline fix +
Section 39's non-fatal retransmit-cap fix, both this session).

CLUSTER STATE: torn down cleanly at end of session (zero exo processes on
both nodes, verified via ps aux | grep -c). Next session needs its own
explicit relaunch go-ahead, same standing rule as always.

=== What's proven and safe ===

Section 30/31 (prior sessions): recovery-handshake side_channel_ deadline
extended to 150s — deployed, holds.
Section 32 (prior session): reset_after_reconnect() recreates batched-decode
glue objects instead of enumerating fields — deployed, verified via a real
A/B-tested regression test, 322/322 suite passing.
Section 38 (this session): p2p_channel_ (the retry-barrier TCP channel) gets
its own principled 300s deadline instead of sharing the 60s data-path
deadline, plus wait-time logging on eventual-success recv() so future
tuning has real data instead of survivorship-biased deadline-exceeded-only
samples. Deployed, compiled clean, real-hardware tested.
Section 39 (this session): send()/recv()'s own internal 15s/40-round
retransmit cap no longer throws fatally — it was a second, redundant
timeout layered on top of a protocol that already has a real liveness
check (p2p_retry_barrier's own TCP recv). Deployed, compiled clean. Held
through 2 real cancel-test runs post-deploy with ZERO crashes and ZERO
re-places — same PIDs the entire time — a first for this whole multi-
session campaign. This is real, verified progress, not a hopeful patch.

=== Summary of tonight's full arc (Sections 33-39) ===

1. Deployed Section 32 (from prior session) to both studios, ran the cancel
   test against it — 3 separate real jaccl transport faults hit during
   testing, ALL THREE recovered with zero runner crashes (first time this
   campaign). But a 4th, subtler gap surfaced: after the 3rd recovery, both
   runners went idle and never resumed dispatching — faulthandler dumps
   showed both ranks waiting on local IPC, not blocked on RDMA. Documented
   as Section 33, corrected in Section 34 (see next).

2. User challenged Section 33's framing directly ("I know apple doesn't
   have hardware support for RC. That's the ENTIRE point of the soft-RC
   that we built"). Went back to the actual jaccl source instead of
   restating the premise: confirmed the soft-RC layer (ack-sync barriers,
   retransmit rounds, p2p_retry_barrier bitmask reconciliation) IS real,
   engaged, and doing its job — MeshGroup::send/recv and the TP collectives
   both route through it, and the ack-sync-pre barrier that closes the
   "peer sends into empty recv queue" race is actually ON by default at
   every real launch (start_cluster.sh overrides the source default).
   Corrected in Section 34: the real open question is narrower — is the
   retry budget sized right, not "are we even syncing".

3. User then asked "why are we doing anything over TCP here??? we have
   backend RDMA, not TCP." Verified the physical fabric first (confirmed
   en3, 192.168.200.x, is genuinely the Thunderbolt bridge on both nodes,
   not a LAN fallback) before answering. Found: jaccl runs TWO independent
   networking stacks over the same physical Thunderbolt cable — real RDMA
   verbs for tensor data, but a completely separate plain TCP/IP socket
   stack for coordination (bootstrap, retry-barrier, collective barriers).
   That TCP coordination traffic is what was actually stalling. Documented
   as Section 36.

4. User directed BOTH: fix the TCP implementation on its own merits (for
   upstream — mlx's own send()/recv() has ZERO reliability logic, an
   unbounded-hang bug on any packet loss, confirmed by reading
   upstream/main directly via the already-configured upstream remote) AND
   separately plan an RDMA migration for exo's own future transport.
   Sections 37-38 laid out both tracks: RDMA migration for exo's own
   per-round hot-path traffic (reusing the proven drain_acks/
   ack_connections_ pattern, NOT for bootstrap/reconnect which are
   structurally TCP-only), plus TCP hardening as real standalone work.

5. Implemented the concrete TCP-hardening piece: p2p_channel_ (previously
   sharing the 60s data-path deadline, deliberately, "should keep failing
   fast") gets its own 300s deadline plus wait-time logging (Section 38).
   Consulted on how to size it — rejected cloning side_channel_'s formula
   (that number is derived from a provable skew bound that doesn't apply
   here) and rejected keeping 60s (already falsified by real hardware).
   Landed on a large default justified by cost asymmetry: false positives
   fire during healthy operation and drop real work; genuinely dead peers
   surface via TCP errors well before any reasonable deadline anyway.

6. Deployed Section 38, relaunched with JACCL_TRACE_PROGRESS=1, ran the
   cancel test. Reproduced a real fault. Found DEADLINE_HIT evidence in
   the trace: a transfer legitimately ran 29 retransmit rounds over 15s
   with p2p_retry_barrier succeeding EVERY round (proving both ranks
   alive) — yet still threw fatally, because send()/recv() had their OWN
   separate, redundant 15s/40-round cap independent of the barrier's
   liveness check. Investigated a possible cross-rank call_id desync as an
   alternate hypothesis first, ruled it out via the code's own documented
   invariant (call_id is a per-process counter, not shared across ranks)
   before concluding — avoided chasing a false lead.

7. User interrupted directly mid-investigation: "you are still trying to
   bandaid it man... the goal is that it's not blocked, or if that can't
   be done for whatever reason that it's never a fatal wait." Implemented
   the real fix immediately: removed the fatal throw from send()/recv()'s
   internal round/deadline caps entirely (Section 39) — they're now purely
   informational logging, and p2p_retry_barrier's own recv (Section 38's
   300s deadline) is the SOLE way this protocol can end in real failure.
   Verified this doesn't introduce unbounded CPU spin (existing sleep-on-
   empty-poll + barrier's own TCP round-trip cost naturally rate-limits
   the loop).

8. Deployed Section 39, real-hardware tested: 2 cancel-test runs, zero
   crashes, zero re-places, same PIDs throughout — direct proof the fix
   works for its target case (legitimately slow-but-alive transfers no
   longer treated as fatal).

9. Further testing (letting the cluster run past the initial verification)
   surfaced the real cost of removing that safety net: found call_id=411
   stuck for OVER 21 MINUTES, p2p_retry_barrier still succeeding every
   round (transport genuinely healthy the whole time) — but faulthandler
   dumps on both nodes show a real MUTUAL DEADLOCK: rank0's recv() is
   waiting FOR rank1, rank1's recv() is simultaneously waiting FOR rank0,
   neither side is calling send(). This is at the scheduler-protocol layer
   (pp_scheduler_wire.py's tick()/recv_header()/
   recv_prefill_chunk_done_ack_message(), pp_batched_decode_glue.py) —
   ABOVE jaccl, not a transport bug at all. jaccl's old crude 15s/40-round
   timeout used to be the ONLY thing that ever force-cleared this specific
   failure mode (via crash + re-place). Section 39 correctly stopped
   punishing slow-but-healthy transfers, but in doing so removed the one
   mechanism that used to (accidentally) recover from this real, separate
   deadlock bug.

=== Concrete unresolved bug for next session ===

A genuine send/recv ordering deadlock exists in the scheduler protocol
layer (pp_scheduler_wire.py / pp_batched_decode_glue.py — NOT jaccl/mesh
layer). Both ranks can end up simultaneously calling recv() waiting on
each other with neither calling send() first. Confirmed via faulthandler
Python-stack dumps on both nodes at the same moment (rank0 PID captured in
this session's dumps, rank1 too — see design doc Section 39 for the exact
stack traces and log excerpts). Evidence signature to grep for: a
call_id's [jaccl-prog] BARRIER lines showing all_recv stuck at a partial
count (e.g. "got_count=0/1 all_recv=0/1") repeating for hundreds/thousands
of rounds with elapsed_us climbing into the tens of minutes, on BOTH sides
simultaneously (both ranks recv()ing the SAME call_id from each other, not
one sending + one receiving).

Needs: (1) trace the actual protocol logic in tick()'s state machine to
find the code path where both ranks can reach a recv-before-send state for
the same logical exchange — likely an ordering/state-machine bug in how
prefill-chunk-done acks are sequenced between rank0 and rank1. (2) Decide
on a recovery mechanism for this class of failure now that jaccl's crude
timeout no longer accidentally provides one — the right answer is almost
certainly application-level deadlock detection (e.g. a watchdog on
scheduler tick() progress) rather than reintroducing another jaccl-layer
timeout, which would just re-create Section 39's exact problem one layer
down.

=== Also still open, unimplemented design work ===

Section 37: RDMA migration for jaccl's per-round control-plane traffic
(p2p_retry_barrier's got-bitmask exchange, in-collective barriers) onto a
self-healing RDMA-native UC exchange, reusing the proven drain_acks/
ack_connections_ pattern. Bootstrap and reconnect-recovery stay on TCP
permanently (verified against the actual code: side_channel_ is
constructed before the RDMA queue pairs even exist, and reconnect_fresh()
tears down and rebuilds ack_connections_ itself, so a recovery path can't
depend on the thing it's recovering). Migration sketch (chunk the
bitmask into fixed-size ACK-QP frames tagged (epoch, round, chunk_index),
OR-merge on receive) is written up but not started. Two open design
details flagged, unresolved: stale-epoch rejection, explicit barrier-
release signal.

=== Next session's concrete priorities, in order ===

1. Root-cause the pp_scheduler_wire.py mutual-deadlock (see above) — this
   is the actual blocker now that Section 39 removed the accidental
   crash-based recovery for it. Read tick()'s full state machine in
   pp_batched_decode_glue.py (Rank0BatchedDecodeGlue at line ~450,
   Rank1BatchedDecodeGlue at line ~1437) alongside pp_scheduler_wire.py's
   recv_header/recv_prefill_chunk_done_ack_message to find the exact
   ordering bug.
2. Once understood, decide + implement the right recovery mechanism
   (likely app-level watchdog, not another jaccl timeout).
3. Run the cancel test 5x clean (established campaign discipline) against
   Section 38+39's fix once the deadlock is also addressed, to get a real
   pass-rate distribution — tonight only reproduced the target scenario
   with a small number of runs before the deadlock investigation took
   priority.
4. Section 37's RDMA migration remains open, unimplemented design work —
   pick up if the TCP-hardening track (38/39) plus the deadlock fix are
   sufficient, or continue in parallel if there's still time pressure to
   move off TCP entirely.

Full technical detail for everything above (exact commands, log excerpts,
stack traces, consult reviews, code diffs discussed): design doc
docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md, Sections 33-39.
