# Dual-cable topology split + jaccl QP budget (2026-08-21)

Session summary: a multi-hour live debugging campaign fixed 9 real jaccl
(RDMA-over-Thunderbolt) transport bugs, split the cluster's two physical
Thunderbolt 5 cables so RDMA and TCP coordination traffic no longer share a
link, and investigated (and rejected, with reasons) collapsing the RDMA data
path's 3 queue pairs down to fewer/one. This doc is the standing reference so
none of this needs re-deriving from scratch later.

## 1. The bug chain (jaccl transport, 9 fixes)

All commits on `adurham/mlx` (fork; none of this exists upstream on
`ml-explore/mlx` — see section 5).

1. `b4871e809` — `reconnect()` segfault when a TP coord subgroup exists:
   the QP-only reset fallback couldn't actually clear a dead-UC wedge on a
   group that had `split()`, but ran anyway and crashed on the next
   collective. Fixed by giving `reconnect_fresh()` a real subgroup
   teardown/rebuild path instead of silently degrading.
2. `9a5921683` — missing MR (memory region) registration for the batched
   data-QP recv pool array (`data_pool_recv_buffers_`), added by an earlier
   commit that allocated the array but never registered it — every fresh
   runner's first `mx.distributed.init()` threw
   `unordered_map::at: key not found`, 100% deterministic.
3. `5d2c42435` — `reconnect()`/`reconnect_fresh()` held no lock while a
   concurrent collective could still touch the group being torn down —
   use-after-free segfault. Fixed with proper `collective_mutex_` coverage
   including subgroups' own mutexes (parent→child lock order).
4. `b27a101dc` — jaccl-v2's standing recv pool was armed *lazily* (on first
   use, after the bootstrap barrier) instead of eagerly, on fresh init —
   exposed a UC silent-drop race identical in shape to 3 other already-fixed
   standing-pool arming bugs, just on this specific pool.
5. `0a8d8a4ee` — `all_gather`'s prefill posted its send immediately after
   `ack_sync_pre()`, which only proves the peer *entered* the call, not that
   its data recv WRs are posted — UC silently drops the early send. Fixed
   with the same "post recvs → confirmed TCP-coordinator barrier → post
   sends" ordering `all_reduce` already used.
6. `f8b77fe5a` — the fix in #5 was unreachable on `split()` subgroups: only
   the top-level ctor ever called `set_coordinator()`, so
   `coordinator_ != nullptr` gates could never engage on a subgroup. Gave
   subgroups their own dedicated TCP coordinator (`coord_channel_`).
7. `5d2c42435`'s sibling / `9d3dc9296`-`45697d100` — diagnostic-only commits
   tracing why the fix in #6 still didn't close the warmup stall.
8. **`fa69bafd1` — the actual root cause: `split()` (RDMA subgroups) cannot
   work under TP on this hardware at all.** See section 2. Every
   coord-subgroup collective was silently falling back to sharing the
   top-level model group's `next_call_id_` counter with live TP traffic —
   the real cause of the deterministic warmup stall investigated across
   fixes #4-7. Fixed with `CoordGroup` (new, QP-less, TCP-only).
9. A late scare: after ~10 rapid restart cycles during this session's
   testing, prefill throughput crashed (130-390 tok/s → ~130-165 tok/s,
   asymmetric GPU power ~7W vs ~20W). Diagnosed as accumulated Thunderbolt
   RDMA hardware/OS-level degradation from repeated teardown (previously
   documented pattern: "reboot fixes it, kill/restart wedges it again").
   **Not a code regression** — confirmed by full reboot of both nodes,
   after which prefill throughput recovered to 334-390 tok/s across
   100K/300K/500K context, matching the pre-session 339 tok/s baseline.

## 2. Why RDMA subgroups (`split()`) can't work under TP: the QP budget

**The hard constraint:** Apple's Thunderbolt RDMA HCA reports `max_qp=3`
(verified live via `ibv_devinfo -v` on both nodes: `max_qp:3, max_cq:6,
max_pd:6`). **This is not documented anywhere by Apple** as far as this
session found (checked TN3205 "Low-latency communication with RDMA over
Thunderbolt" and the Apple developer forums) — it's an empirical value read
off the driver on this specific hardware, not a published spec number.

**Scope: per HCA (per physical cable), not system-wide.** Verified by
reading `create_connections()` in `rdma.cpp`: it opens one `ibv_context` per
device name, and `connections_`/`ack_connections_`/`pool_connections_` all
borrow that *same* context (`owns_ctx=false`). So the 3-QP ceiling applies
per physical Thunderbolt cable/port, not once across the whole machine. This
cluster has two RDMA-capable cables (`rdma_en3` + `rdma_en4`, both
independently `PORT_ACTIVE`, each independently reporting `max_qp=3`) — two
independent 3-QP budgets, not one 3-QP budget shared between them.

**Why the top-level group already uses all 3, under TP:**
- `connections_` — the data QP, raw tensor payloads for `all_sum`/`all_gather`.
- `ack_connections_` — dedicated QP for the ARQ-completion ACK exchange
  (`ack_sync_pre`/`ack_sync_post`), fired **twice per collective**, on
  every one of 43 model layers, every prefill chunk, every decode step.
- `pool_connections_` — standing pre-armed recv pool for jaccl-v2's
  reliable-optimistic fast path on that same hot collective.

PP mode uses a disjoint set (`connections_` + `p2p_retry_connections_` +
`ack_connections_`) — see `jaccl_pipeline_mode_enabled()`'s comment in
`mesh.cpp` for the mode-gating logic (fixed 2026-08-10, commit `49b316d5d`).

**The actual bug (#8 above):** `MeshGroup::split()` — used by exo's
`get_coord_group()` to build an isolated "coord subgroup" for control-plane
collectives (warmup sync, per-decode-step task agreement) — tries to
allocate its *own* `connections_` + `ack_connections_` (2 more QPs) *on the
same device context the top-level group already fully occupies*. Zero
budget left. `split()` threw `RuntimeError("Couldn't create queue pair")`
100% of the time under TP, and `get_coord_group()`'s bare
`except RuntimeError: sub = group` silently swallowed it — every
"coord subgroup" collective was actually running on the shared top-level
group the entire time, colliding call_ids with live model traffic.

## 3. The fix: `CoordGroup`, a QP-less TCP-only collective group

`mlx/distributed/jaccl/lib/jaccl/coord_group.h` (new, commit `fa69bafd1`):
implements the full `jaccl::Group` collective interface
(`all_sum`/`all_max`/`all_min`/`all_gather`/`barrier`) on one dedicated,
reliable, framed TCP socket — **zero ibverbs calls, ever**, so it can never
hit the QP ceiling regardless of sharding mode or what the parent already
allocated. Properties:

- Own `next_call_id_` namespace, carried in-band and cross-checked
  (magic/opcode/call_id/n_bytes) on every op — a desync is a loud throw,
  not a silent mispair.
- Own socket — can't interleave with the parent's `side_channel_` framed
  stream (same corruption class `p2p_channel_` was split out to fix
  2026-07-17).
- Borrows nothing from the parent (no `ibv_context`, no PD/CQ/QP/MRs), so
  creating one does **not** set `has_split_` — the parent keeps full
  `reconnect_fresh()` capability, the only recovery that actually clears a
  dead-UC wedge. The RDMA `split()` path permanently forfeited that.
- `send()`/`recv()` and payloads over 1 MiB throw explicitly — this is a
  control plane, not a bulk-data path.

`MeshGroup::split_tcp_coord(color)` (new) builds one: reserves an ephemeral
port on rank 0, publishes it over the parent's already-locked
`side_channel_`, and returns a `CoordGroup`. Wired through the full stack
(`GroupImpl` base → `JACCLGroup` → public `Group` API → Python binding
`mx.distributed.Group.split_tcp_coord`). `get_coord_group()` (exo,
`utils_mlx.py`) now calls this instead of `split()`, and the silent
`except RuntimeError: sub = group` fallback was deleted for any error other
than "this backend has no TCP coord mechanism at all" (the ring/TCP-only
backend's genuine, intentional case).

**Verified:** compiles clean; a standalone loopback test
(`mlx/distributed/jaccl/lib/examples/coord_group_loopback_test.cpp`, forks
two ranks over `127.0.0.1`, no RDMA hardware needed) built and ran, passing
all_sum/all_max/all_min/all_gather/barrier + the desync tripwire.

## 4. Dual-cable network topology split (separate, additive)

User-directed architectural change, independent of the bug fixes above:
route jaccl's TCP coordination traffic (`side_channel_`, `coord_channel_`,
`p2p_channel_`) onto a **dedicated** physical Thunderbolt cable, keeping the
other cable **RDMA-only** — instead of both cables bonded together and TCP
coordination traffic riding whatever IP the general priority table picked
(which, before this fix, was the shared home LAN).

Took 3 iterations to get right (`src/exo/master/placement_utils.py`), each
closing a real gap found via live evidence, not guessed:

1. **`f9d5a1a89`** — `_select_rdma_cable()` becomes the single source of
   truth for which physical cable RDMA claimed; `get_mlx_jaccl_coordinators`
   excludes that cable's interface from TCP coordinator resolution.
   *Correct by exclusion, but the existing `ring=False` priority table
   still ranked general ethernet/LAN above Thunderbolt, so the coordinator
   kept landing on the home LAN anyway* (verified live: resolved to
   `192.168.86.201`, not a Thunderbolt IP).
2. **`20042552a`** — added `prefer_thunderbolt` to `find_ip_prioritised`,
   used only by jaccl's coordinator path (ring/prefill callers untouched,
   default off): once the RDMA cable is excluded, any *remaining* direct
   Thunderbolt cable outranks the general LAN. Also had to extend the RDMA
   exclusion to `SocketConnection` edges too (by resolving reserved
   interface names to IPs) — a cable that's both RDMA-capable and
   IP-bridged shows up as both edge types, and excluding only the RDMA edge
   let it back in via its socket edge once TB became preferred.
3. **`792138e15`** — verified live the coordinator was landing on
   `169.254.115.69`, a macOS self-assigned link-local (APIPA) address on
   `en8`, a genuinely unrelated USB "Ethernet Adapter" device tied at the
   same `maybe_ethernet` priority rank as the real TB cables (macOS
   re-tags every `enX` other than en0/en1 as `maybe_ethernet` — a real gotcha
   documented in `_get_interface_types_from_networksetup`). Added
   `is_link_local_ipv4()` as the **primary** sort key (link-locality before
   interface type) so a dead-but-fast-looking interface never outranks a
   live-but-slower one, with a loud WARNING if the eventual winner is still
   link-local (meaning every candidate is APIPA — real signal, not
   silenced).

**Verified live, final state:** jaccl coordinator resolves to
`192.168.200.1` (the real, dedicated non-RDMA Thunderbolt cable's static
IP), zero fault signatures across a full deploy, correct inference output.

## 5. Standing architectural principle

User-directed, endorsed with caveats by a consulted second opinion:
**RDMA reserved for genuinely latency-critical hot-path traffic; TCP for
anything where the added latency is negligible relative to the operation
it sits inside.**

Real measured cost on this hardware (prior session, warm memory fact 854):
RDMA collective activation ~0.8ms vs. TCP-over-Thunderbolt-bridge ~2-3ms,
plus ~30-50% higher CPU overhead on the TCP path.

- **Fits the TCP profile, now moved:** one-time warmup handshakes,
  per-decode-step task agreement (`mx_any` — once per *step*, not once per
  *layer*). Small payloads (a few hundred bytes at most), infrequent
  relative to the per-layer hot path.
- **Does NOT fit the TCP profile, stays on RDMA:** `ack_connections_` /
  `pool_connections_` — these fire on every layer's collective, every
  prefill chunk, every decode step (the actual hot path, not overhead
  riding alongside it).

**Collapsing the 3 hot-path QPs into fewer was considered and rejected**
(2026-08-21, see the `mesh.cpp` comment in section 2 for the full reasoning
and the corruption mechanism from the 2026-07-17 pool/data-QP merge that
motivates keeping them separate). Two lower-risk levers exist if more QP
headroom is ever needed instead: (a) the second RDMA-capable cable's entire
3-QP budget currently sits unused (not implemented — flagged for whoever
needs it), (b) moving more genuinely latency-tolerant traffic to
`CoordGroup`-style TCP, following the pattern in section 3.

## 6. This is fork-only work

None of the fixes in this document exist upstream on `ml-explore/mlx`.
`git log upstream/main -- mlx/distributed/jaccl/lib/jaccl/mesh.cpp` shows
only 2 commits total (the original jaccl port + one barrier addition) vs.
28+ commits of QP-budget/ACK-QP/pool-QP/CoordGroup/reconnect hardening on
`adurham/mlx`. This is custom engineering for this cluster's specific
2-node TP=2 Mac Studio setup and the edge cases it hits (the QP ceiling,
dual-cable topology, the specific UC-drop/corruption patterns fixed above)
— not something to compare against "how everyone else does it," since as
far as this fork's history shows, nobody else has published equivalent
work.
