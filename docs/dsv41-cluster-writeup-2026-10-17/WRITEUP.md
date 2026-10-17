# DeepSeek-V4.1-Flash on 2x Mac Studio M4 Max -- how this cluster works

_Final edit 2026-10-17. Assembled from the cluster's own lead sections (§0, 1, 2, 9, 10, 11,
written live on the cluster) and the section drafts (§3-8). Measured numbers trace to the two evidence packs (`evidence.md`, `evidence-live.md`); §3, §4,
§5.1, §5.3 and §6.2 additionally cite the fork's own dated docs and commits (named inline); §8
and §12 draw on the live captures they name. Private identifiers are scrubbed to neutral
placeholders. Independently fact-checked against the evidence packs and copy-edited on
2026-10-17; the review record and every applied fix are in `REVIEW-AND-FIXLOG-2026-10-17.md`._

---

## 0. TL;DR

Two Apple Mac Studio M4 Max (128 GB unified memory each, 256 GB total) linked by a
Thunderbolt 5 cable pair carrying RDMA, running tensor-parallel degree 2, serve a
552B-parameter DeepSeek-V4.1-Flash MoE (16B active) quantized to EXL3 at 2.9 bits per weight
-- 210.68 GB on disk, 108.76 GB per rank -- at a 1,048,576-token context.

None of this is a supported configuration. Apple's Thunderbolt RDMA HCA exposes Unreliable
Connected queue pairs only, no hardware RC, and a `max_qp` of 3 per device (observed, undocumented); upstream exo has
no Thunderbolt path at all. Reliability, loss detection and recovery are ours to build, and
the three forks carry 2,607 / 277 / 527 commits of divergence from upstream to do it.

It works. A 1,000,003-token prompt completed end to end: HTTP 200, `finish_reason=length`,
6,760.5 s wall, 104.2 tok/s prefill over the rows it actually had to prefill, 2.92 tok/s
decode (299,008 of the prompt's rows came from cache -- see §9.4). Decode holds a ~30 tok/s plateau to 160K and then falls off a cliff to 6.0 tok/s at
300K. Prefill degrades with position: 245.1 tok/s at 100K, 195.4 at 300K, 64.9 over the deepest
band of the 1M request.

The memory budget does not close -- from -2.3 GB best case to -9.4 GB worst case per node at 1M
-- and it runs anyway, with ~127 GB of the nodes' 137 GB (128 GiB) in use.

Speculative decode is a production default and worth +34.9% at 100K on the two medians below
(36.63 vs 27.15 median tok/s; the baseline doc's paired analysis states +36.71%), 24 paired
runs, 12/12 pairs -- a fixed-window A/B, a different instrument from the depth-run plateau above. A degeneration loop guard and a reasoning-budget limiter are armed live; both
were exercised and both behaved (the guard's retry semantics remain open, §12.6).

### 0.1 Headline numbers

| metric | value | basis |
|---|---|---|
| 1M prompt, end to end | HTTP 200, `prompt_tokens=1,000,003`, `finish_reason=length` | live run, 6,760.5 s wall |
| 1M prefill | 104.2 tok/s | 700,995 uncached rows / 6,725.29 s (partial prefix cache, 299,008 cached) |
| 1M decode | 2.92 tok/s | client and server channels agree |
| decode plateau | 34.5 / 31.4 / 29.4 / 30.7 / 30.4 tok/s at 1K / 10K / 50K / 100K / 160K | client decode window, spec-ON; the 160K point is server-side only (that run used `stream:false` -- client decode not separable) |
| decode cliff | 6.0 tok/s at 300K | 5x falloff inside 1.5 doublings |
| prefill by depth | 255.2 / 275.1 / 257.7 / 245.1 / 242.0 / 195.4 tok/s at 1K / 10K / 50K / 100K / 160K / 300K | uncached, server-measured |
| spec decode gain | +34.9% from the two medians (36.63 vs 27.15 tok/s; the baseline doc's paired analysis states +36.71%), CI [+28.26,+51.02]%, 12/12 pairs | 24 paired runs, 256-tok fixed window, 100K |
| fork divergence | exo 2,607 (2,568 no-merge), mlx 277 (253), mlx-lm 527 (485) commits ahead of upstream | `git rev-list --count` |
| jaccl mesh.cpp | 32 of 33 all-history commits are fork-local | `git log -- mlx/distributed/jaccl/lib/jaccl/mesh.cpp` |
| weights per rank | 108.76 GB measured | shard headers + the sharding rule the running code applies |
| 1M KV per session | 3.2-4.2 GiB | MLA absorbed form, replicated not TP-sharded |
| headroom at 1M | -2.3 GB (best) to -9.4 GB (worst) | 125.83 GB wired limit (120000 MiB) - 108.76 GB weights - 3.4-4.5 GB KV - 16-22 GB ESTIMATE |
| guard / limiter | live fire at committed=269, period=1; limiter cut 600 -> 96 reasoning tokens exactly | HEAD `5a18b81` |

---

## 1. What the system is

### 1.1 Hardware

Two identical Apple Mac Studios, `Mac16,9`, Apple M4 Max, 16 CPU cores (12 performance, 4
efficiency), 128 GB unified memory each -- 256 GB across the pair. Both run macOS 27.0
(build 26A428), Darwin 27.0.0 (`xnu-13432.1.9~1/RELEASE_ARM64_T6041`), system firmware
20457.1.29. `sysctl hw.memsize` reports 137,438,953,472 B = 128 GiB on each.

The GPU wired limit is raised on both nodes: `sysctl iogpu.wired_limit_mb = 120000`. Every
doc in the corpus pins 115000 MB (120.6 GB); the live value is 120000 MB (120000 MiB =
125.83 GB, the corpus's reading of this knob), and any "% of wired limit" figure copied from a
doc is computed against the wrong denominator. We use the live value.

### 1.2 Topology

The two nodes are joined by a Thunderbolt 5 cable pair. Apple exposes the TB RDMA HCA
through ibverbs as two device names per node, `rdma_en3` and `rdma_en4`, both `status:
active`. The pairing is asymmetric by construction: node0 reaches node1 over its own
`rdma_en3`, node1 reaches node0 over its own `rdma_en4`. The launcher writes exactly that
into `p47_ibv.json` (`[[null, "rdma_en3"], ["rdma_en4", null]]`).

```
        MacBook Pro M4 (controller -- not a serving node)
                    |
                    |  ssh; start_cluster.sh rsyncs ~/repos/exo to both nodes
                    |
        +-----------+-----------------------------+
        |                                         |
      node0                                     node1
  Mac16,9 / Apple M4 Max                    Mac16,9 / Apple M4 Max
  16 cores (12P + 4E), 128 GB               16 cores (12P + 4E), 128 GB
  macOS 27.0 (26A428)                       macOS 27.0 (26A428)
  iogpu.wired_limit_mb = 120000             iogpu.wired_limit_mb = 120000
  rdma_en3, rdma_en4 (PORT_ACTIVE)          rdma_en3, rdma_en4 (PORT_ACTIVE)
        |                                         |
        |   Thunderbolt 5 cable pair, RDMA        |
        |   node0 rdma_en3  <-->  node1 rdma_en4  |
        +-----------------------------------------+
             jaccl UC queue pairs, max_qp = 3 per device
             TP = 2, one rank per node, port 52415
```

There is no switch, no NIC, no fabric. The transport is two cables and a software
reliability layer (§3).

### 1.3 Model

`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` -- a 552B-parameter MoE with 16B
active, quantized to EXL3 at 2.9 bits per weight. The checkpoint is 210.68 GB on disk
(`du -sk` = 205,739,672 KiB), 40 safetensors shards, identical on both nodes. Config:
`quant_method: exl3`, version 1.4.2, `codebook: mul1`, `avg_bits: 2.9`, `head_bits: 6`,
`mtp_bits: 4`.

At TP=2 the per-rank weight footprint is 108.76 GB measured: routed experts 98.02 GB
(width-sharded), DSpark stages 3.84 GB (width-sharded), shared expert 0.40 GB
(width-sharded), and 6.50 GB of attention + embed + head + router + vision + other that is
replicated. `DeepseekV4ShardingStrategy` width-shards three things, not one.
(The component sum -- 2x each width-sharded part plus the replicated 6.50 -- is 211.0 GB,
~0.2% apart from the 210.68 GB on disk; du vs component-sum.)

Context is 1,048,576 tokens (`contextLength = maxKvTokens = 1048576`). V4.1 uses the MLA
absorbed form -- one shared 512-d KV latent per position, `num_key_value_heads = 1`, and no
`kv_lora_rank` key in the config -- so a 1M session costs 3.2-4.2 GiB per node (§5.2). The cache is
replicated, not TP-sharded.

### 1.4 Software stack

Three downstream forks, all under `~/repos/`: exo (fork of exo-explore/exo), mlx (fork of
ml-explore/mlx), mlx-lm (fork of ml-explore/mlx-lm). Deployed exo HEAD through the capture window: `5a18b81` (`deploy/next28`) for the
overnight runs, then `991cf11` (the §12 liveness fix) from its 10:05-10:12 deploy onward;
mlx `b177818`; mlx-lm `676f803`. The mlx and mlx-lm SHAs are
pinned in `uv.lock` and match the submodule gitlinks, an invariant CI enforces
(`src/exo/shared/tests/test_submodule_lock_pin_alignment.py`).

Live versions on the serving node: `exo 0.3.70`, `mlx 0.32.5.dev20261016+b177818`,
`mlx-lm 20261017.dev46+g676f803ed`. The mlx import-time version string differs from the
uv.lock declaration (`0.32.3.dev20261002+b177818`) but the git SHA suffix is the same
commit (§11.2).

Deploy is `start_cluster.sh` run from the controller, which rsyncs the repo to both nodes. The
API listens on port 52415. The instance reports `worldSize=2`, `maxPrefixSessions=4`,
`maxPrefixBytes=12884901888`, `reasoningDialect=tool_conditional`, capabilities
`[text, thinking, thinking_toggle, vision]`.

---

## 2. Why this is bespoke engineering, not a config

Nothing in the upstream projects knows about this hardware. exo upstream has no Thunderbolt
RDMA path; MLX upstream has no jaccl transport; mlx-lm upstream has no EXL3 trellis reader.
Every layer that makes the cluster work is fork-local, and the divergence is large enough
that "just pull upstream" is not an option.

### 2.1 Fork divergence

Counted with `git rev-list --count <upstream-ref>..HEAD` at capture:

| repo | branch | HEAD | upstream | commits ahead | ahead, --no-merges |
|---|---|---|---|---|---|
| exo | deploy/next28 | `5a18b81` | exo-explore/exo `bc412e2` | 2,607 | 2,568 |
| mlx | integrate/upstream-20261016 | `b177818` | ml-explore/mlx `b6642e6` | 277 | 253 |
| mlx-lm | main | `676f803` | ml-explore/mlx-lm `93e6d4a` | 527 | 485 |

The concentration is the point. Of 33 all-history commits touching
`mlx/distributed/jaccl/lib/jaccl/mesh.cpp`, **32 are in the fork divergence** -- the
transport is essentially ours end to end. (The file lives at the upstream-refactored
`lib/jaccl` layout; a path lookup for a top-level `mlx/jaccl/mesh.cpp` returns 0 commits and
silently understates the work.)

The forks are not a thin patch layer. exo's divergence includes the DSv4.1 engine, the
sharding strategy, the launcher, the graceful-shutdown helper and the reboot script. mlx's
includes the jaccl UC reliability layer, StallWatch, `reconnect_fresh`, the scheduler
exception capture and an IOGPU residency-set abort bypass. mlx-lm's includes the EXL3
kernel set, loader and checkpoint reader, plus the spec-rollback ring fix.

### 2.2 Thunderbolt RDMA is the reason

Apple's TB RDMA HCA presents RoCE-like verbs -- an `ibv_context` per device, queue pairs,
completion queues, memory regions -- but it is not a NIC. Three properties force the design:

- **No hardware Reliable Connected queue pairs.** jaccl hardcodes `IBV_QPT_UC` (Unreliable
  Connected) and builds reliability in software. UC is connection-oriented but has no
  hardware retransmit and no hardware acknowledgement: a lost completion is simply gone,
  and the only signal is a poll loop that stops progressing. Delivery, ordering, loss
  detection and recovery are all ours.
- **No Unreliable Datagram path either.** UC only, recorded from the HCA probe.
- **`max_qp = 3` per device.** Read live with `ibv_devinfo -v` (`max_qp:3, max_cq:6,
  max_pd:6`). It is not in Apple's published documentation, so treat it as true for the
  HCAs we tested, not as a spec constant. The scope is per HCA, per physical cable, not
  system-wide -- which is why the second cable is a second, independent 3-QP budget rather
  than a bandwidth-only addition.

Those three facts are the whole reason Sections 3 and 4 exist. A normal cluster gets
reliability, loss detection and recovery from the NIC; here they are application code.

### 2.3 What that costs in practice

- QP construction had to become mode-gated so either sharding mode allocates exactly three
  QPs per peer and fits one HCA's budget (commit `49b316d`). A QP-less, TCP-only
  `CoordGroup` (commit `fa69baf`) exists purely so control-plane collectives cannot hit
  the ceiling in any mode.
- A lost UC completion needed a detector (`StallWatch`, 8 s timeout) and a hard recovery
  (`reconnect_fresh`, ~0.15 s, gated by `MLX_JACCL_RECONNECT_FRESH=1`) because
  `queue_pair_reset()` does not clear a dead-UC-data-path wedge.
- Any RDMA queue teardown can drop the Thunderbolt XDomain link with no recovery short of a
  physical reboot, so reboot became a first-class operation and SIGKILL was banned on the
  data path.
- A jaccl fault used to unwind out of the MLX stream worker thread into `std::terminate` ->
  `SIGABRT`, killing both ranks. Capturing it into a per-`StreamThread` `std::exception_ptr`
  and rethrowing on the Python-facing thread (commit `407a53c`) turned a cluster-killer
  into one failed request.

### 2.4 Platform asymmetry worth knowing

The darwin mlx pin is our fork (`github.com/<user>/mlx`, `sys_platform == 'darwin'`; §11.2).
On Linux, mlx comes from third-party prebuilt wheels on `rltakashige/mlx-jaccl-fix-small-recv`
(CUDA 12/13 variants). So "mlx is our fork" is true on the platform this cluster runs and
not true on Linux.

The EXL3 format is a second bespoke axis: it is a trellis quantizer -- a codebook walk plus
a small matmul, not an affine dequant -- and nothing upstream understood it. The reader,
kernels and loader were built in the mlx-lm fork, including an additive `silu_clamp`
(`EXL3_MOE_CLAMP=10.0`) without which layer 39 diverges (RMS 150.5 vs 34.6).

---

## 3. The transport

Two Mac Studios M4 Max, one Thunderbolt 5 cable pair, tensor-parallel degree 2. The whole
stack moves its per-layer collectives (`all_sum`, `all_gather`) across a Thunderbolt-attached
HCA that Apple exposes through the ibverbs API. At the verbs layer it looks like any other
RoCE device: an `ibv_context` per device name, queue pairs, completion queues, memory
regions. What it does not give you is the queue-pair type everyone builds a reliable
transport on -- no hardware RC, no UD, UC only (§2.2). jaccl, the RDMA backend inside our MLX
fork, hardcodes `IBV_QPT_UC` and provides reliability over UC in software for the TP
collectives (mlx `mesh.cpp:232-234`; the connection setup in `rdma.cpp` fixes the QP type).
UC is connection-oriented but has no hardware retransmit and no hardware acknowledgement: a
send or receive whose completion never arrives is simply gone, and the only signal is a poll
loop that stops making progress. So delivery, ordering, loss detection and recovery -- all of
which a normal cluster gets from the NIC -- are ours to build. The four subsections below are
the four places that obligation actually bit us.

### 3.1 The max_qp=3 ceiling

**Symptom.** A fresh runner booted into a crash loop at jaccl init, throwing "Couldn't create
queue pair" and retrying forever. The retry backoff (`utils_mlx.py::_init_jaccl_with_backoff`)
had been written for a genuinely transient cause -- leaked QPs from a crashed runner -- so it
never gave up, even though the real cause could never clear.

**How we root-caused it.** The ceiling is a hardware/driver value read live off the driver
(§2.2; mlx `mesh.cpp:20-26`; `docs/dual-cable-topology-and-qp-budget-2026-08-21.md:61-66`).
The scope is **per HCA, per physical cable, not system-wide**: `create_connections()` opens
one `ibv_context` per device name, and `connections_`, `ack_connections_` and
`pool_connections_` all borrow that same context (`owns_ctx=false`), so all three QP types
live on one cable's 3-QP budget (`mesh.cpp:28-44`; `dual-cable...:68-75`). The top-level
`MeshGroup` ctor had grown to build four dedicated QPs per peer -- `connections_` (data),
`ack_connections_`, `pool_connections_`, plus a `p2p_retry_connections_` added for the
TCP-to-RDMA p2p migration -- so the fourth `ibv_create_qp` always failed `EBUSY`,
structurally, not transiently (`mesh.cpp:46-53`).

A second, nastier face of the same ceiling: `MeshGroup::split()`, which `get_coord_group()`
used to build an isolated subgroup for control-plane collectives, tried to allocate its *own*
two QPs on a context the top-level group had already fully occupied. It threw
`RuntimeError("Couldn't create queue pair")` 100% of the time under TP, and a bare
`except RuntimeError: sub = group` swallowed it -- so every "coord subgroup" collective
silently ran on the shared top-level group, colliding `call_id`s with live model traffic.
That collision was the deterministic warmup stall we later chased across several fixes
(`dual-cable...:89-98`).

**The fix.** Stop allocating QPs the running mode will never use. QP construction is now
mode-gated: `jaccl_pipeline_mode_enabled()` reads `MLX_JACCL_SHARDING_MODE` (set by
`start_cluster.sh` from `DSV4_SHARDING`) and builds a disjoint third QP per mode -- PP gets
`p2p_retry_connections_`, TP gets `pool_connections_` -- so either mode allocates exactly
three QPs per peer and fits one HCA's budget (commit `49b316d`, `mesh.cpp:135-146`). The
coord-subgroup case got a different answer: `CoordGroup` (`coord_group.h`), a QP-less,
TCP-only collective group that makes zero ibverbs calls and therefore cannot hit the ceiling
in any mode (commit `fa69baf`). Merging the three hot-path QPs into fewer was considered
and rejected: `ack_connections_` fires twice per collective on all 43 layers (the era's count; §4.2), on every
prefill chunk and every decode step, and an earlier attempt to merge `pool_connections_` into
`connections_` produced real corruption, because the pool's uniform recv-buffer size class
and the data path's per-message sizes collide at the ibverbs level below any application
framing (`mesh.cpp:74-100`).

**What it cost.** The ceiling is not something we can raise on this path; the honest framing
is that the cluster has two independent 3-QP budgets, one per cable. The current topology
opens the TP data QP on one cable and dedicates the other to jaccl's TCP coordinator traffic,
which means the second cable's entire 3-QP budget sits unused. If TP ever needs a fourth
RDMA-native QP, the documented route is to open it against the second device's context -- but
that is flagged, not implemented (`mesh.cpp:33-44`; `dual-cable...:204-206`). So today the
transport runs three QPs per peer and we design around that.

### 3.2 XDomain link drops on any RDMA queue teardown

**Symptom.** After an RDMA queue teardown the Thunderbolt link could fail to re-enumerate and
would not recover without a physical reboot. `start_cluster.sh` names it directly: a teardown
leaves the TB link in a state the script can only repair by rebooting, and a related route
fault "does NOT fix it (macOS recreates the same broken route at boot)" on a warm reboot
(`start_cluster.sh:1210-1213`, `:1233-1238`). The failure mode is recorded historically as
the "any RDMA teardown drops the XDomain link" pattern
(`docs/phase3-cluster-validation-blocked-resolved-2026-08-20.md:24-25`).

**How we root-caused it.** Partially. What we have on record is a post-hoc teardown timeline,
not a captured kernel trace. The recorded observation is that the AppleThunderboltRDMA device
tore down at 20:23:22 after a runner was SIGKILLed, the TB link dropped, and all ports
reported "No device connected"; the fix applied was a physical reboot of both Studios, after
which the link came back healthy (`docs/PERFORMANCE_HISTORY.md:4110-4112`;
`docs/dspark-mtp-ab-preregister-2026-08-25.md:543-546`). The one mechanism we can cite from
the repo is that forcing the interface to a non-default MTU (8000) breaks Thunderbolt
XDomain link re-enumeration after an RDMA queue teardown (`start_cluster.sh:1210-1213`):

```
# ... forcing en3 to a non-default MTU (8000), which breaks
# Thunderbolt XDomain link RE-ENUMERATION after an RDMA queue teardown -- the
# link then can't recover without a reboot.
```

A verbatim `/usr/bin/log` kernel excerpt was never captured in the artifacts (still owed on
the next occurrence), so the exact kernel-level mechanism is [UNVERIFIED]; what is
verified is the observed teardown and the "no recovery short of reboot" operational fact. The
standing lesson is conservative: never bounce the TB interface or its MTU on the boot path,
because that renegotiation is the thing that can land the PHY in the link-dead state
(`start_cluster.sh:1233-1240`).

**The fix / what it cost.** We did not find a fix; we found a rule. No recovery short of a
physical reboot was found, so the operational posture became "reboot is a first-class
operation, not an escalation." That decision propagates straight into §8 -- it is the
reason there is a FileVault-safe unattended reboot script at all, and the reason SIGKILL is
banned on the data path (§3.3, §3.4). A separate, documented variation is that a warm reboot
does not clear the *route* corruption while a reinstalled connected route does; the launcher
repairs the route passively and never touches the PHY (`start_cluster.sh:1219-1246`).

### 3.3 SIGKILL leaks QPs and wedges the link; SIGTERM-only

**Symptom.** A runner killed with `SIGKILL` left live RDMA queue pairs on the Thunderbolt NIC.
Those accumulate and wedge the Apple TB stack to "No device connected", whose documented
escape hatch is a node reboot.

**How we root-caused it.** `SIGKILL` skips the C++ static-duration destructors that call
`destroy_qp`/`dealloc_pd`/`close_device`, so the QPs are never freed; a normal interpreter
exit runs them and releases the QPs (`scripts/exo_graceful_shutdown.sh:132-134`;
`start_cluster.sh:1510-1517`). This was not a rare event, because of a compounding bug: the
launcher's liveness probe `ssh "$NODE" "pgrep -f 'python.*exo'"` **matched its own command
line** whenever `$NODE` was the node the deploy ran from -- the ssh client's argv contains
the literal text `python.*exo`, and the ERE matches it -- so the probe reported exo alive
even when it was dead, the 15 s graceful window could never succeed on the coordinator, and
**every** deploy driven from a Studio escalated to `pkill -9`, "silently spending the
RDMA-cleanup guarantee on every single run"
(`docs/incidents/pkill-self-match-forces-sigkill-2026-09-10.md:41-59`;
`start_cluster.sh:1519-1531`). Reproduced 4/4 on the coordinator, 0/4 on the secondary.

**The fix.** Shutdown moved entirely into a self-match-immune, pattern-complete helper,
`scripts/exo_graceful_shutdown.sh`, delivered over `ssh "$NODE" bash -s` stdin so that no
pattern text appears in any argv on either end. It uses the bracket trick (`[p]ython.*exo`),
excludes its own pid and parent, and checks all three things the kill sweep targets --
`python.*exo`, `exo.main` and the `52415,52416` port holders (the API and the file server) -- because the old probe checked
only the first, so "gone" was never actually proven (`exo_graceful_shutdown.sh:49-67`).
`start_cluster.sh` now refuses to fall back to the inline form (`start_cluster.sh:1542-1548`)
and reads a verdict line (`CLEAN_EXIT` / `SIGKILLED` / `SURVIVED_SIGKILL`) off the helper's
output (`start_cluster.sh:1553-1566`).

**What it cost.** Nothing measurable, and it paid for itself immediately: on the first deploy
after the change both nodes reported `CLEAN_EXIT after 1s (NO SIGKILL USED)`, RDMA ports
stayed `PORT_ACTIVE`, the TB link stayed at 0% loss, and wired+compressor drained to 3.6 /
3.8 GB (`pkill-self-match...:105-107`). The rule is now absolute: SIGTERM and wait; SIGKILL
only as a last resort with a loud warning that the node may now need a reboot.

### 3.4 UC completion loss: StallWatch and reconnect_fresh

**Symptom.** Under `c>=2` the two ranks can lead each other by hundreds to thousands of
collectives; when that lead overruns the ACK recv pool, or the NIC's completion ring wedges,
a posted ACK or data work completion is silently lost. The owning collective's poll loop then
spins on a counter that will never reach zero, hanging the runner until the supervisor
SIGKILLs it -- 45 s of supervisor silence, then a 20 s growth-check interval, so a genuinely
wedged round dies at ~65 s -- and force-replaces the whole instance (mlx `mesh_impl.h:136-152`;
`supervisor.py`).

**How we root-caused it.** UC gives no loss signal, so the only observable is the absence of
progress. We turned that into the detector. **StallWatch** is a struct in `mesh_impl.h` that
watches a collective poll loop for a permanently-stuck in-flight counter -- "a lost UC
completion." Its `tick()` is called once per poll iteration with the loop's current progress
metric; any change resets the deadline, and no change for `jaccl_stall_timeout_us()` throws
(`mesh_impl.h:437-470`). The default timeout is `MLX_JACCL_STALL_TIMEOUT_US = 8,000,000`
(8 s), chosen because normal collectives complete well under a second, giving a >8x margin
over the slowest healthy collective observed; 0 disables it back to the legacy
hang-until-SIGKILL behavior. StallWatch landed in `a5be440` (2026-07-04) and was extended
to the send/recv drain loops by `67f22a1` (2026-08-10). The throw is deliberately a plain
`std::runtime_error` -- which only works because of the scheduler fix in §4, where the
same "throw from the worker thread" path used to be `std::terminate`.

**The recovery: reconnect_fresh.** `queue_pair_reset()` -- which preserves the PD/CQ/MRs/context
-- does *not* clear a dead-UC-data-path wedge; a stuck send whose completion never arrives
survives it. **reconnect_fresh()** is the hard recovery: it closes and reopens the ibv device
contexts and rebuilds everything device-side (PD/CQ/QP, buffer MR registrations, the
MeshImpl/RingImpl views), re-exchanging over the surviving TCP side channel. It is the
in-process equivalent of a runner respawn, and per the source it is "the only recovery that
ever cleared the dead-UC-path wedge -- at ~0.15 s instead of a re-place" (mlx `mesh.h:153-164`;
`mesh.cpp` ctor's `reconnect_fresh` comment; `docs/dsv4-c2-serving-handoff-2026-07-06.md:362-368`).
It is gated by `MLX_JACCL_RECONNECT_FRESH=1` inside `reconnect()`. Commit `e399ecf`
(2026-07-06) introduced it; `5d2c424` added the lock coverage it needed (it had been running
with no lock while a concurrent collective could touch the group being torn down -- a
use-after-free segfault), `b4871e8` gave it a real subgroup teardown/rebuild path instead of
silently degrading to the QP-only reset, and `fa69baf`'s QP-less CoordGroup exists partly so
the parent keeps full `reconnect_fresh` capability -- the RDMA `split()` path permanently
forfeited it (mlx `mesh.h:116-121`; `dual-cable...:15-49,108-130`).

**What it cost.** The failure became recoverable in seconds instead of a ~90 s instance
re-place or a reboot. The trade is a detection latency: an 8 s stall timeout per incident,
plus one `mach_absolute_time()` per poll iteration, which the source notes is negligible
beside the `ibv_poll_cq` the loop already performs. It also changes the failure semantics of
everything above it: a stalled collective is now a thrown, catchable error, which is exactly
the contract Section 4 had to make real.

---

## 4. Surviving failures without rebooting

The difference between a demo and a daily driver here was never throughput. It was that three
failure modes that used to take the whole cluster down or corrupt it -- a transport blip that
aborted both ranks, memory leaks that climbed until jetsam killed the runner, and a silent
output-corruption class -- were root-caused, turning the first two into per-instance restarts
with the 108 GB model still resident.

### 4.1 From std::terminate to a catchable per-instance restart

**Mechanism.** A jaccl RDMA collective fault (`[jaccl] all_reduce wc.status=N`) is thrown by
the mesh poll loop from inside a lambda dispatched to the CPU command encoder, which runs on
the MLX stream worker thread. The old scheduler caught the exception only to log-and-re-throw
it *on the worker thread*, which unwinds out of `thread_fn` -> `std::thread` ->
`std::terminate()` -> `SIGABRT`. A single Thunderbolt/RDMA blip therefore aborted the runner
on **both** ranks and took the whole cluster down unrecoverably, with no catchable error ever
reaching Python (mlx commit `407a53c`, whose message states exactly this). A diagnostic
precursor (`b0444ff`) had already confirmed the shape: a silent `SIGABRT` every ~10-12
minutes of decode with only Python's faulthandler stack showing `mx.async_eval` as the crash
site.

**The fix.** Capture the worker-thread exception into a per-`StreamThread`
`std::exception_ptr`, keep the worker alive, and rethrow it on the calling (Python-facing)
thread at the next synchronization point: `throw_if_stream_exception` in `scheduler.h`,
rethrow in `schedule::synchronize(Stream)`, and rethrow at both `eval()` completion paths in
`transforms.cpp` (commit `407a53c`). The fault then surfaces as a normal catchable exception
at the Python `step()` boundary, where exo's runner `try/except` converts it into a
`RunnerTerminationError` and the supervisor restarts just that instance -- "a transport blip
becomes one failed request, not a dead cluster." The exo side is already shaped for this: the
runner's step guard recognizes jaccl fault markers and, where recoverable, resets the QPs and
resumes serving with the 108.76 GB model **resident** instead of a full ~90 s re-place
(`src/exo/worker/runner/runner.py:794-821`); the non-recoverable path raises through bootstrap
into the supervisor's `RunnerTerminationError` handling (`bootstrap.py`, `supervisor.py`). A
follow-on fix, `779c3ad` with `CombinedEvalFailure` (derived from `std::runtime_error` so
existing handlers keep working), closed a case where upstream cleanup could swallow the
captured fault (`mlx/docs/incidents/fe92a0565-jaccl-fault-shadowing-2026-09-10.md`). A
distinct `SIGABRT` on the same hardware needed its own remedy: Apple's IOGPU
`MTLResidencySet::addAllocation` calls `abort()` inside the kernel stack, killing the process
~10-15 min into long-context inference, bypassed by a fork patch (`<user>/mlx@f8bac64`).

### 4.2 The four memory-leak sites

These are the DSv4 KV/prefix-cache leaks that made long sessions climb until macOS jetsam
killed the runner (`docs/dsv4-memory-leaks-2026-06-27.md`). Each was a redundant deepcopy or a
retained intermediate of the full per-layer cache:

| site | symptom | fix commit |
|---|---|---|
| `KVPrefixCache.update_kv_cache` metadata-refresh path (`cache.py:663-672`) | fires on the common per-turn case (`new_length == old_depth`) yet deepcopied all 44 non-sliceable layer caches every turn; with no fix the pattern climbed to ~108 GB at 128K (that era's numbers -- see the note below the table) | `d5f6c42` |
| growth path, `new_length > old_depth` (`cache.py:719`) | a real growing session also deepcopied all 44 layers every turn (the 500K single-prefill test never hit repeated growth; a live session does) | `c0149c5` |
| multi-leaf accumulation (`DSV4_MAX_PREFIX_SESSIONS` 4 -> 1; later restored to 4 by `4d09fcd` for live-session eviction, the launcher keeping the ~12 GB 100K worst case on the record) | a streaming session creates a new leaf per request; with cap 4, four full KV deepcopies at 128K = ~109 GB ("the 98% the user hit") | `9182190` |
| `batched_cache` + `captured_prompt_pre_norm` retention (`generate.py:765-767`, `batch_generate.py:1725-1729`) | full merged `B x L x hidden` KV buffer and the MTP capture never explicitly freed; residual climb during sustained reasoning | `c6da30c` |

The counts here, in §3.1 and in §6.2 are the V4-Flash-era engine's (43 hidden layers, 44 cache
entries); this writeup's V4.1 checkpoint runs 40 hidden layers (`num_hidden_layers = 40`), so
absolute cache counts differ while the leak classes and their fixes are the same.

After the four, a fifth accumulator in the same trie -- two never-pruned `CacheSnapshot` sets
holding a full 21-layer `PoolingCache` set each, +21 per turn -- was bounded by `947d7e5`
(`docs/dsv4-memory-leak-resolution-2026-06-29.md`). Separately, a fork-side Metal-allocator
leak (~770 KB/token of IOAccelerator VM regions per decoded token, invisible to
`mx.metal.get_active_memory()`, which killed the runner around 100K decoded tokens) was fixed
by cache-bucket coalescing in `a6afdd8` (`docs/fork-notes.md`). Net effect: active memory
went from climbing to plateau, holding ~77-80 GB flat across sequential requests.

### 4.3 The ring-alias spec-rollback fix

The subtlest of the three, because it corrupted output rather than crashing. `spec.rollback`
reset `cache.offset` and rebuilt the compressor carry but touched no window ring. The ring is
position-addressed -- slot `p % window` -- so a verify forward that writes all `gamma + 1` rows
overwrites the slots of the already-committed `p - window` whenever `gamma - accepted >= 2`.
The next forward then attends over rejected-draft KV; and on a stack with a layer above the
first, the corrupted forward's hidden states also write that layer's compressed/index caches
at committed positions that are never recomputed, so the perturbation is **permanent per
conversation** -- the strongest form of warm != fresh
(`docs/dsv41-spec-rollback-ring-fix-2026-10-14.md:17-40`). The rollback's own docstring had
licensed the omission with a claim that was false for the ring.

The fix (mlx-lm `c4d1982`) makes `spec.snap` take the next forward's row count (`1 + gamma`)
and, per layer, copy the `min(verify_rows, window)` ring rows the write is about to overwrite
plus the `comp_kv`/`index_k` rows the chunk's closed groups will occupy; `rollback` restores
the rejected ring slots and raises `RuntimeError` on a wrong-span snapshot rather than
corrupting silently. Because the tests were single-rank, the deploy gate added a
**rank-consistency** check: **2536 rejected speculative rounds compared rank-0 vs rank-1, 0
cross-rank mismatches, verdict PASS** (`docs/dsv41-next27-ringfix-construct-2026-10-14.md:227-230`).
It shipped live on `deploy/next27-ringfix`. Together, 4.1, 4.2 and 4.3 are what let us leave
the cluster running instead of babysitting it.

---

## 5. Fitting and running the model

The deployed checkpoint is the one described in §1.3. What follows is how it was made to fit
and run: the EXL3 reader, the 1M KV cache, the prefill cliff, and the memory budget.

### 5.1 EXL3 trellis support in the mlx-lm fork

EXL3 is a trellis quantizer: decode is a codebook walk plus a small matmul, not an affine
dequant, and nothing upstream understood the format. We built the reader in four pieces.

The kernels: the PonyExl3 Metal kernel set vendored as `mlx_lm.models.exl3` -- the minimal
import closure of `EXL3SwitchGLU` and `EXL3Linear` (12 modules plus 4 numpy references;
commit `9ea86f9`, 2026-09-28). The vendoring changed the numerics in one load-bearing way: an
additive `silu_clamp` activation (`EXL3_MOE_CLAMP=10.0`) implementing V4.1's clamped SwiGLU,
without which layer 39 diverges (RMS 150.5 vs 34.6); two ALU optimizations came on by default
(the EXL3 kernels vs the production MXFP4 kernels: decode R=1 at 1.909x, verify R=4 at 1.979x). The loader (commit
`6391efc`): `mlx_lm/models/exl3/loader.py` builds the stacked buffers the kernels consume and
carries the TP slice -- gate/up OUT-tiles plus matching down K-tiles, `gu_svh`/`dn_suh`
replicated -- agreeing to cos 0.9999999 (the two ranks' partials sum to the full result).
`Exl3Checkpoint` is a pread shard reader (LRU of 12 open shards). The format contract was
verified against the real file: trellis int16, `k = packed*16//256`, `.mul1` int32, signs
float16. The layer builder on top holds per-layer cosine >= 0.99983 (`bd1bfd1`). We pin mlx-lm
at `676f803`.

### 5.2 The 1M KV cache

V4.1 uses the MLA absorbed form -- one shared 512-d KV latent per position, not per-head K/V
(`num_key_value_heads = 1`; the config carries no `kv_lora_rank` key -- the absorbed
form's latent width is fixed in the engine; `deepseek_v41/config.py:34`, `cache.py:80`). The live instance
reports `contextLength = maxKvTokens = 1048576`, and the cache is replicated, not TP-sharded
(`docs/dsv41-restart-rankfix-buildnote-2026-10-11.md:55-65`). One 1M session per node: window
ring 5.0 MiB, compressed KV 2.50 GiB, index keys 1.625 GiB = 4.13 GiB (the top of the 3.2-4.2 GiB range), counting `index_k` on
all 8 `index_source_layer_ids`. Count it on only the 4 kv-source layers and you get 3.2 GiB,
the figure in the code's own comment (the components sum to 3.31 GiB; the comment rounds to
3.2; `mlx_lm/models/deepseek_v41/cache.py:29`). Same cache, different bookkeeping -- state
the rule. Storage is bf16 even though the values are pre-quantized: bf16's 8 mantissa bits hold their
<=6-significant-bit values exactly, the code's own rationale.

### 5.3 The prefill cliff

The cliff predates this model. A growing V4-Flash conversation prefilled at ~270 t/s at 100K,
declined gently to ~168 t/s at 340K, then fell off a ledge to ~40-48 t/s past 340K -- not a
steady slowdown but periodic 8-32 s stalls on individual 128-token chunks, alternating with
fast ones (PREFILL_CLIFF_HANDOFF.md:48-51). On the span graph, `attn.indexer` per-call average
stepped 4532 -> 5362 us (+18%) from a 300K to a 360K prefill while every non-scaling op stayed
flat, its max spiking to ~22 ms (PREFILL_CLIFF_HANDOFF.md:78-82). The mechanism
(`docs/prefill-cliff-mechanism-2026-08-24.md:1-24`) is the `active_memory > threshold` family:
the MLX allocator gc-release (allocator.cpp:149-151) and the fork's eval-driver memory-branch
throttle (transforms.cpp:285-299) both ride the forward pass's registered command-buffer bytes
crossing the threshold at ~340K. This is the best standing explanation, not live-confirmed
(`PERFORMANCE_HISTORY.md:2943-2953`; §10.1). The proximate fix was raising `MLX_MAX_MB_PER_BUFFER` from 50
to 200 (the limit is read in mlx `device.cpp:768-773`; the launcher lands it in exo `463ac5d`): a single indexer `argsort` at the era shape
registered ~196 MB in one shot, 3.9x the 50 MB limit but under 200 MB. One correction:
`EXO_DSV4_PREFILL_ARGPARTITION=1` was credited with the fix, but `mx.argpartition` on Metal
delegates to the same `gpu_merge_sort(..., argsort=true)` as `argsort` (sort.cpp:342) and the
340K cliff was already gone before it shipped -- it is a decode-time optimization, not the
cliff fix (deep-context degradation at 1M is a separate regime, §9.3). After: c=1 500K cold prefill ran 1993 s for 251 t/s average, never below 200
(`docs/prefill-throughput-breakthrough-2026-06-24.md:87` -- the fix landed in June; the
mechanism was explained two months later, in the 2026-08-24 note); a 2026-08-24 probe at
381,619 tokens measured 328.6 t/s aggregate, needle PASS. Today, live prefill is 245.1 t/s at 100K and
242.0 at 160K, 195.4 at 300K, and 64.9 over the deepest band of a 1M request (evidence-live.md;
the depth runs' generated text appears in §5.4 and §6.1).

### 5.4 The memory budget does not close, and it runs anyway

The honest number first. Per-rank weights are 108.76 GB measured (routed experts 98.02, DSpark
3.84, shared expert 0.40, replicated attention+embed+head 6.50; components sum to 211.0 GB per
pair vs 210.68 GB on disk, ~0.2% apart -- du vs component-sum). Against the live
`iogpu.wired_limit_mb = 120000` (125.83 GB; 117.19 GiB), plus 3.4-4.5 GB (3.2-4.2 GiB) of 1M KV
and a 16-22 GB activation/scratch/allocator ESTIMATE, headroom is **-2.3 GB (best) to -9.4 GB
(worst)** at 1M. (The evidence pack's D4.4 formula reads the same knob as 120.0 GB; the corpus's
own convention is 120000 MiB = 125.83 GB, used here.) The budget does not close at 1M. The same
tension is the 8.8-14.8 GB weight-vs-cap gap column in
`docs/deepseek-v41-memory-budget-correction-2026-09-27.md:88-95`.

And yet it runs at 1M. Both nodes sit at ~127 GB of the 137 GB (128 GiB) in use while
serving -- real pressure, no wedge. Stated plainly, not papered over: the -8.2 to -15.3 GB
figure is a paper budget whose dominant term is an ESTIMATE (16-22 GB, borrowed from V4-Flash
plus V4.1 additions, not measured), so either the true overhead is under the low end, or the
MLX allocator reuses freed scratch rather than holding the high-water mark resident. We have
no activation measurement to say which. A 1M prompt completed end to end on this budget: HTTP
200, `prompt_tokens=1,000,003`, `finish_reason=length`, wall 6,760.5 s, generated:

```
BOTTLENECK: the `moe.all_sum` cross-rank collective (and its GPU->CPU->GPU drain) is the most frequently discussed performance bottleneck, appearing across dozens of docs from 2026-08-19 through 2026-08-24. It is quantified in `docs/moe-all-sum-dominant-cost-2026-08-19.md`, which measured a 2.63x speedup (62.0% share) at 12K tokens
```

---

## 6. Making it fast

### 6.1 DSpark/MTP speculative decode: +36.7% at 100K

Speculative decode came back as a production default after the depth-gated batched-verify fix.
The baseline config: `EXO_SPECULATIVE=1`, `EXO_DSV4_MTP=1`, `EXO_DSV4_DSPARK=1` with the
native checkpoint-bundled head (`EXO_DSV4_MTP_DEDICATED=0`), `EXO_SPECULATIVE_GAMMA=3`, and
`EXO_DSV4_VERIFY_BATCH=1` with `MIN_CTX=8192` -- batched M=4 verify above 8K context, the
bitwise rowseq path below (`docs/dspark-mtp-production-baseline-2026-08-27.md:8-9,45-66`).

The measurement was 24 paired runs on the same 100K-token prompt, temp=0, a 256-token fixed
window: **spec-ON median 36.63 tok/s vs spec-OFF 27.15** (= +36.71% as the baseline doc states
it; the two quoted medians give +34.9%), 95% bootstrap CI
[+28.26%, +51.02%], 12/12 pairs
(`docs/dspark-mtp-production-baseline-2026-08-27.md:18-21`). Depth is gamma=3. The mechanism
is verify cost, not drafting: the batched M=4 verify forward dropped per-round verify from
83.76 to 60.60 ms mean (MTP-PROF n=1550; C_s 3.20 -> 2.14), and acceptance held at 2.250
versus the rowseq path's 2.118. Commits: mlx-lm `dda9237`, exo `6eba31f`.

Do the headline decode numbers include it? Yes. The production defaults ship spec ON, and the
live instance's decode arrives in speculative rounds of ~2.8-3.0 tokens at every depth
(evidence-live.md, Deliverable 2), which is a gamma=3 signature. So the live decode plateau --
34.5 / 31.4 / 29.4 / 30.7 tok/s at 1K / 10K / 50K / 100K -- is a spec-ON figure. The 100K row
generated, verbatim:

```
BOTTLENECK: The single most frequently discussed performance bottleneck across the reference material is the `moe.all_sum` cross-rank collective in the MoE forward pass, investigated in at least 20 distinct docs spanning 2026-05-14 through 2026-08-21. The document that quantifies it is `docs/moe-all-sum-dominant-cost-2026-08-19.md`, which used a NOP-ablation to measure a 2.63x speedup (62.0% share) at 12K tokens and 2.57x (61.1% share) at 38K tokens when all_sum was bypassed. Later work in `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md` and `docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md` corrected those percentages downward to a real ~2.9- [...]
```

The same effect is not free at every depth. At 352.6K the 2026-08-27 run found a memory
regression -- the ~10.13 GB DSpark head loaded replicated per node, pushing spec-ON residency
to ~99.4 GB/node -- and 4 of 16 ON runs collapsed to ~1 tok/s via mmap clean-page eviction
thrash. TP-sharding the head's MoE FFN (fix 2d85ccd, ~3-3.5 GB/node back) restored residency
and gave +17.6% over stripped-OFF at 352.6K, zero collapses over 8 runs
(`docs/dspark-mtp-production-baseline-2026-08-27.md:77-95`).

### 6.2 The MoE all_sum collective: closed, NEGATIVE

This is the investigation we would most want a reader to trust, so it stays in full. It is a
negative result with methodology, and it pre-empts the obvious "did you try quantizing the
collective" comment.

The headline, originally: a NOP-ablation on the live cluster measured the per-layer TP
`moe.all_sum` at 2.63x and 2.57x speedups at 12K and 38K tokens -- a **62.0% / 61.1% share of
prefill wall time** (`docs/moe-all-sum-dominant-cost-2026-08-19.md:18-19`). That number is
wrong, and the correction is the point.

Correction, step one: jaccl-internal timing -- timestamps inside the C++ transport call
itself, immune to Python/MLX graph laziness -- measured the real decode-time collective at
**median 36.1 us, mean 66.3 us** per call for the actual 8192-byte payload (hidden_size 4096 x
2 bytes/bf16), across 45,666 calls per rank
(`docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md:5-15,97-114`). Only 0.04% of
calls exceeded the earlier sync-span figure; the transport is fast.

Correction, step two: a pure arithmetic reconciliation. The sync-span-measured ~4094 us/call,
summed over exactly 43 `moe.all_sum` calls per token (the V4-Flash-era engine these August
runs measured; §4.2), predicts 176.0 ms/token against a real
measured 53.48 ms/token -- 3.29x the entire per-token budget, which is impossible. The 4094 us
figure was a forced-per-span-synchronization artifact: `mx.synchronize()` at a span boundary
drains the whole pending lazy graph, misattributing preceding layers' compute to the span, and
structurally serializes an execution regime production does not run under. With the real
transport cost, 43 calls x 36.1 us = 1.55 ms/token, or **2.9%-5.3% of decode wall time**
(`docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md:18-51`). The "21.4% of decode
wall" figure from `docs/moe-allsum-collective-cost-confirmed-2026-08-21.md` is the same
artifact. The correction happened, and the corrected share is 2.9-5.3%, not 62%.

The lever that followed the 62% headline was quantizing the collective's wire payload. It is
closed as mathematically infeasible, not merely buggy. `moe.all_sum` is a true elementwise
partial-sum reduction -- both ranks contribute to every output element (`deepseek_v4.py:~2834`)
-- so the zero-padded int8 trick that works for gather-semantics collectives does not apply,
and naive int8-code summation overflows and cannot recover per-rank scales. The live test
confirmed the failure mode: the quantized path hung the collective, and the corrected
measurement was 265 vs 178 ms/call, i.e. 1.49x SLOWER, not faster. The investigation's root
cause: replacing one reliable `all_sum` with two `all_gather`s moved required traffic onto
jaccl's `all_gather` path, which has no reliability layer at all (raw UC posts, a `StallWatch`
that throws on any stall, no retransmit) versus `all_sum`'s soft-RC path
(`docs/moe-allsum-quant-root-cause-and-closure-2026-08-19.md:28-87`). The lever is dead; a
transport-level `all_gather` reliability fix is a jaccl-core project, out of scope for a
config change.

Two smaller entries from the same family, both negative, both kept. The NOP toggle itself --
the instrument that produced the 62% -- was later declared unsafe for throughput measurement
after it destabilized the cluster and forced a relaunch
(`docs/allsum-ablation-unsafe-2026-08-21.md:88`), so the headline figure rests on a methodology
the corpus has since disowned. And the `MLX_JACCL_RELIABLE_MAX_SZ=3` bump measured
statistically identical to sz=2, a clean null; do not re-attempt
(`docs/jaccl-sz3-tested-no-improvement-2026-08-20.md:21`).

---

## 7. Making output usable

Throughput is worthless if the model loops or never stops thinking. Two guards run armed in
production, both visible in the live log at HEAD `5a18b81`.

### 7.1 Degeneration loop guard

The guard watches the committed token stream for exact cycles (`EXO_LOOP_DETECT_WINDOW=160`,
`MAX_PERIOD=24`, `MIN_REPEATS=6`, `ACTION=error`, plus a long-block detector). We fired the
classic trigger -- "output the word banana exactly 40 times" -- and the model reasoned
normally, then collapsed. The last deltas on the wire, verbatim, in order:

```
delta.reasoning_content = "ban"
delta.reasoning_content = "ana"
delta.reasoning_content = " banana"
delta.reasoning_content = " banana"
delta.reasoning_content = " banana"
delta.reasoning_content = " banana"
delta.reasoning_content = " banana"
```

Instead of a normal finish, the turn terminated with a clean error event:

```
data: {"error": {"message": "Generation terminated: repetition-loop degeneration detected (the model collapsed into a repeating token cycle). Failing the turn cleanly so it can be retried.", "type": "InternalServerError", "code": "repetition_loop"}}
```

The runner's own line for the same turn names the mechanism:

```
[DSV41] DEGENERATION DETECTED ... at committed=269 emitted=268: raw-token cycle period=1 repeated>=6x action=error guard_env_digest=633711af prompt_tokens~26 cycle_token_ids=[51406] cycle_text=' banana'
[DSV41] obs turn-end ... committed=269 emitted=268 elapsed=7.4s tok_s=36.5 reason=degeneration-error
```

Period=1, a single token id 51406 that detokenizes to `" banana"`, repeated six or more times,
killed at 268 emitted tokens. `guard_env_digest=633711af` matches the boot line's digest, so
the knobs that fired are the knobs that shipped. The guard's job is to stop the tail, not
retract it: the 268 degenerate tokens were already streamed. A negative control (a 12-section
essay, `max_tokens=1200`) generated all 1,200 tokens with `finish_reason=length` and did not
fire.

### 7.2 Reasoning-budget limiter

Same prompt, same `max_tokens=800`, `enable_thinking: true`; the only difference is the
per-request `reasoning_budget_tokens` field, which overrides the limiter's `0.75 x max_tokens`
default.

| run | `reasoning_budget_tokens` | server `budget=` | `reasoning_tokens` | `completion_tokens` | `finish_reason` |
|---|---|---|---|---|---|
| default | absent | 600 | 600 | 800 | `length` |
| limited | 96 | 96 | 96 | 447 | `stop` |

With the default the model spent its whole 600-token reasoning allowance and then ran into
`max_tokens=800` -- `finish_reason=length`, answer truncated. With a 96-token budget the
engine reported `reasoning_tokens=96` on the nose and the turn completed normally
(`finish_reason=stop`) with the answer in `content`: the next sampled token was forced to the
think-END delimiter. The reasoning the model was allowed, verbatim, all 322 characters of it:

```
We need answer classic. Need explain step by step full reasoning, verify two independent ways. Need be careful. Let ball x, bat x+1.00, total 1.10 => 2x+1=1.10 => x=0.05. Verify: sum 0.05+1.05=1.10, difference 1.00. Another independent way maybe algebra with total minus difference? Let's craft. Need mention common trap $
```

and immediately after it, the start of the `content` channel, which only exists because the
limiter closed the think block:

```
Let the ball cost \(x\) dollars.

The bat costs \(\$1.00\) more than the ball, so the bat costs:
```

96 tokens of reasoning instead of 1,996 characters, a clean `stop`, and an answer that still
lands on 5 cents. The limiter's default budget is capped at 16,384 (observed live as
`budget=16384` on ordinary traffic), so it constrains runaway thinking rather than normal
traffic.

---

## 8. Operating it

This section draws on live reads of the nodes' config and scripts (cited inline below:
`reboot-node.sh`, the live `smctl` config, `docs/xcode-removal-launcher-clt-fallback-2026-08-25.md`,
commit `70e0423`), not on the two evidence packs.

### 8.1 Fan curve via smctl

Both Studios run the `smctl` daemon with a custom curve for the `exo`
profile, deployed 2026-10-11. It was calibrated from a controlled load test -- a 12-core CPU
burn on one Studio, 34.5 W package, with the hottest Tp* sensor climbing 40C -> 58C under
Apple's auto curve while it still held the ~1000 RPM floor. The curve is
`points = [[58, 1000], [65, 1500], [72, 2100], [78, 2700], [85, 3200], [92, "max"]]`,
`hysteresis = 4`, `slew_rate = 500`, `fall_slew_rate = 150`, with an 11-sensor list and a
`safety.temp_ceiling = 100.0` guard (read live from `/etc/smctl/config.toml` on the serving
node, not from the evidence pack). The framing we keep: the one throttle event we actually
chased was a **GPU power-governor** issue, not thermal (`pmset -g therm` clean on both nodes
during it, hang-watchdog incident 2026-09-21), so the fan curve is defense-in-depth against a
thermal throttle we have not measured, not the fix for a measured one. [UNVERIFIED] on that
causal claim.

### 8.2 FileVault-safe reboot

Because §3.2 means reboots are routine, `reboot-node.sh` automates
them. A plain reboot would halt at the FileVault pre-boot unlock screen and never come back on
the network, so the script drives `fdesetup authrestart`: it pulls the FileVault password from
1Password **on the controller**, pipes the authrestart plist over ssh stdin straight into
`fdesetup`, and never lands the password on disk or on the node (`reboot-node.sh:2-6,42-76`).
Sudo is scoped with `NOPASSWD` to exactly `/usr/bin/fdesetup authrestart*`, and a `--check`
dry run verifies credentials and the sudoers rule without rebooting. The cluster returns
SSH-able and unlocked in ~2-3 min.

### 8.3 SIGTERM runbook

`scripts/exo_graceful_shutdown.sh` is the single shutdown path (§3.3):
SIGTERM every target, wait a grace period, escalate to SIGKILL only as a last resort, and
report a verdict line (`CLEAN_EXIT` / `SIGKILLED` / `SURVIVED_SIGKILL`) rather than gating,
because the caller decides. Verified live: `CLEAN_EXIT after 1s (NO SIGKILL USED)` on both
nodes, ports `PORT_ACTIVE`, 0% TB link loss.

### 8.4 Rust toolchain incident

When Xcode.app was removed from both Studios, the
launcher's hardcoded `DEVELOPER_DIR=/Applications/Xcode.app/...` made the isolated maturin
build of the `exo_rs` PyO3 bindings die at `xcrun` and aborted the whole deploy before exo ran
a line, fixed by commit `70e0423` resolving `DEVELOPER_DIR` to Command Line Tools when Xcode
is absent (`docs/xcode-removal-launcher-clt-fallback-2026-08-25.md`).

---

## 9. Measured performance

All numbers below are live runs against the serving instance at HEAD `5a18b81`, captured
from a separate machine (a MacBook Pro M4) so no capture work competed with the GPU. Model
`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`, `temperature=0`,
`enable_thinking=false`, `"stream": true` on every run except the first 160K attempt.

### 9.1 Methodology

**Two independent timing channels, always both.** (1) Client: every SSE frame is
line-stamped with an epoch time; the decode window is last-committed-token delta minus
first-committed-token delta, with `: keep-alive` and `: prefill_progress` comment frames
excluded. (2) Server: the runner's own `[DSV41] obs first-token` and `obs turn-end` lines,
correlated to the request by response `id` == `command_id` -> `task_id`. The two channels
agree to within 0.3% on every deep row, which is what validates the method.

**The `tok_s=` field on a turn-end line is not a decode rate.** It divides committed tokens
by elapsed *including* prefill (0.1 tok/s at 300K, 0.3 at 160K). It is quoted only where the
raw line is quoted.

**Token counts are the API's own `usage` block**, never a local estimate. Local counts come
from the checkpoint-family tokenizer and are calibration only; they agreed with the server
to within 5 tokens at 160K and exactly at 1M.

**Prefill is reported uncached**: `(prompt_tokens - cached_tokens) / server_prefill_s`. A
prefix-cache hit is not prefilled compute. Without that correction the raw column appears to
rise with depth purely because deeper rows got partial cache hits.

**Contention was checked per run.** exo serves one prefill in flight at a time; each run's
measured decode window was verified to contain no other request's `obs first-token`. For the
300K row that check is shown explicitly. Rates are as observed on a busy-but-serialized
instance.

**The runs are needle-style questions over the repo's own docs**, and the answers are correct
against the context each run received. Several are reproduced verbatim in §5.4 and §6.1,
so the throughput figures can be read against real output rather than a token count.

### 9.2 Decode and prefill vs context depth

| depth | prompt_tokens | cached | completion | finish | server prefill s | prefill tok/s (uncached) | client decode s | client decode tok/s | server decode tok/s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1K | 1,003 | 0 | 82 | `stop` | 3.93 | 255.2 | 2.35 | 34.5 | 29.2 | 6.95 |
| 10K | 10,004 | 0 | 152 | `stop` | 36.37 | 275.1 | 4.80 | 31.4 | 31.3 | 41.34 |
| 50K | 50,004 | 8,192 | 163 | `stop` | 162.27 | 257.7 | 5.51 | 29.4 | 29.3 | 167.94 |
| 100K | 100,003 | 49,152 | 200 | `length` | 207.50 | 245.1 | 6.49 | 30.7 | 30.6 | 214.27 |
| 160K | 160,004 | 0 | 200 | `length` | 661.18 | 242.0 | n/a | n/a | 30.4 (server decode window 6.54 s, server-side only) | 668.15 |
| 300K | 300,003 | 0 | 189 | `stop` | 1535.25 | 195.4 | 31.67 | 5.9 | 6.0 | 1600.82 |

Five further points at ~110-119K were harvested from another client's traffic on the same
build and instance -- independent corroboration, not repeats: 26.7 / 32.2 / 32.2 / 37.6 /
30.3 tok/s server decode at 110,780 / 113,424 / 115,457 / 116,680 / 118,665 tokens.

**Two regimes, not one.** Decode is a smooth ~30 tok/s plateau to 160K, then 6.0 tok/s at
300K -- a 5x falloff inside one doubling-and-a-half. The five independent points sit on the
flat part and confirm the cliff starts between ~160K and 300K, not gradually. There is no
second slow mode hiding in the 1K-160K range.

**The 300K cliff is not an artifact.** No contention (the turn's first-token is 02:31:59.523,
turn-end 02:32:31.188, next request's first-token 02:32:38.018 -- after it ended). The burst
structure explains the mechanism: decode arrives in speculative rounds of ~3 tokens at every
depth (2.8-3.0 measured, constant), and what changes is the cost of a round:

| depth | tokens/round | median inter-round gap | implied tok/s |
|---|---|---|---|
| 1K | 2.9 | 0.085 s | 34.1 |
| 10K | 2.9 | 0.086 s | 33.7 |
| 50K | 2.8 | 0.089 s | 31.5 |
| 100K | 3.0 | 0.092 s | 32.6 |
| 300K | 2.9 | 0.486 s | 6.0 |

A constant round size with the round cost rising 5.3x is a per-round memory-side cost (KV
read / residency), not a sampling or parser artifact. Offered as the mechanism the numbers
point at, not as a proven cause.

### 9.3 Prefill degrades with position

Prefill is not flat. 255.2 / 275.1 / 257.7 / 245.1 / 242.0 tok/s at 1K-160K, then 195.4 at
300K, and inside the 1M runs 216.8 tok/s over the first 100K band falling to 64.9 over the
last band. The 1M request's own banded curve, measured within one request (no cross-run
confound):

| absolute position of rows being prefilled | prefill tok/s |
|---|---|
| ~303,104 - 403,456 | 182.2 |
| ~403,456 - 503,808 | 150.1 |
| ~503,808 - 604,160 | 133.6 |
| ~604,160 - 704,512 | 125.5 |
| ~704,512 - 804,864 | 98.1 |
| ~804,864 - 905,216 | 69.2 |
| ~905,216 - 1,000,003 | 64.9 |

Monotonic within the request. Prefill cost is a function of POSITION in the context, not of how many tokens have
been done: the same request prefilled its first 100K rows at 216.8 tok/s in attempt 1
(cold, starting near position 4K) and at 182.2 tok/s in attempt 2 (starting at position
~300K). The deepest stretches are irreducibly the slowest, which is why a 1M prefill cannot
be front-loaded or parallelized away.

### 9.4 The 1M run

Attempt 1 (2 h cap) was PARTIAL: prefill reached 802,816 / 1,000,003 tokens (80.3%) in
7,181 s (111.8 tok/s) before `curl: (28)` killed it; no first token was ever emitted.
Attempt 2 (4 h cap) COMPLETED: HTTP 200, `prompt_tokens=1,000,003`, `completion_tokens=100`,
`finish_reason=length`, wall 6,760.5 s, 104.2 tok/s prefill over 700,995 uncached rows,
2.92 tok/s decode (client and server agree). `curl_exit=0`, no timeout.

The cache caveat, stated plainly: `cached_tokens=299,008`, server `prefix_cache_hit=partial`,
`reuse=299008` -- those rows are attempt 1's partial prefill of the same prompt, left resident
and reused. Generated text is in §5.4. This is *1M prompt served end to end, partial prefix cache* -- NOT a cold 1M
prefill. The `prefill_progress` counter tracks uncached tokens only, which is why its last
frame reads 700,995 while `total_tokens` reads 1,000,003; the turn reached 100% of the work
it had to do. Attempt 1 (~98K cached) is the closer cold proxy; extrapolating its last band
(~68 tok/s) over its remaining ~197K tokens puts a cold 1M prefill at roughly 2.9 h.

### 9.5 Speculative decode

Production default; the full measurement, mechanism and commits are in §6.1. In one line: 24
paired runs, same 100K prompt, temp=0, 256-token fixed window, **spec-ON median 36.63 tok/s
vs spec-OFF 27.15** (+36.71% as the baseline doc states it; the two quoted medians give
+34.9%), 95% bootstrap CI [+28.26%, +51.02]%, 12/12 pairs. The live
plateau above is a spec-ON figure -- the ~3 tokens/round signature is gamma=3.

---

## 10. Open problems

These are genuinely unresolved, not roadmap items. Sources are the corpus's own open-thread
ledger (`PERFORMANCE_HISTORY.md` section 13 and the dated docs) plus what the live runs
could not close.

### 10.1 The two cliffs (mechanisms unproven)

**Decode cliff between 160K and 300K.** Measured, reproducible, mechanism not proven. The
round-cost table in §9.2 points at a per-round memory-side cost (KV read / residency) and is
consistent with the repo's own `deep-context-refault-mechanism` note, but that is an
inference from burst structure, not a captured trace. The refault mechanism itself is
CONFIRMED (page-ins measured, swap/compressor ruled out) while the fix is explicitly NOT
attempted (`docs/deep-context-refault-mechanism-confirmed-2026-09-22.md:95-98`).

**Prefill cliff at ~340K.** The gc_limit-threshold model (§5.3) is the best standing
explanation but is not live-confirmed; `k_eff` is unmeasured (bounded 2-5); the bimodal 8-32 s
stall amplitude at Studio scale is UNREPRODUCED (`PERFORMANCE_HISTORY.md:2943-2953`,
`docs/prefill-cliff-mechanism-2026-08-24.md:48,877`). The mitigation is in and measured (§5.3); the mechanism
underneath is not live-confirmed.

### 10.2 The memory budget does not close

The arithmetic and the ESTIMATE caveat are in §5.4. The prescribed remedy -- streaming the
coldest 9.0-15.1% of expert bytes -- is not implemented
(`docs/deepseek-v41-memory-budget-correction-2026-09-27.md:88-99`).

### 10.3 Measurement holes

- **Collective-wait honesty hole.** A wait hidden inside a command buffer reads as busy. A
  clean closer exists (`EXO_DSV4_ALLSUM_PROBE`, a direct CPU-side collective timer) but it
  forces a blocking eval in place of the production async fence, perturbing what it measures,
  and was never run (`PERFORMANCE_HISTORY.md:3633-3640`).
- **Per-kernel GPU attribution structurally unavailable.** xctrace/Metal System Trace yields
  only generic "Compute" channel names; `mx.metal.start_capture` / Xcode GPU Frame Capture
  stayed unreachable in an SSH-only environment (`PERFORMANCE_HISTORY.md:2344-2352,3745`).
- **Decode roofline floor excludes KV-cache/attention read cost** -- a methodology gap, so
  the 500K attribution figure is likely overstated once KV-read cost is counted
  (`PERFORMANCE_HISTORY.md:2309-2318`).
- **Prefill FLOPs roofline ~1.40x headroom tail not decomposed**; the 28.8% non-GEMM
  remainder minus the 9.5% all_sum slice was never decomposed with decode-style rigor, and
  the sdpa ceiling denominators were never audited for masking/MLA-absorption FLOP inflation
  (`PERFORMANCE_HISTORY.md:2446-2457`).
- **~11 ms/round unaccounted** in the DSv4.1 adaptive-gamma accounting, explicitly not the
  sync (`benchmarks/phase18-dsv41-adaptive-gamma-2026-09-28/README.md:25`).

### 10.4 Unattributed residuals

- **+1.67..+2.52 ms/tok decode depth residual**, all of it on-GPU busy work, collective ruled
  out on both axes; the leading candidate (inter-layer pipelining bias) is unproven
  (`PERFORMANCE_HISTORY.md:3619-3626`).
- **Which sub-cause dominates the 34x all_sum span-vs-transport gap** (34x = the ~4094 us
  sync-span average vs the ~120 us isolated-microbenchmark floor; eval-fence vs dispatch
  vs Python scheduling) -- attributed jointly, "not yet further decomposed, now the correct
  next target" (`PERFORMANCE_HISTORY.md:3015-3017`).
- **Cross-rank all_sum straggler tail asymmetry** (4.2x rank0-leaning in rare severe calls),
  root cause not investigated, ~1.7% of decode wall upper bound, "not conclusively
  ignorable" (`PERFORMANCE_HISTORY.md:2472-2493`).
- **Do 0.5-20 ms decode gaps align with per-layer collective/fence boundaries?** A
  ~20-22 gaps/token check did not cleanly confirm the expected 43/token -- inconclusive
  rather than refuted (`PERFORMANCE_HISTORY.md:2793-2799`).
- **MoE-at-depth interplay vs allocator/memory-pressure as the depth-residual owner**:
  candidate-1 verdict INCONCLUSIVE (`docs/p02c-moe-allocator-depth-residual-2026-08-29.md:176,380`).

### 10.5 Transport and concurrency

- **jaccl C++ transport-layer bugs, "NOT YET FIXED"**: long-session corruption (first-send
  loss on 16380-byte UC chunks) and the `p2p_retry_exchange` stall. A poison-WR probe is
  armed but never fired; the stall is explicitly "still open and genuinely separate"
  (`docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:4444-4446,8233-8241`).
- **N=2 concurrent-admission transport crashes, root cause unidentified**; two different
  crash signatures (parse error vs ECONNRESET) not yet distinguished
  (`docs/batched-decode-n2-admission-handoff-2026-08-05.md:1091-1109`).
- **c=2 speculative decode is not production-ready**: BS>1 batched verify is the Phase-5
  TODO, the c=2 spec-OFF throughput control is outstanding, and two shared bugs remain
  (`docs/dspark-mtp-master-history-2026-08-28.md:333-335`).
- **Rank1 Metal GPU timeout crash forensics**: the best-supported mechanism is
  tracing-procedure memory pressure, but the alternative is not separable with generic
  channels; untraced deep (500K) decode jetsam/swap risk was told to be measured directly,
  not inferred (`PERFORMANCE_HISTORY.md:3273-3286`).

### 10.6 Model-side and quality

- **DSpark FULLBLOCK depth collapse** is carried as an unresolved pre-condition for enabling
  DSpark: "15.9x collapse, worse than no speculation at depth. Unresolved"
  (`docs/p4-scoping-mtp-for-tp-2026-08-24.md:1074`). Note the corpus holds three discordant
  verdicts on it (§10.7).
- **V4.1 full-body runtime unknowns**: prefetch overlap must be engineered (60-80% assumed),
  full-model routing concentration unverified, real end-to-end numbers including acceptance
  need the runtime (`benchmarks/phase8-fork-verification-2026-09-28/README.md:80-88`).
- **Token-63 propagation**: whether the L2-full lever's bf16 tie-break propagates to the
  token-63 output change is not pinned; no output-affecting change ships on this evidence
  (`PERFORMANCE_HISTORY.md:11649,11658`).
- **MoE expert-GEMM prefill kernel headroom is real but not priced or harvested**: loop-2
  verdict INCONCLUSIVE at 51-60% of calibrated ceiling; the prescribed per-segment M/N/K
  autopsy is not executed (`PERFORMANCE_HISTORY.md:11149,11394-11395`).

### 10.7 Unresolved conflicts in the corpus

- **MoE gate+up fusion sign**: -3.8% at c=2 (2026-06-26) vs +3.01% at c=1 short-ctx
  (2026-08-21); `PERFORMANCE_HISTORY.md:1234-1237` records it as unresolved.
- **DSpark FULLBLOCK 15.9x collapse**: the source doc major-corrects it as not reproduced,
  while the p4-scoping doc still lists it unresolved. Any writeup must pick a frame.
- **hc_expand**: rejection upheld on 2026-08-22 vs a fused kernel shipped 2026-08-24; two
  different levers, and the shipped kernel's own 300K/500K depth A/B was INCONCLUSIVE.

### 10.8 What the live session could not close

- A **fully cold 1M prefill** was never measured (attempt 2 had 299,008 cached rows); the
  ~2.9 h figure is an extrapolation from attempt 1's last band.
- The guard's `action` knob (`warn` vs `error`) was not toggled -- it needs a relaunch.
- 1M decode is 2.92 tok/s; whether that is the same mechanism as the 300K cliff or a
  different one is not established.

---

## 11. How to reproduce

### 11.1 What you need

- Two Apple Mac Studio M4 Max, 128 GB unified memory each, macOS 27.0 (26A428) or later.
- One Thunderbolt 5 cable pair between them, both ends enumerated as RDMA devices
  (`rdma_en3` / `rdma_en4`, `status: active`). Verify with `ibv_devinfo -v`; expect
  `max_qp:3, max_cq:6, max_pd:6` per device.
- `sysctl iogpu.wired_limit_mb = 120000` on both nodes.
- A controller machine (a laptop is what we use) with ssh to both nodes.
- The checkpoint `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (210.68 GB, 40
  safetensors shards) present at `~/.exo/models/` on both nodes, identical.

### 11.2 The code

Three forks, all under `~/repos/` on the controller:

| repo | remote | branch | HEAD |
|---|---|---|---|
| exo | `github.com/<user>/exo` | `main` | `b88d65dd330478e8e1b8a565848af3c863be7a06` |
| mlx | `github.com/<user>/mlx` | `main` | `b177818386215ea578c16f8d8de06706e06bcff8` |
| mlx-lm | `github.com/<user>/mlx-lm` | `main` | `676f803ed28f2703adb7a16f74a407baccf05066` |

The mlx row's `main` is the branch the `uv.lock` pin references; the local checkout at capture
was on `integrate/upstream-20261016` at the same SHA. The exo row is the deployed lineage:
later on 2026-10-17 `main` was folded to a superset of `deploy/next28` plus the §12
liveness fix, and that is what `start_cluster.sh` last synced to the nodes.

Check out the branch and the submodules ON THE CONTROLLER first. `EXO_TARGET_BRANCH` is only
a push-check, not a checkout -- setting it without checking out is a silent no-op.

The mlx and mlx-lm SHAs must match both the submodule gitlinks and the `uv.lock` pins
(`uv.lock:1382-1384` for mlx, `:1499-1501` for mlx-lm). CI enforces this
(`src/exo/shared/tests/test_submodule_lock_pin_alignment.py`); a mismatch will fail the
deploy. On darwin, mlx resolves from the fork via `pyproject.toml:90`; on Linux it comes
from third-party wheels on `rltakashige/mlx-jaccl-fix-small-recv`.

### 11.3 Launch

Deploy is `start_cluster.sh` run from the controller. It rsyncs `~/repos/exo` to both nodes
and starts the cluster; the API then listens on port 52415. Do not edit files on the nodes
-- all changes go through git to the fork.

```
cd ~/repos/exo
git checkout main && git submodule update --init --recursive
./start_cluster.sh
```

Shutdown is SIGTERM-only via `scripts/exo_graceful_shutdown.sh`; SIGKILL leaks RDMA queue
pairs and can wedge the Thunderbolt link. If a node does need a reboot, use
`./reboot-node.sh` (it drives `fdesetup authrestart` so FileVault auto-unlocks) -- a raw
`sudo shutdown -r now` leaves the node at the FileVault unlock screen with no network.

### 11.4 Confirm the instance

```
curl -s http://node0.tailnet.ts.net:52415/state
```

Read the placed instance's model id from the response. Expect `contextLength=1048576`,
`maxKvTokens=1048576`, `worldSize=2`, `maxPrefixSessions=4`, `maxPrefixBytes=12884901888`,
`storageSize=210,713,013,432`.

### 11.5 Test

Write prompt bodies to files and pass them with `--data-binary @file` -- never inline a
160K prompt in a shell argument. All runs use `"stream": true` so per-token timing exists;
`"stream": false` collapses `time_starttransfer` onto `time_total` and leaves decode
unmeasurable.

One depth row:

```bash
cd prompts
curl -sS -N --max-time 3600 -X POST \
  http://node0.tailnet.ts.net:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @depth_d100k.json \
  -w 'CURLMETA http=%{http_code} starttransfer=%{time_starttransfer} total=%{time_total}'
```

Body shape (the corpus is the repo's own `docs/**.md` plus the repo's top-level handoff docs,
concatenated in sorted order,
truncated at the target depth, then the question appended):

```json
{
 "model": "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw",
 "max_tokens": 200,
 "temperature": 0.0,
 "stream": true,
 "enable_thinking": false,
 "messages": [ {"role": "user", "content": "<N tokens of repo docs> + the question"} ]
}
```

The 1M run needs a 4-hour window: `--max-time 14400`. A 2-hour cap reaches only ~80% of the
prefill and is cancelled by the client.

### 11.6 Reading the result

The measurement rules are §9.1. Two repro-specific notes: cross-check against the runner's
own `[DSV41] obs first-token` / `obs turn-end` lines in `~/.exo/exo_log/exo.log`, correlated
by response `id` == `command_id` -> `task_id`; and over non-interactive ssh `zstdcat` is not
on PATH -- use the absolute path `/opt/homebrew/bin/zstdcat` when reading rotated archives,
or an empty grep result will look like a real negative.

### 11.7 Guard and limiter

Both are armed in the running process and read at engine construction, so toggling them needs
a relaunch. The guard's knobs and the limiter's behavior are in §7; the classic guard repro
is "output the word banana exactly 40 times", and the limiter honors a per-request
`reasoning_budget_tokens` field over its `0.75 x max_tokens` default (capped at 16384).

---

## 12. The night the cluster killed itself (and the fix)

Everything above describes a cluster that works. This section is the night it did not, and it
is the reason the lead sections of this writeup could be written by a session running *on*
the cluster, and it draws on the incident captures behind the 2026-10-17 PERFORMANCE_HISTORY
entry -- the supervisor log and the detector report -- plus the deploy's own validation logs,
with commits cited inline.

### 12.1 The incident

Overnight 2026-10-16 -> 17, a session on the cluster model was killed repeatedly.
From 04:41 to 07:54 the supervisor SIGKILLed **live, actively-decoding runners roughly every
8.8 minutes -- 22 kills over that 193-minute window, 12 on rank0 and 10 on rank1**. Each kill cost
a re-placement plus a full 210 GB reload (~164 s) on **both** nodes, ~25 kill/re-place
cycles total.

Every kill hit a runner whose own log showed committed tokens advancing at ~30 tok/s within
5-18 s of the kill. Zero guard fires all night. The deaths clustered at ~2,700-2,800
committed tokens because that is where 45-67 s of supervisor silence had elapsed.

### 12.2 Root cause

The generation was a giant **withheld `write_file` tool call**. The parser holds the DSML
block until it closes, so `step()` never returns a result and the runner-loop decode
heartbeat in `handle_generation_tasks` cannot fire. Meanwhile the decode phase emitted
nothing the supervisor counts: LOAD and PREFILL re-emit runner status every <=15 s (the
load warmup beat, the fence hook, the prefill heartbeat -- all from the 10-05 fix), but
DECODE emitted no liveness event at all.

The hang probe then fired on flat footprint + `spin=True` -- both entirely normal during deep
decode, where the RDMA poll threads burn ~1.3 cores. The stack-class classifier (Fix B,
running in shadow) would have killed these too, because `spin=True` is not discriminative for
decode; it was deliberately not armed as the remedy.

Post-mortem decode of the killed streams (server degeneration dumps, 2,048 committed ids
each) shows real draft prose behind a short "Let me write." preamble loop. The kills destroyed
live work.

### 12.3 The fix

Commit `991cf11` on branch `fix/dsv41-decode-liveness`, off `deploy/next28`:

1. **Per-round decode liveness beat** in `Dsv41Engine._rounds`: `self.prefill_heartbeat()` on
   every COMPLETED round with a non-empty committed batch, on both TP ranks, every beat
   backed by a finished collective plus committed tokens. A genuinely wedged round still goes
   silent -- that is the control.
2. **Bounded output**: `resolve_max_output_tokens()` with `EXO_DSV41_DEFAULT_MAX_OUTPUT_TOKENS`
   (8192; 0 = the old 32168 default) and `EXO_DSV41_MAX_OUTPUT_TOKENS_CEILING` (32768; 0 =
   off), plus launcher allow-list entries. These are *output* caps; the limiter's 16384 (§7.2)
   is a separate *reasoning*-budget cap, and the controller's client-side `max_tokens=8192`
   (§12.5) is a third, independent bound.

Tests: 60 pass, 20 of them new (fake-clock cadence, wedge and empty-batch controls, resolver
matrix).

### 12.4 Validation, live, in four steps

Deployed ~10:05-10:12 from the laptop (`EXO_TARGET_BRANCH=fix/dsv41-decode-liveness`); both
nodes at `991cf11`; pre-deploy runner-env snapshots taken and matched post-launch; both
ranks log the same `resolved=8192` on the first request, 23 ms apart.

1. **6-minute streaming generation**: 11,608 tokens, `reason=stop`, ZERO watchdog lines.
2. **Killer-case reproduction**: a 190+ s window with zero client output in which committed
   advanced 2,263 -> 5,863 (~19 tok/s) -- the exact state that killed 22 times. Zero silence
   lines, runner alive.
3. **Negative control**: SIGSTOP rank-1 mid-generation -> SIGKILL at 61 s (inside the 45+20 s
   window), both ranks re-placed, health gate passed. The guard was not disabled.
4. **Re-validation**: a cluster-hosted Hermes session (provider exo) wrote this writeup's
   lead sections (§0, 1, 2, 9, 10, 11) to `scratch/writeup/lead/`:
   22 API calls, 30 minutes, **0 kills**. Self-referential by construction -- the session
   writing this document is the test case -- and a 30-minute window is thin, so read it as
   re-validation, not proof.

### 12.5 Companion changes and the detector that was measured and rejected

Same day: a hermes-agent crash-loop circuit breaker (`e69abff` + `6bb4366`;
`agent.auto_recovery_identical_failure_limit` default 2, where progress means strictly more
streamed than the previous attempt, so a deterministic mid-stream kill still counts), and a
controller-side per-model client cap (`providers.exo.models.<id>.max_tokens=8192`, verified on the
wire; the providers-map `extra_body` does not reach the main request path -- open audit for
`use_prefix_cache` / `service_tier`).

A text-level "varied repetition" detector was the obvious next idea, so it was measured
before shipping. Decoding the four real incident streams (last 2,048 committed ids each)
showed the tails are **coherent draft work** -- the writeup's own section prose and
performance tables -- with only a 4-line planning-phrase cycle. A line-level near-duplicate
sweep (W in {10,16,24} x M in {5,6,7} x J in {0.75,0.8,0.9}) tripped **0/4 positives at every
parameter combo**, while real negatives (session transcripts, the evidence packs, five exo
docs) reached 5-17 matched lines from markdown tables, code fences and headings alone. The
signal is inverted relative to the class: table-heavy real output scores as "more repetitive"
than the incident streams, and a trip would abort a turn that was writing real content.
Verdict: **do not ship a text-level detector for this class**; the existing exact-cycle guards
behaved correctly (they did not fire, because there was no exact cycle). The damage is
addressed by the shipped changes instead.

KV headroom sanity for the caps: DSv4.1 MLA KV at 1M is 4.13 GiB (§5.2) => ~4.2 KiB per
generated token; the new caps add at most ~35-140 MB against ~12.6 GB of headroom (125.83 GB wired
limit - 108.76 GB weights - 4.13 GiB 1M KV = ~12.6 GB). Memory is not the constraint.

### 12.6 Still open

(These are specific to this fix; §10 holds the cluster's wider open set.)

Built, default OFF: `fix/dsv41-no-repeat-ngram` (`31b5367` + hardening `755000e`), a
deterministic greedy loop escape -- no-repeat n-gram ban before the round argmax, masked-argmax
accept path, control-token exemption, per-turn telemetry. The enable ladder (off-equivalence
replay -> escape replay n=5/6 -> quality A/B -> perf -> soak) is not yet run; the knob is OFF
everywhere.

Also open by decision: when a CONFIRMED loop does fire, the guard's `action="error"` fails the
turn, the client retries an identical request, and greedy decode reproduces the loop. A
cleaner termination (return partial content as `finish_reason="length"` plus a marker) would
let the agent move on. Left unchanged deliberately -- it alters a shipped guard's semantics
and deserves its own review.
