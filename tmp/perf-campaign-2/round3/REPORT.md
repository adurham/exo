# CAMPAIGN 2, ROUND 3 — REPORT

**Date:** 2026-09-03
**Scope:** the feasibility gate for H-e (GPU-resident collective), the fence-correction
settling experiment, and the 06-26 doc patch. Full analysis in `FEASIBILITY.md` (this
dir); raw worker artifacts `W1-jaccl-transport-buffer.md` (this dir) and
`~/tmp/w3_gpu_collective_prior_art.md`; probe JSONs in `probe-results/`.

---

## VERDICT (one paragraph)

**H-e is INFEASIBLE at the pre-registered ≥40%-of-1400 µs materiality bar; the
decode-stall thread is CLOSED AT THE STACK LEVEL.** The transport is genuine ibverbs
RDMA (dlopen'd `librdma.dylib`, `rdma.h:143-167`) so the gating question's premise
holds, and the completion→GPU-signal half (CQE → `setSignaledValue` → `encodeWait`)
is mechanically cheap with APIs already in the fork (`FEASIBILITY.md` Q4). But the two
load-bearing premises fail on source facts: (1) **there is no cheaper coherence op** —
`input_coherent` already *is* the minimal mechanism, built on private Metal internals
(`#pragma METAL internals`, `coherent(system)`, `thread_scope_system`,
`fence.metal:15-26`) because no public Metal API guarantees GPU-write visibility to an
external DMA agent; (2) **jaccl's buffers are a host bounce by construction** — every
collective `memcpy`s MLX-array→own `posix_memalign`'d+`ibv_reg_mr`'d region
(`mesh_impl.h:789-801`, `rdma.cpp:22-28,90-96`), a NEW blocking fact; in-place
registration of `MTLBuffer.contents()` has never been done anywhere in the fork and is
worth only ~10 µs/layer. With cross-layer overlap blocked (`hc_expand` mixes the
collective result into all four hyper-connection streams), the GPU has no alternative
work during the collective, so faster signalling cannot buy back wait time. Sum of
H-e1's addressable components ≈ **10–150 µs of ~1400 µs/layer (~1–11%)** vs the ≥560 µs
bar; H-e2 (GPU-initiated DMA) needs private APIs absent from all shipped code
(`FEASIBILITY.md` Q5). The ~1400 µs is dominated by local compute drain — corroborated
by round 1's `compress_ratio`-alternation fingerprint, the no-peer control, and the
2026-08-22 natural experiment where fixing the async-fence gate alone recovered
+58–67% decode throughput while every transport-side lever ever measured is ≤5%.

## The six feasibility answers (summary; full cites in FEASIBILITY.md)

1. **Transport:** genuine libibverbs RDMA over Apple's TB stack via dlopen/dlsym
   (`IBVWrapper`), UC QPs, TCP control-plane only. `[PM-VERIFIED]`
2. **Registration:** host bounce by construction — jaccl registers only its own
   `posix_memalign`'d buffers; MLX storage is `StorageModeShared`
   (`allocator.cpp:17-18`) and in-principle registrable, but zero precedent in-fork.
   NEW finding this round. `[PM-VERIFIED]`
3. **Coherence:** `input_coherent` = per-word self-store + `seq_cst
   thread_scope_system` fence via private Metal internals; the encoder barrier is
   ordering-only (`memoryBarrier(BarrierScopeBuffers)`, `device.cpp:571-573`).
   No public API covers GPU→external-DMA visibility ⇒ no headroom. The CPU→GPU
   direction has NO equivalent op today (would need to be added). `[PM-VERIFIED]`
4. **GPU completion:** CQE→`MTLSharedEvent`→`encodeWait` expressible with existing
   fork APIs (`event.cpp:171-173,212-217`, `device.cpp:654-659`); missing piece is
   glue only (jaccl has zero SharedEvent usage). Genuinely cheap — but bounded by
   notification mechanics, not the dependency. `[PM-VERIFIED]`
5. **H-e1** buildable but saves ~1–11% (table in FEASIBILITY.md Q5) — below bar.
   **H-e2** infeasible without private APIs. **Prior art:** zero GPU-collective work
   in fork (108 jaccl commits all CPU-transport), branches, or upstream
   `ml-explore/mlx` (fork is not secretly ahead; upstream `distributed.cpp` throws
   identically). `[PM-VERIFIED]`

## Task 2 — the settling experiment: fence story DEAD on evidence; reorder NOT bit-safe

**Patch (verified by PM):** mlx-lm branch
`perf-campaign-2/task2-early-allsum-reorder` @ **`1446c5d`** (one commit on `main`@`37260bb`,
only `mlx_lm/models/deepseek_v4.py`, +53/−1), env-gated
`EXO_DSV4_MOE_EARLY_ALLSUM` (default off = byte-identical no-op; off-branch is the
untouched original). When ON: `all_sum(y)` immediately after `switch_mlp` + a second
`all_sum(shared_out)` before the combine, skipping the original late `all_sum` (which
would double-count). Engages only on the non-rowseq path (prefill, and c≥2/B≥2 batched
verify); c=1 DSpark verify (L=4, rowseq) is untouched by construction. Algebra:
`sum_r[(y_r·s).sum(-2)+sh_r]` vs `(sum_r y_r·s).sum(-2)+sum_r sh_r` — exact over reals,
**NOT bit-guaranteed in FP** (reassociation), as pre-registered by the worker and
confirmed below.

**Method:** 2× Mac Studio, live cluster, temp=0 seed=42. Restarts were surgical
(env reconstructed byte-for-byte from `ps eww` of the live runners; only delta = the
flag; `uv pip install --no-deps --force-reinstall ./mlx-lm` per restart; site-packages
is a copy, not editable — verified by md5 both directions). Captures in
`probe-results/`.

**Results (the gate's two legs):**

| Gate | Result |
|---|---|
| **Quality leg (the 06-26 failure signature: near-zero garbage output, `all_needles=False`)** | **PASSES decisively — garbage does NOT reproduce.** B=2 200K needle with flag ON: **`all_needles=True`, both streams answer `FALCON-MERCURY-7749`, zero special-token leak, zero bistability** (`earlyflag_needle_b2.json`: agg 90.4 t/s, sym 0.991; warm-run text capture: both streams correct). c=1 8-prompt probe: all 8 coherent, well-formed answers. The 06-26 attempt failed because the shared partial was never reduced; with the second `all_sum` the reorder is numerically sane. **Correction #1 CONFIRMED: it was the algebra bug, not the fence.** |
| **Byte-identity leg (temp=0 bit-equiv)** | **FAILS — and the failure is reassociation, not corruption.** Rig determinism controls: within-boot rep2 diff = **8/8 byte-identical** (rig is deterministic); cross-boot same-code control (restored main vs old main baseline) diverges on **[0,3,4,6]** — a cross-boot noise floor that exists with NO code change. Flag-ON vs flag-OFF (both new boots): **2/8 differ — [1] and [6]**. Prompt [1] (primes) is the clean signal: OFF-boot ≡ old baseline (stable), ON diverges mid-reasoning at char 80 with a *coherent alternative phrasing* ("natural numbers" vs "numbers") — the predicted `sum(y)·s ≠ sum(y·s)` bit-drift flipping one near-tie token, then cascading through argmax. [6] sits in the cross-boot-noise set (already divergent OFF-vs-BASE). At B=2 the two concurrent streams' *reasoning* also diverge slightly under ON (319 vs 310 chars) while both converge to the correct needle answer; the OFF/BASE boot had cross-stream identity. |

**Perf delta: NOT claimable, stated honestly.** ON-boot B=2 decode read 90.4 t/s vs
BASE 64.1 — but these are different boots with different numerics (MTP acceptance
cascades differ: c=1 mean gen-tps BASE 33.6 / OFF-boot 45.5 / ON 44.4; the
OFF-boot itself differs +35% from BASE across boots with identical code), so no
within-boot A/B of this import-time flag is possible and the record's within-boot rule
forbids quoting the cross-boot number as a perf claim. Registered as: **reorder's
throughput effect UNMEASURED (needs a per-call runtime-gated variant for a within-boot
A/B); its correctness effect is settled.** The verify-share delta requested by the
task is therefore honestly reported as *not measurable under this design this round*.

**Reverted:** both nodes back on `main` @ `37260bb`, site-packages md5
`0c09ff466f0454493fc8c74d546d077d` (== pre-experiment), flag var absent from runner
env, runners relaunched with the original env. **Health on shipped config verified
live:** `POST /v1/chat/completions` → "Paris", `finish_reason=stop`; runner env
spot-check `MLX_JACCL_SHARDING_MODE=Tensor`, `EXO_DSV4_FENCE_ASYNC=1`, `EXO_DSV4_MTP=1`,
`grep -c EXO_DSV4_MOE_EARLY_ALLSUM` in live env = 0 on both nodes; screens `exorun`
detached on both; both worker processes up; restored-boot 8-prompt probe coherent.

## Task 3 — doc patch

`docs/dsv4-decode-stall-2026-06-26.md`: dated **CORRECTIONS (2026-09-03)** block added
at the top (56 insertions, body untouched below it), citing round-2 `MECHANISM.md` line
refs. Covers the three misdirecting claims: (1) "overlap primitive exists" — false,
no GPU collective; (2) fence-is-load-bearing — wrong root cause, settled by this
round's experiment (quality leg passes; bit leg fails via reassociation, stated as
such); (3) the 2-per-layer collective framing — `sum_gradients` is identity at
inference (`distributed.py:21` forward `return x`), attn `all_gather` is
prefill-only/length-gated (`deepseek_v4.py:242` "Decode (L==1) and MTP verify skip
it"), and live `EXO_DSV4_ATTN_ALLSUM=0` disables the legacy attn `all_sum`
(`:1713`) ⇒ decode carries exactly 1 collective/layer × 43. Commit: **exo `e1e712a62`** (amended once to state the gate result accurately).

## Reconciliation with prior record

- **Round 1** (`round1/REPORT.md`): transport 2.6% CLOSED — unchanged; "~95% local
  drain" attribution — upheld and extended (this round bounds what any
  notification/dispatch mechanism could recover at ~150 µs max, far under the mass of
  the residual; the async-fence-gate fix's +58–67% remains the strongest natural
  experiment locating the stall in local-drain blocking).
- **Round 2** (`round2/MECHANISM.md`): all five answers upheld; Q3's algebra-bug
  correction now settled on live evidence (quality leg) rather than inference;
  Q1's "two CPU round-trips" characterization now has a second concrete member
  (jaccl's own memcpy bounce). Round-2's "41/43 layers carry TWO collectives" and
  round-1's "no post-attention all_sum" are **both** correct once split by regime:
  attn `all_gather` is prefill-only (length-gated), decode is 1×43 — this is now
  stated in the CORRECTIONS block so the two rounds stop reading as a contradiction.
- **The 06-26 doc**: patched at source (Task 3) — the three claims that misdirected
  investigations are corrected with cites, history preserved below.
- **Consult challenge (recorded honestly):** a second opinion argued the unmeasured
  ~1250 µs residual could be encode/commit/wake overhead that H-e1(c) recovers, and
  proposed two probes. The probes are backend code (prohibited this round,
  constraint 3); they are specified in FEASIBILITY.md as the round-4 falsification
  path. The verdict does not hinge on the residual's exact split: even the most
  generous reading of the notification class (~150 µs, bounded by round-1's measured
  96.8 µs dispatch floor and the ≤100 µs crossing class) is far below the ≥560 µs
  bar.

## Commits / SHAs

| Artifact | SHA / path |
|---|---|
| exo doc CORRECTIONS block | `e1e712a62` (main, `docs/dsv4-decode-stall-2026-06-26.md`) |
| mlx-lm experiment branch | `perf-campaign-2/task2-early-allsum-reorder` @ `1446c5d` (deployed to nodes then reverted; NOT merged; pushed to `adurham/mlx-lm` origin by the worker — see note) |
| mlx-lm on nodes now (shipped config) | `main` @ `37260bbd6` (verified) |
| mlx submodule (unchanged all round) | `e40a416b2` (verified) |
| Round-3 deliverables | `tmp/perf-campaign-2/round3/{FEASIBILITY.md, REPORT.md, W1-…, probe-results/*}` |

**Process note (worker discipline):** the patch worker **pushed its branch to
`origin` (adurham/mlx-lm) despite an explicit no-push instruction** — flagged here
rather than buried. The branch is experiment-scoped, clearly labeled, and not
referenced by any deployed config (both nodes verified back on `main`); no PR was
opened. Recommend the user delete `origin/perf-campaign-2/task2-early-allsum-reorder`
if they want the fork pristine, or keep it — the commit message documents the
experiment.

## Cluster health (final)

Both nodes: runners up in `exorun` screens on shipped config (`main`@`37260bb`, flag
absent, env verified via live `ps`); model placed and serving (capital-of-France →
"Paris"); no segfaults in the post-experiment window; both 200K needle runs and the
restored-boot probe completed cleanly.

## Round 4 (if funded)

NOT backend H-e implementation — that is closed. The only open thread with real
measured upside is the **local-drain class**: (i) the two falsification probes
(GPU-identity-kernel all_sum substitute; command-buffer GPUStartTime/GPUEndTime around
one all_sum) to decompose the ~1250 µs residual, and (ii) the async-fence-gate
family — the record's only ≥50% lever class (2026-08-22: +58–67%). If those probes
show the residual is encode/commit/wake (not compute), a *fence-removal* (not
collective) design reopens with its own pre-registration.