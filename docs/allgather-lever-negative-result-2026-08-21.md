# attn.all_gather lever test: negative result — 2026-08-21 (session 2, part 3)

## Hypothesis tested

The sync-span kernel breakdown (`docs/prefill-sync-span-kernel-breakdown-2026-08-21.md`)
showed `attn.all_gather` at 8.5% of prefill wall time. Reading
`mlx-lm/mlx_lm/models/deepseek_v4.py`'s seq-split attention code revealed
this cost comes from a workaround: instead of a real subgroup
`all_gather` to reconstruct per-rank output bands, the default path
(`EXO_DSV4_SEQSPLIT_GATHER_VIA_ALLSUM=1`, i.e. unset) zero-pads and uses
`all_sum` on the top-level group at ~2x wire bytes. The code comment
explains why: subgroups have no TCP coordinator, so the reliable ARQ
can't arm there, and large-L bands intermittently hit a UC stuck-send
wedge (observed 2026-07-06, `all_gather STALLED` → failed subgroup
reconnect → full re-place).

Tonight's earlier jaccl work (commit `f8b77fe5a`, "give subgroups their
own TCP coordinator") looked like it might have retired exactly that
precondition. Hypothesis: flip `EXO_DSV4_SEQSPLIT_GATHER_VIA_ALLSUM=0` to
use the real (cheaper, ~half the wire bytes) subgroup `all_gather`, and
see if it's now safe and reduces the 8.5% `attn.all_gather` cost.

## Method

1. Plumbed `EXO_DSV4_SEQSPLIT_GATHER_VIA_ALLSUM` through `start_cluster.sh`
   (previously settable only by editing `deepseek_v4.py` directly — not
   how this repo makes runtime changes). Committed as `b08111e4b`.
2. Relaunched with `EXO_DSV4_SEQSPLIT_GATHER_VIA_ALLSUM=0` (real subgroup
   all_gather) plus tracing/sync-span profiler still on for comparison.
   Verified the flag was live in both nodes' process env via `ps aux`
   before testing.
3. Ran a single 100K-context prefill request via the same benchmark
   script used all session.

## Result: negative — the wedge still reproduces

The real subgroup `all_gather` faulted almost immediately into prefill
(first chunk, ~0.4s after "Starting prefill"):

```
[mlx scheduler] captured St13runtime_error in task (surfacing at next
synchronize): [jaccl] all_gather wc.status=1 wr_id=0x80001 byte_len=4096
jaccl transport fault in generator.step(): [jaccl] all_gather wc.status=1
wr_id=0x80001 byte_len=4096. Attempting in-place reconnect (both ranks)
to avoid a re-place.
```

The runner's in-place-reconnect path caught it cleanly (no crash, no
re-place, `jaccl reconnect complete` within ~0.3s) — but the in-flight
generation request was lost: the benchmark returned `0.0 tok/s prefill`,
empty response, needle check FAIL. This is a clean, fast, disposable
negative result (total test time ~2s to failure), not a hang/wedge that
would have needed a full reboot to clear — confirms the value of testing
small before committing to a change.

## Conclusion

**Tonight's TCP-coordinator fix (`f8b77fe5a`) does NOT cover this fault
path.** The subgroup `all_gather`'s reliability problem is still present
post-fix. Reading the wc.status/wr_id in the fault message
(`wc.status=1`, an ibverbs work-completion error code, not a timeout) — this
looks like the same class of raw-UC transport fault the comment
describes, not a new or different failure mode. The existing
`EXO_DSV4_SEQSPLIT_GATHER_VIA_ALLSUM=1` (all_sum) workaround remains
necessary and should stay the default. **Do not flip this flag without
first fixing the underlying subgroup all_gather reliability gap** — that
would be new jaccl transport work, not a config toggle, and is
out of scope for tonight.

The `attn.all_gather` 8.5% prefill-wall-time cost is confirmed as real,
necessary cost under the current (safe) all_sum-based reconstruction —
not a leftover inefficiency from an already-fixed bug. This closes out
that specific lever as a dead end for now; the ~2x-wire-bytes trade-off
buys correctness that the real all_gather still can't provide.

## Cluster state after this test

Reverted immediately via a clean `./start_cluster.sh` relaunch with no
override env vars — confirmed via `ps aux` on both nodes:
`EXO_DSV4_SEQSPLIT_GATHER_VIA_ALLSUM` absent (default/all_sum path
active), `EXO_TRACING_ENABLED=false`, `EXO_PROFILER` absent. Cluster is
back in normal production configuration. Both `exo` and `mlx` repos
remain clean and pushed (commits `b08111e4b` exo, unchanged mlx —
this was a config-only test, no mlx submodule changes).

## Remaining unexplored lever from the kernel breakdown

`moe.switch_mlp` (30.0% of prefill, the single largest span) is
undecomposed below the Python span level — its own named sub-ops sum to
<0.5% of its own total, meaning >99% of that cost is inside the actual
GatherQMM/expert-matmul kernel call. Going further here needs Metal-level
tracing (Instruments / Metal System Trace), not Python spans — a
different, larger investigation than tonight's, not started.
