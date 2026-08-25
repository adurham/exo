# Xcode removed from both studios — launcher CLT-fallback fix (2026-08-25)

**Date:** 2026-08-25
**Severity:** deploy-blocking (no model/perf impact)
**Fix commit:** `70e0423bc` (`start_cluster.sh`)
**Affected launch:** arm-A of the `hc_collapse` fused-pre A/B

---

## Symptom

The arm-A launch (`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 ./start_cluster.sh`)
aborted early, before a single line of exo ran:

```
Failed to sync on macstudio-m4-1
```

The launcher had already rsynced the tree to `macstudio-m4-1` and reached the
`Syncing dependencies on macstudio-m4-1...` step. `m4-1`'s existing exo
processes had already been killed by the launcher's own teardown, so the
cluster was left in a split state (see *Cluster impact* below).

## Root cause

Three independent facts had to line up:

1. **`/Applications/Xcode.app` is gone from BOTH Mac Studios.** Only
   `/Library/Developer/CommandLineTools` (CLT) remains. Verified live on both
   nodes (see *Verification* below). Nothing in the exo repo removed it — this
   was an out-of-band change to the machines.

2. **`start_cluster.sh` hardcoded the Xcode path.** Three remote-build sites
   exported
   `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` unconditionally,
   plus `PATH=/opt/homebrew/bin:$(dirname $(xcrun -f metal)):$PATH`. On a node
   without Xcode, `DEVELOPER_DIR` points at a nonexistent directory, so every
   `xcrun`/`cc`/`clang` invocation fails, and `xcrun -f metal` prints nothing —
   making `dirname` fail with a usage error and (in the old form) contributing
   a junk PATH segment.

3. **The uv wheel cache had been masking it.** Prior launches reused cached
   wheels and never shelled out to a compiler at all, so the dead
   `DEVELOPER_DIR` was harmless. Commit `782c8cf97` touched `start_cluster.sh`,
   which invalidated the cache; the very next launch was the first one that
   actually had to *build*, and it failed immediately.

The failure surfaced inside `uv sync`'s isolated **maturin** build of the
`exo_rs` PyO3 extension: `maturin pep517 build-wheel` → `cargo` → `cc` →
`xcrun`. Every Rust build script (`serde`, `libc`, `proc-macro2`, `anyhow`,
`quote`, `zerocopy`, `rustversion`, `zmij`, `serde_json`, …) failed to link.

## Evidence

From `/tmp/hccol_armA_launch.log` (arm-A aborted launch):

```
Ensuring Xcode developer directory on macstudio-m4-1...
Syncing repo to macstudio-m4-1 via rsync (canonical source: adams-macbook-pro-m4)...
Ensuring build dependencies on macstudio-m4-1...
Syncing dependencies on macstudio-m4-1...
xcrun: error: missing DEVELOPER_DIR path: /Applications/Xcode.app/Contents/Developer
usage: dirname string [...]
...
  × Failed to build `exo-rs @ file:///Users/adam.durham/repos/exo/rust/exo_rs`
  ├─▶ The build backend returned an error
  ╰─▶ Call to `maturin.build_editable` failed (exit status: 1)
      Running `maturin pep517 build-wheel -i ...`
      error: linking with `cc` failed: exit status: 1
        = note: xcrun: error: missing DEVELOPER_DIR path:
                /Applications/Xcode.app/Contents/Developer
      error: could not compile `serde_json` (build script) due to 1 previous error
      ...
      💥 maturin failed
        Caused by: Failed to build a native library through cargo
        Caused by: Cargo build finished with "exit status: 101"
Failed to sync on macstudio-m4-1
```

Note the `usage: dirname string [...]` on the line right after the first
`xcrun` error — that is the old `$(dirname $(xcrun -f metal))` PATH segment
failing with an empty argument, an independent second symptom of the same
missing toolchain.

## Verification performed (2026-08-25, both nodes)

| check | macstudio-m4-1 | macstudio-m4-2 |
|---|---|---|
| `xcode-select -p` | `/Library/Developer/CommandLineTools` | `/Library/Developer/CommandLineTools` |
| `/Applications/Xcode.app/Contents/Developer` exists | NO | NO |
| `xcrun -f metal` | `error: unable to find utility "metal"` | `error: unable to find utility "metal"` |
| `xcrun -f cc` | `/Library/Developer/CommandLineTools/usr/bin/cc` | `/Library/Developer/CommandLineTools/usr/bin/cc` |

So: CLT is a perfectly valid `DEVELOPER_DIR` for C/C++/Rust linking (`cc` works),
but it ships **no Metal compiler** — only a full Xcode.app does.

## The fix (`70e0423bc`)

1. **Guarded `xcode-select`.** The pre-existing
   `sudo -n xcode-select -s /Applications/Xcode.app/...` call is now wrapped in
   `if [ -d /Applications/Xcode.app/Contents/Developer ]; then ... fi`, so a
   cached sudo credential can never point a node's default toolchain at a
   nonexistent path.

2. **`REMOTE_DEV_ENV` — per-node toolchain prelude, resolved ON THE NODE.**
   A single-quoted shell fragment (so the laptop expands none of it — the
   laptop *still has* Xcode, so deciding locally would reintroduce the exact
   bug) that picks Xcode if present, else CLT:

   ```sh
   REMOTE_DEV_ENV='if [ -d /Applications/Xcode.app/Contents/Developer ]; then
     export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer;
   else export DEVELOPER_DIR=/Library/Developer/CommandLineTools; fi; ...'
   ```

3. **Conditional metal PATH segment.** `dirname $(xcrun -f metal)` is added to
   PATH only when `xcrun -f metal` actually resolves; otherwise the segment is
   omitted entirely rather than poisoning PATH with `dirname`'s `.` fallback.

4. **Three hardcoded `DEVELOPER_DIR` export sites replaced** with
   `"$REMOTE_DEV_ENV"` (uv sync, mlx stamp-check/rebuild, mlx-lm pin).

5. **Fail-fast guard in the mlx stamp-check's `NEED_BUILD=1` branch.** Since
   CLT cannot compile Metal shaders, an mlx C++ rebuild is structurally
   impossible under the fallback. Rather than dying deep inside cmake with a
   confusing error, the launcher now checks `xcrun -f metal` up front and
   aborts with an explicit "reinstall Xcode on this node" banner.

6. **Quoting-bug rework (found in review, before this ever ran).** The
   fail-fast banner's original text contained literal single quotes
   (`has no 'metal' compiler.`, `cmake/clang 'metal: command not found' error.`).
   That text lives inside a single-quoted `zsh -l -c '...'` payload, so on the
   remote shell the inner `'` closes the `-c` string; the second line's now
   unquoted region contains spaces, terminating the `zsh -c` argument
   mid-script → unmatched-quote syntax error → **the whole stamp-check block
   would have failed on EVERY launch**, not just when a rebuild was needed.
   Fixed by removing the inner quotes entirely.

   Validated beyond `bash -n`: a harness replaced `ssh` with a function that
   dumps `"$2"`, with the real `REMOTE_DEV_ENV` and `NODE=testnode`; the inner
   `zsh -l -c` payload was extracted with `shlex.split` and parsed with both
   `zsh -n` and `sh -n`. Post-fix payload = 2213 bytes, 0 single quotes, parses
   clean under `zsh -n`, `sh -n`, `bash -n`. Negative control (same harness
   with the quotes restored) truncates the payload to 1834 bytes and
   `zsh -n` reports `unmatched "` — confirming the bug was real and the test is
   load-bearing.

## Residual risk

**Any future mlx C++ rebuild requires reinstalling Xcode on the studios.**
The CLT fallback makes Python/Rust deploys work, but it cannot build MLX's
Metal kernels. If the `mlx` submodule pin ever moves (or `MLX_FORCE_REINSTALL`
is set, or the venv stamp goes stale), the launcher will **abort loudly** at
the stamp-check with the reinstall-Xcode banner instead of silently producing a
broken build or burning ten minutes inside cmake. That is by design — but it
means a pin advance is now gated on someone reinstalling Xcode first.

The mlx pin is currently `e40a416b2` and both studios' venvs carry a matching
good build, so `NEED_BUILD` stays 0 and the guard does not fire on ordinary
launches.

## Cluster impact

- **macstudio-m4-1:** its exo processes were killed by the aborted launch's own
  teardown before the sync failure, leaving the node with nothing running.
- **macstudio-m4-2:** never reached (the launcher exits on the first node's
  failure), so it kept its **stale** pre-launch build running.
- **Recovery:** none needed beyond the next full relaunch, which tears down and
  rebuilds both nodes from scratch. No model weights, venv state, or checked-in
  code was corrupted — the failure was entirely in the build step.
- **No performance impact.** No model or kernel code changed; this is a
  launcher-only fix.

## Lessons

- A hardcoded absolute toolchain path in a deploy script is a latent failure
  that a build cache can mask for an arbitrarily long time. The trigger was an
  *unrelated* edit to `start_cluster.sh` invalidating that cache — the
  correlation between "the commit that broke it" and "the commit that exposed
  it" is misleading.
- Environment decisions about a remote node must be **resolved on that node**.
  The laptop still had Xcode; any local `if [ -d ... ]` would have been right
  about the laptop and wrong about both studios.
- Nested-quoting in `ssh "$NODE" "... zsh -l -c '...'"` payloads cannot be
  validated by `bash -n` alone — `bash -n` checks only the *outer* script. The
  only trustworthy check is to reconstruct the exact argument bash emits and
  parse the extracted inner payload with the shell that will actually run it.
