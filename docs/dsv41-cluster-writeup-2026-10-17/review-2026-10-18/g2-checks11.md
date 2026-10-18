# G2 — static lint of §11 (`scripts/check-s11.sh`)

Scope: make §11's reproduction instructions machine-checkable without a cluster.
Not a repro run. Nothing was launched; no node was touched.

## What was built

- `scripts/check-s11.sh` — Python 3 (stdlib only, 3.9+), despite the `.sh` name (the
  deliverable named it that way; shebang + exec bit make `./scripts/check-s11.sh` work).
- `.github/workflows/check-s11.yml` — runs it on push / PR / dispatch, `--check-only`.

## Checks, and what each actually proves

1. **Pinned SHA resolves.** Reads the SHA out of §11.2's `exo` table row (not a
   constant in the script — if §11 re-pins, the lint follows), requires full 40-hex,
   then `git cat-file -e <sha>^{commit}`. Rejects abbreviated SHAs explicitly.
   *Note:* the `mlx` / `mlx-lm` row SHAs are foreign to this repo and cannot be
   `cat-file`'d here; check 5 verifies them against the submodule gitlinks instead.
2. **Paths.** 9 references (7 distinct paths + the `docs/` corpus tree). Extracted
   from backticked tokens in §11, resolved with `git ls-tree <sha>` (exact, then
   basename fallback). Annotated `file:NN-MM` refs must also be long enough to
   contain those lines. §11.2's elided form ("`uv.lock:1382-1384` for mlx,
   `:1499-1501` for mlx-lm") is handled by a single document-order pass, so the
   second range attaches to `uv.lock` and not to whatever file is named later.
3. **CLI flags.** §11 itself names **zero** exo-level flags — §11.3's launch is
   `EXO_TARGET_BRANCH=main ./start_cluster.sh`, and the line that script actually
   runs (`start_cluster.sh:3573`) is `python -m exo -v`. So the check does three
   real things rather than vacuously passing:
   - extracts `parse()`'s source from `git show <sha>:src/exo/main.py` and **execs it**
     (with `default_namespace` extracted too, and a namespace holder standing in for
     `ExoConfig.__init__`) — the real argparse definition at the SHA, no exo import,
     no launch, no heavyweight deps;
   - feeds it the argv §11 inherits (`-v`) plus the bare invocation; both must parse;
   - cross-checks §11's stated facts against the parser's defaults: "port 52415" ==
     `--api-port` 52415, and `start_cluster.sh`'s `EXO_ZENOH_PORT` 52414 /
     `EXO_DISCOVERY_PORT` 52413 == `--zenoh-port` / `--discovery-port` defaults.
   Also: the 8 curl flags §11 uses (bundled `-sS` expanded) checked against the
   local `curl --help all`; git's `--init`/`--recursive` against `git submodule
   update -h`.
4. **Prompt pipeline — SKIPPED today.** `scripts/gen-prompts.py` appeared mid-run
   (sibling); `scripts/validate-prompts.py` has not. Skip is loud and named, and the
   status line records it. The machinery was verified in isolation (temp dir, stub
   generator/validator): happy path → PASS, validator rejecting → FAIL, generator
   writing nothing → FAIL. When both files land the check runs for real with
   `--out <tmpdir>` (the generator does take `--out`, confirmed).
5. **Hardcoded constants.** §11 claims `NODES` at `:1113` and peer constants at
   `:1025-1028`. Verified at the SHA: line 1113 is `NODES=(macstudio-m4-1
   macstudio-m4-2)` (2 entries, matching §11's "two nodes"); 1025-1028 hold exactly
   2 `_PEER_ID` (52-char `12D3Koo…`) and 2 `_IP` dotted quads. Also cross-referenced:
   `DSV4_WIRED_LIMIT_MB:-120000` == §11.1's `wired_limit_mb = 120000`; `mlx`
   gitlink == table `b1778183…`; `mlx-lm` gitlink == table `676f803e…`;
   `uv.lock:1384` / `:1501` carry those pins; `pyproject.toml:90` is the darwin fork;
   `reasoning_budget.py:91-92` still the uncapped pass-through §11.7 cites.

## Status line

Appended at the end of §11 (before the `---`), idempotent: a line matching
`^_Static-checked at .*_$` is **replaced**, never stacked (verified: run twice → 1
line, `REPLACED`). Diff to the writeup is +2 lines.

Wording is deliberately constrained: "static only — nothing was launched, so this
does not show that the SHA boots on this hardware, that the hardcoded node/peer
constants fit your cluster, or that any §9 figure reproduces".

`--check-only` (CI) never writes; it fails when the line is missing or names a
different SHA than §11's table — that is the anti-staleness ratchet.

## Negative tests (the lint bites)

- mutate a path → `CHECK 2 FAIL missing at b88d65dd3304: scripts/exo_graceful_shutdownX.sh`
- mutate "port 52415" → `CHECK 3 FAIL §11 says the API listens on 52416 but --api-port default is 52415`
- mutate `NODES` line ref → `CHECK 5 FAIL §11 puts NODES at start_cluster.sh:1100 but it is at [1113]`
- mutate `mlx` table SHA → `CHECK 5 FAIL mlx gitlink is b1778183…ff8 … table says …ff9`
- bogus SHA → `CHECK 1 FAIL`, checks 2-5 SKIP with "the pinned SHA does not resolve"

## CI workflow, verified cheap

`.git` here is 2.8 GB, so `fetch-depth: 0` would be wasteful. The workflow checks out
at depth 2 and shallow-fetches only the commit §11 pins (`git fetch --depth 1 origin
$SHA`), extracted from §11 by an inline step. Simulated locally against this repo:
the shallow fetch produced a 13 MB `.git` and every primitive the lint uses
(`cat-file -e`, `ls-tree`, `show <sha>:<path>`, `rev-parse`, `log -1`) worked; the
full lint then ran in that shallow clone and exited 0 with the status line current.
No secrets, no cluster, standard ubuntu runner, <1 min → $0.

## Honest limits (repeat of the point)

PASS == §11 is well-formed and points at things that exist at the SHA it pins.
PASS != that SHA boots on two Mac Studios; != the hardcoded IPs/peer IDs are right
for anyone; != the model fits; != any §9 figure reproduces. Those still need the
hardware and the relaunch this task explicitly did not do.

## Files

- `scripts/check-s11.sh` (new, executable)
- `.github/workflows/check-s11.yml` (new)
- `docs/dsv41-cluster-writeup-2026-10-17/WRITEUP.md` (+2 lines: status line + blank)

Not committed (per instructions).

END G2
