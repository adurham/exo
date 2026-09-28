# Phase 6 — STRICT-FALSE-LOAD-GUARD made rename-aware (2026-09-28)

**Bottom line.** The `strict=False` main-model-load guard was disabled on
exactly the model it exists to protect. It now works, proven live on node1
with real mlx, and it catches a genuinely dropped weight while staying silent
on a healthy load.

## The defect

exo loads every model it serves with `load_model(model_path, lazy=True,
strict=False)`. That means a checkpoint/model mismatch silently proceeds with
random/default init for whatever didn't load — the same failure shape as the
DSpark checkpoint-key-mismatch incident, but on the main load, for **100% of
production traffic**.

`_run_strict_false_load_guard` was added (2026-09-15 audit) to make that
visible. But it diffed the **raw on-disk key set** against the model's
**parameter tree**. For any architecture defining `Model.sanitize()` — which
renames keys before load — those two are different namespaces, so the diff
would report a large *expected* mismatch on every load.

The first fix for that was to **skip** the diff for such architectures. That
was correct as far as it went (no false alarms) but the skip condition is
`hasattr(model, "sanitize")` — and **DeepSeek-V4 defines `sanitize()`**.

So the guard was silently disabled on:
- the model this cluster actually serves (`DeepSeek-V4-Flash-Vision-Exp`),
- the model family the whole V4.1 port targets,
- i.e. precisely where the strict=False random-init hazard is live.

`_log_dspark_load_guard` (the sibling guard for the optional DSpark draft
head) was separately mis-firing — see `72c678ee`.

## The fix

Capture **both sides of the comparison at load time**, then diff those.

`_RecordLoadedWeightKeys` is a context manager wrapping the *real*
`load_model(...)` call at both call sites (`load_mlx_items` single-device,
`shard_and_load` distributed). It patches `nn.Module.load_weights` and
records, from one instant:

- `keys` — the exact key set delivered to `load_weights`
  (post-sanitize by construction: it is the same set mlx-lm hands over), and
- `params` — the module's own param-tree keys, read **before** the real call,
  which is exactly the value mlx's own `strict=True` branch snapshots as
  `curr_weights`.

That pair *is* the comparison mlx's own `strict=True` performs. Diffing it
needs no knowledge of what any `sanitize()` does, so it is namespace-correct
for every architecture, DeepSeek-V4 included.

Cost: one dict comprehension while the weights are already in hand. **No
second load** — an earlier draft called `load_model` twice to observe it,
which on this cluster would have re-read ~156 GB per load. Do not restore
that shape.

`_run_strict_false_load_guard` now:
1. uses the captured pair when both sides are present (`source=post-sanitize`);
2. otherwise falls back to the raw on-disk diff **only** for architectures
   with no `sanitize()` (`source=raw-on-disk`);
3. otherwise logs one INFO that the raw diff was skipped, and why.

The log line names its namespace (`source=...`) so a reader can immediately
tell which comparison fired.

## Why the capture beats hooking `sanitize()` itself

The previously-documented follow-up was "hook `sanitize()` to capture its
key-rename map". Hooking `load_weights` is strictly better:

- it needs no knowledge of *what* `sanitize` does (no per-architecture
  handling, no maintenance as architectures change);
- it is the same namespace mlx-lm itself compares against in `strict=True`;
- it also survives `sanitize()`-adjacent transformations (fp8 dequant,
  expert stacking, `wo_a` reshape, top-level remaps) without enumerating them.

## Robustness details that are load-bearing

- **Union, not replace.** If a load path ever calls `load_weights` more than
  once, the union of everything delivered is the right set (the question is
  "did every parameter get filled").
- **Params read before the call.** `EXO_DSV4_LMHEAD_MXFP8=1` quantizes
  `lm_head` in place *after* `load_weights`, adding `.scales`/`.biases`;
  reading the tree afterwards would false-alarm there. Reading it at the
  same instant mlx does avoids both that and any update-added keys.
- **`tree_flatten(..., destination={})` returns a dict.** Without
  `destination` it returns a list of `(key, value)` tuples. The capture
  originally iterated it as pairs and raised — *silently*, because the call
  was inside a `contextlib.suppress`. Now it reads `.keys()` under a `cast`,
  and the failure path **logs a warning** instead of suppressing, so a
  regression here cannot quietly downgrade the guard to its fallback.
  (Caught by the new tests, which is why they exist.)
- **Never raises.** Same contract as the sibling DSpark guard: a guard that
  can crash a model load is worse than the hazard it reports.

## Evidence

`scripts/p9_live_guard_proof.py` — run on `macstudio-m4-1` with real mlx
against a tiny DeepSeek-V4 built from the fork's own test config
(it goes through the real `sanitize()` and the real `load_model`):

```
=== (0) build tiny DSv4 + export its weights ===
    30 tensors, e.g. ['lm_head.weight', 'model.embed_tokens.weight', 'model.hc_head.base']

=== (A) HEALTHY load: expect ZERO guard lines ===
    captured=30 keys; guard lines=0 -> PASS

=== (B) DROPPED key: expect ERROR naming it ===
    dropping 'model.layers.0.attn.kv_norm.weight'
   [ERROR] [STRICT-FALSE-LOAD-GUARD] ...: strict=False model load did NOT exactly
   match the post-sanitize weight set's keys — missing=1 extra=0 (model expects
   30 keys total, the sanitized set carried 29; source=post-sanitize). Sample
   missing=['model.layers.0.attn.kv_norm.weight'] extra=[]
    ERROR lines=1; names victim=True -> PASS

=== (C) PRE-sanitize names on disk: raw diff lies, captured diff is right ===
    rewrote 1 key(s) to pre-sanitize names
    raw on-disk  diff: missing=1 extra=1
    captured set diff: missing=0 extra=0
    guard lines=0 -> PASS (silent)

OVERALL: PASS
```

Case (C) is the money shot: with pre-sanitize key names on disk — the exact
situation that forced the old skip — **the raw diff false-alarms (1 missing,
1 extra) while the captured diff is correctly clean.** That is the guard
working on DeepSeek-V4 for the first time.

Full captured output: `raw/live_proof_output.txt`.

Regression tests: `src/exo/worker/engines/mlx/tests/test_strict_false_load_guard.py`
— 20 tests, all passing on node1 with real mlx (`20 passed in 0.26s`),
including three new tests that exercise the capture mechanism itself against
real mlx (both-sides capture, method restoration on exit, and
capture/param namespace identity).

## Verification ledger

| check | result |
|---|---|
| `pytest test_strict_false_load_guard.py` (node1, real mlx) | 20 passed |
| live proof, tiny real DSv4 (node1) | A/B/C all PASS |
| `ruff check` (gateway) | All checks passed |
| `basedpyright` (gateway) | 199 errors = exact pre-existing baseline, zero delta |
| node1 restored bit-exact after the proof overlay | md5 `c162c2f29cb16ad5c8ad27cfa1918a95`, empty git diff |
| production cluster during all of this | process 61922 etime `04-01:39:46`, API 200, never relaunched |

## Not done here (deliberately)

- **Acceptance-rate measurement** via production telemetry. Enabling
  `EXO_DSV4_MTP_LOG_INTERVAL` requires a production relaunch — the user's
  decision, not bundled into this fix.
- The guard is **log-only** and stays that way. Flipping `strict=True` blind
  on the main load would turn a currently-healthy production path into a
  coin-flip outage; see `_log_strict_false_load_guard`'s docstring.
