# PRE-REGISTRATION: fresh-boot control — chunk size vs boot state (2026-09-23)

Written and committed BEFORE the measurement. Bands apply verbatim.

## The confound being resolved

Two variables were changed between the "collapse" control runs and the "fixed"
adaptive run, and they cannot be separated from the existing data:

| run | boot state | prefill chunk (past 500K) |
|---|---|---|
| control r0 | long-lived (4 leaves, baseline 107.32 GB) | 2048 |
| control r1 | long-lived (baseline 114.67 GB) | 2048 |
| **adaptive r0** | **fresh (baseline 87.83 GB)** | **128** |
| adaptive r1 | — | FAILED (watchdog kill, unusable) |

An independent review decomposed adaptive r0's 30 GB peak reduction as ~19.5 GB
boot state + ~10.5 GB chunk size (~2/3 boot, 1/3 chunk). That decomposition is a
*model*, not a measurement. This run makes it a measurement.

## Design

You cannot reproduce an aged boot deterministically in hours (leaves accumulate
slowly, fragmentation is irreproducible), but you CAN reproduce a fresh boot. So
run the A/B inside the controllable state:

- **fresh + 128 chunks** — already measured (adaptive r0: 25.75 t/s, 17,010
  page-ins, peak 87.87 GB)
- **fresh + 2048 chunks** — THIS RUN

The boot-state effect then falls out as (long,2048) - (fresh,2048) using the
existing control runs.

### How to disable the chunk shrink (verified reasoning, not assumed)

`start_cluster.sh` uses `: "${EXO_PREFILL_STEP_SIZE_HIGH_CTX:=128}"`. Because
`:=` assigns when the variable is **unset OR empty**, exporting an EMPTY value
would silently restore 128 and the experiment would be a null test.

Correct disable: `EXO_PREFILL_STEP_SIZE_HIGH_CTX=0`. In mlx_lm/generate.py:
```
_step_high = int(os.environ.get("EXO_PREFILL_STEP_SIZE_HIGH_CTX", "0"))   # -> 0
_adaptive  = _step_high > 0 and _crossover > 0 and _step_low > _step_high # -> False
```
so `_chunk = prefill_step_size` = 2048 for the whole prefill — the control
behaviour, and the launcher does forward a literal "0" (`[ -n "0" ]` is true).

## Pre-registered bands (apply verbatim)

**Validity gate (run is INVALID if unmet):**
- `[MEM] before prefill` baseline must be **87.8 ± 2 GB**. Materially higher
  means the boot was not fresh and it is not a clean comparison.

**Decision table:**
- **Peak ~97-100 GB, page-ins near the low end (~17K), t/s ~25**
  -> FRESH BOOT ALONE SUFFICES at this depth. The adaptive run's gain was mostly
  boot state; keep the chunk change as creep-absorption margin only.
- **Peak > 115 GB, page-ins >= 40K, t/s ~12-15**
  -> CHUNK SIZE IS LOAD-BEARING even on a fresh boot; the transient model was
  underestimated.
- **Peak 100-115 GB, partial degradation**
  -> BOTH contribute; ship both and add a headroom guard.

**Quality gate:** `needle_hit` must be True, else the run is invalid regardless
of the numbers.

## Secondary reads (recorded, not gated)

- Prefill rate (chunk size should affect this: 128-chunk vs 2048-chunk).
- Peak memory AND the `[MEM]` before/after pair, so the transient is separable
  from the baseline.
- Page-ins, as the cross-boot comparator (an event count, not a level).

## Mandatory restore

After the run, relaunch with PRODUCTION defaults
(`EXO_PREFILL_STEP_SIZE_HIGH_CTX=128`, `CROSSOVER=500000`) and verify
mechanically from the live process env on the node — not from the boot log.
The cluster must not be left in the experiment config.
