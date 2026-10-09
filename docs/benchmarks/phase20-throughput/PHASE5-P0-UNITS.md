# PHASE 5 — P0 unit reconciliation (per-pass vs per-round)

Author: Phase-5 P0 subagent. Opened 2026-10-08. **Bench-only, OFFLINE.** No cluster access (no ssh,
no relaunch, no API POST); source-read + arithmetic only. Committed on branch
`deploy/phase20-campaign` (worktree `/private/tmp/phase20-campaign`).

Purpose: fix the **per-pass vs per-round tangle** in `PHASE4-P5-ROOFLINE.md` (§2.2/§2.4) before P1
(lever-2) and P2 (dense EXL3) build on it. A wrong unit here flips the P2 route: the campaign
budget framed the choice as *"4 passes → dense ≈240 GB/s → 2.1× (bf16 dequant-cache) —vs— 1 pass →
8.5× (GEMV tuning)"*. **The code says the second framing is the correct one**, and this doc shows
why, with `file:line` for every claim.

Sources read (read-only):
- engine, exo `deploy/next18-identity @ 576e9d279` — `/private/tmp/next18-exo`:
  `src/exo/worker/engines/mlx/dsv41/{rounds,engine,load}.py`
- mlx-lm `deploy/next18-lever2 @ 938b811` — `/private/tmp/next18-lever2`:
  `mlx_lm/models/deepseek_v41/{model,attention,mtp,indexer,_gates}.py`
- PROF-capable build, exo `deploy/next16-instr` — `/private/tmp/next16-instr`:
  `src/exo/worker/engines/mlx/dsv41/rounds.py` (the frozen `verify_block 92.4 / round_total 93.8`
  brackets)
- prior doc: `PHASE4-P5-ROOFLINE.md`; baselines `PHASE3-M3.md`, `PHASE3B-SHIP-VALIDATION.md`.
- companion Phase-5 measurement (concurrent): `PHASE5-P2-DENSE.md` (real-weight dense rate + byte census).

> **Path correction to the dispatch brief.** The brief points at
> `mlx_lm/models/deepseek_v41/rounds.py` for the `_one_round` bracket / the `t_draft 591` /
> `t_verify 622` / `t_tail 655` timer lines. That file does **not** exist in the mlx-lm fork
> (`find /private/tmp/next18-lever2 -name rounds.py` → ∅). The round definition lives in the **exo**
> worker: `src/exo/worker/engines/mlx/dsv41/rounds.py`. The timer lines land at
> `next16-instr …/rounds.py:591/622/655` (the PROF build) and the same statements are at
> `next18-exo …/rounds.py:408/419–429` (no PROF timers in this build). All citations below use the
> exo path.

---

## 0. HEADLINE — the brief's premise is wrong; the P5 doc's `2.04 GB` stands

The brief asks to confirm that a decode round contains **3× m=1 (draft) + 1× m=4 (verify) = 4
indexer passes**, and therefore that per-round dense bytes are `4 × 2.04 = ~8.2 GB`. **The code
does not support this, on two independent grounds:**

1. **There is exactly ONE body forward per round, at m=4.** The engine calls the 40-layer body
   model **once** per round: `model(verify_in, …)` with `verify_in = concat([anchor], drafted)`
   shape `[1, 1+gamma] = [1, 4]` (`rounds.py:419`, `:424–427`).
2. **The "3 draft forwards" are not body forwards at all.** The draft is a **separate 3-stage
   `DSparkHead`** (`mtp.py:255`), *one parallel forward* over `[anchor, noise×(block−1)]`
   (`mtp.py:295–313`) — it does not run the 40-layer body, and it contains **no `Indexer` object**
   (draft attention is local-only, `mtp.py:18–19`).

Consequences, each of which the rest of this doc derives from source:
- **passes/round = 1** (a single m=4 body/indexer pass), **not 4**;
- per-round dense EXL3 bytes = **2.04 GB** (unchanged from the P5 doc), floor **≈4.1 ms @497 GB/s**,
  **not** 8.2 GB / 16.5 ms;
- the dense slice therefore sits at **~9.4× its floor (~53–60 GB/s effective)**, **not ~2.1×/~240
  GB/s** — i.e. the P5 doc's `8.5×` is right after all;
- the brief's **52 % / 56 ms arithmetic (iv) checks out** and the **KV-share (iii) correction is
  valid (2.23 %, not 0.2 %)** — those two parts of the brief are correct; only (i)/(ii) are not.

If the brief's `8.2 GB` were used downstream, P2 would go measure a *4×-inflated* dense byte figure
and pick the wrong route (the bf16-dequant-cache route that assumes dense is near-floor). The
correct route is the GEMV/latency route the 1-pass framing implies.

**Second opinion taken** (`consult`): independently reproduced this reading from the same code
facts and concurred — "the brief is wrong … 1 body pass per round, ~2.04 GB dense bytes/round,
Indexer invoked only at n=4"; and specifically that the draft's local-attention einsum "is not an
Indexer pass".

**Independent measurement corroborates this (Phase-5 P2, commit `41ad0e3fa`).** The concurrent P2
subagent measured the dense slice on **real weights** (`PHASE5-P2-DENSE.md`) and reached the same
conclusion by a different route — **"Passes/round for the dense slice = 1 (the m=4 verify)"**
(`PHASE5-P2-DENSE.md:117`) — with an internal-consistency falsifier: 4 dense passes would cost
`3×39.5 + 51.2 = 169.7 ms/round > the whole 93.8–101.06 ms round → impossible` (`:121–124`). P2 also
picked the route the 1-pass reading implies: **Reading 2 (GEMV/latency tuning)**, Reading 1's bf16
dequant-cache measured **dead** (`:133–160`). So this is two independent source readings + one
independent real-weight measurement, all agreeing against the brief's `×4`.

---

## (i) How many indexer/score passes run per decode round, at m=1 and at the m=4 verify

**Answer: ONE pass per round, at m=4. The draft path does NOT call the Indexer module. There are no
m=1 indexer passes in the steady-state round (gamma=3).**

### The draft-forward call — `head.draft(...)` is NOT a body forward

| what | where |
|---|---|
| engine calls the draft **once** per round | `rounds.py:408` `drafted = head.draft(anchor.reshape(-1), model.embed, model.head, draft_state, width=gamma)` |
| `head` is the DSpark head, not the body | `load.py:207–237` (`build_draft_head` → `loaded.head = head`); `head = self.loaded.head` at `engine.py:981` |
| the draft is `DSparkHead._draft` | `mtp.py:290` (`def draft`) → `mtp.py:295` (`def _draft`) |
| **one parallel forward**, not gamma sequential steps | `mtp.py:300–313`: `block_ids = concat([anchor, noise×(bs−1)])`; `for stage, c in zip(self.stages, caches): x, pre_mix = stage(x, pre_mix, c)` |
| draft is a **3-stage** head | `mtp.py:255` (`class DSparkHead`), `n_stages = args.n_mtp_layers or 3` (`mtp.py:261`) |
| draft attention is **local-only, no indexer** | `mtp.py:18–19` `"draft attention is LOCAL ONLY — pure sliding window, no compressed-KV / indexer / candidate machinery (compress_ratios[40:43] == [0,0,0])"`; `DraftAttention.draft_block` `mtp.py:136` is a dense bidirectional einsum over its own rotating window |
| `mtp.py` imports **no** indexer module | verified: `grep -n "index\|Indexer" mtp.py` → only the docstring lines 18–19 |

So the 3 gamma draft rows are produced by **one** forward through a **3-stage draft head**, and
`gamma` is the number of *tokens produced*, not a pass count (`mtp.py:26`: "ONE parallel forward …
producing base logits for every block position, then a rank-markov … sequential sampling loop" —
the sequential part, `mtp.py:322–333`, is a host-free `mx` argmax loop with no forward).

### The verify-forward call — the only body forward, at m=4

| what | where |
|---|---|
| verify rows assembled | `rounds.py:419–422` `verify_in = mx.concatenate([anchor.reshape(1,1), drafted.reshape(1,gamma)], axis=1)` → `[1, 1+gamma]` = `[1, 4]` |
| **the** body forward (one call) | `rounds.py:424` / `rounds.py:427` `logits, taps = model(verify_in, cache, return_taps=True, argmax=True)` |
| its single sync | `rounds.py:429` `mx.eval(logits)` |
| body = 40 layers, each calls `self.attn` | `model.py:277` `for layer in self.layers:` → `Block.__call__`/`_fused_call` → `self.attn(...)` at `model.py:124`/`:142` |
| `Indexer` **instantiated** only here | `attention.py:118–119` `if self.is_index_source: self.indexer = Indexer(args, layer_id)` |
| `Indexer` **invoked** only here | `attention.py:187,197` `if self.is_index_source: … cidx = self.indexer(x, qr, start_pos, offset, self._freqvec, index_k, shared)` |
| the row-count guard (lever-2) | `indexer.py:591` `if _HIER and n > _FENCE_MIN_ROWS:`; threshold `_gates.py:43` `_FENCE_MIN_ROWS = int(env("DSV41_SPARSE_FENCE_MIN_ROWS", "16"))` |

Within the single `model(verify_in, …)` call, the body runs all 40 layers (`model.py:277`). On every
**index-source** layer, `Attention.__call__` calls `Indexer.__call__` **once** (`attention.py:197`).
The index-source set is `args.index_source_layers` (`attention.py:115,118`) and the repo documents
it as **8 layers (2/8/14/20 + 24/28/32/36)** (`indexer.py:5–9`; `PHASE4-CAMPAIGN.md:211` "consumer
index layers (24/28/32/36)"). So per round the Indexer runs **8 times — once per index-source layer
— all at `n = 4`, inside the one verify forward**. There is **no** second Indexer invocation set and
no m=1 invocation set.

At `n = 4`, `_HIER and n > 16` is **False** → the **fallback** path runs (`indexer.py:591` else →
tiled path `indexer.py:~615`); the hierarchical path only runs at prefill (`n > 16`). This is the
lever-2 mechanism, and its correctness at `n = 4` is exactly what `tests/test_dsv41_indexer_smallm_hier.py`
tests.

### Why the "3× m=1" reading is a category error

- **`gamma` ≠ passes.** gamma=3 is one *token-count* emitted from **one** draft forward and
  **one** 4-row verify forward.
- **The draft and the body are different modules.** `head.draft` → `mtp.DSparkHead` (3 stages,
  local-only attention, no Indexer). `model(verify_in)` → the 40-layer body (which owns the
  Indexer). Multiplying a **body-weight byte figure** (the P5 `2.04 GB`) by **draft-head activity**
  is a category error: even if the draft ran 3×, the correct extra term would be *draft-head* weight
  bytes (a separate, much smaller quantity), not `3 × 2.04 GB`.
- **The suite's `n=1` cells are guard-boundary tests, not a pass census.** The Phase-4 suite
  iterates `N = (1, 2, 3, 4, 16, 17)` (`tests/test_dsv41_indexer_smallm_hier.py:225`) because the
  guard must be *correct* at `n=1` and `n=4` — not because production runs three m=1 passes. In
  production, `n=1` on the body occurs only in the **greedy** path (`head=None`, `rounds.py:367`) or
  the round-1-with-head warm-up (`rounds.py:389`); the steady-state `gamma=3` round is always the
  m=4 verify.

**→ Passes per round = 1 (one m=4 body/indexer pass). Not 4.**

**Corroboration & one nit.** P2 (`PHASE5-P2-DENSE.md:117–120`) agrees the dense stack runs **1
pass/round**, and says the "3× m=1 + 1× m=4" pattern "applies to the tiny indexer/score sub-module,
not the full 40-layer dense stack". **That parenthetical is itself not source-supported**: the m=1
draft forwards run `mtp.DSparkHead`, which contains **no `Indexer`** (`mtp.py:18–19`; `grep -n
"index\|Indexer" mtp.py` → the docstring lines only), so there are **no** m=1 indexer passes either.
The correct and simplest statement is: **the Indexer runs once per index-source layer inside the one
body forward, and the body runs once per round — 1 pass, at n=4.** (P2's headline — 1 pass, Reading-2
route — is right; only its concession to the indexer sub-module is too generous.)

---

## (ii) Per-round dense bytes/floor — the `2.04 GB` is per-PASS *and* per-ROUND

The P5 §2.2 table is correct to label `2.04 GB` as **"bytes READ in one verify round (m=4)"** —
because there *is* only one (m=4) verify per round, "per m=4 pass" and "per round" are the **same
thing**. `PHASE4-P5-ROOFLINE.md:176`:

```
dense / shared / attention EXL3 = 141.0 M params/layer × 40 layers × 2.9 bits/8 = 2.04 GB
```

Dense/shared/attn **weight** bytes are read once per body forward **regardless of row count** —
`m=1` vs `m=4` changes activation compute, not weight streaming. So per pass = per round = 2.04 GB.

### Arithmetic, explicitly (per-pass vs per-round)

```
dense bytes per body pass  = 141.0e6 × 40 × 2.9/8  = 2.0445e9 B      = 2.04 GB
passes per round           = 1                                      (see §(i))
dense bytes per round      = 2.04 GB × 1            = 2.04 GB
```

Floor at the **MEASURED 497 GB/s** (`PHASE4-P5-ROOFLINE.md:210–216`):

```
per pass (= per round):  2.04 GB / 497 GB/s = 4.11 ms     (4.54 ms @450 GB/s)
```

Whole-round total bytes (`PHASE4-P5-ROOFLINE.md:179`, per rank): **6.65 GB** →
floor `6.65/497 = 13.38 ms`; measured verify `92.4 ms` → `92.4/13.38 = 6.9×`; achieved
`6.65 GB / 92.4 ms = 72 GB/s = 14.5 %` of 497 — reproducing the P5 §2.3/§2.4 figures exactly.

### The disproven 4-pass reading, shown for the record

```
HYPOTHETICAL (if 3× m=1 body passes did exist):
  4 passes/round  →  2.04 × 4 = 8.16 GB/round
  floor @497 GB/s           = 16.45 ms     (18.17 ms @450)
  dense slice 34.0 ms → 8.16 / 0.034 s = 240.5 GB/s effective  →  "~2.1× its floor"
```

That `~240 GB/s` / `~2.1×` / `16.45 ms` triple is exactly what the brief asks to certify. The code
shows the three m=1 body passes **do not exist**, so the correct triple is:

```
ACTUAL (1 body pass/round, m=4):
  dense bytes/round = 2.04 GB ;  floor @497 = 4.11 ms
  dense slice time  = 38.5 ms (microbench, PHASE4-P5-ROOFLINE.md:245) → 34.0 ms (gap-share, :245)
  effective rate    = 2.04 / 0.0385 = 53.1 GB/s   (at 34.0 ms: 60.1 GB/s)
  ratio to floor    = 38.5 / 4.11 = 9.4×          (34.0 ms: 8.3×)
```

The P5 doc's stated **`8.5×`** (`:245`, `:263–270`) is the correct figure; the brief's implied
`~2.1×` is the artifact of the phantom 4-pass multiplier.

**The `2.04 GB` figure is itself ~34 % low vs the real byte count (measured by P2).** The `2.04 GB`
uses the card's `2.9 bpw` label. P2's byte census on the real checkpoint headers (all 40 layers,
`PHASE5-P2-DENSE.md:90–108`) finds the dense/attn trellis is packed **80 → k=5 (~5.0 bpw)** and the
shared experts packed **64 → k=4 (4.0 bpw)**, so the real dense per-rank per-pass count is
**2.734 GB/rank** — **~34 % higher** than `2.04 GB`. This does **not** change the unit conclusion
(1 pass/round either way) and does **not** rescue the `×4`; it only means the *correct* per-round
dense bytes are **2.734 GB** → floor **5.50 ms @497 GB/s** (not 4.11 ms), still **~9.3×** the slice
time. Use **2.734 GB/rank/pass** as the byte unit downstream; treat `2.04 GB` as the label-based
lower bound.

**What the `2.04 GB` figure does *not* cover (and why the `4×` cannot be repaired).** The
`141.0 M params/layer × 40` term is strictly the **body** dense/shared/attn weights — `wq_a, wq_b,
wkv, wo_b, wo_a, shared w1/w2/w3` (`PHASE4-P5-ROOFLINE.md:167–169`). It contains **no draft-head
weights** and **no expert weights**. The draft head's own weights (`mtp.0/1/2`: 3 × 128-experts
top-3 + local attn + markov head) are a separate, unmeasured-but-small quantity; if anyone wants
them they must be **measured and added as their own term**, not synthesized by scaling the body
figure. (Phase-17 measured the draft *round* at ~11.1 ms incl. its own head/attn/markov MoE —
`phase17-…/README.md:32–34` — a one-off figure, not a per-pass byte multiplier.)

---

## (iii) KV share correction — `0.148 / 6.65 = 2.23 %`, not `0.2 %`

Confirmed. `PHASE4-P5-ROOFLINE.md:190` states the indexer index-scan is `0.12 GB — 0.2 % of the
total`. That mixes two quantities and is wrong by ~11×:

```
total bytes/round                              = 6.65 GB        (PHASE4-P5-ROOFLINE.md:179)
KV term (window 5.2 MB + top-k gather 26.2 MB
         + index-scan 116.5 MB)                 = 0.148 GB      (:177)
  → KV share        = 0.148 / 6.65 = 0.02226     = 2.226 %   (≈ 2.2 %)
index-scan alone    = 0.12  / 6.65 = 0.01805     = 1.81  %   (≈ 1.8 %)
doc "0.2 %" factor-of-error = 2.226 / 0.2 ≈ 11.1×
```

**Flag: this is a decimal-place error to correct.** The P5 doc's `0.2 %` is ~11× too low; the KV
scan is **≈1.8 %** (index-scan alone) or **≈2.23 %** (the full KV-read term). Consequence: the
doc's conclusion "**the KV read is not a material byte term at 91K**" (`:191`) survives *a fortiori*
— 2.2 % is still immaterial to the verdict — but the number itself must be cited as **2.23 %**, not
`0.2 %`. (The doc's own §2.4 "KV read ~0.3 ms / 1× floor / ~0 % of gap" slice remains consistent:
~0.3 ms is ~1–2 % of the 92.4 ms round.)

---

## (iv) 52 % / 56 ms vs 27 / 29 ms — arithmetic is self-consistent (PASS)

Confirmed from `PHASE3B-SHIP-VALIDATION.md:161–168`, agentic 91K g3, 800 tok:

| arm | ms/round | source |
|---|---:|---|
| production (HIER=1, no levers) | **157.10** | `PHASE3B-SHIP-VALIDATION.md:159` |
| next17-defaults (lever-1 code) | **130.07** | `:160` |
| next17 + `DSV41_INDEXER_HIER=0` (lever-1 code, HIER off) | **101.06** | `:161` |

```
total win vs production = 157.10 − 101.06 = 56.04 ms/round
  lever-1 (COLSPLIT code guard) = 157.10 − 130.07 = 27.03 ms/round  →  27.03 / 56.04 = 48.23 %
  lever-2 (INDEXER_HIER)        = 130.07 − 101.06 = 29.01 ms/round  →  29.01 / 56.04 = 51.77 %
rounded as 27 / 29 / 56:          29 / 56                          =  51.79 %
```

**29.0 / 56.0 = 51.8 % ≈ 52 %, and 27.0 + 29.0 = 56.0 ms. Self-consistent. PASS.**

Cross-checks (stated, not hidden): the win is **larger on agentic (56.04 ms) than benign
(145.75 − 118.56 = 27.19 ms** for lever-1 alone, `:73,:104` — benign lever-2 not separately measured,
see (v)); and the lever-2 share is **robust to the lever-1 attribution** — if lever-1's benign
cross-check (`PHASE3-M3.md:124–126`: next17-defaults − M3-OFF ≈ 24 ms benign / 29 ms agentic) is
used instead, lever-2's ~29 ms is unchanged. The 1.5–8× disagreements between *different* boot
pairs (M3 §2 vs §2b) are the known cross-boot thermal-drift confound (`PHASE3-M3.md:67–71`,
`:148–150`); the same-session chain above is the drift-free one.

Proven production anchors for context (`PHASE3B-SHIP-VALIDATION.md:73–74`): prod benign **145.75** /
agentic **157.10** ms; next17 **118.56** / **130.07** ms; **next17 + HIER=0 agentic = 101.06 ms**.

---

## (v) The MISSING data point — `HIER=0` BENIGN was never measured

Confirmed: only the **agentic** `next17 + DSV41_INDEXER_HIER=0` arm exists (**101.06 ms**,
`PHASE3B-SHIP-VALIDATION.md:161`). The matching **`HIER=0` benign** arm was **never run in any
session** — the benign `101.06` does not exist; benign has only `prod 145.75` and `next17 118.56`
(`:73,:104`).

**Disposition: folded into the Phase-3 R1 validation session (free, rides the relaunch) — NOT a
dedicated relaunch.** `PHASE5-CAMPAIGN.md:29` declares it in the relaunch ledger as `— P3 bench-only
… HIER=0 benign rides R1 free`, and `:84` lists it as an R1 deliverable ("**missing HIER=0 benign
point**"). It shares R1's boot/launcher/idle-gate with the fixed-replay A/B, so it costs **0
additional relaunches**. (Useful as a G1b/G2 cross-check: benign lever-2 share = next17-benign −
HIER=0-benign, to pair with the agentic 29 ms.)

---

## Corrected numbers for the round (cite this downstream)

**Authoritative units for the Phase-5 round doc (supersedes the P5 unit tangle):**

| quantity | corrected value | was (P5/brief) | cite |
|---|---|---|---|
| **body/indexer passes per decode round** | **1** (one m=4 verify forward) | 4 (3×m=1 + 1×m=4) | §(i) — `rounds.py:408/419/427`, `mtp.py:295/313` |
| **Indexer calls per round** | **8** (one per index-source layer, **all at n=4**) | — | `attention.py:119,197`; `indexer.py:591` |
| **draft forwards per round** | **1** (DSparkHead, 3 stages, **no Indexer**) | 3 (m=1) | `mtp.py:295/313`; `mtp.py:18–19` |
| **dense/shared/attn EXL3 bytes** | **2.734 GB/rank/pass** (measured census; `2.04 GB` = 2.9-bpw *label* lower bound) | 2.04 GB/pass → 8.2 GB/round | §(ii) — `PHASE5-P2-DENSE.md:98`; `PHASE4-P5-ROOFLINE.md:176` |
| **dense floor @497 GB/s** | **5.50 ms** (measured bytes; 4.11 ms on the label) | 16.5 ms (4-pass) | §(ii) |
| **dense effective rate** | **53–69 GB/s** (≈9.3× floor) — measured | ~240 GB/s (~2.1×) | §(ii) — `PHASE5-P2-DENSE.md:72–75` |
| **whole-round bytes/rank** | **6.65 GB** → floor **13.38 ms** @497 | (unchanged) | `PHASE4-P5-ROOFLINE.md:179,:222` |
| **round measured vs floor** | **92.4 ms = 6.9× floor**; 72 GB/s = 14.5 % | (unchanged) | `PHASE4-P5-ROOFLINE.md:224` |
| **KV share of total bytes** | **2.23 %** (index-scan alone: **1.81 %**) | 0.2 % ❌ decimal error | §(iii) — `PHASE4-P5-ROOFLINE.md:190` |
| **lever-1 share (agentic)** | **27.03 ms = 48.2 %** | — | §(iv) — `PHASE3B-SHIP-VALIDATION.md:166–168` |
| **lever-2 share (agentic)** | **29.01 ms = 51.8 % ≈ 52 %** | — | §(iv) |
| **total win (agentic)** | **56.04 ms** | — | §(iv) |
| **`HIER=0` benign arm** | **MISSING** — rides R1, 0 extra relaunches | — | §(v) — `PHASE5-CAMPAIGN.md:29,84` |

**Downstream impact (P2):** the correct dense reading is **1 pass → 2.734 GB measured (2.04 GB
label) → ~5.5 ms floor → dense at ~53–69 GB/s (≈9.3× floor)**, i.e. the bf16-dequant-cache route's
premise ("dense is near-floor, only ~2.1×") is **not** supported; the GEMV-bandwidth/latency route is
the one the evidence points to. P2 has **already measured this on real weights** and landed on
**Reading 2 / GO** (`PHASE5-P2-DENSE.md:7,190–209`) — consistent with the units established here.
Any further dense work must use the **2.734 GB/rank/pass** byte unit, not `2.04 GB` and never `8.2 GB`.

---

## Citations (every number)

**Pass / forward structure**
- `src/exo/worker/engines/mlx/dsv41/rounds.py:408` (`head.draft(...)` — the draft-forward call),
  `:419–422` (`verify_in` = `[1, 1+gamma]`), `:424`/`:427` (`model(verify_in, …)` — the verify-forward
  call), `:429` (`mx.eval(logits)`), `:367`/`:389` (greedy / round-1-with-head single-row paths).
- `src/exo/worker/engines/mlx/dsv41/engine.py:971` (`_rounds`), `:981` (`head = self.loaded.head`),
  `:989` (`while n < max_tokens`), `:993` (`_one_round(...)`), `:1019` (`yield`).
- `src/exo/worker/engines/mlx/dsv41/load.py:207–237` (`build_draft_head` → `loaded.head = head`).
- `mlx_lm/models/deepseek_v41/model.py:277` (`for layer in self.layers` — the 40-layer body loop),
  `:124`/`:142` (`self.attn(...)`).
- `mlx_lm/models/deepseek_v41/attention.py:115,118` (`is_index_source` / `Indexer(...)`),
  `:187,197` (`self.indexer(x, qr, …)` — the Indexer call site).
- `mlx_lm/models/deepseek_v41/mtp.py:255` (`class DSparkHead`), `:261` (`n_stages … or 3`),
  `:287`/`:313` (`for stage, c in zip(self.stages, caches)`), `:290`/`:295` (`draft`/`_draft`),
  `:136` (`draft_block`), `:300–313` (block assembly + one forward), `:18–19` (local-only, no
  indexer), `:322–333` (host-free markov loop).
- `mlx_lm/models/deepseek_v41/indexer.py:524` (`class Indexer`), `:565` (`__call__`), `:591`
  (`if _HIER and n > _FENCE_MIN_ROWS:`), `:5–9` (the 8 index-source layers).
- `mlx_lm/models/deepseek_v41/_gates.py:43` (`_FENCE_MIN_ROWS` default 16).
- `tests/test_dsv41_indexer_smallm_hier.py:225` (`N = (1,2,3,4,16,17)` — guard-boundary cells, not
  a pass census).
- PROF build brackets: `src/exo/worker/engines/mlx/dsv41/rounds.py:591` (`t_draft`), `:622`
  (`t_verify`), `:655` (`t_tail`) — in `/private/tmp/next16-instr` (the file the brief mis-paths
  into the mlx-lm tree).

**Bytes / floors / arithmetic**
- `PHASE4-P5-ROOFLINE.md:176` (dense `2.04 GB`/pass), `:177` (KV `0.148 GB`), `:179` (total
  `6.65 GB`), `:190–191` (the erroneous `0.2 %` KV claim), `:210–216` (measured `497 GB/s`),
  `:218–226` (floors + `72 GB/s` achieved), `:245–270` (dense slice `38.5`/`34.0 ms`, `8.5×`),
  `:228–233` (FLOPs floor `3.8 ms`), `:167–169` (dense roster → the `141.0 M`).
- `PHASE3B-SHIP-VALIDATION.md:159–161` (157.10 / 130.07 / 101.06), `:166–168` (56.04 / 27.03 /
  29.01), `:73–74` (benign 145.75 / 118.56).
- `PHASE3-M3.md:44–49` (frozen brackets `92.4`/`93.8`), `:124–126` (lever-2 by subtraction),
  `:67–71,:148–150` (cross-boot drift caveat).
- `phase17-dsv41-verify-cost-2026-09-28/README.md:16–23` (verify-row ablation), `:32–34` (draft
  round ~11.1 ms).
- `PHASE5-CAMPAIGN.md:23,29,72–75,84,94–100` (P0 brief, ledger, R1 deliverables, P2 route).

**Second opinion** — `consult` (reference model), this session: concurred, independently, that the
round has **1 body pass at m=4**, dense bytes stay **2.04 GB**, the Indexer runs only at **n=4**, and
the draft's local-attention einsum is **not** an Indexer pass; the `×4` is a category error.

---

## Limitations

1. **Index-source layer count = 8 is taken from repo docs** (`indexer.py:5–9`; `PHASE4-CAMPAIGN.md:211`),
   not from the live `config.json` (the model is on the nodes, not read here). The pass *structure*
   (one verify forward, one Indexer call per index-source layer) is source-derived and independent of
   whether the count is 8 or the "20" the P5 doc uses for the index-scan bytes (`:177`); the count
   changes *how many* Indexer calls, not *how many body passes* (1).
2. **The dense slice time (38.5 / 34.0 ms) is as-quoted from P5** (microbench-extrapolated +
   gap-share). This doc corrects the *units* (per-pass vs per-round) and the *floor*; it does not
   re-derive the slice time — that is P2's job on real weights.
3. **`HIER=0` benign remains unmeasured** (that is the (v) finding, and its fix rides R1).
4. **No cluster, no GPU, no relaunch, no API POST.** All statements are source-read + arithmetic.
   The one `consult` call was a text reasoning check, not a cluster action.
