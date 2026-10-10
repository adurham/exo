# N3 — Engram coverage under `DSV41_DENSE=affine6` (offline code-trace, no GPU)

**Round:** ROUND-Q1C-MEASURE §2 N3. **Date:** 2026-10-10 (CDT). **Author:** mid-tier worker (delegation).
**Mode:** offline code-trace only — no GPU, no cluster, no code change, no commit. **FLAG-ONLY.**

---

## VERDICT: **COVERED** — engram dense tensors load through `_dense()` → `affine6`

The served `engram` dense weight group (the census's ~8 % bucket) **is** governed by `DSV41_DENSE=affine6`:
it loads through `_dense()` and falls to the **base affine6** mode (the shipped policy selector does **not**
match it). It is **not** EXL3 in the shipped build.

⇒ N3's "FLAG ONLY if it remains EXL3" condition **does NOT fire.** There is **no ~8 % "free extra"** sitting
in EXL3 for a future experts re-encode to pick up — that group is already affine6.

---

## 1. How the engram weights load — exact call path

Two physically distinct engram objects exist per engram layer; only one is a "dense byte" and it is covered.

### (a) `engram.wkv` — the dense EXL3 group → **affine6 (COVERED)**

`mlx_lm/models/deepseek_v41/exl3_build.py`:

- **`:550`** `DENSE_MODE = os.environ.get("DSV41_DENSE", "exl3")` → served value **`affine6`**.
- **`:635`** `_DENSE_POLICY = _parse_dense_policy(os.environ.get("DSV41_DENSE_POLICY", ""))` → served value
  `[("layers.*.ffn.shared_experts.*", "q8g64")]`.
- **`:564-569`** `_POLICY_MODES` (`q8g64 → ("affine", 8, 64)`).
- **`:615-632`** `_resolve_dense_mode(name, policy, base_mode)`: LAST `fnmatch.fnmatchcase` hit wins; no hit
  + `base_mode` startswith `"affine"` → `("affine", int(base_mode[6:]), 64)`; else `("exl3", None, None)`.
- **`:638-644`** `_dense(ck, name)` → `AffineProj(load_dense_layer(...), bits, group)` when kind == `"affine"`,
  else `Exl3Proj(...)`.
- **`:688`** `dense = {g for g in _groups(ck, pre) if ".ffn.experts." not in g}` — `_groups` collects the base
  name of every `…​.trellis` key (`:545-547`). The served EXL3 checkpoint index carries
  `layers.{1,14}.engram.wkv.trellis`, so **`layers.{1,14}.engram.wkv` ∈ `dense`**.
- **`:713-724`** the dense load loop — `engram.wkv` is neither `attn.wq_b/wo_b` nor `ffn.shared_experts.*`, so
  it takes the `else` branch at **`:724`**: `_set(blk, tail, _dense(ck, g))` with
  `tail = "engram.wkv"`, `g = "layers.1.engram.wkv"` (resp. `14`).
- **`:615-632`** resolution of `"layers.1.engram.wkv"`:
  `fnmatch("layers.1.engram.wkv", "layers.*.ffn.shared_experts.*")` → **False** (no `.ffn.shared_experts.`
  substring) ⇒ no policy hit ⇒ base `affine6` ⇒ `("affine", 6, 64)` ⇒ **`AffineProj` (6-bit, group 64)**.

So `DSV41_DENSE=affine6` **does** quantize the engram dense group to affine6. It is not sharded (the census
does not halve it — consistent with `:713-724` `else` branch = replicated).

`model.py:107-110` builds `self.engram = Engram(...)` for `layer_id in args.engram_layer_ids`;
`engram.py:251-254` (`Engram.__init__`) initially creates `self.embed = EngramEmbedding(...)` and
`self.wkv = nn.Linear(...)`. The dense loop then **replaces `engram.wkv` with `AffineProj`**.

### (b) `engram.embed` — native fp8 table → **not `_dense`, not EXL3** (and not a "dense byte")

`exl3_build.py:730-733`:
```
if blk.engram is not None:
    if native is None:
        raise ValueError(f"layer {layer_id} has engram; pass the native release")
    blk.engram.embed = LazyEngramTable(native, layer_id)
```
The hash table is read **row-on-demand from the *native* release** (`LazyEngramTable`, `:449-529`) — fp8
`…​.engram.embed.weight` + `…​.scale`, dequantized on lookup (`:525-527`). It is **never materialized** and is
**not a `…​.trellis` group**, so it is not in `dense`, not routed through `_dense`, and **not counted** in the
"dense bytes" census. (This is the ~203 GB table the docstring at `engram.py:222` warns about.)

### (c) `engram.q_weight` / `k_weight` — plain fp32 (neither quantized path)

Loaded via `_plain(...)` in the generic tail loop (`:754-756`); tiny fp32 vectors; not `…​.trellis`; not in
`dense`.

**Served EXL3 checkpoint index** (`index.json`, 39 shards) contains exactly **12** engram keys — for each of
layers `1` and `14`: `engram.k_weight`, `engram.q_weight`, `engram.wkv.{mul1,suh,svh,trellis}`. The
`engram.embed.*` table is **absent** from the EXL3 index (it lives only in the native release). So the only
*quantized-dense* engram tensors are `engram.wkv` — and those are the ones the census counts.

---

## 2. Served config — `engram_layer_ids` is **NOT empty** ⇒ engram layers DO exist

Served model `dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`, `config.json` (`text_config`):

- `engram_layer_ids: [1, 14]` (also `engram_num_embeddings: [384006168, 384016682]`,
  `engram_head_dim: 256`, `engram_n_heads: 8`, `engram_max_ngram_size: 4`,
  `engram_compressed_vocab_size: 99092`, `engram_table_dir: null`).

The config default is `()` (`config.py:92`), but the **served** config sets `[1, 14]`. Corroborated
independently by the phase-4 raw dump `docs/benchmarks/phase4-exl3-prod-mtp-compat-2026-09-28/raw/p7c.out:50`
(`engram_layer_ids=[1, 14]`, same model). ⇒ This item is **not N/A**; engram layers 1 and 14 are live.

---

## 3. The census source — where "attn 74 % / shared 18 % / engram 8 %" came from

- **Reported in:** `docs/benchmarks/phase20-throughput/ROUND-Q1B-FIX.md` §4 (lines 101-103) and the ledger
  (line 207): *"D2 offline dense byte census (host-side, read-only, 0 restarts): per-rank dense split —
  attention ≈74 %, shared_experts ≈18 %, engram ≈8 %."* It is the D2 row of that round's ledger.
- **Class of source:** a **host-side, read-only byte census of the served EXL3 checkpoint's safetensors
  headers** (same method as the committed `bench/p2_dense_census.py` → `raw/p5/dense_census.json`, which
  counts every non-expert `…​.trellis` key at full and at the world-2 rank-0 slice).
- **How "engram" is defined there:** the engram **dense/trellis** group — i.e. `layers.{1,14}.engram.wkv`
  (the only engram tensors present in the served EXL3 index with a `…​.trellis` payload). The native fp8
  `engram.embed` table and the fp32 `q_weight`/`k_weight` are **not** part of the census (not `…​.trellis`;
  embed is not even in the EXL3 checkpoint).
- **No standalone D2 JSON was committed.** The 74/18/8 figures appear only in `ROUND-Q1B-FIX.md`;
  a repo-wide grep finds no other artifact carrying the split (only `ROUND-Q1C-MEASURE.md` restates it).
- **Independent reconstruction** from the committed `raw/p5/dense_census.json`: the engram delta on layers
  1 and 14 (vs same-class non-engram baselines: L1 vs L0, L14 vs L8) is `97.2 MB + 74.2 MB = 171.4 MB`.
  Against the per-rank dense trellis total `2.734 GB` that is **6.3 %**; against the `2.04 GB` label figure
  the doc also cites it is **8.4 %** — i.e. the doc's rounded **"≈8 %"** is the engram group, order-of-magnitude
  and class confirmed. (The exact % depends on the denominator; immaterial to the verdict.)

---

## 4. FLAG-ONLY statement

- **No code change, no encode, no cluster action** was taken or is proposed here. This is a code-trace verdict.
- The engram dense group is **already affine6** under the shipped env — so a future **experts** re-encode
  (which targets the **routed experts**, `Exl3Experts` / `load_experts`, `exl3_build.py:684`) has **no ~8 %
  engram "free extra"** to reclaim; that group is not EXL3.
- Residual, out-of-scope observations (noted, not acted on):
  - The native `engram.embed` fp8 table is streamed (row-on-demand) and lives outside both the EXL3 and
    affine dense paths; it is unaffected by `DSV41_DENSE` and is not a quant-campaign candidate.
  - The shipped winner policy raises only the **shared-expert** class to `q8g64`; the engram dense group
    stays at the base **q6g64 (affine6)** — consistent with the D3 arms, which never isolated the engram
    class. Recorded for completeness only.

---

### Evidence index (file:line)

| fact | evidence |
|---|---|
| engram dense group ∈ `dense` set | `exl3_build.py:545-547, 688` |
| engram dense group routes to `_dense()` | `exl3_build.py:713-724` (else branch, `:724`) |
| `_dense` → affine vs exl3 | `exl3_build.py:638-644` |
| base affine6 fallback (no policy hit) | `exl3_build.py:615-632, 550, 635` |
| policy modes / selector is shared-experts only | `exl3_build.py:564-569`, served `DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64` |
| engram.embed from native (not dense, not EXL3) | `exl3_build.py:730-733, 449-529`; `engram.py:219-236` |
| Engram module built for engram layers | `model.py:107-110`; `engram.py:239-256`; `config.py:92` |
| served `engram_layer_ids=[1,14]` | served `config.json` `text_config`; corroborated `phase4…/raw/p7c.out:50` |
| served index engram keys (12) | served `index.json` `weight_map` (`layers.{1,14}.engram.{q_weight,k_weight,wkv.{trellis,suh,svh,mul1}}`) |
| census source | `ROUND-Q1B-FIX.md:101-103, 207`; method class `bench/p2_dense_census.py` → `raw/p5/dense_census.json` |
