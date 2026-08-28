# Eagle K=1 Patch Regression — Diagnosis and Proposed Fix

## 1. Verdict

The pre-predict `broadcast_from_canonical(soft_emb, coord_group)` introduced in commit `21ba40db` is structurally wrong in two ways that together turn a ~150 s c=2 100K iter into a ~6.5-min one: **(a) it adds a synchronously-completing JACCL collective on coord_group on the *critical path between two predict() calls*, where every chained MTP forward at i≥1 now stalls on a bf16 16 KB all_gather completion before its first op (enorm(emb)) can dispatch — turning compute/comm overlap into compute/comm serialization for every step beyond i=0**; and **(b) it does so completely unnecessarily, because K=1's soft-emb is provably identical to `embed_tokens(tok_arr)` and `tok_arr` is *already* cross-rank-broadcast at the end of the previous iter** — the existing broadcast is the determinism source we already have, and the patch ignores it. The 17× slowdown is the cost of forcing every i≥1 chained MTP forward to drain a comm round-trip on its critical path; the "c=2 parallelism lost" symptom is the same effect viewed through the cycle-wall lens, because the slowest chain step in the cycle paces both streams in lockstep through verify, acceptance, and the post-cycle JACCL broadcasts.

## 2. Mechanism trace

Setup: c=2 batched draft loop at `dsv4_mtp.py:1549-1605`, γ=2, FENCE_EVERY_N_LAYERS=4, EXO_DSV4_MTP_EAGLE_K=1. The MTP module is unsharded (`sharding_group=None`, set at `deepseek_v4.py:2913-2941` which shards only `model.layers`, not `model.mtp`), so `mtp.predict()` itself issues no model-TP collectives — every collective in the draft loop lives on `coord_group`.

Chain-step trace at i=1 (the first iter where Eagle engages, because `prev_logits is None` at i=0):

1. `_compute_eagle_soft_emb(prev_logits=logits_i0, embed, K=1)` builds a lazy subgraph: `softmax → argsort → take_along_axis → divide-by-sum → embed_tokens-lookup → multiply-by-broadcast-weights → sum`. Output is a `(B=2, 1, hidden=4096)` bf16 tensor — 16 KB. (`mtp_module.py:591-622`).
2. `broadcast_from_canonical(soft_emb, coord_group)` schedules a `mx.distributed.all_gather` primitive on coord_group with that 16 KB input (`mtp_module.py:558-588`, `mlx/mlx/distributed/ops.cpp:79-101`). The consuming op `enorm(soft_emb_broadcast)` inside `DeepseekV4MTPModule.__call__` (`deepseek_v4.py:2500-2505`) cannot begin executing on the GPU stream until the comm stream signals completion of this gather. **This is the new critical-path stall: i=1's first compute op now waits for a 16 KB RDMA round-trip that did not exist at i=0.**
3. `self.mtp.set_eagle_soft_emb(soft_emb)` mutates `_EAGLE_CTX["soft_emb"]` (`deepseek_v4.py:195-197`). The Python dict read at `__call__` time (`deepseek_v4.py:2500`) captures the post-broadcast soft_emb tensor by reference into the MTP forward graph.
4. `self.mtp.predict(h, tok_arr, ...)` (`dsv4_mtp.py:1575-1577`) builds the MTP forward graph. Inside the MTP `__call__` the very first emit is `e_normed = self.enorm(emb)` where `emb` IS the all_gather output — so the entire MTP forward (1 attn + 1 MoE block + final norm + lm_head) is dependency-pinned behind the gather completion.
5. After predict returns: `tok_arr = broadcast_from_canonical(argmax(logits).astype(int32), coord_group)` — this is the *existing* post-argmax broadcast (`dsv4_mtp.py:1583-1588`). 4·B = 8-byte payload. Critically it is *not* on the critical path between this iter's predict and the next: the next iter's predict can launch its first GPU dispatch (an embed_tokens lookup in the hard-embed path) before this 8-byte all_gather completes, because the comm stream can deliver an int32 vector in the time the GPU stream is still setting up its first compute kernel.

The hard-embed baseline runs exactly the same chain MINUS steps 1–2. The only critical-path collective per chain step is the 8-byte tok_arr broadcast at the end of each iter, and JACCL/MLX can overlap it with the start of the next predict.

With the Eagle patch, every i≥1 predict now has a 16 KB comm round-trip *immediately before* its first GPU dispatch, with no intervening compute to hide latency behind. Each chain step gains one full uncovered comm round-trip plus the time to walk a 16 KB bf16 payload through the JACCL UC path.

A second-order amplifier worth flagging (which I can plausibility-check but not rule in or out from code alone): the chained-collective tail documented in `mtp_module.py:669-681` says that γ≥2 chained predicts already queue multiple lazy collectives in the GPU/comm command buffer, and the peer-CQE arrival tail produces ~75–107 ms outliers when nothing forces a per-step drain. The c=1 path papers over that with an explicit `mx.eval(tok_arr)` at `mtp_module.py:757-758`. **The c=2 batched path has NO such per-step fence.** So in the c=2 chain, the new soft_emb broadcast adds a *second* coord_group collective per chain step that JACCL must order behind the existing tok_arr broadcast, doubling the per-step chance of falling into the outlier-CQE tail. Over γ=2 chains compounded across all decode cycles in a 100K-context generation, this is consistent with the observed 17× magnitude even though pure-µs accounting on the 16 KB payload alone does not predict it. The "c=2 parallelism lost" pattern (six commands admitted at ~6.5-min cadence) is downstream: cycle walls inflated enough that the BatchGenerator's task-admission step rate collapses to roughly cycle-wall-spaced.

## 3. Proper fix — broadcast the IDs, not the embedding

Two sub-fixes, applied together. Together they eliminate the new critical-path collective entirely at K=1 (the only K the user has run), and replace the 16 KB bf16 broadcast with a sub-100-byte int32+bf16 broadcast at K>1.

### 3a. K=1 short-circuit: reuse `tok_arr`

`_compute_eagle_soft_emb` at K=1 collapses algebraically to `embed_tokens(argmax(prev_logits))` (Section 3 of `2026-05-22_eagle_k1_debug_report.md` proves this — argsort-vs-argmax tiebreak on bf16 logits over a 129 K-entry vocab is vanishingly improbable). And `tok_arr` at the start of iter i is exactly `argmax(prev_logits)` *already cross-rank-broadcast* on coord_group by the post-argmax step at the end of iter i-1. So at K=1 there is no need to compute a soft-emb at all — `embed_tokens(tok_arr)` gives the correct, cross-rank-deterministic value with zero new collectives.

### 3b. K>1 fix: broadcast `topk_ids` + `topk_probs` (tiny), recompute mixture locally

For K>1 the mixture cannot be reconstructed from `tok_arr` alone. But it *can* be reconstructed locally on every rank from `topk_ids` + `topk_probs`, both of which are tiny (4·B·K bytes int32 + 2·B·K bytes bf16 — single-digit bytes at K=2, B=2). Broadcast both on coord_group, then compute `(embed_tokens(topk_ids) * topk_probs[..., None]).sum(axis=-2)` locally. Every rank produces a bit-identical soft_emb without putting a 16 KB collective on the chain critical path.

### Unified diff (c=2 batched path)

`src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` around 1549-1573:

```diff
 for i in range(gamma):
     hidden_for_dump = h if drift_dump else None
     _eagle_installed = _eagle_active and prev_logits is not None
     if _eagle_installed:
-        from .mtp_module import _compute_eagle_soft_emb
-        soft_emb = _compute_eagle_soft_emb(
-            prev_logits, _eagle_embed, _eagle_k
-        )
-        # Cross-rank determinism: prev_logits is rank-local and ...
-        if sync_drafts:
-            soft_emb = broadcast_from_canonical(
-                soft_emb, coord_group
-            )
+        if _eagle_k == 1:
+            # K=1 algebraically equals embed_tokens(argmax(prev_logits)),
+            # and tok_arr was already cross-rank-broadcast at the end of
+            # the previous iter (line 1586). Reuse it as our determinism
+            # source — no new collective on the chain critical path.
+            soft_emb = _eagle_embed(tok_arr)  # (B, 1, hidden)
+        else:
+            # K>1: broadcast the small (topk_ids, topk_probs) tensors
+            # instead of the 16 KB bf16 mixture. Each rank then computes
+            # the mixture locally from identical inputs, so soft_emb is
+            # bit-identical across ranks without putting a large bf16
+            # gather on the chain critical path. Payload is K*B int32 +
+            # K*B bf16 ≈ 12 bytes at K=2, B=2.
+            logits3d = prev_logits if prev_logits.ndim == 3 else prev_logits[:, None, :]
+            probs = mx.softmax(logits3d, axis=-1)
+            topk_ids = mx.argsort(-logits3d, axis=-1)[..., :_eagle_k]   # (B,1,K) i32
+            topk_probs = mx.take_along_axis(probs, topk_ids, axis=-1)   # (B,1,K)
+            topk_probs = topk_probs / topk_probs.sum(axis=-1, keepdims=True)
+            if sync_drafts:
+                topk_ids = broadcast_from_canonical(
+                    topk_ids.astype(mx.int32), coord_group
+                )
+                topk_probs = broadcast_from_canonical(topk_probs, coord_group)
+            topk_embs = _eagle_embed(topk_ids)                          # (B,1,K,hidden)
+            soft_emb = (topk_embs * topk_probs[..., None]).sum(axis=-2) # (B,1,hidden)
         self.mtp.set_eagle_soft_emb(soft_emb)
```

`src/exo/worker/engines/mlx/speculative/mtp_module.py` around 688-702 — identical shape, with the c=1 nuance that the broadcast group name in this function is the parameter `sync_group` (which the caller at `dsv4_mtp.py:1693-1695` already passes as `coord_group`, so no semantic change):

```diff
 _eagle_installed = _eagle_active and prev_logits is not None
 if _eagle_installed:
-    soft_emb = _compute_eagle_soft_emb(prev_logits, _eagle_embed, _eagle_k)
-    if sync_drafts and not _disable_broadcast:
-        soft_emb = broadcast_from_canonical(soft_emb, sync_group)
+    if _eagle_k == 1:
+        soft_emb = _eagle_embed(tok_arr)  # (1,1,hidden)
+    else:
+        logits3d = prev_logits if prev_logits.ndim == 3 else prev_logits[:, None, :]
+        probs = mx.softmax(logits3d, axis=-1)
+        topk_ids = mx.argsort(-logits3d, axis=-1)[..., :_eagle_k]
+        topk_probs = mx.take_along_axis(probs, topk_ids, axis=-1)
+        topk_probs = topk_probs / topk_probs.sum(axis=-1, keepdims=True)
+        if sync_drafts and not _disable_broadcast:
+            topk_ids = broadcast_from_canonical(topk_ids.astype(mx.int32), sync_group)
+            topk_probs = broadcast_from_canonical(topk_probs, sync_group)
+        topk_embs = _eagle_embed(topk_ids)
+        soft_emb = (topk_embs * topk_probs[..., None]).sum(axis=-2)
     mtp_pred.set_eagle_soft_emb(soft_emb)
```

### Why this placement avoids H6/H7/H8 (and the regression observed)

- **H6 (lazy-graph entanglement / pre-predict critical-path collective):** at K=1 there is *zero* new collective. The chain step is bit-identical in collective sequence to the hard-embed baseline; the only added work is a local `embed_tokens(tok_arr)` lookup, which is the same work the MTP `__call__` would have done internally via the `else` branch. At K>1 the new collective is on a payload of single-digit bytes, which JACCL/MLX can fire-and-overlap with subsequent GPU work the same way the existing tok_arr broadcast already overlaps — and there is no 16 KB bf16 walk through the UC path.
- **H7 (sync_group vs coord_group interleaving):** the new collective is on coord_group (or the c=1 `sync_group` parameter which the caller binds to `coord_group`). Same group as the existing tok_arr broadcast. So this is not a fix — it was already fine in the broken patch — but it's worth noting because the broken patch did not introduce a group-mismatch issue, so the fix doesn't need to address one.
- **H8 (dict-mutation race with lazy eval):** the prior debug report (`2026-05-22_eagle_k1_debug_report.md`, section 4) correctly argues that `_EAGLE_CTX.get("soft_emb")` is read at `__call__` time (graph build), captures the tensor by reference, and is unaffected by subsequent dict mutation. The fix preserves this contract exactly — same `set / predict / set None` pattern.
- **H9 (broadcast slicing in B>1):** `broadcast_from_canonical` is correct for B>1 (rank 0's contribution sits in `gathered[:n0]` because all_gather concatenates rank-stride along axis 0). The 16 KB payload was algorithmically right, just operationally wrong; the fix replaces it with small payloads that have the same correctness property.

## 4. Risk assessment

**Where the new patch is bit-equivalent to the current `21ba40db` patch (modulo determinism):**

- K=1 short-circuit: `soft_emb = embed_tokens(tok_arr)` is exactly `embed_tokens(broadcast(argmax(prev_logits)))`. The current broken patch computes `embed_tokens(argmax_local(prev_logits))` then all_gathers the bf16 emb. Both produce the same `embed_tokens(rank_0_argmax(prev_logits))` on every rank — bit-identical output, no behavioral diff except eliminating the regression.
- K>1: the fix broadcasts `topk_ids` + `topk_probs` and reconstructs the mixture locally with rank-0's IDs and probs. The current broken patch computes the mixture rank-locally then broadcasts the assembled bf16. Both produce the same mixture-over-rank-0's-topk on every rank. Bit-identical modulo the order of `multiply → sum` vs `sum-on-broadcast-output` arithmetic; both reduce to identical floating-point sequences when inputs are bit-equal, which they are after broadcast.

**Where the fix could still not work:**

1. **K=1 short-circuit assumes `tok_arr` carries the post-broadcast argmax at the point of the eagle install.** In the c=2 batched path that is true: `tok_arr` is rebound to the broadcast output at the end of iter i-1 (line 1586-1588) and iter i's `_eagle_installed = (prev_logits is not None)` is only true at i≥1. In the c=1 path same invariant holds (`mtp_module.py:712-716`). If a future refactor moves the tok_arr broadcast or changes its binding, the short-circuit silently regresses to rank-local argmax. Mitigation: assertion or a one-line invariant comment at the install site documenting the dependency on the upstream broadcast.
2. **K>1 fix relies on the two small broadcasts being scheduled together by JACCL.** If they end up scheduled with sufficient interleaving to fall into the same outlier-CQE tail described in `mtp_module.py:669-681`, K>1 could still suffer (less than the current patch, but more than baseline). Mitigation: fuse the int32 topk_ids and bf16 topk_probs into one broadcast (concat int32-view + bf16-view as bytes), so K>1 adds exactly one extra coord_group collective per chain step instead of two. Lower priority since K=1 is what's blocking right now and the user's H1 list explicitly notes K=1 was supposed to be a sanity gate.
3. **K=1 algebraic equivalence to hard-embed is ALSO true.** Which means K=1 buys nothing functionally vs `EXO_DSV4_MTP_EAGLE_K=0`. It only validates the side-channel plumbing end-to-end. If a benchmark of K=1-fixed matches hard-embed baseline within noise, that confirms the integration is sound; if it does not, look at the argsort-vs-argmax tiebreak path. (For K>1 the lift is a quality question, not a correctness question — see prior report Section 7.)
4. **The all_sum / all_gather call_id sequence on coord_group changes from K=0 (default) to K=1 (short-circuit).** With K=1 short-circuit there is no new coord_group collective per chain step — sequence is identical to K=0. So this is not a risk at K=1. At K>1 the sequence gains 1 (or 2 with un-fused) coord_group collective per chain step beyond i=0, advancing `next_call_id_` accordingly. Both ranks issue identically; no asymmetry. No additional risk vs K=0 beyond the small-payload latency.

## 5. K=1 sanity bonus

Yes — at K=1 the short-circuit IS the right answer regardless of the regression, because K=1 is mathematically a no-op vs hard-embed (mixture over top-1 collapses to embed(argmax)). The fact that the K=1 short-circuit is bit-identical to hard-embed is exactly why it eliminates the regression: it removes the unnecessary computation and the unnecessary collective. The K=1 path becomes "validate the side-channel plumbing fires (the `_EAGLE_CTX.get` branch is taken inside `__call__`) without changing model behavior." That's a useful integration test that the K>1 paths build on.

The K>1 generalization is what the user's task brief suggests: broadcast `topk_ids` instead of soft_emb. That alone (without broadcasting `topk_probs`) does not give bit-identical soft_emb across ranks — local probs from rank-local prev_logits drift by ~1 ulp, so the mixture weights drift by ~1 ulp, and soft_emb drifts by ~1 ulp. Whether that's *good enough* depends on a downstream question: does the MTP forward's drift amplify ulp-level soft_emb drift into argmax flips on the *next* step? If yes, broadcasting topk_ids alone is insufficient and we need topk_probs too. If no (drift stays bounded), topk_ids alone suffices. The safe choice — and the one I'm recommending in the diff above — is to broadcast both. It's still tiny (single-digit bytes) and removes the risk question entirely.

The c=1 path (`mtp_module.py:625-758`) shares the same shape of fix, with the wrinkle that its post-step `mx.eval(tok_arr)` fence at line 757-758 means the c=1 path was probably less catastrophically affected by the current broken patch than c=2 (the fence drains each iter's collectives one step at a time, eliminating the chained-collective outlier-CQE tail amplifier). c=1 still benefits from removing the unnecessary 16 KB collective.

## 6. Action plan

1. **Open a feature branch** off current `origin/main` (which carries the broken `21ba40db` patch). Do not revert `21ba40db` — instead replace its two added blocks with the K=1 short-circuit + K>1 small-broadcast pattern. Keeping the commit history dense around this code makes future bisect targeted.
2. **Edit `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`** in the chain block at lines 1549-1573 — replace the unconditional `_compute_eagle_soft_emb` + `broadcast_from_canonical(soft_emb, coord_group)` with the K=1 / K>1 branching shown in §3. Keep the existing comment about cross-rank determinism but rewrite it to point to the actual fix (the tok_arr broadcast + small-ID broadcast).
3. **Edit `src/exo/worker/engines/mlx/speculative/mtp_module.py`** in `draft_tokens` at lines 688-702 with the matching pattern. Preserve the `EXO_DSV4_MTP_NO_BROADCAST` diagnostic switch for the K>1 broadcasts; at K=1 there's no broadcast to gate, so the switch is a no-op (which is correct — turning off all coord_group collectives at K=1 leaves a self-consistent rank-local path that matches the K=0 hard-embed baseline).
4. **Verify locally**: `uv run pytest src/exo/worker/engines/mlx/speculative/tests -k eagle` plus `uv run basedpyright && uv run ruff check && nix fmt`. No new tests required for the K=1 short-circuit beyond the existing eagle-path tests, because K=1-fixed should match K=0 (hard-embed) exactly modulo the `_EAGLE_CTX` side-channel firing.
5. **Commit + push** (start_cluster.sh hard-resets to origin/main, per the `feedback_commit_push_before_deploy.md` memory).
6. **Re-bench at K=1 first**: expectation is K=1 matches the FENCE=4 hard-embed baseline within noise (~23.3 symmetric per-stream at iter 0; ~34.16 ± 0.07 aggregate sustained per the `c2_batched_prefill_results_2026_05_08.md` memory). If K=1 reproduces the regression, the soft_emb computation isn't the only thing in the patch causing it — look next at whether `set_eagle_soft_emb(None)` in `finally` (which only matters when `_eagle_installed=True`) is hitting some `__call__`-completion ordering that the cleared dict invalidates after the graph builds. (I don't expect this to surface, but it's the next thing to check if K=1 doesn't recover.)
7. **Then bench K=2 / K=4** with the small-broadcast K>1 path enabled. Expectation: matches K=0 baseline within noise OR shows the quality lift the original Plan B.2 was investigating. Either outcome is a clean read (no determinism bug entangled).

End-state: K=1 is a true equivalence to K=0 with the Eagle plumbing exercised, K>1 is a pure quality investigation.
