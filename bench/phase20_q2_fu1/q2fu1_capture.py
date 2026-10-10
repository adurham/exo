# Copyright © 2026 Adam Durham (hermes-gw). Bench harness -- NOT deployed.
"""Round Q2-FU1 capture hook -- VERIFY-SHAPE divergence instrumentation (DSv4.1).

WHAT THIS IS
------------
An operator-staged, env-gated in-process capture harness for the DSv4.1
(DeepSeek-V4.1 EXL3, TP=2) engine.  Round Q2-Gamma found the speculative
draft depth ``gamma`` changes decoded argmaxes at CONFIDENT positions while
the two same-gamma arms were byte-identical.  This hook produces the raw
per-shape tensors needed to classify the cause:

  H1  MoE router top-k amplification (benign): tiny upstream numeric delta
      flips a near-tie router selection -> different expert -> ~1 nat logit
      move.  Read-out: small |delta-logit| UNDER IDENTICAL ROUTING.
  H2  kernel-path / precision discontinuity across the verify row count M.
      Read-out: large |delta-logit| under identical routing, monotonic in M.
  H3  semantic attention/KV/indexer non-equivalence (rotating-KV in-place
      writes, compressor/indexer reduction order).  Read-out: large
      |delta-logit| + rollback-component signature (see the A/B arm).

HOW IT IS INSTALLED (one bootstrap point, mirrors ``next18_capture``)
---------------------------------------------------------------------
``exo.worker.engines.mlx.dsv41.engine._maybe_install_q2fu1_capture()`` (called
from ``Dsv41Engine.__post_init__``) imports THIS module from PYTHONPATH IFF
``DSV41_Q2FU1_CAPTURE`` names an output directory, then calls
``install(outdir)``.  With the env unset the import never runs: production is
byte-for-byte unchanged and zero-cost.  The hook monkeypatches mlx-lm and exo
classes AT RUNTIME -- mlx-lm source is never modified.

WHAT IT RECORDS (pinned; a separate analyzer consumes it; do not change)
------------------------------------------------------------------------
One ``.npz`` per captured forward, ``<arm>_M<M>_<tag>.npz`` under the outdir:

  scalar     : ``M`` int, ``start_pos`` int, ``gamma`` int, ``round_idx`` int
               (-1 for a diagnostic forward), ``arm`` str, ``rank`` int,
               ``vocab_lo`` int, ``vocab_hi`` int, ``n_captured_layers`` int
  int32[M]   : ``rows_ids``  -- token ids fed as this forward's rows
  float32[M, V]      : ``logits`` -- FINAL fp32 logits; this rank's vocab
               slice ``[vocab_lo, vocab_hi)`` filled from ``ShardedHead.local``
               (values exactly as computed, BEFORE the cross-rank all_sum),
               all other columns NaN.  V = 129280.  The analyzer merges ranks.
  float32[M, D]      : ``hidden_in``  -- hc-mean residual entering layer 0
  float32[M, D]      : ``hidden_out`` -- hc-mean residual leaving last layer
  per body layer L=0..39:
      ``idx_L``   int32[M, topk]   -- Gate top-k expert ids (topk=6 body)
      ``w_L``     float32[M, topk] -- Gate weights (post norm + route_scale)
      ``hid_L``   float32[M, D]    -- hc-mean residual ENTERING block L
      ``margin_L`` float32[M]      -- k-th minus (k+1)-th BIASED router score,
               recomputed at dump time from the captured pre-topk x (a pure
               post-eval side computation, never an ancestor of the forward)
  str (Unicode npz arrays): ``note``, ``dtype_logits``, ``flags``,
               ``rows_txt`` (the fed rows, human-readable)

  ``hid_*`` are hc-mean summaries of the [b, n, hc_mult, dim] hyper-connection
  residual stream (the quantity the DSpark taps carry); the full 4x5120 stack
  is deliberately not stored (memory).  Documented deviation from the letter
  of the spec, reported to the PM.

FAITHFULNESS (why holding references does not perturb the graph)
----------------------------------------------------------------
The wrappers only KEEP PYTHON REFERENCES to tensors the production forward
already computes (block inputs/outputs, gate ids/weights/pre-topk x, head
input and local slice).  They never call mx.eval / mx.sync / .item() /
astype inside the forward, never reorder ops, never alter return values, so
no op enters or leaves the lazy graph and no command-buffer boundary moves.
numpy conversion happens only AFTER the engine's own eval of the round; the
held refs are ancestors of the arrays the engine already evaluated, so the
conversion forces nothing new.  The router margins are recomputed post-eval.

DRIVE SURFACE (request-level; no reboot between arms)
-----------------------------------------------------
A chat request whose ``correlation_id`` starts with ``q2fu1`` activates the
harness for THAT request; the request is served normally.  Grammar
(semicolon separated):

  q2fu1; arm=<name>; sweep=<0|1>; ms=<N>; nospec=<0|1>; rollback=<0|1>;
  dmax=<N>; ksweep=<M,M,..>; rounds=<N>

  sweep=1   : the M-shape program at the fixed prefix (post-prefill anchor):
              (a) M=1 sequential greedy chain, K=6 single-row forwards from
                  the anchor (each = the M=1 reference at that position);
              (b) exact full cache snapshot (rings, compressor carries +
                  chunk stashes, comp_kv/index_k rows in group range, engram
                  slice, offset, draft windows);
              (c) SINGLE forwards of rows[:M] for M in ksweep {1,3,4,5,6},
                  each preceded by an exact full restore (same bytes, same
                  absolute positions across shapes);
              (d) M=4 repeat (noise floor, tag M4rep2);
              (e) allsum arm: runtime flip of ``moe._MOE_ALLSUM_BF16`` ->
                  False, M=1 + M=4 (tags M1fp32/M4fp32);
              (f) fence arm: ``indexer._FENCE_MIN_ROWS`` and
                  ``sparse_attention._FENCE_MIN_ROWS`` -> 0, M=1 + M=4
                  (tags M1fence0/M4fence0);
              (g) rollback component arm per M>=3: verify rows[:M], capture
                  the post-verify state, apply production ``spec.rollback``
                  to target=P0+2 (acc=1), then an M=1 forward of the
                  committed next token:
                    ``g<M-1>_A_m1`` -- production rollback only;
                    ``g<M-1>_B_m1`` -- rollback + exact restore of ONLY the
                    rejected rows' window-ring slots (from the pre-verify
                    bytes);
                  A vs B isolates the window-ring clobber.
              (h) integrity: final M=1 (tag M1final) must equal chain step 0
                  bitwise (ids + logits) and the post-sweep cache digest must
                  equal the pre-sweep digest; logged as ``[q2fu1] INTEGRITY``.
              The cache is left at the restored anchor: production decode
              continues from exactly where it would have without the sweep.
  ms=N      : passively capture the first N real verify rounds (tag r<i>).
  nospec=1  : serve this request without speculation (pure-greedy reference).
  rollback=1: run the A/B rollback probes inside each real round (first
              ``rounds`` rounds): tags g<g>_fwd_r<i> / A_m1_r<i> / B_m1_r<i>.

ENV (all read once at install; the launcher forwards them)
----------------------------------------------------------
  DSV41_Q2FU1_CAPTURE  output DIRECTORY (unset => module inert).
  DSV41_Q2FU1_MS / DSV41_Q2FU1_ROUNDS / DSV41_Q2FU1_DMAX / DSV41_Q2FU1_KSWEEP
                       default request knobs (see parse_spec).

STAGING: copy this file to ``/tmp/q2fu1/q2fu1_capture.py`` on BOTH nodes and
use ``/tmp/q2fu1/out`` as the capture dir (the launcher derives PYTHONPATH
from the capture dir's PARENT when EXO_PYTHONPATH_DIAG is unset).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import traceback

import numpy as np

try:  # mlx is required only when actually installing inside the server
    import mlx.core as mx
except Exception:  # pragma: no cover - import smoke on hosts without mlx
    mx = None

_LOG = "[q2fu1]"

DEFAULT_KSWEEP = (1, 3, 4, 5, 6)
DEFAULT_MS = 0
DEFAULT_ROUNDS = 24
DEFAULT_DMAX = 6
#: Sequential greedy chain length K used to derive the fixed continuation.
CHAIN_K = 6
#: Logprob width used by every captured forward (mirrors a client request with
#: top_logprobs=2 -- the Q2-Gamma requests -- so the head call style matches).
LP_K = 2


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_ms_list(name: str, default) -> list[int]:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return list(default)
    try:
        return [int(x) for x in raw.replace(" ", "").split(",") if x]
    except Exception:
        return list(default)


def parse_spec(correlation_id) -> dict | None:
    """Decode the ``q2fu1`` correlation-id grammar; None when not ours."""
    if not isinstance(correlation_id, str):
        return None
    cid = correlation_id.strip()
    if not cid.startswith("q2fu1"):
        return None
    spec: dict = {
        "arm": "q2fu1",
        "sweep": False,
        "ms": _env_int("DSV41_Q2FU1_MS", DEFAULT_MS),
        "nospec": False,
        "rollback": False,
        "dmax": _env_int("DSV41_Q2FU1_DMAX", DEFAULT_DMAX),
        "ksweep": _env_ms_list("DSV41_Q2FU1_KSWEEP", DEFAULT_KSWEEP),
        "rounds": _env_int("DSV41_Q2FU1_ROUNDS", DEFAULT_ROUNDS),
    }
    for chunk in cid.split(";"):
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        k, v = k.strip(), v.strip()
        if k == "arm" and v:
            spec["arm"] = _san(v)
        elif k == "sweep":
            spec["sweep"] = v not in ("0", "false", "no", "")
        elif k == "ms":
            spec["ms"] = int(v or 0)
        elif k == "nospec":
            spec["nospec"] = v not in ("0", "false", "no", "")
        elif k == "rollback":
            spec["rollback"] = v not in ("0", "false", "no", "")
        elif k == "dmax":
            spec["dmax"] = int(v or 0)
        elif k == "rounds":
            spec["rounds"] = int(v or 0)
        elif k == "ksweep":
            spec["ksweep"] = [int(x) for x in v.split(",") if x.strip()]
    return spec


def _san(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))[:64]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _digest_mx(a) -> str:
    """sha256[:16] over the materialized bytes of an mx array (bf16-safe)."""
    if a is None:
        return "-"
    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    arr = np.ascontiguousarray(np.array(a))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _hc_mean_rows(a) -> np.ndarray:
    """[b, n, hc, d] or [b, n, d] -> [b*n, d] fp32 mean over the hc axis."""
    arr = np.asarray(np.array(a), dtype=np.float32)
    if arr.ndim == 4:
        arr = arr.mean(axis=2)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr.astype(np.float32)


def _copy_eval(*arrays):
    """Materialized mx copies (safe to hold across later cache writes)."""
    out = []
    for a in arrays:
        if a is None:
            out.append(None)
            continue
        c = mx.array(a)
        out.append(c)
    real = [c for c in out if c is not None]
    if real:
        mx.eval(real)
    return out


# ---------------------------------------------------------------------------
# the capture singleton
# ---------------------------------------------------------------------------
class Q2fu1Capture:
    """Installed once per engine process; inert until a q2fu1 request lands."""

    def __init__(self, outdir: str):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self._installed = False
        self._engine_patched = False
        self._orig: dict = {}
        self._stack: list[dict] = []
        self._pending: tuple | None = None
        self.active: dict | None = None
        self.round_idx = 0
        self.rank = -1
        self.written: list[str] = []
        self.integrity: list[dict] = []
        self._outids: dict[str, np.ndarray] = {}
        self._logits_by_tag: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------- install --
    def install(self) -> None:
        if self._installed:
            return
        assert mx is not None, "mlx not importable; capture cannot install"
        from mlx_lm.models.deepseek_v41 import exl3_build as M_BUILD
        from mlx_lm.models.deepseek_v41 import model as M_MODEL
        from mlx_lm.models.deepseek_v41 import moe as M_MOE
        from mlx_lm.models.deepseek_v41 import mtp as M_MTP

        cap = self

        # ---- Model.__call__ : open/close a per-forward record --------------
        orig_model_call = M_MODEL.Model.__call__

        def model_call(m, input_ids, cache, last_logit_only=False,
                       return_taps=False, argmax=False, logprobs=0):
            pend = cap._pending
            rec = None
            if pend is not None and pend[0] is m:
                cap._pending = None
                rec = cap._begin(m, pend[1], input_ids, cache)
            out = orig_model_call(m, input_ids, cache, last_logit_only=last_logit_only,
                                  return_taps=return_taps, argmax=argmax,
                                  logprobs=logprobs)
            if rec is not None:
                cap._end(rec, out)
            return out

        self._orig["model_call"] = orig_model_call
        M_MODEL.Model.__call__ = model_call

        # ---- Block.__call__ / DraftStage.__call__ : residual-stream refs ---
        # Signature-agnostic: Block is (x, pre_mix, start_pos, cache, shared),
        # DraftStage is (x, pre_mix, cache); both return (x, ffn_pre).
        def make_block_wrapper(orig):
            def block_call(blk, x, *args, **kwargs):
                rec = cap._stack[-1] if cap._stack else None
                role = lid = None
                if rec is not None:
                    role, lid = cap._owner_role(rec["m"], blk)
                hid_in = x if role is not None else None
                out = orig(blk, x, *args, **kwargs)
                if role is not None:
                    slot = rec["layers"].setdefault(lid, {})
                    slot["role"] = role
                    slot["block_in"] = hid_in
                    slot["block_out"] = out[0]
                return out
            return block_call

        self._orig["block_call"] = M_MODEL.Block.__call__
        M_MODEL.Block.__call__ = make_block_wrapper(M_MODEL.Block.__call__)
        self._orig["stage_call"] = M_MTP.DraftStage.__call__
        M_MTP.DraftStage.__call__ = make_block_wrapper(M_MTP.DraftStage.__call__)

        # ---- Gate.__call__ : router ids/weights + pre-topk x ---------------
        orig_gate_call = M_MOE.Gate.__call__

        def gate_call(g, x, image_mask=None):
            rec = cap._stack[-1] if cap._stack else None
            out = orig_gate_call(g, x, image_mask=image_mask)
            if rec is not None:
                rec.setdefault("gates", []).append((g, x, out[0], out[1]))
            return out

        self._orig["gate_call"] = orig_gate_call
        M_MOE.Gate.__call__ = gate_call

        # ---- ShardedHead : head input + LOCAL vocab slice ------------------
        if hasattr(M_BUILD, "ShardedHead"):
            orig_head_call = M_BUILD.ShardedHead.__call__
            orig_head_local = M_BUILD.ShardedHead.local

            def head_call(hd, h):
                rec = cap._stack[-1] if cap._stack else None
                out = orig_head_call(hd, h)
                cap._note_head(rec, h, out, hd)
                return out

            def head_local(hd, h):
                rec = cap._stack[-1] if cap._stack else None
                out = orig_head_local(hd, h)
                cap._note_head(rec, h, out, hd)
                return out

            self._orig["head_call"] = orig_head_call
            self._orig["head_local"] = orig_head_local
            M_BUILD.ShardedHead.__call__ = head_call
            M_BUILD.ShardedHead.local = head_local

        self._installed = True
        print(f"{_LOG} capture hook ACTIVE -> {self.outdir} (pid={os.getpid()})",
              flush=True)

    @staticmethod
    def _note_head(rec, h, out, hd) -> None:
        if rec is None:
            return
        if getattr(rec["m"], "head", None) is hd:
            rec["head_in"] = h
            rec["head_local"] = out
            rec["head_mod"] = hd

    def _owner_role(self, m, blk) -> tuple[str | None, int | None]:
        lid = getattr(blk, "layer_id", None)
        layers = getattr(m, "layers", None)
        if lid is not None and layers is not None and int(lid) < len(layers) \
                and layers[int(lid)] is blk:
            return "body", int(lid)
        si = getattr(blk, "stage_idx", None)
        mtp = getattr(m, "mtp", None)
        stages = getattr(mtp, "stages", None) if mtp is not None else None
        if si is not None and stages is not None and int(si) < len(stages) \
                and stages[int(si)] is blk:
            return "draft", int(si)
        return None, None

    # -------------------------------------------------- engine-side hooks ---
    def install_engine_hooks(self, engine_mod, engine_cls, rounds_mod) -> None:
        if self._engine_patched:
            return
        cap = self

        orig_gen = engine_cls._generate

        def generate(engine, params, prompt, *, task_id):
            spec = parse_spec(getattr(params, "correlation_id", None))
            gen = orig_gen(engine, params, prompt, task_id=task_id)
            if spec is None:
                return gen
            cap.rank = int(getattr(engine, "device_rank", -1))
            return cap._request_gen(engine, gen, spec)

        self._orig["engine_generate"] = orig_gen
        engine_cls._generate = generate

        orig_round = rounds_mod._one_round

        def one_round(engine, *, model, cache, token, head, policy,
                      draft_state=None, logprobs=0, lp_out=None):
            spec = cap.active
            idx = cap.round_idx
            cap.round_idx = idx + 1
            try:
                if spec is not None and spec.get("nospec"):
                    head = None                       # pure-greedy arm
                if spec is not None and spec.get("rollback") and head is not None \
                        and idx < max(1, int(spec.get("rounds", 0))):
                    snap_pre = None
                    try:
                        if hasattr(cache, "layers"):
                            snap_pre = cap._full_snapshot(cache, draft_state)
                        return cap._rollback_round(
                            engine, rounds_mod, orig_round, model=model, cache=cache,
                            token=token, head=head, policy=policy,
                            draft_state=draft_state, logprobs=logprobs,
                            lp_out=lp_out, spec=spec, idx=idx)
                    except Exception:
                        print(f"{_LOG} rollback arm failed; restoring the pre-round "
                              f"state and falling back to the production round:\n"
                              f"{traceback.format_exc()}", flush=True)
                        if snap_pre is not None:
                            try:
                                cap._restore_full(cache, snap_pre, draft_state)
                            except Exception:
                                print(f"{_LOG} pre-round restore FAILED (cache may "
                                      f"be inconsistent):\n{traceback.format_exc()}",
                                      flush=True)
                if spec is not None and idx < int(spec.get("ms", 0)):
                    gamma_hint = None
                    try:
                        gamma_hint = int(policy.next()) if policy is not None else 1
                    except Exception:
                        gamma_hint = None
                    cap._pending = (model, {
                        "arm": spec.get("arm", "q2fu1"), "tag": f"r{idx}",
                        "gamma": gamma_hint, "round_idx": idx,
                        "flags": "verify_passive",
                    })
                return orig_round(engine, model=model, cache=cache, token=token,
                                  head=head, policy=policy, draft_state=draft_state,
                                  logprobs=logprobs, lp_out=lp_out)
            finally:
                cap._pending = None

        self._orig["one_round"] = orig_round
        # NOTE the BY-VALUE import: ``exo...dsv41.engine`` does
        # ``from .rounds import _one_round`` at module import, so patching only
        # ``rounds._one_round`` would NOT reach the engine's ``_rounds`` loop.
        # Patch BOTH bindings (and the engine module binding restored on
        # uninstall).
        self._orig["engine_mod_one_round"] = engine_mod._one_round
        rounds_mod._one_round = one_round
        engine_mod._one_round = one_round
        self._engine_patched = True

    # -------------------------------------------------------- request drive --
    def _request_gen(self, engine, gen, spec: dict):
        """Wrap one request generator: sweep at the anchor, pass items through."""
        self.active = spec
        self.round_idx = 0
        prev_speculative = None
        try:
            if spec.get("nospec"):
                prev_speculative = engine.speculative
                engine.speculative = False
            swept = False
            while True:
                try:
                    item = gen.send(None)
                except StopIteration:
                    break
                if not swept and spec.get("sweep"):
                    try:
                        loc = {}
                        fr = getattr(gen, "gi_frame", None)
                        if fr is not None:
                            loc = fr.f_locals
                        self._run_sweep(engine, loc, spec)
                    except Exception:
                        print(f"{_LOG} SWEEP FAILED:\n{traceback.format_exc()}",
                              flush=True)
                    swept = True
                yield item
        finally:
            if prev_speculative is not None:
                engine.speculative = prev_speculative
            self.active = None
            self._write_summary(spec)

    # --------------------------------------------------------- per-forward --
    def _begin(self, m, spec: dict, input_ids, cache) -> dict:
        rows = np.array(input_ids)
        rec = {
            "m": m, "spec": spec,
            "rows": rows.astype(np.int32).reshape(-1),
            "M": int(rows.shape[-1]),
            "start_pos": int(cache.offset),
            "layers": {}, "gates": [],
            "head_in": None, "head_local": None, "head_mod": None,
        }
        self._stack.append(rec)
        return rec

    def _end(self, rec, out) -> None:
        if self._stack and self._stack[-1] is rec:
            self._stack.pop()
        rec["out"] = out
        try:
            self._write_record(rec)
        except Exception:
            print(f"{_LOG} record dump FAILED (forward unaffected):\n"
                  f"{traceback.format_exc()}", flush=True)

    # -------------------------------------------------------------- dumps ---
    def _write_record(self, rec) -> None:
        spec = rec["spec"]
        m = rec["m"]
        M = rec["M"]
        out = rec["out"]
        first = out[0] if isinstance(out, tuple) else out
        if getattr(first, "ndim", 1) == 3:      # full logits row(s)
            ids = np.argmax(np.array(first), axis=-1).astype(np.int32)
        else:
            ids = np.array(first).astype(np.int32)
        ids = ids.reshape(-1)
        self._outids[spec["tag"]] = ids

        V = 129280
        lo, hi = 0, V
        hl = rec.get("head_local")
        hm = rec.get("head_mod")
        if hm is not None:
            V = int(getattr(hm, "_vocab", None) or V)
        logits = np.full((M, V), np.nan, dtype=np.float32)
        if hl is not None:
            y = np.array(hl).astype(np.float32)
            y = y.reshape(-1, y.shape[-1])
            lo = int(getattr(hm, "_lo", 0))
            hi = lo + int(y.shape[-1])
            logits[:, lo:hi] = y[:M]
        elif isinstance(out, tuple) and getattr(out[0], "ndim", 0) == 3:
            y = np.array(out[0]).astype(np.float32).reshape(-1, out[0].shape[-1])
            logits[:, :y.shape[-1]] = y[:M]
            hi = int(y.shape[-1])

        arrays: dict = {
            "M": np.int64(M),
            "start_pos": np.int64(rec["start_pos"]),
            "gamma": np.int64(int(spec.get("gamma") or 0)),
            "round_idx": np.int64(int(spec.get("round_idx", -1))),
            "arm": np.array(str(spec.get("arm", "q2fu1"))),
            "rank": np.int64(self.rank),
            "vocab_lo": np.int64(lo),
            "vocab_hi": np.int64(hi),
            "rows_ids": rec["rows"].astype(np.int32),
            "rows_txt": np.array(str([int(x) for x in rec["rows"]])),
            "out_ids": ids.astype(np.int32),
            "logits": logits,
            "flags": np.array(str(spec.get("flags", ""))),
            "dtype_logits": np.array(str(np.asarray(hl).dtype) if hl is not None
                                     else "none"),
            "note": np.array(
                f"start_pos={rec['start_pos']} M={M} gamma={spec.get('gamma')} "
                f"arm={spec.get('arm')} tag={spec['tag']} rank={self.rank}; "
                f"logits=ShardedHead.local slice [lo,hi) pre-all_sum; "
                f"hid_*=hc-mean residual"),
        }
        lay = rec["layers"]
        body_Ls = sorted(L for L, v in lay.items() if v.get("role") == "body")
        arrays["n_captured_layers"] = np.int64(len(body_Ls))
        if body_Ls:
            first = lay[body_Ls[0]]
            last = lay[body_Ls[-1]]
            if first.get("block_in") is not None:
                arrays["hidden_in"] = _hc_mean_rows(first["block_in"])[:M]
            if last.get("block_out") is not None:
                arrays["hidden_out"] = _hc_mean_rows(last["block_out"])[:M]
        # gate bucket: layer -> (x, w, idx, gate module)
        gate_by_L: dict[int, tuple] = {}
        for g, x, w, idx in rec.get("gates", []):
            L = self._gate_layer(m, g)
            if L is not None:
                gate_by_L[L] = (g, x, w, idx)
        for L in body_Ls:
            slot = lay[L]
            if slot.get("block_in") is not None:
                arrays[f"hid_{L}"] = _hc_mean_rows(slot["block_in"])[:M]
            gx = gate_by_L.get(L)
            if gx is None:
                continue
            g, x, w, idx = gx
            arrays[f"idx_{L}"] = np.array(idx).astype(np.int32).reshape(M, -1)
            arrays[f"w_{L}"] = np.array(w).astype(np.float32).reshape(M, -1)
            try:
                arrays[f"margin_{L}"] = self._router_margin(g, x, M)
            except Exception:
                pass
        stem = f"{_san(spec.get('arm', 'q2fu1'))}_M{M}_{_san(spec['tag'])}"
        path = os.path.join(self.outdir, stem + ".npz")
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, path)
        self.written.append(stem)
        print(f"{_LOG} dump {stem}.npz "
              f"(start={rec['start_pos']} ids={ids[:8].tolist()}"
              f"{'...' if M > 8 else ''})", flush=True)

    @staticmethod
    def _gate_layer(m, g):
        layers = getattr(m, "layers", None) or []
        for i, blk in enumerate(layers):
            if getattr(getattr(blk, "ffn", None), "gate", None) is g:
                return i
        mtp = getattr(m, "mtp", None)
        stages = getattr(mtp, "stages", None) if mtp is not None else None
        for i, st in enumerate(stages or []):
            if getattr(getattr(st, "ffn", None), "gate", None) is g:
                return None              # draft-stage gate: not a body layer key
        return None

    @staticmethod
    def _router_margin(g, x, M) -> np.ndarray:
        """k-th minus (k+1)-th BIASED score, post-eval side computation."""
        x32 = np.array(x).astype(np.float32).reshape(-1, int(g.weight.shape[1]))
        w32 = np.array(g.weight).astype(np.float32)
        temp = float(getattr(g, "gate_temp", 1.0) or 1.0)
        s = np.sqrt(np.logaddexp(0.0, (x32 @ w32.T) / temp))
        biased = s + np.array(g.bias).astype(np.float32)[None, :]
        k = int(g.topk)
        out = np.full((s.shape[0],), np.nan, dtype=np.float32)
        for r in range(s.shape[0]):
            if biased.shape[1] > k:
                part = np.partition(biased[r], -k)
                out[r] = float(part[-k] - part[-(k + 1)])
        return out[:M]

    # ------------------------------------------------------ full snapshots --
    def _full_snapshot(self, cache, draft_state=None, k: int = CHAIN_K + 2) -> dict:
        """Exact materialized copy of every mutable cache byte this hook can
        touch: rings, compressor state + chunk stashes, the comp_kv / index_k
        rows a <=k-row forward can write, the engram slice, offset, draft
        windows.  Also a digest of the same content."""
        off = int(cache.offset)
        layers = []
        h = hashlib.sha256()
        h.update(off.to_bytes(8, "little"))
        for lc in cache.layers:
            ring = mx.array(lc.win_kv)
            cs = lc.comp_state
            kv_state = mx.array(cs.kv_state) if cs is not None else None
            sc_state = mx.array(cs.score_state) if cs is not None else None
            chunk_kv = (mx.array(cs.chunk_kv) if cs is not None
                        and cs.chunk_kv is not None else None)
            chunk_sc = (mx.array(cs.chunk_score) if cs is not None
                        and cs.chunk_score is not None else None)
            chunk_start = int(cs.chunk_start) if cs is not None else 0
            rows = []
            r = int(lc.ratio or 0)
            if r > 0:
                j0 = max(0, off // r - 1)
                j1 = (off + k) // r + 2
                for name in ("comp_kv", "index_k"):
                    buf = getattr(lc, name, None)
                    if buf is not None:
                        j1c = min(int(buf.shape[1]), j1)
                        if j1c > j0:
                            rows.append((name, j0, j1c, mx.array(buf[:, j0:j1c])))
            layers.append((lc, ring, kv_state, sc_state, chunk_kv, chunk_sc,
                           chunk_start, rows))
        to_eval = [t for (_lc, ring, kv, sc, ck, csc, _st, rows) in layers
                   for t in (ring, kv, sc, ck, csc) if t is not None]
        to_eval += [t[3] for (_lc, *_, rows) in layers for t in rows]
        if to_eval:
            mx.eval(to_eval)
        # digest (rings + carries + saved rows + offset)
        h.update(f"{len(layers)}".encode())
        for (_lc, ring, kv, sc, ck, csc, st, rows) in layers:
            for t in (ring, kv, sc, ck, csc):
                if t is not None:
                    h.update(_digest_mx(t).encode())
            h.update(str(st).encode())
        engram = None
        if cache.engram_ids is not None:
            e0 = max(0, off - 2)
            e1 = min(int(cache.engram_ids.shape[1]), off + k + 2)
            engram = (e0, e1, np.array(cache.engram_ids[:, e0:e1], copy=True))
            h.update(hashlib.sha256(engram[2].tobytes()).hexdigest().encode())
        draft = None
        if draft_state is not None:
            draft = [(mx.array(w.win_kv), int(w.n_ctx)) for w in draft_state]
            mx.eval([d[0] for d in draft])
            for kv, nctx in draft:
                h.update(_digest_mx(kv).encode())
                h.update(str(nctx).encode())
        return {"offset": off, "layers": layers, "engram": engram,
                "draft": draft, "digest": h.hexdigest()[:16]}

    def _restore_full(self, cache, snap: dict, draft_state=None) -> None:
        for (_lc, ring, kv, sc, ck, csc, st, rows) in snap["layers"]:
            lc = _lc
            lc.win_kv[:] = ring
            cs = lc.comp_state
            if cs is not None:
                if kv is not None:
                    cs.kv_state[:] = kv
                if sc is not None:
                    cs.score_state[:] = sc
                cs.chunk_kv = None if ck is None else mx.array(ck)
                cs.chunk_score = None if csc is None else mx.array(csc)
                cs.chunk_start = st
            for name, j0, j1, rows_arr in rows:
                buf = getattr(lc, name)
                buf[:, j0:j1] = rows_arr
        if snap["engram"] is not None:
            e0, e1, data = snap["engram"]
            cache.engram_ids[:, e0:e1] = data
        cache.offset = snap["offset"]
        if draft_state is not None and snap["draft"] is not None:
            for w, (kv, nctx) in zip(draft_state, snap["draft"]):
                w.win_kv[:] = kv
                w.n_ctx = nctx

    # --------------------------------------------------------- sweep engine --
    def _run_sweep(self, engine, loc, spec: dict) -> None:
        model = engine.loaded.model
        session = loc.get("session")
        anchor = loc.get("anchor")
        if session is None or anchor is None:
            raise RuntimeError("sweep: generator frame has no session/anchor")
        cache = session.cache.cache
        draft_state = getattr(session, "draft_state", None)
        arm = spec.get("arm", "q2fu1")
        P0 = int(cache.offset)
        print(f"{_LOG} SWEEP start at P0={P0} anchor={int(anchor)} rank={self.rank}",
              flush=True)
        # (b) exact snapshot of the ANCHOR state, taken BEFORE anything moves.
        snap = self._full_snapshot(cache, draft_state)
        # (a) M=1 sequential greedy chain (K single-row forwards from the
        # anchor; each is the M=1 reference at that absolute position).
        chain = [int(anchor)]
        for i in range(CHAIN_K):
            self._forward_rows_raw(model, cache, [chain[-1]], arm=arm,
                                   tag=f"s{i}", gamma=0, round_idx=-1,
                                   flags="m1_chain")
            chain.append(int(self._outids[f"s{i}"].reshape(-1)[0]))
        rows = chain[:CHAIN_K]
        print(f"{_LOG} sweep chain={chain} P0={P0}", flush=True)
        # (c) the M sweep.
        ksweep = [int(x) for x in spec.get("ksweep", DEFAULT_KSWEEP)]
        for M in ksweep:
            self._restore_full(cache, snap, draft_state)
            self._forward_rows_raw(model, cache, rows[:M], arm=arm, tag=f"M{M}",
                                   gamma=0, round_idx=-1, flags="sweep")
            self._logits_by_tag[f"M{M}"] = None      # (files carry the data)
        # (d) noise-floor repeat at M=4.
        if 4 in ksweep:
            self._restore_full(cache, snap, draft_state)
            self._forward_rows_raw(model, cache, rows[:4], arm=arm, tag="M4rep2",
                                   gamma=0, round_idx=-1, flags="sweep_rep")
        # (e) allsum dtype arm.
        from mlx_lm.models.deepseek_v41 import moe as M_MOE
        prev_allsum = M_MOE._MOE_ALLSUM_BF16
        try:
            M_MOE._MOE_ALLSUM_BF16 = False
            self._restore_full(cache, snap, draft_state)
            self._forward_rows_raw(model, cache, rows[:1], arm=arm, tag="M1fp32",
                                   gamma=0, round_idx=-1, flags="allsum_fp32")
            self._restore_full(cache, snap, draft_state)
            self._forward_rows_raw(model, cache, rows[:4], arm=arm, tag="M4fp32",
                                   gamma=0, round_idx=-1, flags="allsum_fp32")
        finally:
            M_MOE._MOE_ALLSUM_BF16 = prev_allsum
        # (f) fence arm.
        from mlx_lm.models.deepseek_v41 import indexer as M_IX
        from mlx_lm.models.deepseek_v41 import sparse_attention as M_SA
        prev_ix, prev_sa = M_IX._FENCE_MIN_ROWS, M_SA._FENCE_MIN_ROWS
        try:
            M_IX._FENCE_MIN_ROWS = 0
            M_SA._FENCE_MIN_ROWS = 0
            self._restore_full(cache, snap, draft_state)
            self._forward_rows_raw(model, cache, rows[:1], arm=arm, tag="M1fence0",
                                   gamma=0, round_idx=-1, flags="fence0")
            self._restore_full(cache, snap, draft_state)
            self._forward_rows_raw(model, cache, rows[:4], arm=arm, tag="M4fence0",
                                   gamma=0, round_idx=-1, flags="fence0")
        finally:
            M_IX._FENCE_MIN_ROWS = prev_ix
            M_SA._FENCE_MIN_ROWS = prev_sa
        # (g) rollback component arm (acc=1; M = gamma+1).
        for M in [x for x in ksweep if x >= 3]:
            self._rollback_component_arm(model, cache, rows[:M], M, spec)
        # (h) integrity: M=1 after the sweep must equal chain step 0 bitwise,
        # and the restored cache digest must equal the pre-chain digest.
        self._restore_full(cache, snap, draft_state)
        self._forward_rows_raw(model, cache, rows[:1], arm=arm, tag="M1final",
                               gamma=0, round_idx=-1, flags="m1_chain_final")
        dig_after = self._full_snapshot(cache, draft_state)["digest"]
        # NOTE: the M1final forward itself advanced the cache by one row, so
        # digest the state as restored (not as advanced): restore again first.
        self._restore_full(cache, snap, draft_state)
        dig_restored = self._full_snapshot(cache, draft_state)["digest"]
        ids_ok = bool(np.array_equal(self._outids.get("s0"),
                                     self._outids.get("M1final")))
        ok = (dig_restored == snap["digest"])
        verdict = {"cache_restore_ok": bool(ok), "m1_repro_ok": ids_ok,
                   "digest_pre": snap["digest"], "digest_post": dig_restored,
                   "digest_after_m1": dig_after, "P0": P0, "chain": chain,
                   "rank": self.rank}
        self.integrity.append(verdict)
        level = "OK" if (ok and ids_ok) else "*** FAIL ***"
        print(f"{_LOG} INTEGRITY {level} cache_restore={ok} m1_repro={ids_ok} "
              f"pre={snap['digest']} post={dig_restored} P0={P0} chain={chain}",
              flush=True)

    @staticmethod
    def _stash_copies(stashes):
        """Eval'd mx copies of ``spec.stashes`` (np arrays break mx.concatenate
        inside ``spec._rebuild``; live buffers could be rewritten later)."""
        out = []
        for kv, sc in stashes:
            out.append((None if kv is None else mx.array(kv),
                        None if sc is None else mx.array(sc)))
        real = [t for pair in out for t in pair if t is not None]
        if real:
            mx.eval(real)
        return out

    # --------------------------------------------------- rollback A/B arm ---
    def _rollback_component_arm(self, model, cache, row_ids, M, spec) -> None:
        """verify rows -> production rollback (acc=1) -> A/B next-anchor M=1.

        A = production rollback only.
        B = rollback + exact restore of ONLY the REJECTED rows' ring slots
            (from the pre-verify bytes) -- the semantic-fix candidate.
        Both start from identical POST-verify bytes.
        """
        from mlx_lm.models.deepseek_v41 import spec as SPEC
        arm = spec.get("arm", "q2fu1")
        P0 = int(cache.offset)
        snapshot = SPEC.snap(cache, P0)                 # production snapshot
        rings_pre = [mx.array(lc.win_kv) for lc in cache.layers]
        mx.eval(rings_pre)
        # the verify forward (M rows in ONE forward; gamma = M-1).
        self._forward_rows_raw(model, cache, row_ids, arm=arm, tag=f"g{M - 1}_fwd",
                               gamma=M - 1, round_idx=-1, flags="rollback_warm")
        stashes = SPEC.stashes(cache)
        stashes_mx = self._stash_copies(stashes)
        target_ids = self._outids[f"g{M - 1}_fwd"]
        full_post = self._full_snapshot(cache)          # POST-verify bytes
        target = P0 + 2                                 # acc=1
        nxt1 = int(target_ids[0])                       # row-0 target token

        def _ring_restore_rejected():
            for lc, old in zip(cache.layers, rings_pre):
                for q in range(target, P0 + M):
                    s = q % lc.window
                    lc.win_kv[:, s] = old[:, s]

        # A: production rollback only.
        SPEC.rollback(cache, snapshot, target, stashes_mx)
        self._forward_rows_raw(model, cache, [nxt1], arm=arm, tag=f"g{M - 1}_A_m1",
                               gamma=M - 1, round_idx=-1, flags="rollback_A")
        # B: same rollback + rejected-slot ring restore.
        self._restore_full(cache, full_post)
        SPEC.rollback(cache, snapshot, target, stashes_mx)
        _ring_restore_rejected()
        self._forward_rows_raw(model, cache, [nxt1], arm=arm, tag=f"g{M - 1}_B_m1",
                               gamma=M - 1, round_idx=-1,
                               flags="rollback_B_ringrestore")
        # leave the cache at the (B-shaped) post-rollback state: identical to
        # production except the rejected slots, which no live read needs.
        self._restore_full(cache, full_post)
        SPEC.rollback(cache, snapshot, target, stashes_mx)
        _ring_restore_rejected()

    def _rollback_round(self, engine, rounds_mod, orig_round, *, model, cache, token,
                        head, policy, draft_state, logprobs, lp_out, spec, idx) -> tuple:
        """A production-shaped speculative round with the A/B probes inserted."""
        import time as _time
        from mlx_lm.models.deepseek_v41 import spec as SPEC
        started = _time.perf_counter()
        arm = spec.get("arm", "q2fu1")
        anchor = mx.array([token], dtype=mx.int32).reshape(1, 1)
        if draft_state is None:
            draft_state = engine._draft_windows.get(rounds_mod._DRAFT_KEY)
        if draft_state is None:
            # priming round: run production (counted by the caller).
            return orig_round(engine, model=model, cache=cache, token=token,
                              head=head, policy=policy, draft_state=draft_state,
                              logprobs=logprobs, lp_out=lp_out)
        gamma = int(policy.next()) if policy is not None else 1
        rounds_mod._ensure_capacity_for_round(cache, 1 + gamma)
        position = int(cache.offset)
        drafted = head.draft(anchor.reshape(-1), getattr(model, "embed", None),
                             getattr(model, "head", None), draft_state, width=gamma)
        if isinstance(drafted, tuple):
            drafted = drafted[0]
        drafted = drafted.astype(mx.int32)
        verify_in = mx.concatenate([anchor.reshape(1, 1), drafted.reshape(1, gamma)],
                                   axis=1)
        snapshot = SPEC.snap(cache, position)
        rings_pre = [mx.array(lc.win_kv) for lc in cache.layers]
        mx.eval(rings_pre)
        self._pending = (model, {"arm": arm, "tag": f"g{gamma}_fwd_r{idx}",
                                 "gamma": gamma, "round_idx": idx,
                                 "flags": "rollback_round_fwd"})
        if logprobs:
            logits, taps, lp = model(verify_in, cache, return_taps=True,
                                     argmax=True, logprobs=logprobs)
        else:
            logits, taps = model(verify_in, cache, return_taps=True, argmax=True)
            lp = None
        mx.eval(logits, drafted)
        stashes = SPEC.stashes(cache)
        stashes_mx = self._stash_copies(stashes)
        full_post = self._full_snapshot(cache, draft_state) if draft_state else \
            self._full_snapshot(cache)
        target = [int(v) for v in logits[0]]
        draft = [int(v) for v in drafted[0]]
        accepted = 0
        while accepted < gamma and accepted < len(target) \
                and target[accepted] == draft[accepted]:
            accepted += 1
        committed = draft[:accepted] + [target[min(accepted, len(target) - 1)]]
        committed_position = position + accepted + 1
        taps_ids = list(model.args.dspark_target_layer_ids)

        def tapcat(t):
            return mx.concatenate([t[L] for L in taps_ids], axis=-1)

        def _ring_restore_rejected():
            for lc, old in zip(cache.layers, rings_pre):
                for q in range(committed_position, position + 1 + gamma):
                    s = q % lc.window
                    lc.win_kv[:, s] = old[:, s]

        # A: exactly the production sequence.
        SPEC.rollback(cache, snapshot, committed_position, stashes)
        head.append_ctx(tapcat(taps)[:, : accepted + 1], draft_state)
        self._forward_rows_raw(model, cache, [committed[-1]], arm=arm,
                               tag=f"g{gamma}_A_m1_r{idx}", gamma=gamma,
                               round_idx=idx, flags="rollback_A_real")
        # B: rollback + rejected-slot ring restore.
        self._restore_full(cache, full_post,
                           draft_state if draft_state else None)
        SPEC.rollback(cache, snapshot, committed_position, stashes_mx)
        _ring_restore_rejected()
        head.append_ctx(tapcat(taps)[:, : accepted + 1], draft_state)
        self._forward_rows_raw(model, cache, [committed[-1]], arm=arm,
                               tag=f"g{gamma}_B_m1_r{idx}", gamma=gamma,
                               round_idx=idx, flags="rollback_B_real")
        # restore the exact PRODUCTION post-round state and return it: the
        # byte state production's own rollback would leave, i.e. the A shape
        # (NO rejected-slot ring restore -- production keeps the clobber, and
        # the request's continuation must be byte-identical to a plain round).
        self._restore_full(cache, full_post,
                           draft_state if draft_state else None)
        SPEC.rollback(cache, snapshot, committed_position, stashes_mx)
        head.append_ctx(tapcat(taps)[:, : accepted + 1], draft_state)
        if lp is not None and lp_out is not None:
            rounds_mod._rows_logprobs(lp, len(committed), lp_out)
        return (committed, (_time.perf_counter() - started) * 1e3, accepted, gamma)

    # ------------------------------------------------------- forward helper --
    def _forward_rows_raw(self, model, cache, row_ids, *, arm, tag, gamma,
                          round_idx, flags) -> None:
        self._pending = (model, {"arm": arm, "tag": tag, "gamma": gamma,
                                 "round_idx": round_idx, "flags": flags})
        vin = mx.array([list(row_ids)], dtype=mx.int32)
        try:
            out = model(vin, cache, return_taps=True, argmax=True, logprobs=LP_K)
        finally:
            self._pending = None          # never leak a stale spec into a call
        if isinstance(out, tuple):
            mx.eval(out[0])
            if len(out) > 1 and isinstance(out[1], dict):
                mx.eval(list(out[1].values()))
            else:
                mx.eval(out[1])
        else:
            mx.eval(out)

    # ------------------------------------------------------------- summary --
    def _write_summary(self, spec: dict) -> None:
        try:
            path = os.path.join(self.outdir,
                                f"summary_{_san(spec.get('arm', 'q2fu1'))}"
                                f"_rank{self.rank}.json")
            payload = {
                "arm": spec.get("arm"), "spec": {k: v for k, v in spec.items()
                                                 if k != "ksweep"},
                "ksweep": spec.get("ksweep"), "rank": self.rank,
                "written": list(self.written), "integrity": self.integrity,
                "outids": {k: list(map(int, v)) for k, v in self._outids.items()},
            }
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1)
            os.replace(tmp, path)
            print(f"{_LOG} summary -> {path}", flush=True)
        except Exception:
            print(f"{_LOG} summary write failed:\n{traceback.format_exc()}",
                  flush=True)
        self.written = []
        self.integrity = []
        self._outids = {}


# ---------------------------------------------------------------------------
# module API (mirrors next18_capture)
# ---------------------------------------------------------------------------
_CAP: Q2fu1Capture | None = None


def install(path: str | None = None) -> Q2fu1Capture:
    """Create/return the singleton; idempotent.  Also patches the exo engine."""
    global _CAP
    if _CAP is None:
        outdir = path or os.environ.get("DSV41_Q2FU1_CAPTURE") or "."
        _CAP = Q2fu1Capture(outdir)
        _CAP.install()
        try:
            from exo.worker.engines.mlx.dsv41 import engine as ENG
            from exo.worker.engines.mlx.dsv41 import rounds as RND
            _CAP.install_engine_hooks(ENG, ENG.Dsv41Engine, RND)
        except Exception:
            print(f"{_LOG} engine hooks not installed (offline?):\n"
                  f"{traceback.format_exc()}", flush=True)
    return _CAP


def uninstall() -> None:
    global _CAP
    if _CAP is not None and _CAP._installed:
        from mlx_lm.models.deepseek_v41 import exl3_build as M_BUILD
        from mlx_lm.models.deepseek_v41 import model as M_MODEL
        from mlx_lm.models.deepseek_v41 import moe as M_MOE
        from mlx_lm.models.deepseek_v41 import mtp as M_MTP
        M_MODEL.Model.__call__ = _CAP._orig["model_call"]
        M_MODEL.Block.__call__ = _CAP._orig["block_call"]
        M_MTP.DraftStage.__call__ = _CAP._orig["stage_call"]
        M_MOE.Gate.__call__ = _CAP._orig["gate_call"]
        if "head_call" in _CAP._orig:
            M_BUILD.ShardedHead.__call__ = _CAP._orig["head_call"]
            M_BUILD.ShardedHead.local = _CAP._orig["head_local"]
        if _CAP._engine_patched:
            from exo.worker.engines.mlx.dsv41 import engine as ENG
            from exo.worker.engines.mlx.dsv41 import rounds as RND
            ENG.Dsv41Engine._generate = _CAP._orig["engine_generate"]
            ENG._one_round = _CAP._orig["engine_mod_one_round"]
            RND._one_round = _CAP._orig["one_round"]
        _CAP._installed = False
    _CAP = None


def flush() -> None:
    """No-op: records are written per-forward; kept for harness parity."""
    return None


def active() -> Q2fu1Capture | None:
    return _CAP
