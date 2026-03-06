# Fix: Warmup Pipeline Deadlock (EOS desync)

## Root Cause

During warmup in pipeline parallel:
1. `pp_prefill_mode` is active (no all_sum token sync)
2. Each node samples independently — only the tail has correct logits
3. `stream_generate` checks for EOS tokens at line 1051 and breaks early
4. If one node hits EOS before others, it exits the generation loop
5. The exited node stops participating in pipeline recv/send
6. Remaining nodes deadlock waiting for RDMA operations that never come

## Fix

### Option A: Pass `ignore_eos=True` to stream_generate during warmup (Recommended)

Add an `ignore_eos` parameter to `stream_generate` and `generate_step`. When True, skip the EOS check at line 1051. Set it True for warmup calls.

This is the cleanest fix because:
- Warmup doesn't care about EOS (it just warms up the JIT)
- All nodes always generate exactly `max_tokens` steps
- Pipeline stays synchronized

### Changes

1. **`stream_generate()`** (line ~990): Add `ignore_eos: bool = False` parameter. When True, skip the `if token in tokenizer.eos_token_ids: break` at line 1051.

2. **`warmup_inference()`** (line ~419): Pass `ignore_eos=True` to `stream_generate()`.

That's it — two small changes. No architectural changes needed.
