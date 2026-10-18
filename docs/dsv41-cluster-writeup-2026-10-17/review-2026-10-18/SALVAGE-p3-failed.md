# Salvage — failed p3 agent (deleg_3cf17139 task-2), 2026-10-18

Failed on reasoning-token exhaustion: it ran the recompute but never wrote its file.
Raw recompute output below (log lines are truncated with "(+N chars)" — the full trace is at
`/Users/adam.durham/.hermes/cache/delegation/live/deleg_3cf17139/task-2.log`).
NOTE: its tolerance was too tight — it compared rounded
displayed values against full-precision recomputes, so most 'BAD' lines are rounding
artifacts, not defects. Re-verify at DISPLAYED precision.

## Its recompute script output
result   | terminal ok 0.9s: {"output": "BAD L117 prefill 239.5 239.4732\nBAD L132 srv prefill 242.0 241.9976\nBAD L133 srv decode 30.4 30.581\nL133 window 6.54\nBAD L168 d1k prefill 255.2 255.2163\nBAD L168 d1k cdec 34.5 34.8936\nBAD L168 d1k sdec 29.2 29.6029\nBAD L169 d10k prefill 275.1 275.0619\nBAD L169 cdec 31.4 31.6667\nBAD L169 sdec 31.3 31.47\nBAD L170 d50k prefill 257.7 257.6693\nBAD L170 cdec 29.4 29.5826\nBAD L170 …(+718 chars)
tool     | -> execute_code(from hermes_tools import read_file, terminal p="/Users/adam.durham/repos/exo/docs/dsv41-cluster-writeup-2026-10-17/review-2026-10-18/p3-pack-internal.md" r=terminal(f"ls -la {p}; wc -l {p}") print(r["output"]) print(read …(+31 chars))

## Its two findings from the final message
final    | status=failed duration=220.11s summary: {"output_file": "/Users/adam.durham/repos/exo/docs/dsv41-cluster-writeup-2026-10-17/review-2026-10-18/p3-pack-internal.md", "most_serious": ["evidence-live:L118 (with L133): 668.15 s total minus 661.18 s server prefill leaves only 6.97 s, less than the 6.54 s decode window that L133 derives from the server's own turn-end minus first-token latency, so the 160K run's decode window cannot be as large …(+100 chars)
