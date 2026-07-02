#!/usr/bin/awk -f
# Gap analysis for exo prefill progress lines. Usage:
#   awk -f gap_analysis.awk -v total=727262 exo.log
# Prints chunks, median-ish stats, and all gaps > thresh (default 10s).
BEGIN { prev_t = -1; prev_k = -1; n = 0; if (thresh == 0) thresh = 10 }
$0 ~ ("Prefill progress: [0-9]+/" total) {
  # line: [ 2026-07-01 23:24:05.004 | DEBUG ...
  ts = $3               # HH:MM:SS.mmm
  split(ts, hms, ":")
  t = hms[1] * 3600 + hms[2] * 60 + hms[3]
  # tokens
  match($0, /Prefill progress: [0-9]+/)
  k = substr($0, RSTART + 18, RLENGTH - 18) + 0
  if (prev_t >= 0 && k > prev_k) {
    dt = t - prev_t
    if (dt < 0) dt += 86400
    n++
    sum += dt
    if (dt > max) { max = dt; max_at = prev_k }
    if (dt > thresh) printf "GAP %.1fs at token %d (%s)\n", dt, prev_k, ts
  }
  prev_t = t; prev_k = k
}
END {
  printf "chunks=%d total_wall=%.0fs mean=%.2fs max=%.1fs at %d\n", n, sum, sum/n, max, max_at
}