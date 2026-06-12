#!/bin/zsh
# jaccl_wedge_watchdog.sh — detect a stalled exo runner and capture a stack
# sample on THIS node the moment generation wedges.
#
# Wedge signature (from the 2026-06-11 deadlock): the runner process stays
# alive and the GPU stays pegged, but runner stderr stops advancing — no new
# "MTP ACCEPT" / "Prefill progress" lines — while control-plane /state still
# shows a TextGeneration task Running. That is a stuck JACCL collective.
#
# This script polls the runner stderr mtime. If the log goes quiet for
# STALL_SECS while the runner PID is still alive, it fires `sample` (native,
# no root needed) and dumps the result to ~/.exo/wedge_dumps/. It writes one
# dump per wedge episode (debounced) so a long hang doesn't spam.
#
# Run one instance per node (see --install). Cross-rank diff the two dumps:
# the rank blocked in jaccl recv vs the rank that returned early from the
# step gate is the smoking gun for an asymmetric-collective deadlock.

set -u

STALL_SECS="${STALL_SECS:-45}"      # log silent this long => suspect wedge
POLL_SECS="${POLL_SECS:-10}"        # how often we check
SAMPLE_SECS="${SAMPLE_SECS:-3}"     # sample duration when firing
STDERR_LOG="$HOME/.exo/exo_log/runner_log/stderr.log"
DUMP_DIR="$HOME/.exo/wedge_dumps"
HOSTTAG="$(hostname -s)"

mkdir -p "$DUMP_DIR"

HEARTBEAT_LOG="$DUMP_DIR/watchdog.log"
log() {
  local line="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  print -r -- "$line"
  print -r -- "$line" >> "$HEARTBEAT_LOG"
}

# Largest-footprint spawn_main child of `-m exo` = the model runner.
find_runner_pid() {
  ps axo pid,rss,command \
    | grep 'spawn_main' | grep -v grep \
    | sort -k2 -n | tail -1 | sed -E 's/^[[:space:]]*([0-9]+).*/\1/'
}

mtime_of() { stat -f '%m' "$1" 2>/dev/null || echo 0; }

log "watchdog up on $HOSTTAG (stall=${STALL_SECS}s poll=${POLL_SECS}s sample=${SAMPLE_SECS}s)"
log "watching $STDERR_LOG"

last_episode_mtime=0   # debounce: don't re-dump the same stall episode
last_heartbeat=0       # emit a liveness line at most every 5 min

while true; do
  sleep "$POLL_SECS"
  now_hb="$(date +%s)"
  if (( now_hb - last_heartbeat >= 300 )); then
    log "heartbeat: alive, watching (last_episode_mtime=$last_episode_mtime)"
    last_heartbeat="$now_hb"
  fi
  pid="$(find_runner_pid)"
  [[ -z "$pid" ]] && { log "no runner pid (idle/restarting)"; continue; }

  lm="$(mtime_of "$STDERR_LOG")"
  now="$(date +%s)"
  quiet=$(( now - lm ))

  # Only suspect a wedge when the log has been quiet AND there is an active
  # generation task on the cluster. Quiet-while-idle is normal.
  if (( quiet >= STALL_SECS )); then
    # Is a TextGeneration actually Running right now? (cheap local API hit)
    running="$(curl -s --max-time 5 http://127.0.0.1:52415/state 2>/dev/null \
      | grep -c '"TextGeneration"' )"
    if (( running == 0 )); then
      continue   # genuinely idle, not a wedge
    fi
    # Debounce: only one dump per episode (keyed on the frozen mtime).
    if (( lm == last_episode_mtime )); then
      continue
    fi
    last_episode_mtime="$lm"
    ts="$(date '+%Y%m%d_%H%M%S')"
    out="$DUMP_DIR/wedge_${HOSTTAG}_${ts}_pid${pid}.sample.txt"
    log "WEDGE SUSPECTED: stderr quiet ${quiet}s, runner pid=$pid alive, TextGeneration Running -> sampling"
    {
      echo "=== wedge dump $HOSTTAG pid=$pid quiet=${quiet}s at $ts ==="
      echo "--- /state task summary ---"
      curl -s --max-time 5 http://127.0.0.1:52415/state 2>/dev/null \
        | python3 -c 'import sys,json,collections;
d=json.load(sys.stdin); c=collections.Counter();
[c.update([(t,v.get("taskStatus"))]) for k,w in d["tasks"].items() for t,v in w.items()];
print("\n".join(f"  {n}  {k}" for k,n in sorted(c.items())))' 2>/dev/null
      echo "--- sample $pid (${SAMPLE_SECS}s) ---"
      sample "$pid" "$SAMPLE_SECS" 2>&1
    } > "$out"
    log "wrote $out ($(wc -l < "$out") lines)"
  fi
done
