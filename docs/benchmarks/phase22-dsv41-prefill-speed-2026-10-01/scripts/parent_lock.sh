#!/bin/bash
# parent_lock.sh take|drop -- hold ~/dsv41-gpu.lock on BOTH Macs for a two-node run.
#
# Each take uses a unique token and its holder only watches ITS OWN file
# (~/parent-lock.want.<token>). The old version shared one filename: a take
# issued within ~2 s of a drop re-created the file before the previous holder's
# poll saw it disappear, so that holder kept the lock indefinitely (the
# ~90-minute stall on 2026-09-29). Holders also self-release after MAXHOLD
# seconds, so a crashed gateway-side run can never pin the GPUs.
S=~/.hermes/cache/scratch/exl3patch
TOKF=$S/.parent_lock_token
MAXHOLD=${PARENT_LOCK_MAXHOLD:-7200}
case "$1" in
take)
  TOK=$(date +%s)-$$-$RANDOM; echo "$TOK" > $TOKF
  # Parent priority: flock is not FIFO, so a queue of agent jobs can starve the
  # parent for hours. SIGSTOP every QUEUED lockf (not the current holder) so it
  # cannot win the lock when the holder finishes; drop SIGCONTs them. Nothing
  # queued has started yet, so no agent work is lost.
  # (SIGSTOP-pausing of queued lockf was REMOVED 2026-09-29: a lockf can win the
  # lock between the scan and the signal, and a stopped holder pins the GPU
  # forever -- it caused a 31-minute stall. Agents are capped at 20-min holds.)
  for n in macstudio-m4-1 macstudio-m4-2; do
    ssh $n "touch ~/parent-lock.want.$TOK; nohup lockf -k ~/dsv41-gpu.lock sh -c 'touch ~/parent-lock.held.$TOK; t=0; while [ -f ~/parent-lock.want.$TOK ] && [ \$t -lt $MAXHOLD ]; do sleep 2; t=\$((t+2)); done; rm -f ~/parent-lock.held.$TOK ~/parent-lock.want.$TOK' >/dev/null 2>&1 &"
  done
  for n in macstudio-m4-1 macstudio-m4-2; do
    ssh $n "i=0; while [ ! -f ~/parent-lock.held.$TOK ]; do sleep 5; i=\$((i+1)); [ \$i -gt 720 ] && { echo \"\$(hostname -s) LOCK TIMEOUT\"; exit 1; }; done; echo \"\$(hostname -s) lock held\"" || { "$0" drop; exit 1; }
  done ;;
drop)
  TOK=$(cat $TOKF 2>/dev/null)
  for n in macstudio-m4-1 macstudio-m4-2; do
    ssh $n "rm -f ~/parent-lock.want.$TOK; i=0; while [ -f ~/parent-lock.held.$TOK ] && [ \$i -lt 10 ]; do sleep 1; i=\$((i+1)); done; [ -f ~/parent-lock.held.$TOK ] && echo \"\$(hostname -s) WARNING holder still present\"; [ -f ~/parent-stopped.pids ] && { xargs kill -CONT < ~/parent-stopped.pids 2>/dev/null; rm -f ~/parent-stopped.pids; }; echo \"\$(hostname -s) lock released\""
  done ;;
esac
