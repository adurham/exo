#!/bin/sh
# p25 launcher -- starts the take-3 characterization soak detached on this node.
# Refuses to start if a soak is already running.
cd "$HOME" || exit 1
if pgrep -f p23_ssd_soak2 >/dev/null 2>&1; then
  echo "P25_SKIP ALREADY_RUNNING"
  exit 1
fi
NODE=$(hostname -s)
nohup sh "$HOME/p25_driver.sh" 2400 > "$HOME/p25-driver-$NODE.log" 2>&1 &
echo "P25_LAUNCHED pid=$! node=$NODE"
