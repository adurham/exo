#!/bin/sh
# p25 -- sustained-load soak take 3: full-length run to characterize the
# late-run device degradation seen in p24 (onset ~20 min in, both nodes,
# worse on m4-2). Same harness as p24 (iostat device truth + p23_ssd_soak2.py)
# plus a 60 s environment sampler (swap / vm_stat / therm).
cd "$HOME" || exit 1
NODE=$(hostname -s)
DUR="${1:-2400}"
echo "p25 driver start $(date '+%F %T') node=$NODE dur=${DUR}s"
nohup iostat -d -c "$DUR" -w 1 disk0 > "$HOME/p25-iostat-$NODE.txt" 2>&1 &
IPID=$!
echo "iostat pid $IPID"
nohup sh "$HOME/p25_env.sh" "$DUR" > "$HOME/p25-env-$NODE.log" 2>&1 &
EPID=$!
echo "env pid $EPID"
python3 "$HOME/p23_ssd_soak2.py" "$((DUR-60))" 8 > "$HOME/p25-soak-$NODE.log" 2>&1
echo "python rc=$?"
sleep 2
kill $IPID 2>/dev/null
kill $EPID 2>/dev/null
echo "p25 driver done $(date '+%F %T')"
tail -30 "$HOME/p25-soak-$NODE.log"
