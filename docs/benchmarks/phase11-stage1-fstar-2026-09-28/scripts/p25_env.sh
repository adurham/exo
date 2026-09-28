#!/bin/sh
# p25 environment sampler -- one sample per minute (swap, vm pages, thermal).
DUR="${1:-2400}"
i=0
while [ "$i" -lt "$DUR" ]; do
  echo "## t=${i}s $(date '+%F %T')"
  sysctl vm.swapusage 2>/dev/null
  vm_stat 2>/dev/null | sed -n '2,4p'
  pmset -g therm 2>/dev/null | sed -n '2,3p'
  i=$((i+60))
  sleep 60
done
