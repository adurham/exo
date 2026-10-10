#!/bin/sh
# q2b: READ-ONLY capture of the memory counters on both cluster nodes (ssh aliases studio1/studio2). No writes, no signals.
# usage: sh q2b_capture_node_readonly.sh   (run from a laptop that has the ssh aliases)
# Runner pid = the multiprocessing.spawn child of `python -m exo`; pass overrides as: R1=<pid> R2=<pid> sh ...
R1=${R1:-49330}   # studio1 runner (rank 1)   in the 2026-10-10 capture
R2=${R2:-59829}   # studio2 runner (rank 0)
for spec in "studio1 $R1" "studio2 $R2"; do
  set -- $spec
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$1" "date '+%F %T %Z'; echo '### '\$(hostname)' runner pid $2';
    sysctl iogpu.wired_limit_mb hw.memsize vm.swapusage;
    echo '### footprint -f bytes'; footprint -f bytes -p $2 | grep -E 'Footprint:|phys_footprint|IOAccelerator \(graphics\)';
    echo '### footprint --wired --swapped (binary units)'; footprint -p $2 --wired --swapped | sed -n '1,8p';
    echo '### footprint --sysFootprint'; footprint --sysFootprint --noCategories -p $2 | sed -n '/Auxiliary/,\$p';
    echo '### vmmap -summary'; vmmap -summary $2 | grep -E 'Physical footprint|IOAccelerator \(graphics\)|^TOTAL ';
    echo '### vm_stat'; vm_stat | grep -E 'Pages free|Pages wired|stored in compressor|occupied by compressor|Swapouts|Swapins';
    echo '### memory_pressure'; memory_pressure | tail -2"
done
