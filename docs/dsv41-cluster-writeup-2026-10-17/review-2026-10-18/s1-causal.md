[FIX] WRITEUP.md:L236 "fork, hardcodes `IBV_QPT_UC` and provides reliability over UC in software for the TP"
SRC: WRITEUP.md:682 "all_gather path, which has no reliability layer at all"; mesh_impl.h:1935 "all_gather was left on the bare"
FIX: provides reliability over UC in software for the all_reduce family; all_gather stays on raw UC posts with a StallWatch only (§6.2).
[BLOCK] WRITEUP.md:L405 "use-after-free segfault), `b4871e8` gave it a real subgroup teardown/rebuild path instead of"
SRC: mlx: git log -1 --format=%s b4871e8 -> "fix(jaccl): reconnect() must reset the v2 pool QP - TP post-reconnect segfault"
FIX: `b4871e8` made that QP-only fallback reset and eagerly re-arm the v2 pool QP, which it had silently skipped.
[BLOCK] WRITEUP.md:L529 "carries the TP slice -- gate/up OUT-tiles plus matching down K-tiles, `gu_svh`/`dn_suh`"
SRC: exl3/loader.py:11 "gu_suh and dn_svh replicated, partial outputs all_summed"; loader.py:203-207 slices gu_svh and dn_suh
FIX: carries the TP slice -- gate/up OUT-tiles plus matching down K-tiles, `gu_suh`/`dn_svh` replicated
[NOTE] WRITEUP.md:L564 "to 200 (the limit is read in mlx `device.cpp:768-773`; the launcher lands it in exo `463ac5d`)"
SRC: grep -n "max_mb_per_buffer_ = env" mlx/backend/metal/device.cpp -> "807:  max_mb_per_buffer_ = env::max_mb_per_buffer(max_mb_per_buffer_);"
FIX: (the limit is read in mlx `device.cpp:807`; the launcher lands it in exo `463ac5d`)
CHECKED: 58
END S1
