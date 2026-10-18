[FIX] WRITEUP.md:1105 "The guard's `action` knob (`warn` vs `error`) was not toggled -- it needs a relaunch."
SRC: WRITEUP.md:1389 "CHANGED from `action=\"error\"` to `action=\"truncate\"` (`4c0fdb35b`, client rule `f21f8ad24c`)"
FIX: The guard's `action` knob now defaults to `truncate` (§12.6); `warn` vs `truncate` was not A/B-toggled -- it needs a relaunch.

[FIX] WRITEUP.md:207 "reboot became a first-class operation and SIGKILL was banned on the data path"
SRC: WRITEUP.md:329 "the reason SIGKILL is banned on the deploy path -- the watchdog's own SIGKILL is a separate"
FIX: reboot became a first-class operation and SIGKILL was banned on the deploy path (the watchdog's own SIGKILL is separate).

[FIX] WRITEUP.md:68 "Every doc in the corpus pins 115000 MB (120.6 GB)"
SRC: WRITEUP.md:583 "The evidence pack's D4.4 formula reads the same knob as 120.0 GB; the corpus's own convention"
FIX: Most corpus docs pin 115000 MB (120.6 GB) and the pack's D4.4 formula reads 120.0 GB; we use 125.83 GB.

[FIX] WRITEUP.md:54 "guard / limiter | live fire at committed=269, period=1"
SRC: WRITEUP.md:1386 "the period-1 banana repro no longer guard-killed"
FIX: append to basis: "(pre-§12.6 rule; the contiguous-period escape ships default ON and no longer kills this repro)".

CHECKED: 34

END S3
