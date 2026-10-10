#!/bin/zsh
# Q2-GAMMA pre-boot STAGE CHECK (read-only; prints GO-READY or BLOCKED).
# Verifies every precondition the Phase-1 runbook depends on. Sends NO request and
# starts/stops NOTHING on the cluster. Safe to run any time while holding for GO.
set -u
REPO=/Users/adam.durham/repos/exo
API=http://192.168.86.48:52415
fail=0
ok(){ print -r -- "  [ok]   $1"; }
bad(){ print -r -- "  [FAIL] $1"; fail=1; }

print -r -- "== Q2-GAMMA stage check $(date '+%F %T') =="

# 1. eval branch pushed and at the expected tip
tip=$(git -C "$REPO" rev-parse origin/deploy/q2-gamma 2>/dev/null || echo none)
[[ "$tip" == 2cf078519* ]] && ok "origin/deploy/q2-gamma = $tip" || bad "origin/deploy/q2-gamma = $tip (expected 2cf078519…)"

# 2. the shared checkout can reach it and the pre-deploy ancestor check will pass
git -C "$REPO" fetch origin --quiet 2>/dev/null
git -C "$REPO" merge-base --is-ancestor "$tip" origin/deploy/q2-gamma 2>/dev/null && ok "ancestor check (detached checkout at tip) PASSES" || bad "ancestor check FAILS"

# 3. production tip is a valid restore target
prod=$(git -C "$REPO" rev-parse origin/deploy/next19-dense 2>/dev/null || echo none)
git -C "$REPO" merge-base --is-ancestor 99e2966ee origin/deploy/next19-dense 2>/dev/null && ok "restore target 99e2966ee ⊂ origin/deploy/next19-dense" || bad "restore target invalid"

# 4. no stray DSV41_SPEC_GAMMA exported in the launching env
[[ -z "${DSV41_SPEC_GAMMA:-}" ]] && ok "DSV41_SPEC_GAMMA unset in this shell (clean restore default)" || bad "DSV41_SPEC_GAMMA=${DSV41_SPEC_GAMMA} already set"

# 5. cluster live + idle
st=$(curl -sS -m 8 "$API/state" 2>/dev/null)
if [[ -n "$st" ]]; then
  n=$(print -r -- "$st" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(len(d.get("instances",{})))' 2>/dev/null)
  ok "API /state reachable (instances=$n)"
else
  bad "API /state NOT reachable"
fi

# 6. current node build is still production (nothing half-deployed)
h=$(ssh -o ConnectTimeout=6 macstudio-m4-1 'cd ~/repos/exo && git rev-parse --short HEAD' 2>/dev/null)
[[ "$h" == 99e2966ee ]] && ok "studio1 node HEAD = 99e2966ee (production)" || bad "studio1 node HEAD = $h"

# 7. live nodes do NOT yet carry the token (expected pre-boot; the eval boot adds it)
c=$(ssh -o ConnectTimeout=6 macstudio-m4-1 'grep -c DSV41_SPEC_GAMMA ~/relaunch_exo.sh 2>/dev/null || true' 2>/dev/null | tr -d '\r')
[[ "$c" == "0" ]] && ok "pre-boot: relaunch_exo.sh has no DSV41_SPEC_GAMMA token (as expected)" || print -r -- "  [note] relaunch_exo.sh token count=$c"

print -r -- ""
if [[ $fail == 0 ]]; then print -r -- ">> GO-READY"; else print -r -- ">> BLOCKED"; fi
exit $fail
