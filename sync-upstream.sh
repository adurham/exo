#!/usr/bin/env bash
# sync-upstream.sh — pull upstream changes into the experimentation forks.
#
# WHY THIS EXISTS
#   exo + its mlx / mlx-lm submodules are long-lived experimentation forks that
#   pin RAW submodule SHAs (no branch tracking in .gitmodules). That means:
#     * MERGE, never rebase. Rebasing rewrites SHAs and orphans the exo pin.
#     * The same conflicts recur every pull (e.g. the deliberate "Revert Jaccl
#       refactor #3412" makes upstream's lib/jaccl/* edits conflict forever).
#   git rerere (enabled in all three repos) records each resolution once and
#   replays it automatically, so recurring conflicts stop being manual work.
#
# WHAT IT DOES
#   For each repo: fetch upstream, report ahead/behind, and (only with --go)
#   merge upstream into the CURRENTLY CHECKED-OUT branch. It NEVER pushes and
#   NEVER touches the exo submodule pin — you validate (build + quality-probe),
#   then commit/push/bump the pin yourself.
#
# USAGE
#   ./sync-upstream.sh            # dry-run: show what each repo is ahead/behind
#   ./sync-upstream.sh --go       # actually merge upstream into current branches
#   ./sync-upstream.sh --go exo   # only operate on a subset (exo|mlx|mlx-lm)
#
set -uo pipefail

EXO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# label : path-relative-to-EXO_ROOT : upstream-remote : upstream-branch
REPOS=(
  "exo:.:upstream:main"
  "mlx-lm:mlx-lm:upstream:main"
  "mlx:mlx:upstream:main"
)

GO=0
ONLY=""
for arg in "$@"; do
  case "$arg" in
    --go) GO=1 ;;
    exo|mlx|mlx-lm) ONLY="$arg" ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

c_bold=$'\e[1m'; c_grn=$'\e[32m'; c_yel=$'\e[33m'; c_red=$'\e[31m'; c_rst=$'\e[0m'

for entry in "${REPOS[@]}"; do
  IFS=':' read -r label rel remote ubr <<<"$entry"
  [[ -n "$ONLY" && "$ONLY" != "$label" ]] && continue
  dir="$EXO_ROOT/$rel"
  echo "${c_bold}======== $label  ($rel) ========${c_rst}"

  if ! git -C "$dir" remote get-url "$remote" >/dev/null 2>&1; then
    echo "  ${c_red}no '$remote' remote — skipping${c_rst}"; echo; continue
  fi

  git -C "$dir" fetch "$remote" --quiet
  cur="$(git -C "$dir" rev-parse --abbrev-ref HEAD)"
  read -r ahead behind < <(git -C "$dir" rev-list --left-right --count "HEAD...$remote/$ubr")
  echo "  branch: ${c_bold}$cur${c_rst}   vs $remote/$ubr -> ${c_grn}ahead $ahead${c_rst} / ${c_yel}behind $behind${c_rst}"

  if [[ "$behind" -eq 0 ]]; then
    echo "  ${c_grn}up to date — nothing to pull${c_rst}"; echo; continue
  fi

  echo "  ${c_yel}$behind upstream commit(s) to pull:${c_rst}"
  git -C "$dir" log --format='    %h %s' "HEAD..$remote/$ubr" | head -15
  [[ "$behind" -gt 15 ]] && echo "    ... ($((behind-15)) more)"

  # conflict preview (no working-tree changes)
  base="$(git -C "$dir" merge-base HEAD "$remote/$ubr")"
  conflicts="$(git -C "$dir" merge-tree --write-tree HEAD "$remote/$ubr" 2>/dev/null | grep -c '^CONFLICT' || true)"
  if [[ "${conflicts:-0}" -gt 0 ]]; then
    echo "  ${c_yel}predicted conflicts: $conflicts (rerere will auto-resolve known ones)${c_rst}"
  else
    echo "  ${c_grn}predicted: clean merge${c_rst}"
  fi

  if [[ "$GO" -eq 0 ]]; then
    echo "  ${c_bold}(dry-run — pass --go to merge)${c_rst}"; echo; continue
  fi

  if ! git -C "$dir" diff --quiet || ! git -C "$dir" diff --cached --quiet; then
    echo "  ${c_red}working tree dirty — commit/stash first, skipping merge${c_rst}"; echo; continue
  fi

  echo "  ${c_bold}merging $remote/$ubr into $cur ...${c_rst}"
  if git -C "$dir" merge --no-edit "$remote/$ubr"; then
    echo "  ${c_grn}merged cleanly${c_rst}"
  else
    # rerere may have resolved everything; check for remaining unmerged paths
    unresolved="$(git -C "$dir" diff --name-only --diff-filter=U)"
    if [[ -z "$unresolved" ]]; then
      echo "  ${c_grn}rerere resolved all conflicts — staging + completing merge${c_rst}"
      git -C "$dir" add -A && git -C "$dir" commit --no-edit
    else
      echo "  ${c_red}MANUAL RESOLUTION NEEDED in:${c_rst}"
      echo "$unresolved" | sed 's/^/    /'
      echo "  resolve, 'git add', 'git commit' — rerere will remember for next time."
    fi
  fi
  echo "  ${c_bold}validate before pinning:${c_rst} build + quality_probe_dsv4.py, then bump exo pin"
  echo
done

echo "${c_bold}Done.${c_rst} Nothing pushed. After validating a submodule, bump the exo pin:"
echo "  git -C $EXO_ROOT add mlx mlx-lm && git -C $EXO_ROOT commit -m 'chore: bump submodule pins'"
