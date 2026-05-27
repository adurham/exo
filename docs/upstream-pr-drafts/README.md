# Upstream PR drafts

Tracked-in-repo copies of PR descriptions, paired with branches at `adurham:pr/*`.

- **Files for PRs that have been opened** kept for reference + diff history.
- **Files for held PRs (not yet opened)** documented in `../upstream-prs.md`
  → "Held / drafted but not yet opened" section.

| File | Status | PR / branch |
|---|---|---|
| `02-mlx-sdpa-chunked.md` | **HELD** | `adurham:pr/sdpa-chunked-dispatch` (stacked on mlx#3594) |
| `04-mlx-head-dim-192-256.md` | **HELD** | `adurham:pr/sdpa-head-dim-192-256` (float32 limitation pending) |
| `06-mlx-allocator-coalesce.md` | **OPENED** | [mlx#3596](https://github.com/ml-explore/mlx/pull/3596) |

(PRs #1, #3, #5 are committed via the PR body itself on GitHub — we kept the
drafts in `/tmp/pr-drafts/` during the session but only the held ones are
worth tracking long-term in the repo.)

To open a held PR:

```bash
# example: PR for chunked SDPA, once #3594 lands
cd ~/repos/mlx
gh pr create \
  --repo ml-explore/mlx \
  --head adurham:pr/sdpa-chunked-dispatch \
  --base main \
  --title "feat: chunked SDPA dispatch for long key sequences (>65K)" \
  --body-file ~/repos/exo/docs/upstream-pr-drafts/02-mlx-sdpa-chunked.md
```
