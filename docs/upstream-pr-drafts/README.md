# Upstream PR drafts (held, not yet opened)

Branches in this directory's `.md` files are built + tested clean on M4 Max but held
from public submission for the reasons documented in
[`../upstream-prs.md` → "Held / drafted but not yet opened"](../upstream-prs.md).

These are the as-prepared PR descriptions, kept here so they don't drift out of sync
with the branches at `adurham:pr/*` if/when we decide to open them.

To open one:

```bash
# example: PR #2 (chunked SDPA)
cd ~/repos/mlx
gh pr create \
  --repo ml-explore/mlx \
  --head adurham:pr/sdpa-chunked-dispatch \
  --base main \
  --title "feat: chunked SDPA dispatch for long key sequences (>65K)" \
  --body-file ~/repos/exo/docs/upstream-pr-drafts/02-mlx-sdpa-chunked.md
```
