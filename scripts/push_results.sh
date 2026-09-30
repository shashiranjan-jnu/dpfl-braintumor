#!/usr/bin/env bash
# Usage  bash scripts/push_results.sh "commit message"
# Run inside the results working copy on the dev branch.
set -u
MSG="${1:-Add results}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

git add results/ docs/lab_notebook.md 2>/dev/null
if git diff --cached --quiet; then
  echo "Nothing to push."
  exit 0
fi
git commit -q -m "$MSG"

for i in 1 2 3; do
  if git pull -q --rebase --autostash origin "$BRANCH" && git push -q origin "$BRANCH"; then
    echo "Pushed on attempt $i."
    exit 0
  fi
  echo "Attempt $i failed, retrying in $((i * 10)) seconds."
  sleep $((i * 10))
done
echo "Push failed after 3 attempts." >&2
exit 1
