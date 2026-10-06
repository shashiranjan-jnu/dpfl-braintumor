#!/usr/bin/env bash
# Usage  bash scripts/push_results.sh "commit message"
# Run inside a results working copy. If GIT_PUSH_URL is set, it is used as the
# remote, so the token never needs to be stored in .git/config.
set -u
MSG="${1:-Add results}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
REMOTE="${GIT_PUSH_URL:-origin}"

git add results/ docs/lab_notebook.md 2>/dev/null
if git diff --cached --quiet; then
  echo "Nothing to push."
  exit 0
fi
git commit -q -m "$MSG"

for i in 1 2 3; do
  if git pull -q --rebase --autostash "$REMOTE" "$BRANCH" 2>/dev/null && git push -q "$REMOTE" "HEAD:$BRANCH" 2>/dev/null; then
    echo "Pushed on attempt $i."
    exit 0
  fi
  echo "Attempt $i failed, retrying in $((i * 10)) seconds."
  sleep $((i * 10))
done
echo "Push failed after 3 attempts." >&2
exit 1
