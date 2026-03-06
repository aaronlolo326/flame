#!/bin/bash
set -e

echo "On branch: $(git branch --show-current)"
git status

echo
read -p "Stage new files too? (y/N): " stage_new
if [[ "$stage_new" =~ ^[Yy]$ ]]; then
  git add -A
else
  git add -u
fi

echo
read -p "Commit description: " desc
desc="${desc:-default}"

echo
read -p "Commit and push? (y/N): " ok
if [[ ! "$ok" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

git commit -m "$desc"
git push