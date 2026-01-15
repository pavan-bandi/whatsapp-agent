#!/bin/bash

FILE="activity.txt"

# Ensure inside a git repo
if [ ! -d ".git" ]; then
  echo "Not a git repository!"
  exit 1
fi

START_DATE="2026-01-15"
END_DATE="2026-01-25"

current="$START_DATE"

while [[ "$current" < "$(date -I -d "$END_DATE + 1 day")" ]]; do

  # Random number of commits per day (1–4)
  commit_count=$((RANDOM % 4 + 1))

  for ((i=1; i<=commit_count; i++)); do

    # Random time
    hour=$(printf "%02d" $((RANDOM % 14 + 8)))   # 08–21
    minute=$(printf "%02d" $((RANDOM % 60)))
    second=$(printf "%02d" $((RANDOM % 60)))

    commit_datetime="$current $hour:$minute:$second"

    # Make file change
    echo "Update at $commit_datetime" >> "$FILE"

    # Stage changes
    git add .

    # Commit with backdated timestamp
    GIT_AUTHOR_DATE="$commit_datetime" \
    GIT_COMMITTER_DATE="$commit_datetime" \
    git commit -m "Update on $commit_datetime"

  done

  # Move to next day
  current=$(date -I -d "$current + 1 day")
done

echo "Done! Commits created from Jan 15 to Jan 25, 2026."