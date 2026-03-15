#!/bin/bash

FILE="activity.txt"

if [ ! -d ".git" ]; then
  echo "Not a git repository!"
  exit 1
fi

START_DATE="2026-03-15"
END_DATE="2026-04-25"

MESSAGES=(
  "fix bug"
  "small update"
  "refactor"
  "cleanup"
  "minor changes"
  "improve logic"
  "update docs"
  "adjust config"
  "code improvements"
  "quick fix"
  "testing changes"
  "feature tweaks"
)

current="$START_DATE"

while [[ "$current" < "$(date -I -d "$END_DATE + 1 day")" ]]; do

  # 50% chance to skip entire day
  if (( RANDOM % 100 < 50 )); then
    echo "Skipping $current"
    current=$(date -I -d "$current + 1 day")
    continue
  fi

  # Random commits today (1–6)
  commit_count=$((RANDOM % 6 + 1))

  # Random base hour
  base_hour=$((RANDOM % 12 + 8))

  echo "Creating $commit_count commits on $current"

  for ((i=1; i<=commit_count; i++)); do

    # Sometimes cluster commits together
    if (( RANDOM % 100 < 35 )); then
      hour=$base_hour
      minute=$((RANDOM % 20))
    else
      hour=$((RANDOM % 16 + 6))
      minute=$((RANDOM % 60))
    fi

    second=$((RANDOM % 60))

    commit_datetime="$current $(printf "%02d:%02d:%02d" "$hour" "$minute" "$second")"

    # Small random file change
    echo "$(date +%s)-$RANDOM" >> "$FILE"

    # Sometimes create extra file noise
    if (( RANDOM % 100 < 30 )); then
      echo "temp $RANDOM" >> temp.log
    fi

    git add .

    # Random commit message
    msg=${MESSAGES[$RANDOM % ${#MESSAGES[@]}]}

    GIT_AUTHOR_DATE="$commit_datetime" \
    GIT_COMMITTER_DATE="$commit_datetime" \
    git commit -m "$msg"

  done

  current=$(date -I -d "$current + 1 day")
done

echo "Random commit history generated!"