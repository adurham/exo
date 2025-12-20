#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <hostname> <query>"
  exit 1
fi

HOST="$1"
shift
QUERY="$*"

MODEL="${MODEL:-mlx-community/Qwen3-235B-Instruct-4bit}"

curl -sN -X POST "http://$HOST:52415/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
        \"model\": \"${MODEL}\",
        \"stream\": true,
        \"messages\": [{ \"role\": \"user\",   \"content\": \"$QUERY\"}]
      }" |
  grep --line-buffered '^data:' |
  grep --line-buffered -v 'data: \[DONE\]' |
  cut -d' ' -f2- |
  jq -r --unbuffered '.choices[].delta.content // empty' |
  awk '{ORS=""; print; fflush()} END {print "\n"}'
