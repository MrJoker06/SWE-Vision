#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo "Error: .env file not found"
  exit 1
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL is not set}"
: "${MODEL_NAME:=gpt-5.2}"

IMAGE_PATH="${1:-./assets/test_image.png}"
QUERY="${2:-What is the gap between GPT5.2 and 6-year-olds from the chart?}"

python -m swe_vision.cli \
  --image "$IMAGE_PATH" \
  --model "$MODEL_NAME" \
  --reasoning \
  "$QUERY"