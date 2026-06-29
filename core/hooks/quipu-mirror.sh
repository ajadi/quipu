#!/usr/bin/env sh
# quipu-mirror.sh — one-way Quipu DB → memory/*.md sync hook.
#
# Register as a PostToolUse hook on task-close or run on-demand.
# Guard: if QUIPU_PROJECT_ID is unset, exits quietly (no-op).
set -eu

if [ -z "${QUIPU_PROJECT_ID:-}" ]; then
    exit 0
fi

python -m quipu mirror \
    --project-id "$QUIPU_PROJECT_ID" \
    --output-dir memory
