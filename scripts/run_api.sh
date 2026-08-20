#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"
export TASK_ROOT="${TASK_ROOT:-$PROJECT_ROOT/runtime/tasks}"
export MARKETING_AGENT_ROOT="${MARKETING_AGENT_ROOT:-$PROJECT_ROOT}"
export AGENT_TOOLSET="${AGENT_TOOLSET:-production}"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/runtime/venv/gpu/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python runtime is missing or not executable: $PYTHON_BIN" >&2
  exit 1
fi

exec "$PYTHON_BIN" -m uvicorn marketing_agent.api:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}"
