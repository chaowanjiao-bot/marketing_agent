#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_BUILDER="${ENV_BUILDER:-$PROJECT_ROOT/runtime/venv/gpu/bin/python}"

create_env() {
  local name="$1"
  local requirements="$2"
  local environment="$PROJECT_ROOT/runtime/venv/$name"
  if [[ ! -x "$environment/bin/python" ]]; then
    "$ENV_BUILDER" -m virtualenv "$environment"
  fi
  "$environment/bin/python" -m pip install --upgrade pip
  if [[ "$name" == "vqascore" ]]; then
    "$environment/bin/python" -m pip install \
      torch==2.7.1 torchvision==0.22.1 \
      --index-url https://download.pytorch.org/whl/cu128
  fi
  "$environment/bin/python" -m pip install -r "$requirements"
}

case "${1:-all}" in
  powerpaint) create_env powerpaint "$PROJECT_ROOT/requirements/powerpaint.txt" ;;
  vqascore) create_env vqascore "$PROJECT_ROOT/requirements/vqascore.txt" ;;
  all)
    create_env powerpaint "$PROJECT_ROOT/requirements/powerpaint.txt"
    create_env vqascore "$PROJECT_ROOT/requirements/vqascore.txt"
    ;;
  *) echo "usage: $0 [powerpaint|vqascore|all]" >&2; exit 2 ;;
esac
