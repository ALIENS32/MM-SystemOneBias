#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
profile="${1:-core}"
if [[ ! -f "configs/environments/${profile}.txt" ]]; then
  echo "Unknown environment profile: ${profile}" >&2
  echo "Choose core, vlm, gemma, clef, d1, d3, or autotrust" >&2
  exit 2
fi
env_dir="${MM_SYSTEMONEBIAS_ENV:-$HOME/.cache/mm-systemonebias/envs/$profile}"
python_bin="${PYTHON_BIN:-}"
if [[ -z "$python_bin" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python_bin=python3.12
  elif command -v python3 >/dev/null 2>&1; then
    python_bin=python3
  else
    echo "Python 3.12+ is required" >&2
    exit 1
  fi
fi
if ! "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "Python 3.12+ is required (found: $("$python_bin" --version 2>&1))" >&2
  exit 1
fi
"$python_bin" -m venv "$env_dir"
"$env_dir/bin/pip" install --upgrade pip
"$env_dir/bin/pip" install -e .
"$env_dir/bin/pip" install -r "configs/environments/${profile}.txt"
printf 'environment: %s\n' "$env_dir"
