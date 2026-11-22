#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 OUTPUT_DIR PREDICTIONS.jsonl [PREDICTIONS.jsonl ...]" >&2
  exit 2
fi
output_dir=$1
shift
exec "$PYTHON_BIN" -m mm_systemonebias.metrics.cli --predictions "$@" --output-dir "$output_dir"
