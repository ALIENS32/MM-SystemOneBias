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
model="${1:?usage: smoke.sh MODEL_ID --dataset-path PATH [options]}"
shift
exec "$PYTHON_BIN" -m mm_systemonebias.inference --config "$ROOT/configs/models/${model}.yaml" --output-dir "${MM_SYSTEMONEBIAS_RUNS:-$HOME/.cache/mm-systemonebias/runs}/${model}-smoke" --limit-originals-per-group 1 "$@"
