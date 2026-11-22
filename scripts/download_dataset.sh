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
$PYTHON_BIN - "$@" <<'PY'
import argparse
from mm_systemonebias.data import DEFAULT_REPO_ID, download_dataset
p = argparse.ArgumentParser(description="Download MM-SystemOneBias dataset")
p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
p.add_argument("--revision")
p.add_argument("--output-dir")
p.add_argument("--allow-patterns", nargs="+")
a = p.parse_args()
download_dataset(a.repo_id, a.revision, a.output_dir, a.allow_patterns)
PY
