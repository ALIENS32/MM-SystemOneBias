#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
env_dir="${MM_SYSTEMONEBIAS_AUTOTRUST_ENV:-$HOME/.cache/mm-systemonebias/envs/autotrust}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$env_dir/bin/python" -m mm_systemonebias.models.serve_autotrust "$@"
