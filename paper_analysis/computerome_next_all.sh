#!/usr/bin/env bash
set -euo pipefail
# One background process, one GPU; all female/male suites run sequentially.
STUDY_GPU="${1:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source paper_analysis/next_steps_environment.sh computerome
study_check_gpu_environment
exec "$STUDY_PYTHON" paper_analysis/run_plaque_study.py --platform computerome --suite all
