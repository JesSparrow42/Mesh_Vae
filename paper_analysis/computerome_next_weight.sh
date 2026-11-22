#!/usr/bin/env bash
set -euo pipefail
# Direct DPREACT GPU-host runner, matching the existing Computerome workflow.
# Positional argument 1 selects the GPU. F and M run sequentially on that GPU.
STUDY_GPU="${1:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source paper_analysis/next_steps_environment.sh computerome
study_check_gpu_environment
exec "$STUDY_PYTHON" paper_analysis/run_plaque_study.py --platform computerome --suite weight
