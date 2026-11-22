#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source paper_analysis/next_steps_environment.sh gefion
exec "$STUDY_PYTHON" paper_analysis/run_plaque_study.py --platform gefion --audit-only

