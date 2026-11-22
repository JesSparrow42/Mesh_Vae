#!/usr/bin/env bash
# Payload for a Computerome GPU job array. Submit from MQVAE_new with slots 1-50.
set -euo pipefail
JOB_CODE_ROOT="${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-}}"
ARRAY_SLOT="${PBS_ARRAYID:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "$JOB_CODE_ROOT" || ! "$ARRAY_SLOT" =~ ^([1-9]|[1-4][0-9]|50)$ ]]; then
  echo "Submit from MQVAE_new as a GPU job array with slots 1-50." >&2
  exit 2
fi
STUDY_TASK_INDEX="$((ARRAY_SLOT - 1))"
cd "$JOB_CODE_ROOT"
if [[ ! -f paper_analysis/run_plaque_study.py ]]; then
  echo "Study code not found in scheduler submit directory: $JOB_CODE_ROOT" >&2
  exit 3
fi
mkdir -p logs
exec > "logs/next_task_${STUDY_TASK_INDEX}.log" 2>&1
echo "[JOB] Host: $(hostname)"
echo "[JOB] Code: $PWD"
echo "[JOB] Task: $STUDY_TASK_INDEX"
echo "[JOB] Array slot: $ARRAY_SLOT"
echo "[JOB] Scheduler: ${PBS_JOBID:-${SLURM_JOB_ID:-unknown}}"
echo "[JOB] GPU visibility: ${CUDA_VISIBLE_DEVICES:-scheduler did not set a mask}"
source paper_analysis/next_steps_environment.sh computerome
study_check_gpu_environment
exec "$STUDY_PYTHON" paper_analysis/run_plaque_study.py \
  --platform computerome --suite all --task-index "$STUDY_TASK_INDEX"
