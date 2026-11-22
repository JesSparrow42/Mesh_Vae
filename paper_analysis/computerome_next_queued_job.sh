#!/usr/bin/env bash
# Submit this payload from MQVAE_new after requesting one GPU via PBS or Slurm.
set -euo pipefail
JOB_CODE_ROOT="${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$JOB_CODE_ROOT" ]]; then
  echo "Submit this script through the GPU scheduler from the MQVAE_new folder." >&2
  exit 2
fi
cd "$JOB_CODE_ROOT"
if [[ ! -f paper_analysis/computerome_next_all.sh ]]; then
  echo "Computerome study scripts not found in scheduler submit directory: $JOB_CODE_ROOT" >&2
  exit 3
fi
mkdir -p logs
echo "[JOB] Host: $(hostname)"
echo "[JOB] Code: $PWD"
echo "[JOB] GPU visibility: ${CUDA_VISIBLE_DEVICES:-scheduler did not set a mask}"
echo "[JOB] Output: $PWD/logs/next_all_gpu0.log"
exec bash paper_analysis/computerome_next_all.sh 0 > logs/next_all_gpu0.log 2>&1
