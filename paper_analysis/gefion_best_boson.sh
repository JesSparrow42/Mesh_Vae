#!/bin/bash -l
#SBATCH --account=cu_0080
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=3-00:00:00
#SBATCH --array=0-7%2
#SBATCH --job-name=best_boson
#SBATCH --output=best_boson_%A_%a.out
#SBATCH --error=best_boson_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the unpacked code folder}"
source paper_analysis/next_steps_environment.sh gefion
"$STUDY_PYTHON" -c "import torch; assert torch.cuda.is_available(); print('[GPU]', torch.cuda.get_device_name(0))"

# Two selected models x F/M x Gaussian-MMD/Boson-MMD, one training seed (42).
# Original Gaussian-KL runs are reused as references and are never overwritten.
exec "$STUDY_PYTHON" paper_analysis/selected_boson_study.py run \
  --task-index "${SLURM_ARRAY_TASK_ID:?Missing Slurm array index}" \
  --reference-tag "${1:-next_v1}" --run-tag "${2:-best_boson_v1}"
