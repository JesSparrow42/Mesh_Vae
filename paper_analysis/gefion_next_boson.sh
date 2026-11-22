#!/bin/bash -l
#SBATCH --account=cu_0080
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=3-00:00:00
#SBATCH --array=0-11%2
#SBATCH --job-name=plaque_boson
#SBATCH --output=next_boson_%A_%a.out
#SBATCH --error=next_boson_%A_%a.err

set -euo pipefail
# Each sex uses the same participants for both priors and all three training seeds.
# 0..5 = F, 6..11 = M; within each sex: seed42/43/44 x Gaussian-MMD/Boson-MMD.
cd "${SLURM_SUBMIT_DIR:?Submit this script using sbatch from the code folder}"
source paper_analysis/next_steps_environment.sh gefion
study_check_gpu_environment
TASK="${SLURM_ARRAY_TASK_ID:?Missing Slurm array index}"
if (( TASK < 0 || TASK > 11 )); then
  echo "Boson array index must be between 0 and 11" >&2
  exit 2
fi
GENDERS=(F M)
ARMS=(gaussian_mmd boson_mmd)
GENDER="${GENDERS[TASK / 6]}"
SEED="$((42 + (TASK % 6) / 2))"
ARM="${ARMS[TASK % 2]}"
REFERENCE_TAG="${1:-next_v1}"
REFERENCE_ROOT="${2:-/dcai/projects/cu_0080/results/checkpoints}"
REFERENCE_SEED="${3:-42}"
RUN_TAG="${4:-next_boson_v1}"
REFERENCE_RUN="$REFERENCE_ROOT/GEFION_NEXT_plaque_all_log_${GENDER}_${REFERENCE_TAG}_seed${REFERENCE_SEED}"
exec "$STUDY_PYTHON" paper_analysis/run_plaque_study.py \
  --platform gefion --suite plaque --experiment all_log --gender "$GENDER" \
  --seed "$SEED" --prior-arm "$ARM" --run-tag "$RUN_TAG" \
  --reference-run "$REFERENCE_RUN"
