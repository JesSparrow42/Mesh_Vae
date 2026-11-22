#!/bin/bash -l
#SBATCH --account=cu_0080
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=3-00:00:00
#SBATCH --array=0-13%2
#SBATCH --job-name=plaque_territory_volume
#SBATCH --output=next_territory_volume_%A_%a.out
#SBATCH --error=next_territory_volume_%A_%a.err

set -euo pipefail
# Submit from the MQVAE directory. Both F and M are already in the array.
cd "${SLURM_SUBMIT_DIR:?Submit this script using sbatch from MQVAE}"
source paper_analysis/next_steps_environment.sh gefion
"$STUDY_PYTHON" -c "import torch, numpy, pandas, trimesh, open3d; assert torch.cuda.is_available(), 'No GPU available'"
exec "$STUDY_PYTHON" paper_analysis/run_plaque_study.py --platform gefion --suite territory_volume --task-index "${SLURM_ARRAY_TASK_ID:?Missing Slurm array index}"

