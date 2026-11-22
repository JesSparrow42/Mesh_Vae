#!/usr/bin/env bash
# Source this file; no installations, upgrades, module purge, or --export settings.
set -euo pipefail
STUDY_PLATFORM="${1:?platform required}"
STUDY_CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$STUDY_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
case "$STUDY_PLATFORM" in
  gefion)
    module load GCC/12.3.0 OpenMPI/4.1.5
    module load PyTorch/2.1.2-CUDA-12.1.1
    module load trimesh/3.23.5
    STUDY_PYTHON="${MQVAE_PYTHON:-$(command -v python)}"
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/dcai/projects/cu_0080/mqvae_extra_packages"
    ;;
  computerome)
    STUDY_PYTHON="${MQVAE_PYTHON:-/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python}"
    # Keep a scheduler GPU mask. Torque can instead list the assigned device
    # as <hostname>-gpu<index> in PBS_GPUFILE.
    if [[ -n "${PBS_JOBID:-}" && -z "${CUDA_VISIBLE_DEVICES+x}" && -n "${PBS_GPUFILE:-}" ]]; then
      if [[ ! -r "$PBS_GPUFILE" ]]; then
        echo "Cannot read assigned GPU list: $PBS_GPUFILE" >&2
        return 4
      fi
      while IFS= read -r assigned_device; do
        if [[ "$assigned_device" =~ -gpu([0-9]+)$ ]]; then
          export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
          break
        fi
      done < "$PBS_GPUFILE"
    fi
    if [[ -n "${PBS_JOBID:-}${SLURM_JOB_ID:-}" && -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
      echo "GPU job lacks CUDA_VISIBLE_DEVICES or a readable PBS_GPUFILE GPU entry." >&2
      return 4
    fi
    if [[ -z "${PBS_JOBID:-}${SLURM_JOB_ID:-}" ]]; then
      export CUDA_VISIBLE_DEVICES="${STUDY_GPU:-0}"
    fi
    ;;
  *) echo "Unknown platform: $STUDY_PLATFORM" >&2; return 2 ;;
esac
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

study_check_gpu_environment() {
    "$STUDY_PYTHON" -c "import sys, torch, numpy, pandas, scipy, sklearn, trimesh, open3d, matplotlib; import meshvae.conditioning; assert torch.cuda.is_available(), 'No GPU available'; print('[ENV] Python:', sys.executable); print('[ENV] GPU:', torch.cuda.get_device_name(0)); print('[ENV] Required imports work')"
}
