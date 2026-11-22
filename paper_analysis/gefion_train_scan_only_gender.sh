#!/bin/bash -l
#SBATCH --account=cu_0080
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=3-00:00:00
#SBATCH --job-name=mqvae_scan_gender
#SBATCH --output=/dcai/projects/cu_0080/results/mqvae_scan_gender_%j.out
#SBATCH --error=/dcai/projects/cu_0080/results/mqvae_scan_gender_%j.err

set -euo pipefail

# Submit with, for example:
#   sbatch --export=ALL,GENDER=F,SEED=42,RUN_TAG=paper_v1 paper_analysis/gefion_train_scan_only_gender.sh
#   sbatch --export=ALL,GENDER=M,SEED=42,RUN_TAG=paper_v1 paper_analysis/gefion_train_scan_only_gender.sh

GENDER="${GENDER:-F}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-paper_v1}"
if [[ "$GENDER" != "F" && "$GENDER" != "M" ]]; then
    echo "GENDER must be F or M, got: $GENDER" >&2
    exit 2
fi

ACCOUNT_ROOT=/dcai/projects/cu_0080
TRANSFER_ROOT="$ACCOUNT_ROOT/xfer/sftp/inbox/REACT-Q"
CODE_ROOT="$ACCOUNT_ROOT/MQVAE"
DATA_ROOT="$ACCOUNT_ROOT/results/MQVAE_gefion_20260823"
SPLIT_ROOT="$ACCOUNT_ROOT/datasets"
CHECKPOINT_ROOT="$ACCOUNT_ROOT/results/checkpoints"
EXTRA_PACKAGES="$ACCOUNT_ROOT/mqvae_extra_packages"

RUN_NAME="GEFION_SCANONLY_ANYPLAQ_${GENDER}_${RUN_TAG}_seed${SEED}"
PREP_NAME="${RUN_NAME}_metadata"
TRAIN_SPLIT="$SPLIT_ROOT/3d_bodyscan_20260817_gefion_20260823_train_${GENDER}.tsv"
TEST_SPLIT="$SPLIT_ROOT/3d_bodyscan_20260817_gefion_20260823_test_${GENDER}.tsv"
METADATA="$TRANSFER_ROOT/body_scan_metadata20260817.tsv"

# This is the exact ABI-compatible stack that cleared all project imports.
# Do not module purge: Gefion's default/slurm modules are required for sbatch.
module load GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1
module load trimesh/3.23.5

PYTHON="${MQVAE_PYTHON:-$(command -v python)}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$EXTRA_PACKAGES"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$CODE_ROOT"
mkdir -p "$CHECKPOINT_ROOT"

for required in \
    "$DATA_ROOT/simplified_padded_tensors.npz" \
    "$DATA_ROOT/scan_geometry_features.npz" \
    "$TRAIN_SPLIT" \
    "$TEST_SPLIT" \
    "$METADATA"
do
    if [[ ! -f "$required" ]]; then
        echo "Missing required input: $required" >&2
        exit 3
    fi
done

echo "========== ENVIRONMENT =========="
nvidia-smi
"$PYTHON" -c "import sys, torch, numpy, pandas, scipy, sklearn, trimesh, open3d, matplotlib; assert sys.version_info[:2] == (3, 11); assert torch.cuda.is_available(); print('[ENV] ALL REQUIRED IMPORTS WORK'); print('[ENV] GPU', torch.cuda.get_device_name(0))"

echo "========== BUILD TARGET-ENRICHED SPLITS =========="
"$PYTHON" run_cardiovascular_target_benchmark.py \
  --name_prefix "$PREP_NAME" \
  --dataroot "$DATA_ROOT" \
  --metadata_tsv "$METADATA" \
  --train_split_tsv "$TRAIN_SPLIT" \
  --test_split_tsv "$TEST_SPLIT" \
  --checkpoints_dir "$CHECKPOINT_ROOT" \
  --only_gender "$GENDER" \
  --targets anyplaq \
  --variants conv_pointlabels_scan_age \
  --audit_only

ENRICHED_ROOT="$CHECKPOINT_ROOT/$PREP_NAME/enriched_splits"

echo "========== TRAIN SCAN-ONLY ENCODER =========="
# The encoder receives no age, BMI, sex, or disease condition. The plaque head
# is supervised using training labels, while the held-out test remains untouched.
# Open3D TriangleMesh objects are not pickleable, so num_threads must remain 0.
"$PYTHON" train_mesh_vae.py \
  --phase train \
  --name "$RUN_NAME" \
  --checkpoints_dir "$CHECKPOINT_ROOT" \
  --dataroot "$DATA_ROOT" \
  --cond_tsv "$ENRICHED_ROOT/train_enriched.tsv" \
  --feature_source scan \
  --only_gender "$GENDER" \
  --gpu_ids 0 \
  --seed "$SEED" \
  --split_seed "$SEED" \
  --split_ratio 0.10 \
  --niter 500 \
  --niter_decay 1500 \
  --batch_size 8 \
  --num_threads 0 \
  --save_epoch_freq 250 \
  --beta 0.001 \
  --beta_warmup_epochs 30 \
  --use_preprocessed \
  --use_pointcloud \
  --no-use_cond \
  --encoder_cond_mode none \
  --encoder_arch meshconv \
  --encoder_max_points 4096 \
  --mesh_use_knn \
  --mesh_knn_k 12 \
  --mesh_knn_max_points 4096 \
  --use_scan_geometry_features \
  --require_scan_geometry_features \
  --scan_geometry_features_file scan_geometry_features.npz \
  --use_point_labels \
  --point_label_pooling \
  --point_label_regions 6 \
  --aux_predict_binary \
  --no-aux_predict_regression \
  --aux_binary_col benchmark_anyplaq \
  --aux_binary_weight 10 \
  --aux_binary_pos_weight 0 \
  --no-use_amp \
  --no-cond_athero \
  --no-cond_plaque \
  --no-aux_predict_athero \
  --no-aux_predict_plaque \
  --no-balance_athero

echo "========== HELD-OUT TEST =========="
"$PYTHON" diagnose_mesh_binary.py \
  --name "$RUN_NAME" \
  --checkpoints_dir "$CHECKPOINT_ROOT" \
  --dataroot "$DATA_ROOT" \
  --cond_tsv "$ENRICHED_ROOT/test_enriched.tsv" \
  --feature_source scan \
  --only_gender "$GENDER" \
  --gpu_ids 0 \
  --which_epoch best \
  --batch_size 8 \
  --use_preprocessed \
  --use_pointcloud \
  --no-use_cond

PREDICTIONS=$(find "$CHECKPOINT_ROOT/$RUN_NAME/binary_diagnostics" -type f -name predictions.csv | sort | tail -n 1)
if [[ -z "$PREDICTIONS" ]]; then
    echo "Could not locate held-out predictions for $RUN_NAME" >&2
    exit 4
fi

echo "========== PARTICIPANT-BOOTSTRAP CONFIDENCE INTERVALS =========="
"$PYTHON" paper_analysis/bootstrap_metrics.py \
  --input "${RUN_NAME}=${PREDICTIONS}" \
  --task binary \
  --bootstraps 5000 \
  --seed "$SEED" \
  --output-dir "$CHECKPOINT_ROOT/$RUN_NAME/paper_ci"

echo "========== EXTRACT DETERMINISTIC LATENTS =========="
"$PYTHON" paper_analysis/extract_mesh_latents.py \
  --name "$RUN_NAME" \
  --dataroot "$DATA_ROOT" \
  --checkpoints-dir "$CHECKPOINT_ROOT" \
  --train-tsv "$ENRICHED_ROOT/train_enriched.tsv" \
  --test-tsv "$ENRICHED_ROOT/test_enriched.tsv" \
  --gender "$GENDER" \
  --which-epoch best \
  --gpu-ids 0 \
  --batch-size 8 \
  --output-dir "$CHECKPOINT_ROOT/$RUN_NAME/latents" \
  --use_preprocessed \
  --use_pointcloud \
  --no-use_cond

echo "========== COMPLETE =========="
echo "Run:         $RUN_NAME"
echo "Summary:     $CHECKPOINT_ROOT/$RUN_NAME/binary_summary.csv"
echo "Confidence:  $CHECKPOINT_ROOT/$RUN_NAME/paper_ci/bootstrap_metric_intervals.csv"
echo "Latents:     $CHECKPOINT_ROOT/$RUN_NAME/latents"
