#!/bin/bash -l
set -euo pipefail

ACCOUNT_ROOT=/dcai/projects/cu_0080
TRANSFER_ROOT="$ACCOUNT_ROOT/xfer/sftp/inbox/REACT-Q"
CODE_ROOT="$ACCOUNT_ROOT/MQVAE"
DATA_ROOT="$ACCOUNT_ROOT/results/MQVAE_gefion_20260823"
SPLIT_ROOT="$ACCOUNT_ROOT/datasets"
CHECKPOINT_ROOT="$ACCOUNT_ROOT/results/checkpoints"
EXTRA_PACKAGES="$ACCOUNT_ROOT/mqvae_extra_packages"

RUN_NAME=GEFION_LONG_ANYPLAQ_F_20260823
PREP_NAME="${RUN_NAME}_metadata"
TRAIN_SPLIT="$SPLIT_ROOT/3d_bodyscan_20260817_gefion_20260823_train_F.tsv"
TEST_SPLIT="$SPLIT_ROOT/3d_bodyscan_20260817_gefion_20260823_test_F.tsv"
METADATA="$TRANSFER_ROOT/body_scan_metadata20260817.tsv"

# Open3D in mqvae_extra_packages is a CPython 3.11 binary. This module stack
# deliberately uses the matching Python ABI and a CUDA-enabled PyTorch build.
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

echo "========== GEFION GPU CHECK =========="
nvidia-smi
"$PYTHON" -c "import sys, torch, numpy, pandas, scipy, sklearn, trimesh, open3d; assert sys.version_info[:2] == (3, 11), sys.version; assert torch.cuda.is_available(), 'PyTorch cannot see the allocated GPU'; print('[ENV] ALL REQUIRED IMPORTS WORK'); print('[ENV] Python', sys.version.split()[0]); print('[ENV] PyTorch', torch.__version__); print('[ENV] Open3D', open3d.__version__); print('[ENV] GPU', torch.cuda.get_device_name(0))"

echo "========== BUILD ENRICHED ANYPLAQ SPLITS =========="
"$PYTHON" run_cardiovascular_target_benchmark.py \
  --name_prefix "$PREP_NAME" \
  --dataroot "$DATA_ROOT" \
  --metadata_tsv "$METADATA" \
  --train_split_tsv "$TRAIN_SPLIT" \
  --test_split_tsv "$TEST_SPLIT" \
  --checkpoints_dir "$CHECKPOINT_ROOT" \
  --only_gender F \
  --targets anyplaq \
  --variants conv_pointlabels_scan_age \
  --audit_only

ENRICHED_ROOT="$CHECKPOINT_ROOT/$PREP_NAME/enriched_splits"

echo "========== TRAIN 2,000-EPOCH GPU MODEL =========="
# Open3D TriangleMesh objects are not pickleable, so DataLoader workers must
# stay in the training process. Model computation still runs on the GPU.
"$PYTHON" train_mesh_vae.py \
  --phase train \
  --name "$RUN_NAME" \
  --checkpoints_dir "$CHECKPOINT_ROOT" \
  --dataroot "$DATA_ROOT" \
  --cond_tsv "$ENRICHED_ROOT/train_enriched.tsv" \
  --feature_source scan \
  --only_gender F \
  --gpu_ids 0 \
  --seed 42 \
  --split_seed 42 \
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
  --use_cond \
  --encoder_arch meshconv \
  --encoder_cond_mode demo \
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

echo "========== HELD-OUT TEST OF BEST CHECKPOINT =========="
"$PYTHON" diagnose_mesh_binary.py \
  --name "$RUN_NAME" \
  --checkpoints_dir "$CHECKPOINT_ROOT" \
  --dataroot "$DATA_ROOT" \
  --cond_tsv "$ENRICHED_ROOT/test_enriched.tsv" \
  --feature_source scan \
  --only_gender F \
  --gpu_ids 0 \
  --which_epoch best \
  --batch_size 8 \
  --use_cond

echo "========== COMPLETE =========="
echo "Results: $CHECKPOINT_ROOT/$RUN_NAME/binary_summary.csv"
