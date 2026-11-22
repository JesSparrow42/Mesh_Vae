#!/usr/bin/env bash

set -euo pipefail

# Run directly on a Computerome DPREACT GPU host. Example:
#   nohup env GENDER=F SEED=42 RUN_TAG=paper_v1 GPU=0 \
#     bash paper_analysis/computerome_train_scan_only_gender.sh \
#     > logs/computerome_scan_F_paper_v1_seed42.log 2>&1 &

GENDER="${GENDER:-F}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-paper_v1}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ "$GENDER" != "F" && "$GENDER" != "M" ]]; then
    echo "GENDER must be F or M, got: $GENDER" >&2
    exit 2
fi

PROJECT_ROOT=/data/projects/3D_scans
CODE_ROOT="$PROJECT_ROOT/ML/MQVAE"
DATA_ROOT="$PROJECT_ROOT/ML/MQVAE_combined_fast_20260808"
SPLIT_ROOT="$PROJECT_ROOT/datasets"
CHECKPOINT_ROOT="$CODE_ROOT/checkpoints"
PYTHON="${MQVAE_PYTHON:-/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python}"

RUN_NAME="COMPUTEROME_SCANONLY_ANYPLAQ_${GENDER}_${RUN_TAG}_seed${SEED}"
PREP_NAME="${RUN_NAME}_metadata"
TRAIN_SPLIT="$SPLIT_ROOT/3d_bodyscan_20260604_scanfix_20260811_train_${GENDER}.tsv"
TEST_SPLIT="$SPLIT_ROOT/3d_bodyscan_20260604_scanfix_20260811_test_${GENDER}.tsv"
METADATA="$PROJECT_ROOT/3d_bodyscan_20260604.tsv"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cd "$CODE_ROOT"
mkdir -p "$CHECKPOINT_ROOT" logs

for required in \
    "$PYTHON" \
    "$DATA_ROOT/simplified_padded_tensors.npz" \
    "$DATA_ROOT/scan_geometry_features.npz" \
    "$TRAIN_SPLIT" \
    "$TEST_SPLIT" \
    "$METADATA" \
    "$CODE_ROOT/run_cardiovascular_target_benchmark.py" \
    "$CODE_ROOT/train_mesh_vae.py" \
    "$CODE_ROOT/diagnose_mesh_binary.py" \
    "$CODE_ROOT/paper_analysis/bootstrap_metrics.py" \
    "$CODE_ROOT/paper_analysis/extract_mesh_latents.py"
do
    if [[ ! -e "$required" ]]; then
        echo "Missing required input: $required" >&2
        exit 3
    fi
done

echo "========== ENVIRONMENT =========="
"$PYTHON" -c "import sys, torch, numpy, pandas, scipy, sklearn, trimesh, open3d, matplotlib; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'; print('[ENV] ALL REQUIRED IMPORTS WORK'); print('[ENV] Python', sys.version.split()[0]); print('[ENV] PyTorch', torch.__version__); print('[ENV] Open3D', open3d.__version__); print('[ENV] GPU', torch.cuda.get_device_name(0))"

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
# num_threads stays at zero because Open3D TriangleMesh objects cannot be
# pickled safely by PyTorch DataLoader workers.
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
  --batch_size "$BATCH_SIZE" \
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
  --batch_size "$BATCH_SIZE" \
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
  --batch-size "$BATCH_SIZE" \
  --output-dir "$CHECKPOINT_ROOT/$RUN_NAME/latents" \
  --use_preprocessed \
  --use_pointcloud \
  --no-use_cond

echo "========== COMPLETE =========="
echo "Run:         $RUN_NAME"
echo "Summary:     $CHECKPOINT_ROOT/$RUN_NAME/binary_summary.csv"
echo "Confidence:  $CHECKPOINT_ROOT/$RUN_NAME/paper_ci/bootstrap_metric_intervals.csv"
echo "Latents:     $CHECKPOINT_ROOT/$RUN_NAME/latents"

