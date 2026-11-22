#!/usr/bin/env bash

set -euo pipefail

GENDER="${GENDER:-F}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-paper_v1}"
GPU="${GPU:-0}"

if [[ "$GENDER" != "F" && "$GENDER" != "M" ]]; then
    echo "GENDER must be F or M, got: $GENDER" >&2
    exit 2
fi

PROJECT_ROOT=/data/projects/3D_scans
CODE_ROOT="$PROJECT_ROOT/ML/MQVAE"
DATA_ROOT="$PROJECT_ROOT/ML/MQVAE_combined_fast_20260808"
CHECKPOINT_ROOT="$CODE_ROOT/checkpoints"
PYTHON="${MQVAE_PYTHON:-/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python}"
RUN_NAME="COMPUTEROME_SCANONLY_ANYPLAQ_${GENDER}_${RUN_TAG}_seed${SEED}"
TEST_TSV="$CHECKPOINT_ROOT/${RUN_NAME}_metadata/enriched_splits/test_enriched.tsv"
OUTPUT_DIR="$CHECKPOINT_ROOT/$RUN_NAME/paper_reconstructions"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

cd "$CODE_ROOT"
mkdir -p logs

for required in \
    "$PYTHON" \
    "$CHECKPOINT_ROOT/$RUN_NAME/best_vae_net.pth" \
    "$CHECKPOINT_ROOT/$RUN_NAME/cond_stats.json" \
    "$DATA_ROOT/simplified_padded_tensors.npz" \
    "$TEST_TSV" \
    "$CODE_ROOT/paper_analysis/export_model_reconstructions.py" \
    "$CODE_ROOT/paper_analysis/make_reconstruction_figure.py"
do
    if [[ ! -e "$required" ]]; then
        echo "Missing required input: $required" >&2
        exit 3
    fi
done

echo "========== EXPORT ACTUAL HELD-OUT RECONSTRUCTIONS =========="
"$PYTHON" paper_analysis/export_model_reconstructions.py \
  --name "$RUN_NAME" \
  --dataroot "$DATA_ROOT" \
  --checkpoints-dir "$CHECKPOINT_ROOT" \
  --test-tsv "$TEST_TSV" \
  --gender "$GENDER" \
  --which-epoch best \
  --gpu-ids 0 \
  --examples-per-class 1 \
  --output-dir "$OUTPUT_DIR"

echo "========== BUILD PAPER FIGURE =========="
"$PYTHON" paper_analysis/make_reconstruction_figure.py \
  --bundle "$OUTPUT_DIR/reconstruction_bundle.npz" \
  --manifest "$OUTPUT_DIR/reconstruction_manifest.csv" \
  --output-dir "$OUTPUT_DIR/figure"

echo "========== COMPLETE =========="
echo "Point clouds: $OUTPUT_DIR"
echo "Figure:       $OUTPUT_DIR/figure/reconstruction_examples.png"
echo "Caption:      $OUTPUT_DIR/figure/reconstruction_examples_caption.txt"

