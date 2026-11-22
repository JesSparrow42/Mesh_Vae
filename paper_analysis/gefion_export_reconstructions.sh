#!/bin/bash -l
#SBATCH --account=cu_0080
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --job-name=mqvae_reconstruct
#SBATCH --output=/dcai/projects/cu_0080/results/mqvae_reconstruct_%j.out
#SBATCH --error=/dcai/projects/cu_0080/results/mqvae_reconstruct_%j.err

set -euo pipefail

GENDER="${GENDER:-F}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-paper_v1}"
if [[ "$GENDER" != "F" && "$GENDER" != "M" ]]; then
    echo "GENDER must be F or M, got: $GENDER" >&2
    exit 2
fi

ACCOUNT_ROOT=/dcai/projects/cu_0080
CODE_ROOT="$ACCOUNT_ROOT/MQVAE"
DATA_ROOT="$ACCOUNT_ROOT/results/MQVAE_gefion_20260823"
CHECKPOINT_ROOT="$ACCOUNT_ROOT/results/checkpoints"
EXTRA_PACKAGES="$ACCOUNT_ROOT/mqvae_extra_packages"
RUN_NAME="GEFION_SCANONLY_ANYPLAQ_${GENDER}_${RUN_TAG}_seed${SEED}"
TEST_TSV="$CHECKPOINT_ROOT/${RUN_NAME}_metadata/enriched_splits/test_enriched.tsv"
OUTPUT_DIR="$CHECKPOINT_ROOT/$RUN_NAME/paper_reconstructions"

# Keep the working Gefion stack. Do not module purge.
module load GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1
module load trimesh/3.23.5

PYTHON="${MQVAE_PYTHON:-$(command -v python)}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$EXTRA_PACKAGES"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd "$CODE_ROOT"

for required in \
    "$CHECKPOINT_ROOT/$RUN_NAME/best_vae_net.pth" \
    "$CHECKPOINT_ROOT/$RUN_NAME/cond_stats.json" \
    "$DATA_ROOT/simplified_padded_tensors.npz" \
    "$TEST_TSV"
do
    if [[ ! -f "$required" ]]; then
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

echo "========== BUILD PAPER FIGURE FROM EXPORTED POINT CLOUDS =========="
"$PYTHON" paper_analysis/make_reconstruction_figure.py \
  --bundle "$OUTPUT_DIR/reconstruction_bundle.npz" \
  --manifest "$OUTPUT_DIR/reconstruction_manifest.csv" \
  --output-dir "$OUTPUT_DIR/figure"

echo "========== COMPLETE =========="
echo "Point clouds: $OUTPUT_DIR"
echo "Figure:       $OUTPUT_DIR/figure/reconstruction_examples.png"
echo "Caption:      $OUTPUT_DIR/figure/reconstruction_examples_caption.txt"
