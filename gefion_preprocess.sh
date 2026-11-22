#!/bin/bash -l
set -euo pipefail

ACCOUNT_ROOT=/dcai/projects/cu_0080
TRANSFER_ROOT="$ACCOUNT_ROOT/xfer/sftp/inbox/REACT-Q"
CODE_ROOT="$ACCOUNT_ROOT/MQVAE"
OUTPUT_ROOT="$ACCOUNT_ROOT/results/MQVAE_gefion_20260823"
DATASET_ROOT="$ACCOUNT_ROOT/datasets"
CHECKPOINT_ROOT="$OUTPUT_ROOT/checkpoints"
EXTRA_PACKAGES="$ACCOUNT_ROOT/mqvae_extra_packages"

# Keep the Python ABI aligned with the existing CPython 3.11 Open3D install.
module load GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1
module load trimesh/3.23.5

PYTHON="${MQVAE_PYTHON:-$(command -v python)}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$EXTRA_PACKAGES"

GODSTRUP_ROOT=$(find "$TRANSFER_ROOT" -mindepth 1 -maxdepth 1 -type d -iname '*dstrup' -print -quit)
if [[ -z "$GODSTRUP_ROOT" ]]; then
    echo "Could not find the Godstrup data folder below $TRANSFER_ROOT" >&2
    exit 2
fi
echo "[ENV] Python: $PYTHON"
"$PYTHON" -c "import sys, torch, numpy, pandas, scipy, sklearn, trimesh, open3d; assert sys.version_info[:2] == (3, 11), sys.version; print('[ENV] ALL REQUIRED IMPORTS WORK'); print('[ENV] Python', sys.version.split()[0]); print('[ENV] PyTorch', torch.__version__); print('[ENV] Open3D', open3d.__version__)"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPEN3D_CPU_RENDERING=true

mkdir -p "$OUTPUT_ROOT/sites" "$OUTPUT_ROOT/empty_input" "$DATASET_ROOT" "$CHECKPOINT_ROOT"
cd "$CODE_ROOT"

run_site() {
    local site_name=$1
    local obj_root=$2
    local site_output="$OUTPUT_ROOT/sites/$site_name"
    mkdir -p "$site_output"
    if [[ -f "$site_output/simplified_padded_tensors.npz" ]]; then
        echo "[SITE] skipping completed $site_name: $site_output/simplified_padded_tensors.npz"
        return 0
    fi
    echo "[SITE] starting $site_name from $obj_root at $(date -Iseconds)"
    "$PYTHON" train_mesh_vae.py \
      --phase preprocess \
      --name "PREPROCESS_GEFION_${site_name}_20260823" \
      --checkpoints_dir "$CHECKPOINT_ROOT" \
      --dataroot "$site_output" \
      --preprocess_roots "$obj_root" \
      --target_tris 20000 \
      --n_points 25000 \
      > "$site_output/preprocess.log" 2>&1
}

# Large raw meshes have high transient memory use. Run sites sequentially so
# their Open3D working sets never compete for the node's memory allocation.
run_site Glostrup "$TRANSFER_ROOT/Glostrup"
run_site Godstrup "$GODSTRUP_ROOT"
run_site Herlev "$TRANSFER_ROOT/Herlev"
run_site RH "$TRANSFER_ROOT/RH"

SITE_NPZS=(
  "$OUTPUT_ROOT/sites/Glostrup/simplified_padded_tensors.npz"
  "$OUTPUT_ROOT/sites/Godstrup/simplified_padded_tensors.npz"
  "$OUTPUT_ROOT/sites/Herlev/simplified_padded_tensors.npz"
  "$OUTPUT_ROOT/sites/RH/simplified_padded_tensors.npz"
)
for npz in "${SITE_NPZS[@]}"; do
    if [[ ! -f "$npz" ]]; then
        echo "Expected site NPZ was not created: $npz" >&2
        exit 4
    fi
done

echo "[MERGE] starting at $(date -Iseconds)"
"$PYTHON" train_mesh_vae.py \
  --phase preprocess \
  --name PREPROCESS_GEFION_ALL_20260823 \
  --checkpoints_dir "$CHECKPOINT_ROOT" \
  --dataroot "$OUTPUT_ROOT" \
  --preprocess_roots "$OUTPUT_ROOT/empty_input" \
  --preprocess_existing_npz "${SITE_NPZS[@]}" \
  --target_tris 20000 \
  --n_points 25000

echo "[GEOMETRY] starting at $(date -Iseconds)"
"$PYTHON" build_scan_geometry_features.py \
  --npz "$OUTPUT_ROOT/simplified_padded_tensors.npz" \
  --obj_roots \
    "$TRANSFER_ROOT/Glostrup" \
    "$GODSTRUP_ROOT" \
    "$TRANSFER_ROOT/Herlev" \
    "$TRANSFER_ROOT/RH" \
  --out "$OUTPUT_ROOT/scan_geometry_features.npz" \
  --workers 4 \
  --checkpoint_every 25

echo "[SPLIT] starting at $(date -Iseconds)"
"$PYTHON" -m meshvae.make_train_test_split \
  --input_tsv "$TRANSFER_ROOT/body_scan_metadata20260817.tsv" \
  --out_dir "$DATASET_ROOT" \
  --prefix 3d_bodyscan_20260817_gefion_20260823 \
  --match_npz "$OUTPUT_ROOT/simplified_padded_tensors.npz" \
  --id_col participant_id \
  --test_size 0.20 \
  --seed 42

echo "[DONE] finished at $(date -Iseconds)"
echo "[DONE] scan data: $OUTPUT_ROOT"
echo "[DONE] female train: $DATASET_ROOT/3d_bodyscan_20260817_gefion_20260823_train_F.tsv"
echo "[DONE] female test:  $DATASET_ROOT/3d_bodyscan_20260817_gefion_20260823_test_F.tsv"
