# Combined Scan Data

The preprocessor can build one combined `simplified_padded_tensors.npz` from:

- the old preprocessed NPZ, and
- one or more new OBJ folders searched recursively.

By default, duplicate cleaned scan IDs such as `RIGDET...` or `GOEDET...` are skipped after the first occurrence. When using `--preprocess_existing_npz`, old samples are added first, so duplicates in the new folder are skipped.

## Recommended Combined Preprocess

Use a new output dataroot so the old NPZ is not overwritten:

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python train_mesh_vae.py \
  --phase preprocess \
  --name PREPROCESS_combined_20260715 \
  --dataroot /data/projects/3D_scans/ML/MQVAE_combined \
  --preprocess_existing_npz /data/projects/3D_scans/ML/MQVAE/simplified_padded_tensors.npz \
  --preprocess_roots /data/projects/3D_scans/3d_bodyscans \
  --target_tris 20000 \
  --n_points 25000
```

The new root `/data/projects/3D_scans/3d_bodyscans` is searched recursively, so all subfolders under it are included automatically.

## Fast Pack From Already Simplified OBJs

If `/data/projects/3D_scans/ML/MQVAE_combined/simplified` already contains `*_simplified.obj` files, do not run the full preprocessing again. Pack them into the NPZ directly:

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python train_mesh_vae.py \
  --phase preprocess \
  --name PACK_combined_simplified_20260808 \
  --dataroot /data/projects/3D_scans/ML/MQVAE_combined \
  --preprocess_roots /data/projects/3D_scans/ML/MQVAE_combined/simplified \
  --preprocess_assume_simplified \
  --point_sample_method uniform \
  --n_points 25000
```

This skips normalization, decimation, and rewriting simplified OBJ files. It only builds `simplified_padded_tensors.npz` and samples the point clouds.

During preprocessing, the script prints progress like:

```text
[PREPROC] starting 125/2000 (  6.25%) elapsed=12m10s eta=3h02m
[PREPROC] completed 126/2000 (  6.30%) elapsed=12m42s eta=3h03m
```

## Training On The Combined NPZ

After preprocessing, use the combined dataroot in training and diagnosis:

```bash
--dataroot /data/projects/3D_scans/ML/MQVAE_combined
```

Training still requires TSV metadata and labels. Any scan whose cleaned ID is not present in `--cond_tsv` is dropped, so use train/test TSV files that include the new scan IDs.
