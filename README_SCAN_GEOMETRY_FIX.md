# Physical-scale scan fix

## What was wrong

The old preprocessing centered every participant and divided that participant
by their own maximum radius.  This erased absolute height, width, depth, and
body-volume information before training.  Missing plaque measurements were
also converted to healthy labels.

The new path:

1. reads only vertex records from the original OBJ files;
2. recovers physical size and shape descriptors without re-simplifying meshes;
3. restores between-participant point-cloud scale using one training-set scale;
4. saves training normalization with each checkpoint;
5. drops unknown disease outcomes instead of calling them healthy.

## 1. Build the resumable geometry sidecar

```bash
cd /data/projects/3D_scans/ML/MQVAE && \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python build_scan_geometry_features.py \
  --npz /data/projects/3D_scans/ML/MQVAE_combined_fast_20260808/simplified_padded_tensors.npz \
  --obj_roots /data/projects/3D_scans/3d_bodyscans \
  --out /data/projects/3D_scans/ML/MQVAE_combined_fast_20260808/scan_geometry_features.npz \
  --workers 2 \
  --checkpoint_every 5
```

This is not mesh preprocessing.  It streams OBJ vertex lines and prints a
percentage and ETA.  It resumes from `scan_geometry_features.partial.csv`.

## 2. Rebuild target-safe train/test splits

```bash
cd /data/projects/3D_scans/ML/MQVAE && \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python -m meshvae.make_train_test_split \
  --input_tsv /data/projects/3D_scans/3d_bodyscan_20260604.tsv \
  --out_dir /data/projects/3D_scans/datasets \
  --prefix 3d_bodyscan_20260604_scanfix_20260811 \
  --match_npz /data/projects/3D_scans/ML/MQVAE_combined_fast_20260808/simplified_padded_tensors.npz \
  --id_col participant_id \
  --target_missing_policy drop \
  --test_size 0.20 \
  --seed 42
```

## 3. Run a short scan-only gate

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python run_model_comparison.py \
  --name SCAN_GATE_F_20260811 \
  --dataroot /data/projects/3D_scans/ML/MQVAE_combined_fast_20260808 \
  --train_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_train_F.tsv \
  --test_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_test_F.tsv \
  --out_dir /data/projects/3D_scans/ML/MQVAE/checkpoints/SCAN_GATE_F_20260811 \
  --only_gender F \
  --gpu_ids 0 \
  --epochs 100 \
  --n_points 4096 \
  --batch_size_point 8 \
  --use_scan_geometry_features \
  --require_scan_geometry_features \
  --restore_metric_scale \
  --target_missing_policy drop \
  --experiments logistic_scan_noage convolution_scan_noage logistic_scan_age convolution_scan_age
```

## 4. Screen all safe metadata columns

```bash
cd /data/projects/3D_scans/ML/MQVAE && \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python screen_metadata_predictors.py \
  --train_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_train_F.tsv \
  --test_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_test_F.tsv \
  --out_dir /data/projects/3D_scans/ML/MQVAE/checkpoints/METADATA_SCREEN_F_20260811 \
  --only_gender F \
  --age_col age \
  --target_missing_policy drop \
  --folds 5 \
  --seed 42
```

## 5. Run all 15 models

```bash
cd /data/projects/3D_scans/ML/MQVAE && \
nohup env CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python run_all_15_models.py \
  --name_prefix FULL15_F_scanfix_20260811 \
  --dataroot /data/projects/3D_scans/ML/MQVAE_combined_fast_20260808 \
  --train_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_train_F.tsv \
  --test_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_test_F.tsv \
  --only_gender F \
  --gpu_ids 0 \
  --niter 150 \
  --batch_size 1 \
  --baseline_epochs 100 \
  --baseline_n_points 4096 \
  --baseline_batch_size 8 \
  --use_scan_geometry_features \
  --require_scan_geometry_features \
  --restore_metric_scale \
  --target_missing_policy drop \
  > /data/projects/3D_scans/ML/MQVAE/FULL15_F_scanfix_20260811.log 2>&1 &
```

Monitor with:

```bash
tail -f /data/projects/3D_scans/ML/MQVAE/FULL15_F_scanfix_20260811.log
```
