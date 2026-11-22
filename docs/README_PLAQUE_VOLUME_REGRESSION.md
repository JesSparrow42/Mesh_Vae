# Total plaque-volume regression

Run the commands in this guide from the repository root.

> **Historical experiment template:** the retained `scripts/benchmarks/run_full_meshvae_regression_variants.py` is a training-module fragment, not a complete command-line runner. The launch example below records the earlier experiment interface and is not runnable as written. Use the current [cardiovascular benchmark](README_CARDIOVASCULAR_TARGET_BENCHMARK.md) or [paper-analysis workflow](../paper_analysis/README.md) for available orchestration commands.

The continuous target is derived for each participant as:

```text
plaque_volume_total_corrected =
    plaque_volume_right_carotid_corrected
  + plaque_volume_left_carotid_corrected
  + plaque_volume_right_femoral_corrected
  + plaque_volume_left_femoral_corrected
```

All four measurements must be present. A measured zero is retained as zero;
a row with any missing component is excluded. Training predicts
`log1p(plaque_volume_total_corrected)`, then z-normalizes that value using only
the training data. Held-out diagnostics report both original-volume metrics
(`R2`, `MAE`, `RMSE`, Pearson and Spearman) and log-scale metrics
(`TransformedR2`, `TransformedMAE`).

## Historical female scan-and-age launch

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python -m scripts.benchmarks.run_full_meshvae_regression_variants \
  --name_prefix PLAQUEVOL_F_scanfix_20260815 \
  --dataroot /data/projects/3D_scans/ML/MQVAE_combined_fast_20260808 \
  --train_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_train_F.tsv \
  --test_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_test_F.tsv \
  --target_col plaque_volume_total_corrected \
  --target_transform log1p \
  --target_source_cols \
    plaque_volume_right_carotid_corrected \
    plaque_volume_left_carotid_corrected \
    plaque_volume_right_femoral_corrected \
    plaque_volume_left_femoral_corrected \
  --only_gender F \
  --gpu_ids 0 \
  --niter 150 \
  --batch_size 1 \
  --variants normal_scan_age
```

The earlier experiment used the name
`PLAQUEVOL_F_scanfix_20260815_vae_scan_age`, with training followed by diagnosis
of the best checkpoint on the held-out test TSV. Its concise result path was:

```text
/data/projects/3D_scans/ML/MQVAE/checkpoints/PLAQUEVOL_F_scanfix_20260815_vae_scan_age/regression_summary.csv
```

The run also writes per-participant predictions and a full `metrics.json`
under the experiment's `regression_diagnostics/latest_*` directory.
