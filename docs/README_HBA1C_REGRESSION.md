# HbA1c Regression

Run the commands in this guide from the repository root.

`run_all_15_hba1c_models.py` repeats the 15-row experiment design with
`glycohemoglobin_hba1c` as a continuous outcome. It does not turn HbA1c into an
arbitrary binary class. Results therefore use R-squared, MAE, RMSE, Pearson
correlation, and Spearman correlation instead of accuracy, sensitivity, and
specificity.

The six scan-based VAE rows still call the full, long-running
`train_mesh_vae.py` pipeline. Their latent disease-classification head is
disabled and replaced by a one-output regression head. The former logistic
classification comparison is implemented as linear regression, which is the
corresponding baseline for a continuous target.

Target normalization is fitted from the training TSV only and saved in each
checkpoint's `cond_stats.json`. Rows with missing HbA1c are dropped separately
from the training and held-out test datasets. The evaluator reports a
training-mean baseline alongside the model metrics.

## Female 15-Model Run

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python run_all_15_hba1c_models.py \
  --name_prefix HBA1C_FULL15_F_scanfix_20260813 \
  --dataroot /data/projects/3D_scans/ML/MQVAE_combined_fast_20260808 \
  --train_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_train_F.tsv \
  --test_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_test_F.tsv \
  --target_col glycohemoglobin_hba1c \
  --only_gender F \
  --gpu_ids 0 \
  --niter 150 \
  --batch_size 1 \
  --baseline_epochs 100 \
  --baseline_n_points 4096 \
  --baseline_batch_size 8
```

Resume is enabled by default. Re-running the same command skips completed
components and checkpoints, while finishing missing diagnoses. The final table
is written to:

```text
/data/projects/3D_scans/ML/MQVAE/checkpoints/HBA1C_FULL15_F_scanfix_20260813/hba1c_model_comparison_results.csv
```
