# 15-Model Body-Scan Comparison

Run the commands in this guide from the repository root.

`run_all_15_models.py` is the main experiment runner. It uses the same body-scan
NPZ and metadata splits for every trainable row, resumes completed work, and
writes one partial/final comparison table.

The six scan-based VAE experiments are full long-running models trained through
`train_mesh_vae.py` and evaluated through `diagnose_mesh_vae.py`:

- Ordinary VAE: permutation-invariant point encoder, with and without age.
- Convolution VAE: local kNN edge convolution, with and without age.
- Point-label VAE: six pseudo anatomical regions with region-wise latent
  pooling, with and without age.

The remaining trainable experiments are two age-only VAEs, three logistic
regression baselines, and three convolutional classifier baselines. The requested
age-only point-label row remains in the 15-row table as `not_applicable`, because
there are no body points to label when the point cloud is excluded.

The diagnosis stage always loads `best_vae_net.pth`, selected by validation AUC
(with balanced accuracy as a tie-break contribution), and thresholds the direct
latent disease-classifier probability on the held-out internal validation subset.

## Full Female Body-Scan Run

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python run_all_15_models.py \
  --name_prefix FULL15_F_combined_fast_20260808_v2_20260809 \
  --dataroot /data/projects/3D_scans/ML/MQVAE_combined_fast_20260808 \
  --train_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_combined_fast_20260808_train_F.tsv \
  --test_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_combined_fast_20260808_test_F.tsv \
  --only_gender F \
  --gpu_ids 0 \
  --niter 150 \
  --batch_size 1 \
  --baseline_epochs 100 \
  --baseline_n_points 4096 \
  --baseline_batch_size 8
```

Run the exact same command after an interruption. Resume mode is enabled by
default: completed comparison rows and diagnosed full VAEs are skipped, while a
trained VAE whose diagnosis is missing is diagnosed without retraining.

The continuously updated outputs are:

```text
/data/projects/3D_scans/ML/MQVAE/checkpoints/FULL15_F_combined_fast_20260808_v2_20260809/model_comparison_results_partial.csv
/data/projects/3D_scans/ML/MQVAE/checkpoints/FULL15_F_combined_fast_20260808_v2_20260809/model_comparison_results_partial.md
```

Final outputs are:

```text
/data/projects/3D_scans/ML/MQVAE/checkpoints/FULL15_F_combined_fast_20260808_v2_20260809/model_comparison_results.csv
/data/projects/3D_scans/ML/MQVAE/checkpoints/FULL15_F_combined_fast_20260808_v2_20260809/model_comparison_results.md
```

The point-label runs print a six-region count vector and a label fingerprint.
The model loader also prints the checkpoint name, path, and SHA-256 prefix. These
messages make it possible to verify that the labeled and ordinary variants use
different inputs and checkpoints.
