# Clinical feature sanity-check patch

This patch keeps the old command entry points and adds one switch:

```bash
--feature_source scan        # default, old behavior
--feature_source ldl         # use ldl_cholesterol instead of body scan
--feature_source hbalc       # use glycohemoglobin_hba1c instead of body scan
--feature_source ldl_hbalc   # use both numeric features instead of body scan
```

Clinical feature mode encodes the selected numeric TSV feature(s) as a tiny
synthetic point cloud, so the existing MeshVAE training and diagnostic scripts
still run with the same code path. LDL and HBALC are inputs/features in this
mode; the diagnostic target remains the atherosclerosis label from the cohort
TSV. It is intended as a sanity check, not as the final body-scan model.

## Make train/test TSVs

```bash
cd /data/projects/3D_scans/ML/MQVAE
python make_train_test_split.py \
  --input_tsv /data/projects/3D_scans/3d_bodyscan_20260604.tsv \
  --out_dir /data/projects/3D_scans/datasets \
  --prefix 3d_bodyscan_20260604 \
  --age_bins 18 30 40 50 60 71
```

This writes:

- `/data/projects/3D_scans/datasets/3d_bodyscan_20260604_train.tsv`
- `/data/projects/3D_scans/datasets/3d_bodyscan_20260604_test.tsv`
- `/data/projects/3D_scans/datasets/3d_bodyscan_20260604_train_F.tsv`
- `/data/projects/3D_scans/datasets/3d_bodyscan_20260604_test_F.tsv`
- equivalent `_M.tsv` files

If the TSV lacks `ho_carotis ve_carotis ho_fem ve_fem`, they are derived from
corrected plaque volume columns as `volume > 0`.

## Train using LDL only

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python train_mesh_vae.py \
  --name REACT_F_18_70_ldl_sanity \
  --cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_train_F.tsv \
  --feature_source ldl \
  --only_gender F \
  --age_bins 18 30 40 50 60 71 \
  --niter 150
```

## Train using HBALC only

Change only the feature source and name:

```bash
  --name REACT_F_18_70_hbalc_sanity \
  --feature_source hbalc \
```

## Diagnose the LDL sanity model

Use the same `--name` and the same `--feature_source` as training:

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python diagnose_mesh_vae.py \
  --name REACT_F_18_70_ldl_sanity \
  --feature_source ldl \
  --only_gender F \
  --age_bins 18 30 40 50 60 71 \
  --calib_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_train_F.tsv \
  --test_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_test_F.tsv
```
