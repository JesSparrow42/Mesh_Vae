# Supervised Diagnostic Model

Run the commands in this guide from the repository root.

This is the fresh direct classifier path. It predicts the binary
atherosclerosis target directly from either body-scan point clouds or
LDL/HBA1C sanity-check features.

It saves and loads:

```text
checkpoints/<name>/supervised_latest.pt
```

Diagnosis uses `--name` to find that checkpoint.

## Body Scans

Train:

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python train_diagnostic_model.py \
  --name SUP_F_scan_20260618 \
  --dataroot /data/projects/3D_scans/ML/MQVAE \
  --cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_train_F.tsv \
  --feature_source scan \
  --only_gender F \
  --batch_size 16 \
  --n_points 4096 \
  --epochs 100
```

Evaluate:

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python diagnose_diagnostic_model.py \
  --name SUP_F_scan_20260618 \
  --dataroot /data/projects/3D_scans/ML/MQVAE \
  --test_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_test_F.tsv \
  --feature_source scan \
  --only_gender F \
  --batch_size 16 \
  --n_points 4096
```

If the GPU runs out of memory, lower `--batch_size` to `8` or `4`.

## LDL

Train:

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python train_diagnostic_model.py \
  --name SUP_F_ldl_20260618 \
  --cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_train_F.tsv \
  --feature_source ldl \
  --only_gender F \
  --batch_size 128 \
  --epochs 100
```

Evaluate:

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python diagnose_diagnostic_model.py \
  --name SUP_F_ldl_20260618 \
  --test_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_test_F.tsv \
  --feature_source ldl \
  --only_gender F \
  --batch_size 128
```

## HBA1C

Train:

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python train_diagnostic_model.py \
  --name SUP_F_hba1c_20260618 \
  --cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_train_F.tsv \
  --feature_source hba1c \
  --only_gender F \
  --batch_size 128 \
  --epochs 100
```

Evaluate:

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python diagnose_diagnostic_model.py \
  --name SUP_F_hba1c_20260618 \
  --test_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_test_F.tsv \
  --feature_source hba1c \
  --only_gender F \
  --batch_size 128
```

## LDL + HBA1C

Train:

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python train_diagnostic_model.py \
  --name SUP_F_ldl_hba1c_20260618 \
  --cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_train_F.tsv \
  --feature_source ldl_hba1c \
  --only_gender F \
  --batch_size 128 \
  --epochs 100
```

Evaluate:

```bash
cd /data/projects/3D_scans/ML/MQVAE && CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python diagnose_diagnostic_model.py \
  --name SUP_F_ldl_hba1c_20260618 \
  --test_cond_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_test_F.tsv \
  --feature_source ldl_hba1c \
  --only_gender F \
  --batch_size 128
```
