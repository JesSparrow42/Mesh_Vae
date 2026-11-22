# Step-by-step Computerome workflow

Computerome uses the existing DPREACT Python environment directly. Do not load
Gefion modules and do not use `sbatch` for this workflow.

## 1. Verify the environment and files

```bash
cd /data/projects/3D_scans/ML/MQVAE

PY=/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python

"$PY" -c "import torch, open3d, trimesh, numpy, pandas, scipy, sklearn, matplotlib; print('ALL IMPORTS WORK'); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

ls -lh /data/projects/3D_scans/ML/MQVAE_combined_fast_20260808/simplified_padded_tensors.npz
ls -lh /data/projects/3D_scans/ML/MQVAE_combined_fast_20260808/scan_geometry_features.npz
ls -lh /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_train_F.tsv
ls -lh /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_test_F.tsv
ls -lh paper_analysis/computerome_train_scan_only_gender.sh
```

Every command must succeed before training.

## 2. Start the female model

```bash
cd /data/projects/3D_scans/ML/MQVAE
mkdir -p logs

nohup env GENDER=F SEED=42 RUN_TAG=paper_v1 GPU=0 BATCH_SIZE=8 \
  bash paper_analysis/computerome_train_scan_only_gender.sh \
  > logs/computerome_scan_F_paper_v1_seed42.log 2>&1 &

echo $! | tee logs/computerome_scan_F_paper_v1_seed42.pid
```

The printed number is the Linux process ID, not a Slurm job number.

## 3. Monitor training

```bash
tail -f /data/projects/3D_scans/ML/MQVAE/logs/computerome_scan_F_paper_v1_seed42.log
```

Press `Ctrl+C` to stop following the log. Training continues in the background.

Check whether the process still exists:

```bash
PID=$(cat /data/projects/3D_scans/ML/MQVAE/logs/computerome_scan_F_paper_v1_seed42.pid)
ps -p "$PID" -o pid,etime,%cpu,%mem,cmd
```

Check GPU activity:

```bash
nvidia-smi
```

## 4. Verify completion

```bash
F_RUN=/data/projects/3D_scans/ML/MQVAE/checkpoints/COMPUTEROME_SCANONLY_ANYPLAQ_F_paper_v1_seed42

tail -n 80 /data/projects/3D_scans/ML/MQVAE/logs/computerome_scan_F_paper_v1_seed42.log
ls -lh "$F_RUN/best_vae_net.pth"
cat "$F_RUN/binary_summary.csv"
cat "$F_RUN/paper_ci/bootstrap_metric_intervals.csv"
ls -lh "$F_RUN/latents/train_latent_embeddings.csv"
ls -lh "$F_RUN/latents/test_latent_embeddings.csv"
```

The log must end with `========== COMPLETE ==========`. A checkpoint alone does
not prove that held-out evaluation and confidence intervals completed.

## 5. Start the male model after the female model finishes

```bash
nohup env GENDER=M SEED=42 RUN_TAG=paper_v1 GPU=0 BATCH_SIZE=8 \
  bash paper_analysis/computerome_train_scan_only_gender.sh \
  > logs/computerome_scan_M_paper_v1_seed42.log 2>&1 &

echo $! | tee logs/computerome_scan_M_paper_v1_seed42.pid
```

Do not start both models on GPU 0 simultaneously.

## 6. Export genuine held-out reconstructions

After the corresponding model completes:

```bash
nohup env GENDER=F SEED=42 RUN_TAG=paper_v1 GPU=0 \
  bash paper_analysis/computerome_export_reconstructions.sh \
  > logs/computerome_reconstruction_F_paper_v1_seed42.log 2>&1 &

echo $! | tee logs/computerome_reconstruction_F_paper_v1_seed42.pid
tail -f logs/computerome_reconstruction_F_paper_v1_seed42.log
```

The polished reconstruction figure is written to:

```text
/data/projects/3D_scans/ML/MQVAE/checkpoints/COMPUTEROME_SCANONLY_ANYPLAQ_F_paper_v1_seed42/paper_reconstructions/figure/reconstruction_examples.png
```
