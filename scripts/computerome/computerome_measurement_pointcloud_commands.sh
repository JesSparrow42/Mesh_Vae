#!/usr/bin/env bash
(
set -euo pipefail

cd /data/projects/3D_scans/ML/MQVAE_new
PY=/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

"$PY" run_body_measurements_pointcloud.py build-cache \
  --baseline-run measurement_runs/n2894_baseline_01 \
  --obj-root /data/preprocessed/3D_scans \
  --points 2048 --workers 2 --seed 42 \
  --out-dir measurement_runs/n2894_point_cache_2048

"$PY" run_body_measurements_pointcloud.py train \
  --baseline-run measurement_runs/n2894_baseline_01 \
  --cache measurement_runs/n2894_point_cache_2048 \
  --device auto --batch-size 8 --threads 2 \
  --epochs 150 --patience 25 --bootstrap 2000 --seed 42 \
  --out-dir measurement_runs/n2894_pointcloud_01
)
