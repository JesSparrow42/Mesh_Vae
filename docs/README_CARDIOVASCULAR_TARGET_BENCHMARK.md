# Cardiovascular target benchmark

Run the commands in this guide from the repository root.

`scripts/benchmarks/run_cardiovascular_target_benchmark.py` preserves the existing female
train/test participant split, enriches those rows from the newer master
metadata TSV, audits every target, and runs the same two architectures for
each usable target:

1. Convolutional MeshVAE with body scan and age.
2. The same convolutional MeshVAE with six-region point labels and
   region-aware pooling.

The benchmark contains eleven outcomes and therefore at most 22 training
runs. Continuous outcomes use a regression head. Binary outcomes use a
class-weighted binary head. AUC is the primary binary metric; threshold
metrics use 0.5.

## Outcomes

Continuous targets:

- right, left, and combined carotid plaque volume
- right, left, and combined femoral plaque volume
- coronary plaque volume (`pv_coro`)
- CCTA segment involvement score (`ccta_sis`)
- coronary calcium score (`cacs_coronary`)

Binary targets:

- any coronary artery disease (`ccta_anycad`)
- plaque at any measured site (`anyplaq`)

Plaque volumes and calcium score are modeled after `log1p` transformation,
but held-out R2, MAE, and RMSE are reported in the original units. The runner
uses the supplied `pv_caro` and `pv_femo` columns when present and fills only
missing values from the corresponding left-plus-right measurements. It writes
an audit showing whether supplied totals disagree with those sums.

## Full female run

```bash
cd /data/projects/3D_scans/ML/MQVAE && nohup env CUDA_VISIBLE_DEVICES=0 \
/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python -m scripts.benchmarks.run_cardiovascular_target_benchmark \
  --name_prefix CARDIO_TARGETS_F_20260817 \
  --dataroot /data/projects/3D_scans/ML/MQVAE_combined_fast_20260808 \
  --metadata_tsv /data/projects/3D_scans/3d_bodyscan_20260604.tsv \
  --train_split_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_train_F.tsv \
  --test_split_tsv /data/projects/3D_scans/datasets/3d_bodyscan_20260604_scanfix_20260811_test_F.tsv \
  --only_gender F \
  --gpu_ids 0 \
  --niter 150 \
  --batch_size 1 \
  > /data/projects/3D_scans/ML/MQVAE/CARDIO_TARGETS_F_20260817.log 2>&1 &
```

Follow progress:

```bash
tail -f /data/projects/3D_scans/ML/MQVAE/CARDIO_TARGETS_F_20260817.log
```

The command is resumable by default. Running it again skips completed
checkpoints and completed diagnosis files.

## Results

The continuously updated table is:

```text
/data/projects/3D_scans/ML/MQVAE/checkpoints/CARDIO_TARGETS_F_20260817/cardiovascular_target_results_partial.csv
```

After completion, the final files are:

```text
/data/projects/3D_scans/ML/MQVAE/checkpoints/CARDIO_TARGETS_F_20260817/cardiovascular_target_results.csv
/data/projects/3D_scans/ML/MQVAE/checkpoints/CARDIO_TARGETS_F_20260817/cardiovascular_target_results.md
```

Target counts and derivation checks are written to
`target_availability_audit.csv` and `target_derivation_audit.json` in the same
directory. Enriched copies of the unchanged participant split are kept under
`enriched_splits/`.

## Latent body-measurement R2

The final cell in [`notebooks/tsne_plaque_latent.ipynb`](../notebooks/tsne_plaque_latent.ipynb) fits Ridge regressions to the
original scan-only latent dimensions using training participants only. It
reports cross-validated training R2 and one held-out test R2 for BMI, waist,
hip, height, weight, and waist-to-hip ratio. It does not regress on the two
t-SNE display coordinates.
