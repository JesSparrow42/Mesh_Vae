# Step-by-step Gefion workflow

This guide is intentionally linear. Complete one checkpoint before moving to the
next. All commands are run on Gefion from:

```bash
cd /dcai/projects/cu_0080/MQVAE
```

Do not run `module purge`. The job scripts load the exact module combination that
already passed the project import test.

## What is actually being trained?

There are two layers of analysis:

1. **Deep scan model:** one female and one male MeshVAE are trained on full-body
   point clouds. The encoder receives scans only. A supervised auxiliary head
   learns the `benchmark_anyplaq` target from the latent representation.
2. **Fair modality comparison:** after the deep models finish, one fixed
   logistic-regression pipeline is trained repeatedly using age, BMI, scan latent,
   and their combinations. Age-only and BMI-only therefore do not require a new
   MeshVAE. Only the feature columns change; participants, split, classifier, and
   tuning procedure remain fixed.

The confirmatory result is the paired held-out AUC difference:

```text
age + BMI + scan  minus  age + BMI
```

reported separately in women and men.

## Checkpoint 1: confirm that Gefion has the new files

```bash
ls -l paper_analysis/gefion_train_scan_only_gender.sh
ls -l paper_analysis/gefion_export_reconstructions.sh
ls -l paper_analysis/export_model_reconstructions.py
ls -l paper_analysis/make_reconstruction_figure.py
```

All four commands must print a file. If one says `No such file or directory`,
upload/synchronize the current `paper_analysis` directory before continuing.

## Checkpoint 2: submit the female training job

```bash
sbatch --export=ALL,GENDER=F,SEED=42,RUN_TAG=paper_v1 \
  paper_analysis/gefion_train_scan_only_gender.sh
```

Gefion prints something like:

```text
Submitted batch job 123456
```

Write down that number. In the examples below it is called `JOBID`.

The job requests one GPU, 128 GB RAM, eight CPUs, and at most three days. It
trains for 2,000 epochs: 500 at the initial schedule plus 1,500 decay epochs.
`best_vae_net.pth` is selected using internal-validation AUC, not test AUC.

## Checkpoint 3: watch the job

Replace `JOBID` with the real number:

```bash
squeue -j JOBID
```

Follow normal output:

```bash
tail -f /dcai/projects/cu_0080/results/mqvae_scan_gender_JOBID.out
```

Press `Ctrl+C` to stop following the file. This does not stop the job.

Look at errors in a second terminal:

```bash
tail -n 80 /dcai/projects/cu_0080/results/mqvae_scan_gender_JOBID.err
```

Normal output should eventually contain:

```text
[ENV] ALL REQUIRED IMPORTS WORK
========== TRAIN SCAN-ONLY ENCODER ==========
--- Epoch 1 ---
[epoch 1] batch ... loss = ...
```

An empty error file is good. Messages about modules being reloaded are not model
errors. A Python traceback, `OOM`, or a non-zero exit code is an error.

When the job disappears from `squeue`, check its final state:

```bash
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Continue only when the main row says `COMPLETED` and `ExitCode` is `0:0`.

## Checkpoint 4: confirm the female outputs

```bash
F_RUN=/dcai/projects/cu_0080/results/checkpoints/GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42

ls -lh "$F_RUN/best_vae_net.pth"
cat "$F_RUN/binary_summary.csv"
cat "$F_RUN/paper_ci/bootstrap_metric_intervals.csv"
ls -lh "$F_RUN/latents/train_latent_embeddings.csv"
ls -lh "$F_RUN/latents/test_latent_embeddings.csv"
```

The five checks establish that training, held-out evaluation, bootstrap CIs, and
latent export all completed. Do not infer success merely because a checkpoint
file exists.

## Checkpoint 5: submit and verify the male job

```bash
sbatch --export=ALL,GENDER=M,SEED=42,RUN_TAG=paper_v1 \
  paper_analysis/gefion_train_scan_only_gender.sh
```

Monitor it exactly as above. After completion:

```bash
M_RUN=/dcai/projects/cu_0080/results/checkpoints/GEFION_SCANONLY_ANYPLAQ_M_paper_v1_seed42

ls -lh "$M_RUN/best_vae_net.pth"
cat "$M_RUN/binary_summary.csv"
cat "$M_RUN/paper_ci/bootstrap_metric_intervals.csv"
ls -lh "$M_RUN/latents/train_latent_embeddings.csv"
ls -lh "$M_RUN/latents/test_latent_embeddings.csv"
```

## Checkpoint 6: train the age/BMI/scan comparison models

These are regularized logistic models, not new VAEs. Set the paths:

```bash
F_RUN=/dcai/projects/cu_0080/results/checkpoints/GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42
M_RUN=/dcai/projects/cu_0080/results/checkpoints/GEFION_SCANONLY_ANYPLAQ_M_paper_v1_seed42
F_META=/dcai/projects/cu_0080/results/checkpoints/GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42_metadata/enriched_splits
M_META=/dcai/projects/cu_0080/results/checkpoints/GEFION_SCANONLY_ANYPLAQ_M_paper_v1_seed42_metadata/enriched_splits
OUT=/dcai/projects/cu_0080/results/paper_analysis/modality_primary
```

Run all seven controlled comparisons:

```bash
python paper_analysis/run_modality_benchmark.py \
  --train-latent "$F_RUN/latents/train_latent_embeddings.csv" "$M_RUN/latents/train_latent_embeddings.csv" \
  --test-latent "$F_RUN/latents/test_latent_embeddings.csv" "$M_RUN/latents/test_latent_embeddings.csv" \
  --train-tsv "$F_META/train_enriched.tsv" "$M_META/train_enriched.tsv" \
  --test-tsv "$F_META/test_enriched.tsv" "$M_META/test_enriched.tsv" \
  --target-col benchmark_anyplaq \
  --genders F M \
  --reference age_bmi \
  --bootstraps 5000 \
  --output-dir "$OUT"
```

The key paper result is:

```bash
cat "$OUT/primary_age_bmi_scan_vs_age_bmi.csv"
```

The complete estimates and confidence intervals are:

```bash
cat "$OUT/modality_metric_intervals.csv"
```

The figure with error bars is:

```text
/dcai/projects/cu_0080/results/paper_analysis/modality_primary/modality_auc_by_sex.pdf
```

## Checkpoint 7: export real model reconstructions

This is a separate short GPU job. It does not use the old poster figure.

For the female model:

```bash
sbatch --export=ALL,GENDER=F,SEED=42,RUN_TAG=paper_v1 \
  paper_analysis/gefion_export_reconstructions.sh
```

Monitor the new job number with `squeue`, `tail`, and `sacct` as above. Its output
directory is:

```text
/dcai/projects/cu_0080/results/checkpoints/GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42/paper_reconstructions
```

It creates:

- `reconstruction_bundle.npz`: actual input and reconstructed point arrays;
- `reconstruction_manifest.csv`: target, probability, Chamfer error, and selection rule;
- `example_00_input.ply` and `example_00_reconstruction.ply`;
- corresponding files for the second target group;
- `figure/reconstruction_examples.png`, `.tiff`, and `.pdf`;
- `figure/reconstruction_examples_caption.txt`.

By default, the selected plaque-free and plaque-positive examples are the held-out
participants closest to their class's median Chamfer reconstruction error. This is
more defensible than choosing whichever bodies look most attractive.

To export specific held-out IDs instead, run the Python exporter directly and
repeat `--participant-id`:

```bash
python paper_analysis/export_model_reconstructions.py \
  --name GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42 \
  --dataroot /dcai/projects/cu_0080/results/MQVAE_gefion_20260823 \
  --checkpoints-dir /dcai/projects/cu_0080/results/checkpoints \
  --test-tsv "$F_META/test_enriched.tsv" \
  --gender F \
  --which-epoch best \
  --gpu-ids 0 \
  --participant-id PARTICIPANT_ONE \
  --participant-id PARTICIPANT_TWO \
  --output-dir "$F_RUN/paper_reconstructions_selected"
```

## Checkpoint 8: repolish an existing reconstruction bundle

This stage does not reload the model. It reads the exported point arrays:

```bash
RECON="$F_RUN/paper_reconstructions"

python paper_analysis/make_reconstruction_figure.py \
  --bundle "$RECON/reconstruction_bundle.npz" \
  --manifest "$RECON/reconstruction_manifest.csv" \
  --output-dir "$RECON/figure"
```

The renderer automatically infers body height, frontal width, and depth axes. If
the body appears rotated, explicitly choose axes, for example:

```bash
python paper_analysis/make_reconstruction_figure.py \
  --bundle "$RECON/reconstruction_bundle.npz" \
  --manifest "$RECON/reconstruction_manifest.csv" \
  --vertical-axis z \
  --front-axis x \
  --output-dir "$RECON/figure_zx"
```

Try `x`, `y`, or `z`; vertical and front axes must be different.

## Checkpoint 9: analyze latent signal and t-SNE

Run women first:

```bash
python paper_analysis/analyze_latent_signal.py \
  --train-latent "$F_RUN/latents/train_latent_embeddings.csv" \
  --test-latent "$F_RUN/latents/test_latent_embeddings.csv" \
  --train-tsv "$F_META/train_enriched.tsv" \
  --test-tsv "$F_META/test_enriched.tsv" \
  --target-col benchmark_anyplaq \
  --genders F \
  --permutations 999 \
  --bootstraps 5000 \
  --output-dir /dcai/projects/cu_0080/results/paper_analysis/latent_F
```

Repeat with the male paths and `--genders M`. Interpret the JSON statistics from
the original latent vectors. Treat the t-SNE PDF as visualization only.

## Checkpoint 10: files to bring back for the paper

Copy/download these result directories from Gefion:

```text
/dcai/projects/cu_0080/results/paper_analysis/modality_primary
/dcai/projects/cu_0080/results/paper_analysis/latent_F
/dcai/projects/cu_0080/results/paper_analysis/latent_M
/dcai/projects/cu_0080/results/checkpoints/GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42/paper_reconstructions
/dcai/projects/cu_0080/results/checkpoints/GEFION_SCANONLY_ANYPLAQ_M_paper_v1_seed42/paper_reconstructions
```

At that point the manuscript placeholders can be replaced with final estimates,
paired confidence intervals, cohort counts, latent tests, and genuine checkpoint
reconstruction figures.
