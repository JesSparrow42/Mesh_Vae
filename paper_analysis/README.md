# REACT-Q paper analysis workflow

This directory turns the existing MeshVAE/Gefion outputs into one reproducible paper analysis. It does not replace or modify the working training code.

## The one-sentence goal

Determine whether scan-derived body morphology adds held-out information about any measured plaque beyond age and BMI, separately in women and men.

## Files

| File | Purpose |
|---|---|
| `bootstrap_metrics.py` | Participant-level confidence intervals for binary or regression prediction CSVs; paired model differences when IDs overlap. |
| `bootstrap_result_tree.py` | Recursively audits all held-out prediction files and produces one CI table plus an explicit failure inventory. |
| `collect_all_results.py` | Collects saved model results and CI artifacts from either server, with source paths and an inventory; merges the server exports. See `NEXT_STEPS.md`. |
| `results_overview.ipynb` | Reads both portable server exports, computes missing participant CIs, and saves performance, total-volume, Boson and t-SNE plots. |
| `gefion_train_scan_only_gender.sh` | Reproducible scan-only GPU run for one sex and seed, followed by diagnosis, CIs, and latent extraction. |
| `STEP_BY_STEP_GEFION.md` | Literal training, monitoring, reconstruction, and analysis instructions with checkpoints. |
| `extract_mesh_latents.py` | Exports deterministic posterior means from a trained checkpoint, or batches completed NEXT studies using their saved inputs. |
| `gefion_next_boson.sh` | Matched Gaussian-MMD/Boson-MMD array for the saved all-log total-volume study, both sexes and three training seeds. |
| `run_modality_benchmark.py` | Fixed-classifier age/BMI/scan comparison and primary paired delta AUC. |
| `build_cohort_table.py` | Unique-participant cohort flow and Table 1 summaries with leakage checks. |
| `plot_calibration.py` | Calibration curves, Brier score, slope, and intercept for the primary models. |
| `analyze_latent_signal.py` | Quantitative original-latent cluster and body-measurement analyses plus a t-SNE figure, optionally colored by measured plaque volume. |
| `export_model_reconstructions.py` | Loads a real checkpoint and exports representative held-out input/reconstruction point clouds. |
| `gefion_export_reconstructions.sh` | Short GPU job that exports real reconstructions and renders the paper figure. |
| `make_reconstruction_figure.py` | Builds a 300-dpi paper figure from the exported model point clouds, never from poster pixels. |
| `project_conclusions.md` | Main goal, current evidence, claim boundaries, and pre-specified interpretation rules. |
| `paper_draft.md` | Manuscript scaffold, current interpretation, and result placeholders. |

## 1. Run the two confirmatory GPU jobs on Gefion

The job script preserves the module stack that already cleared every required import. It deliberately does **not** run `module purge`.

From `/dcai/projects/cu_0080/MQVAE`:

```bash
sbatch --export=ALL,GENDER=F,SEED=42,RUN_TAG=paper_v1 \
  paper_analysis/gefion_train_scan_only_gender.sh

sbatch --export=ALL,GENDER=M,SEED=42,RUN_TAG=paper_v1 \
  paper_analysis/gefion_train_scan_only_gender.sh
```

Check them with:

```bash
squeue -u "$USER"
sacct -j JOB_NUMBER --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Each completed run writes:

- `binary_summary.csv`;
- participant-level `binary_diagnostics/latest_*/predictions.csv`;
- `paper_ci/bootstrap_metric_intervals.csv`;
- `latents/train_latent_embeddings.csv` and `latents/test_latent_embeddings.csv`.

The model is intentionally scan-only at the encoder: `--no-use_cond --encoder_cond_mode none`. The linear-probe step adds age and BMI in a controlled way later.

## 2. Run the clean age/BMI/scan comparison

Set the two run directories after both jobs finish:

```bash
F_RUN=/dcai/projects/cu_0080/results/checkpoints/GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42
M_RUN=/dcai/projects/cu_0080/results/checkpoints/GEFION_SCANONLY_ANYPLAQ_M_paper_v1_seed42
OUT=/dcai/projects/cu_0080/results/paper_analysis/modality_primary
```

Then run one analysis containing both sexes. Multiple paths after each input flag are combined, while the models remain sex-stratified:

```bash
python paper_analysis/run_modality_benchmark.py \
  --train-latent "$F_RUN/latents/train_latent_embeddings.csv" "$M_RUN/latents/train_latent_embeddings.csv" \
  --test-latent "$F_RUN/latents/test_latent_embeddings.csv" "$M_RUN/latents/test_latent_embeddings.csv" \
  --train-tsv "$F_RUN/../GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42_metadata/enriched_splits/train_enriched.tsv" \
              "$M_RUN/../GEFION_SCANONLY_ANYPLAQ_M_paper_v1_seed42_metadata/enriched_splits/train_enriched.tsv" \
  --test-tsv  "$F_RUN/../GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42_metadata/enriched_splits/test_enriched.tsv" \
              "$M_RUN/../GEFION_SCANONLY_ANYPLAQ_M_paper_v1_seed42_metadata/enriched_splits/test_enriched.tsv" \
  --target-col benchmark_anyplaq \
  --genders F M \
  --reference age_bmi \
  --bootstraps 5000 \
  --output-dir "$OUT"
```

The primary answer is in:

```text
primary_age_bmi_scan_vs_age_bmi.csv
```

The paper figure is:

```text
modality_auc_by_sex.pdf
```

The command above uses the enriched split paths created by each run. If the directory names differ, locate them with:

```bash
find /dcai/projects/cu_0080/results/checkpoints -path '*_metadata/enriched_splits/train_enriched.tsv'
```

Build the participant-level cohort table from those same frozen splits:

```bash
python paper_analysis/build_cohort_table.py \
  --train-tsv \
    "$F_RUN/../GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42_metadata/enriched_splits/train_enriched.tsv" \
    "$M_RUN/../GEFION_SCANONLY_ANYPLAQ_M_paper_v1_seed42_metadata/enriched_splits/train_enriched.tsv" \
  --test-tsv \
    "$F_RUN/../GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42_metadata/enriched_splits/test_enriched.tsv" \
    "$M_RUN/../GEFION_SCANONLY_ANYPLAQ_M_paper_v1_seed42_metadata/enriched_splits/test_enriched.tsv" \
  --target-col benchmark_anyplaq \
  --output-dir /dcai/projects/cu_0080/results/paper_analysis/cohort
```

The manifest counts unique participant IDs and stops immediately if any person
appears in both train and test.

## 3. Add confidence intervals to existing results

To audit every held-out prediction file below the checkpoint tree in one pass:

```bash
cd /dcai/projects/cu_0080/MQVAE

python paper_analysis/bootstrap_result_tree.py \
  --root /dcai/projects/cu_0080/results/checkpoints \
  --output-dir /dcai/projects/cu_0080/results/paper_ci_all \
  --bootstraps 5000
```

This writes `all_result_intervals.csv`, a complete `result_inventory.csv`, and
`result_failures.csv` for files that need attention. Participant IDs are required
by default so the script cannot accidentally report row-level intervals as
participant-level evidence.

For a specific paired comparison, use the targeted commands below. Paired CIs
are more informative than two separate model CIs because the same held-out people
are resampled in both models.

For a binary prediction file:

```bash
python paper_analysis/bootstrap_metrics.py \
  --input any_plaque=/path/to/binary_diagnostics/latest_TIMESTAMP/predictions.csv \
  --task binary \
  --bootstraps 5000 \
  --output-dir /dcai/projects/cu_0080/results/paper_analysis/ci_any_plaque
```

For a continuous target:

```bash
python paper_analysis/bootstrap_metrics.py \
  --input combined_carotid=/path/to/regression_diagnostics/latest_TIMESTAMP/predictions.csv \
  --task regression \
  --bootstraps 5000 \
  --output-dir /dcai/projects/cu_0080/results/paper_analysis/ci_combined_carotid
```

For two models evaluated on the same participants:

```bash
python paper_analysis/bootstrap_metrics.py \
  --input age_bmi=/path/to/age_bmi_predictions.csv \
  --input age_bmi_scan=/path/to/age_bmi_scan_predictions.csv \
  --reference age_bmi \
  --task binary \
  --bootstraps 5000 \
  --output-dir /dcai/projects/cu_0080/results/paper_analysis/ci_primary_pair
```

Do not relabel architecture min–max ranges as confidence intervals. The output of this script is participant-sampling uncertainty; the Boson five-seed intervals are optimization-seed uncertainty.

Add calibration for the two primary models after the modality benchmark:

```bash
python paper_analysis/plot_calibration.py \
  --input /dcai/projects/cu_0080/results/paper_analysis/modality/modality_test_predictions.csv \
  --models age_bmi age_bmi_scan \
  --genders F M \
  --bootstraps 5000 \
  --output-dir /dcai/projects/cu_0080/results/paper_analysis/calibration
```

The bin-level error bars are Wilson intervals for observed plaque prevalence.
Calibration slope/intercept and Brier score receive participant-bootstrap CIs.

## 4. Quantify the t-SNE/cluster story

Run once for each sex-specific latent export. Example for women:

```bash
python paper_analysis/analyze_latent_signal.py \
  --train-latent "$F_RUN/latents/train_latent_embeddings.csv" \
  --test-latent "$F_RUN/latents/test_latent_embeddings.csv" \
  --train-tsv /path/to/F/train_enriched.tsv \
  --test-tsv /path/to/F/test_enriched.tsv \
  --target-col benchmark_anyplaq \
  --genders F \
  --permutations 999 \
  --bootstraps 5000 \
  --output-dir /dcai/projects/cu_0080/results/paper_analysis/latent_F
```

Use `gender_F/latent_signal_summary.json` for the cluster statistics and `gender_F/tsne_latent_signal.pdf` for the figure. The statistical claims come from original latent vectors, not from t-SNE geometry.

## 5. Export and polish actual checkpoint reconstructions

After a sex-specific model finishes, submit the dedicated reconstruction job:

```bash
sbatch --export=ALL,GENDER=F,SEED=42,RUN_TAG=paper_v1 \
  paper_analysis/gefion_export_reconstructions.sh
```

The job loads `best_vae_net.pth`, reconstructs every held-out participant,
selects the example nearest the median Chamfer error in each binary target group,
exports original/reconstructed PLY and NPZ data, and then creates PNG, TIFF, and
PDF figures. It does not read or crop the poster image.

See `paper_analysis/STEP_BY_STEP_GEFION.md` for the exact output paths, manual
participant selection, axis correction, monitoring commands, and the complete
training-to-paper sequence.

## Final reporting rules

1. Report the target-specific number of participants for every metric.
2. Call a test set “held-out,” not “external,” unless it comes from a genuinely independent cohort.
3. Keep transformed R² and raw-scale R² visibly distinct.
4. Interpret AUC as pairwise ranking, not percent correct.
5. Use the paired AUC change for the primary scan-increment claim.
6. Treat eleven-target and Boson analyses as secondary/exploratory.
7. Do not claim t-SNE clusters without the original-latent permutation result.
