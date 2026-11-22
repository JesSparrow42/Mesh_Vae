# Selected Gefion boson comparison

Two model configurations, each trained separately for women and men. These were
selected from `Gefion_plaque_results_visualized.pptx`.

| Configuration | Inputs | Main comparison metric | Original female result | Original male result |
| --- | --- | --- | --- | --- |
| `binary_age` | Scans + age | AUC, higher is better | 0.854 (N=412) | 0.840 (N=383) |
| `all_log` | Scans + age + weight | MAE on original volume scale, lower is better | 37.440 (N=395) | 69.994 (N=368) |

The classifier predicts any peripheral plaque. The regression predicts total
peripheral plaque volume (carotid + femoral); neither target includes coronary
volume. The regression uses the original log1p target transform and Huber loss,
with the full eligible cohort, including zeros. The log-volume configuration was
chosen for its low average absolute error; other configurations had higher R².
Scores above are rounded values from the presentation. The collector reads the
actual saved predictions and recomputes scores at full precision.

## Upload and submit

Upload `gefion_best_boson.zip` into the existing Gefion transfer directory:
`/dcai/projects/cu_0080/xfer/sftp/inbox/REACT-Q/`.

Run this block once on the Gefion login node:

```bash
mkdir /dcai/projects/cu_0080/MQVAE_boson_best && \
unzip '/dcai/projects/cu_0080/xfer/sftp/inbox/REACT-Q/gefion_best_boson.zip' \
  -d /dcai/projects/cu_0080/MQVAE_boson_best && \
cd /dcai/projects/cu_0080/MQVAE_boson_best && \
source paper_analysis/next_steps_environment.sh gefion && \
"$STUDY_PYTHON" paper_analysis/selected_boson_study.py check && \
sbatch paper_analysis/gefion_best_boson.sh
```

The separate code directory avoids replacing the old checkout. If that directory
already exists from unpacking this bundle, continue from `cd` instead of unpacking
again. The read-only `check` verifies all four original runs before submission;
if it fails, fix the reported missing reference rather than starting a fresh
cohort. There are no package installations or upgrades in these commands.

The submission creates **eight GPU tasks, at most two active at a time**:

| Array index | Model | Sex | New prior/loss |
| --- | --- | --- | --- |
| 0 | binary_age | F | Gaussian / MMD |
| 1 | binary_age | F | Boson / MMD |
| 2 | binary_age | M | Gaussian / MMD |
| 3 | binary_age | M | Boson / MMD |
| 4 | all_log | F | Gaussian / MMD |
| 5 | all_log | F | Boson / MMD |
| 6 | all_log | M | Gaussian / MMD |
| 7 | all_log | M | Boson / MMD |

This is four boson fits and four Gaussian controls, all with training seed 42.
Original Gaussian/KL runs are included in the final comparison without being
retrained or overwritten. Both new arms use MMD with 64 buffered latent samples
and at least 64 prior samples; deterministic-kernel preference is enabled for
both. Their other saved training options, including epochs, batch size, feature
inputs, learning rate and model-selection settings, are reused from the original
run. Training, validation and held-out TSVs are copied from the original saved
study, without regenerating splits or selecting a new subset. This does not
guarantee identical initial weights or reproduce an unarchived old code version.

The two new arms hold the training loss fixed so their comparison can assess the
change of prior. Comparing against the original Gaussian/KL result also changes
the regularizer. All original reference studies must have status `completed` and
saved manifests, training commands, splits, checkpoints and held-out predictions.

## Check progress and logs

```bash
squeue -u "$USER"
```

After returning to the code directory, print the final 40 lines of each task's
stdout and stderr:

```bash
cd /dcai/projects/cu_0080/MQVAE_boson_best
tail -n 40 best_boson_*.out best_boson_*.err
```

Slurm `COMPLETED` and exit code `0:0` should accompany a study status of
`completed`. The launcher exits unsuccessfully if the study is unavailable or
training/evaluation fails. The collector below can be run while jobs are still
active and shows which of the eight have completed.

## Print and collect the results

```bash
cd /dcai/projects/cu_0080/MQVAE_boson_best
source paper_analysis/next_steps_environment.sh gefion
"$STUDY_PYTHON" paper_analysis/selected_boson_study.py collect
```

This requires no GPU. It prints statuses, all available original/Gaussian-MMD/
Boson-MMD scores and paired differences, then saves:

- `best_boson_results/comparison.csv`: AUC, average precision, MAE, R² and other
  applicable held-out metrics. Twelve rows when all eight new runs are complete.
- `best_boson_results/paired_differences.csv`: Boson minus Gaussian-MMD and Boson
  minus original Gaussian-KL, using AUC for classification and MAE for volume.
- `best_boson_results/status.csv`: all four originals and eight new run states;
  missing/failed runs are not reported as zero scores.
- `best_boson_results/comparison_notes.json`: comparison and interval definitions.

For classification, a positive AUC difference favors boson. For regression, a
negative MAE difference favors boson. The collector aligns participants and
targets, verifies frozen reference hashes and saved training settings, and adds
95% paired participant-bootstrap intervals (2,000 resamples). These intervals
describe fixed fitted models, not variation across training seeds. Selection
used these existing held-out results, so this is an exploratory comparison rather
than an independent confirmatory evaluation. AUC ranking and absolute volume
error remain separate questions; high AUC does not imply accurate volume.

Download the CSVs from
`/dcai/projects/cu_0080/MQVAE_boson_best/best_boson_results/`
with the same file-transfer method used for earlier results. Reconstructions are
not required to obtain these prediction scores.

## Reference and output locations

Default checkpoint root: `/dcai/projects/cu_0080/results/checkpoints`.

Originals:

```text
GEFION_NEXT_weight_binary_age_F_next_v1_seed42
GEFION_NEXT_weight_binary_age_M_next_v1_seed42
GEFION_NEXT_plaque_all_log_F_next_v1_seed42
GEFION_NEXT_plaque_all_log_M_next_v1_seed42
```

New outputs use a distinct arm and tag, for example:

```text
GEFION_NEXT_weight_binary_age_boson_mmd_F_best_boson_v1_seed42
GEFION_NEXT_plaque_all_log_gaussian_mmd_M_best_boson_v1_seed42
```

If the original study used a different tag, use it consistently when checking,
submitting and collecting:

```bash
"$STUDY_PYTHON" paper_analysis/selected_boson_study.py check --reference-tag YOUR_ORIGINAL_TAG
sbatch paper_analysis/gefion_best_boson.sh YOUR_ORIGINAL_TAG best_boson_v1
"$STUDY_PYTHON" paper_analysis/selected_boson_study.py collect --reference-tag YOUR_ORIGINAL_TAG
```

Do not resubmit an already active array. Completed runs can be evaluated again;
an incomplete checkpoint is not silently restarted. To start an intentional new
comparison, pass a new run tag as the second script argument and use the same
`--run-tag` when collecting.

## Sampler and validation

This uses the checked-in Sparrow photonic **simulator**, not a hardware sampling
service. The bundled sampler source is selected explicitly to avoid a different
installed version. Existing defaults use eight modes, four photons, 100 loops,
eight cached circuits and a 4,096-vector cache, with linear settings. No sampler
physics code is edited. Each boson task runs the existing integrity preflight
(Hong–Ou–Mandel probabilities, photon conservation, shape, finite values and
diversity) before training and logs the sampler source, version and configuration.

Local checks cover all eight command routes, exact saved participant membership,
unchanged training settings, both covariate configurations, paired result
alignment, missing runs and rejection of changed references/settings. Local
shell syntax validation passed. The local bundled Python does not include
PyTorch, so the actual sampler/GPU preflight and real training must be verified
on Gefion. The launch reuses the existing Gefion module stack and extra-package
directory; it does not install or upgrade dependencies.
