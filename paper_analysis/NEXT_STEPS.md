# Territory, sex-specific, and plaque-burden experiments

All new launchers include **female and male models separately**. Nothing installs
or upgrades Python packages. The existing scan-only scripts and their experiment
names are unchanged. This is a new study using the existing MeshVAE architecture.

## Copy the implementation, not just the shell scripts

Upload the updated `next_steps_upload.zip` to your code directory on the chosen cluster and
extract it there. It contains the required updates under `meshvae/` and the new
files under `paper_analysis/`. Alternatively synchronize the files listed in
`next_steps_upload_manifest.txt`. Keep the directory structure. Uploading only the
shell scripts will leave the new training flags unsupported.

The bundle contains source and instructions, not participant data or packages.
It is an update for the full repository, not a standalone project.

## What is included

| Suite | Models, separately in F and M | Array tasks / model fits |
|---|---|---:|
| territory_classification | Plaque presence in right/left/combined carotid, right/left/combined femoral, coronary | 14 / 14 |
| territory_volume | Plaque volume in those seven territories; log1p + Huber loss | 14 / 14 |
| plaque | All-participant raw Huber, all-participant log1p Huber, positive-only log1p Huber, positive-only log1p median, two-part hurdle | 10 / 12 |
| sensitivity | Upper age-specific tail and highest 5% plaque burden | 4 / 4 |
| weight | Matched scan + age versus scan + age + weight, for presence and volume | 8 / 8 |

Running everything means **50 tasks and 52 neural model fits**, before any
unavailable targets are skipped. The initial comparison uses **150 epochs, no
decay, batch size 8**, consistently across the new experiments. This is a screening
schedule, not a claim of convergence. Change `niter`, `niter_decay`, and `run_tag`
in `paper_analysis/next_steps_config.json` for a longer validation-selected study.
Keep a new run tag when changing any settings or metadata. Do not choose the final
model by repeatedly optimizing against the existing held-out test results.

## Gefion: plain sbatch submissions

Run from `/dcai/projects/cu_0080/MQVAE`. First check the local metadata and targets
without training or allocating a GPU:

```bash
bash paper_analysis/gefion_next_audit.sh
```

This writes each experiment's `study/manifest.json`, selected TSVs, and
`study/status.json` under the configured checkpoint directory. Check missing
columns, target availability, and cohort sizes before submitting. Weight must be
a measured weight in **kg**. If the column is not recognized, set `weight_col` in
the JSON configuration to its exact name. `age_col` can also be specified.

Submit whichever suites you want:

```bash
sbatch paper_analysis/gefion_next_territory_classification.sh
sbatch paper_analysis/gefion_next_territory_volume.sh
sbatch paper_analysis/gefion_next_plaque.sh
sbatch paper_analysis/gefion_next_sensitivity.sh
sbatch paper_analysis/gefion_next_weight.sh
```

No `--export`, no gender edits, and no separate male submission. Both sexes are
already included in each array. Every array task requests one GPU, 8 CPUs, 128 GB,
and at most three days. Each suite allows at most two array tasks simultaneously;
submitting all five permits up to ten concurrent GPUs. The hurdle task trains its
two components sequentially within the same allocation.

Logs are written **in the directory you submitted from**, for example
`next_plaque_123456_0.out` and `.err`. The first number is the array job ID and the
second is the task index. The indices for each suite are female experiments first,
then male experiments, in the order printed by:

```bash
source paper_analysis/next_steps_environment.sh gefion
"$STUDY_PYTHON" paper_analysis/run_plaque_study.py --platform gefion --suite plaque --list
```

## Computerome: queued GPU job

`dpreact-002` may have no GPU. `nohup` alone keeps a process alive but does not
request a GPU, so a run there stops at the preflight check. Audit and collection
work on the CPU session; training requires a queued GPU allocation. Both `qsub`
and `sbatch` clients may be installed; select the scheduler and queue actually
available to your account. Do not infer GPU access from the presence of either
client. Request one GPU, eight CPUs, sufficient RAM, and walltime in the queue
command. The Python launcher keeps the GPU mask assigned by PBS or Slurm.
If Torque exposes the GPU through `PBS_GPUFILE` instead, the launcher reads the
assigned device there. A queued task stops before training when neither method
identifies its GPU.

For the full study, submit `computerome_next_array_job.sh` with array slots **1-50**
and at most two concurrent GPU jobs. Each task trains one sex and one experiment,
except the hurdle task, which fits its two components sequentially. The queue
script maps slots 1-50 to study task indices 0-49 and writes
`logs/next_task_<index>.log` under the submitted checkout. Computerome's
[workshop guide](https://delphiapp.computerome.dk/strapi/uploads/Computerome_New_User_Workshop_03032022_e5e9c38a9e.pdf)
documents `qsub -t 1-100%10` arrays and `$PBS_ARRAYID` on its PBS system. The
script also accepts Slurm's `$SLURM_ARRAY_TASK_ID` if that is the scheduler your
account uses. Use `--list` to see task numbers before submitting:

```bash
source paper_analysis/next_steps_environment.sh computerome
"$STUDY_PYTHON" paper_analysis/run_plaque_study.py --platform computerome --suite all --list
```

From `MQVAE_new`, first check that the PBS server responds, then submit the
array on Computerome's documented Torque/PBS system:

```bash
qstat -q
qsub -N mqvae_next -l nodes=1:ppn=8:gpus=1,mem=128gb,walltime=24:00:00 \
  -t 1-50%2 paper_analysis/computerome_next_array_job.sh
qstat -t -u "$USER"
```

This requests one GPU and eight CPU cores per task, with at most two tasks running
at once. The resource line follows [Torque's GPU request syntax](https://docs.adaptivecomputing.com/torque/3-0-5/3.7schedulinggpus.php).
Computerome account, queue, and walltime limits are site-specific; if `qstat`
cannot contact its server or `qsub` rejects the resource line, inspect the exact
message before trying the Slurm client. The presence of `/usr/bin/sbatch` alone
does not identify a Slurm GPU partition.

For a single sequential GPU job, submit `computerome_next_queued_job.sh` from
the checkout root. It runs all experiments and writes `logs/next_all_gpu0.log`.

The code can live in **`/data/projects/3D_scans/ML/MQVAE_new`**. The Computerome
checkpoint setting is now relative to the repository containing the scripts:
`checkpoints` means `MQVAE_new/checkpoints` when launched from this checkout.
The shared scan archive, metadata, and train/test membership files retain their
existing paths. Renaming the code folder does not require copying those datasets.
The launchers also put this checkout on `PYTHONPATH`, including for child processes.

Upload the latest bundle to `MQVAE_new`, then apply it before launching:

```bash
cd /data/projects/3D_scans/ML/MQVAE_new
unzip -o next_steps_upload.zip
```

Run the audit first. It checks both sexes, target availability, measured weight,
and cohort sizes without training. Its first lines must show the checkpoint root
under `MQVAE_new`. A target marked `unavailable` is skipped with its reason; a
`[FAILED]` message needs attention before the full run.

```bash
bash paper_analysis/computerome_next_audit.sh
```

Only after an array task begins, watch its log from the checkout:

```bash
tail -f logs/next_task_0.log
```

The array runs all suites and both sexes as independent GPU tasks. Do not launch
the sequential job or individual suite scripts at the same time. Ctrl+C stops
`tail` only. Individual failures are logged per task. The status collector shows
completed, unavailable, and failed experiments.
For only one suite, use the corresponding command instead:

```bash
bash paper_analysis/computerome_next_territory_classification.sh 0
```

The other independent launchers are:

```text
paper_analysis/computerome_next_territory_volume.sh
paper_analysis/computerome_next_plaque.sh
paper_analysis/computerome_next_sensitivity.sh
paper_analysis/computerome_next_weight.sh
```

Each accepts a GPU index as its first argument. Run them only inside a GPU
allocation, one at a time per GPU.
They use the existing `/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python`.
Before training, they check the required imports and GPU once, and print the
Python executable used. A missing dependency fails this check without attempting
all model fits. The scripts do not install or upgrade anything.

## Recommended plaque analysis

The number of zeros is not by itself proof of why R2 is low. Weak signal, skewed
positive values, measurement noise, sample size, and model fitting can also matter.
Compare these candidates using internal validation, and report the prespecified
held-out results without promising an R2 improvement:

1. **All-participant log1p + Huber** is the practical reference. It preserves true
   zeros and reduces the influence of very large volumes. This transform already
   existed in the code; it is not a new remedy by itself.
2. **Positive-only regression** answers how much plaque is present among people
   with plaque. Its held-out metrics are conditional on plaque positivity and
   cannot replace whole-population metrics.
3. **Two-part hurdle** estimates plaque probability on all volume-observed
   participants and burden on positive participants. The binary model uses
   unweighted BCE. The positive model uses squared error on standardized log1p
   volume. A residual-smearing factor estimated on positive internal-validation
   predictions corrects retransformation before multiplying by plaque probability.
   The combined model is evaluated on all eligible held-out participants, including
   zeros. This is an approximate mean predictor: it assumes the log-residual
   distribution is sufficiently stable across predictors, and unweighted training
   does not guarantee calibrated probabilities. Inspect the component predictions
   and calibration before interpreting predicted volumes clinically. The calibration
   split is also used for early model selection; an independent calibration split
   would be preferable in a larger confirmatory study.
4. **Median model** uses absolute error on log1p positive volume, with checkpoint
   selection by validation log MAE. It targets a robust conditional median,
   not the mean; R2 is not its optimization criterion.
5. **Age-tail and top-5% models** are exploratory comparisons only. They can reveal
   different patterns but cannot establish better whole-population prediction or
   a causal link between body size and plaque.

Do **not replace true zeros with a small positive number**. [`log1p(0)=0`](https://numpy.org/doc/stable/reference/generated/numpy.log1p.html) already
handles zeros, while replacing them changes the outcome without solving the
underlying mixture. The reference reports include a training-median constant
baseline, so a model that merely predicts the large zero mass is visible.

Method references: [two-part models and retransformation](https://pubmed.ncbi.nlm.nih.gov/10180918/)
and [median/quantile regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html).
These support the analysis choices, not a claim of success on this dataset.

## Target definitions and eligibility

- Volume uses corrected right/left carotid and femoral fields when present.
  Combined carotid/femoral volume uses `pv_caro`/`pv_femo` when available, otherwise
  the sum of both measured sides. Direct-versus-summed discrepancies are audited.
- Coronary volume uses `pv_coro`, with `ct_qa_cor_tot_plaq_vol` or
  `coronary_plaque_volume` as fallback. Verify consistent units in your source data.
- Classification is **measured plaque presence**, volume > 0, not stenosis severity,
  calcium score, or a substitution of CCTA any-CAD for plaque presence.
- `peripheral_total` is the existing four-site sum: right/left carotid plus
  right/left femoral. All four volumes must be observed. Coronary volume is not
  silently added to this total.
- A measured zero stays zero. Negative/nonfinite volumes are unknown. For combined
  presence, one positive observed side establishes positivity, but a negative label
  requires complete negative measurements or a valid directly measured total.
- Participants must have a unique scan, finite adult age, measured positive weight,
  and complete scan-geometry features. Both weight-comparison variants use identical
  participants, including the age-only variant. This is a complete-case study;
  exclusions are counted in the manifest and should be reported in the paper.
- Existing outer train/test membership is preserved and checked for overlap.
  The development split is divided into training and validation (15%) before
  fitting transformations, weight/geometry normalization, or subgroup thresholds.
  The same split is shared across experiments within each sex. Outcomes from the
  test split never determine whether a model is trained.
- Training requires at least 50 participants and validation at least 10, with
  target variation; classification also needs at least two observations per class
  in both splits. These are minimum guards, not evidence of adequate statistical
  power for a neural network. Unavailable experiments are recorded and skipped.

## Age-tail and highest 5% details

The highest 5% means the upper tail above the **training-sex-specific 95th
percentile of peripheral volume**, also requiring volume > 0. It does not mean
five individuals. The threshold is frozen and applied to validation and test.

The age-tail default retains the upper **20% within fixed 10-year age bands**,
separately by sex. Change `age_top_fraction` and `age_band_width` in the config
before the study if a different prespecified experiment is desired. Thresholds
come only from training participants; unseen validation/test age bands are
excluded. A zero cutoff retains zeros, consistent with the proposed age strategy.
Ties at either type of cutoff are retained, so the selected fraction can exceed
5% or 20%; the exact counts are recorded. No test-set quantiles are fitted.

Because selection requires knowing the true plaque measurement, these are
retrospective subgroup models, not a screening rule for identifying unknown
high-plaque patients. Restricting the outcome range can reduce or destabilize R2.
Small selected cohorts may be skipped even when their full-cohort model can train.

## Weight and body-size comparisons

New primary models feed scans, age encoding, and standardized measured weight
into the encoder. Weight mean/std are fitted on training participants and saved in
`cond_stats.json`; validation and test reuse them. Sexes have separate fitted
models and statistics. BMI is not substituted for weight. Existing models have
`cond_weight=False` by default and retain their original dimensions.

Each evaluated model gets same-participant reference comparisons: training mean,
training median for regression, age, weight, age + weight, age + scan geometry,
and age + weight + scan geometry. The regression baselines are raw-volume ridge
regressions; classification baselines are unweighted regularized logistic models.
They use fixed regularization and training-only scaling, not optimized test fits.
These small models use numpy/pandas and require no new packages.

Regression reports also include held-out weight/geometry correlations with plaque,
Spearman correlations, and partial correlations with log1p plaque after adjusting
for age and age squared. These are exploratory associations, not causal estimates;
no multiplicity-adjusted significance claims are made.

## Results and restart behavior

Each run uses a separate name such as
`GEFION_NEXT_plaque_all_log_F_next_v1_seed42` under the platform checkpoint root.
Look in `study/heldout/` for:

- `comparison.csv`: model and baseline metrics (R2, MAE, median absolute error,
  RMSE, Pearson, Spearman, log MAE; AUC/AP/Brier for classification).
- `predictions.csv`: paired participant-level predictions.
- `intervals.json`: participant-bootstrap 95% intervals for AUC or MAE, plus the
  paired difference versus the age + weight baseline; fixed models, not retraining.
- `exploratory_body_size_associations.csv`: regression associations described above.

Small held-out groups retain their sample count and undefined metrics as missing.
If no eligible test participant exists, no performance claim is produced.
R2 comparisons across different cohorts are not equivalent; focus on matched
baselines within each cohort and report the cohort definition alongside metrics.

`study/status.json` records audited/unavailable/running/failed/dry_run/completed states.
Successful training gets an explicit completion marker. An existing partial best
checkpoint is **not** treated as finished: after interrupted training, use a new
run tag to restart. Re-running a completed task reuses its finished training and
can repeat diagnosis. This runner does not claim to resume optimizer state.

Collect results on Gefion:

```bash
source paper_analysis/next_steps_environment.sh gefion
"$STUDY_PYTHON" paper_analysis/collect_plaque_study.py --checkpoints-dir /dcai/projects/cu_0080/results/checkpoints --run-tag next_v1
```

On Computerome, the collector reads the same configuration as training:

```bash
cd /data/projects/3D_scans/ML/MQVAE_new
bash paper_analysis/computerome_next_results.sh
```

This can be run during training or after it finishes. It writes
`MQVAE_new/next_steps_results/status.csv` and `comparisons.csv`, and prints a
status-count summary. The comparisons contain held-out results and matched
baselines, not the validation score used to select checkpoints. Detailed
predictions, intervals, and body-size associations stay in each run's
`checkpoints/COMPUTEROME_NEXT_*/study/heldout/` directory.

To package the summary files for transfer after collection:

```bash
tar -czf next_steps_results.tar.gz next_steps_results
```

Download `MQVAE_new/next_steps_results.tar.gz` using the existing SFTP connection.
No reconstruction export is required to obtain AUC or regression metrics for
these new experiments. If you deliberately set an absolute `checkpoints` path in
the config, both training and this collector honor it; the default keeps new
checkpoints in the new checkout.

## One overview across Computerome and Gefion

Use `collect_all_results.py` from the existing Python environment on each server.
It reads saved results; it does not train models, allocate GPUs, or copy participant
predictions unless `--include-plot-data` is supplied. It collects older comparisons, binary/regression diagnoses, NEXT
studies and Boson ablations. Original fields and source paths are retained.
The plotting workflow uses that flag to export participant predictions, latent
embeddings and matching metadata with a portable path map. No model weights or
raw body scans are exported. The notebook cannot reconstruct t-SNE from a summary
score; missing embeddings must first be extracted from the trained checkpoint.

Copy the changed source files directly with SFTP, preserving their `paper_analysis/`
paths. The full results workflow also uses the updated extractor, bootstrap helpers,
modality loader and notebook. Matched Boson training needs the updated study runner
and `gefion_next_boson.sh`. No archive extraction is needed.

On Computerome, from your code checkout:

```bash
cd /data/projects/3D_scans/ML/MQVAE_new
PY=/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

"$PY" paper_analysis/collect_all_results.py \
  --platform computerome \
  --root /data/projects/3D_scans/ML/MQVAE/checkpoints \
  --root /data/projects/3D_scans/ML/MQVAE/checkpoints_pairwise \
  --root /data/projects/3D_scans/ML/MQVAE/checkpoints_supervised \
  --root /data/projects/3D_scans/ML/MQVAE_new/checkpoints \
  --output-dir results_overview_computerome --include-plot-data

```

These are the three checkpoint folders in `MQVAE` plus the single checkpoint
folder in `MQVAE_new`. Each row retains `source_root` and `run_directory`, so
identical experiment names in different folders remain separate. If a root is
unavailable, the inventory records it.

On Gefion:

```bash
cd /dcai/projects/cu_0080/MQVAE
source paper_analysis/next_steps_environment.sh gefion
"$STUDY_PYTHON" paper_analysis/collect_all_results.py \
  --platform gefion \
  --root /dcai/projects/cu_0080/results/checkpoints \
  --output-dir results_overview_gefion --include-plot-data

```

Download both complete export directories with your existing SFTP connection into
`outputs/model_overview/` in this repository. Their directory names must
remain `results_overview_computerome` and `results_overview_gefion`. Open
`paper_analysis/results_overview.ipynb` in Jupyter or VS Code and run all cells.
The notebook automatically combines these two inputs. It computes/caches missing
95% participant intervals from verified held-out predictions using the existing
bootstrap code and draws performance plots, total-volume tables, Boson comparisons
and checkpoint-specific t-SNE figures. Unknown split provenance, missing CIs,
missing embeddings and metadata exclusions remain visible in its inventories.
Figures and tables are saved under `outputs/model_overview/plots/`.

An explicit merged export is optional:

```bash
python paper_analysis/collect_all_results.py \
  --merge results_overview_computerome/all_results.csv results_overview_gefion/all_results.csv \
  --output-dir results_overview_combined
```

Start with `all_results.csv`; inspect `result_inventory.csv` for missing evaluations and
unreadable files. `preferred` distinguishes current result artifacts from older,
partial, or repeated reports where the saved formats establish that relationship.
Do not count all exported rows as distinct trained models: a study also reports
baselines, and wrappers can repeat component results. Saved confidence intervals
are collected separately in `confidence_intervals.csv`; their source and
resampling unit are retained. An unavailable CI stays unavailable.
`plot_artifacts.csv` maps original remote paths to relative files under `plot_data/`;
keep it with its export folder when transferring files. Its failure rows identify
referenced metadata that could not be copied.

Compare models within the same target, cohort, sex, evaluation split and metric
scale. Combining the exports is an overview, not pooling the two servers' test
participants or ranking unrelated experiments together.

## Optional: confidence intervals on the server

The notebook can calculate these after export. Alternatively compute them on the
server with the commands below. The PowerPoint was transcribed from screenshots,
so its rounded scores cannot
provide confidence intervals. The new study already saves primary AUC/MAE
intervals in `study/heldout/intervals.json`. For 95% intervals on R2, MAE,
Spearman and the other model metrics, reuse the existing participant bootstrap:

```bash
CKPT=/dcai/projects/cu_0080/results/checkpoints
for STUDY in "$CKPT"/*_NEXT_*/study; do
  if [ -f "$STUDY/test_predictions.csv" ]; then
    python paper_analysis/bootstrap_metrics.py \
      --input "meshvae=$STUDY/test_predictions.csv" --task auto \
      --bootstraps 5000 --confidence 0.95 --output-dir "$STUDY/paper_ci"
  fi
done
```

Set `CKPT` to the appropriate Computerome root to repeat there. Then rerun the
collector to include `bootstrap_metric_intervals.csv`. Use `estimate`, `ci_low`
and `ci_high` as the point and error-bar limits. Raw-volume and transformed-volume
metrics are different rows. The wide `study/heldout/predictions.csv` includes
baselines and needs one model column selected before using this bootstrap tool.
Summary-only legacy comparisons cannot acquire participant CIs by collection alone.

`bootstrap_result_tree.py` now inventories non-test artifacts and defaults to
known test filenames or diagnoses referenced by their summary. Historical
evaluations can still appear as separate artifacts; select the saved evaluation
for the intended checkpoint before plotting. A diagnosis whose input split is
not recorded is not established as a held-out test merely by its folder name.

## Total volume and t-SNE inputs

Total peripheral plaque volume is already in presentation slides **6–10 and
18–19**. The whole-cohort rows are `all_raw`, `all_log` and `hurdle`; positive-only
and upper-tail cohorts answer separate questions. Its target is
`volume_peripheral_total` (older name `plaque_volume_total_corrected`): the four
carotid/femoral sides. Coronary volume remains a separate endpoint.

If completed NEXT models do not yet have latent embeddings, run this once on
Computerome **before collection**. `--gpu-ids -1` uses CPU inference and needs no
GPU allocation, but extraction can take substantial time. It extracts all completed
NEXT studies found in the four roots. Existing embeddings are reused by this
explicit `--skip-existing` option; remove it to refresh embeddings after changing
a checkpoint. Older runs without saved study metadata remain listed as missing
inputs instead of guessing their participant splits.

```bash
"$PY" paper_analysis/extract_mesh_latents.py \
  --platform computerome \
  --study-root /data/projects/3D_scans/ML/MQVAE/checkpoints \
  --study-root /data/projects/3D_scans/ML/MQVAE/checkpoints_pairwise \
  --study-root /data/projects/3D_scans/ML/MQVAE/checkpoints_supervised \
  --study-root /data/projects/3D_scans/ML/MQVAE_new/checkpoints \
  --gpu-ids -1 --skip-existing
```

The matching Gefion command, after sourcing its environment above, is:

```bash
"$STUDY_PYTHON" paper_analysis/extract_mesh_latents.py \
  --platform gefion --study-root /dcai/projects/cu_0080/results/checkpoints \
  --gpu-ids -1 --skip-existing --output-dir latent_extraction_gefion
```

Inside an allocated GPU job `--gpu-ids 0`
can accelerate the same inference. The notebook can then visualize all exported
checkpoint spaces separately. To extract/analyze one particular whole-cohort volume
model manually, replace the following example name with one present in the overview:

```bash
CKPT=/dcai/projects/cu_0080/results/checkpoints
NAME=GEFION_NEXT_plaque_all_log_F_next_v1_seed42
RUN="$CKPT/$NAME"
python paper_analysis/extract_mesh_latents.py \
  --name "$NAME" --checkpoints-dir "$CKPT" \
  --dataroot /dcai/projects/cu_0080/results/MQVAE_gefion_20260823 \
  --train-tsv "$RUN/study/train.tsv" --test-tsv "$RUN/study/test.tsv" \
  --gender F --which-epoch best --output-dir "$RUN/latents" \
  --cond_stregkode_col participant_id --cond_age_col study_age --cond_gender_col study_gender

python paper_analysis/analyze_latent_signal.py \
  --train-latent "$RUN/latents/train_latent_embeddings.csv" \
  --test-latent "$RUN/latents/test_latent_embeddings.csv" \
  --train-tsv "$RUN/study/train.tsv" --test-tsv "$RUN/study/test.tsv" \
  --target-col presence_peripheral_total --volume-col volume_peripheral_total \
  --id-col participant_id --age-col study_age --gender-col study_gender --genders F \
  --representation-label "Scan + age + weight latent representation" \
  --bootstraps 5000 --output-dir "$RUN/latent_analysis"
```

Repeat with the male run and `--gender M` / `--genders M`. This analysis requires
both plaque-presence classes and complete age/BMI metadata; it reports the
resulting exclusions. It adds a measured-volume color panel without treating
continuous volume as a binary target. Scaler/PCA fitting uses training latents.
t-SNE dots do not receive metric confidence intervals; the performance and latent
probe plots do. Each checkpoint has its own coordinates. A hurdle has separate
presence and positive-volume encoders, not one combined latent space.

## Gefion: matched Boson training for total plaque volume

The new launcher reuses the completed **all-participant log-volume models in both
sexes**, their saved `study/train_command.json`, manifest configuration and exact
train/validation/test TSVs. It preserves architecture, age/weight inputs, point
labels, loss and training budget. It compares Gaussian-MMD to Boson-MMD at training
seeds 42, 43 and 44. All six fits for each sex use the original frozen participants.
Existing Gaussian-KL results remain a separate reference.

From the updated Gefion checkout submit:

```bash
cd /dcai/projects/cu_0080/MQVAE
sbatch paper_analysis/gefion_next_boson.sh \
  next_v1 /dcai/projects/cu_0080/results/checkpoints 42 next_boson_v1

squeue -u "$USER"
```

This schedules **12 tasks**, with at most two concurrent GPUs, one GPU/128 GB/three
days requested per task. The arguments are the original reference run tag,
checkpoint root, reference training seed and new output tag. The defaults reference
`GEFION_NEXT_plaque_all_log_F_next_v1_seed42` and its male counterpart. A missing or
incomplete reference stops that task instead of training an unmatched substitute.
The default uses the existing linear Sparrow Boson sampler and checks it before
training. New run names include the prior arm and seed; original checkpoints are
retained. Logs are `next_boson_<job>_<task>.out`/`.err` in the submission folder.

After the tasks finish, extract any desired latents, rerun Gefion collection and
download the refreshed export; rerunning the notebook includes the new results.
Seed-aggregate Boson intervals describe variation
over training runs; participant-bootstrap intervals describe test-sample
uncertainty. Keep those two kinds of interval explicitly labeled.

## Validation performed locally

The CPU-only tests exercise target derivation, missing/negative values, participant
isolation, sex separation, training-only cutoffs, frozen weight normalization,
backward-compatible condition dimensions, all generated training flags, metrics,
two-part retransformation, and an end-to-end dry run with synthetic metadata.
Batches of female/male dry runs in a renamed checkout also verify that training
and collection use that checkout's results even when invoked from another cwd.
Bash syntax is checked for all new shell scripts. Run tests with:

```bash
python -m unittest paper_analysis.test_plaque_study -v
```

No real participant data or cluster connection is available for these checks.
Full GPU training and cluster scheduling must be
validated on the target host; the audit is designed to expose metadata issues first.
