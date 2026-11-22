# Predicting manual body measurements from scans

Run the commands in this guide from the repository root.

## Your validation cohort

Your screenshots identify original OBJ files under
`/data/preprocessed/3D_scans` and labels in
`/data/projects/3D_scans/bodyscan_variables_for_validation_n1715.csv`.
The [configured targets](../body_measurements/targets.validation_n1715.json)
map the 17 pictured measurements to the 17 exact `*_cm_3d` headers visible in
that CSV. The leading unnamed column contains sequence numbers (`1`, `2`, …),
**not OBJ scan IDs**. Never pair these labels to OBJs by alphabetical file order.
The only safe training join is an explicit table linking each sequence number to
its OBJ filename/scan code, or a previously merged table containing both the
measurement columns and scan ID.

On the server, start with a metadata-only audit and look for a suitable mapping
table among the nearby CSV/TSV files:

```bash
python -m scripts.measurements.run_body_measurements audit \
  --metadata /data/projects/3D_scans/bodyscan_variables_for_validation_n1715.csv \
  --config body_measurements/targets.validation_n1715.json \
  --out-dir measurement_runs/validation_metadata_audit

python -m scripts.measurements.run_body_measurements inspect-sources \
  --metadata /data/projects/3D_scans/bodyscan_variables_for_validation_n1715.csv \
  --obj-root /data/preprocessed/3D_scans \
  --candidate-root /data/projects/3D_scans \
  --out-dir measurement_runs/source_inspection
```

Read `candidate_tables.csv`. It lists, for each nearby table, the number of
measurement headers it shares with the validation CSV, its strongest OBJ ID
column, and any column overlapping the numbered row key. Candidate filenames
visible in your screenshot include `cohort_with_obj.csv`,
`obj_to_cohort_merged.csv`, and `obj_files_with_stregkode.csv`; their contents
are not visible, so inspect their headers before choosing one. A high overlap
is a clue, not proof of a correct row-level join.

Generate the physical feature sidecar **directly from those OBJs**. This reads
only vertex records, requires one unique scan ID per OBJ, and can resume from
its partial CSV if interrupted:

```bash
python -m scripts.measurements.run_body_measurements build-features \
  --obj-root /data/preprocessed/3D_scans \
  --workers 2 \
  --out-dir measurement_runs/validation_obj_features
```

Check `build_audit.json`, especially `bbox_long_median_raw_units`, against the
expected body height in the OBJ coordinate units. The folder name
`preprocessed` does not establish that the meshes still have physical scale.
If the OBJ coordinates are already normalized, locate the original physical
scans or recorded scale factors before fitting centimetre targets.

Once the mapping table and its column names are confirmed, link the labels:

```bash
python -m scripts.measurements.run_body_measurements link-labels \
  --metadata /data/projects/3D_scans/bodyscan_variables_for_validation_n1715.csv \
  --mapping /path/to/confirmed_row_obj_mapping.csv \
  --map-row-col ACTUAL_ROW_NUMBER_COLUMN \
  --map-obj-col ACTUAL_OBJ_FILENAME_COLUMN \
  --features measurement_runs/validation_obj_features/scan_geometry_features.npz \
  --out-dir measurement_runs/validation_linked
```

The command fails on missing/duplicate row keys, repeated OBJ IDs, and mapped
scans without valid features. Its output
`validation_linked/metadata_with_scan_id.csv` has a verified `scan_id` column.
Then run the full target coverage audit and baseline:

```bash
python -m scripts.measurements.run_body_measurements audit \
  --metadata measurement_runs/validation_linked/metadata_with_scan_id.csv \
  --id-col scan_id \
  --features measurement_runs/validation_obj_features/scan_geometry_features.npz \
  --config body_measurements/targets.validation_n1715.json \
  --out-dir measurement_runs/validation_join_audit

python -m scripts.measurements.run_body_measurements train \
  --metadata measurement_runs/validation_linked/metadata_with_scan_id.csv \
  --id-col scan_id \
  --features measurement_runs/validation_obj_features/scan_geometry_features.npz \
  --config body_measurements/targets.validation_n1715.json \
  --scan-units mm \
  --out-dir measurement_runs/validation_baseline
```

Replace `--scan-units mm` with the **verified OBJ coordinate units**. If the
mapping table already contains all 17 measurement columns alongside a unique
OBJ ID, you can use that table directly with `--id-col` after checking row-level
agreement; the separate link step is then unnecessary. As the participant
count grows, keep these measurement column names and rerun the same audits.

The headers suggest two definitions that deserve special review:
`cross_shoulder_over_neck_cm_3d` describes an over-neck path, while the figure
calls N “shoulder width”; `hip_thigh_girth_cm_3d` is mapped to H “total thigh
circumference”. Check the original measuring protocol before interpreting
either prediction. The `*_3d` names also do not establish whether a value was
recorded manually or derived by measurement software; verify label provenance.

This is a separate supervised regression workflow. The inputs are physical
features extracted from the scans; the outputs are measurements in centimetres.
Manual measurements are training/evaluation labels only. Prediction on a new
scan needs no manual measurements, age, sex, weight, BMI, or clinical metadata.

The implementation is a **first baseline**, not a trained or validated replacement
for manual measurement. No cohort metadata or original scans are available in
this local checkout. Its tests use synthetic data and establish software behavior,
not accuracy on people.

## Measurements and target mapping

`body_measurements/targets.example.json` lists all 17 labels from the reference:

| Letter | Measurement | Letter | Measurement |
| --- | --- | --- | --- |
| A | Bust circumference | J | Abdominal circumference |
| B | Underbust circumference | K | Maximum abdominal circumference |
| C | High waist circumference | L | Waistband circumference |
| D | Waist circumference | M | Body height |
| E | High hip circumference | N | Shoulder width |
| F | Hip circumference | O | Wrist circumference |
| G | Buttock circumference | P | Left thigh circumference |
| H | Total thigh circumference | Q | Right thigh circumference |
| I | Neck circumference | | |

The general template starts disabled (`column: null`). Copy it for a different
cohort and fill only confirmed metadata columns, their units (`cm`, `mm`, or `m`), and optionally
`min_cm`/`max_cm` bounds based on your measurement protocol. Output is always cm.
For example, the repository already references the following three columns;
verify that they exist and mean the same measurements in the current export:

```json
{
  "targets": {
    "waist_circumference": {"column": "waist_circumference_cm", "unit": "cm"},
    "hip_circumference": {"column": "hip_circumference_cm", "unit": "cm"},
    "body_height": {"column": "height_cm", "unit": "cm"}
  }
}
```

Do not treat waist, high waist, waistband, and abdominal circumference as
interchangeable. Confirm landmarks, tape path, posture, measurement date relative
to scan date, wrist side, and the definitions of shoulder width and total thigh.
A target without historical labels cannot yet be learned. Distinct targets cannot
share one metadata column accidentally.

## 1. Prepare physical scan features

Use the existing `scan_geometry_features.npz` built from **original, unnormalized
OBJs**, or generate it with the existing script:

```bash
python -m scripts.preprocessing.build_scan_geometry_features \
  --npz /path/to/simplified_padded_tensors.npz \
  --obj_roots /path/to/original_bodyscans \
  --out /path/to/scan_geometry_features.npz \
  --workers 2
```

The packed NPZ supplies scan names; this baseline reads the sidecar's physical
features, not its normalized points. Existing normalization divides each person's
scan by that person's radius, removing absolute size. The sidecar's `scale_proxy`
is half a bounding-box diagonal, **not** the exact inverse of that normalization.
Do not assume that multiplying normalized points by it restores exact metric
geometry. This workflow does not perform that operation.

The sidecar contains global extents, shape ratios, coordinate spread, and vertex
count. Coordinate units are whatever the source OBJ uses; confirm them from the
scanner/export settings. All scans must use the same physical units and export
convention. `--scan-units` records that declaration and enforces it at prediction;
it does not detect actual units or repair mixed units. It is independent of label
units in the target config.
Matching feature names and declared units does not independently verify raw OBJ
provenance; check the sidecar's source paths and how it was generated.

## 2. Audit the metadata before choosing labels

From this repository directory, using an environment with NumPy, pandas,
scikit-learn, and joblib (`body_measurements/requirements.txt`):

```bash
python -m scripts.measurements.run_body_measurements audit \
  --metadata /path/to/manual_measurements.tsv \
  --id-col participant_id \
  --features /path/to/scan_geometry_features.npz \
  --out-dir measurement_runs/audit_01
```

Read `metadata_columns.csv` for exact headers and numeric coverage, `audit.json`
for ID overlap, and edit `targets.template.json` into your target config.
Repeat the audit in a new output directory with `--config /path/to/targets.json`
to check per-target usable labels and invalid-value counts. `--features` is
optional for a metadata-only audit. Malformed rows and duplicate cleaned IDs
fail explicitly; zeros, negative values, invalid numbers, and missing labels
are never converted into real measurements. Decimal commas are supported in
quoted CSV, TSV, and semicolon-separated exports.

Previously recorded source paths, **not verified from this machine**, include:

- Computerome metadata: `/data/projects/3D_scans/3d_bodyscan_20260604.tsv`
- Computerome scan data: `/data/projects/3D_scans/ML/MQVAE_combined_fast_20260808`
- Gefion metadata: `/dcai/projects/cu_0080/xfer/sftp/inbox/REACT-Q/body_scan_metadata20260817.tsv`

## 3. Train and evaluate

Example below assumes the raw OBJ coordinates are in millimetres; replace `mm`
with the verified units for your scans.

```bash
python -m scripts.measurements.run_body_measurements train \
  --metadata /path/to/manual_measurements.tsv \
  --id-col participant_id \
  --features /path/to/scan_geometry_features.npz \
  --config /path/to/targets.json \
  --scan-units mm \
  --out-dir measurement_runs/baseline_01
```

If the join column identifies scans rather than people, also supply
`--participant-col person_id` with the stable person ID. All visits/scans for the
same person then stay together. The legacy DET filename convention extracts one
ID from a filename, so duplicate DET IDs are rejected; disambiguate visits and
their labels explicitly instead of averaging or silently choosing one scan.
One metadata row must correspond to each scan and its contemporaneous labels.

The default split is approximately 60% training, 20% validation, 20% test by
participant, with a fixed seed. Each measurement uses its available labels:
missing wrist values do not discard otherwise useful waist labels. Models are:

- A training-median baseline, which ignores scan features.
- Standardized Ridge regression with four candidate regularization strengths.
- Extra Trees with fixed hyperparameters.

Preprocessing and model fitting use labelled training rows only. Validation MAE
selects the model independently for each target, including the median if it wins.
Models are not refitted on validation/test data. Only the selected model and
median comparator are evaluated on test. Too few training labels skips a target;
too few validation labels produces an explicitly labelled median-only fallback.
Defaults of 10 training and 3 validation labels are software gates, not evidence
that such small cohorts support reliable estimates.
These minimum counts refer to labelled scan rows, not unique participants.

Outputs include:

| File | Purpose |
| --- | --- |
| `audit.json` | Matched scans, missing/invalid geometry and target coverage |
| `splits.csv` | Reproducible scan and participant assignments |
| `label_coverage.csv` | Label and participant counts in each split |
| `model_selection.csv` | Selected model, validation MAE, skipped/fallback status |
| `metrics.csv` | MAE, RMSE, R², bias, descriptive limits of agreement, fractions within 1/2 cm |
| `evaluation_predictions.csv` | Validation/test predictions with observed labels where available |
| `model.joblib` | Selected estimators, target mapping, schema and units |
| `run.json` | Input hashes, settings, versions, and completion/failure status |

Reuse `--splits measurement_runs/baseline_01/splits.csv` for later models on the
same cohort. Reusing a split requires coverage of every retained scan and checks
participant separation. Do not keep trying seeds or choosing models by test
results. To add new participants, make an explicit cohort/split decision. Keep
the existing test participants held out when developing a larger model.

Metrics use available labelled scans per target; different targets may therefore
have different evaluation cohorts. Repeated scans count as separate observations
in training and metrics. Limits of agreement are residual mean ± 1.96 sample SD,
not confidence intervals or calibrated per-person prediction intervals. Small
test sets or repeated scans need further participant-level uncertainty analysis.
The 1/2 cm summaries are diagnostics, not assumed acceptance criteria.

## 4. Predict on new scans

Build features from new original scans using the same exporter and coordinate
units, then run without any measurement metadata:

```bash
python -m scripts.measurements.run_body_measurements predict \
  --features /path/to/new_scan_geometry_features.npz \
  --model measurement_runs/baseline_01/model.joblib \
  --scan-units mm \
  --out-dir measurement_runs/new_predictions_01
```

`predictions_cm.csv` has one scan per row and one column per fitted measurement.
Inspect `prediction_audit.json` for excluded scans with invalid geometry and
`model_selection.csv` for targets using only the median fallback. Predictions
are raw regression estimates: no anatomical constraints or clipping are applied.
Existing nonempty output directories are refused to preserve earlier results.
Load only trusted NPZ sidecars and joblib checkpoints, which can contain Python
objects.

## Detailed point-cloud experiment

`scripts/measurements/run_body_measurements_pointcloud.py` adds an experimental local point-cloud
regressor while leaving the original baseline implementation unchanged. This
is a candidate to test, with no claim of improved participant accuracy before
the Computerome run has been evaluated.

The cache samples 2,048 vertices per original OBJ and uses deterministic
farthest-point anchors (128, then 32). Two local nearest-neighbour pooling stages
learn detail at increasing spatial scales. Their output joins global point
features, the original 14 geometry features, and frozen baseline predictions.
A shared regression head predicts 17 corrections to the baseline, in centimetres
after undoing training-only target standardization. Layer sizes are in the
script's `MeasurementPointNet` class. This is a small custom local PointNet-style
model inspired by [hierarchical point-set learning](https://arxiv.org/abs/1706.02413),
not a reproduction of the full PointNet++ architecture.

The script reads the exact metadata/config/features paths and hashes from the
completed baseline's `run.json`, then reuses its `splits.csv`. It refuses changed
source tables or a different cohort. It reads raw OBJs referenced by the geometry
sidecar, with `--obj-root` as a fallback for missing paths. It keeps physical
scale and raw scanner orientation, translates scans to their bounding-box
midpoint, and converts coordinates to metres. It does not apply PCA, mirroring,
rotation or individual radius normalization. A consistent scan pose/export
orientation remains necessary, especially for left/right measurements. Group
separation follows the identities recorded by the baseline; unknown repeated
visits under different scan IDs cannot be inferred by this script.

Training masks unavailable labels. It selects a checkpoint by validation macro
MAE in cm and then chooses a residual weight from 0, 0.25, 0.5 or 1 for each target
using validation MAE. Weight 0 preserves the baseline for that target. The
zero-correction epoch-0 checkpoint also competes. These choices provide a
validation comparison, not a guarantee of lower test error. Test labels do not
choose the checkpoint or the residual weights.

On Computerome, first paste the contents of
[`scripts/computerome/computerome_measurement_pointcloud_setup.sh`](../scripts/computerome/computerome_measurement_pointcloud_setup.sh) into the terminal. That installs
only the new Python script in `MQVAE_new`, without an archive or dependency
installation. Then run:

```bash
cd /data/projects/3D_scans/ML/MQVAE_new
PY=/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

"$PY" -m scripts.measurements.run_body_measurements_pointcloud build-cache \
  --baseline-run measurement_runs/n2894_baseline_01 \
  --obj-root /data/preprocessed/3D_scans \
  --points 2048 --workers 2 \
  --out-dir measurement_runs/n2894_point_cache_2048

"$PY" -m scripts.measurements.run_body_measurements_pointcloud train \
  --baseline-run measurement_runs/n2894_baseline_01 \
  --cache measurement_runs/n2894_point_cache_2048 \
  --device auto --batch-size 8 --threads 2 \
  --epochs 150 --patience 25 --bootstrap 2000 \
  --out-dir measurement_runs/n2894_pointcloud_01
```

The raw-OBJ cache is a one-time resumable step. It refuses to silently drop a
failed scan. Training uses a visible CUDA GPU when available, otherwise CPU.
Use an allocated GPU host for practical training speed; no scheduler settings
are invented by these commands. `--device cuda:0` can require a GPU explicitly.
The training output directory must be new or empty.

The new run contains `best_model.pt`, the required `baseline_model.joblib`,
`run.json`, `splits.csv`, `training_history.csv`, `model_selection.csv`,
`evaluation_predictions.csv`, `validation_metrics.csv`, and `test_metrics.csv`.
The test metrics report MAE in cm, MAE as a percentage of the test mean, and 95%
participant bootstrap intervals. They also report paired new-minus-baseline MAE
on the same test rows; negative `delta_mae_cm` means lower error. Intervals
describe fixed fitted-model test uncertainty, not variability across training
seeds or individual prediction intervals. All scans belonging to a sampled
person stay together in each bootstrap sample.

For inference, preserve the whole new model run, including its frozen baseline.
Build the usual physical feature sidecar for new scans, then:

```bash
"$PY" -m scripts.measurements.run_body_measurements_pointcloud build-cache \
  --features measurement_runs/new_obj_features/scan_geometry_features.npz \
  --scan-units m --points 2048 --workers 2 \
  --out-dir measurement_runs/new_point_cache_2048

"$PY" -m scripts.measurements.run_body_measurements_pointcloud predict \
  --model-run measurement_runs/n2894_pointcloud_01 \
  --cache measurement_runs/new_point_cache_2048 \
  --device auto \
  --output measurement_runs/new_predictions_pointcloud_cm.csv
```

Use the same scan units, sampling schema, pose and orientation as training. No
measurement metadata is needed for inference. Continue to assess measurement
definitions, label quality, acceptable per-target tolerances and performance
across body sizes and scan acquisition conditions.

## Verify the implementation

```bash
python -m unittest discover -s body_measurements/tests -v
```

Tests cover unit conversion, partial labels, malformed metadata, duplicate IDs,
person-level separation, held-out-label isolation, schema checking, saved-model
prediction, and the full synthetic audit/train/predict workflow. No GPU or VAE
checkpoint is required.

## Expand the labelled scan collection

`scripts/training/train_body_measurements_expanded.py` orchestrates the existing feature builder,
baseline trainer and point-cloud trainer for a larger OBJ collection. Paste the
contents of [`scripts/computerome/computerome_train_measurement_expanded_setup.sh`](../scripts/computerome/computerome_train_measurement_expanded_setup.sh) into Computerome,
then run:

```bash
cd /data/projects/3D_scans/ML/MQVAE_new
PY=/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

nohup "$PY" -u -m scripts.training.train_body_measurements_expanded \
  --obj-root /data/projects/3D_scans/3d_bodyscans \
  --old-baseline measurement_runs/n2894_baseline_01 \
  --out-dir measurement_runs/n2894_expanded_01 \
  > measurement_runs/n2894_expanded_01.log 2>&1 &

tail -n 50 -F measurement_runs/n2894_expanded_01.log
```

Only scans with at least one valid manual target enter the supervised cohort.
Ambiguous IDs in the new collection are excluded. For original cohort IDs, the
original validated OBJ sources take precedence over other exports, preserving
the old test inputs and resolving those new-export ambiguities explicitly.
The permanently excluded scan ID RIGDET050925 remains excluded. Original files
and fitted models are retained.

All original participant split assignments are preserved. Known additional
scans of an original participant inherit that person's assignment; new people
are assigned approximately 80% to training and 20% to validation. No new people
are added to the test set. Grouping uses the original metadata participant
column, or scan IDs when no separate person column was recorded. Different
scan IDs for the same unknown person cannot be inferred automatically.

The workflow screens new scans for broad physical-size and dominant-axis
compatibility and stops on obvious format differences. This is not a complete
pose, rotation or left/right check. Original geometry must reproduce its saved
sidecar. Invalid added OBJ geometry is listed in `features/*.failures.csv` and
the retained cohort is recorded; loss of any original scan stops the workflow.

The output folder must be new. It contains source and split manifests,
`expansion_audit.json`, `features/`, a newly trained `baseline/`,
`point_cache_2048/` and the advanced `pointcloud/` run. The latter contains
`test_metrics.csv` and the saved deployment files. Defaults are 2,048 points,
150 epochs, early stopping after 25 epochs without validation improvement,
and 2,000 bootstrap replicates. `--device auto` uses a visible CUDA GPU or CPU.
Use an allocated compute environment for this work; these commands do not
request scheduler resources.
