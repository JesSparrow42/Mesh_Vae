# MeshVAE

Mesh and point-cloud models for body-scan analysis, clinical target prediction,
and body-measurement regression. The repository contains the current MeshVAE
implementation, experiment runners, and paper-analysis workflows.

## Repository layout

```text
meshvae/              Current MeshVAE implementation
body_measurements/    Measurement models, target configurations, and tests
paper_analysis/       Study workflows, result analysis, and paper-specific guides
third_party/          Vendored photonic sampler and upstream attribution
scripts/
  computerome/        Computerome setup and terminal scripts
  gefion/             Gefion preprocessing and training scripts
tools/                Builders for terminal installers and delivery scripts
notebooks/            Current exploratory analysis notebook
docs/
  commands/           Saved cluster command snippets
  history/            Historical patch notes and original file inventory
archive/              Earlier implementations, experiments, and work notebooks
outputs/              Saved delivery helpers and result manifests
*.py                  Supported command entry points and experiment runners
```

Run Python commands from the repository root. The root entry points and package
paths are retained so existing training commands and imports continue to work.
Files in `archive/` are historical references; use the root scripts and current
packages for new runs.

## Getting started

Use the Python environment configured for your experiment. The baseline
measurement dependencies are listed in
[`body_measurements/requirements.txt`](body_measurements/requirements.txt);
the VAE and point-cloud training workflows need their additional model and GPU
dependencies. Cluster commands and data paths are described in the linked guides.

Inspect the available commands before supplying your own data and output paths:

```bash
python train_mesh_vae.py --help
python run_body_measurements.py --help
python run_body_measurements_pointcloud.py --help
```

Datasets, participant metadata, trained checkpoints, installed environments, and
credentials stay outside version control. Configure their local or cluster paths
for your own run. Clear saved outputs from notebooks before committing them.

## Guides

| Workflow | Guide |
| --- | --- |
| Model layout and conditioning | [MeshVAE refactor](docs/README_REFACTOR.md) |
| Scan preprocessing | [Combined scan data](docs/README_COMBINED_SCAN_DATA.md) |
| Physical scan scale and target handling | [Scan geometry fix](docs/README_SCAN_GEOMETRY_FIX.md) |
| Measurement baselines and point clouds | [Body measurements](docs/README_BODY_MEASUREMENTS.md) |
| Clinical features | [Clinical conditioning](docs/README_CLINICAL_FEATURES.md) |
| Model comparison | [Comparison runners](docs/README_MODEL_COMPARISON.md) |
| Supervised diagnosis | [Diagnostic workflow](docs/README_SUPERVISED_DIAGNOSTIC.md) |
| HbA1c prediction | [HbA1c regression](docs/README_HBA1C_REGRESSION.md) |
| Plaque-volume prediction | [Plaque-volume regression](docs/README_PLAQUE_VOLUME_REGRESSION.md) |
| Cardiovascular targets | [Target benchmark](docs/README_CARDIOVASCULAR_TARGET_BENCHMARK.md) |
| Paper analysis and cluster workflows | [Paper-analysis overview](paper_analysis/README.md) |

The guides' Python commands are run from the repository root, even though the
guides live in `docs/`. General cluster shell scripts are grouped under
[`scripts/computerome/`](scripts/computerome/) and
[`scripts/gefion/`](scripts/gefion/). Paper-specific launchers remain in
`paper_analysis/` alongside the study configuration and analysis code.

## Checks

With the required dependencies available, run the measurement tests from the
repository root:

```bash
python -m unittest discover -s body_measurements/tests -v
```

For the paper-analysis metric and calibration checks:

```bash
python -m unittest paper_analysis.tests.test_bootstrap_metrics paper_analysis.tests.test_calibration -v
```

These checks use synthetic fixtures rather than participant datasets. Model
training and cluster jobs require their separate data and runtime configuration.
