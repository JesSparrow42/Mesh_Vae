# 🧭 Command & launcher index

Python entry points are grouped by workflow. Run module commands from the
repository root, or use the installed console commands listed in the
[main README](../README.md).

| Folder | Purpose | Example |
| --- | --- | --- |
| `training/` | VAE training and measurement-cohort expansion | `python -m scripts.training.train_mesh_vae --help` |
| `measurements/` | Measurement audits, baselines, and point clouds | `python -m scripts.measurements.run_body_measurements --help` |
| `benchmarks/` | Model comparisons and clinical-target experiments | `python -m scripts.benchmarks.run_all_15_models --help` |
| `diagnostics/` | Model scoring, checks, and result collection | `python -m scripts.diagnostics.diagnose_mesh_vae --help` |
| `preprocessing/` | Scan geometry and metadata inspection | `python -m scripts.preprocessing.build_scan_geometry_features --help` |
| `analysis/` | Metadata-predictor screening | `python -m scripts.analysis.screen_metadata_predictors --help` |
| `computerome/` | Computerome setup and shell launchers | See the [measurement guide](../docs/README_BODY_MEASUREMENTS.md). |
| `gefion/` | Gefion preprocessing and training launchers | `bash scripts/gefion/gefion_preprocess.sh` |

Cluster launchers retain the project paths, module loads, environments, and compute
requirements configured for their server. Read the launcher and its workflow
guide before running a cluster job.

Paper-specific launchers remain beside their study configuration in
[`paper_analysis/`](../paper_analysis/README.md). Copy-and-paste command references
are in [`docs/commands/`](../docs/commands/).
