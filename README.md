<div align="center">

# 🧬 MeshVAE

**From 3D body scans to learned representations and clinical research.**

Conditional mesh and point-cloud VAEs · body measurements · reproducible study workflows

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Package version](https://img.shields.io/badge/package-v0.1.0-0EA5E9)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](meshvae/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](#-development)
[![Last commit](https://img.shields.io/github/last-commit/JesSparrow42/Mesh_Vae?color=8B5CF6)](https://github.com/JesSparrow42/Mesh_Vae/commits/main)

[Quick start](#-quick-start) · [Commands](#-commands) · [Project map](#-project-map) · [Documentation](#-documentation)

</div>

## ✨ What’s inside

- **Mesh & point-cloud VAEs** with conditional age/gender inputs, reconstruction losses, and an optional adversarial age objective.
- **Gaussian & Boson priors** with a vendored photonic sampler and experiment runners for comparing model variants.
- **Body-measurement models** using physical scan features, baseline regressors, and local point-cloud learning.
- **Clinical research workflows** for plaque and HbA1c targets, held-out evaluation, calibration, and confidence intervals.
- **Cluster-ready workflows** for Computerome and Gefion, plus analysis notebooks and paper figures.

## 🚀 Quick start

```bash
git clone https://github.com/JesSparrow42/Mesh_Vae.git
cd Mesh_Vae

python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

meshvae-measurements --help
```

Add the model and analysis dependencies when you need them:

```bash
python -m pip install -e ".[vae,measurement,analysis,dev]"
```

Choose a PyTorch build that matches your CPU or CUDA environment. The cluster
scripts document the separate module stacks and paths used on each server.

| Install group | Adds support for |
| --- | --- |
| Base install (`.`) | NumPy, pandas, SciPy, scikit-learn, and joblib for scan features and tabular baselines |
| `measurement` | PyTorch for point-cloud measurement models |
| `vae` | Mesh/point-cloud training, geometry, and model diagnostics |
| `analysis` | Study analysis, plots, and reporting |
| `dev` | Running the test suite |

## 🧭 Commands

Run module commands from the repository root. Installed console commands are
available after the editable install above.

| Task | Console command | Module command |
| --- | --- | --- |
| Train, preprocess, test, or sample a VAE | `meshvae-train` | `python -m scripts.training.train_mesh_vae` |
| Audit labels and train measurement baselines | `meshvae-measurements` | `python -m scripts.measurements.run_body_measurements` |
| Build caches and train point-cloud regressors | `meshvae-pointcloud` | `python -m scripts.measurements.run_body_measurements_pointcloud` |
| Run VAE diagnostics | `meshvae-diagnose` | `python -m scripts.diagnostics.diagnose_mesh_vae` |
| Run the plaque-study workflow | `meshvae-plaque-study` | `python -m paper_analysis.run_plaque_study` |

Add `--help` to inspect a command’s arguments. Dataset paths, metadata columns,
target definitions, and checkpoint locations are supplied by your experiment.

For example, inspect the training options or the full comparison runner:

```bash
python -m scripts.training.train_mesh_vae --help
python -m scripts.benchmarks.run_all_15_models --help
```

## 📦 Package & environment

The project package is **`meshvae-research`**, version **`0.1.0`**, with
**Python ≥ 3.11**. Package metadata and dependency groups live in
[`pyproject.toml`](pyproject.toml). This is the repository’s package version;
installation is from the checkout.

The following versions identify the checked Windows CPU environment, rather than
a lockfile for every experiment. Cluster CUDA environments use their own stacks.

| Component | Reference version | Role |
| --- | --- | --- |
| Python | 3.12.7 | Runtime |
| PyTorch | 2.6.0+cpu | Neural models |
| NumPy / pandas | 1.26.4 / 2.2.2 | Arrays and cohort tables |
| SciPy / scikit-learn | 1.13.1 / 1.5.1 | Statistics, baselines, and metrics |
| trimesh | 4.6.8 | Mesh processing |
| pytest | 7.4.4 | Tests |
| Vendored `bosonsampler` | 0.1.0 | Photonic sampling; [source attribution](third_party/nonlinear_photonic_sampler/UPSTREAM.md) |

## 🗂️ Project map

```text
Mesh_Vae/
├── meshvae/                VAE models, geometry, losses, and conditioning
├── body_measurements/      Measurement models, target configs, and tests
├── scripts/
│   ├── training/           Training entry points and expanded-cohort workflow
│   ├── measurements/       Measurement baselines and point-cloud commands
│   ├── benchmarks/         Model comparisons and clinical-target experiments
│   ├── diagnostics/        Scoring, checks, and result collection
│   ├── preprocessing/      Scan features and metadata inspection
│   ├── analysis/           Metadata-predictor analysis
│   ├── computerome/        Computerome shell launchers and setup scripts
│   └── gefion/             Gefion preprocessing and training launchers
├── paper_analysis/         Study configuration, evaluation, figures, and notebooks
├── notebooks/              Current exploratory analysis
├── docs/                   Workflow guides and command references
├── tools/                  Installer builders and delivery helpers
├── third_party/            Vendored photonic sampler
├── outputs/                Delivery helpers and saved result manifests
├── archive/                Earlier implementations and exploratory work
└── pyproject.toml          Package metadata, extras, and command entry points
```

## 📚 Documentation

Start with the [workflow index](docs/README.md), the
[body-measurement guide](docs/README_BODY_MEASUREMENTS.md), or the
[paper-analysis overview](paper_analysis/README.md).

<details>
<summary><strong>Browse all workflow guides</strong></summary>

| Topic | Guide |
| --- | --- |
| Model structure and conditioning | [MeshVAE architecture](docs/README_REFACTOR.md) |
| Scan preprocessing and physical scale | [Combined scans](docs/README_COMBINED_SCAN_DATA.md) · [Geometry features](docs/README_SCAN_GEOMETRY_FIX.md) |
| Body-measurement prediction | [Baselines, point clouds, and cohort expansion](docs/README_BODY_MEASUREMENTS.md) |
| Clinical inputs | [Conditioning features](docs/README_CLINICAL_FEATURES.md) |
| Model comparisons | [Comparison runners](docs/README_MODEL_COMPARISON.md) |
| Continuous targets | [HbA1c](docs/README_HBA1C_REGRESSION.md) · [Plaque volume](docs/README_PLAQUE_VOLUME_REGRESSION.md) |
| Cardiovascular target study | [Benchmark guide](docs/README_CARDIOVASCULAR_TARGET_BENCHMARK.md) |
| Cluster launchers | [Scripts index](scripts/README.md) |
| Earlier implementations | [Archive index](archive/README.md) |

</details>

## 🧪 Development

```bash
python -m pip install -e ".[vae,measurement,analysis,dev]"
python -m pytest body_measurements/tests paper_analysis
```

Tests use synthetic fixtures. Dataset-dependent training and cluster jobs are
configured separately in the workflow guides. Keep datasets, model checkpoints,
and installed environments outside Git; clear notebook outputs before committing.

For changes, work in the active packages and categorized scripts, run the relevant
tests, and update the matching guide. Archived snapshots retain their original
layout for reference.
