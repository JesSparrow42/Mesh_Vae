# 📚 Workflow guides

Run examples from the repository root. Python entry points are grouped under
`scripts/`; the preferred form is `python -m scripts.<workflow>.<command>`.

| Start here | Guide |
| --- | --- |
| Understand the model package | [Architecture and conditioning](README_REFACTOR.md) |
| Prepare scans | [Combined scan data](README_COMBINED_SCAN_DATA.md) |
| Preserve physical scale | [Scan geometry and targets](README_SCAN_GEOMETRY_FIX.md) |
| Predict body measurements | [Baselines, point clouds, and cohort expansion](README_BODY_MEASUREMENTS.md) |
| Configure clinical inputs | [Clinical features](README_CLINICAL_FEATURES.md) |
| Compare models | [Model comparison](README_MODEL_COMPARISON.md) |
| Predict continuous outcomes | [HbA1c](README_HBA1C_REGRESSION.md) · [Plaque volume](README_PLAQUE_VOLUME_REGRESSION.md) |
| Run cardiovascular benchmarks | [Target benchmark](README_CARDIOVASCULAR_TARGET_BENCHMARK.md) |
| Reproduce the paper workflows | [Paper-analysis overview](../paper_analysis/README.md) |
| Find launchers and commands | [Script index](../scripts/README.md) · [Saved command snippets](commands/) |

The [earlier supervised-diagnostic reference](README_SUPERVISED_DIAGNOSTIC.md)
describes scripts that are not distributed in the current checkout. Original
patch notes and source inventories are preserved in [`history/`](history/).
