"""Repository paths shared by organized command entry points."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Basenames are retained so launchers can resolve an entry point without
# depending on the folder of the launcher itself.
SCRIPT_PATHS = {
    'benchmark_hba1c_body_size.py': 'scripts/benchmarks/benchmark_hba1c_body_size.py',
    'build_scan_geometry_features.py': 'scripts/preprocessing/build_scan_geometry_features.py',
    'check_boson_prior.py': 'scripts/diagnostics/check_boson_prior.py',
    'collect_diagnostic_results.py': 'scripts/diagnostics/collect_diagnostic_results.py',
    'diagnose_mesh_binary.py': 'scripts/diagnostics/diagnose_mesh_binary.py',
    'diagnose_mesh_regression.py': 'scripts/diagnostics/diagnose_mesh_regression.py',
    'diagnose_mesh_vae.py': 'scripts/diagnostics/diagnose_mesh_vae.py',
    'find_metadata_tsv_for_obj_folder.py': 'scripts/preprocessing/find_metadata_tsv_for_obj_folder.py',
    'inspect_npz_tsv_overlap.py': 'scripts/preprocessing/inspect_npz_tsv_overlap.py',
    'inspect_scan_metadata_overlap.py': 'scripts/preprocessing/inspect_scan_metadata_overlap.py',
    'linear_regression_hba1c.py': 'scripts/benchmarks/linear_regression_hba1c.py',
    'mesh_geom_pipeline.py': 'meshvae/compat/mesh_geom_pipeline.py',
    'mesh_vae.py': 'meshvae/compat/mesh_vae.py',
    'run_all_15_hba1c_models.py': 'scripts/benchmarks/run_all_15_hba1c_models.py',
    'run_all_15_models.py': 'scripts/benchmarks/run_all_15_models.py',
    'run_body_measurements.py': 'scripts/measurements/run_body_measurements.py',
    'run_body_measurements_pointcloud.py': 'scripts/measurements/run_body_measurements_pointcloud.py',
    'run_boson_prior_ablation.py': 'scripts/benchmarks/run_boson_prior_ablation.py',
    'run_cardiovascular_target_benchmark.py': 'scripts/benchmarks/run_cardiovascular_target_benchmark.py',
    'run_full_meshvae_regression_variants.py': 'scripts/benchmarks/run_full_meshvae_regression_variants.py',
    'run_full_meshvae_variants.py': 'scripts/benchmarks/run_full_meshvae_variants.py',
    'run_hba1c_model_comparison.py': 'scripts/benchmarks/run_hba1c_model_comparison.py',
    'run_model_comparison.py': 'scripts/benchmarks/run_model_comparison.py',
    'screen_metadata_predictors.py': 'scripts/analysis/screen_metadata_predictors.py',
    'train_body_measurements_expanded.py': 'scripts/training/train_body_measurements_expanded.py',
    'train_mesh_vae.py': 'scripts/training/train_mesh_vae.py',
}


def script_path(name: str) -> Path:
    """Return the absolute path of a named project entry point."""
    try:
        relative = SCRIPT_PATHS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown project entry point: {name}") from exc
    return REPO_ROOT / relative


def module_name(name: str) -> str:
    """Return the import name used by ``python -m`` for an entry point."""
    return SCRIPT_PATHS[name].removesuffix(".py").replace("/", ".")
