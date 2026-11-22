#!/usr/bin/env python3
"""Run the complete 15-row HbA1c regression experiment and write one table."""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_full_meshvae_variants import VARIANTS


EXPERIMENTS = [
    ("vae_scan_age", "VAE", "Yes", "Yes", "", "Point-set MeshVAE"),
    ("vae_scan_noage", "VAE", "No", "Yes", "", "Point-set MeshVAE"),
    ("vae_age_only", "VAE", "Yes", "No", "", "Feature VAE"),
    ("vae_conv_scan_age", "VAE", "Yes", "Yes", "Conv", "Local-kNN MeshVAE"),
    ("vae_conv_scan_noage", "VAE", "No", "Yes", "Conv", "Local-kNN MeshVAE"),
    ("vae_conv_age_only", "VAE", "Yes", "No", "Conv", "Convolutional feature VAE"),
    ("vae_pointlabels_scan_age", "VAE", "Yes", "Yes", "Point Label", "Anatomical-region MeshVAE"),
    ("vae_pointlabels_scan_noage", "VAE", "No", "Yes", "Point Label", "Anatomical-region MeshVAE"),
    ("vae_pointlabels_age_only", "VAE", "Yes", "No", "Point Label", "Not applicable"),
    ("logistic_scan_age", "Linear regression", "Yes", "Yes", "", "Linear regression"),
    ("logistic_scan_noage", "Linear regression", "No", "Yes", "", "Linear regression"),
    ("logistic_age_only", "Linear regression", "Yes", "No", "", "Linear regression"),
    ("convolution_scan_age", "Convolution", "Yes", "Yes", "", "PointNet regressor"),
    ("convolution_scan_noage", "Convolution", "No", "Yes", "", "PointNet regressor"),
    ("convolution_age_only", "Convolution", "Yes", "No", "", "1D age-feature CNN"),
]

FULL_VARIANTS = {
    "normal_scan_age": "vae_scan_age",
    "normal_scan_noage": "vae_scan_noage",
    "conv_scan_age": "vae_conv_scan_age",
    "conv_scan_noage": "vae_conv_scan_noage",
    "pointlabels_scan_age": "vae_pointlabels_scan_age",
    "pointlabels_scan_noage": "vae_pointlabels_scan_noage",
}
COMPONENT_KEYS = [
    "vae_age_only", "vae_conv_age_only", "vae_pointlabels_age_only",
    "logistic_scan_age", "logistic_scan_noage", "logistic_age_only",
    "convolution_scan_age", "convolution_scan_noage", "convolution_age_only",
]


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(shlex.quote(str(value)) for value in command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return {str(row.get("Experiment", "")): dict(row) for row in csv.DictReader(handle)}


def collect(args, failures: dict[str, str]) -> list[dict[str, Any]]:
    base = Path(args.checkpoints_dir)
    component_dir = base / args.name_prefix / "hba1c_components"
    component_path = component_dir / "regression_comparison_results.csv"
    if not component_path.exists():
        component_path = component_dir / "regression_comparison_results_partial.csv"
    components = read_rows(component_path)
    suffix_to_key = {VARIANTS[variant]["suffix"]: key for variant, key in FULL_VARIANTS.items()}
    rows = []
    for index, (key, model, age, scan, modification, architecture) in enumerate(EXPERIMENTS, start=1):
        row: dict[str, Any] = {
            "Run": index, "Experiment": key, "Model": model, "Age": age, "Pointcloud": scan,
            "Modification": modification, "Architecture": architecture, "Status": "pending",
        }
        if key in components:
            row.update(components[key])
            row["Run"] = index
        else:
            suffix = next((suffix for suffix, mapped in suffix_to_key.items() if mapped == key), None)
            if suffix is not None:
                run_dir = base / f"{args.name_prefix}_{suffix}"
                summary = read_rows(run_dir / "regression_summary.csv")
                if summary:
                    source = next(iter(summary.values()))
                    row.update({column: source.get(column, "") for column in ["Status", "N", "R2", "MAE", "RMSE", "Pearson", "Spearman", "ResultsDir"]})
                    row["NTest"] = source.get("N", "")
                    row["Checkpoint"] = str(run_dir / "best_vae_net.pth")
                    training_summary = run_dir / "training_summary.json"
                    if training_summary.exists():
                        training = json.loads(training_summary.read_text(encoding="utf-8", errors="replace"))
                        row["BestEpoch"] = training.get("best_epoch", "")
                        row["NTrain"] = training.get("train_samples", "")
                elif (run_dir / "best_vae_net.pth").exists():
                    row.update({"Status": "trained_not_diagnosed", "Checkpoint": str(run_dir / "best_vae_net.pth")})
        if key in failures and row.get("Status") not in {"completed", "not_applicable"}:
            row["Status"] = "failed"
            row["Error"] = failures[key]
        rows.append(row)
    return rows


def write_table(rows, args, final):
    out_dir = Path(args.checkpoints_dir) / args.name_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "hba1c_model_comparison_results" if final else "hba1c_model_comparison_results_partial"
    frame = __import__("pandas").DataFrame(rows)
    frame.to_csv(out_dir / f"{stem}.csv", index=False)
    columns = ["Run", "Model", "Age", "Pointcloud", "Modification", "Status", "R2", "MAE", "RMSE", "Pearson", "Spearman"]
    display = frame.reindex(columns=columns)
    lines = [
        "# 15-Model HbA1c Regression Comparison", "",
        f"Target: `{args.target_col}`", f"Train split: `{args.train_cond_tsv}`",
        f"Test split: `{args.test_cond_tsv}`", "", "```text", display.to_string(index=False), "```", "",
        "The point-label age-only row is not applicable because there are no scan points to label.",
        "For this continuous outcome, the former logistic comparison is correctly implemented as linear regression.", "",
    ]
    (out_dir / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[TABLE] wrote {out_dir / f'{stem}.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_prefix", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--train_cond_tsv", required=True)
    parser.add_argument("--test_cond_tsv", required=True)
    parser.add_argument("--checkpoints_dir", default="/data/projects/3D_scans/ML/MQVAE/checkpoints")
    parser.add_argument("--target_col", default="glycohemoglobin_hba1c")
    parser.add_argument("--only_gender", choices=["M", "F"], default="F")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--niter", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--baseline_epochs", type=int, default=100)
    parser.add_argument("--baseline_n_points", type=int, default=4096)
    parser.add_argument("--baseline_batch_size", type=int, default=8)
    parser.add_argument("--use_scan_geometry_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scan_geometry_features_file", default="scan_geometry_features.npz")
    parser.add_argument("--require_scan_geometry_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restore_metric_scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_components", action="store_true")
    parser.add_argument("--skip_full_vae", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    failures: dict[str, str] = {}
    write_table(collect(args, failures), args, final=False)

    geometry_args = []
    if args.use_scan_geometry_features:
        geometry_args.extend(["--use_scan_geometry_features", "--scan_geometry_features_file", args.scan_geometry_features_file])
    else:
        geometry_args.append("--no-use_scan_geometry_features")
    if args.require_scan_geometry_features:
        geometry_args.append("--require_scan_geometry_features")
    if args.restore_metric_scale:
        geometry_args.append("--restore_metric_scale")
    else:
        geometry_args.append("--no-restore_metric_scale")

    if not args.skip_components:
        command = [
            sys.executable, str(root / "run_hba1c_model_comparison.py"),
            "--name", f"{args.name_prefix}_components", "--dataroot", args.dataroot,
            "--train_cond_tsv", args.train_cond_tsv, "--test_cond_tsv", args.test_cond_tsv,
            "--out_dir", str(Path(args.checkpoints_dir) / args.name_prefix / "hba1c_components"),
            "--target_col", args.target_col, "--only_gender", args.only_gender, "--gpu_ids", args.gpu_ids,
            "--n_points", str(args.baseline_n_points), "--batch_size_point", str(args.baseline_batch_size),
            "--epochs", str(args.baseline_epochs), "--experiments", *COMPONENT_KEYS, "--resume", *geometry_args,
        ]
        try:
            run(command, args.dry_run)
        except subprocess.CalledProcessError as exc:
            for key in COMPONENT_KEYS:
                failures[key] = str(exc)
            if not args.continue_on_error:
                raise
        write_table(collect(args, failures), args, final=False)

    if not args.skip_full_vae:
        for variant, experiment_key in FULL_VARIANTS.items():
            command = [
                sys.executable, str(root / "run_full_meshvae_regression_variants.py"),
                "--name_prefix", args.name_prefix, "--dataroot", args.dataroot,
                "--checkpoints_dir", args.checkpoints_dir,
                "--train_cond_tsv", args.train_cond_tsv, "--test_cond_tsv", args.test_cond_tsv,
                "--target_col", args.target_col, "--only_gender", args.only_gender, "--gpu_ids", args.gpu_ids,
                "--niter", str(args.niter), "--batch_size", str(args.batch_size),
                "--variants", variant, "--resume", *geometry_args,
            ]
            try:
                run(command, args.dry_run)
            except subprocess.CalledProcessError as exc:
                failures[experiment_key] = str(exc)
                if not args.continue_on_error:
                    raise
            write_table(collect(args, failures), args, final=False)
    write_table(collect(args, failures), args, final=not args.dry_run)


if __name__ == "__main__":
    main()
