#!/usr/bin/env python3
"""Run the complete body-scan experiment matrix and maintain one result table."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_full_meshvae_variants import VARIANTS


EXPERIMENTS = [
    {"key": "vae_scan_age", "model": "VAE", "age": "Yes", "pointcloud": "Yes", "modification": "", "architecture": "Point-set MeshVAE"},
    {"key": "vae_scan_noage", "model": "VAE", "age": "No", "pointcloud": "Yes", "modification": "", "architecture": "Point-set MeshVAE"},
    {"key": "vae_age_only", "model": "VAE", "age": "Yes", "pointcloud": "No", "modification": "", "architecture": "FeatureVAEClassifier"},
    {"key": "vae_conv_scan_age", "model": "VAE", "age": "Yes", "pointcloud": "Yes", "modification": "Conv", "architecture": "Local-kNN MeshVAE"},
    {"key": "vae_conv_scan_noage", "model": "VAE", "age": "No", "pointcloud": "Yes", "modification": "Conv", "architecture": "Local-kNN MeshVAE"},
    {"key": "vae_conv_age_only", "model": "VAE", "age": "Yes", "pointcloud": "No", "modification": "Conv", "architecture": "ConvFeatureVAEClassifier"},
    {"key": "vae_pointlabels_scan_age", "model": "VAE", "age": "Yes", "pointcloud": "Yes", "modification": "Point Label", "architecture": "Anatomical-region MeshVAE"},
    {"key": "vae_pointlabels_scan_noage", "model": "VAE", "age": "No", "pointcloud": "Yes", "modification": "Point Label", "architecture": "Anatomical-region MeshVAE"},
    {"key": "vae_pointlabels_age_only", "model": "VAE", "age": "Yes", "pointcloud": "No", "modification": "Point Label", "architecture": "Not applicable: no points to label"},
    {"key": "logistic_scan_age", "model": "Logistic", "age": "Yes", "pointcloud": "Yes", "modification": "", "architecture": "LinearClassifier"},
    {"key": "logistic_scan_noage", "model": "Logistic", "age": "No", "pointcloud": "Yes", "modification": "", "architecture": "LinearClassifier"},
    {"key": "logistic_age_only", "model": "Logistic", "age": "Yes", "pointcloud": "No", "modification": "", "architecture": "LinearClassifier"},
    {"key": "convolution_scan_age", "model": "Convolution", "age": "Yes", "pointcloud": "Yes", "modification": "", "architecture": "PointNet classifier"},
    {"key": "convolution_scan_noage", "model": "Convolution", "age": "No", "pointcloud": "Yes", "modification": "", "architecture": "PointNet classifier"},
    {"key": "convolution_age_only", "model": "Convolution", "age": "Yes", "pointcloud": "No", "modification": "", "architecture": "1D age-feature CNN"},
]

FULL_VARIANT_TO_EXPERIMENT = {
    "normal_scan_age": "vae_scan_age",
    "normal_scan_noage": "vae_scan_noage",
    "conv_scan_age": "vae_conv_scan_age",
    "conv_scan_noage": "vae_conv_scan_noage",
    "pointlabels_scan_age": "vae_pointlabels_scan_age",
    "pointlabels_scan_noage": "vae_pointlabels_scan_noage",
}

COMPARISON_KEYS = [
    "vae_age_only",
    "vae_conv_age_only",
    "vae_pointlabels_age_only",
    "logistic_scan_age",
    "logistic_scan_noage",
    "logistic_age_only",
    "convolution_scan_age",
    "convolution_scan_noage",
    "convolution_age_only",
]

OUTPUT_COLUMNS = [
    "Run",
    "Experiment",
    "Model",
    "Age",
    "Pointcloud",
    "Modification",
    "EffectiveInput",
    "Architecture",
    "Status",
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "BalancedAccuracy",
    "AUC",
    "Precision",
    "F1",
    "TP",
    "TN",
    "FP",
    "FN",
    "Threshold",
    "BestEpoch",
    "NTrain",
    "NTest",
    "Checkpoint",
    "ResultsDir",
    "Error",
]


def print_command(cmd: list[str]) -> None:
    print(" ".join(shlex.quote(str(x)) for x in cmd), flush=True)


def run_command(cmd: list[str], dry_run: bool) -> None:
    print_command(cmd)
    if not dry_run:
        subprocess.run(cmd, check=True)


def read_first_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return dict(rows[0]) if rows else {}


def comparison_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = {str(row.get("Experiment", "")): row for row in csv.DictReader(handle)}
    return all(
        key in rows and str(rows[key].get("Status", "")) in {"completed", "not_applicable"}
        for key in COMPARISON_KEYS
    )


def latest_diagnostic(run_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(
        (run_dir / "diagnostics").glob("*/summary_row.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    csv_path = candidates[0]
    row = read_first_csv(csv_path)
    json_path = csv_path.parent / "summary.json"
    if json_path.exists():
        summary = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        test = summary.get("test", {}) or {}
        calibration = summary.get("calibration", {}) or {}
        row.setdefault("test_accuracy", test.get("accuracy", ""))
        row.setdefault("test_tp", test.get("tp", ""))
        row.setdefault("test_tn", test.get("tn", ""))
        row.setdefault("test_fp", test.get("fp", ""))
        row.setdefault("test_fn", test.get("fn", ""))
        row.setdefault("test_n", test.get("n_scored", ""))
        row.setdefault("calib_n", calibration.get("n_scored", ""))
    row["summary_file"] = str(csv_path)
    return row


def blank_row(index: int, spec: dict[str, str]) -> dict[str, Any]:
    input_parts = []
    if spec["pointcloud"] == "Yes":
        input_parts.append("body scan")
    if spec["age"] == "Yes":
        input_parts.append("age")
    return {
        "Run": index,
        "Experiment": spec["key"],
        "Model": spec["model"],
        "Age": spec["age"],
        "Pointcloud": spec["pointcloud"],
        "Modification": spec["modification"],
        "EffectiveInput": " + ".join(input_parts),
        "Architecture": spec["architecture"],
        "Status": "pending",
    }


def value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and str(row[name]).strip() not in {"", "nan", "None"}:
            return row[name]
    return ""


def merge_full_result(out: dict[str, Any], source: dict[str, Any], checkpoint: Path) -> None:
    training_summary_path = checkpoint.parent / "training_summary.json"
    training_summary = {}
    if training_summary_path.exists():
        training_summary = json.loads(training_summary_path.read_text(encoding="utf-8", errors="replace"))
    out.update({
        "Status": "completed",
        "Accuracy": value(source, "test_accuracy"),
        "Sensitivity": value(source, "test_sensitivity"),
        "Specificity": value(source, "test_specificity"),
        "BalancedAccuracy": value(source, "test_balanced_accuracy"),
        "AUC": value(source, "test_auc"),
        "Precision": value(source, "test_precision"),
        "F1": value(source, "test_f1"),
        "TP": value(source, "test_tp"),
        "TN": value(source, "test_tn"),
        "FP": value(source, "test_fp"),
        "FN": value(source, "test_fn"),
        "Threshold": value(source, "chosen_thresholds", "chosen_threshold"),
        "BestEpoch": training_summary.get("best_epoch", ""),
        "NTrain": value(source, "calib_n"),
        "NTest": value(source, "test_n"),
        "Checkpoint": str(checkpoint),
        "ResultsDir": value(source, "results_dir"),
    })


def merge_comparison_result(out: dict[str, Any], source: dict[str, Any]) -> None:
    for column in [
        "EffectiveInput", "Architecture", "Status", "Accuracy", "Sensitivity",
        "Specificity", "BalancedAccuracy", "AUC", "Precision", "F1", "TP",
        "TN", "FP", "FN", "Threshold", "BestEpoch", "NTrain", "NTest",
        "Checkpoint", "Error",
    ]:
        if column in source:
            out[column] = source[column]


def collect_rows(args, failures: dict[str, str]) -> list[dict[str, Any]]:
    checkpoints = Path(args.checkpoints_dir)
    component_csv = checkpoints / args.name_prefix / "comparison_components" / "model_comparison_results.csv"
    component_rows: dict[str, dict[str, Any]] = {}
    if component_csv.exists():
        with component_csv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                component_rows[str(row.get("Experiment", ""))] = dict(row)

    suffix_by_key = {
        experiment_key: str(VARIANTS[variant_key]["suffix"])
        for variant_key, experiment_key in FULL_VARIANT_TO_EXPERIMENT.items()
    }

    rows = []
    for index, spec in enumerate(EXPERIMENTS, start=1):
        out = blank_row(index, spec)
        key = spec["key"]
        if key in suffix_by_key:
            run_dir = checkpoints / f"{args.name_prefix}_{suffix_by_key[key]}"
            checkpoint = run_dir / "best_vae_net.pth"
            diagnostic = latest_diagnostic(run_dir)
            if diagnostic is not None:
                merge_full_result(out, diagnostic, checkpoint)
            elif checkpoint.exists():
                out.update({"Status": "trained_not_diagnosed", "Checkpoint": str(checkpoint)})
        elif key in component_rows:
            merge_comparison_result(out, component_rows[key])
        if key in failures and out.get("Status") != "completed":
            out["Status"] = "failed"
            out["Error"] = failures[key]
        rows.append(out)
    return rows


def display_metric(raw: Any) -> str:
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return ""
    return f"{number:.4f}" if math.isfinite(number) else ""


def write_results(rows: list[dict[str, Any]], args, final: bool) -> None:
    out_dir = Path(args.checkpoints_dir) / args.name_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "model_comparison_results" if final else "model_comparison_results_partial"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 15-Model Body-Scan Comparison",
        "",
        f"Train split: `{args.train_cond_tsv}`",
        f"Test split: `{args.test_cond_tsv}`",
        f"Scan data: `{args.dataroot}`",
        "",
        "| Run | Model | Age | Point cloud | Modification | Status | Accuracy | Sensitivity | Specificity | Balanced accuracy | AUC | F1 |",
        "|---:|---|:---:|:---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {Run} | {Model} | {Age} | {Pointcloud} | {Modification} | {Status} | {Accuracy} | {Sensitivity} | {Specificity} | {BalancedAccuracy} | {AUC} | {F1} |".format(
                **{
                    **row,
                    "Accuracy": display_metric(row.get("Accuracy")),
                    "Sensitivity": display_metric(row.get("Sensitivity")),
                    "Specificity": display_metric(row.get("Specificity")),
                    "BalancedAccuracy": display_metric(row.get("BalancedAccuracy")),
                    "AUC": display_metric(row.get("AUC")),
                    "F1": display_metric(row.get("F1")),
                }
            )
        )
    lines.extend([
        "",
        "The age-only point-label row is retained from the requested design as not applicable: without body-scan points there are no body regions to label.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    completed = sum(str(row.get("Status")) == "completed" for row in rows)
    print(f"[TABLE] {completed}/14 trainable experiments completed; wrote {csv_path}", flush=True)
    print(f"[TABLE] wrote {md_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_prefix", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--train_cond_tsv", required=True)
    parser.add_argument("--test_cond_tsv", required=True)
    parser.add_argument("--checkpoints_dir", default="/data/projects/3D_scans/ML/MQVAE/checkpoints")
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
    parser.add_argument("--target_missing_policy", choices=["drop", "zero", "error"], default="drop")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_full_vae", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    failures: dict[str, str] = {}
    Path(args.checkpoints_dir, args.name_prefix).mkdir(parents=True, exist_ok=True)
    write_results(collect_rows(args, failures), args, final=False)

    if not args.skip_baselines:
        component_dir = Path(args.checkpoints_dir) / args.name_prefix / "comparison_components"
        component_csv = component_dir / "model_comparison_results.csv"
        if args.resume and comparison_is_complete(component_csv):
            print(f"[RESUME] comparison components already exist: {component_csv}", flush=True)
        else:
            command = [
                sys.executable,
                str(root / "run_model_comparison.py"),
                "--name", f"{args.name_prefix}_components",
                "--dataroot", args.dataroot,
                "--train_cond_tsv", args.train_cond_tsv,
                "--test_cond_tsv", args.test_cond_tsv,
                "--feature_source", "scan",
                "--only_gender", args.only_gender,
                "--gpu_ids", args.gpu_ids,
                "--out_dir", str(component_dir),
                "--n_points", str(args.baseline_n_points),
                "--batch_size_point", str(args.baseline_batch_size),
                "--epochs", str(args.baseline_epochs),
                "--patience", "25",
                "--target_missing_policy", args.target_missing_policy,
                "--experiments", *COMPARISON_KEYS,
                "--resume",
            ]
            if args.use_scan_geometry_features:
                command.extend(["--use_scan_geometry_features", "--scan_geometry_features_file", args.scan_geometry_features_file])
            else:
                command.append("--no-use_scan_geometry_features")
            if args.require_scan_geometry_features:
                command.append("--require_scan_geometry_features")
            if args.restore_metric_scale:
                command.append("--restore_metric_scale")
            else:
                command.append("--no-restore_metric_scale")
            try:
                run_command(command, args.dry_run)
            except subprocess.CalledProcessError as exc:
                message = f"comparison components exited with status {exc.returncode}"
                print(f"[ERROR] {message}", flush=True)
                for key in COMPARISON_KEYS:
                    failures[key] = message
                if not args.continue_on_error:
                    raise
        write_results(collect_rows(args, failures), args, final=False)

    if not args.skip_full_vae:
        for run_number, (variant, experiment_key) in enumerate(FULL_VARIANT_TO_EXPERIMENT.items(), start=1):
            suffix = str(VARIANTS[variant]["suffix"])
            run_dir = Path(args.checkpoints_dir) / f"{args.name_prefix}_{suffix}"
            diagnostic = latest_diagnostic(run_dir)
            if args.resume and diagnostic is not None:
                print(f"[RESUME] full VAE {run_number}/6 already diagnosed: {experiment_key}", flush=True)
                continue

            command = [
                sys.executable,
                str(root / "run_full_meshvae_variants.py"),
                "--name_prefix", args.name_prefix,
                "--dataroot", args.dataroot,
                "--checkpoints_dir", args.checkpoints_dir,
                "--train_cond_tsv", args.train_cond_tsv,
                "--test_cond_tsv", args.test_cond_tsv,
                "--only_gender", args.only_gender,
                "--gpu_ids", args.gpu_ids,
                "--niter", str(args.niter),
                "--batch_size", str(args.batch_size),
                "--target_missing_policy", args.target_missing_policy,
                "--scan_geometry_features_file", args.scan_geometry_features_file,
                "--variants", variant,
            ]
            if args.use_scan_geometry_features:
                command.append("--use_scan_geometry_features")
            else:
                command.append("--no-use_scan_geometry_features")
            if args.require_scan_geometry_features:
                command.append("--require_scan_geometry_features")
            else:
                command.append("--no-require_scan_geometry_features")
            if args.restore_metric_scale:
                command.append("--restore_metric_scale")
            else:
                command.append("--no-restore_metric_scale")
            if (
                args.resume
                and (run_dir / "best_vae_net.pth").exists()
                and (run_dir / "training_summary.json").exists()
            ):
                command.append("--skip_train")

            print(f"\n[MASTER] full VAE {run_number}/6: {experiment_key}", flush=True)
            try:
                run_command(command, args.dry_run)
            except subprocess.CalledProcessError as exc:
                failures[experiment_key] = f"full VAE exited with status {exc.returncode}"
                print(f"[ERROR] {experiment_key}: {failures[experiment_key]}", flush=True)
                if not args.continue_on_error:
                    raise
            write_results(collect_rows(args, failures), args, final=False)

    rows = collect_rows(args, failures)
    write_results(rows, args, final=True)
    print("\n========== FINAL 15-MODEL STATUS ==========", flush=True)
    for row in rows:
        print(
            f"{int(row['Run']):02d}. {row['Experiment']:<32} {row.get('Status', ''):<22} "
            f"bal_acc={display_metric(row.get('BalancedAccuracy')):<6} auc={display_metric(row.get('AUC')):<6}",
            flush=True,
        )


if __name__ == "__main__":
    main()
