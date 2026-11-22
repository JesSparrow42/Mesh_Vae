#!/usr/bin/env python3
"""Run a paired Gaussian-versus-Boson latent-prior ablation."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ARMS = {
    "gaussian_kl": {
        "label": "Gaussian prior + KL",
        "latent_prior": "gaussian",
        "prior_match": "kl",
    },
    "gaussian_mmd": {
        "label": "Gaussian prior + MMD",
        "latent_prior": "gaussian",
        "prior_match": "mmd",
    },
    "boson_mmd": {
        "label": "Sparrow Boson prior + MMD",
        "latent_prior": "boson",
        "prior_match": "mmd",
    },
}

HIGHER_IS_BETTER = ["R2", "Pearson", "Spearman", "TransformedR2"]
LOWER_IS_BETTER = [
    "MAE",
    "RMSE",
    "TransformedMAE",
    "HeldOutChamfer",
    "HeldOutLoss",
    "LatentMMD",
    "TrainingSeconds",
]


def _run(command: list[str], dry_run: bool) -> float:
    print("\n[COMMAND] " + " ".join(shlex.quote(str(value)) for value in command), flush=True)
    if dry_run:
        return 0.0
    started = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - started


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return frame.iloc[0].to_dict() if len(frame) else {}


def _bootstrap_ci(values: np.ndarray, seed: int = 8675309) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _collect_rows(args: argparse.Namespace, failures: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for arm_key in args.arms:
            experiment = f"{args.name_prefix}_{arm_key}_seed{seed}"
            run_dir = Path(args.checkpoints_dir) / experiment
            summary = _read_summary(run_dir / "regression_summary.csv")
            row: dict[str, Any] = {
                "Experiment": experiment,
                "Seed": int(seed),
                "Arm": arm_key,
                "Model": ARMS[arm_key]["label"],
                "LatentPrior": ARMS[arm_key]["latent_prior"],
                "PriorMatch": ARMS[arm_key]["prior_match"],
                "Status": "pending",
            }
            if summary:
                row.update(summary)
                row.update(
                    {
                        "Experiment": experiment,
                        "Seed": int(seed),
                        "Arm": arm_key,
                        "Model": ARMS[arm_key]["label"],
                        "LatentPrior": ARMS[arm_key]["latent_prior"],
                        "PriorMatch": ARMS[arm_key]["prior_match"],
                    }
                )
                runtime_path = run_dir / "benchmark_runtime_seconds.txt"
                if runtime_path.exists():
                    row["TrainingSeconds"] = float(runtime_path.read_text().strip())
            elif (run_dir / "best_vae_net.pth").exists():
                row["Status"] = "trained_not_diagnosed"
            if experiment in failures and row["Status"] != "completed":
                row.update({"Status": "failed", "Error": failures[experiment]})
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    complete = frame[frame["Status"] == "completed"].copy()
    metrics = [metric for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER if metric in complete]
    rows = []
    for arm_key in frame["Arm"].drop_duplicates():
        arm = complete[complete["Arm"] == arm_key]
        row: dict[str, Any] = {
            "Arm": arm_key,
            "Model": ARMS[arm_key]["label"],
            "CompletedSeeds": int(len(arm)),
        }
        for metric in metrics:
            values = pd.to_numeric(arm[metric], errors="coerce")
            row[f"{metric}_Mean"] = float(values.mean())
            row[f"{metric}_SD"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_boson_effect(frame: pd.DataFrame) -> pd.DataFrame:
    complete = frame[frame["Status"] == "completed"].copy()
    gaussian = complete[complete["Arm"] == "gaussian_mmd"].set_index("Seed")
    boson = complete[complete["Arm"] == "boson_mmd"].set_index("Seed")
    paired_seeds = sorted(set(gaussian.index) & set(boson.index))
    rows = []
    for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        if metric not in gaussian or metric not in boson:
            continue
        raw_delta = (
            pd.to_numeric(boson.loc[paired_seeds, metric], errors="coerce").to_numpy()
            - pd.to_numeric(gaussian.loc[paired_seeds, metric], errors="coerce").to_numpy()
        )
        direction = 1.0 if metric in HIGHER_IS_BETTER else -1.0
        improvement = direction * raw_delta
        finite = np.isfinite(improvement)
        improvement = improvement[finite]
        delta_finite = raw_delta[finite]
        low, high = _bootstrap_ci(improvement)
        rows.append(
            {
                "Metric": metric,
                "PairedSeeds": int(len(improvement)),
                "BosonMinusGaussian": float(np.mean(delta_finite)) if len(delta_finite) else float("nan"),
                "MeanDirectionalImprovement": float(np.mean(improvement)) if len(improvement) else float("nan"),
                "Improvement95CI_Low": low,
                "Improvement95CI_High": high,
                "BosonWins": int(np.sum(improvement > 0.0)),
                "Ties": int(np.sum(np.isclose(improvement, 0.0))),
            }
        )
    return pd.DataFrame(rows)


def _write_results(frame: pd.DataFrame, args: argparse.Namespace, final: bool) -> None:
    out_dir = Path(args.checkpoints_dir) / args.name_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if final else "_partial"
    raw_path = out_dir / f"boson_ablation_runs{suffix}.csv"
    aggregate_path = out_dir / f"boson_ablation_aggregate{suffix}.csv"
    paired_path = out_dir / f"boson_ablation_paired_effect{suffix}.csv"
    frame.to_csv(raw_path, index=False)
    aggregate = _aggregate(frame)
    paired = _paired_boson_effect(frame)
    aggregate.to_csv(aggregate_path, index=False)
    paired.to_csv(paired_path, index=False)

    display_columns = [
        "Seed", "Arm", "Status", "TransformedR2", "Spearman",
        "R2", "MAE", "HeldOutChamfer", "LatentMMD", "TrainingSeconds",
    ]
    display = frame.reindex(columns=display_columns)
    lines = [
        "# Boson Prior Ablation",
        "",
        "The causal Boson comparison is `boson_mmd` versus `gaussian_mmd`; both use the same MMD objective.",
        "`gaussian_kl` is the current standard-model reference.",
        "",
        "## Individual runs",
        "",
        "```text",
        display.to_string(index=False),
        "```",
        "",
        "## Arm means",
        "",
        "```text",
        aggregate.to_string(index=False),
        "```",
        "",
        "## Paired Boson effect",
        "",
        "```text",
        paired.to_string(index=False),
        "```",
        "",
        "A positive MeanDirectionalImprovement favors Boson for every metric. "
        "Intervals are paired bootstrap intervals over model seeds and are descriptive when few seeds are used.",
    ]
    report_path = out_dir / f"boson_ablation_report{suffix}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[TABLE] {raw_path}", flush=True)
    print(f"[TABLE] {aggregate_path}", flush=True)
    print(f"[TABLE] {paired_path}", flush=True)
    print(f"[REPORT] {report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name_prefix", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--train_cond_tsv", required=True)
    parser.add_argument("--test_cond_tsv", required=True)
    parser.add_argument("--checkpoints_dir", default="/data/projects/3D_scans/ML/MQVAE/checkpoints")
    parser.add_argument("--target_col", default="plaque_volume_total_corrected")
    parser.add_argument(
        "--target_source_cols",
        nargs="+",
        default=[
            "plaque_volume_right_carotid_corrected",
            "plaque_volume_left_carotid_corrected",
            "plaque_volume_right_femoral_corrected",
            "plaque_volume_left_femoral_corrected",
        ],
    )
    parser.add_argument("--target_transform", choices=["identity", "log1p"], default="log1p")
    parser.add_argument("--only_gender", choices=["M", "F"], default="F")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--niter", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split_ratio", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--beta_warmup_epochs", type=int, default=30)
    parser.add_argument("--regression_weight", type=float, default=20.0)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--encoder_max_points", type=int, default=4096)
    parser.add_argument("--mesh_knn_k", type=int, default=12)
    parser.add_argument("--mesh_knn_max_points", type=int, default=4096)
    parser.add_argument("--mmd_buffer_size", type=int, default=64)
    parser.add_argument("--mmd_prior_samples", type=int, default=64)
    parser.add_argument("--scan_geometry_features_file", default="scan_geometry_features.npz")
    parser.add_argument(
        "--boson_sampler_path",
        default=None,
        help="Optional path to a newer Non_Linear_Photonic_Sampler checkout.",
    )
    parser.add_argument(
        "--boson_params",
        default=json.dumps(
            {
                "m": 8,
                "num_sources": 4,
                "input_state": [1, 0, 1, 0, 1, 0, 1, 0],
                "num_loops": 100,
                "detector_inefficiency": 1.0,
                "bs_loss": 1.0,
                "mode_loss": [1.0] * 8,
                "use_advanced_nonlinearity": False,
                "cache_size": 4096,
                "cache_circuits": 8,
            },
            separators=(",", ":"),
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--diagnose_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicate values.")
    boson_config = json.loads(args.boson_params)
    if boson_config.__class__ is not dict:
        raise ValueError("--boson_params must be a JSON object.")
    boson_variant = (
        "nonlinear QD" if bool(boson_config.get("use_advanced_nonlinearity", False))
        else "linear"
    )
    ARMS["boson_mmd"]["label"] = f"Sparrow {boson_variant} Boson prior + MMD"
    print(f"[BOSON] benchmark variant: {boson_variant}", flush=True)

    root = Path(__file__).resolve().parent
    failures: dict[str, str] = {}
    if "boson_mmd" in args.arms and not args.diagnose_only and not args.skip_preflight:
        preflight_command = [
            sys.executable,
            str(root / "check_boson_prior.py"),
            "--latent_dim",
            str(args.latent_dim),
            "--samples",
            "512",
            "--boson_params",
            args.boson_params,
        ]
        if args.boson_sampler_path:
            preflight_command.extend(["--boson_sampler_path", args.boson_sampler_path])
        _run(preflight_command, args.dry_run)

    _write_results(_collect_rows(args, failures), args, final=False)
    for seed in args.seeds:
        for arm_key in args.arms:
            arm = ARMS[arm_key]
            experiment = f"{args.name_prefix}_{arm_key}_seed{seed}"
            run_dir = Path(args.checkpoints_dir) / experiment
            checkpoint = run_dir / "best_vae_net.pth"
            try:
                if not args.diagnose_only and not (args.resume and checkpoint.exists()):
                    train_command = [
                        sys.executable,
                        str(root / "train_mesh_vae.py"),
                        "--phase", "train",
                        "--name", experiment,
                        "--checkpoints_dir", args.checkpoints_dir,
                        "--dataroot", args.dataroot,
                        "--cond_tsv", args.train_cond_tsv,
                        "--feature_source", "scan",
                        "--only_gender", args.only_gender,
                        "--gpu_ids", args.gpu_ids,
                        "--seed", str(seed),
                        "--deterministic",
                        "--split_seed", str(seed),
                        "--niter", str(args.niter),
                        "--batch_size", str(args.batch_size),
                        "--split_ratio", str(args.split_ratio),
                        "--beta", str(args.beta),
                        "--beta_warmup_epochs", str(args.beta_warmup_epochs),
                        "--latent_dim", str(args.latent_dim),
                        "--latent_prior", arm["latent_prior"],
                        "--prior_match_mode", arm["prior_match"],
                        "--mmd_buffer_size", str(args.mmd_buffer_size),
                        "--mmd_prior_samples", str(args.mmd_prior_samples),
                        "--use_preprocessed", "--use_pointcloud", "--use_cond",
                        "--encoder_arch", "meshconv",
                        "--encoder_cond_mode", "demo",
                        "--encoder_max_points", str(args.encoder_max_points),
                        "--mesh_use_knn",
                        "--mesh_knn_k", str(args.mesh_knn_k),
                        "--mesh_knn_max_points", str(args.mesh_knn_max_points),
                        "--use_scan_geometry_features",
                        "--require_scan_geometry_features",
                        "--scan_geometry_features_file", args.scan_geometry_features_file,
                        "--restore_metric_scale",
                        "--no-use_amp",
                        "--no-use_point_labels",
                        "--no-point_label_pooling",
                        "--no-cond_athero", "--no-cond_plaque",
                        "--no-aux_predict_athero", "--no-aux_predict_plaque",
                        "--no-balance_athero",
                        "--aux_predict_regression",
                        "--no-aux_predict_binary",
                        "--aux_regression_col", args.target_col,
                        "--aux_regression_source_cols", *args.target_source_cols,
                        "--aux_regression_transform", args.target_transform,
                        "--aux_regression_weight", str(args.regression_weight),
                    ]
                    if arm["latent_prior"] == "boson":
                        train_command.extend(["--boson_params", args.boson_params])
                        if args.boson_sampler_path:
                            train_command.extend(
                                ["--boson_sampler_path", args.boson_sampler_path]
                            )
                    elapsed = _run(train_command, args.dry_run)
                    if not args.dry_run:
                        (run_dir / "benchmark_runtime_seconds.txt").write_text(
                            f"{elapsed:.6f}\n", encoding="utf-8"
                        )
                elif args.diagnose_only and not checkpoint.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
                else:
                    print(f"[RESUME] checkpoint exists: {checkpoint}", flush=True)

                summary_path = run_dir / "regression_summary.csv"
                if not (args.resume and summary_path.exists()):
                    diagnose_command = [
                        sys.executable,
                        str(root / "diagnose_mesh_regression.py"),
                        "--name", experiment,
                        "--checkpoints_dir", args.checkpoints_dir,
                        "--dataroot", args.dataroot,
                        "--cond_tsv", args.test_cond_tsv,
                        "--feature_source", "scan",
                        "--only_gender", args.only_gender,
                        "--gpu_ids", args.gpu_ids,
                        "--seed", str(seed),
                        "--deterministic",
                        "--which_epoch", "best",
                        "--batch_size", str(args.batch_size),
                        "--use_cond",
                    ]
                    if arm["latent_prior"] == "boson" and args.boson_sampler_path:
                        diagnose_command.extend(
                            ["--boson_sampler_path", args.boson_sampler_path]
                        )
                    _run(diagnose_command, args.dry_run)
                else:
                    print(f"[RESUME] diagnosis exists: {summary_path}", flush=True)
            except Exception as exc:
                failures[experiment] = f"{type(exc).__name__}: {exc}"
                print(f"[ERROR] {experiment}: {failures[experiment]}", flush=True)
                if not args.continue_on_error:
                    _write_results(_collect_rows(args, failures), args, final=False)
                    raise
            _write_results(_collect_rows(args, failures), args, final=False)

    final_frame = _collect_rows(args, failures)
    _write_results(final_frame, args, final=not args.dry_run)
    complete = int((final_frame["Status"] == "completed").sum())
    expected = len(args.seeds) * len(args.arms)
    print(f"\n========== BOSON ABLATION COMPLETE: {complete}/{expected} ==========" , flush=True)


if __name__ == "__main__":
    main()
