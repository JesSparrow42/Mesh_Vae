#!/usr/bin/env python3
"""Run real, long MeshVAE train+diagnose variants.

This script intentionally calls train_mesh_vae.py and diagnose_mesh_vae.py.
It does not use the lightweight comparison models from run_model_comparison.py.

Only scan-based MeshVAE variants are included because a true MeshVAE needs
geometry to reconstruct. Age-only/no-scan rows are classifier baselines, not
full MeshVAE experiments.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "normal_scan_age": {
        "suffix": "vae_scan_age",
        "train_args": [
            "--encoder_cond_mode", "demo",
            "--encoder_arch", "pointnet",
            "--encoder_max_points", "4096",
        ],
        "diagnose_args": [],
        "description": "Full VAE with a permutation-invariant point encoder, body scan + age.",
    },
    "normal_scan_noage": {
        "suffix": "vae_scan_noage",
        "train_args": [
            "--encoder_cond_mode", "none",
            "--encoder_arch", "pointnet",
            "--encoder_max_points", "4096",
        ],
        "diagnose_args": [],
        "description": "Full VAE with a permutation-invariant point encoder, body scan only.",
    },
    "conv_scan_age": {
        "suffix": "vae_conv_scan_age",
        "train_args": [
            "--encoder_cond_mode", "demo",
            "--encoder_arch", "meshconv",
            "--encoder_max_points", "2048",
            "--mesh_use_knn", "--mesh_knn_k", "12", "--mesh_knn_max_points", "2048",
        ],
        "diagnose_args": [],
        "description": "Full VAE with local kNN edge convolution, body scan + age.",
    },
    "conv_scan_noage": {
        "suffix": "vae_conv_scan_noage",
        "train_args": [
            "--encoder_cond_mode", "none",
            "--encoder_arch", "meshconv",
            "--encoder_max_points", "2048",
            "--mesh_use_knn", "--mesh_knn_k", "12", "--mesh_knn_max_points", "2048",
        ],
        "diagnose_args": [],
        "description": "Full VAE with local kNN edge convolution, body scan only.",
    },
    "pointlabels_scan_age": {
        "suffix": "vae_pointlabels_scan_age",
        "train_args": [
            "--encoder_cond_mode", "demo",
            "--encoder_arch", "pointnet",
            "--encoder_max_points", "4096",
            "--use_point_labels", "--point_label_regions", "6", "--point_label_pooling",
        ],
        "diagnose_args": [],
        "description": "Full VAE with anatomical pseudo labels and region-wise pooling, body scan + age.",
    },
    "pointlabels_scan_noage": {
        "suffix": "vae_pointlabels_scan_noage",
        "train_args": [
            "--encoder_cond_mode", "none",
            "--encoder_arch", "pointnet",
            "--encoder_max_points", "4096",
            "--use_point_labels", "--point_label_regions", "6", "--point_label_pooling",
        ],
        "diagnose_args": [],
        "description": "Full VAE with anatomical pseudo labels and region-wise pooling, body scan only.",
    },
}


def print_cmd(cmd: list[str]) -> None:
    print(" ".join(shlex.quote(str(x)) for x in cmd), flush=True)


def run(cmd: list[str], dry_run: bool) -> None:
    print_cmd(cmd)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name_prefix", required=True)
    ap.add_argument("--dataroot", default="/data/projects/3D_scans/ML/MQVAE")
    ap.add_argument("--checkpoints_dir", default="/data/projects/3D_scans/ML/MQVAE/checkpoints")
    ap.add_argument("--train_cond_tsv", required=True)
    ap.add_argument("--test_cond_tsv", required=True)
    ap.add_argument("--only_gender", choices=["M", "F"], default="F")
    ap.add_argument("--niter", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gpu_ids", default="0")
    ap.add_argument("--use_scan_geometry_features", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--scan_geometry_features_file", default="scan_geometry_features.npz")
    ap.add_argument("--require_scan_geometry_features", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--restore_metric_scale", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--target_missing_policy", choices=["drop", "zero", "error"], default="drop")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
    )
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_diagnose", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    train_script = str(root / "train_mesh_vae.py")
    diagnose_script = str(root / "diagnose_mesh_vae.py")

    for key in args.variants:
        cfg = VARIANTS[key]
        name = f"{args.name_prefix}_{cfg['suffix']}"
        print(f"\n========== {name} ==========")
        print(cfg["description"], flush=True)

        common = [
            "--name", name,
            "--dataroot", args.dataroot,
            "--checkpoints_dir", args.checkpoints_dir,
            "--feature_source", "scan",
            "--only_gender", args.only_gender,
            "--gpu_ids", args.gpu_ids,
            "--target_missing_policy", args.target_missing_policy,
        ]
        if args.use_scan_geometry_features:
            common.extend(["--use_scan_geometry_features", "--scan_geometry_features_file", args.scan_geometry_features_file])
        else:
            common.append("--no-use_scan_geometry_features")
        if args.require_scan_geometry_features:
            common.append("--require_scan_geometry_features")
        if args.restore_metric_scale:
            common.append("--restore_metric_scale")
        else:
            common.append("--no-restore_metric_scale")

        if not args.skip_train:
            train_cmd = [
                sys.executable,
                train_script,
                *common,
                "--cond_tsv", args.train_cond_tsv,
                "--niter", str(args.niter),
                "--batch_size", str(args.batch_size),
                "--no-use_amp",
                "--cond_athero_mode", "any",
                "--aux_predict_athero",
                "--aux_athero_mode", "any",
                "--aux_athero_weight", "20.0",
                "--aux_athero_pos_weight", "1.0",
                "--balance_athero",
                "--beta", "0.001",
                "--beta_warmup_epochs", "30",
                "--split_ratio", "0.20",
                *cfg["train_args"],
            ]
            run(train_cmd, args.dry_run)

        if not args.skip_diagnose:
            diagnose_cmd = [
                sys.executable,
                diagnose_script,
                *common,
                "--calib_cond_tsv", args.train_cond_tsv,
                "--test_cond_tsv", args.test_cond_tsv,
                "--which_epoch", "best",
                "--score_method", "classifier_prob_sick",
                "--policy_mode", "global",
                "--calib_use_training_validation",
                *cfg["diagnose_args"],
            ]
            run(diagnose_cmd, args.dry_run)


if __name__ == "__main__":
    main()
