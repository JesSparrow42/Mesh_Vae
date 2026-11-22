#!/usr/bin/env python3
"""Evaluate a trained MeshVAE continuous-target head on a held-out TSV split."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from meshvae.conditioning import (
    clean_stregkode,
    inverse_regression_target,
    transform_regression_target,
)
from meshvae.datasets import PreprocessedMeshVAEDataset, mesh_collate_fn
from meshvae.math_utils import mmd_imq
from meshvae.model import MeshVAEModel
from meshvae.options import TestOptions
from meshvae.train_eval import regression_metrics


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy()


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    true_rank = _rank(y_true)
    pred_rank = _rank(y_pred)
    if np.std(true_rank) <= 0.0 or np.std(pred_rank) <= 0.0:
        return float("nan")
    return float(np.corrcoef(true_rank, pred_rank)[0, 1])


def main() -> None:
    opt = TestOptions().parse()
    seed = int(getattr(opt, "seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if bool(getattr(opt, "deterministic", False)):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    if not bool(getattr(opt, "aux_predict_regression", False)):
        raise ValueError(
            "This checkpoint is not configured for continuous regression. "
            "Train it with --aux_predict_regression first."
        )
    if not getattr(opt, "cond_tsv", None):
        raise ValueError("Pass the held-out metadata split with --cond_tsv.")

    print("\n========== TEST: MeshVAE regression ==========", flush=True)
    dataset = PreprocessedMeshVAEDataset(opt.dataroot, opt)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(getattr(opt, "batch_size", 1))),
        shuffle=False,
        num_workers=0,
        collate_fn=mesh_collate_fn,
    )

    model = MeshVAEModel(opt)
    model.load_network(opt.which_epoch)
    model.eval()

    target_mean = float(opt.aux_regression_mean)
    target_std = float(opt.aux_regression_std)
    target_transform = str(getattr(opt, "aux_regression_transform", "identity"))
    rows: list[dict[str, float | str]] = []
    diagnostic_sums: dict[str, float] = {}
    diagnostic_count = 0
    latent_samples: list[torch.Tensor] = []
    latent_conditions: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_index, data in enumerate(loader):
            model.set_input(data)
            torch.manual_seed(seed + batch_index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + batch_index)
            result = model.test()
            batch_count = int(result.get("nexamples", 1))
            diagnostic_count += batch_count
            for key in ("loss", "chamfer", "prior_match"):
                value = model.last_diag.get(key)
                if value is not None and np.isfinite(value):
                    diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(value) * batch_count
            latent = result.get("latent_sample")
            if latent is not None:
                latent_samples.append(latent)
                if model.cond is not None:
                    latent_conditions.append(model.cond.detach().cpu())
            pred_norm = result.get("regression_pred")
            if pred_norm is None:
                raise RuntimeError("Checkpoint produced no regression prediction.")
            pred_transformed = pred_norm.numpy().reshape(-1) * target_std + target_mean
            pred = inverse_regression_target(pred_transformed, target_transform)
            target = data["regression_target_raw"].detach().cpu().numpy().reshape(-1)
            paths = data.get("path", [""] * len(pred))
            for path, true_value, pred_value, pred_transformed_value in zip(
                paths,
                target,
                pred,
                pred_transformed,
            ):
                rows.append(
                    {
                        "participant_id": clean_stregkode(path) or str(path),
                        "target": float(true_value),
                        "prediction": float(pred_value),
                        "target_transformed": float(transform_regression_target([true_value], target_transform)[0]),
                        "prediction_transformed": float(pred_transformed_value),
                        "error": float(pred_value - true_value),
                        "absolute_error": float(abs(pred_value - true_value)),
                    }
                )

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("No held-out samples had both a scan and a finite regression target.")

    y = predictions["target"].to_numpy(dtype=np.float64)
    pred = predictions["prediction"].to_numpy(dtype=np.float64)
    metrics = regression_metrics(y, pred)
    metrics["spearman"] = _spearman(y, pred)
    y_transformed = transform_regression_target(y, target_transform)
    pred_transformed = predictions["prediction_transformed"].to_numpy(dtype=np.float64)
    transformed_metrics = regression_metrics(y_transformed, pred_transformed)
    metrics.update({f"transformed_{key}": value for key, value in transformed_metrics.items()})
    metrics["n"] = int(len(predictions))
    baseline_value = float(inverse_regression_target([target_mean], target_transform)[0])
    baseline = np.full_like(y, baseline_value)
    baseline_metrics = regression_metrics(y, baseline)
    metrics["mean_baseline_mae"] = baseline_metrics["mae"]
    metrics["mean_baseline_rmse"] = baseline_metrics["rmse"]
    metrics["target_column"] = str(opt.aux_regression_col)
    metrics["target_source_columns"] = list(getattr(opt, "aux_regression_source_cols", None) or [])
    metrics["target_transform"] = target_transform
    metrics["checkpoint"] = str(opt.which_epoch)
    for key, total in diagnostic_sums.items():
        metrics[f"heldout_{key}"] = total / max(diagnostic_count, 1)

    if latent_samples:
        latent = torch.cat(latent_samples, dim=0).to(model.device)
        cond = (
            torch.cat(latent_conditions, dim=0).to(model.device)
            if len(latent_conditions) == len(latent_samples)
            else None
        )
        base_net = model.net.module if isinstance(model.net, torch.nn.DataParallel) else model.net
        with torch.no_grad():
            torch.manual_seed(seed + 100000)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + 100000)
            prior_samples = base_net.sample_prior(
                int(latent.shape[0]), model.device, cond=cond
            ).to(latent.dtype)
            metrics["latent_mmd"] = float(mmd_imq(latent, prior_samples).detach().cpu())
    else:
        metrics["latent_mmd"] = float("nan")
    metrics["latent_prior"] = str(getattr(opt, "latent_prior", "gaussian"))
    metrics["prior_match_mode"] = str(getattr(opt, "prior_match_mode", "auto"))

    run_dir = (
        Path(opt.checkpoints_dir)
        / opt.name
        / "regression_diagnostics"
        / f"latest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary_row = {
        "Experiment": opt.name,
        "Status": "completed",
        "N": int(metrics["n"]),
        "R2": metrics["r2"],
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "Pearson": metrics["pearson"],
        "Spearman": metrics["spearman"],
        "TransformedR2": metrics["transformed_r2"],
        "TransformedMAE": metrics["transformed_mae"],
        "HeldOutChamfer": metrics.get("heldout_chamfer", float("nan")),
        "HeldOutLoss": metrics.get("heldout_loss", float("nan")),
        "LatentMMD": metrics.get("latent_mmd", float("nan")),
        "LatentPrior": metrics["latent_prior"],
        "PriorMatch": metrics["prior_match_mode"],
        "Target": str(opt.aux_regression_col),
        "TargetTransform": target_transform,
        "ResultsDir": str(run_dir),
    }
    summary_path = Path(opt.checkpoints_dir) / opt.name / "regression_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    print("\n========== HELD-OUT REGRESSION RESULTS ==========", flush=True)
    print(
        pd.DataFrame([summary_row])[
            [
                "N", "R2", "MAE", "RMSE", "Pearson", "Spearman",
                "TransformedR2", "TransformedMAE", "HeldOutChamfer", "LatentMMD",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(
        f"[BASELINE] inverse transformed-train-mean MAE={metrics['mean_baseline_mae']:.4f} "
        f"RMSE={metrics['mean_baseline_rmse']:.4f}"
    )
    print(f"[OUT] predictions -> {run_dir / 'predictions.csv'}")
    print(f"[OUT] metrics     -> {run_dir / 'metrics.json'}")
    print(f"[OUT] summary     -> {summary_path}")


if __name__ == "__main__":
    main()
