#!/usr/bin/env python3
"""Evaluate a trained MeshVAE binary-target head on a held-out TSV split."""
from __future__ import annotations

# Allow direct execution as well as ``python -m`` from the repository checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from meshvae.conditioning import clean_stregkode
from meshvae.datasets import PreprocessedMeshVAEDataset, mesh_collate_fn
from meshvae.math_utils import mmd_imq
from meshvae.model import MeshVAEModel
from meshvae.options import TestOptions
from meshvae.train_eval import _roc_auc


def _average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


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
    if not bool(getattr(opt, "aux_predict_binary", False)):
        raise ValueError(
            "This checkpoint is not configured for a generic binary target. "
            "Train it with --aux_predict_binary first."
        )
    if not getattr(opt, "cond_tsv", None):
        raise ValueError("Pass the held-out metadata split with --cond_tsv.")

    print("\n========== TEST: MeshVAE binary target ==========", flush=True)
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
    rows: list[dict[str, float | str | int]] = []
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
            probability = result.get("binary_prob")
            if probability is None:
                raise RuntimeError("Checkpoint produced no generic binary prediction.")
            probability = probability.numpy().reshape(-1)
            target = data["binary_target"].detach().cpu().numpy().reshape(-1)
            paths = data.get("path", [""] * len(probability))
            for path, true_value, prob in zip(paths, target, probability):
                rows.append(
                    {
                        "participant_id": clean_stregkode(path) or str(path),
                        "target": int(round(float(true_value))),
                        "probability": float(prob),
                        "prediction_0_5": int(float(prob) >= 0.5),
                    }
                )

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("No held-out samples had both a scan and a finite binary target.")

    y = predictions["target"].to_numpy(dtype=np.int64) == 1
    probability = predictions["probability"].to_numpy(dtype=np.float64)
    pred = probability >= 0.5
    tp = int(np.sum(y & pred))
    tn = int(np.sum((~y) & (~pred)))
    fp = int(np.sum((~y) & pred))
    fn = int(np.sum(y & (~pred)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    metrics = {
        "n": int(len(y)),
        "negative": int((~y).sum()),
        "positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / max(len(y), 1)),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12)),
        "roc_auc": _roc_auc(y, probability),
        "average_precision": _average_precision(y, probability),
        "threshold": 0.5,
        "target_column": str(opt.aux_binary_col),
        "checkpoint": str(opt.which_epoch),
    }
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
        / "binary_diagnostics"
        / f"latest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary_row = {
        "Experiment": opt.name,
        "Status": "completed",
        "Target": str(opt.aux_binary_col),
        "N": metrics["n"],
        "Positive": metrics["positive"],
        "Prevalence": metrics["prevalence"],
        "AUC": metrics["roc_auc"],
        "AveragePrecision": metrics["average_precision"],
        "BalancedAccuracy": metrics["balanced_accuracy"],
        "Accuracy": metrics["accuracy"],
        "Sensitivity": metrics["sensitivity"],
        "Specificity": metrics["specificity"],
        "Precision": metrics["precision"],
        "F1": metrics["f1"],
        "Threshold": metrics["threshold"],
        "HeldOutChamfer": metrics.get("heldout_chamfer", float("nan")),
        "HeldOutLoss": metrics.get("heldout_loss", float("nan")),
        "LatentMMD": metrics.get("latent_mmd", float("nan")),
        "LatentPrior": metrics["latent_prior"],
        "PriorMatch": metrics["prior_match_mode"],
        "ResultsDir": str(run_dir),
    }
    summary_path = Path(opt.checkpoints_dir) / opt.name / "binary_summary.csv"
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    print("\n========== HELD-OUT BINARY RESULTS ==========", flush=True)
    print(
        pd.DataFrame([summary_row])[
            [
                "N", "Positive", "Prevalence", "AUC", "AveragePrecision",
                "BalancedAccuracy", "Accuracy", "Sensitivity", "Specificity", "F1",
                "HeldOutChamfer", "LatentMMD",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"[OUT] predictions -> {run_dir / 'predictions.csv'}")
    print(f"[OUT] metrics     -> {run_dir / 'metrics.json'}")
    print(f"[OUT] summary     -> {summary_path}")


if __name__ == "__main__":
    main()
