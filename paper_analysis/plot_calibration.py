#!/usr/bin/env python3
"""Calibration analysis for the primary age+BMI and age+BMI+scan models."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def calibration_coefficients(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Fit target ~ intercept + slope*logit(probability) by Newton iteration."""
    y = np.asarray(target, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))
    design = np.column_stack([np.ones(len(y)), logit])
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(100):
        fitted = sigmoid(design @ beta)
        weight = np.clip(fitted * (1.0 - fitted), 1e-8, None)
        information = design.T @ (weight[:, None] * design)
        information += np.eye(2) * 1e-8
        score = design.T @ (y - fitted)
        step = np.linalg.solve(information, score)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(beta[0]), float(beta[1])


def calibration_metrics(frame: pd.DataFrame) -> dict[str, float]:
    target = frame["target"].to_numpy(dtype=float)
    probability = frame["probability"].to_numpy(dtype=float)
    intercept, slope = calibration_coefficients(target, probability)
    return {
        "brier": float(np.mean((probability - target) ** 2)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def bootstrap_intervals(
    frame: pd.DataFrame,
    bootstraps: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    estimates = calibration_metrics(frame)
    target = frame["target"].to_numpy(dtype=int)
    groups = [np.flatnonzero(target == value) for value in (0, 1)]
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in estimates}
    for _ in range(bootstraps):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        values = calibration_metrics(frame.iloc[indices])
        for metric, value in values.items():
            if np.isfinite(value):
                samples[metric].append(value)
    alpha = 1.0 - confidence
    rows = []
    for metric, estimate in estimates.items():
        values = np.asarray(samples[metric], dtype=float)
        low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
        rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "ci_low": float(low),
                "ci_high": float(high),
                "confidence_level": confidence,
                "n_participants": len(frame),
                "n_bootstrap_valid": len(values),
            }
        )
    return pd.DataFrame(rows)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    spread = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2)) / denominator
    return float(centre - spread), float(centre + spread)


def calibration_bins(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    work = frame[["target", "probability"]].sort_values("probability").reset_index(drop=True)
    effective_bins = min(int(bins), max(2, len(work) // 20))
    work["bin"] = pd.qcut(
        work["probability"].rank(method="first"), effective_bins, labels=False, duplicates="drop"
    )
    rows = []
    for index, group in work.groupby("bin", sort=True):
        positives = int(group["target"].sum())
        low, high = wilson_interval(positives, len(group))
        rows.append(
            {
                "bin": int(index),
                "n": len(group),
                "mean_predicted_probability": float(group["probability"].mean()),
                "observed_prevalence": float(group["target"].mean()),
                "observed_ci_low": low,
                "observed_ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=("age_bmi", "age_bmi_scan"))
    parser.add_argument("--genders", nargs="+", choices=("F", "M", "ALL"), default=("F", "M"))
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = pd.read_csv(args.input)
    required = {"participant_id", "target", "probability", "model", "gender"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing columns {sorted(missing)} in {args.input}")
    frame = frame[frame["model"].isin(args.models) & frame["gender"].isin(args.genders)].copy()
    if frame.empty:
        raise ValueError("No rows remain after filtering models and genders.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    interval_frames = []
    bin_frames = []
    for gender_index, gender in enumerate(args.genders):
        for model_index, model in enumerate(args.models):
            group = frame[(frame["gender"] == gender) & (frame["model"] == model)].copy()
            if group.empty:
                continue
            if group["participant_id"].duplicated().any():
                raise ValueError(f"Duplicate participant IDs for gender={gender}, model={model}")
            if group["target"].nunique() != 2:
                raise ValueError(f"Both classes are required for gender={gender}, model={model}")
            intervals = bootstrap_intervals(
                group,
                args.bootstraps,
                args.confidence,
                args.seed + gender_index * 1000 + model_index,
            )
            intervals.insert(0, "gender", gender)
            intervals.insert(1, "model", model)
            interval_frames.append(intervals)
            binned = calibration_bins(group, args.bins)
            binned.insert(0, "gender", gender)
            binned.insert(1, "model", model)
            bin_frames.append(binned)

    intervals = pd.concat(interval_frames, ignore_index=True)
    binned = pd.concat(bin_frames, ignore_index=True)
    intervals.to_csv(output_dir / "calibration_metric_intervals.csv", index=False)
    binned.to_csv(output_dir / "calibration_bins.csv", index=False)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib unavailable; CSV outputs were still created.", flush=True)
    else:
        present_genders = [gender for gender in args.genders if gender in set(binned["gender"])]
        fig, axes = plt.subplots(1, len(present_genders), figsize=(6.0 * len(present_genders), 5.2), squeeze=False)
        colors = {"age_bmi": "#6D7F8B", "age_bmi_scan": "#0F7C82"}
        labels = {"age_bmi": "Age + BMI", "age_bmi_scan": "Age + BMI + scan"}
        for axis, gender in zip(axes[0], present_genders):
            for model in args.models:
                group = binned[(binned["gender"] == gender) & (binned["model"] == model)]
                if group.empty:
                    continue
                observed = group["observed_prevalence"].to_numpy(dtype=float)
                low = group["observed_ci_low"].to_numpy(dtype=float)
                high = group["observed_ci_high"].to_numpy(dtype=float)
                axis.errorbar(
                    group["mean_predicted_probability"],
                    observed,
                    yerr=np.vstack([observed - low, high - observed]),
                    marker="o",
                    linewidth=1.8,
                    capsize=2.5,
                    color=colors.get(model),
                    label=labels.get(model, model),
                )
            axis.plot([0, 1], [0, 1], linestyle="--", color="#8A99A3", linewidth=1)
            axis.set_title({"F": "Women", "M": "Men", "ALL": "Pooled"}[gender])
            axis.set_xlabel("Mean predicted probability")
            axis.set_ylabel("Observed plaque prevalence")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid(color="#E1E8EB", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0, 0].legend(frameon=False, loc="upper left")
        fig.suptitle("Calibration of primary held-out plaque models")
        fig.tight_layout()
        fig.savefig(output_dir / "calibration_primary_models.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / "calibration_primary_models.pdf", bbox_inches="tight")
        plt.close(fig)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.expanduser().resolve()),
        "models": list(args.models),
        "genders": list(args.genders),
        "bins": args.bins,
        "bin_intervals": "Wilson 95% intervals for observed prevalence",
        "metric_intervals": "participant-level stratified percentile bootstrap",
        "bootstrap_replicates": args.bootstraps,
        "interpretation": (
            "Perfect calibration follows the diagonal. Calibration slope near 1 and intercept near 0 are ideal; "
            "Brier score is better when lower."
        ),
    }
    (output_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[OUT] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
