#!/usr/bin/env python3
"""Participant-level bootstrap confidence intervals for held-out predictions.

Examples
--------
One model::

    python paper_analysis/bootstrap_metrics.py \
      --input scan_model=/path/to/predictions.csv \
      --output-dir /path/to/paper_ci/scan_model

Paired model comparison on the same held-out participants::

    python paper_analysis/bootstrap_metrics.py \
      --input age_bmi=/path/to/age_bmi.csv \
      --input age_bmi_scan=/path/to/age_bmi_scan.csv \
      --reference age_bmi \
      --output-dir /path/to/paper_ci/primary_comparison

The bootstrap resampling unit is the participant. Binary outcomes are
stratified by case/control status. Paired comparisons use exactly the same
participant resample for both models.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

try:
    from .data_utils import clean_participant_id, find_column
except ImportError:  # Direct execution from the repository root.
    from data_utils import clean_participant_id, find_column


ID_ALIASES = ("participant_id", "participantid", "stregkode", "subject_id", "id")
BINARY_TARGET_ALIASES = ("target", "label_sick", "y_true", "true_label", "observed")
SCORE_ALIASES = (
    "probability",
    "score_value",
    "prob_sick",
    "classifier_prob_sick",
    "sample_prob_sick",
    "y_score",
)
PROBABILITY_ALIASES = {"probability", "prob_sick", "classifier_prob_sick", "sample_prob_sick"}
BINARY_PREDICTION_ALIASES = ("prediction_0_5", "pred_sick", "prediction", "y_pred")
REGRESSION_TARGET_ALIASES = ("target", "observed", "y_true", "label")
REGRESSION_PREDICTION_ALIASES = ("prediction", "predicted", "y_pred")
TRANSFORMED_TARGET_ALIASES = ("target_transformed", "observed_transformed", "y_true_transformed")
TRANSFORMED_PREDICTION_ALIASES = (
    "prediction_transformed",
    "predicted_transformed",
    "y_pred_transformed",
)


@dataclass(frozen=True)
class Metric:
    name: str
    function: Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]
    higher_is_better: bool


def _finite_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    keep = np.isfinite(left) & np.isfinite(right)
    return left[keep], right[keep]


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y, score = _finite_pair(y_true, score)
    y = y.astype(int)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    rank_sum = float(ranks[y == 1].sum())
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y, score = _finite_pair(y_true, score)
    y = y.astype(int)
    positives = int(np.sum(y == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ranked = y[order]
    # A threshold includes every observation with the same score. Treating ties
    # one row at a time makes AP depend on CSV row order (and bootstrap order).
    endpoints = np.r_[np.flatnonzero(np.diff(score[order])), len(score) - 1]
    true_positive = np.cumsum(ranked)[endpoints]
    precision = true_positive / (endpoints + 1)
    return float(np.sum(np.diff(np.r_[0, true_positive]) * precision) / positives)


def _confusion(y_true: np.ndarray, prediction: np.ndarray) -> tuple[int, int, int, int]:
    y, prediction = _finite_pair(y_true, prediction)
    y = y.astype(int) == 1
    prediction = prediction.astype(int) == 1
    return (
        int(np.sum(y & prediction)),
        int(np.sum((~y) & (~prediction))),
        int(np.sum((~y) & prediction)),
        int(np.sum(y & (~prediction))),
    )


def _binary_metric(name: str, y: np.ndarray, score: np.ndarray, prediction: np.ndarray | None) -> float:
    if name == "auc":
        return roc_auc(y, score)
    if name == "average_precision":
        return average_precision(y, score)
    if name == "brier":
        left, right = _finite_pair(y, score)
        return float(np.mean((right - left) ** 2)) if len(left) else float("nan")
    if prediction is None:
        raise ValueError(f"Metric {name} requires binary predictions.")
    tp, tn, fp, fn = _confusion(y, prediction)
    n = tp + tn + fp + fn
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    if name == "accuracy":
        return float((tp + tn) / max(n, 1))
    if name == "balanced_accuracy":
        return float(0.5 * (sensitivity + specificity))
    if name == "sensitivity":
        return float(sensitivity)
    if name == "specificity":
        return float(specificity)
    if name == "precision":
        return float(precision)
    if name == "f1":
        return float(2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12))
    raise KeyError(name)


def _regression_metric(name: str, target: np.ndarray, prediction: np.ndarray) -> float:
    target, prediction = _finite_pair(target, prediction)
    if not len(target):
        return float("nan")
    residual = prediction - target
    if name == "r2":
        denominator = float(np.sum((target - target.mean()) ** 2))
        return float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else float("nan")
    if name == "mae":
        return float(np.mean(np.abs(residual)))
    if name == "rmse":
        return float(np.sqrt(np.mean(residual**2)))
    if name == "pearson":
        if len(target) < 2 or np.std(target) <= 0 or np.std(prediction) <= 0:
            return float("nan")
        return float(np.corrcoef(target, prediction)[0, 1])
    if name == "spearman":
        target_rank = pd.Series(target).rank(method="average").to_numpy()
        prediction_rank = pd.Series(prediction).rank(method="average").to_numpy()
        if len(target_rank) < 2 or np.std(target_rank) <= 0 or np.std(prediction_rank) <= 0:
            return float("nan")
        return float(np.corrcoef(target_rank, prediction_rank)[0, 1])
    raise KeyError(name)


BINARY_METRICS = (
    ("auc", True),
    ("average_precision", True),
    ("balanced_accuracy", True),
    ("accuracy", True),
    ("sensitivity", True),
    ("specificity", True),
    ("precision", True),
    ("f1", True),
    ("brier", False),
)
REGRESSION_METRICS = (
    ("r2", True),
    ("mae", False),
    ("rmse", False),
    ("pearson", True),
    ("spearman", True),
)


def _resolve(frame: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> str | None:
    column = find_column(frame, aliases)
    if column is None and required:
        raise KeyError(f"None of {list(aliases)} was found. Available columns: {list(frame.columns)[:80]}")
    return column


def infer_task(frame: pd.DataFrame) -> str:
    if find_column(frame, SCORE_ALIASES) is not None:
        return "binary"
    if find_column(frame, REGRESSION_PREDICTION_ALIASES) is not None:
        return "regression"
    raise ValueError("Could not infer task from prediction columns; pass --task explicitly.")


def _coerce_binary(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(
        {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "positive": 1.0,
            "negative": 0.0,
            "case": 1.0,
            "control": 0.0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.notna(), mapped)


def _handle_duplicates(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    duplicates = frame["_id"].duplicated(keep=False)
    if not duplicates.any():
        return frame.reset_index(drop=True)
    ids = frame.loc[duplicates, "_id"].drop_duplicates().head(10).tolist()
    if policy == "error":
        raise ValueError(
            f"Participant IDs are duplicated (examples: {ids}). Participant-level CIs require one row "
            "per participant. Use --duplicate-policy mean only when repeated rows are genuinely repeated "
            "predictions for the same outcome."
        )
    if policy == "first":
        return frame.drop_duplicates("_id", keep="first").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for participant_id, group in frame.groupby("_id", sort=False):
        targets = group["_target"].dropna().unique()
        if len(targets) > 1 and not np.allclose(targets, targets[0], equal_nan=True):
            raise ValueError(f"Participant {participant_id} has inconsistent targets across duplicate rows.")
        row: dict[str, object] = {"_id": participant_id, "_target": float(targets[0])}
        for column in group.columns:
            if column in row or column == "_target":
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(values.mean()) if values.notna().any() else group[column].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def load_predictions(
    path: str | Path,
    task: str,
    threshold: float,
    duplicate_policy: str,
    *,
    require_participant_id: bool = True,
) -> tuple[pd.DataFrame, str]:
    path = Path(path)
    frame = pd.read_csv(path, dtype=str)
    split_column = _resolve(frame, ("split",), required=False)
    excluded_rows = 0
    if split_column is not None:
        heldout = frame[split_column].str.strip().str.lower().isin(("test", "heldout", "held_out", "held-out"))
        excluded_rows = int((~heldout).sum())
        frame = frame.loc[heldout].copy()
    if frame.empty:
        raise ValueError(f"No held-out prediction rows remain in {path}.")
    resolved_task = infer_task(frame) if task == "auto" else task
    id_column = _resolve(frame, ID_ALIASES, required=False)
    # Older MeshVAE diagnostics write paths instead of IDs in this exact file.
    # Restrict the fallback to that contract, rather than guessing generic CSVs.
    if id_column is None and path.name.lower() == "test_predictions.csv":
        id_column = _resolve(frame, ("path",), required=False)
    if id_column is None:
        if require_participant_id:
            raise ValueError("No participant ID column; participant-level CIs require one ID per person.")
        frame["_id"] = [f"row_{index:08d}" for index in range(len(frame))]
    else:
        ids = frame[id_column]
        if id_column.lower() == "path":
            ids = ids.map(lambda value: Path(str(value).replace("\\", "/")).name if pd.notna(value) else None)
        frame["_id"] = ids.map(clean_participant_id)
        if frame["_id"].isna().any():
            raise ValueError(f"Missing participant IDs in {path}.")

    def require_finite(column: str) -> None:
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite or nonnumeric predictions in {path}: {column}.")

    if resolved_task == "binary":
        target_column = _resolve(frame, BINARY_TARGET_ALIASES)
        score_column = _resolve(frame, SCORE_ALIASES)
        score_is_probability = score_column.lower() in PROBABILITY_ALIASES
        prediction_column = _resolve(frame, BINARY_PREDICTION_ALIASES, required=False)
        frame["_target"] = _coerce_binary(frame[target_column])
        frame["_score"] = pd.to_numeric(frame[score_column], errors="coerce")
        require_finite("_score")
        if score_is_probability and not frame["_score"].between(0, 1).all():
            raise ValueError(f"Probability predictions in {path} must be between 0 and 1.")
        if not np.isfinite(threshold):
            raise ValueError("Binary decision threshold must be finite.")
        if prediction_column is None:
            frame["_pred"] = (frame["_score"] >= float(threshold)).astype(float)
        else:
            frame["_pred"] = _coerce_binary(frame[prediction_column])
        if not frame["_pred"].isin([0, 1]).all():
            raise ValueError(f"Binary decisions in {path} must be 0 or 1.")
        if (frame["_target"].notna() & ~frame["_target"].isin([0, 1])).any():
            raise ValueError(f"Binary targets in {path} must be 0 or 1.")
        frame = frame.dropna(subset=["_target"]).copy()
        if frame["_target"].nunique() != 2:
            raise ValueError(f"Binary predictions in {path} do not contain both classes.")
    elif resolved_task == "regression":
        target_column = _resolve(frame, REGRESSION_TARGET_ALIASES)
        prediction_column = _resolve(frame, REGRESSION_PREDICTION_ALIASES)
        frame["_target"] = pd.to_numeric(frame[target_column], errors="coerce")
        frame["_prediction"] = pd.to_numeric(frame[prediction_column], errors="coerce")
        require_finite("_prediction")
        transformed_target = _resolve(frame, TRANSFORMED_TARGET_ALIASES, required=False)
        transformed_prediction = _resolve(frame, TRANSFORMED_PREDICTION_ALIASES, required=False)
        if (transformed_target is None) != (transformed_prediction is None):
            raise ValueError("Transformed target and prediction columns must be supplied together.")
        if transformed_target is not None and transformed_prediction is not None:
            frame["_target_transformed"] = pd.to_numeric(frame[transformed_target], errors="coerce")
            frame["_prediction_transformed"] = pd.to_numeric(
                frame[transformed_prediction], errors="coerce"
            )
            require_finite("_prediction_transformed")
        frame = frame.dropna(subset=["_target"]).copy()
        for column in ("_target", "_target_transformed"):
            if column in frame and not np.isfinite(frame[column].to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite targets in {path}: {column}.")
    else:
        raise ValueError(f"Unsupported task: {resolved_task}")

    if frame.empty:
        raise ValueError(f"No labelled held-out predictions remain in {path}.")
    had_duplicates = frame["_id"].duplicated().any()
    frame = _handle_duplicates(frame, duplicate_policy)
    if resolved_task == "binary" and duplicate_policy == "mean" and had_duplicates:
        # A mean of class decisions is not itself a binary decision.
        frame["_pred"] = (frame["_score"] >= threshold).astype(float)
    frame.attrs["excluded_non_test_rows"] = excluded_rows
    if resolved_task == "binary":
        frame.attrs["score_is_probability"] = score_is_probability
    return frame.reset_index(drop=True), resolved_task


def _strata(frame: pd.DataFrame, task: str, regression_strata: str) -> list[np.ndarray]:
    if task == "binary":
        return [np.flatnonzero(frame["_target"].to_numpy() == value) for value in (0, 1)]
    values = frame["_target"].to_numpy(dtype=float)
    if regression_strata in {"auto", "zero"}:
        zero = np.isclose(values, 0.0)
        if zero.any() and (~zero).any():
            return [np.flatnonzero(zero), np.flatnonzero(~zero)]
    return [np.arange(len(frame), dtype=int)]


def bootstrap_indices(
    frame: pd.DataFrame,
    task: str,
    n_bootstrap: int,
    seed: int,
    regression_strata: str = "auto",
) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = [group for group in _strata(frame, task, regression_strata) if len(group)]
    for _ in range(int(n_bootstrap)):
        sampled = [rng.choice(group, size=len(group), replace=True) for group in groups]
        indices = np.concatenate(sampled)
        rng.shuffle(indices)
        yield indices


def metric_values(frame: pd.DataFrame, task: str) -> dict[str, float]:
    if task == "binary":
        y = frame["_target"].to_numpy(dtype=float)
        score = frame["_score"].to_numpy(dtype=float)
        pred = frame["_pred"].to_numpy(dtype=float)
        # In-memory standardized frames are probability outputs; CSV loading
        # explicitly marks generic scores/margins, even if they fall in [0, 1].
        probability = frame.attrs.get("score_is_probability", True)
        if probability and (not np.isfinite(score).all() or ((score < 0) | (score > 1)).any()):
            raise ValueError("Probability predictions must be finite and between 0 and 1.")
        return {name: float("nan") if name == "brier" and not probability else _binary_metric(name, y, score, pred)
                for name, _ in BINARY_METRICS}

    target = frame["_target"].to_numpy(dtype=float)
    prediction = frame["_prediction"].to_numpy(dtype=float)
    values = {name: _regression_metric(name, target, prediction) for name, _ in REGRESSION_METRICS}
    if {"_target_transformed", "_prediction_transformed"}.issubset(frame.columns):
        transformed_target = frame["_target_transformed"].to_numpy(dtype=float)
        transformed_prediction = frame["_prediction_transformed"].to_numpy(dtype=float)
        values.update(
            {
                f"transformed_{name}": _regression_metric(name, transformed_target, transformed_prediction)
                for name, _ in REGRESSION_METRICS
            }
        )
    return values


def _higher_is_better(metric: str) -> bool:
    base = metric.removeprefix("transformed_")
    lookup = dict(BINARY_METRICS) | dict(REGRESSION_METRICS)
    return bool(lookup[base])


def percentile_interval(values: Iterable[float], confidence: float) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), 0
    alpha = 1.0 - float(confidence)
    low, high = np.quantile(array, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high), int(len(array))


def summarize_model(
    label: str,
    frame: pd.DataFrame,
    task: str,
    n_bootstrap: int,
    seed: int,
    confidence: float,
    regression_strata: str,
) -> pd.DataFrame:
    estimates = metric_values(frame, task)
    samples = {metric: [] for metric in estimates}
    for indices in bootstrap_indices(frame, task, n_bootstrap, seed, regression_strata):
        values = metric_values(frame.iloc[indices], task)
        for metric, value in values.items():
            samples[metric].append(value)
    rows = []
    for metric, estimate in estimates.items():
        low, high, valid = percentile_interval(samples[metric], confidence)
        rows.append(
            {
                "model": label,
                "task": task,
                "metric": metric,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "confidence_level": confidence,
                "n_participants": len(frame),
                "n_bootstrap_requested": n_bootstrap,
                "n_bootstrap_valid": valid,
                "higher_is_better": _higher_is_better(metric),
                "resampling_unit": "participant",
                "interval_method": "stratified percentile bootstrap",
            }
        )
    return pd.DataFrame(rows)


def paired_difference(
    reference_label: str,
    reference: pd.DataFrame,
    comparison_label: str,
    comparison: pd.DataFrame,
    task: str,
    n_bootstrap: int,
    seed: int,
    confidence: float,
    regression_strata: str,
) -> pd.DataFrame:
    ref_columns = [column for column in reference.columns if column.startswith("_")]
    cmp_columns = [column for column in comparison.columns if column.startswith("_")]
    merged = reference[ref_columns].merge(
        comparison[cmp_columns],
        on="_id",
        how="inner",
        suffixes=("_ref", "_cmp"),
        validate="one_to_one",
    )
    if not len(merged):
        raise ValueError(f"No participant IDs overlap between {reference_label} and {comparison_label}.")
    ref_target = merged["_target_ref"].to_numpy(dtype=float)
    cmp_target = merged["_target_cmp"].to_numpy(dtype=float)
    if not np.allclose(ref_target, cmp_target, equal_nan=True):
        raise ValueError(
            f"Targets disagree after participant matching for {reference_label} and {comparison_label}."
        )

    def rebuild(suffix: str) -> pd.DataFrame:
        data = {"_id": merged["_id"], "_target": merged[f"_target_{suffix}"]}
        for base in (
            "_score",
            "_pred",
            "_prediction",
            "_target_transformed",
            "_prediction_transformed",
        ):
            column = f"{base}_{suffix}"
            if column in merged:
                data[base] = merged[column]
        aligned = pd.DataFrame(data)
        aligned.attrs.update(reference.attrs if suffix == "ref" else comparison.attrs)
        return aligned

    reference_aligned = rebuild("ref")
    comparison_aligned = rebuild("cmp")
    ref_estimates = metric_values(reference_aligned, task)
    cmp_estimates = metric_values(comparison_aligned, task)
    common_metrics = sorted(set(ref_estimates) & set(cmp_estimates))
    deltas = {metric: [] for metric in common_metrics}
    for indices in bootstrap_indices(
        reference_aligned, task, n_bootstrap, seed, regression_strata
    ):
        ref_values = metric_values(reference_aligned.iloc[indices], task)
        cmp_values = metric_values(comparison_aligned.iloc[indices], task)
        for metric in common_metrics:
            deltas[metric].append(cmp_values[metric] - ref_values[metric])

    rows = []
    for metric in common_metrics:
        raw_delta = cmp_estimates[metric] - ref_estimates[metric]
        low, high, valid = percentile_interval(deltas[metric], confidence)
        higher = _higher_is_better(metric)
        improvement = raw_delta if higher else -raw_delta
        improvement_low, improvement_high = (low, high) if higher else (-high, -low)
        rows.append(
            {
                "reference_model": reference_label,
                "comparison_model": comparison_label,
                "task": task,
                "metric": metric,
                "reference_estimate": ref_estimates[metric],
                "comparison_estimate": cmp_estimates[metric],
                "delta_comparison_minus_reference": raw_delta,
                "delta_ci_low": low,
                "delta_ci_high": high,
                "directional_improvement": improvement,
                "improvement_ci_low": improvement_low,
                "improvement_ci_high": improvement_high,
                "confidence_level": confidence,
                "n_paired_participants": len(merged),
                "n_bootstrap_valid": valid,
                "higher_is_better": higher,
                "interval_method": "paired stratified percentile bootstrap",
            }
        )
    return pd.DataFrame(rows)


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use --input LABEL=/path/to/predictions.csv")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Prediction label cannot be empty.")
    return label, Path(path).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_input)
    parser.add_argument("--task", choices=("auto", "binary", "regression"), default="auto")
    parser.add_argument("--reference", default=None, help="Label used as paired comparison reference.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--duplicate-policy", choices=("error", "first", "mean"), default="error")
    parser.add_argument("--regression-strata", choices=("auto", "none", "zero"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between zero and one.")
    if args.bootstraps < 100:
        raise ValueError("Use at least 100 bootstrap replicates; 5000 is recommended for final results.")

    loaded: dict[str, pd.DataFrame] = {}
    tasks: dict[str, str] = {}
    paths: dict[str, str] = {}
    for label, path in args.input:
        if label in loaded:
            raise ValueError(f"Duplicate input label: {label}")
        frame, resolved_task = load_predictions(
            path, args.task, args.threshold, args.duplicate_policy
        )
        loaded[label] = frame
        tasks[label] = resolved_task
        paths[label] = str(path.resolve())
        print(f"[INPUT] {label}: task={resolved_task} n={len(frame)} path={path}", flush=True)

    if len(set(tasks.values())) != 1:
        raise ValueError(f"All inputs in one invocation must have the same task. Resolved tasks: {tasks}")
    task = next(iter(tasks.values()))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for offset, (label, frame) in enumerate(loaded.items()):
        summaries.append(
            summarize_model(
                label,
                frame,
                task,
                args.bootstraps,
                args.seed + offset * 100003,
                args.confidence,
                args.regression_strata,
            )
        )
    summary = pd.concat(summaries, ignore_index=True)
    summary_path = output_dir / "bootstrap_metric_intervals.csv"
    summary.to_csv(summary_path, index=False)

    paired_frames = []
    if args.reference is not None:
        if args.reference not in loaded:
            raise KeyError(f"Reference {args.reference!r} is not one of {list(loaded)}")
        for offset, (label, frame) in enumerate(loaded.items()):
            if label == args.reference:
                continue
            paired_frames.append(
                paired_difference(
                    args.reference,
                    loaded[args.reference],
                    label,
                    frame,
                    task,
                    args.bootstraps,
                    args.seed + 500000 + offset * 100003,
                    args.confidence,
                    args.regression_strata,
                )
            )
    paired = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    paired_path = output_dir / "paired_model_differences.csv"
    paired.to_csv(paired_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "inputs": paths,
        "reference": args.reference,
        "bootstrap_replicates": args.bootstraps,
        "confidence_level": args.confidence,
        "seed": args.seed,
        "duplicate_policy": args.duplicate_policy,
        "regression_strata": args.regression_strata,
        "resampling_unit": "participant",
        "interval_method": "stratified percentile bootstrap",
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"[OUT] metric intervals -> {summary_path}", flush=True)
    if not paired.empty:
        print(f"[OUT] paired differences -> {paired_path}", flush=True)
        primary = paired[paired["metric"].isin(["auc", "transformed_r2", "r2"])]
        if len(primary):
            print("\nPrimary paired results:\n" + primary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

