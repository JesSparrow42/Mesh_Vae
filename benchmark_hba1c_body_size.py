#!/usr/bin/env python3
"""Benchmark HbA1c prediction from measured body size on fixed TSV splits."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from meshvae.conditioning import _detect_sep, _pick_col, _to_numeric_series, clean_column_names


DEFAULT_FEATURES = {
    "age": (["age"], "age"),
    "waist": (["waist_circumference_cm"], "waist"),
    "hip": (["hip_circumference_cm"], "hip"),
    "height": (["height_cm"], "height"),
    "weight": (["weight_kg"], "weight"),
    "bmi": (["body_mass_index", "bmi"], "body_mass"),
}


def read_condition_tsv(path: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=_detect_sep(path),
        dtype=str,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        engine="python",
    )
    return clean_column_names(frame)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target, prediction = target[valid], prediction[valid]
    residual = prediction - target
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "r2": float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0.0 else float("nan"),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "pearson": (
            float(np.corrcoef(target, prediction)[0, 1])
            if len(target) > 1 and np.std(target) > 0.0 and np.std(prediction) > 0.0
            else float("nan")
        ),
    }


def resolve_columns(frame: pd.DataFrame, target_name: str) -> tuple[str, dict[str, str]]:
    target = target_name if target_name in frame.columns else _pick_col(frame, [target_name], fallback_contains="hba1c")
    if target is None:
        raise KeyError(f"Could not find HbA1c target. Available: {list(frame.columns)[:100]}")
    resolved = {}
    for display, (preferred, contains) in DEFAULT_FEATURES.items():
        column = _pick_col(frame, preferred, fallback_contains=contains)
        if column is not None:
            resolved[display] = str(column)
    required = {"age", "waist", "hip", "height", "weight", "bmi"}
    missing = sorted(required - set(resolved))
    if missing:
        raise KeyError(f"Missing expected body-size columns: {missing}. Available: {list(frame.columns)[:100]}")
    return str(target), resolved


def numeric_frame(frame: pd.DataFrame, target: str, columns: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame({"target": _to_numeric_series(frame[target])})
    for display, column in columns.items():
        out[display] = _to_numeric_series(frame[column])
    return out.dropna(subset=["target"]).reset_index(drop=True)


def polynomial_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    columns = [values]
    for left in range(values.shape[1]):
        for right in range(left, values.shape[1]):
            columns.append((values[:, left] * values[:, right]).reshape(-1, 1))
    return np.concatenate(columns, axis=1)


def fit_transform(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    median = np.nanmedian(train, axis=0)
    train = np.where(np.isfinite(train), train, median[None, :])
    test = np.where(np.isfinite(test), test, median[None, :])
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    train = (train - mean[None, :]) / std[None, :]
    test = (test - mean[None, :]) / std[None, :]
    return train, test


def fit_ridge(train: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    design = np.concatenate([np.ones((len(train), 1)), train], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ target


def predict_ridge(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((len(values), 1)), values], axis=1)
    return design @ weights


def target_folds(target: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    quantiles = np.asarray(
        pd.qcut(target, q=min(10, max(2, len(target) // 20)), labels=False, duplicates="drop"),
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[] for _ in range(folds)]
    for label in np.unique(quantiles):
        indices = np.flatnonzero(quantiles == label)
        rng.shuffle(indices)
        for fold, split in enumerate(np.array_split(indices, folds)):
            parts[fold].extend(split.tolist())
    return [np.asarray(sorted(part), dtype=np.int64) for part in parts]


def spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    left = pd.Series(target).rank(method="average").to_numpy()
    right = pd.Series(prediction).rank(method="average").to_numpy()
    if len(left) < 2 or np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def evaluate_model(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_names: list[str],
    polynomial: bool,
    args,
) -> dict:
    train_values = train_frame[feature_names].to_numpy(np.float64)
    test_values = test_frame[feature_names].to_numpy(np.float64)
    if polynomial:
        train_values = polynomial_features(train_values)
        test_values = polynomial_features(test_values)
    target = train_frame["target"].to_numpy(np.float64)
    test_target = test_frame["target"].to_numpy(np.float64)
    alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
    folds = target_folds(target, args.folds, args.seed)
    best_alpha = 0.0
    best_score = -float("inf")
    best_oof = None
    all_indices = np.arange(len(target))
    for alpha in alphas:
        oof = np.full(len(target), np.nan, dtype=np.float64)
        for validation in folds:
            fit_indices = np.setdiff1d(all_indices, validation, assume_unique=True)
            fit_x, validation_x = fit_transform(train_values[fit_indices], train_values[validation])
            weights = fit_ridge(fit_x, target[fit_indices], alpha)
            oof[validation] = predict_ridge(validation_x, weights)
        score = regression_metrics(target, oof)["r2"]
        if score > best_score:
            best_score, best_alpha, best_oof = score, alpha, oof.copy()

    train_x, test_x = fit_transform(train_values, test_values)
    weights = fit_ridge(train_x, target, best_alpha)
    prediction = predict_ridge(test_x, weights)
    test_metrics = regression_metrics(test_target, prediction)
    test_metrics["spearman"] = spearman(test_target, prediction)
    cv_metrics = regression_metrics(target, best_oof)
    return {
        "Model": "+".join(feature_names) + (" (quadratic)" if polynomial else ""),
        "Features": ", ".join(feature_names),
        "Polynomial": polynomial,
        "NTrain": len(train_frame),
        "NTest": len(test_frame),
        "Alpha": best_alpha,
        "CV_R2": cv_metrics["r2"],
        "CV_MAE": cv_metrics["mae"],
        "Test_R2": test_metrics["r2"],
        "Test_MAE": test_metrics["mae"],
        "Test_RMSE": test_metrics["rmse"],
        "Test_Pearson": test_metrics["pearson"],
        "Test_Spearman": test_metrics["spearman"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tsv", required=True)
    parser.add_argument("--test_tsv", required=True)
    parser.add_argument("--target_col", default="glycohemoglobin_hba1c")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_train = read_condition_tsv(args.train_tsv)
    raw_test = read_condition_tsv(args.test_tsv)
    target, columns = resolve_columns(raw_train, args.target_col)
    test_target, test_columns = resolve_columns(raw_test, args.target_col)
    if target != test_target or columns != test_columns:
        raise ValueError("Training and test metadata columns differ.")
    train = numeric_frame(raw_train, target, columns)
    test = numeric_frame(raw_test, target, columns)

    body = ["waist", "hip", "height", "weight", "bmi"]
    definitions = [
        (["age"], False),
        (body, False),
        (["age", *body], False),
        (body, True),
        (["age", *body], True),
    ]
    rows = [evaluate_model(train, test, features, quadratic, args) for features, quadratic in definitions]
    results = pd.DataFrame(rows).sort_values("CV_R2", ascending=False).reset_index(drop=True)

    correlations = []
    for feature in ["age", *body]:
        pair = train[[feature, "target"]].dropna()
        correlations.append({
            "Feature": feature,
            "N": len(pair),
            "Pearson": float(pair.corr(method="pearson").iloc[0, 1]),
            "Spearman": float(pair.corr(method="spearman").iloc[0, 1]),
        })
    correlation_frame = pd.DataFrame(correlations).sort_values("Spearman", key=np.abs, ascending=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "hba1c_body_size_benchmark.csv", index=False)
    correlation_frame.to_csv(out_dir / "hba1c_feature_correlations.csv", index=False)
    manifest = {
        "target_column": target,
        "resolved_columns": columns,
        "train_tsv": args.train_tsv,
        "test_tsv": args.test_tsv,
    }
    (out_dir / "hba1c_body_size_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n========== HBA1C FEATURE CORRELATIONS (TRAIN ONLY) ==========", flush=True)
    print(correlation_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print("\n========== HBA1C BODY-SIZE BENCHMARK (FIXED TEST) ==========", flush=True)
    print(
        results[["Model", "CV_R2", "Test_R2", "Test_MAE", "Test_RMSE", "Test_Pearson", "Test_Spearman"]]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        flush=True,
    )
    print(f"[OUT] {out_dir / 'hba1c_body_size_benchmark.csv'}", flush=True)


if __name__ == "__main__":
    main()
