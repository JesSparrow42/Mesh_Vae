#!/usr/bin/env python3
"""Plain ordinary least-squares HbA1c benchmark on fixed metadata splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_hba1c_body_size import (
    numeric_frame,
    read_condition_tsv,
    regression_metrics,
    resolve_columns,
    spearman,
)


MODEL_FEATURES = [
    ("Mean baseline", []),
    ("BMI", ["bmi"]),
    ("Age", ["age"]),
    ("BMI + age", ["bmi", "age"]),
    ("Waist", ["waist"]),
    ("Waist + age", ["waist", "age"]),
    ("Body size", ["waist", "hip", "height", "weight", "bmi"]),
    ("Body size + age", ["waist", "hip", "height", "weight", "bmi", "age"]),
]


def fit_ols(values: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.concatenate([np.ones((len(values), 1)), values], axis=1)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ coefficients
    degrees_freedom = max(len(target) - design.shape[1], 1)
    residual_variance = float(residual @ residual) / degrees_freedom
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return coefficients, standard_errors


def predict_ols(values: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((len(values), 1)), values], axis=1)
    return design @ coefficients


def evaluate(name: str, features: list[str], train: pd.DataFrame, test: pd.DataFrame) -> dict:
    required = ["target", *features]
    train_complete = train[required].dropna().copy()
    test_complete = test[required].dropna().copy()
    y_train = train_complete["target"].to_numpy(np.float64)
    y_test = test_complete["target"].to_numpy(np.float64)

    if features:
        X_train = train_complete[features].to_numpy(np.float64)
        X_test = test_complete[features].to_numpy(np.float64)
        coefficients, standard_errors = fit_ols(X_train, y_train)
        train_prediction = predict_ols(X_train, coefficients)
        test_prediction = predict_ols(X_test, coefficients)
        coefficient_map = {"Intercept": float(coefficients[0])}
        interval_map = {}
        for index, feature in enumerate(features, start=1):
            coefficient_map[feature] = float(coefficients[index])
            interval_map[feature] = [
                float(coefficients[index] - 1.96 * standard_errors[index]),
                float(coefficients[index] + 1.96 * standard_errors[index]),
            ]
    else:
        mean = float(np.mean(y_train))
        train_prediction = np.full_like(y_train, mean)
        test_prediction = np.full_like(y_test, mean)
        coefficient_map = {"Intercept": mean}
        interval_map = {}

    train_metrics = regression_metrics(y_train, train_prediction)
    test_metrics = regression_metrics(y_test, test_prediction)
    return {
        "Model": name,
        "Features": ", ".join(features) if features else "intercept only",
        "NTrain": len(y_train),
        "NTest": len(y_test),
        "Train_R2": train_metrics["r2"],
        "Test_R2": test_metrics["r2"],
        "Test_MAE": test_metrics["mae"],
        "Test_RMSE": test_metrics["rmse"],
        "Test_Pearson": test_metrics["pearson"],
        "Test_Spearman": spearman(y_test, test_prediction),
        "Coefficients": json.dumps(coefficient_map, sort_keys=True),
        "Coefficient_95CI": json.dumps(interval_map, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tsv", required=True)
    parser.add_argument("--test_tsv", required=True)
    parser.add_argument("--target_col", default="glycohemoglobin_hba1c")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    train_raw = read_condition_tsv(args.train_tsv)
    test_raw = read_condition_tsv(args.test_tsv)
    target, columns = resolve_columns(train_raw, args.target_col)
    test_target, test_columns = resolve_columns(test_raw, args.target_col)
    if target != test_target or columns != test_columns:
        raise ValueError("Training and test metadata columns differ.")
    train = numeric_frame(train_raw, target, columns)
    test = numeric_frame(test_raw, target, columns)

    rows = [evaluate(name, features, train, test) for name, features in MODEL_FEATURES]
    results = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "hba1c_ordinary_least_squares.csv"
    results.to_csv(output, index=False)

    print("\n========== ORDINARY LEAST-SQUARES HBA1C RESULTS ==========", flush=True)
    print(
        results[
            ["Model", "NTrain", "NTest", "Train_R2", "Test_R2", "Test_MAE", "Test_RMSE", "Test_Pearson", "Test_Spearman"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        flush=True,
    )
    print("\n========== RAW-UNIT COEFFICIENTS ==========", flush=True)
    for row in rows:
        print(f"{row['Model']}: {row['Coefficients']}", flush=True)
    print(f"[OUT] {output}", flush=True)


if __name__ == "__main__":
    main()
