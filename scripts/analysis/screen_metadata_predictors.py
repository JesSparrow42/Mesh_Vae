#!/usr/bin/env python3
"""Screen metadata columns for atherosclerosis prediction without test leakage.

Each usable column is evaluated alone and together with age.  Model selection
and threshold fitting use out-of-fold predictions from the training split;
the supplied test split is evaluated once with that fixed model and threshold.
Outcome-derived columns are excluded conservatively.
"""
from __future__ import annotations

# Allow direct execution as well as ``python -m`` from the repository checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from meshvae.conditioning import (
    _detect_sep,
    _pick_col,
    _to_numeric_series,
    apply_target_missing_policy,
    clean_column_names,
    clean_stregkode,
    ensure_athero_columns,
)


def binary_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(pred).astype(bool)
    tp = int(np.sum(y & p))
    tn = int(np.sum((~y) & (~p)))
    fp = int(np.sum((~y) & p))
    fn = int(np.sum(y & (~p)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    return {
        "accuracy": (tp + tn) / max(len(y), 1),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, dict[str, float]]:
    values = np.unique(np.asarray(prob, dtype=np.float64))
    if len(values) <= 1:
        candidates = values if len(values) else np.asarray([0.5])
    else:
        candidates = np.concatenate([
            [values[0] - 1e-9],
            (values[:-1] + values[1:]) / 2.0,
            [values[-1] + 1e-9],
        ])
    best = None
    best_value = 0.5
    for value in candidates:
        metrics = binary_metrics(y_true, prob >= value)
        if best is None or metrics["balanced_accuracy"] > best["balanced_accuracy"]:
            best = metrics
            best_value = float(value)
    return best_value, best or binary_metrics(y_true, prob >= 0.5)


def evaluate_probs(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    out = binary_metrics(y_true, np.asarray(prob) >= threshold)
    y = np.asarray(y_true).astype(bool)
    scores = np.asarray(prob, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        out["auc"] = float("nan")
    else:
        order = np.argsort(scores)
        ranks = np.empty(len(scores), dtype=np.float64)
        i = 0
        while i < len(scores):
            j = i + 1
            while j < len(scores) and scores[order[j]] == scores[order[i]]:
                j += 1
            ranks[order[i:j]] = (i + 1 + j) / 2.0
            i = j
        out["auc"] = float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    out["threshold"] = float(threshold)
    return out


LEAKAGE_PATTERNS = (
    "athero",
    "plaque",
    "carotid",
    "carotis",
    "femoral",
    "stenosis",
    "calcification",
    "calcium_score",
    "agatston",
)
ID_PATTERNS = ("participant", "stregkode", "scan_id", "subject_id", "person_id")
AGE_NAMES = {"age", "alder", "agecut", "age_cut", "age_bin", "agebin", "age_group", "agegroup"}


def read_tsv(path: str, sep: str | None) -> pd.DataFrame:
    detected = _detect_sep(path, forced=sep)
    df = pd.read_csv(
        path,
        sep=detected,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        engine="python",
    )
    df = clean_column_names(df)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            values = df[col].astype(str).str.strip().str.strip('"').str.strip("'").str.strip()
            values = values.mask(values.str.lower().isin({"", "nan", "none", "null", "<na>"}))
            df[col] = values
    return df


def prepare_frame(path: str, args) -> tuple[pd.DataFrame, str, str | None]:
    df = read_tsv(path, args.sep)
    df = ensure_athero_columns(df, args.target_cols)
    df = apply_target_missing_policy(
        df, args.target_cols, args.target_missing_policy, context=path
    )
    id_col = args.id_col or _pick_col(
        df,
        ["participant_id", "participantid", "stregkode", "scan_id", "id"],
        fallback_contains="participant",
    )
    gender_col = args.gender_col or _pick_col(df, ["sexMF", "sex", "gender"], fallback_contains="sex")
    if id_col is None:
        raise KeyError(f"Could not identify participant ID column in {path}.")
    df["_screen_id"] = df[id_col].apply(clean_stregkode)
    df = df.dropna(subset=["_screen_id"]).drop_duplicates("_screen_id").copy()
    if args.only_gender:
        if gender_col is None:
            raise KeyError(f"--only_gender was set but no gender column was found in {path}.")
        gender = (
            df[gender_col].astype(str).str.strip().str.upper()
            .replace({"MALE": "M", "FEMALE": "F"})
        )
        before = len(df)
        df = df.loc[gender == args.only_gender].copy()
        print(f"[META] {path}: gender={args.only_gender} kept {len(df)}/{before} rows.", flush=True)
    df["_target"] = (df[args.target_cols].astype(float).max(axis=1) > 0.5).astype(np.int64)
    return df.reset_index(drop=True), id_col, gender_col


def exclusion_reason(
    col: str,
    train: pd.DataFrame,
    protected: set[str],
    args,
) -> tuple[str | None, str | None]:
    lower = str(col).lower()
    if col in protected or lower.startswith("_screen") or lower == "_target":
        return "protected/id/target", None
    if lower in AGE_NAMES or re.search(r"(^|_)age($|_)", lower):
        return "age baseline column", None
    if any(pattern in lower for pattern in LEAKAGE_PATTERNS):
        return "possible outcome leakage", None
    if any(pattern in lower for pattern in ID_PATTERNS) or lower in {"id", "name", "path", "filename"}:
        return "identifier-like column", None

    series = train[col]
    nonmissing = series.notna()
    missing_fraction = 1.0 - float(nonmissing.mean())
    unique = int(series[nonmissing].nunique(dropna=True))
    if missing_fraction > args.max_missing_fraction:
        return f"missing fraction {missing_fraction:.3f} exceeds limit", None
    if unique < 2:
        return "constant or empty", None

    numeric = _to_numeric_series(series)
    numeric_fraction = float(numeric.notna().sum() / max(int(nonmissing.sum()), 1))
    if numeric_fraction >= args.numeric_fraction:
        return None, "numeric"
    if unique > args.max_categories:
        return f"categorical cardinality {unique} exceeds limit", None
    if unique / max(len(series), 1) > args.max_category_fraction:
        return f"categorical uniqueness fraction {unique / max(len(series), 1):.3f} exceeds limit", None
    return None, "categorical"


class TabularEncoder:
    def __init__(self, columns: list[str], types: dict[str, str]):
        self.columns = list(columns)
        self.types = dict(types)
        self.numeric_stats: dict[str, tuple[float, float]] = {}
        self.levels: dict[str, list[str]] = {}

    def fit(self, frame: pd.DataFrame) -> "TabularEncoder":
        for col in self.columns:
            if self.types[col] == "numeric":
                values = _to_numeric_series(frame[col]).to_numpy(np.float64)
                finite = values[np.isfinite(values)]
                median = float(np.median(finite)) if len(finite) else 0.0
                filled = np.where(np.isfinite(values), values, median)
                mean = float(filled.mean())
                std = float(filled.std())
                self.numeric_stats[col] = (median, std if np.isfinite(std) and std > 1e-8 else 1.0)
            else:
                values = frame[col].fillna("__MISSING__").astype(str)
                counts = values.value_counts()
                levels = sorted(str(level) for level, count in counts.items() if int(count) >= 2)
                self.levels[col] = levels or ["__MISSING__"]
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks = []
        for col in self.columns:
            if self.types[col] == "numeric":
                median, std = self.numeric_stats[col]
                values = _to_numeric_series(frame[col]).to_numpy(np.float64)
                values = np.where(np.isfinite(values), values, median)
                # Center on the fitted median.  Median centering is stable for
                # skewed laboratory values; the intercept absorbs any offset.
                blocks.append(((values - median) / std).reshape(-1, 1))
            else:
                values = frame[col].fillna("__MISSING__").astype(str).to_numpy()
                levels = self.levels[col]
                blocks.append(np.stack([(values == level).astype(np.float64) for level in levels], axis=1))
        return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(frame), 0), dtype=np.float64)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 80) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.concatenate([np.ones((len(X), 1), dtype=np.float64), X], axis=1)
    weights = np.zeros(design.shape[1], dtype=np.float64)
    n_pos = max(float(y.sum()), 1.0)
    n_neg = max(float(len(y) - y.sum()), 1.0)
    sample_weight = np.where(y > 0.5, len(y) / (2.0 * n_pos), len(y) / (2.0 * n_neg))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    for _ in range(max_iter):
        logits = np.clip(design @ weights, -35.0, 35.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        curvature = sample_weight * np.maximum(prob * (1.0 - prob), 1e-6)
        gradient = design.T @ (sample_weight * (prob - y)) + penalty @ weights
        hessian = (design.T * curvature) @ design + penalty + np.eye(design.shape[1]) * 1e-8
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        weights -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return weights


def predict_logistic(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((len(X), 1), dtype=np.float64), np.asarray(X, dtype=np.float64)], axis=1)
    logits = np.clip(design @ weights, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-logits))


def stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    fold_parts: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in [0, 1]:
        idx = np.flatnonzero(np.asarray(y) == cls)
        rng.shuffle(idx)
        for fold, part in enumerate(np.array_split(idx, n_splits)):
            fold_parts[fold].extend(part.tolist())
    return [np.asarray(sorted(part), dtype=np.int64) for part in fold_parts]


def fit_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    types: dict[str, str],
    args,
) -> dict:
    y_train = train["_target"].to_numpy(np.int64)
    y_test = test["_target"].to_numpy(np.int64)
    min_class = int(np.bincount(y_train, minlength=2).min())
    n_splits = min(int(args.folds), min_class)
    if n_splits < 2:
        raise ValueError(f"Too few examples in one target class for cross-validation: {np.bincount(y_train)}")
    oof = np.full(len(train), np.nan, dtype=np.float64)
    all_idx = np.arange(len(train), dtype=np.int64)
    for val_idx in stratified_folds(y_train, n_splits, args.seed):
        fit_idx = np.setdiff1d(all_idx, val_idx, assume_unique=True)
        encoder = TabularEncoder(columns, types).fit(train.iloc[fit_idx])
        X_fit = encoder.transform(train.iloc[fit_idx])
        X_val = encoder.transform(train.iloc[val_idx])
        weights = fit_logistic(X_fit, y_train[fit_idx])
        oof[val_idx] = predict_logistic(X_val, weights)
    if not np.isfinite(oof).all():
        raise RuntimeError("Internal cross-validation error: some training rows were not scored.")
    threshold, _ = best_threshold(y_train, oof)
    cv_metrics = evaluate_probs(y_train, oof, threshold)
    encoder = TabularEncoder(columns, types).fit(train)
    X_train = encoder.transform(train)
    X_test = encoder.transform(test)
    weights = fit_logistic(X_train, y_train)
    test_prob = predict_logistic(X_test, weights)
    test_metrics = evaluate_probs(y_test, test_prob, threshold)
    return {
        "cv": cv_metrics,
        "test": test_metrics,
        "threshold": float(threshold),
        "n_splits": int(n_splits),
    }


def metric_fields(prefix: str, result: dict) -> dict:
    cv = result["cv"]
    test = result["test"]
    return {
        f"{prefix}CV_AUC": cv["auc"],
        f"{prefix}CV_BalancedAccuracy": cv["balanced_accuracy"],
        f"{prefix}Test_AUC": test["auc"],
        f"{prefix}Test_Accuracy": test["accuracy"],
        f"{prefix}Test_Sensitivity": test["sensitivity"],
        f"{prefix}Test_Specificity": test["specificity"],
        f"{prefix}Test_BalancedAccuracy": test["balanced_accuracy"],
        f"{prefix}Test_F1": test["f1"],
        f"{prefix}Threshold": result["threshold"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--test_tsv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--only_gender", choices=["M", "F"], default="F")
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--gender_col", default=None)
    ap.add_argument("--age_col", default=None)
    ap.add_argument("--sep", default=None)
    ap.add_argument("--target_cols", nargs=4, default=["ho_carotis", "ve_carotis", "ho_fem", "ve_fem"])
    ap.add_argument("--target_missing_policy", choices=["drop", "zero", "error"], default="drop")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_missing_fraction", type=float, default=0.50)
    ap.add_argument("--numeric_fraction", type=float, default=0.80)
    ap.add_argument("--max_categories", type=int, default=30)
    ap.add_argument("--max_category_fraction", type=float, default=0.10)
    args = ap.parse_args()

    train, train_id, train_gender = prepare_frame(args.train_tsv, args)
    test, test_id, test_gender = prepare_frame(args.test_tsv, args)
    overlap = set(train["_screen_id"]) & set(test["_screen_id"])
    if overlap:
        raise ValueError(f"Train/test participant leakage: {len(overlap)} IDs occur in both splits.")
    if train["_target"].nunique() != 2 or test["_target"].nunique() != 2:
        raise ValueError("Both train and test splits must contain positive and negative targets.")

    common_columns = [c for c in train.columns if c in test.columns]
    age_col = args.age_col or _pick_col(train, ["age", "alder"], fallback_contains="age")
    if age_col is None or age_col not in test.columns:
        raise KeyError("A continuous age column is required for the age-adjusted screening baseline.")
    age_numeric = _to_numeric_series(train[age_col])
    if age_numeric.notna().mean() < args.numeric_fraction:
        raise ValueError(f"Selected age column {age_col!r} is not sufficiently numeric.")
    train[age_col] = _to_numeric_series(train[age_col])
    test[age_col] = _to_numeric_series(test[age_col])

    protected = set(args.target_cols) | {train_id, test_id, "_screen_id", "_target"}
    if train_gender:
        protected.add(train_gender)
    if test_gender:
        protected.add(test_gender)
    candidates: list[str] = []
    types: dict[str, str] = {age_col: "numeric"}
    exclusions = []
    for col in common_columns:
        reason, feature_type = exclusion_reason(col, train, protected, args)
        if reason is None and feature_type is not None:
            candidates.append(col)
            types[col] = feature_type
        else:
            exclusions.append({"Column": col, "Reason": reason or "unknown"})

    for col in candidates:
        if types[col] == "numeric":
            train[col] = _to_numeric_series(train[col])
            test[col] = _to_numeric_series(test[col])

    print(f"[SCREEN] age={age_col} candidates={len(candidates)} excluded={len(exclusions)}", flush=True)
    age_result = fit_evaluate(train, test, [age_col], types, args)
    rows = [{
        "RankBasis": "baseline",
        "Column": "__AGE_ONLY__",
        "Type": "numeric",
        "MissingFractionTrain": float(train[age_col].isna().mean()),
        "UniqueTrain": int(train[age_col].nunique(dropna=True)),
        **metric_fields("AgeOnly_", age_result),
    }]

    for index, col in enumerate(candidates, start=1):
        print(f"[SCREEN] {index}/{len(candidates)} {col} ({types[col]})", flush=True)
        only = fit_evaluate(train, test, [col], types, args)
        adjusted = fit_evaluate(train, test, [age_col, col], types, args)
        row = {
            "RankBasis": "age-adjusted CV delta AUC",
            "Column": col,
            "Type": types[col],
            "MissingFractionTrain": float(train[col].isna().mean()),
            "UniqueTrain": int(train[col].nunique(dropna=True)),
            **metric_fields("FeatureOnly_", only),
            **metric_fields("AgePlusFeature_", adjusted),
            "AgePlusFeature_CV_DeltaAUC": adjusted["cv"]["auc"] - age_result["cv"]["auc"],
            "AgePlusFeature_Test_DeltaAUC": adjusted["test"]["auc"] - age_result["test"]["auc"],
        }
        rows.append(row)

    if candidates:
        all_result = fit_evaluate(train, test, [age_col, *candidates], types, args)
        rows.append({
            "RankBasis": "combined",
            "Column": "__AGE_PLUS_ALL_SAFE_METADATA__",
            "Type": "mixed",
            "MissingFractionTrain": np.nan,
            "UniqueTrain": np.nan,
            **metric_fields("AgePlusFeature_", all_result),
            "AgePlusFeature_CV_DeltaAUC": all_result["cv"]["auc"] - age_result["cv"]["auc"],
            "AgePlusFeature_Test_DeltaAUC": all_result["test"]["auc"] - age_result["test"]["auc"],
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    rank_col = "AgePlusFeature_CV_DeltaAUC"
    result_df = result_df.sort_values(rank_col, ascending=False, na_position="last").reset_index(drop=True)
    result_path = out_dir / "metadata_predictor_results.csv"
    exclusions_path = out_dir / "metadata_predictor_exclusions.csv"
    markdown_path = out_dir / "metadata_predictor_results.md"
    manifest_path = out_dir / "metadata_predictor_manifest.json"
    result_df.to_csv(result_path, index=False)
    pd.DataFrame(exclusions).to_csv(exclusions_path, index=False)

    display = result_df.loc[result_df["Column"] != "__AGE_ONLY__"].head(25)
    lines = [
        "# Metadata Predictor Screening",
        "",
        "Columns are ranked by out-of-fold training AUC improvement over age. The held-out test split was not used for ranking or threshold selection.",
        "",
        "| Column | Type | CV AUC with age | CV delta vs age | Test AUC with age | Test balanced accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in display.to_dict(orient="records"):
        def fmt(key: str) -> str:
            value = row.get(key, np.nan)
            return f"{float(value):.4f}" if pd.notna(value) else ""
        lines.append(
            f"| {row['Column']} | {row['Type']} | {fmt('AgePlusFeature_CV_AUC')} | "
            f"{fmt('AgePlusFeature_CV_DeltaAUC')} | {fmt('AgePlusFeature_Test_AUC')} | "
            f"{fmt('AgePlusFeature_Test_BalancedAccuracy')} |"
        )
    lines.extend([
        "",
        "Treat small improvements cautiously because many columns were screened. Outcome-derived variables were excluded before fitting.",
        "",
    ])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "train_tsv": args.train_tsv,
        "test_tsv": args.test_tsv,
        "age_col": age_col,
        "target_cols": args.target_cols,
        "target_missing_policy": args.target_missing_policy,
        "only_gender": args.only_gender,
        "n_train": len(train),
        "n_test": len(test),
        "n_candidates": len(candidates),
        "ranking": "out-of-fold training delta AUC over age-only",
    }, indent=2), encoding="utf-8")
    print(f"[OUT] results   -> {result_path}", flush=True)
    print(f"[OUT] report    -> {markdown_path}", flush=True)
    print(f"[OUT] exclusions -> {exclusions_path}", flush=True)


if __name__ == "__main__":
    main()
