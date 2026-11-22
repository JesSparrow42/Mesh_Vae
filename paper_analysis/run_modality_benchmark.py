#!/usr/bin/env python3
"""Clean held-out comparison of age, BMI, and scan-derived latent features.

This script keeps the classifier and held-out participants fixed while only
changing the available input features.  It is intended to replace comparisons
that mixed different neural architectures with different input modalities.

The primary paper contrast is ``age_bmi_scan`` minus ``age_bmi``, evaluated
separately in women and men with a paired participant bootstrap.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .bootstrap_metrics import paired_difference, summarize_model
    from .data_utils import (
        AGE_ALIASES,
        BMI_ALIASES,
        GENDER_ALIASES,
        clean_participant_id,
        find_column,
        latent_columns,
        normalize_gender,
        numeric,
        read_table,
        resolve_binary_target,
        resolve_id_column,
        resolve_numeric_column,
    )
except ImportError:
    from bootstrap_metrics import paired_difference, summarize_model
    from data_utils import (
        AGE_ALIASES,
        BMI_ALIASES,
        GENDER_ALIASES,
        clean_participant_id,
        find_column,
        latent_columns,
        normalize_gender,
        numeric,
        read_table,
        resolve_binary_target,
        resolve_id_column,
        resolve_numeric_column,
    )


MODEL_FEATURES = {
    "age": ("age",),
    "bmi": ("bmi",),
    "scan": ("scan",),
    "age_bmi": ("age", "bmi"),
    "age_scan": ("age", "scan"),
    "bmi_scan": ("bmi", "scan"),
    "age_bmi_scan": ("age", "bmi", "scan"),
}
DISPLAY_NAMES = {
    "age": "Age",
    "bmi": "BMI",
    "scan": "Scan latent",
    "age_bmi": "Age + BMI",
    "age_scan": "Age + scan",
    "bmi_scan": "BMI + scan",
    "age_bmi_scan": "Age + BMI + scan",
}


def load_latent(path: Path, split: str, explicit_id: str | None) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path, dtype=str)
    if "split" in frame.columns:
        split_mask = frame["split"].str.strip().str.lower() == split.lower()
        frame = frame.loc[split_mask].copy()
        if frame.empty:
            raise ValueError(f"No {split} latent rows in {path}.")
    id_column = resolve_id_column(frame, explicit_id)
    columns = latent_columns(frame)
    out = pd.DataFrame({"participant_id": frame[id_column].map(clean_participant_id)})
    for column in columns:
        out[column] = pd.to_numeric(frame[column], errors="coerce")
    out = out.dropna(subset=["participant_id"]).copy()
    if out["participant_id"].duplicated().any():
        examples = out.loc[out["participant_id"].duplicated(False), "participant_id"].head().tolist()
        raise ValueError(f"Duplicate participant IDs in {path}: {examples}")
    return out.reset_index(drop=True), columns


def load_latent_files(paths: list[Path], split: str, explicit_id: str | None) -> tuple[pd.DataFrame, list[str]]:
    loaded = [load_latent(path, split, explicit_id) for path in paths]
    common = sorted(set.intersection(*(set(columns) for _, columns in loaded)))
    if not common:
        raise ValueError(f"No common latent columns across {split} files: {paths}")
    frame = pd.concat(
        [item[["participant_id", *common]] for item, _ in loaded], ignore_index=True
    )
    if frame["participant_id"].duplicated().any():
        examples = frame.loc[frame["participant_id"].duplicated(False), "participant_id"].head().tolist()
        raise ValueError(f"Duplicate IDs after combining {split} latent files: {examples}")
    return frame, common


def load_metadata(
    path: Path,
    target_column: str,
    id_column: str | None,
    age_column: str | None,
    bmi_column: str | None,
    gender_column: str | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = read_table(path)
    resolved_id = resolve_id_column(frame, id_column)
    resolved_age = resolve_numeric_column(frame, age_column, AGE_ALIASES, "age")
    resolved_bmi = resolve_numeric_column(frame, bmi_column, BMI_ALIASES, "BMI")
    resolved_gender = (
        gender_column if gender_column in frame.columns else find_column(frame, GENDER_ALIASES)
    )
    if resolved_gender is None:
        raise KeyError(f"Could not find sex/gender column in {path}")
    out = pd.DataFrame(
        {
            "participant_id": frame[resolved_id].map(clean_participant_id),
            "age": numeric(frame[resolved_age]),
            "bmi": numeric(frame[resolved_bmi]),
            "gender": normalize_gender(frame[resolved_gender]),
            "target": resolve_binary_target(frame, target_column),
        }
    )
    out = out.dropna(subset=["participant_id"]).copy()
    if out["participant_id"].duplicated().any():
        examples = out.loc[out["participant_id"].duplicated(False), "participant_id"].head().tolist()
        raise ValueError(f"Duplicate participant IDs in metadata {path}: {examples}")
    resolved = {
        "participant_id": str(resolved_id),
        "age": str(resolved_age),
        "bmi": str(resolved_bmi),
        "gender": str(resolved_gender),
        "target": str(target_column),
    }
    return out.reset_index(drop=True), resolved


def load_metadata_files(
    paths: list[Path],
    target_column: str,
    id_column: str | None,
    age_column: str | None,
    bmi_column: str | None,
    gender_column: str | None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    loaded = [
        load_metadata(
            path,
            target_column,
            id_column,
            age_column,
            bmi_column,
            gender_column,
        )
        for path in paths
    ]
    frame = pd.concat([item for item, _ in loaded], ignore_index=True)
    if frame["participant_id"].duplicated().any():
        examples = frame.loc[frame["participant_id"].duplicated(False), "participant_id"].head().tolist()
        raise ValueError(f"Duplicate IDs after combining metadata files: {examples}")
    return frame, [resolved for _, resolved in loaded]


def merge_split(metadata: pd.DataFrame, latent: pd.DataFrame, latent_cols: list[str], split: str) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(metadata)
    merged = metadata.merge(latent, on="participant_id", how="inner", validate="one_to_one")
    finite_latent = np.isfinite(merged[latent_cols].to_numpy(dtype=float)).all(axis=1)
    complete = (
        merged[["age", "bmi", "target"]].notna().all(axis=1)
        & merged["gender"].isin(["F", "M"])
        & finite_latent
        & merged["target"].isin([0, 1])
    )
    complete_frame = merged.loc[complete].reset_index(drop=True)
    flow = {
        "split": split,
        "metadata_rows": before,
        "with_latent_match": len(merged),
        "complete_case_rows": len(complete_frame),
        "female_complete": int((complete_frame["gender"] == "F").sum()),
        "male_complete": int((complete_frame["gender"] == "M").sum()),
        "positive_complete": int(complete_frame["target"].sum()),
    }
    return complete_frame, flow


def feature_matrix(frame: pd.DataFrame, model: str, latent_cols: list[str], pooled: bool) -> np.ndarray:
    blocks: list[np.ndarray] = []
    features = MODEL_FEATURES[model]
    if "age" in features:
        blocks.append(frame[["age"]].to_numpy(dtype=float))
    if "bmi" in features:
        blocks.append(frame[["bmi"]].to_numpy(dtype=float))
    if "scan" in features:
        blocks.append(frame[latent_cols].to_numpy(dtype=float))
    if pooled:
        # Every pooled model receives the same sex indicator, so input comparisons
        # are not accidentally driven by unequal sex composition.
        blocks.append((frame[["gender"]].to_numpy() == "M").astype(float))
    return np.concatenate(blocks, axis=1)


def choose_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], probability, [1.0])))
    if len(candidates) > 1001:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 1001))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        prediction = probability >= threshold
        positive = target == 1
        sensitivity = float(np.sum(prediction & positive) / max(np.sum(positive), 1))
        specificity = float(np.sum((~prediction) & (~positive)) / max(np.sum(~positive), 1))
        score = 0.5 * (sensitivity + specificity)
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def fit_one_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model: str,
    latent_cols: list[str],
    pooled: bool,
    cv_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise ImportError(
            "run_modality_benchmark.py requires scikit-learn. On Gefion it is provided by the "
            "working module stack used by gefion_train_long.sh."
        ) from error

    x_train = feature_matrix(train, model, latent_cols, pooled)
    x_test = feature_matrix(test, model, latent_cols, pooled)
    y_train = train["target"].to_numpy(dtype=int)
    y_test = test["target"].to_numpy(dtype=int)
    minority = int(min(np.sum(y_train == 0), np.sum(y_train == 1)))
    folds = min(int(cv_folds), minority)
    if folds < 2:
        raise ValueError(f"Not enough cases in the training split for {model}: minority class n={minority}")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )
    grid = {"logistic__C": np.logspace(-4, 4, 17)}
    search = GridSearchCV(
        pipeline,
        grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_
    oof_probability = cross_val_predict(
        best,
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold = choose_threshold(y_train, oof_probability)
    probability = best.predict_proba(x_test)[:, 1]
    predictions = pd.DataFrame(
        {
            "participant_id": test["participant_id"].to_numpy(),
            "target": y_test,
            "probability": probability,
            "prediction": (probability >= threshold).astype(int),
        }
    )
    details = {
        "best_c": float(search.best_params_["logistic__C"]),
        "cv_auc": float(search.best_score_),
        "cv_folds": folds,
        "threshold_from_oof_train": threshold,
        "n_features": int(x_train.shape[1]),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_positive": int(y_test.sum()),
        "test_prevalence": float(y_test.mean()),
    }
    return predictions, details


def canonical_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_id": frame["participant_id"],
            "_target": frame["target"].astype(float),
            "_score": frame["probability"].astype(float),
            "_pred": frame["prediction"].astype(float),
        }
    )


def make_auc_figure(metrics: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib unavailable; skipping modality_auc_by_sex figure.", flush=True)
        return
    auc = metrics[metrics["metric"] == "auc"].copy()
    if auc.empty:
        return
    order = list(MODEL_FEATURES)
    labels = [DISPLAY_NAMES[item] for item in order]
    colors = {"F": "#D4A72C", "M": "#0F6F73", "ALL": "#735DA5"}
    genders = [gender for gender in ("F", "M", "ALL") if gender in set(auc["gender"])]
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    base = np.arange(len(order), dtype=float)
    offsets = np.linspace(-0.18, 0.18, max(len(genders), 1))
    for offset, gender in zip(offsets, genders):
        subset = auc[auc["gender"] == gender].set_index("model").reindex(order)
        estimate = subset["estimate"].to_numpy(dtype=float)
        low = subset["ci_low"].to_numpy(dtype=float)
        high = subset["ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            estimate,
            base + offset,
            xerr=np.vstack([estimate - low, high - estimate]),
            fmt="o",
            markersize=7,
            capsize=3,
            color=colors[gender],
            ecolor=colors[gender],
            label={"F": "Women", "M": "Men", "ALL": "Pooled"}[gender],
        )
    ax.axvline(0.5, color="#8696A0", linestyle="--", linewidth=1, label="Chance AUC")
    ax.set_yticks(base, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out ROC AUC (95% participant-bootstrap CI)")
    ax.set_xlim(0.45, 1.0)
    ax.set_title("Does scan morphology add information beyond age and BMI?")
    ax.grid(axis="x", color="#D9E3E8", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(output_dir / "modality_auc_by_sex.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "modality_auc_by_sex.pdf", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-latent", required=True, type=Path, nargs="+")
    parser.add_argument("--test-latent", required=True, type=Path, nargs="+")
    parser.add_argument("--train-tsv", required=True, type=Path, nargs="+")
    parser.add_argument("--test-tsv", required=True, type=Path, nargs="+")
    parser.add_argument("--target-col", default="anyplaq")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--age-col", default=None)
    parser.add_argument("--bmi-col", default=None)
    parser.add_argument("--gender-col", default=None)
    parser.add_argument("--genders", nargs="+", choices=("F", "M", "ALL"), default=("F", "M"))
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_FEATURES), default=tuple(MODEL_FEATURES))
    parser.add_argument("--reference", choices=tuple(MODEL_FEATURES), default="age_bmi")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_latent, train_latent_cols = load_latent_files(args.train_latent, "train", args.id_col)
    test_latent, test_latent_cols = load_latent_files(args.test_latent, "test", args.id_col)
    common_latent = sorted(set(train_latent_cols) & set(test_latent_cols))
    if not common_latent:
        raise ValueError("Train and test latent files have no common latent columns.")
    if len(common_latent) != len(train_latent_cols) or len(common_latent) != len(test_latent_cols):
        print(
            f"[WARN] Using {len(common_latent)} common latent columns; train={len(train_latent_cols)} "
            f"test={len(test_latent_cols)}.",
            flush=True,
        )
    train_latent = train_latent[["participant_id", *common_latent]]
    test_latent = test_latent[["participant_id", *common_latent]]

    train_meta, train_resolved = load_metadata_files(
        args.train_tsv,
        args.target_col,
        args.id_col,
        args.age_col,
        args.bmi_col,
        args.gender_col,
    )
    test_meta, test_resolved = load_metadata_files(
        args.test_tsv,
        args.target_col,
        args.id_col,
        args.age_col,
        args.bmi_col,
        args.gender_col,
    )
    train, train_flow = merge_split(train_meta, train_latent, common_latent, "train")
    test, test_flow = merge_split(test_meta, test_latent, common_latent, "test")
    overlap = sorted(set(train["participant_id"]) & set(test["participant_id"]))
    if overlap:
        raise ValueError(f"Train/test participant leakage detected (examples: {overlap[:10]}).")
    pd.DataFrame([train_flow, test_flow]).to_csv(output_dir / "cohort_flow.csv", index=False)

    metric_frames: list[pd.DataFrame] = []
    paired_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    specification_rows: list[dict[str, Any]] = []

    for gender_index, gender in enumerate(args.genders):
        pooled = gender == "ALL"
        train_group = train.copy() if pooled else train[train["gender"] == gender].reset_index(drop=True)
        test_group = test.copy() if pooled else test[test["gender"] == gender].reset_index(drop=True)
        if train_group["target"].nunique() != 2 or test_group["target"].nunique() != 2:
            raise ValueError(f"Gender={gender} does not contain both target classes in train and test.")
        print(
            f"[COHORT] gender={gender} train={len(train_group)} test={len(test_group)} "
            f"test_positive={int(test_group['target'].sum())}",
            flush=True,
        )
        canonical: dict[str, pd.DataFrame] = {}
        for model_index, model in enumerate(args.models):
            predictions, details = fit_one_model(
                train_group,
                test_group,
                model,
                common_latent,
                pooled,
                args.cv_folds,
                args.seed + gender_index * 1000 + model_index,
            )
            canonical[model] = canonical_predictions(predictions)
            model_predictions = predictions.copy()
            model_predictions.insert(1, "gender", gender)
            model_predictions.insert(2, "model", model)
            model_predictions["display_name"] = DISPLAY_NAMES[model]
            model_predictions["threshold_from_oof_train"] = details["threshold_from_oof_train"]
            prediction_frames.append(model_predictions)

            intervals = summarize_model(
                model,
                canonical[model],
                "binary",
                args.bootstraps,
                args.seed + 100000 + gender_index * 1000 + model_index,
                args.confidence,
                "auto",
            )
            intervals.insert(0, "gender", gender)
            intervals["display_name"] = DISPLAY_NAMES[model]
            for key, value in details.items():
                intervals[key] = value
            metric_frames.append(intervals)
            specification_rows.append({"gender": gender, "model": model, **details})
            auc = intervals.loc[intervals["metric"] == "auc", "estimate"].iloc[0]
            print(f"[MODEL] gender={gender} {model}: held-out AUC={auc:.4f}", flush=True)

        if args.reference not in canonical:
            raise ValueError(f"Reference model {args.reference} was not run for gender={gender}.")
        for model_index, model in enumerate(args.models):
            if model == args.reference:
                continue
            paired = paired_difference(
                args.reference,
                canonical[args.reference],
                model,
                canonical[model],
                "binary",
                args.bootstraps,
                args.seed + 200000 + gender_index * 1000 + model_index,
                args.confidence,
                "auto",
            )
            paired.insert(0, "gender", gender)
            paired_frames.append(paired)

    metrics = pd.concat(metric_frames, ignore_index=True)
    paired = pd.concat(paired_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    specifications = pd.DataFrame(specification_rows)
    metrics.to_csv(output_dir / "modality_metric_intervals.csv", index=False)
    paired.to_csv(output_dir / "modality_paired_differences.csv", index=False)
    predictions.to_csv(output_dir / "modality_test_predictions.csv", index=False)
    specifications.to_csv(output_dir / "modality_model_specifications.csv", index=False)
    make_auc_figure(metrics, output_dir)

    primary = paired[
        (paired["comparison_model"] == "age_bmi_scan") & (paired["metric"] == "auc")
    ].copy()
    primary.to_csv(output_dir / "primary_age_bmi_scan_vs_age_bmi.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": "Does scan morphology add held-out predictive information beyond age and BMI?",
        "primary_contrast": "age_bmi_scan minus age_bmi",
        "primary_metric": "held-out ROC AUC",
        "genders": list(args.genders),
        "models": list(args.models),
        "target": args.target_col,
        "train_columns": train_resolved,
        "test_columns": test_resolved,
        "latent_dimensions": len(common_latent),
        "complete_case_policy": "Same complete-case participants used for every modality within sex and split.",
        "threshold_policy": "Balanced-accuracy threshold chosen from out-of-fold training predictions only.",
        "interval_method": "Paired participant-level stratified percentile bootstrap.",
        "bootstrap_replicates": args.bootstraps,
        "seed": args.seed,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nPrimary comparison:\n" + primary.to_string(index=False), flush=True)
    print(f"[OUT] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
