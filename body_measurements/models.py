"""Leakage-conscious baselines for predicting manual measurements in centimetres.

Only scan-derived numeric features belong in ``X``. The data preparation layer
is responsible for excluding metadata, identifiers, and target-derived features.
Each candidate is fitted on labelled training rows for its own target; validation
selects a model and the held-out test partition is used only for reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FORMAT_VERSION = 1
_PARTITIONS = ("train", "validation", "test")


def _numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with named scan geometry features.")
    if X.shape[1] == 0:
        raise ValueError("At least one scan geometry feature is required.")
    if not X.index.is_unique or not X.columns.is_unique:
        raise ValueError("Feature row identifiers and column names must be unique.")
    if not all(isinstance(name, str) for name in X.columns):
        raise ValueError("Feature names must be strings.")
    try:
        numeric = X.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Scan geometry features must be numeric.") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Scan geometry features contain missing or nonfinite values.")
    return numeric


def _align_rows(value: pd.DataFrame | pd.Series, X: pd.DataFrame, name: str):
    if not value.index.is_unique:
        raise ValueError(f"{name} row identifiers must be unique.")
    if len(value) != len(X) or not X.index.isin(value.index).all():
        raise ValueError(f"{name} must have exactly the same scan identifiers as X.")
    return value.reindex(X.index)


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    labelled = np.isfinite(actual)
    actual, predicted = actual[labelled], predicted[labelled]
    n = len(actual)
    result = dict.fromkeys(
        ("mae_cm", "rmse_cm", "r2", "bias_cm", "loa_lower_cm", "loa_upper_cm",
         "within_1_cm", "within_2_cm"), float("nan")
    )
    result["n"] = n
    if not n:
        return result
    residual = predicted - actual
    bias = float(residual.mean())
    result.update(
        mae_cm=float(np.abs(residual).mean()),
        rmse_cm=float(np.sqrt(np.square(residual).mean())),
        bias_cm=bias,
        within_1_cm=float((np.abs(residual) <= 1.0).mean()),
        within_2_cm=float((np.abs(residual) <= 2.0).mean()),
    )
    if n >= 2:
        spread = 1.96 * float(residual.std(ddof=1))
        result.update(loa_lower_cm=bias - spread, loa_upper_cm=bias + spread)
        denominator = float(np.square(actual - actual.mean()).sum())
        if denominator > 0:
            result["r2"] = float(1.0 - np.square(residual).sum() / denominator)
    return result


def fit_models(
    X: pd.DataFrame,
    y: pd.DataFrame,
    split: pd.Series,
    *,
    seed: int = 42,
    min_train_labels: int = 10,
    min_validation_labels: int = 3,
    n_estimators: int = 300,
    n_jobs: int = -1,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit training-only models and return bundle, metrics, predictions, selection.

    Targets must already be expressed in centimetres. Missing targets are masked
    independently. A target with insufficient training labels is skipped; with
    insufficient validation labels its training median is the selected fallback.
    Validation MAE selects among the median, four scaled Ridge regressions, and
    Extra Trees. Ties prefer the first candidate, starting with the median.
    No model is refitted after selection. Validation reports every candidate;
    test reports only the selected model and the median comparator. Metrics use
    labelled rows; predictions also include unlabelled rows from those splits.
    """
    X = _numeric_features(X)
    if not isinstance(y, pd.DataFrame) or not isinstance(split, pd.Series):
        raise TypeError("y must be a DataFrame and split must be a Series.")
    if not y.columns.is_unique or not all(isinstance(name, str) for name in y.columns):
        raise ValueError("Target names must be unique strings.")
    if y.shape[1] == 0:
        raise ValueError("At least one measurement target is required.")
    y = _align_rows(y, X, "Targets")
    split = _align_rows(split, X, "Split")
    try:
        y = y.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Measurement targets must be numeric centimetres or missing.") from exc
    if np.isinf(y.to_numpy()).any():
        raise ValueError("Measurement targets cannot contain infinite values.")
    if not split.isin(_PARTITIONS).all():
        raise ValueError("Split labels must be train, validation, or test with no missing values.")
    for name, value in (("min_train_labels", min_train_labels),
                        ("min_validation_labels", min_validation_labels),
                        ("n_estimators", n_estimators)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    train = split.eq("train")
    if not train.any():
        raise ValueError("A nonempty training partition is required.")

    selected_models: dict = {}
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[dict] = []
    for target in y.columns:
        labelled = y[target].notna()
        counts = {part: int((split.eq(part) & labelled).sum()) for part in _PARTITIONS}
        selection = {
            "target": target,
            "n_train": counts["train"],
            "n_validation": counts["validation"],
            "n_test": counts["test"],
            "selected_model": None,
            "validation_mae_cm": float("nan"),
        }
        if counts["train"] < min_train_labels:
            selection.update(
                status="skipped",
                reason=f"Only {counts['train']} training labels; need {min_train_labels}.",
            )
            selection_rows.append(selection)
            continue

        candidates = {"train_median": DummyRegressor(strategy="median")}
        for alpha in (0.1, 1, 10, 100):
            candidates[f"ridge_alpha_{alpha}"] = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        candidates["extra_trees"] = ExtraTreesRegressor(
            n_estimators=n_estimators, min_samples_leaf=2, random_state=seed, n_jobs=n_jobs
        )
        train_labelled = train & labelled
        for estimator in candidates.values():
            estimator.fit(X.loc[train_labelled], y.loc[train_labelled, target])

        # Complete selection before inspecting test outcomes or computing test metrics.
        validation = split.eq("validation")
        validation_predictions = {
            name: estimator.predict(X.loc[validation]) if validation.any() else np.empty(0)
            for name, estimator in candidates.items()
        }
        validation_metrics = {
            name: _regression_metrics(y.loc[validation, target].to_numpy(), predicted)
            for name, predicted in validation_predictions.items()
        }
        if counts["validation"] >= min_validation_labels:
            selected_name = min(candidates, key=lambda name: validation_metrics[name]["mae_cm"])
            selection.update(status="trained", reason="Selected by validation MAE.")
        else:
            selected_name = "train_median"
            selection.update(
                status="baseline_only_insufficient_validation",
                reason=(f"Only {counts['validation']} validation labels; need "
                        f"{min_validation_labels}. Using training median."),
            )
        selected_models[target] = candidates[selected_name]
        selection.update(
            selected_model=selected_name,
            validation_mae_cm=validation_metrics[selected_name]["mae_cm"],
        )
        selection_rows.append(selection)

        for part in ("validation", "test"):
            rows = split.eq(part)
            actual = y.loc[rows, target].to_numpy()
            for name, estimator in candidates.items():
                if part == "test" and name not in {"train_median", selected_name}:
                    continue
                if part == "validation":
                    predicted = validation_predictions[name]
                    scores = validation_metrics[name]
                else:
                    predicted = estimator.predict(X.loc[rows]) if rows.any() else np.empty(0)
                    scores = _regression_metrics(actual, predicted)
                metric_rows.append({
                    "target": target, "model": name, "split": part,
                    "selected": name == selected_name, **scores,
                })
                prediction_frames.append(pd.DataFrame({
                    "scan_id": X.index[rows], "target": target, "split": part,
                    "actual_cm": actual, "predicted_cm": predicted,
                    "model": name, "selected": name == selected_name,
                }))

    selection_table = pd.DataFrame(selection_rows)
    if not selected_models:
        reasons = "; ".join(f"{row['target']}: {row['reason']}" for row in selection_rows)
        raise ValueError(f"No measurement models could be fitted. {reasons}")
    bundle = {
        "format_version": FORMAT_VERSION,
        "feature_names": X.columns.tolist(),
        "target_names": list(selected_models),
        "target_units": "cm",
        "models": selected_models,
        "selection": selection_rows,
        "seed": seed,
    }
    return bundle, pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True), selection_table


def predict(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Predict centimetres, safely reordering the exact saved feature schema."""
    if bundle.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported body measurement model bundle format_version.")
    X = _numeric_features(X)
    expected = bundle["feature_names"]
    missing = [name for name in expected if name not in X.columns]
    additional = [name for name in X.columns if name not in expected]
    if missing or additional:
        raise ValueError(f"Feature schema mismatch; missing={missing}, additional={additional}.")
    X = X.loc[:, expected]
    if not len(X):
        return pd.DataFrame(index=X.index, columns=bundle["target_names"], dtype=float)
    return pd.DataFrame(
        {target: bundle["models"][target].predict(X) for target in bundle["target_names"]},
        index=X.index,
    )
