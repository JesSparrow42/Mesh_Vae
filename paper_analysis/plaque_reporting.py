"""Held-out plaque metrics, simple covariate baselines, and exploratory associations."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from meshvae.conditioning import clean_stregkode


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else float("nan")


def metrics(y, prediction, kind="regression"):
    y, p = np.asarray(y, float), np.asarray(prediction, float)
    if len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Predictions must be aligned, finite, and complete.")
    if not len(y):
        return {"N": 0}
    if kind == "binary":
        n1, n0 = int((y == 1).sum()), int((y == 0).sum())
        ranks = pd.Series(p).rank().to_numpy()
        auc = (ranks[y == 1].sum() - n1*(n1+1)/2)/(n0*n1) if n0 and n1 else float("nan")
        order = np.argsort(-p, kind="stable")
        sy, sp = y[order], p[order]
        ends = np.r_[np.flatnonzero(np.diff(sp) != 0), len(sp)-1]
        tp = np.cumsum(sy)[ends]
        ap = float(np.sum(np.diff(np.r_[0, tp]) * tp/(ends+1))/n1) if n1 else float("nan")
        return {"N": len(y), "Positive": n1, "AUC": float(auc), "AveragePrecision": ap,
                "Brier": float(np.mean((y-p)**2)), "Accuracy": float(np.mean((p >= .5) == y))}
    error = p-y
    total = float(np.sum((y-y.mean())**2))
    return {"N": len(y), "R2": float(1-np.sum(error**2)/total) if total > 0 else float("nan"),
            "MAE": float(np.mean(abs(error))), "MedianAbsoluteError": float(np.median(abs(error))),
            "RMSE": float(np.sqrt(np.mean(error**2))), "Pearson": correlation(y, p),
            "Spearman": correlation(pd.Series(y).rank(), pd.Series(p).rank()),
            "LogMAE": float(np.mean(abs(np.log1p(y)-np.log1p(np.maximum(p, 0)))))}


def fit_predict(x, y, test_x, kind):
    """Fixed ridge strength; all scaling and model fitting use training rows only."""
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-6)
    x = np.c_[np.ones(len(x)), (x-mean)/std]
    test_x = np.c_[np.ones(len(test_x)), (test_x-mean)/std]
    penalty = np.eye(x.shape[1]); penalty[0, 0] = 0
    if kind == "binary":
        beta = np.zeros(x.shape[1])
        for _ in range(100):
            probability = 1/(1+np.exp(-np.clip(x@beta, -35, 35)))
            weight = np.maximum(probability*(1-probability), 1e-8)
            step = np.linalg.solve(x.T@(weight[:, None]*x)+penalty, x.T@(probability-y)+penalty@beta)
            beta -= step
            if np.max(abs(step)) < 1e-8:
                break
        return 1/(1+np.exp(-np.clip(test_x@beta, -35, 35)))
    beta = np.linalg.solve(x.T@x+penalty, x.T@y)
    return np.maximum(test_x@beta, 0)


def attach_geometry(frame, dataroot):
    with np.load(Path(dataroot)/"scan_geometry_features.npz", allow_pickle=True) as data:
        geometry = pd.DataFrame(data["features"], columns=[f"geometry_{x}" for x in data["feature_names"]])
        geometry["participant_id"] = [clean_stregkode(x) for x in data["names"]]
    if geometry.participant_id.duplicated().any():
        raise ValueError("Duplicate participant IDs in geometry sidecar")
    columns = [c for c in geometry if c.startswith("geometry_")]
    result = frame.merge(geometry, on="participant_id", how="left", validate="one_to_one")
    if not np.isfinite(result[columns].to_numpy(float)).all():
        raise ValueError("Selected participants have missing/nonfinite scan geometry.")
    return result, columns


def report(train, test, predictions, target, kind, outdir, dataroot, seed=42, bootstraps=500):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    train, geometry = attach_geometry(train, dataroot)
    test, _ = attach_geometry(test, dataroot)
    if predictions.participant_id.duplicated().any() or set(predictions.participant_id) != set(test.participant_id):
        raise ValueError("Diagnosis must provide exactly one prediction for every requested held-out participant.")
    predcol = "probability" if kind == "binary" else "prediction"
    joined = test.merge(predictions[["participant_id", predcol]], on="participant_id", validate="one_to_one")
    y, train_y = joined[target].to_numpy(float), train[target].to_numpy(float)
    model_predictions = {"meshvae": joined[predcol].to_numpy(float)}
    model_predictions["training_mean"] = np.full(len(y), train_y.mean())
    if kind != "binary":
        model_predictions["training_median"] = np.full(len(y), np.median(train_y))
    for name, cols in {"age": ["study_age"], "weight": ["study_weight_kg"],
                       "age_weight": ["study_age", "study_weight_kg"],
                       "age_geometry": ["study_age", *geometry],
                       "age_weight_geometry": ["study_age", "study_weight_kg", *geometry]}.items():
        model_predictions[name] = fit_predict(train[cols].to_numpy(float), train_y, joined[cols].to_numpy(float), kind)
    rows = [{"Model": name, **metrics(y, p, kind)} for name, p in model_predictions.items()]
    pd.DataFrame(rows).to_csv(outdir/"comparison.csv", index=False)
    pd.DataFrame({"participant_id": joined.participant_id, "target": y, **model_predictions}).to_csv(outdir/"predictions.csv", index=False)
    rng = np.random.default_rng(seed)
    primary = "AUC" if kind == "binary" else "MAE"
    intervals, differences = [], []
    if len(y) >= 2:
        for _ in range(bootstraps):
            idx = rng.integers(0, len(y), len(y))
            score = metrics(y[idx], model_predictions["meshvae"][idx], kind).get(primary, np.nan)
            baseline = metrics(y[idx], model_predictions["age_weight"][idx], kind).get(primary, np.nan)
            if np.isfinite(score):
                intervals.append(score)
            if np.isfinite(score) and np.isfinite(baseline):
                differences.append(score-baseline)
    ci = {"metric": primary, "requested_bootstraps": bootstraps, "valid_bootstraps": len(intervals),
          "meshvae_95_percentile_ci": np.quantile(intervals, [.025, .975]).tolist() if intervals else None,
          "meshvae_minus_age_weight_95_percentile_ci": np.quantile(differences, [.025, .975]).tolist() if differences else None,
          "interpretation": "fixed fitted models; participant bootstrap; no correction for multiple exploratory comparisons"}
    (outdir/"intervals.json").write_text(json.dumps(ci, indent=2), encoding="utf-8")
    if kind == "regression":
        association = []
        age_design = np.c_[np.ones(len(joined)), joined.study_age, joined.study_age**2]
        response = np.log1p(y)
        residual_y = response-age_design@np.linalg.lstsq(age_design, response, rcond=None)[0]
        for col in ["study_weight_kg", *geometry]:
            x = joined[col].to_numpy(float)
            residual_x = x-age_design@np.linalg.lstsq(age_design, x, rcond=None)[0]
            association.append({"Feature": col, "N": len(y), "Pearson_volume": correlation(x, y),
                                "Spearman_volume": correlation(pd.Series(x).rank(), pd.Series(y).rank()),
                                "PartialPearson_log1p_adjusted_age_age2": correlation(residual_x, residual_y)})
        pd.DataFrame(association).to_csv(outdir/"exploratory_body_size_associations.csv", index=False)
    return rows


def combine_hurdle(binary, positive, validation_positive, outdir):
    """Approximate population mean using validation residual smearing, never test residuals."""
    joined = binary[["participant_id", "probability"]].merge(positive, on="participant_id", validate="one_to_one")
    if len(joined) != len(binary) or len(joined) != len(positive):
        raise ValueError("Hurdle components must predict exactly the same participants")
    residual = validation_positive.target_transformed-validation_positive.prediction_transformed
    if len(residual) < 10 or not np.isfinite(residual).all():
        raise ValueError("Hurdle calibration needs at least ten finite positive validation residuals")
    with np.errstate(over="raise", invalid="raise"):
        smearing = float(np.exp(residual).mean())
        conditional_mean = np.maximum(np.exp(joined.prediction_transformed)*smearing-1, 0)
    joined["prediction"] = joined.probability*conditional_mean
    Path(outdir, "hurdle_calibration.json").write_text(json.dumps({
        "smearing_factor": smearing, "calibration_n": len(residual),
        "calibration_split": "positive internal validation; no held-out residuals",
        "assumptions": "unweighted presence probabilities; shared log-residual distribution; approximate conditional mean",
    }, indent=2), encoding="utf-8")
    return joined[["participant_id", "target", "prediction"]]
