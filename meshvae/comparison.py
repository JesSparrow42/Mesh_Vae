from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .conditioning import (
    apply_target_missing_policy,
    _clean_agecut_label,
    _detect_sep,
    _labels_from_agecut,
    _pick_col,
    _to_numeric_series,
    clean_column_names,
    clean_stregkode,
    ensure_athero_columns,
)
from .datasets import point_label_channels
from .scan_geometry import (
    ScanGeometrySpec,
    fit_scan_geometry_spec,
    load_aligned_scan_geometry,
    transform_scan_geometry,
)


@dataclass(frozen=True)
class Experiment:
    key: str
    display_model: str
    use_age: bool
    use_pointcloud: bool
    modification: str
    family: str
    conv: bool = False
    point_label: bool = False

    @property
    def slug(self) -> str:
        return self.key


def experiments() -> list[Experiment]:
    return [
        Experiment("vae_scan_age", "VAE", True, True, "", "vae_feature"),
        Experiment("vae_scan_noage", "VAE", False, True, "", "vae_feature"),
        Experiment("vae_age_only", "VAE", True, False, "", "vae_feature"),
        Experiment("vae_conv_scan_age", "VAE", True, True, "Conv", "vae_point", conv=True),
        Experiment("vae_conv_scan_noage", "VAE", False, True, "Conv", "vae_point", conv=True),
        Experiment("vae_conv_age_only", "VAE", True, False, "Conv", "vae_feature", conv=True),
        Experiment("vae_pointlabels_scan_age", "VAE", True, True, "Point Label", "vae_point", point_label=True),
        Experiment("vae_pointlabels_scan_noage", "VAE", False, True, "Point Label", "vae_point", point_label=True),
        Experiment("vae_pointlabels_age_only", "VAE", True, False, "Point Label", "vae_feature", point_label=True),
        Experiment("logistic_scan_age", "Logistic", True, True, "", "logistic"),
        Experiment("logistic_scan_noage", "Logistic", False, True, "", "logistic"),
        Experiment("logistic_age_only", "Logistic", True, False, "", "logistic"),
        Experiment("convolution_scan_age", "Convolution", True, True, "", "pointnet", conv=True),
        Experiment("convolution_scan_noage", "Convolution", False, True, "", "pointnet", conv=True),
        Experiment("convolution_age_only", "Convolution", True, False, "", "tabular_conv", conv=True),
    ]


@dataclass
class MetadataSpec:
    st_col: str
    gender_col: str | None
    age_col: str | None
    agecut_col: str | None
    age_mode: str
    age_labels: list[str]
    age_mean: float
    age_std: float
    target_cols: list[str]
    clinical_cols: list[str]
    clinical_mean: list[float]
    clinical_std: list[float]


def normalize_feature_source(feature_source: str) -> str:
    fs = str(feature_source or "scan").strip().lower().replace("-", "_")
    if fs == "hbalc":
        fs = "hba1c"
    if fs in {"both", "ldl_hbalc"}:
        fs = "ldl_hba1c"
    return fs


def make_device(gpu_ids: str) -> torch.device:
    gpu_ids = str(gpu_ids)
    if gpu_ids.strip() in {"", "-1", "cpu"} or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:0")


def read_condition_tsv(path: str, sep: str | None = None) -> pd.DataFrame:
    detected = _detect_sep(path, forced=sep)
    df = pd.read_csv(
        path,
        sep=detected,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        engine="python",
    )
    return clean_column_names(df)


def resolve_clinical_columns(df: pd.DataFrame, args) -> list[str]:
    explicit = list(getattr(args, "clinical_feature_cols", []) or [])
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise KeyError(f"Missing clinical feature columns {missing}. Available: {list(df.columns)[:80]}")
        return explicit

    fs = normalize_feature_source(getattr(args, "feature_source", "scan"))
    requested: list[tuple[str, str]] = []
    if fs == "scan":
        return []
    if fs == "ldl":
        requested = [("ldl_cholesterol", "ldl")]
    elif fs == "hba1c":
        requested = [("glycohemoglobin_hba1c", "hba1c")]
    elif fs == "ldl_hba1c":
        requested = [("ldl_cholesterol", "ldl"), ("glycohemoglobin_hba1c", "hba1c")]
    else:
        raise ValueError(f"Unsupported --feature_source={getattr(args, 'feature_source', None)}")

    cols = []
    for preferred, contains in requested:
        col = _pick_col(df, [preferred], fallback_contains=contains)
        if col is None:
            raise KeyError(f"Could not find clinical feature {preferred}. Available: {list(df.columns)[:80]}")
        cols.append(col)
    return cols


def age_feature_matrix(out: pd.DataFrame, args, spec: MetadataSpec | None, fit: bool) -> tuple[np.ndarray, MetadataSpec | None, str, str | None, str | None, list[str], float, float]:
    age_source = str(getattr(args, "age_source", "auto")).lower()
    agecut_col = None if spec is None else spec.agecut_col
    age_col = None if spec is None else spec.age_col

    if fit:
        agecut_col = getattr(args, "cond_agecut_col", None) or _pick_col(
            out,
            ["agecut", "age_cut", "age_bin", "agebin", "age_group", "agegroup"],
            fallback_contains="agecut",
        )
        age_col = getattr(args, "cond_age_col", None) or _pick_col(out, ["alder", "age"], fallback_contains="ald")
        if age_col is None:
            age_col = _pick_col(out, ["age"], fallback_contains="age")

    use_agecut = age_source == "agecut" or (age_source == "auto" and agecut_col is not None)
    if use_agecut:
        if agecut_col is None:
            raise KeyError("Age source is agecut, but no agecut column was found.")
        labels = _labels_from_agecut(out[agecut_col]) if fit or spec is None else list(spec.age_labels)
        if not labels:
            raise ValueError("No usable agecut labels found.")
        clean = out[agecut_col].apply(_clean_agecut_label)
        keep = clean.isin(labels)
        if not bool(keep.all()):
            dropped = int((~keep).sum())
            print(f"[META] Dropping {dropped} rows with unknown agecut labels.", flush=True)
            out.drop(out.index[~keep], inplace=True)
            clean = clean.loc[out.index]
        mat = np.zeros((len(out), len(labels)), dtype=np.float32)
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        for i, lab in enumerate(clean.tolist()):
            mat[i, label_to_idx[str(lab)]] = 1.0
        return mat, spec, "agecut", age_col, agecut_col, labels, 0.0, 1.0

    if age_col is None:
        raise KeyError("Could not find a numeric age column.")
    age = _to_numeric_series(out[age_col])
    keep = age.notna()
    if not bool(keep.all()):
        dropped = int((~keep).sum())
        print(f"[META] Dropping {dropped} rows with missing numeric age.", flush=True)
        out.drop(out.index[~keep], inplace=True)
        age = age.loc[out.index]
    if fit or spec is None:
        mean = float(age.mean())
        std = float(age.std(ddof=0))
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
    else:
        mean = float(spec.age_mean)
        std = float(spec.age_std)
    mat = ((age.to_numpy(np.float32) - mean) / std).reshape(-1, 1).astype(np.float32)
    return mat, spec, "age", age_col, None, ["age_z"], mean, std


def build_metadata_arrays(cond_tsv: str, args, spec: MetadataSpec | None = None, fit: bool = True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray | None, MetadataSpec]:
    df = read_condition_tsv(cond_tsv, sep=getattr(args, "cond_sep", None))

    target_cols = list(getattr(args, "target_cols", []) or ["ho_carotis", "ve_carotis", "ho_fem", "ve_fem"])
    df = ensure_athero_columns(df, target_cols)
    df = apply_target_missing_policy(
        df,
        target_cols,
        policy=getattr(args, "target_missing_policy", "drop"),
        context=str(cond_tsv),
    )

    if fit or spec is None:
        st_col = getattr(args, "cond_stregkode_col", None) or _pick_col(
            df, ["stregkode", "participant_id", "participantid", "id"], fallback_contains="stregkode"
        )
        if st_col is None:
            st_col = _pick_col(df, ["participant_id", "participantid"], fallback_contains="participant")
        gender_col = getattr(args, "cond_gender_col", None) or _pick_col(
            df, ["sex", "gender", "sexMF"], fallback_contains="sex"
        )
        clinical_cols = resolve_clinical_columns(df, args)
    else:
        st_col = spec.st_col
        gender_col = spec.gender_col
        clinical_cols = list(spec.clinical_cols)

    if st_col is None:
        raise KeyError(f"Could not find participant id column in {cond_tsv}. Available: {list(df.columns)[:80]}")

    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise KeyError(f"Missing target columns {missing_targets}. Available: {list(df.columns)[:80]}")

    use_cols = [st_col] + target_cols
    if gender_col is not None:
        use_cols.append(gender_col)
    if fit or spec is None:
        maybe_agecut = getattr(args, "cond_agecut_col", None) or _pick_col(
            df, ["agecut", "age_cut", "age_bin", "agebin", "age_group", "agegroup"], fallback_contains="agecut"
        )
        maybe_age = getattr(args, "cond_age_col", None) or _pick_col(df, ["alder", "age"], fallback_contains="ald")
        if maybe_age is None:
            maybe_age = _pick_col(df, ["age"], fallback_contains="age")
    else:
        maybe_agecut = spec.agecut_col
        maybe_age = spec.age_col
    for c in [maybe_agecut, maybe_age, *clinical_cols]:
        if c is not None and c not in use_cols:
            use_cols.append(c)

    out = df[use_cols].copy()
    out["code"] = out[st_col].apply(clean_stregkode)
    out = out.dropna(subset=["code"]).copy()

    if gender_col is not None:
        g = out[gender_col].astype(str).str.strip().str.upper().str.strip('"').str.strip("'")
        g = g.replace({"MALE": "M", "FEMALE": "F"})
        out["gender_clean"] = g
        only_gender = getattr(args, "only_gender", None)
        if only_gender:
            before = len(out)
            out = out[out["gender_clean"] == str(only_gender).upper()].copy()
            print(f"[META] gender={only_gender}: kept {len(out)}/{before} rows.", flush=True)
    elif getattr(args, "only_gender", None):
        raise KeyError("--only_gender was set, but no sex/gender column was found.")

    for c in target_cols:
        out[c] = _to_numeric_series(out[c]).clip(0, 1)

    age_mat, _, age_mode, age_col, agecut_col, age_labels, age_mean, age_std = age_feature_matrix(out, args, spec, fit)
    y = (out[target_cols].to_numpy(np.float32).max(axis=1) > 0.5).astype(np.float32)
    out["target_any"] = y

    clinical_norm = None
    clinical_mean: list[float] = [] if spec is None else list(spec.clinical_mean)
    clinical_std: list[float] = [] if spec is None else list(spec.clinical_std)
    if clinical_cols:
        X = np.stack([_to_numeric_series(out[c]).to_numpy(np.float32) for c in clinical_cols], axis=1)
        keep = np.isfinite(X).all(axis=1)
        if not bool(keep.all()):
            dropped = int((~keep).sum())
            print(f"[META] Dropping {dropped} rows with missing clinical features {clinical_cols}.", flush=True)
            out = out.iloc[np.flatnonzero(keep)].copy()
            y = y[keep]
            age_mat = age_mat[keep]
            X = X[keep]
        if fit or spec is None:
            mean = np.nanmean(X, axis=0).astype(np.float32)
            std = np.nanstd(X, axis=0).astype(np.float32)
            mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
            std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
            clinical_mean = mean.tolist()
            clinical_std = std.tolist()
        else:
            mean = np.asarray(spec.clinical_mean, dtype=np.float32)
            std = np.asarray(spec.clinical_std, dtype=np.float32)
        clinical_norm = ((X - mean[None, :]) / std[None, :]).astype(np.float32)

    out["_row_pos"] = np.arange(len(out), dtype=np.int64)
    out = out.drop_duplicates(subset=["code"]).reset_index(drop=True)
    keep_first = out["_row_pos"].to_numpy(dtype=np.int64)
    y = y[keep_first]
    age_mat = age_mat[keep_first]
    if clinical_norm is not None:
        clinical_norm = clinical_norm[keep_first]
    out = out.drop(columns=["_row_pos"])

    new_spec = MetadataSpec(
        st_col=str(st_col),
        gender_col=None if gender_col is None else str(gender_col),
        age_col=None if age_col is None else str(age_col),
        agecut_col=None if agecut_col is None else str(agecut_col),
        age_mode=age_mode,
        age_labels=list(age_labels),
        age_mean=float(age_mean),
        age_std=float(age_std),
        target_cols=list(target_cols),
        clinical_cols=list(clinical_cols),
        clinical_mean=list(clinical_mean),
        clinical_std=list(clinical_std),
    )
    pos = int(y.sum())
    print(f"[META] {cond_tsv}: usable={len(out)} target_pos={pos} target_neg={len(out) - pos}", flush=True)
    return out, age_mat.astype(np.float32), y.astype(np.float32), clinical_norm, new_spec


def features_to_pointcloud(X_norm: np.ndarray, n_points: int) -> np.ndarray:
    X_norm = np.asarray(X_norm, dtype=np.float32)
    n, k = X_norm.shape
    e = max(2, int(n_points))
    grid = np.linspace(-1.0, 1.0, e, dtype=np.float32)
    points = np.zeros((n, e, 3), dtype=np.float32)
    points[:, :, 2] = grid[None, :]
    if k >= 1:
        points[:, :, 0] = X_norm[:, 0:1]
    if k >= 2:
        points[:, :, 1] = X_norm[:, 1:2]
    if k > 2:
        points[:, :, 2] += 0.05 * X_norm[:, 2:].mean(axis=1, keepdims=True)
    return points


def load_points_from_npz(
    args,
    meta_df: pd.DataFrame,
    age_mat: np.ndarray,
    y: np.ndarray,
    scan_spec: ScanGeometrySpec | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, ScanGeometrySpec | None]:
    npz_path = Path(args.dataroot) / getattr(args, "npz_name", "simplified_padded_tensors.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Preprocessed NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    if "points" not in data.files:
        raise KeyError(f"{npz_path} has no 'points' array. Re-run preprocessing with point clouds enabled.")
    points_all = data["points"].astype(np.float32)
    names_all = data["names"]
    row_by_code = {str(code): i for i, code in enumerate(meta_df["code"].tolist())}

    keep_npz = []
    keep_meta = []
    names = []
    for i, nm in enumerate(names_all):
        code = clean_stregkode(nm)
        j = row_by_code.get(str(code))
        if j is None:
            continue
        keep_npz.append(i)
        keep_meta.append(j)
        names.append(str(nm))

    if not keep_npz:
        raise ValueError(
            "No overlap between NPZ names and condition TSV participant ids. "
            "This usually means the train/test TSVs and simplified_padded_tensors.npz are from different cohorts."
        )

    dropped = len(names_all) - len(keep_npz)
    print(f"[DATA] matched {len(keep_npz)}/{len(names_all)} NPZ scans to {getattr(args, 'cond_tsv', 'TSV')} (dropped {dropped}).", flush=True)
    keep_npz_arr = np.asarray(keep_npz, dtype=np.int64)
    matched_points = points_all[keep_npz_arr]
    scan_features = np.zeros((len(keep_npz_arr), 0), dtype=np.float32)
    if bool(getattr(args, "use_scan_geometry_features", False)):
        raw_features, raw_scales, feature_names, valid, _ = load_aligned_scan_geometry(
            args.dataroot,
            getattr(args, "scan_geometry_features_file", "scan_geometry_features.npz"),
            names_all,
            require_complete=False,
        )
        raw_features = raw_features[keep_npz_arr]
        raw_scales = raw_scales[keep_npz_arr]
        valid = valid[keep_npz_arr]
        if bool(getattr(args, "require_scan_geometry_features", False)) and not bool(valid.all()):
            examples = [names[i] for i in np.flatnonzero(~valid)[:20]]
            raise ValueError(
                f"{int((~valid).sum())} selected scans have no physical geometry sidecar row. "
                f"First missing names: {examples}"
            )
        if fit or scan_spec is None:
            scan_spec = fit_scan_geometry_spec(raw_features, raw_scales, feature_names)
        elif list(scan_spec.feature_names) != list(feature_names):
            raise ValueError("Training and test scan geometry feature columns differ.")
        scan_features, coordinate_factor, _ = transform_scan_geometry(raw_features, raw_scales, scan_spec)
        if bool(getattr(args, "restore_metric_scale", True)):
            matched_points *= coordinate_factor[:, None, None]
            print(
                f"[SCAN GEOMETRY] restored point-cloud scale with training median "
                f"{scan_spec.coordinate_scale:.6g} raw units.",
                flush=True,
            )
        print(f"[SCAN GEOMETRY] added {scan_features.shape[1]} scan-derived global features.", flush=True)

    return (
        matched_points,
        age_mat[np.asarray(keep_meta, dtype=np.int64)],
        y[np.asarray(keep_meta, dtype=np.int64)],
        names,
        scan_features,
        scan_spec,
    )


def build_arrays(
    args,
    fit: bool = True,
    spec: MetadataSpec | None = None,
    scan_spec: ScanGeometrySpec | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, MetadataSpec, ScanGeometrySpec | None]:
    meta_df, age_mat, y, clinical_norm, spec = build_metadata_arrays(args.cond_tsv, args, spec=spec, fit=fit)
    fs = normalize_feature_source(getattr(args, "feature_source", "scan"))
    if fs == "scan":
        points, age_mat, y, names, scan_features, scan_spec = load_points_from_npz(
            args, meta_df, age_mat, y, scan_spec=scan_spec, fit=fit
        )
    else:
        if clinical_norm is None:
            raise ValueError(f"--feature_source={fs} needs clinical features, but none were built.")
        points = features_to_pointcloud(clinical_norm, int(getattr(args, "clinical_points", 64)))
        names = meta_df["code"].astype(str).tolist()
        scan_features = np.zeros((len(points), 0), dtype=np.float32)
    return (
        points.astype(np.float32),
        age_mat.astype(np.float32),
        y.astype(np.float32),
        names,
        scan_features.astype(np.float32),
        spec,
        scan_spec,
    )


def roc_auc_manual(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prob, dtype=np.float64)
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and sorted_p[j] == sorted_p[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true) > 0.5
    p = np.asarray(pred).astype(bool)
    tp = int(np.sum(y & p))
    tn = int(np.sum((~y) & (~p)))
    fp = int(np.sum((~y) & p))
    fn = int(np.sum(y & (~p)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / max(len(y), 1)
    f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "precision": float(precision),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def evaluate_probs(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    out = binary_metrics(y_true, np.asarray(prob) >= float(threshold))
    out["auc"] = roc_auc_manual(y_true, prob)
    out["threshold"] = float(threshold)
    return out


def stratified_train_val_indices(y: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for cls in [0, 1]:
        idx = np.flatnonzero((y > 0.5) == bool(cls))
        rng.shuffle(idx)
        if len(idx) <= 1:
            train_idx.extend(idx.tolist())
            continue
        n_val = max(1, int(round(len(idx) * float(val_ratio))))
        n_val = min(n_val, len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    if not val_idx:
        all_idx = np.arange(len(y))
        rng.shuffle(all_idx)
        split = max(1, int(round(len(y) * (1.0 - float(val_ratio)))))
        train_idx = all_idx[:split].tolist()
        val_idx = all_idx[split:].tolist()
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def point_summary_features(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float32)
    xyz = p[:, :, :3]
    mean = xyz.mean(axis=1)
    std = xyz.std(axis=1)
    mn = xyz.min(axis=1)
    mx = xyz.max(axis=1)
    qs = np.quantile(xyz, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).transpose(1, 0, 2).reshape(len(xyz), -1)
    radius = np.linalg.norm(xyz, axis=2)
    rstats = np.stack(
        [radius.mean(axis=1), radius.std(axis=1), radius.min(axis=1), radius.max(axis=1)],
        axis=1,
    )
    # Cross-sectional profiles retain information about waist, hip, torso, and
    # limb shape that whole-body extrema cannot represent.  The longest robust
    # axis is treated as body height; each band records two horizontal widths,
    # radial spread, and occupancy.
    section_rows = []
    n_sections = 12
    for sample in xyz:
        lo = np.quantile(sample, 0.01, axis=0)
        hi = np.quantile(sample, 0.99, axis=0)
        spans = np.maximum(hi - lo, 1e-6)
        vertical = int(np.argmax(spans))
        horizontal = [axis for axis in range(3) if axis != vertical]
        height = np.clip((sample[:, vertical] - lo[vertical]) / spans[vertical], 0.0, 1.0)
        profile = []
        for section in range(n_sections):
            low = section / n_sections
            high = (section + 1) / n_sections
            mask = (height >= low) & (height < high if section + 1 < n_sections else height <= high)
            band = sample[mask]
            if len(band) < 5:
                profile.extend([0.0, 0.0, 0.0, 0.0])
                continue
            band_center = np.median(band[:, horizontal], axis=0)
            width_a = float(np.quantile(band[:, horizontal[0]], 0.95) - np.quantile(band[:, horizontal[0]], 0.05))
            width_b = float(np.quantile(band[:, horizontal[1]], 0.95) - np.quantile(band[:, horizontal[1]], 0.05))
            radial_band = np.linalg.norm(band[:, horizontal] - band_center[None, :], axis=1)
            profile.extend([
                width_a,
                width_b,
                float(np.quantile(radial_band, 0.90)),
                float(len(band) / max(len(sample), 1)),
            ])
        section_rows.append(profile)
    sections = np.asarray(section_rows, dtype=np.float32)
    return np.concatenate(
        [mean, std, mn, mx, qs.astype(np.float32), rstats, sections], axis=1
    ).astype(np.float32)


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0).astype(np.float32)
    std = np.nanstd(X, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.where(np.isfinite(X), X, mean[None, :])
    return ((X - mean[None, :]) / std[None, :]).astype(np.float32)


def body_region_channels(points: np.ndarray, n_regions: int = 6) -> np.ndarray:
    return point_label_channels(points, n_regions)


class BenchmarkPointDataset(Dataset):
    def __init__(
        self,
        points: np.ndarray,
        meta: np.ndarray,
        y: np.ndarray,
        n_points: int,
        scan_features: np.ndarray | None = None,
        include_labels: bool = False,
        deterministic: bool = False,
        seed: int = 0,
    ):
        self.points = np.asarray(points, dtype=np.float32)
        self.meta = np.asarray(meta, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.n_points = int(n_points)
        self.scan_features = (
            np.zeros((len(self.y), 0), dtype=np.float32)
            if scan_features is None
            else np.asarray(scan_features, dtype=np.float32)
        )
        self.include_labels = bool(include_labels)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self._visits = np.zeros(len(self.y), dtype=np.int64)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        p = self.points[idx]
        labels = body_region_channels(p) if self.include_labels else None
        if self.deterministic:
            sample_seed = self.seed + idx
        else:
            sample_seed = self.seed + idx + int(self._visits[idx]) * max(len(self), 1)
            self._visits[idx] += 1
        rng = np.random.default_rng(sample_seed)
        replace = p.shape[0] < self.n_points
        take = rng.choice(p.shape[0], size=self.n_points, replace=replace)
        p = p[take].astype(np.float32)
        if self.include_labels:
            p = np.concatenate([p[:, :3], labels[take]], axis=1).astype(np.float32)
        else:
            p = p[:, :3].astype(np.float32)
        if self.scan_features.shape[1] > 0:
            global_features = np.broadcast_to(
                self.scan_features[idx][None, :],
                (p.shape[0], self.scan_features.shape[1]),
            )
            p = np.concatenate([p, global_features], axis=1).astype(np.float32)
        return (
            torch.from_numpy(p),
            torch.from_numpy(self.meta[idx].astype(np.float32)),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        if meta is not None and meta.numel() > 0:
            x = torch.cat([x, meta], dim=1)
        return self.fc(x).squeeze(1)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        if meta is not None and meta.numel() > 0:
            x = torch.cat([x, meta], dim=1)
        return self.net(x).squeeze(1)


class TabularConvClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        if meta is not None and meta.numel() > 0:
            x = torch.cat([x, meta], dim=1)
        h = self.conv(x.unsqueeze(1)).flatten(1)
        return self.head(h).squeeze(1)


class FlexiblePointNet(nn.Module):
    def __init__(self, in_channels: int, meta_dim: int, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + meta_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode(self, points: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        h = self.stem(points.transpose(1, 2).contiguous())
        pooled = torch.cat([h.max(dim=2).values, h.mean(dim=2)], dim=1)
        if meta is not None and meta.numel() > 0:
            pooled = torch.cat([pooled, meta], dim=1)
        return pooled

    def forward(self, points: torch.Tensor, meta: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.encode(points, meta)).squeeze(1)


class FeatureVAEClassifier(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 32, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor | None = None):
        if meta is not None and meta.numel() > 0:
            x = torch.cat([x, meta], dim=1)
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 4.0)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        recon = self.decoder(z)
        logits = self.classifier(mu).squeeze(1)
        return logits, recon, mu, logvar, x


class ConvFeatureVAEClassifier(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 32, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        conv_hidden = max(16, hidden // 2)
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, conv_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_hidden, conv_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(conv_hidden * in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor | None = None):
        if meta is not None and meta.numel() > 0:
            x = torch.cat([x, meta], dim=1)
        h = self.encoder_conv(x.unsqueeze(1)).flatten(1)
        h = self.encoder_fc(h)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 4.0)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        recon = self.decoder(z)
        logits = self.classifier(mu).squeeze(1)
        return logits, recon, mu, logvar, x


class PointVAEClassifier(nn.Module):
    def __init__(self, in_channels: int, meta_dim: int, latent_dim: int = 64, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.encoder = FlexiblePointNet(in_channels, meta_dim, hidden=hidden, dropout=dropout)
        enc_dim = hidden * 2 + meta_dim
        self.mu = nn.Linear(enc_dim, latent_dim)
        self.logvar = nn.Linear(enc_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 12),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def point_target(points: torch.Tensor) -> torch.Tensor:
        xyz = points[:, :, :3]
        return torch.cat(
            [
                xyz.mean(dim=1),
                xyz.std(dim=1),
                xyz.min(dim=1).values,
                xyz.max(dim=1).values,
            ],
            dim=1,
        )

    def forward(self, points: torch.Tensor, meta: torch.Tensor | None = None):
        h = self.encoder.encode(points, meta)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 4.0)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        recon = self.decoder(z)
        logits = self.classifier(mu).squeeze(1)
        target = self.point_target(points)
        return logits, recon, mu, logvar, target


def loader_predictions(model: nn.Module, loader: DataLoader, device: torch.device, is_vae: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    ys = []
    with torch.no_grad():
        for x, meta, y in loader:
            x = x.to(device)
            meta = meta.to(device)
            if is_vae:
                logits = model(x, meta)[0]
            else:
                logits = model(x, meta)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, dict[str, float]]:
    vals = np.unique(np.asarray(prob, dtype=np.float64))
    if len(vals) <= 1:
        cands = vals if len(vals) else np.array([0.5])
    else:
        mids = (vals[:-1] + vals[1:]) / 2.0
        cands = np.concatenate([[vals[0] - 1e-9], mids, [vals[-1] + 1e-9]])
    best_thr = 0.5
    best_met: dict[str, float] | None = None
    for thr in cands:
        met = binary_metrics(y_true, prob >= thr)
        if best_met is None or met["balanced_accuracy"] > best_met["balanced_accuracy"]:
            best_thr = float(thr)
            best_met = met
    assert best_met is not None
    return best_thr, best_met


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    args,
    device: torch.device,
    is_vae: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    model.to(device)
    pos = float(np.sum(y_train > 0.5))
    neg = float(np.sum(y_train <= 0.5))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_score = -float("inf")
    best: dict[str, Any] = {}
    patience = int(args.patience)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, meta, y in train_loader:
            x = x.to(device)
            meta = meta.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            if is_vae:
                logits, recon, mu, logvar, target = model(x, meta)
                cls_loss = bce(logits, y)
                recon_loss = mse(recon, target)
                kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
                loss = cls_loss + args.vae_recon_weight * recon_loss + args.vae_beta * kl
            else:
                logits = model(x, meta)
                loss = bce(logits, y)
            if not torch.isfinite(loss):
                print("[WARN] non-finite batch skipped", flush=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        y_val, p_val = loader_predictions(model, val_loader, device, is_vae=is_vae)
        thr, _ = best_threshold(y_val, p_val)
        val_metrics = evaluate_probs(y_val, p_val, thr)
        score = val_metrics["auc"]
        if not np.isfinite(score):
            score = val_metrics["balanced_accuracy"]
        improved = score > best_score
        if improved:
            best_score = float(score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best = {
                "epoch": epoch,
                "threshold": float(thr),
                "val": val_metrics,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
            }
            patience = int(args.patience)
        else:
            patience -= 1
        if epoch == 1 or epoch % args.print_every == 0 or improved:
            print(
                f"    epoch={epoch:03d} loss={np.mean(losses) if losses else np.nan:.4f} "
                f"val_auc={val_metrics['auc']:.4f} val_bal={val_metrics['balanced_accuracy']:.4f}"
                f"{' *' if improved else ''}",
                flush=True,
            )
        if patience <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def make_feature_loaders(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    exp: Experiment,
    args,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    tr_idx, va_idx = stratified_train_val_indices(y_train, args.val_ratio, args.seed)
    mean, std = fit_standardizer(X_train[tr_idx])
    X_train_s = apply_standardizer(X_train, mean, std)
    X_test_s = apply_standardizer(X_test, mean, std)
    train_ds = TensorDataset(
        torch.from_numpy(X_train_s[tr_idx]),
        torch.zeros((len(tr_idx), 0), dtype=torch.float32),
        torch.from_numpy(y_train[tr_idx].astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_train_s[va_idx]),
        torch.zeros((len(va_idx), 0), dtype=torch.float32),
        torch.from_numpy(y_train[va_idx].astype(np.float32)),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test_s),
        torch.zeros((len(X_test_s), 0), dtype=torch.float32),
        torch.from_numpy(y_test.astype(np.float32)),
    )
    return (
        DataLoader(train_ds, batch_size=args.batch_size_tabular, shuffle=True, num_workers=args.num_threads),
        DataLoader(val_ds, batch_size=args.batch_size_tabular, shuffle=False, num_workers=args.num_threads),
        DataLoader(test_ds, batch_size=args.batch_size_tabular, shuffle=False, num_workers=args.num_threads),
        y_train[tr_idx],
        X_train_s,
    )


def make_point_loaders(
    points_train: np.ndarray,
    points_test: np.ndarray,
    scan_features_train: np.ndarray,
    scan_features_test: np.ndarray,
    meta_train: np.ndarray,
    meta_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    exp: Experiment,
    args,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    tr_idx, va_idx = stratified_train_val_indices(y_train, args.val_ratio, args.seed)
    include_labels = bool(exp.point_label)
    train_ds = BenchmarkPointDataset(
        points_train[tr_idx],
        meta_train[tr_idx],
        y_train[tr_idx],
        args.n_points,
        scan_features=scan_features_train[tr_idx],
        include_labels=include_labels,
        deterministic=False,
        seed=args.seed,
    )
    val_ds = BenchmarkPointDataset(
        points_train[va_idx],
        meta_train[va_idx],
        y_train[va_idx],
        args.n_points,
        scan_features=scan_features_train[va_idx],
        include_labels=include_labels,
        deterministic=True,
        seed=args.seed + 1000,
    )
    test_ds = BenchmarkPointDataset(
        points_test,
        meta_test,
        y_test,
        args.n_points,
        scan_features=scan_features_test,
        include_labels=include_labels,
        deterministic=True,
        seed=args.seed + 2000,
    )
    return (
        DataLoader(train_ds, batch_size=args.batch_size_point, shuffle=True, num_workers=args.num_threads),
        DataLoader(val_ds, batch_size=args.batch_size_point, shuffle=False, num_workers=args.num_threads),
        DataLoader(test_ds, batch_size=args.batch_size_point, shuffle=False, num_workers=args.num_threads),
        y_train[tr_idx],
    )


def run_one(exp: Experiment, data: dict[str, Any], args, device: torch.device, out_dir: Path) -> dict[str, Any]:
    print(f"\n========== {exp.slug} ==========", flush=True)
    if exp.point_label and not exp.use_pointcloud:
        print(
            "[N/A] Point-label modification requires body-scan points; this age-only row is retained "
            "in the requested 15-row design but is not trained.",
            flush=True,
        )
        return {
            "Experiment": exp.slug,
            "Model": exp.display_model,
            "Age": "Yes",
            "Pointcloud": "No",
            "Modification": exp.modification,
            "EffectiveInput": "age only",
            "Architecture": "not applicable: no points to label",
            "Status": "not_applicable",
            "Accuracy": np.nan,
            "Sensitivity": np.nan,
            "Specificity": np.nan,
            "BalancedAccuracy": np.nan,
            "AUC": np.nan,
            "Precision": np.nan,
            "F1": np.nan,
            "TP": np.nan,
            "TN": np.nan,
            "FP": np.nan,
            "FN": np.nan,
            "Threshold": np.nan,
            "BestEpoch": np.nan,
            "NTrain": int(len(data["y_train"])),
            "NTest": int(len(data["y_test"])),
            "Checkpoint": "",
        }

    meta_train = data["meta_train"] if exp.use_age else np.zeros((len(data["y_train"]), 0), dtype=np.float32)
    meta_test = data["meta_test"] if exp.use_age else np.zeros((len(data["y_test"]), 0), dtype=np.float32)
    is_vae = exp.family.startswith("vae")

    if exp.family in {"pointnet", "vae_point"} and exp.use_pointcloud:
        train_loader, val_loader, test_loader, y_fit = make_point_loaders(
            data["points_train"],
            data["points_test"],
            data["scan_features_train"],
            data["scan_features_test"],
            meta_train,
            meta_test,
            data["y_train"],
            data["y_test"],
            exp,
            args,
        )
        in_ch = 3 + (6 if exp.point_label else 0) + int(data["scan_features_train"].shape[1])
        if exp.family == "pointnet":
            model = FlexiblePointNet(in_channels=in_ch, meta_dim=meta_train.shape[1], hidden=args.hidden, dropout=args.dropout)
        else:
            model = PointVAEClassifier(in_channels=in_ch, meta_dim=meta_train.shape[1], latent_dim=args.latent_dim, hidden=args.hidden, dropout=args.dropout)
    else:
        feats_train = []
        feats_test = []
        if exp.use_pointcloud:
            feats_train.append(data["point_features_train"])
            feats_test.append(data["point_features_test"])
            if data["scan_features_train"].shape[1] > 0:
                feats_train.append(data["scan_features_train"])
                feats_test.append(data["scan_features_test"])
        if exp.use_age:
            feats_train.append(data["meta_train"])
            feats_test.append(data["meta_test"])
        if not feats_train:
            raise ValueError(f"Experiment has no input features: {exp}")
        X_train = np.concatenate(feats_train, axis=1).astype(np.float32)
        X_test = np.concatenate(feats_test, axis=1).astype(np.float32)
        train_loader, val_loader, test_loader, y_fit, X_train_s = make_feature_loaders(
            X_train, X_test, data["y_train"], data["y_test"], exp, args
        )
        if exp.family == "logistic":
            model = LinearClassifier(in_dim=X_train_s.shape[1])
        elif exp.family == "tabular_conv":
            model = TabularConvClassifier(in_dim=X_train_s.shape[1], hidden=max(16, args.hidden // 4), dropout=args.dropout)
        elif exp.family == "vae_feature":
            vae_cls = ConvFeatureVAEClassifier if exp.conv else FeatureVAEClassifier
            model = vae_cls(in_dim=X_train_s.shape[1], latent_dim=args.latent_dim, hidden=max(32, args.hidden // 2), dropout=args.dropout)
        else:
            model = MLPClassifier(in_dim=X_train_s.shape[1], hidden=args.hidden, dropout=args.dropout)

    model, best = train_model(model, train_loader, val_loader, y_fit, args, device, is_vae=is_vae)
    threshold = float(best.get("threshold", 0.5))
    y_test, p_test = loader_predictions(model, test_loader, device, is_vae=is_vae)
    test_metrics = evaluate_probs(y_test, p_test, threshold)
    y_val, p_val = loader_predictions(model, val_loader, device, is_vae=is_vae)
    val_metrics = evaluate_probs(y_val, p_val, threshold)

    ckpt_path = out_dir / f"{exp.slug}.pt"
    torch.save(
        {
            "experiment": exp.__dict__,
            "state_dict": model.state_dict(),
            "threshold": threshold,
            "val": val_metrics,
            "test": test_metrics,
        },
        ckpt_path,
    )
    row = {
        "Experiment": exp.slug,
        "Model": exp.display_model,
        "Age": "Yes" if exp.use_age else "No",
        "Pointcloud": "Yes" if exp.use_pointcloud else "No",
        "Modification": exp.modification,
        "EffectiveInput": "+".join(filter(None, ["pointcloud" if exp.use_pointcloud else "", "age" if exp.use_age else ""])),
        "Architecture": model.__class__.__name__,
        "Status": "completed",
        "Accuracy": test_metrics["accuracy"],
        "Sensitivity": test_metrics["sensitivity"],
        "Specificity": test_metrics["specificity"],
        "BalancedAccuracy": test_metrics["balanced_accuracy"],
        "AUC": test_metrics["auc"],
        "Precision": test_metrics["precision"],
        "F1": test_metrics["f1"],
        "TP": test_metrics["tp"],
        "TN": test_metrics["tn"],
        "FP": test_metrics["fp"],
        "FN": test_metrics["fn"],
        "Threshold": threshold,
        "BestEpoch": best.get("epoch", np.nan),
        "NTrain": int(len(data["y_train"])),
        "NTest": int(len(data["y_test"])),
        "Checkpoint": str(ckpt_path),
    }
    print(
        f"[RESULT] acc={row['Accuracy']:.4f} bal={row['BalancedAccuracy']:.4f} "
        f"auc={row['AUC']:.4f} sens={row['Sensitivity']:.4f} spec={row['Specificity']:.4f}",
        flush=True,
    )
    return row


def write_report(df: pd.DataFrame, out_dir: Path, args) -> None:
    csv_path = out_dir / "model_comparison_results.csv"
    md_path = out_dir / "model_comparison_report.md"
    df.to_csv(csv_path, index=False)
    table = df.to_string(index=False)
    lines = [
        "# Model Comparison Results",
        "",
        f"Train TSV: `{args.train_cond_tsv}`",
        f"Test TSV: `{args.test_cond_tsv}`",
        f"Dataroot: `{args.dataroot}`",
        "",
        "Point Label uses a geometry-derived pseudo body-region label per point.",
        "",
        "```text",
        table,
        "```",
        "",
    ]
    md_path.write_text("\n".join(lines))
    print(f"[OUT] wrote {csv_path}")
    print(f"[OUT] wrote {md_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataroot", default="/data/projects/3D_scans/ML/MQVAE")
    ap.add_argument("--train_cond_tsv", required=True)
    ap.add_argument("--test_cond_tsv", required=True)
    ap.add_argument(
        "--feature_source",
        default="scan",
        choices=["scan", "ldl", "hba1c", "hbalc", "ldl_hba1c", "ldl_hbalc", "both"],
        help="scan uses simplified_padded_tensors.npz; ldl/hba1c use TSV clinical values as a sanity-check input.",
    )
    ap.add_argument("--npz_name", default="simplified_padded_tensors.npz")
    ap.add_argument("--clinical_points", type=int, default=64)
    ap.add_argument("--clinical_feature_cols", nargs="*", default=None)
    ap.add_argument("--target_cols", nargs="+", default=["ho_carotis", "ve_carotis", "ho_fem", "ve_fem"])
    ap.add_argument("--target_missing_policy", choices=["drop", "zero", "error"], default="drop")
    ap.add_argument("--use_scan_geometry_features", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--scan_geometry_features_file", default="scan_geometry_features.npz")
    ap.add_argument("--require_scan_geometry_features", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--restore_metric_scale", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--only_gender", choices=["M", "F"], default="F")
    ap.add_argument("--cond_sep", default=None)
    ap.add_argument("--cond_stregkode_col", default=None)
    ap.add_argument("--cond_age_col", default=None)
    ap.add_argument("--cond_agecut_col", default=None)
    ap.add_argument("--cond_gender_col", default=None)
    ap.add_argument("--age_source", choices=["auto", "agecut", "age"], default="auto")
    ap.add_argument("--gpu_ids", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_points", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--batch_size_point", type=int, default=16)
    ap.add_argument("--batch_size_tabular", type=int, default=128)
    ap.add_argument("--num_threads", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--print_every", type=int, default=5)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--vae_beta", type=float, default=0.01)
    ap.add_argument("--vae_recon_weight", type=float, default=0.10)
    available_experiments = [exp.slug for exp in experiments()]
    ap.add_argument(
        "--experiments",
        nargs="+",
        choices=available_experiments,
        default=available_experiments,
        help="Subset of comparison experiment keys to run.",
    )
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse completed rows from model_comparison_results_partial.csv.",
    )
    args = ap.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    args.feature_source = normalize_feature_source(args.feature_source)
    args.cond_tsv = args.train_cond_tsv

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.dataroot) / "checkpoints" / args.name / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = make_device(args.gpu_ids)
    print(f"[DEVICE] {device}")

    (
        points_train,
        meta_train,
        y_train,
        names_train,
        scan_features_train,
        meta_spec,
        scan_spec,
    ) = build_arrays(args, fit=True)
    test_args = argparse.Namespace(**vars(args))
    test_args.cond_tsv = args.test_cond_tsv
    (
        points_test,
        meta_test,
        y_test,
        names_test,
        scan_features_test,
        _,
        _,
    ) = build_arrays(test_args, spec=meta_spec, scan_spec=scan_spec, fit=False)

    print(f"[DATA] train={len(y_train)} test={len(y_test)} point_shape={points_train.shape[1:]} meta_dim={meta_train.shape[1]}")
    data = {
        "points_train": points_train,
        "points_test": points_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
        "scan_features_train": scan_features_train,
        "scan_features_test": scan_features_test,
        "y_train": y_train,
        "y_test": y_test,
        "point_features_train": point_summary_features(points_train),
        "point_features_test": point_summary_features(points_test),
    }

    partial_path = out_dir / "model_comparison_results_partial.csv"
    completed_rows: dict[str, dict[str, Any]] = {}
    if args.resume and partial_path.exists():
        previous = pd.read_csv(partial_path)
        if "Experiment" in previous.columns:
            for row in previous.to_dict(orient="records"):
                if str(row.get("Status", "")) in {"completed", "not_applicable"}:
                    completed_rows[str(row["Experiment"])] = row
            print(f"[RESUME] found {len(completed_rows)} completed comparison rows in {partial_path}", flush=True)

    rows = []
    selected = [exp for exp in experiments() if exp.slug in set(args.experiments)]
    print(f"[RUN] selected {len(selected)} comparison experiments: {[e.slug for e in selected]}", flush=True)
    for exp in selected:
        if exp.slug in completed_rows:
            print(f"[RESUME] skipping completed {exp.slug}", flush=True)
            rows.append(completed_rows[exp.slug])
            continue
        try:
            rows.append(run_one(exp, data, args, device, out_dir))
        except Exception as e:
            print(f"[ERROR] {exp.slug}: {e}", flush=True)
            rows.append({
                "Experiment": exp.slug,
                "Model": exp.display_model,
                "Age": "Yes" if exp.use_age else "No",
                "Pointcloud": "Yes" if exp.use_pointcloud else "No",
                "Modification": exp.modification,
                "EffectiveInput": "+".join(filter(None, ["pointcloud" if exp.use_pointcloud else "", "age" if exp.use_age else ""])),
                "Architecture": "",
                "Status": "failed",
                "Accuracy": np.nan,
                "Sensitivity": np.nan,
                "Specificity": np.nan,
                "BalancedAccuracy": np.nan,
                "AUC": np.nan,
                "Precision": np.nan,
                "F1": np.nan,
                "TP": np.nan,
                "TN": np.nan,
                "FP": np.nan,
                "FN": np.nan,
                "Threshold": np.nan,
                "BestEpoch": np.nan,
                "NTrain": int(len(data["y_train"])),
                "NTest": int(len(data["y_test"])),
                "Checkpoint": "",
                "Error": str(e),
            })
        pd.DataFrame(rows).to_csv(partial_path, index=False)

    df = pd.DataFrame(rows)
    write_report(df, out_dir, args)


if __name__ == "__main__":
    main()
