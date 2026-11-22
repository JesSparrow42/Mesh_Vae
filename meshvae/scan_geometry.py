from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .conditioning import clean_stregkode


@dataclass
class ScanGeometrySpec:
    feature_names: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    coordinate_scale: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "ScanGeometrySpec":
        return cls(
            feature_names=[str(x) for x in raw["feature_names"]],
            feature_mean=[float(x) for x in raw["feature_mean"]],
            feature_std=[float(x) for x in raw["feature_std"]],
            coordinate_scale=float(raw["coordinate_scale"]),
        )


def resolve_scan_geometry_path(dataroot: str | Path, filename: str | Path) -> Path:
    path = Path(filename)
    return path if path.is_absolute() else Path(dataroot) / path


def load_aligned_scan_geometry(
    dataroot: str | Path,
    filename: str | Path,
    names: np.ndarray | list,
    require_complete: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, Path]:
    """Load a sidecar and align its rows to an NPZ names array."""
    path = resolve_scan_geometry_path(dataroot, filename)
    if not path.exists():
        raise FileNotFoundError(
            f"Scan geometry sidecar not found: {path}. Run build_scan_geometry_features.py first."
        )
    data = np.load(path, allow_pickle=True)
    required = {"names", "features", "feature_names", "scale_proxy"}
    missing_keys = sorted(required - set(data.files))
    if missing_keys:
        raise KeyError(f"{path} is missing arrays: {missing_keys}")

    side_names = data["names"]
    side_features = data["features"].astype(np.float32)
    side_scale = data["scale_proxy"].astype(np.float32).reshape(-1)
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    if side_features.ndim != 2 or side_features.shape[0] != len(side_names):
        raise ValueError(f"Invalid features shape in {path}: {side_features.shape}")
    if side_features.shape[1] != len(feature_names):
        raise ValueError(f"Feature names and columns disagree in {path}.")

    by_code: dict[str, int] = {}
    for i, raw_name in enumerate(side_names):
        code = clean_stregkode(raw_name)
        if code and code not in by_code:
            by_code[code] = i

    aligned = np.full((len(names), side_features.shape[1]), np.nan, dtype=np.float32)
    scales = np.full(len(names), np.nan, dtype=np.float32)
    valid = np.zeros(len(names), dtype=bool)
    for i, raw_name in enumerate(names):
        code = clean_stregkode(raw_name)
        j = by_code.get(code) if code else None
        if j is None:
            continue
        row = side_features[j]
        scale = float(side_scale[j])
        if np.isfinite(row).all() and np.isfinite(scale) and scale > 0.0:
            aligned[i] = row
            scales[i] = scale
            valid[i] = True

    missing_count = int((~valid).sum())
    print(
        f"[SCAN GEOMETRY] matched {int(valid.sum())}/{len(valid)} NPZ scans to {path} "
        f"(missing {missing_count}).",
        flush=True,
    )
    if require_complete and missing_count:
        examples = [str(names[i]) for i in np.flatnonzero(~valid)[:20]]
        raise ValueError(
            f"{missing_count} selected scans have no physical geometry features. "
            f"First missing names: {examples}"
        )
    return aligned, scales, feature_names, valid, path


def fit_scan_geometry_spec(
    features: np.ndarray,
    scales: np.ndarray,
    feature_names: list[str],
) -> ScanGeometrySpec:
    features = np.asarray(features, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    valid_rows = np.isfinite(features).all(axis=1) & np.isfinite(scales) & (scales > 0.0)
    if not bool(valid_rows.any()):
        raise ValueError("No valid scan geometry rows are available to fit training statistics.")
    fit = features[valid_rows]
    mean = np.mean(fit, axis=0).astype(np.float32)
    std = np.std(fit, axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    coordinate_scale = float(np.median(scales[valid_rows]))
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0.0:
        raise ValueError(f"Invalid fitted coordinate scale: {coordinate_scale}")
    return ScanGeometrySpec(
        feature_names=list(feature_names),
        feature_mean=mean.tolist(),
        feature_std=std.tolist(),
        coordinate_scale=coordinate_scale,
    )


def transform_scan_geometry(
    features: np.ndarray,
    scales: np.ndarray,
    spec: ScanGeometrySpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    mean = np.asarray(spec.feature_mean, dtype=np.float32)
    std = np.asarray(spec.feature_std, dtype=np.float32)
    if features.shape[1] != len(mean):
        raise ValueError(
            f"Scan geometry feature count changed: sidecar has {features.shape[1]}, stats expect {len(mean)}."
        )
    valid = np.isfinite(features).all(axis=1) & np.isfinite(scales) & (scales > 0.0)
    imputed = np.where(np.isfinite(features), features, mean[None, :])
    standardized = ((imputed - mean[None, :]) / std[None, :]).astype(np.float32)
    coordinate_factor = np.ones(len(scales), dtype=np.float32)
    coordinate_factor[valid] = scales[valid] / np.float32(spec.coordinate_scale)
    return standardized, coordinate_factor, valid


def save_scan_geometry_spec(spec: ScanGeometrySpec, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")


def load_scan_geometry_spec(path: str | Path) -> ScanGeometrySpec:
    path = Path(path)
    return ScanGeometrySpec.from_dict(json.loads(path.read_text(encoding="utf-8")))
