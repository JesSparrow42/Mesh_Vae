"""Strict scan/metadata alignment, explicit target units, and person-level splits."""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


_DET_ID = re.compile(r"([A-Z]+DET\d{5,10})", re.IGNORECASE)
_NULL_TEXT = {"", "nan", "none", "null", "na", "n/a", "<na>", "nat"}
_TO_CM = {"cm": 1.0, "mm": 0.1, "m": 100.0}
_SCAN_SUFFIXES = {".obj", ".ply", ".stl", ".off", ".npz", ".npy"}
UNNAMED_ROW_COLUMN = "__row_number__"


def _text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    result = str(value).strip().strip('"').strip("'").strip()
    return None if result.casefold() in _NULL_TEXT else result


def clean_id(value) -> str | None:
    """Extract legacy DET IDs, or retain a generic identifier/file stem.

    Generic identifiers are case sensitive and leading zeroes are preserved.
    DET extraction follows the existing repository's scan naming convention.
    Any collisions introduced by extraction are rejected during alignment.
    """
    value = _text(value)
    if value is None:
        return None
    basename = value.replace("\\", "/").rsplit("/", 1)[-1]
    match = _DET_ID.search(basename)
    if match:
        return match.group(1).upper()
    suffix = Path(basename).suffix
    if suffix.lower() in _SCAN_SUFFIXES:
        basename = basename[:-len(suffix)]
    return _text(basename)


def clean_participant_id(value) -> str | None:
    """Preserve an explicit person's ID, including slashes and DET substrings."""
    return _text(value)


def load_config(path: str | Path) -> list[dict]:
    """Read active target mappings; every active target needs explicit units."""
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    definitions = raw.get("targets") if isinstance(raw, dict) else None
    if not isinstance(definitions, dict):
        raise ValueError("Target config must contain an object named 'targets'.")
    result: list[dict] = []
    used_columns: set[str] = set()
    for name, specification in definitions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Target names must be nonempty strings.")
        if not isinstance(specification, dict):
            raise ValueError(f"Target {name!r} must be an object.")
        column = specification.get("column")
        if column is None:
            continue
        if not isinstance(column, str) or not column.strip():
            raise ValueError(f"Target {name!r} needs a nonempty metadata column.")
        if column in used_columns:
            raise ValueError(f"Metadata column {column!r} is mapped to more than one target.")
        unit = specification.get("unit")
        if not isinstance(unit, str) or unit not in _TO_CM:
            raise ValueError(f"Target {name!r} requires an explicit unit: cm, mm, or m.")
        target = {**specification, "name": name, "column": column, "unit": unit}
        for bound in ("min_cm", "max_cm"):
            if target.get(bound) is not None:
                try:
                    target[bound] = float(target[bound])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Target {name!r}: {bound} must be numeric.") from exc
                if not np.isfinite(target[bound]) or target[bound] <= 0:
                    raise ValueError(f"Target {name!r}: {bound} must be finite and positive.")
        if (target.get("min_cm") is not None and target.get("max_cm") is not None
                and target["min_cm"] > target["max_cm"]):
            raise ValueError(f"Target {name!r}: min_cm exceeds max_cm.")
        used_columns.add(column)
        result.append(target)
    if not result:
        raise ValueError("No active targets: set at least one column and its unit in the config.")
    return result


def read_metadata(path: str | Path) -> pd.DataFrame:
    """Read quoted TSV/CSV exports without silently dropping malformed rows.

    Values stay as strings so identifiers retain leading zeroes and decimal
    commas can be interpreted only after the delimiter has been resolved.
    """
    path = Path(path)
    contents = path.read_bytes()
    try:
        text = contents.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        text = contents.decode("cp1252")
        encoding = "cp1252"
    if "\x00" in text:
        raise ValueError(f"{path}: unsupported encoding; export as UTF-8 TSV/CSV.")
    if not text.strip():
        raise ValueError(f"{path}: metadata is empty.")

    # Inspect the parsed header for each delimiter: separators inside quotes
    # must not influence detection, nor should decimal commas in later rows.
    candidates = []
    for delimiter in ("\t", ";", ","):
        try:
            reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
            header = next((row for row in reader if row), [])
            candidates.append((len(header), delimiter))
        except csv.Error:
            continue
    if not candidates:
        raise ValueError(f"{path}: cannot parse the metadata header.")
    width, delimiter = max(candidates, key=lambda item: item[0])
    if width == 0:
        raise ValueError(f"{path}: metadata has no header.")
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    header = None
    rows = []
    try:
        for row in reader:
            if not row:
                continue
            if header is None:
                header = [value.lstrip("\ufeff").strip() for value in row]
                # pandas-exported row indices often have an empty first header.
                # Give that field an explicit name, but never treat it as a scan ID.
                if header and header[0] == "":
                    header[0] = UNNAMED_ROW_COLUMN
                if any(not value for value in header) or len(set(header)) != len(header):
                    raise ValueError(f"{path}: metadata column names must be nonempty and unique.")
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"{path}: malformed metadata row ending at line {reader.line_num}: "
                    f"expected {len(header)} fields, found {len(row)}."
                )
            rows.append(row)
    except csv.Error as exc:
        raise ValueError(f"{path}: malformed CSV near line {reader.line_num}: {exc}") from exc
    result = pd.DataFrame(rows, columns=header, dtype=str)
    result.attrs.update({"encoding": encoding, "delimiter": delimiter})
    return result


def load_features(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load physical geometry sidecars and discard explicitly invalid rows.

    Existing sidecars store names as object arrays, so only load trusted local
    files. The feature matrix is never augmented with labels or metadata.
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        required = {"names", "features", "feature_names"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}.")
        names = data["names"]
        raw_feature_names = data["feature_names"]
        try:
            values = np.asarray(data["features"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: features must be numeric.") from exc
        if names.ndim != 1 or raw_feature_names.ndim != 1 or values.ndim != 2:
            raise ValueError(f"{path}: names/feature_names must be vectors and features a matrix.")
        if values.shape != (len(names), len(raw_feature_names)) or not len(raw_feature_names):
            raise ValueError(f"{path}: feature names, scan names, and matrix dimensions disagree.")
        feature_names = [v.decode("utf-8") if isinstance(v, bytes) else str(v)
                         for v in raw_feature_names]
        if (any(not value.strip() for value in feature_names)
                or len(set(feature_names)) != len(feature_names)):
            raise ValueError(f"{path}: feature names must be nonempty and unique.")
        ids = [clean_id(value) for value in names]
        if any(value is None for value in ids):
            raise ValueError(f"{path}: scan names contain missing identifiers.")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate scan IDs after cleaning; resolve repeated scans explicitly.")
        valid = np.isfinite(values).all(axis=1)
        nonfinite_rows = int((~valid).sum())
        flagged_invalid_rows = 0
        invalid_scale_rows = 0
        if "valid" in data.files:
            declared = data["valid"]
            if declared.shape != (len(names),) or not np.isin(declared, [False, True]).all():
                raise ValueError(f"{path}: valid must be a boolean (or 0/1) vector with one entry per scan.")
            declared = declared.astype(bool)
            flagged_invalid_rows = int((~declared).sum())
            valid &= declared
        if "scale_proxy" in data.files:
            try:
                scale = np.asarray(data["scale_proxy"], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: scale_proxy must be numeric.") from exc
            if scale.shape != (len(names),):
                raise ValueError(f"{path}: scale_proxy must have one entry per scan.")
            valid_scale = np.isfinite(scale) & (scale > 0)
            invalid_scale_rows = int((~valid_scale).sum())
            valid &= valid_scale
    frame = pd.DataFrame(values[valid], index=pd.Index(np.asarray(ids)[valid], name="scan_id"),
                         columns=feature_names)
    audit = {"rows_total": len(ids), "rows_valid": int(valid.sum()),
             "rows_invalid": int((~valid).sum()), "nonfinite_feature_rows": nonfinite_rows,
             "flagged_invalid_rows": flagged_invalid_rows, "invalid_scale_rows": invalid_scale_rows,
             "feature_names": feature_names}
    return frame, audit


def align_training(
    features: pd.DataFrame, metadata: pd.DataFrame, targets: list[dict],
    id_col: str, participant_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Match unique scans, mask unavailable labels, and retain person groups."""
    if not targets:
        raise ValueError("At least one active target is required.")
    if not metadata.columns.is_unique or not features.index.is_unique:
        raise ValueError("Metadata columns and feature scan IDs must be unique.")
    required = {id_col, *(target["column"] for target in targets)}
    if participant_col is not None:
        required.add(participant_col)
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing requested columns: {sorted(missing)}.")
    ids = metadata[id_col].map(clean_id)
    duplicate = ids.notna() & ids.duplicated(keep=False)
    if duplicate.any():
        raise ValueError(
            f"Duplicate metadata scan IDs after cleaning ({int(duplicate.sum())} rows). "
            "Resolve repeated visits/scans explicitly before training."
        )
    available = metadata.loc[ids.notna()].copy()
    available.index = pd.Index(ids.loc[ids.notna()].tolist(), name="scan_id")
    matched_ids = features.index[features.index.isin(available.index)]
    aligned = available.loc[matched_ids]
    X = features.loc[matched_ids].copy()
    if participant_col is None:
        groups = pd.Series(matched_ids.astype(str), index=matched_ids, name="participant_id")
    else:
        groups = aligned[participant_col].map(clean_participant_id).rename("participant_id")
        if groups.isna().any():
            raise ValueError("Matched metadata rows have missing participant IDs; person-level splitting is unsafe.")
    y = pd.DataFrame(index=matched_ids)
    target_audit = {}
    for target in targets:
        name, column, unit = target["name"], target["column"], target["unit"]
        if unit not in _TO_CM:
            raise ValueError(f"Target {name!r}: unsupported unit {unit!r}.")
        raw = aligned[column].map(_text)
        decimal_text = raw.map(lambda value: value.replace(",", ".") if value is not None else None)
        numbers = pd.to_numeric(decimal_text, errors="coerce").astype(float)
        numbers *= _TO_CM[unit]
        finite = pd.Series(np.isfinite(numbers.to_numpy()), index=numbers.index)
        positive = finite & (numbers > 0)
        below = positive & (numbers < target["min_cm"]) if target.get("min_cm") is not None else positive & False
        above = positive & (numbers > target["max_cm"]) if target.get("max_cm") is not None else positive & False
        valid = positive & ~below & ~above
        y[name] = numbers.where(valid)
        target_audit[name] = {
            "column": column, "unit": unit, "output_unit": "cm",
            "available": int(valid.sum()), "missing": int((~valid).sum()),
            "coverage_fraction": float(valid.mean()) if len(valid) else 0.0,
            "source_missing": int(raw.isna().sum()),
            "invalid_numeric": int((raw.notna() & ~finite).sum()),
            "nonpositive": int((finite & (numbers <= 0)).sum()),
            "below_min": int(below.sum()), "above_max": int(above.sum()),
        }
    keep = y.notna().any(axis=1)
    audit = {
        "metadata_rows_total": len(metadata), "metadata_rows_empty_id": int(ids.isna().sum()),
        "metadata_rows_without_scan": int((~available.index.isin(features.index)).sum()),
        "feature_rows_total": len(features), "matched_rows": len(matched_ids),
        "feature_rows_without_metadata": len(features) - len(matched_ids),
        "rows_without_any_target": int((~keep).sum()), "rows_retained": int(keep.sum()),
        "participants_retained": int(groups.loc[keep].nunique()), "targets": target_audit,
        "participant_column": participant_col,
    }
    return X.loc[keep], y.loc[keep], groups.loc[keep], audit


def make_splits(
    groups: pd.Series, seed: int, test_fraction: float = 0.2, val_fraction: float = 0.2,
) -> pd.Series:
    """Assign complete participants to deterministic train/validation/test sets.

    Fractions apply to participant counts. Each held-out set receives at least
    one person; flooring leaves the remaining participants in training.
    """
    if (not np.isfinite(test_fraction) or not np.isfinite(val_fraction)
            or test_fraction <= 0 or val_fraction <= 0 or test_fraction + val_fraction >= 1):
        raise ValueError("Split fractions must be positive and leave a positive training fraction.")
    normalized = groups.map(clean_participant_id)
    if normalized.isna().any():
        raise ValueError("Participant groups cannot be missing.")
    participants = np.asarray(sorted(normalized.unique()), dtype=object)
    if len(participants) < 3:
        raise ValueError("At least three participants are required for train/validation/test splitting.")
    n_test = max(1, int(np.floor(len(participants) * test_fraction)))
    n_val = max(1, int(np.floor(len(participants) * val_fraction)))
    if n_test + n_val >= len(participants):
        raise ValueError("Insufficient participants for the requested split fractions.")
    shuffled = np.random.default_rng(seed).permutation(participants)
    mapping = {participant: "train" for participant in participants}
    mapping.update({participant: "test" for participant in shuffled[:n_test]})
    mapping.update({participant: "validation" for participant in shuffled[n_test:n_test + n_val]})
    return normalized.map(mapping).rename("split")
