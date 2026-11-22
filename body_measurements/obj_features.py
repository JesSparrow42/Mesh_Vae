"""Build the existing physical geometry feature schema directly from raw OBJs."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from build_scan_geometry_features import FEATURE_NAMES, parse_obj_vertices
from .data import clean_id


def find_raw_obj_paths(root: str | Path) -> dict[str, Path]:
    """Index raw scan files, rejecting ambiguous cleaned identifiers."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"OBJ root is not a directory: {root}")
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".obj")
    if not paths:
        raise ValueError(f"No OBJ files found under {root}")
    by_id: dict[str, Path] = {}
    duplicates = {}
    for path in paths:
        code = clean_id(path.name)
        if code is None:
            raise ValueError(f"OBJ has no usable scan ID: {path}")
        if code in by_id:
            duplicates.setdefault(code, [by_id[code]]).append(path)
        else:
            by_id[code] = path
    if duplicates:
        examples = {code: [str(path) for path in values] for code, values in list(duplicates.items())[:5]}
        raise ValueError(f"Duplicate cleaned OBJ IDs; select one raw scan per ID: {examples}")
    return by_id


def _current(path: Path, row: dict | None) -> bool:
    if not row or row.get("status") != "ok" or row.get("path") != str(path):
        return False
    stats = path.stat()
    try:
        return (int(row["source_size"]) == stats.st_size
                and int(row["source_mtime_ns"]) == stats.st_mtime_ns
                and np.isfinite([float(row[name]) for name in FEATURE_NAMES]).all())
    except (KeyError, TypeError, ValueError):
        return False


def _save_partial(path: Path, rows: dict[str, dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame([rows[key] for key in sorted(rows)]).to_csv(temporary, index=False)
    os.replace(temporary, path)


def build_obj_features(
    obj_root: str | Path,
    out_path: str | Path,
    *,
    workers: int = 2,
    checkpoint_every: int = 10,
) -> dict:
    """Stream OBJ vertices without mesh normalization or remeshing.

    The sidecar uses the same 14-feature schema as build_scan_geometry_features.
    A partial CSV is resumable only when path, byte size, and mtime still match.
    """
    if workers < 1 or checkpoint_every < 1:
        raise ValueError("workers and checkpoint_every must be positive integers")
    out_path = Path(out_path)
    if out_path.exists():
        raise ValueError(f"Output already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".partial.csv")
    by_id = find_raw_obj_paths(obj_root)
    rows = {}
    if partial.exists():
        saved = pd.read_csv(partial, dtype={"scan_id": str}).to_dict(orient="records")
        if len(saved) != len({row["scan_id"] for row in saved}):
            raise ValueError(f"Duplicate IDs in partial file: {partial}")
        rows = {row["scan_id"]: row for row in saved}
    pending = [(code, path) for code, path in sorted(by_id.items()) if not _current(path, rows.get(code))]
    print(f"[OBJ] {len(by_id)} scans, {len(pending)} to process; partial={partial}", flush=True)
    started = time.monotonic()
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for done, ((code, path), result) in enumerate(
                zip(pending, executor.map(parse_obj_vertices, (str(path) for _, path in pending))), 1
            ):
                stats = path.stat()
                rows[code] = {"scan_id": code, **result,
                              "source_size": stats.st_size, "source_mtime_ns": stats.st_mtime_ns}
                if done % checkpoint_every == 0 or done == len(pending):
                    _save_partial(partial, rows)
                    elapsed = max(time.monotonic() - started, 0.001)
                    eta_min = (len(pending) - done) * elapsed / done / 60
                    print(f"[OBJ] {done}/{len(pending)} processed, ETA {eta_min:.1f} min", flush=True)
    codes = sorted(by_id)
    matrix = np.full((len(codes), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    scale_proxy = np.full(len(codes), np.nan, dtype=np.float32)
    valid = np.zeros(len(codes), dtype=bool)
    source_paths = []
    failures = []
    for index, code in enumerate(codes):
        path = by_id[code]
        source_paths.append(str(path))
        row = rows[code]
        if not _current(path, row):
            failures.append({"scan_id": code, "path": str(path), "error": str(row.get("error", "invalid geometry"))})
            continue
        features = np.asarray([float(row[name]) for name in FEATURE_NAMES], dtype=np.float32)
        scale = float(row["scale_proxy"])
        if not np.isfinite(features).all() or not np.isfinite(scale) or scale <= 0:
            failures.append({"scan_id": code, "path": str(path), "error": "nonfinite feature or scale"})
            continue
        matrix[index] = features
        scale_proxy[index] = scale
        valid[index] = True
    if not valid.any():
        raise ValueError(f"No valid OBJ features; inspect {partial}")
    np.savez_compressed(
        out_path,
        names=np.asarray(codes, dtype=object), features=matrix,
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        scale_proxy=scale_proxy, valid=valid,
        source_paths=np.asarray(source_paths, dtype=object),
    )
    pd.DataFrame(failures, columns=["scan_id", "path", "error"]).to_csv(
        out_path.with_suffix(".failures.csv"), index=False,
    )
    long_index = FEATURE_NAMES.index("bbox_long")
    long_values = matrix[valid, long_index]
    return {"obj_root": str(Path(obj_root).resolve()), "out": str(out_path.resolve()),
            "rows_total": len(codes), "rows_valid": int(valid.sum()),
            "rows_invalid": int((~valid).sum()), "rows_reused": len(codes) - len(pending),
            "bbox_long_p05_raw_units": float(np.quantile(long_values, 0.05)),
            "bbox_long_median_raw_units": float(np.median(long_values)),
            "bbox_long_p95_raw_units": float(np.quantile(long_values, 0.95))}
