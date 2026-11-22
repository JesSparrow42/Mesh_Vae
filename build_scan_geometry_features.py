#!/usr/bin/env python3
"""Recover physical body-size features from raw OBJ files.

The existing combined NPZ was normalized separately for every participant, so
absolute body dimensions were lost.  This script reads only OBJ vertex records
(it does not simplify meshes or resample points), writes resumable progress,
and creates a small sidecar aligned to the existing NPZ names array.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from meshvae.conditioning import clean_stregkode


FEATURE_NAMES = [
    "bbox_extent_x",
    "bbox_extent_y",
    "bbox_extent_z",
    "bbox_long",
    "bbox_mid",
    "bbox_short",
    "bbox_diagonal",
    "bbox_volume_cuberoot",
    "vertex_std_long",
    "vertex_std_mid",
    "vertex_std_short",
    "bbox_mid_to_long",
    "bbox_short_to_long",
    "log1p_vertex_count",
]


def parse_obj_vertices(path_text: str) -> dict:
    path = Path(path_text)
    count = 0
    xyz_min = np.full(3, np.inf, dtype=np.float64)
    xyz_max = np.full(3, -np.inf, dtype=np.float64)
    xyz_sum = np.zeros(3, dtype=np.float64)
    outer_sum = np.zeros((3, 3), dtype=np.float64)
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if len(line) < 3 or line[0] != "v" or not line[1].isspace():
                    continue
                parts = line.split(maxsplit=4)
                if len(parts) < 4:
                    continue
                try:
                    xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                except ValueError:
                    continue
                if not np.isfinite(xyz).all():
                    continue
                count += 1
                xyz_min = np.minimum(xyz_min, xyz)
                xyz_max = np.maximum(xyz_max, xyz)
                xyz_sum += xyz
                outer_sum += np.outer(xyz, xyz)

        if count < 4:
            raise ValueError(f"found only {count} valid vertices")
        extent = xyz_max - xyz_min
        if not np.isfinite(extent).all() or float(extent.max()) <= 0.0:
            raise ValueError(f"invalid bounding box {extent.tolist()}")

        mean = xyz_sum / float(count)
        cov = (outer_sum / float(count)) - np.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        eig = np.linalg.eigvalsh(cov)
        std_sorted = np.sqrt(np.maximum(eig, 0.0))[::-1]
        extent_sorted = np.sort(extent)[::-1]
        long, mid, short = [float(x) for x in extent_sorted]
        diag = float(np.linalg.norm(extent))
        volume_cuberoot = float(np.cbrt(max(float(np.prod(extent)), 0.0)))
        values = [
            float(extent[0]), float(extent[1]), float(extent[2]),
            long, mid, short, diag, volume_cuberoot,
            float(std_sorted[0]), float(std_sorted[1]), float(std_sorted[2]),
            mid / max(long, 1e-12), short / max(long, 1e-12), math.log1p(count),
        ]
        return {
            "path": str(path),
            "status": "ok",
            "error": "",
            "vertex_count": int(count),
            "scale_proxy": 0.5 * diag,
            **dict(zip(FEATURE_NAMES, values)),
        }
    except Exception as exc:
        return {
            "path": str(path),
            "status": "error",
            "error": str(exc),
            "vertex_count": int(count),
            "scale_proxy": np.nan,
            **{name: np.nan for name in FEATURE_NAMES},
        }


def choose_obj_paths(roots: list[str]) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = {}
    found = 0
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            raise FileNotFoundError(f"OBJ root not found: {root}")
        paths = [root] if root.is_file() else root.rglob("*.obj")
        for path in paths:
            found += 1
            code = clean_stregkode(path.name)
            if code:
                candidates.setdefault(code, []).append(path)
    print(f"[INDEX] found {found} OBJ files and {len(candidates)} cleaned participant IDs.", flush=True)

    chosen = {}
    duplicate_ids = 0
    for code, paths in candidates.items():
        if len(paths) > 1:
            duplicate_ids += 1
        # Prefer a non-simplified file, then the largest file.  A raw scan is
        # normally much larger than a derived simplified copy.
        def rank(path: Path):
            simplified = "simplified" in path.name.lower() or "simplified" in str(path.parent).lower()
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            return (simplified, -size, str(path))
        chosen[code] = sorted(paths, key=rank)[0]
    print(f"[INDEX] duplicate IDs resolved deterministically: {duplicate_ids}.", flush=True)
    return chosen


def load_partial(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    rows = {str(row["code"]): row for row in df.to_dict(orient="records") if row.get("status") == "ok"}
    print(f"[RESUME] loaded {len(rows)} completed scans from {path}.", flush=True)
    return rows


def write_partial(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [rows[key] for key in sorted(rows)]
    pd.DataFrame(ordered).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def progress_line(done: int, total: int, started: float) -> str:
    elapsed = max(time.time() - started, 1e-6)
    rate = done / elapsed
    remaining = max(total - done, 0)
    eta_hours = (remaining / rate / 3600.0) if rate > 0 else float("inf")
    return f"[PROGRESS] {done}/{total} ({100.0 * done / max(total, 1):.2f}%) ETA={eta_hours:.2f}h"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Existing simplified_padded_tensors.npz")
    ap.add_argument("--obj_roots", nargs="+", required=True, help="Raw OBJ roots searched recursively")
    ap.add_argument("--out", default=None, help="Output sidecar. Default: scan_geometry_features.npz beside --npz")
    ap.add_argument("--partial_csv", default=None, help="Resumable intermediate CSV path")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--checkpoint_every", type=int, default=10)
    args = ap.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    output = Path(args.out) if args.out else npz_path.parent / "scan_geometry_features.npz"
    partial = Path(args.partial_csv) if args.partial_csv else output.with_suffix(".partial.csv")
    packed = np.load(npz_path, allow_pickle=True)
    if "names" not in packed.files:
        raise KeyError(f"{npz_path} has no names array.")
    names = packed["names"]
    requested_codes = [clean_stregkode(name) for name in names]
    requested_set = {code for code in requested_codes if code}
    print(f"[NPZ] {len(names)} rows, {len(requested_set)} unique usable IDs.", flush=True)

    obj_by_code = choose_obj_paths(args.obj_roots)
    missing_obj = sorted(requested_set - set(obj_by_code))
    print(
        f"[MATCH] raw OBJ available for {len(requested_set) - len(missing_obj)}/{len(requested_set)} NPZ IDs; "
        f"missing {len(missing_obj)}.",
        flush=True,
    )
    if missing_obj:
        print(f"[MATCH] first missing OBJ IDs: {missing_obj[:30]}", flush=True)

    completed = load_partial(partial)
    pending = [(code, obj_by_code[code]) for code in sorted(requested_set) if code in obj_by_code and code not in completed]
    print(f"[RUN] pending scans={len(pending)} workers={max(1, args.workers)}", flush=True)
    started = time.time()
    checkpoint_every = max(1, int(args.checkpoint_every))

    if pending:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            paths = [str(path) for _, path in pending]
            for done, ((code, _), result) in enumerate(zip(pending, pool.map(parse_obj_vertices, paths)), start=1):
                completed[code] = {"code": code, **result}
                if result["status"] != "ok":
                    print(f"[ERROR] {code}: {result['error']} ({result['path']})", flush=True)
                if done % checkpoint_every == 0 or done == len(pending):
                    write_partial(partial, completed)
                    print(progress_line(done, len(pending), started), flush=True)

    feature_matrix = np.full((len(names), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    scale_proxy = np.full(len(names), np.nan, dtype=np.float32)
    valid = np.zeros(len(names), dtype=bool)
    source_paths = np.full(len(names), "", dtype=object)
    for i, code in enumerate(requested_codes):
        row = completed.get(str(code)) if code else None
        if not row or row.get("status") != "ok":
            continue
        values = np.asarray([row[name] for name in FEATURE_NAMES], dtype=np.float32)
        scale = float(row["scale_proxy"])
        if not np.isfinite(values).all() or not np.isfinite(scale) or scale <= 0.0:
            continue
        feature_matrix[i] = values
        scale_proxy[i] = scale
        valid[i] = True
        source_paths[i] = str(row["path"])

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        names=np.asarray(names, dtype=object),
        features=feature_matrix,
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        scale_proxy=scale_proxy,
        valid=valid,
        source_paths=source_paths,
    )
    audit = pd.DataFrame(feature_matrix[valid], columns=FEATURE_NAMES).describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    audit_path = output.with_name(output.stem + "_audit.csv")
    audit.to_csv(audit_path)
    print(f"[OUT] sidecar -> {output} valid={int(valid.sum())}/{len(valid)}", flush=True)
    print(f"[OUT] audit   -> {audit_path}", flush=True)
    print(f"[OUT] resume  -> {partial}", flush=True)
    if bool(valid.any()):
        qs = np.quantile(scale_proxy[valid], [0.01, 0.5, 0.99])
        print(f"[AUDIT] scale_proxy p01={qs[0]:.3f} median={qs[1]:.3f} p99={qs[2]:.3f}", flush=True)


if __name__ == "__main__":
    main()
