#!/usr/bin/env python3
"""Physical point-cloud residual regression, with the original cohort held fixed.

Only trusted local joblib/PyTorch checkpoints may be loaded. This experiment is
not guaranteed to improve held-out accuracy. Baseline and residual blend weights
are selected using validation labels only; the test partition is report-only.
"""
from __future__ import annotations

# Allow direct execution as well as ``python -m`` from the repository checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from body_measurements.data import align_training, clean_id, load_config, load_features, read_metadata
from body_measurements.models import predict as baseline_predict
from scripts.measurements.run_body_measurements import fingerprint, load_split_file, prepare_output, write_json


FORMAT = 1
METRES = {"m": 1.0, "cm": 0.01, "mm": 0.001}
EXCLUDED_IDS = {"RIGDET050925"}


def checked_source(spec: dict) -> Path:
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Original baseline input is missing: {path}")
    if fingerprint(str(path))["sha256"] != spec["sha256"]:
        raise ValueError(f"Baseline input has changed since fitting: {path}")
    return path


def load_baseline(directory: str | Path):
    root = Path(directory)
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete" or run.get("scan_units") not in METRES:
        raise ValueError("Baseline must be complete and declare scan_units m, cm, or mm.")
    sources = {key: checked_source(run[key]) for key in ("features", "metadata", "config")}
    X, _ = load_features(sources["features"])
    X, y, groups, _ = align_training(X, read_metadata(sources["metadata"]),
                                     load_config(sources["config"]), run["id_col"], run.get("participant_col"))
    manifest = read_metadata(root / "splits.csv")
    wanted = manifest.scan_id.map(clean_id)
    if wanted.isna().any() or wanted.duplicated().any():
        raise ValueError("Baseline split manifest has empty or duplicate scan IDs.")
    if set(wanted) != set(X.index):
        raise ValueError("Reconstructed cohort differs from the baseline split manifest; refusing cohort changes.")
    if EXCLUDED_IDS.intersection(wanted):
        raise ValueError("Ambiguous excluded scan RIGDET050925 is unexpectedly in the baseline cohort.")
    X, y, groups = X.loc[wanted], y.loc[wanted], groups.loc[wanted]
    splits = load_split_file(str(root / "splits.csv"), groups)
    bundle = joblib.load(root / "model.joblib")
    if bundle["feature_names"] != list(X) or bundle["target_names"] != list(y):
        raise ValueError("Baseline model feature/target schema differs from original cohort.")
    baseline = baseline_predict(bundle, X)
    saved = pd.read_csv(root / "evaluation_predictions.csv", dtype={"scan_id": str})
    required = {"scan_id", "target", "split", "selected", "actual_cm", "predicted_cm"}
    if not required.issubset(saved):
        raise ValueError("Baseline evaluation predictions lack the required comparison columns.")
    saved = saved.loc[saved.selected.astype(str).str.lower().isin(["true", "1"])]
    if saved.duplicated(["scan_id", "target", "split"]).any():
        raise ValueError("Baseline has repeated selected evaluation predictions.")
    for part in ("validation", "test"):
        for target in y:
            ids = X.index[splits.eq(part)]
            rows = saved.loc[saved.split.eq(part) & saved.target.eq(target)].set_index("scan_id")
            if set(rows.index) != set(ids):
                raise ValueError(f"Baseline saved evaluation cohort differs for {part}/{target}.")
            if not np.allclose(rows.loc[ids, "actual_cm"], y.loc[ids, target], equal_nan=True, rtol=1e-6, atol=1e-5):
                raise ValueError(f"Baseline saved labels differ for {part}/{target}.")
            if not np.allclose(rows.loc[ids, "predicted_cm"], baseline.loc[ids, target], rtol=1e-6, atol=1e-5):
                raise ValueError(f"Baseline checkpoint no longer reproduces saved predictions for {part}/{target}.")
    return root, run, sources, X, y, groups, splits, bundle


def farthest_points(points: np.ndarray, count: int) -> np.ndarray:
    """Deterministic coverage anchors; physical distances are not normalized."""
    count = min(count, len(points))
    indices = np.empty(count, dtype=np.int64)
    distance = np.full(len(points), np.inf)
    selected = int(np.argmax(np.sum((points - points.mean(0)) ** 2, axis=1)))
    for i in range(count):
        indices[i] = selected
        distance = np.minimum(distance, np.sum((points - points[selected]) ** 2, axis=1))
        selected = int(np.argmax(distance))
    return indices


def sample_obj(task):
    scan_id, path_text, count, anchors1, anchors2, units, seed = task
    result = {"scan_id": scan_id, "path": path_text, "status": "error"}
    try:
        path = Path(path_text)
        before = path.stat()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{scan_id}".encode()).digest()[:8], "little"))
        sample = np.empty((count, 3), dtype=np.float64)
        low, high = np.full(3, np.inf), np.full(3, -np.inf)
        vertices, skipped_vertices = 0, 0
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                if len(line) < 3 or line[0] != "v" or not line[1].isspace():
                    continue
                fields = line.split(maxsplit=4)
                if len(fields) < 4:
                    skipped_vertices += 1
                    continue
                try:
                    point = np.asarray([float(value) for value in fields[1:4]])
                except ValueError:
                    skipped_vertices += 1
                    continue
                if not np.isfinite(point).all():
                    skipped_vertices += 1
                    continue
                low, high = np.minimum(low, point), np.maximum(high, point)
                if vertices < count:
                    sample[vertices] = point
                else:
                    index = int(rng.integers(vertices + 1))
                    if index < count:
                        sample[index] = point
                vertices += 1
        if vertices < max(32, anchors1):
            raise ValueError(f"Only {vertices} vertices; cannot represent requested local geometry")
        sample = sample[:min(count, vertices)]
        if vertices < count:
            sample = np.concatenate([sample, sample[rng.integers(len(sample), size=count - len(sample))]])
        sample = sample[rng.permutation(count)]
        extent = (high - low) * METRES[units]
        if not (0.5 <= float(extent.max()) <= 3.0):
            raise ValueError(f"Physical longest extent {extent.max():.4g} m is implausible; verify declared units")
        # Translation only: retain body scale, orientation, and left/right.
        sample = ((sample - (low + high) / 2) * METRES[units]).astype(np.float32)
        a1 = farthest_points(sample, anchors1)
        a2 = farthest_points(sample[a1], anchors2)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("OBJ changed while it was being read")
        result.update(status="ok", source_size=after.st_size, source_mtime_ns=after.st_mtime_ns,
                      vertex_count=vertices, skipped_vertex_records=skipped_vertices,
                      extent_m=extent.tolist(), points=sample, a1=a1, a2=a2)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def raw_paths(features_path: Path, wanted: list[str], fallback: str | None) -> dict[str, str]:
    with np.load(features_path, allow_pickle=True) as sidecar:
        if "source_paths" not in sidecar.files:
            raise ValueError("Geometry sidecar needs its source_paths array.")
        names = [clean_id(value) for value in sidecar["names"]]
        source_paths = sidecar["source_paths"]
        if source_paths.shape != (len(names),):
            raise ValueError("Sidecar source_paths dimensions disagree.")
        by_id = dict(zip(names, [str(value) for value in source_paths]))
    result = {name: by_id[name] for name in wanted if name in by_id and Path(by_id[name]).is_file()}
    missing = set(wanted) - set(result)
    if missing and fallback:
        candidates = {}
        for path in Path(fallback).rglob("*"):
            if path.is_file() and path.suffix.lower() == ".obj" and clean_id(path.name) in missing:
                code = clean_id(path.name)
                if code in candidates:
                    raise ValueError(f"Ambiguous fallback OBJs for requested scan {code}: {candidates[code]}, {path}")
                candidates[code] = str(path.resolve())
        result.update(candidates)
    missing = set(wanted) - set(result)
    if missing:
        raise ValueError(f"Missing raw sources for {len(missing)} scans, examples: {sorted(missing)[:10]}")
    for name, path_text in result.items():
        if clean_id(Path(path_text).name) != name:
            raise ValueError(f"Raw source filename does not agree with scan ID {name}: {path_text}")
    return result


def build_cache(args):
    if args.baseline_run:
        _, run, sources, X, _, _, _, _ = load_baseline(args.baseline_run)
        features_path, units = sources["features"], run["scan_units"]
        baseline_hash = fingerprint(str(Path(args.baseline_run) / "splits.csv"))["sha256"]
    else:
        if not args.features or not args.scan_units:
            raise ValueError("For new scans, provide --features and --scan-units instead of --baseline-run.")
        features_path, units = Path(args.features), args.scan_units
        X, _ = load_features(features_path)
        baseline_hash = None
        if EXCLUDED_IDS.intersection(X.index):
            raise ValueError("Exclude ambiguous RIGDET050925 from new-scan features explicitly.")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = {"format_version": FORMAT, "points": args.points, "anchors1": args.anchors1,
                "anchors2": args.anchors2, "seed": args.seed, "scan_units": units,
                "features": fingerprint(str(features_path)), "split_sha256": baseline_hash,
                "scan_ids": list(X.index), "feature_names": list(X),
                "centering": "full_OBJ_bounding_box_midpoint", "coordinate_units": "m",
                "normalization": "translation_only_no_individual_scale_or_rotation",
                "sampling": "deterministic_uniform_valid_vertex_reservoir_not_uniform_mesh_surface"}
    settings_path = output / "cache_settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("Existing cache settings differ. Use a new output directory.")
    elif any(output.iterdir()):
        raise ValueError("Cache output must be new/empty or have matching cache_settings.json.")
    else:
        write_json(settings_path, settings)
    paths = raw_paths(features_path, list(X.index), args.obj_root)
    parts = output / "scans"
    parts.mkdir(exist_ok=True)
    rows, tasks = [], []
    for code in X.index:
        destination = parts / (hashlib.sha256(code.encode()).hexdigest() + ".npz")
        stats = Path(paths[code]).stat()
        if destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                reusable = (saved["scan_id"].item() == code and saved["source_path"].item() == paths[code]
                            and saved["source_size"].item() == stats.st_size
                            and saved["source_mtime_ns"].item() == stats.st_mtime_ns)
            if reusable:
                rows.append({"scan_id": code, "status": "ok", "reused": True})
                continue
        tasks.append((code, paths[code], args.points, args.anchors1, args.anchors2, units, args.seed))
    print(f"[CACHE] {len(X)} exact scans; {len(tasks)} to read; {len(rows)} reused", flush=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for done, result in enumerate(executor.map(sample_obj, tasks), 1):
            points, a1, a2 = result.pop("points", None), result.pop("a1", None), result.pop("a2", None)
            rows.append(result)
            if result["status"] == "ok":
                code = result["scan_id"]
                expected_bbox = [f"bbox_extent_{axis}" for axis in "xyz"]
                if all(name in X for name in expected_bbox):
                    expected = X.loc[code, expected_bbox].to_numpy(float)*METRES[units]
                    if not np.allclose(result["extent_m"], expected, rtol=1e-5, atol=1e-6):
                        result.update(status="error", error="Raw OBJ bounding box differs from frozen feature sidecar")
                if "log1p_vertex_count" in X and not np.isclose(np.log1p(result["vertex_count"]), X.loc[code, "log1p_vertex_count"], rtol=1e-6, atol=1e-6):
                    result.update(status="error", error="Raw OBJ vertex count differs from frozen feature sidecar")
            if result["status"] == "ok":
                code = result["scan_id"]
                np.savez_compressed(parts / (hashlib.sha256(code.encode()).hexdigest() + ".npz"),
                                    scan_id=code, source_path=result["path"], source_size=result["source_size"],
                                    source_mtime_ns=result["source_mtime_ns"], points=points, a1=a1, a2=a2)
            if done % 10 == 0 or done == len(tasks):
                print(f"[CACHE] {done}/{len(tasks)} read, ETA {(len(tasks)-done)*(time.monotonic()-started)/done/60:.1f} min", flush=True)
                pd.DataFrame(rows).to_csv(output / "cache_audit.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "cache_audit.csv", index=False)
    failures = [row for row in rows if row["status"] != "ok"]
    if failures:
        raise ValueError(f"{len(failures)} OBJ failures; see cache_audit.csv. Cohort was not silently reduced.")
    arrays, anchor1, anchor2 = [], [], []
    for code in X.index:
        with np.load(parts / (hashlib.sha256(code.encode()).hexdigest() + ".npz"), allow_pickle=False) as saved:
            arrays.append(saved["points"])
            anchor1.append(saved["a1"])
            anchor2.append(saved["a2"])
    temporary = output / "point_cache.tmp.npz"
    np.savez_compressed(temporary, scan_ids=np.asarray(X.index, dtype=str), points=np.stack(arrays),
                        a1=np.stack(anchor1), a2=np.stack(anchor2), features=X.to_numpy(np.float32),
                        feature_names=np.asarray(X.columns, dtype=str))
    os.replace(temporary, output / "point_cache.npz")
    print(f"[OUT] {(output / 'point_cache.npz').resolve()}", flush=True)


def gather(values, indices):
    return values[torch.arange(values.shape[0], device=values.device).view(-1, *([1] * (indices.ndim - 1))), indices]


def mlp(sizes):
    layers = []
    for source, destination in zip(sizes[:-1], sizes[1:]):
        layers.extend([nn.Linear(source, destination), nn.LayerNorm(destination), nn.GELU()])
    return nn.Sequential(*layers)


class MeasurementPointNet(nn.Module):
    """Two local kNN abstraction stages plus physical geometry and baseline skip."""
    def __init__(self, features, targets, neighbors1=32, neighbors2=16):
        super().__init__()
        self.neighbors1, self.neighbors2 = neighbors1, neighbors2
        self.stem = mlp([3, 32, 32])
        self.local1 = mlp([36, 64, 64])
        self.local2 = mlp([132, 128, 128])
        self.geometry = mlp([features, 32, 32])
        self.head = nn.Sequential(mlp([512 + 64 + 32 + targets, 256, 128]), nn.Dropout(.1), nn.Linear(128, targets))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def abstraction(self, xyz, features, anchor_indices, neighbors, network):
        centers = gather(xyz, anchor_indices)
        with torch.no_grad():
            indices = torch.cdist(centers, xyz).topk(min(neighbors, xyz.shape[1]), largest=False).indices
        relative = gather(xyz, indices) - centers[:, :, None]
        local = torch.cat([relative, torch.linalg.vector_norm(relative, dim=-1, keepdim=True), gather(features, indices)], -1)
        encoded = network(local)
        return centers, torch.cat([encoded.max(2).values, encoded.mean(2)], -1)

    def forward(self, xyz, geometry, baseline_standardized, a1, a2):
        initial = self.stem(xyz)
        xyz1, features1 = self.abstraction(xyz, initial, a1, self.neighbors1, self.local1)
        _, features2 = self.abstraction(xyz1, features1, a2, self.neighbors2, self.local2)
        encoded = torch.cat([features2.max(1).values, features2.mean(1), initial.max(1).values,
                             initial.mean(1), self.geometry(geometry), baseline_standardized], -1)
        return self.head(encoded)


class ScanDataset(Dataset):
    def __init__(self, cache, indices, geometry, baseline, labels):
        self.cache, self.indices = cache, np.asarray(indices)
        self.geometry, self.baseline, self.labels = geometry, baseline, labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        return tuple(torch.from_numpy(np.array(value, copy=True)) for value in
                     (self.cache["points"][row], self.geometry[row], self.baseline[row],
                      self.cache["a1"][row], self.cache["a2"][row], self.labels[row]))


def masked_loss(predicted, labels):
    mask = torch.isfinite(labels)
    if not mask.any():
        return predicted.sum()*0
    safe = torch.where(mask, labels, predicted.detach())
    errors = torch.nn.functional.smooth_l1_loss(predicted, safe, reduction="none")
    per_target = errors.sum(0) / mask.sum(0).clamp_min(1)
    return per_target[mask.any(0)].mean()


def macro_mae(actual, predicted):
    return float(np.nanmean(np.nanmean(np.abs(actual - predicted), axis=0)))


def device_for(value):
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Use --device auto, cpu, or cuda:0 (never cuda:-1).") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or a valid cuda device.")
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f"Requested {device}, but that GPU is unavailable.")
    return device


def infer(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for points, geometry, baseline, a1, a2, _ in loader:
            rows.append(model(points.to(device), geometry.to(device), baseline.to(device), a1.to(device), a2.to(device)).cpu().numpy())
    return np.concatenate(rows)


def read_cache(path):
    root = Path(path)
    settings = json.loads((root / "cache_settings.json").read_text())
    if settings.get("format_version") != FORMAT or settings.get("coordinate_units") != "m":
        raise ValueError("Unsupported point-cache format or physical units.")
    with np.load(root / "point_cache.npz", allow_pickle=False) as packed:
        cache = {key: packed[key] for key in packed.files}
    ids = cache["scan_ids"].tolist()
    if len(ids) != len(set(ids)) or ids != settings["scan_ids"]:
        raise ValueError("Cache scan IDs are duplicated or differ from cache manifest.")
    for key in ("points", "features"):
        if not np.isfinite(cache[key]).all():
            raise ValueError(f"Cache contains invalid {key}.")
    if cache["points"].shape != (len(ids), settings["points"], 3):
        raise ValueError("Cached point dimensions differ from declared schema.")
    if cache["feature_names"].tolist() != settings["feature_names"] or cache["features"].shape != (len(ids), len(settings["feature_names"])):
        raise ValueError("Cached feature dimensions/names differ from declared schema.")
    for key, count, limit in (("a1", settings["anchors1"], settings["points"]),
                              ("a2", settings["anchors2"], settings["anchors1"])):
        values = cache[key]
        if values.shape != (len(ids), count) or not np.issubdtype(values.dtype, np.integer) or (values < 0).any() or (values >= limit).any():
            raise ValueError(f"Invalid point-anchor indices: {key}")
    return cache, settings


def metrics_table(actual, predicted, names, groups, baseline, bootstrap, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for j, target in enumerate(names):
        valid = np.isfinite(actual[:, j])
        truth, prediction, old = actual[valid, j], predicted[valid, j], baseline[valid, j]
        people = np.asarray(groups)[valid]
        error, old_error = np.abs(truth-prediction), np.abs(truth-old)
        unique = np.unique(people)
        # Cluster bootstrap participants, retaining every scan from each sampled person.
        mapping = {person: np.flatnonzero(people == person) for person in unique}
        boot = []
        for _ in range(bootstrap):
            indices = np.concatenate([mapping[person] for person in rng.choice(unique, len(unique), replace=True)])
            mae = float(error[indices].mean())
            boot.append([mae, float((error[indices]-old_error[indices]).mean()), 100*mae/float(truth[indices].mean())])
        interval = np.quantile(boot, [.025, .975], axis=0) if boot else np.full((2, 3), np.nan)
        baseline_mae = float(old_error.mean())
        mae = float(error.mean())
        denominator = float(np.sum((truth-truth.mean())**2))
        rows.append({"target": target, "n": len(truth), "participants": len(unique),
                     "mean_measured_cm": float(truth.mean()), "mae_cm": mae,
                     "mae_ci95_low_cm": interval[0, 0], "mae_ci95_high_cm": interval[1, 0],
                     "mae_pct_of_mean": 100*mae/float(truth.mean()),
                     "mae_pct_ci95_low": interval[0, 2], "mae_pct_ci95_high": interval[1, 2],
                     "baseline_mae_cm": baseline_mae, "delta_mae_cm": mae-baseline_mae,
                     "delta_ci95_low_cm": interval[0, 1], "delta_ci95_high_cm": interval[1, 1],
                     "improvement_pct": 100*(baseline_mae-mae)/baseline_mae if baseline_mae else np.nan,
                     "rmse_cm": float(np.sqrt(np.mean((truth-prediction)**2))),
                     "r2": 1-float(np.sum((truth-prediction)**2))/denominator if denominator else np.nan,
                     "within_1_cm": float((error <= 1).mean()), "within_2_cm": float((error <= 2).mean())})
    return pd.DataFrame(rows)


def train(args):
    root, run, sources, X, y, groups, splits, bundle = load_baseline(args.baseline_run)
    cache, cache_settings = read_cache(args.cache)
    if cache["scan_ids"].tolist() != list(X.index) or cache["feature_names"].tolist() != list(X):
        raise ValueError("Cache cohort/order/schema differs from exact baseline.")
    if cache_settings["features"] != fingerprint(str(sources["features"])) or cache_settings["split_sha256"] != fingerprint(str(root / "splits.csv"))["sha256"]:
        raise ValueError("Cache feature provenance or split fingerprint differs from baseline.")
    if cache_settings["scan_units"] != run["scan_units"] or not np.allclose(cache["features"], X.to_numpy(), rtol=1e-6, atol=1e-6):
        raise ValueError("Cached geometry or units differ from baseline inputs.")
    torch.set_num_threads(getattr(args, "threads", 2))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = device_for(args.device)
    masks = {part: splits.eq(part).to_numpy() for part in ("train", "validation", "test")}
    actual = y.to_numpy(np.float32)
    for part in masks:
        if (np.isfinite(actual[masks[part]]).sum(0) < (10 if part == "train" else 3)).any():
            raise ValueError(f"Each target needs >=10 train and >=3 held-out labels; insufficient {part} coverage.")
    # Fit statistics exclusively on training scans, never validation or test.
    training = actual[masks["train"]]
    target_mean = np.nanmean(training, 0)
    target_std = np.maximum(np.nanstd(training, 0), .1)
    geom_mean = X.to_numpy()[masks["train"]].mean(0)
    geom_std = np.maximum(X.to_numpy()[masks["train"]].std(0), 1e-6)
    geometry = ((X.to_numpy()-geom_mean)/geom_std).astype(np.float32)
    baseline = baseline_predict(bundle, X).to_numpy(np.float32)
    standardized_base = ((baseline-target_mean)/target_std).astype(np.float32)
    residual_labels = ((actual-baseline)/target_std).astype(np.float32)
    loaders = {}
    for part, mask in masks.items():
        data = ScanDataset(cache, np.flatnonzero(mask), geometry, standardized_base, residual_labels)
        loaders[part] = DataLoader(data, batch_size=args.batch_size, shuffle=part == "train", num_workers=0)
    model = MeasurementPointNet(X.shape[1], y.shape[1], args.neighbors1, args.neighbors2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=8)
    output = prepare_output(args.out_dir)
    shutil.copy2(root / "model.joblib", output / "baseline_model.joblib")
    shutil.copy2(root / "splits.csv", output / "splits.csv")
    settings = {"format_version": FORMAT, "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "training", "baseline_run": str(root.resolve()), "baseline_run_file": fingerprint(str(root / "run.json")),
                "baseline_model": fingerprint(str(root / "model.joblib")), "split_source": fingerprint(str(root / "splits.csv")),
                "baseline_inputs": {key: fingerprint(str(path)) for key, path in sources.items()},
                "feature_names": list(X), "target_names": list(y), "scan_units": run["scan_units"],
                "geometry_mean": geom_mean.tolist(), "geometry_std": geom_std.tolist(),
                "target_mean_cm": target_mean.tolist(), "target_std_cm": target_std.tolist(),
                "cache_settings": cache_settings, "training_options": vars(args),
                "architecture": "local_knn_residual_pointnet_two_stages_with_frozen_baseline",
                "no_augmentation": "No rotations, flips, scale changes, PCA alignment, or per-scan radius normalization.",
                "selection": "Validation macro MAE checkpoint, then per-target validation blend in [0,.25,.5,1]. Test report only.",
                "versions": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    settings["training_options"].pop("function", None)
    write_json(output / "run.json", settings)
    best = macro_mae(actual[masks["validation"]], baseline[masks["validation"]])
    best_epoch, stale = 0, 0
    history = [{"epoch": 0, "validation_macro_mae_cm": best, "train_loss": None}]
    def save_checkpoint(epoch, metric):
        torch.save({"format_version": FORMAT, "state_dict": model.state_dict(), "epoch": epoch,
                    "validation_macro_mae_cm": metric, "settings": settings}, output / "best_model.pt")
    save_checkpoint(0, best)
    print(f"[TRAIN] {len(X)} scans, {y.shape[1]} targets, device={device}, parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"[BASELINE] validation macro MAE={best:.4f} cm; epoch 0 is a valid saved competitor", flush=True)
    try:
        for epoch in range(1, args.epochs+1):
            started = time.monotonic()
            model.train()
            losses = []
            for step, (points, geom, old, a1, a2, labels) in enumerate(loaders["train"], 1):
                optimizer.zero_grad(set_to_none=True)
                predicted = model(points.to(device), geom.to(device), old.to(device), a1.to(device), a2.to(device))
                loss = masked_loss(predicted, labels.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if step % args.log_every == 0:
                    print(f"[TRAIN] epoch {epoch} batch {step}/{len(loaders['train'])}, loss={np.mean(losses):.5f}", flush=True)
            correction = infer(model, loaders["validation"], device)*target_std
            validation_mae = macro_mae(actual[masks["validation"]], baseline[masks["validation"]]+correction)
            scheduler.step(validation_mae)
            improved = validation_mae < best-1e-5
            if improved:
                best, best_epoch, stale = validation_mae, epoch, 0
                save_checkpoint(epoch, best)
            else:
                stale += 1
            history.append({"epoch": epoch, "validation_macro_mae_cm": validation_mae, "train_loss": float(np.mean(losses))})
            pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
            print(f"[EPOCH] {epoch}/{args.epochs}, validation MAE={validation_mae:.4f} cm, best={best:.4f} (epoch {best_epoch}), {time.monotonic()-started:.1f}s", flush=True)
            if stale >= args.patience:
                print(f"[STOP] No validation improvement for {args.patience} epochs", flush=True)
                break
        checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        validation_correction = infer(model, loaders["validation"], device)*target_std
        blend_weights, selected = [], []
        for j, target in enumerate(y):
            scores = {weight: float(np.nanmean(np.abs(actual[masks["validation"], j] - (baseline[masks["validation"], j]+weight*validation_correction[:, j]))))
                      for weight in (0., .25, .5, 1.)}
            weight = min(scores, key=scores.get)
            blend_weights.append(weight)
            selected.append({"target": target, "residual_weight": weight, "validation_mae_cm": scores[weight],
                             "baseline_validation_mae_cm": scores[0.], "checkpoint_epoch": best_epoch})
        pd.DataFrame(selected).to_csv(output / "model_selection.csv", index=False)
        checkpoint["blend_weights"] = blend_weights
        # Test is read only after checkpoint and per-target blend selection are frozen.
        tables = []
        for part in ("validation", "test"):
            correction = validation_correction if part == "validation" else infer(model, loaders[part], device)*target_std
            prediction = baseline[masks[part]]+correction*np.asarray(blend_weights)
            idx = np.flatnonzero(masks[part])
            frame = pd.DataFrame({"scan_id": np.repeat(np.asarray(X.index)[idx], len(y.columns)),
                                  "participant_id": np.repeat(groups.to_numpy()[idx], len(y.columns)),
                                  "target": np.tile(y.columns, len(idx)), "split": part,
                                  "actual_cm": actual[idx].ravel(), "predicted_cm": prediction.ravel(),
                                  "baseline_predicted_cm": baseline[idx].ravel(),
                                  "full_residual_predicted_cm": (baseline[idx]+correction).ravel(),
                                  "model": "validation_selected_pointcloud_residual", "selected": True})
            tables.append(frame)
            metrics_table(actual[idx], prediction, list(y), groups.to_numpy()[idx], baseline[idx],
                          args.bootstrap if part == "test" else 0, args.seed).to_csv(output / f"{part}_metrics.csv", index=False)
        pd.concat(tables, ignore_index=True).to_csv(output / "evaluation_predictions.csv", index=False)
        settings.update(status="complete", best_epoch=best_epoch, best_validation_macro_mae_cm=best,
                        residual_weights=dict(zip(y.columns, blend_weights)),
                        test_intervals="95% participant cluster bootstrap; same sampled people used in paired MAE difference. Fixed fitted model only.")
        checkpoint["settings"] = dict(settings)
        torch.save(checkpoint, output / "best_model.pt")
        settings["checkpoint"] = fingerprint(str(output / "best_model.pt"))
        write_json(output / "run.json", settings)
        print(pd.read_csv(output / "test_metrics.csv")[["target", "n", "mae_cm", "mae_pct_of_mean", "baseline_mae_cm", "delta_mae_cm"]].round(3).to_string(index=False))
        print(f"[OUT] {output.resolve()}", flush=True)
    except Exception as exc:
        settings.update(status="failed", error=str(exc))
        write_json(output / "run.json", settings)
        raise


def predict_new(args):
    root = Path(args.model_run)
    torch.set_num_threads(getattr(args, "threads", 2))
    device = device_for(args.device)
    run = json.loads((root / "run.json").read_text())
    if run.get("status") != "complete" or fingerprint(str(root / "best_model.pt"))["sha256"] != run.get("checkpoint", {}).get("sha256"):
        raise ValueError("Model run is incomplete/failed or checkpoint fingerprint differs from final manifest.")
    checkpoint = torch.load(root / "best_model.pt", map_location=device, weights_only=False)
    if checkpoint.get("format_version") != FORMAT or checkpoint.get("settings", {}).get("status") != "complete":
        raise ValueError("Checkpoint is not a completed measurement point-cloud experiment.")
    settings = checkpoint["settings"]
    cache, current = read_cache(args.cache)
    expected = settings["cache_settings"]
    for key in ("points", "anchors1", "anchors2", "seed", "scan_units", "feature_names", "centering", "normalization"):
        if current[key] != expected[key]:
            raise ValueError(f"New scan cache differs from training schema: {key}")
    X = pd.DataFrame(cache["features"], index=cache["scan_ids"], columns=cache["feature_names"])
    baseline_path = root / "baseline_model.joblib"
    if fingerprint(str(baseline_path))["sha256"] != settings["baseline_model"]["sha256"]:
        raise ValueError("Deployed baseline model fingerprint differs from training.")
    baseline = baseline_predict(joblib.load(baseline_path), X).to_numpy(np.float32)
    mean, std = np.asarray(settings["target_mean_cm"]), np.asarray(settings["target_std_cm"])
    geometry = ((X.to_numpy()-settings["geometry_mean"])/settings["geometry_std"]).astype(np.float32)
    dataset = ScanDataset(cache, np.arange(len(X)), geometry, ((baseline-mean)/std).astype(np.float32), np.zeros_like(baseline))
    options = settings["training_options"]
    model = MeasurementPointNet(X.shape[1], baseline.shape[1], options["neighbors1"], options["neighbors2"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    correction = infer(model, DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0), device)*std
    prediction = baseline+correction*np.asarray(checkpoint["blend_weights"])
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite new scan predictions")
    destination = Path(args.output)
    if destination.exists():
        raise ValueError("Prediction output exists; choose a new filename.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction, index=pd.Index(X.index, name="scan_id"), columns=settings["target_names"]).to_csv(destination)
    print(f"[OUT] {destination.resolve()} (all measurement columns in cm)", flush=True)


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be >=1")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("build-cache", help="Read exact baseline raw OBJ cohort, or a new-scan feature sidecar")
    source = cache.add_mutually_exclusive_group(required=True)
    source.add_argument("--baseline-run")
    source.add_argument("--features")
    cache.add_argument("--scan-units", choices=METRES)
    cache.add_argument("--obj-root", help="Fallback only if saved raw source paths no longer exist")
    cache.add_argument("--out-dir", required=True)
    cache.add_argument("--points", type=positive, default=2048)
    cache.add_argument("--anchors1", type=positive, default=128)
    cache.add_argument("--anchors2", type=positive, default=32)
    cache.add_argument("--workers", type=positive, default=2)
    cache.add_argument("--seed", type=int, default=42)
    cache.set_defaults(function=build_cache)
    fit = sub.add_parser("train", help="Validation-selected residual experiment; frozen original participant splits")
    fit.add_argument("--baseline-run", required=True)
    fit.add_argument("--cache", required=True)
    fit.add_argument("--out-dir", required=True)
    fit.add_argument("--device", default="auto")
    fit.add_argument("--epochs", type=positive, default=150)
    fit.add_argument("--patience", type=positive, default=25)
    fit.add_argument("--batch-size", type=positive, default=8)
    fit.add_argument("--threads", type=positive, default=2)
    fit.add_argument("--neighbors1", type=positive, default=32)
    fit.add_argument("--neighbors2", type=positive, default=16)
    fit.add_argument("--learning-rate", type=float, default=.0005)
    fit.add_argument("--weight-decay", type=float, default=.0001)
    fit.add_argument("--bootstrap", type=int, default=2000)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--log-every", type=positive, default=25)
    fit.set_defaults(function=train)
    prediction = sub.add_parser("predict", help="Apply the saved baseline plus validation-selected residual blend")
    prediction.add_argument("--model-run", required=True)
    prediction.add_argument("--cache", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--device", default="auto")
    prediction.add_argument("--batch-size", type=positive, default=8)
    prediction.add_argument("--threads", type=positive, default=2)
    prediction.set_defaults(function=predict_new)
    args = parser.parse_args()
    if args.command == "build-cache" and not (args.anchors2 <= args.anchors1 <= args.points):
        parser.error("Require anchors2 <= anchors1 <= points")
    if args.command == "train" and (args.bootstrap < 0 or args.learning_rate <= 0 or args.weight_decay < 0):
        parser.error("bootstrap/weight-decay must be nonnegative and learning-rate positive")
    args.function(args)


if __name__ == "__main__":
    main()
