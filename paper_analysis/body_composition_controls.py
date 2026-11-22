#!/usr/bin/env python3
"""Six prespecified scan-encoder positive controls; no model or package changes.

Use audit to inspect eligibility, then prepare to freeze splits, train on a GPU,
and collect the saved held-out predictions. BMI labels never enter conditioning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from meshvae.conditioning import clean_stregkode

TARGETS = ("bmi", "subcutaneous_fat_ml", "skeletal_muscle_ml")
GEOMETRY_COLUMNS = {"bbox_extent_x", "bbox_extent_y", "bbox_extent_z", "bbox_long", "bbox_mid", "bbox_short", "bbox_diagonal", "bbox_volume_cuberoot", "vertex_std_long", "vertex_std_mid", "vertex_std_short", "bbox_mid_to_long", "bbox_short_to_long", "log1p_vertex_count"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_table(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig") as handle:
        head = handle.readline()
    frame = pd.read_csv(path, sep="\t" if head.count("\t") > head.count(",") else ",", dtype=str)
    frame.columns = [str(x).strip().lstrip("\ufeff").strip('"') for x in frame]
    return frame


def identified(frame, context):
    columns = {str(c).lower(): c for c in frame}
    column = next((columns[c] for c in ("participant_id", "stregkode", "participantid", "id") if c in columns), None)
    if column is None:
        raise ValueError(f"{context}: no participant ID column")
    out = frame.copy()
    raw = out[column].astype("string")
    out["participant_id"] = raw.map(lambda x: None if pd.isna(x) or not str(x).strip() else clean_stregkode(x))
    if out.participant_id.isna().any() or out.participant_id.duplicated().any():
        raise ValueError(f"{context}: missing or duplicate cleaned participant IDs")
    return out.set_index("participant_id", drop=False)


def numeric(frame, column):
    if column not in frame:
        raise ValueError(f"Missing column {column!r}; available columns: {list(frame)}")
    return pd.to_numeric(frame[column].astype(str).str.strip().str.strip('"').str.replace(",", ".", regex=False), errors="coerce")


def bmi_values(frame, bmi_col=None, weight_col=None, height_col=None, height_unit=None):
    if bmi_col:
        return numeric(frame, bmi_col), {"source": bmi_col, "unit": "kg/m2", "definition": "explicit source BMI"}
    if not (weight_col and height_col and height_unit in ("m", "cm")):
        raise ValueError("For BMI pass --bmi-col NAME, or --bmi-weight-col NAME --bmi-height-col NAME --bmi-height-unit {m,cm}. Weight must be kg; units are never inferred.")
    weight, height = numeric(frame, weight_col), numeric(frame, height_col)
    if height_unit == "cm":
        height = height / 100
    values = (weight / height.pow(2)).where((weight > 0) & (height > 0))
    return values, {"weight_source": weight_col, "weight_unit": "kg", "height_source": height_col, "height_unit": height_unit, "unit": "kg/m2", "definition": "weight_kg / height_m squared"}


def tissue_labels(path):
    tissue = identified(read_table(path), "tissue file")
    if "ts_status" not in tissue:
        raise ValueError("Tissue file lacks ts_status")
    valid_status = tissue.ts_status.astype(str).str.strip().str.lower().eq("ok")
    audit = {"source": str(Path(path).resolve()), "status_counts": tissue.ts_status.value_counts(dropna=False).to_dict(), "accepted_status": "ok", "modality": "unconfirmed; do not infer CT or visceral anatomy from names"}
    for target in TARGETS[1:]:
        value = numeric(tissue, target)
        audit[target] = {"zero": int((value == 0).sum()), "negative": int((value < 0).sum()), "nonfinite": int((~np.isfinite(value)).sum())}
        tissue[target] = value.where(valid_status & np.isfinite(value) & (value > 0))
    audit["possible_aliases"] = {}
    for left, right in (("vat_ml", "torso_fat_ml"), ("sat_ml", "subcutaneous_fat_ml")):
        if left in tissue and right in tissue:
            a, b = numeric(tissue, left), numeric(tissue, right)
            overlap = np.isfinite(a) & np.isfinite(b)
            audit["possible_aliases"][f"{left}=={right}"] = {"paired_rows": int(overlap.sum()), "equal_rows": int(np.isclose(a[overlap], b[overlap]).sum()), "used_as_target": False}
    return tissue, audit


def distribution(values):
    values = np.asarray(values, float)
    return {"n": len(values), "min": float(np.min(values)) if len(values) else None, "median": float(np.median(values)) if len(values) else None, "max": float(np.max(values)) if len(values) else None, "zeros": int(np.sum(values == 0))}


def make_splits(metadata, train, test, scan_ids, gender, targets, age_col, gender_col, seed=42, val_fraction=.15):
    """Freeze membership before target filtering. All tasks share the same sex split."""
    if set(train.index) & set(test.index):
        raise ValueError("Outer train/test participant overlap")
    prepared = {}
    for split_name, membership in (("development", train), ("test", test)):
        # The supplied split files define membership only. They must never fill
        # a missing authoritative label rejected by the tissue/BMI audit.
        frame = metadata.reindex(membership.index).copy()
        frame["participant_id"] = membership.index
        frame["study_age"] = numeric(frame, age_col)
        frame["study_gender"] = frame[gender_col].astype(str).str.strip().str.strip('"').str.upper().replace({"FEMALE": "F", "MALE": "M"})
        keep = frame.participant_id.isin(scan_ids) & frame.study_gender.eq(gender) & np.isfinite(frame.study_age) & (frame.study_age >= 18)
        prepared[split_name] = frame.loc[keep].sort_index().reset_index(drop=True)
    development = prepared.pop("development")
    order = np.random.default_rng(seed).permutation(len(development))
    nval = min(max(1, round(len(development) * val_fraction)), max(0, len(development) - 1))
    mask = np.zeros(len(development), bool)
    mask[order[:nval]] = True
    prepared["train"] = development.loc[~mask].copy()
    prepared["validation"] = development.loc[mask].copy()
    output = {}
    for target in targets:
        output[target] = {}
        for split_name, frame in prepared.items():
            y = numeric(frame, target)
            keep = np.isfinite(y) & (y > 0)
            selected = frame.loc[keep, ["participant_id", "study_age", "study_gender"]].copy()
            selected["control_target"] = y[keep].to_numpy()
            output[target][split_name] = selected.reset_index(drop=True)
    return output


def command(name, paths, target, gender, settings, epochs=150, seed=42):
    transform = "identity" if target == "bmi" else "log1p"
    return [sys.executable, str(ROOT / "train_mesh_vae.py"), "--name", name, "--checkpoints_dir", str(settings["checkpoints"]), "--dataroot", str(settings["dataroot"]), "--cond_tsv", str(paths["train"]), "--val_cond_tsv", str(paths["validation"]), "--only_gender", gender, "--gpu_ids", "0", "--seed", str(seed), "--split_seed", str(seed), "--niter", str(epochs), "--niter_decay", "0", "--batch_size", "8", "--num_threads", "0", "--save_epoch_freq", "250", "--beta", "0.001", "--beta_warmup_epochs", "30", "--feature_source", "scan", "--use_preprocessed", "--use_pointcloud", "--use_cond", "--cond_stregkode_col", "participant_id", "--cond_age_col", "study_age", "--cond_gender_col", "study_gender", "--cond_age_source", "age", "--age_bins", "18", "35", "50", "60", "200", "--cond_age_mode", "rbf", "--encoder_cond_mode", "none", "--encoder_arch", "meshconv", "--encoder_max_points", "4096", "--mesh_use_knn", "--mesh_knn_k", "12", "--mesh_knn_max_points", "4096", "--use_scan_geometry_features", "--require_scan_geometry_features", "--restore_metric_scale", "--scan_geometry_features_file", "scan_geometry_features.npz", "--use_point_labels", "--point_label_pooling", "--point_label_regions", "6", "--no-use_amp", "--no-cond_weight", "--no-cond_athero", "--no-cond_plaque", "--no-aux_predict_athero", "--no-aux_predict_plaque", "--no-aux_predict_binary", "--no-balance_athero", "--latent_prior", "gaussian", "--prior_match_mode", "kl", "--aux_predict_regression", "--aux_regression_col", "control_target", "--aux_regression_transform", transform, "--aux_regression_loss", "huber", "--aux_regression_weight", "20"]


def metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if len(y) != len(prediction) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require aligned complete finite predictions")
    residual = prediction - y
    variance = np.sum((y - y.mean()) ** 2)
    ranks_y, ranks_p = pd.Series(y).rank(), pd.Series(prediction).rank()
    return {"N": len(y), "R2": float(1 - np.sum(residual ** 2) / variance) if variance > 0 else None, "MAE": float(np.abs(residual).mean()), "RMSE": float(np.sqrt(np.mean(residual ** 2))), "Spearman": float(ranks_y.corr(ranks_p)) if ranks_y.std() > 0 and ranks_p.std() > 0 else None}


def baseline_predictions(train, test):
    y = train.control_target.to_numpy(float)
    result = {"training_mean": np.full(len(test), y.mean()), "training_median": np.full(len(test), np.median(y))}
    mean, scale = float(train.study_age.mean()), max(float(train.study_age.std(ddof=0)), 1e-6)
    x = np.c_[np.ones(len(train)), (train.study_age.to_numpy(float) - mean) / scale]
    xt = np.c_[np.ones(len(test)), (test.study_age.to_numpy(float) - mean) / scale]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    result["age_linear"] = np.maximum(xt @ beta, 0)
    return result, {"age_mean": mean, "age_scale": scale, "coefficients": beta.tolist(), "fitted_on": "train only"}


def geometry_arrays(train, test, dataroot):
    from meshvae.scan_geometry import load_aligned_scan_geometry
    x, _, names, _, _ = load_aligned_scan_geometry(dataroot, "scan_geometry_features.npz", train.participant_id.tolist(), require_complete=True)
    xt, _, test_names, _, _ = load_aligned_scan_geometry(dataroot, "scan_geometry_features.npz", test.participant_id.tolist(), require_complete=True)
    if names != test_names:
        raise ValueError("Geometry feature columns differ")
    return x.astype(float), xt.astype(float), names


def geometry_baselines(train, test, dataroot, seed=42):
    # Existing project environment includes sklearn; no installation is performed.
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x, xt, names = geometry_arrays(train, test, dataroot)
    y = train.control_target.to_numpy(float)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    trees = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=1, random_state=seed)
    fitted = {"geometry_ridge": ridge, "geometry_extra_trees": trees}
    predictions = {}
    for name, model in fitted.items():
        model.fit(x, y)
        predictions[name] = model.predict(xt)
    return predictions, {"fitted_on": "train only", "features": names, "ridge_alpha": 1.0, "trees": 200, "min_samples_leaf": 5, "n_jobs": 1, "seed": seed, "target_scale": "original units; no target transform or test tuning"}


def paired_mae_interval(y, model, comparator, seed=42, bootstraps=1000):
    # Paired participant bootstrap of fixed predictions; not training-seed uncertainty.
    differences = np.abs(np.asarray(model, float) - y) - np.abs(np.asarray(comparator, float) - y)
    rng = np.random.default_rng(seed)
    values = [float(differences[rng.integers(0, len(y), len(y))].mean()) for _ in range(bootstraps)]
    lower, upper = np.quantile(values, [.025, .975])
    return {"MeshVAEMinusComparatorMAE": float(differences.mean()), "CI95Lower": float(lower), "CI95Upper": float(upper), "Bootstraps": bootstraps, "BetterDirection": "negative", "Uncertainty": "fixed fitted models; paired participant bootstrap"}


def settings_for(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    settings = dict(config["platforms"][args.platform])
    for key in ("dataroot", "metadata", "checkpoints"):
        value = getattr(args, key, None) or settings[key]
        settings[key] = str((ROOT / value).resolve()) if not Path(value).is_absolute() else str(Path(value))
    return settings


def audit(args, freeze=False):
    settings = settings_for(args)
    metadata = identified(read_table(settings["metadata"]), "metadata")
    age_col = args.age_col or next((c for c in ("age", "alder", "age_years") if c in metadata), None)
    gender_col = args.gender_col or next((c for c in ("sexMF", "sex", "gender") if c in metadata), None)
    if not age_col or not gender_col:
        raise ValueError("Pass --age-col and --gender-col matching metadata")
    provenance = {"metadata": {"path": settings["metadata"], "sha256": digest(settings["metadata"])}}
    audit_report = {"platform": args.platform, "targets": list(args.targets), "label_definitions": {}, "tasks": [], "provenance": provenance, "encoder_input": "scan points and geometry only", "decoder_conditioning": "age and sex only; no weight, height, BMI, tissue or disease label", "caution": "A successful control shows this pipeline can learn this target; it cannot prove all causes of poor plaque performance are resolved."}
    if "bmi" in args.targets:
        metadata["bmi"], audit_report["label_definitions"]["bmi"] = bmi_values(metadata, args.bmi_col, args.bmi_weight_col, args.bmi_height_col, args.bmi_height_unit)
    if set(args.targets) & set(TARGETS[1:]):
        if not args.tissue:
            raise ValueError("Pass --tissue /data/projects/3D_scans/tissue_types.csv")
        tissues, tissue_audit = tissue_labels(args.tissue)
        audit_report["tissue_audit"] = tissue_audit
        provenance["tissue"] = {"path": str(Path(args.tissue).resolve()), "sha256": digest(args.tissue)}
        for target in TARGETS[1:]:
            metadata[target] = tissues[target].reindex(metadata.index)
    scan_path = Path(settings["dataroot"]) / "simplified_padded_tensors.npz"
    geometry_path = Path(settings["dataroot"]) / "scan_geometry_features.npz"
    with np.load(scan_path, allow_pickle=True) as archive:
        scan_ids = [clean_stregkode(x) for x in archive["names"]]
    if None in scan_ids or len(set(scan_ids)) != len(scan_ids):
        raise ValueError("Scans have duplicate or empty participant IDs")
    with np.load(geometry_path, allow_pickle=True) as sidecar:
        geometry_ids = [clean_stregkode(x) for x in sidecar["names"]]
        feature_names = [str(x) for x in sidecar["feature_names"]]
    if None in geometry_ids or len(set(geometry_ids)) != len(geometry_ids):
        raise ValueError("Geometry sidecar has duplicate or empty participant IDs")
    if not feature_names or not set(feature_names) <= GEOMETRY_COLUMNS:
        raise ValueError("Geometry sidecar has unexpected features; only the established raw OBJ geometry features are allowed")
    audit_report["geometry_features"] = feature_names
    from meshvae.scan_geometry import load_aligned_scan_geometry
    provenance["geometry"] = {"path": str(geometry_path), "sha256": digest(geometry_path)}
    provenance["scans"] = {"path": str(scan_path), "size": scan_path.stat().st_size, "mtime_ns": scan_path.stat().st_mtime_ns, "id_sha256": hashlib.sha256("\n".join(sorted(scan_ids)).encode()).hexdigest()}
    study = Path(args.output).resolve()
    if freeze and (study / "manifest.json").exists():
        raise ValueError(f"Already prepared: {study}; use another --output for a new study")
    files = {}
    for gender in ("F", "M"):
        split_paths = {split: settings[f"{split}_template"].format(gender=gender) for split in ("train", "test")}
        splits = {k: identified(read_table(v), f"{gender} {k}") for k, v in split_paths.items()}
        for split, path in split_paths.items():
            provenance[f"{gender}_{split}"] = {"path": path, "sha256": digest(path)}
        tasks = make_splits(metadata, splits["train"], splits["test"], set(scan_ids), gender, args.targets, age_col, gender_col, args.seed)
        for target, frames in tasks.items():
            for frame in frames.values():
                if len(frame):
                    load_aligned_scan_geometry(settings["dataroot"], "scan_geometry_features.npz", frame.participant_id.tolist(), require_complete=True)
            problem = [f"{s}: {len(frames[s])} rows (minimum {minimum})" for s, minimum in (("train", 50), ("validation", 10), ("test", 2)) if len(frames[s]) < minimum]
            if frames["train"].control_target.nunique() < 2 or frames["validation"].control_target.nunique() < 2:
                problem.append("training/validation label has no variation")
            name = f"{args.platform.upper()}_CONTROL_{target}_{gender}_{args.run_tag}_seed{args.seed}"
            task = {"name": name, "target": target, "gender": gender, "unit": "kg/m2" if target == "bmi" else "ml", "counts": {s: distribution(f.control_target) for s, f in frames.items()}, "ready": not problem, "problems": problem, "splits": {}}
            audit_report["tasks"].append(task)
            from meshvae.options import TrainOptions
            paths_for_check = {s: study / f"{name}/{s}.tsv" for s in frames}
            TrainOptions().parser.parse_args(command(name, paths_for_check, target, gender, settings, args.epochs, args.seed)[2:])
            for split, frame in frames.items():
                relative = f"{name}/{split}.tsv"
                files[relative] = frame
                task["splits"][split] = relative
    audit_report["passed"] = all(t["ready"] for t in audit_report["tasks"])
    audit_report["settings"] = settings
    audit_report["epochs"], audit_report["seed"] = args.epochs, args.seed
    print(pd.DataFrame([{**{k: t[k] for k in ("target", "gender", "ready")}, **{s: t["counts"][s]["n"] for s in ("train", "validation", "test")}, "problems": "; ".join(t["problems"])} for t in audit_report["tasks"]]).to_string(index=False))
    study.mkdir(parents=True, exist_ok=True)
    write_json(study / "audit.json", audit_report)
    if not audit_report["passed"]:
        raise ValueError(f"Audit did not pass; see {study / 'audit.json'}. No training is allowed.")
    if freeze:
        hashes = {}
        for relative, frame in files.items():
            path = study / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, sep="\t", index=False)
            hashes[relative] = digest(path)
        audit_report["frozen_split_sha256"] = hashes
        write_json(study / "manifest.json", audit_report)
        print(f"[PASS] Frozen {len(audit_report['tasks'])} tasks in {study}")
    return audit_report


def frozen(args):
    study = Path(args.output).resolve()
    manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["passed"]:
        raise ValueError("Study audit has not passed")
    for relative, expected in manifest["frozen_split_sha256"].items():
        if digest(study / relative) != expected:
            raise ValueError(f"Frozen split changed: {relative}")
    for key, source in manifest["provenance"].items():
        if "sha256" in source and digest(source["path"]) != source["sha256"]:
            raise ValueError(f"Source changed since audit: {key}")
        if key == "scans":
            stat = Path(source["path"]).stat()
            if stat.st_size != source["size"] or stat.st_mtime_ns != source["mtime_ns"]:
                raise ValueError("Scan archive changed since audit; prepare a new study")
    return study, manifest


def run(args):
    study, manifest = frozen(args)
    tasks = manifest["tasks"]
    if args.task_index is not None:
        if not 0 <= args.task_index < len(tasks):
            raise ValueError(f"task-index must be 0..{len(tasks)-1}")
        tasks = [tasks[args.task_index]]
    if not args.dry_run:
        subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'No GPU available'; print(torch.cuda.get_device_name(0))"], check=True)
    for task in tasks:
        folder = study / task["name"]
        status_path = folder / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("state") == "completed":
            print(f"[SKIP] completed {task['name']}")
            continue
        paths = {k: study / v for k, v in task["splits"].items()}
        cmd = command(task["name"], paths, task["target"], task["gender"], manifest["settings"], manifest["epochs"], manifest["seed"])
        write_json(folder / "train_command.json", cmd)
        if args.dry_run:
            print(shlex.join(cmd))
            continue
        checkpoint = Path(manifest["settings"]["checkpoints"]) / task["name"]
        try:
            if status.get("state") not in ("trained", "evaluation_failed"):
                if (checkpoint / "best_vae_net.pth").exists() or (checkpoint / "latest_vae_net.pth").exists():
                    raise ValueError(f"Partial/existing checkpoint: {checkpoint}. Use a new study/run tag; no automatic overwrite.")
                write_json(status_path, {"state": "training", "task": task["name"]})
                subprocess.run(cmd, check=True)
                if not (checkpoint / "best_vae_net.pth").is_file():
                    raise ValueError("Training returned without best checkpoint")
                write_json(status_path, {"state": "trained", "task": task["name"]})
            diagnosis = [sys.executable, str(ROOT / "diagnose_mesh_regression.py"), "--name", task["name"], "--checkpoints_dir", manifest["settings"]["checkpoints"], "--dataroot", manifest["settings"]["dataroot"], "--cond_tsv", str(paths["test"]), "--only_gender", task["gender"], "--gpu_ids", "0", "--which_epoch", "best", "--batch_size", "8", "--seed", str(manifest["seed"]), "--cond_stregkode_col", "participant_id", "--cond_age_col", "study_age", "--cond_gender_col", "study_gender"]
            try:
                subprocess.run(diagnosis, check=True)
                summary = pd.read_csv(checkpoint / "regression_summary.csv")
                predictions = pd.read_csv(Path(summary.iloc[0].ResultsDir) / "predictions.csv", dtype={"participant_id": str})
                expected = read_table(paths["test"])
                validate_predictions(expected, predictions)
                predictions.to_csv(folder / "predictions.csv", index=False)
            except Exception:
                write_json(status_path, {"state": "evaluation_failed", "task": task["name"]})
                raise
            write_json(status_path, {"state": "completed", "task": task["name"]})
        except Exception as error:
            latest = json.loads(status_path.read_text()) if status_path.exists() else {}
            if latest.get("state") == "training":
                write_json(status_path, {"state": "failed", "task": task["name"], "reason": str(error)})
            print(f"[ERROR] {task['name']}: {error}", file=sys.stderr)
            raise


def validate_predictions(expected, predictions):
    if predictions.participant_id.duplicated().any() or set(expected.participant_id) != set(predictions.participant_id):
        raise ValueError("Held-out predictions missing/duplicate/unexpected IDs")
    aligned = expected.set_index("participant_id").loc[predictions.participant_id]
    if not np.allclose(aligned.control_target.astype(float), predictions.target, rtol=1e-5, atol=1e-4):
        raise ValueError("Held-out target values differ from frozen labels")
    if not np.isfinite(predictions.prediction).all():
        raise ValueError("Nonfinite prediction")


def collect(args):
    study, manifest = frozen(args)
    rows, statuses, differences = [], [], []
    for task in manifest["tasks"]:
        folder = study / task["name"]
        state = json.loads((folder / "status.json").read_text()).get("state") if (folder / "status.json").exists() else "not_started"
        statuses.append({"target": task["target"], "gender": task["gender"], "status": state})
        if state != "completed":
            continue
        train = pd.read_csv(study / task["splits"]["train"], sep="\t")
        test = pd.read_csv(study / task["splits"]["test"], sep="\t")
        prediction = pd.read_csv(folder / "predictions.csv", dtype={"participant_id": str})
        validate_predictions(test, prediction)
        test = test.set_index("participant_id").loc[prediction.participant_id].reset_index()
        baselines, baseline_fit = baseline_predictions(train, test)
        geometry_predictions, geometry_fit = geometry_baselines(train, test, manifest["settings"]["dataroot"], manifest["seed"])
        baselines.update(geometry_predictions)
        baseline_fit["geometry_models"] = geometry_fit
        write_json(folder / "baseline_fit.json", baseline_fit)
        all_predictions = {"meshvae": prediction.prediction.to_numpy(float), **baselines}
        for model, values in all_predictions.items():
            rows.append({"Target": task["target"], "Gender": task["gender"], "Model": model, "Unit": task["unit"], **metrics(test.control_target, values)})
        y = test.control_target.to_numpy(float)
        for comparator in geometry_predictions:
            differences.append({"Target": task["target"], "Gender": task["gender"], "Comparator": comparator, **paired_mae_interval(y, all_predictions["meshvae"], all_predictions[comparator], manifest["seed"])})
        pd.DataFrame({"participant_id": test.participant_id, "target": test.control_target, **all_predictions}).to_csv(folder / "comparison_predictions.csv", index=False)
    pd.DataFrame(statuses).to_csv(study / "status.csv", index=False)
    pd.DataFrame(rows).to_csv(study / "comparison.csv", index=False)
    pd.DataFrame(differences).to_csv(study / "paired_geometry_differences.csv", index=False)
    print(pd.DataFrame(statuses).to_string(index=False))
    if rows:
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Completed controls: {sum(s['status']=='completed' for s in statuses)}/{len(statuses)}\nResults: {study}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "prepare", "train", "collect"))
    parser.add_argument("--platform", choices=("computerome", "gefion"), default="computerome")
    parser.add_argument("--config", default=str(ROOT / "paper_analysis/next_steps_config.json"))
    parser.add_argument("--output", default=str(ROOT / "body_control_results"))
    parser.add_argument("--tissue", help="tissue_types.csv; modality and anatomical definitions unconfirmed")
    parser.add_argument("--bmi-col")
    parser.add_argument("--bmi-weight-col")
    parser.add_argument("--bmi-height-col")
    parser.add_argument("--bmi-height-unit", choices=("m", "cm"))
    parser.add_argument("--age-col")
    parser.add_argument("--gender-col")
    parser.add_argument("--metadata")
    parser.add_argument("--dataroot")
    parser.add_argument("--checkpoints")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--run-tag", default="control_v1")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.action in ("audit", "prepare"):
            try:
                from sklearn.ensemble import ExtraTreesRegressor  # noqa: F401
                from sklearn.linear_model import Ridge  # noqa: F401
                from sklearn.preprocessing import StandardScaler  # noqa: F401
            except ImportError as error:
                raise ValueError("Existing interpreter must provide scikit-learn for the fixed geometry baselines. No packages were installed; audit stopped before training.") from error
            audit(args, freeze=args.action == "prepare")
        elif args.action == "train":
            run(args)
        else:
            collect(args)
    except (ValueError, FileNotFoundError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"[ERROR] {error}")


if __name__ == "__main__":
    main()
