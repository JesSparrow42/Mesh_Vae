#!/usr/bin/env python3
"""Export saved training membership, retaining validation separately and auditing gaps."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if globals().get("__file__", "<stdin>") != "<stdin>" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from meshvae.conditioning import build_age_gender_lookup, clean_stregkode
from meshvae.options import BaseOptions, TrainOptions
from meshvae.scan_geometry import load_aligned_scan_geometry

MEMBERSHIP_COLUMNS = ["scan_id", "npz_scan_name", "obj_basename", "split", "gender", "run", "run_directory", "source_tsv", "npz_file", "membership_evidence"]
INVENTORY_COLUMNS = ["run", "run_directory", "status", "reason", "training_rows", "validation_rows", "source_tsv", "npz_file", "completion_evidence"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def option_values(path):
    values = {}
    if path.is_file():
        for key, raw in re.findall(r"^\s*([A-Za-z_]\w*)\s*:\s*(.*?)\s*(?:\[default:.*)?$", path.read_text(encoding="utf-8-sig"), re.M):
            try:
                values[key] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                values[key] = raw
    return values


def command_values(path):
    command = read_json(path)
    if not isinstance(command, list):
        return {}
    parser = TrainOptions().parser
    tokens = command[2:] if len(command) > 1 and str(command[1]).endswith(".py") else command
    opt, unknown = parser.parse_known_args(tokens)
    if unknown:
        raise ValueError("Saved training command contains unknown options: " + " ".join(unknown))
    return vars(opt)


def eligible_names(opt, cache):
    if opt.feature_source != "scan" or not opt.use_preprocessed:
        raise ValueError("Only preprocessed scan models are supported; clinical/raw-OBJ runs need a saved membership list")
    archive = Path(opt.dataroot) / "simplified_padded_tensors.npz"
    if archive not in cache:
        with np.load(archive, allow_pickle=True) as data:
            cache[archive] = np.asarray(data["names"]).copy()
    names = cache[archive]
    needs_metadata = any(getattr(opt, field, False) for field in ("use_cond", "aux_predict_plaque", "aux_predict_regression", "aux_predict_binary"))
    cond, group, aux = build_age_gender_lookup(opt) if needs_metadata else ({}, {}, {})
    keep = []
    for name in names:
        code = clean_stregkode(name)
        target = aux.get(code, {})
        if opt.aux_predict_regression and "regression_target" not in target:
            continue
        if opt.aux_predict_binary and "binary_target" not in target:
            continue
        if opt.use_cond and code not in cond and not opt.keep_missing_cond:
            continue
        keep.append(str(name))
    ids = [clean_stregkode(name) for name in keep]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("Missing or duplicate cleaned scan IDs in selected NPZ names")
    if opt.use_scan_geometry_features and opt.require_scan_geometry_features:
        load_aligned_scan_geometry(opt.dataroot, opt.scan_geometry_features_file, keep, require_complete=True)
    return dict(zip(ids, keep)), group, archive


def study_models(root):
    claimed = set()
    for manifest_path in sorted(root.rglob("study/manifest.json")):
        study, run = manifest_path.parent, manifest_path.parent.parent
        try:
            manifest = read_json(manifest_path)
            state = read_json(study / "status.json").get("status", "missing")
        except Exception as error:
            claimed.add(run)
            yield run, study, {"_export_error": f"{type(error).__name__}: {error}"}, "malformed_metadata"
            continue
        components = ("presence", "positive") if manifest.get("experiment", {}).get("kind") == "hurdle" else (None,)
        for component in components:
            owner = run.with_name(run.name + "_" + component) if component else run
            inputs = study / component if component else study
            claimed.add(owner)
            yield owner, inputs, manifest, state
    for owner in sorted({path.parent for filename in ("opt.txt", "training_summary.json", "cond_stats.json", "best_vae_net.pth", "latest_vae_net.pth") for path in root.rglob(filename)} - claimed):
        yield owner, owner, {}, "legacy"


def export_ids(roots, output, default_dataroot=None):
    rows, inventory, cache, seen = [], [], {}, set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            inventory.append(dict(run="", run_directory=str(root), status="unresolved", reason="Missing checkpoint root"))
            continue
        for run, inputs, manifest, study_status in study_models(root):
            if run in seen:
                continue
            seen.add(run)
            entry = dict(run=run.name, run_directory=str(run), status="unresolved", reason="", training_rows=0, validation_rows=0)
            try:
                if manifest.get("_export_error"):
                    raise ValueError(manifest["_export_error"])
                summary = read_json(run / "training_summary.json")
                done = read_json(inputs / "training_completed.json")
                checkpoint = next((path for path in (run / "best_vae_net.pth", run / "latest_vae_net.pth") if path.is_file()), None)
                completed = bool(summary) or done.get("status") == "completed" or study_status == "completed"
                if not completed or checkpoint is None:
                    entry.update(status="skipped_incomplete", reason=f"Study status={study_status}; no completed training plus checkpoint evidence")
                    inventory.append(entry)
                    continue
                entry["completion_evidence"] = "training_summary.json" if summary else "training_completed.json" if done else "study/status.json"
                opt = TrainOptions().parser.parse_args(["--name", run.name])
                settings = command_values(inputs / "train_command.json")
                settings.update(option_values(run / "opt.txt"))
                if not settings and not (run / "cond_stats.json").is_file():
                    raise ValueError("Missing saved training command/options/conditioning statistics")
                vars(opt).update(settings)
                opt.name, opt.checkpoints_dir = run.name, str(run.parent)
                stats = read_json(run / "cond_stats.json")
                if "aux_predict_plaque" not in settings and stats.get("aux_plaque_cols"):
                    opt.aux_predict_plaque = True
                    opt.aux_plaque_cols = stats["aux_plaque_cols"]
                if "cond_tsv" not in settings and stats.get("cond_path"):
                    opt.cond_tsv = stats["cond_path"]
                if "dataroot" not in settings:
                    if not default_dataroot:
                        raise ValueError("Saved dataroot unavailable; supply --dataroot explicitly")
                    opt.dataroot = str(default_dataroot)
                opt.phase, opt.is_train, opt.is_test = "test", False, True
                BaseOptions._load_cond_stats_for_eval(None, opt)
                selected, groups, archive = eligible_names(opt, cache)
                entry.update(source_tsv=str(opt.cond_tsv or ""), npz_file=str(archive))
                actual_validation = summary.get("validation_ids")
                if manifest:
                    expected_train = {clean_stregkode(code) for code in pd.read_csv(inputs / "train.tsv", sep="\t", dtype=str)["participant_id"]}
                    if set(selected) != expected_train:
                        raise ValueError("Saved training membership differs from current eligible NPZ/metadata; data may have changed")
                    validation_opt = argparse.Namespace(**vars(opt))
                    validation_opt.cond_tsv = str(inputs / "validation.tsv")
                    validation, validation_groups, _ = eligible_names(validation_opt, cache)
                    expected_val = {clean_stregkode(code) for code in pd.read_csv(inputs / "validation.tsv", sep="\t", dtype=str)["participant_id"]}
                    if set(validation) != expected_val:
                        raise ValueError("Saved validation membership differs from current eligible NPZ/metadata")
                    if actual_validation is not None and {clean_stregkode(code) for code in actual_validation} != set(validation):
                        raise ValueError("Saved summary validation IDs differ from the saved validation TSV")
                    evidence = "saved_training_tsv_and_completed_run"
                elif getattr(opt, "val_cond_tsv", None):
                    validation_opt = argparse.Namespace(**vars(opt))
                    validation_opt.cond_tsv = opt.val_cond_tsv
                    validation, validation_groups, _ = eligible_names(validation_opt, cache)
                    if actual_validation is not None and {clean_stregkode(code) for code in actual_validation} != set(validation):
                        raise ValueError("Explicit validation IDs differ from the completed summary")
                    evidence = "saved_explicit_split_and_summary_counts"
                else:
                    if actual_validation is None:
                        raise ValueError("Legacy internal validation IDs were not saved; refusing to treat all development rows as training")
                    val_ids = {clean_stregkode(code) for code in actual_validation}
                    if None in val_ids or not val_ids.issubset(selected):
                        raise ValueError("Saved validation IDs are missing from current eligible scans")
                    validation = {code: selected[code] for code in val_ids}
                    validation_groups = groups
                    selected = {code: name for code, name in selected.items() if code not in val_ids}
                    evidence = "reconstructed_eligibility_minus_saved_validation_ids"
                if set(selected) & set(validation):
                    raise ValueError("Training/validation overlap")
                if summary and (len(selected) != int(summary["train_samples"]) or len(validation) != int(summary["validation_samples"])):
                    raise ValueError("Current split counts differ from the completed training summary; inputs may have changed")
                for split, membership, group_map, source in (("train", selected, groups, opt.cond_tsv), ("validation", validation, validation_groups, validation_opt.cond_tsv if manifest or getattr(opt, "val_cond_tsv", None) else opt.cond_tsv)):
                    for code, name in sorted(membership.items()):
                        basename = Path(name).name
                        rows.append(dict(scan_id=code, npz_scan_name=name, obj_basename=basename if basename.lower().endswith(".obj") else "", split=split,
                                         gender=manifest.get("gender") or opt.only_gender or group_map.get(code, "").split("_")[0], run=run.name, run_directory=str(run),
                                         source_tsv=str(source or ""), npz_file=str(archive), membership_evidence=evidence))
                entry.update(status="exported", training_rows=len(selected), validation_rows=len(validation))
            except Exception as error:
                entry["reason"] = f"{type(error).__name__}: {error}"
            inventory.append(entry)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    membership = pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS)
    membership.to_csv(output / "training_obj_ids_by_run.csv", index=False)
    unique = []
    for code, group in membership.loc[membership.split.eq("train")].groupby("scan_id", sort=True):
        unique.append(dict(scan_id=code, npz_scan_names=json.dumps(sorted(set(group.npz_scan_name))), obj_basenames=json.dumps(sorted(set(group.obj_basename) - {""})), gender=";".join(sorted(set(group.gender) - {""})), n_runs=group.run.nunique()))
    pd.DataFrame(unique, columns=["scan_id", "npz_scan_names", "obj_basenames", "gender", "n_runs"]).to_csv(output / "training_obj_ids.csv", index=False)
    audit = pd.DataFrame(inventory, columns=INVENTORY_COLUMNS)
    audit.to_csv(output / "training_id_inventory.csv", index=False)
    print(f"[OUT] {output}\n[COUNTS] {len(unique)} unique training IDs; {int(audit.status.eq('exported').sum())} exported runs; {int(audit.status.ne('exported').sum())} skipped/unresolved runs")
    return membership, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataroot", type=Path, help="Explicit fallback for legacy runs whose saved options lack a dataroot")
    args = parser.parse_args()
    export_ids(args.root, args.out_dir, args.dataroot)


if __name__ == "__main__":
    main()
