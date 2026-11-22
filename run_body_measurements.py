#!/usr/bin/env python3
"""Audit, train, and apply scan-only body-measurement regression baselines."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from body_measurements.data import (
    align_training, clean_id, clean_participant_id, load_config, load_features, make_splits, read_metadata,
)
from body_measurements.models import fit_models, predict
from body_measurements.joins import link_validation_rows, scan_candidate_tables
from body_measurements.obj_features import build_obj_features


TEMPLATE = Path(__file__).parent / "body_measurements" / "targets.example.json"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_output(path: str) -> Path:
    output = Path(path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def fingerprint(path: str) -> dict:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(source.resolve()), "sha256": digest.hexdigest()}


def column_inventory(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in metadata:
        text = metadata[column].astype("string").str.strip().str.strip('\"').str.strip("'")
        numbers = pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce")
        numbers = numbers[np.isfinite(numbers.fillna(np.nan).to_numpy(dtype=float))].astype(float)
        rows.append({
            "column": column,
            "nonempty_rows": int((text.notna() & text.ne("")).sum()),
            "finite_numeric_rows": len(numbers),
            "nonpositive_numeric_rows": int((numbers <= 0).sum()),
            "min": numbers.min(), "median": numbers.median(), "max": numbers.max(),
        })
    return pd.DataFrame(rows)


def audit(args: argparse.Namespace) -> None:
    metadata = read_metadata(args.metadata)
    report = {
        "metadata_rows": len(metadata), "metadata_columns": list(metadata.columns),
    }
    ids = None
    if args.id_col:
        if args.id_col not in metadata:
            raise ValueError(f"ID column {args.id_col!r} is absent. Columns: {list(metadata)}")
        ids = metadata[args.id_col].map(clean_id)
        report.update({
            "id_col": args.id_col, "empty_id_rows": int(ids.isna().sum()),
            "duplicate_id_rows": int((ids.notna() & ids.duplicated(keep=False)).sum()),
            "unique_ids": int(ids.nunique()),
        })
    if args.features:
        features, feature_audit = load_features(args.features)
        report["features"] = feature_audit
        if ids is not None:
            report["matching_unique_ids"] = len(set(features.index) & set(ids.dropna()))
        if args.config:
            if not args.id_col:
                raise ValueError("--config alignment audit needs --id-col; link numbered validation rows to OBJs first")
            targets = load_config(args.config)
            _, _, _, alignment = align_training(
                features, metadata, targets, args.id_col, args.participant_col,
            )
            report["alignment"] = alignment
    elif args.config:
        targets = load_config(args.config)
        missing = [target["column"] for target in targets if target["column"] not in metadata]
        if missing:
            raise ValueError(f"Configured target columns missing from metadata: {missing}")
        report["configured_targets"] = [target["name"] for target in targets]
    output = prepare_output(args.out_dir)
    column_inventory(metadata).to_csv(output / "metadata_columns.csv", index=False)
    write_json(output / "audit.json", report)
    # A template is deliberately unfilled: a similar header is not a confirmed label definition.
    (output / "targets.template.json").write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[AUDIT] {len(metadata)} metadata rows, {len(metadata.columns)} columns")
    if "matching_unique_ids" in report:
        print(f"[AUDIT] {report['matching_unique_ids']} unique IDs match valid scan features")
    print(f"[OUT] {output.resolve()}")


def inspect_sources(args: argparse.Namespace) -> None:
    table = scan_candidate_tables(args.metadata, args.obj_root, args.candidate_root)
    output = prepare_output(args.out_dir)
    table.to_csv(output / "candidate_tables.csv", index=False)
    print(table[[column for column in ("path", "rows", "measurement_headers_shared",
                                       "best_obj_column", "best_obj_matches",
                                       "best_row_column", "best_row_matches", "error")
                 if column in table]].to_string(index=False))
    print(f"[OUT] {output.resolve()}")


def link_labels(args: argparse.Namespace) -> None:
    validation = read_metadata(args.metadata)
    mapping = read_metadata(args.mapping)
    available = None
    if args.features:
        features, _ = load_features(args.features)
        available = set(features.index)
    linked, join_audit = link_validation_rows(
        validation, mapping, validation_row_col=args.validation_row_col,
        map_row_col=args.map_row_col, map_obj_col=args.map_obj_col,
        available_scan_ids=available,
    )
    output = prepare_output(args.out_dir)
    linked.to_csv(output / "metadata_with_scan_id.csv", index=False)
    write_json(output / "join_audit.json", {
        **join_audit, "validation": fingerprint(args.metadata),
        "mapping": fingerprint(args.mapping),
        "features": fingerprint(args.features) if args.features else None,
    })
    print(f"[LINK] {join_audit['joined_rows']} uniquely linked validation rows")
    print(f"[OUT] {output.resolve()}")


def build_features(args: argparse.Namespace) -> None:
    output = Path(args.out_dir)
    if output.exists():
        allowed = {"scan_geometry_features.partial.csv"}
        if not output.is_dir() or any(path.name not in allowed for path in output.iterdir()):
            raise ValueError(f"Build output must be new or contain only its resumable partial CSV: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = build_obj_features(
        args.obj_root, output / "scan_geometry_features.npz",
        workers=args.workers, checkpoint_every=args.checkpoint_every,
    )
    write_json(output / "build_audit.json", report)
    print(f"[OUT] {output.resolve()}")


def load_split_file(path: str, groups: pd.Series) -> pd.Series:
    """Reuse an existing manifest while enforcing person-level separation."""
    frame = read_metadata(path)
    if not {"scan_id", "split"}.issubset(frame.columns):
        raise ValueError("Split file must have scan_id and split columns.")
    ids = frame.scan_id.map(clean_id)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Split file contains empty or duplicate scan IDs.")
    splits = pd.Series(frame["split"].astype(str).str.strip().to_numpy(), index=ids, name="split")
    if not splits.isin(["train", "validation", "test"]).all():
        raise ValueError("Split values must be train, validation, or test.")
    missing = groups.index.difference(splits.index)
    if len(missing):
        raise ValueError(f"Split file lacks {len(missing)} retained scan IDs.")
    # Check all manifest rows with person identities, including rows without usable labels.
    if "participant_id" in frame.columns:
        people = frame.participant_id.map(clean_participant_id)
        if people.isna().any():
            raise ValueError("Split file has missing participant IDs.")
        if pd.DataFrame({"person": people.to_numpy(), "split": splits.to_numpy()}).groupby("person")["split"].nunique().gt(1).any():
            raise ValueError("Split file assigns one participant to multiple splits.")
        saved_groups = pd.Series(people.to_numpy(), index=ids)
        if not saved_groups.loc[groups.index].eq(groups).all():
            raise ValueError("Split file participant IDs disagree with current metadata.")
    aligned = splits.loc[groups.index]
    if pd.DataFrame({"group": groups, "split": aligned}).groupby("group")["split"].nunique().gt(1).any():
        raise ValueError("A participant occurs in more than one split.")
    if set(aligned) != {"train", "validation", "test"}:
        raise ValueError("Retained data must include train, validation, and test rows.")
    return aligned


def train(args: argparse.Namespace) -> None:
    targets = load_config(args.config)
    features, feature_audit = load_features(args.features)
    metadata = read_metadata(args.metadata)
    X, y, groups, alignment = align_training(
        features, metadata, targets, args.id_col, args.participant_col,
    )
    splits = (
        load_split_file(args.splits, groups) if args.splits
        else make_splits(groups, args.seed, args.test_fraction, args.val_fraction)
    )
    split_counts = []
    for name in ("train", "validation", "test"):
        rows = splits.eq(name)
        for target in y:
            observed = rows & y[target].notna()
            split_counts.append({
                "split": name, "target": target, "scan_rows": int(rows.sum()),
                "participants": int(groups[rows].nunique()), "labels": int(observed.sum()),
                "participants_with_labels": int(groups[observed].nunique()),
            })
    output = prepare_output(args.out_dir)
    write_json(output / "audit.json", {"features": feature_audit, "alignment": alignment})
    pd.DataFrame({"scan_id": X.index, "participant_id": groups.to_numpy(), "split": splits.to_numpy()}).to_csv(
        output / "splits.csv", index=False,
    )
    pd.DataFrame(split_counts).to_csv(output / "label_coverage.csv", index=False)
    run = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "features": fingerprint(args.features), "metadata": fingerprint(args.metadata),
        "config": fingerprint(args.config), "targets": targets,
        "id_col": args.id_col, "participant_col": args.participant_col,
        "scan_units": args.scan_units, "output_units": "cm",
        "seed": args.seed, "split_source": fingerprint(args.splits) if args.splits else "generated",
        "test_fraction": args.test_fraction, "validation_fraction": args.val_fraction,
        "min_train_labels": args.min_train_labels, "min_validation_labels": args.min_validation_labels,
        "n_estimators": args.n_estimators, "n_jobs": args.n_jobs,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "feature_source": "scan_geometry_features_only",
        "status": "fitting",
    }
    write_json(output / "run.json", run)
    print(f"[TRAIN] {len(X)} scan rows, {groups.nunique()} participants, {len(y.columns)} mapped targets", flush=True)
    try:
        bundle, metrics, predictions, selection = fit_models(
            X, y, splits, seed=args.seed, min_train_labels=args.min_train_labels,
            min_validation_labels=args.min_validation_labels,
            n_estimators=args.n_estimators, n_jobs=args.n_jobs,
        )
    except Exception as exc:
        run.update(status="failed", error=str(exc))
        write_json(output / "run.json", run)
        raise
    bundle.update({"scan_units": args.scan_units, "target_config": targets, "run": run})
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "evaluation_predictions.csv", index=False)
    selection.to_csv(output / "model_selection.csv", index=False)
    run["status"] = "complete"
    joblib.dump(bundle, output / "model.joblib")
    write_json(output / "run.json", run)
    print(selection.to_string(index=False))
    print(f"[OUT] {output.resolve()}")


def predict_new(args: argparse.Namespace) -> None:
    # joblib models are Python objects: load only checkpoints you trust.
    bundle = joblib.load(args.model)
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported body-measurement model format.")
    if args.scan_units != bundle.get("scan_units"):
        raise ValueError("Scan units differ from training. Regenerate features in the original training units.")
    X, audit_report = load_features(args.features)
    result = predict(bundle, X)
    result = result.rename(columns={name: f"{name}_cm" for name in result.columns})
    result.index.name = "scan_id"
    output = prepare_output(args.out_dir)
    result.to_csv(output / "predictions_cm.csv")
    write_json(output / "prediction_audit.json", {
        "features": audit_report, "rows_predicted": len(result),
        "model": fingerprint(args.model), "input": fingerprint(args.features),
        "scan_units": args.scan_units, "output_units": "cm",
    })
    print(f"[PREDICT] {len(result)} scans, {len(result.columns)} measurements; units: cm")
    print(f"[OUT] {output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    a = commands.add_parser("audit", help="Inspect metadata columns, IDs, and optional label coverage")
    i = commands.add_parser("inspect-sources", help="Find candidate row/OBJ mapping tables beside the validation CSV")
    l = commands.add_parser("link-labels", help="Link numbered validation rows to explicit OBJ scan IDs")
    b = commands.add_parser("build-features", help="Extract physical scan features directly from raw OBJs")
    t = commands.add_parser("train", help="Select per-measurement regressors using validation MAE")
    p = commands.add_parser("predict", help="Predict centimetres from scans without measurement metadata")
    for command in (a, t):
        command.add_argument("--metadata", required=True, help="Manual-measurement CSV or TSV")
        command.add_argument("--id-col", required=command is t,
                             help="Metadata column matching scan filenames/IDs; omit for index-only audit")
        command.add_argument("--participant-col", help="Stable person ID if each person can have multiple scan IDs")
        command.add_argument("--config", required=command is t, help="Explicit target-column and unit mapping JSON")
    for command in (a, t, p, i, l, b):
        command.add_argument("--out-dir", required=True, help="New or empty output directory")
    for command in (a, t, p):
        command.add_argument("--features", required=command is not a, help="Physical scan_geometry_features.npz")
    i.add_argument("--metadata", required=True, help="The numbered validation CSV")
    i.add_argument("--obj-root", required=True, help="Folder with original OBJs")
    i.add_argument("--candidate-root", required=True, help="Folder with possible mapping CSV/TSV files")
    l.add_argument("--metadata", required=True, help="The numbered validation CSV")
    l.add_argument("--mapping", required=True, help="Table with explicit row key and OBJ filename/code")
    l.add_argument("--validation-row-col", default="__row_number__",
                   help="Validation row key; unnamed first header is called __row_number__")
    l.add_argument("--map-row-col", required=True, help="Mapping column with validation row key")
    l.add_argument("--map-obj-col", required=True, help="Mapping column with matching OBJ filename/code")
    l.add_argument("--features", help="Optional scan features; require every mapped OBJ to be present")
    b.add_argument("--obj-root", required=True, help="Recursively searched raw OBJ folder")
    b.add_argument("--workers", type=int, default=2)
    b.add_argument("--checkpoint-every", type=int, default=10)
    for command in (t, p):
        command.add_argument("--scan-units", required=True, choices=("mm", "cm", "m"),
                             help="Actual raw OBJ coordinate units used to build this sidecar (not target units)")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--test-fraction", type=float, default=0.2)
    t.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of all participants")
    t.add_argument("--splits", help="Existing splits.csv to reuse a held-out cohort")
    t.add_argument("--min-train-labels", type=int, default=10)
    t.add_argument("--min-validation-labels", type=int, default=3)
    t.add_argument("--n-estimators", type=int, default=300)
    t.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--model", required=True, help="Trusted model.joblib from train")
    a.set_defaults(handler=audit)
    i.set_defaults(handler=inspect_sources)
    l.set_defaults(handler=link_labels)
    b.set_defaults(handler=build_features)
    t.set_defaults(handler=train)
    p.set_defaults(handler=predict_new)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
