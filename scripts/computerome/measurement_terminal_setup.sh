bash <<'BODY_MEASUREMENTS_TERMINAL_INSTALL_END'
set -eu
TASK_CODE_DIR="${BODY_MEASUREMENTS_CODE_DIR:-/data/projects/3D_scans/ML/MQVAE_new}"
PY="${BODY_MEASUREMENTS_PYTHON:-/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python}"
META="${BODY_MEASUREMENTS_METADATA:-/data/projects/3D_scans/bodyscan_variables_for_validation_n2894.csv}"
cd "$TASK_CODE_DIR"
"$PY" -c 'import numpy, pandas, sklearn, joblib; from meshvae.conditioning import clean_stregkode'
test -f "$META" || { echo "[ERROR] Metadata file missing: $META"; exit 1; }
STAGE_DIR=$(mktemp -d "$PWD/.measurement_install_XXXXXXXX")
mkdir -p "$STAGE_DIR/body_measurements"
cat > "$STAGE_DIR/run_body_measurements.py" <<'BODY_MEASUREMENTS_SOURCE_00_END'
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
BODY_MEASUREMENTS_SOURCE_00_END
cat > "$STAGE_DIR/build_scan_geometry_features.py" <<'BODY_MEASUREMENTS_SOURCE_01_END'
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
BODY_MEASUREMENTS_SOURCE_01_END
cat > "$STAGE_DIR/body_measurements/__init__.py" <<'BODY_MEASUREMENTS_SOURCE_02_END'
"""Supervised prediction of manual body measurements from scan geometry."""

__version__ = "0.1.0"
BODY_MEASUREMENTS_SOURCE_02_END
cat > "$STAGE_DIR/body_measurements/data.py" <<'BODY_MEASUREMENTS_SOURCE_03_END'
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
BODY_MEASUREMENTS_SOURCE_03_END
cat > "$STAGE_DIR/body_measurements/joins.py" <<'BODY_MEASUREMENTS_SOURCE_04_END'
"""Explicitly link numbered validation rows to OBJ scan identifiers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import UNNAMED_ROW_COLUMN, clean_id, read_metadata


def _unique_nonempty(series: pd.Series, label: str, normalizer) -> pd.Series:
    values = series.map(normalizer)
    if values.isna().any():
        raise ValueError(f"{label} has {int(values.isna().sum())} empty values")
    if values.duplicated().any():
        examples = values[values.duplicated(keep=False)].unique().tolist()[:10]
        raise ValueError(f"{label} has duplicate IDs: {examples}")
    return values


def link_validation_rows(
    validation: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    validation_row_col: str = UNNAMED_ROW_COLUMN,
    map_row_col: str,
    map_obj_col: str,
    available_scan_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Join by an explicit shared row key, rejecting ambiguous and unmatched rows."""
    if not validation.columns.is_unique or not mapping.columns.is_unique:
        raise ValueError("Validation and mapping columns must be unique")
    if validation_row_col not in validation:
        raise ValueError(f"Validation row column {validation_row_col!r} absent: {list(validation)}")
    for column in (map_row_col, map_obj_col):
        if column not in mapping:
            raise ValueError(f"Mapping column {column!r} absent: {list(mapping)}")
    if "scan_id" in validation and validation_row_col != "scan_id":
        raise ValueError("Validation table already has a scan_id column; use that directly after auditing it")

    def row_key(value) -> str | None:
        if value is None or pd.isna(value):
            return None
        result = str(value).strip()
        return result if result and result.casefold() not in {"nan", "none", "null"} else None

    validation_key = _unique_nonempty(validation[validation_row_col], "validation row key", row_key)
    mapping_key = _unique_nonempty(mapping[map_row_col], "mapping row key", row_key)
    scan_ids = _unique_nonempty(mapping[map_obj_col], "mapping OBJ", clean_id)
    by_key = pd.Series(scan_ids.to_numpy(), index=mapping_key.to_numpy())
    missing = validation_key[~validation_key.isin(by_key.index)].tolist()
    if missing:
        raise ValueError(f"Mapping lacks {len(missing)} validation row keys; examples: {missing[:10]}")
    linked_ids = by_key.loc[validation_key.to_numpy()].to_numpy()
    extra_mapping = len(set(mapping_key) - set(validation_key))
    if available_scan_ids is not None:
        absent = sorted(set(linked_ids) - available_scan_ids)
        if absent:
            raise ValueError(f"{len(absent)} mapped OBJ IDs have no scan features; examples: {absent[:10]}")
    if "scan_id" in validation.columns:
        raise ValueError("Validation metadata already contains scan_id")
    linked = validation.copy()
    linked.insert(0, "scan_id", linked_ids)
    audit = {
        "validation_rows": len(validation), "mapping_rows": len(mapping),
        "joined_rows": len(linked), "mapping_rows_not_in_validation": extra_mapping,
        "unique_scan_ids": len(set(linked_ids)),
        "validation_row_col": validation_row_col, "map_row_col": map_row_col,
        "map_obj_col": map_obj_col,
    }
    return linked, audit


def scan_candidate_tables(
    validation_path: str | Path,
    obj_root: str | Path,
    candidate_root: str | Path,
) -> pd.DataFrame:
    """Summarize nearby CSV/TSV headers and overlaps without guessing a join."""
    validation = read_metadata(validation_path)
    candidate_root = Path(candidate_root)
    obj_root = Path(obj_root)
    if not candidate_root.is_dir() or not obj_root.is_dir():
        raise FileNotFoundError("Candidate root and OBJ root must both exist")
    scan_ids = {clean_id(path.name) for path in obj_root.rglob("*")
                if path.is_file() and path.suffix.lower() == ".obj"}
    scan_ids.discard(None)
    measure_cols = set(validation.columns) - {UNNAMED_ROW_COLUMN}
    rows = []
    for path in sorted(candidate_root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".tsv"} or path == Path(validation_path):
            continue
        try:
            table = read_metadata(path)
            per_column = {}
            for column in table:
                ids = set(table[column].map(clean_id).dropna()) & scan_ids
                if ids:
                    per_column[column] = len(ids)
            row_candidates = []
            if UNNAMED_ROW_COLUMN in validation:
                expected = set(validation[UNNAMED_ROW_COLUMN].astype(str).str.strip())
                for column in table:
                    values = table[column].astype(str).str.strip()
                    overlap = len(set(values) & expected)
                    if overlap:
                        row_candidates.append((column, overlap))
            rows.append({
                "path": str(path), "rows": len(table),
                "measurement_headers_shared": len(measure_cols & set(table.columns)),
                "best_obj_column": max(per_column, key=per_column.get) if per_column else "",
                "best_obj_matches": max(per_column.values(), default=0),
                "best_row_column": max(row_candidates, key=lambda pair: pair[1])[0] if row_candidates else "",
                "best_row_matches": max((overlap for _, overlap in row_candidates), default=0),
                "columns": " | ".join(table.columns), "error": "",
            })
        except (ValueError, UnicodeError) as exc:
            rows.append({"path": str(path), "error": str(exc)})
    result = pd.DataFrame(rows)
    if len(result):
        result.sort_values(["best_obj_matches", "best_row_matches", "measurement_headers_shared"],
                           ascending=False, na_position="last", inplace=True)
    return result
BODY_MEASUREMENTS_SOURCE_04_END
cat > "$STAGE_DIR/body_measurements/models.py" <<'BODY_MEASUREMENTS_SOURCE_05_END'
"""Leakage-conscious baselines for predicting manual measurements in centimetres.

Only scan-derived numeric features belong in ``X``. The data preparation layer
is responsible for excluding metadata, identifiers, and target-derived features.
Each candidate is fitted on labelled training rows for its own target; validation
selects a model and the held-out test partition is used only for reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FORMAT_VERSION = 1
_PARTITIONS = ("train", "validation", "test")


def _numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with named scan geometry features.")
    if X.shape[1] == 0:
        raise ValueError("At least one scan geometry feature is required.")
    if not X.index.is_unique or not X.columns.is_unique:
        raise ValueError("Feature row identifiers and column names must be unique.")
    if not all(isinstance(name, str) for name in X.columns):
        raise ValueError("Feature names must be strings.")
    try:
        numeric = X.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Scan geometry features must be numeric.") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Scan geometry features contain missing or nonfinite values.")
    return numeric


def _align_rows(value: pd.DataFrame | pd.Series, X: pd.DataFrame, name: str):
    if not value.index.is_unique:
        raise ValueError(f"{name} row identifiers must be unique.")
    if len(value) != len(X) or not X.index.isin(value.index).all():
        raise ValueError(f"{name} must have exactly the same scan identifiers as X.")
    return value.reindex(X.index)


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    labelled = np.isfinite(actual)
    actual, predicted = actual[labelled], predicted[labelled]
    n = len(actual)
    result = dict.fromkeys(
        ("mae_cm", "rmse_cm", "r2", "bias_cm", "loa_lower_cm", "loa_upper_cm",
         "within_1_cm", "within_2_cm"), float("nan")
    )
    result["n"] = n
    if not n:
        return result
    residual = predicted - actual
    bias = float(residual.mean())
    result.update(
        mae_cm=float(np.abs(residual).mean()),
        rmse_cm=float(np.sqrt(np.square(residual).mean())),
        bias_cm=bias,
        within_1_cm=float((np.abs(residual) <= 1.0).mean()),
        within_2_cm=float((np.abs(residual) <= 2.0).mean()),
    )
    if n >= 2:
        spread = 1.96 * float(residual.std(ddof=1))
        result.update(loa_lower_cm=bias - spread, loa_upper_cm=bias + spread)
        denominator = float(np.square(actual - actual.mean()).sum())
        if denominator > 0:
            result["r2"] = float(1.0 - np.square(residual).sum() / denominator)
    return result


def fit_models(
    X: pd.DataFrame,
    y: pd.DataFrame,
    split: pd.Series,
    *,
    seed: int = 42,
    min_train_labels: int = 10,
    min_validation_labels: int = 3,
    n_estimators: int = 300,
    n_jobs: int = -1,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit training-only models and return bundle, metrics, predictions, selection.

    Targets must already be expressed in centimetres. Missing targets are masked
    independently. A target with insufficient training labels is skipped; with
    insufficient validation labels its training median is the selected fallback.
    Validation MAE selects among the median, four scaled Ridge regressions, and
    Extra Trees. Ties prefer the first candidate, starting with the median.
    No model is refitted after selection. Validation reports every candidate;
    test reports only the selected model and the median comparator. Metrics use
    labelled rows; predictions also include unlabelled rows from those splits.
    """
    X = _numeric_features(X)
    if not isinstance(y, pd.DataFrame) or not isinstance(split, pd.Series):
        raise TypeError("y must be a DataFrame and split must be a Series.")
    if not y.columns.is_unique or not all(isinstance(name, str) for name in y.columns):
        raise ValueError("Target names must be unique strings.")
    if y.shape[1] == 0:
        raise ValueError("At least one measurement target is required.")
    y = _align_rows(y, X, "Targets")
    split = _align_rows(split, X, "Split")
    try:
        y = y.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Measurement targets must be numeric centimetres or missing.") from exc
    if np.isinf(y.to_numpy()).any():
        raise ValueError("Measurement targets cannot contain infinite values.")
    if not split.isin(_PARTITIONS).all():
        raise ValueError("Split labels must be train, validation, or test with no missing values.")
    for name, value in (("min_train_labels", min_train_labels),
                        ("min_validation_labels", min_validation_labels),
                        ("n_estimators", n_estimators)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    train = split.eq("train")
    if not train.any():
        raise ValueError("A nonempty training partition is required.")

    selected_models: dict = {}
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[dict] = []
    for target in y.columns:
        labelled = y[target].notna()
        counts = {part: int((split.eq(part) & labelled).sum()) for part in _PARTITIONS}
        selection = {
            "target": target,
            "n_train": counts["train"],
            "n_validation": counts["validation"],
            "n_test": counts["test"],
            "selected_model": None,
            "validation_mae_cm": float("nan"),
        }
        if counts["train"] < min_train_labels:
            selection.update(
                status="skipped",
                reason=f"Only {counts['train']} training labels; need {min_train_labels}.",
            )
            selection_rows.append(selection)
            continue

        candidates = {"train_median": DummyRegressor(strategy="median")}
        for alpha in (0.1, 1, 10, 100):
            candidates[f"ridge_alpha_{alpha}"] = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        candidates["extra_trees"] = ExtraTreesRegressor(
            n_estimators=n_estimators, min_samples_leaf=2, random_state=seed, n_jobs=n_jobs
        )
        train_labelled = train & labelled
        for estimator in candidates.values():
            estimator.fit(X.loc[train_labelled], y.loc[train_labelled, target])

        # Complete selection before inspecting test outcomes or computing test metrics.
        validation = split.eq("validation")
        validation_predictions = {
            name: estimator.predict(X.loc[validation]) if validation.any() else np.empty(0)
            for name, estimator in candidates.items()
        }
        validation_metrics = {
            name: _regression_metrics(y.loc[validation, target].to_numpy(), predicted)
            for name, predicted in validation_predictions.items()
        }
        if counts["validation"] >= min_validation_labels:
            selected_name = min(candidates, key=lambda name: validation_metrics[name]["mae_cm"])
            selection.update(status="trained", reason="Selected by validation MAE.")
        else:
            selected_name = "train_median"
            selection.update(
                status="baseline_only_insufficient_validation",
                reason=(f"Only {counts['validation']} validation labels; need "
                        f"{min_validation_labels}. Using training median."),
            )
        selected_models[target] = candidates[selected_name]
        selection.update(
            selected_model=selected_name,
            validation_mae_cm=validation_metrics[selected_name]["mae_cm"],
        )
        selection_rows.append(selection)

        for part in ("validation", "test"):
            rows = split.eq(part)
            actual = y.loc[rows, target].to_numpy()
            for name, estimator in candidates.items():
                if part == "test" and name not in {"train_median", selected_name}:
                    continue
                if part == "validation":
                    predicted = validation_predictions[name]
                    scores = validation_metrics[name]
                else:
                    predicted = estimator.predict(X.loc[rows]) if rows.any() else np.empty(0)
                    scores = _regression_metrics(actual, predicted)
                metric_rows.append({
                    "target": target, "model": name, "split": part,
                    "selected": name == selected_name, **scores,
                })
                prediction_frames.append(pd.DataFrame({
                    "scan_id": X.index[rows], "target": target, "split": part,
                    "actual_cm": actual, "predicted_cm": predicted,
                    "model": name, "selected": name == selected_name,
                }))

    selection_table = pd.DataFrame(selection_rows)
    if not selected_models:
        reasons = "; ".join(f"{row['target']}: {row['reason']}" for row in selection_rows)
        raise ValueError(f"No measurement models could be fitted. {reasons}")
    bundle = {
        "format_version": FORMAT_VERSION,
        "feature_names": X.columns.tolist(),
        "target_names": list(selected_models),
        "target_units": "cm",
        "models": selected_models,
        "selection": selection_rows,
        "seed": seed,
    }
    return bundle, pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True), selection_table


def predict(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Predict centimetres, safely reordering the exact saved feature schema."""
    if bundle.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported body measurement model bundle format_version.")
    X = _numeric_features(X)
    expected = bundle["feature_names"]
    missing = [name for name in expected if name not in X.columns]
    additional = [name for name in X.columns if name not in expected]
    if missing or additional:
        raise ValueError(f"Feature schema mismatch; missing={missing}, additional={additional}.")
    X = X.loc[:, expected]
    if not len(X):
        return pd.DataFrame(index=X.index, columns=bundle["target_names"], dtype=float)
    return pd.DataFrame(
        {target: bundle["models"][target].predict(X) for target in bundle["target_names"]},
        index=X.index,
    )
BODY_MEASUREMENTS_SOURCE_05_END
cat > "$STAGE_DIR/body_measurements/obj_features.py" <<'BODY_MEASUREMENTS_SOURCE_06_END'
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
BODY_MEASUREMENTS_SOURCE_06_END
cat > "$STAGE_DIR/body_measurements/targets.validation_n1715.json" <<'BODY_MEASUREMENTS_SOURCE_07_END'
{
  "description": "Exact manual measurement headers visible in bodyscan_variables_for_validation_n1715.csv; verify the source export and units before fitting.",
  "targets": {
    "bust_circumference": {"letter": "A", "column": "bust_chest_girth_cm_3d", "unit": "cm"},
    "underbust_circumference": {"letter": "B", "column": "underbust_circumference_horizontal_cm_3d", "unit": "cm"},
    "high_waist_circumference": {"letter": "C", "column": "high_waist_girth_cm_3d", "unit": "cm"},
    "waist_circumference": {"letter": "D", "column": "waist_girth_cm_3d", "unit": "cm"},
    "high_hip_circumference": {"letter": "E", "column": "high_hip_girth_cm_3d", "unit": "cm"},
    "hip_circumference": {"letter": "F", "column": "hip_girth_cm_3d", "unit": "cm"},
    "buttock_circumference": {"letter": "G", "column": "buttock_girth_cm_3d", "unit": "cm"},
    "total_thigh_circumference": {"letter": "H", "column": "hip_thigh_girth_cm_3d", "unit": "cm"},
    "neck_circumference": {"letter": "I", "column": "mid_neck_girth_cm_3d", "unit": "cm"},
    "abdominal_circumference": {"letter": "J", "column": "belly_circumference_cm_3d", "unit": "cm"},
    "max_abdominal_circumference": {"letter": "K", "column": "maximum_belly_circumference_cm_3d", "unit": "cm"},
    "waistband_circumference": {"letter": "L", "column": "waist_band_cm_3d", "unit": "cm"},
    "body_height": {"letter": "M", "column": "body_height_cm_3d", "unit": "cm"},
    "shoulder_width": {"letter": "N", "column": "cross_shoulder_over_neck_cm_3d", "unit": "cm"},
    "wrist_circumference": {"letter": "O", "column": "wrist_girth_cm_3d", "unit": "cm"},
    "left_thigh_circumference": {"letter": "P", "column": "thigh_girth_left_horizontal_cm_3d", "unit": "cm"},
    "right_thigh_circumference": {"letter": "Q", "column": "thigh_girth_right_horizontal_cm_3d", "unit": "cm"}
  }
}
BODY_MEASUREMENTS_SOURCE_07_END
cat > "$STAGE_DIR/body_measurements/targets.example.json" <<'BODY_MEASUREMENTS_SOURCE_08_END'
{
  "description": "Map confirmed manual-measurement columns and their source units. Null columns disable targets. Predictions are in cm.",
  "targets": {
    "bust_circumference": {"letter": "A", "column": null, "unit": null},
    "underbust_circumference": {"letter": "B", "column": null, "unit": null},
    "high_waist_circumference": {"letter": "C", "column": null, "unit": null},
    "waist_circumference": {"letter": "D", "column": null, "unit": null},
    "high_hip_circumference": {"letter": "E", "column": null, "unit": null},
    "hip_circumference": {"letter": "F", "column": null, "unit": null},
    "buttock_circumference": {"letter": "G", "column": null, "unit": null},
    "total_thigh_circumference": {"letter": "H", "column": null, "unit": null},
    "neck_circumference": {"letter": "I", "column": null, "unit": null},
    "abdominal_circumference": {"letter": "J", "column": null, "unit": null},
    "max_abdominal_circumference": {"letter": "K", "column": null, "unit": null},
    "waistband_circumference": {"letter": "L", "column": null, "unit": null},
    "body_height": {"letter": "M", "column": null, "unit": null},
    "shoulder_width": {"letter": "N", "column": null, "unit": null},
    "wrist_circumference": {"letter": "O", "column": null, "unit": null},
    "left_thigh_circumference": {"letter": "P", "column": null, "unit": null},
    "right_thigh_circumference": {"letter": "Q", "column": null, "unit": null}
  }
}
BODY_MEASUREMENTS_SOURCE_08_END
cat > "$STAGE_DIR/body_measurements/requirements.txt" <<'BODY_MEASUREMENTS_SOURCE_09_END'
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
joblib>=1.3
BODY_MEASUREMENTS_SOURCE_09_END
"$PY" - "$STAGE_DIR" <<'BODY_MEASUREMENTS_VERIFY_END'
from pathlib import Path
import hashlib
import json
import sys
expected = {'run_body_measurements.py': 'fd9f5f39e55cc13a2c7aec1a4a46b9d1957b7986b89f363579f9888d7cdc8473', 'build_scan_geometry_features.py': '0fe6a2129105a7992f5d9c484a40882fe39d6826d25a0bdafee505e79d91e637', 'body_measurements/__init__.py': 'aa72db8921c08bac60e888f8bec60878e21c947e16693169330834791b57cef5', 'body_measurements/data.py': '71f12f27dd1c1ba4af5315c8f0217c4bb79ab11e7d02bdd046a95a927aa545bf', 'body_measurements/joins.py': '6a1b9fe39f0a25d088b0c47f6ef314ecb0298035146a61fd95188d1f33b45a7a', 'body_measurements/models.py': '433da514dbd45ec0c72fadf6c5d55b6ac897903694cd115ce10c0a8e9de337ec', 'body_measurements/obj_features.py': 'b68fe20d4a1d9b79f7bfeaf0c6265e64d772c25df32955f40dc54e6a10fa584e', 'body_measurements/targets.validation_n1715.json': 'cc0e8a09ed63343bcba84b871e06313e588332d1468fad91002e0d5a659bd773', 'body_measurements/targets.example.json': '608bfc901e304a30011c9287af6698d57ccebfbfd37f20dd81950939420cdca0', 'body_measurements/requirements.txt': '7c3f12fba42757fefa424e0bd5ca28913427500a9fd90b319110b3b2557beb6d'}
stage = Path(sys.argv[1])
for name, checksum in expected.items():
    content = (stage / name).read_bytes()
    if hashlib.sha256(content).hexdigest() != checksum:
        raise SystemExit('Paste verification failed for ' + name + '; no existing files replaced.')
    if name.endswith('.py'):
        compile(content, name, 'exec')
    elif name.endswith('.json'):
        json.loads(content)
print('[CHECK] All 10 pasted files verified.')
BODY_MEASUREMENTS_VERIFY_END
BACKUP_DIR=$(mktemp -d "$PWD/measurement_code_backup_XXXXXXXX")
mkdir -p "$BACKUP_DIR/body_measurements" body_measurements
for FILE in run_body_measurements.py build_scan_geometry_features.py body_measurements/__init__.py body_measurements/data.py body_measurements/joins.py body_measurements/models.py body_measurements/obj_features.py body_measurements/targets.validation_n1715.json body_measurements/targets.example.json body_measurements/requirements.txt; do
    if [ -e "$FILE" ]; then
        cp -p "$FILE" "$BACKUP_DIR/$FILE"
    fi
done
for FILE in run_body_measurements.py build_scan_geometry_features.py body_measurements/__init__.py body_measurements/data.py body_measurements/joins.py body_measurements/models.py body_measurements/obj_features.py body_measurements/targets.validation_n1715.json body_measurements/targets.example.json body_measurements/requirements.txt; do
    cp "$STAGE_DIR/$FILE" "$FILE"
done
"$PY" run_body_measurements.py --help
printf '[INSTALLED] Measurement code in %s\n' "$PWD"
printf '[BACKUP] Previous files saved in %s\n' "$BACKUP_DIR"
mkdir -p measurement_runs
AUDIT_DIR=$(mktemp -d "$PWD/measurement_runs/n2894_metadata_audit_XXXXXXXX")
"$PY" run_body_measurements.py audit \
    --metadata "$META" \
    --id-col participant_id \
    --config body_measurements/targets.validation_n1715.json \
    --out-dir "$AUDIT_DIR"
cat "$AUDIT_DIR/audit.json"
BODY_MEASUREMENTS_TERMINAL_INSTALL_END
