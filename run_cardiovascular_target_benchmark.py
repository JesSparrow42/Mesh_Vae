#!/usr/bin/env python3
"""Train and test matched MeshVAE variants across cardiovascular targets."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from meshvae.conditioning import clean_column_names, clean_stregkode


@dataclass(frozen=True)
class TargetSpec:
    key: str
    column: str
    label: str
    kind: str
    transform: str = "identity"


TARGETS = [
    TargetSpec("carotid_right", "benchmark_carotid_right", "Right carotid plaque volume", "regression", "log1p"),
    TargetSpec("carotid_left", "benchmark_carotid_left", "Left carotid plaque volume", "regression", "log1p"),
    TargetSpec("carotid_total", "benchmark_carotid_total", "Combined carotid plaque volume", "regression", "log1p"),
    TargetSpec("femoral_right", "benchmark_femoral_right", "Right femoral plaque volume", "regression", "log1p"),
    TargetSpec("femoral_left", "benchmark_femoral_left", "Left femoral plaque volume", "regression", "log1p"),
    TargetSpec("femoral_total", "benchmark_femoral_total", "Combined femoral plaque volume", "regression", "log1p"),
    TargetSpec("coronary_volume", "benchmark_coronary_volume", "Coronary plaque volume", "regression", "log1p"),
    TargetSpec("ccta_sis", "benchmark_ccta_sis", "CCTA segment involvement score", "regression"),
    TargetSpec("cacs_coronary", "benchmark_cacs_coronary", "Coronary calcium score", "regression", "log1p"),
    TargetSpec("ccta_anycad", "benchmark_ccta_anycad", "Any coronary artery disease", "binary"),
    TargetSpec("anyplaq", "benchmark_anyplaq", "Any plaque at any measured site", "binary"),
]

VARIANTS = {
    "conv_scan_age": {
        "label": "MeshVAE convolution, scan + age",
        "point_labels": False,
    },
    "conv_pointlabels_scan_age": {
        "label": "MeshVAE convolution + point labels, scan + age",
        "point_labels": True,
    },
}


def _read_table(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline()
    sep = "\t" if header.count("\t") >= header.count(",") else ","
    return clean_column_names(
        pd.read_csv(path, sep=sep, dtype=str, engine="python", on_bad_lines="warn")
    )


def _resolve_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias in frame.columns:
            return alias
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _id_column(frame: pd.DataFrame) -> str:
    column = _resolve_column(frame, ["participant_id", "stregkode", "participantid", "id"])
    if column is None:
        raise KeyError(f"Could not find a participant ID column. Columns: {list(frame.columns)[:60]}")
    return column


def _numeric(frame: pd.DataFrame, aliases: list[str]) -> tuple[pd.Series, str | None]:
    column = _resolve_column(frame, aliases)
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64), None
    values = pd.to_numeric(
        frame[column].astype(str).str.strip().str.strip('"').str.strip("'").str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return values.astype(np.float64), column


def _enrich_split(master: pd.DataFrame, split: pd.DataFrame, split_name: str) -> pd.DataFrame:
    master = master.copy()
    split = split.copy()
    master["_clean_id"] = master[_id_column(master)].map(clean_stregkode)
    split["_clean_id"] = split[_id_column(split)].map(clean_stregkode)
    master = master.dropna(subset=["_clean_id"]).drop_duplicates("_clean_id", keep="first")
    split = split.dropna(subset=["_clean_id"]).drop_duplicates("_clean_id", keep="first")
    master_by_id = master.set_index("_clean_id", drop=False)
    split_by_id = split.set_index("_clean_id", drop=False)
    ids = split["_clean_id"].tolist()
    enriched = master_by_id.reindex(ids).copy()
    missing_master = enriched["_clean_id"].isna()
    if bool(missing_master.any()):
        missing_ids = [ids[index] for index in np.flatnonzero(missing_master.to_numpy())]
        print(
            f"[SPLIT] {split_name}: {len(missing_ids)} IDs were absent from master metadata; "
            "their existing split values will be retained.",
            flush=True,
        )
        enriched.loc[missing_ids, "_clean_id"] = missing_ids
    for column in split.columns:
        if column == "_clean_id":
            continue
        split_values = split_by_id.reindex(ids)[column]
        if column not in enriched.columns:
            enriched[column] = split_values
        else:
            enriched[column] = enriched[column].where(enriched[column].notna(), split_values)
    enriched["participant_id"] = ids
    enriched = enriched.reset_index(drop=True).drop(columns=["_clean_id"], errors="ignore")
    print(f"[SPLIT] {split_name}: preserved {len(enriched)} participant IDs.", flush=True)
    return enriched


def _filter_gender(frame: pd.DataFrame, gender: str, split_name: str) -> pd.DataFrame:
    column = _resolve_column(frame, ["sexMF", "sex", "gender"])
    if column is None:
        raise KeyError(f"Could not find sex/gender column in enriched {split_name} metadata.")
    clean = (
        frame[column].astype(str).str.strip().str.strip('"').str.strip("'").str.upper()
        .replace({"FEMALE": "F", "MALE": "M"})
    )
    keep = clean == str(gender).upper()
    if not bool(keep.all()):
        print(
            f"[SPLIT] {split_name}: removed {int((~keep).sum())} rows outside gender={gender}.",
            flush=True,
        )
    return frame.loc[keep].reset_index(drop=True)


def _prepare_targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.copy()
    source_columns: dict[str, Any] = {}

    right_caro, source_columns["carotid_right"] = _numeric(
        frame, ["plaque_volume_right_carotid_corrected", "plaque_volume_right_carotid"]
    )
    left_caro, source_columns["carotid_left"] = _numeric(
        frame, ["plaque_volume_left_carotid_corrected", "plaque_volume_left_carotid"]
    )
    right_femo, source_columns["femoral_right"] = _numeric(
        frame, ["plaque_volume_right_femoral_corrected", "plaque_volume_right_femoral"]
    )
    left_femo, source_columns["femoral_left"] = _numeric(
        frame, ["plaque_volume_left_femoral_corrected", "plaque_volume_left_femoral"]
    )

    caro_direct, source_columns["carotid_total"] = _numeric(frame, ["pv_caro"])
    femo_direct, source_columns["femoral_total"] = _numeric(frame, ["pv_femo"])
    coronary, coronary_primary = _numeric(frame, ["pv_coro"])
    coronary_fallback, coronary_secondary = _numeric(
        frame, ["ct_qa_cor_tot_plaq_vol", "coronary_plaque_volume"]
    )
    coronary = coronary.where(coronary.notna(), coronary_fallback)
    source_columns["coronary_volume"] = [
        column for column in [coronary_primary, coronary_secondary] if column is not None
    ]
    sis, source_columns["ccta_sis"] = _numeric(frame, ["ccta_sis"])
    cacs, source_columns["cacs_coronary"] = _numeric(
        frame, ["cacs_coronary", "coronary_calcium_score", "cacs"]
    )
    any_cad, source_columns["ccta_anycad"] = _numeric(
        frame, ["ccta_anycad", "any_cad", "ccta_any_cad"]
    )
    any_plaque, source_columns["anyplaq"] = _numeric(
        frame, ["anyplaq", "any_plaq", "any_plaque"]
    )

    caro_sum = pd.concat([right_caro, left_caro], axis=1).sum(axis=1, min_count=2)
    femo_sum = pd.concat([right_femo, left_femo], axis=1).sum(axis=1, min_count=2)
    caro_total = caro_direct.where(caro_direct.notna(), caro_sum)
    femo_total = femo_direct.where(femo_direct.notna(), femo_sum)
    location_values = pd.concat(
        [right_caro, left_caro, right_femo, left_femo, coronary], axis=1
    )
    derived_any = (location_values.max(axis=1, skipna=True) > 0.0).astype(np.float64)
    derived_any[location_values.notna().sum(axis=1) == 0] = np.nan
    any_plaque = any_plaque.where(any_plaque.notna(), derived_any)

    frame["benchmark_carotid_right"] = right_caro
    frame["benchmark_carotid_left"] = left_caro
    frame["benchmark_carotid_total"] = caro_total
    frame["benchmark_femoral_right"] = right_femo
    frame["benchmark_femoral_left"] = left_femo
    frame["benchmark_femoral_total"] = femo_total
    frame["benchmark_coronary_volume"] = coronary
    frame["benchmark_ccta_sis"] = sis
    frame["benchmark_cacs_coronary"] = cacs
    frame["benchmark_ccta_anycad"] = any_cad
    frame["benchmark_anyplaq"] = any_plaque

    audit: dict[str, Any] = {"source_columns": source_columns}
    for key, direct, derived in [
        ("pv_caro_vs_left_plus_right", caro_direct, caro_sum),
        ("pv_femo_vs_left_plus_right", femo_direct, femo_sum),
    ]:
        comparable = direct.notna() & derived.notna()
        mismatch = comparable & ~np.isclose(direct, derived, rtol=1e-4, atol=1e-6)
        audit[key] = {
            "comparable": int(comparable.sum()),
            "mismatches": int(mismatch.sum()),
        }
    return frame, audit


def _audit_target(frame: pd.DataFrame, spec: TargetSpec, split: str) -> dict[str, Any]:
    values = pd.to_numeric(frame[spec.column], errors="coerce")
    finite = values[np.isfinite(values)]
    row: dict[str, Any] = {
        "target_key": spec.key,
        "target": spec.label,
        "column": spec.column,
        "kind": spec.kind,
        "split": split,
        "rows": int(len(frame)),
        "available": int(len(finite)),
        "missing": int(len(frame) - len(finite)),
        "unique": int(finite.nunique()),
    }
    if spec.kind == "binary":
        row["negative"] = int(np.isclose(finite, 0.0).sum())
        row["positive"] = int(np.isclose(finite, 1.0).sum())
        row["invalid"] = int((~np.isclose(finite, 0.0) & ~np.isclose(finite, 1.0)).sum())
        row["prevalence"] = float(np.mean(finite == 1.0)) if len(finite) else float("nan")
    elif len(finite):
        quantiles = finite.quantile([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
        row.update(
            {
                "zero": int(np.isclose(finite, 0.0).sum()),
                "negative": int((finite < 0.0).sum()),
                "min": float(quantiles.loc[0.0]),
                "q25": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.5]),
                "q75": float(quantiles.loc[0.75]),
                "q90": float(quantiles.loc[0.9]),
                "max": float(quantiles.loc[1.0]),
            }
        )
    return row


def _target_is_usable(train_audit: dict[str, Any], test_audit: dict[str, Any], spec: TargetSpec) -> tuple[bool, str]:
    if train_audit["available"] < 50 or test_audit["available"] < 20:
        return False, "fewer than 50 training or 20 held-out observations"
    if spec.kind == "binary":
        if train_audit.get("invalid", 0) or test_audit.get("invalid", 0):
            return False, "binary target contains values other than 0 and 1"
        if min(train_audit.get("negative", 0), train_audit.get("positive", 0)) < 5:
            return False, "training split has fewer than five observations in one class"
        if min(test_audit.get("negative", 0), test_audit.get("positive", 0)) < 1:
            return False, "held-out split does not contain both classes"
    elif min(train_audit["unique"], test_audit["unique"]) < 2:
        return False, "continuous target has no variation in one split"
    elif spec.transform == "log1p" and (
        train_audit.get("negative", 0) > 0 or test_audit.get("negative", 0) > 0
    ):
        return False, "log1p target contains negative values"
    return True, ""


def _run(command: list[str], dry_run: bool) -> None:
    print("\n[COMMAND] " + " ".join(shlex.quote(str(value)) for value in command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return frame.iloc[0].to_dict() if len(frame) else {}


def _collect_rows(
    selected_targets: list[TargetSpec],
    selected_variants: list[str],
    args: argparse.Namespace,
    unavailable: dict[str, str],
    failures: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for spec in selected_targets:
        for variant_key in selected_variants:
            experiment = f"{args.name_prefix}_{spec.key}_{variant_key}"
            run_dir = Path(args.checkpoints_dir) / experiment
            row: dict[str, Any] = {
                "Experiment": experiment,
                "TargetKey": spec.key,
                "Target": spec.label,
                "TargetColumn": spec.column,
                "OutcomeType": spec.kind,
                "Transform": spec.transform if spec.kind == "regression" else "",
                "Variant": variant_key,
                "Model": VARIANTS[variant_key]["label"],
                "Age": "Yes",
                "BodyScan": "Yes",
                "PointLabels": "Yes" if VARIANTS[variant_key]["point_labels"] else "No",
                "Status": "pending",
            }
            if spec.key in unavailable:
                row.update({"Status": "unavailable", "Error": unavailable[spec.key]})
            else:
                summary_name = "regression_summary.csv" if spec.kind == "regression" else "binary_summary.csv"
                summary = _read_summary(run_dir / summary_name)
                if summary:
                    row.update(summary)
                    row.update(
                        {
                            "Experiment": experiment,
                            "TargetKey": spec.key,
                            "Target": spec.label,
                            "TargetColumn": spec.column,
                            "OutcomeType": spec.kind,
                            "Transform": spec.transform if spec.kind == "regression" else "",
                            "Variant": variant_key,
                            "Model": VARIANTS[variant_key]["label"],
                            "Age": "Yes",
                            "BodyScan": "Yes",
                            "PointLabels": "Yes" if VARIANTS[variant_key]["point_labels"] else "No",
                        }
                    )
                elif (run_dir / "best_vae_net.pth").exists():
                    row["Status"] = "trained_not_diagnosed"
            if experiment in failures and row["Status"] != "completed":
                row.update({"Status": "failed", "Error": failures[experiment]})
            rows.append(row)
    return rows


def _write_results(rows: list[dict[str, Any]], args: argparse.Namespace, final: bool) -> None:
    out_dir = Path(args.checkpoints_dir) / args.name_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "cardiovascular_target_results" if final else "cardiovascular_target_results_partial"
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / f"{stem}.csv", index=False)

    regression_columns = ["Target", "PointLabels", "Status", "N", "R2", "MAE", "RMSE", "Pearson", "Spearman"]
    binary_columns = [
        "Target", "PointLabels", "Status", "N", "Positive", "Prevalence", "AUC",
        "AveragePrecision", "BalancedAccuracy", "Sensitivity", "Specificity", "F1",
    ]
    regression = frame[frame["OutcomeType"] == "regression"].reindex(columns=regression_columns)
    binary = frame[frame["OutcomeType"] == "binary"].reindex(columns=binary_columns)
    lines = [
        "# Cardiovascular Target Benchmark", "",
        f"Train split: `{args.train_split_tsv}`", f"Test split: `{args.test_split_tsv}`", "",
        "## Continuous targets", "", "```text", regression.to_string(index=False), "```", "",
        "## Binary targets", "", "```text", binary.to_string(index=False), "```", "",
        "R2, MAE, and RMSE apply to continuous targets. AUC and average precision are the primary binary metrics.",
        "Binary threshold metrics use a fixed probability threshold of 0.5.", "",
    ]
    (out_dir / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[TABLE] wrote {out_dir / f'{stem}.csv'}", flush=True)
    print(f"[TABLE] wrote {out_dir / f'{stem}.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark convolutional scan+age MeshVAEs across cardiovascular targets."
    )
    parser.add_argument("--name_prefix", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--metadata_tsv", required=True)
    parser.add_argument("--train_split_tsv", required=True)
    parser.add_argument("--test_split_tsv", required=True)
    parser.add_argument("--checkpoints_dir", default="/data/projects/3D_scans/ML/MQVAE/checkpoints")
    parser.add_argument("--only_gender", choices=["M", "F"], default="F")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--niter", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--split_ratio", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--beta_warmup_epochs", type=int, default=30)
    parser.add_argument("--regression_weight", type=float, default=20.0)
    parser.add_argument("--binary_weight", type=float, default=10.0)
    parser.add_argument("--encoder_max_points", type=int, default=4096)
    parser.add_argument("--mesh_knn_k", type=int, default=12)
    parser.add_argument("--mesh_knn_max_points", type=int, default=4096)
    parser.add_argument("--point_label_regions", type=int, default=6)
    parser.add_argument("--scan_geometry_features_file", default="scan_geometry_features.npz")
    parser.add_argument("--targets", nargs="+", choices=[target.key for target in TARGETS], default=None)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--audit_only",
        action="store_true",
        help="Build enriched splits and report target availability without training models.",
    )
    parser.add_argument("--diagnose_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = Path(args.checkpoints_dir) / args.name_prefix
    split_dir = out_dir / "enriched_splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    master = _read_table(Path(args.metadata_tsv))
    raw_train = _read_table(Path(args.train_split_tsv))
    raw_test = _read_table(Path(args.test_split_tsv))
    train = _enrich_split(master, raw_train, "train")
    test = _enrich_split(master, raw_test, "test")
    train = _filter_gender(train, args.only_gender, "train")
    test = _filter_gender(test, args.only_gender, "test")
    overlap = set(train["participant_id"]) & set(test["participant_id"])
    if overlap:
        raise ValueError(f"Train/test participant overlap detected: {sorted(overlap)[:20]}")

    combined, derivation_audit = _prepare_targets(
        pd.concat([train.assign(_benchmark_split="train"), test.assign(_benchmark_split="test")], ignore_index=True)
    )
    train = combined[combined["_benchmark_split"] == "train"].drop(columns="_benchmark_split").copy()
    test = combined[combined["_benchmark_split"] == "test"].drop(columns="_benchmark_split").copy()
    train_path = split_dir / "train_enriched.tsv"
    test_path = split_dir / "test_enriched.tsv"
    train.to_csv(train_path, sep="\t", index=False)
    test.to_csv(test_path, sep="\t", index=False)
    print(f"[OUT] enriched train metadata -> {train_path}", flush=True)
    print(f"[OUT] enriched test metadata  -> {test_path}", flush=True)

    selected_keys = set(args.targets or [target.key for target in TARGETS])
    selected_targets = [target for target in TARGETS if target.key in selected_keys]
    selected_variants = list(args.variants)
    audits = []
    unavailable: dict[str, str] = {}
    for spec in selected_targets:
        train_audit = _audit_target(train, spec, "train")
        test_audit = _audit_target(test, spec, "test")
        audits.extend([train_audit, test_audit])
        usable, reason = _target_is_usable(train_audit, test_audit, spec)
        if not usable:
            unavailable[spec.key] = reason
        print(
            f"[TARGET] {spec.key}: train={train_audit['available']} test={test_audit['available']} "
            f"status={'usable' if usable else reason}",
            flush=True,
        )
    pd.DataFrame(audits).to_csv(out_dir / "target_availability_audit.csv", index=False)
    (out_dir / "target_derivation_audit.json").write_text(
        json.dumps(derivation_audit, indent=2), encoding="utf-8"
    )

    if args.audit_only:
        print(
            f"\n[AUDIT] complete; no models were trained. Results: "
            f"{out_dir / 'target_availability_audit.csv'}",
            flush=True,
        )
        return

    failures: dict[str, str] = {}
    _write_results(
        _collect_rows(selected_targets, selected_variants, args, unavailable, failures),
        args,
        final=False,
    )

    for spec in selected_targets:
        if spec.key in unavailable:
            continue
        for variant_key in selected_variants:
            experiment = f"{args.name_prefix}_{spec.key}_{variant_key}"
            run_dir = Path(args.checkpoints_dir) / experiment
            checkpoint = run_dir / "best_vae_net.pth"
            try:
                if not args.diagnose_only and not (args.resume and checkpoint.exists()):
                    command = [
                        sys.executable, str(root / "train_mesh_vae.py"),
                        "--phase", "train", "--name", experiment,
                        "--checkpoints_dir", args.checkpoints_dir,
                        "--dataroot", args.dataroot, "--cond_tsv", str(train_path),
                        "--feature_source", "scan", "--only_gender", args.only_gender,
                        "--gpu_ids", args.gpu_ids, "--niter", str(args.niter),
                        "--batch_size", str(args.batch_size), "--split_ratio", str(args.split_ratio),
                        "--split_seed", str(args.split_seed), "--beta", str(args.beta),
                        "--beta_warmup_epochs", str(args.beta_warmup_epochs),
                        "--use_preprocessed", "--use_pointcloud", "--use_cond",
                        "--encoder_arch", "meshconv", "--encoder_cond_mode", "demo",
                        "--encoder_max_points", str(args.encoder_max_points),
                        "--mesh_use_knn", "--mesh_knn_k", str(args.mesh_knn_k),
                        "--mesh_knn_max_points", str(args.mesh_knn_max_points),
                        "--use_scan_geometry_features", "--require_scan_geometry_features",
                        "--scan_geometry_features_file", args.scan_geometry_features_file,
                        "--restore_metric_scale", "--no-use_amp", "--no-cond_athero",
                        "--no-cond_plaque", "--no-aux_predict_athero", "--no-aux_predict_plaque",
                        "--no-balance_athero",
                    ]
                    if VARIANTS[variant_key]["point_labels"]:
                        command.extend(
                            [
                                "--use_point_labels", "--point_label_pooling",
                                "--point_label_regions", str(args.point_label_regions),
                            ]
                        )
                    else:
                        command.extend(["--no-use_point_labels", "--no-point_label_pooling"])
                    if spec.kind == "regression":
                        command.extend(
                            [
                                "--aux_predict_regression", "--no-aux_predict_binary",
                                "--aux_regression_col", spec.column,
                                "--aux_regression_transform", spec.transform,
                                "--aux_regression_weight", str(args.regression_weight),
                            ]
                        )
                    else:
                        command.extend(
                            [
                                "--aux_predict_binary", "--no-aux_predict_regression",
                                "--aux_binary_col", spec.column,
                                "--aux_binary_weight", str(args.binary_weight),
                                "--aux_binary_pos_weight", "0",
                            ]
                        )
                    _run(command, args.dry_run)
                elif args.diagnose_only and not checkpoint.exists():
                    raise FileNotFoundError(f"Missing checkpoint for --diagnose_only: {checkpoint}")
                else:
                    print(f"[RESUME] checkpoint exists; skipping training: {checkpoint}", flush=True)

                summary_name = "regression_summary.csv" if spec.kind == "regression" else "binary_summary.csv"
                summary_path = run_dir / summary_name
                if not (args.resume and summary_path.exists()):
                    diagnosis_script = (
                        root / "diagnose_mesh_regression.py"
                        if spec.kind == "regression"
                        else root / "diagnose_mesh_binary.py"
                    )
                    diagnosis = [
                        sys.executable, str(diagnosis_script),
                        "--name", experiment, "--checkpoints_dir", args.checkpoints_dir,
                        "--dataroot", args.dataroot,
                        "--cond_tsv", str(test_path), "--feature_source", "scan",
                        "--only_gender", args.only_gender, "--gpu_ids", args.gpu_ids,
                        "--which_epoch", "best", "--batch_size", str(args.batch_size), "--use_cond",
                    ]
                    _run(diagnosis, args.dry_run)
                else:
                    print(f"[RESUME] diagnosis exists; skipping: {summary_path}", flush=True)
            except Exception as exc:
                failures[experiment] = f"{type(exc).__name__}: {exc}"
                print(f"[ERROR] {experiment}: {failures[experiment]}", flush=True)
                if not args.continue_on_error:
                    raise
            _write_results(
                _collect_rows(selected_targets, selected_variants, args, unavailable, failures),
                args,
                final=False,
            )

    _write_results(
        _collect_rows(selected_targets, selected_variants, args, unavailable, failures),
        args,
        final=not args.dry_run,
    )
    print("\n========== CARDIOVASCULAR TARGET BENCHMARK COMPLETE ==========", flush=True)


if __name__ == "__main__":
    main()
