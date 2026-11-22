#!/usr/bin/env python3
"""Inventory saved result tables; no training, ranking, or metric recomputation.

Collect on each server with repeated --root and one --platform label, then merge
the exported all_results.csv files with --merge. Original source fields and
provenance are retained. Unknown evaluation splits are explicitly unverified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd


OWN = {"all_results.csv", "confidence_intervals.csv", "result_inventory.csv", "plot_artifacts.csv"}
METADATA = ("opt.txt", "training_summary.json", "run.json", "analysis_manifest.json", "study/manifest.json")
ALIASES = {
    "model": ("Model", "model", "Architecture", "architecture", "Arm", "experiment"),
    "run": ("Experiment", "experiment", "name", "Run", "run"),
    "target": ("TargetColumn", "target_column", "aux_regression_col", "aux_binary_col", "Target", "target"),
    "cohort": ("Cohort", "cohort", "experiment.cohort"),
    "gender": ("Gender", "gender", "sex", "Sex", "only_gender", "settings.only_gender"),
    "seed": ("Seed", "seed", "config.seed"),
    "prior": ("LatentPrior", "latent_prior", "prior"),
    "task": ("OutcomeType", "task", "experiment.kind"),
    "target_transform": ("TargetTransform", "target_transform", "aux_regression_transform", "Transform"),
}


def flatten(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in value.items():
        key = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, key))
        else:
            result[key] = json.dumps(item) if isinstance(item, list) else item
    return result


def pick(mapping: dict, names, default=""):
    for name in names:
        value = mapping.get(name)
        if value is not None and str(value).strip().lower() not in {"", "nan", "none", "null"}:
            return value
    return default


def kind(path: Path) -> str | None:
    name = path.name.lower()
    if name in OWN or any(word in name for word in ("prediction", "latent", "correlation", "availability", "cohort_flow")):
        return None
    if path.suffix.lower() == ".json":
        if name == "intervals.json":
            return "confidence_interval"
        return "json_fallback" if name in {"summary.json", "metrics.json"} else None
    if path.suffix.lower() != ".csv":
        return None
    if any(word in name for word in ("interval", "paired_difference", "paired_model_difference")):
        return "confidence_interval"
    if name == "transcribed_results.csv":
        return "screenshot_transcription"
    if name.startswith("boson_ablation_aggregate"):
        return "seed_aggregate"
    if name.startswith("boson_ablation_paired_effect"):
        return "seed_paired_effect"
    if ("summary" in name or "results" in name
            or name.startswith("boson_ablation_runs") or name in {
                "comparison.csv", "metrics.csv", "hba1c_body_size_benchmark.csv", "hba1c_ordinary_least_squares.csv"}):
        return "saved_summary"
    return None


def read_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("Summary JSON must be an object")
        if path.name == "intervals.json":
            rows = []
            for field, model in (("meshvae_95_percentile_ci", "meshvae"), ("meshvae_minus_age_weight_95_percentile_ci", "meshvae_minus_age_weight")):
                bounds = value.get(field)
                if isinstance(bounds, list) and len(bounds) == 2:
                    rows.append({**flatten(value), "model": model, "ci_low": bounds[0], "ci_high": bounds[1], "resampling_unit": "participant", "ci_source_field": field})
            if not rows:
                raise ValueError("No supported saved plaque-study confidence intervals")
            return rows
        return [flatten(value)]
    return pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")


def context(path: Path, root: Path) -> tuple[dict, Path, list[str]]:
    """Nearest metadata wins; a result root need not contain opt.txt."""
    data, errors, owner, owner_found = {}, [], path.parent, False
    parents = [path.parent, *path.parent.parents]
    parents = parents[:parents.index(root) + 1]
    for directory in parents:
        for filename in METADATA:
            source = directory / filename
            if not source.is_file():
                continue
            try:
                if filename == "opt.txt":
                    parsed = dict(re.findall(r"^\s*([A-Za-z_][\w]*)\s*:\s*(.*?)\s*(?:\[default:.*)?$", source.read_text(encoding="utf-8-sig"), re.M))
                else:
                    parsed = flatten(json.loads(source.read_text(encoding="utf-8-sig")))
                for key, value in parsed.items():
                    data.setdefault(key, value)
            except Exception as exc:
                errors.append(f"{source}: {type(exc).__name__}: {exc}")
        if not owner_found and any((directory / name).exists() for name in ("opt.txt", "training_summary.json", "study/manifest.json", "binary_summary.csv", "regression_summary.csv", "summary_row.csv")):
            owner = directory
            owner_found = True
    # Diagnostic families identify their owning run even without metadata.
    for directory in parents:
        if directory.name in {"diagnostics", "binary_diagnostics", "regression_diagnostics"}:
            owner = directory.parent
            break
    return data, owner, errors


def normalize(raw: dict, path: Path, root: Path, platform: str, source_kind: str, number: int, ctx: tuple) -> dict:
    metadata, owner, _ = ctx
    row = dict(raw)
    row["original_fields_json"] = json.dumps(raw, ensure_ascii=False, default=str)
    for key, aliases in ALIASES.items():
        meta_aliases = [name for name in aliases if not name.startswith("aux_") or str(metadata.get("aux_predict_binary" if "binary" in name else "aux_predict_regression", "")).lower() in {"true", "1"}]
        row[key] = pick(raw, aliases, pick(metadata, meta_aliases))
    row["run"] = row["run"] or owner.name
    row["model"] = row["model"] or row["run"]
    row["reference_run"] = pick(metadata, ("reference.run",))
    row["reference_fingerprint"] = pick(metadata, ("reference.fingerprint",))
    row["reference_split_hashes"] = json.dumps({key.rsplit(".", 1)[-1]: value for key, value in metadata.items()
                                               if key.startswith("reference.split_file_sha256.")}, sort_keys=True)
    if not row["target"] and metadata.get("experiment.territory"):
        row["target"] = ("presence_" if row["task"] == "binary" else "volume_") + str(metadata["experiment.territory"])
    if not row["task"]:
        keys = {key.lower() for key in raw if pick(raw, (key,)) != ""}
        if any(key in keys for key in ("auc", "test_auc", "test.roc_auc", "roc_auc", "balancedaccuracy")):
            row["task"] = "binary"
        elif any(key in keys for key in ("r2", "test_r2", "mae", "test_mae", "rmse")):
            row["task"] = "regression"
    metric = str(pick(raw, ("metric", "Metric"))).lower()
    row["metric_scale"] = "original_target_units" if row["task"] in {"regression", "hurdle"} else "unspecified"
    if metric.startswith("transformed_"):
        row["metric_scale"] = "transformed"
    elif any(key.lower().startswith("transformed") for key in raw):
        row["metric_scale"] = "original_and_transformed_columns"
    split = pick(raw, ("split", "Split", "evaluation_split"))
    if not split and ("heldout" in path.parts or any(key.lower().startswith(("test_", "test.")) for key in raw)):
        split = "test"
    row.update(platform=platform, source_file=str(path), source_row=number, source_root=str(root),
               source_kind=source_kind, run_directory=str(owner), evaluation_split=split or "unverified",
               preferred=True, preference_reason="only_saved_source", ci_available=False, ci_files="", evidence_level="saved_result")
    if source_kind == "screenshot_transcription":
        row.update(preferred=False, preference_reason="transcribed_reference", evaluation_split="unverified", evidence_level="transcribed_screenshot")
        group = str(raw.get("group", ""))
        if not row["target"] and group in {"total_volume", "weight_classification", "weight_regression"}:
            row["target"] = "peripheral_total"
        elif not row["target"] and group in {"territory_classification", "territory_volume"}:
            row["target"] = re.sub(r"_(presence|volume)$", "", str(raw.get("experiment", "")))
        row["normalization_note"] = "Target territory inferred from transcribed group/experiment labels; original run unavailable"
    row["result_scope"] = source_kind if source_kind in {"seed_aggregate", "seed_paired_effect"} else "model_evaluation"
    return row


def choose_preferred(rows: list[dict]) -> None:
    """Prefer final tables and direct summaries, retaining every alternative."""
    groups = {}
    files = {row["source_file"] for row in rows}
    for row in rows:
        path = Path(row["source_file"])
        if row["source_kind"] == "screenshot_transcription":
            continue
        if "_partial" in path.stem and str(path.with_name(path.name.replace("_partial", ""))) in files:
            row.update(preferred=False, preference_reason="partial_with_final_available")
            continue
        if "_partial" in path.stem:
            row["preference_reason"] = "partial_only"
        ref = pick(row, ("ResultsDir", "results_dir", "Checkpoint", "checkpoint"))
        if not ref and row["source_kind"] == "json_fallback":
            ref = str(path.parent)
        if not ref:
            continue
        # A result pointer is reliable across transferred Windows/Linux paths.
        ref = str(ref).replace("\\", "/").rstrip("/")
        parts = ref.split("/")
        for family in ("diagnostics", "binary_diagnostics", "regression_diagnostics"):
            if family in parts:
                ref = "/".join(parts[max(0, parts.index(family) - 1):])
                break
        key = (row["platform"], row["source_root"], ref, str(row["target"]), str(row["evaluation_split"]))
        priority = 3 if path.name in {"binary_summary.csv", "regression_summary.csv", "summary_row.csv"} else 2
        if row["source_kind"] == "json_fallback":
            priority = 1
        groups.setdefault(key, []).append((priority, row))
    for entries in groups.values():
        # Keep all rows of a multi-model table: only suppress another artifact.
        selected = max(entries, key=lambda item: (item[0], item[1]["source_file"]))[1]["source_file"]
        for _, row in entries:
            if row["source_file"] != selected:
                row.update(preferred=False, preference_reason="reference_copy")
    # Timestamped diagnostic exports are historical when a later folder exists.
    families, pointers = {}, {}
    for row in rows:
        path = Path(row["source_file"])
        ref = str(pick(row, ("ResultsDir", "results_dir"))).replace("\\", "/").split("/")
        if path.name in {"binary_summary.csv", "regression_summary.csv"} and len(ref) >= 2:
            pointers[(row["platform"], str(path.parent / ref[-2]))] = ref[-1]
        if re.fullmatch(r"latest_\d{8}_\d{6}", path.parent.name):
            families.setdefault((row["platform"], str(path.parent.parent)), []).append(row)
    for key, family in families.items():
        latest = pointers.get(key) or max(Path(row["source_file"]).parent.name for row in family)
        for row in family:
            if Path(row["source_file"]).parent.name != latest:
                row.update(preferred=False, preference_reason="historical_diagnostic_timestamp")


def collect(roots: list[Path], platform: str, output: Path):
    results, intervals, inventory, valid_roots, seen = [], [], [], [], set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            inventory.append(dict(platform=platform, source_file=str(root), status="missing_root", error="Directory unavailable"))
            continue
        valid_roots.append(str(root))
        markers, summarized = set(), set()
        for path in sorted(root.rglob("*")):
            if not path.is_file() or output in path.parents or path in seen:
                continue
            seen.add(path)
            if path.name == "opt.txt" or path.suffix.lower() in {".pth", ".pt"}:
                markers.add(path.parent)
            source_kind = kind(path)
            if source_kind is None:
                continue
            entry = dict(platform=platform, source_file=str(path), source_root=str(root), source_kind=source_kind, status="failed", rows=0, error="")
            try:
                ctx = context(path, root)
                records = read_records(path)
                normalized = [normalize(raw, path, root, platform, source_kind, i + 1, ctx) for i, raw in enumerate(records)]
                if source_kind == "confidence_interval":
                    for row in normalized:
                        row["resampling_unit"] = pick(row, ("resampling_unit",), "unspecified_in_source")
                    intervals.extend(normalized)
                else:
                    results.extend(normalized)
                    if source_kind in {"seed_aggregate", "seed_paired_effect"}:
                        for raw, row in zip(records, normalized):
                            if any(re.search(r"(?:95ci|_ci_|^ci_)", key.lower()) for key in raw):
                                intervals.append({**row, "resampling_unit": "seed"})
                summarized.update((path.parent, ctx[1]))
                entry.update(status="collected" if records else "empty", rows=len(records), metadata_errors=" | ".join(ctx[2]))
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            inventory.append(entry)
        for directory in sorted(markers - summarized):
            inventory.append(dict(platform=platform, source_file=str(directory), source_root=str(root), status="saved_run_without_recognized_summary", rows=0, error=""))
    if not valid_roots:
        raise ValueError("None of the requested roots exists as a directory")
    choose_preferred(results)
    # Match explicit model names; a paired contrast is not a direct-model CI.
    for owner in {row["run_directory"] for row in intervals}:
        matching = [row for row in results if row["run_directory"] == owner and row["preferred"]]
        identities = {(row["model"], row["target"]) for row in matching}
        direct = [row for row in intervals if row["run_directory"] == owner and "minus" not in str(row["model"]) and not pick(row, ("reference_model", "comparison_model"))]
        for row in matching:
            ci_files = sorted({ci["source_file"] for ci in direct if ci["model"] == row["model"] or (len(identities) == 1 and not ci["target"] and not row["target"])})
            if not ci_files and len(identities) == 1 and len({ci["model"] for ci in direct}) == 1:
                ci_files = sorted({ci["source_file"] for ci in direct if not ci["target"] or ci["target"] == row["target"]})
            if ci_files:
                row.update(ci_available=True, ci_files=json.dumps(ci_files))
    return results, intervals, inventory, valid_roots


def merge(paths: list[Path]):
    tables = {name: [] for name in OWN}
    for source in paths:
        source = source.expanduser().resolve()
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
        if not {"platform", "source_file", "source_row"}.issubset(frame):
            raise ValueError(f"{source}: expected an all_results.csv export from this collector")
        frame["export_file"] = str(source)
        tables["all_results.csv"].extend(frame.to_dict("records"))
        for name in ("confidence_intervals.csv", "result_inventory.csv", "plot_artifacts.csv"):
            sibling = source.parent / name
            if sibling.exists():
                extra = pd.read_csv(sibling, dtype=str, keep_default_na=False)
                extra["export_file"] = str(source)
                tables[name].extend(extra.to_dict("records"))
    return tables


def export_plot_data(roots: list[Path], platform: str, output: Path) -> list[dict]:
    """Copy plot inputs with an original-path map; omit weights and raw scans."""
    rows, seen, linked = [], set(), []

    def copy(source: Path, destination: Path, root: Path, artifact_kind: str, owner: Path):
        source = source.expanduser().resolve()
        if source in seen:
            return
        seen.add(source)
        row = dict(platform=platform, source_root=str(root), source_file=str(source),
                   exported_file=destination.relative_to(output).as_posix(),
                   artifact_kind=artifact_kind, run_directory=str(owner), status="failed", error="")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            row["status"] = "copied"
        except OSError as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    for number, root in enumerate(roots, start=1):
        root = root.expanduser().resolve()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or output in path.parents:
                continue
            name = path.name.lower()
            artifact_kind = None
            if path.suffix.lower() == ".csv":
                if "predictions" in name:
                    artifact_kind = "prediction"
                elif name.endswith("latent_embeddings.csv"):
                    artifact_kind = "latent"
                elif name == "tsne_coordinates.csv":
                    artifact_kind = "coordinates"
            elif path.suffix.lower() == ".tsv" and (name in {"train.tsv", "test.tsv"} or name.endswith("_enriched.tsv")):
                artifact_kind = "metadata"
            elif name in {"opt.txt", "latent_manifest.json", "train_command.json"} or (name == "manifest.json" and "study" in path.parts):
                artifact_kind = "configuration"
            if artifact_kind is None:
                continue
            owner = context(path, root)[1]
            destination = output / "plot_data" / f"root_{number:02d}_{root.name}" / path.relative_to(root)
            copy(path, destination, root, artifact_kind, owner)
            if name == "latent_manifest.json":
                try:
                    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
                    for key in ("train_tsv", "test_tsv"):
                        if manifest.get(key):
                            source = Path(manifest[key]).expanduser()
                            if not source.is_absolute():
                                source = path.parent / source
                            linked.append((source.resolve(), root, owner))
                except (OSError, ValueError) as exc:
                    rows[-1]["error"] = f"Manifest metadata references unavailable: {exc}"
    for source, root, owner in linked:
        digest = hashlib.sha256(str(source).encode()).hexdigest()[:12]
        copy(source, output / "plot_data" / "external" / f"{digest}_{source.name}", root, "metadata", owner)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--root", type=Path, action="append", help="Repeat for additional checkpoint/result roots")
    mode.add_argument("--merge", type=Path, nargs="+", metavar="CSV", help="Combine server all_results.csv exports")
    parser.add_argument("--platform", help="Required with --root, e.g. computerome or gefion")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-plot-data", action="store_true",
                        help="Export saved prediction/latent CSVs and matching metadata for the local plotting notebook; no model weights.")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if args.root and not args.platform:
        parser.error("--root requires --platform")
    if args.merge and args.include_plot_data:
        parser.error("--include-plot-data is used on the server with --root, not --merge")
    if args.root and output in [path.expanduser().resolve() for path in args.root]:
        parser.error("--output-dir must differ from each input root")
    try:
        if args.merge:
            tables = merge(args.merge)
            inputs = [str(path) for path in args.merge]
        else:
            results, intervals, inventory, inputs = collect(args.root, args.platform, output)
            tables = {"all_results.csv": results, "confidence_intervals.csv": intervals, "result_inventory.csv": inventory}
        output.mkdir(parents=True, exist_ok=True)
        if args.root and args.include_plot_data:
            tables["plot_artifacts.csv"] = export_plot_data(args.root, args.platform, output)
        for name, rows in tables.items():
            columns = ["platform", "source_file", "source_row"] if name not in {"result_inventory.csv", "plot_artifacts.csv"} else ["platform", "source_file", "status", "error"]
            frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
            frame.drop_duplicates().to_csv(output / name, index=False)
        (output / "collection_manifest.json").write_text(json.dumps({"inputs": inputs, "platform": args.platform, "mode": "merge" if args.merge else "collect", "counts": {key: len(value) for key, value in tables.items()}, "includes_plot_data": args.include_plot_data, "note": "Saved results only; no metrics or confidence intervals computed. Unknown split provenance remains unverified."}, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"[OUT] {output}: {len(tables['all_results.csv'])} result rows, {len(tables['confidence_intervals.csv'])} CI rows")
    if args.include_plot_data:
        artifacts = tables["plot_artifacts.csv"]
        failed = sum(row.get("status") != "copied" for row in artifacts)
        print(f"[PLOT DATA] {len(artifacts) - failed} files copied; {failed} failures (see plot_artifacts.csv)")


if __name__ == "__main__":
    main()
