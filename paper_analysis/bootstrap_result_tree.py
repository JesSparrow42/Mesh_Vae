#!/usr/bin/env python3
"""Compute participant-level confidence intervals for every prediction file in a result tree.

This is the batch companion to ``bootstrap_metrics.py``. It recursively discovers
held-out prediction CSV files, infers binary versus regression tasks, and writes
one paper-ready table containing estimates and confidence intervals. Files that
cannot be interpreted are retained in an inventory with the error message rather
than silently disappearing.

Example
-------
Run this on Gefion after the held-out diagnostics have finished::

    python paper_analysis/bootstrap_result_tree.py \
      --root /dcai/projects/cu_0080/results/checkpoints \
      --output-dir /dcai/projects/cu_0080/results/paper_ci_all
"""
from __future__ import annotations

import argparse
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from .bootstrap_metrics import load_predictions, summarize_model
except ImportError:  # Direct execution from the repository root.
    from bootstrap_metrics import load_predictions, summarize_model


DEFAULT_PATTERNS = (
    "test_predictions.csv",
    "*_test_predictions.csv",
)
INVENTORY_PATTERNS = ("predictions.csv", "*predictions.csv", "*predictions_*.csv")


def discover(root: Path, patterns: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path.resolve() for path in root.rglob(pattern) if path.is_file())
    return sorted(files, key=lambda path: str(path).lower())


def referenced_diagnostics(root: Path) -> set[Path]:
    """Accept only the diagnostic directory recorded by its own run summary."""
    referenced: set[Path] = set()
    for task in ("binary", "regression"):
        directory = f"{task}_diagnostics"
        for summary in root.rglob(f"{task}_summary.csv"):
            try:
                frame = pd.read_csv(summary, dtype=str)
            except (OSError, ValueError):
                continue  # Its prediction files remain explicitly ambiguous in the inventory.
            for value in frame.get("ResultsDir", pd.Series(dtype=str)).dropna():
                recorded = Path(str(value).replace("\\", "/"))
                if recorded.parent.name != directory:
                    continue
                # Resolve within the summary's run, also allowing relocated result snapshots.
                prediction = summary.parent / directory / recorded.name / "predictions.csv"
                if prediction.is_file():
                    referenced.add(prediction.resolve())
    return referenced


def heldout_evidence(root: Path, path: Path, referenced: set[Path], explicit: bool) -> tuple[str, str]:
    relative = path.relative_to(root)
    excluded = {"train", "training", "validation", "val", "calibration", "calib"}
    tokens = set(path.stem.lower().replace("-", "_").split("_"))
    directories = {part.lower() for part in relative.parts[:-1]}
    if (tokens | directories) & excluded:
        return "excluded_non_test", "Training, validation, or calibration prediction file."
    if any(path.match(pattern) for pattern in DEFAULT_PATTERNS):
        return "eligible", "Held-out test filename contract."
    if path in referenced:
        # A diagnostic can evaluate validation data too; ResultsDir alone is
        # not held-out provenance. Require explicit test rows in its split column.
        try:
            split_frame = pd.read_csv(path, dtype=str, usecols=lambda name: str(name).lower() == "split")
        except (OSError, ValueError):
            split_frame = pd.DataFrame()
        if len(split_frame.columns):
            labels = set(split_frame.iloc[:, 0].dropna().str.strip().str.lower())
            if labels & {"test", "heldout", "held_out", "held-out"}:
                return "eligible", "Run summary references diagnostic with explicit held-out split rows."
            if labels and labels <= excluded | {"development"}:
                return "excluded_non_test", "Referenced diagnostic contains only non-test split rows."
        if not explicit:
            return "ambiguous", "ResultsDir does not establish test membership; no explicit held-out split rows."
    if explicit:
        return "eligible", "Explicit --pattern selection; any split column is filtered to held-out rows."
    return "ambiguous", "No held-out filename or summary reference; use canonical study/test_predictions.csv."


def result_label(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    parent = relative.parent.as_posix()
    if parent in {"", "."}:
        return path.stem
    return parent.replace("/", "__")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help="Explicit held-out filename pattern. Defaults score test filenames and summary-referenced diagnostics with explicit test split rows; other predictions are inventoried only.",
    )
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--duplicate-policy", choices=("error", "first", "mean"), default="error")
    parser.add_argument("--regression-strata", choices=("auto", "none", "zero"), default="auto")
    parser.add_argument(
        "--require-participant-id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject files without a participant ID column (recommended for paper CIs).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstraps < 100:
        raise ValueError("Use at least 100 bootstrap replicates; 5000 is recommended for final results.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between zero and one.")

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = args.patterns or list(INVENTORY_PATTERNS)
    paths = discover(root, patterns)
    referenced = referenced_diagnostics(root)
    if not paths:
        raise FileNotFoundError(f"No prediction files found below {root} using {patterns}")

    interval_frames: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    used_labels: dict[str, int] = {}

    for path in paths:
        relative = path.relative_to(root).as_posix()
        base_label = result_label(root, path)
        used_labels[base_label] = used_labels.get(base_label, 0) + 1
        label = base_label if used_labels[base_label] == 1 else f"{base_label}__{used_labels[base_label]}"
        row: dict[str, object] = {
            "label": label,
            "relative_path": relative,
            "absolute_path": str(path),
            "status": "failed",
            "task": None,
            "n_participants": None,
            "error": None,
        }
        eligibility, evidence = heldout_evidence(root, path, referenced, explicit=bool(args.patterns))
        row["heldout_evidence"] = evidence
        if eligibility != "eligible":
            row.update(status=eligibility, error=evidence)
            inventory.append(row)
            print(f"[SKIP] {relative}: {evidence}", flush=True)
            continue
        try:
            frame, task = load_predictions(
                path,
                task="auto",
                threshold=args.threshold,
                duplicate_policy=args.duplicate_policy,
                require_participant_id=args.require_participant_id,
            )
            stable_offset = zlib.crc32(relative.encode("utf-8"))
            summary = summarize_model(
                label=label,
                frame=frame,
                task=task,
                n_bootstrap=args.bootstraps,
                seed=args.seed + stable_offset,
                confidence=args.confidence,
                regression_strata=args.regression_strata,
            )
            summary.insert(0, "prediction_file", relative)
            interval_frames.append(summary)
            row.update(status="completed", task=task, n_participants=len(frame),
                       excluded_non_test_rows=frame.attrs.get("excluded_non_test_rows", 0))
            print(f"[OK] {relative}: task={task} n={len(frame)}", flush=True)
        except Exception as exc:  # Preserve every failure in the audit table.
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[SKIP] {relative}: {row['error']}", flush=True)
        inventory.append(row)

    intervals = pd.concat(interval_frames, ignore_index=True) if interval_frames else pd.DataFrame()
    inventory_frame = pd.DataFrame(inventory)
    intervals_path = output_dir / "all_result_intervals.csv"
    inventory_path = output_dir / "result_inventory.csv"
    failures_path = output_dir / "result_failures.csv"
    intervals.to_csv(intervals_path, index=False)
    inventory_frame.to_csv(inventory_path, index=False)
    inventory_frame[inventory_frame["status"] != "completed"].to_csv(failures_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "patterns": args.patterns or list(DEFAULT_PATTERNS),
        "inventory_patterns": patterns,
        "prediction_files_discovered": len(paths),
        "prediction_files_completed": int((inventory_frame["status"] == "completed").sum()),
        "prediction_files_failed": int((inventory_frame["status"] == "failed").sum()),
        "prediction_files_ambiguous": int((inventory_frame["status"] == "ambiguous").sum()),
        "prediction_files_excluded_non_test": int((inventory_frame["status"] == "excluded_non_test").sum()),
        "bootstrap_replicates": args.bootstraps,
        "confidence_level": args.confidence,
        "seed": args.seed,
        "require_participant_id": args.require_participant_id,
        "duplicate_policy": args.duplicate_policy,
        "resampling_unit": "participant",
        "interval_method": "stratified percentile bootstrap",
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"[OUT] intervals -> {intervals_path}", flush=True)
    print(f"[OUT] inventory -> {inventory_path}", flush=True)
    if (inventory_frame["status"] != "completed").any():
        print(f"[CHECK] failures -> {failures_path}", flush=True)
    if intervals.empty:
        raise RuntimeError("No prediction file could be analyzed; inspect result_failures.csv.")


if __name__ == "__main__":
    main()
