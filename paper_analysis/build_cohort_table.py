#!/usr/bin/env python3
"""Create participant-level cohort flow and Table 1 summaries from frozen splits."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .run_modality_benchmark import load_metadata_files
except ImportError:  # Direct execution from the repository root.
    from run_modality_benchmark import load_metadata_files


def describe_group(frame: pd.DataFrame, split: str, gender: str) -> dict[str, object]:
    selected = frame if gender == "ALL" else frame[frame["gender"] == gender]
    complete = selected.dropna(subset=["age", "bmi", "target"])
    complete = complete[complete["target"].isin([0, 1])]
    age = complete["age"].to_numpy(dtype=float)
    bmi = complete["bmi"].to_numpy(dtype=float)
    positives = int(complete["target"].sum())

    def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
        if not len(values):
            return {
                f"{prefix}_mean": np.nan,
                f"{prefix}_sd": np.nan,
                f"{prefix}_median": np.nan,
                f"{prefix}_q1": np.nan,
                f"{prefix}_q3": np.nan,
            }
        return {
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            f"{prefix}_median": float(np.median(values)),
            f"{prefix}_q1": float(np.quantile(values, 0.25)),
            f"{prefix}_q3": float(np.quantile(values, 0.75)),
        }

    return {
        "split": split,
        "gender": gender,
        "n_unique_participants": int(selected["participant_id"].nunique()),
        "n_complete_primary_analysis": int(len(complete)),
        "n_plaque_positive": positives,
        "n_plaque_free": int(len(complete) - positives),
        "plaque_prevalence": float(positives / len(complete)) if len(complete) else np.nan,
        "n_missing_age": int(selected["age"].isna().sum()),
        "n_missing_bmi": int(selected["bmi"].isna().sum()),
        "n_missing_target": int(selected["target"].isna().sum()),
        **stats(age, "age"),
        **stats(bmi, "bmi"),
    }


def format_value(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def table_markdown(summary: pd.DataFrame) -> str:
    rows = [
        "| Split | Population | N | Plaque positive, n (%) | Age, mean (SD) | BMI, mean (SD) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"F": "Women", "M": "Men", "ALL": "All"}
    for record in summary.to_dict("records"):
        positive = int(record["n_plaque_positive"])
        prevalence = 100.0 * float(record["plaque_prevalence"])
        rows.append(
            "| {split} | {population} | {n} | {positive} ({prevalence:.1f}%) | "
            "{age_mean} ({age_sd}) | {bmi_mean} ({bmi_sd}) |".format(
                split=str(record["split"]).capitalize(),
                population=labels[str(record["gender"])],
                n=int(record["n_complete_primary_analysis"]),
                positive=positive,
                prevalence=prevalence,
                age_mean=format_value(record["age_mean"]),
                age_sd=format_value(record["age_sd"]),
                bmi_mean=format_value(record["bmi_mean"]),
                bmi_sd=format_value(record["bmi_sd"]),
            )
        )
    return "\n".join(rows) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tsv", required=True, type=Path, nargs="+")
    parser.add_argument("--test-tsv", required=True, type=Path, nargs="+")
    parser.add_argument("--target-col", default="anyplaq")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--age-col", default=None)
    parser.add_argument("--bmi-col", default=None)
    parser.add_argument("--gender-col", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train, train_columns = load_metadata_files(
        args.train_tsv,
        args.target_col,
        args.id_col,
        args.age_col,
        args.bmi_col,
        args.gender_col,
    )
    test, test_columns = load_metadata_files(
        args.test_tsv,
        args.target_col,
        args.id_col,
        args.age_col,
        args.bmi_col,
        args.gender_col,
    )
    overlap = sorted(set(train["participant_id"]) & set(test["participant_id"]))
    if overlap:
        raise ValueError(f"Train/test participant leakage detected (examples: {overlap[:10]}).")

    rows = []
    for split, frame in (("train", train), ("test", test)):
        for gender in ("F", "M", "ALL"):
            rows.append(describe_group(frame, split, gender))
    summary = pd.DataFrame(rows)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "cohort_characteristics.csv", index=False)
    (output_dir / "cohort_characteristics.md").write_text(
        table_markdown(summary), encoding="utf-8"
    )
    flow = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": args.target_col,
        "train_unique_participants": int(train["participant_id"].nunique()),
        "test_unique_participants": int(test["participant_id"].nunique()),
        "total_unique_participants": int(
            pd.concat([train["participant_id"], test["participant_id"]]).nunique()
        ),
        "train_test_overlap": 0,
        "train_columns": train_columns,
        "test_columns": test_columns,
        "train_files": [str(path.resolve()) for path in args.train_tsv],
        "test_files": [str(path.resolve()) for path in args.test_tsv],
        "counting_unit": "unique participant ID",
    }
    (output_dir / "cohort_manifest.json").write_text(
        json.dumps(flow, indent=2), encoding="utf-8"
    )
    print(table_markdown(summary), flush=True)
    print(f"[OUT] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
