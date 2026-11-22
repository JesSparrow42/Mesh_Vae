#!/usr/bin/env python3
"""Collect MeshVAE diagnostic summary rows across many checkpoint runs."""
from __future__ import annotations

# Allow direct execution as well as ``python -m`` from the repository checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
import csv
import json
import math
from pathlib import Path


METRIC_COLUMNS = [
    "test_accuracy",
    "test_balanced_accuracy",
    "test_auc",
    "test_sensitivity",
    "test_specificity",
    "test_precision",
    "test_f1",
    "calib_balanced_accuracy",
    "calib_auc",
]


OUTPUT_COLUMNS = [
    "rank",
    "name",
    "variant",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_auc",
    "test_sensitivity",
    "test_specificity",
    "test_precision",
    "test_f1",
    "calib_balanced_accuracy",
    "calib_auc",
    "policy_mode",
    "chosen_score",
    "only_gender",
    "results_dir",
]


def parse_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def format_float(value) -> str:
    x = parse_float(value)
    if not math.isfinite(x):
        return ""
    return f"{x:.4f}"


def read_summary_row(path: Path) -> dict:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return dict(rows[0])


def read_summary_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    test = obj.get("test", {}) or {}
    calib = obj.get("calibration", {}) or {}
    return {
        "name": path.parents[2].name,
        "which_epoch": obj.get("which_epoch", ""),
        "only_gender": (obj.get("settings", {}) or {}).get("only_gender", ""),
        "policy_mode": obj.get("policy_mode", ""),
        "chosen_score": obj.get("chosen_score", ""),
        "prevalence_cutoff": obj.get("prevalence_cutoff", ""),
        "test_accuracy": test.get("accuracy", ""),
        "test_balanced_accuracy": test.get("balanced_accuracy", ""),
        "test_auc": test.get("roc_auc", ""),
        "test_sensitivity": test.get("sensitivity", ""),
        "test_specificity": test.get("specificity", ""),
        "test_precision": test.get("precision", ""),
        "test_f1": test.get("f1", ""),
        "calib_balanced_accuracy": calib.get("balanced_accuracy", ""),
        "calib_auc": calib.get("roc_auc", ""),
        "results_dir": str(path.parent),
    }


def latest_summary_for_run(run_dir: Path) -> tuple[Path, dict] | None:
    diag_root = run_dir / "diagnostics"
    if not diag_root.exists():
        return None

    candidates = sorted(
        list(diag_root.glob("*/summary_row.csv")) + list(diag_root.glob("*/summary.json")),
        key=lambda p: (p.stat().st_mtime, str(p)),
        reverse=True,
    )
    if not candidates:
        return None

    # Prefer summary_row.csv when both files exist in the same latest folder.
    latest_dir = candidates[0].parent
    row_path = latest_dir / "summary_row.csv"
    json_path = latest_dir / "summary.json"
    if row_path.exists():
        return row_path, read_summary_row(row_path)
    if json_path.exists():
        return json_path, read_summary_json(json_path)
    return None


def variant_from_name(name: str, prefix: str) -> str:
    if prefix and name.startswith(prefix):
        return name[len(prefix):].lstrip("_")
    return name


def collect(checkpoints_dir: Path, prefix: str) -> list[dict]:
    rows = []
    for run_dir in sorted(checkpoints_dir.glob(f"{prefix}*")):
        if not run_dir.is_dir():
            continue
        latest = latest_summary_for_run(run_dir)
        if latest is None:
            continue
        summary_path, row = latest
        row = dict(row)
        row["name"] = row.get("name") or run_dir.name
        row["variant"] = variant_from_name(row["name"], prefix)
        row["summary_file"] = str(summary_path)
        rows.append(row)

    rows.sort(
        key=lambda r: (
            parse_float(r.get("test_balanced_accuracy")),
            parse_float(r.get("test_auc")),
        ),
        reverse=True,
    )
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("[RESULTS] No diagnostic summaries found.")
        return

    cols = [
        "rank",
        "variant",
        "test_balanced_accuracy",
        "test_accuracy",
        "test_auc",
        "test_sensitivity",
        "test_specificity",
        "test_f1",
    ]
    widths = {c: len(c) for c in cols}
    for row in rows:
        for c in cols:
            value = format_float(row.get(c)) if c in METRIC_COLUMNS else str(row.get(c, ""))
            widths[c] = max(widths[c], len(value))

    print("\n========== DIAGNOSTIC RESULTS ==========")
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for row in rows:
        vals = []
        for c in cols:
            vals.append((format_float(row.get(c)) if c in METRIC_COLUMNS else str(row.get(c, ""))).ljust(widths[c]))
        print("  ".join(vals))

    best = rows[0]
    print(
        f"\n[BEST] {best.get('variant')} "
        f"bal_acc={format_float(best.get('test_balanced_accuracy'))} "
        f"auc={format_float(best.get('test_auc'))} "
        f"sens={format_float(best.get('test_sensitivity'))} "
        f"spec={format_float(best.get('test_specificity'))}"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS + ["summary_file"], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[OUT] wrote {path}")


def write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MeshVAE Diagnostic Results",
        "",
        "| Rank | Variant | Test balanced accuracy | Test AUC | Sensitivity | Specificity | F1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | {variant} | {bal} | {auc} | {sens} | {spec} | {f1} |".format(
                rank=row.get("rank", ""),
                variant=row.get("variant", ""),
                bal=format_float(row.get("test_balanced_accuracy")),
                auc=format_float(row.get("test_auc")),
                sens=format_float(row.get("test_sensitivity")),
                spec=format_float(row.get("test_specificity")),
                f1=format_float(row.get("test_f1")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OUT] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_dir", default="/data/projects/3D_scans/ML/MQVAE/checkpoints")
    ap.add_argument("--name_prefix", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    rows = collect(Path(args.checkpoints_dir), args.name_prefix)
    print(f"[RESULTS] found {len(rows)} completed diagnostic runs for prefix {args.name_prefix!r}")
    print_table(rows)

    if args.out_csv:
        write_csv(Path(args.out_csv), rows)
    if args.out_md:
        write_markdown(Path(args.out_md), rows)


if __name__ == "__main__":
    main()
