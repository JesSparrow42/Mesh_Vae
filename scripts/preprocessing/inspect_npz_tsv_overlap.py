#!/usr/bin/env python3
"""Inspect how many scan IDs are present in train/test TSVs.

The scan IDs can come either from a preprocessed NPZ or directly from an OBJ
folder.  The folder mode is useful when you want to check only a new source
folder before mixing it with older preprocessed scans.
"""
from __future__ import annotations

# Allow direct execution as well as ``python -m`` from the repository checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

CODE_RE = re.compile(r"([A-Z]+DET\d{5,10})", re.IGNORECASE)


def prefix_of(code: str | None) -> str:
    if not code:
        return "EMPTY"
    m = re.match(r"([A-Z]+)", str(code))
    return m.group(1) if m else "OTHER"


def extract_stregkoder(x) -> set[str]:
    if x is None:
        return set()
    s = str(x).strip()
    s = s.strip().strip('"').strip("'")
    return {m.group(1).upper() for m in CODE_RE.finditer(s)}


def clean_stregkode(x) -> str | None:
    codes = sorted(extract_stregkoder(x))
    return codes[0] if codes else None


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff").strip().strip('"').strip("'") for c in out.columns]
    return out


def _pick_col(df: pd.DataFrame, preferred: list[str] | None = None, fallback_contains: str | None = None) -> str | None:
    if preferred:
        for cand in preferred:
            if cand and cand in df.columns:
                return cand
            if cand:
                for col in df.columns:
                    if col.lower() == cand.lower():
                        return col
    if fallback_contains:
        needle = fallback_contains.lower()
        for col in df.columns:
            if needle in col.lower():
                return col
    return None


def _detect_sep(path: str, forced: str | None = None) -> str:
    if forced:
        return forced
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.readline()
    return "\t" if head.count("\t") >= head.count(",") else ","


def load_npz_ids(path: Path) -> list[str]:
    data = np.load(path, allow_pickle=True)
    if "names" not in data.files:
        raise KeyError(f"{path} does not contain a 'names' array. Keys: {data.files}")
    ids = []
    for nm in data["names"]:
        ids.extend(sorted(extract_stregkoder(nm)))
    return ids


def load_obj_folder_ids(root: Path) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(f"OBJ root does not exist: {root}")
    ids = []
    for path in sorted(root.rglob("*.obj")):
        ids.extend(sorted(extract_stregkoder(str(path))))
    return ids


def load_tsv_ids(path: Path, sep: str | None = None, id_col: str | None = None) -> tuple[list[str], str]:
    detected_sep = _detect_sep(str(path), forced=sep)
    df = pd.read_csv(
        path,
        sep=detected_sep,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        engine="python",
    )
    df = clean_column_names(df)
    col = id_col or _pick_col(
        df,
        ["stregkode", "participant_id", "participantid", "id", "name", "scan_id"],
        fallback_contains="stregkode",
    )
    if col is None:
        col = _pick_col(df, ["participant_id", "participantid"], fallback_contains="participant")
    if col is None:
        raise KeyError(f"Could not find an ID column in {path}. Available columns: {list(df.columns)[:80]}")
    ids = []
    for value in df[col].tolist():
        ids.extend(sorted(extract_stregkoder(value)))
    return ids, col


def print_prefix_counts(title: str, ids: set[str]) -> None:
    counts: dict[str, int] = {}
    for code in ids:
        counts[prefix_of(code)] = counts.get(prefix_of(code), 0) + 1
    print(title)
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")


def report(label: str, scan_ids: list[str], tsv_ids: list[str], id_col: str) -> set[str]:
    scan_set = set(scan_ids)
    tsv_set = set(tsv_ids)
    matched = scan_set & tsv_set
    unmatched_scans = sorted(scan_set - tsv_set)
    tsv_without_npz = sorted(tsv_set - scan_set)
    denom = max(len(scan_set), 1)
    pct = 100.0 * len(matched) / denom

    print(f"\n========== {label} ==========")
    print(f"[TSV] id_col={id_col} rows_with_id={len(tsv_ids)} unique_ids={len(tsv_set)}")
    print(f"[MATCH] scan IDs with TSV metadata: {len(matched)}/{len(scan_set)} ({pct:.2f}%)")
    print(f"[DROP] scan IDs missing from TSV: {len(unmatched_scans)}")
    print(f"[EXTRA] TSV IDs not found in scan source: {len(tsv_without_npz)}")
    print_prefix_counts("[MATCH] matched prefixes:", matched)
    print_prefix_counts("[DROP] missing-from-TSV prefixes:", set(unmatched_scans))
    if unmatched_scans:
        print(f"[DROP] first missing scan IDs: {unmatched_scans[:20]}")
    if tsv_without_npz:
        print(f"[EXTRA] first TSV IDs not in scan source: {tsv_without_npz[:20]}")
    return matched


def write_csv(path: Path, scan_ids: list[str], train_ids: list[str], test_ids: list[str]) -> None:
    train_set = set(train_ids)
    test_set = set(test_ids)
    rows = []
    for code in sorted(set(scan_ids)):
        rows.append(
            {
                "id": code,
                "prefix": prefix_of(code),
                "in_train_tsv": int(code in train_set),
                "in_test_tsv": int(code in test_set),
                "in_any_tsv": int(code in train_set or code in test_set),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n[OUT] wrote per-scan overlap table -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", default=None, help="Path to simplified_padded_tensors.npz")
    src.add_argument("--obj_root", default=None, help="Recursively scan this folder for .obj files.")
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--test_tsv", required=True)
    ap.add_argument("--sep", default=None, help="Force TSV delimiter. Default: auto-detect.")
    ap.add_argument("--id_col", default=None, help="Force participant/scan ID column.")
    ap.add_argument("--out_csv", default=None, help="Optional per-scan ID overlap CSV.")
    args = ap.parse_args()

    npz_path = Path(args.npz) if args.npz else None
    obj_root = Path(args.obj_root) if args.obj_root else None
    train_path = Path(args.train_tsv)
    test_path = Path(args.test_tsv)

    if npz_path is not None:
        scan_ids = load_npz_ids(npz_path)
        source_label = "NPZ"
        source_path = npz_path
    else:
        scan_ids = load_obj_folder_ids(obj_root)
        source_label = "OBJ"
        source_path = obj_root

    train_ids, train_col = load_tsv_ids(train_path, sep=args.sep, id_col=args.id_col)
    test_ids, test_col = load_tsv_ids(test_path, sep=args.sep, id_col=args.id_col)

    scan_set = set(scan_ids)
    print(f"[{source_label}] {source_path}")
    print(f"[{source_label}] samples_with_id={len(scan_ids)} unique_ids={len(scan_set)}")
    print_prefix_counts(f"[{source_label}] prefixes:", scan_set)

    report("TRAIN TSV", scan_ids, train_ids, train_col)
    report("TEST TSV", scan_ids, test_ids, test_col)
    report("TRAIN + TEST TSV", scan_ids, train_ids + test_ids, f"{train_col}/{test_col}")

    if args.out_csv:
        write_csv(Path(args.out_csv), scan_ids, train_ids, test_ids)


if __name__ == "__main__":
    main()
