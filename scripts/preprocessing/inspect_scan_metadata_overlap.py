#!/usr/bin/env python3
"""Compare scan IDs against one full metadata TSV/CSV.

This is stricter than the older split-file checker:
  - scan IDs are only accepted when they look like PREFIXDET123456
  - metadata IDs are tested in every column and ranked by overlap
  - non-UTF-8 files and tab/comma/semicolon delimiters are handled
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
ENCODINGS = ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin1"]


def extract_codes(value) -> set[str]:
    if value is None:
        return set()
    return {m.group(1).upper() for m in CODE_RE.finditer(str(value))}


def prefix_of(code: str | None) -> str:
    if not code:
        return "EMPTY"
    m = re.match(r"([A-Z]+)", str(code))
    return m.group(1) if m else "OTHER"


def prefix_counts(ids: set[str]) -> str:
    counts: dict[str, int] = {}
    for code in ids:
        counts[prefix_of(code)] = counts.get(prefix_of(code), 0) + 1
    return ", ".join(f"{k}:{counts[k]}" for k in sorted(counts)) if counts else "-"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff").strip().strip('"').strip("'") for c in out.columns]
    return out


def detect_sep(path: Path, encoding: str, forced: str | None) -> str:
    if forced:
        return forced
    with path.open("r", encoding=encoding, errors="replace") as f:
        head = ""
        for line in f:
            if line.strip():
                head = line
                break
    counts = {"\t": head.count("\t"), ",": head.count(","), ";": head.count(";")}
    return max(counts, key=counts.get)


def read_table(path: Path, forced_sep: str | None) -> tuple[pd.DataFrame, str, str]:
    errors = []
    for enc in ENCODINGS:
        try:
            sep = detect_sep(path, enc, forced_sep)
            df = pd.read_csv(
                path,
                sep=sep,
                dtype=str,
                quoting=csv.QUOTE_NONE,
                on_bad_lines="skip",
                engine="python",
                encoding=enc,
                encoding_errors="replace",
            )
            return clean_columns(df), sep, enc
        except Exception as exc:
            errors.append(f"{enc}: {exc}")
    raise RuntimeError("Could not read metadata file with common encodings: " + " | ".join(errors))


def load_obj_ids(root: Path) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(f"OBJ root does not exist: {root}")
    ids = []
    for path in sorted(root.rglob("*.obj")):
        ids.extend(sorted(extract_codes(str(path))))
    return ids


def load_npz_ids(path: Path) -> list[str]:
    data = np.load(path, allow_pickle=True)
    if "names" not in data.files:
        raise KeyError(f"{path} does not contain names. Keys: {data.files}")
    ids = []
    for value in data["names"]:
        ids.extend(sorted(extract_codes(value)))
    return ids


def normalize_gender(value) -> str:
    s = str(value).strip().strip('"').strip("'").strip().upper()
    return {"FEMALE": "F", "WOMAN": "F", "KVINDE": "F", "MALE": "M", "MAN": "M", "MAND": "M"}.get(s, s)


def pick_gender_col(df: pd.DataFrame) -> str | None:
    preferred = ["sexMF", "sex", "gender", "gender_clean", "koen", "køn"]
    for cand in preferred:
        for col in df.columns:
            if col.lower() == cand.lower():
                return col
    for needle in ["gender", "sex", "køn", "koen"]:
        for col in df.columns:
            if needle in col.lower():
                return col
    return None


def codes_by_row(series: pd.Series) -> list[set[str]]:
    return [extract_codes(v) for v in series.tolist()]


def score_columns(df: pd.DataFrame, scan_set: set[str]) -> list[dict]:
    rows = []
    for col in df.columns:
        per_row = codes_by_row(df[col])
        ids = set().union(*per_row) if per_row else set()
        if not ids:
            continue
        matched = ids & scan_set
        rows.append(
            {
                "column": col,
                "rows_with_id": sum(1 for codes in per_row if codes),
                "metadata_unique_ids": len(ids),
                "matches": len(matched),
                "match_pct_of_scans": 100.0 * len(matched) / max(len(scan_set), 1),
                "matched_prefixes": prefix_counts(matched),
                "first_matches": sorted(matched)[:10],
                "first_missing_scan_ids": sorted(scan_set - ids)[:10],
            }
        )
    rows.sort(key=lambda r: (r["matches"], r["metadata_unique_ids"], r["rows_with_id"]), reverse=True)
    return rows


def write_outputs(out_prefix: Path, scan_set: set[str], metadata_ids: set[str], best_col: str, scores: list[dict]) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(f"{out_prefix}_column_scores.csv", index=False)
    pd.DataFrame({"id": sorted(scan_set & metadata_ids)}).to_csv(f"{out_prefix}_matched_ids.csv", index=False)
    pd.DataFrame({"id": sorted(scan_set - metadata_ids)}).to_csv(f"{out_prefix}_scan_ids_missing_metadata.csv", index=False)
    pd.DataFrame({"id": sorted(metadata_ids - scan_set)}).to_csv(f"{out_prefix}_metadata_ids_without_scan.csv", index=False)
    print(f"\n[OUT] wrote {out_prefix}_column_scores.csv")
    print(f"[OUT] wrote {out_prefix}_matched_ids.csv")
    print(f"[OUT] wrote {out_prefix}_scan_ids_missing_metadata.csv")
    print(f"[OUT] wrote {out_prefix}_metadata_ids_without_scan.csv")
    print(f"[OUT] best metadata ID column: {best_col}")


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--obj_root", default=None)
    src.add_argument("--npz", default=None)
    ap.add_argument("--metadata_tsv", required=True)
    ap.add_argument("--sep", default=None)
    ap.add_argument("--only_gender", default=None, help="Optional metadata filter, e.g. F or M.")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out_prefix", default=None)
    args = ap.parse_args()

    if args.obj_root:
        source_path = Path(args.obj_root)
        scan_ids = load_obj_ids(source_path)
        source_label = "OBJ"
    else:
        source_path = Path(args.npz)
        scan_ids = load_npz_ids(source_path)
        source_label = "NPZ"

    scan_set = set(scan_ids)
    print(f"[{source_label}] source={source_path}")
    print(f"[{source_label}] IDs found={len(scan_ids)} unique_ids={len(scan_set)} prefixes={prefix_counts(scan_set)}")

    metadata_path = Path(args.metadata_tsv)
    df, sep, encoding = read_table(metadata_path, args.sep)
    print(f"[META] file={metadata_path}")
    print(f"[META] rows={len(df)} cols={len(df.columns)} sep={repr(sep)} encoding={encoding}")
    print(f"[META] first columns={list(df.columns[:30])}")

    if args.only_gender:
        gender_col = pick_gender_col(df)
        if gender_col is None:
            raise KeyError(f"--only_gender {args.only_gender} requested, but no gender column found.")
        before = len(df)
        want = normalize_gender(args.only_gender)
        keep = df[gender_col].apply(normalize_gender) == want
        df = df.loc[keep].copy()
        print(f"[META] gender filter {gender_col} == {want}: kept {len(df)}/{before} rows")

    scores = score_columns(df, scan_set)
    if not scores:
        raise ValueError("No PREFIXDET12345-style IDs found in any metadata column.")

    print("\n========== BEST ID COLUMNS ==========")
    for i, row in enumerate(scores[: max(1, args.top)], start=1):
        print(
            f"{i:02d}. matches={row['matches']}/{len(scan_set)} "
            f"({row['match_pct_of_scans']:.2f}%) rows_with_id={row['rows_with_id']} "
            f"unique_ids={row['metadata_unique_ids']} column={row['column']}"
        )
        print(f"    matched_prefixes={row['matched_prefixes']}")
        print(f"    first_matches={row['first_matches']}")

    best = scores[0]
    best_col = best["column"]
    metadata_ids = set()
    for value in df[best_col].tolist():
        metadata_ids.update(extract_codes(value))

    matched = scan_set & metadata_ids
    print("\n========== SUMMARY ==========")
    print(f"[BEST] id_col={best_col}")
    print(f"[MATCH] scan IDs with metadata: {len(matched)}/{len(scan_set)} ({100.0 * len(matched) / max(len(scan_set), 1):.2f}%)")
    print(f"[DROP] scan IDs missing metadata: {len(scan_set - metadata_ids)}")
    print(f"[EXTRA] metadata IDs without scan: {len(metadata_ids - scan_set)}")
    print(f"[MATCH] prefixes={prefix_counts(matched)}")
    print(f"[DROP] first missing scan IDs={sorted(scan_set - metadata_ids)[:30]}")
    print(f"[EXTRA] first metadata IDs without scan={sorted(metadata_ids - scan_set)[:30]}")

    if args.out_prefix:
        write_outputs(Path(args.out_prefix), scan_set, metadata_ids, best_col, scores)


if __name__ == "__main__":
    main()
