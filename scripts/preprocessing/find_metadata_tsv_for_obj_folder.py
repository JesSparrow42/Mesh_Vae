#!/usr/bin/env python3
"""Rank metadata files by overlap with OBJ scan IDs.

Use this when a scan folder contains many subfolders and you need to find the
TSV/CSV file that actually contains metadata rows for those scan names.
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
ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1"]


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


def detect_sep(path: Path, forced: str | None = None, encoding: str = "utf-8") -> str:
    if forced:
        return forced
    with path.open("r", encoding=encoding, errors="replace") as f:
        head = f.readline()
    counts = {"\t": head.count("\t"), ",": head.count(","), ";": head.count(";")}
    return max(counts, key=counts.get)


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


def load_obj_ids(root: Path) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(f"OBJ root does not exist: {root}")
    ids = []
    for path in sorted(root.rglob("*.obj")):
        ids.extend(sorted(extract_stregkoder(str(path))))
    return ids


def candidate_files(root: Path, extensions: set[str], max_size_mb: float | None) -> list[Path]:
    files = []
    max_bytes = None if max_size_mb is None else int(max_size_mb * 1024 * 1024)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        if max_bytes is not None and path.stat().st_size > max_bytes:
            print(f"[SKIP] too large ({path.stat().st_size / 1024 / 1024:.1f} MB): {path}", flush=True)
            continue
        files.append(path)
    return files


def base_result_for_file(path: Path, obj_set: set[str]) -> dict:
    path_ids = extract_stregkoder(str(path))
    matched = obj_set & path_ids
    return {
        "path": str(path),
        "sep": "",
        "encoding": "",
        "rows": 0,
        "id_col": "[file path/name]",
        "metadata_unique_ids": int(len(path_ids)),
        "matches": int(len(matched)),
        "matched_prefixes": prefix_counts(matched),
        "first_matches": sorted(matched)[:10],
        "first_missing_obj_ids": sorted(obj_set - path_ids)[:10],
        "error": "",
    }


def read_csv_any_encoding(path: Path, sep: str | None) -> tuple[pd.DataFrame, str, str]:
    errors = []
    for enc in ENCODINGS:
        try:
            use_sep = detect_sep(path, sep, encoding=enc)
            df = pd.read_csv(
                path,
                sep=use_sep,
                dtype=str,
                quoting=csv.QUOTE_NONE,
                on_bad_lines="skip",
                engine="python",
                encoding=enc,
                encoding_errors="replace",
            )
            return clean_column_names(df), use_sep, enc
        except Exception as exc:
            errors.append(f"{enc}: {exc}")
    raise RuntimeError(" | ".join(errors[:4]))


def best_overlap_for_file(path: Path, obj_set: set[str], sep: str | None) -> dict | None:
    best = base_result_for_file(path, obj_set)
    path_ids = extract_stregkoder(str(path))
    try:
        df, use_sep, encoding = read_csv_any_encoding(path, sep)
    except Exception as exc:
        best["error"] = str(exc)
        return best

    for col in df.columns:
        codes = set()
        for value in df[col].tolist():
            codes.update(extract_stregkoder(value))
        if not codes and not path_ids:
            continue
        code_set = path_ids | codes
        matched = obj_set & code_set
        score = len(matched)
        if score > best["matches"] or (score == best["matches"] and len(code_set) > best["metadata_unique_ids"]):
            best = {
                "path": str(path),
                "sep": repr(use_sep),
                "encoding": encoding,
                "rows": int(len(df)),
                "id_col": "[file path/name]" if not codes else str(col),
                "metadata_unique_ids": int(len(code_set)),
                "matches": int(score),
                "matched_prefixes": prefix_counts(matched),
                "first_matches": sorted(matched)[:10],
                "first_missing_obj_ids": sorted(obj_set - code_set)[:10],
                "error": "",
            }

    return best


def print_path_id_summary(files: list[Path], obj_set: set[str], search_root: Path) -> None:
    all_path_ids = set()
    dir_ids: dict[Path, set[str]] = {}
    for path in files:
        ids = extract_stregkoder(str(path))
        if not ids:
            continue
        all_path_ids.update(ids)
        for parent in [path.parent, *path.parents]:
            try:
                parent.relative_to(search_root)
            except ValueError:
                continue
            dir_ids.setdefault(parent, set()).update(ids)
            if parent == search_root:
                break

    matched = obj_set & all_path_ids
    print("\n========== METADATA IDS FROM FILE/PATH NAMES ==========")
    print(f"[PATH IDS] unique_ids={len(all_path_ids)} matches={len(matched)}/{len(obj_set)} ({100.0 * len(matched) / max(len(obj_set), 1):.2f}%)")
    print(f"[PATH IDS] matched_prefixes={prefix_counts(matched)}")
    print("[PATH IDS] top folders by overlap:")
    ranked_dirs = sorted(
        ((len(obj_set & ids), len(ids), path) for path, ids in dir_ids.items()),
        key=lambda item: (item[0], item[1], str(item[2])),
        reverse=True,
    )
    for matches, n_ids, path in ranked_dirs[:10]:
        print(f"  matches={matches}/{len(obj_set)} unique_path_ids={n_ids} folder={path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj_root", required=True, help="Folder containing the new OBJ scans. Searched recursively.")
    ap.add_argument("--search_root", default=None, help="Folder to search for metadata files. Default: --obj_root.")
    ap.add_argument("--extensions", nargs="+", default=[".tsv", ".csv", ".txt"])
    ap.add_argument("--sep", default=None, help="Force delimiter for all files. Default: auto-detect.")
    ap.add_argument("--max_size_mb", type=float, default=1000.0, help="Skip files larger than this. Use 0 for no limit.")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    obj_root = Path(args.obj_root)
    search_root = Path(args.search_root) if args.search_root else obj_root
    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}
    max_size_mb = None if args.max_size_mb == 0 else args.max_size_mb

    obj_ids = load_obj_ids(obj_root)
    obj_set = set(obj_ids)
    print(f"[OBJ] root={obj_root}")
    print(f"[OBJ] files_with_id={len(obj_ids)} unique_ids={len(obj_set)}")
    print(f"[OBJ] prefixes={prefix_counts(obj_set)}")

    files = candidate_files(search_root, extensions, max_size_mb)
    print(f"[SEARCH] root={search_root}")
    print(f"[SEARCH] candidate metadata files={len(files)} extensions={sorted(extensions)}")
    print_path_id_summary(files, obj_set, search_root)

    results = []
    for i, path in enumerate(files, start=1):
        print(f"[SCAN] {i}/{len(files)} {path}", flush=True)
        result = best_overlap_for_file(path, obj_set, args.sep)
        if result is not None:
            denom = max(len(obj_set), 1)
            result["pct"] = 100.0 * float(result.get("matches", 0)) / denom
            results.append(result)

    results.sort(key=lambda r: (int(r.get("matches", 0)), int(r.get("metadata_unique_ids", 0))), reverse=True)

    print("\n========== BEST METADATA CANDIDATES ==========")
    for rank, r in enumerate(results[: max(1, args.top)], start=1):
        print(
            f"{rank:02d}. matches={r.get('matches', 0)}/{len(obj_set)} "
            f"({r.get('pct', 0.0):.2f}%) rows={r.get('rows', '')} "
            f"unique_ids={r.get('metadata_unique_ids', '')} id_col={r.get('id_col', '')} "
            f"path={r.get('path', '')}"
        )
        if r.get("error"):
            print(f"    error={r['error']}")
        print(f"    matched_prefixes={r.get('matched_prefixes', '-')}")
        print(f"    first_matches={r.get('first_matches', [])}")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"\n[OUT] wrote ranked candidates -> {out}")


if __name__ == "__main__":
    main()
