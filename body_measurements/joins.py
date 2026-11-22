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
