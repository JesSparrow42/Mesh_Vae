#!/usr/bin/env python3
"""Make reproducible train/test TSVs for MeshVAE cohorts.

Default input is the new 2026 TSV:
  /data/projects/3D_scans/3d_bodyscan_20260604.tsv

The script also derives the legacy binary atherosclerosis columns
ho_carotis/ve_carotis/ho_fem/ve_fem from corrected plaque volumes when those
binary columns are not already present.

Run from the MQVAE repository root with:
  python -m meshvae.make_train_test_split
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from meshvae.conditioning import (
    apply_target_missing_policy,
    _clean_agecut_label,
    _detect_sep,
    _pick_col,
    _to_numeric_series,
    clean_column_names,
    clean_stregkode,
    ensure_athero_columns,
)

SCAN_CODE_RE = re.compile(r"([A-Z]+DET\d{5,10})", re.IGNORECASE)


def age_bin_series(age: pd.Series, age_bins: list[float] | None) -> pd.Series:
    a = _to_numeric_series(age)
    if not age_bins or len(age_bins) < 2:
        # Coarse fallback bins keep the stratification stable without requiring args.
        age_bins = [18, 30, 40, 50, 60, 71, 200]
    bins = list(map(float, age_bins))
    labels = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        labels.append(f"{int(lo)}+" if hi >= 199.5 else f"{int(lo)}-{int(hi)}")
    return pd.cut(a, bins=bins, labels=labels, right=False, include_lowest=True).astype(str).fillna("ageNA")


def clean_string_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip wrapping quotes/spaces from exported TSV string values.

    The 2026 export is parsed with QUOTE_NONE so values like "F" remain as
    literal quotes.  If we do not normalize them, the convenience train_F/test_F
    files become empty because gender is seen as '"F"' instead of 'F'.
    """
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip().str.strip('"').str.strip("'").str.strip()
            df.loc[df[c].str.lower().isin(["nan", "none", "null", "<na>"]), c] = np.nan
    return df


def strict_scan_code(value) -> str | None:
    if value is None:
        return None
    m = SCAN_CODE_RE.search(str(value))
    if m:
        return m.group(1).upper()
    # Fallback keeps compatibility with older clean IDs, but the DET regex above
    # prevents accidental matches like TESTSIMPLIFIED from dominating new scans.
    code = clean_stregkode(value)
    if code and SCAN_CODE_RE.fullmatch(code):
        return code.upper()
    return None


def load_match_ids(match_npz: str | None, match_obj_root: str | None) -> set[str] | None:
    if not match_npz and not match_obj_root:
        return None

    ids: list[str] = []
    if match_npz:
        npz_path = Path(match_npz)
        data = np.load(npz_path, allow_pickle=True)
        if "names" not in data.files:
            raise KeyError(f"{npz_path} does not contain a names array. Keys: {data.files}")
        for nm in data["names"]:
            code = strict_scan_code(nm)
            if code:
                ids.append(code)
        print(f"[MATCH] Loaded {len(ids)} scan IDs from NPZ: {npz_path}")

    if match_obj_root:
        root = Path(match_obj_root)
        before = len(ids)
        for path in sorted(root.rglob("*.obj")):
            code = strict_scan_code(str(path))
            if code:
                ids.append(code)
        print(f"[MATCH] Loaded {len(ids) - before} scan IDs from OBJ root: {root}")

    id_set = set(ids)
    print(f"[MATCH] unique scan IDs available for split: {len(id_set)}")
    return id_set


def filter_metadata_to_scan_ids(df: pd.DataFrame, scan_ids: set[str] | None, id_col: str | None) -> pd.DataFrame:
    if scan_ids is None:
        return df

    col = id_col or _pick_col(
        df,
        ["participant_id", "participantid", "stregkode", "id", "scan_id", "name"],
        fallback_contains="participant",
    )
    if col is None:
        col = _pick_col(df, ["stregkode"], fallback_contains="stregkode")
    if col is None:
        raise KeyError(f"Could not find participant/scan ID column. Available columns: {list(df.columns)[:100]}")

    codes = df[col].apply(strict_scan_code)
    keep = codes.isin(scan_ids)
    before = len(df)
    matched_unique = set(codes[keep].dropna().tolist())
    df = df.loc[keep].copy()
    print(
        f"[MATCH] Filter metadata to scans using column '{col}': "
        f"kept {len(df)}/{before} rows, matched_unique_ids={len(matched_unique)}/{len(scan_ids)}"
    )
    missing = sorted(scan_ids - matched_unique)
    if missing:
        print(f"[MATCH] first scan IDs missing metadata: {missing[:30]}")
    if df.empty:
        raise ValueError("No metadata rows matched the requested scan IDs.")
    return df


def age_group_series(df: pd.DataFrame, age_col: str | None, agecut_col: str | None, age_bins: list[float] | None) -> pd.Series:
    if agecut_col is not None:
        out = df[agecut_col].apply(_clean_agecut_label).fillna("ageNA").astype(str)
        print(f"[SPLIT] Stratifying by existing agecut column: {agecut_col}")
        return out
    if age_col is None:
        raise KeyError("Could not find age or agecut column for stratification.")
    print(f"[SPLIT] Stratifying by numeric age column: {age_col}")
    return age_bin_series(df[age_col], age_bins)


def stratified_split_indices(strata: pd.Series, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    test_idx = []
    train_idx = []

    for _key, idx_vals in strata.groupby(strata).groups.items():
        idx = np.asarray(list(idx_vals), dtype=np.int64)
        rng.shuffle(idx)
        n = len(idx)
        if n <= 1:
            n_test = 0
        else:
            n_test = int(round(n * float(test_size)))
            n_test = max(1, min(n - 1, n_test))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    # Safety fallback: avoid an empty test split on tiny cohorts.
    if len(test_idx) == 0 and len(train_idx) > 1:
        rng.shuffle(train_idx)
        test_idx.append(train_idx.pop())

    return np.asarray(sorted(train_idx), dtype=np.int64), np.asarray(sorted(test_idx), dtype=np.int64)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    print(f"[OUT] {path} rows={len(df)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tsv", default="/data/projects/3D_scans/3d_bodyscan_20260604.tsv")
    ap.add_argument("--out_dir", default="/data/projects/3D_scans/datasets")
    ap.add_argument("--prefix", default=None, help="Output filename prefix. Default: input stem.")
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sep", default=None, help="Force input delimiter. Default: auto-detect tab/comma.")
    ap.add_argument("--age_bins", nargs="+", type=float, default=[18, 30, 40, 50, 60, 71, 200],
                    help="Fallback numeric age bins. Ignored when agecut exists.")
    ap.add_argument("--agecut_col", default=None, help="Agecut column to use for stratification. Default: auto-detect agecut.")
    ap.add_argument("--target_cols", nargs=4, default=["ho_carotis", "ve_carotis", "ho_fem", "ve_fem"])
    ap.add_argument("--target_missing_policy", choices=["drop", "zero", "error"], default="drop")
    ap.add_argument("--id_col", default=None, help="Participant/scan ID column. Default: auto-detect.")
    match = ap.add_mutually_exclusive_group()
    match.add_argument("--match_npz", default=None, help="Keep only metadata rows whose ID exists in this NPZ names array.")
    match.add_argument("--match_obj_root", default=None, help="Keep only metadata rows whose ID exists under this OBJ root.")
    args = ap.parse_args()

    input_path = Path(args.input_tsv)
    out_dir = Path(args.out_dir)
    prefix = args.prefix or input_path.stem

    sep = _detect_sep(str(input_path), forced=args.sep)
    df = pd.read_csv(
        input_path,
        sep=sep,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        engine="python",
    )
    df = clean_column_names(df)
    df = clean_string_values(df)
    scan_ids = load_match_ids(args.match_npz, args.match_obj_root)
    df = filter_metadata_to_scan_ids(df, scan_ids, args.id_col)
    df = df.reset_index(drop=True)
    df = ensure_athero_columns(df, list(args.target_cols))
    df = apply_target_missing_policy(
        df,
        list(args.target_cols),
        policy=args.target_missing_policy,
        context=str(input_path),
    ).reset_index(drop=True)

    age_col = _pick_col(df, ["age", "alder"], fallback_contains="age") or _pick_col(df, ["alder"], fallback_contains="ald")
    agecut_col = args.agecut_col or _pick_col(
        df, ["agecut", "age_cut", "age_bin", "agebin", "age_group", "agegroup"], fallback_contains="agecut"
    )
    gender_col = _pick_col(df, ["sexMF", "sex", "gender"], fallback_contains="sex")

    missing_targets = [c for c in args.target_cols if c not in df.columns]
    if missing_targets:
        raise KeyError(
            f"Could not find or derive target columns {missing_targets}. Available columns: {list(df.columns)[:100]}"
        )
    if (age_col is None and agecut_col is None) or gender_col is None:
        raise KeyError(
            f"Could not find age/agecut/gender columns. age={age_col}, agecut={agecut_col}, "
            f"gender={gender_col}, columns={list(df.columns)[:100]}"
        )

    g = (
        df[gender_col]
        .astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip("'")
        .str.strip()
        .str.upper()
        .replace({"MALE": "M", "FEMALE": "F"})
        .reset_index(drop=True)
    )
    sick_bits = []
    for c in args.target_cols:
        sick_bits.append((_to_numeric_series(df[c]) > 0.0).astype(int).reset_index(drop=True))
    sick_any = pd.concat(sick_bits, axis=1).max(axis=1).astype(int)
    age_groups = age_group_series(df, age_col, agecut_col, args.age_bins).reset_index(drop=True)
    strata = g.fillna("sexNA") + "_" + age_groups.fillna("ageNA") + "_sick" + sick_any.astype(str)

    train_idx, test_idx = stratified_split_indices(strata, args.test_size, args.seed)
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    write_tsv(train_df, out_dir / f"{prefix}_train.tsv")
    write_tsv(test_df, out_dir / f"{prefix}_test.tsv")

    # Convenience gender-specific files for your current --only_gender F runs.
    for sex in ["F", "M"]:
        write_tsv(train_df[g.iloc[train_idx].to_numpy() == sex], out_dir / f"{prefix}_train_{sex}.tsv")
        write_tsv(test_df[g.iloc[test_idx].to_numpy() == sex], out_dir / f"{prefix}_test_{sex}.tsv")

    print("\n[COUNTS] all split x sick_any")
    print(pd.crosstab(pd.Series(["train"] * len(train_df) + ["test"] * len(test_df), name="split"),
                      pd.concat([sick_any.iloc[train_idx], sick_any.iloc[test_idx]], ignore_index=True),
                      dropna=False).to_string())

    print("\n[COUNTS] gender x split")
    tmp = pd.DataFrame({
        "split": ["train"] * len(train_idx) + ["test"] * len(test_idx),
        "gender": pd.concat([g.iloc[train_idx], g.iloc[test_idx]], ignore_index=True),
    })
    print(pd.crosstab(tmp["gender"], tmp["split"], dropna=False).to_string())


if __name__ == "__main__":
    main()
