#!/usr/bin/env python3
"""Make reproducible train/test TSVs for MeshVAE cohorts.

Default input is the new 2026 TSV:
  /data/projects/3D_scans/3d_bodyscan_20260604.tsv

The script also derives the legacy binary atherosclerosis columns
ho_carotis/ve_carotis/ho_fem/ve_fem from corrected plaque volumes when those
binary columns are not already present.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from meshvae.conditioning import (
    _clean_agecut_label,
    _detect_sep,
    _pick_col,
    _to_numeric_series,
    clean_column_names,
    ensure_athero_columns,
)


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
    df = ensure_athero_columns(df, list(args.target_cols))

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
    )
    sick_bits = []
    for c in args.target_cols:
        sick_bits.append((_to_numeric_series(df[c]).fillna(0.0) > 0.0).astype(int))
    sick_any = pd.concat(sick_bits, axis=1).max(axis=1).astype(int)
    age_groups = age_group_series(df, age_col, agecut_col, args.age_bins)
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
