from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

AGE_BINS = [18, 35, 50, 60, 200]

def n_age_dims(opt) -> int:
    labels = getattr(opt, "age_labels", None)
    if labels:
        return len(list(labels))
    return len(getattr(opt, "age_bins", AGE_BINS)) - 1

def _build_age_labels(age_bins: list[float]) -> list[str]:
    labels = []
    for lo, hi in zip(age_bins[:-1], age_bins[1:]):
        if hi >= 199.5:
            labels.append(f"{int(lo)}+")
        else:
            labels.append(f"{int(lo)}-{int(hi)}")
    return labels

AGE_LABELS = _build_age_labels(AGE_BINS)


def cond_dim_from_opt(opt) -> int:
    if not getattr(opt, "use_cond", False):
        return 0
    extra = 0
    if getattr(opt, "cond_athero", False):
        mode = getattr(opt, "cond_athero_mode", "any")
        extra += 1 if mode == "any" else 4
    if getattr(opt, "cond_plaque", False):
        extra += 4
    return 2 + n_age_dims(opt) + extra


def age_slice_from_opt(opt) -> slice:
    return slice(2, 2 + n_age_dims(opt))


def athero_slice_from_opt(opt) -> slice | None:
    if not getattr(opt, "cond_athero", False):
        return None
    off = 2 + n_age_dims(opt)
    width = 1 if getattr(opt, "cond_athero_mode", "any") == "any" else 4
    return slice(off, off + width)


def plaque_slice_from_opt(opt) -> slice | None:
    off = 2 + n_age_dims(opt)
    if getattr(opt, "cond_athero", False):
        off += 1 if getattr(opt, "cond_athero_mode", "any") == "any" else 4
    if not getattr(opt, "cond_plaque", False):
        return None
    return slice(off, off + 4)


def _age_to_bin(
    age: float,
    age_bins: list[float] | None = None,
    age_labels: list[str] | None = None,
) -> str | None:
    if age is None:
        return None
    try:
        a = float(age)
    except Exception:
        return None

    bins = list(age_bins) if age_bins is not None else list(AGE_BINS)
    labels = list(age_labels) if age_labels is not None else list(_build_age_labels(bins))

    for i in range(len(labels)):
        lo, hi = bins[i], bins[i + 1]
        if (a >= lo) and (a < hi or (i == len(labels) - 1 and a >= lo)):
            return labels[i]
    return None


def _age_centers(age_bins: list[float]) -> np.ndarray:
    centers = []
    for lo, hi in zip(age_bins[:-1], age_bins[1:]):
        hi_eff = 85.0 if hi >= 199.5 else hi
        centers.append(0.5 * (lo + hi_eff))
    return np.asarray(centers, dtype=np.float32)


def _age_to_vec(age: float, opt) -> np.ndarray:
    bins = list(map(float, getattr(opt, "age_bins", AGE_BINS)))
    mode = str(getattr(opt, "cond_age_mode", "hard"))
    centers = _age_centers(bins)
    n = len(bins) - 1
    a = float(age)

    if mode == "hard":
        vec = np.zeros(n, dtype=np.float32)
        idx = max(0, min(n - 1, int(np.searchsorted(np.asarray(bins[1:-1]), a, side="right"))))
        vec[idx] = 1.0
        return vec

    if mode == "soft":
        vec = np.zeros(n, dtype=np.float32)
        if a <= centers[0]:
            vec[0] = 1.0
            return vec
        if a >= centers[-1]:
            vec[-1] = 1.0
            return vec

        for i in range(len(centers) - 1):
            c0, c1 = float(centers[i]), float(centers[i + 1])
            if c0 <= a <= c1:
                t = (a - c0) / max(c1 - c0, 1e-6)
                vec[i] = 1.0 - t
                vec[i + 1] = t
                return vec

        vec[-1] = 1.0
        return vec

    if mode == "rbf":
        gamma = float(getattr(opt, "age_rbf_gamma", 0.015))
        w = np.exp(-gamma * (a - centers) ** 2).astype(np.float32)
        w /= max(float(w.sum()), 1e-6)
        return w

    raise ValueError(f"Unknown --cond_age_mode={mode}")

def clean_stregkode(x) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    s = Path(s).name
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"\.obj$", "", s, flags=re.IGNORECASE)
    s = s.strip(" !").upper()
    m = re.search(r"([A-Z]{2,}DET\d{5,10})", s)
    if m:
        return m.group(1)
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s if s else None

def _pick_col(df, preferred: list[str] | None, fallback_contains: str | None = None):
    if preferred:
        for cand in preferred:
            if cand and cand in df.columns:
                return cand
            if cand:
                for col in df.columns:
                    if col.lower() == cand.lower():
                        return col
    if fallback_contains:
        for col in df.columns:
            if fallback_contains in col.lower():
                return col
    return None

def _detect_sep(path: str, forced: str | None = None) -> str:
    if forced:
        return forced
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.readline()
        # crude but effective: whichever delimiter appears more
        return "\t" if head.count("\t") >= head.count(",") else ","
    except Exception:
        return "\t"


def _clean_colname(col) -> str:
    """Make messy exported TSV headers usable: strip quotes/BOM/spaces."""
    return str(col).strip().lstrip("\ufeff").strip().strip('"').strip("'")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_colname(c) for c in df.columns]
    return df


def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Numeric conversion that also handles decimal commas like '40,15'."""
    return pd.to_numeric(
        s.astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip("'")
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _clean_agecut_label(x) -> str | None:
    """Normalize cohort agecut labels like '18-29', '30-39', '60-70'."""
    if x is None:
        return None
    s = str(x).strip().strip('"').strip("'").strip()
    if not s or s.lower() in {"nan", "none", "null", "na", "<na>"}:
        return None
    s = s.replace("–", "-").replace("—", "-").replace(" ", "")
    return s


def _agecut_sort_key(label: str):
    """Sort agecut labels by their leading number, then by trailing number."""
    s = _clean_agecut_label(label) or ""
    m = re.match(r"^(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?)|\+)?$", s)
    if not m:
        return (float("inf"), float("inf"), s)
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) is not None else float("inf")
    return (lo, hi, s)


def _labels_from_agecut(series: pd.Series) -> list[str]:
    vals = [_clean_agecut_label(x) for x in series.tolist()]
    vals = [v for v in vals if v is not None]
    return sorted(dict.fromkeys(vals), key=_agecut_sort_key)


def _agecut_to_vec(label: str, labels: list[str]) -> np.ndarray:
    vec = np.zeros(len(labels), dtype=np.float32)
    lab = _clean_agecut_label(label)
    if lab in labels:
        vec[labels.index(lab)] = 1.0
    return vec


def _plaque_source_candidates_for_athero_col(col: str) -> list[str]:
    """Fallback mappings from binary atherosclerosis columns to plaque volume columns.

    The old pipeline expects ho_carotis/ve_carotis/ho_fem/ve_fem.  The new
    2026 TSV shown in your screenshot appears to contain corrected plaque
    volumes instead.  If the binary columns are absent, we derive them as
    volume > 0 from these columns.
    """
    c = str(col).lower()
    mapping = {
        "ho_carotis": [
            "plaque_volume_right_carotid_corrected",
            "right_carotid_plaque_volume",
            "ct_ho_carot_plaq_vol",
        ],
        "ve_carotis": [
            "plaque_volume_left_carotid_corrected",
            "left_carotid_plaque_volume",
            "ct_ve_carot_plaq_vol",
        ],
        "ho_fem": [
            "plaque_volume_right_femoral_corrected",
            "right_femoral_plaque_volume",
            "ct_ho_fem_plaq_vol",
        ],
        "ve_fem": [
            "plaque_volume_left_femoral_corrected",
            "left_femoral_plaque_volume",
            "ct_ve_fem_plaq_vol",
        ],
    }
    return mapping.get(c, [])


def ensure_athero_columns(df: pd.DataFrame, athero_cols: list[str]) -> pd.DataFrame:
    """Ensure disease target columns exist, deriving them from plaque volumes if needed."""
    df = df.copy()
    lower_to_col = {str(c).lower(): c for c in df.columns}

    for target_col in list(athero_cols or []):
        if target_col in df.columns:
            continue
        match = None
        for cand in _plaque_source_candidates_for_athero_col(target_col):
            if cand in df.columns:
                match = cand
                break
            if cand.lower() in lower_to_col:
                match = lower_to_col[cand.lower()]
                break
        if match is not None:
            df[target_col] = (_to_numeric_series(df[match]).fillna(0.0) > 0.0).astype(np.float32)
            print(f"[COND] Derived {target_col} = ({match} > 0)")

    return df


def build_age_gender_lookup(opt):
    """
    Returns:
      code2cond: dict stregkode_clean -> np.array(cond_dim,) float32
      code2group: dict stregkode_clean -> 'F_50-60', etc.
      code2aux: dict stregkode_clean -> {"plaque_target": np.array(4,), "age": float32}
    """
    need_any_meta = (
        getattr(opt, "use_cond", False)
        or getattr(opt, "aux_predict_plaque", False)
    )
    if not need_any_meta:
        return {}, {}, {}

    cond_path = opt.cond_tsv
    if cond_path is None:
        cands = sorted(Path("/data/projects/3D_scans").glob("cohort_*"))
        if not cands:
            raise FileNotFoundError("No --cond_tsv provided and no cohort_* found under /data/projects/3D_scans")
        cond_path = str(cands[0])

    sep = _detect_sep(cond_path, forced=getattr(opt, "cond_sep", None))

    df = pd.read_csv(
        cond_path,
        sep=sep,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        engine="python",
    )
    df = clean_column_names(df)

    st_col = opt.cond_stregkode_col or _pick_col(
        df, ["stregkode", "participant_id", "participantid", "id"], fallback_contains="stregkode"
    )
    if st_col is None:
        st_col = _pick_col(df, ["participant_id", "participantid"], fallback_contains="participant")
    age_col = opt.cond_age_col or _pick_col(df, ["alder", "age"], fallback_contains="ald")
    if age_col is None:
        age_col = _pick_col(df, ["age"], fallback_contains="age")
    agecut_col = getattr(opt, "cond_agecut_col", None) or _pick_col(
        df,
        ["agecut", "age_cut", "age_bin", "agebin", "age_group", "agegroup"],
        fallback_contains="agecut",
    )
    g_col = opt.cond_gender_col or _pick_col(df, ["sex", "gender", "sexMF"], fallback_contains="sex")

    age_source = str(getattr(opt, "cond_age_source", "auto")).strip().lower()
    use_agecut = age_source == "agecut" or (age_source == "auto" and agecut_col is not None)

    if st_col is None or g_col is None or (age_col is None and not use_agecut):
        raise KeyError(
            f"Could not find columns in {cond_path}. "
            f"stregkode={st_col}, age={age_col}, agecut={agecut_col}, gender={g_col}. "
            f"Available: {list(df.columns)[:80]}"
        )
    if use_agecut and agecut_col is None:
        raise KeyError(
            f"--cond_age_source agecut was requested, but no agecut column was found in {cond_path}. "
            f"Available: {list(df.columns)[:80]}"
        )

    athero_cols = list(getattr(opt, "cond_athero_cols", [])) if getattr(opt, "cond_athero", False) else []
    if athero_cols:
        df = ensure_athero_columns(df, athero_cols)

    cond_plaque_cols = list(getattr(opt, "cond_plaque_cols", [])) if getattr(opt, "cond_plaque", False) else []
    aux_plaque_cols = list(getattr(opt, "aux_plaque_cols", [])) if getattr(opt, "aux_predict_plaque", False) else []

    base_cols = [st_col, g_col]
    if age_col is not None:
        base_cols.append(age_col)
    if use_agecut and agecut_col is not None:
        base_cols.append(agecut_col)
    use_cols = list(dict.fromkeys(base_cols + athero_cols + cond_plaque_cols + aux_plaque_cols))
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing metadata columns in {cond_path}: {missing}. "
            f"Available: {list(df.columns)[:80]}"
        )

    out = df[use_cols].copy()
    rename_map = {st_col: "stregkode_raw", g_col: "gender_raw"}
    if age_col is not None:
        rename_map[age_col] = "age_raw"
    if use_agecut and agecut_col is not None:
        rename_map[agecut_col] = "agecut_raw"
    out = out.rename(columns=rename_map)

    out["stregkode_clean"] = out["stregkode_raw"].apply(clean_stregkode)
    if "age_raw" in out.columns:
        out["age_num"] = _to_numeric_series(out["age_raw"])
    else:
        out["age_num"] = np.nan
    if use_agecut:
        out["agecut_label"] = out["agecut_raw"].apply(_clean_agecut_label)
    g = out["gender_raw"].astype(str).str.strip().str.upper().str.strip('"').str.strip("'").replace({"MALE": "M", "FEMALE": "F"})
    out["gender_clean"] = g

    drop_cols = ["stregkode_clean"] + ([] if use_agecut else ["age_num"])
    out = out.dropna(subset=drop_cols).copy()
    if not use_agecut:
        out = out[out["age_num"] >= 18].copy()
    else:
        out = out.dropna(subset=["agecut_label"]).copy()
        if "age_num" in out.columns and out["age_num"].notna().any():
            out = out[(out["age_num"].isna()) | (out["age_num"] >= 18)].copy()
    out = out[out["gender_clean"].isin(["M", "F"])].copy()

    only_gender = getattr(opt, "only_gender", None)
    if only_gender is not None:
        only_gender = str(only_gender).upper()
        out = out[out["gender_clean"] == only_gender].copy()
        if len(out) == 0:
            raise ValueError(f"No rows left after filtering only_gender={only_gender} in {cond_path}")
        
    if use_agecut:
        age_bins = list(map(float, getattr(opt, "age_bins", []) or []))
        age_labels = list(getattr(opt, "age_labels", []) or [])
        if not age_labels:
            age_labels = _labels_from_agecut(out["agecut_label"])
        if not age_labels:
            raise ValueError(f"No usable agecut labels found in {cond_path}")
        out["age_bin"] = out["agecut_label"].apply(_clean_agecut_label)
        before_age = len(out)
        out = out[out["age_bin"].isin(age_labels)].copy()
        dropped_age = before_age - len(out)
        if dropped_age > 0:
            print(f"[COND] Dropped {dropped_age} rows with agecut labels not seen during training: {age_labels}")
        opt.cond_age_source = "agecut"
        opt.cond_agecut_col = agecut_col
        opt.age_labels = age_labels
        opt.age_bins = age_bins
    else:
        age_bins = list(map(float, getattr(opt, "age_bins", AGE_BINS)))
        if len(age_bins) < 2:
            raise ValueError("Numeric age conditioning requires --age_bins with at least two edges.")
        age_labels = _build_age_labels(age_bins)
        out["age_bin"] = out["age_num"].apply(lambda a: _age_to_bin(a, age_bins, age_labels))
        out = out.dropna(subset=["age_bin"]).copy()
        opt.cond_age_source = "age"
        opt.age_labels = age_labels
        opt.age_bins = age_bins

    print(f"[COND] age_source={getattr(opt, 'cond_age_source', 'age')} labels={age_labels}")

    if athero_cols:
        for c in athero_cols:
            out[c] = _to_numeric_series(out[c]).fillna(0.0).clip(0, 1)
        out["athero_any"] = (out[athero_cols].max(axis=1) > 0.5).astype(np.float32)

    def _normalize_plaque(cols: list[str], norm_name: str):
        if not cols:
            return None, None, None

        P_raw = []
        for c in cols:
            P_raw.append(_to_numeric_series(out[c]).fillna(0.0).to_numpy(np.float32))
        P_raw = np.stack(P_raw, axis=1)  # (N,4)

        if norm_name == "log1p_z":
            P = np.log1p(P_raw)
            mean = P.mean(axis=0).astype(np.float32)
            std = (P.std(axis=0).astype(np.float32) + 1e-6)
            Pn = (P - mean[None, :]) / std[None, :]
        else:
            Pn = P_raw.astype(np.float32)
            mean = np.zeros(Pn.shape[1], dtype=np.float32)
            std = np.ones(Pn.shape[1], dtype=np.float32)

        return Pn, mean, std

    cond_plaque_mat, cond_plaque_mean, cond_plaque_std = _normalize_plaque(
        cond_plaque_cols, getattr(opt, "cond_plaque_norm", "log1p_z")
    )
    aux_plaque_mat, aux_plaque_mean, aux_plaque_std = _normalize_plaque(
        aux_plaque_cols, getattr(opt, "aux_plaque_norm", "log1p_z")
    )

    cond_names = ["gender_M", "gender_F"] + [f"age_{lab}" for lab in age_labels]
    if athero_cols:
        if getattr(opt, "cond_athero_mode", "any") == "any":
            cond_names += ["athero_any"]
        else:
            cond_names += [f"athero_{c}" for c in athero_cols]
    if cond_plaque_cols:
        cond_names += [f"plaque_{c}" for c in cond_plaque_cols]

    demo_cond_dim = 2 + len(age_labels)
    cond_dim = demo_cond_dim
    if athero_cols:
        cond_dim += (1 if getattr(opt, "cond_athero_mode", "any") == "any" else 4)
    if cond_plaque_cols:
        cond_dim += 4

    if getattr(opt, "use_cond", False):
        opt.demo_cond_dim = demo_cond_dim
        opt.cond_dim = cond_dim
    else:
        opt.demo_cond_dim = 0
        opt.cond_dim = 0

    conds = []
    groups = []
    aux_rows = []

    for row_idx, (_, row) in enumerate(out.iterrows()):
        cvec = np.zeros(cond_dim, dtype=np.float32)

        gender = row["gender_clean"]
        ab = row["age_bin"]

        cvec[0 if gender == "M" else 1] = 1.0
        if use_agecut:
            cvec[2:demo_cond_dim] = _agecut_to_vec(str(row["age_bin"]), age_labels)
        else:
            cvec[2:demo_cond_dim] = _age_to_vec(float(row["age_num"]), opt)
        off = demo_cond_dim

        if athero_cols:
            mode = getattr(opt, "cond_athero_mode", "any")
            if mode == "any":
                cvec[off] = float(row["athero_any"])
                off += 1
            else:
                cvec[off:off + 4] = row[athero_cols].to_numpy(np.float32)
                off += 4

        if cond_plaque_cols and cond_plaque_mat is not None:
            cvec[off:off + 4] = cond_plaque_mat[row_idx]
            off += 4

        conds.append(cvec)
        groups.append(f"{gender}_{ab}")

        aux_entry = {
            "age": np.float32(row["age_num"]),
        }
        if aux_plaque_mat is not None:
            aux_entry["plaque_target"] = aux_plaque_mat[row_idx].astype(np.float32)
        aux_rows.append(aux_entry)

    out["cond"] = conds
    out["group"] = groups
    out["aux"] = aux_rows

    out = out.drop_duplicates(subset=["stregkode_clean"]).reset_index(drop=True)

    if athero_cols and getattr(opt, "is_train", False):
        if "athero_any" not in out.columns:
            out["athero_any"] = (out[athero_cols].max(axis=1) > 0.5).astype(np.float32)

        out["athero_any_int"] = (out["athero_any"] > 0.5).astype(int)

        age_tab = (
            out.groupby(["age_bin", "athero_any_int"])
            .size()
            .unstack(fill_value=0)
            .rename(columns={0: "athero0", 1: "athero1"})
            .reset_index()
            .sort_values(["age_bin"])
        )

        ga_tab = (
            out.groupby(["gender_clean", "age_bin", "athero_any_int"])
            .size()
            .unstack(fill_value=0)
            .rename(columns={0: "athero0", 1: "athero1"})
            .reset_index()
            .sort_values(["gender_clean", "age_bin"])
        )

        stats_dir = Path(opt.checkpoints_dir) / opt.name
        stats_dir.mkdir(parents=True, exist_ok=True)

        age_path = stats_dir / "athero_counts_by_age.csv"
        ga_path = stats_dir / "athero_counts_by_gender_age.csv"

        age_tab.to_csv(age_path, index=False)
        ga_tab.to_csv(ga_path, index=False)

        print("\n[COND] Atherosclerosis any-bit counts by age:")
        print(age_tab.to_string(index=False))
        print(f"[COND] wrote -> {age_path}")

        print("\n[COND] Atherosclerosis any-bit counts by gender+age:")
        print(ga_tab.to_string(index=False))
        print(f"[COND] wrote -> {ga_path}")

    code2cond = dict(zip(out["stregkode_clean"], out["cond"]))
    code2group = dict(zip(out["stregkode_clean"], out["group"]))
    code2aux = dict(zip(out["stregkode_clean"], out["aux"]))

    try:
        stats = {
            "cond_dim": int(cond_dim),
            "cond_names": cond_names,
            "cond_path": str(cond_path),
            "feature_source": str(getattr(opt, "feature_source", "scan")),
            "clinical_points": int(getattr(opt, "clinical_points", 0)),
            "ninput_edges": int(getattr(opt, "ninput_edges", 0)),
            "clinical_feature_cols": list(getattr(opt, "clinical_feature_cols", []) or []),
            "sep": sep,
            "age_bins": age_bins,
            "age_labels": age_labels,
            "cond_age_source": str(getattr(opt, "cond_age_source", "age")),
            "agecut_col": agecut_col if use_agecut else None,
            "cond_age_mode": str(getattr(opt, "cond_age_mode", "hard")),
            "age_rbf_gamma": float(getattr(opt, "age_rbf_gamma", 0.015)),
            "only_gender": getattr(opt, "only_gender", None),
            "athero_cols": athero_cols,
            "plaque_cols": cond_plaque_cols,
            "plaque_norm": getattr(opt, "cond_plaque_norm", "log1p_z"),
            "plaque_mean": cond_plaque_mean.tolist() if cond_plaque_mean is not None else None,
            "plaque_std": cond_plaque_std.tolist() if cond_plaque_std is not None else None,
            "aux_plaque_cols": aux_plaque_cols,
            "aux_plaque_norm": getattr(opt, "aux_plaque_norm", "log1p_z"),
            "aux_plaque_mean": aux_plaque_mean.tolist() if aux_plaque_mean is not None else None,
            "aux_plaque_std": aux_plaque_std.tolist() if aux_plaque_std is not None else None,
            "aux_predict_athero": bool(getattr(opt, "aux_predict_athero", False)),
            "aux_athero_mode": str(getattr(opt, "aux_athero_mode", "any")),
            "aux_athero_weight": float(getattr(opt, "aux_athero_weight", 0.0)),
            "aux_athero_pos_weight": float(getattr(opt, "aux_athero_pos_weight", 1.0)),
            "aux_athero_dropout": float(getattr(opt, "aux_athero_dropout", 0.15)),

            "use_cond_prior": bool(getattr(opt, "use_cond_prior", False)),
            "demo_cond_dim": int(getattr(opt, "demo_cond_dim", demo_cond_dim)),
            "encoder_cond_mode": str(getattr(opt, "encoder_cond_mode", "none")),
            "encoder_use_cond": False,
            "adv_predict_age": bool(getattr(opt, "adv_predict_age", False)),
            "adv_age_weight": float(getattr(opt, "adv_age_weight", 0.0)),
            "adv_age_hidden": int(getattr(opt, "adv_age_hidden", 128)),
            "adv_age_dropout": float(getattr(opt, "adv_age_dropout", 0.15)),
            "adv_age_grl_lambda": float(getattr(opt, "adv_age_grl_lambda", 1.0)),

            "athero_mode": getattr(opt, "cond_athero_mode", "any") if athero_cols else None,
        }
        if getattr(opt, "is_train", False):
            stats_path = Path(opt.checkpoints_dir) / opt.name / "cond_stats.json"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"[WARN] Could not write cond_stats.json: {e}", flush=True)

    return code2cond, code2group, code2aux
