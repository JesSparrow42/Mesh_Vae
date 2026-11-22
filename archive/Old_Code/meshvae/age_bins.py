from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .conditioning import _build_age_labels, _detect_sep, _pick_col

def build_dynamic_base_edges(ages: np.ndarray, opt) -> list[float]:
    min_age = max(18, int(np.floor(np.min(ages))))
    max_age = int(np.ceil(np.max(ages)))

    switch_age = int(getattr(opt, "age_search_switch_age", 50))
    step_y = int(getattr(opt, "age_search_step_young", 10))
    step_o = int(getattr(opt, "age_search_step_old", 5))

    edges = [18.0, 20.0]

    a = 20
    while a < min(max_age + 1, switch_age):
        a += step_y
        if a < switch_age:
            edges.append(float(a))

    if edges[-1] != float(switch_age):
        edges.append(float(switch_age))

    a = switch_age
    while a < max_age:
        a += step_o
        if a < 200:
            edges.append(float(a))

    if edges[-1] != 200.0:
        edges.append(200.0)

    edges = sorted(set(edges))
    return edges

def bin_counts(ages: np.ndarray, edges: list[float]) -> np.ndarray:
    counts, _ = np.histogram(ages, bins=np.asarray(edges, dtype=np.float64))
    return counts

def bin_prevalences(ages: np.ndarray, labels: np.ndarray, edges: list[float]) -> np.ndarray:
    prev = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (ages >= lo) & (ages < hi if hi < 199.5 else ages >= lo)
        if mask.sum() == 0:
            prev.append(np.nan)
        else:
            prev.append(float(labels[mask].mean()))
    return np.asarray(prev, dtype=np.float64)

def merge_adjacent_bins(edges: list[float], merge_idx: int) -> list[float]:
    # remove one internal edge
    new_edges = edges[:merge_idx + 1] + edges[merge_idx + 2:]
    return new_edges

def generate_dynamic_age_bin_candidates(opt) -> list[list[float]]:
    cond_path = opt.cond_tsv
    sep = _detect_sep(cond_path, forced=getattr(opt, "cond_sep", None))
    df = pd.read_csv(cond_path, sep=sep, dtype=str, quoting=csv.QUOTE_NONE, on_bad_lines="skip", engine="python")

    age_col = opt.cond_age_col or _pick_col(df, ["alder", "age"], fallback_contains="ald")
    g_col = opt.cond_gender_col or _pick_col(df, ["sex", "gender"], fallback_contains="sex")
    athero_cols = list(getattr(opt, "cond_athero_cols", []))

    out = df[[age_col, g_col] + athero_cols].copy()
    out = out.rename(columns={age_col: "age_raw", g_col: "gender_raw"})
    out["age_num"] = pd.to_numeric(out["age_raw"], errors="coerce")
    out["gender_clean"] = out["gender_raw"].astype(str).str.strip().str.upper().replace({"MALE": "M", "FEMALE": "F"})
    out = out.dropna(subset=["age_num"]).copy()
    out = out[out["age_num"] >= 18].copy()
    out = out[out["gender_clean"].isin(["M", "F"])].copy()

    only_gender = getattr(opt, "only_gender", None)
    if only_gender is not None:
        out = out[out["gender_clean"] == str(only_gender).upper()].copy()

    for c in athero_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(0, 1)
    out["label"] = (out[athero_cols].max(axis=1) > 0.5).astype(np.float32)

    ages = out["age_num"].to_numpy(dtype=np.float64)
    labels = out["label"].to_numpy(dtype=np.float32)

    min_bin_size = int(getattr(opt, "age_auto_min_bin_size", 80))
    min_bins = int(getattr(opt, "age_search_min_bins", 4))
    max_bins = int(getattr(opt, "age_search_max_bins", 10))

    edges = build_dynamic_base_edges(ages, opt)

    # first enforce minimum bin size by greedy merging
    while True:
        counts = bin_counts(ages, edges)
        if counts.min() >= min_bin_size or len(edges) - 1 <= min_bins:
            break

        prev = bin_prevalences(ages, labels, edges)
        bad = int(np.argmin(counts))

        candidates = []
        if bad > 0:
            # merge with left
            candidates.append((abs(prev[bad] - prev[bad - 1]) if not np.isnan(prev[bad - 1]) else 1e9, bad - 1))
        if bad < len(counts) - 1:
            # merge with right
            candidates.append((abs(prev[bad] - prev[bad + 1]) if not np.isnan(prev[bad + 1]) else 1e9, bad))

        _, merge_idx = min(candidates, key=lambda t: t[0])
        edges = merge_adjacent_bins(edges, merge_idx)

    candidates = [edges.copy()]

    # keep generating coarser candidates by merging the most similar adjacent bins
    while len(edges) - 1 > min_bins:
        prev = bin_prevalences(ages, labels, edges)
        adj = []
        for i in range(len(prev) - 1):
            if np.isnan(prev[i]) or np.isnan(prev[i + 1]):
                continue
            adj.append((abs(prev[i] - prev[i + 1]), i))
        if not adj:
            break
        _, merge_idx = min(adj, key=lambda t: t[0])
        edges = merge_adjacent_bins(edges, merge_idx)
        if len(edges) - 1 <= max_bins:
            candidates.append(edges.copy())

    return _dedupe_candidates(candidates)

def _finalize_candidate_edges(edges):
    mids = [int(round(float(x))) for x in edges[1:-1]]
    mids = sorted(mids)
    out = []
    for x in mids:
        if 18 < x < 200 and (len(out) == 0 or x > out[-1]):
            out.append(x)
    if len(out) != 3:
        return None
    return [18.0, float(out[0]), float(out[1]), float(out[2]), 200.0]


def _candidate_has_min_bin_size(ages: np.ndarray, edges: list[float], min_bin_size: int) -> bool:
    counts, _ = np.histogram(ages, bins=np.asarray(edges, dtype=np.float64))
    return bool(np.all(counts >= int(min_bin_size)))


def _dedupe_candidates(cands: list[list[float]]) -> list[list[float]]:
    seen = set()
    out = []
    for c in cands:
        key = tuple(int(round(x)) for x in c)
        if key not in seen:
            seen.add(key)
            out.append([float(x) for x in c])
    return out


def generate_prevalence_bin_candidates(opt) -> list[list[float]]:
    if not getattr(opt, "use_cond", False):
        raise ValueError("Prevalence-based age search requires --use_cond")
    if not getattr(opt, "cond_athero", False):
        raise ValueError("Prevalence-based age search requires --cond_athero")

    cond_path = opt.cond_tsv
    if cond_path is None:
        raise ValueError("--cond_tsv is required for prevalence-based age search")

    sep = _detect_sep(cond_path, forced=getattr(opt, "cond_sep", None))
    df = pd.read_csv(
        cond_path,
        sep=sep,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        engine="python",
    )

    st_col = opt.cond_stregkode_col or _pick_col(df, ["stregkode"], fallback_contains="stregkode")
    age_col = opt.cond_age_col or _pick_col(df, ["alder", "age"], fallback_contains="ald")
    g_col = opt.cond_gender_col or _pick_col(df, ["sex", "gender"], fallback_contains="sex")
    athero_cols = list(getattr(opt, "cond_athero_cols", []))

    use_cols = [st_col, age_col, g_col] + athero_cols
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for prevalence search in {cond_path}: {missing}")

    out = df[use_cols].copy()
    out = out.rename(columns={st_col: "stregkode_raw", age_col: "age_raw", g_col: "gender_raw"})
    out["age_num"] = pd.to_numeric(out["age_raw"], errors="coerce")
    out["gender_clean"] = (
        out["gender_raw"].astype(str).str.strip().str.upper().replace({"MALE": "M", "FEMALE": "F"})
    )
    out = out.dropna(subset=["age_num"]).copy()
    out = out[out["age_num"] >= 18].copy()
    out = out[out["gender_clean"].isin(["M", "F"])].copy()

    only_gender = getattr(opt, "only_gender", None)
    if only_gender is not None:
        out = out[out["gender_clean"] == str(only_gender).upper()].copy()

    for c in athero_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(0, 1)

    out["label"] = (out[athero_cols].max(axis=1) > 0.5).astype(np.float32)

    ages = out["age_num"].to_numpy(dtype=np.float64)
    pos_ages = out.loc[out["label"] > 0.5, "age_num"].to_numpy(dtype=np.float64)

    min_bin_size = int(getattr(opt, "age_auto_min_bin_size", 80))
    max_candidates = int(getattr(opt, "age_auto_n_candidates", 10))
    smooth_window = int(getattr(opt, "age_auto_smooth_window", 5))

    cand_sets = []

    # 1) age-quantile candidates
    for qs in ([0.20, 0.40, 0.60], [0.25, 0.50, 0.75], [0.30, 0.55, 0.80]):
        edges = [18.0] + [float(np.quantile(ages, q)) for q in qs] + [200.0]
        edges = _finalize_candidate_edges(edges)
        if edges is not None and _candidate_has_min_bin_size(ages, edges, min_bin_size):
            cand_sets.append(edges)

    # 2) positive-case age-quantile candidates
    if len(pos_ages) >= max(20, min_bin_size):
        for qs in ([0.20, 0.40, 0.60], [0.25, 0.50, 0.75]):
            edges = [18.0] + [float(np.quantile(pos_ages, q)) for q in qs] + [200.0]
            edges = _finalize_candidate_edges(edges)
            if edges is not None and _candidate_has_min_bin_size(ages, edges, min_bin_size):
                cand_sets.append(edges)

    # 3) prevalence-change candidates
    by_age = (
        out.assign(age_int=np.floor(out["age_num"]).astype(int))
           .groupby("age_int")["label"]
           .mean()
           .reset_index()
    )

    full_age = pd.DataFrame({"age_int": np.arange(max(18, int(by_age["age_int"].min())), int(by_age["age_int"].max()) + 1)})
    by_age = full_age.merge(by_age, on="age_int", how="left")
    by_age["label"] = by_age["label"].interpolate().bfill().ffill()
    by_age["prev_s"] = by_age["label"].rolling(window=smooth_window, center=True, min_periods=1).mean()
    by_age["delta"] = by_age["prev_s"].diff().abs().fillna(0.0)

    top_peaks = (
        by_age.sort_values("delta", ascending=False)["age_int"]
        .drop_duplicates()
        .tolist()
    )
    top_peaks = sorted([int(x) for x in top_peaks[:8] if 18 < x < 90])

    if len(top_peaks) >= 3:
        edges = _finalize_candidate_edges([18.0, top_peaks[0], top_peaks[len(top_peaks)//2], top_peaks[-1], 200.0])
        if edges is not None and _candidate_has_min_bin_size(ages, edges, min_bin_size):
            cand_sets.append(edges)

    cand_sets.append([18.0, 35.0, 50.0, 60.0, 200.0])  # keep your old default as a baseline

    cand_sets = _dedupe_candidates(cand_sets)
    cand_sets = cand_sets[:max_candidates]

    if len(cand_sets) == 0:
        raise ValueError("No valid automatic age-bin candidates were generated.")

    return cand_sets

def search_best_age_bins(opt):
    import copy
    from .diagnostics import evaluate_search_candidate_from_indices
    from .training import train

    if str(getattr(opt, "age_search_source", "json")) == "dynamic":
        candidate_sets = generate_dynamic_age_bin_candidates(opt)
    elif str(getattr(opt, "age_search_source", "json")) == "prevalence":
        candidate_sets = generate_prevalence_bin_candidates(opt)
    else:
        if not opt.age_bin_candidates_json:
            raise ValueError("--age_bin_candidates_json is required for --phase search_age_bins when --age_search_source json")
        candidate_sets = json.loads(Path(opt.age_bin_candidates_json).read_text())

    results = []

    for edges in candidate_sets:
        if len(edges) < 3:
            continue

        trial = copy.deepcopy(opt)
        trial.phase = "train"
        trial.is_train = True
        trial.age_bins = [float(x) for x in edges]
        trial.age_labels = _build_age_labels(trial.age_bins)
        trial.niter = int(opt.age_search_epochs)
        trial.niter_decay = 0
        trial.name = f"{opt.name}_agebins_{'_'.join(str(int(x)) for x in edges[:-1])}"

        print(f"\n[AGE-SEARCH] trying age_bins={trial.age_bins}  -> name={trial.name}")
        out = train(trial)

        diag = evaluate_search_candidate_from_indices(
            dataset=out["dataset"],
            calib_idx=out["train_idx"],
            test_idx=out["val_idx"],
            model=out["model"],
            opt=trial,
        )

        results.append({
            "trial_name": trial.name,
            "age_bins": json.dumps(trial.age_bins),
            "n_bins": len(trial.age_bins) - 1,
            "best_val": float(out["best_val"]),
            "best_epoch": int(out["best_epoch"]),
            "diag_score_name": diag["chosen_score"],
            "diag_bal_acc": float(diag["balanced_accuracy"]),
            "diag_auc": float(diag["roc_auc"]),
            "diag_sens": float(diag["sensitivity"]),
            "diag_spec": float(diag["specificity"]),
        })

    df = pd.DataFrame(results)

    objective = str(getattr(opt, "age_search_objective", "diagnostic_bal_acc"))
    if objective == "diagnostic_auc":
        df = df.sort_values(["diag_auc", "diag_bal_acc"], ascending=[False, False]).reset_index(drop=True)
    elif objective == "val_loss":
        df = df.sort_values("best_val", ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values(["diag_bal_acc", "diag_auc"], ascending=[False, False]).reset_index(drop=True)

    out_csv = Path(opt.checkpoints_dir) / opt.name / "age_bin_search.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"[AGE-SEARCH] wrote -> {out_csv}")
    return df

# =========================================
# Train / val split
# =========================================
