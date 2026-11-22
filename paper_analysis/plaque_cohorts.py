"""Auditable plaque targets and training-fitted exploratory cohorts (numpy/pandas only)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from meshvae.conditioning import clean_stregkode
from meshvae.scan_geometry import load_aligned_scan_geometry
from scripts.benchmarks.run_cardiovascular_target_benchmark import _read_table, _id_column, _resolve_column, _numeric

TERRITORIES = ("carotid_right", "carotid_left", "carotid_total", "femoral_right",
               "femoral_left", "femoral_total", "coronary_volume")
WEIGHT_ALIASES = ["weight_kg", "weight", "body_weight_kg", "body_weight", "vaegt", "vægt", "weightkg"]


def identified(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["participant_id"] = frame[_id_column(frame)].map(clean_stregkode)
    if frame["participant_id"].isna().any() or frame["participant_id"].duplicated().any():
        raise ValueError(f"{context}: missing or duplicate participant IDs; resolve before training.")
    return frame.set_index("participant_id", drop=False)


def derive_targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out, audit = frame.copy(), {}
    values = {}
    for site in ("carotid", "femoral"):
        for side in ("right", "left"):
            key = f"{site}_{side}"
            raw, source = _numeric(out, [f"plaque_volume_{side}_{site}_corrected", f"plaque_volume_{side}_{site}"])
            valid = np.isfinite(raw) & (raw >= 0)
            values[key] = raw.where(valid)
            audit[key] = {"source": source, "invalid": int((raw.notna() & ~valid).sum())}
        direct, source = _numeric(out, [f"pv_{'caro' if site == 'carotid' else 'femo'}"])
        direct = direct.where(np.isfinite(direct) & (direct >= 0))
        summed = pd.concat([values[f"{site}_right"], values[f"{site}_left"]], axis=1).sum(axis=1, min_count=2)
        values[f"{site}_total"] = direct.fillna(summed)
        overlap = direct.notna() & summed.notna()
        audit[f"{site}_total"] = {"source": source, "fallback": "both measured sides",
            "direct_sum_disagreements": int((overlap & ~np.isclose(direct, summed)).sum())}
    coronary, source = _numeric(out, ["pv_coro"])
    fallback, fallback_source = _numeric(out, ["ct_qa_cor_tot_plaq_vol", "coronary_plaque_volume"])
    coronary = coronary.where(np.isfinite(coronary) & (coronary >= 0))
    fallback = fallback.where(np.isfinite(fallback) & (fallback >= 0))
    values["coronary_volume"] = coronary.fillna(fallback)
    audit["coronary_volume"] = {"source": source, "fallback": fallback_source}
    sides = [values[x] for x in ("carotid_right", "carotid_left", "femoral_right", "femoral_left")]
    values["peripheral_total"] = pd.concat(sides, axis=1).sum(axis=1, min_count=4)
    audit["peripheral_total"] = {"definition": "sum of four carotid/femoral sides; all four required; excludes coronary"}
    for key, volume in values.items():
        out[f"volume_{key}"] = volume
        out[f"presence_{key}"] = (volume > 0).astype(float).where(volume.notna())
    # Binary territory totals can be known positive even when the volume sum is unknown.
    for key, components in [("carotid_total", sides[:2]), ("femoral_total", sides[2:]), ("peripheral_total", sides)]:
        measured = pd.concat(components, axis=1)
        any_positive = measured.gt(0).any(axis=1)
        complete = measured.notna().all(axis=1)
        derived = any_positive.astype(float).where(any_positive | complete)
        out[f"presence_{key}"] = out[f"presence_{key}"].fillna(derived)
    return out, audit


def load_splits(metadata: Path, train_path: Path, test_path: Path, dataroot: Path,
                gender: str, seed: int = 42, val_fraction: float = .15,
                weight_col: str | None = None, age_col: str | None = None) -> tuple[dict, dict]:
    if not 0 < val_fraction < .5:
        raise ValueError("Validation fraction must lie between 0 and 0.5.")
    master = identified(_read_table(metadata), "master metadata")
    train = identified(_read_table(train_path), "outer training")
    test = identified(_read_table(test_path), "held-out test")
    if set(train.index) & set(test.index):
        raise ValueError("Outer training/test participant overlap.")
    with np.load(dataroot / "simplified_padded_tensors.npz", allow_pickle=True) as scans:
        scan_ids = [clean_stregkode(x) for x in scans["names"]]
    if None in scan_ids or len(scan_ids) != len(set(scan_ids)):
        raise ValueError("Scan archive must have one nonempty ID per participant.")
    available = set(scan_ids)
    # Small sidecar only: check scan features before reserving GPU time.
    with np.load(dataroot / "scan_geometry_features.npz", allow_pickle=True) as sidecar:
        side_ids = [clean_stregkode(x) for x in sidecar["names"]]
    if None in side_ids or len(set(side_ids)) != len(side_ids):
        raise ValueError("Geometry sidecar must contain unique nonempty participant IDs")
    prepared, audit = {}, {"gender": gender, "seed": seed, "splits": {}}
    for key, split in [("development", train), ("test", test)]:
        # Preserve the supplied membership; prefer master values, use split metadata as fallback.
        frame = master.reindex(split.index).combine_first(split).reindex(split.index).reset_index(drop=True)
        frame["participant_id"] = split.index.to_numpy()
        gcol = _resolve_column(frame, ["sexMF", "sex", "gender"])
        if gcol is None:
            raise KeyError("No sex/gender column in metadata.")
        sex = frame[gcol].astype(str).str.strip().str.strip('\"').str.strip("'").str.upper().replace({"FEMALE": "F", "MALE": "M"})
        age, acol = _numeric(frame, [age_col] if age_col else ["age", "alder", "age_years"])
        weight, wcol = _numeric(frame, [weight_col] if weight_col else WEIGHT_ALIASES)
        if acol is None or wcol is None:
            raise KeyError(f"Numeric age/weight columns required: age={acol}, weight={wcol}. Set age_col/weight_col in paper_analysis/next_steps_config.json; weight must be kg.")
        frame["study_age"], frame["study_weight_kg"], frame["study_gender"] = age, weight, sex
        frame, derivation = derive_targets(frame)
        keep = (sex == gender) & frame["participant_id"].isin(available) & np.isfinite(age) & (age >= 18) & np.isfinite(weight) & (weight > 0)
        prepared[key] = frame.loc[keep].sort_values("participant_id").reset_index(drop=True)
        if len(prepared[key]):
            load_aligned_scan_geometry(dataroot, "scan_geometry_features.npz",
                                       prepared[key].participant_id.to_numpy(), require_complete=True)
        audit["splits"][key] = {"input": len(frame), "eligible": int(keep.sum()), "excluded": int((~keep).sum()),
                               "age_column": acol, "weight_column": wcol, "target_derivation": derivation}
    dev = prepared.pop("development")
    rng = np.random.default_rng(seed)
    val_indices = []
    # Stable within sex and shared across all targets/ablations, stratified on known peripheral presence.
    labels = dev["presence_peripheral_total"].fillna(-1)
    for value in sorted(labels.unique()):
        indices = np.flatnonzero(labels.to_numpy() == value)
        rng.shuffle(indices)
        nval = min(max(1, round(len(indices) * val_fraction)), max(0, len(indices) - 1))
        val_indices.extend(indices[:nval])
    val_mask = np.zeros(len(dev), dtype=bool)
    val_mask[val_indices] = True
    prepared["train"] = dev.loc[~val_mask].reset_index(drop=True)
    prepared["validation"] = dev.loc[val_mask].reset_index(drop=True)
    audit["internal_split"] = {"train": len(prepared["train"]), "validation": len(prepared["validation"]), "fraction": val_fraction}
    return prepared, audit


def select_cohort(splits: dict, target: str, policy: str, age_top_fraction: float = .2,
                  age_band_width: int = 10) -> tuple[dict, dict]:
    if policy not in {"all", "positive", "top5", "age_top"}:
        raise ValueError(f"Unknown cohort {policy}")
    if not 0 < age_top_fraction < 1 or age_band_width < 1:
        raise ValueError("Invalid age-tail fraction or band width")
    available = {k: v.loc[np.isfinite(v[target])].copy() for k, v in splits.items()}
    fit = available["train"]
    rule = {"policy": policy, "fitted_on": "training only", "target": target}
    if policy == "top5":
        rule["threshold"] = float(fit[target].quantile(.95)) if len(fit) else None
        rule["positive_required"] = True
    elif policy == "age_top":
        bands = np.floor(fit.study_age / age_band_width).astype(int) * age_band_width
        rule.update({"age_band_width": age_band_width, "top_fraction": age_top_fraction,
                     "thresholds": {str(int(b)): float(g[target].quantile(1-age_top_fraction))
                                    for b, g in fit.groupby(bands)}, "unseen_age_band": "excluded"})
    selected = {}
    rule["counts"] = {}
    for key, frame in available.items():
        if policy == "positive":
            keep = frame[target] > 0
        elif policy == "top5":
            keep = (frame[target] > 0) & (frame[target] >= rule["threshold"]) if rule["threshold"] is not None else np.zeros(len(frame), bool)
        elif policy == "age_top":
            bands = (np.floor(frame.study_age / age_band_width).astype(int) * age_band_width).astype(str)
            cutoff = bands.map(rule["thresholds"])
            keep = cutoff.notna() & (frame[target] >= cutoff)
        else:
            keep = np.ones(len(frame), bool)
        selected[key] = frame.loc[keep].reset_index(drop=True)
        rule["counts"][key] = {"available": len(frame), "selected": len(selected[key]),
                               "zeros": int((selected[key][target] == 0).sum()),
                               "positive": int((selected[key][target] > 0).sum()),
                               "median": float(selected[key][target].median()) if len(selected[key]) else None}
    return selected, rule


def training_problem(splits: dict, target: str, kind: str, minimum_train: int = 50,
                     minimum_validation: int = 10) -> str | None:
    # Never decide whether to train from held-out outcomes.
    for key, minimum in [("train", minimum_train), ("validation", minimum_validation)]:
        values = splits[key][target]
        if len(values) < minimum:
            return f"{key}: {len(values)} participants, requires {minimum}"
        if values.nunique() < 2:
            return f"{key}: target has no variation"
        if kind == "binary" and (not values.isin([0, 1]).all() or min(int((values == 0).sum()), int((values == 1).sum())) < 2):
            return f"{key}: binary target requires at least two participants per class"
    return None
