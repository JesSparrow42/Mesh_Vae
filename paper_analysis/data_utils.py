"""Shared, deliberately conservative data-loading helpers.

The original project receives metadata from several exports with slightly
different column names.  These helpers resolve known aliases while refusing
to silently invent missing variables.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ID_ALIASES = (
    "participant_id",
    "participantid",
    "stregkode",
    "subject_id",
    "subjectid",
    "scan_id",
)
AGE_ALIASES = ("age", "alder", "age_years")
BMI_ALIASES = ("body_mass_index", "bmi", "bodymassindex")
GENDER_ALIASES = ("sexmf", "sex", "gender", "koen", "køn")

PLAQUE_COMPONENT_ALIASES = (
    ("plaque_volume_right_carotid_corrected", "plaque_volume_right_carotid"),
    ("plaque_volume_left_carotid_corrected", "plaque_volume_left_carotid"),
    ("plaque_volume_right_femoral_corrected", "plaque_volume_right_femoral"),
    ("plaque_volume_left_femoral_corrected", "plaque_volume_left_femoral"),
    ("pv_coro", "ct_qa_cor_tot_plaq_vol", "coronary_plaque_volume"),
)


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [
        str(column).strip().lstrip("\ufeff").strip('"').strip("'")
        for column in out.columns
    ]
    return out


def detect_separator(path: str | Path) -> str:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        sample = handle.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        return "\t" if sample.count("\t") >= sample.count(",") else ","


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return clean_column_names(
        pd.read_csv(
            path,
            sep=detect_separator(path),
            dtype=str,
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
            engine="python",
        )
    )


def find_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    contains: str | None = None,
) -> str | None:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        match = lower.get(str(candidate).lower())
        if match is not None:
            return match
    if contains:
        token = contains.lower()
        matches = [str(column) for column in frame.columns if token in str(column).lower()]
        if len(matches) == 1:
            return matches[0]
    return None


_SCAN_CODE = re.compile(r"([A-Z]+DET\d{5,10})", re.IGNORECASE)


def clean_participant_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    match = _SCAN_CODE.search(text)
    if match:
        return match.group(1).upper()
    basename = Path(text.replace("\\", "/")).name
    stem = basename.split(".")[0].strip()
    return stem.upper() if stem else None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip("'")
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def bmi_plot_data(frame: pd.DataFrame, bmi_column: str | None = None) -> pd.DataFrame:
    """Keep source BMI and independently check it only against unit-bearing columns."""
    if bmi_column and bmi_column not in frame:
        raise KeyError(f"BMI column {bmi_column!r} is missing.")
    source_column = bmi_column or find_column(frame, BMI_ALIASES)
    weight_column = find_column(frame, ("weight_kg", "body_weight_kg", "study_weight_kg"))
    height_column = find_column(frame, ("height_m", "body_height_m", "height_cm", "body_height_cm", "body_height_cm_3d"))
    result = pd.DataFrame(index=frame.index)
    result["bmi"] = numeric(frame[source_column]) if source_column else np.nan
    result["bmi_source_column"] = source_column or ""
    result["bmi_source_value"] = frame[source_column].astype(str) if source_column else ""
    result["bmi_weight_column"] = weight_column or ""
    result["bmi_height_column"] = height_column or ""
    result["bmi_weight_kg"] = numeric(frame[weight_column]) if weight_column else np.nan
    result["bmi_height_m"] = numeric(frame[height_column]) if height_column else np.nan
    if height_column and "_cm" in height_column.lower():
        result["bmi_height_m"] /= 100.0
    valid_components = (
        np.isfinite(result["bmi_weight_kg"]) & result["bmi_weight_kg"].gt(0)
        & np.isfinite(result["bmi_height_m"]) & result["bmi_height_m"].gt(0)
    )
    result["bmi_computed_check"] = (
        result["bmi_weight_kg"] / result["bmi_height_m"].pow(2)
    ).where(valid_components)
    result["bmi_difference"] = result["bmi"] - result["bmi_computed_check"]
    result["bmi_check_status"] = np.select(
        [result["bmi_computed_check"].isna(), ~np.isfinite(result["bmi"]),
         result["bmi_difference"].abs().gt(1.0)],
        ["unit-bearing components unavailable", "source BMI unavailable", "difference above 1 kg/m2"],
        default="within 1 kg/m2",
    )
    return result


def binary(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.strip('"').str.strip("'").str.lower()
    mapped = cleaned.map(
        {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "case": 1.0,
            "control": 0.0,
            "positive": 1.0,
            "negative": 0.0,
        }
    )
    parsed = numeric(series)
    return parsed.where(parsed.notna(), mapped)


def normalize_gender(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip("'")
        .str.upper()
        .replace(
            {
                "FEMALE": "F",
                "KVINDE": "F",
                "WOMAN": "F",
                "2": "F",
                "MALE": "M",
                "MAND": "M",
                "MAN": "M",
                "1": "M",
            }
        )
    )


def resolve_id_column(frame: pd.DataFrame, explicit: str | None = None) -> str:
    column = explicit if explicit in frame.columns else find_column(frame, ID_ALIASES)
    if column is None:
        raise KeyError(f"Could not find participant ID column. Available columns: {list(frame.columns)[:80]}")
    return str(column)


def resolve_numeric_column(
    frame: pd.DataFrame,
    explicit: str | None,
    aliases: Iterable[str],
    label: str,
) -> str:
    column = explicit if explicit in frame.columns else find_column(frame, aliases)
    if column is None:
        raise KeyError(f"Could not find {label} column. Available columns: {list(frame.columns)[:80]}")
    return str(column)


def derive_any_plaque(frame: pd.DataFrame) -> pd.Series:
    direct = find_column(frame, ("benchmark_anyplaq", "anyplaq", "any_plaq", "any_plaque"))
    if direct is not None:
        direct_values = binary(frame[direct])
    else:
        direct_values = pd.Series(np.nan, index=frame.index, dtype=float)

    component_values: list[pd.Series] = []
    for aliases in PLAQUE_COMPONENT_ALIASES:
        column = find_column(frame, aliases)
        if column is not None:
            component_values.append(numeric(frame[column]))
    if not component_values:
        if direct_values.notna().any():
            return direct_values
        raise KeyError(
            "Could not find any-plaque target or plaque-volume components. "
            f"Available columns: {list(frame.columns)[:100]}"
        )

    components = pd.concat(component_values, axis=1)
    derived = (components.max(axis=1, skipna=True) > 0.0).astype(float)
    derived[components.notna().sum(axis=1) == 0] = np.nan
    return direct_values.where(direct_values.notna(), derived)


def resolve_binary_target(frame: pd.DataFrame, target_column: str) -> pd.Series:
    if target_column.lower() in {"anyplaq", "any_plaque", "benchmark_anyplaq"}:
        return derive_any_plaque(frame)
    column = target_column if target_column in frame.columns else find_column(frame, (target_column,))
    if column is None:
        raise KeyError(f"Target {target_column!r} was not found. Available: {list(frame.columns)[:100]}")
    return binary(frame[column])


def latent_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("latent_", "mu_", "z_")
    columns = [str(column) for column in frame.columns if str(column).lower().startswith(prefixes)]
    if not columns:
        excluded = {
            "participant_id",
            "split",
            "target",
            "probability",
            "prediction",
            "age",
            "bmi",
            "gender",
            "tsne_1",
            "tsne_2",
        }
        columns = [
            str(column)
            for column in frame.columns
            if str(column).lower() not in excluded
            and pd.to_numeric(frame[column], errors="coerce").notna().mean() > 0.95
        ]
    if not columns:
        raise KeyError("No latent columns were found (expected names such as latent_000).")
    return sorted(columns)

