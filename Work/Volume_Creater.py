from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import trimesh
from matplotlib.path import Path as MplPath

try:
    from scipy import ndimage as ndi
except ImportError as exc:
    raise ImportError(
        "This script requires scipy for connected-components and morphology. "
        "Install it with: pip install scipy"
    ) from exc


COLUMN_ALIASES = {
    "subject_id": ["stregkode", "stregcode", "record_id", "subject_id"],

    "body_height_cm": [
        "measure_Body height (cm)",
        "measure_Body_height (cm)",
        "Body height (cm)",
        "height",
    ],
    "neck_height_cm": [
        "measure_Neck height front (cm)",
        "measure_Neck height (cm)",
        "Neck height front (cm)",
    ],
    "neck_height_front_cm": [
        "measure_Neck height front (cm)",
        "measure_Neck height (cm)",
        "Neck height front (cm)",
    ],
    "waist_height_cm": [
        "measure_Waist height (cm)",
        "Waist height (cm)",
    ],
    "buttock_height_cm": [
        "measure_Buttock height (cm)",
        "Buttock height (cm)",
    ],
    "hip_height_cm": [
        "measure_Hip height (cm)",
        "Hip height (cm)",
    ],
    "crotch_height_cm": [
        "measure_Crotch height (cm)",
        "Crotch height (cm)",
    ],
    "breast_height_cm": [
        "measure_Breast height (cm)",
        "Breast height (cm)",
    ],
    "shoulder_blades_height_cm": [
        "measure_Height of shoulder blades (cm)",
        "Height of shoulder blades (cm)",
    ],

    "left_shoulder_width_cm": [
        "measure_Shoulder width left (cm)",
        "measure_Shoulder breadth left (cm)",
    ],
    "right_shoulder_width_cm": [
        "measure_Shoulder width right (cm)",
        "measure_Shoulder breadth right (cm)",
    ],
    "cross_shoulder_cm": [
        "measure_Cross shoulder over neck (cm)",
        "measure_Distance shoulder blades (cm)",
        "cross_shoulder_over_neck",
    ],

    "mid_neck_girth_cm": [
        "measure_Mid neck girth (cm)",
        "Mid neck girth (cm)",
    ],
    "bust_girth_cm": [
        "measure_Bust/chest girth (cm)",
        "Bust/chest girth (cm)",
    ],
    "underbust_circumference_cm": [
        "measure_Underbust circumference (horizontal) (cm)",
        "Underbust circumference (horizontal) (cm)",
    ],
    "waist_girth_cm": [
        "measure_Waist girth (cm)",
        "Waist girth (cm)",
    ],
    "high_waist_girth_cm": [
        "measure_High waist girth (cm)",
        "High waist girth (cm)",
    ],
    "high_hip_girth_cm": [
        "measure_High hip girth (cm)",
        "High hip girth (cm)",
    ],
    "buttock_girth_cm": [
        "measure_Buttock girth (cm)",
        "Buttock girth (cm)",
    ],
    "hip_girth_cm": [
        "measure_Hip girth (cm)",
        "Hip girth (cm)",
    ],
    "hip_thigh_girth_cm": [
        "measure_Hip/thigh girth (cm)",
        "Hip/thigh girth (cm)",
    ],

    "distance_breast_to_vertical_cm": [
        "measure_Distance breast to vertical (cm)",
        "Distance breast to vertical (cm)",
    ],
    "distance_back_breast_to_vertical_cm": [
        "measure_Distance back in breast height to vertical (cm)",
        "Distance back in breast height to vertical (cm)",
    ],
    "distance_belly_to_vertical_cm": [
        "measure_Distance belly to vertical (cm)",
        "Distance belly to vertical (cm)",
    ],
    "distance_back_belly_to_vertical_cm": [
        "measure_Distance back in belly height to vertical (cm)",
        "Distance back in belly height to vertical (cm)",
    ],
    "distance_abdomen_to_vertical_cm": [
        "measure_Distance abdomen to vertical (cm)",
        "Distance abdomen to vertical (cm)",
    ],
    "distance_buttock_to_vertical_cm": [
        "measure_Distance buttock to vertical (cm)",
        "Distance buttock to vertical (cm)",
    ],

    "arm_length_left_cm": [
        "measure_Arm length left (cm)",
        "Arm length left (cm)",
    ],
    "arm_length_right_cm": [
        "measure_Arm length right (cm)",
        "Arm length right (cm)",
    ],
    "arm_length_to_neck_back_left_cm": [
        "measure_Arm length to neck back left (cm)",
        "Arm length to neck back left (cm)",
    ],
    "arm_length_to_neck_back_right_cm": [
        "measure_Arm length to neck back right (cm)",
        "Arm length to neck back right (cm)",
    ],
    "wrist_girth_cm": [
        "measure_Wrist girth (cm)",
        "Wrist girth (cm)",
    ],

    "inseam_left_cm": [
        "measure_Inseam left (cm)",
        "Inseam left (cm)",
    ],
    "inseam_right_cm": [
        "measure_Inseam right (cm)",
        "Inseam right (cm)",
    ],
    "thigh_girth_left_cm": [
        "measure_Thigh girth left (horizontal) (cm)",
        "Thigh girth left (horizontal) (cm)",
    ],
    "thigh_girth_right_cm": [
        "measure_Thigh girth right (horizontal) (cm)",
        "Thigh girth right (horizontal) (cm)",
    ],
}

OBJ_PATH_ALIASES = ["obj_path", "mesh_path"]


@dataclass
class BodyPriors:
    crotch_z: float
    waist_z: float
    breast_z: float
    shoulder_z: float
    neck_base_z: float
    center_x: float
    torso_profile_z: np.ndarray
    torso_profile_half_width: np.ndarray
    min_leg_component_area_cm2: float
    bust_width_cm: float
    waist_width_cm: float
    hip_width_cm: float
    shoulder_core_width_cm: float


def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"{minutes}m {remainder:.1f}s"


def clean_string(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    return text


def parse_float(value: object, default: float = np.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        value_float = float(value)
        if np.isnan(value_float):
            return default
        return value_float
    text = clean_string(value)
    if not text:
        return default
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return default


def get_value(row: pd.Series, keys: Iterable[str], default: float = np.nan) -> float:
    for key in keys:
        if key in row.index and pd.notna(row[key]):
            value = parse_float(row[key], default=np.nan)
            if not np.isnan(value):
                return value
    return default


def get_measure(row: pd.Series, logical_name: str, default: float = np.nan) -> float:
    aliases = COLUMN_ALIASES.get(logical_name)
    if aliases is None:
        return default
    return get_value(row, aliases, default=default)


def first_present(columns: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    column_set = set(columns)
    for alias in aliases:
        if alias in column_set:
            return alias
    return None


def get_subject_id(row: pd.Series) -> str:
    for key in COLUMN_ALIASES["subject_id"]:
        if key in row.index and pd.notna(row[key]):
            text = clean_string(row[key])
            if text:
                return text
    return ""


def read_table(path: Path, sep: Optional[str] = None) -> pd.DataFrame:
    resolved_sep = sep
    if resolved_sep is None:
        resolved_sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=resolved_sep, low_memory=False)


def required_logical_columns_present(df: pd.DataFrame) -> dict[str, bool]:
    return {
        logical: first_present(df.columns, aliases) is not None
        for logical, aliases in COLUMN_ALIASES.items()
    }


def merge_lookup(primary_df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    primary_join = first_present(primary_df.columns, COLUMN_ALIASES["subject_id"])
    lookup_join = first_present(lookup_df.columns, COLUMN_ALIASES["subject_id"])
    if primary_join is None or lookup_join is None:
        raise ValueError("Could not find a subject id column for merging.")

    lookup = lookup_df.copy()
    if lookup_join != primary_join:
        lookup = lookup.rename(columns={lookup_join: primary_join})

    lookup = lookup.drop_duplicates(subset=[primary_join], keep="first")
    merged = primary_df.merge(lookup, on=primary_join, how="left", suffixes=("", "__lookup"))

    for column in list(lookup.columns):
        if column == primary_join:
            continue
        lookup_column = f"{column}__lookup"
        if column in primary_df.columns and lookup_column in merged.columns:
            left = merged[column]
            right = merged[lookup_column]
            left_missing = left.isna() | (left.astype(str).str.strip() == "")
            merged.loc[left_missing, column] = right[left_missing]
            merged = merged.drop(columns=[lookup_column])

    return merged


def resolve_mesh_path(row: pd.Series, data_dir: Optional[Path]) -> Path:
    for key in OBJ_PATH_ALIASES:
        if key in row.index and pd.notna(row[key]):
            raw = clean_string(row[key])
            if raw:
                candidate = Path(raw)
                if candidate.exists():
                    return candidate
                if data_dir is not None:
                    joined = data_dir / candidate.name
                    if joined.exists():
                        return joined

    subject_id = get_subject_id(row)
    if not subject_id:
        raise ValueError("Missing subject id")
    if data_dir is None:
        raise ValueError("No data directory provided and obj_path column not found")

    candidates = [
        data_dir / f"{subject_id}.obj",
        data_dir / f"!{subject_id}!.obj",
    ]

    if subject_id.startswith("!") and subject_id.endswith("!"):
        candidates.append(data_dir / f"{subject_id}.obj")
    if subject_id.startswith("!") and subject_id.endswith("!.obj"):
        candidates.append(data_dir / subject_id)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def load_mesh(obj_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(obj_path, force="mesh", process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        geometries = [
            geometry
            for geometry in mesh.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not geometries:
            raise ValueError(f"No mesh geometry inside scene: {obj_path}")
        mesh = trimesh.util.concatenate(geometries)

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh)!r}")

    return mesh


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()

    try:
        mesh.remove_infinite_values()
    except Exception:
        pass
    try:
        mesh.update_faces(mesh.unique_faces())
    except Exception:
        pass
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    try:
        mesh.process(validate=True)
    except Exception:
        pass
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    try:
        mesh.fill_holes()
    except Exception:
        pass

    return mesh


def orient_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, float]]:
    """
    Make the longest mesh extent vertical (z) and make the wider horizontal
    torso extent left-right (x). This is only an axis permutation, not a full PCA.
    """
    mesh = mesh.copy()
    original_extents = mesh.bounds[1] - mesh.bounds[0]

    vertical_axis_source = int(np.argmax(original_extents))
    if vertical_axis_source != 2:
        remaining = [axis for axis in [0, 1, 2] if axis != vertical_axis_source]
        order = [remaining[0], remaining[1], vertical_axis_source]
        mesh.vertices = mesh.vertices[:, order]
        try:
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass

    vertices = mesh.vertices
    z = vertices[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())
    height = z_max - z_min

    torso_band = vertices[(z >= z_min + 0.35 * height) & (z <= z_min + 0.85 * height)]
    if len(torso_band) < 100:
        torso_band = vertices

    span_x = float(torso_band[:, 0].max() - torso_band[:, 0].min())
    span_y = float(torso_band[:, 1].max() - torso_band[:, 1].min())

    if span_y > span_x:
        mesh.vertices = mesh.vertices[:, [1, 0, 2]]
        try:
            trimesh.repair.fix_normals(mesh)
        except Exception:
            pass

    info = {
        "vertical_axis_source": float(vertical_axis_source),
        "torso_span_x_raw": span_x,
        "torso_span_y_raw": span_y,
    }
    return mesh, info


def normalize_mesh_to_cm(mesh: trimesh.Trimesh, body_height_cm: float) -> tuple[trimesh.Trimesh, float]:
    mesh = mesh.copy()
    z_min, z_max = mesh.bounds[:, 2]
    current_height = float(z_max - z_min)
    if current_height <= 0:
        raise ValueError("Mesh has zero height after orientation")

    scale = body_height_cm / current_height
    mesh.apply_scale(scale)

    z_min_after = float(mesh.bounds[0, 2])
    mesh.apply_translation([0.0, 0.0, -z_min_after])
    return mesh, scale


def mean_finite(*values: float) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def sum_finite(a: float, b: float) -> float:
    if np.isfinite(a) and np.isfinite(b):
        return float(a + b)
    return np.nan


def pick_first_finite(*values: float, default: float = np.nan) -> float:
    for value in values:
        if np.isfinite(value):
            return float(value)
    return default


def circle_area_from_girth(girth_cm: float) -> float:
    if not np.isfinite(girth_cm) or girth_cm <= 0:
        return np.nan
    return (girth_cm ** 2) / (4.0 * math.pi)


def stadium_width_from_girth_and_depth(girth_cm: float, depth_cm: float) -> float:
    """
    Approximate full left-right width from circumference and AP depth.
    Uses a stadium cross-section:
        C ~= 2*W + (pi - 2)*D
        => W ~= (C - (pi - 2)*D) / 2
    """
    if np.isfinite(girth_cm) and girth_cm > 0 and np.isfinite(depth_cm) and depth_cm > 0:
        width = (girth_cm - (math.pi - 2.0) * depth_cm) / 2.0
        return max(width, 0.6 * depth_cm)

    if np.isfinite(girth_cm) and girth_cm > 0:
        return girth_cm / math.pi

    return np.nan


def validate_heights_v2(
    body: float,
    crotch: float,
    waist: float,
    breast: float,
    neck_front: float,
) -> None:
    required = [body, crotch, waist, breast, neck_front]
    if not all(np.isfinite(v) for v in required):
        raise ValueError("Missing one or more required height landmarks")

    if not (0.0 < crotch < waist < breast < neck_front < body + 5.0):
        raise ValueError(
            f"Unexpected height ordering: "
            f"crotch={crotch}, waist={waist}, breast={breast}, neck_front={neck_front}, body={body}"
        )


def estimate_mesh_width_at_height(
    vertices: np.ndarray,
    z_target_cm: float,
    half_window_cm: float = 1.5,
    min_points: int = 100,
) -> float:
    if not np.isfinite(z_target_cm):
        return np.nan

    z = vertices[:, 2]
    window = half_window_cm
    for _ in range(4):
        band = vertices[(z >= z_target_cm - window) & (z <= z_target_cm + window)]
        if len(band) >= min_points:
            return float(band[:, 0].max() - band[:, 0].min())
        window *= 1.75

    return np.nan


def estimate_center_x(vertices: np.ndarray, anchor_height_cm: float) -> float:
    z = vertices[:, 2]

    if np.isfinite(anchor_height_cm):
        band = vertices[(z >= anchor_height_cm - 5.0) & (z <= anchor_height_cm + 5.0)]
        if len(band) >= 50:
            return float(np.median(band[:, 0]))

    mid_z = 0.45 * float(z.max())
    band = vertices[(z >= mid_z - 10.0) & (z <= mid_z + 10.0)]
    if len(band) < 50:
        band = vertices

    return float(np.median(band[:, 0]))


def rasterize_section_mask(path2d, planar_pitch_cm: float):
    if path2d is None:
        return None, None, None

    try:
        discrete = path2d.discrete
    except Exception:
        return None, None, None

    if discrete is None or len(discrete) == 0:
        return None, None, None

    loops = []
    for curve in discrete:
        arr = np.asarray(curve, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
            continue

        if np.linalg.norm(arr[0] - arr[-1]) > max(1e-9, 0.5 * planar_pitch_cm):
            arr = np.vstack([arr, arr[0]])

        if np.ptp(arr[:, 0]) < 1e-9 and np.ptp(arr[:, 1]) < 1e-9:
            continue

        loops.append(arr)

    if not loops:
        return None, None, None

    all_points = np.vstack(loops)
    bounds = np.array([all_points.min(axis=0), all_points.max(axis=0)], dtype=float)

    origin = bounds[0] - 2.0 * planar_pitch_cm
    span = np.maximum(bounds[1] - origin, planar_pitch_cm)
    resolution = np.ceil(span / planar_pitch_cm).astype(int) + 2
    resolution = np.maximum(resolution, 8)

    nx = int(resolution[0])
    ny = int(resolution[1])

    x_centers = origin[0] + (np.arange(nx) + 0.5) * planar_pitch_cm
    y_centers = origin[1] + (np.arange(ny) + 0.5) * planar_pitch_cm

    filled = np.zeros((ny, nx), dtype=bool)

    for loop in loops:
        min_x = loop[:, 0].min() - planar_pitch_cm
        max_x = loop[:, 0].max() + planar_pitch_cm
        min_y = loop[:, 1].min() - planar_pitch_cm
        max_y = loop[:, 1].max() + planar_pitch_cm

        col_idx = np.where((x_centers >= min_x) & (x_centers <= max_x))[0]
        row_idx = np.where((y_centers >= min_y) & (y_centers <= max_y))[0]

        if len(col_idx) == 0 or len(row_idx) == 0:
            continue

        xx, yy = np.meshgrid(x_centers[col_idx], y_centers[row_idx])
        pts = np.column_stack([xx.ravel(), yy.ravel()])

        poly = MplPath(loop, closed=True)
        inside = poly.contains_points(pts, radius=1e-12).reshape(len(row_idx), len(col_idx))

        filled[np.ix_(row_idx, col_idx)] ^= inside

    if not filled.any():
        return None, None, None

    return filled, x_centers, y_centers


def connected_component_stats(mask: np.ndarray, x_centers: np.ndarray, planar_pitch_cm: float):
    labels, n = ndi.label(mask)
    pixel_area = planar_pitch_cm ** 2
    comps = []

    for lab in range(1, n + 1):
        comp = labels == lab
        pix = int(comp.sum())
        if pix == 0:
            continue
        rr, cc = np.where(comp)
        comps.append({
            "label": lab,
            "pixels": pix,
            "area_cm2": float(pix * pixel_area),
            "cx": float(x_centers[cc].mean()),
            "xmin": float(x_centers[cc].min()),
            "xmax": float(x_centers[cc].max()),
        })

    return labels, comps


def refine_neck_base_height(
    mesh: trimesh.Trimesh,
    row: pd.Series,
    planar_pitch_cm: float,
    search_half_window_cm: float = 4.0,
    step_cm: float = 0.5,
) -> float:
    approx = pick_first_finite(
        get_measure(row, "neck_height_front_cm"),
        get_measure(row, "neck_height_cm"),
    )
    shoulder_z = get_measure(row, "shoulder_blades_height_cm")
    neck_girth = get_measure(row, "mid_neck_girth_cm")

    if not np.isfinite(approx):
        raise ValueError("Missing neck height")

    target_area = circle_area_from_girth(neck_girth)

    z0 = max(0.0, approx - search_half_window_cm)
    z1 = approx + search_half_window_cm

    if np.isfinite(shoulder_z):
        z0 = max(z0, shoulder_z - 1.0)

    zs = np.arange(z0, z1 + 1e-9, step_cm)
    if len(zs) == 0:
        return float(approx)

    sections = mesh.section_multiplane(
        plane_origin=[0.0, 0.0, 0.0],
        plane_normal=[0.0, 0.0, 1.0],
        heights=zs,
    )

    areas = []
    for path in sections:
        if path is None:
            areas.append(np.inf)
            continue
        mask, _, _ = rasterize_section_mask(path, planar_pitch_cm)
        if mask is None or not mask.any():
            areas.append(np.inf)
        else:
            areas.append(float(mask.sum()) * planar_pitch_cm ** 2)

    areas = np.asarray(areas, dtype=float)
    smooth = areas.copy()

    for i in range(1, len(areas) - 1):
        neighborhood = areas[i - 1:i + 2]
        if np.isfinite(neighborhood).all():
            smooth[i] = float(np.mean(neighborhood))

    if np.isfinite(target_area):
        score = np.where(np.isfinite(smooth), np.abs(smooth - target_area), np.inf)
    else:
        score = np.where(np.isfinite(smooth), smooth, np.inf)

    best_idx = int(np.argmin(score))
    if not np.isfinite(score[best_idx]):
        return float(approx)

    return float(zs[best_idx])


def refine_crotch_height(
    mesh: trimesh.Trimesh,
    row: pd.Series,
    planar_pitch_cm: float,
    center_x: float,
    search_half_window_cm: float = 4.0,
    step_cm: float = 0.5,
) -> float:
    approx = pick_first_finite(
        get_measure(row, "crotch_height_cm"),
        mean_finite(get_measure(row, "inseam_left_cm"), get_measure(row, "inseam_right_cm")),
        default=np.nan,
    )
    if not np.isfinite(approx):
        return np.nan

    waist_z = get_measure(row, "waist_height_cm")
    hip_z = get_measure(row, "hip_height_cm")

    z0 = max(0.0, approx - search_half_window_cm)
    z1 = approx + search_half_window_cm

    if np.isfinite(hip_z):
        z1 = min(z1, hip_z + 6.0)
    if np.isfinite(waist_z):
        z1 = min(z1, waist_z - 1.0)

    zs = np.arange(z0, z1 + 1e-9, step_cm)
    if len(zs) == 0:
        return float(approx)

    sections = mesh.section_multiplane(
        plane_origin=[0.0, 0.0, 0.0],
        plane_normal=[0.0, 0.0, 1.0],
        heights=zs,
    )

    best_score = -np.inf
    best_z = float(approx)

    for z_val, path in zip(zs, sections):
        if path is None:
            continue

        mask, x_centers, _ = rasterize_section_mask(path, planar_pitch_cm)
        if mask is None or not mask.any():
            continue

        _, comps = connected_component_stats(mask, x_centers, planar_pitch_cm)
        if not comps:
            continue

        left_candidates = [c for c in comps if c["cx"] < center_x]
        right_candidates = [c for c in comps if c["cx"] >= center_x]

        left_area = max((c["area_cm2"] for c in left_candidates), default=0.0)
        right_area = max((c["area_cm2"] for c in right_candidates), default=0.0)

        center_band = np.abs(x_centers - center_x) <= 2.0
        center_area = float(mask[:, center_band].sum()) * (planar_pitch_cm ** 2) if center_band.any() else 0.0

        if left_area <= 0.0 or right_area <= 0.0:
            continue

        score = min(left_area, right_area) - 0.75 * center_area + 0.10 * (left_area + right_area)
        if score > best_score:
            best_score = score
            best_z = float(z_val)

    return best_z


def build_body_priors(
    row: pd.Series,
    mesh: trimesh.Trimesh,
    planar_pitch_cm: float,
) -> BodyPriors:
    vertices = mesh.vertices

    body_height = get_measure(row, "body_height_cm")
    waist_z = get_measure(row, "waist_height_cm")
    breast_z = get_measure(row, "breast_height_cm")
    shoulder_z = get_measure(row, "shoulder_blades_height_cm")
    neck_front_z = get_measure(row, "neck_height_front_cm")
    neck_z = get_measure(row, "neck_height_cm")
    hip_z = get_measure(row, "hip_height_cm")
    buttock_z = get_measure(row, "buttock_height_cm")

    crotch_measured = pick_first_finite(
        get_measure(row, "crotch_height_cm"),
        mean_finite(get_measure(row, "inseam_left_cm"), get_measure(row, "inseam_right_cm")),
        default=np.nan,
    )

    validate_heights_v2(
        body=body_height,
        crotch=crotch_measured,
        waist=waist_z,
        breast=breast_z,
        neck_front=pick_first_finite(neck_front_z, neck_z),
    )

    center_anchor = pick_first_finite(hip_z, buttock_z, crotch_measured, waist_z, default=np.nan)
    center_x = estimate_center_x(vertices, center_anchor)

    crotch_z = refine_crotch_height(
        mesh=mesh,
        row=row,
        planar_pitch_cm=planar_pitch_cm,
        center_x=center_x,
    )
    if not np.isfinite(crotch_z):
        crotch_z = float(crotch_measured)

    # AP depths
    bust_depth = sum_finite(
        get_measure(row, "distance_breast_to_vertical_cm"),
        get_measure(row, "distance_back_breast_to_vertical_cm"),
    )
    waist_depth = pick_first_finite(
        sum_finite(
            get_measure(row, "distance_belly_to_vertical_cm"),
            get_measure(row, "distance_back_belly_to_vertical_cm"),
        ),
        sum_finite(
            get_measure(row, "distance_abdomen_to_vertical_cm"),
            get_measure(row, "distance_back_belly_to_vertical_cm"),
        ),
        default=np.nan,
    )
    hip_depth = pick_first_finite(
        sum_finite(
            get_measure(row, "distance_abdomen_to_vertical_cm"),
            get_measure(row, "distance_buttock_to_vertical_cm"),
        ),
        waist_depth,
        default=np.nan,
    )

    # Torso widths from measurements, with mesh fallbacks
    bust_width = pick_first_finite(
        stadium_width_from_girth_and_depth(
            pick_first_finite(
                get_measure(row, "bust_girth_cm"),
                get_measure(row, "underbust_circumference_cm"),
                default=np.nan,
            ),
            bust_depth,
        ),
        estimate_mesh_width_at_height(vertices, breast_z),
        default=np.nan,
    )

    waist_width = pick_first_finite(
        stadium_width_from_girth_and_depth(
            pick_first_finite(
                get_measure(row, "waist_girth_cm"),
                get_measure(row, "high_waist_girth_cm"),
                default=np.nan,
            ),
            waist_depth,
        ),
        estimate_mesh_width_at_height(vertices, waist_z),
        default=np.nan,
    )

    hip_width = pick_first_finite(
        stadium_width_from_girth_and_depth(
            pick_first_finite(
                get_measure(row, "hip_girth_cm"),
                get_measure(row, "buttock_girth_cm"),
                get_measure(row, "high_hip_girth_cm"),
                get_measure(row, "hip_thigh_girth_cm"),
                default=np.nan,
            ),
            hip_depth,
        ),
        estimate_mesh_width_at_height(vertices, pick_first_finite(hip_z, buttock_z, crotch_z)),
        default=np.nan,
    )

    shoulder_span = pick_first_finite(
        get_measure(row, "cross_shoulder_cm"),
        sum_finite(
            get_measure(row, "left_shoulder_width_cm"),
            get_measure(row, "right_shoulder_width_cm"),
        ),
        default=np.nan,
    )

    shoulder_core_width = pick_first_finite(
        0.72 * shoulder_span if np.isfinite(shoulder_span) else np.nan,
        0.82 * estimate_mesh_width_at_height(vertices, shoulder_z),
        default=np.nan,
    )

    z_points = []
    half_widths = []

    def add_point(z_val: float, full_width: float) -> None:
        if np.isfinite(z_val) and np.isfinite(full_width) and full_width > 0:
            z_points.append(float(z_val))
            half_widths.append(0.5 * float(full_width))

    add_point(crotch_z + 2.0, hip_width)
    add_point(hip_z, hip_width)
    add_point(waist_z, waist_width)
    add_point(breast_z, bust_width)
    add_point(shoulder_z, shoulder_core_width)

    if len(z_points) < 2:
        raise ValueError("Not enough torso width priors to split torso/arms reliably")

    order = np.argsort(z_points)
    z_points = np.asarray(z_points, dtype=float)[order]
    half_widths = np.asarray(half_widths, dtype=float)[order]

    neck_base_z = refine_neck_base_height(
        mesh=mesh,
        row=row,
        planar_pitch_cm=planar_pitch_cm,
    )

    wrist_area = circle_area_from_girth(get_measure(row, "wrist_girth_cm"))
    thigh_area = mean_finite(
        circle_area_from_girth(get_measure(row, "thigh_girth_left_cm")),
        circle_area_from_girth(get_measure(row, "thigh_girth_right_cm")),
    )

    min_leg_component_area_cm2 = pick_first_finite(
        0.18 * thigh_area if np.isfinite(thigh_area) else np.nan,
        2.0 * wrist_area if np.isfinite(wrist_area) else np.nan,
        default=30.0,
    )

    return BodyPriors(
        crotch_z=float(crotch_z),
        waist_z=float(waist_z),
        breast_z=float(breast_z),
        shoulder_z=float(shoulder_z),
        neck_base_z=float(neck_base_z),
        center_x=float(center_x),
        torso_profile_z=z_points,
        torso_profile_half_width=half_widths,
        min_leg_component_area_cm2=float(min_leg_component_area_cm2),
        bust_width_cm=float(bust_width),
        waist_width_cm=float(waist_width),
        hip_width_cm=float(hip_width),
        shoulder_core_width_cm=float(shoulder_core_width),
    )


def torso_half_width_at_z(priors: BodyPriors, z_cm: float) -> float:
    return float(
        np.interp(
            z_cm,
            priors.torso_profile_z,
            priors.torso_profile_half_width,
            left=priors.torso_profile_half_width[0],
            right=priors.torso_profile_half_width[-1],
        )
    )

def neck_core_half_width_cm(row: pd.Series, mesh: trimesh.Trimesh, neck_base_z: float) -> float:
    neck_girth = get_measure(row, "mid_neck_girth_cm")
    neck_width_from_girth = neck_girth / math.pi if np.isfinite(neck_girth) and neck_girth > 0 else np.nan
    neck_width_from_mesh = estimate_mesh_width_at_height(mesh.vertices, neck_base_z, half_window_cm=1.0, min_points=60)

    width = pick_first_finite(
        0.90 * neck_width_from_girth if np.isfinite(neck_width_from_girth) else np.nan,
        0.55 * neck_width_from_mesh if np.isfinite(neck_width_from_mesh) else np.nan,
        default=12.0,
    )
    return max(4.5, 0.5 * float(width))


def classify_head_slice(
    mask: np.ndarray,
    x_centers: np.ndarray,
    planar_pitch_cm: float,
    center_x: float,
    neck_base_z: float,
    z_center: float,
    neck_half_width_cm: float,
    transition_height_cm: float = 5.0,
) -> tuple[float, float]:
    """
    Returns (head_area_cm2, upper_torso_area_cm2)

    In the first few cm above neck_base_z, only the central core is head.
    Lateral spill stays upper_torso. Above the transition zone, the whole slice is head.
    """
    pixel_area = planar_pitch_cm ** 2

    if z_center >= neck_base_z + transition_height_cm:
        total_area = float(mask.sum()) * pixel_area
        return total_area, 0.0

    t = max(0.0, min(1.0, (z_center - neck_base_z) / transition_height_cm))

    # Start narrow at neck, widen gradually into the lower skull
    head_half_width_cm = neck_half_width_cm + t * 3.0

    head_cols = np.abs(x_centers - center_x) <= head_half_width_cm

    head_area = float(mask[:, head_cols].sum()) * pixel_area if head_cols.any() else 0.0
    total_area = float(mask.sum()) * pixel_area
    torso_spill = max(0.0, total_area - head_area)

    return head_area, torso_spill

def classify_upper_slice(
    mask: np.ndarray,
    x_centers: np.ndarray,
    planar_pitch_cm: float,
    center_x: float,
    torso_half_width_cm: float,
) -> tuple[float, float, float]:
    """
    Returns (left_arm_area_cm2, torso_area_cm2, right_arm_area_cm2)
    """
    pixel_area = planar_pitch_cm ** 2

    erode_iters = max(1, int(round(1.0 / planar_pitch_cm)))
    eroded = ndi.binary_erosion(mask, iterations=erode_iters)
    work = eroded if eroded.any() else mask

    labels, n = ndi.label(work)
    if n == 0:
        total = float(mask.sum()) * pixel_area
        return 0.0, total, 0.0

    core_cols = np.abs(x_centers - center_x) <= (torso_half_width_cm + 1.0)

    best_lab = None
    best_score = -np.inf
    for lab in range(1, n + 1):
        comp = labels == lab
        overlap = float(comp[:, core_cols].sum()) if core_cols.any() else 0.0
        size = float(comp.sum())
        score = 5.0 * overlap + size
        if score > best_score:
            best_score = score
            best_lab = lab

    trunk = labels == best_lab
    rr, cc = np.where(trunk)

    if len(cc) == 0:
        total = float(mask.sum()) * pixel_area
        return 0.0, total, 0.0

    x0 = min(float(x_centers[cc].min()) - 1.0, center_x - 0.5 * torso_half_width_cm)
    x1 = max(float(x_centers[cc].max()) + 1.0, center_x + 0.5 * torso_half_width_cm)

    left_cols = x_centers < x0
    mid_cols = (x_centers >= x0) & (x_centers <= x1)
    right_cols = x_centers > x1

    left_area = float(mask[:, left_cols].sum()) * pixel_area if left_cols.any() else 0.0
    torso_area = float(mask[:, mid_cols].sum()) * pixel_area if mid_cols.any() else 0.0
    right_area = float(mask[:, right_cols].sum()) * pixel_area if right_cols.any() else 0.0

    return left_area, torso_area, right_area


def classify_lower_slice(
    mask: np.ndarray,
    x_centers: np.ndarray,
    planar_pitch_cm: float,
    center_x: float,
    min_leg_component_area_cm2: float,
) -> tuple[float, float]:
    """
    Returns (left_leg_area_cm2, right_leg_area_cm2)
    """
    pixel_area = planar_pitch_cm ** 2
    labels, n = ndi.label(mask)

    if n == 0:
        return 0.0, 0.0

    if n == 1:
        left_cols = x_centers < center_x
        right_cols = x_centers >= center_x
        left_area = float(mask[:, left_cols].sum()) * pixel_area if left_cols.any() else 0.0
        right_area = float(mask[:, right_cols].sum()) * pixel_area if right_cols.any() else 0.0
        return left_area, right_area

    comps = []
    for lab in range(1, n + 1):
        comp = labels == lab
        pix = int(comp.sum())
        if pix == 0:
            continue
        rr, cc = np.where(comp)
        area = pix * pixel_area
        cx = float(x_centers[cc].mean())
        comps.append((lab, area, cx))

    left_candidates = [
        (lab, area)
        for lab, area, cx in comps
        if cx < center_x and area >= min_leg_component_area_cm2
    ]
    right_candidates = [
        (lab, area)
        for lab, area, cx in comps
        if cx >= center_x and area >= min_leg_component_area_cm2
    ]

    if left_candidates:
        left_lab = max(left_candidates, key=lambda x: x[1])[0]
        left_area = float((labels == left_lab).sum()) * pixel_area
    else:
        left_cols = x_centers < center_x
        left_area = float(mask[:, left_cols].sum()) * pixel_area if left_cols.any() else 0.0

    if right_candidates:
        right_lab = max(right_candidates, key=lambda x: x[1])[0]
        right_area = float((labels == right_lab).sum()) * pixel_area
    else:
        right_cols = x_centers >= center_x
        right_area = float(mask[:, right_cols].sum()) * pixel_area if right_cols.any() else 0.0

    return left_area, right_area


def integrate_section_volumes(
    mesh: trimesh.Trimesh,
    row: pd.Series,
    slice_cm: float,
    planar_pitch_cm: float,
) -> tuple[dict[str, float], dict[str, float]]:
    body_height_cm = get_measure(row, "body_height_cm")
    if np.isnan(body_height_cm):
        raise ValueError("Missing body height")

    mesh = clean_mesh(mesh)
    mesh, orient_info = orient_mesh(mesh)
    mesh, scale = normalize_mesh_to_cm(mesh, body_height_cm)

    priors = build_body_priors(row=row, mesh=mesh, planar_pitch_cm=planar_pitch_cm)
    neck_half_width_cm = neck_core_half_width_cm(row=row, mesh=mesh, neck_base_z=priors.neck_base_z)    

    bounds = mesh.bounds
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]

    n_slices = max(1, int(math.ceil(body_height_cm / slice_cm)))
    z_centers = np.arange(n_slices, dtype=float) * slice_cm + 0.5 * slice_cm
    z_centers = z_centers[z_centers < body_height_cm]

    sections = mesh.section_multiplane(
        plane_origin=[0.0, 0.0, 0.0],
        plane_normal=[0.0, 0.0, 1.0],
        heights=z_centers,
    )

    volumes_cm3 = {
        "head": 0.0,
        "upper_torso": 0.0,
        "lower_torso": 0.0,
        "left_arm": 0.0,
        "right_arm": 0.0,
        "left_leg": 0.0,
        "right_leg": 0.0,
        "total": 0.0,
    }

    nonempty_slice_count = 0
    total_section_pixels = 0.0

    for z_center, path in zip(z_centers, sections):
        if path is None:
            continue

        mask, x_centers, y_centers = rasterize_section_mask(path, planar_pitch_cm)
        if mask is None or not mask.any():
            continue

        pixel_area = planar_pitch_cm ** 2
        total_area_cm2 = float(mask.sum()) * pixel_area
        if total_area_cm2 <= 0.0:
            continue

        nonempty_slice_count += 1
        total_section_pixels += float(mask.sum())
        volumes_cm3["total"] += total_area_cm2 * slice_cm

        if z_center >= priors.neck_base_z:
            head_area, shoulder_spill_area = classify_head_slice(
                mask=mask,
                x_centers=x_centers,
                planar_pitch_cm=planar_pitch_cm,
                center_x=priors.center_x,
                neck_base_z=priors.neck_base_z,
                z_center=z_center,
                neck_half_width_cm=neck_half_width_cm,
                transition_height_cm=5.0,
            )
            volumes_cm3["head"] += head_area * slice_cm
            volumes_cm3["upper_torso"] += shoulder_spill_area * slice_cm
            continue

        if z_center < priors.crotch_z:
            left_leg_area, right_leg_area = classify_lower_slice(
                mask=mask,
                x_centers=x_centers,
                planar_pitch_cm=planar_pitch_cm,
                center_x=priors.center_x,
                min_leg_component_area_cm2=priors.min_leg_component_area_cm2,
            )
            volumes_cm3["left_leg"] += left_leg_area * slice_cm
            volumes_cm3["right_leg"] += right_leg_area * slice_cm
            continue

        torso_half_width_cm = torso_half_width_at_z(priors, z_center)

        left_arm_area, torso_area, right_arm_area = classify_upper_slice(
            mask=mask,
            x_centers=x_centers,
            planar_pitch_cm=planar_pitch_cm,
            center_x=priors.center_x,
            torso_half_width_cm=torso_half_width_cm,
        )

        volumes_cm3["left_arm"] += left_arm_area * slice_cm
        volumes_cm3["right_arm"] += right_arm_area * slice_cm

        if z_center >= priors.waist_z:
            volumes_cm3["upper_torso"] += torso_area * slice_cm
        else:
            volumes_cm3["lower_torso"] += torso_area * slice_cm

    part_sum_cm3 = sum(v for k, v in volumes_cm3.items() if k != "total")

    diagnostics = {
        "slice_cm": slice_cm,
        "planar_pitch_cm": planar_pitch_cm,
        "slice_count": float(len(z_centers)),
        "nonempty_slice_count": float(nonempty_slice_count),
        "total_section_pixels": total_section_pixels,
        "center_x_cm": priors.center_x,
        "crotch_z_cm": priors.crotch_z,
        "waist_z_cm": priors.waist_z,
        "breast_z_cm": priors.breast_z,
        "shoulder_z_cm": priors.shoulder_z,
        "neck_base_z_cm": priors.neck_base_z,
        "min_leg_component_area_cm2": priors.min_leg_component_area_cm2,
        "torso_half_width_crotch_cm": float(torso_half_width_at_z(priors, priors.crotch_z + 1.0)),
        "torso_half_width_waist_cm": float(torso_half_width_at_z(priors, priors.waist_z)),
        "torso_half_width_breast_cm": float(torso_half_width_at_z(priors, priors.breast_z)),
        "torso_half_width_shoulder_cm": float(torso_half_width_at_z(priors, priors.shoulder_z)),
        "bust_width_prior_cm": priors.bust_width_cm,
        "waist_width_prior_cm": priors.waist_width_cm,
        "hip_width_prior_cm": priors.hip_width_cm,
        "shoulder_core_width_prior_cm": priors.shoulder_core_width_cm,
        "bbox_x_cm": float(x_max - x_min),
        "bbox_y_cm": float(y_max - y_min),
        "bbox_z_cm": float(z_max - z_min),
        "vertical_axis_source": float(orient_info["vertical_axis_source"]),
        "scale_factor": scale,
        "part_sum_cm3": part_sum_cm3,
        "part_sum_diff_cm3": volumes_cm3["total"] - part_sum_cm3,
        "body_height_cm": body_height_cm,
        "neck_core_half_width_cm": neck_half_width_cm,
        "neck_height_cm": get_measure(row, "neck_height_cm"),
        "neck_height_front_cm": get_measure(row, "neck_height_front_cm"),
        "waist_height_cm": get_measure(row, "waist_height_cm"),
        "crotch_height_cm": get_measure(row, "crotch_height_cm"),
        "breast_height_cm": get_measure(row, "breast_height_cm"),
        "buttock_height_cm": get_measure(row, "buttock_height_cm"),
        "hip_height_cm": get_measure(row, "hip_height_cm"),
        "shoulder_blades_height_cm": get_measure(row, "shoulder_blades_height_cm"),
        "left_shoulder_width_cm": get_measure(row, "left_shoulder_width_cm"),
        "right_shoulder_width_cm": get_measure(row, "right_shoulder_width_cm"),
        "cross_shoulder_cm": get_measure(row, "cross_shoulder_cm"),
    }

    if mesh.is_watertight:
        diagnostics["mesh_volume_cm3"] = abs(float(mesh.volume))
    else:
        diagnostics["mesh_volume_cm3"] = np.nan

    return volumes_cm3, diagnostics


def process_subject(
    row: pd.Series,
    data_dir: Optional[Path],
    slice_cm: float,
    planar_pitch_cm: float,
) -> dict[str, float]:
    subject_id = get_subject_id(row)
    if not subject_id:
        raise ValueError("Missing subject id")

    mesh_path = resolve_mesh_path(row, data_dir)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing mesh: {mesh_path}")

    log(f"  Loading {subject_id} from {mesh_path.name}")
    t0 = time.perf_counter()
    mesh = load_mesh(mesh_path)
    t1 = time.perf_counter()
    log(f"  Mesh loaded in {fmt_seconds(t1 - t0)} (vertices={len(mesh.vertices):,}, faces={len(mesh.faces):,})")

    log(f"  Integrating sections at {slice_cm:.2f} cm, planar raster pitch {planar_pitch_cm:.2f} cm")
    t2 = time.perf_counter()
    volumes_cm3, diagnostics = integrate_section_volumes(
        mesh,
        row,
        slice_cm=slice_cm,
        planar_pitch_cm=planar_pitch_cm,
    )
    t3 = time.perf_counter()
    log(f"  Section integration finished in {fmt_seconds(t3 - t2)}")

    cleaned = clean_mesh(mesh)
    result = {
        "subject_id": subject_id,
        "mesh_path": str(mesh_path),
        "method": "section_integral_measurement_guided_adaptive_split_v2",
        "slice_cm": slice_cm,
        "planar_pitch_cm": planar_pitch_cm,
        "mesh_is_watertight": bool(cleaned.is_watertight),
        "body_height_cm": diagnostics["body_height_cm"],
        "neck_height_cm": diagnostics["neck_height_cm"],
        "neck_height_front_cm": diagnostics["neck_height_front_cm"],
        "waist_height_cm": diagnostics["waist_height_cm"],
        "crotch_height_cm": diagnostics["crotch_height_cm"],
        "breast_height_cm": diagnostics["breast_height_cm"],
        "buttock_height_cm": diagnostics["buttock_height_cm"],
        "hip_height_cm": diagnostics["hip_height_cm"],
        "shoulder_blades_height_cm": diagnostics["shoulder_blades_height_cm"],
        "left_shoulder_width_cm": diagnostics["left_shoulder_width_cm"],
        "right_shoulder_width_cm": diagnostics["right_shoulder_width_cm"],
        "cross_shoulder_cm": diagnostics["cross_shoulder_cm"],
        "total_volume_cm3": volumes_cm3["total"],
        "total_volume_l": volumes_cm3["total"] / 1000.0,
    }

    mesh_volume_cm3 = diagnostics.get("mesh_volume_cm3", np.nan)
    result["mesh_volume_cm3"] = mesh_volume_cm3
    result["mesh_volume_l"] = mesh_volume_cm3 / 1000.0 if not np.isnan(mesh_volume_cm3) else np.nan

    for part in ["head", "upper_torso", "lower_torso", "left_arm", "right_arm", "left_leg", "right_leg"]:
        result[f"{part}_volume_cm3"] = volumes_cm3[part]
        result[f"{part}_volume_l"] = volumes_cm3[part] / 1000.0

    result["part_sum_cm3"] = diagnostics["part_sum_cm3"]
    result["part_sum_l"] = diagnostics["part_sum_cm3"] / 1000.0
    result["part_sum_diff_cm3"] = diagnostics["part_sum_diff_cm3"]
    result["part_sum_diff_l"] = diagnostics["part_sum_diff_cm3"] / 1000.0

    if not np.isnan(mesh_volume_cm3) and mesh_volume_cm3 > 0:
        result["section_to_mesh_ratio"] = volumes_cm3["total"] / mesh_volume_cm3
    else:
        result["section_to_mesh_ratio"] = np.nan

    result["volume_plausible"] = bool(25.0 <= result["total_volume_l"] <= 150.0)
    result["slice_count"] = diagnostics["slice_count"]
    result["nonempty_slice_count"] = diagnostics["nonempty_slice_count"]
    result["total_section_pixels"] = diagnostics["total_section_pixels"]
    result["center_x_cm"] = diagnostics["center_x_cm"]
    result["crotch_z_cm"] = diagnostics["crotch_z_cm"]
    result["waist_z_cm"] = diagnostics["waist_z_cm"]
    result["breast_z_cm"] = diagnostics["breast_z_cm"]
    result["shoulder_z_cm"] = diagnostics["shoulder_z_cm"]
    result["neck_core_half_width_cm"] = diagnostics["neck_core_half_width_cm"]
    result["neck_base_z_cm"] = diagnostics["neck_base_z_cm"]
    result["min_leg_component_area_cm2"] = diagnostics["min_leg_component_area_cm2"]
    result["torso_half_width_crotch_cm"] = diagnostics["torso_half_width_crotch_cm"]
    result["torso_half_width_waist_cm"] = diagnostics["torso_half_width_waist_cm"]
    result["torso_half_width_breast_cm"] = diagnostics["torso_half_width_breast_cm"]
    result["torso_half_width_shoulder_cm"] = diagnostics["torso_half_width_shoulder_cm"]
    result["bust_width_prior_cm"] = diagnostics["bust_width_prior_cm"]
    result["waist_width_prior_cm"] = diagnostics["waist_width_prior_cm"]
    result["hip_width_prior_cm"] = diagnostics["hip_width_prior_cm"]
    result["shoulder_core_width_prior_cm"] = diagnostics["shoulder_core_width_prior_cm"]
    result["bbox_x_cm"] = diagnostics["bbox_x_cm"]
    result["bbox_y_cm"] = diagnostics["bbox_y_cm"]
    result["bbox_z_cm"] = diagnostics["bbox_z_cm"]
    result["vertical_axis_source"] = diagnostics["vertical_axis_source"]
    result["scale_factor"] = diagnostics["scale_factor"]

    log(
        "  Volumes (L): "
        f"total={result['total_volume_l']:.2f}, "
        f"head={result['head_volume_l']:.2f}, "
        f"upper_torso={result['upper_torso_volume_l']:.2f}, "
        f"lower_torso={result['lower_torso_volume_l']:.2f}, "
        f"arms={result['left_arm_volume_l'] + result['right_arm_volume_l']:.2f}, "
        f"legs={result['left_leg_volume_l'] + result['right_leg_volume_l']:.2f}"
    )

    return result


def write_outputs(results: list[dict], errors: list[dict], out_path: Path) -> None:
    pd.DataFrame(results).to_csv(out_path, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_path.with_suffix(".errors.csv"), index=False)


def maybe_auto_lookup(primary_path: Path, explicit_lookup: Optional[Path]) -> Optional[Path]:
    if explicit_lookup is not None:
        return explicit_lookup
    candidate = primary_path.parent / "combined_cohort_measures.csv"
    if candidate.exists() and candidate != primary_path:
        return candidate
    return None


def summarize_columns(df: pd.DataFrame) -> str:
    preview = list(df.columns[:12])
    return ", ".join(preview) + (" ..." if len(df.columns) > 12 else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", required=True, type=Path, help="Primary CSV/TSV input table")
    parser.add_argument("--data-dir", type=Path, default=None, help="Folder containing OBJ files")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV")
    parser.add_argument("--sep", default=None, help="Separator for primary table, default auto-detect")
    parser.add_argument("--lookup-table", type=Path, default=None, help="Optional second CSV/TSV with extra measurement columns to merge in")
    parser.add_argument("--lookup-sep", default=None, help="Separator for lookup table, default auto-detect")
    parser.add_argument("--slice-cm", type=float, default=0.5, help="Cross-section spacing in cm")
    parser.add_argument("--planar-pitch-cm", type=float, default=0.4, help="2D raster pitch for each cross-section in cm")
    parser.add_argument("--pitch-cm", type=float, default=None, help="Deprecated alias for --slice-cm")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows")
    parser.add_argument("--subject-id", default=None, help="Only process one subject id")
    parser.add_argument("--save-every", type=int, default=10, help="Write partial results every N successful rows")
    parser.add_argument("--audit-only", action="store_true", help="Only inspect the input table and exit")
    args = parser.parse_args()

    if args.pitch_cm is not None:
        args.slice_cm = args.pitch_cm
    if args.planar_pitch_cm <= 0:
        raise ValueError("--planar-pitch-cm must be > 0")
    if args.slice_cm <= 0:
        raise ValueError("--slice-cm must be > 0")

    total_start = time.perf_counter()
    log("==== Volume_Creater started ====")
    log(f"Primary table    : {args.measurements}")
    log(f"Data dir         : {args.data_dir}")
    log(f"Output           : {args.out}")
    log(f"Slice spacing    : {args.slice_cm} cm")
    log(f"Planar pitch     : {args.planar_pitch_cm} cm")

    df = read_table(args.measurements, sep=args.sep)
    log(f"Loaded primary table with {len(df):,} rows and {len(df.columns):,} columns")
    log(f"Columns: {summarize_columns(df)}")

    present = required_logical_columns_present(df)
    log(f"Required logical columns present in primary: {present}")

    lookup_path = maybe_auto_lookup(args.measurements, args.lookup_table)
    primary_missing = [name for name, ok in present.items() if name != "subject_id" and not ok]
    if primary_missing and lookup_path is not None:
        log(f"Primary table is missing some measurement columns, merging lookup table: {lookup_path}")
        lookup_df = read_table(lookup_path, sep=args.lookup_sep)
        log(f"Loaded lookup table with {len(lookup_df):,} rows and {len(lookup_df.columns):,} columns")
        df = merge_lookup(df, lookup_df)
        present_after = required_logical_columns_present(df)
        log(f"Required logical columns after merge: {present_after}")

    if args.audit_only:
        log("Audit only requested. Exiting without processing meshes.")
        return

    if args.subject_id:
        subject_column = first_present(df.columns, COLUMN_ALIASES["subject_id"])
        if subject_column is None:
            raise ValueError("Could not find subject id column to filter on")
        df = df[df[subject_column].astype(str) == args.subject_id].copy()
        log(f"Subject filter applied: {args.subject_id}; rows remaining={len(df)}")

    if args.limit is not None:
        df = df.head(args.limit).copy()
        log(f"Limit applied: processing first {len(df)} rows")

    if len(df) == 0:
        raise ValueError("No rows to process after filtering")

    results: list[dict] = []
    errors: list[dict] = []

    total_rows = len(df)
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        subject_id = get_subject_id(row) or "MISSING"
        row_start = time.perf_counter()
        log("")
        log(f"--- [{index}/{total_rows}] {subject_id} ---")
        try:
            result = process_subject(
                row=row,
                data_dir=args.data_dir,
                slice_cm=args.slice_cm,
                planar_pitch_cm=args.planar_pitch_cm,
            )
            results.append(result)
            if len(results) % args.save_every == 0:
                log(f"Writing partial results to {args.out}")
                write_outputs(results, errors, args.out)
        except Exception as exc:
            log(f"  ERROR: {exc}")
            errors.append({"subject_id": subject_id, "error": str(exc)})

        row_end = time.perf_counter()
        log(f"Row runtime: {fmt_seconds(row_end - row_start)}")
        log(f"Progress: {len(results)} successes, {len(errors)} errors")

    log("")
    log("Final write of outputs")
    write_outputs(results, errors, args.out)

    total_end = time.perf_counter()
    log(f"Wrote {len(results)} results to {args.out}")
    if errors:
        log(f"Wrote {len(errors)} errors to {args.out.with_suffix('.errors.csv')}")
    log(f"Total runtime: {fmt_seconds(total_end - total_start)}")
    log("==== Volume_Creater finished ====")


if __name__ == "__main__":
    main()