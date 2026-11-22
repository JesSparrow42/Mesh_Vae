from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import trimesh

from .geometry import list_obj_files, simplify_mesh, meshes_to_padded_tensors, clean_mesh, normalize_mesh
from .conditioning import (
    build_age_gender_lookup,
    clean_stregkode,
    clean_column_names,
    _detect_sep,
    _pick_col,
    _to_numeric_series,
)
from .scan_geometry import (
    fit_scan_geometry_spec,
    load_aligned_scan_geometry,
    load_scan_geometry_spec,
    save_scan_geometry_spec,
    transform_scan_geometry,
)


def normalize_feature_source_name(feature_source: str) -> str:
    fs = str(feature_source or "scan").strip().lower().replace("-", "_")
    if fs == "hbalc":
        fs = "hba1c"
    if fs in {"both", "ldl_hbalc"}:
        fs = "ldl_hba1c"
    return fs


POINT_LABEL_REGION_NAMES = (
    "feet_lower_legs",
    "upper_legs",
    "pelvis",
    "torso",
    "arms_shoulders",
    "neck_head",
)


def point_label_ids(points: np.ndarray, n_regions: int = 6) -> np.ndarray:
    """Assign deterministic geometry-derived body regions to normalized points.

    For six regions, the longest robust axis is treated as body height.  Its
    direction is made stable by placing the narrower end (normally the head) at
    the top.  Height bands identify legs, pelvis, torso, and head; laterally
    distant points in the upper body are identified as arms/shoulders.

    These are pseudo labels, not manual anatomical annotations.  They are
    intentionally deterministic and use only the scan geometry.
    """
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    regions = max(1, int(n_regions))
    if xyz.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)

    lo = np.quantile(xyz, 0.02, axis=0)
    hi = np.quantile(xyz, 0.98, axis=0)
    spans = np.maximum(hi - lo, 1e-6)
    vertical_axis = int(np.argmax(spans))
    horizontal_axes = [a for a in range(3) if a != vertical_axis]

    height = np.clip(
        (xyz[:, vertical_axis] - lo[vertical_axis]) / spans[vertical_axis],
        0.0,
        1.0,
    )
    horizontal = xyz[:, horizontal_axes]
    middle = (height >= 0.35) & (height <= 0.70)
    center = np.median(horizontal[middle] if np.any(middle) else horizontal, axis=0)
    horizontal_scale = np.maximum(spans[horizontal_axes], 1e-6)
    radial = np.linalg.norm((horizontal - center[None, :]) / horizontal_scale[None, :], axis=1)

    # PCA/axis alignment has a sign ambiguity.  The head end of a body scan is
    # usually narrower than the two-feet end, so orient the narrower end upward.
    low_end = radial[height <= 0.12]
    high_end = radial[height >= 0.88]
    if low_end.size and high_end.size:
        low_width = float(np.quantile(low_end, 0.80))
        high_width = float(np.quantile(high_end, 0.80))
        if low_width < high_width:
            height = 1.0 - height

    if regions != 6:
        ids = np.floor(height * regions).astype(np.int64)
        return np.clip(ids, 0, regions - 1)

    ids = np.full(len(height), 3, dtype=np.int64)  # torso
    ids[height < 0.18] = 0
    ids[(height >= 0.18) & (height < 0.40)] = 1
    ids[(height >= 0.40) & (height < 0.52)] = 2
    ids[height >= 0.80] = 5

    upper_body = (height >= 0.48) & (height < 0.82)
    torso_band = radial[(height >= 0.52) & (height < 0.72)]
    arm_cutoff = float(np.quantile(torso_band, 0.82)) if torso_band.size else float(np.quantile(radial, 0.82))
    ids[upper_body & (radial > arm_cutoff)] = 4
    return ids


def point_label_channels(points: np.ndarray, n_regions: int = 6) -> np.ndarray:
    """Return one-hot channels for deterministic pseudo anatomical regions."""
    regions = max(1, int(n_regions))
    ids = point_label_ids(points, regions)

    onehot = np.zeros((len(ids), regions), dtype=np.float32)
    onehot[np.arange(len(ids)), ids] = 1.0
    return onehot


def point_label_summary(points: np.ndarray, n_regions: int = 6) -> tuple[list[int], str]:
    """Return region counts and a short fingerprint for run verification."""
    labels = point_label_channels(points, n_regions)
    counts = labels.sum(axis=0).astype(np.int64).tolist()
    digest = hashlib.sha256(labels.tobytes()).hexdigest()[:12]
    return counts, digest


def add_point_label_channels(points: np.ndarray, opt) -> np.ndarray:
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    if not bool(getattr(opt, "use_point_labels", False)):
        return xyz
    labels = point_label_channels(xyz, int(getattr(opt, "point_label_regions", 6)))
    return np.concatenate([xyz, labels], axis=1).astype(np.float32)


def add_scan_input_channels(
    points: np.ndarray,
    opt,
    scan_features: np.ndarray | None = None,
) -> np.ndarray:
    """Build per-point encoder input, including scan-level physical features."""
    out = add_point_label_channels(points, opt)
    if scan_features is None:
        return out
    global_features = np.asarray(scan_features, dtype=np.float32).reshape(1, -1)
    repeated = np.broadcast_to(global_features, (out.shape[0], global_features.shape[1]))
    return np.concatenate([out, repeated], axis=1).astype(np.float32)



def clinical_feature_columns_for_opt(opt, df: pd.DataFrame) -> list[str]:
    """Resolve TSV column names for --feature_source clinical sanity mode."""
    explicit = list(getattr(opt, "clinical_feature_cols", []) or [])
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing --clinical_feature_cols in {getattr(opt, 'cond_tsv', None)}: {missing}. "
                f"Available: {list(df.columns)[:80]}"
            )
        return explicit

    fs = normalize_feature_source_name(getattr(opt, "feature_source", "scan"))

    requested = []
    if fs == "ldl":
        requested = [("ldl_cholesterol", "ldl")]
    elif fs == "hba1c":
        requested = [("glycohemoglobin_hba1c", "hba1c")]
    elif fs == "ldl_hba1c":
        requested = [("ldl_cholesterol", "ldl"), ("glycohemoglobin_hba1c", "hba1c")]
    else:
        raise ValueError(f"Clinical feature mode got unsupported --feature_source={fs}")

    cols = []
    for preferred, contains in requested:
        col = _pick_col(df, [preferred], fallback_contains=contains)
        if col is None:
            raise KeyError(
                f"Could not find clinical feature column for {preferred} in {getattr(opt, 'cond_tsv', None)}. "
                f"Available: {list(df.columns)[:80]}"
            )
        cols.append(col)
    return cols


def _clinical_stats_path(opt) -> Path:
    return Path(opt.checkpoints_dir) / opt.name / "clinical_feature_stats.json"


def _load_or_fit_clinical_stats(opt, X_raw: np.ndarray, feature_cols: list[str]) -> dict:
    stats_path = _clinical_stats_path(opt)
    is_train = bool(getattr(opt, "is_train", False))

    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        old_cols = list(stats.get("feature_cols", []))
        if old_cols and old_cols != list(feature_cols):
            raise ValueError(
                f"Clinical feature columns in checkpoint stats do not match this run: "
                f"checkpoint={old_cols}, current={list(feature_cols)}. Use a new --name."
            )
        return stats

    if not is_train:
        print(
            f"[CLINICAL] WARNING: {stats_path} not found; fitting feature normalization "
            "from this TSV. For final evaluation, train first with the same --name.",
            flush=True,
        )

    mean = np.nanmean(X_raw, axis=0).astype(np.float32)
    std = np.nanstd(X_raw, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)

    stats = {
        "feature_source": normalize_feature_source_name(getattr(opt, "feature_source", "scan")),
        "feature_cols": list(feature_cols),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "clinical_points": int(getattr(opt, "clinical_points", 32)),
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[CLINICAL] wrote feature normalization -> {stats_path}")
    return stats


def _features_to_pointcloud(X_norm: np.ndarray, n_points: int) -> np.ndarray:
    """Encode 1-2 normalized numeric features as a tiny deterministic 3D point cloud."""
    X_norm = np.asarray(X_norm, dtype=np.float32)
    n, k = X_norm.shape
    e = max(2, int(n_points))
    grid = np.linspace(-1.0, 1.0, e, dtype=np.float32)
    P = np.zeros((n, e, 3), dtype=np.float32)
    P[:, :, 2] = grid[None, :]

    if k >= 1:
        P[:, :, 0] = X_norm[:, 0:1]
    if k >= 2:
        P[:, :, 1] = X_norm[:, 1:2]
    if k > 2:
        # Rare custom case: fold extra features into a small sinusoidal trace.
        extra = X_norm[:, 2:].mean(axis=1, keepdims=True)
        P[:, :, 2] = P[:, :, 2] + 0.05 * extra
    return P


class ClinicalFeatureMeshVAEDataset(Dataset):
    """
    Uses numeric TSV features instead of body scans.

    This is a sanity-check path for the existing VAE/diagnostic code: LDL/HBA1C
    are encoded as a small point cloud so the model can be trained and diagnosed
    with the same scripts and one extra argument: --feature_source.
    """

    def __init__(self, folder, opt):
        super().__init__()
        self.opt = opt
        cond_path = getattr(opt, "cond_tsv", None) or getattr(opt, "cohort_tsv", None)
        if cond_path is None:
            raise ValueError("Clinical feature mode requires --cond_tsv pointing to the train/test TSV.")

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

        st_col = getattr(opt, "cond_stregkode_col", None) or _pick_col(
            df, ["stregkode", "participant_id", "participantid", "id"], fallback_contains="stregkode"
        )
        if st_col is None:
            st_col = _pick_col(df, ["participant_id", "participantid"], fallback_contains="participant")
        if st_col is None:
            raise KeyError(f"Could not find participant id column in {cond_path}. Available: {list(df.columns)[:80]}")

        feature_cols = clinical_feature_columns_for_opt(opt, df)
        X_raw_all = np.stack(
            [_to_numeric_series(df[c]).to_numpy(dtype=np.float32) for c in feature_cols],
            axis=1,
        )

        code2cond, code2group, _code2aux = build_age_gender_lookup(opt)

        codes_all = df[st_col].apply(clean_stregkode).tolist()
        keep_idx = []
        names = []
        conds = []
        groups = []
        for i, code in enumerate(codes_all):
            c = code2cond.get(code)
            if c is None and getattr(opt, "use_cond", False) and not getattr(opt, "keep_missing_cond", False):
                continue
            row_x = X_raw_all[i]
            if np.all(~np.isfinite(row_x)):
                continue
            keep_idx.append(i)
            names.append(code or f"row_{i}")
            if getattr(opt, "use_cond", False):
                conds.append(c if c is not None else np.zeros(int(getattr(opt, "cond_dim", 6)), dtype=np.float32))
                groups.append(code2group.get(code, "UNK"))

        if not keep_idx:
            raise ValueError(
                f"No usable rows left in {cond_path} for --feature_source={getattr(opt, 'feature_source', None)}. "
                "Check participant ids, gender/age filters, disease columns, and missing clinical values."
            )

        X_raw = X_raw_all[np.asarray(keep_idx, dtype=np.int64)]
        stats = _load_or_fit_clinical_stats(opt, X_raw, feature_cols)
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        X_imp = np.where(np.isfinite(X_raw), X_raw, mean[None, :]).astype(np.float32)
        X_norm = (X_imp - mean[None, :]) / std[None, :]
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        self.feature_cols = list(feature_cols)
        self.raw_features = X_raw.astype(np.float32)
        self.norm_features = X_norm.astype(np.float32)
        self.names = np.asarray(names, dtype=object)
        self.points = _features_to_pointcloud(X_norm, int(getattr(opt, "clinical_points", 32)))
        self.verts = self.points.astype(np.float32)
        self.verts_mask = np.ones(self.verts.shape[:2], dtype=bool)
        self.faces = np.zeros((self.verts.shape[0], 0, 3), dtype=np.int64)
        self.faces_mask = np.zeros((self.verts.shape[0], 0), dtype=bool)

        self.use_cond = bool(getattr(self.opt, "use_cond", False))
        self.cond = np.stack(conds, axis=0).astype(np.float32) if self.use_cond else None
        self.group = np.asarray(groups, dtype=object) if self.use_cond else None
        self.plaque_target = None
        self.age_scalar = None

        print(f"[CLINICAL] Loaded {len(self.names)} rows from {cond_path}")
        print(f"[CLINICAL] feature_source={getattr(opt, 'feature_source', None)} cols={self.feature_cols}")
        print(f"[CLINICAL] points shape: {self.points.shape}")

    def __len__(self):
        return int(self.points.shape[0])

    def __getitem__(self, idx):
        P = self.points[idx].astype(np.float32)
        V_mask = self.verts_mask[idx]
        F = self.faces[idx]
        F_mask = self.faces_mask[idx]

        data = {
            "path": str(self.names[idx]),
            "simplified": None,
            "verts": torch.from_numpy(P),
            "faces": torch.from_numpy(F),
            "verts_mask": torch.from_numpy(V_mask),
            "faces_mask": torch.from_numpy(F_mask),
            "mesh": [],
            "edge_features": torch.from_numpy(add_point_label_channels(P, self.opt).T).unsqueeze(0),
        }

        if self.use_cond and self.cond is not None:
            data["cond"] = torch.from_numpy(self.cond[idx])
            data["group"] = str(self.group[idx])

        return data


def load_obj_paths_from_cohort(opt):
    # Resolve cohort TSV from --subset_group if provided
    cohort_path = opt.cohort_tsv
    if cohort_path is None and opt.subset_group is not None:
        group = opt.subset_group
        cohort_path = f"/data/projects/3D_scans/sorted_data/{group}/cohort_{group}.tsv"

    if cohort_path is None:
        return None  # means: fall back to scanning dataroot

    df = pd.read_csv(cohort_path, sep="\t")
    if opt.cohort_obj_col not in df.columns:
        raise KeyError(
            f"Column '{opt.cohort_obj_col}' not found in {cohort_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Clean path strings (strip quotes/spaces)
    paths = (
        df[opt.cohort_obj_col]
        .astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip("'")
        .tolist()
    )

    # Keep only .obj paths
    paths = [p for p in paths if p.lower().endswith(".obj")]

    # Optional: require file exists
    if opt.require_existing:
        from pathlib import Path
        before = len(paths)
        paths = [p for p in paths if Path(p).exists()]
        print(f"[COHORT] require_existing: kept {len(paths)}/{before} existing OBJ paths")

    print(f"[COHORT] Loaded {len(paths)} OBJ paths from {cohort_path}")
    return paths


class MeshVAEDataset(Dataset):
    """
    Loads OBJ files, simplifies them, converts to padded tensors.
    """

    def __init__(self, folder, opt):
        super().__init__()
        self.opt = opt
        cohort_paths = load_obj_paths_from_cohort(opt)

        if cohort_paths is not None:
            self.files = [Path(p) for p in cohort_paths]
        else:
            self.files = list_obj_files(Path(folder))

        self.files = self.files[: opt.max_dataset_size]
        print(f"[DATA] Using {len(self.files)} OBJ meshes.")

        self.meshes = []
        for p in self.files:
            import open3d as o3d

            m = o3d.io.read_triangle_mesh(str(p), enable_post_processing=True)
            if m.is_empty():
                print(f"[DATA] WARNING: empty mesh skipped: {p}")
                continue
            m = clean_mesh(m)
            self.meshes.append((p, m))

        print(f"[DATA] Loaded {len(self.meshes)} meshes.")

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        path, mesh = self.meshes[idx]

        # Keep a copy of the raw mesh if you care about it later
        raw_mesh = mesh

        # 1) Clean
        mesh = clean_mesh(mesh)

        # 2) Normalize (center + PCA + unit sphere)
        mesh = normalize_mesh(mesh, center_mode="bbox", align_axes=True, scale_mode="unit_sphere")

        # 3) Simplify
        simp = simplify_mesh(mesh, target_tris=self.opt.target_tris)

        # Geometry arrays (already normalized)
        V = np.asarray(simp.vertices, dtype=np.float64)
        F = np.asarray(simp.triangles, dtype=np.int64)

        # Trimesh should match what model sees
        tri = trimesh.Trimesh(vertices=V, faces=F, process=False)

        # Edge features from SAME normalized vertices
        feats_np = add_point_label_channels(V.astype(np.float32), self.opt)
        edge_features = torch.from_numpy(feats_np.T).unsqueeze(0)  # (1, C, N)

        # 4) Convert to padded tensors
        batch = meshes_to_padded_tensors([simp])

        data = {
            "path": str(path),
            "original": raw_mesh,                 # raw Open3D (optional)
            "simplified": simp,                   # normalized + simplified Open3D
            "verts": batch["verts"][0],
            "faces": batch["faces"][0],
            "verts_mask": batch["verts_mask"][0],
            "faces_mask": batch["faces_mask"][0],
            "mesh": [tri],                        # list of Trimesh
            "edge_features": edge_features,       # (1, C, N)
        }
        return data


class PreprocessedMeshVAEDataset(Dataset):
    """
    Loads meshes from the offline-preprocessed NPZ:
      simplified_padded_tensors.npz

    - verts: (B, Vmax, 3)
    - verts_mask: (B, Vmax)
    - faces: (B, Fmax, 3)
    - faces_mask: (B, Fmax)
    - names: (B,)
    - points: (B, n_points, 3)  [optional, for point-cloud input]
    """
    def __init__(self, folder, opt):
        super().__init__()
        self.opt = opt
        root = Path(folder)
        npz_path = root / "simplified_padded_tensors.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Preprocessed NPZ not found at {npz_path}. "
                                    f"Run with --phase preprocess first.")

        self.data = np.load(npz_path, allow_pickle=True)
        self.verts = self.data["verts"].astype(np.float32)        # (B, Vmax, 3)
        self.verts_mask = self.data["verts_mask"].astype(bool)    # (B, Vmax)
        self.faces = self.data["faces"].astype(np.int64)          # (B, Fmax, 3)
        self.faces_mask = self.data["faces_mask"].astype(bool)    # (B, Fmax)
        self.names = self.data["names"]
        self.points = self.data["points"].astype(np.float32) if "points" in self.data.files else None
        self.scan_features = None
        self.scan_features_raw = None
        self.scan_scale_proxy = None
        self.scan_geometry_valid = None
        self.scan_geometry_feature_names: list[str] = []
        self.scan_geometry_path = None
        if bool(getattr(self.opt, "use_scan_geometry_features", False)):
            (
                self.scan_features_raw,
                self.scan_scale_proxy,
                self.scan_geometry_feature_names,
                self.scan_geometry_valid,
                self.scan_geometry_path,
            ) = load_aligned_scan_geometry(
                root,
                getattr(self.opt, "scan_geometry_features_file", "scan_geometry_features.npz"),
                self.names,
                require_complete=False,
            )
        if getattr(self.opt, "use_pointcloud", False) and self.points is not None:
            point_count = int(self.points.shape[1])
            self.opt.ninput_edges = point_count
            if hasattr(self.opt, "chamfer_npoints"):
                self.opt.chamfer_npoints = min(int(self.opt.chamfer_npoints), point_count)
            if hasattr(self.opt, "metric_npoints"):
                self.opt.metric_npoints = min(int(self.opt.metric_npoints), point_count)
            print(f"[DATA] point-cloud length -> ninput_edges={point_count}")

        # ---------- conditioning ----------
        self.use_cond = bool(getattr(self.opt, "use_cond", False))
        self.use_aux_plaque = bool(getattr(self.opt, "aux_predict_plaque", False))
        self.use_aux_regression = bool(getattr(self.opt, "aux_predict_regression", False))
        self.use_aux_binary = bool(getattr(self.opt, "aux_predict_binary", False))

        self.cond = None
        self.group = None
        self.plaque_target = None
        self.age_scalar = None
        self.regression_target = None
        self.regression_target_raw = None
        self.binary_target = None

        if self.use_cond or self.use_aux_plaque or self.use_aux_regression or self.use_aux_binary:
            code2cond, code2group, code2aux = build_age_gender_lookup(self.opt)

            orig_n = len(self.names)
            keep = []
            conds = []
            groups = []
            plaque_targets = []
            age_scalars = []
            regression_targets = []
            regression_targets_raw = []
            binary_targets = []
            missing_meta_examples = []

            for i, nm in enumerate(self.names):
                code = clean_stregkode(nm)
                c = code2cond.get(code)
                aux = code2aux.get(code, {})

                if self.use_aux_regression and "regression_target" not in aux:
                    if len(missing_meta_examples) < 10:
                        missing_meta_examples.append(str(nm))
                    continue
                if self.use_aux_binary and "binary_target" not in aux:
                    if len(missing_meta_examples) < 10:
                        missing_meta_examples.append(str(nm))
                    continue

                if self.use_cond and c is None:
                    if len(missing_meta_examples) < 10:
                        missing_meta_examples.append(str(nm))
                    if getattr(self.opt, "keep_missing_cond", False):
                        keep.append(i)
                        conds.append(np.zeros(int(getattr(self.opt, "cond_dim", 6)), dtype=np.float32))
                        groups.append("UNK")
                    else:
                        continue
                else:
                    keep.append(i)
                    if self.use_cond:
                        conds.append(c if c is not None else np.zeros(int(getattr(self.opt, "cond_dim", 6)), dtype=np.float32))
                        groups.append(code2group.get(code, "UNK"))

                if self.use_aux_plaque:
                    plaque_targets.append(
                        np.asarray(aux.get("plaque_target", np.zeros(4, dtype=np.float32)), dtype=np.float32)
                    )
                    age_scalars.append(np.float32(aux.get("age", np.nan)))
                if self.use_aux_regression:
                    regression_targets.append(np.float32(aux["regression_target"]))
                    regression_targets_raw.append(np.float32(aux["regression_target_raw"]))
                if self.use_aux_binary:
                    binary_targets.append(np.float32(aux["binary_target"]))

            keep = np.asarray(keep, dtype=np.int64)

            if keep.size != orig_n:
                self.verts = self.verts[keep]
                self.verts_mask = self.verts_mask[keep]
                self.faces = self.faces[keep]
                self.faces_mask = self.faces_mask[keep]
                self.names = self.names[keep]
                if self.points is not None:
                    self.points = self.points[keep]
                if self.scan_features_raw is not None:
                    self.scan_features_raw = self.scan_features_raw[keep]
                    self.scan_scale_proxy = self.scan_scale_proxy[keep]
                    self.scan_geometry_valid = self.scan_geometry_valid[keep]
                print(f"[COND] Kept {len(self.names)}/{orig_n} with metadata (dropped {orig_n - len(self.names)}).")
                print(
                    f"[COND] Metadata lookup size={len(code2cond)}. "
                    f"If this drop is unexpected, the TSV split and preprocessed NPZ likely come from different cohorts."
                )
                if missing_meta_examples:
                    print(f"[COND] First unmatched NPZ names: {missing_meta_examples}")

            if self.use_cond:
                self.cond = np.stack(conds, axis=0).astype(np.float32)
                self.group = np.array(groups, dtype=object)

            if self.use_aux_plaque:
                self.plaque_target = np.stack(plaque_targets, axis=0).astype(np.float32)
                self.age_scalar = np.asarray(age_scalars, dtype=np.float32)
            if self.use_aux_regression:
                self.regression_target = np.asarray(regression_targets, dtype=np.float32)
                self.regression_target_raw = np.asarray(regression_targets_raw, dtype=np.float32)
            if self.use_aux_binary:
                self.binary_target = np.asarray(binary_targets, dtype=np.float32)

        if self.scan_features_raw is not None:
            require_features = bool(getattr(self.opt, "require_scan_geometry_features", False))
            missing_geometry = ~self.scan_geometry_valid
            if require_features and bool(missing_geometry.any()):
                examples = [str(self.names[i]) for i in np.flatnonzero(missing_geometry)[:20]]
                raise ValueError(
                    f"{int(missing_geometry.sum())} selected scans are missing physical geometry features. "
                    f"First missing names: {examples}"
                )

            stats_path = Path(self.opt.checkpoints_dir) / self.opt.name / "scan_geometry_stats.json"
            if bool(getattr(self.opt, "is_train", False)):
                scan_spec = fit_scan_geometry_spec(
                    self.scan_features_raw,
                    self.scan_scale_proxy,
                    self.scan_geometry_feature_names,
                )
                save_scan_geometry_spec(scan_spec, stats_path)
                print(f"[SCAN GEOMETRY] wrote training-only normalization -> {stats_path}", flush=True)
            else:
                if not stats_path.exists():
                    raise FileNotFoundError(
                        f"Training scan geometry statistics not found: {stats_path}. "
                        "Use the same --name that was used for training."
                    )
                scan_spec = load_scan_geometry_spec(stats_path)
                if list(scan_spec.feature_names) != list(self.scan_geometry_feature_names):
                    raise ValueError(
                        "Scan geometry sidecar columns do not match the checkpoint's training statistics."
                    )

            self.scan_features, coordinate_factor, transformed_valid = transform_scan_geometry(
                self.scan_features_raw,
                self.scan_scale_proxy,
                scan_spec,
            )
            self.scan_geometry_valid = transformed_valid
            if bool(getattr(self.opt, "restore_metric_scale", True)):
                if self.points is not None:
                    self.points *= coordinate_factor[:, None, None]
                self.verts *= coordinate_factor[:, None, None]
                print(
                    f"[SCAN GEOMETRY] restored between-person scale using one training median "
                    f"({scan_spec.coordinate_scale:.6g} raw units).",
                    flush=True,
                )

            feature_dim = int(self.scan_features.shape[1])
            self.opt.scan_geometry_feature_dim = feature_dim
            self.opt.input_nc = (
                3
                + (int(getattr(self.opt, "point_label_regions", 6)) if bool(getattr(self.opt, "use_point_labels", False)) else 0)
                + feature_dim
            )
            print(
                f"[SCAN GEOMETRY] encoder channels: xyz=3 geometry={feature_dim} "
                f"total={self.opt.input_nc}",
                flush=True,
            )

            # build_age_gender_lookup writes cond_stats before geometry is loaded.
            # Amend it now so diagnosis reconstructs the exact training input.
            if bool(getattr(self.opt, "is_train", False)):
                cond_stats_path = Path(self.opt.checkpoints_dir) / self.opt.name / "cond_stats.json"
                if cond_stats_path.exists():
                    cond_stats = json.loads(cond_stats_path.read_text(encoding="utf-8"))
                    cond_stats.update({
                        "input_nc": int(self.opt.input_nc),
                        "use_scan_geometry_features": True,
                        "require_scan_geometry_features": bool(getattr(self.opt, "require_scan_geometry_features", False)),
                        "scan_geometry_features_file": str(getattr(self.opt, "scan_geometry_features_file", "scan_geometry_features.npz")),
                        "scan_geometry_feature_dim": feature_dim,
                        "scan_geometry_feature_names": list(self.scan_geometry_feature_names),
                        "restore_metric_scale": bool(getattr(self.opt, "restore_metric_scale", True)),
                    })
                    cond_stats_path.write_text(json.dumps(cond_stats, indent=2), encoding="utf-8")

        print("vertices shape:", self.verts.shape)
        print("verts_mask shape:", self.verts_mask.shape)
        print("faces shape:", self.faces.shape)
        print("faces_mask shape:", self.faces_mask.shape)
        print("points:", None if self.points is None else self.points.shape)

        print(f"[DATA] Loaded preprocessed NPZ: {npz_path}")
        print(f"[DATA]   meshes={self.verts.shape[0]}, Vmax={self.verts.shape[1]}, Fmax={self.faces.shape[1]}")
        if bool(getattr(self.opt, "use_point_labels", False)) and len(self.names) > 0:
            sample = self.points[0] if self.points is not None else self.verts[0][self.verts_mask[0]]
            if self.points is not None:
                sample = sample[np.lexsort((sample[:, 2], sample[:, 1], sample[:, 0]))]
            counts, digest = point_label_summary(sample, int(getattr(self.opt, "point_label_regions", 6)))
            print(
                f"[POINT LABEL] scheme=anatomical_regions regions={len(counts)} "
                f"counts={counts} fingerprint={digest}",
                flush=True,
            )
            if bool(getattr(self.opt, "is_train", False)):
                debug_path = Path(self.opt.checkpoints_dir) / self.opt.name / "point_label_debug.json"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_path.write_text(
                    json.dumps(
                        {
                            "scheme": "anatomical_regions_v2",
                            "sample": str(self.names[0]),
                            "regions": int(len(counts)),
                            "counts": counts,
                            "fingerprint": digest,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"[POINT LABEL] wrote verification -> {debug_path}", flush=True)

    def __len__(self):
        return self.verts.shape[0]

    def __getitem__(self, idx):
        V_pad = self.verts[idx]         # (Vmax, 3)
        V_mask = self.verts_mask[idx]   # (Vmax,)
        F_pad = self.faces[idx]         # (Fmax, 3)
        F_mask = self.faces_mask[idx]   # (Fmax,)

        # Unpadded geometry
        V = V_pad[V_mask]               # (V_real, 3)
        F = F_pad[F_mask]               # (F_real, 3)

        # Trimesh for the VAE (even though pool_mode=skip currently)
        tri = trimesh.Trimesh(vertices=V.astype(np.float64),
                              faces=F.astype(np.int64),
                              process=False)

        # Choose representation for edge_features:
        #  - vertices (default) or
        #  - point cloud (if use_pointcloud + points present)
        if getattr(self.opt, "use_pointcloud", False) and (self.points is not None):
            P = self.points[idx].astype(np.float32)   # (n_points, 3)
            # Canonical ordering: sort points by (x, y, z)
            order = np.lexsort((P[:, 2], P[:, 1], P[:, 0]))
            P = P[order]

            scan_features = self.scan_features[idx] if self.scan_features is not None else None
            feats_np = add_scan_input_channels(P, self.opt, scan_features).T
        else:
            scan_features = self.scan_features[idx] if self.scan_features is not None else None
            feats_np = add_scan_input_channels(V.astype(np.float32), self.opt, scan_features).T

        edge_features = torch.from_numpy(feats_np).unsqueeze(0)    # (1, C, E)

        # Rebuild simple Open3D mesh from simplified geometry (for stats / redetail)
        import open3d as o3d
        simp_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(V.astype(np.float64)),
            triangles=o3d.utility.Vector3iVector(F.astype(np.int64)),
        )
        simp_mesh = clean_mesh(simp_mesh)

        data = {
            "path": str(self.names[idx]),
            "simplified": simp_mesh,            # Open3D mesh, collate -> [simp_mesh]
            "verts": torch.from_numpy(V_pad),   # padded
            "faces": torch.from_numpy(F_pad),
            "verts_mask": torch.from_numpy(V_mask),
            "faces_mask": torch.from_numpy(F_mask),
            "mesh": [tri],                      # list of Trimesh (what the model expects)
            "edge_features": edge_features,     # (1, C, E)
        }

        if self.use_cond and self.cond is not None:
            data["cond"] = torch.from_numpy(self.cond[idx])
            data["group"] = str(self.group[idx])

        if self.use_aux_plaque and self.plaque_target is not None:
            data["plaque_target"] = torch.from_numpy(self.plaque_target[idx])
            data["age_scalar"] = torch.tensor(self.age_scalar[idx], dtype=torch.float32)
        if self.use_aux_regression and self.regression_target is not None:
            data["regression_target"] = torch.tensor(self.regression_target[idx], dtype=torch.float32)
            data["regression_target_raw"] = torch.tensor(self.regression_target_raw[idx], dtype=torch.float32)
        if self.use_aux_binary and self.binary_target is not None:
            data["binary_target"] = torch.tensor(self.binary_target[idx], dtype=torch.float32)

        return data


def mesh_collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        v0 = vals[0]
        if isinstance(v0, torch.Tensor):
            if k == "edge_features":
                # vals[i]: (1, C, E) → out: (B, C, E)
                out[k] = torch.cat(vals, dim=0)
            else:
                out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out
