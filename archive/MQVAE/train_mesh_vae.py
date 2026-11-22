#!/usr/bin/env python3
"""
Clean training script for MeshVAE + Boson prior + geometry pipeline.

To run:

python train_mesh_vae.py \
    --dataroot ./data \
    --arch meshvae \
    --phase train \
    --name REACT_vae_boson \
    --boson_params '{"m":16,"num_sources":8,"num_loops":1,"input_loss":0.0,"coupling_efficiency":1.0,"detector_inefficiency":0.0,"multi_photon_prob":0.0,"mu":1.0,"temporal_mismatch":0.0,"spectral_mismatch":0.0,"arrival_time_jitter":0.0,"bs_loss":0.0,"bs_jitter":0.0,"phase_noise_std":0.0,"systematic_phase_offset":0.0,"mode_loss":[0.0],"dark_count_rate":0.0,"use_advanced_nonlinearity":false,"chi":0.0}' \
    --ncf 64 128 256 256 \
    --pool_res 900 450 300 180 \
    --latent_dim 64 \
    --beta 1.0 \
    --ninput_edges 10000 \
    --pool_mode skip \
    --latent_prior boson \
    --gpu_ids 0 \
    --split_ratio 0.2 \
    --split_seed 0 \
    --dump_padded_obj \
    --dump_padded_max 8 \
    --debug_kernels \
    --use_amp \
    --max_dataset_size 2
"""

import os
import json
from pathlib import Path
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
import trimesh
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import pandas as pd
from datetime import datetime
import re
import csv

from mesh_geom_pipeline import (
    list_obj_files,
    simplify_mesh,
    mesh_stats_o3d,
    meshes_to_padded_tensors,
    tensors_to_meshes,
    clean_mesh,
    preprocess_obj_folder,
    normalize_mesh,   # <-- add this
)

from mesh_vae import MeshVAEModel


# =========================================
# Base options
# =========================================

class BaseOptions:
    def __init__(self):
        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument("--dataroot", required=True)
        parser.add_argument("--name", required=True)
        parser.add_argument("--checkpoints_dir", default="./checkpoints")

        parser.add_argument("--gpu_ids", type=str, default="0")
        parser.add_argument(
            "--phase",
            type=str,
            default="train",
            choices=["preprocess", "train", "test", "sample", "search_age_bins", "train_baseline"],
        )

        # Preprocessing / geometry
        parser.add_argument("--target_tris", type=int, default=20000,
                            help="Target number of triangles per simplified mesh (offline preprocessing).")
        parser.add_argument("--n_points", type=int, default=0,
                            help="If > 0, also precompute point clouds with this many points per mesh.")
        parser.add_argument("--use_preprocessed", action="store_true",
                            help="Load meshes from simplified_padded_tensors.npz instead of raw OBJs.")
        parser.add_argument("--use_pointcloud", action="store_true",
                            help="Use precomputed point clouds as input features instead of mesh vertices.")

        parser.add_argument("--arch", type=str, default="meshvae")
        parser.add_argument("--pool_mode", type=str, default="skip")

        # Model sizing
        parser.add_argument("--input_nc", type=int, default=3)  # xyz features
        parser.add_argument("--ncf", nargs="+", type=int, default=[64, 128, 256])
        parser.add_argument("--latent_dim", type=int, default=64)
        parser.add_argument("--ninput_edges", type=int, default=10000)

        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--latent_prior", type=str, default="gaussian")
        parser.add_argument("--boson_params", type=str, default=None)

        parser.add_argument("--lr", type=float, default=0.0002)
        parser.add_argument("--beta1", type=float, default=0.5)
        parser.add_argument("--lr_policy", type=str, default="lambda")
        parser.add_argument("--lr_decay_iters", type=int, default=50)
        parser.add_argument("--epoch_count", type=int, default=1)
        parser.add_argument("--niter", type=int, default=100)
        parser.add_argument("--niter_decay", type=int, default=50)
        parser.add_argument("--init_type", type=str, default="xavier")
        parser.add_argument("--init_gain", type=float, default=0.02)

        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_threads", type=int, default=0)

        parser.add_argument("--max_dataset_size", type=int, default=99999)

        parser.add_argument("--split_ratio", type=float, default=0.2)
        parser.add_argument("--split_seed", type=int, default=0)

        parser.add_argument("--use_amp", action="store_true")
        parser.add_argument("--dump_padded_obj", action="store_true")
        parser.add_argument("--dump_padded_max", type=int, default=8)
        parser.add_argument("--debug_kernels", action="store_true")

        parser.add_argument(
            "--outroot", 
            type=str, 
            default=".",
            help="Output root for test reconstructions."
        )

        parser.add_argument("--cohort_tsv", type=str, default=None,
                    help="Path to cohort TSV containing a column with OBJ paths (e.g. obj_path).")
        parser.add_argument("--cohort_obj_col", type=str, default="obj_path",
                            help="Column name in cohort TSV that contains full paths to .obj files.")
        parser.add_argument("--subset_group", type=str, default=None,
                            help="Shortcut group name like 'F_18-40' -> will look for "
                                "/data/projects/3D_scans/sorted_data/<group>/cohort_<group>.tsv")
        parser.add_argument("--require_existing", action="store_true",
                            help="Drop cohort rows whose obj_path does not exist on disk.")
        parser.add_argument("--chamfer_npoints", type=int, default=2048,
                            help="Subsample size for Chamfer loss when --use_pointcloud is set.")

        # ----------------- Conditioning (age + gender) -----------------
        parser.add_argument("--use_cond", action="store_true",
                            help="Train conditional VAE using age+gender from a cohort TSV.")
        parser.add_argument("--cond_tsv", type=str, default=None,
                            help="TSV with stregkode + age + sex/gender columns (e.g. cohort_20251031.tsv).")
        parser.add_argument("--cond_stregkode_col", type=str, default=None,
                            help="Column name for stregkode (auto-detect if None).")
        parser.add_argument("--cond_age_col", type=str, default=None,
                            help="Column name for age (auto-detect if None).")
        parser.add_argument("--cond_gender_col", type=str, default=None,
                            help="Column name for sex/gender (auto-detect if None).")
        parser.add_argument("--cond_embed_dim", type=int, default=16,
                            help="Embedding size used inside the network for conditioning.")
        parser.add_argument("--keep_missing_cond", action="store_true",
                            help="Keep samples missing metadata (cond=all zeros). Default: drop them.")
        
        parser.add_argument(
            "--only_gender",
            type=str,
            default=None,
            choices=["M", "F"],
            help="Optional: keep only one gender throughout dataset construction. "
                "Use this for gender-separated training / diagnosis."
        )

        # ---- Bin Search for age-based sampling policy (diagnosis) ----
        parser.add_argument(
            "--cond_age_mode",
            type=str,
            default="hard",
            choices=["hard", "soft", "rbf"],
            help="Age encoding inside the 4 existing age dims: "
                "'hard' = one-hot, 'soft' = linear interpolation, 'rbf' = RBF over age-bin centers."
        )
        parser.add_argument(
            "--age_bins",
            nargs="+",
            type=float,
            default=[18, 35, 50, 60, 200],
            help="Age-bin edges. Number of bins is dynamic."
        )
        parser.add_argument("--age_search_min_bins", type=int, default=4)
        parser.add_argument("--age_search_max_bins", type=int, default=10)
        parser.add_argument("--age_search_switch_age", type=float, default=50.0)
        parser.add_argument("--age_search_step_young", type=float, default=10.0)
        parser.add_argument("--age_search_step_old", type=float, default=5.0)
        parser.add_argument("--age_search_objective", type=str, default="diagnostic_bal_acc",
                            choices=["diagnostic_bal_acc", "diagnostic_auc", "val_loss"])
        parser.add_argument("--age_search_policy_mode", type=str, default="global",
                            choices=["global", "age", "prevalence"])
        parser.add_argument(
            "--age_rbf_gamma",
            type=float,
            default=0.015,
            help="RBF gamma when --cond_age_mode=rbf."
        )

        parser.add_argument(
            "--encoder_use_cond",
            action="store_true",
            help="Allow metadata into encoder. Leave OFF for scan-driven diagnosis."
        )
        parser.add_argument(
            "--use_cond_prior",
            action="store_true",
            help="Use p(z|cond). Leave OFF for scan-driven diagnosis."
        )

        # ---- Supervised latent atherosclerosis head ----
        parser.add_argument(
            "--aux_predict_athero",
            action="store_true",
            help="Add a supervised classifier head on latent mu(X). Keeps the model as a VAE but makes latent space disease-predictive."
        )
        parser.add_argument(
            "--aux_athero_mode",
            type=str,
            default="any",
            choices=["any", "4bit"],
            help="'any' = binary sick/healthy head. '4bit' = one auxiliary target per atherosclerosis location."
        )
        parser.add_argument(
            "--aux_athero_weight",
            type=float,
            default=0.0,
            help="Weight for supervised atherosclerosis BCE loss on latent mu(X). Suggested first try: 1.0 to 3.0."
        )
        parser.add_argument(
            "--aux_athero_pos_weight",
            type=float,
            default=1.0,
            help="Positive-class BCE weight for atherosclerosis classifier. Use >1 if positives are rare."
        )
        parser.add_argument(
            "--aux_athero_dropout",
            type=float,
            default=0.15,
            help="Dropout inside the latent atherosclerosis classifier head."
        )

        parser.add_argument(
            "--aux_predict_plaque",
            action="store_true",
            help="Predict plaque volumes from latent as an auxiliary task."
        )
        parser.add_argument(
            "--aux_plaque_cols",
            nargs=4,
            default=[
                "plaque_volume_right_carotid_corrected",
                "plaque_volume_left_carotid_corrected",
                "plaque_volume_right_femoral_corrected",
                "plaque_volume_left_femoral_corrected",
            ],
            help="4 plaque target columns for latent auxiliary regression."
        )
        parser.add_argument(
            "--aux_plaque_norm",
            type=str,
            default="log1p_z",
            choices=["none", "log1p_z"],
            help="Normalization for auxiliary plaque targets."
        )
        parser.add_argument(
            "--aux_plaque_weight",
            type=float,
            default=0.25,
            help="Weight for auxiliary plaque regression loss."
        )

        parser.add_argument(
            "--mesh_use_knn",
            action="store_true",
            help="Augment face-adjacency edges with kNN edges in the mesh encoder."
        )
        parser.add_argument(
            "--mesh_knn_k",
            type=int,
            default=8,
        )
        parser.add_argument(
            "--mesh_knn_max_points",
            type=int,
            default=4096,
        )

        parser.add_argument(
            "--age_bin_candidates_json",
            type=str,
            default=None,
            help="JSON file with candidate 5-edge age-bin sets for --phase search_age_bins."
        )
        parser.add_argument(
            "--age_search_epochs",
            type=int,
            default=8,
            help="Short training length per candidate during age-bin search."
        )

        parser.add_argument(
            "--age_search_source",
            type=str,
            default="json",
            choices=["json", "prevalence", "dynamic"],
            help="How to create age-bin candidates during --phase search_age_bins."
        )
        parser.add_argument(
            "--age_auto_n_candidates",
            type=int,
            default=10,
            help="Maximum number of automatically generated prevalence-based candidates."
        )
        parser.add_argument(
            "--age_auto_min_bin_size",
            type=int,
            default=80,
            help="Minimum number of samples required in each auto-generated age bin."
        )
        parser.add_argument(
            "--age_auto_smooth_window",
            type=int,
            default=5,
            help="Window size in years for smoothing the age-prevalence curve."
        )

        # ---- Conditioning extras (atherosclerosis / plaque volume) ----
        parser.add_argument("--cond_sep", type=str, default=None,
                            help="Optional: force separator for --cond_tsv. Default: auto-detect tab vs comma.")

        parser.add_argument("--cond_athero", action="store_true",
                            help="Include 4 binary atherosclerosis flags in cond vector.")
        parser.add_argument("--cond_athero_cols", nargs=4,
                            default=["ho_carotis", "ve_carotis", "ho_fem", "ve_fem"],
                            help="4 column names (0/1). Order: ho_carotis ve_carotis ho_fem ve_fem")

        parser.add_argument("--cond_plaque", action="store_true",
                            help="Include 4 plaque volume values in cond vector.")
        parser.add_argument("--cond_plaque_cols", nargs=4,
                            default=[
                                "plaque_volume_right_carotid_corrected",
                                "plaque_volume_left_carotid_corrected",
                                "plaque_volume_right_femoral_corrected",
                                "plaque_volume_left_femoral_corrected",
                            ],
                            help="4 plaque volume column names. Order: ho_carotis ve_carotis ho_fem ve_fem")
        parser.add_argument("--cond_plaque_norm", type=str, default="log1p_z",
                            choices=["none", "log1p_z"],
                            help="Normalize plaque volumes before feeding to net. Recommended: log1p_z")

        # --- Atherosclerosis encoding mode ---
        parser.add_argument(
            "--cond_athero_mode",
            type=str,
            default="any",
            choices=["any", "4bit"],
            help="How to encode atherosclerosis. "
                "'any' = single dim (1 if any of the 4 cols is 1). "
                "'4bit' = keep all 4 dims."
        )

        # --- Sampling helper for athero_mode=any ---
        parser.add_argument(
            "--sample_athero_any",
            type=int,
            default=None,
            help="For --phase sample when --cond_athero_mode=any: 0 or 1."
        )

        # --- Training imbalance fix ---
        parser.add_argument(
            "--balance_athero",
            action="store_true",
            help="Oversample atherosclerosis-positive examples during training (WeightedRandomSampler)."
        )

        # conditioning
        parser.add_argument("--cond_scale", type=float, default=1.0,
                            help="Scale factor applied to conditional embedding (encoder/decoder).")

        # Helps sampling quality (recommended when you want random z sampling to work)
        parser.add_argument("--beta_warmup_epochs", type=int, default=0,
                            help="Warmup KL beta linearly from 0->--beta over this many epochs.")

        # ----------------- Sampling from trained model -----------------
        parser.add_argument("--which_epoch", type=str, default="latest",
                            help="Checkpoint tag: 'latest' or e.g. 'epoch_50'")
        parser.add_argument("--sample_n", type=int, default=8,
                            help="Number of samples to generate.")
        parser.add_argument("--sample_gender", type=str, default=None,
                            help="M or F")
        parser.add_argument("--sample_age", type=float, default=None,
                            help="Numeric age (will be binned).")
        parser.add_argument("--sample_age_bin", type=str, default=None,
                            help="One of: 18-35, 35-50, 50-60, 60+")
        parser.add_argument("--temperature", type=float, default=1.0,
                            help="Scale z (higher => more variety).")
        parser.add_argument("--sample_athero", nargs=4, type=int, default=None,
                            help="For --phase sample: 4 ints (0/1) for ho_carotis ve_carotis ho_fem ve_fem.")

        parser.add_argument("--sample_plaque", nargs=4, type=float, default=None,
                            help="For --phase sample: 4 floats (raw plaque volumes) for ho_carotis ve_carotis ho_fem ve_fem.")

        # ----------------- Supervised baseline phase -----------------
        parser.add_argument(
            "--baseline_test_cond_tsv",
            type=str,
            default=None,
            help="Held-out test TSV for --phase train_baseline."
        )
        parser.add_argument(
            "--baseline_models",
            nargs="+",
            default=["logreg", "ridge", "random_forest", "hist_gbm", "mlp"],
            help="Baseline models: logreg ridge svm_rbf random_forest hist_gbm mlp cnn pointnet all_sklearn all"
        )
        parser.add_argument(
            "--baseline_feature_set",
            type=str,
            default="shape_meta",
            choices=["shape", "meta", "shape_meta"],
            help="Features for sklearn baselines."
        )
        parser.add_argument("--baseline_calib_ratio", type=float, default=0.2)
        parser.add_argument("--baseline_n_points", type=int, default=2048)
        parser.add_argument("--baseline_epochs", type=int, default=40)
        parser.add_argument("--baseline_lr", type=float, default=1e-3)


        self.parser = parser

    def parse(self):
        opt = self.parser.parse_args()
        opt.gpu_ids = [int(i) for i in opt.gpu_ids.split(",") if str(i).strip() != ""]

        if opt.only_gender is not None:
            opt.only_gender = str(opt.only_gender).upper()

        # Parse boson params JSON
        if opt.boson_params:
            opt.boson_cfg = json.loads(opt.boson_params)
        else:
            opt.boson_cfg = None

        # Always keep age config available
        opt.age_bins = list(map(float, getattr(opt, "age_bins", [18, 35, 50, 60, 200])))
        opt.age_labels = _build_age_labels(opt.age_bins)

        # ----------------- Compute cond_dim -----------------
        if getattr(opt, "use_cond", False):
            extra = 0
            if getattr(opt, "cond_athero", False):
                mode = getattr(opt, "cond_athero_mode", "any")
                extra += (1 if mode == "any" else 4)
            if getattr(opt, "cond_plaque", False):
                extra += 4
            opt.cond_dim = 6 + extra
        else:
            opt.cond_dim = 0

        # In sample/test: auto-load training-time cond config if it exists
        if getattr(opt, "use_cond", False) and getattr(opt, "phase", "") in ("sample", "test"):
            stats_path = Path(opt.checkpoints_dir) / opt.name / "cond_stats.json"
            if stats_path.exists():
                stats = json.loads(stats_path.read_text())
                opt.cond_dim = int(stats.get("cond_dim", opt.cond_dim))
                opt.cond_athero = bool(stats.get("athero_cols", []))
                opt.cond_plaque = bool(stats.get("plaque_cols", []))
                opt.cond_athero_cols = stats.get("athero_cols", getattr(opt, "cond_athero_cols", None))
                opt.cond_plaque_cols = stats.get("plaque_cols", getattr(opt, "cond_plaque_cols", None))
                opt.cond_plaque_norm = stats.get("plaque_norm", getattr(opt, "cond_plaque_norm", "log1p_z"))
                opt.cond_athero_mode = stats.get("athero_mode", getattr(opt, "cond_athero_mode", "4bit"))
                opt.only_gender = stats.get("only_gender", getattr(opt, "only_gender", None))

                opt.age_bins = list(map(float, stats.get("age_bins", getattr(opt, "age_bins", [18, 35, 50, 60, 200]))))
                opt.age_labels = stats.get("age_labels", getattr(opt, "age_labels", _build_age_labels(opt.age_bins)))
                opt.cond_age_mode = stats.get("cond_age_mode", getattr(opt, "cond_age_mode", "hard"))

        return opt



class TrainOptions(BaseOptions):
    pass


class TestOptions(BaseOptions):
    pass


# =========================================
# Conditioning helpers (age + gender)
# =========================================

AGE_BINS = [18, 35, 50, 60, 200]

def n_age_dims(opt) -> int:
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

    st_col = opt.cond_stregkode_col or _pick_col(df, ["stregkode"], fallback_contains="stregkode")
    age_col = opt.cond_age_col or _pick_col(df, ["alder", "age"], fallback_contains="ald")
    g_col = opt.cond_gender_col or _pick_col(df, ["sex", "gender"], fallback_contains="sex")

    if st_col is None or age_col is None or g_col is None:
        raise KeyError(
            f"Could not find columns in {cond_path}. "
            f"stregkode={st_col}, age={age_col}, gender={g_col}. "
            f"Available: {list(df.columns)[:80]}"
        )

    athero_cols = list(getattr(opt, "cond_athero_cols", [])) if getattr(opt, "cond_athero", False) else []
    cond_plaque_cols = list(getattr(opt, "cond_plaque_cols", [])) if getattr(opt, "cond_plaque", False) else []
    aux_plaque_cols = list(getattr(opt, "aux_plaque_cols", [])) if getattr(opt, "aux_predict_plaque", False) else []

    use_cols = list(dict.fromkeys([st_col, age_col, g_col] + athero_cols + cond_plaque_cols + aux_plaque_cols))
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing metadata columns in {cond_path}: {missing}. "
            f"Available: {list(df.columns)[:80]}"
        )

    out = df[use_cols].copy()
    out = out.rename(columns={st_col: "stregkode_raw", age_col: "age_raw", g_col: "gender_raw"})

    out["stregkode_clean"] = out["stregkode_raw"].apply(clean_stregkode)
    out["age_num"] = pd.to_numeric(out["age_raw"], errors="coerce")
    g = out["gender_raw"].astype(str).str.strip().str.upper().replace({"MALE": "M", "FEMALE": "F"})
    out["gender_clean"] = g

    out = out.dropna(subset=["stregkode_clean", "age_num"]).copy()
    out = out[out["age_num"] >= 18].copy()
    out = out[out["gender_clean"].isin(["M", "F"])].copy()

    only_gender = getattr(opt, "only_gender", None)
    if only_gender is not None:
        only_gender = str(only_gender).upper()
        out = out[out["gender_clean"] == only_gender].copy()
        if len(out) == 0:
            raise ValueError(f"No rows left after filtering only_gender={only_gender} in {cond_path}")
        
    age_bins = list(map(float, getattr(opt, "age_bins", AGE_BINS)))
    age_labels = _build_age_labels(age_bins)
    out["age_bin"] = out["age_num"].apply(lambda a: _age_to_bin(a, age_bins, age_labels))
    out = out.dropna(subset=["age_bin"]).copy()

    if athero_cols:
        for c in athero_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(0, 1)
        out["athero_any"] = (out[athero_cols].max(axis=1) > 0.5).astype(np.float32)

    def _normalize_plaque(cols: list[str], norm_name: str):
        if not cols:
            return None, None, None

        P_raw = []
        for c in cols:
            P_raw.append(pd.to_numeric(out[c], errors="coerce").fillna(0.0).to_numpy(np.float32))
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

    age_labels = _build_age_labels(age_bins)

    cond_names = ["gender_M", "gender_F"] + [f"age_{lab}" for lab in age_labels]

    cond_dim = 2 + len(age_labels)
    if athero_cols:
        cond_dim += (1 if getattr(opt, "cond_athero_mode", "any") == "any" else 4)
    if cond_plaque_cols:
        cond_dim += 4


    if getattr(opt, "use_cond", False):
        extra = 0
        if getattr(opt, "cond_athero", False):
            mode = getattr(opt, "cond_athero_mode", "any")
            extra += (1 if mode == "any" else 4)
        if getattr(opt, "cond_plaque", False):
            extra += 4
        opt.cond_dim = 2 + n_age_dims(opt) + extra
    else:
        opt.cond_dim = 0

    conds = []
    groups = []
    aux_rows = []

    for row_idx, (_, row) in enumerate(out.iterrows()):
        cvec = np.zeros(cond_dim, dtype=np.float32)

        gender = row["gender_clean"]
        ab = row["age_bin"]

        cvec[2:2 + len(age_labels)] = _age_to_vec(float(row["age_num"]), opt)
        off = 2 + len(age_labels)

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

    if athero_cols:
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
            "sep": sep,
            "age_bins": age_bins,
            "age_labels": age_labels,
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
            "encoder_use_cond": bool(getattr(opt, "encoder_use_cond", False)),

            "athero_mode": getattr(opt, "cond_athero_mode", "any") if athero_cols else None,
        }
        stats_path = Path(opt.checkpoints_dir) / opt.name / "cond_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"[WARN] Could not write cond_stats.json: {e}", flush=True)

    return code2cond, code2group, code2aux


# =========================================
# Dataset + collate
# =========================================
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
        edge_features = torch.from_numpy(V.astype(np.float32).T).unsqueeze(0)  # (1, 3, N)

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
            "edge_features": edge_features,       # (1, 3, N)
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

        # ---------- conditioning ----------
        self.use_cond = bool(getattr(self.opt, "use_cond", False))
        self.use_aux_plaque = bool(getattr(self.opt, "aux_predict_plaque", False))

        self.cond = None
        self.group = None
        self.plaque_target = None
        self.age_scalar = None

        if self.use_cond or self.use_aux_plaque:
            code2cond, code2group, code2aux = build_age_gender_lookup(self.opt)

            orig_n = len(self.names)
            keep = []
            conds = []
            groups = []
            plaque_targets = []
            age_scalars = []

            for i, nm in enumerate(self.names):
                code = clean_stregkode(nm)
                c = code2cond.get(code)
                aux = code2aux.get(code, {})

                if self.use_cond and c is None:
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

            keep = np.asarray(keep, dtype=np.int64)

            if keep.size != orig_n:
                self.verts = self.verts[keep]
                self.verts_mask = self.verts_mask[keep]
                self.faces = self.faces[keep]
                self.faces_mask = self.faces_mask[keep]
                self.names = self.names[keep]
                if self.points is not None:
                    self.points = self.points[keep]
                print(f"[COND] Kept {len(self.names)}/{orig_n} with metadata (dropped {orig_n - len(self.names)}).")

            if self.use_cond:
                self.cond = np.stack(conds, axis=0).astype(np.float32)
                self.group = np.array(groups, dtype=object)

            if self.use_aux_plaque:
                self.plaque_target = np.stack(plaque_targets, axis=0).astype(np.float32)
                self.age_scalar = np.asarray(age_scalars, dtype=np.float32)

        print("vertices shape:", self.verts.shape)
        print("verts_mask shape:", self.verts_mask.shape)
        print("faces shape:", self.faces.shape)
        print("faces_mask shape:", self.faces_mask.shape)
        print("points:", None if self.points is None else self.points.shape)

        print(f"[DATA] Loaded preprocessed NPZ: {npz_path}")
        print(f"[DATA]   meshes={self.verts.shape[0]}, Vmax={self.verts.shape[1]}, Fmax={self.faces.shape[1]}")

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

            feats_np = P.T                            # (3, n_points)
        else:
            feats_np = V.astype(np.float32).T         # (3, V_real)

        edge_features = torch.from_numpy(feats_np).unsqueeze(0)    # (1, 3, E)

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
            "edge_features": edge_features,     # (1, 3, E)
        }

        if self.use_cond and self.cond is not None:
            data["cond"] = torch.from_numpy(self.cond[idx])
            data["group"] = str(self.group[idx])

        if self.use_aux_plaque and self.plaque_target is not None:
            data["plaque_target"] = torch.from_numpy(self.plaque_target[idx])
            data["age_scalar"] = torch.tensor(self.age_scalar[idx], dtype=torch.float32)

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

@torch.no_grad()
def evaluate_subset(model, subset, opt):
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=mesh_collate_fn,
    )

    rows = []
    for data in loader:
        model.set_input(data)
        model.test()
        rows.append(dict(model.last_diag))

    if len(rows) == 0:
        return {"loss": float("inf")}

    df = pd.DataFrame(rows)
    out = {}
    for k in ["loss", "chamfer", "kl", "plaque_loss"]:
        if k in df.columns:
            out[k] = float(df[k].mean())
    return out

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
    from diagnose_mesh_vae import evaluate_search_candidate_from_indices

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

def split_dataset(dataset, ratio, seed):
    n = len(dataset)
    n_val = int(n * ratio)
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    from torch.utils.data import Subset

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# =========================================
# Redetailing
# =========================================

def redetail_to_simplified_edges(
    mesh_recon,
    mesh_simplified,
    max_target_edges: int = 100_000,
    max_steps: int = 4,
):
    """
    Upsample recon mesh until its edge count is in the same ballpark as the
    *simplified* mesh, but never above max_target_edges and at most max_steps
    subdivision steps.

    This is deliberately conservative so it doesn't hang or explode in size.
    """
    import open3d as o3d

    mesh_recon = clean_mesh(mesh_recon)
    mesh_simplified = clean_mesh(mesh_simplified)

    target_edges = mesh_stats_o3d(mesh_simplified)["num_edges"]
    target_edges = min(target_edges, max_target_edges)

    print(f"[REDETAIL] target edges from simplified = {target_edges}")

    prev_edges = -1
    m = mesh_recon
    for step in range(max_steps):
        cur_edges = mesh_stats_o3d(m)["num_edges"]
        print(f"[REDETAIL] step {step}: edges={cur_edges}")

        # close enough or no progress → stop
        if cur_edges >= target_edges or cur_edges == prev_edges:
            break

        prev_edges = cur_edges
        m = m.subdivide_midpoint(number_of_iterations=1)
        m = clean_mesh(m)

    return m

# =========================================
# Training loop
# =========================================

def train(opt):
    print("\n========== TRAIN: MeshVAE ==========\n")

    if getattr(opt, "use_preprocessed", False):
        dataset = PreprocessedMeshVAEDataset(opt.dataroot, opt)
    else:
        dataset = MeshVAEDataset(opt.dataroot, opt)

    if hasattr(dataset, "meshes"):
        print("[DATA] First 5 files:", [str(p) for p, _ in dataset.meshes[:5]])
    elif hasattr(dataset, "names"):
        print("[DATA] First 5 files:", [str(n) for n in dataset.names[:5]])
    else:
        print("[DATA] No 'meshes' or 'names' attribute found in dataset.")

    train_data, val_data = split_dataset(dataset, opt.split_ratio, opt.split_seed)

    sampler = None
    if getattr(opt, "balance_athero", False) and getattr(opt, "use_cond", False) and getattr(opt, "cond_athero", False):
        base_ds = dataset
        if hasattr(base_ds, "cond") and base_ds.cond is not None:
            idxs = train_data.indices if hasattr(train_data, "indices") else list(range(len(train_data)))
            mode = getattr(opt, "cond_athero_mode", "any")

            labels = []
            for j in idxs:
                c = base_ds.cond[j]
                if mode == "any":
                    y = int(c[6] > 0.5)
                else:
                    y = int(np.max(c[6:10]) > 0.5)
                labels.append(y)

            labels = np.asarray(labels, dtype=np.int64)
            cnt0 = int((labels == 0).sum())
            cnt1 = int((labels == 1).sum())
            print(f"[BALANCE] train split athero_any counts: 0={cnt0}  1={cnt1}")

            if cnt0 > 0 and cnt1 > 0:
                w0 = 1.0 / cnt0
                w1 = 1.0 / cnt1
                weights = np.where(labels == 1, w1, w0).astype(np.float64)
                sampler = WeightedRandomSampler(
                    weights=torch.from_numpy(weights),
                    num_samples=len(weights),
                    replacement=True,
                )

    train_loader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=opt.num_threads,
        collate_fn=mesh_collate_fn,
    )

    model = MeshVAEModel(opt)
    total_iters = 0

    metrics_dir = Path(opt.checkpoints_dir) / opt.name / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_csv = metrics_dir / f"train_metrics_{run_id}.csv"
    val_csv = metrics_dir / f"val_metrics_{run_id}.csv"
    train_png = metrics_dir / f"train_metrics_{run_id}.png"

    train_rows = []
    val_rows = []
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        print(f"\n--- Epoch {epoch} ---")

        if getattr(opt, "beta_warmup_epochs", 0) > 0:
            warm = float(opt.beta_warmup_epochs)
            frac = min(1.0, (epoch - opt.epoch_count + 1) / warm)
            cur_beta = float(opt.beta) * frac
            model.vi.beta = cur_beta
            print(f"[BETA] warmup beta={cur_beta:g} (frac={frac:.3f})")

        for i, data in enumerate(train_loader):
            total_iters += 1
            model.set_input(data)
            model.optimize_parameters()

            row = {"epoch": epoch, "iter": i, "global_iter": total_iters}
            row.update(getattr(model, "last_diag", {}) or {})
            if "loss" not in row and hasattr(model, "loss"):
                row["loss"] = float(model.loss.detach().cpu())
            train_rows.append(row)

            if i % 10 == 0:
                print(f"[epoch {epoch}] iter {i} loss = {model.loss.item():.4f}")

        model.update_learning_rate()
        model.save_network(f"epoch_{epoch}")
        model.save_network("latest")

        val_metrics = evaluate_subset(model, val_data, opt)
        val_metrics["epoch"] = epoch
        val_rows.append(val_metrics)

        val_loss = float(val_metrics.get("loss", float("inf")))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch

        print("[VAL] " + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items() if k != "epoch"))

        run_test(opt, model, val_data)

        pd.DataFrame(train_rows).to_csv(train_csv, index=False)
        pd.DataFrame(val_rows).to_csv(val_csv, index=False)
        print(f"[METRICS] wrote train -> {train_csv}")
        print(f"[METRICS] wrote val   -> {val_csv}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            df = pd.DataFrame(train_rows)
            plt.figure()
            if "global_iter" in df.columns and "loss" in df.columns:
                plt.plot(df["global_iter"], df["loss"], label="loss")
            if "global_iter" in df.columns and "chamfer" in df.columns:
                plt.plot(df["global_iter"], df["chamfer"], label="chamfer")
            if "global_iter" in df.columns and "kl" in df.columns:
                plt.plot(df["global_iter"], df["kl"], label="kl")
            if "global_iter" in df.columns and "plaque_loss" in df.columns:
                plt.plot(df["global_iter"], df["plaque_loss"], label="plaque_loss")

            plt.xlabel("step")
            plt.ylabel("value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(train_png)
            plt.close()
            print(f"[METRICS] wrote plot -> {train_png}")
        except Exception as e:
            print(f"[METRICS] plot skipped: {e}")

    print("\nTraining complete.")
    train_idx = np.asarray(getattr(train_data, "indices", np.arange(len(train_data))), dtype=np.int64)
    val_idx = np.asarray(getattr(val_data, "indices", np.arange(len(val_data))), dtype=np.int64)

    return {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "model": model,
        "dataset": dataset,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


# =========================================
# Baseline Models
# =========================================

def train_baseline_phase(opt):
    """
    Optional wrapper around meshvae_baselines.py.

    This assumes meshvae_baselines.py is in the same folder as train_mesh_vae.py.
    It runs supervised baseline classifiers without touching the VAE code.
    """
    print("\n========== TRAIN_BASELINE ==========\n")

    from meshvae_baselines import run_baseline_from_dataset

    # Build training dataset with the normal metadata machinery.
    train_ds = PreprocessedMeshVAEDataset(opt.dataroot, opt)

    test_ds = None
    if getattr(opt, "baseline_test_cond_tsv", None):
        import copy
        test_opt = copy.copy(opt)
        test_opt.cond_tsv = opt.baseline_test_cond_tsv
        test_ds = PreprocessedMeshVAEDataset(opt.dataroot, test_opt)

    models = [str(m).lower() for m in getattr(opt, "baseline_models", [])]
    if "all" in models:
        models = ["logreg", "ridge", "svm_rbf", "random_forest", "hist_gbm", "mlp", "cnn", "pointnet"]
    elif "all_sklearn" in models:
        models = ["logreg", "ridge", "svm_rbf", "random_forest", "hist_gbm", "mlp"]

    out_dir = (
        Path(opt.checkpoints_dir)
        / opt.name
        / "baselines"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    run_baseline_from_dataset(
        train_ds,
        opt,
        out_dir,
        test_dataset=test_ds,
        model_names=models,
        feature_set=str(getattr(opt, "baseline_feature_set", "shape_meta")),
        calib_ratio=float(getattr(opt, "baseline_calib_ratio", 0.2)),
        seed=int(getattr(opt, "split_seed", 3)),
        n_points=int(getattr(opt, "baseline_n_points", 2048)),
        epochs=int(getattr(opt, "baseline_epochs", 40)),
        batch_size=int(getattr(opt, "batch_size", 16)),
        lr=float(getattr(opt, "baseline_lr", 1e-3)),
    )

# =========================================
# Testing / reconstruction
# =========================================

def run_test(opt, model, val_subset):
    print("\n========== TESTING ==========\n")

    loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=mesh_collate_fn,
    )

    save_root = Path(opt.checkpoints_dir) / opt.name / "reconstructions"
    save_root.mkdir(parents=True, exist_ok=True)

    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

    for i, data in enumerate(loader):
        sample_path = Path(data["path"][0])
        base_name = sample_path.stem  # e.g. "Scan_aug_010"

        print(f"\n[TEST] sample {i}")
        print(f"       source path : {sample_path}")
        print(f"       base name   : {base_name}")

        # -------------------------------------------------
        # Forward pass through VAE
        # -------------------------------------------------
        model.set_input(data)
        out = model.test()

        x = out["x"][0]        # (C, E)
        x_mean = out["x_mean"][0]  # (C, E)
        print(f"[TEST] x shape      = {tuple(x.shape)}")
        print(f"[TEST] x_mean shape = {tuple(x_mean.shape)}")

        # -------------------------------------------------
        # POINT CLOUD MODE: save original + recon as .ply
        # -------------------------------------------------
        if getattr(opt, "use_pointcloud", False):
            # original point cloud
            pc_orig = x.permute(1, 0).cpu().numpy()      # (N, 3)
            pcd_orig = o3d.geometry.PointCloud()
            pcd_orig.points = o3d.utility.Vector3dVector(pc_orig.astype(np.float64))
            orig_path = save_root / f"{base_name}_orig_points.ply"
            o3d.io.write_point_cloud(str(orig_path), pcd_orig)
            print(f"[TEST] wrote original point cloud -> {orig_path.name}")

            # reconstructed point cloud
            pc_recon = x_mean.permute(1, 0).cpu().numpy()  # (N, 3)
            pcd_recon = o3d.geometry.PointCloud()
            pcd_recon.points = o3d.utility.Vector3dVector(pc_recon.astype(np.float64))
            recon_path = save_root / f"{base_name}_recon_points.ply"
            o3d.io.write_point_cloud(str(recon_path), pcd_recon)
            print(f"[TEST] wrote recon point cloud    -> {recon_path.name}")

            # done for this sample
            continue

        # -------------------------------------------------
        # MESH MODE: save simplified input mesh + recon mesh
        # -------------------------------------------------
        # 1) input mesh (simplified)
        simp_mesh = data["simplified"][0]
        simp_path = save_root / f"{base_name}_input_mesh.obj"
        o3d.io.write_triangle_mesh(str(simp_path), simp_mesh)
        print(f"[TEST] wrote input mesh -> {simp_path.name}")

        # 2) reconstructed mesh from x_mean using stored faces/mask
        batch = {
            "verts": out["x_mean"].permute(0, 2, 1),  # (1, E, C)
            "verts_mask": data["verts_mask"],         # (1, V)
            "faces": data["faces"],                   # (1, F, 3)
            "faces_mask": data["faces_mask"],         # (1, F)
        }
        recon_mesh = tensors_to_meshes(batch)[0]
        recon_path = save_root / f"{base_name}_recon_mesh.obj"
        o3d.io.write_triangle_mesh(str(recon_path), recon_mesh)
        print(f"[TEST] wrote recon mesh -> {recon_path.name}")


def sample_conditional(opt):
    if not getattr(opt, "use_cond", False):
        raise ValueError("Sampling requires --use_cond.")
    if not getattr(opt, "use_pointcloud", False):
        raise ValueError("Sampling expects --use_pointcloud (decoder outputs xyz points).")

    # load cond stats if present (needed for plaque normalization)
    stats_path = Path(opt.checkpoints_dir) / opt.name / "cond_stats.json"
    stats = None
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
        except Exception:
            stats = None

    opt.is_train = False
    model = MeshVAEModel(opt)
    model.load_network(opt.which_epoch)

    # resolve gender / age-bin using training-time age labels if available
    labels = list(stats.get("age_labels", getattr(opt, "age_labels", AGE_LABELS))) if stats else list(getattr(opt, "age_labels", AGE_LABELS))
    bins = list(stats.get("age_bins", getattr(opt, "age_bins", AGE_BINS))) if stats else list(getattr(opt, "age_bins", AGE_BINS))

    gender = (opt.sample_gender or "F").strip().upper()
    if getattr(opt, "only_gender", None) is not None:
        only_gender = str(opt.only_gender).upper()
        if gender != only_gender:
            raise ValueError(f"This checkpoint is gender-filtered with only_gender={only_gender}, but sample_gender={gender}")
    if opt.sample_age_bin:
        age_bin = opt.sample_age_bin.strip()
    else:
        age_bin = _age_to_bin(opt.sample_age, bins, labels) or labels[min(2, len(labels) - 1)]

    if gender not in ("M", "F"):
        raise ValueError("sample_gender must be M or F")
    if age_bin not in labels:
        raise ValueError(f"sample_age_bin must be one of {labels}")

    # cond_dim: trust opt (parse() already tries to load from cond_stats.json)
    cond_dim = int(getattr(opt, "cond_dim", 6))

    # base one-hot [M, F, age0, age1, age2, age3]
    c = np.zeros(cond_dim, dtype=np.float32)
    c[0 if gender == "M" else 1] = 1.0

    age_map = {lab: 2 + i for i, lab in enumerate(labels)}
    c[age_map[age_bin]] = 1.0

    off = 6

    # ---- athero ----
    if getattr(opt, "cond_athero", False):
        mode = getattr(opt, "cond_athero_mode", "any")

        if mode == "any":
            # prefer explicit --sample_athero_any
            if opt.sample_athero_any is not None:
                a_any = int(opt.sample_athero_any)
            else:
                # fallback: if user provided 4 bits, collapse to any-bit
                vals4 = opt.sample_athero if opt.sample_athero is not None else [0, 0, 0, 0]
                vals4 = np.asarray(vals4, dtype=np.float32).clip(0, 1)
                a_any = int(vals4.max() > 0.5)

            c[off] = float(a_any)
            off += 1

        else:  # "4bit"
            vals4 = opt.sample_athero if opt.sample_athero is not None else [0, 0, 0, 0]
            if len(vals4) != 4:
                raise ValueError("--sample_athero must have 4 ints (0/1) when --cond_athero_mode=4bit.")
            vals4 = np.asarray(vals4, dtype=np.float32).clip(0, 1)
            c[off:off+4] = vals4
            off += 4


    # ---- plaque volumes ----
    if getattr(opt, "cond_plaque", False):
        raw = opt.sample_plaque if opt.sample_plaque is not None else [0.0, 0.0, 0.0, 0.0]
        if len(raw) != 4:
            raise ValueError("--sample_plaque must have 4 floats.")
        raw = np.asarray(raw, dtype=np.float32)

        norm = getattr(opt, "cond_plaque_norm", "log1p_z")
        if norm == "log1p_z":
            v = np.log1p(raw)
            if stats and stats.get("plaque_mean") is not None and stats.get("plaque_std") is not None:
                mu = np.asarray(stats["plaque_mean"], dtype=np.float32)
                sd = np.asarray(stats["plaque_std"], dtype=np.float32)
                v = (v - mu) / sd
            # else: fallback = log1p only
        else:
            v = raw

        c[off:off+4] = v.astype(np.float32)
        off += 4

    N = int(opt.sample_n)
    cond = torch.from_numpy(np.repeat(c[None, :], N, axis=0)).to(model.device)

    net = model.net.module if isinstance(model.net, torch.nn.DataParallel) else model.net

    with torch.no_grad():
        pz = net.prior(N, int(getattr(opt, "latent_dim", 64)), device=model.device, cond=cond)
        z = (pz.rsample() if hasattr(pz, "rsample") else pz.sample()).to(model.device)
        z = z * float(getattr(opt, "temperature", 1.0))
        x_mean = net.decoder(z, int(opt.ninput_edges), cond=cond)  # (N,3,E)

    out_dir = Path(opt.checkpoints_dir) / opt.name / "samples_cond"
    out_dir.mkdir(parents=True, exist_ok=True)

    import open3d as o3d
    # make filename informative
    tag = f"{gender}_{age_bin}"
    if getattr(opt, "cond_athero", False):
        mode = getattr(opt, "cond_athero_mode", "any")
        if mode == "any":
            a_any = opt.sample_athero_any
            if a_any is None:
                vals4 = opt.sample_athero if opt.sample_athero is not None else [0,0,0,0]
                a_any = int(np.asarray(vals4).max() > 0.5)
            tag += f"_ath{int(a_any)}"
        else:
            a = (opt.sample_athero if opt.sample_athero is not None else [0,0,0,0])
            tag += f"_ath{''.join(str(int(x)) for x in a)}"
    if getattr(opt, "cond_plaque", False):
        p = (opt.sample_plaque if opt.sample_plaque is not None else [0,0,0,0])
        tag += f"_plq{'-'.join(f'{x:g}' for x in p)}"

    for i in range(N):
        pts = x_mean[i].permute(1, 0).detach().cpu().numpy()  # (E,3)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        out_path = out_dir / f"{tag}_T{opt.temperature:g}_{opt.which_epoch}_s{i:03d}.ply"
        o3d.io.write_point_cloud(str(out_path), pcd)

    print(f"[SAMPLE] wrote {N} samples -> {out_dir}")



# =========================================
# Main
# =========================================

if __name__ == "__main__":
    opt = BaseOptions().parse()

    os.makedirs(Path(opt.checkpoints_dir) / opt.name, exist_ok=True)

    if opt.phase == "preprocess":
        # 🔹 Offline preprocessing only – no training
        n_points = opt.n_points if getattr(opt, "n_points", 0) > 0 else None
        print("\n========== PREPROCESS (offline) ==========\n")
        preprocess_obj_folder(
            input_root=opt.dataroot,
            out_root=opt.dataroot,
            target_tris=opt.target_tris,
            save_npz=True,
            n_points=n_points,
        )
        print("\nPreprocessing complete. You can now run with --use_preprocessed.\n")

    elif opt.phase == "train":
        opt.is_train = True
        train(opt)

    elif opt.phase == "test":
        opt.is_train = False
        model = MeshVAEModel(opt)
        dataset = PreprocessedMeshVAEDataset(opt.dataroot, opt) if getattr(opt, "use_preprocessed", False) \
                  else MeshVAEDataset(opt.dataroot, opt)
        model.load_network(opt.which_epoch)
        run_test(opt, model, dataset)

    elif opt.phase == "sample":
        opt.is_train = False
        sample_conditional(opt)

    elif opt.phase == "search_age_bins":
        opt.is_train = True
        search_best_age_bins(opt)

    elif opt.phase == "train_baseline":
        opt.is_train = False
        train_baseline_phase(opt)

