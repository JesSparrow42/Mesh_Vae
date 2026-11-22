#!/usr/bin/env python3
"""
Diagnostic pipeline for a conditional MeshVAE trained on point-cloud meshes.

What this script does
---------------------
1. Loads a trained conditional MeshVAE checkpoint.
2. Uses a calibration split (typically your training TSV) to:
   - infer empirical healthy/sick atherosclerosis patterns per demographic group,
   - generate class-conditioned reference samples from the model,
   - fit decision thresholds.
3. Scores a held-out test split by comparing each test sample to generated
   healthy/sick reference samples using Chamfer distance.
4. Also computes class-conditional reconstruction / ELBO scores as a fallback,
   then automatically selects the best score on calibration unless a score is
   requested explicitly.
5. Writes per-sample CSVs and a JSON summary with metrics.

Notes
-----
- This script assumes the model was trained in point-cloud mode
  (--use_preprocessed --use_pointcloud --use_cond).
- Thresholds are fit on the calibration split, NOT on the test split.
- For cond_athero_mode=4bit, healthy is 0000 and sick is treated as the set of
  non-zero patterns observed in the calibration split for the same age/gender
  group. This preserves location information while still producing a binary
  healthy-vs-sick decision.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

from .options import TestOptions
from .datasets import PreprocessedMeshVAEDataset, ClinicalFeatureMeshVAEDataset
from .conditioning import AGE_LABELS, cond_dim_from_opt
from .model import MeshVAEModel
from .losses import chamfer_distance_subsample

UNDER50_BINS = {"18-35", "35-50"}
OVER50_BINS = {"50-60", "60+"}



# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

class DiagnosticOptions(TestOptions):
    def __init__(self):
        super().__init__()
        p = self.parser

        def add_once(*args, **kwargs):
            existing = {opt for action in p._actions for opt in getattr(action, "option_strings", [])}
            requested = {a for a in args if isinstance(a, str) and a.startswith("--")}
            if requested & existing:
                return None
            return p.add_argument(*args, **kwargs)

        add_once("--calib_cond_tsv", type=str, required=True,
                       help="Calibration TSV. Use the train split here.")
        add_once("--test_cond_tsv", type=str, required=True,
                       help="Held-out test TSV.")
        add_once("--results_subdir", type=str, default="diagnostics",
                       help="Subdirectory under checkpoints/<name>/ for outputs.")

        # Defaults from your preferred diagnostic command
        add_once("--metric_npoints", type=int, default=4096,
                       help="Subsample size used for Chamfer during diagnostics.")
        add_once("--metric_chunk", type=int, default=4096,
                       help="Chunk size inside Chamfer computation.")
        add_once("--distance_batch_size", type=int, default=2,
                       help="How many reference samples to compare at once.")

        add_once("--samples_per_class", type=int, default=64,
                       help="Total generated reference samples per class per group.")
        add_once("--distance_temperature", type=float, default=0.05,
                       help="Temperature for distance-based pseudo-likelihood.")

        add_once("--score_method", type=str, default="auto",
                       choices=["auto", "sample_prob_sick", "sample_margin", "recon_margin", "elbo_margin"],
                       help="Which score to threshold. 'auto' picks the best one on calibration.")

        add_once("--threshold_metric", type=str, default="balanced_accuracy",
                       choices=["balanced_accuracy", "youden_j"],
                       help="Criterion for threshold selection.")

        add_once("--use_group_priors", action="store_true",
                       help="Use age/gender-specific prevalence as prior in pseudo-probability.")

        add_once("--calib_max_samples", type=int, default=0,
                       help="Optional cap for calibration samples (0 = use all).")
        add_once("--test_max_samples", type=int, default=0,
                       help="Optional cap for test samples (0 = use all).")
        add_once("--seed", type=int, default=42)

        add_once("--save_reference_samples", action="store_true",
                       help="Save generated reference point clouds as .pt for inspection.")

        add_once("--prior_mode", type=str, default="global",
                       choices=["flat", "global", "group"],
                       help="Prior used when converting distance scores to a sick probability.")

        add_once("--neutralize_plaque_in_refs", action="store_true",
                       help="Use one neutral plaque vector for both healthy and sick refs.")

        add_once(
            "--policy_mode",
            type=str,
            default="global",
            choices=["global", "age", "prevalence"],
            help=(
                "Thresholding mode: "
                "'global' = one threshold for all samples, "
                "'age' = separate thresholds for under/over-50, "
                "'prevalence' = separate thresholds for low/high prevalence groups."
            ),
        )
        add_once(
            "--prevalence_band_mode",
            type=str,
            default="global",
            choices=["global", "median", "fixed"],
            help="How to choose the prevalence cutoff when --policy_mode prevalence.",
        )
        add_once(
            "--prevalence_cutoff",
            type=float,
            default=None,
            help="Used only when --prevalence_band_mode fixed.",
        )
        
# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MetaInfo:
    group: str
    gender: str
    age_bin: str
    sick: int
    pattern: Tuple[int, ...]
    pattern_str: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_net(model: MeshVAEModel):
    return model.net.module if isinstance(model.net, torch.nn.DataParallel) else model.net


def load_cond_stats_into_opt(opt) -> None:
    stats_path = Path(opt.checkpoints_dir) / opt.name / "cond_stats.json"
    if not stats_path.exists():
        return

    stats = json.loads(stats_path.read_text())
    opt.cond_dim = int(stats.get("cond_dim", getattr(opt, "cond_dim", 0)))
    if int(stats.get("ninput_edges", 0) or 0) > 0:
        opt.ninput_edges = int(stats["ninput_edges"])
    opt.cond_athero = bool(stats.get("athero_cols", []))
    opt.cond_plaque = bool(stats.get("plaque_cols", []))
    opt.cond_athero_cols = stats.get("athero_cols", getattr(opt, "cond_athero_cols", []))
    opt.cond_plaque_cols = stats.get("plaque_cols", getattr(opt, "cond_plaque_cols", []))
    opt.cond_plaque_norm = stats.get("plaque_norm", getattr(opt, "cond_plaque_norm", "log1p_z"))
    opt.cond_athero_mode = stats.get("athero_mode", getattr(opt, "cond_athero_mode", "any"))

    opt.age_bins = stats.get("age_bins", [18, 35, 50, 60, 200])
    opt.age_labels = stats.get("age_labels", AGE_LABELS)
    opt.cond_age_source = stats.get("cond_age_source", getattr(opt, "cond_age_source", "auto"))
    opt.cond_agecut_col = stats.get("agecut_col", getattr(opt, "cond_agecut_col", None))
    opt.cond_age_mode = stats.get("cond_age_mode", "hard")
    opt.only_gender = stats.get("only_gender", getattr(opt, "only_gender", None))

    # Rebuild optional supervised latent classifier head at diagnosis time.
    opt.aux_predict_athero = bool(stats.get("aux_predict_athero", getattr(opt, "aux_predict_athero", False)))
    opt.aux_athero_mode = stats.get("aux_athero_mode", getattr(opt, "aux_athero_mode", "any"))
    opt.aux_athero_weight = float(stats.get("aux_athero_weight", getattr(opt, "aux_athero_weight", 0.0)))
    opt.aux_athero_pos_weight = float(stats.get("aux_athero_pos_weight", getattr(opt, "aux_athero_pos_weight", 1.0)))
    opt.aux_athero_dropout = float(stats.get("aux_athero_dropout", getattr(opt, "aux_athero_dropout", 0.15)))

    opt.use_cond_prior = bool(stats.get("use_cond_prior", getattr(opt, "use_cond_prior", False)))
    opt.demo_cond_dim = int(stats.get("demo_cond_dim", 2 + len(getattr(opt, "age_labels", AGE_LABELS))))
    opt.encoder_cond_mode = stats.get("encoder_cond_mode", getattr(opt, "encoder_cond_mode", "none"))
    opt.encoder_use_cond = False
    opt.adv_predict_age = bool(stats.get("adv_predict_age", getattr(opt, "adv_predict_age", False)))
    opt.adv_age_weight = float(stats.get("adv_age_weight", getattr(opt, "adv_age_weight", 0.0)))
    opt.adv_age_hidden = int(stats.get("adv_age_hidden", getattr(opt, "adv_age_hidden", 128)))
    opt.adv_age_dropout = float(stats.get("adv_age_dropout", getattr(opt, "adv_age_dropout", 0.15)))
    opt.adv_age_grl_lambda = float(stats.get("adv_age_grl_lambda", getattr(opt, "adv_age_grl_lambda", 1.0)))

    # Clinical feature sanity-check metadata, if present.  This lets diagnose
    # infer --feature_source from the checkpoint when you forget to pass it.
    stats_feature = str(stats.get("feature_source", getattr(opt, "feature_source", "scan")))
    if stats_feature != "scan" and str(getattr(opt, "feature_source", "scan")) == "scan":
        opt.feature_source = stats_feature
    if int(stats.get("clinical_points", 0) or 0) > 0:
        opt.clinical_points = int(stats.get("clinical_points"))

    clinical_stats_path = Path(opt.checkpoints_dir) / opt.name / "clinical_feature_stats.json"
    if clinical_stats_path.exists():
        try:
            clinical_stats = json.loads(clinical_stats_path.read_text())
            stats_feature = str(clinical_stats.get("feature_source", getattr(opt, "feature_source", "scan")))
            if stats_feature != "scan" and str(getattr(opt, "feature_source", "scan")) == "scan":
                opt.feature_source = stats_feature
            if int(clinical_stats.get("clinical_points", 0) or 0) > 0:
                opt.clinical_points = int(clinical_stats.get("clinical_points"))
        except Exception as e:
            print(f"[CLINICAL] WARNING: could not read {clinical_stats_path}: {e}")

def agebin_to_index(age_bin: str, opt) -> int:
    labels = list(getattr(opt, "age_labels", AGE_LABELS))
    return 2 + {lab: i for i, lab in enumerate(labels)}[age_bin]

def n_age_dims(opt) -> int:
    return len(list(getattr(opt, "age_labels", AGE_LABELS)))

def cond_to_meta(cond: np.ndarray, opt) -> MetaInfo:
    gender = "M" if cond[0] >= cond[1] else "F"
    age_labels = list(getattr(opt, "age_labels", AGE_LABELS))
    na = n_age_dims(opt)
    age_idx = int(np.argmax(cond[2:2 + na]))
    age_bin = age_labels[age_idx]

    athero_slice, _ = cond_layout(opt)
    if athero_slice is None:
        pattern = (0,)
        sick = 0
    else:
        bits = tuple(int(x > 0.5) for x in cond[athero_slice])
        pattern = bits
        sick = int(any(bits))

    pattern_str = "".join(str(x) for x in pattern)
    return MetaInfo(
        group=f"{gender}_{age_bin}",
        gender=gender,
        age_bin=age_bin,
        sick=sick,
        pattern=pattern,
        pattern_str=pattern_str,
    )


def split_group(group: str) -> Tuple[str, str]:
    gender, age_bin = group.split("_", 1)
    return gender, age_bin


def cond_layout(opt) -> Tuple[slice | None, slice | None]:
    off = 2 + n_age_dims(opt)
    athero_slice = None
    plaque_slice = None

    if getattr(opt, "cond_athero", False):
        width = 1 if getattr(opt, "cond_athero_mode", "any") == "any" else 4
        athero_slice = slice(off, off + width)
        off += width

    if getattr(opt, "cond_plaque", False):
        plaque_slice = slice(off, off + 4)
        off += 4

    if getattr(opt, "cond_dim", off) != off:
        raise ValueError(f"cond_dim mismatch: expected {off}, got {getattr(opt, 'cond_dim', None)}")

    return athero_slice, plaque_slice

def make_cond_from_group_pattern(
    group: str,
    pattern: Tuple[int, ...],
    opt,
    plaque_lookup: Dict[Tuple[str, Tuple[int, ...]], np.ndarray] | None = None,
    group_plaque_lookup: Dict[str, np.ndarray] | None = None,
    global_plaque_mean: np.ndarray | None = None,
) -> np.ndarray:
    cond = np.zeros(int(opt.cond_dim), dtype=np.float32)
    gender, age_bin = split_group(group)

    cond[0 if gender == "M" else 1] = 1.0
    # cond[agebin_to_index(age_bin)] = 1.0
    cond[agebin_to_index(age_bin, opt)] = 1.0

    athero_slice, plaque_slice = cond_layout(opt)
    if athero_slice is not None:
        vals = np.asarray(pattern, dtype=np.float32)
        if vals.size != (athero_slice.stop - athero_slice.start):
            raise ValueError(f"Pattern size mismatch for group={group}: {pattern}")
        cond[athero_slice] = vals

    if plaque_slice is not None:
        if getattr(opt, "neutralize_plaque_in_refs", False):
            if global_plaque_mean is None:
                plaque_vec = np.zeros(plaque_slice.stop - plaque_slice.start, dtype=np.float32)
            else:
                plaque_vec = np.asarray(global_plaque_mean, dtype=np.float32)
        else:
            plaque_vec = None
            if plaque_lookup is not None:
                plaque_vec = plaque_lookup.get((group, tuple(pattern)))
            if plaque_vec is None and group_plaque_lookup is not None:
                plaque_vec = group_plaque_lookup.get(group)
            if plaque_vec is None:
                plaque_vec = global_plaque_mean
            if plaque_vec is None:
                plaque_vec = np.zeros(plaque_slice.stop - plaque_slice.start, dtype=np.float32)

        cond[plaque_slice] = np.asarray(plaque_vec, dtype=np.float32)

    return cond


def sample_indices_balanced(dataset: PreprocessedMeshVAEDataset, max_samples: int, seed: int) -> np.ndarray:
    idx = np.arange(len(dataset))
    if max_samples <= 0 or max_samples >= len(idx):
        return idx

    rng = np.random.default_rng(seed)
    labels = np.array([cond_to_meta(c, dataset.opt).sick for c in dataset.cond], dtype=np.int64)

    pos = idx[labels == 1]
    neg = idx[labels == 0]

    if len(pos) == 0 or len(neg) == 0:
        rng.shuffle(idx)
        return np.sort(idx[:max_samples])

    half = max_samples // 2
    n_pos = min(len(pos), half)
    n_neg = min(len(neg), max_samples - n_pos)
    take = list(rng.choice(pos, size=n_pos, replace=False)) + list(rng.choice(neg, size=n_neg, replace=False))

    if len(take) < max_samples:
        remaining = np.setdiff1d(idx, np.array(take, dtype=np.int64), assume_unique=False)
        if len(remaining) > 0:
            extra_n = min(len(remaining), max_samples - len(take))
            take.extend(rng.choice(remaining, size=extra_n, replace=False).tolist())

    take = np.array(sorted(set(int(x) for x in take)), dtype=np.int64)
    return take


def build_dataset(opt, cond_tsv: str) -> PreprocessedMeshVAEDataset:
    opt.cond_tsv = cond_tsv
    if str(getattr(opt, "feature_source", "scan")) != "scan":
        dataset = ClinicalFeatureMeshVAEDataset(opt.dataroot, opt)
    else:
        dataset = PreprocessedMeshVAEDataset(opt.dataroot, opt)
    return dataset


def gather_empirical_stats(dataset: PreprocessedMeshVAEDataset, opt, indices: np.ndarray | None = None):
    by_group_pattern: Dict[str, Counter] = defaultdict(Counter)
    global_pattern = Counter()
    group_counts = Counter()
    group_sick_counts = Counter()

    athero_slice, plaque_slice = cond_layout(opt)
    plaque_by_group_pattern: Dict[Tuple[str, Tuple[int, ...]], List[np.ndarray]] = defaultdict(list)
    plaque_by_group: Dict[str, List[np.ndarray]] = defaultdict(list)
    plaque_all: List[np.ndarray] = []

    if indices is None:
        indices = np.arange(len(dataset), dtype=np.int64)

    for idx in indices.tolist():
        cond = np.asarray(dataset.cond[idx], dtype=np.float32)
        group = str(dataset.group[idx])

        meta = cond_to_meta(cond, opt)
        pat = tuple(meta.pattern)

        by_group_pattern[group][pat] += 1
        global_pattern[pat] += 1
        group_counts[group] += 1
        group_sick_counts[group] += int(meta.sick)

        if plaque_slice is not None:
            plaque_vec = np.asarray(cond[plaque_slice], dtype=np.float32)
            plaque_by_group_pattern[(group, pat)].append(plaque_vec)
            plaque_by_group[group].append(plaque_vec)
            plaque_all.append(plaque_vec)

    plaque_lookup = {
        key: np.mean(np.stack(vals, axis=0), axis=0).astype(np.float32)
        for key, vals in plaque_by_group_pattern.items()
        if len(vals) > 0
    }
    group_plaque_lookup = {
        key: np.mean(np.stack(vals, axis=0), axis=0).astype(np.float32)
        for key, vals in plaque_by_group.items()
        if len(vals) > 0
    }
    global_plaque_mean = None
    if plaque_all:
        global_plaque_mean = np.mean(np.stack(plaque_all, axis=0), axis=0).astype(np.float32)

    return {
        "by_group_pattern": by_group_pattern,
        "global_pattern": global_pattern,
        "group_counts": group_counts,
        "group_sick_counts": group_sick_counts,
        "plaque_lookup": plaque_lookup,
        "group_plaque_lookup": group_plaque_lookup,
        "global_plaque_mean": global_plaque_mean,
    }


def default_positive_patterns(opt) -> List[Tuple[int, ...]]:
    if getattr(opt, "cond_athero_mode", "any") == "any":
        return [(1,)]

    pats: List[Tuple[int, ...]] = []
    for i in range(1, 16):
        bits = tuple(int(x) for x in f"{i:04b}")
        pats.append(bits)
    return pats


def zero_pattern(opt) -> Tuple[int, ...]:
    if getattr(opt, "cond_athero_mode", "any") == "any":
        return (0,)
    return (0, 0, 0, 0)


def choose_patterns_for_group(group: str, stats: dict, opt) -> Tuple[List[Tuple[int, ...]], List[int]]:
    cnt = stats["by_group_pattern"].get(group, Counter())
    pos = [(pat, n) for pat, n in cnt.items() if any(pat)]

    if not pos:
        pos = [(pat, n) for pat, n in stats["global_pattern"].items() if any(pat)]

    if not pos:
        pos = [(pat, 1) for pat in default_positive_patterns(opt)]

    pats = [pat for pat, _ in pos]
    counts = [int(n) for _, n in pos]
    return pats, counts


def allocate_counts(total: int, patterns: Sequence[Tuple[int, ...]], weights: Sequence[int]) -> Dict[Tuple[int, ...], int]:
    if total <= 0:
        raise ValueError("total must be > 0")
    if len(patterns) == 0:
        return {}

    if len(patterns) == 1:
        return {tuple(patterns[0]): int(total)}

    patterns = [tuple(p) for p in patterns]
    weights = np.asarray(weights, dtype=np.float64)
    if np.all(weights <= 0):
        weights = np.ones_like(weights)

    order = np.argsort(-weights)
    patterns = [patterns[i] for i in order]
    weights = weights[order]

    if total < len(patterns):
        chosen = patterns[:total]
        return {p: 1 for p in chosen}

    alloc = {p: 1 for p in patterns}
    remaining = total - len(patterns)
    w = weights / weights.sum()
    raw = w * remaining
    extra = np.floor(raw).astype(int)

    for p, n in zip(patterns, extra.tolist()):
        alloc[p] += int(n)

    short = total - sum(alloc.values())
    if short > 0:
        frac = raw - np.floor(raw)
        frac_order = np.argsort(-frac)
        for idx in frac_order[:short]:
            alloc[patterns[idx]] += 1

    return alloc


def generate_reference_sets(model: MeshVAEModel, opt, stats: dict, out_dir: Path | None = None):
    net = get_net(model)
    ref_cache = {}

    age_labels = list(getattr(opt, "age_labels", AGE_LABELS))
    all_groups = sorted(
        set(stats["group_counts"].keys()) |
        {f"{g}_{a}" for g in ["M", "F"] for a in age_labels}
    )
    if not all_groups:
        raise RuntimeError("No demographic groups found in calibration set.")

    for group in all_groups:
        healthy_pat = zero_pattern(opt)
        sick_pats, sick_counts = choose_patterns_for_group(group, stats, opt)
        sick_alloc = allocate_counts(int(opt.samples_per_class), sick_pats, sick_counts)

        healthy_cond = make_cond_from_group_pattern(
            group,
            healthy_pat,
            opt,
            plaque_lookup=stats["plaque_lookup"],
            group_plaque_lookup=stats["group_plaque_lookup"],
            global_plaque_mean=stats["global_plaque_mean"],
        )
        healthy_conds = np.repeat(healthy_cond[None, :], int(opt.samples_per_class), axis=0)

        sick_cond_list = []
        sick_pattern_labels = []
        sick_unique_conds = []
        for pat, n in sick_alloc.items():
            c = make_cond_from_group_pattern(
                group,
                pat,
                opt,
                plaque_lookup=stats["plaque_lookup"],
                group_plaque_lookup=stats["group_plaque_lookup"],
                global_plaque_mean=stats["global_plaque_mean"],
            )
            sick_unique_conds.append((pat, c.copy()))
            for _ in range(int(n)):
                sick_cond_list.append(c.copy())
                sick_pattern_labels.append("".join(str(x) for x in pat))

        sick_conds = np.stack(sick_cond_list, axis=0).astype(np.float32)
        healthy_conds = healthy_conds.astype(np.float32)

        with torch.no_grad():
            cond_h = torch.from_numpy(healthy_conds).float().to(model.device)
            pz_h = net.prior(cond_h.shape[0], int(opt.latent_dim), device=model.device, cond=cond_h)
            z_h = pz_h.rsample() if hasattr(pz_h, "rsample") else pz_h.sample()
            z_h = z_h * float(opt.temperature)
            refs_h = net.decoder(z_h, int(opt.ninput_edges), cond=cond_h).detach().cpu()

            cond_s = torch.from_numpy(sick_conds).float().to(model.device)
            pz_s = net.prior(cond_s.shape[0], int(opt.latent_dim), device=model.device, cond=cond_s)
            z_s = pz_s.rsample() if hasattr(pz_s, "rsample") else pz_s.sample()
            z_s = z_s * float(opt.temperature)
            refs_s = net.decoder(z_s, int(opt.ninput_edges), cond=cond_s).detach().cpu()

        group_total = float(stats["group_counts"].get(group, 0))
        group_sick = float(stats["group_sick_counts"].get(group, 0))
        global_total = float(sum(stats["group_counts"].values()))
        global_sick = float(sum(stats["group_sick_counts"].values()))

        global_pi = (global_sick / global_total) if global_total > 0 else 0.5
        group_pi = (group_sick / group_total) if group_total > 0 else global_pi

        if opt.prior_mode == "group":
            pi_sick = group_pi
        elif opt.prior_mode == "global":
            pi_sick = global_pi
        else:
            pi_sick = 0.5

        pi_sick = float(np.clip(pi_sick, 1e-4, 1.0 - 1e-4))

        ref_cache[group] = {
            "healthy_refs": refs_h,
            "sick_refs": refs_s,
            "healthy_cond": healthy_cond,
            "sick_unique_conds": sick_unique_conds,
            "sick_pattern_labels": sick_pattern_labels,
            "pi_sick": pi_sick,
        }

        if out_dir is not None and opt.save_reference_samples:
            torch.save(refs_h, out_dir / f"{group}_healthy_refs.pt")
            torch.save(refs_s, out_dir / f"{group}_sick_refs.pt")

    return ref_cache


def logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def chamfer_to_reference_set(
    x: torch.Tensor,
    refs: torch.Tensor,
    n_points: int,
    chunk: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dists = []
    x = x.to(device)
    for start in range(0, refs.shape[0], batch_size):
        r = refs[start:start + batch_size].to(device)
        xb = x.expand(r.shape[0], -1, -1)
        d = chamfer_distance_subsample(xb, r, n_points=n_points, chunk=chunk)
        dists.append(d.detach().cpu().numpy())
    return np.concatenate(dists, axis=0)


def score_sample_against_refs(x: torch.Tensor, group: str, ref_cache: dict, opt) -> dict:
    info = ref_cache[group]
    dh = chamfer_to_reference_set(
        x=x,
        refs=info["healthy_refs"],
        n_points=int(opt.metric_npoints),
        chunk=int(opt.metric_chunk),
        batch_size=int(opt.distance_batch_size),
        device=x.device,
    )
    ds = chamfer_to_reference_set(
        x=x,
        refs=info["sick_refs"],
        n_points=int(opt.metric_npoints),
        chunk=int(opt.metric_chunk),
        batch_size=int(opt.distance_batch_size),
        device=x.device,
    )

    tau = float(opt.distance_temperature)
    ll_h = logmeanexp(-dh / tau)
    ll_s = logmeanexp(-ds / tau)

    pi_s = float(info["pi_sick"])
    pi_h = 1.0 - pi_s
    log_post_h = math.log(pi_h) + ll_h
    log_post_s = math.log(pi_s) + ll_s
    delta = float(log_post_s - log_post_h)
    if delta >= 0:
        prob_s = 1.0 / (1.0 + math.exp(-delta))
    else:
        ed = math.exp(delta)
        prob_s = ed / (1.0 + ed)

    return {
        "sample_d_healthy_min": float(np.min(dh)),
        "sample_d_healthy_mean": float(np.mean(dh)),
        "sample_d_sick_min": float(np.min(ds)),
        "sample_d_sick_mean": float(np.mean(ds)),
        "sample_ll_healthy": float(ll_h),
        "sample_ll_sick": float(ll_s),
        "sample_logpost_margin": float(log_post_s - log_post_h),
        "sample_margin": float(np.mean(dh) - np.mean(ds)),
        "sample_prob_sick": float(prob_s),
    }


@torch.no_grad()
def evaluate_condition_fit(
    model: MeshVAEModel,
    item: dict,
    cond_vec: np.ndarray,
) -> dict:
    net = get_net(model)
    x = item["edge_features"].to(model.device)
    cond = torch.from_numpy(cond_vec[None, :]).float().to(model.device)

    mesh_batch = {
        "verts": item["verts"].unsqueeze(0).to(model.device),
        "faces": item["faces"].unsqueeze(0).to(model.device),
        "verts_mask": item["verts_mask"].unsqueeze(0).to(model.device),
        "faces_mask": item["faces_mask"].unsqueeze(0).to(model.device),
    }

    loss, diag, _ = model.vi(net, x, mesh_batch, cond=cond)

    out = {
        "loss": float(loss.detach().cpu()),
        "elbo": float(diag["elbo"][0].detach().cpu()),
        "log_px": float(diag["log_px"][0].detach().cpu()),
    }
    if "kl" in diag:
        out["kl"] = float(diag["kl"][0].detach().cpu())
    if "prior_match" in diag:
        out["prior_match"] = float(diag["prior_match"][0].detach().cpu())
    if "chamfer" in diag:
        out["chamfer"] = float(diag["chamfer"][0].detach().cpu())
    else:
        out["chamfer"] = float(-diag["log_px"][0].detach().cpu())
    if "spread" in diag:
        out["spread"] = float(diag["spread"][0].detach().cpu())
    if "repulsion" in diag:
        out["repulsion"] = float(diag["repulsion"][0].detach().cpu())
    return out


def score_sample_by_reconstruction(
    model: MeshVAEModel,
    item: dict,
    group: str,
    ref_cache: dict,
) -> dict:
    info = ref_cache[group]

    healthy_fit = evaluate_condition_fit(model, item, info["healthy_cond"])

    sick_fits = []
    for pat, cond_vec in info["sick_unique_conds"]:
        fit = evaluate_condition_fit(model, item, cond_vec)
        fit["pattern"] = "".join(str(i) for i in pat)
        sick_fits.append(fit)

    sick_best_recon = min(sick_fits, key=lambda z: z["chamfer"])
    sick_best_elbo = max(sick_fits, key=lambda z: z["elbo"])

    return {
        "recon_healthy": float(healthy_fit["chamfer"]),
        "recon_sick_best": float(sick_best_recon["chamfer"]),
        "recon_best_sick_pattern": str(sick_best_recon["pattern"]),
        "recon_margin": float(healthy_fit["chamfer"] - sick_best_recon["chamfer"]),
        "elbo_healthy": float(healthy_fit["elbo"]),
        "elbo_sick_best": float(sick_best_elbo["elbo"]),
        "elbo_best_sick_pattern": str(sick_best_elbo["pattern"]),
        "elbo_margin": float(sick_best_elbo["elbo"] - healthy_fit["elbo"]),
        "log_px_healthy": float(healthy_fit["log_px"]),
        "log_px_sick_best_elbo": float(sick_best_elbo["log_px"]),
    }

@torch.no_grad()
def score_sample_by_latent_classifier(
    model: MeshVAEModel,
    item: dict,
    cond_true: np.ndarray,
) -> dict:
    """
    Score a sample with the supervised latent classifier head, if present.

    This uses the trained encoder to get mu(X), then the athero_head predicts
    disease from latent geometry. If the checkpoint was not trained with
    --aux_predict_athero, this returns an empty dict.
    """
    net = get_net(model)
    if not hasattr(net, "athero_head") or net.athero_head is None:
        return {}

    x = item["edge_features"].to(model.device)
    cond = torch.from_numpy(cond_true[None, :]).float().to(model.device)

    mesh_batch = {
        "verts": item["verts"].unsqueeze(0).to(model.device),
        "faces": item["faces"].unsqueeze(0).to(model.device),
        "verts_mask": item["verts_mask"].unsqueeze(0).to(model.device),
        "faces_mask": item["faces_mask"].unsqueeze(0).to(model.device),
    }

    out = net(x, mesh_batch, cond=cond)
    logits = out.get("athero_logits", None)
    if logits is None:
        return {}

    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
    logits_np = logits.detach().cpu().numpy()[0]

    row = {}

    if probs.shape[0] == 1:
        row["classifier_prob_sick"] = float(probs[0])
        row["classifier_logit_sick"] = float(logits_np[0])
    else:
        # For a 4-bit head, use probability of any positive location.
        # This is smoother than max(prob) and corresponds to:
        # P(any sick) = 1 - product_locs P(location healthy)
        prob_any = 1.0 - float(np.prod(1.0 - probs))
        row["classifier_prob_sick"] = prob_any
        row["classifier_logit_sick"] = float(np.max(logits_np))

        for k, name in enumerate(["ho_carotis", "ve_carotis", "ho_fem", "ve_fem"]):
            row[f"classifier_prob_{name}"] = float(probs[k])
            row[f"classifier_logit_{name}"] = float(logits_np[k])

    return row

def score_dataset(
    name: str,
    dataset: PreprocessedMeshVAEDataset,
    model: MeshVAEModel,
    ref_cache: dict,
    opt,
    indices: np.ndarray | None = None,
) -> pd.DataFrame:
    if indices is None:
        indices = np.arange(len(dataset), dtype=np.int64)

    rows = []
    for pos, idx in enumerate(indices.tolist()):
        item = dataset[idx]
        x = item["edge_features"].to(model.device)
        mesh = item.get("mesh")
        cond_true = np.asarray(dataset.cond[idx], dtype=np.float32)
        meta = cond_to_meta(cond_true, opt)

        if meta.group not in ref_cache:
            raise KeyError(f"No reference cache found for group={meta.group}")

        row = {
            "split": name,
            "row_idx": int(idx),
            "path": str(item["path"]),
            "group": meta.group,
            "gender": meta.gender,
            "age_bin": meta.age_bin,
            "label_sick": int(meta.sick),
            "true_pattern": meta.pattern_str,
        }

        row.update(score_sample_against_refs(x, meta.group, ref_cache, opt))
        row.update(score_sample_by_reconstruction(model, item, meta.group, ref_cache))
        row.update(score_sample_by_latent_classifier(model, item, cond_true))
        rows.append(row)

        if (pos + 1) % 25 == 0 or (pos + 1) == len(indices):
            print(f"[{name}] scored {pos + 1}/{len(indices)} samples", flush=True)

    return pd.DataFrame(rows)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / max(len(y_true), 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    rec = sens
    f1 = (2.0 * prec * rec) / max(prec + rec, 1e-12)
    bal_acc = 0.5 * (sens + spec)
    youden_j = sens + spec - 1.0

    return {
        "n": int(len(y_true)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "youden_j": float(youden_j),
    }


def roc_auc_manual(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)

def prevalence_policy_band(prev: float, cutoff: float) -> str:
    return "lowprev" if float(prev) < float(cutoff) else "highprev"

def choose_threshold_for_prevalence_band(y_true: np.ndarray, scores: np.ndarray, band: str) -> dict:
    best = None

    for thr in threshold_candidates(scores):
        pred = (scores >= thr).astype(int)
        mets = binary_metrics(y_true, pred)

        if band == "lowprev":
            # prioritize catching disease, but do not allow total specificity collapse
            feasible = mets["specificity"] >= 0.20
            crit = 0.70 * mets["sensitivity"] + 0.30 * mets["specificity"]
        else:
            # prioritize ruling out false positives, but do not allow total sensitivity collapse
            feasible = mets["sensitivity"] >= 0.20
            crit = 0.70 * mets["specificity"] + 0.30 * mets["sensitivity"]

        tie = mets["balanced_accuracy"]

        if (
            (best is None)
            or (int(feasible) > int(best["feasible"]))
            or (
                int(feasible) == int(best["feasible"])
                and (
                    crit > best["criterion"]
                    or (math.isclose(crit, best["criterion"]) and tie > best["tie"])
                )
            )
        ):
            best = {
                "threshold": float(thr),
                "criterion": float(crit),
                "tie": float(tie),
                "feasible": bool(feasible),
                "metrics": mets,
            }

    return best

def fit_prevalence_policy_thresholds_for_score(calib_df: pd.DataFrame, score_name: str, cutoff: float) -> dict:
    tmp = calib_df.copy()
    tmp["policy_band"] = tmp["group_prevalence"].apply(lambda p: prevalence_policy_band(p, cutoff))

    thr_map = {}
    band_metrics = {}

    for band in ["lowprev", "highprev"]:
        sub = tmp[tmp["policy_band"] == band]
        if len(sub) == 0:
            continue

        y = sub["label_sick"].to_numpy().astype(int)
        s = sub[score_name].to_numpy(dtype=np.float64)
        best = choose_threshold_for_prevalence_band(y, s, band)

        thr_map[band] = best["threshold"]
        band_metrics[band] = best["metrics"]

    return {"thresholds": thr_map, "band_metrics": band_metrics}

def apply_prevalence_policy_thresholds(df: pd.DataFrame, score_name: str, thr_map: dict, cutoff: float) -> pd.DataFrame:
    out = df.copy()
    out["policy_band"] = out["group_prevalence"].apply(lambda p: prevalence_policy_band(p, cutoff))
    out["chosen_score"] = score_name

    low_thr = thr_map.get("lowprev", thr_map.get("highprev", np.nan))
    high_thr = thr_map.get("highprev", thr_map.get("lowprev", np.nan))

    lookup = {
        "lowprev": float(low_thr),
        "highprev": float(high_thr),
    }

    out["chosen_threshold"] = out["policy_band"].map(lookup).astype(float)
    out["score_value"] = out[score_name].astype(float)

    score_vals = out["score_value"].to_numpy(dtype=np.float64)
    thr_vals = out["chosen_threshold"].to_numpy(dtype=np.float64)
    out["pred_sick"] = (score_vals >= thr_vals).astype(int)
    out["correct"] = (out["pred_sick"].astype(int) == out["label_sick"].astype(int)).astype(int)
    return out

def fit_prevalence_policy_score_table(calib_df: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    rows = []

    for score_name in available_score_names(calib_df):
        fitted = fit_prevalence_policy_thresholds_for_score(calib_df, score_name, cutoff)
        thr_map = fitted["thresholds"]

        pred_df = apply_prevalence_policy_thresholds(calib_df, score_name, thr_map, cutoff)

        lowprev = pred_df[pred_df["policy_band"] == "lowprev"]
        highprev = pred_df[pred_df["policy_band"] == "highprev"]

        mets_l = (
            binary_metrics(lowprev["label_sick"].to_numpy(), lowprev["pred_sick"].to_numpy())
            if len(lowprev) > 0
            else {"sensitivity": float("nan"), "specificity": float("nan"), "balanced_accuracy": float("nan")}
        )
        mets_h = (
            binary_metrics(highprev["label_sick"].to_numpy(), highprev["pred_sick"].to_numpy())
            if len(highprev) > 0
            else {"sensitivity": float("nan"), "specificity": float("nan"), "balanced_accuracy": float("nan")}
        )

        policy_objective = 0.5 * (
            0.70 * mets_l["sensitivity"] + 0.30 * mets_l["specificity"]
        ) + 0.5 * (
            0.70 * mets_h["specificity"] + 0.30 * mets_h["sensitivity"]
        )

        rows.append({
            "score_name": score_name,
            "thr_lowprev": float(thr_map.get("lowprev", np.nan)),
            "thr_highprev": float(thr_map.get("highprev", np.nan)),
            "prevalence_cutoff": float(cutoff),
            "policy_objective": float(policy_objective),
            "lowprev_sensitivity": float(mets_l["sensitivity"]),
            "lowprev_specificity": float(mets_l["specificity"]),
            "lowprev_bal_acc": float(mets_l["balanced_accuracy"]),
            "highprev_sensitivity": float(mets_h["sensitivity"]),
            "highprev_specificity": float(mets_h["specificity"]),
            "highprev_bal_acc": float(mets_h["balanced_accuracy"]),
            "roc_auc": roc_auc_manual(
                calib_df["label_sick"].to_numpy(),
                calib_df[score_name].to_numpy(dtype=np.float64),
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["policy_objective", "roc_auc"], ascending=[False, False])
        .reset_index(drop=True)
    )

def evaluate_search_candidate_from_indices(dataset, calib_idx, test_idx, model, opt) -> dict:
    """
    Used by train_mesh_vae.py --phase search_age_bins.

    Important:
    During age-bin search, opt comes from train_mesh_vae.BaseOptions,
    not DiagnosticOptions. Therefore diagnostic-only options like
    samples_per_class, metric_npoints, distance_temperature, etc. may not exist.
    We set safe defaults here.
    """

    # ------------------------------------------------------------------
    # Defaults needed because age-bin search calls this from train_mesh_vae.py
    # ------------------------------------------------------------------
    opt.samples_per_class = int(getattr(opt, "samples_per_class", 32))
    opt.metric_npoints = int(getattr(opt, "metric_npoints", getattr(opt, "chamfer_npoints", 2048)))
    opt.metric_chunk = int(getattr(opt, "metric_chunk", 1024))
    opt.distance_batch_size = int(getattr(opt, "distance_batch_size", 4))
    opt.distance_temperature = float(getattr(opt, "distance_temperature", 0.05))
    opt.prior_mode = str(getattr(opt, "prior_mode", "flat"))
    opt.temperature = float(getattr(opt, "temperature", 1.0))
    opt.neutralize_plaque_in_refs = bool(getattr(opt, "neutralize_plaque_in_refs", False))
    opt.save_reference_samples = bool(getattr(opt, "save_reference_samples", False))

    # For age-bin search, force the interpretable diagnostic score unless explicitly set.
    opt.score_method = str(getattr(opt, "score_method", "sample_prob_sick"))

    stats = gather_empirical_stats(dataset, opt, indices=calib_idx)
    ref_cache = generate_reference_sets(model=model, opt=opt, stats=stats, out_dir=None)

    calib_df = score_dataset(
        "calibration",
        dataset,
        model,
        ref_cache,
        opt,
        indices=np.asarray(calib_idx, dtype=np.int64),
    )
    test_df = score_dataset(
        "test",
        dataset,
        model,
        ref_cache,
        opt,
        indices=np.asarray(test_idx, dtype=np.int64),
    )

    calib_df = add_group_prevalence(calib_df, stats)
    test_df = add_group_prevalence(test_df, stats)

    policy_mode = str(getattr(opt, "age_search_policy_mode", "global"))
    score_method = str(getattr(opt, "score_method", "sample_prob_sick"))

    if policy_mode == "global":
        thr_df = fit_all_thresholds(calib_df, criterion="balanced_accuracy")

        if score_method == "auto":
            chosen_score = str(thr_df.iloc[0]["score_name"])
        else:
            chosen_score = score_method

        if chosen_score not in set(thr_df["score_name"]):
            raise ValueError(
                f"Requested score_method={chosen_score} not found in prevalence threshold table. "
                f"Available: {list(thr_df['score_name'])}"
            )

        chosen_row = thr_df.loc[thr_df["score_name"] == chosen_score].iloc[0]
        test_pred_df = apply_threshold(test_df, chosen_score, float(chosen_row["threshold"]))

    elif policy_mode == "age":
        thr_df = fit_age_policy_score_table(calib_df)

        if score_method == "auto":
            chosen_score = str(thr_df.iloc[0]["score_name"])
        else:
            chosen_score = score_method

        if chosen_score not in set(thr_df["score_name"]):
            raise ValueError(
                f"Requested score_method={chosen_score} not found in age threshold table. "
                f"Available: {list(thr_df['score_name'])}"
            )

        chosen_row = thr_df.loc[thr_df["score_name"] == chosen_score].iloc[0]
        thr_map = {
            "under50": float(chosen_row["thr_under50"]),
            "over50": float(chosen_row["thr_over50"]),
        }
        test_pred_df = apply_age_policy_thresholds(test_df, chosen_score, thr_map)

    elif policy_mode == "prevalence":
        prevalence_cutoff = resolve_prevalence_cutoff(calib_df, opt)
        thr_df = fit_prevalence_policy_score_table(calib_df, prevalence_cutoff)

        if score_method == "auto":
            chosen_score = "sample_prob_sick"
        else:
            chosen_score = score_method

        if chosen_score not in set(thr_df["score_name"]):
            raise ValueError(
                f"Requested score_method={chosen_score} not found in prevalence threshold table. "
                "For prevalence mode, use sample_prob_sick."
            )

        chosen_row = thr_df.loc[thr_df["score_name"] == chosen_score].iloc[0]
        thr_map = {
            "lowprev": float(chosen_row["thr_lowprev"]),
            "highprev": float(chosen_row["thr_highprev"]),
        }
        test_pred_df = apply_prevalence_policy_thresholds(
            test_df,
            chosen_score,
            thr_map,
            prevalence_cutoff,
        )

    else:
        raise ValueError(f"Unknown age_search_policy_mode={policy_mode}")

    mets = binary_metrics(
        test_pred_df["label_sick"].to_numpy(),
        test_pred_df["pred_sick"].to_numpy(),
    )
    auc = roc_auc_manual(
        test_df["label_sick"].to_numpy(),
        test_df[chosen_score].to_numpy(dtype=np.float64),
    )

    return {
        "chosen_score": chosen_score,
        "balanced_accuracy": float(mets["balanced_accuracy"]),
        "roc_auc": float(auc),
        "sensitivity": float(mets["sensitivity"]),
        "specificity": float(mets["specificity"]),
    }

def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    vals = np.unique(np.asarray(scores, dtype=np.float64))
    if len(vals) == 1:
        return np.array([vals[0]], dtype=np.float64)
    mids = (vals[:-1] + vals[1:]) / 2.0
    extra = np.array([vals[0] - 1e-9, vals[-1] + 1e-9], dtype=np.float64)
    return np.concatenate([extra[:1], mids, extra[1:]])

def choose_threshold(y_true: np.ndarray, scores: np.ndarray, criterion: str = "balanced_accuracy") -> dict:
    best = None
    for thr in threshold_candidates(scores):
        pred = (scores >= thr).astype(int)
        mets = binary_metrics(y_true, pred)
        crit = mets[criterion]
        if (best is None) or (crit > best["criterion"]) or (
            math.isclose(crit, best["criterion"]) and abs(thr) < abs(best["threshold"])
        ):
            best = {
                "threshold": float(thr),
                "criterion": float(crit),
                "metrics": mets,
            }
    return best

def available_score_names(df: pd.DataFrame) -> List[str]:
    """
    Scores that can be thresholded. Some exist only if the checkpoint
    was trained with the supervised latent classifier head.
    """
    preferred = [
        "classifier_prob_sick",
        "sample_prob_sick",
        "sample_margin",
        "recon_margin",
        "elbo_margin",
    ]
    return [s for s in preferred if s in df.columns]

def fit_all_thresholds(calib_df: pd.DataFrame, criterion: str) -> pd.DataFrame:
    y = calib_df["label_sick"].to_numpy().astype(int)
    rows = []

    for score_name in available_score_names(calib_df):
        scores = calib_df[score_name].to_numpy(dtype=np.float64)
        best = choose_threshold(y, scores, criterion=criterion)
        mets = best["metrics"]
        rows.append({
            "score_name": score_name,
            "threshold": best["threshold"],
            "criterion": best["criterion"],
            "roc_auc": roc_auc_manual(y, scores),
            **mets,
        })

    if not rows:
        raise RuntimeError("No usable score columns found for threshold fitting.")

    out = (
        pd.DataFrame(rows)
        .sort_values([criterion, "roc_auc"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return out


def apply_threshold(df: pd.DataFrame, score_name: str, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["chosen_score"] = score_name
    out["chosen_threshold"] = float(threshold)
    out["score_value"] = out[score_name].astype(float)
    out["pred_sick"] = (out[score_name].to_numpy(dtype=np.float64) >= float(threshold)).astype(int)
    out["correct"] = (out["pred_sick"].astype(int) == out["label_sick"].astype(int)).astype(int)
    return out


def maybe_plot_scores(calib_df: pd.DataFrame, test_df: pd.DataFrame, score_name: str, threshold: float, out_png: Path) -> None:
    if not _HAS_MPL:
        return

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    for label, title in [(0, "healthy"), (1, "sick")]:
        vals = calib_df.loc[calib_df["label_sick"] == label, score_name].to_numpy(dtype=np.float64)
        if len(vals) > 0:
            plt.hist(vals, bins=30, alpha=0.6, density=True, label=f"calib {title}")
    plt.axvline(threshold, linestyle="--")
    plt.title(f"Calibration: {score_name}")
    plt.xlabel("score")
    plt.ylabel("density")
    plt.legend()

    plt.subplot(1, 2, 2)
    for label, title in [(0, "healthy"), (1, "sick")]:
        vals = test_df.loc[test_df["label_sick"] == label, score_name].to_numpy(dtype=np.float64)
        if len(vals) > 0:
            plt.hist(vals, bins=30, alpha=0.6, density=True, label=f"test {title}")
    plt.axvline(threshold, linestyle="--")
    plt.title(f"Test: {score_name}")
    plt.xlabel("score")
    plt.ylabel("density")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def maybe_plot_scores_age_policy(
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_name: str,
    thr_map: dict,
    out_png: Path,
) -> None:
    if not _HAS_MPL:
        return

    plt.figure(figsize=(12, 8))

    panels = [
        ("calibration", calib_df, "under50"),
        ("calibration", calib_df, "over50"),
        ("test", test_df, "under50"),
        ("test", test_df, "over50"),
    ]

    for i, (split_name, df, band) in enumerate(panels, start=1):
        sub = df[df["age_policy_band"] == band]
        plt.subplot(2, 2, i)

        for label, title in [(0, "healthy"), (1, "sick")]:
            vals = sub.loc[sub["label_sick"] == label, score_name].to_numpy(dtype=np.float64)
            if len(vals) > 0:
                plt.hist(vals, bins=30, alpha=0.6, density=True, label=f"{split_name} {title}")

        if band in thr_map:
            plt.axvline(float(thr_map[band]), linestyle="--")

        plt.title(f"{split_name}: {band} | {score_name}")
        plt.xlabel("score")
        plt.ylabel("density")
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def age_policy_band(age_bin: str) -> str:
    s = str(age_bin).strip()
    if s.endswith("+"):
        lo = float(s[:-1])
        return "over50" if lo >= 50 else "under50"

    lo_s, hi_s = s.split("-", 1)
    lo = float(lo_s)
    hi = float(hi_s)

    if hi <= 50:
        return "under50"
    if lo >= 50:
        return "over50"

    # bin crosses 50; put it where its midpoint lies
    mid = 0.5 * (lo + hi)
    return "under50" if mid < 50 else "over50"


def choose_threshold_for_band(y_true: np.ndarray, scores: np.ndarray, band: str) -> dict:
    best = None
    for thr in threshold_candidates(scores):
        pred = (scores >= thr).astype(int)
        mets = binary_metrics(y_true, pred)

        # Clinical objective:
        # under50  -> maximize sensitivity
        # 50+      -> maximize specificity
        crit = mets["sensitivity"] if band == "under50" else mets["specificity"]
        tie = mets["balanced_accuracy"]

        if (best is None) or (crit > best["criterion"]) or (
            math.isclose(crit, best["criterion"]) and tie > best["tie"]
        ):
            best = {
                "threshold": float(thr),
                "criterion": float(crit),
                "tie": float(tie),
                "metrics": mets,
            }
    return best


def fit_age_policy_thresholds_for_score(calib_df: pd.DataFrame, score_name: str) -> dict:
    tmp = calib_df.copy()
    tmp["age_policy_band"] = tmp["age_bin"].apply(age_policy_band)

    thr_map = {}
    band_metrics = {}

    for band in ["under50", "over50"]:
        sub = tmp[tmp["age_policy_band"] == band]
        if len(sub) == 0:
            continue

        y = sub["label_sick"].to_numpy().astype(int)
        s = sub[score_name].to_numpy(dtype=np.float64)
        best = choose_threshold_for_band(y, s, band)

        thr_map[band] = best["threshold"]
        band_metrics[band] = best["metrics"]

    return {"thresholds": thr_map, "band_metrics": band_metrics}


def apply_age_policy_thresholds(df: pd.DataFrame, score_name: str, thr_map: dict) -> pd.DataFrame:
    out = df.copy()
    out["age_policy_band"] = out["age_bin"].apply(age_policy_band)
    out["chosen_score"] = score_name
    out["chosen_threshold"] = out["age_policy_band"].map(thr_map).astype(float)
    out["score_value"] = out[score_name].astype(float)

    score_vals = out["score_value"].to_numpy(dtype=np.float64)
    thr_vals = out["chosen_threshold"].to_numpy(dtype=np.float64)
    out["pred_sick"] = (score_vals >= thr_vals).astype(int)
    out["correct"] = (out["pred_sick"].astype(int) == out["label_sick"].astype(int)).astype(int)
    return out


def fit_age_policy_score_table(calib_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # leave out sample_prob_sick because it bakes in priors
    for score_name in available_score_names(calib_df):
        fitted = fit_age_policy_thresholds_for_score(calib_df, score_name)
        thr_map = fitted["thresholds"]

        pred_df = apply_age_policy_thresholds(calib_df, score_name, thr_map)

        under50 = pred_df[pred_df["age_policy_band"] == "under50"]
        over50 = pred_df[pred_df["age_policy_band"] == "over50"]

        mets_u = binary_metrics(
            under50["label_sick"].to_numpy(),
            under50["pred_sick"].to_numpy(),
        ) if len(under50) > 0 else {"sensitivity": float("nan"), "balanced_accuracy": float("nan")}

        mets_o = binary_metrics(
            over50["label_sick"].to_numpy(),
            over50["pred_sick"].to_numpy(),
        ) if len(over50) > 0 else {"specificity": float("nan"), "balanced_accuracy": float("nan")}

        # combined selection objective
        policy_objective = 0.5 * mets_u["sensitivity"] + 0.5 * mets_o["specificity"]

        rows.append({
            "score_name": score_name,
            "thr_under50": float(thr_map.get("under50", np.nan)),
            "thr_over50": float(thr_map.get("over50", np.nan)),
            "policy_objective": float(policy_objective),
            "under50_sensitivity": float(mets_u["sensitivity"]),
            "under50_bal_acc": float(mets_u["balanced_accuracy"]),
            "over50_specificity": float(mets_o["specificity"]),
            "over50_bal_acc": float(mets_o["balanced_accuracy"]),
            "roc_auc": roc_auc_manual(
                calib_df["label_sick"].to_numpy(),
                calib_df[score_name].to_numpy(dtype=np.float64),
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["policy_objective", "roc_auc"], ascending=[False, False])
        .reset_index(drop=True)
    )

def add_group_prevalence(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    lookup = {}
    for g, total in stats["group_counts"].items():
        sick = float(stats["group_sick_counts"].get(g, 0))
        lookup[g] = sick / max(float(total), 1.0)

    out["group_prevalence"] = out["group"].map(lookup).fillna(0.5).astype(float)
    return out


def resolve_prevalence_cutoff(calib_df: pd.DataFrame, opt) -> float:
    mode = str(getattr(opt, "prevalence_band_mode", "global"))

    if mode == "fixed":
        if getattr(opt, "prevalence_cutoff", None) is None:
            raise ValueError("--prevalence_cutoff must be set when --prevalence_band_mode fixed")
        return float(opt.prevalence_cutoff)

    if mode == "median":
        return float(np.median(calib_df["group_prevalence"].to_numpy(dtype=np.float64)))

    # default: split at empirical global calibration prevalence
    return float(calib_df["label_sick"].mean())


def prevalence_to_weight(prev: np.ndarray, prev_min: float, prev_max: float) -> np.ndarray:
    prev = np.asarray(prev, dtype=np.float64)
    denom = max(float(prev_max - prev_min), 1e-8)
    w = (prev - float(prev_min)) / denom
    return np.clip(w, 0.0, 1.0)


def threshold_grid(scores: np.ndarray, n_grid: int = 61) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    vals = np.unique(scores)
    if len(vals) <= n_grid:
        return threshold_candidates(scores)

    qs = np.linspace(0.02, 0.98, n_grid)
    grid = np.quantile(vals, qs)
    return np.unique(grid.astype(np.float64))


def apply_prevalence_schedule_thresholds(
    df: pd.DataFrame,
    score_name: str,
    sched: dict,
) -> pd.DataFrame:
    out = df.copy()

    prev = out["group_prevalence"].to_numpy(dtype=np.float64)
    w = prevalence_to_weight(prev, sched["prev_min"], sched["prev_max"])

    thr = sched["thr_low"] + w * (sched["thr_high"] - sched["thr_low"])

    out["chosen_score"] = score_name
    out["score_value"] = out[score_name].astype(float)
    out["prevalence_weight"] = w
    out["chosen_threshold"] = thr
    out["pred_sick"] = (out["score_value"].to_numpy(dtype=np.float64) >= thr).astype(int)
    out["correct"] = (out["pred_sick"].astype(int) == out["label_sick"].astype(int)).astype(int)

    # optional reporting buckets, only for summaries/plots
    out["policy_band"] = np.where(w < 0.5, "lowprev", "highprev")
    return out


def prevalence_schedule_objective(pred_df: pd.DataFrame) -> dict:
    group_rows = []

    for group, sub in pred_df.groupby("group"):
        mets = binary_metrics(
            sub["label_sick"].to_numpy(),
            sub["pred_sick"].to_numpy(),
        )
        prev = float(sub["group_prevalence"].iloc[0])

        # smooth shift:
        # low prevalence -> care more about sensitivity
        # high prevalence -> care more about specificity
        utility = (1.0 - prev) * mets["sensitivity"] + prev * mets["specificity"]

        # discourage trivial collapse
        penalty = 0.0
        if mets["sensitivity"] < 0.15:
            penalty += 0.25 * (0.15 - mets["sensitivity"])
        if mets["specificity"] < 0.15:
            penalty += 0.25 * (0.15 - mets["specificity"])

        group_rows.append({
            "group": group,
            "n": len(sub),
            "prevalence": prev,
            "utility": utility - penalty,
            "balanced_accuracy": mets["balanced_accuracy"],
            "sensitivity": mets["sensitivity"],
            "specificity": mets["specificity"],
        })

    gdf = pd.DataFrame(group_rows)
    if len(gdf) == 0:
        return {
            "policy_objective": float("-inf"),
            "balanced_accuracy": float("nan"),
            "group_table": gdf,
        }

    weights = gdf["n"].to_numpy(dtype=np.float64)
    weights = weights / weights.sum()

    return {
        "policy_objective": float(np.sum(weights * gdf["utility"].to_numpy(dtype=np.float64))),
        "balanced_accuracy": float(np.sum(weights * gdf["balanced_accuracy"].to_numpy(dtype=np.float64))),
        "group_table": gdf,
    }


def fit_prevalence_schedule_for_score(calib_df: pd.DataFrame, score_name: str) -> dict:
    prev = calib_df["group_prevalence"].to_numpy(dtype=np.float64)
    prev_min = float(prev.min())
    prev_max = float(prev.max())

    cand = threshold_grid(calib_df[score_name].to_numpy(dtype=np.float64), n_grid=61)

    best = None
    for thr_low in cand:
        for thr_high in cand:
            # enforce monotone schedule: higher prevalence => higher threshold
            if thr_high < thr_low:
                continue

            sched = {
                "thr_low": float(thr_low),
                "thr_high": float(thr_high),
                "prev_min": prev_min,
                "prev_max": prev_max,
            }

            pred_df = apply_prevalence_schedule_thresholds(calib_df, score_name, sched)
            obj = prevalence_schedule_objective(pred_df)

            if (best is None) or (obj["policy_objective"] > best["policy_objective"]) or (
                math.isclose(obj["policy_objective"], best["policy_objective"])
                and obj["balanced_accuracy"] > best["balanced_accuracy"]
            ):
                best = {
                    **sched,
                    "policy_objective": obj["policy_objective"],
                    "balanced_accuracy": obj["balanced_accuracy"],
                    "group_table": obj["group_table"],
                }

    return best


def fit_prevalence_schedule_score_table(calib_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for score_name in available_score_names(calib_df):
        best = fit_prevalence_schedule_for_score(calib_df, score_name)

        rows.append({
            "score_name": score_name,
            "thr_low": float(best["thr_low"]),
            "thr_high": float(best["thr_high"]),
            "prev_min": float(best["prev_min"]),
            "prev_max": float(best["prev_max"]),
            "policy_objective": float(best["policy_objective"]),
            "balanced_accuracy": float(best["balanced_accuracy"]),
            "roc_auc": roc_auc_manual(
                calib_df["label_sick"].to_numpy(),
                calib_df[score_name].to_numpy(dtype=np.float64),
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["policy_objective", "roc_auc"], ascending=[False, False])
        .reset_index(drop=True)
    )

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    opt = DiagnosticOptions().parse()
    if not getattr(opt, "calib_cond_tsv", None) or not getattr(opt, "test_cond_tsv", None):
        raise ValueError("diagnose_mesh_vae.py requires --calib_cond_tsv and --test_cond_tsv")
    set_seed(int(opt.seed))

    opt.phase = "diagnose"
    opt.is_train = False
    opt.use_cond = True
    opt.use_preprocessed = True
    opt.use_pointcloud = True
    opt.chamfer_npoints = int(opt.metric_npoints)

    load_cond_stats_into_opt(opt)
    if str(getattr(opt, "feature_source", "scan")) != "scan":
        opt.use_preprocessed = False
        opt.use_pointcloud = True
        opt.clinical_points = max(2, int(getattr(opt, "clinical_points", 32)))
        opt.ninput_edges = opt.clinical_points
        opt.metric_npoints = min(int(opt.metric_npoints), opt.clinical_points)
        opt.chamfer_npoints = opt.metric_npoints

    if int(getattr(opt, "cond_dim", 0)) == 0:
        extra = 0
        if getattr(opt, "cond_athero", False):
            extra += (1 if getattr(opt, "cond_athero_mode", "any") == "any" else 4)
        if getattr(opt, "cond_plaque", False):
            extra += 4
        opt.demo_cond_dim = 2 + n_age_dims(opt)
        opt.cond_dim = opt.demo_cond_dim + extra

    if not getattr(opt, "cond_athero", False):
        raise ValueError("This diagnostic script expects atherosclerosis conditioning to be enabled.")

    print("\n========== BUILD DATASETS ==========")
    calib_ds = build_dataset(opt, opt.calib_cond_tsv)
    test_ds = build_dataset(opt, opt.test_cond_tsv)

    calib_idx = sample_indices_balanced(
        calib_ds,
        int(opt.calib_max_samples),
        seed=int(opt.seed),
    )
    test_idx = (
        sample_indices_balanced(
            test_ds,
            int(opt.test_max_samples),
            seed=int(opt.seed) + 1,
        )
        if int(opt.test_max_samples) > 0
        else np.arange(len(test_ds), dtype=np.int64)
    )

    print(f"Calibration set: {len(calib_ds)} total, using {len(calib_idx)}")
    print(f"Test set:        {len(test_ds)} total, using {len(test_idx)}")

    print("\n========== LOAD MODEL ==========")
    model = MeshVAEModel(opt)
    model.load_network(opt.which_epoch)
    model.net.eval()

    print("\n========== BUILD REFERENCE SAMPLES ==========")
    calib_stats = gather_empirical_stats(calib_ds, opt)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(opt.checkpoints_dir) / opt.name / opt.results_subdir / f"{opt.which_epoch}_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_cache = generate_reference_sets(
        model=model,
        opt=opt,
        stats=calib_stats,
        out_dir=out_dir,
    )

    print("\n========== SCORE CALIBRATION ==========")
    calib_df = score_dataset("calibration", calib_ds, model, ref_cache, opt, indices=calib_idx)
    calib_df = add_group_prevalence(calib_df, calib_stats)
    calib_csv = out_dir / "calibration_scores.csv"
    calib_df.to_csv(calib_csv, index=False)
    print(f"[OUT] wrote {calib_csv}")

    print("\n========== FIT THRESHOLDS ==========")
    policy_mode = str(getattr(opt, "policy_mode", "prevalence"))

    if policy_mode == "global":
        prevalence_cutoff = None
        thr_df = fit_all_thresholds(calib_df, criterion=str(opt.threshold_metric))
    elif policy_mode == "prevalence":
        prevalence_cutoff = resolve_prevalence_cutoff(calib_df, opt)
        thr_df = fit_prevalence_policy_score_table(calib_df, prevalence_cutoff)
    else:
        prevalence_cutoff = None
        thr_df = fit_age_policy_score_table(calib_df)

    thr_csv = out_dir / "threshold_search.csv"
    thr_df.to_csv(thr_csv, index=False)
    print(thr_df.to_string(index=False))
    print(f"[OUT] wrote {thr_csv}")

    if opt.score_method == "auto":
        chosen_score = str(thr_df.iloc[0]["score_name"])
    else:
        chosen_score = str(opt.score_method)
        if chosen_score not in set(thr_df["score_name"]):
            raise ValueError(f"Requested score_method={chosen_score} not found in threshold table")

    chosen_row = thr_df.loc[thr_df["score_name"] == chosen_score].iloc[0]

    if policy_mode == "global":
        chosen_threshold = float(chosen_row["threshold"])
        thr_map = {"global": chosen_threshold}
        calib_pred_df = apply_threshold(calib_df, chosen_score, chosen_threshold)

    elif policy_mode == "prevalence":
        thr_map = {
            "lowprev": float(chosen_row["thr_lowprev"]),
            "highprev": float(chosen_row["thr_highprev"]),
        }
        calib_pred_df = apply_prevalence_policy_thresholds(calib_df, chosen_score, thr_map, prevalence_cutoff)

    else:
        thr_map = {
            "under50": float(chosen_row["thr_under50"]),
            "over50": float(chosen_row["thr_over50"]),
        }
        calib_pred_df = apply_age_policy_thresholds(calib_df, chosen_score, thr_map)

    calib_pred_csv = out_dir / "calibration_predictions.csv"
    calib_pred_df.to_csv(calib_pred_csv, index=False)
    print(f"[OUT] wrote {calib_pred_csv}")

    print("\n========== SCORE TEST ==========")
    test_df = score_dataset("test", test_ds, model, ref_cache, opt, indices=test_idx)
    test_df = add_group_prevalence(test_df, calib_stats)
    test_csv = out_dir / "test_scores.csv"
    test_df.to_csv(test_csv, index=False)
    print(f"[OUT] wrote {test_csv}")

    if policy_mode == "global":
        test_pred_df = apply_threshold(test_df, chosen_score, chosen_threshold)
    elif policy_mode == "prevalence":
        test_pred_df = apply_prevalence_policy_thresholds(test_df, chosen_score, thr_map, prevalence_cutoff)
    else:
        test_pred_df = apply_age_policy_thresholds(test_df, chosen_score, thr_map)

    test_pred_csv = out_dir / "test_predictions.csv"
    test_pred_df.to_csv(test_pred_csv, index=False)
    print(f"[OUT] wrote {test_pred_csv}")

    calib_metrics = binary_metrics(
        calib_pred_df["label_sick"].to_numpy(),
        calib_pred_df["pred_sick"].to_numpy(),
    )
    test_metrics = binary_metrics(
        test_pred_df["label_sick"].to_numpy(),
        test_pred_df["pred_sick"].to_numpy(),
    )

    calib_auc = roc_auc_manual(
        calib_df["label_sick"].to_numpy(),
        calib_df[chosen_score].to_numpy(dtype=np.float64),
    )
    test_auc = roc_auc_manual(
        test_df["label_sick"].to_numpy(),
        test_df[chosen_score].to_numpy(dtype=np.float64),
    )

    if policy_mode == "prevalence":
        calib_low = calib_pred_df[calib_pred_df["policy_band"] == "lowprev"]
        calib_high = calib_pred_df[calib_pred_df["policy_band"] == "highprev"]
        test_low = test_pred_df[test_pred_df["policy_band"] == "lowprev"]
        test_high = test_pred_df[test_pred_df["policy_band"] == "highprev"]

        calib_low_metrics = binary_metrics(calib_low["label_sick"].to_numpy(), calib_low["pred_sick"].to_numpy()) if len(calib_low) > 0 else {}
        calib_high_metrics = binary_metrics(calib_high["label_sick"].to_numpy(), calib_high["pred_sick"].to_numpy()) if len(calib_high) > 0 else {}
        test_low_metrics = binary_metrics(test_low["label_sick"].to_numpy(), test_low["pred_sick"].to_numpy()) if len(test_low) > 0 else {}
        test_high_metrics = binary_metrics(test_high["label_sick"].to_numpy(), test_high["pred_sick"].to_numpy()) if len(test_high) > 0 else {}

    elif policy_mode == "age":
        calib_low = calib_pred_df[calib_pred_df["age_policy_band"] == "under50"]
        calib_high = calib_pred_df[calib_pred_df["age_policy_band"] == "over50"]
        test_low = test_pred_df[test_pred_df["age_policy_band"] == "under50"]
        test_high = test_pred_df[test_pred_df["age_policy_band"] == "over50"]

        calib_low_metrics = binary_metrics(calib_low["label_sick"].to_numpy(), calib_low["pred_sick"].to_numpy()) if len(calib_low) > 0 else {}
        calib_high_metrics = binary_metrics(calib_high["label_sick"].to_numpy(), calib_high["pred_sick"].to_numpy()) if len(calib_high) > 0 else {}
        test_low_metrics = binary_metrics(test_low["label_sick"].to_numpy(), test_low["pred_sick"].to_numpy()) if len(test_low) > 0 else {}
        test_high_metrics = binary_metrics(test_high["label_sick"].to_numpy(), test_high["pred_sick"].to_numpy()) if len(test_high) > 0 else {}

    else:
        calib_low_metrics = {}
        calib_high_metrics = {}
        test_low_metrics = {}
        test_high_metrics = {}

    summary = {
        "which_epoch": str(opt.which_epoch),
        "policy_mode": str(getattr(opt, "policy_mode", "prevalence")),
        "chosen_score": chosen_score,
        "chosen_thresholds": thr_map,
        "prevalence_cutoff": prevalence_cutoff,
        "threshold_metric": (
            str(opt.threshold_metric) if policy_mode == "global" else "policy_objective"
        ),
        "score_table": json.loads(thr_df.to_json(orient="records")),
        "calibration": {
            **calib_metrics,
            "roc_auc": calib_auc,
            "n_scored": int(len(calib_df)),
            "lowprev": calib_low_metrics,
            "highprev": calib_high_metrics,
        },
        "test": {
            **test_metrics,
            "roc_auc": test_auc,
            "n_scored": int(len(test_df)),
            "lowprev": test_low_metrics,
            "highprev": test_high_metrics,
        },
        "paths": {
            "calibration_scores_csv": str(calib_csv),
            "test_scores_csv": str(test_csv),
            "threshold_search_csv": str(thr_csv),
            "calibration_predictions_csv": str(calib_pred_csv),
            "test_predictions_csv": str(test_pred_csv),
        },
        "settings": {
            "metric_npoints": int(opt.metric_npoints),
            "metric_chunk": int(opt.metric_chunk),
            "distance_batch_size": int(opt.distance_batch_size),
            "samples_per_class": int(opt.samples_per_class),
            "distance_temperature": float(opt.distance_temperature),
            "prior_mode": str(getattr(opt, "prior_mode", "flat")),
            "policy_mode": str(getattr(opt, "policy_mode", "prevalence")),
            "prevalence_band_mode": str(getattr(opt, "prevalence_band_mode", "global")),
            "cond_athero_mode": str(getattr(opt, "cond_athero_mode", "any")),
            "only_gender": getattr(opt, "only_gender", None),
        },
    }

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"[OUT] wrote {summary_json}")
    # Compact one-row CSV for easy comparison across runs.
    summary_row = {
        "name": str(opt.name),
        "which_epoch": str(opt.which_epoch),
        "only_gender": getattr(opt, "only_gender", None),
        "policy_mode": str(getattr(opt, "policy_mode", "prevalence")),
        "chosen_score": chosen_score,
        "prevalence_cutoff": prevalence_cutoff,
        "calib_balanced_accuracy": calib_metrics.get("balanced_accuracy", np.nan),
        "calib_auc": calib_auc,
        "calib_sensitivity": calib_metrics.get("sensitivity", np.nan),
        "calib_specificity": calib_metrics.get("specificity", np.nan),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy", np.nan),
        "test_auc": test_auc,
        "test_sensitivity": test_metrics.get("sensitivity", np.nan),
        "test_specificity": test_metrics.get("specificity", np.nan),
        "test_precision": test_metrics.get("precision", np.nan),
        "test_f1": test_metrics.get("f1", np.nan),
        "results_dir": str(out_dir),
    }
    summary_csv = out_dir / "summary_row.csv"
    pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)
    print(f"[OUT] wrote {summary_csv}")

    print("\n========== DONE ==========")
    print(json.dumps({
        "chosen_score": chosen_score,
        "policy_mode": summary["policy_mode"],
        "chosen_thresholds": summary["chosen_thresholds"],
        "prevalence_cutoff": summary["prevalence_cutoff"],
        "calibration": summary["calibration"],
        "test": summary["test"],
        "results_dir": str(out_dir),
    }, indent=2))

if __name__ == "__main__":
    main()
