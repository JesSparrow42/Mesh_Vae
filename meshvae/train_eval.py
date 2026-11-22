from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .datasets import mesh_collate_fn
from .conditioning import inverse_regression_target
from .geometry import clean_mesh, mesh_stats_o3d


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(pred)
    y = y[valid]
    pred = pred[valid]
    if y.size == 0:
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"), "pearson": float("nan")}
    residual = pred - y
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / denom) if denom > 0.0 else float("nan")
    pearson = (
        float(np.corrcoef(y, pred)[0, 1])
        if y.size > 1 and np.std(y) > 0.0 and np.std(pred) > 0.0
        else float("nan")
    )
    return {"r2": r2, "mae": mae, "rmse": rmse, "pearson": pearson}

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
    for k in [
        "loss", "chamfer", "kl", "plaque_loss", "athero_loss", "athero_acc",
        "binary_loss", "binary_acc", "regression_loss", "age_adv_loss", "age_adv_acc",
    ]:
        if k in df.columns:
            out[k] = float(df[k].mean())
    if {"athero_target_any", "athero_prob_sick"}.issubset(df.columns):
        y = df["athero_target_any"].to_numpy(dtype=np.float64) >= 0.5
        prob = df["athero_prob_sick"].to_numpy(dtype=np.float64)
        pred = prob >= 0.5
        tp = int(np.sum(y & pred))
        tn = int(np.sum((~y) & (~pred)))
        fp = int(np.sum((~y) & pred))
        fn = int(np.sum(y & (~pred)))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        out.update({
            "athero_auc": _roc_auc(y, prob),
            "athero_balanced_accuracy": 0.5 * (sensitivity + specificity),
            "athero_sensitivity": sensitivity,
            "athero_specificity": specificity,
        })
    if {"binary_target", "binary_prob"}.issubset(df.columns):
        y = df["binary_target"].to_numpy(dtype=np.float64) >= 0.5
        prob = df["binary_prob"].to_numpy(dtype=np.float64)
        pred = prob >= 0.5
        tp = int(np.sum(y & pred))
        tn = int(np.sum((~y) & (~pred)))
        fp = int(np.sum((~y) & pred))
        fn = int(np.sum(y & (~pred)))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        out.update({
            "binary_auc": _roc_auc(y, prob),
            "binary_balanced_accuracy": 0.5 * (sensitivity + specificity),
            "binary_sensitivity": sensitivity,
            "binary_specificity": specificity,
        })
    if {"regression_target", "regression_pred"}.issubset(df.columns):
        target_mean = float(getattr(opt, "aux_regression_mean", 0.0))
        target_std = float(getattr(opt, "aux_regression_std", 1.0))
        transform = str(getattr(opt, "aux_regression_transform", "identity"))
        y_transformed = df["regression_target"].to_numpy(dtype=np.float64) * target_std + target_mean
        pred_transformed = df["regression_pred"].to_numpy(dtype=np.float64) * target_std + target_mean
        transformed_metrics = regression_metrics(y_transformed, pred_transformed)
        out.update({f"regression_transformed_{key}": value for key, value in transformed_metrics.items()})
        y = inverse_regression_target(y_transformed, transform)
        pred = inverse_regression_target(pred_transformed, transform)
        metrics = regression_metrics(y, pred)
        out.update({f"regression_{key}": value for key, value in metrics.items()})
    return out

def split_dataset(dataset, ratio, seed, labels=None):
    n = len(dataset)
    rng = np.random.default_rng(seed)

    if labels is None and hasattr(dataset, "cond") and dataset.cond is not None:
        try:
            from .conditioning import athero_slice_from_opt
            sl = athero_slice_from_opt(dataset.opt)
            if sl is not None:
                labels = (np.max(dataset.cond[:, sl], axis=1) > 0.5).astype(np.int64)
        except Exception:
            labels = None

    if labels is not None and len(np.unique(labels)) > 1:
        labels = np.asarray(labels).astype(np.int64)
        train_parts = []
        val_parts = []
        for cls in np.unique(labels):
            cls_idx = np.flatnonzero(labels == cls)
            rng.shuffle(cls_idx)
            n_val_cls = max(1, int(round(len(cls_idx) * float(ratio))))
            n_val_cls = min(n_val_cls, max(len(cls_idx) - 1, 1))
            val_parts.append(cls_idx[:n_val_cls])
            train_parts.append(cls_idx[n_val_cls:])
        val_idx = np.concatenate(val_parts)
        train_idx = np.concatenate(train_parts)
        rng.shuffle(val_idx)
        rng.shuffle(train_idx)
        print(
            f"[SPLIT] stratified train={len(train_idx)} val={len(val_idx)} "
            f"train_pos={int(labels[train_idx].sum())} val_pos={int(labels[val_idx].sum())}",
            flush=True,
        )
    else:
        n_val = max(1, int(round(n * float(ratio))))
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        print(f"[SPLIT] random train={len(train_idx)} val={len(val_idx)}", flush=True)

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
