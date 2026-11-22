from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .datasets import mesh_collate_fn
from .geometry import clean_mesh, mesh_stats_o3d

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
    for k in ["loss", "chamfer", "kl", "plaque_loss", "athero_loss", "athero_acc", "age_adv_loss", "age_adv_acc"]:
        if k in df.columns:
            out[k] = float(df[k].mean())
    return out

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
