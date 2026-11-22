from __future__ import print_function
from os.path import join
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from threading import Thread
import numpy as np
from heapq import heappop, heapify
from torch.nn import ConstantPad2d
import os, sys, re
import time
import argparse
import torch.utils.data as data
import numpy as np
import pickle
from glob import glob   # ← NEW
from tempfile import mkstemp
from shutil import move
import ntpath
from pathlib import Path
import csv
from collections import Counter
import copy
import math
import json
import yaml  # optional, only used if available
from datetime import datetime
import matplotlib.pyplot as plt
import math
import trimesh
from models.mesh_conv_model import MeshConv, MeshPool, MeshUnpool, MeshUnion
from typing import List, Optional
from pathlib import Path
import random
import ast
from torch.utils.data import DataLoader as TorchDataLoader

# train -> Mesh Clas/data/util_writer/trainoptions/test
# test -> testoptions
# data -> basedata
# trainoptions -> baseoptions
# Mesh Clas -> util/Networks 
# Networks -> layers conv/pool/unpool
# pool -> union



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

MESH_EXTENSIONS = [
    '.obj',
]

def is_mesh_file(filename):
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)

### LOGGING ###
def _exp_dir(opt):
    return os.path.join(opt.checkpoints_dir, opt.name)

def _metrics_csv_path(opt):
    os.makedirs(_exp_dir(opt), exist_ok=True)
    return os.path.join(_exp_dir(opt), "metrics.csv")

def _append_metrics_csv(opt, row: dict):
    path = _metrics_csv_path(opt)
    new_file = not os.path.isfile(path)
    # keep column order stable
    fieldnames = ["epoch", "lr", "loss", "elbo", "log_px", "kl", "n"]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        # coerce to plain Python scalars/None for CSV
        clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        w.writerow(clean)

def _plot_metrics_from_csv(opt):
    """Create loss.png, elbo.png, log_px.png, kl.png in the experiment folder."""
    path = _metrics_csv_path(opt)
    if not os.path.isfile(path):
        print(f"[WARN] metrics CSV not found at {path}, skipping plots.")
        return
    # read
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        print("[WARN] metrics CSV empty, skipping plots.")
        return
    # convert columns
    def col(name, cast=float):
        out = []
        for row in rows:
            try:
                out.append(cast(row[name]))
            except Exception:
                out.append(float("nan"))
        return out

    epochs = col("epoch", int)
    series = {
        "loss":   col("loss", float),
        "elbo":   col("elbo", float),
        "log_px": col("log_px", float),
        "kl":     col("kl", float),
    }

    for key, vals in series.items():
        try:
            plt.figure()
            plt.plot(epochs, vals)
            plt.xlabel("epoch")
            plt.ylabel(key)
            plt.title(key)
            plt.grid(True, alpha=0.3)
            out_png = os.path.join(_exp_dir(opt), f"{key}.png")
            plt.savefig(out_png, bbox_inches="tight")
            plt.close()
            print(f"[PLOT] wrote {out_png}")
        except Exception as e:
            print(f"[WARN] failed to plot {key}: {e}")



# -------------------------------
# 1) Edge selection + padding
# -------------------------------


def _select_edge_indices(total_edges: int,
                         target_edges: int,
                         strategy: str = "stride",
                         seed: int = 0) -> np.ndarray:
    if target_edges <= 0 or total_edges <= 0:
        return np.zeros(0, dtype=np.int64)
    if total_edges == target_edges:
        return np.arange(total_edges, dtype=np.int64)

    if total_edges > target_edges:
        if strategy == "random":
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(total_edges, target_edges, replace=False)).astype(np.int64)
        elif strategy == "head":
            idx = np.arange(target_edges, dtype=np.int64)
        else:
            idx = np.linspace(0, total_edges - 1, num=target_edges, dtype=np.int64)
        print(f"[PAD] truncate edges {total_edges} -> {target_edges} (strategy='{strategy}')")
        return idx

    # padding: duplicate evenly (no -1s)
    idx = np.round(np.linspace(0, total_edges - 1, num=target_edges)).astype(np.int64)
    idx = np.clip(idx, 0, total_edges - 1)
    print(f"[PAD] pad edges {total_edges} -> {target_edges} (duplicates via '{strategy}')")
    return idx


def pad(input_arr,
        ninput_edges: int,
        pad_value: float = 0.0,
        truncate_strategy: str = "stride",
        seed: int = 0):
    is_tensor = False
    try:
        import torch
        if torch.is_tensor(input_arr):
            is_tensor = True
            device = input_arr.device
            dtype  = input_arr.dtype
            arr = input_arr.detach().cpu().numpy()
        else:
            arr = np.asarray(input_arr)
    except Exception:
        arr = np.asarray(input_arr)

    if arr.ndim != 2:
        raise ValueError(f"pad() expects (C, E) array, got shape {arr.shape}")

    C, E_raw = arr.shape
    E_tgt = int(ninput_edges)
    if E_raw == E_tgt:
        out = arr.copy()
    else:
        idx = _select_edge_indices(E_raw, E_tgt, strategy=truncate_strategy, seed=seed)
        out = arr[:, idx]

    if is_tensor:
        import torch
        return torch.from_numpy(out).to(device=device, dtype=dtype)
    return out.astype(arr.dtype, copy=False)



# -----------------------------------------
# 2) Geometry extraction + OBJ save helpers
# -----------------------------------------

def _get_mesh_geometry(mesh, fallback_path=None):
    """
    Extract vertices & faces from a mesh-like object; if missing, try to load from fallback OBJ path.
    Returns (vertices, faces) or (None, None). Prints shapes for debugging.
    """
    verts = faces = None

    for v_name in ("vertices", "verts", "vs", "V", "v"):
        if hasattr(mesh, v_name):
            verts = getattr(mesh, v_name)
            break
    for f_name in ("faces", "F", "triangles", "tris"):
        if hasattr(mesh, f_name):
            faces = getattr(mesh, f_name)
            break

    if (verts is None or faces is None) and fallback_path:
        try:
            import trimesh
            tm = trimesh.load(fallback_path, force='mesh')
            verts = tm.vertices
            faces = tm.faces
        except Exception as e:
            print(f"[GEOM] trimesh load failed for {fallback_path}: {e}")
            return None, None

    if verts is not None:
        verts = np.asarray(verts, dtype=np.float32)
    if faces is not None:
        faces = np.asarray(faces, dtype=np.int64)

    if verts is not None:
        print(f"[GEOM] _get_mesh_geometry: verts={verts.shape}")
    if faces is not None:
        print(f"[GEOM] _get_mesh_geometry: faces={faces.shape}")

    return verts, faces


def _parse_obj_edges_from_file(path: str):
    edges = []
    try:
        with open(path, 'r', errors='ignore') as f:
            for line in f:
                s = line.lstrip()
                if not s.startswith('l '):
                    continue
                parts = s.split()
                # tokens after 'l' can be "i", "i/j/k", or more than 2 -> split into segments
                idxs = []
                for tok in parts[1:]:
                    tok = tok.split('/')[0]
                    try:
                        idxs.append(int(tok) - 1)  # OBJ 1-based -> 0-based
                    except Exception:
                        pass
                for a, b in zip(idxs, idxs[1:]):
                    if a >= 0 and b >= 0:
                        u, v = (a, b) if a < b else (b, a)
                        edges.append((u, v))
    except Exception as e:
        print(f"[GEOM] _parse_obj_edges_from_file failed: {e}")
        return np.empty((0, 2), dtype=np.int64)

    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.asarray(edges, dtype=np.int64), axis=0)


def extract_geometry_and_edges(mesh_or_path):
    """
    Return (V, F, E_idx), robust to lists/tuples/np arrays/paths/mesh-like.
    Falls back to:
      - faces->edges,
      - OBJ 'l' lines (wireframe),
      - filters invalid edges.
    """
    V = F = E_idx = None
    obj = None
    path = None

    # --- Unwrap carriers (list/tuple/np.ndarray/scalars) to a usable thing ---
    def _unwrap(x):
        import numpy as _np
        # numpy array?
        if isinstance(x, _np.ndarray):
            # scalar array
            if x.ndim == 0:
                return _unwrap(x.item())
            # 1D array (typical collate)
            if x.ndim == 1:
                # prefer first element; caller should pass the correct i-th element,
                # but this keeps us alive even if not.
                return _unwrap(x[0])
            # fallback to first flat element
            return _unwrap(next(iter(x.flat)))
        # single-item list/tuple?
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return _unwrap(x[0])
        return x

    mesh_or_path = _unwrap(mesh_or_path)

    # if it's a string-like path at this point
    if isinstance(mesh_or_path, (str, bytes, os.PathLike)):
        path = str(mesh_or_path)
        print(f"[GEOM] input path: {path}")
    else:
        obj = mesh_or_path

    # Pull from mesh-like object
    if obj is not None:
        for p_attr in ("obj_path", "fullfilename", "file", "path", "filepath"):
            if hasattr(obj, p_attr) and isinstance(getattr(obj, p_attr), str):
                cand = getattr(obj, p_attr)
                if os.path.isfile(cand):
                    path = cand
                    print(f"[GEOM] using mesh path from attribute '{p_attr}': {path}")
                    break

        # vertices
        for v_attr in ("vertices", "verts", "vs", "V", "v", "pos"):
            if hasattr(obj, v_attr):
                Vv = getattr(obj, v_attr)
                try:
                    import torch
                    if isinstance(Vv, torch.Tensor):
                        Vv = Vv.detach().cpu().numpy()
                except Exception:
                    pass
                Vv = np.asarray(Vv, dtype=np.float32)
                if Vv.ndim == 2 and Vv.shape[1] in (2, 3):
                    if Vv.shape[1] == 2:
                        Vv = np.concatenate([Vv, np.zeros((Vv.shape[0], 1), dtype=np.float32)], axis=1)
                    V = Vv
                break

        # faces
        for f_attr in ("faces", "F", "triangles", "tris"):
            if hasattr(obj, f_attr):
                Fv = getattr(obj, f_attr)
                try:
                    import torch
                    if isinstance(Fv, torch.Tensor):
                        Fv = Fv.detach().cpu().numpy()
                except Exception:
                    pass
                Fv = np.asarray(Fv, dtype=np.int64)
                if Fv.ndim == 2 and Fv.shape[1] == 3:
                    F = Fv
                break

        # edges if present
        for e_attr in ("edges", "e2v", "ev", "edge_index", "edges_to_vertices"):
            if hasattr(obj, e_attr):
                Ev = getattr(obj, e_attr)
                try:
                    import torch
                    if isinstance(Ev, torch.Tensor):
                        Ev = Ev.detach().cpu().numpy()
                except Exception:
                    pass
                E_val = np.asarray(Ev)
                if E_val.ndim == 2 and E_val.shape[1] == 2:
                    pass
                elif E_val.ndim == 2 and E_val.shape[0] == 2:
                    E_val = E_val.T
                else:
                    E_val = np.empty((0, 2), dtype=np.int64)
                E_idx = E_val.astype(np.int64, copy=False)
                break

    # If still missing, load from file
    if (V is None or F is None) and path and os.path.isfile(path):
        try:
            import trimesh
            tm = trimesh.load(path, force='mesh', process=False)
            if V is None: V = np.asarray(tm.vertices, dtype=np.float32)
            if F is None and tm.faces is not None and len(tm.faces) > 0:
                F = np.asarray(tm.faces, dtype=np.int64)
        except Exception as e:
            print(f"[GEOM] trimesh load failed: {e}")

    # faces -> edges
    if (E_idx is None or E_idx.size == 0) and F is not None and np.asarray(F).size > 0:
        es = set()
        F_arr = np.asarray(F, dtype=np.int64).reshape(-1, 3)
        for (a, b, c) in F_arr:
            for u, v in ((a, b), (b, c), (c, a)):
                if u > v: u, v = v, u
                es.add((int(u), int(v)))
        E_idx = np.array(sorted(es), dtype=np.int64)
        print(f"[GEOM] derived edges from faces: {E_idx.shape}")

    # parse wireframe edges if faces missing
    if (E_idx is None or E_idx.size == 0) and path and os.path.isfile(path):
        parsed = _parse_obj_edges_from_file(path)
        if parsed.size:
            E_idx = parsed
            print(f"[GEOM] parsed edges from OBJ 'l' lines: {E_idx.shape}")

    # Final checks
    if V is None:
        print("[GEOM] No vertices available.")
        return None, None, None
    if F is None and (E_idx is None or E_idx.size == 0):
        print("[GEOM] No faces or edges available.")
        return None, None, None

    if E_idx is not None and E_idx.size:
        Vn = V.shape[0]
        mask = (E_idx[:, 0] >= 0) & (E_idx[:, 0] < Vn) & (E_idx[:, 1] >= 0) & (E_idx[:, 1] < Vn)
        if np.count_nonzero(mask) != E_idx.shape[0]:
            print(f"[GEOM] filtered {E_idx.shape[0] - np.count_nonzero(mask)} invalid edges")
        E_idx = E_idx[mask]
        if E_idx.size:
            u = np.minimum(E_idx[:, 0], E_idx[:, 1])
            v = np.maximum(E_idx[:, 0], E_idx[:, 1])
            E_idx = np.unique(np.stack([u, v], axis=1), axis=0)

    print(f"[GEOM] final: V={V.shape}, F={None if F is None else F.shape}, E={None if E_idx is None else E_idx.shape}")
    return V, F, E_idx



def save_obj(vertices, faces, filepath, edges=None):
    try:
        import torch
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        if faces is not None and isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()
        if edges is not None and isinstance(edges, torch.Tensor):
            edges = edges.detach().cpu().numpy()
    except Exception:
        pass

    V = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    F = None if faces is None else np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    E = None if edges is None else np.asarray(edges, dtype=np.int64).reshape(-1, 2)

    with open(filepath, "w") as f:
        for (x, y, z) in V:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        if F is not None and F.size:
            for (a, b, c) in F:
                f.write(f"f {int(a)+1} {int(b)+1} {int(c)+1}\n")
        elif E is not None and E.size:
            for (i, j) in E:
                f.write(f"l {int(i)+1} {int(j)+1}\n")

    print(f"[SAVE_OBJ] wrote: {filepath}  | V={V.shape[0]}  F={0 if F is None else F.shape[0]}  E={0 if E is None else E.shape[0]}")



# -----------------------------------------
# 3) Deformation for reconstruction + saver
# -----------------------------------------

def adjust_vertices_to_edges(vertices, edge_index, target_lengths, iterations=10):
    V_new = np.asarray(vertices, dtype=np.float32).copy()
    E_idx = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    tgt = np.asarray(target_lengths, dtype=np.float32).reshape(-1)

    for _ in range(int(iterations)):
        for e, (i, j) in enumerate(E_idx):
            if e >= tgt.shape[0]:
                break
            L_target = float(max(tgt[e], 1e-8))
            vi, vj = V_new[i], V_new[j]
            vec = vj - vi
            L = float(np.linalg.norm(vec))
            if L < 1e-8:
                continue
            delta = (L_target - L)
            diru = vec / L
            V_new[i] -= 0.5 * delta * diru
            V_new[j] += 0.5 * delta * diru
    return V_new


def _save_reconstructions(opt, epoch, x, x_mean, meshes=None, k=2):
    import matplotlib.pyplot as plt

    if x is None or x_mean is None:
        print("[RECON] no tensors provided.")
        return

    # Output dir
    exp_base = getattr(opt, "checkpoints_dir", "./checkpoints")
    exp_name = getattr(opt, "name", "experiment")
    out_dir = os.path.join(exp_base, exp_name, "recon", f"epoch_{int(epoch):03d}")
    os.makedirs(out_dir, exist_ok=True)

    # To numpy
    x_np = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    x_mean_np = x_mean.detach().cpu().numpy() if hasattr(x_mean, "detach") else np.asarray(x_mean)
    if x_np.ndim == 4 and x_np.shape[-1] == 1:        x_np = x_np[..., 0]
    if x_mean_np.ndim == 4 and x_mean_np.shape[-1] == 1: x_mean_np = x_mean_np[..., 0]

    B = min(int(k), x_np.shape[0])
    print(f"[RECON] x shape={x_np.shape}, x_mean shape={x_mean_np.shape}, saving {B}")

    # Heatmaps + NPZs (always saved)
    for i in range(B):
        np.savez_compressed(os.path.join(out_dir, f"original_{i}.npz"), edge_features=x_np[i])
        np.savez_compressed(os.path.join(out_dir, f"recon_{i}.npz"),    edge_features=x_mean_np[i])
        plt.figure(); plt.imshow(x_np[i], aspect='auto');    plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"original {i} (C×E)"); plt.savefig(os.path.join(out_dir, f"original_{i}.png"), bbox_inches="tight"); plt.close()
        plt.figure(); plt.imshow(x_mean_np[i], aspect='auto'); plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"recon {i} (C×E)");    plt.savefig(os.path.join(out_dir, f"recon_{i}.png"),    bbox_inches="tight"); plt.close()

    # Helper to pick the i-th carrier robustly
    def _pick_carrier(M, i):
        import numpy as _np
        if M is None:
            return None
        # list/tuple
        if isinstance(M, (list, tuple)):
            if len(M) == 0:
                return None
            if len(M) == 1 and isinstance(M[0], _np.ndarray):
                # common case: [ array([...paths...]) ]
                arr = M[0]
                if arr.ndim == 0:
                    return arr.item()
                if arr.ndim == 1:
                    return arr[i if i < arr.shape[0] else 0]
                return next(iter(arr.flat))
            return M[i if i < len(M) else 0]

        # numpy array at top-level
        if isinstance(M, _np.ndarray):
            if M.ndim == 0:
                return M.item()
            if M.ndim == 1:
                return M[i if i < M.shape[0] else 0]
            return next(iter(M.flat))

        # single object
        return M

    # Mesh OBJs
    for i in range(B):
        try:
            carrier = _pick_carrier(meshes, i)

            # Try several ways to get geometry:
            # 1) direct (handles path, mesh-like, and arrays of either)
            V, F, E_idx = extract_geometry_and_edges(carrier)

            # 2) If that failed and the carrier has an obj_path attribute, try that
            if (V is None or (F is None and (E_idx is None or E_idx.size == 0))) and hasattr(carrier, "obj_path"):
                V, F, E_idx = extract_geometry_and_edges(getattr(carrier, "obj_path"))

            if V is None or (F is None and (E_idx is None or E_idx.size == 0)):
                print(f"[RECON] sample {i}: after unwrapping & fallbacks, geometry still missing. Skipping OBJ write.")
                continue

            # Align edges with model features
            feats_i = x_np[i]      # (C, E_model)
            recon_i = x_mean_np[i] # (C, E_model)
            E_model = feats_i.shape[1]
            E_geom  = E_idx.shape[0] if E_idx is not None else 0
            if E_geom == 0 or E_model == 0:
                print(f"[RECON] sample {i}: zero edges after alignment, skipping OBJ write.")
                continue
            use_E = min(E_geom, E_model)
            if use_E != E_model or use_E != E_geom:
                print(f"[RECON] sample {i}: edge mismatch geom={E_geom}, model={E_model} -> first {use_E}")
            E_idx  = E_idx[:use_E]
            feats_i = feats_i[:, :use_E]
            recon_i = recon_i[:, :use_E]

            # Pick channel most correlated with true lengths
            vecs = V[E_idx[:, 1]] - V[E_idx[:, 0]]
            true_lengths = np.linalg.norm(vecs, axis=1)
            best_c, best_corr = 0, -1.0
            for c in range(feats_i.shape[0]):
                col = feats_i[c]
                s = float(np.std(col))
                if s < 1e-9: 
                    continue
                corr = np.corrcoef(col, true_lengths)[0, 1]
                if not np.isnan(corr) and abs(corr) > best_corr:
                    best_corr = abs(corr)
                    best_c = c

            tgt = np.maximum(recon_i[best_c], 1e-8)
            V_recon = adjust_vertices_to_edges(V, E_idx, tgt, iterations=10)

            save_obj(V,       F, os.path.join(out_dir, f"original_{i}.obj"), edges=E_idx)
            save_obj(V_recon, F, os.path.join(out_dir, f"recon_{i}.obj"),   edges=E_idx)

        except Exception as e:
            print(f"[RECON] sample {i}: exception during OBJ export: {e}")


# ----- metrics logging (minimal, safe to import) -----
class CSVLogger:
    def __init__(self, path: str, fieldnames: list[str]):
        self.path = path
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._csv = csv
        self._file_exists = os.path.isfile(self.path)

    def write(self, row: dict):
        # keep unknown keys out to avoid column drift
        row = {k: row.get(k, "") for k in self.fieldnames}
        with open(self.path, "a", newline="") as f:
            w = self._csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._file_exists:
                w.writeheader()
                self._file_exists = True
            w.writerow(row)


### SAMPLE FUNCTION ###
def run_sample_prior(opt):
    """
    Offline sampling entry: load checkpoint, build a tiny test loader to obtain a mesh,
    then draw --sample_k prior samples and decode them.
    """
    import copy as _copy

    # build a minimal "test" copy of options (to get a mesh)
    test_opt = _copy.deepcopy(opt)
    test_opt.phase = 'test'
    # make sure we can actually load a few meshes
    if not hasattr(test_opt, 'split_ratio') or test_opt.split_ratio is None:
        test_opt.split_ratio = 0.2
    if not hasattr(test_opt, 'split_seed') or test_opt.split_seed is None:
        test_opt.split_seed = 0
    # use a tiny batch
    test_opt.batch_size = max(1, min(getattr(test_opt, 'batch_size', 1), 2))

    # dataset + model
    dataset = DataLoader(test_opt)
    if len(dataset) == 0:
        raise RuntimeError("No meshes in test loader; cannot sample.")
    model = create_model(test_opt)
    model.load_network(opt.which_epoch)

    # take one batch to populate model.mesh
    first = next(iter(dataset))
    model.set_input(first)
    # remember the paths for the sampler
    if isinstance(first, dict) and "obj_path" in first:
        setattr(model, "mesh_paths", first["obj_path"])

    # seeding (optional)
    if getattr(opt, 'sample_seed', 0):
        torch.manual_seed(int(opt.sample_seed))
        np.random.seed(int(opt.sample_seed))

    # choose out_dir
    if opt.sample_out:
        out_dir = opt.sample_out
        os.makedirs(out_dir, exist_ok=True)
    else:
        # use checkpoint epoch/label
        tag = str(opt.which_epoch).lower()
        out_dir = os.path.join(_exp_dir(opt), "samples", f"epoch_{tag}")

    _decode_prior_samples(opt, model, B=int(opt.sample_k), out_dir=out_dir)

### HELPERS ###
@torch.no_grad()
def _decode_prior_samples(opt, model, B: int, out_dir: str):
    """
    Draw B latents from the model prior and decode to edge features [C,E].
    Keeps NPZ + quick PNG (as before). NEW: writes OBJ meshes when geometry
    is available from the cached test mesh (or mesh file paths). If we fall
    back to MeshStub, we skip OBJ (no vertices/faces there).
    """

    os.makedirs(out_dir, exist_ok=True)

    # -------- choose / build meshes (unchanged) --------
    meshes = None
    start_E = None

    # Prefer a cached small mesh saved during training
    cached = getattr(model, "mesh", None)
    if isinstance(cached, (list, tuple)) and len(cached) > 0 and hasattr(cached[0], "edges_count"):
        base = cached[0]
        try:
            start_E = int(getattr(base, "edges_count"))
        except Exception:
            start_E = None

        if start_E is not None:
            meshes = [copy.deepcopy(base) for _ in range(B)]
            print(f"[SAMPLE] using cached test mesh; edges_count={start_E}")

    # Fall back to minimal stubs with the correct E0 (unchanged)
    if meshes is None:
        E0 = int(getattr(opt, "ninput_edges", 0) or 0)
        if E0 <= 0:
            raise RuntimeError("[SAMPLE] Need a valid --ninput_edges for decoder start_E.")

        class _MeshStub:
            __slots__ = ("edges_count", "gemm_edges")
            def __init__(self, e: int):
                self.edges_count = int(e)
                idx  = np.arange(e, dtype=np.int64)
                prev = (idx - 1) % e
                nxt  = (idx + 1) % e
                # columns: [self, prev, next, self] -- valid shape and dtype
                self.gemm_edges = np.stack([idx, prev, nxt, idx], axis=1)

        meshes = [_MeshStub(E0) for _ in range(B)]
        start_E = E0
        print(f"[SAMPLE] built {B} mesh stubs with edges_count={E0}")

    print(f"[SAMPLE] start_E = {start_E}")

    # -------- p(z) and decode (unchanged) --------
    pz = model.net.prior(B, model.net.latent_dim, device=model.device)  # Gaussian or Boson
    z  = pz.rsample() if hasattr(pz, "rsample") else pz.sample()
    if isinstance(z, (tuple, list)):
        z = z[0]
    print("[SAMPLE][DBG] z shape:", tuple(z.shape))

    x_mean = model.net.decoder(z, meshes, start_E)
    print("[SAMPLE][DBG] raw decode shape:", tuple(x_mean.shape))

    # normalize to [B, C, E]
    if x_mean.dim() == 4 and x_mean.shape[-1] == 1:
        x_mean = x_mean.squeeze(-1)
    if x_mean.dim() == 2:
        x_mean = x_mean.unsqueeze(0)
    if x_mean.dim() == 3:
        pass
    elif x_mean.dim() == 3:
        pass
    elif x_mean.dim() != 3:
        raise RuntimeError(f"[SAMPLE] Unexpected decoded tensor shape {tuple(x_mean.shape)}")

    B_out, C, E = x_mean.shape
    print(f"[SAMPLE][DBG] normalized decode shape: B={B_out}, C={C}, E={E}")

    # -------- save NPZ + PNG (unchanged) --------
    for i in range(B_out):
        x_np = x_mean[i].detach().cpu().numpy()  # (C, E)

        # NPZ
        np.savez_compressed(os.path.join(out_dir, f"sample_{i}.npz"),
                            edge_features=x_np)

        # quick PNG
        img = x_np
        if x_np.ndim == 1:
            img = x_np[None, :] ; Cimg = 1
        else:
            Cimg = x_np.shape[0]
        plt.figure()
        plt.imshow(img, aspect="auto", interpolation="nearest")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"sample {i} (C={Cimg})")
        plt.savefig(os.path.join(out_dir, f"sample_{i}.png"), bbox_inches="tight")
        plt.close()

    # -------- NEW: also write OBJ when geometry is available --------
    # helper that accepts either a mesh object OR a file path
    def _extract_geometry_and_edges(carrier):
        import numpy as _np
        V = F = E_idx = None

        # path case
        if isinstance(carrier, (str, os.PathLike)):
            try:
                tm = trimesh.load(carrier, force='mesh')
                V = _np.asarray(tm.vertices)
                F = _np.asarray(tm.faces, dtype=_np.int64)
            except Exception:
                return None, None, None
        else:
            # mesh object case
            for a in ("vertices", "vs", "v", "V"):
                if hasattr(carrier, a):
                    V = _np.asarray(getattr(carrier, a))
                    break
            for a in ("faces", "fs", "f", "triangles", "F"):
                if hasattr(carrier, a):
                    F = _np.asarray(getattr(carrier, a), dtype=_np.int64)
                    break
            for a in ("edges", "e2v", "ev", "edge_index", "edges_to_vertices"):
                if hasattr(carrier, a):
                    E_idx = _np.asarray(getattr(carrier, a), dtype=_np.int64)
                    break

        # derive edges from faces if missing
        if E_idx is None and F is not None:
            S = set()
            for (a, b, c) in F:
                for (i, j) in ((a, b), (b, c), (c, a)):
                    if i > j: i, j = j, i
                    S.add((int(i), int(j)))
            E_idx = _np.array(sorted(S), dtype=_np.int64)

        return V, F, E_idx

    # choose length channel for sampling (no GT to correlate with)
    length_ch = int(getattr(opt, "length_channel", 0))

    # also allow the model to provide file paths when mesh objects lack geometry
    paths = getattr(model, "mesh_paths", None)

    # try per-sample, using either a mesh object or a file path
    for i in range(B_out):
        try:
            carrier = None
            if isinstance(meshes, (list, tuple)) and i < len(meshes):
                carrier = meshes[i]
            if carrier is None and isinstance(paths, (list, tuple)) and len(paths) > 0:
                carrier = paths[i] if i < len(paths) else paths[0]

            if carrier is None:
                print(f"[SAMPLE][OBJ] skip {i}: no mesh object or path available.")
                continue

            V, F, E_idx = _extract_geometry_and_edges(carrier)
            if V is None or F is None or E_idx is None:
                print(f"[SAMPLE][OBJ] skip {i}: missing geometry/edges.")
                continue

            xmean_i = x_mean[i].detach().cpu().numpy()  # (C,E)
            E_model = xmean_i.shape[1]
            if E_idx.shape[0] != E_model:
                use_E = min(E_idx.shape[0], E_model)
                if use_E == 0:
                    print(f"[SAMPLE][OBJ] skip {i}: zero edges.")
                    continue
                E_idx   = E_idx[:use_E]
                xmean_i = xmean_i[:, :use_E]
                print(f"[SAMPLE][OBJ] warn: E mismatch (mesh={E_idx.shape[0]}, model={E_model}); using first {use_E}.")

            tgt_lengths = np.maximum(xmean_i[length_ch], 1e-8)
            V_gen = adjust_vertices_to_edges(V, E_idx, tgt_lengths, iterations=10)
            save_obj(V_gen, F, os.path.join(out_dir, f"sample_{i}.obj"))
        except Exception as e:
            print(f"[SAMPLE][OBJ] item {i}: failed to save OBJ: {e}")

    print(f"[SAMPLE] wrote {B_out} samples to {out_dir}")


def seg_accuracy(predicted, ssegs, meshes):
    correct = 0
    ssegs = ssegs.squeeze(-1)
    correct_mat = ssegs.gather(2, predicted.cpu().unsqueeze(dim=2))
    for mesh_id, mesh in enumerate(meshes):
        correct_vec = correct_mat[mesh_id, :mesh.edges_count, 0]
        edge_areas = torch.from_numpy(mesh.get_edge_areas())
        correct += (correct_vec.float() * edge_areas).sum()
    return correct

def print_network(net):
    """Print the total number of parameters in the network
    Parameters:
        network
    """
    print('---------- Network initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')

def get_heatmap_color(value, minimum=0, maximum=1):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


def normalize_np_array(np_array):
    min_value = np.min(np_array)
    max_value = np.max(np_array)
    return (np_array - min_value) / (max_value - min_value)


def calculate_entropy(np_array):
    entropy = 0
    np_array /= np.sum(np_array)
    for a in np_array:
        if a != 0:
            entropy -= a * np.log(a)
    entropy /= np.log(np_array.shape[0])
    return entropy



###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance', num_groups=1):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, affine=True, num_groups=num_groups)
    elif norm_type == 'none':
        norm_layer = NoNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_norm_args(norm_layer, nfeats_list):
    if hasattr(norm_layer, '__name__') and norm_layer.__name__ == 'NoNorm':
        norm_args = [{'fake': True} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'GroupNorm':
        norm_args = [{'num_channels': f} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'BatchNorm':
        norm_args = [{'num_features': f} for f in nfeats_list]
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_layer.func.__name__)
    return norm_args

class NoNorm(nn.Module): #todo with abstractclass and pass
    def __init__(self, fake=True):
        self.fake = fake
        super(NoNorm, self).__init__()
    def forward(self, x):
        return x
    def __call__(self, x):
        return self.forward(x)

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def define_classifier(input_nc, ncf, ninput_edges, nclasses, opt, gpu_ids, arch, init_type, init_gain):
    net = None
    norm_layer = get_norm_layer(norm_type=opt.norm, num_groups=opt.num_groups)

    if arch == 'mconvnet':
        net = MeshConvNet(norm_layer, input_nc, ncf, nclasses, ninput_edges, opt.pool_res, opt.fc_n,
                          opt.resblocks)
    elif arch == 'meshunet':
        down_convs = [input_nc] + ncf
        up_convs   = ncf[::-1] + [nclasses]
        pool_res   = [ninput_edges] + opt.pool_res
        net = MeshEncoderDecoder(pool_res, down_convs, up_convs, blocks=opt.resblocks,
                                 transfer_data=True)
    elif arch == 'mconvregnet':
        extra = int(getattr(opt, 'tab_dim', 0))
        net = MeshRegNet(norm_layer, input_nc, ncf, nclasses, ninput_edges, opt.pool_res, opt.fc_n,
                         opt.resblocks, extra_dim=extra)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_loss(opt):
    if getattr(opt, 'regression_targets', None):
        if getattr(opt, 'regression_loss', 'smoothl1').lower() == 'mse':
            return torch.nn.MSELoss()
        else:
            return torch.nn.SmoothL1Loss(beta=1.0)  # Huber
    if opt.dataset_mode == 'classification':
        if getattr(opt, 'multi_label', False):
            # BCE with logits for multi-label; optional class weighting
            pw = None
            if hasattr(opt, 'pos_weight') and opt.pos_weight:
                pw = torch.tensor(opt.pos_weight, dtype=torch.float32)
                if getattr(opt, 'gpu_ids', None):
                    pw = pw.cuda(opt.gpu_ids[0])
            return torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            return torch.nn.CrossEntropyLoss()
    elif opt.dataset_mode == 'segmentation':
        return torch.nn.CrossEntropyLoss(ignore_index=-1)
    else:
        raise NotImplementedError(f"Unknown dataset_mode {opt.dataset_mode}")

def fill_mesh(mesh2fill, file: str, opt):
    """
    Populate `mesh2fill` either from a cached .npz (preferred) or by rebuilding
    from the raw .obj and saving a fresh cache.

    Expects helper `get_mesh_path(file, num_aug)` and `from_scratch(file, opt)`
    to already exist in your codebase.
    """
    load_path = get_mesh_path(file, opt.num_aug)

    # what a valid cache must contain
    REQUIRED = {
        "vs", "edges", "gemm_edges", "edges_count",
        "ve", "v_mask", "filename", "sides",
        "edge_lengths", "edge_areas", "features",
    }

    # ---------------- helpers ---------------- #

    def to_np(x, dtype=None):
        """Coerce to numpy array; optionally cast dtype (no copy if not needed)."""
        a = np.asarray(x)
        return a.astype(dtype, copy=False) if dtype is not None else a

    def to_1d(x, dtype=None):
        """Coerce list / 1-D array (possibly object/ragged) to a flat 1-D numpy."""
        if x is None:
            return None
        a = np.asarray(x)
        if a.dtype == object:
            a = np.array(list(a), dtype=(dtype or np.float32))
        elif dtype is not None:
            a = a.astype(dtype, copy=False)
        return a

    def to_obj_array(x):
        """Coerce list-of-arrays (ragged) → dtype=object array for safe savez."""
        try:
            a = np.asarray(x)
            if a.dtype == object:
                return a
        except Exception:
            pass
        return np.array(list(x), dtype=object)

    def save_fresh(mesh_src):
        # numeric, dense tensors/arrays
        vs           = to_np(mesh_src.vs,             np.float32)
        edges        = to_np(mesh_src.edges,          np.int32)
        v_mask       = to_1d(getattr(mesh_src, "v_mask", []), np.int32)
        edge_lengths = to_1d(mesh_src.edge_lengths,   np.float32)
        edge_areas   = to_1d(mesh_src.edge_areas,     np.float32)
        features     = to_np(mesh_src.features,       np.float32)

        # sides can be dense or ragged across datasets; store what we have
        try:
            s = np.asarray(mesh_src.sides)
            sides = s
        except Exception:
            sides = to_obj_array(mesh_src.sides)

        # ragged fields
        gemm_edges = to_obj_array(mesh_src.gemm_edges)
        ve         = to_obj_array(getattr(mesh_src, "ve", []))

        # scalars / meta
        edges_count = int(getattr(mesh_src, "edges_count", len(edges)))
        filename    = str(getattr(mesh_src, "filename", os.path.basename(file)))

        save_kwargs = dict(
            vs=vs, edges=edges, gemm_edges=gemm_edges,
            edges_count=edges_count, ve=ve, v_mask=v_mask,
            filename=filename, sides=sides, edge_lengths=edge_lengths,
            edge_areas=edge_areas, features=features
        )

        #print(f"[DEBUG] Saving mesh keys to {load_path}: {list(save_kwargs.keys())}")
        np.savez_compressed(load_path, allow_pickle=True, **save_kwargs)

    # --------------- load / (re)build --------------- #

    need_rebuild = False
    if os.path.exists(load_path):
        mesh_data = np.load(load_path, encoding="latin1", allow_pickle=True)
        loaded_keys = set(mesh_data.keys()) if hasattr(mesh_data, "keys") else set(mesh_data.files)
        #print(f"[DEBUG] Loaded mesh keys from {load_path}: {list(loaded_keys)}")
        if not REQUIRED.issubset(loaded_keys):
            print(f"[WARN] Incomplete cache (missing {sorted(REQUIRED - loaded_keys)}). Rebuilding…")
            try:
                os.remove(load_path)
            except OSError:
                pass
            need_rebuild = True
    else:
        need_rebuild = True

    if need_rebuild:
        # Catch geometry assertions during build (e.g., zero-area faces)
        try:
            src = from_scratch(file, opt)
        except AssertionError as e:
            print(f"[WARN] Skipping mesh due to bad geometry: {file} :: {e}")
            return  # do not save or load; skip this sample
        except Exception as e:
            print(f"[WARN] Skipping mesh due to unexpected error: {file} :: {e}")
            return
        save_fresh(src)
        mesh_data = np.load(load_path, encoding="latin1", allow_pickle=True)
        #print(f"[DEBUG] Reloaded keys: {list(mesh_data.keys())}")

    # --------------- copy into mesh2fill --------------- #
    def getk(k, default=None):
        """Fetch key from npz (NpzFile acts like a dict with .files)."""
        if hasattr(mesh_data, "files"):
            return mesh_data[k] if k in mesh_data.files else default
        return mesh_data.get(k, default)

    # ---------- copy geometry / metadata ----------
    mesh2fill.vs        = to_np(getk("vs"),        np.float32)   # [V,3]
    mesh2fill.edges     = to_np(getk("edges"),     np.int32)     # [E,2]
    mesh2fill.gemm_edges= getk("gemm_edges")                      # ragged (object)
    mesh2fill.ve        = getk("ve")                               # ragged (object)
    mesh2fill.v_mask    = to_1d(getk("v_mask", []), np.int32)      # [V] or []
    mesh2fill.filename  = str(getk("filename", os.path.basename(file)))
    mesh2fill.sides     = getk("sides")                            # dense or ragged
    mesh2fill.features  = to_np(getk("features"), np.float32)      # [C, E_pad] (padded)

    # Prefer a cached edges_count, but trust the geometry if they disagree.
    E_cached            = int(getk("edges_count", len(mesh2fill.edges)))
    E_geom              = int(len(mesh2fill.edges))
    E                   = min(E_cached, E_geom)
    mesh2fill.edges_count = E

    # ---------- per-edge invariants (must be length E) ----------
    # try cache
    edge_lengths = getk("edge_lengths", None)
    edge_areas   = getk("edge_areas",   None)

    mesh2fill.edge_lengths = (to_1d(edge_lengths, np.float32)[:E]
                            if edge_lengths is not None else None)
    mesh2fill.edge_areas   = (to_1d(edge_areas,   np.float32)[:E]
                            if edge_areas   is not None else None)

    # If lengths missing or wrong length, recompute from vs & edges
    if mesh2fill.edge_lengths is None or mesh2fill.edge_lengths.shape[0] != E:
        v = mesh2fill.vs                        # [V,3]
        e = mesh2fill.edges[:E]                 # [E,2]
        le = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).astype(np.float32)
        mesh2fill.edge_lengths = le

    # Sanitize numbers (NaN/Inf → 0)
    if not np.isfinite(mesh2fill.edge_lengths).all():
        mesh2fill.edge_lengths = np.nan_to_num(
            mesh2fill.edge_lengths, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

    # Areas are optional for MeshCNN; zero safely if missing/bad
    if (mesh2fill.edge_areas is None or
        mesh2fill.edge_areas.shape[0] != E or
        not np.isfinite(mesh2fill.edge_areas).all()):
        mesh2fill.edge_areas = np.zeros(E, dtype=np.float32)



def get_mesh_path(file: str, num_aug: int):
    filename, _ = os.path.splitext(file)
    dir_name = os.path.dirname(filename)
    prefix = os.path.basename(filename)
    load_dir = os.path.join(dir_name, 'cache')
    load_file = os.path.join(load_dir, '%s_%03d.npz' % (prefix, np.random.randint(0, num_aug)))
    if not os.path.isdir(load_dir):
        os.makedirs(load_dir, exist_ok=True)
    return load_file

def from_scratch(file, opt):

    class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

    mesh_data = MeshPrep()
    mesh_data.vs = mesh_data.edges = None
    mesh_data.gemm_edges = mesh_data.sides = None
    mesh_data.edges_count = None
    mesh_data.ve = None
    mesh_data.v_mask = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.edge_areas = []
    mesh_data.vs, faces = fill_from_file(mesh_data, file)
    mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)
    faces, face_areas = remove_non_manifolds(mesh_data, faces)
    if opt.num_aug > 1:
        faces = augmentation(mesh_data, opt, faces)
    build_gemm(mesh_data, faces, face_areas)
    if opt.num_aug > 1:
        post_augmentation(mesh_data, opt)
    mesh_data.features = extract_features(mesh_data)
    return mesh_data

def fill_from_file(mesh, file):
    mesh.filename = ntpath.split(file)[1]
    mesh.fullfilename = file
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


def remove_non_manifolds(mesh, faces):
    mesh.ve = [[] for _ in mesh.vs]
    edges_set = set()
    mask = np.ones(len(faces), dtype=bool)
    _, face_areas = compute_face_normals_and_areas(mesh, faces)

    if np.any(face_areas == 0):
        print(f"[WARN] Skipping mesh with zero-area face: {mesh.filename}")
        return None   # or raise a custom exception

    for face_id, face in enumerate(faces):
        if face_areas[face_id] == 0:
            mask[face_id] = False
            continue
        faces_edges = []
        is_manifold = False
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            if cur_edge in edges_set:
                is_manifold = True
                break
            else:
                faces_edges.append(cur_edge)
        if is_manifold:
            mask[face_id] = False
        else:
            for idx, edge in enumerate(faces_edges):
                edges_set.add(edge)
    return faces[mask], face_areas[mask]


def build_gemm(mesh, faces, face_areas):
    """
    gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
    sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an edge is in the gemm_edge entry of the 4 neighboring edges
    for example edge i -> gemm_edges[gemm_edges[i], sides[i]] == [i, i, i, i]
    """
    mesh.ve = [[] for _ in mesh.vs]
    edge_nb = []
    sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                mesh.ve[edge[0]].append(edges_count)
                mesh.ve[edge[1]].append(edges_count)
                mesh.edge_areas.append(0)
                nb_count.append(0)
                edges_count += 1
            mesh.edge_areas[edge2key[edge]] += face_areas[face_id] / 3
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
            sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
    mesh.edges = np.array(edges, dtype=np.int32)
    mesh.gemm_edges = np.array(edge_nb, dtype=np.int64)
    mesh.sides = np.array(sides, dtype=np.int64)
    mesh.edges_count = edges_count
    mesh.edge_areas = np.array(mesh.edge_areas, dtype=np.float32) / np.sum(face_areas) #todo whats the difference between edge_areas and edge_lenghts?


def compute_face_normals_and_areas(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    eps = 1e-12
    face_normals = face_normals / np.clip(face_areas[:, None], eps, None)
    #face_normals /= face_areas[:, np.newaxis]
    assert (not np.any(face_areas[:, np.newaxis] == 0)), 'has zero area face: %s' % mesh.filename
    face_areas *= 0.5
    return face_normals, face_areas


# Data augmentation methods
def augmentation(mesh, opt, faces=None):
    if hasattr(opt, 'scale_verts') and opt.scale_verts:
        scale_verts(mesh)
    if hasattr(opt, 'flip_edges') and opt.flip_edges:
        faces = flip_edges(mesh, opt.flip_edges, faces)
    return faces


def post_augmentation(mesh, opt):
    if hasattr(opt, 'slide_verts') and opt.slide_verts:
        slide_verts(mesh, opt.slide_verts)


def slide_verts(mesh, prct):
    edge_points = get_edge_points(mesh)
    dihedral = dihedral_angle(mesh, edge_points).squeeze() #todo make fixed_division epsilon=0
    thr = np.mean(dihedral) + np.std(dihedral)
    vids = np.random.permutation(len(mesh.ve))
    target = int(prct * len(vids))
    shifted = 0
    for vi in vids:
        if shifted < target:
            edges = mesh.ve[vi]
            if min(dihedral[edges]) > 2.65:
                edge = mesh.edges[np.random.choice(edges)]
                vi_t = edge[1] if vi == edge[0] else edge[0]
                nv = mesh.vs[vi] + np.random.uniform(0.2, 0.5) * (mesh.vs[vi_t] - mesh.vs[vi])
                mesh.vs[vi] = nv
                shifted += 1
        else:
            break
    mesh.shifted = shifted / len(mesh.ve)


def scale_verts(mesh, mean=1, var=0.1):
    for i in range(mesh.vs.shape[1]):
        mesh.vs[:, i] = mesh.vs[:, i] * np.random.normal(mean, var)


def angles_from_faces(mesh, edge_faces, faces):
    normals = [None, None]
    for i in range(2):
        edge_a = mesh.vs[faces[edge_faces[:, i], 2]] - mesh.vs[faces[edge_faces[:, i], 1]]
        edge_b = mesh.vs[faces[edge_faces[:, i], 1]] - mesh.vs[faces[edge_faces[:, i], 0]]
        normals[i] = np.cross(edge_a, edge_b)
        div = fixed_division(np.linalg.norm(normals[i], ord=2, axis=1), epsilon=0)
        normals[i] /= div[:, np.newaxis]
    dot = np.sum(normals[0] * normals[1], axis=1).clip(-1, 1)
    angles = np.pi - np.arccos(dot)
    return angles


def flip_edges(mesh, prct, faces):
    edge_count, edge_faces, edges_dict = get_edge_faces(faces)
    dihedral = angles_from_faces(mesh, edge_faces[:, 2:], faces)
    edges2flip = np.random.permutation(edge_count)
    # print(dihedral.min())
    # print(dihedral.max())
    target = int(prct * edge_count)
    flipped = 0
    for edge_key in edges2flip:
        if flipped == target:
            break
        if dihedral[edge_key] > 2.7:
            edge_info = edge_faces[edge_key]
            if edge_info[3] == -1:
                continue
            new_edge = tuple(sorted(list(set(faces[edge_info[2]]) ^ set(faces[edge_info[3]]))))
            if new_edge in edges_dict:
                continue
            new_faces = np.array(
                [[edge_info[1], new_edge[0], new_edge[1]], [edge_info[0], new_edge[0], new_edge[1]]])
            if check_area(mesh, new_faces):
                del edges_dict[(edge_info[0], edge_info[1])]
                edge_info[:2] = [new_edge[0], new_edge[1]]
                edges_dict[new_edge] = edge_key
                rebuild_face(faces[edge_info[2]], new_faces[0])
                rebuild_face(faces[edge_info[3]], new_faces[1])
                for i, face_id in enumerate([edge_info[2], edge_info[3]]):
                    cur_face = faces[face_id]
                    for j in range(3):
                        cur_edge = tuple(sorted((cur_face[j], cur_face[(j + 1) % 3])))
                        if cur_edge != new_edge:
                            cur_edge_key = edges_dict[cur_edge]
                            for idx, face_nb in enumerate(
                                    [edge_faces[cur_edge_key, 2], edge_faces[cur_edge_key, 3]]):
                                if face_nb == edge_info[2 + (i + 1) % 2]:
                                    edge_faces[cur_edge_key, 2 + idx] = face_id
                flipped += 1
    # print(flipped)
    return faces


def rebuild_face(face, new_face):
    new_point = list(set(new_face) - set(face))[0]
    for i in range(3):
        if face[i] not in new_face:
            face[i] = new_point
            break
    return face

def check_area(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_areas *= 0.5
    return face_areas[0] > 0 and face_areas[1] > 0


def get_edge_faces(faces):
    edge_count = 0
    edge_faces = []
    edge2keys = dict()
    for face_id, face in enumerate(faces):
        for i in range(3):
            cur_edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if cur_edge not in edge2keys:
                edge2keys[cur_edge] = edge_count
                edge_count += 1
                edge_faces.append(np.array([cur_edge[0], cur_edge[1], -1, -1]))
            edge_key = edge2keys[cur_edge]
            if edge_faces[edge_key][2] == -1:
                edge_faces[edge_key][2] = face_id
            else:
                edge_faces[edge_key][3] = face_id
    return edge_count, np.array(edge_faces), edge2keys


def set_edge_lengths(mesh, edge_points=None):
    if edge_points is not None:
        edge_points = get_edge_points(mesh)
    edge_lengths = np.linalg.norm(mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1)
    mesh.edge_lengths = edge_lengths


def extract_features(mesh):
    features = []
    edge_points = get_edge_points(mesh)
    set_edge_lengths(mesh, edge_points)
    with np.errstate(divide='raise'):
        try:
            for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios]:
                feature = extractor(mesh, edge_points)
                features.append(feature)
            return np.concatenate(features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad features')


def dihedral_angle(mesh, edge_points):
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    return angles


def symmetric_opposite_angles(mesh, edge_points):
    """ computes two angles: one for each face shared between the edge
        the angle is in each face opposite the edge
        sort handles order ambiguity
    """
    angles_a = get_opposite_angles(mesh, edge_points, 0)
    angles_b = get_opposite_angles(mesh, edge_points, 3)
    angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0)
    angles = np.sort(angles, axis=0)
    return angles


def symmetric_ratios(mesh, edge_points):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios_a = get_ratios(mesh, edge_points, 0)
    ratios_b = get_ratios(mesh, edge_points, 3)
    ratios = np.concatenate((np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0)
    return np.sort(ratios, axis=0)


def get_edge_points(mesh):
    """ returns: edge_points (#E x 4) tensor, with four vertex ids per edge
        for example: edge_points[edge_id, 0] and edge_points[edge_id, 1] are the two vertices which define edge_id 
        each adjacent face to edge_id has another vertex, which is edge_points[edge_id, 2] or edge_points[edge_id, 3]
    """
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
        # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)
    return edge_points


def get_side_points(mesh, edge_id):
    # if mesh.gemm_edges[edge_id, side] == -1:
    #     return mesh.get_side_points(edge_id, ((side + 2) % 4))
    # else:
    edge_a = mesh.edges[edge_id]

    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]


def get_normals(mesh, edge_points, side):
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
    normals /= div[:, np.newaxis]
    return normals

def get_opposite_angles(mesh, edge_points, side):
    edges_a = mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    edges_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]

    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    return np.arccos(dot)


def get_ratios(mesh, edge_points, side):
    edges_lengths = np.linalg.norm(mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, 1 - side // 2]],
                                   ord=2, axis=1)
    point_o = mesh.vs[edge_points[:, side // 2 + 2]]
    point_a = mesh.vs[edge_points[:, side // 2]]
    point_b = mesh.vs[edge_points[:, 1 - side // 2]]
    line_ab = point_b - point_a
    projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / fixed_division(
        np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
    closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
    return d / edges_lengths

def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div


class Mesh:

    def __init__(self, file=None, opt=None, hold_history=False, export_folder=''):
        self.vs = self.v_mask = self.filename = self.features = self.edge_areas = None
        self.edges = self.gemm_edges = self.sides = None
        self.pool_count = 0
        fill_mesh(self, file, opt)
        self.export_folder = export_folder
        self.history_data = None
        if hold_history:
            self.init_history()
        self.export()
        self.obj_path = file

    def extract_features(self):
        return self.features

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.edges[edge_id]
        v_a = self.vs[edge[0]]
        v_b = self.vs[edge[1]]
        # update pA
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        self.v_mask[edge[1]] = False
        mask = self.edges == edge[1]
        self.ve[edge[0]].extend(self.ve[edge[1]])
        self.edges[mask] = edge[0]

    def remove_vertex(self, v):
        self.v_mask[v] = False

    def remove_edge(self, edge_id):
        """
        Remove `edge_id` from each vertex's incident-edge list `self.ve`.
        Works whether self.ve[v] is a list, tuple, or numpy array.
        Silently does nothing if the id isn't present.
        """
        if not hasattr(self, "ve") or self.ve is None:
            return

        vs = self.edges[edge_id]  # pair [va, vb]
        e = int(edge_id)
        for v in vs:
            lst = self.ve[v]
            # normalize to a Python list
            if isinstance(lst, np.ndarray):
                lst = lst.tolist()
            else:
                lst = list(lst)

            # drop all occurrences (defensive)
            if lst:
                lst = [x for x in lst if x != e]

            self.ve[v] = lst



    def clean(self, edges_mask, groups):
        """
        Remove pooled edges (False in `edges_mask`), remap adjacency, and slice all
        per-edge arrays consistently. Keeps -1 as sentinel in adjacency lists.

        NumPy 2.x safe (we do not pass copy=False to np.asarray on ragged/object
        sources). Also rebuilds self.ve as lists of ints.
        """
        # 1) Normalize inputs / dtypes
        edges_mask = np.asarray(edges_mask, dtype=bool)
        E = int(self.edges.shape[0])                       # real edges rows in self.edges
        assert edges_mask.shape[0] == E, f"mask len {edges_mask.shape[0]} != E {E}"

        # 2) Ensure index-like arrays are integers (allow copy for NumPy 2.x)
        self.gemm_edges   = np.asarray(self.gemm_edges,   dtype=np.int64)   # [E, K] with -1
        self.edges        = np.asarray(self.edges,        dtype=np.int64)   # [E, ...]
        if hasattr(self, "sides"):
            self.sides    = np.asarray(self.sides,        dtype=np.int64)   # [E, ...]
        if hasattr(self, "edge_lengths"):
            self.edge_lengths = np.asarray(self.edge_lengths, dtype=np.float32)
        if hasattr(self, "edge_areas"):
            self.edge_areas   = np.asarray(self.edge_areas,   dtype=np.float32)
        if hasattr(self, "v_mask"):
            self.v_mask       = np.asarray(self.v_mask,       dtype=bool)

        # 3) Build old -> new index map
        new_idx = np.full(E, -1, dtype=np.int64)
        kept    = np.flatnonzero(edges_mask)                               # positions we keep
        new_idx[kept] = np.arange(kept.size, dtype=np.int64)

        # 4) Remap adjacency (only valid >=0 entries), THEN drop rows
        ge   = self.gemm_edges
        valid = ge >= 0
        remapped = np.where(valid, new_idx[ge], -1)                         # -1 stays -1
        self.gemm_edges = remapped[edges_mask]                              # drop pooled rows

        # 5) Slice all per-edge arrays with the mask
        self.edges = self.edges[edges_mask]
        if hasattr(self, "edge_lengths"):
            self.edge_lengths = self.edge_lengths[edges_mask]
        if hasattr(self, "edge_areas"):
            self.edge_areas   = self.edge_areas[edges_mask]
        if hasattr(self, "sides"):
            self.sides        = self.sides[edges_mask]
        if hasattr(self, "v_mask"):   # vertex mask is vertex-wise; leave as-is
            self.v_mask = self.v_mask

        # 6) Update vertex->incident-edges lists (self.ve) using new_idx
        #    Make sure each entry becomes a **Python list of ints**
        if hasattr(self, "ve") and self.ve is not None:
            new_v = []
            for lst in self.ve:
                arr = np.asarray(lst, dtype=np.int64)      # tolerate list / np array / tuple
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                mapped = np.where(arr >= 0, new_idx[arr], -1)   # map; keep -1 for invalid
                mapped = mapped[mapped >= 0]                    # drop invalids
                new_v.append(mapped.tolist())                   # **store as list**
            self.ve = new_v

        # 7) Bookkeeping / history hooks
        self.edges_count = int(kept.size)

        # if your history expects a torch mask, supply it here
        torch_mask = torch.from_numpy(edges_mask.copy())
        if hasattr(self, "_clean_history"):
            self._clean_history(groups, torch_mask)





    def export(self, file=None, vcolor=None):
        if file is None:
            if self.export_folder:
                filename, file_extension = os.path.splitext(self.filename)
                file = '%s/%s_%d%s' % (self.export_folder, filename, self.pool_count, file_extension)
            else:
                return
        faces = []
        vs = self.vs[self.v_mask]
        gemm = np.array(self.gemm_edges)
        new_indices = np.zeros(self.v_mask.shape[0], dtype=np.int32)
        new_indices[self.v_mask] = np.arange(0, np.ma.where(self.v_mask)[0].shape[0])
        for edge_index in range(len(gemm)):
            cycles = self.__get_cycle(gemm, edge_index)
            for cycle in cycles:
                faces.append(self.__cycle_to_face(cycle, new_indices))
        with open(file, 'w+') as f:
            for vi, v in enumerate(vs):
                vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
                f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
            for face_id in range(len(faces) - 1):
                f.write("f %d %d %d\n" % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1))
            f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
            for edge in self.edges:
                f.write("\ne %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1))

    def export_segments(self, segments):
        if not self.export_folder:
            return
        cur_segments = segments
        for i in range(self.pool_count + 1):
            filename, file_extension = os.path.splitext(self.filename)
            file = '%s/%s_%d%s' % (self.export_folder, filename, i, file_extension)
            fh, abs_path = mkstemp()
            edge_key = 0
            with os.fdopen(fh, 'w') as new_file:
                with open(file) as old_file:
                    for line in old_file:
                        if line[0] == 'e':
                            new_file.write('%s %d' % (line.strip(), cur_segments[edge_key]))
                            if edge_key < len(cur_segments):
                                edge_key += 1
                                new_file.write('\n')
                        else:
                            new_file.write(line)
            os.remove(file)
            move(abs_path, file)
            if i < len(self.history_data['edges_mask']):
                cur_segments = segments[:len(self.history_data['edges_mask'][i])]
                cur_segments = cur_segments[self.history_data['edges_mask'][i]]

    def __get_cycle(self, gemm, edge_id):
        cycles = []
        for j in range(2):
            next_side = start_point = j * 2
            next_key = edge_id
            if gemm[edge_id, start_point] == -1:
                continue
            cycles.append([])
            for i in range(3):
                tmp_next_key = gemm[next_key, next_side]
                tmp_next_side = self.sides[next_key, next_side]
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2)
                gemm[next_key, next_side] = -1
                gemm[next_key, next_side + 1 - 2 * (next_side % 2)] = -1
                next_key = tmp_next_key
                next_side = tmp_next_side
                cycles[-1].append(next_key)
        return cycles

    def __cycle_to_face(self, cycle, v_indices):
        face = []
        for i in range(3):
            v = list(set(self.edges[cycle[i]]) & set(self.edges[cycle[(i + 1) % 3]]))[0]
            face.append(v_indices[v])
        return face

    def init_history(self):
        self.history_data = {
                               'groups': [],
                               'gemm_edges': [self.gemm_edges.copy()],
                               'occurrences': [],
                               'old2current': np.arange(self.edges_count, dtype=np.int32),
                               'current2old': np.arange(self.edges_count, dtype=np.int32),
                               'edges_mask': [torch.ones(self.edges_count,dtype=torch.bool)],
                               'edges_count': [self.edges_count],
                              }
        if self.export_folder:
            self.history_data['collapses'] = MeshUnion(self.edges_count)

    def union_groups(self, source, target):
        if self.export_folder and self.history_data:
            self.history_data['collapses'].union(self.history_data['current2old'][source], self.history_data['current2old'][target])
        return

    def remove_group(self, index):
        if self.history_data is not None:
            self.history_data['edges_mask'][-1][self.history_data['current2old'][index]] = 0
            self.history_data['old2current'][self.history_data['current2old'][index]] = -1
            if self.export_folder:
                self.history_data['collapses'].remove_group(self.history_data['current2old'][index])

    def get_groups(self):
        return self.history_data['groups'].pop()

    def get_occurrences(self):
        return self.history_data['occurrences'].pop()
    
    def __clean_history(self, groups, pool_mask):
        if self.history_data is not None:
            mask = self.history_data['old2current'] != -1
            self.history_data['old2current'][mask] = np.arange(self.edges_count, dtype=np.int32)
            self.history_data['current2old'][0: self.edges_count] = np.ma.where(mask)[0]
            if self.export_folder != '':
                self.history_data['edges_mask'].append(self.history_data['edges_mask'][-1].clone())
            self.history_data['occurrences'].append(groups.get_occurrences())
            self.history_data['groups'].append(groups.get_groups(pool_mask))
            self.history_data['gemm_edges'].append(self.gemm_edges.copy())
            self.history_data['edges_count'].append(self.edges_count)
    
    def unroll_gemm(self):
        self.history_data['gemm_edges'].pop()
        self.gemm_edges = self.history_data['gemm_edges'][-1]
        self.history_data['edges_count'].pop()
        self.edges_count = self.history_data['edges_count'][-1]

    def get_edge_areas(self):
        return self.edge_areas

##############################################################################
# Classes For Classification / Segmentation Networks
##############################################################################

class MeshConvNet(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """
    def __init__(self, norm_layer, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
                 nresblocks=3):
        super(MeshConvNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))


        self.gp = torch.nn.AvgPool1d(self.res[-1])
        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def forward(self, x, mesh):

        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)
        x = x.view(-1, self.k[-1])

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MResConv(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1):
        super(MResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        self.conv0 = MeshConv(self.in_channels, self.out_channels, bias=False)
        for i in range(self.skips):
            setattr(self, 'bn{}'.format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConv(self.out_channels, self.out_channels, bias=False))

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skips):
            x = getattr(self, 'bn{}'.format(i + 1))(F.relu(x))
            x = getattr(self, 'conv{}'.format(i + 1))(x, mesh)
        x += x1
        x = F.relu(x)
        return x


class MeshEncoderDecoder(nn.Module):
    """Network for fully-convolutional tasks (segmentation)
    """
    def __init__(self, pools, down_convs, up_convs, blocks=0, transfer_data=True):
        super(MeshEncoderDecoder, self).__init__()
        self.transfer_data = transfer_data
        self.encoder = MeshEncoder(pools, down_convs, blocks=blocks)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoder(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data)

    def forward(self, x, meshes):
        fe, before_pool = self.encoder((x, meshes))
        fe = self.decoder((fe, meshes), before_pool)
        return fe

    def __call__(self, x, meshes):
        return self.forward(x, meshes)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, pool=0, opt=None):
        super().__init__()
        self.bn = []
        self.pool = None

        self.conv1 = MeshConv(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConv(out_channels, out_channels))
        self.conv2 = nn.ModuleList(self.conv2)

        for _ in range(blocks + 1):
            self.bn.append(nn.InstanceNorm2d(out_channels))
        self.bn = nn.ModuleList(self.bn)

        # construct pool ONLY if we are not skipping
        if pool and (opt is None or getattr(opt, "pool_mode", "target") != "skip"):
            self.pool = MeshPool(
                target=pool,
                multi_thread=False,
                mode=getattr(opt, "pool_mode", "target"),
                keep_frac=float(getattr(opt, "pool_keep", 1.0))
            )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        fe, meshes = x
        if fe.dim() == 4 and fe.shape[-1] == 1:
            print(f"[DownConv] squeeze trailing 1 from {tuple(fe.shape)}", flush=True)
            fe = fe.squeeze(-1)

        x1 = self.conv1(fe, meshes)
        x1 = self.bn[0](x1)
        x1 = F.relu(x1)
        if True: print(f"[DownConv] conv1 -> {tuple(x1.shape)}", flush=True)

        x2 = x1
        for idx, conv in enumerate(self.conv2):
            y = conv(x2, meshes)
            y = self.bn[idx + 1](y)
            x2 = F.relu(y)
            if True: print(f"[DownConv] conv2[{idx}] -> {tuple(x2.shape)}", flush=True)

        # features sometimes [B,C,E,1]; squeeze
        if x2.dim() == 4 and x2.shape[-1] == 1:
            x2 = x2.squeeze(-1)

        if self.pool is not None:
            x2 = self.pool(x2, meshes)
            if True: print(f"[DownConv] pool -> {tuple(x2.shape)}", flush=True)

        return x2, meshes
               


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, unroll=0, residual=True,
                 batch_norm=True, transfer_data=True):
        super(UpConv, self).__init__()
        self.residual = residual
        self.bn = []
        self.unroll = None
        self.transfer_data = transfer_data
        self.up_conv = MeshConv(in_channels, out_channels)
        if transfer_data:
            self.conv1 = MeshConv(2 * out_channels, out_channels)
        else:
            self.conv1 = MeshConv(out_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConv(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        if batch_norm:
            for _ in range(blocks + 1):
                self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if unroll:
            self.unroll = MeshUnpool(unroll)

    def __call__(self, x, from_down=None):
        return self.forward(x, from_down)

    def forward(self, x, from_down):
        from_up, meshes = x
        x1 = self.up_conv(from_up, meshes).squeeze(3)
        if self.unroll:
            x1 = self.unroll(x1, meshes)
        if self.transfer_data:
            x1 = torch.cat((x1, from_down), 1)
        x1 = self.conv1(x1, meshes)
        if self.bn:
            x1 = self.bn[0](x1)
        x1 = F.relu(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2
        x2 = x2.squeeze(3)
        return x2

class MeshEncoder(nn.Module):
    def __init__(self, pools, convs, fcs=None, blocks=0, global_pool=None):
        super(MeshEncoder, self).__init__()
        self.fcs = None
        self.convs = []
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool))
        self.global_pool = None
        if fcs is not None:
            self.fcs = []
            self.fcs_bn = []
            last_length = convs[-1]
            if global_pool is not None:
                if global_pool == 'max':
                    self.global_pool = nn.MaxPool1d(pools[-1])
                elif global_pool == 'avg':
                    self.global_pool = nn.AvgPool1d(pools[-1])
                else:
                    assert False, 'global_pool %s is not defined' % global_pool
            else:
                last_length *= pools[-1]
            if fcs[0] == last_length:
                fcs = fcs[1:]
            for length in fcs:
                self.fcs.append(nn.Linear(last_length, length))
                self.fcs_bn.append(nn.InstanceNorm1d(length))
                last_length = length
            self.fcs = nn.ModuleList(self.fcs)
            self.fcs_bn = nn.ModuleList(self.fcs_bn)
        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x):
        fe, meshes = x
        encoder_outs = []
        for conv in self.convs:
            fe, before_pool = conv((fe, meshes))
            encoder_outs.append(before_pool)
        if self.fcs is not None:
            if self.global_pool is not None:
                fe = self.global_pool(fe)
            fe = fe.contiguous().view(fe.size()[0], -1)
            for i in range(len(self.fcs)):
                fe = self.fcs[i](fe)
                if self.fcs_bn:
                    x = fe.unsqueeze(1)
                    fe = self.fcs_bn[i](x).squeeze(1)
                if i < len(self.fcs) - 1:
                    fe = F.relu(fe)
        return fe, encoder_outs

    def __call__(self, x):
        return self.forward(x)


class MeshDecoder(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True):
        super(MeshDecoder, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConv(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                        batch_norm=batch_norm, transfer_data=transfer_data))
        self.final_conv = UpConv(convs[-2], convs[-1], blocks=blocks, unroll=False,
                                 batch_norm=batch_norm, transfer_data=False)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def forward(self, x, encoder_outs=None):
        fe, meshes = x
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            fe = up_conv((fe, meshes), before_pool)
        fe = self.final_conv((fe, meshes))
        return fe

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)

def reset_params(model): # todo replace with my init
    for i, m in enumerate(model.modules()):
        weight_init(m)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class ClassifierModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None

        #
        self.nclasses = opt.nclasses

        # load/define networks
        self.net = define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        self.criterion = define_loss(opt).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def _safe_net_forward(self, x, mesh, x_tab=None):
        try:
            return self.net(x, mesh, x_tab)   # works for regressor nets
        except TypeError:
            return self.net(x, mesh)          # classifier nets ignore tabular arg

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data['edge_features']).float()
        if getattr(self.opt, 'multi_label', False):
            labels = torch.from_numpy(data['label']).float()
        else:
            labels = torch.tensor(data['label']).long()

        # set inputs
        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device)
        self.mesh = data['mesh']
        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label = torch.from_numpy(data['soft_label'])

    def forward(self):
        return self._safe_net_forward(self.edge_features, self.mesh)

    def backward(self, out):
        self.loss = self.criterion(out, self.labels)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number_correct, total_count (so you can compute accuracy)
        """
        with torch.no_grad():
            out = self.forward()
            if getattr(self.opt, 'multi_label', False):
                probs = torch.sigmoid(out)
                th = float(getattr(self.opt, 'ml_threshold', 0.5))
                pred = (probs > th).float()
                correct = (pred.eq(self.labels)).sum().item()
                total = self.labels.numel()
                return correct, total
            else:
                pred_class = out.data.max(1)[1]
                label_class = self.labels
                self.export_segmentation(pred_class.cpu())
                correct = self.get_accuracy(pred_class, label_class)
                return correct, len(label_class)

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation """
        if self.opt.dataset_mode == 'classification':
            correct = pred.eq(labels).sum()
        elif self.opt.dataset_mode == 'segmentation':
            correct = seg_accuracy(pred, self.soft_label, self.mesh)
        return correct

    def export_segmentation(self, pred_seg):
        if self.opt.dataset_mode == 'segmentation':
            for meshi, mesh in enumerate(self.mesh):
                mesh.export_segments(pred_seg[meshi, :])

#######################################################################
# Classes for Regression
#######################################################################

class MeshRegNet(nn.Module):
    """
    Mesh trunk + optional tabular features -> regression (nout dims).
    extra_dim = number of tabular features concatenated to the pooled mesh descriptor.
    """
    def __init__(self, norm_layer, nf0, conv_res, nout, input_res, pool_res, fc_n, nresblocks=3, extra_dim=0):
        super(MeshRegNet, self).__init__()
        self.F = F

        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, f'conv{i}', MResConv(ki, self.k[i + 1], nresblocks))
            setattr(self, f'norm{i}', norm_layer(**norm_args[i]))
            setattr(self, f'pool{i}', MeshPool(self.res[i + 1]))

        self.gp = torch.nn.AvgPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1] + int(extra_dim), fc_n)
        self.fc2 = nn.Linear(fc_n, nout)

    def forward(self, x, mesh, x_tab=None):
        for i in range(len(self.k) - 1):
            x = getattr(self, f'conv{i}')(x, mesh)
            x = self.F.relu(getattr(self, f'norm{i}')(x))
            x = getattr(self, f'pool{i}')(x, mesh)

        x = self.gp(x)
        x = x.view(-1, self.k[-1])

        if x_tab is not None:
            x = torch.cat([x, x_tab], dim=1)

        x = self.F.relu(self.fc1(x))
        return self.fc2(x)

class RegressorModel(nn.Module):
    """Train a mesh (and optional tabular) regressor for continuous targets."""
    def __init__(self, opt):
        super(RegressorModel, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train

        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)

        # output dim = number of regression targets
        self.nout = int(getattr(opt, 'nclasses', 1))

        # build network
        self.net = define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, self.nout, opt,
                                     self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        self.criterion = define_loss(opt).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

        self.edge_features = None
        self.labels = None
        self.tab_feats = None
        self.mesh = None

    
    def _safe_net_forward(self, x, mesh, x_tab=None):
        try:
            return self.net(x, mesh, x_tab)   # works for regressor nets
        except TypeError:
            return self.net(x, mesh)          # classifier nets ignore tabular arg

    def set_input(self, data):
        x = torch.from_numpy(data['edge_features']).float()
        y = torch.from_numpy(data['label']).float()  # continuous targets
        self.edge_features = x.to(self.device).requires_grad_(self.is_train)
        self.labels = y.to(self.device)
        self.mesh = data['mesh']
        if 'tab_feats' in data:
            self.tab_feats = torch.from_numpy(data['tab_feats']).float().to(self.device)
        else:
            self.tab_feats = None

    def forward(self):
        return self._safe_net_forward(self.edge_features, self.mesh, self.tab_feats)

    def backward(self, out):
        self.loss = self.criterion(out, self.labels)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

    # ---------- I/O helpers (reuse from ClassifierModel) ----------
    def load_network(self, which_epoch):
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_network(self, which_epoch):
        save_filename = '%s_net.pth' % which_epoch
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # ---------- evaluation ----------
    def test(self):
        """
        Returns: (sum_abs_error, count_values)
        so you can compute MAE = sum_abs_error / count_values

        Also sets attributes:
          self.last_mae_per_target, self.last_rmse_per_target (lists)
        """
        with torch.no_grad():
            out = self.forward()               # [B, nout]
            err = out - self.labels            # [B, nout]
            mae_per_t = torch.mean(torch.abs(err), dim=0)   # per-target MAE
            rmse_per_t = torch.sqrt(torch.mean(err**2, dim=0))

            self.last_mae_per_target = mae_per_t.detach().cpu().tolist()
            self.last_rmse_per_target = rmse_per_t.detach().cpu().tolist()

            sum_abs = torch.sum(torch.abs(err)).item()
            total   = err.numel()
            return sum_abs, total

#def create_model(opt):
#    model = ClassifierModel(opt)
#    return model

# train_mesh_cnn.py  (or wherever create_model lives)

def create_model(opt):
    """
    Factory for model wrappers that match the training loop API.
    Chooses between:
      - ClassifierModel (mconvnet / meshunet)
      - RegressorModel  (mconvregnet or when regression targets are requested)
      - MeshVAEmodel    (meshvae / vae)
    """
    arch = getattr(opt, "arch", "mconvnet").lower().strip()
    dataset_mode = getattr(opt, "dataset_mode", "classification").lower().strip()

    # --- Mesh VAE wins if explicitly requested ---
    if arch in ("meshvae", "vae"):
        from mesh_vae_model import MeshVAEModel
        return MeshVAEModel(opt)

    # --- Regressor wins if explicitly requested OR regression targets are set ---
    targets = getattr(opt, "regression_targets", None)
    has_reg_targets = (
        targets is not None
        and str(targets).strip().lower() not in ("", "none", "[]")
    )
    if dataset_mode == "regression" or has_reg_targets or arch in ("mconvregnet", "regressor", "reg"):
        return RegressorModel(opt)

    # --- Classifier (default) ---
    if arch in ("mconvnet", "meshunet", "classifier", "cls"):
        return ClassifierModel(opt)

    raise ValueError(
        f"Unknown --arch '{opt.arch}'. Use one of: mconvnet, meshunet, mconvregnet, meshvae"
    )


### Dataloading

class BaseDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.mean = 0
        self.std = 1
        self.ninput_channels = None

        # dataset location & phase
        self.root = os.path.abspath(getattr(opt, 'dataroot', '.'))
        self.phase = getattr(opt, 'phase', 'train')
        self.split_ratio = float(getattr(opt, 'split_ratio', 0.8))
        self.split_seed  = int(getattr(opt, 'split_seed', 0))

        # BaseDataset.__init__
        self.mesh_paths = self._discover_meshes()
        cap = getattr(opt, "max_dataset_size", float("inf"))
        if isinstance(cap, (int,)) and cap > 0:
            self.mesh_paths = self.mesh_paths[:cap]
        elif isinstance(cap, float) and math.isfinite(cap) and cap > 0:
            self.mesh_paths = self.mesh_paths[:int(cap)]
        # else: cap is inf/None/<=0 -> no limit
        self.size = len(self.mesh_paths)

        super(BaseDataset, self).__init__()

    # BaseDataset.__len__
    def __len__(self):
        cap = getattr(self.opt, "max_dataset_size", None)
        if isinstance(cap, (int,)) and cap > 0:
            return min(self.size, cap)
        if isinstance(cap, float) and math.isfinite(cap) and cap > 0:
            return min(self.size, int(cap))
        return self.size


    # ---------- helpers ----------
    def _discover_meshes(self):
        """Return list of .obj paths for current phase (SHREC-style or flat folder)."""
        train_dir = os.path.join(self.root, 'train')
        test_dir  = os.path.join(self.root, 'test')

        # Case 1: SHREC-like structure
        if os.path.isdir(train_dir) or os.path.isdir(test_dir):
            base = train_dir if self.phase == 'train' else test_dir
            paths = sorted(glob(os.path.join(base, '**', '*.obj'), recursive=True))
            if not paths:
                raise FileNotFoundError(f'No .obj files found under {base}')
            return paths

        # Case 2: flat folder -> deterministic split
        all_paths = sorted(glob(os.path.join(self.root, '*.obj')))
        if not all_paths:
            raise FileNotFoundError(f'No .obj files found in {self.root}')
        rng = np.random.RandomState(self.split_seed)
        idx = np.arange(len(all_paths)); rng.shuffle(idx)
        k = int(len(all_paths) * self.split_ratio)
        sel = idx[:k] if self.phase == 'train' else idx[k:]
        return [all_paths[i] for i in sel]

    def _writable_stats_path(self, root: str) -> str:
        """
        Prefer <root>/cache/mean_std_cache.p.
        If <root> is not writable, fall back to ~/.cache/meshcnn/<slug>/mean_std_cache.p
        """
        # try <root>/cache
        cache_dir = os.path.join(root, "cache")
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            test_path = os.path.join(cache_dir, ".touch")
            with open(test_path, "wb") as f:
                f.write(b"")
            os.remove(test_path)
            return os.path.join(cache_dir, "mean_std_cache.p")
        except Exception:
            pass

        # fallback to ~/.cache
        slug = os.path.basename(os.path.abspath(root)) or "dataset"
        home_dir = os.path.join(os.path.expanduser("~"), ".cache", "meshcnn", slug)
        Path(home_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(home_dir, "mean_std_cache.p")

    def get_mean_std(self):
        """Computes Mean and Standard Deviation from Training Data"""
        mean_std_cache = self._writable_stats_path(self.root)

        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')
            num_aug = self.opt.num_aug
            self.opt.num_aug = 1

            mean = None   # will init on first valid sample
            std  = None
            count = 0

            for idx in range(len(self)):  # respects --max_dataset_size via __len__
                data_i = self[idx]
                if data_i is None:
                    continue
                feats = data_i.get('edge_features')
                if feats is None:
                    continue

                # feats expected shape: (C, E)
                # initialize accumulators with correct length/dtype
                if mean is None:
                    C = feats.shape[0]
                    mean = np.zeros(C, dtype=np.float64)
                    std  = np.zeros(C, dtype=np.float64)

                m = feats.mean(axis=1).astype(np.float64, copy=False)
                s = feats.std(axis=1).astype(np.float64, copy=False)

                mean += m
                std  += s
                count += 1

                if count % 500 == 0:
                    print(f'{count} of {self.size}')

            if count == 0:
                raise RuntimeError(f'No valid samples for mean/std. dataroot="{self.root}"')

            mean /= count
            std  /= count

            transform_dict = {
                'mean': mean[:, np.newaxis],   # (C,1) for broadcasting
                'std':  std[:,  np.newaxis],
                'ninput_channels': int(mean.shape[0]),
            }
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved:', mean_std_cache)
            self.opt.num_aug = num_aug

        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
        print('loaded mean / std from cache')
        self.mean = transform_dict['mean']
        self.std = transform_dict['std']
        self.ninput_channels = transform_dict['ninput_channels']


# keep your collate_fn that drops None batches
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    keys = batch[0].keys()
    meta = {k: np.array([d[k] for d in batch]) for k in keys}
    return meta

class ClassificationData(BaseDataset):

    def __init__(self, opt):
        # base init
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.gpu_ids else torch.device('cpu'))
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)

        # ---------- CSV (multi-label) path ----------
        labels_csv = getattr(opt, 'labels_csv', '')
        csv_key_col = getattr(opt, 'csv_key_col', 'stregkode')
        label_cols  = getattr(opt, 'label_cols', None)
        self.csv_labels = False

        if labels_csv:
            if not os.path.isfile(labels_csv):
                raise FileNotFoundError(f'labels_csv not found: {labels_csv}')

            # read CSV
            with open(labels_csv, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                headers = reader.fieldnames

            # choose label columns
            if not label_cols:
                preferred = [c for c in ['ho_carotis', 've_carotis', 'ho_fem', 've_fem'] if c in headers]
                if preferred:
                    label_cols = preferred
                else:
                    def is_binary(col):
                        vals = [r[col] for r in rows]
                        seen = 0
                        for v in vals:
                            if v == '' or v is None:
                                continue
                            try:
                                x = float(v)
                            except Exception:
                                return False
                            if x not in (0.0, 1.0):
                                return False
                            seen += 1
                        return seen > 0
                    label_cols = [c for c in headers if is_binary(c)]
                    if not label_cols:
                        raise RuntimeError("Could not infer label columns from CSV. Please set opt.label_cols.")

            if csv_key_col not in headers:
                for candidate in ['stregkode', 'stregkode_2', 'filename', 'obj', 'obj_path']:
                    if candidate in headers:
                        csv_key_col = candidate
                        break
                else:
                    raise RuntimeError(f"Key column '{csv_key_col}' was not found in CSV.")

            # find all .obj files and build map from normalized key -> path
            all_objs = sorted(glob(os.path.join(self.dir, '**', '*.obj'), recursive=True))
            file_map = {self._normalize_key(p): p for p in all_objs}

            # join CSV rows to OBJ files (with validation + logging)
            paths = []
            missing = 0

            problems = []                 # list of (key, column, raw_value, message)
            ptype = Counter()             # counts per problem type
            max_print = int(getattr(opt, "print_csv_issues_max", 50))  # limit how many lines to print

            def coerce01(cell, col, key):
                """Return 0/1 float; record problems for empty, non-numeric, or non-binary values."""
                s = str(cell).strip()

                if s == "" or s.lower() in ("nan", "na", "none", "null"):
                    problems.append((key, col, cell, "blank/NA -> set 0"))
                    ptype["blank/NA"] += 1
                    return 0.0

                try:
                    x = float(s)
                except Exception:
                    problems.append((key, col, cell, "non-numeric -> set 0"))
                    ptype["non-numeric"] += 1
                    return 0.0

                # not exactly 0/1? note it; we still threshold to {0,1}
                if x < 0 or x > 1:
                    problems.append((key, col, x, "out of [0,1] -> thresholded to {0,1}"))
                    ptype["out-of-range"] += 1
                elif x not in (0.0, 1.0):
                    problems.append((key, col, x, "not exactly 0/1 -> thresholded to {0,1}"))
                    ptype["not-binary"] += 1

                return 1.0 if x >= 0.5 else 0.0

            for r in rows:
                key = self._normalize_key(r[csv_key_col])
                if key in file_map:
                    vals = [coerce01(r.get(c, ""), c, key) for c in label_cols]
                    vec = np.asarray(vals, dtype=np.float32)
                    paths.append((file_map[key], vec))
                else:
                    missing += 1

            # Print diagnostics (without flooding logs)
            if problems:
                n = len(problems)
                print(f"[CSV label issues] {n} problematic cells detected "
                    f"(showing first {min(max_print, n)}):")
                for key, col, raw, msg in problems[:max_print]:
                    print(f"  - sample={key}  column={col}  value={raw!r}  => {msg}")
                print(f"[CSV label issues] counts by type: {dict(ptype)}")
            if missing:
                print(f"[CSV join] {missing} CSV rows had no matching .obj (after normalization).")

            # split if needed
            has_phase_tokens = any((os.sep + 'train' + os.sep) in p for p, _ in paths) or \
                            any((os.sep + 'test'  + os.sep) in p for p, _ in paths)
            if has_phase_tokens:
                tgt = 'train' if getattr(opt, 'phase', 'train') == 'train' else 'test'
                self.paths = [(p, y) for p, y in paths if (os.sep + tgt + os.sep) in p]
            else:
                split_ratio = float(getattr(opt, 'split_ratio', 0.8))
                split_seed  = int(getattr(opt, 'split_seed', 0))
                rng = np.random.RandomState(split_seed)
                idx = np.arange(len(paths)); rng.shuffle(idx)
                k = int(len(paths) * split_ratio)
                sel = idx[:k] if getattr(opt, 'phase', 'train') == 'train' else idx[k:]
                self.paths = [paths[i] for i in sel]

            self.size = len(self.paths)
            self.classes = label_cols
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.nclasses = len(self.classes)
            self.csv_labels = True

            # pos_weight for BCE (optional but helpful)
            if getattr(opt, 'phase', 'train') == 'train' and self.size > 0:
                Y = np.stack([y for _, y in self.paths], axis=0)  # [N, C]
                pos = Y.sum(axis=0)
                neg = Y.shape[0] - pos
                pos_weight = (neg / np.maximum(pos, 1)).astype(np.float32)
                opt.pos_weight = pos_weight.tolist()

            opt.multi_label = True  # tell the model/loss to use BCE

            print(f"[ClassificationData] multi-label CSV loaded: classes={self.nclasses} "
                f"names={self.classes} samples={self.size} (missing CSV->OBJ matches: {missing}) "
                f"dataroot=\"{self.dir}\"")
        else:
            # ---------- original behavior (no CSV) ----------
            self.classes, self.class_to_idx = self.find_classes(self.dir)

            def _is_obj_path(p):  # robust extension check
                return os.path.splitext(p)[1].lower() == '.obj'

            def collect_from_class_dirs():
                items = []
                for cls in self.classes:
                    clsdir = os.path.join(self.dir, cls)
                    for r, _, fnames in os.walk(clsdir):
                        for fn in sorted(fnames):
                            if _is_obj_path(fn):
                                items.append((os.path.join(r, fn), self.class_to_idx[cls]))
                return items

            used_fallback = False
            if self.classes:
                all_items = collect_from_class_dirs()
                if len(all_items) > 0:
                    has_phase_tokens = any((os.sep + 'train' + os.sep) in p for p, _ in all_items) or \
                                    any((os.sep + 'test'  + os.sep) in p for p, _ in all_items)
                    if has_phase_tokens:
                        tgt = 'train' if getattr(opt, 'phase', 'train') == 'train' else 'test'
                        self.paths = [(p, y) for p, y in all_items if (os.sep + tgt + os.sep) in p]
                    else:
                        split_ratio = float(getattr(opt, 'split_ratio', 0.8))
                        split_seed = int(getattr(opt, 'split_seed', 0))
                        rng = np.random.RandomState(split_seed)
                        idx = np.arange(len(all_items))
                        rng.shuffle(idx)
                        k = int(len(all_items) * split_ratio)
                        sel = idx[:k] if getattr(opt, 'phase', 'train') == 'train' else idx[k:]
                        self.paths = [all_items[i] for i in sel]
                    self.size = len(self.paths)
                    self.nclasses = len(self.classes)
                else:
                    used_fallback = True
            else:
                used_fallback = True

            if used_fallback:
                all_objs = sorted([p for p in glob(os.path.join(self.dir, '*.obj')) if _is_obj_path(p)])
                if not all_objs:
                    raise FileNotFoundError(f'No .obj files found in "{self.dir}"')
                split_ratio = float(getattr(opt, 'split_ratio', 0.8))
                split_seed = int(getattr(opt, 'split_seed', 0))
                rng = np.random.RandomState(split_seed)
                idx = np.arange(len(all_objs)); rng.shuffle(idx)
                k = int(len(all_objs) * split_ratio)
                sel = idx[:k] if getattr(opt, 'phase', 'train') == 'train' else idx[k:]
                self.classes = ['-']
                self.class_to_idx = {'-': 0}
                self.nclasses = 1
                self.paths = [(all_objs[i], 0) for i in sel]
                self.size = len(self.paths)

            print(f"[ClassificationData] phase={getattr(opt,'phase','?')} "
                f"classes={self.nclasses} samples={self.size} dataroot=\"{self.dir}\"")

        # cap dataset if requested
        cap = getattr(opt, "max_dataset_size", float("inf"))
        if math.isfinite(cap) and cap > 0:
            self.paths = self.paths[:int(cap)]
            self.size = len(self.paths)

        # compute/load normalization
        self.get_mean_std()

        # wire into options for model creation
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path, label = self.paths[index]
        try:
            # load mesh
            mesh = Mesh(
                file=path,
                opt=self.opt,
                hold_history=True,
                export_folder=getattr(self.opt, "export_folder", getattr(self.opt, "exp_dir", ""))
            )

            # features (expected shape (C, E))
            edge_features = mesh.extract_features()
            if edge_features is None:
                raise ValueError("extract_features() returned None")
            edge_features = np.asarray(edge_features)

            # keep original before pad/trunc, then pad to ninput_edges
            edge_features_raw = edge_features
            E_raw = int(edge_features_raw.shape[1])

            # NOTE: use same selector logic as pad() to guarantee identical indices
            trunc_strategy = getattr(self.opt, "truncate_strategy", "stride")
            E_tgt = int(self.opt.ninput_edges)
            idx_map = _select_edge_indices(E_raw, E_tgt, strategy=trunc_strategy, seed=0)

            edge_features = pad(
                edge_features_raw,
                self.opt.ninput_edges,
                pad_value=0.0,
                truncate_strategy=trunc_strategy,
            )

            # ---- DEBUG DUMP: save geometry (first N only) ----
            if getattr(self.opt, "dump_padded_obj", False) and index < int(getattr(self.opt, "dump_padded_max", 8)):
                import os
                from pathlib import Path

                E_used = int(edge_features.shape[1])  # == ninput_edges
                stem = Path(path).stem
                if E_raw > E_used:
                    print(f"[PAD][{stem}] TRUNCATE {E_raw} -> {E_used}")
                elif E_raw < E_used:
                    print(f"[PAD][{stem}] PAD {E_raw} -> {E_used}")
                else:
                    print(f"[PAD][{stem}] NO-OP {E_raw} edges")

                # output dir under experiment folder
                exp_dir = getattr(
                    self.opt, "exp_dir",
                    os.path.join(getattr(self.opt, "checkpoints_dir", "./checkpoints"),
                                getattr(self.opt, "name", "exp"))
                )
                out_root = os.path.join(exp_dir, "padded_debug")
                os.makedirs(out_root, exist_ok=True)
                out_path = os.path.join(out_root, f"{stem}__Eraw{E_raw}__Eused{E_used}.obj")

                # robust geometry fetch including EDGES
                try:
                    # prefer the same function we use later elsewhere; if not available, fall back to faces-only
                    try:
                        V, F, E_idx = extract_geometry_and_edges(path)
                    except NameError:
                        # minimal fallback when helper not imported
                        V, F = _get_mesh_geometry(mesh, fallback_path=path)
                        if F is None or (isinstance(F, np.ndarray) and F.size == 0):
                            # no faces -> no edges; write vertices only
                            E_idx = np.empty((0, 2), dtype=np.int64)
                        else:
                            # derive edges from faces
                            es = set()
                            for a, b, c in np.asarray(F, dtype=np.int64):
                                for u, v in ((a, b), (b, c), (c, a)):
                                    if u > v: u, v = v, u
                                    es.add((int(u), int(v)))
                            E_idx = np.array(sorted(es), dtype=np.int64)

                    if V is None or V.size == 0:
                        print(f"[PAD][dump] no vertices; skipping OBJ for {path}")
                    else:
                        # map selection indices to actual edges
                        if E_idx is None or E_idx.size == 0:
                            # No explicit edges -> just save vertices/faces (best effort)
                            kept_F = np.asarray(F, dtype=np.int64) if F is not None else np.empty((0, 3), np.int64)
                            save_obj(V, kept_F if kept_F.size else None, out_path)
                            print(f"[PAD][dump] wrote (no edges) {out_path}")
                        else:
                            # Select/duplicate edges exactly like pad()
                            sel = idx_map  # length E_used
                            sel = np.clip(sel, 0, E_idx.shape[0]-1) if E_idx.shape[0] > 0 else sel
                            kept_edges = E_idx[sel] if sel.size > 0 else np.empty((0, 2), dtype=np.int64)

                            # Try to keep only faces fully composed of kept edges
                            kept_edge_set = {(min(int(u), int(v)), max(int(u), int(v))) for (u, v) in kept_edges.tolist()} if kept_edges.size else set()
                            kept_faces = []
                            if F is not None and np.asarray(F).size and kept_edge_set:
                                F_arr = np.asarray(F, dtype=np.int64).reshape(-1, 3)
                                for (a, b, c) in F_arr:
                                    tri_edges = {(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))}
                                    if tri_edges.issubset(kept_edge_set):
                                        kept_faces.append([a, b, c])
                            kept_faces = np.asarray(kept_faces, dtype=np.int64)

                            if kept_faces.size > 0:
                                # Save reduced surface (faces that survive)
                                save_obj(V, kept_faces, out_path)
                                print(f"[PAD][dump] wrote faces-only (subset) {out_path}  (faces={kept_faces.shape[0]}, edges_kept={kept_edges.shape[0]})")
                            else:
                                # Save wireframe of kept edges to visualize truncation
                                save_obj(V, None, out_path, edges=kept_edges)
                                print(f"[PAD][dump] wrote wireframe {out_path}  (edges_kept={kept_edges.shape[0]})")
                except Exception as e:
                    print(f"[PAD][dump] failed to save OBJ for {path}: {e}")
            # --------------------------------------------------

            # return sample (keep obj_path so downstream can reload geometry if needed)
            meta = {
                "mesh": mesh,
                "obj_path": path,
                "label": label if isinstance(label, np.ndarray) else int(label),
                "edge_features": (edge_features - self.mean) / self.std,
            }
            return meta

        except Exception as e:
            print(f"[WARN] Skipping sample {path}: {e}")
            return None



    def __len__(self):
        return self.size

    # original helpers (unchanged)
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_mesh_file(fname) and (root.count(phase) == 1):
                        path = os.path.join(root, fname)
                        meshes.append((path, class_to_idx[target]))
        return meshes

    @staticmethod
    def _normalize_key(s):
        base = os.path.basename(str(s))
        base = os.path.splitext(base)[0]
        base = base.replace("'", "").replace('"', '')
        base = re.sub(r'[^A-Za-z0-9]', '', base)
        return base.upper()


class RegressionData(BaseDataset):
    """
    Dataset for mesh-based regression with optional tabular CSV features.
    Expects:
      opt.labels_csv: path to CSV
      opt.csv_key_col: column whose value identifies the mesh (matches OBJ filename ignoring punctuation)
      opt.regression_targets: list[str], e.g. ["measure_Body height (cm)", "measure_Hip height (cm)"]
      opt.regression_tab_cols: optional list[str] of extra CSV columns to fuse (numeric or 'M'/'F')
    """
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.gpu_ids else torch.device('cpu'))
        self.dir = os.path.join(opt.dataroot)

        # --- read CSV ---
        csv_path = getattr(opt, 'labels_csv', '')
        if not csv_path:
            raise RuntimeError("RegressionData requires opt.labels_csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'labels_csv not found: {csv_path}')

        key_col = getattr(opt, 'csv_key_col', 'stregkode')
        y_cols  = getattr(opt, 'regression_targets', None)
        if not y_cols or not isinstance(y_cols, (list, tuple)) or len(y_cols) < 1:
            raise RuntimeError("Set opt.regression_targets = [list of target columns]")

        tab_cols = getattr(opt, 'regression_tab_cols', []) or []

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            headers = reader.fieldnames

        if key_col not in headers:
            raise RuntimeError(f"Key column '{key_col}' not found in CSV headers {headers[:10]}...")

        for c in y_cols:
            if c not in headers:
                raise RuntimeError(f"Target column '{c}' not found in CSV")

        # ---- map normalized base name -> OBJ full path ----
        all_objs = sorted(glob(os.path.join(self.dir, '**', '*.obj'), recursive=True))
        file_map = {self._normalize_key(p): p for p in all_objs}

        # ---- join CSV rows to OBJ files ----
        self.samples = []      # list of (obj_path, y_vec, tab_vec or None)
        missing = 0

        def num_or_nan(v):
            s = str(v).strip()
            if s == '' or s.lower() in ('nan', 'na', 'none', 'null'):
                return np.nan
            try:
                return float(s)
            except Exception:
                # common category: 'M'/'F'
                u = s.upper()
                if u in ('M', 'F'):
                    return 1.0 if u == 'M' else 0.0
                return np.nan

        for r in rows:
            key = self._normalize_key(r[key_col])
            if key not in file_map:
                missing += 1
                continue

            # targets (vector)
            y = [num_or_nan(r.get(c, '')) for c in y_cols]
            # drop row if any target is NaN
            if any(np.isnan(y)):
                continue
            y = np.asarray(y, dtype=np.float32)

            # optional tabular features
            x_tab = None
            if tab_cols:
                vals = [num_or_nan(r.get(c, '')) for c in tab_cols]
                # replace NaNs with column means later; keep as float now
                x_tab = np.asarray(vals, dtype=np.float32)

            self.samples.append((file_map[key], y, x_tab))

        if not self.samples:
            raise RuntimeError("No usable rows after joining CSV to OBJ and filtering NaN targets.")

        # ---- train/test split (if no explicit /train/ or /test/ in paths) ----
        has_phase_tokens = any((os.sep + 'train' + os.sep) in p for p, _, __ in self.samples) or \
                           any((os.sep + 'test'  + os.sep) in p for p, _, __ in self.samples)

        if has_phase_tokens:
            tgt = 'train' if getattr(opt, 'phase', 'train') == 'train' else 'test'
            self.samples = [t for t in self.samples if (os.sep + tgt + os.sep) in t[0]]
        else:
            split_ratio = float(getattr(opt, 'split_ratio', 0.8))
            split_seed  = int(getattr(opt, 'split_seed', 0))
            rng = np.random.RandomState(split_seed)
            idx = np.arange(len(self.samples)); rng.shuffle(idx)
            k = int(len(self.samples) * split_ratio)
            sel = idx[:k] if getattr(opt, 'phase', 'train') == 'train' else idx[k:]
            self.samples = [self.samples[i] for i in sel]

        # ---- tabular normalization (fit on train; reuse on val/test) ----
        self.tab_cols = tab_cols
        self.tab_dim  = len(tab_cols)
        self.tab_mean = None
        self.tab_std  = None
        if self.tab_dim > 0:
            if getattr(opt, 'phase', 'train') == 'train' or not hasattr(opt, 'tab_mean'):
                # fit on this split
                X = [s[2] for s in self.samples if s[2] is not None]
                X = np.stack(X, axis=0)
                # NaN -> column mean (computed ignoring NaN)
                col_mean = np.nanmean(X, axis=0)
                inds = np.where(np.isnan(X))
                X[inds] = np.take(col_mean, inds[1])
                col_std = np.nanstd(X, axis=0)
                col_std[col_std < 1e-8] = 1.0
                opt.tab_mean = col_mean.tolist()
                opt.tab_std  = col_std.tolist()

            self.tab_mean = np.asarray(getattr(opt, 'tab_mean'), dtype=np.float32)
            self.tab_std  = np.asarray(getattr(opt, 'tab_std'), dtype=np.float32)

        # ---- finalize ----
        self.size = len(self.samples)
        self.nout = len(y_cols)
        self.target_names = y_cols

        # wire into options for model creation
        opt.nclasses = self.nout          # reuse the same field the nets already use for output dim
        opt.input_nc = self.ninput_channels
        opt.tab_dim  = self.tab_dim       # for the regressor head
        opt.target_names = self.target_names

        print(f"[RegressionData] phase={getattr(opt,'phase','?')} samples={self.size} "
              f"targets={self.target_names} tab_dim={self.tab_dim} "
              f"(CSV rows w/o OBJ: {missing}) dataroot=\"{self.dir}\"")

        # compute/load edge feature normalization like ClassificationData does
        self.get_mean_std()

    def __getitem__(self, index):
        path, y, x_tab = self.samples[index]
        try:
            mesh = Mesh(file=path, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder)
            edge_features = mesh.extract_features()
            if edge_features is None:
                raise ValueError("extract_features() returned None")
            edge_features = pad(edge_features, self.opt.ninput_edges)


            # normalize tabular features if present
            tab_feats = None
            if self.tab_dim > 0 and x_tab is not None:
                vals = x_tab.copy()
                # fill NaN with mean, then standardize
                nan_mask = np.isnan(vals)
                if np.any(nan_mask):
                    vals[nan_mask] = self.tab_mean[nan_mask]
                tab_feats = (vals - self.tab_mean) / self.tab_std
                tab_feats = tab_feats.astype(np.float32)

            meta = {
                'mesh': mesh,
                'label': y.astype(np.float32),  # continuous targets
                'edge_features': (edge_features - self.mean) / self.std,
            }
            if tab_feats is not None:
                meta['tab_feats'] = tab_feats
            return meta
        except Exception as e:
            print(f"[WARN] Skipping sample {path}: {e}")
            return None

    def __len__(self):
        return self.size

    @staticmethod
    def _normalize_key(s):
        """Turn '/.../!RIGDET001925!.obj' or 'RIGDET001925' -> 'RIGDET001925'."""
        base = os.path.basename(str(s))
        base = os.path.splitext(base)[0]
        base = base.replace("'", "").replace('"', '')
        base = re.sub(r'[^A-Za-z0-9]', '', base)
        return base.upper()






class SegmentationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = self.make_dataset(self.dir)
        self.seg_paths = self.get_seg_files(self.paths, os.path.join(self.root, 'seg'), seg_ext='.eseg')
        self.sseg_paths = self.get_seg_files(self.paths, os.path.join(self.root, 'sseg'), seg_ext='.seseg')
        self.classes, self.offset = self.get_n_segs(os.path.join(self.root, 'classes.txt'), self.seg_paths)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()
        # # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index]
        mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
        meta = {}
        meta['mesh'] = mesh
        label = read_seg(self.seg_paths[index]) - self.offset
        label = pad(label, self.opt.ninput_edges, val=-1, dim=0)
        meta['label'] = label
        soft_label = read_sseg(self.sseg_paths[index])
        meta['soft_label'] = pad(soft_label, self.opt.ninput_edges, val=-1, dim=0)
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size

    @staticmethod
    def get_seg_files(paths, seg_dir, seg_ext='.seg'):
        segs = []
        for path in paths:
            segfile = os.path.join(seg_dir, os.path.splitext(os.path.basename(path))[0] + seg_ext)
            assert(os.path.isfile(segfile))
            segs.append(segfile)
        return segs

    @staticmethod
    def get_n_segs(classes_file, seg_files):
        if not os.path.isfile(classes_file):
            all_segs = np.array([], dtype='float64')
            for seg in seg_files:
                all_segs = np.concatenate((all_segs, read_seg(seg)))
            segnames = np.unique(all_segs)
            np.savetxt(classes_file, segnames, fmt='%d')
        classes = np.loadtxt(classes_file)
        offset = classes[0]
        classes = classes - offset
        return classes, offset

    @staticmethod
    def make_dataset(path):
        meshes = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if is_mesh_file(fname):
                    path = os.path.join(root, fname)
                    meshes.append(path)

        return meshes


def read_seg(seg):
    seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
    return seg_labels


def read_sseg(sseg_file):
    sseg_labels = read_seg(sseg_file)
    sseg_labels = np.array(sseg_labels > 0, dtype=np.int32)
    return sseg_labels


def CreateDataset(opt):
    if opt.dataset_mode == 'classification':
        dataset = ClassificationData(opt)
    elif opt.dataset_mode == 'regression':
        dataset = RegressionData(opt)
    elif opt.dataset_mode == 'segmentation':
        dataset = SegmentationData(opt)
    else:
        raise NotImplementedError(f"dataset_mode {opt.dataset_mode} not recognized")
    return dataset



class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = TorchDataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            collate_fn=collate_fn,
        )

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, batch in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield batch


### Training


try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None

class Writer:
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_name = os.path.join(self.save_dir, 'loss_log.txt')
        self.testacc_log = os.path.join(self.save_dir, 'testacc_log.txt')
        self.start_logs()
        self.nexamples = 0
        self.ncorrect = 0
        #
        if opt.is_train and not opt.no_vis and SummaryWriter is not None:
            self.display = SummaryWriter(comment=opt.name)
        else:
            self.display = None

    def start_logs(self):
        """ creates test / train log files """
        if self.opt.is_train:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
        else:
            with open(self.testacc_log, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Testing Acc (%s) ================\n' % now)

    def print_current_losses(self, epoch, i, losses, t, t_data):
        """ prints train loss to terminal / file """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) loss: %.3f ' \
                  % (epoch, i, t, t_data, losses.item())
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_loss(self, loss, epoch, i, n):
        iters = i + (epoch - 1) * n
        if self.display:
            self.display.add_scalar('data/train_loss', loss, iters)

    def plot_model_wts(self, model, epoch):
        if self.opt.is_train and self.display:
            for name, param in model.net.named_parameters():
                self.display.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    def print_acc(self, epoch, acc):
        """ prints test accuracy to terminal / file """
        message = 'epoch: {}, TEST ACC: [{:.5} %]\n' \
            .format(epoch, acc * 100)
        print(message)
        with open(self.testacc_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_acc(self, acc, epoch):
        if self.display:
            self.display.add_scalar('data/test_acc', acc, epoch)

    def reset_counter(self):
        """
        counts # of correct examples
        """
        self.ncorrect = 0
        self.nexamples = 0

    def update_counter(self, ncorrect, nexamples):
        self.ncorrect += ncorrect
        self.nexamples += nexamples

    @property
    def acc(self):
        return float(self.ncorrect) / self.nexamples

    def close(self):
        if self.display is not None:
            self.display.close()


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data params
        self.parser.add_argument('--dataroot', required=True, help='path to meshes (should have subfolders train, test)')
        self.parser.add_argument('--dataset_mode',
            choices=('classification', 'segmentation', 'regression'),
            default='classification',
            help='classification | segmentation | regression')
        self.parser.add_argument('--ninput_edges', type=int, default=750,
            help='# of input edges (will include dummy edges)')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
            help='Maximum number of samples per epoch')

        # network params
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self.parser.add_argument('--resblocks', type=int, default=0, help='# of res blocks')
        self.parser.add_argument('--fc_n', type=int, default=100, help='# between fc and nclasses')
        self.parser.add_argument('--ncf', nargs='+', default=[16, 32, 32], type=int, help='conv filters')
        self.parser.add_argument('--pool_res', nargs='+', default=[1140, 780, 580], type=int, help='pooling res')
        self.parser.add_argument('--norm', type=str, default='batch', help='instance/batch/group norm')
        self.parser.add_argument('--num_groups', type=int, default=16, help='# groups for groupnorm')
        self.parser.add_argument('--init_type', type=str, default='xavier', help='normal|xavier|kaiming|orthogonal')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for init')

        # general params
        self.parser.add_argument('--num_threads', default=3, type=int, help='# threads for loading data')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids; use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='debug', help='experiment name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true', help='sequential batches')
        self.parser.add_argument('--seed', type=int, help='if specified, uses seed')

        # 🔧 debug / perf toggles
        self.parser.add_argument(
            '--debug_kernels', action='store_true', default=False,
            help='Enable autograd anomaly detection + deterministic cuDNN (slow; for debugging NaNs/cudnn errors).'
        )
        self.parser.add_argument(
            '--use_amp', action='store_true', default=False,
            help='Enable mixed precision training (autocast + GradScaler).'
        )



        # visualization / exports
        self.parser.add_argument('--export_folder', type=str, default='', help='exports under checkpoints/<name>/...')

        # run/test + recon saving
        self.parser.add_argument('--only_test', action='store_true', help='run one eval on test split and exit')
        self.parser.add_argument('--save_recon_freq', type=int, default=1, help='save reconstructions every N epochs (0=never)')
        self.parser.add_argument('--recon_k', type=int, default=2, help='how many recon pairs to save')

        # NEW: padded debug flags
        self.parser.add_argument('--dump_padded_obj', action='store_true',
                                 help='Dump padded/truncated mesh geometry to OBJ (debugging)')
        self.parser.add_argument('--dump_padded_max', type=int, default=8,
                                 help='Max number of debug OBJ files to save')

        # architecture / VAE / sampling (unchanged)
        self.parser.add_argument('--regression_targets', type=str, default=None,
            help='Comma-separated columns for regression (e.g. "age,bmi").')
        self.parser.add_argument('--arch', type=str, default='mconvnet',
            choices=['mconvnet', 'meshunet', 'mconvregnet', 'meshvae'],
            help='mconvnet/meshunet=classifier, mconvregnet=regressor, meshvae=VAE')
        self.parser.add_argument('--pool_mode', type=str, default='target',
            choices=['target','frac','skip'], help='pooling strategy')
        self.parser.add_argument('--pool_keep', type=float, default=1.0,
            help='for pool_mode=frac: keep this fraction each down-step')

        self.parser.add_argument('--latent_prior', type=str, choices=('gaussian','boson'),
            default='gaussian', help='VAE latent prior')
        self.parser.add_argument('--boson_params', type=str, default='',
            help='Boson sampler JSON (path or inline)')
        self.parser.add_argument('--boson_n_photons', type=int, default=8)
        self.parser.add_argument('--boson_n_modes', type=int, default=16)
        self.parser.add_argument('--boson_loss', type=float, default=0.0)
        self.parser.add_argument('--boson_seed', type=int, default=0)
        self.parser.add_argument('--latent_dim', type=int, default=64)
        self.parser.add_argument('--beta', type=float, default=1.0)
        self.parser.add_argument('--sample_k', type=int, default=8)
        self.parser.add_argument('--sample_seed', type=int, default=0)
        self.parser.add_argument('--sample_out', type=str, default='')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train

        # GPUs
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for sid in str_ids:
            i = int(sid)
            if i >= 0:
                self.opt.gpu_ids.append(i)
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
            self.opt.device = torch.device(f"cuda:{self.opt.gpu_ids[0]}")
        else:
            # No GPU or user requested CPU (-1), so use CPU
            self.opt.device = torch.device("cpu")

        # ---- ALWAYS define experiment dir + export folder ----
        # (this fixes the 'Namespace has no attribute exp_dir' crashes)
        def mkdir(p):
            if not os.path.isdir(p):
                os.makedirs(p, exist_ok=True)

        self.opt.exp_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        mkdir(self.opt.exp_dir)

        if self.opt.export_folder:
            self.opt.export_folder = os.path.join(self.opt.exp_dir, self.opt.export_folder)
            mkdir(self.opt.export_folder)

        # boson params parsing (unchanged)
        def _read_json_or_path(s: str):
            if not s or str(s).strip() == '':
                return None
            try:
                if os.path.isfile(s):
                    with open(s, 'r') as f:
                        return json.load(f)
                try:
                    return json.loads(s)
                except Exception:
                    return ast.literal_eval(s)
            except Exception:
                print('[WARN] failed to parse --boson_params, ignoring.')
                return None

        _boson_cfg = _read_json_or_path(self.opt.boson_params)
        if getattr(self.opt, 'latent_prior', 'gaussian').lower() == 'boson':
            if not isinstance(_boson_cfg, dict) or len(_boson_cfg) == 0:
                raise ValueError("You set --latent_prior boson but --boson_params is empty.")
        setattr(self.opt, 'boson_cfg', _boson_cfg)

        # Seeds
        if self.opt.seed is not None:
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        # Print & save options only in train
        if self.is_train:
            args = vars(self.opt)
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print(f'{k}: {v}')
            print('-------------- End ----------------')

            file_name = os.path.join(self.opt.exp_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write(f'{k}: {v}\n')
                opt_file.write('-------------- End ----------------\n')

        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=250, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--run_test_freq', type=int, default=1, help='frequency of running test in training script')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str,
            choices=['train', 'test', 'val', 'sample'], default='train',
            help='train | test | val | sample (draw from prior)')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=500, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # data augmentation stuff
        self.parser.add_argument('--num_aug', type=int, default=10, help='# of augmentation files')
        self.parser.add_argument('--scale_verts', action='store_true', help='non-uniformly scale the mesh e.g., in x, y or z')
        self.parser.add_argument('--slide_verts', type=float, default=0, help='percent vertices which will be shifted along the mesh surface')
        self.parser.add_argument('--flip_edges', type=float, default=0, help='percent of edges to randomly flip')
        # tensorboard visualization
        self.parser.add_argument('--no_vis', action='store_true', help='will not use tensorboard')
        self.parser.add_argument('--verbose_plot', action='store_true', help='plots network weights, etc.')
        self.is_train = True

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc') #todo delete.
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--num_aug', type=int, default=1, help='# of augmentation files')
        self.is_train = False

def coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def run_test(epoch, model, writer):
    """
    Test loop that works for classifier, regressor, and VAE.
    Robust to missing/invalid split params and empty splits.
    """

    # --- build a test copy of the options (never mutates train opts) ---
    test_opt = copy.deepcopy(opt)
    test_opt.phase = 'test'

    # --- coalesce helper (kept) ---
    def coalesce(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    # ---------- SAFETY: ensure required fields are valid numbers ----------
    # split_ratio: prefer existing value, else opt value, else default 0.2
    sr = coalesce(getattr(test_opt, 'split_ratio', None),
                  getattr(opt,      'split_ratio', None),
                  0.2)
    try:
        sr = float(sr)
    except Exception:
        print(f"[TEST] invalid split_ratio '{sr}', forcing to 0.2", flush=True)
        sr = 0.2
    # clamp into (0, 1) to avoid empty split
    if not (0.0 < sr < 1.0):
        print(f"[TEST] split_ratio {sr} not in (0,1); forcing to 0.2", flush=True)
        sr = 0.2
    test_opt.split_ratio = sr

    # split seed (keep your pattern)
    seed = coalesce(getattr(test_opt, 'split_seed', None),
                    getattr(opt,      'split_seed', None), 0)
    try:
        seed = int(seed)
    except Exception:
        seed = 0
    test_opt.split_seed = seed

    # cap (max dataset size) – keep whatever you already accept
    test_opt.max_dataset_size = coalesce(getattr(test_opt, 'max_dataset_size', None),
                                         getattr(opt,      'max_dataset_size', None))

    # --- safe diagnostics (kept) ---
    print(f"[TEST] phase={test_opt.phase} "
          f"split={test_opt.split_ratio} "
          f"seed={test_opt.split_seed} "
          f"cap={test_opt.max_dataset_size}", flush=True)

    # ---------- build dataset; retry once if empty ----------
    def _build_loader(_opt):
        ds = DataLoader(_opt)
        print(f"[TEST] #meshes in loader: {len(ds)}", flush=True)
        return ds

    try:
        dataset = _build_loader(test_opt)
    except Exception as e:
        # if a framework-level sampler NumSamples error happens here, try fallback split
        print(f"[TEST] loader error on split={test_opt.split_ratio}: {e}", flush=True)
        test_opt.split_ratio = 0.2
        print("[TEST] retrying with split_ratio=0.2 ...", flush=True)
        dataset = _build_loader(test_opt)

    # If it still ends up empty, skip test for this epoch
    if len(dataset) == 0:
        print("[TEST] dataset is empty after fallback; skipping test this epoch.", flush=True)
        return 0.0

    # ---------- switch to eval safely (works whether model has .eval() or not) ----------
    had_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        if hasattr(model, "eval"):
            model.eval()
    except Exception:
        pass
    try:
        if hasattr(model, "net") and hasattr(model.net, "eval"):
            model.net.eval()
    except Exception:
        pass

    writer.reset_counter()

    # --- aggregation buckets (kept) ---
    vae_seen     = 0
    vae_sum_elbo = 0.0
    vae_sum_logpx= 0.0
    vae_sum_kl   = 0.0
    vae_sum_loss = 0.0
    is_vae       = None
    #first_x, first_x_mean = None, None
    # near the start of run_test()
    first_x, first_x_mean, first_meshes = None, None, None

    # regression detection from options (kept)
    reg_mode = bool(getattr(test_opt, 'regression_targets', None)) or \
               getattr(test_opt, 'dataset_mode', '').lower() == 'regression'

    # regression aggregators (kept)
    reg_sum_abs = 0.0      # overall |err| sum across all targets
    reg_total   = 0        # total number of scalar values
    reg_sum_abs_t = None   # tensor [T]
    reg_sum_sq_t  = None   # tensor [T]
    reg_seen      = 0      # number of examples (rows)

    # ---------------- evaluation loop ----------------
    for data in dataset:
        if data is None:
            continue

        model.set_input(data)
        out = model.test()   # VAE -> dict; classifier/regressor -> tuple

        # --- grab example visuals once (VAE path) ---
        if first_x is None and isinstance(out, dict):
            k = getattr(getattr(model, "opt", None), "sample_k", 8)
            x      = out.get("x", None)
            x_mean = out.get("x_mean", None)
            if x is not None:      first_x      = x.detach()[:k].cpu()
            if x_mean is not None: first_x_mean = x_mean.detach()[:k].cpu()

            # >>>>>>> FIXED: robustly collect carriers (paths or mesh objects) <<<<<<<
            if first_meshes is None and isinstance(data, dict):
                import numpy as _np
                if "obj_path" in data:
                    p = data["obj_path"]
                    if isinstance(p, _np.ndarray):
                        first_meshes = p[:k].tolist()
                    elif isinstance(p, (list, tuple)):
                        first_meshes = list(p)[:k]
                    else:
                        first_meshes = [p]
                elif "mesh" in data:
                    m = data["mesh"]
                    if isinstance(m, _np.ndarray):
                        first_meshes = m[:k].tolist()
                    elif isinstance(m, (list, tuple)):
                        first_meshes = list(m)[:k]
                    else:
                        first_meshes = [m]

            if first_meshes is None:
                m = getattr(model, "mesh", None)
                if m is not None:
                    try:
                        import numpy as _np
                        if isinstance(m, _np.ndarray):
                            first_meshes = m[:k].tolist()
                        elif isinstance(m, (list, tuple)):
                            first_meshes = list(m)[:k]
                        else:
                            first_meshes = [m]
                    except Exception:
                        first_meshes = [m]


        # infer pathway once (kept)
        if is_vae is None:
            is_vae = isinstance(out, dict) or getattr(model, "task", "").lower() == "vae"

        if not is_vae and reg_mode:
            # --- Regression path (kept) ---
            # recompute error robustly even if model.test() returned something else
            with torch.no_grad():
                pred = model.forward()           # [B, T]
                err  = pred - model.labels       # [B, T]
                abs_err = torch.abs(err)         # [B, T]

                if reg_sum_abs_t is None:
                    reg_sum_abs_t = torch.zeros(abs_err.shape[1], device=abs_err.device)
                    reg_sum_sq_t  = torch.zeros(abs_err.shape[1], device=abs_err.device)

                reg_sum_abs_t += abs_err.sum(dim=0)
                reg_sum_sq_t  += (err ** 2).sum(dim=0)
                reg_seen      += err.shape[0]

            # overall aggregates (sum over all dims)
            reg_sum_abs += float(abs_err.sum().item())
            reg_total   += int(err.numel())

        elif not is_vae:
            # --- Classification path (unchanged) ---
            ncorrect, nexamples = out
            writer.update_counter(ncorrect, nexamples)

        else:
            # --- VAE path (kept) ---
            if not isinstance(out, dict):
                raise RuntimeError("MeshVAEmodel.test() must return a dict for diagnostics")
            diag = out.get("diagnostics", {})
            b = int(out.get("nexamples", 0))
            if b <= 0:
                # fall back to batch size if not provided
                if isinstance(data, dict):
                    for v in data.values():
                        if torch.is_tensor(v):
                            b = v.shape[0]
                            break
                if b <= 0:
                    b = 1

            def _sum_or_scalar(x):
                if x is None: return 0.0
                if torch.is_tensor(x): return x.detach().float().sum().item()
                try: return float(x)
                except Exception: return 0.0

            vae_sum_elbo += _sum_or_scalar(diag.get("elbo"))
            vae_sum_logpx+= _sum_or_scalar(diag.get("log_px"))
            vae_sum_kl   += _sum_or_scalar(diag.get("kl"))
            vae_sum_loss += _sum_or_scalar(out.get("loss"))
            vae_seen     += b

    # ---------- restore train state (no-op if not present) ----------
    try:
        if hasattr(model, "train"):
            model.train()
    except Exception:
        pass
    try:
        if hasattr(model, "net") and hasattr(model.net, "train"):
            model.net.train()
    except Exception:
        pass
    torch.set_grad_enabled(had_grad)

    # ---------- reporting ----------
    if is_vae:
        if vae_seen > 0:
            avg_elbo  = vae_sum_elbo  / vae_seen
            avg_logpx = vae_sum_logpx / vae_seen
            avg_kl    = vae_sum_kl    / vae_seen
            avg_loss  = vae_sum_loss  / vae_seen
        else:
            avg_elbo = avg_logpx = avg_kl = avg_loss = float('nan')

        print(f"[TEST/VAE] loss:{avg_loss:.4f}  elbo:{avg_elbo:.4f}  "
              f"log_px:{avg_logpx:.4f}  kl:{avg_kl:.4f}  (n={vae_seen})", flush=True)

        # ---- CSV row (one per epoch) ----
        try:
            lr = float(getattr(model.optimizer.param_groups[0], "lr", 0.0)) if hasattr(model, "optimizer") else 0.0
        except Exception:
            lr = 0.0

        _append_metrics_csv(opt, {
            "epoch": int(epoch),
            "lr": lr,
            "loss": float(avg_loss),
            "elbo": float(avg_elbo),
            "log_px": float(avg_logpx),
            "kl": float(avg_kl),
            "n": int(vae_seen),
        })

        # ---- optional recon saving ----
        freq = int(getattr(opt, "save_recon_freq", 0))
        k    = int(getattr(opt, "recon_k", 2))

        if freq > 0 and int(epoch) % freq == 0 and first_x is not None and first_x_mean is not None:
            out_dir = os.path.join(_exp_dir(opt), "recon", f"epoch_{int(epoch):03d}")
            # helpful print so you know we’re firing the save path:
            print(f"[RECON][TRIGGER] epoch={int(epoch)}  k={k}  out_dir={out_dir}  "
                f"meshes={0 if first_meshes is None else len(first_meshes)}")
            _save_reconstructions(
                opt, epoch,
                first_x, first_x_mean,
                meshes=first_meshes,   # <-- pass meshes or paths
                k=k
            )

        # Keep return type similar to classification but meaningful for VAE
        return {"loss": avg_loss, "elbo": avg_elbo, "log_px": avg_logpx, "kl": avg_kl}

    if reg_mode:
        mae = reg_sum_abs / max(1, reg_total)

        # per-target metrics if we saw any rows
        if reg_seen > 0:
            per_mae  = (reg_sum_abs_t / reg_seen).detach().cpu().tolist()
            per_rmse = (reg_sum_sq_t.sqrt() / reg_seen).detach().cpu().tolist()
            names = getattr(test_opt, 'target_names', [])
            s_mae  = ", ".join(f"{n}:{v:.2f}" for n, v in zip(names, per_mae))
            s_rmse = ", ".join(f"{n}:{v:.2f}" for n, v in zip(names, per_rmse))
            print(f"[TEST/REG] epoch {epoch:03d}  MAE={mae:.2f}  | per-target MAE [{s_mae}]  | RMSE [{s_rmse}]", flush=True)
        else:
            print(f"[TEST/REG] epoch {epoch:03d}  MAE={mae:.2f}", flush=True)
        return mae

    # classification (default)
    writer.print_acc(epoch, writer.acc)
    return writer.acc



if __name__ == '__main__':
    opt = TrainOptions().parse()

    # ---- DEBUG NUMERICS / CUDNN GUARDS ----
    if getattr(opt, 'debug_kernels', False):
        torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("[dbg] anomaly detection ON, cuDNN deterministic ON, benchmark OFF", flush=True)
    else:
        # sensible fast defaults (flip benchmark on if your input sizes are fairly stable)
        torch.autograd.set_detect_anomaly(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print("[dbg] anomaly detection OFF, cuDNN deterministic OFF, benchmark ON", flush=True)

    # ---- offline sampling mode (runs only when --phase sample) ----
    if opt.phase.lower() == 'sample':
        run_sample_prior(opt)      # function you added earlier
        sys.exit(0)

    # --- tell the dataset about your CSV + labels ---
    opt.labels_csv  = "/data/projects/3D_scans/combined_cohort_measures.csv"
    opt.csv_key_col = "stregkode"
    opt.label_cols  = ["ho_carotis", "ve_carotis", "ho_fem", "ve_fem"]

    # optional (used in test()):
    opt.ml_threshold = 0.5

    # Which continuous targets to regress:
    opt.regression_targets = ["measure_Body height (cm)", "measure_Hip height (cm)"]

    # Optional: extra tabular features from the CSV to fuse with the mesh embedding
    # (strings OK; we'll coerce numerics; 'M'/'F' -> 1/0)
    opt.regression_tab_cols = ["sex", "birth_year", "waist"]  # <- adjust / leave [] for mesh-only

    # Optional: pick the regression loss
    opt.regression_loss = "smoothl1"   # "mse" or "smoothl1" (Huber). Huber is robust to outliers.

    # Original
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)

    # printouts for debugging
    #print(f"[DEBUG] arch={opt.arch} dataset_mode={getattr(opt,'dataset_mode',None)} "
    #    f"regression_targets={getattr(opt,'regression_targets',None)!r}")

    model = create_model(opt)
    model = model.to(opt.device)

    # --- GPU sanity ---
    print("[gpu] model device:", next(model.parameters()).device, flush=True)

    writer = Writer(opt)
    total_steps = 0

    if getattr(opt, 'only_test', False):
        acc = run_test(epoch=0, model=model, writer=writer)
        print(f"[TEST] final acc = {acc:.4f}")
        sys.exit(0)


    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            if data is None:
                continue  # whole batch filtered out
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            # after model.set_input(data)
            ef = getattr(model, "edge_features", None)
            if isinstance(ef, torch.Tensor) and not torch.isfinite(ef).all():
                bad = (~torch.isfinite(ef)).sum().item()
                print(f"[dbg] skipping batch: non-finite edge_features ({bad} bad values)", flush=True)
                continue

            # --- GPU sanity (print once or occasionally) ---
            if i == 0:  # or: if i % 50 == 0
                try:
                    print("[gpu] edge_features device:", model.edge_features.device, flush=True)
                except Exception:
                    pass

            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest')
            model.save_network(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        model.update_learning_rate()

        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            acc = run_test(epoch, model, writer)   # ✅
            writer.plot_acc(acc, epoch)
        
        try:
            _plot_metrics_from_csv(opt)
        except Exception as e:
            print(f"[WARN] failed to plot metrics at end of run: {e}")

    writer.close()


