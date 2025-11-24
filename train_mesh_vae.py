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
import trimesh
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mesh_geom_pipeline import (
    list_obj_files,
    simplify_mesh,
    mesh_stats_o3d,
    meshes_to_padded_tensors,
    tensors_to_meshes,
    clean_mesh,
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
        parser.add_argument("--phase", type=str, default="train")

        parser.add_argument("--arch", type=str, default="meshvae")
        parser.add_argument("--pool_mode", type=str, default="skip")

        # Model sizing
        parser.add_argument("--input_nc", type=int, default=3)  # xyz features
        parser.add_argument("--ncf", nargs="+", type=int, default=[64, 128, 256])
        parser.add_argument("--pool_res", nargs="+", type=int, default=[1500, 750, 375])
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

        self.parser = parser

    def parse(self):
        opt = self.parser.parse_args()
        opt.gpu_ids = [int(i) for i in opt.gpu_ids.split(",")]

        # Parse boson params JSON
        if opt.boson_params:
            opt.boson_cfg = json.loads(opt.boson_params)
        else:
            opt.boson_cfg = None

        return opt


class TrainOptions(BaseOptions):
    pass


class TestOptions(BaseOptions):
    pass


# =========================================
# Dataset + collate
# =========================================

class MeshVAEDataset(Dataset):
    """
    Loads OBJ files, simplifies them, converts to padded tensors.
    """

    def __init__(self, folder, opt):
        super().__init__()
        self.opt = opt
        self.files = list_obj_files(Path(folder))
        self.files = self.files[: opt.max_dataset_size]

        print(f"[DATA] Found {len(self.files)} OBJ meshes.")

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

        # 1) Clean and simplify
        mesh = clean_mesh(mesh)
        simp = simplify_mesh(mesh, target_tris=None, ratio=0.02)

        # 2) Convert Open3D → Trimesh for model use
        V = np.asarray(simp.vertices, dtype=np.float64)
        F = np.asarray(simp.triangles, dtype=np.int64)
        tri = trimesh.Trimesh(vertices=V, faces=F, process=False)

        # 3) Convert to padded tensors (for saving / reconstruction)
        batch = meshes_to_padded_tensors([simp])

        data = {
            "path": str(path),
            "original": mesh,                # Open3D
            "simplified": simp,              # Open3D
            "verts": batch["verts"][0],
            "faces": batch["faces"][0],
            "verts_mask": batch["verts_mask"][0],
            "faces_mask": batch["faces_mask"][0],
            "mesh": [tri],                   # 🟢 list of Trimesh (what model expects)
            "edge_features": torch.from_numpy(V.T).float().unsqueeze(0),  # (1, 3, V)
        }

        return data


def mesh_collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        v0 = vals[0]
        if isinstance(v0, torch.Tensor):
            # avoid adding an extra dimension for edge_features
            if k == "edge_features" and v0.ndim == 2:
                # (C, E) → (B, C, E)
                out[k] = torch.stack(vals, dim=0)
            elif k == "edge_features" and v0.ndim == 3:
                # already (1, C, E), just concatenate
                out[k] = torch.cat(vals, dim=0)
            else:
                out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


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

# def redetail_to_original_edges(mesh_recon, mesh_original):
#     """
#     Increase triangle count until reconstructed mesh roughly
#     matches the original edge count.
#     """
#     m = mesh_recon
#     target_edges = mesh_stats_o3d(mesh_original)["num_edges"]
#     print(f"[REDETAIL] Target edges = {target_edges}")

#     prev_edges = -1
#     for i in range(8):
#         cur_edges = mesh_stats_o3d(m)["num_edges"]
#         print(f"[REDETAIL] step {i}: edges={cur_edges}")
#         if cur_edges >= target_edges or cur_edges == prev_edges:
#             break
#         prev_edges = cur_edges

#         m = m.subdivide_midpoint(number_of_iterations=1)
#         m = clean_mesh(m)

#     return m
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

    dataset = MeshVAEDataset(opt.dataroot, opt)
    train_data, val_data = split_dataset(dataset, opt.split_ratio, opt.split_seed)

    train_loader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_threads,
        collate_fn=mesh_collate_fn,
    )

    model = MeshVAEModel(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        print(f"\n--- Epoch {epoch} ---")

        for i, data in enumerate(train_loader):
            total_iters += 1

            model.set_input(data)
            model.optimize_parameters()

            if i % 10 == 0:
                print(f"[epoch {epoch}] iter {i} loss = {model.loss.item():.4f}")

        model.update_learning_rate()

        # Save model every epoch
        model.save_network(f"epoch_{epoch}")

        # Evaluate on val
        run_test(opt, model, val_data)

    print("\nTraining complete.")


# =========================================
# Testing / reconstruction
# =========================================

# def run_test(opt, model, val_subset):
#     print("\n========== TESTING ==========\n")

#     loader = DataLoader(
#         val_subset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=mesh_collate_fn,
#     )

#     save_root = Path(opt.checkpoints_dir) / opt.name / "reconstructions"
#     save_root.mkdir(parents=True, exist_ok=True)

#     for i, data in enumerate(loader):
#         model.set_input(data)
#         out = model.test()

#         # B=1 here
#         orig_mesh = data["original"][0]
#         simp_mesh = data["simplified"][0]

#         # x_mean: (B, C, E)
#         x_mean = out["x_mean"]          # (1, C, E)

#         # Build batch for tensors_to_meshes
#         batch = {
#             "verts": x_mean.permute(0, 2, 1),   # (1, E, C)
#             "verts_mask": data["verts_mask"],   # (1, V)
#             "faces": data["faces"],             # (1, F, 3)
#             "faces_mask": data["faces_mask"],   # (1, F)
#         }

#         recon_mesh = tensors_to_meshes(batch)[0]

#         # Redetail to original resolution
#         recon_detail = redetail_to_original_edges(recon_mesh, orig_mesh)

#         import open3d as o3d

#         out_path = save_root / f"recon_{i}.obj"
#         out_path2 = save_root / f"recon_{i}_redetail.obj"
#         o3d.io.write_triangle_mesh(str(out_path), recon_mesh)
#         o3d.io.write_triangle_mesh(str(out_path2), recon_detail)

#         print(f"[TEST] Saved {out_path}")
#         print(f"[TEST] Saved {out_path2}")

# def run_test(opt, model, val_subset):
#     print("\n========== TESTING ==========\n")

#     loader = DataLoader(
#         val_subset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=mesh_collate_fn,   # keep your custom collate
#     )

#     save_root = Path(opt.checkpoints_dir) / opt.name / "reconstructions"
#     save_root.mkdir(parents=True, exist_ok=True)

#     import open3d as o3d
#     # Silence info spam; warning about normals will still show once
#     o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

#     for i, data in enumerate(loader):
#         print(f"\n[TEST] ==== sample {i} ====", flush=True)

#         # 1) Forward pass
#         print("[TEST] setting input...", flush=True)
#         model.set_input(data)
#         print("[TEST] running model.test()...", flush=True)
#         out = model.test()
#         print("[TEST] forward pass done.", flush=True)

#         # 2) original + simplified meshes from dataset
#         orig_mesh = data["original"][0]      # Open3D TriangleMesh
#         simp_mesh = data["simplified"][0]    # Open3D TriangleMesh

#         orig_path = save_root / f"original_{i}.obj"
#         simp_path = save_root / f"simplified_{i}.obj"

#         # Only write these once (they don't change between epochs)
#         if not orig_path.exists():
#             print(f"[TEST] writing original mesh -> {orig_path}", flush=True)
#             o3d.io.write_triangle_mesh(str(orig_path), orig_mesh)
#         else:
#             print(f"[TEST] original mesh already exists -> {orig_path}", flush=True)

#         if not simp_path.exists():
#             print(f"[TEST] writing simplified mesh -> {simp_path}", flush=True)
#             o3d.io.write_triangle_mesh(str(simp_path), simp_mesh)
#         else:
#             print(f"[TEST] simplified mesh already exists -> {simp_path}", flush=True)

#         # 3) reconstruction from VAE output (x_mean: (B, C, E), B=1)
#         x_mean = out["x_mean"]  # (1, C, E)
#         print(f"[TEST] x_mean shape = {tuple(x_mean.shape)}", flush=True)

#         # Build batch dict expected by tensors_to_meshes()
#         batch = {
#             "verts": x_mean.permute(0, 2, 1),     # (1, E, C)
#             "verts_mask": data["verts_mask"],     # (1, V)
#             "faces": data["faces"],               # (1, F, 3)
#             "faces_mask": data["faces_mask"],     # (1, F)
#         }

#         print("[TEST] converting tensors -> recon mesh", flush=True)
#         recon_mesh = tensors_to_meshes(batch)[0]
#         recon_stats = mesh_stats_o3d(recon_mesh)
#         print(f"[TEST] recon mesh: V={recon_stats['num_vertices']}, "
#               f"F={recon_stats['num_triangles']}, "
#               f"E={recon_stats['num_edges']}", flush=True)

#         # 4) re-detail recon to match original edge scale (clamped)
#         print("[TEST] starting redetail_to_original_edges()", flush=True)
#         recon_detail = redetail_to_original_edges(recon_mesh, orig_mesh)
#         detail_stats = mesh_stats_o3d(recon_detail)
#         print(f"[TEST] recon_redetail mesh: V={detail_stats['num_vertices']}, "
#               f"F={detail_stats['num_triangles']}, "
#               f"E={detail_stats['num_edges']}", flush=True)

#         # 5) Save both recon meshes
#         recon_path = save_root / f"recon_{i}.obj"
#         recon_redetail_path = save_root / f"recon_{i}_redetail.obj"

#         print(f"[TEST] writing recon mesh -> {recon_path}", flush=True)
#         o3d.io.write_triangle_mesh(str(recon_path), recon_mesh)

#         print(f"[TEST] writing recon_redetail mesh -> {recon_redetail_path}", flush=True)
#         o3d.io.write_triangle_mesh(str(recon_redetail_path), recon_detail)

#         print(f"[TEST] DONE sample {i}", flush=True)


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
        # -------------------------------------------------
        # Identify which file this sample corresponds to
        # -------------------------------------------------
        sample_path = Path(data["path"][0])
        base_name = sample_path.stem  # e.g. "Scan_aug_000"

        print(f"\n[TEST] sample {i}")
        print(f"       source path : {sample_path}")
        print(f"       base name   : {base_name}")

        # -------------------------------------------------
        # Forward pass
        # -------------------------------------------------
        model.set_input(data)
        out = model.test()

        # -------------------------------------------------
        # 1) simplified mesh (Open3D) – save it
        # -------------------------------------------------
        simp_mesh = data["simplified"][0]
        simp_stats = mesh_stats_o3d(simp_mesh)
        print(
            f"[TEST] simplified V={simp_stats['num_vertices']} "
            f"F={simp_stats['num_triangles']} E={simp_stats['num_edges']}"
        )

        simp_path = save_root / f"{base_name}_simplified.obj"
        o3d.io.write_triangle_mesh(str(simp_path), simp_mesh)
        print(f"[TEST] wrote simplified -> {simp_path.name}")

        # -------------------------------------------------
        # 2) reconstruction from VAE output
        # -------------------------------------------------
        x_mean = out["x_mean"]  # (1, C, E)
        print(f"[TEST] x_mean shape = {tuple(x_mean.shape)}")

        batch = {
            "verts": x_mean.permute(0, 2, 1),   # (1, E, C)
            "verts_mask": data["verts_mask"],   # (1, V)
            "faces": data["faces"],             # (1, F, 3)
            "faces_mask": data["faces_mask"],   # (1, F)
        }

        recon_mesh = tensors_to_meshes(batch)[0]
        recon_stats = mesh_stats_o3d(recon_mesh)
        print(
            f"[TEST] recon      V={recon_stats['num_vertices']} "
            f"F={recon_stats['num_triangles']} E={recon_stats['num_edges']}"
        )

        # -------------------------------------------------
        # 3) re-detail recon to match simplified resolution
        # -------------------------------------------------
        recon_detail = redetail_to_simplified_edges(recon_mesh, simp_mesh)
        detail_stats = mesh_stats_o3d(recon_detail)
        print(
            f"[TEST] recon_redetail V={detail_stats['num_vertices']} "
            f"F={detail_stats['num_triangles']} E={detail_stats['num_edges']}"
        )

        # -------------------------------------------------
        # 4) Save recon + recon_redetail
        # -------------------------------------------------
        recon_path = save_root / f"{base_name}_recon.obj"
        recon_redetail_path = save_root / f"{base_name}_recon_redetail.obj"

        o3d.io.write_triangle_mesh(str(recon_path), recon_mesh)
        o3d.io.write_triangle_mesh(str(recon_redetail_path), recon_detail)

        print(f"[TEST] wrote recon          -> {recon_path.name}")
        print(f"[TEST] wrote recon_redetail -> {recon_redetail_path.name}")


# =========================================
# Main
# =========================================

if __name__ == "__main__":
    opt = BaseOptions().parse()

    os.makedirs(Path(opt.checkpoints_dir) / opt.name, exist_ok=True)

    if opt.phase == "train":
        opt.is_train = True
        train(opt)

    elif opt.phase == "test":
        opt.is_train = False
        model = MeshVAEModel(opt)
        model.load_network("latest")
        run_test(opt, model, MeshVAEDataset(opt.dataroot, opt))
