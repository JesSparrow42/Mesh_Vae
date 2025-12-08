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
    preprocess_obj_folder,
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
        simp = simplify_mesh(mesh, target_tris=None, ratio=1.0)

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

    #dataset = MeshVAEDataset(opt.dataroot, opt)
    if getattr(opt, "use_preprocessed", False):
        dataset = PreprocessedMeshVAEDataset(opt.dataroot, opt)
    else:
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
        model.load_network("latest")
        run_test(opt, model, dataset)





