from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
try:
    import open3d as o3d
except ModuleNotFoundError:
    class _MissingOpen3D:
        def __getattr__(self, name):
            raise ImportError("open3d is required for test/sample point-cloud output")
    o3d = _MissingOpen3D()

from .conditioning import AGE_BINS, AGE_LABELS, _age_to_bin, athero_slice_from_opt
from .datasets import MeshVAEDataset, PreprocessedMeshVAEDataset, ClinicalFeatureMeshVAEDataset, mesh_collate_fn
from .geometry import mesh_stats_o3d, tensors_to_meshes, preprocess_obj_folder
from .model import MeshVAEModel
from .train_eval import split_dataset, evaluate_subset

def train(opt):
    print("\n========== TRAIN: MeshVAE ==========\n")

    if str(getattr(opt, "feature_source", "scan")) != "scan":
        dataset = ClinicalFeatureMeshVAEDataset(opt.dataroot, opt)
    elif getattr(opt, "use_preprocessed", False):
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
            sl = athero_slice_from_opt(opt)

            labels = []
            for j in idxs:
                c = base_ds.cond[j]
                vals = c[sl] if sl is not None else np.zeros(1, dtype=np.float32)
                y = int(np.max(vals) > 0.5)
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
    n_train_batches = len(train_loader)
    print(
        f"[TRAIN] train samples={len(train_data)}  batch_size={opt.batch_size}  "
        f"batches/epoch={n_train_batches}  epochs={opt.niter + opt.niter_decay - opt.epoch_count + 1}",
        flush=True,
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

            if i % 10 == 0 or (i + 1) == n_train_batches:
                print(
                    f"[epoch {epoch}] batch {i + 1}/{n_train_batches} "
                    f"global_iter={total_iters} loss = {model.loss.item():.4f}"
                )

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
            if "global_iter" in df.columns and "age_adv_loss" in df.columns:
                plt.plot(df["global_iter"], df["age_adv_loss"], label="age_adv_loss")
            if "global_iter" in df.columns and "age_adv_acc" in df.columns:
                plt.plot(df["global_iter"], df["age_adv_acc"], label="age_adv_acc")

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

    if str(getattr(opt, "feature_source", "scan")) != "scan":
        print("[TEST] Skipping reconstruction file dump for clinical feature_source mode.")
        return

    loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=mesh_collate_fn,
    )

    save_root = Path(opt.checkpoints_dir) / opt.name / "reconstructions"
    save_root.mkdir(parents=True, exist_ok=True)

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
    cond_dim = int(getattr(opt, "cond_dim", 2 + len(labels)))

    # base one-hot [M, F, age0, age1, age2, age3]
    c = np.zeros(cond_dim, dtype=np.float32)
    c[0 if gender == "M" else 1] = 1.0

    age_map = {lab: 2 + i for i, lab in enumerate(labels)}
    c[age_map[age_bin]] = 1.0

    off = 2 + len(labels)

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
