# meshvae/geometry.py
#
# Geometry pipeline to plug into your existing MeshVAE:
#   - Load .obj meshes
#   - Simplify them (reduce triangles/edges) -> "padded" / low-res meshes
#   - Convert simplified meshes <-> padded PyTorch tensors (verts/faces)
#   - Save simplified OBJs and tensor NPZs
#   - After VAE recon: re-detail (upsample) recon meshes back towards
#     original edge count.
#
# You can use this module both in scripts and notebooks.

from __future__ import annotations
import os
import re
import time
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import numpy as np
import torch
try:
    import open3d as o3d
except ModuleNotFoundError:  # allows importing non-geometry modules in minimal environments
    class _MissingOpen3D:
        def __getattr__(self, name):
            raise ImportError("open3d is required for mesh geometry operations")
    o3d = _MissingOpen3D()
import trimesh


# --------------------------
# Small general helpers
# --------------------------

def ensure_dir(path: Path | str):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_obj_files(root: Path | str) -> List[Path]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".obj":
        return [root]
    if root.is_dir():
        return sorted(root.rglob("*.obj"))
    raise FileNotFoundError(f"No .obj found at {root}")


def list_obj_files_many(roots: Iterable[Path | str]) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        try:
            found = list_obj_files(root)
        except FileNotFoundError as e:
            print(f"[PREPROC] WARNING: {e}", flush=True)
            continue
        print(f"[PREPROC] root={root} OBJ files={len(found)}", flush=True)
        out.extend(found)
    unique: List[Path] = []
    seen: set[Path] = set()
    for path in out:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _clean_scan_name(x) -> str:
    try:
        from .conditioning import clean_stregkode

        code = clean_stregkode(x)
    except Exception:
        code = None
    if code:
        return code
    stem = Path(str(x)).stem.strip().strip('"').strip("'").strip(" !")
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return stem or "scan"


def _unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}__dup{i}" in used:
        i += 1
    name = f"{base}__dup{i}"
    used.add(name)
    return name


def _masked_rows(arr: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    return [arr[i][mask[i]].copy() for i in range(arr.shape[0])]


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _progress_line(prefix: str, done: int, total: int, start_time: float) -> str:
    total = max(int(total), 1)
    done = max(0, min(int(done), total))
    pct = 100.0 * done / total
    elapsed = time.time() - start_time
    if done > 0:
        eta = elapsed * (total - done) / done
        eta_s = _format_seconds(eta)
    else:
        eta_s = "unknown"
    return (
        f"{prefix} {done}/{total} ({pct:6.2f}%) "
        f"elapsed={_format_seconds(elapsed)} eta={eta_s}"
    )


def unique_edge_count(triangles: np.ndarray) -> int:
    """Number of unique undirected edges in a triangle array (N,3)."""
    if triangles.size == 0:
        return 0
    t = triangles.astype(np.int64)
    e01 = t[:, [0, 1]]
    e12 = t[:, [1, 2]]
    e20 = t[:, [2, 0]]
    edges = np.concatenate([e01, e12, e20], axis=0)
    edges = np.sort(edges, axis=1)  # undirected
    view = np.ascontiguousarray(edges).view([('', edges.dtype)] * edges.shape[1])
    return np.unique(view).shape[0]


def mesh_stats_o3d(mesh: o3d.geometry.TriangleMesh) -> Dict[str, int]:
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    return {
        "num_vertices": int(V.shape[0]),
        "num_triangles": int(F.shape[0]),
        "num_edges": unique_edge_count(F),
    }


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_unreferenced_vertices()
    mesh.triangle_normals = o3d.utility.Vector3dVector()
    mesh.compute_vertex_normals()
    return mesh

# def align_mesh(
#     mesh: o3d.geometry.TriangleMesh,
#     center_mode: str = "bbox",
#     align_axes: bool = True,
# ) -> o3d.geometry.TriangleMesh:
#     """
#     Put the mesh in a common coordinate frame:
#       - subtract a common center (bounding-box center or centroid)
#       - align principal axes via PCA
#     Does *not* change physical scale.
#     """
#     V = np.asarray(mesh.vertices)
#     if V.size == 0:
#         return mesh

#     # 1) Centering
#     if center_mode == "centroid":
#         center = V.mean(axis=0)
#     else:
#         vmin = V.min(axis=0)
#         vmax = V.max(axis=0)
#         center = 0.5 * (vmin + vmax)
#     V_centered = V - center

#     if not align_axes:
#         mesh.vertices = o3d.utility.Vector3dVector(V_centered.astype(np.float64))
#         mesh.compute_vertex_normals()
#         return mesh

#     # 2) PCA-based orientation (principal axes)
#     cov = np.cov(V_centered.T)
#     eigvals, eigvecs = np.linalg.eigh(cov)
#     # sort by descending variance
#     order = np.argsort(eigvals)[::-1]
#     R = eigvecs[:, order]

#     # enforce right-handed, consistent orientation
#     if np.linalg.det(R) < 0:
#         R[:, 0] *= -1

#     V_aligned = V_centered @ R
#     mesh.vertices = o3d.utility.Vector3dVector(V_aligned.astype(np.float64))
#     mesh.compute_vertex_normals()
#     return mesh

def normalize_mesh(
    mesh: o3d.geometry.TriangleMesh,
    center_mode: str = "bbox",
    align_axes: bool = True,
    scale_mode: str = "unit_sphere",  # "unit_sphere" or "unit_box"
    eps: float = 1e-8,
) -> o3d.geometry.TriangleMesh:
    """
    Normalize mesh into a common coordinate frame:
      - center (bbox center or centroid)
      - optionally align principal axes via PCA
      - scale to unit sphere (max radius = 1) or unit box (max extent = 1)
    """
    V = np.asarray(mesh.vertices)
    if V.size == 0:
        return mesh

    # 1) Centering
    if center_mode == "centroid":
        center = V.mean(axis=0)
    else:
        vmin = V.min(axis=0)
        vmax = V.max(axis=0)
        center = 0.5 * (vmin + vmax)
    Vc = V - center

    # 2) PCA alignment
    if align_axes:
        cov = np.cov(Vc.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        R = eigvecs[:, order]
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        Vc = Vc @ R

    # 3) Scale normalization
    if scale_mode == "unit_box":
        vmin = Vc.min(axis=0)
        vmax = Vc.max(axis=0)
        scale = float(np.max(vmax - vmin)) + eps
    else:
        # unit sphere by default
        scale = float(np.linalg.norm(Vc, axis=1).max()) + eps

    Vn = Vc / scale

    mesh.vertices = o3d.utility.Vector3dVector(Vn.astype(np.float64))
    mesh.compute_vertex_normals()
    return mesh



def mesh_to_pointcloud(
    mesh: o3d.geometry.TriangleMesh,
    n_points: int,
    method: str = "poisson",
) -> np.ndarray:
    """
    Sample exactly n_points from the mesh surface.

    Returns:
        (n_points, 3) float32 array.
    """
    if n_points <= 0:
        raise ValueError("n_points must be > 0")

    mesh = clean_mesh(mesh)
    if method == "poisson":
        pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)
    else:
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    
    pts = np.asarray(pcd.points, dtype=np.float32)

    # Ensure exactly n_points (repeat or truncate) WITHOUT injecting origin zeros
    if pts.shape[0] == 0:
        raise RuntimeError("Point sampling returned 0 points (mesh may be invalid).")

    if pts.shape[0] < n_points:
        extra = n_points - pts.shape[0]
        idx = np.random.choice(pts.shape[0], size=extra, replace=True)
        pts = np.concatenate([pts, pts[idx]], axis=0)
    elif pts.shape[0] > n_points:
        pts = pts[:n_points]

    return pts

# --------------------------
# Open3D <-> NumPy <-> torch
# --------------------------

def simplify_mesh(
    mesh: o3d.geometry.TriangleMesh,
    target_tris: int | None = 20000,
    ratio: float | None = None,
) -> o3d.geometry.TriangleMesh:
    """Quadric decimation to reduce triangles/edges."""
    tris = np.asarray(mesh.triangles)
    if tris.size == 0:
        return mesh
    if target_tris is None:
        assert ratio is not None and 0.0 < ratio <= 1.0
        target_tris = max(4, int(len(tris) * float(ratio)))
    target_tris = int(max(4, min(target_tris, len(tris))))
    simp = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)
    return clean_mesh(simp)


def to_numpy(mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.triangles, dtype=np.int64)
    return V, F


def pad_batch(arrs: List[np.ndarray], pad_value: float | int, width: int):
    """
    Pad a list of (N_i, width) arrays into:
      out  : (B, N_max, width)
      mask : (B, N_max)    True = real, False = padded
    """
    if not arrs:
        return np.zeros((0, 0, width)), np.zeros((0, 0), dtype=bool)
    B = len(arrs)
    N_max = max(a.shape[0] for a in arrs)
    out = np.full(
        (B, N_max, width),
        pad_value,
        dtype=arrs[0].dtype,
    )
    mask = np.zeros((B, N_max), dtype=bool)
    for i, a in enumerate(arrs):
        n = a.shape[0]
        if n:
            out[i, :n, :] = a
            mask[i, :n] = True
    return out, mask


def meshes_to_padded_tensors(
    meshes: List[o3d.geometry.TriangleMesh],
) -> Dict[str, torch.Tensor]:
    """
    Convert list of meshes to padded tensors:
      verts      : (B, Vmax, 3) float32
      verts_mask : (B, Vmax)    bool
      faces      : (B, Fmax, 3) int64   (-1 padded)
      faces_mask : (B, Fmax)    bool
    """
    V_list, F_list = [], []
    for m in meshes:
        V, F = to_numpy(m)
        V_list.append(V)
        F_list.append(F)
    V_pad, V_mask = pad_batch(V_list, pad_value=0.0, width=3)
    F_pad, F_mask = pad_batch(F_list, pad_value=-1, width=3)
    return {
        "verts": torch.from_numpy(V_pad),       # float32
        "verts_mask": torch.from_numpy(V_mask), # bool
        "faces": torch.from_numpy(F_pad),       # int64
        "faces_mask": torch.from_numpy(F_mask), # bool
    }


def tensors_to_meshes(batch: Dict[str, torch.Tensor]) -> List[o3d.geometry.TriangleMesh]:
    """
    Inverse of meshes_to_padded_tensors, for a batch of meshes.
    """
    B = batch["verts"].shape[0]
    out: List[o3d.geometry.TriangleMesh] = []
    for i in range(B):
        V = batch["verts"][i][batch["verts_mask"][i]].cpu().numpy().reshape(-1, 3)
        F = batch["faces"][i][batch["faces_mask"][i]].cpu().numpy().reshape(-1, 3)
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(V.astype(np.float64)),
            triangles=o3d.utility.Vector3iVector(F.astype(np.int64)),
        )
        out.append(clean_mesh(mesh))
    return out


# --------------------------
# Pre-processing for dataset
# --------------------------

def preprocess_obj_folder(
    input_root: str | Path | Iterable[str | Path],
    out_root: str | Path,
    target_tris: int = 20000,
    save_npz: bool = True,
    n_points: int | None = None,
    existing_npz: Iterable[str | Path] | None = None,
    dedupe_by_id: bool = True,
    assume_simplified: bool = False,
    point_sample_method: str = "poisson",
) -> Tuple[List[o3d.geometry.TriangleMesh], List[str]]:
    """
    Preprocess a folder (or single .obj) into simplified meshes + optional
    per-mesh tensor NPZs. Returns the list of simplified meshes and their names.

    For each input OBJ:
      - read → clean → simplify
      - save simplified OBJ
      - put into a list for batch tensorization later
    """
    out_root = Path(out_root)
    out_simpl = out_root / "simplified"
    ensure_dir(out_simpl)

    if isinstance(input_root, (str, Path)):
        input_roots = [input_root]
    else:
        input_roots = list(input_root)

    obj_files = list_obj_files_many(input_roots)
    if not obj_files:
        if not existing_npz:
            raise RuntimeError(f"No .obj found under {input_roots}")
        print("[PREPROC] No new OBJ files found; only merging existing NPZ input.", flush=True)

    simplified_meshes: List[o3d.geometry.TriangleMesh] = []
    verts_list: List[np.ndarray] = []
    faces_list: List[np.ndarray] = []
    names: List[str] = []
    points_list: List[np.ndarray] = []
    source_paths: List[str] = []
    seen_ids: set[str] = set()
    used_names: set[str] = set()

    for npz_path in list(existing_npz or []):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Existing NPZ not found: {npz_path}")
        old = np.load(npz_path, allow_pickle=True)
        required = {"verts", "verts_mask", "faces", "faces_mask", "names"}
        missing = sorted(required - set(old.files))
        if missing:
            raise KeyError(f"{npz_path} is missing arrays: {missing}")

        old_names = old["names"]
        old_verts = _masked_rows(old["verts"].astype(np.float32), old["verts_mask"].astype(bool))
        old_faces = _masked_rows(old["faces"].astype(np.int64), old["faces_mask"].astype(bool))
        old_points = old["points"].astype(np.float32) if "points" in old.files else None
        if n_points is not None and n_points > 0:
            if old_points is None:
                raise KeyError(
                    f"{npz_path} has no points array, but --n_points={n_points} was requested. "
                    "Re-preprocess the old OBJ root instead of merging this NPZ."
                )
            if int(old_points.shape[1]) != int(n_points):
                raise ValueError(
                    f"{npz_path} points length is {old_points.shape[1]}, but --n_points={n_points}. "
                    "Use the same point count for old and new data."
                )

        kept = 0
        skipped = 0
        for i, raw_name in enumerate(old_names):
            scan_id = _clean_scan_name(raw_name)
            if dedupe_by_id and scan_id in seen_ids:
                skipped += 1
                continue
            seen_ids.add(scan_id)
            out_name = _unique_name(scan_id, used_names)
            names.append(out_name)
            verts_list.append(old_verts[i])
            faces_list.append(old_faces[i])
            if old_points is not None:
                points_list.append(old_points[i].astype(np.float32))
            source_paths.append(f"{npz_path}:{raw_name}")
            kept += 1
        print(f"[PREPROC] merged existing NPZ {npz_path}: kept={kept} skipped_duplicates={skipped}", flush=True)

    print(f"[PREPROC] Found {len(obj_files)} new OBJ files")
    if assume_simplified:
        print("[PREPROC] assume_simplified=True: skipping normalize/simplify/write-OBJ for input OBJs.", flush=True)

    total_obj = len(obj_files)
    progress_start = time.time()
    processed_obj = 0
    skipped_obj = 0

    for obj_idx, path in enumerate(obj_files, start=1):
        scan_id = _clean_scan_name(path)
        print("\n" + _progress_line("[PREPROC] starting", obj_idx - 1, total_obj, progress_start), flush=True)
        if dedupe_by_id and scan_id in seen_ids:
            print(f"\n[PREPROC] {path.name} -> duplicate id {scan_id}, skipped.")
            skipped_obj += 1
            print(_progress_line("[PREPROC] completed", obj_idx, total_obj, progress_start), flush=True)
            continue
        seen_ids.add(scan_id)
        nm = _unique_name(scan_id, used_names)

        print(f"[PREPROC] file {obj_idx}/{total_obj}: {path.name} -> {nm}", flush=True)
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
        if mesh.is_empty():
            print("  -> EMPTY, skipped.")
            skipped_obj += 1
            print(_progress_line("[PREPROC] completed", obj_idx, total_obj, progress_start), flush=True)
            continue
        mesh = clean_mesh(mesh)

        # 🔹 Align patient body: same center + orientation for all
        # mesh = align_mesh(mesh, center_mode="bbox", align_axes=True)
        if assume_simplified:
            simp = mesh
            s0 = mesh_stats_o3d(simp)
            print(f"  packed existing simplified mesh: V={s0['num_vertices']} F={s0['num_triangles']} E={s0['num_edges']}")
        else:
            mesh = normalize_mesh(mesh, center_mode="bbox", align_axes=True, scale_mode="unit_sphere")
            s0 = mesh_stats_o3d(mesh)
            print(f"  original (aligned): V={s0['num_vertices']} F={s0['num_triangles']} E={s0['num_edges']}")

        # 🔹 Fixed-detail simplification (same target_tris for everyone)
        if not assume_simplified:
            simp = simplify_mesh(mesh, target_tris=target_tris)
        if n_points is not None and n_points > 0:
            pc = mesh_to_pointcloud(simp, n_points=n_points, method=point_sample_method)
            points_list.append(pc)

        s1 = mesh_stats_o3d(simp)
        print(f"  simplified: V={s1['num_vertices']} F={s1['num_triangles']} E={s1['num_edges']}")

        if not assume_simplified:
            out_obj = out_simpl / f"{nm}_simplified.obj"
            o3d.io.write_triangle_mesh(str(out_obj), simp, print_progress=False)
            print(f"  saved simplified OBJ -> {out_obj}")

        V, F = to_numpy(simp)
        verts_list.append(V)
        faces_list.append(F)
        simplified_meshes.append(simp)
        names.append(nm)
        source_paths.append(str(path))
        processed_obj += 1
        print(_progress_line("[PREPROC] completed", obj_idx, total_obj, progress_start), flush=True)

    print(
        f"[PREPROC] OBJ loop done: processed={processed_obj} skipped={skipped_obj} total={total_obj}",
        flush=True,
    )

    # Optionally, convert all simplified meshes to *one* big batch of tensors
    if save_npz and names:
        print(f"[PREPROC] packing arrays for {len(names)} total samples...", flush=True)
        V_pad, V_mask = pad_batch(verts_list, pad_value=0.0, width=3)
        F_pad, F_mask = pad_batch(faces_list, pad_value=-1, width=3)
        out_npz = out_root / "simplified_padded_tensors.npz"

        npz_kwargs = {
            "verts": V_pad,
            "verts_mask": V_mask,
            "faces": F_pad,
            "faces_mask": F_mask,
            "names": np.array(names, dtype=object),
            "source_paths": np.array(source_paths, dtype=object),
        }
        if points_list and len(points_list) == len(names):
            npz_kwargs["points"] = np.stack(points_list, axis=0)  # (B, n_points, 3)
        elif n_points is not None and n_points > 0:
            raise RuntimeError(
                "Internal preprocessing error: points were requested, but not every sample has points."
            )

        print(f"[PREPROC] writing compressed NPZ -> {out_npz}", flush=True)
        np.savez_compressed(out_npz, **npz_kwargs)
        print(f"\n[PREPROC] Saved padded tensor batch -> {out_npz}")
        print(f"[PREPROC] Total samples in combined NPZ: {len(names)}")


    return simplified_meshes, names


# --------------------------
# Re-detail (upsample) recon
# --------------------------

def refine_mesh_to_edge_count(
    mesh: o3d.geometry.TriangleMesh,
    target_edges: int,
    tol: float = 0.05,
    max_subdiv: int = 5,
) -> o3d.geometry.TriangleMesh:
    """
    Take a low-res mesh (e.g. VAE reconstruction on the simplified mesh) and
    upsample it using midpoint subdivision until its edge count is roughly
    target_edges (e.g. the original mesh edge count).

    This does NOT magically recreate all fine detail, but it puts geometry
    back into a similar resolution regime, which is useful for inspection.
    """
    mesh = clean_mesh(mesh)
    cur_stats = mesh_stats_o3d(mesh)
    cur_E = cur_stats["num_edges"]
    print(f"[REFINE] start edges={cur_E}, target={target_edges}")

    if target_edges <= 0 or cur_E == 0:
        return mesh

    # If we're already close, we can just return
    if abs(cur_E - target_edges) / target_edges <= tol:
        print("[REFINE] already within tolerance, no subdivision.")
        return mesh

    m = mesh
    for i in range(max_subdiv):
        if cur_E >= target_edges and (cur_E - target_edges) / target_edges <= tol:
            print(f"[REFINE] reached edges={cur_E} within tol at step {i}")
            break
        print(f"[REFINE] subdivision step {i+1}")
        m = m.subdivide_midpoint(number_of_iterations=1)
        m = clean_mesh(m)
        cur_stats = mesh_stats_o3d(m)
        cur_E = cur_stats["num_edges"]
        print(f"         edges={cur_E}")
        if cur_E >= target_edges and abs(cur_E - target_edges) / target_edges <= tol:
            print("[REFINE] done.")
            break

    return m


def redetail_reconstruction_pair(
    orig_obj: str | Path,
    recon_obj: str | Path,
    out_obj: str | Path,
    tol: float = 0.05,
    max_subdiv: int = 5,
):
    """
    Load (original, reconstructed) OBJs, compute original edge count,
    subdivide reconstructed until ~same edges, then save.

    Use this AFTER your VAE has created recon_obj from the simplified mesh.
    """
    orig_obj = Path(orig_obj)
    recon_obj = Path(recon_obj)
    out_obj = Path(out_obj)
    ensure_dir(out_obj.parent)

    # Original (for target edge count)
    m_orig = o3d.io.read_triangle_mesh(str(orig_obj), enable_post_processing=True)
    m_orig = clean_mesh(m_orig)
    s0 = mesh_stats_o3d(m_orig)
    target_edges = s0["num_edges"]
    print(f"[REDETAIL] {orig_obj.name}: original edges={target_edges}")

    # Reconstruction to be refined
    m_rec = o3d.io.read_triangle_mesh(str(recon_obj), enable_post_processing=True)
    m_rec = clean_mesh(m_rec)

    m_ref = refine_mesh_to_edge_count(
        m_rec, target_edges=target_edges, tol=tol, max_subdiv=max_subdiv
    )
    o3d.io.write_triangle_mesh(str(out_obj), m_ref, print_progress=False)
    print(f"[REDETAIL] saved refined recon -> {out_obj}")


def redetail_folder(
    orig_dir: str | Path,
    recon_dir: str | Path,
    out_dir: str | Path,
    orig_suffix: str = "_simplified.obj",
    recon_prefix: str = "recon_",
    recon_ext: str = ".obj",
):
    """
    Convenience: loop over all recon_i.obj in recon_dir, find matching original
    simplified OBJ in orig_dir, and save refined reconstructions into out_dir.

    This is designed to pair with your current `_save_reconstructions` / 
    `_decode_prior_samples` in `train_mesh_cnn.py` which produce files like:
        original_0.obj, recon_0.obj, original_1.obj, ...
    """
    orig_dir = Path(orig_dir)
    recon_dir = Path(recon_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    recon_files = sorted(
        [p for p in recon_dir.iterdir() if p.is_file() and p.name.startswith(recon_prefix)]
    )
    if not recon_files:
        print(f"[REDETAIL] No recon files in {recon_dir}")
        return

    print(f"[REDETAIL] Found {len(recon_files)} recon files in {recon_dir}")

    for rf in recon_files:
        # Example: recon_0.obj → try to use same index to pick original_0.obj
        name = rf.stem  # e.g. "recon_0"
        idx_str = name.split("_")[-1]
        try:
            idx = int(idx_str)
        except ValueError:
            print(f"  [skip] cannot parse index from {rf.name}")
            continue

        orig_name = f"original_{idx}.obj"
        orig_path = recon_dir / orig_name
        if not orig_path.exists():
            # fallback: if you want to use global simplified dataset, change this mapping
            print(f"  [warn] {orig_name} not found next to recon; skipping index {idx}")
            continue

        out_path = out_dir / f"recon_refined_{idx}.obj"
        redetail_reconstruction_pair(orig_path, rf, out_path)
