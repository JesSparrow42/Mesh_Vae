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
import os
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
import sys
import re
import csv
from collections import Counter
import copy
from typing import Optional

class MeshUnion:
    """
    Memory-safe union structure for edges.
    - Keeps your union-find (parent/rank) exactly as-is.
    - Builds a sparse COO (E_feat x G) group matrix on demand in `prepare_groups`.
    - Uses sparse.mm for the pooling feature rebuild (sum/average per group).
    """
    def __init__(self, n: int, device: torch.device = torch.device("cpu")):
        self.device = device
        self.n = int(n)
        self.parent = torch.arange(self.n, device=self.device)
        self.rank = torch.zeros(self.n, dtype=torch.int8, device=self.device)

        # --- cached mapping and sparse groups for the *latest* features width ---
        self._col_for_edge: Optional[torch.Tensor] = None   # LongTensor [E_feat] (−1 for masked-out)
        self._num_groups: int = 0
        self._groups_sparse: Optional[torch.Tensor] = None  # COO [E_feat, G]
        self._occ_counts: Optional[torch.Tensor] = None     # Float [G]
        self._E_feat_cached: int = -1

    # ----------------- union-find helpers (unchanged) -----------------
    def _find(self, x: torch.Tensor) -> torch.Tensor:
        p = self.parent
        while True:
            px = p[x]
            gx = p[px]
            changed = (px != gx)
            if not changed.any():
                break
            p[x[changed]] = gx[changed]
            x = gx
        return p[x]

    def _find_all(self) -> torch.Tensor:
        p = self.parent
        while True:
            newp = p[p]
            if torch.equal(newp, p):
                break
            p = newp
        self.parent = p
        return p

    # ----------------- union ops (unchanged) -----------------
    def remove_group(self, index: int):
        return  # (no-op for this union style)

    def union(self, source: int, target: int) -> None:
        a = int(source); b = int(target); n = int(self.n)
        if a < 0 or b < 0 or a >= n or b >= n:
            return
        ra = int(self._find(torch.tensor(a, device=self.device)))
        rb = int(self._find(torch.tensor(b, device=self.device)))
        if ra == rb:
            return
        if int(self.rank[ra]) < int(self.rank[rb]):
            self.parent[ra] = rb
        elif int(self.rank[rb]) < int(self.rank[ra]):
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    # ----------------- pooling helpers (sparse-aware) -----------------
    def prepare_groups(self, features: torch.Tensor, mask: np.ndarray) -> None:
        """
        Build a compact mapping edge -> group (0..G-1) for edges selected by `mask`.
        Also builds a sparse COO matrix `self._groups_sparse` of shape [E_feat, G]
        with 1 at (edge, group) membership.
        """
        assert features.dim() == 2, f"features must be [C, E], got {tuple(features.shape)}"
        E_feat = int(features.shape[1])

        # Fast path: reuse if already built for same E_feat
        if self._groups_sparse is not None and self._E_feat_cached == E_feat:
            return

        # Flatten current representatives
        roots = self._find_all()

        # boolean mask, clamped to E_feat so it always matches features
        mask_t = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if mask_t.numel() != E_feat:
            mask_t = mask_t[:E_feat]
            print(f"[WARN] prepare_groups: clamped mask to {E_feat}")

        selected = torch.nonzero(mask_t).squeeze(1)  # [N_selected] in [0..E_feat-1]
        if selected.numel() == 0:
            # nothing selected -> no groups
            self._col_for_edge = torch.full((E_feat,), -1, device=self.device, dtype=torch.long)
            self._num_groups = 0
            self._groups_sparse = torch.sparse_coo_tensor(
                indices=torch.zeros((2, 0), dtype=torch.long, device=self.device),
                values=torch.zeros((0,), dtype=torch.float32, device=self.device),
                size=(E_feat, 0), device=self.device).coalesce()
            self._occ_counts = torch.zeros((0,), dtype=torch.float32, device=self.device)
            self._E_feat_cached = E_feat
            return

        selected_roots = roots[selected]
        unique_roots = torch.unique(selected_roots)
        G = int(unique_roots.numel())

        # root -> compact group id [0..G-1]
        col_for_root = torch.full((self.n,), -1, device=self.device, dtype=torch.long)
        col_for_root[unique_roots] = torch.arange(G, device=self.device, dtype=torch.long)

        # per-edge column for ALL edges (use only first E_feat)
        col_for_edge = col_for_root[roots]
        if col_for_edge.numel() != E_feat:
            col_for_edge = col_for_edge[:E_feat]
            print(f"[WARN] prepare_groups: clamped col_for_edge to {E_feat}")

        self._col_for_edge = col_for_edge
        self._num_groups = G

        # ---- Build sparse COO membership: rows = edge (0..E_feat-1), cols = group (0..G-1) ----
        valid = col_for_edge >= 0
        if valid.any():
            row_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)            # edges
            col_idx = col_for_edge[valid]                                        # group for each valid edge
            vals    = torch.ones_like(row_idx, dtype=torch.float32, device=self.device)

            self._groups_sparse = torch.sparse_coo_tensor(
                indices=torch.stack([row_idx, col_idx], dim=0),
                values=vals,
                size=(E_feat, G),
                device=self.device,
                dtype=torch.float32
            ).coalesce()

            # occurrences per group (size of each group)
            occ = torch.bincount(col_idx, minlength=G).to(torch.float32)
        else:
            self._groups_sparse = torch.sparse_coo_tensor(
                indices=torch.zeros((2, 0), dtype=torch.long, device=self.device),
                values=torch.zeros((0,), dtype=torch.float32, device=self.device),
                size=(E_feat, 0), device=self.device
            ).coalesce()
            occ = torch.zeros((0,), dtype=torch.float32, device=self.device)

        self._occ_counts = occ.to(self.device)
        self._E_feat_cached = E_feat

    def rebuild_features_average(
        self,
        features: torch.Tensor,   # [C, E_feat]
        mask: np.ndarray,         # bool (may be longer than E_feat)
        target_edges: int
    ) -> torch.Tensor:
        """
        Average features in each pooled group using a sparse group matrix.
        Returns [C, target_edges] (padded/truncated).
        """
        assert features.dim() == 2, f"features must be [C, E], got {tuple(features.shape)}"
        C, E_feat = int(features.shape[0]), int(features.shape[1])

        # 1) force mask to match E_feat
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != E_feat:
            mask = mask[:E_feat]
            print(f"[WARN] rebuild_features_average: clamped mask to {E_feat}")

        # 2) ensure sparse groups exist for this E
        self.prepare_groups(features, mask)
        G = int(self._num_groups)

        if G == 0:
            out = features.new_zeros(C, 0)
        else:
            # groups: [E_feat, G] (sparse). We want out = features [C,E] * groups [E,G] -> [C,G]
            # PyTorch supports sparse.mm only as (sparse @ dense) -> dense, so do (groups^T @ features^T)^T
            groups_T = self._groups_sparse.transpose(0, 1)                     # [G, E_feat] (sparse)
            sums_T = torch.sparse.mm(groups_T, features.transpose(0, 1))       # [G, C] dense
            out = sums_T.transpose(0, 1)                                        # [C, G]

            # divide by occurrences to average
            occ = self._occ_counts.clamp_min_(1.0).to(out.dtype)                # [G]
            out = out / occ.unsqueeze(0)                                        # broadcast over C

        # pad/truncate to target_edges
        if out.shape[1] < target_edges:
            pad = features.new_zeros(C, target_edges - out.shape[1])
            out = torch.cat([out, pad], dim=1)
        elif out.shape[1] > target_edges:
            out = out[:, :target_edges]

        return out

    # --- kept for compatibility with your code ---
    def get_group(self, edge_key: int) -> torch.Tensor:
        root = int(self._find(torch.tensor(edge_key, device=self.device)))
        return (self._find_all() == root).nonzero().squeeze(1)

    def get_occurrences(self) -> int:
        # number of non-empty groups among currently pooled ones (unchanged semantics)
        return self._num_groups


class MeshConv(nn.Module):
    """
    Convolution between edges and their 4 incident neighbors (1-ring).
    Inputs:
      x:    [B, C, E] (or [B, C, E, 1])  — E is the *feature width* you pass in
      mesh: list/tuple of B mesh objects, each with .edges_count and .gemm_edges
    Returns:
      [B, C_out, E, 1]
    """
    def __init__(self, in_channels, out_channels, k=5, bias=True):
        super().__init__()
        assert k == 5, "This implementation expects self + 4 neighbors (k=5)."
        self.k = k
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, k), bias=bias)

    def __call__(self, edge_f, mesh):
        return self.forward(edge_f, mesh)

    @staticmethod
    def _finite_or_zero(t: torch.Tensor) -> torch.Tensor:
        # replace NaN/Inf; optionally clamp absurd magnitudes
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        # guard against huge spikes that blow up gradients
        return torch.clamp(t, -1e6, 1e6)

    # ------------------------------- forward --------------------------------
    def forward(self, x: torch.Tensor, mesh):
        """
        x:    [B, C, E] or [B, C, E, 1]
        mesh: IGNORED in this simplified version.

        We build a fake neighbor index Gi where *every* edge just points to itself
        5 times, so the conv still has the right shapes but does not depend on
        any mesh connectivity (no edges_count / gemm_edges needed).
        """
        # Accept legacy [B, C, E, 1]
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        assert x.dim() == 3, f"x must be [B, C, E], got {tuple(x.shape)}"

        # sanitize input
        x = self._finite_or_zero(x)

        B, C, E = x.shape
        device, dtype = x.device, x.dtype

        # ---- Build fake neighbor indices Gi, completely independent of `mesh` ----
        # Each edge uses itself as "self" + 4 neighbors, so Gi[b, e, :] = e
        base = torch.arange(E, device=device, dtype=torch.long)  # [E]
        Gi = base.view(1, E, 1).repeat(B, 1, self.k)             # [B, E, 5]

        # ---- Append sentinel row (index E) as in original code (unused here) ----
        x_ext = torch.cat(
            [x, torch.zeros(B, C, 1, device=device, dtype=dtype)],
            dim=2
        )  # [B, C, E+1]

        # ---- Build “fake image” [B, C, E, 5] and apply 1×5 conv ----
        x_img = self.create_GeMM(x_ext, Gi)          # [B, C, E, 5]
        out = self.conv(x_img)                       # [B, C_out, E, 1]

        out = self._finite_or_zero(out)
        return out

    # ------------------------------ helpers ---------------------------------
    @staticmethod
    def _to_dense_gemm(gemm_edges, K: int = 4) -> np.ndarray:
        """Ragged -> dense [E, K] with -1 padding. If already dense, return as int64."""
        arr = np.asarray(gemm_edges, dtype=object)
        if arr.dtype != object and arr.ndim == 2:
            return arr.astype(np.int64, copy=False)
        E = len(arr)
        out = np.full((E, K), -1, dtype=np.int64)
        for i, nbrs in enumerate(arr):
            if nbrs is None:
                continue
            n = np.asarray(nbrs, dtype=np.int64).ravel()
            out[i, :min(K, n.size)] = n[:K]
        return out

    def _pad_gemm_using_E(self, m, E_feat: int, device: torch.device) -> torch.Tensor:
        """
        Build [1, E_feat, 5] for this mesh using E_feat as the “truth”.
        - rows >= m.edges_count become fully sentinel
        - any neighbor id <0 or >= E_feat -> sentinel E_feat
        - for rows beyond true edges, the 'self' id is sentinel
        """
        E_true = int(m.edges_count)
        # dense neighbors of *true* mesh
        nbrs_np = self._to_dense_gemm(m.gemm_edges, K=4)    # [E_true, 4] with -1
        # make a full [E_feat, 4] filled with -1 and copy the valid top part
        nbrs_full = np.full((E_feat, 4), -1, dtype=np.int64)
        copy_n = min(E_true, E_feat)
        if copy_n > 0:
            nbrs_full[:copy_n, :] = nbrs_np[:copy_n, :]

        gemm = torch.as_tensor(nbrs_full, device=device, dtype=torch.long)   # [E_feat, 4]
        # self column: valid ids for first copy_n rows, sentinel for the rest
        self_ids = torch.arange(E_feat, device=device, dtype=torch.long)     # [E_feat]
        if copy_n < E_feat:
            self_ids[copy_n:] = E_feat

        G = torch.cat([self_ids.unsqueeze(1), gemm], dim=1)                  # [E_feat, 5]
        # clamp invalid / out-of-range neighbors to sentinel E_feat
        G[:, 1:] = torch.where((G[:, 1:] >= 0) & (G[:, 1:] < E_feat), G[:, 1:], torch.full_like(G[:, 1:], E_feat))
        return G.unsqueeze(0)                                                # [1, E_feat, 5]

    @staticmethod
    def _flatten_gemm_inds(Gi: torch.Tensor) -> torch.Tensor:
        """[B, E, 5] -> flat indices for x_ext.view(B*(E+1), C)."""
        B, E, K = Gi.shape
        step = E + 1
        base = torch.arange(B, device=Gi.device).view(B, 1, 1) * step  # [B,1,1]
        return (Gi + base).view(B, E, K)                                # [B, E, 5]

    # --------------------- keep-named helper for compatibility --------------
    def create_GeMM(self, x_ext: torch.Tensor, Gi: torch.Tensor) -> torch.Tensor:
        """
        Build the “fake image” by gathering x_ext with Gi and applying MeshCNN
        symmetric functions. x_ext is [B, C, E+1]; Gi is [B, E, 5] with values in [0..E].
        Returns [B, C, E, 5] (order: self, sum12, sum34, abs12, abs34).
        """
        B, C, E1 = x_ext.shape
        E = E1 - 1

        Gi_flat = self._flatten_gemm_inds(Gi).view(-1).long()               # [B*E*5]
        x_flat  = x_ext.permute(0, 2, 1).contiguous().view(B*(E+1), C)      # [B*(E+1), C]
        f = x_flat.index_select(0, Gi_flat).view(B, E, 5, C).permute(0, 3, 1, 2)  # [B, C, E, 5]

        # MeshCNN equivariant combos
        x1 = f[:, :, :, 1] + f[:, :, :, 2]
        x2 = f[:, :, :, 3] + f[:, :, :, 4]
        x3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 2])
        x4 = torch.abs(f[:, :, :, 3] - f[:, :, :, 4])
        x_img = torch.stack([f[:, :, :, 0], x1, x2, x3, x4], dim=3)         # [B, C, E, 5]
        return x_img

class MeshPool(nn.Module):
    def __init__(
        self,
        target: int,
        multi_thread: bool = False,
        mode: str = 'target',       # 'target' | 'frac' | 'skip'
        keep_frac: float = 1.0,     # used only when mode == 'frac'
        debug: bool = False,
    ):
        super().__init__()
        self.__out_target  = int(target)         # width after this pool stage
        self.__multi_thread = multi_thread
        self.__pool_mode    = str(mode)
        self.__keep_frac    = float(keep_frac)
        self.__debug        = bool(debug)

        # runtime slots
        self.__fe         = None
        self.__meshes     = None
        self.__updated_fe = None
        self.__merge_edges = [-1, -1]

    def __call__(self, fe, meshes):
        return self.forward(fe, meshes)

    def forward(self, fe, meshes):
        """
        fe:     [B, C, E] or [B, C, E, 1]
        meshes: list of B mesh objects; this op mutates them (records groups)
        returns:
            out_features: [B, C, self.__out_target]  (or close, depending on keep_frac)
        """
        # dim guard
        if fe.dim() == 4 and fe.shape[-1] == 1:
            print(f"[POOL] squeeze trailing 1 from {tuple(fe.shape)}", flush=True)
            fe = fe.squeeze(-1)

        self.__updated_fe = [[] for _ in range(len(meshes))]
        self.__fe = fe
        self.__meshes = meshes

        if self.__multi_thread:
            threads = []
            for mesh_index in range(len(meshes)):
                th = Thread(target=self.__pool_main, args=(mesh_index,))
                threads.append(th); th.start()
            for mesh_index in range(len(meshes)):
                threads[mesh_index].join()
        else:
            for mesh_index in range(len(meshes)):
                self.__pool_main(mesh_index)

        out = torch.stack(self.__updated_fe, dim=0)   # [B, C, target]
        if True: print(f"[POOL] out {tuple(out.shape)} (target≈{self._out_target})", flush=True)
        return out

    # ------------ internals (your code + stop rules cleaned) ------------
    def __pool_main(self, mesh_index: int):
        import math, numpy as np
        from heapq import heapify, heappop

        mesh = self.__meshes[mesh_index]
        fe_raw = self.__fe[mesh_index]             # [C, E_feat]
        if fe_raw.dim() == 3 and fe_raw.size(-1) == 1:
            fe_raw = fe_raw.squeeze(-1)
        assert fe_raw.dim() == 2, f"fe_raw must be [C, E], got {tuple(fe_raw.shape)}"

        C, E_feat = int(fe_raw.shape[0]), int(fe_raw.shape[1])
        E_mesh    = int(mesh.edges.shape[0])
        E_used    = min(E_mesh, E_feat)            # safe number of edge rows we can touch
        fe_full   = fe_raw[:, :E_used].contiguous()

        # initial queue from your scoring function
        queue = self.__build_queue(fe_full)        # keep your existing method
        if not queue:
            # no merges suggested -> just pass through & pad to target
            out = fe_full
            K = min(out.shape[1], self.__out_target)
            out = out[:, :K]
            if K < self.__out_target:
                out = torch.nn.functional.pad(out, (0, self.__out_target - K))
            self.__updated_fe[mesh_index] = out
            return
        heapify(queue)

        # union structure operates in feature width
        edge_groups = MeshUnion(E_used, device=fe_full.device)

        # bookkeeping
        merges = 0
        LOG_EVERY = 50
        start_t = time.time() if self.__debug else None

        while queue:
            # ---- compute stop based on mode ----
            E_rows = int(mesh.edges.shape[0])  # CURRENT rows after previous merges
            if self.__pool_mode == 'frac':
                # keep roughly keep_frac * current rows, but never below 0
                stop_at = max(0, int(math.ceil(E_rows * self.__keep_frac)))
            else:
                # 'target' (original behavior)
                stop_at = int(self.__out_target)

            if mesh.edges_count <= stop_at:
                break

            # fresh full-length mask for CURRENT rows
            mask_full = np.ones(E_rows, dtype=np.bool_)

            # best edge to merge now
            _, edge_id = heappop(queue)
            eid = int(edge_id)
            if eid < 0 or eid >= E_rows:
                continue  # stale id

            ok = self.__pool_edge(mesh, eid, mask_full, edge_groups)
            if not ok:
                continue

            merges += 1
            if self.__debug and (merges % LOG_EVERY == 0):
                print(f"[POOL] mesh={mesh_index} merges={merges} edges={mesh.edges_count} elapsed={time.time()-start_t:.1f}s",
                      flush=True)

            # your code likely updates queue incrementally here as well
            # (left as-is in your implementation)

        # ---- rebuild features (average per group), then pad to out_target ----
        short_mask = np.ones(min(E_used, mesh.edges_count), dtype=np.bool_)
        K = int(mesh.edges_count)
        fe = edge_groups.rebuild_features_average(fe_full, short_mask, target_edges=K)

        if fe.shape[1] < self.__out_target:
            pad_cols = int(self.__out_target - fe.shape[1])
            fe = torch.nn.functional.pad(fe, (0, pad_cols))
        elif fe.shape[1] > self.__out_target:
            fe = fe[:, :self.__out_target]

        if self.__debug:
            print(f"[POOL] mesh={mesh_index} done: merges={merges}, final_edges={mesh.edges_count}, "
                  f"time={time.time()-start_t:.1f}s", flush=True)

        self.__updated_fe[mesh_index] = fe


    def __pool_edge(self, mesh, edge_id, mask, edge_groups):
        if self.has_boundaries(mesh, edge_id):
            return False
        elif self.__clean_side(mesh, edge_id, mask, edge_groups, 0)\
            and self.__clean_side(mesh, edge_id, mask, edge_groups, 2) \
            and self.__is_one_ring_valid(mesh, edge_id):
            self.__merge_edges[0] = self.__pool_side(mesh, edge_id, mask, edge_groups, 0)
            self.__merge_edges[1] = self.__pool_side(mesh, edge_id, mask, edge_groups, 2)
            mesh.merge_vertices(edge_id)
            mask[edge_id] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_id)
            mesh.edges_count -= 1
            return True
        else:
            return False

    def __clean_side(self, mesh, edge_id, mask, edge_groups, side):
        if mesh.edges_count <= self.__out_target:
            return False
        invalid_edges = MeshPool.__get_invalids(mesh, edge_id, edge_groups, side)
        while len(invalid_edges) != 0 and mesh.edges_count > self.__out_target:
            self.__remove_triplete(mesh, mask, edge_groups, invalid_edges)
            if mesh.edges_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh, edge_id):
                return False
            invalid_edges = self.__get_invalids(mesh, edge_id, edge_groups, side)
        return True

    @staticmethod
    def has_boundaries(mesh, edge_id):
        for edge in mesh.gemm_edges[edge_id]:
            if edge == -1 or -1 in mesh.gemm_edges[edge]:
                return True
        return False

    @staticmethod
    def __is_one_ring_valid(mesh, edge_id):
        v_a = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 0]]].reshape(-1))
        v_b = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 1]]].reshape(-1))
        shared = v_a & v_b - set(mesh.edges[edge_id])
        return len(shared) == 2

    def __pool_side(self, mesh, edge_id, mask, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2, other_keys_b[0], mesh.sides[key_b, other_side_b])
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2 + 1, other_keys_b[1], mesh.sides[key_b, other_side_b + 1])
        MeshPool.__union_groups(mesh, edge_groups, key_b, key_a)
        MeshPool.__union_groups(mesh, edge_groups, edge_id, key_a)
        mask[key_b] = False
        MeshPool.__remove_group(mesh, edge_groups, key_b)
        mesh.remove_edge(key_b)
        mesh.edges_count -= 1
        return key_a

    @staticmethod
    def __get_invalids(mesh, edge_id, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b = info
        shared_items = MeshPool.__get_shared_items(other_keys_a, other_keys_b)
        if len(shared_items) == 0:
            return []
        else:
            assert (len(shared_items) == 2)
            middle_edge = other_keys_a[shared_items[0]]
            update_key_a = other_keys_a[1 - shared_items[0]]
            update_key_b = other_keys_b[1 - shared_items[1]]
            update_side_a = mesh.sides[key_a, other_side_a + 1 - shared_items[0]]
            update_side_b = mesh.sides[key_b, other_side_b + 1 - shared_items[1]]
            MeshPool.__redirect_edges(mesh, edge_id, side, update_key_a, update_side_a)
            MeshPool.__redirect_edges(mesh, edge_id, side + 1, update_key_b, update_side_b)
            MeshPool.__redirect_edges(mesh, update_key_a, MeshPool.__get_other_side(update_side_a), update_key_b, MeshPool.__get_other_side(update_side_b))
            MeshPool.__union_groups(mesh, edge_groups, key_a, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_b, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_a, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, key_b, update_key_b)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_b)
            return [key_a, key_b, middle_edge]

    @staticmethod
    def __redirect_edges(mesh, edge_a_key, side_a, edge_b_key, side_b):
        mesh.gemm_edges[edge_a_key, side_a] = edge_b_key
        mesh.gemm_edges[edge_b_key, side_b] = edge_a_key
        mesh.sides[edge_a_key, side_a] = side_b
        mesh.sides[edge_b_key, side_b] = side_a

    @staticmethod
    def __get_shared_items(list_a, list_b):
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]:
                    shared_items.extend([i, j])
        return shared_items

    @staticmethod
    def __get_other_side(side):
        return side + 1 - 2 * (side % 2)

    @staticmethod
    def __get_face_info(mesh, edge_id, side):
        key_a = mesh.gemm_edges[edge_id, side]
        key_b = mesh.gemm_edges[edge_id, side + 1]
        side_a = mesh.sides[edge_id, side]
        side_b = mesh.sides[edge_id, side + 1]
        other_side_a = (side_a - (side_a % 2) + 2) % 4
        other_side_b = (side_b - (side_b % 2) + 2) % 4
        other_keys_a = [mesh.gemm_edges[key_a, other_side_a], mesh.gemm_edges[key_a, other_side_a + 1]]
        other_keys_b = [mesh.gemm_edges[key_b, other_side_b], mesh.gemm_edges[key_b, other_side_b + 1]]
        return key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b

    @staticmethod
    def __remove_triplete(mesh, mask, edge_groups, invalid_edges):
        vertex = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            vertex &= set(mesh.edges[edge_key])
            mask[edge_key] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_key)
        mesh.edges_count -= 3
        vertex = list(vertex)
        assert(len(vertex) == 1)
        mesh.remove_vertex(vertex[0])

    def __build_queue(self, features: torch.Tensor) -> list:
        """
        Build a Python min-heap of (score, edge_id) from edge features.

        Accepts feature tensors in shapes:
        [C, E], [E, C], [1, C, E], [C, E, 1], [E, C, 1], [1, E, C]
        We normalize to 2D and infer which axis is edges.
        """
        f = features

        # ---- normalize to 2D ----
        # Drop size-1 dims (batch / trailing image dim) conservatively.
        while f.dim() > 2 and (f.size(0) == 1 or f.size(-1) == 1):
            # prefer to squeeze leading/trailing singleton dims only
            if f.size(0) == 1:
                f = f.squeeze(0)
            if f.dim() > 2 and f.size(-1) == 1:
                f = f.squeeze(-1)

        # Fallback: if still >2D, squeeze any other singleton dims (safety net)
        if f.dim() > 2:
            squeeze_dims = [d for d in range(f.dim()) if f.size(d) == 1]
            for d in reversed(squeeze_dims):
                f = f.squeeze(d)

        if f.dim() != 2:
            raise ValueError(f"__build_queue: unexpected features shape {tuple(f.shape)}")

        # ---- decide which axis is edges ----
        # We prefer channels-first: [C, E]. If that isn't true, flip.
        C, E = f.shape
        if C > E:       # looks like [E, C]
            f = f.t()
            C, E = f.shape

        # ---- compute scores per edge ----
        # scores shape: [E]
        mags = (f * f).sum(dim=0)  # [C, E] -> sum over C
        # Make sure no NaNs/Infs break the heap
        if not torch.isfinite(mags).all():
            mags = torch.nan_to_num(mags, nan=0.0, posinf=0.0, neginf=0.0)

        scores = mags.detach().cpu().tolist()
        heap = list(zip(scores, range(len(scores))))
        # min-heap is built by caller (heapify)
        return heap


    @staticmethod
    def __union_groups(mesh, edge_groups, source, target):
        edge_groups.union(source, target)
        mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, edge_groups, index):
        edge_groups.remove_group(index)
        mesh.remove_group(index)

class MeshUnpool(nn.Module):
    def __init__(self, unroll_target: int, chunk_cols: int = 500):
        super().__init__()
        self.unroll_target = unroll_target  # integer; 10**9 used as "full"
        self.chunk_cols = int(chunk_cols)

    def __call__(self, features, meshes):
        return self.forward(features, meshes)

    # ---------------- helpers ----------------
    @staticmethod
    def _sparse_resize(sp: torch.Tensor, new_rows: int, new_cols: int) -> torch.Tensor:
        """
        Return a coalesced COO with the same indices/values clamped to new size.
        Used to 'pad' or 'crop' without densifying.
        """
        idx = sp._indices()
        val = sp._values()
        if idx.numel() == 0:
            return torch.sparse_coo_tensor(
                indices=idx, values=val, size=(new_rows, new_cols),
                device=sp.device, dtype=sp.dtype).coalesce()
        row_ok = idx[0] < new_rows
        col_ok = idx[1] < new_cols
        keep = row_ok & col_ok
        idx2 = idx[:, keep]
        val2 = val[keep]
        return torch.sparse_coo_tensor(
            indices=idx2, values=val2, size=(new_rows, new_cols),
            device=sp.device, dtype=sp.dtype).coalesce()

    @staticmethod
    def _sparse_narrow_cols(sp: torch.Tensor, start: int, length: int, n_rows: Optional[int] = None) -> torch.Tensor:
        """
        Column-narrow a COO tensor without densifying. Keeps rows as-is (optionally clamps to n_rows).
        Returns a new COO with columns shifted to [0..length-1].
        """
        if length <= 0:
            return torch.sparse_coo_tensor(
                indices=torch.zeros((2, 0), dtype=torch.long, device=sp.device),
                values=torch.zeros((0,), dtype=sp.dtype, device=sp.device),
                size=(n_rows if n_rows is not None else sp.shape[0], 0),
                device=sp.device, dtype=sp.dtype).coalesce()
        idx = sp._indices()
        val = sp._values()
        if idx.numel() == 0:
            return torch.sparse_coo_tensor(
                indices=idx, values=val,
                size=(n_rows if n_rows is not None else sp.shape[0], length),
                device=sp.device, dtype=sp.dtype).coalesce()
        # select columns in [start, start+length)
        col = idx[1]
        mask = (col >= start) & (col < start + length)
        idx2 = idx[:, mask].clone()
        idx2[1] = idx2[1] - start
        if n_rows is not None and n_rows != sp.shape[0]:
            row_mask = idx2[0] < n_rows
            idx2 = idx2[:, row_mask]
            val = val[mask][row_mask]
        else:
            val = val[mask]
        return torch.sparse_coo_tensor(
            indices=idx2, values=val,
            size=(n_rows if n_rows is not None else sp.shape[0], length),
            device=sp.device, dtype=sp.dtype).coalesce()

    def pad_groups(self, group: torch.Tensor, unroll_start: int) -> torch.Tensor:
        """
        group expected shape: [start_E, end_E] (your recorded history).
        Works for both dense and sparse groups.
        """
        start, end = group.shape
        pad_rows = unroll_start - start
        pad_cols = int(self.unroll_target) - end

        if getattr(group, "is_sparse", False):
            # Sparse 'pad' is just changing the declared size (and cropping if too big)
            new_rows = start + max(pad_rows, 0)
            new_cols = end + max(pad_cols, 0)
            # if negative pads, we crop to target
            if pad_rows < 0:
                new_rows = unroll_start
            if pad_cols < 0:
                new_cols = int(self.unroll_target)
            return self._sparse_resize(group, new_rows, new_cols)

        # Dense path (your original)
        if pad_rows != 0 or pad_cols != 0:
            # crop if negative, then pad
            if pad_rows < 0 or pad_cols < 0:
                new_r = start + min(pad_rows, 0)
                new_c = end + min(pad_cols, 0)
                group = group[:new_r, :new_c]
                pad_rows = unroll_start - new_r
                pad_cols = int(self.unroll_target) - new_c
            padding = nn.ConstantPad2d((0, pad_cols, 0, pad_rows), 0)
            group = padding(group)
        return group

    @staticmethod
    def pad_occurrences(occ: torch.Tensor, target_cols: int) -> torch.Tensor:
        """
        Pad/crop occurrences to length `target_cols`. Fills with 1s (neutral).
        """
        if occ is None:
            return None
        L = int(occ.shape[0])
        if L == target_cols:
            return occ
        if L > target_cols:
            return occ[:target_cols]
        pad = target_cols - L
        return nn.functional.pad(occ, (0, pad), mode="constant", value=1)

    def forward(self, features: torch.Tensor, meshes):
        # -------- DIM GUARD --------
        if features.dim() == 4 and features.shape[-1] == 1:
            features = features.squeeze(-1)
        assert features.dim() == 3, f"[UNPOOL] expected 3D [B, C, E], got {tuple(features.shape)}"

        B, C, E_now = features.shape
        device = features.device
        target_cols = int(self.unroll_target) if (getattr(self, "unroll_target", None) is not None and self.unroll_target < 10**8) else E_now

        # ---- HISTORY PATH (fast when groups were recorded) ----
        def safe_groups(m):
            try:
                return m.get_groups(E_now)
            except Exception:
                return None

        groups_per_mesh = [safe_groups(m) for m in meshes]
        all_have_history = all(g is not None for g in groups_per_mesh)

        if all_have_history:
            # Gather/pad occurrences if provided
            def safe_occ(m):
                try:
                    o = m.get_occurrences()
                    # accept vector occurrences only; otherwise ignore
                    if isinstance(o, torch.Tensor) and o.dim() == 1:
                        return o.to(device)
                except Exception:
                    pass
                return None

            occ_list = [safe_occ(m) for m in meshes]
            occ_list = [self.pad_occurrences(o, target_cols) if o is not None else None for o in occ_list]

            # Case A: ALL groups are dense -> use your original fast path
            all_dense = all(not getattr(g, "is_sparse", False) for g in groups_per_mesh)
            if all_dense:
                groups = [self.pad_groups(g.to(device), E_now) for g in groups_per_mesh]   # [E_now, target_cols] each
                unroll_mat = torch.cat(groups, dim=0).view(B, E_now, -1)                   # [B, E_now, target_cols]

                # occurrences multiplication (preserve your semantics)
                if any(o is not None for o in occ_list):
                    occ_stack = []
                    for o in occ_list:
                        if o is None:
                            o = torch.ones(target_cols, device=device, dtype=unroll_mat.dtype)
                        occ_stack.append(o)
                    occurrences = torch.stack(occ_stack, dim=0).view(B, 1, target_cols)
                    unroll_mat = unroll_mat * occurrences

                # update connectivity history
                for m in meshes:
                    try:
                        m.unroll_gemm()
                    except Exception:
                        pass

                return torch.matmul(features, unroll_mat.to(device))  # [B, C, target_cols]

            # Case B: groups include sparse -> do chunked slices per batch
            out_batch = []
            for b in range(B):
                g = self.pad_groups(groups_per_mesh[b].to(device), E_now)  # [E_now, target_cols] (sparse or dense)
                f = features[b]                                           # [C, E_now]
                # per-mesh occurrences (vector)
                occ = occ_list[b] if (b < len(occ_list)) else None
                if occ is None:
                    occ = torch.ones(target_cols, device=device, dtype=f.dtype)

                # Iterate over column chunks
                col = 0
                chunks = []
                while col < target_cols:
                    length = min(self.chunk_cols, target_cols - col)

                    if getattr(g, "is_sparse", False):
                        g_slice_sp = self._sparse_narrow_cols(g, col, length, n_rows=E_now)  # [E_now, length] (sparse)
                        g_slice = g_slice_sp.to_dense()                                      # small dense slice
                    else:
                        g_slice = g[:, col:col + length]                                     # already dense

                    # occurrences multiplication (your semantics): broadcast across rows
                    occ_slice = occ[col:col + length].to(g_slice.dtype)
                    g_slice = g_slice * occ_slice.unsqueeze(0)                                # [E_now, length]

                    # matmul: [C, E_now] @ [E_now, length] -> [C, length]
                    chunk_out = torch.matmul(f, g_slice)
                    chunks.append(chunk_out)

                    col += length

                out_b = torch.cat(chunks, dim=1)  # [C, target_cols]
                out_batch.append(out_b.unsqueeze(0))

            # update connectivity history (like your code)
            for m in meshes:
                try:
                    m.unroll_gemm()
                except Exception:
                    pass

            return torch.cat(out_batch, dim=0)  # [B, C, target_cols]

        # ---- FALLBACK (NO HISTORY / SKIP MODE) ----
        # choose a single batch‑wide target (keep-equal if unroll_target ~ "full")
        stage_target = target_cols
        idx = torch.linspace(0, E_now - 1, stage_target, device=device).round().long()
        return features[:, :, idx]
