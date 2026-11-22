#!/usr/bin/env python3
"""
Clean MeshVAE model with boson-sampler latent prior.

- MeshEncoder / MeshDecoder built from MeshConv / MeshPool / MeshUnpool
- MeshVAE with Gaussian or Boson prior
- BosonSamplerTorch + BosonLatentGenerator + BosonPrior
- MeshVariationalInference (ELBO)
- MeshVAEModel wrapper: training, testing, save/load, LR scheduler

This is designed to work with your existing mesh_conv_model.py:
    from models.mesh_conv_model import MeshConv, MeshPool, MeshUnpool
"""

import os
import math
import itertools
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Distribution

# Optional boson flag (we keep it simple / always available in this file)
_BOSON_AVAILABLE = True


# ==============================================================================
# Boson prior & boson sampling utilities
# ==============================================================================

class BosonPrior(Distribution):
    """
    Prior distribution wrapper that delegates rsample() to a boson latent generator.
    Falls back to a standard Normal log_prob for KL calculation.
    """
    arg_constraints = {}
    has_rsample = True

    def __init__(self, boson_sampler, batch_size, latent_features,
                 discriminator_iter: int = 0, validate_args=None):
        super().__init__(validate_args=validate_args)
        self.boson_sampler = boson_sampler
        self.batch_size = batch_size
        self.latent_features = latent_features
        self.discriminator_iter = discriminator_iter

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        total_samples = (1 + self.discriminator_iter) * self.batch_size
        latent = self.boson_sampler(total_samples).to(self.device)  # (total, latent_dim)

        if self.discriminator_iter and self.discriminator_iter > 0:
            chunks = torch.chunk(latent, 1 + self.discriminator_iter, dim=0)
            return chunks[0]  # (batch_size, latent_dim)
        else:
            return latent[: self.batch_size]  # (batch_size, latent_dim)

    def log_prob(self, z):
        # standard Normal log_prob (diagonal), summed over last dim
        mean = torch.zeros_like(z)
        std = torch.ones_like(z)
        log_scale = torch.log(std)
        log_prob = -0.5 * (
            ((z - mean) ** 2) / (std ** 2)
            + 2 * log_scale
            + torch.log(torch.tensor(2 * torch.pi, device=z.device))
        )
        return log_prob.sum(dim=-1)


# ==============================================================================
# Boson sampling simulation helpers
# ==============================================================================

def generate_unitary(m, bs_loss, bs_jitter,
                     phase_noise_std=0.0,
                     systematic_phase_offset=0.0,
                     mode_loss=None):
    random_matrix = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    Q, _ = np.linalg.qr(random_matrix)

    # interpret 0 as no loss, 1 as full loss
    bs_trans = float(np.clip(1.0 - bs_loss, 1e-6, 1.0))
    Q *= np.sqrt(bs_trans)

    if bs_jitter > 0.0:
        Q = Q + bs_jitter * (np.random.randn(m, m) + 1j * np.random.randn(m, m))

    if phase_noise_std > 0.0:
        phase_noise = np.exp(1j * np.random.normal(0.0, phase_noise_std, size=(m, m)))
        Q *= phase_noise

    phase = np.asarray(systematic_phase_offset, dtype=np.float32)
    Q *= np.exp(1j * phase)

    if mode_loss is not None:
        ml = np.asarray(mode_loss, dtype=np.float32).reshape(m)
        mode_trans = np.clip(1.0 - ml, 1e-6, 1.0)
        Q = np.array([Q[i, :] * np.sqrt(mode_trans[i]) for i in range(m)], dtype=np.complex64)

    return Q


def compute_permanent(A: torch.Tensor) -> torch.Tensor:
    """
    Brute-force permanent for small matrices (complex dtype).
    """
    n = A.shape[0]
    perm_sum = torch.zeros([], dtype=A.dtype, device=A.device)

    for sigma in itertools.permutations(range(n)):
        product_term = torch.tensor(1.0, dtype=A.dtype, device=A.device)
        for i in range(n):
            product_term = product_term * A[i, sigma[i]]
        perm_sum = perm_sum + product_term

    return perm_sum


def get_submatrix(U, input_modes, output_modes):
    return U[np.ix_(input_modes, output_modes)]


def kerr_phase_factor(output_modes, m, chi):
    """
    Kerr phase factor exp(-1j * chi * n * (n-1)) per mode.
    """
    counts = np.bincount(output_modes, minlength=m)
    factors = [np.exp(-1j * chi * n * (n - 1)) for n in counts]
    return factors


def boson_sampling_probability(U, input_modes, output_modes, effective_mu):
    if len(input_modes) == 0 or len(output_modes) == 0:
        return 0.0

    U_sub = get_submatrix(U, input_modes, output_modes)
    A_sub_torch = torch.tensor(U_sub, dtype=torch.complex64)
    permanent_val = compute_permanent(A_sub_torch)

    classical_probability = np.prod(np.abs(U_sub) ** 2)
    permanent_val_np = permanent_val.detach().cpu().numpy()
    probability = (
        effective_mu * np.abs(permanent_val_np) ** 2
        + (1 - effective_mu) * classical_probability
    )
    return float(probability)


def boson_sampling_probability_advanced(U, input_modes, output_modes, effective_mu, m, chi):
    if len(input_modes) == 0 or len(output_modes) == 0:
        return 0.0

    U_sub = get_submatrix(U, input_modes, output_modes)
    A_sub_torch = torch.tensor(U_sub, dtype=torch.complex64)
    permanent_val = compute_permanent(A_sub_torch)

    classical_probability = np.prod(np.abs(U_sub) ** 2)
    nl_factor_list = kerr_phase_factor(output_modes, m, chi)
    nl_factor_val = np.prod(nl_factor_list)

    permanent_val_np = permanent_val.detach().cpu().numpy()
    combined_val = nl_factor_val * permanent_val_np
    probability = (
        effective_mu * np.abs(combined_val) ** 2
        + (1 - effective_mu) * classical_probability
    )
    return float(probability)


def sample_input_state(m, num_sources,
                       input_loss, coupling_efficiency,
                       detector_inefficiency, multi_photon_prob):
    interfering_input_modes = []
    total_photons = 0

    # interpret 0 loss as ideal, 1 loss as dead
    p_survive = (1.0 - input_loss) * coupling_efficiency * (1.0 - detector_inefficiency)
    p_survive = float(np.clip(p_survive, 0.0, 1.0))

    for _ in range(num_sources):
        if np.random.rand() < p_survive:
            n_emit = 2 if np.random.rand() < multi_photon_prob else 1
            total_photons += n_emit
            modes = np.random.choice(m, n_emit, replace=True)
            interfering_input_modes.extend(modes.tolist())

    return np.asarray(interfering_input_modes, dtype=np.int64), total_photons


def apply_dark_counts(probability, m, dark_count_rate):
    dark_counts = dark_count_rate * m
    return float(probability + dark_counts)


def boson_sampling_simulation(
    m, num_sources, num_loops,
    input_loss, coupling_efficiency, detector_inefficiency, multi_photon_prob,
    mu, temporal_mismatch, spectral_mismatch, arrival_time_jitter,
    bs_loss, bs_jitter, phase_noise_std, systematic_phase_offset, mode_loss,
    dark_count_rate,
    use_advanced_nonlinearity=False, chi=0.0,
):
    """
    Run multiple boson sampling simulation loops and return avg transition prob.
    """
    probabilities = []
    effective_mu = (
        mu
        * (1 - temporal_mismatch)
        * (1 - spectral_mismatch)
        * (1 - arrival_time_jitter)
    )

    for _ in range(num_loops):
        U = generate_unitary(m, bs_loss, bs_jitter,
                             phase_noise_std, systematic_phase_offset, mode_loss)
        input_modes, interfering_count = sample_input_state(
            m, num_sources,
            input_loss, coupling_efficiency,
            detector_inefficiency, multi_photon_prob,
        )
        if interfering_count == 0:
            probabilities.append(0.0)
            continue

        output_modes = np.random.choice(m, interfering_count, replace=True)
        if use_advanced_nonlinearity:
            probability = boson_sampling_probability_advanced(
                U, input_modes, output_modes, effective_mu, m, chi
            )
        else:
            probability = boson_sampling_probability(
                U, input_modes, output_modes, effective_mu
            )

        probability = apply_dark_counts(probability, m, dark_count_rate)
        probabilities.append(probability)

    return float(np.mean(probabilities))


# ==============================================================================
# PyTorch modules wrapping the boson simulator
# ==============================================================================

class BosonSamplerTorch(nn.Module):
    def __init__(self, latent_dim: int, m: int, num_sources: int, num_loops: int,
                 input_loss: float, coupling_efficiency: float,
                 detector_inefficiency: float, multi_photon_prob: float,
                 mu: float, temporal_mismatch: float, spectral_mismatch: float,
                 arrival_time_jitter: float, bs_loss: float, bs_jitter: float,
                 phase_noise_std: float, systematic_phase_offset, mode_loss,
                 dark_count_rate: float, use_advanced_nonlinearity: bool = False,
                 chi: float = 0.0):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.m = m
        self.num_sources = num_sources
        self.num_loops = num_loops
        self.input_loss = input_loss
        self.coupling_efficiency = coupling_efficiency
        self.detector_inefficiency = detector_inefficiency
        self.multi_photon_prob = multi_photon_prob
        self.mu = mu
        self.temporal_mismatch = temporal_mismatch
        self.spectral_mismatch = spectral_mismatch
        self.arrival_time_jitter = arrival_time_jitter
        self.bs_loss = bs_loss
        self.bs_jitter = bs_jitter
        self.phase_noise_std = phase_noise_std
        self.dark_count_rate = dark_count_rate
        self.use_advanced_nonlinearity = use_advanced_nonlinearity
        self.chi = chi

        def _to_len_m(x, m, dtype=np.float32):
            arr = np.asarray(x, dtype=dtype)
            if arr.ndim == 0:
                arr = np.full((m,), float(arr), dtype=dtype)
            elif arr.size != m:
                arr = np.resize(arr, (m,))
            return arr

        self.register_buffer("mode_loss", torch.from_numpy(_to_len_m(mode_loss, m)).float())
        self.register_buffer("systematic_phase_offset", torch.from_numpy(_to_len_m(systematic_phase_offset, m)).float())

    def _one_latent(self):
        phase_offset_np = self.systematic_phase_offset.cpu().numpy()
        mode_loss_np = self.mode_loss.cpu().numpy()

        vals = []
        for _ in range(self.latent_dim):
            p = boson_sampling_simulation(
                self.m, self.num_sources, self.num_loops,
                self.input_loss, self.coupling_efficiency, self.detector_inefficiency,
                self.multi_photon_prob, self.mu, self.temporal_mismatch,
                self.spectral_mismatch, self.arrival_time_jitter,
                self.bs_loss, self.bs_jitter, self.phase_noise_std,
                phase_offset_np, mode_loss_np,
                self.dark_count_rate, self.use_advanced_nonlinearity, self.chi,
            )
            vals.append(np.log(p + 1e-8))

        z = np.asarray(vals, dtype=np.float32)
        z = (z - z.mean()) / (z.std() + 1e-6)
        return z

    def forward(self, batch_size: int):
        arr = np.stack([self._one_latent() for _ in range(batch_size)], axis=0)
        return torch.from_numpy(arr).to(self.systematic_phase_offset.device)


class BosonLatentGenerator(nn.Module):
    def __init__(self, latent_dim: int, boson_sampler_module: BosonSamplerTorch):
        super().__init__()
        self.boson_sampler_module = boson_sampler_module

    def forward(self, batch_size: int):
        return self.boson_sampler_module(batch_size)


def _normalize_boson_cfg(raw: dict, latent_dim: int) -> dict:
    """
    Normalize CLI/JSON boson params into full config for BosonSamplerTorch.
    """
    p = dict(raw or {})
    p.setdefault("latent_dim", latent_dim)

    # Required ints
    p.setdefault("m", latent_dim)
    p.setdefault("num_sources", max(1, latent_dim // 2))
    p.setdefault("num_loops", 1)
    ...

    # Float defaults
    for k, v in {
        "input_loss": 0.0,
        "coupling_efficiency": 1.0,
        "detector_inefficiency": 0.0,
        "multi_photon_prob": 0.0,
        "mu": 1.0,
        "temporal_mismatch": 0.0,
        "spectral_mismatch": 0.0,
        "arrival_time_jitter": 0.0,
        "bs_loss": 0.0,
        "bs_jitter": 0.0,
        "phase_noise_std": 0.0,
        "systematic_phase_offset": 0.0,
        "dark_count_rate": 0.0,
        "chi": 0.0,
    }.items():
        p.setdefault(k, v)

    # Bool
    p.setdefault("use_advanced_nonlinearity", False)

    # mode_loss: list -> np.ndarray of length m
    mode_loss = p.get("mode_loss")
    if mode_loss is None:
        mode_loss = [0.0] * int(p["m"])
    p["mode_loss"] = np.asarray(mode_loss, dtype=np.float32)

    return p

# ------------------------------------------
# New helpers
# ------------------------------------------

def _sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).t()
    return x2 + y2 - 2.0 * (x @ y.t())

def mmd_rbf(x: torch.Tensor, y: torch.Tensor, scales=(0.5, 1.0, 2.0, 4.0)) -> torch.Tensor:
    dxx = _sq_dist(x, x)
    dyy = _sq_dist(y, y)
    dxy = _sq_dist(x, y)

    kxx = 0.0
    kyy = 0.0
    kxy = 0.0
    for s in scales:
        gamma = 1.0 / (2.0 * s * s)
        kxx = kxx + torch.exp(-gamma * dxx)
        kyy = kyy + torch.exp(-gamma * dyy)
        kxy = kxy + torch.exp(-gamma * dxy)

    return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()


# ------------------------------------------
# FiLM helper
# ------------------------------------------
def _apply_film(feat: torch.Tensor, film_layer: nn.Module, c_embed: torch.Tensor) -> torch.Tensor:
    """
    FiLM: feat -> feat * (1 + gamma) + beta
    feat: (B, C, E)
    c_embed: (B, D)
    film_layer outputs (B, 2C) => gamma,beta each (B,C)
    """
    gb = film_layer(c_embed)                 # (B, 2C)
    gamma, beta = gb.chunk(2, dim=1)         # (B, C), (B, C)
    return feat * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

# ==============================================================================
# Mesh Encoder Helpers
# ==============================================================================

def faces_to_edge_index(faces: torch.Tensor | None, num_nodes: int, device=None) -> torch.Tensor:
    if device is None:
        if faces is not None:
            device = faces.device
        else:
            device = torch.device("cpu")

    if num_nodes <= 1:
        idx = torch.zeros(1, dtype=torch.long, device=device)
        return torch.stack([idx, idx], dim=0)

    if faces is None or faces.numel() == 0:
        idx = torch.arange(num_nodes, device=device)
        src = idx[:-1]
        dst = idx[1:]
        return torch.cat([
            torch.stack([src, dst], dim=0),
            torch.stack([dst, src], dim=0),
        ], dim=1)

    tri = faces.long().view(-1, 3)
    e = torch.cat([
        tri[:, [0, 1]], tri[:, [1, 0]],
        tri[:, [1, 2]], tri[:, [2, 1]],
        tri[:, [2, 0]], tri[:, [0, 2]],
    ], dim=0)

    valid = (
        (e[:, 0] >= 0) & (e[:, 1] >= 0) &
        (e[:, 0] < num_nodes) & (e[:, 1] < num_nodes)
    )
    e = e[valid]

    if e.numel() == 0:
        idx = torch.arange(num_nodes, device=device)
        src = idx[:-1]
        dst = idx[1:]
        return torch.cat([
            torch.stack([src, dst], dim=0),
            torch.stack([dst, src], dim=0),
        ], dim=1)

    e = torch.unique(e, dim=0)
    return e.t().contiguous()


def knn_edge_index(points: torch.Tensor, k: int) -> torch.Tensor:
    n = int(points.shape[0])
    if n <= 1:
        idx = torch.zeros(1, dtype=torch.long, device=points.device)
        return torch.stack([idx, idx], dim=0)

    k_eff = max(1, min(int(k), n - 1))
    with torch.no_grad():
        d = torch.cdist(points, points)
        d.fill_diagonal_(float("inf"))
        nbr = d.topk(k_eff, largest=False).indices

    dst = torch.arange(n, device=points.device).repeat_interleave(k_eff)
    src = nbr.reshape(-1)

    e = torch.stack([src, dst], dim=0)
    rev = torch.stack([dst, src], dim=0)
    e = torch.cat([e, rev], dim=1)
    e = torch.unique(e.t(), dim=0).t().contiguous()
    return e


def merge_edge_index(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a is None or a.numel() == 0:
        return b
    if b is None or b.numel() == 0:
        return a
    return torch.unique(torch.cat([a, b], dim=1).t(), dim=0).t().contiguous()


class MeshEdgeConvBlock(nn.Module):
    """
    Message passing over mesh edges:
      msg(i <- j) = MLP([x_i, x_j - x_i, p_j - p_i])
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 3, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
        self.self_lin = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        if src.numel() == 0:
            out = self.self_lin(x)
            return F.relu(self.norm(out))

        rel_x = x[src] - x[dst]
        rel_p = pos[src] - pos[dst]
        msg_in = torch.cat([x[dst], rel_x, rel_p], dim=1)
        msg = self.msg_mlp(msg_in)

        out = torch.zeros(x.shape[0], msg.shape[1], device=x.device, dtype=msg.dtype)
        out.index_add_(0, dst, msg)

        deg = torch.bincount(dst, minlength=x.shape[0]).to(msg.dtype).unsqueeze(1).clamp(min=1.0)
        out = out / deg
        out = out + self.self_lin(x)
        out = self.norm(out)
        return F.relu(out)

# ==============================================================================
# Mesh encoder / decoder  (encoder stays as before, decoder fixed)
# ==============================================================================

class MeshEncoder(nn.Module):
    """
    Real mesh encoder:
      - uses verts/faces/verts_mask/faces_mask from the dataset
      - face-adjacency message passing
      - optional kNN augmentation
      - global max + mean pooling

    Leave encoder conditioning OFF for scan-driven diagnosis.
    """
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        latent_dim: int,
        k: int = 5,
        opt=None,
        cond_dim: int = 0,
        cond_embed_dim: int = 16,
    ):
        super().__init__()
        assert len(channels) >= 1

        self.use_knn = bool(getattr(opt, "mesh_use_knn", False)) if opt is not None else False
        self.knn_k = int(getattr(opt, "mesh_knn_k", 8)) if opt is not None else 8
        self.knn_max_points = int(getattr(opt, "mesh_knn_max_points", 4096)) if opt is not None else 4096

        self.input_proj = nn.Linear(in_channels, channels[0])

        blocks = []
        c_prev = channels[0]
        for c_out in channels[1:]:
            blocks.append(MeshEdgeConvBlock(c_prev, c_out))
            c_prev = c_out
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = c_prev

        self.cond_dim = int(cond_dim)
        self.cond_embed_dim = int(cond_embed_dim) if self.cond_dim > 0 else 0
        self.cond_proj = nn.Linear(self.cond_dim, self.cond_embed_dim) if self.cond_dim > 0 else None
        self.cond_scale = float(getattr(opt, "cond_scale", 1.0)) if (self.cond_dim > 0 and opt is not None) else 1.0

        pooled_dim = 2 * self.out_dim + (self.cond_embed_dim if self.cond_dim > 0 else 0)
        self.latent_mu = nn.Linear(pooled_dim, latent_dim)
        self.latent_logsigma = nn.Linear(pooled_dim, latent_dim)

    def _encode_one(self, pos: torch.Tensor, faces: torch.Tensor | None) -> torch.Tensor:
        h = F.relu(self.input_proj(pos))

        edge_index = faces_to_edge_index(faces, num_nodes=pos.shape[0], device=pos.device)
        if self.use_knn and pos.shape[0] <= self.knn_max_points:
            edge_index = merge_edge_index(edge_index, knn_edge_index(pos, self.knn_k))

        for block in self.blocks:
            h = block(h, pos, edge_index)

        pooled = torch.cat([h.max(dim=0).values, h.mean(dim=0)], dim=0)
        return pooled

    def forward(self, x: torch.Tensor, mesh_batch: dict | None, cond: torch.Tensor | None = None):
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        assert x.dim() == 3, f"[ENC] expected [B,C,E], got {tuple(x.shape)}"

        pooled_list = []
        B = x.shape[0]

        for b in range(B):
            faces = None

            if isinstance(mesh_batch, dict) and mesh_batch.get("verts") is not None:
                V_pad = mesh_batch["verts"][b]
                V_mask = mesh_batch.get("verts_mask", None)

                if V_mask is not None:
                    pos = V_pad[V_mask[b].bool()].float()
                else:
                    pos = V_pad.float()

                F_pad = mesh_batch.get("faces", None)
                if F_pad is not None:
                    F_mask = mesh_batch.get("faces_mask", None)
                    if F_mask is not None:
                        faces = F_pad[b][F_mask[b].bool()].long()
                    else:
                        faces = F_pad[b].long()

                if pos.numel() == 0:
                    pos = x[b].permute(1, 0).contiguous().float()
                    faces = None
            else:
                pos = x[b].permute(1, 0).contiguous().float()

            pooled_list.append(self._encode_one(pos, faces))

        f_global = torch.stack(pooled_list, dim=0)

        if self.cond_proj is not None:
            if cond is None:
                raise ValueError("Model expects cond but got None.")
            c_embed = F.relu(self.cond_proj(cond.float())) * self.cond_scale
            f_global = torch.cat([f_global, c_embed], dim=1)

        mu = self.latent_mu(f_global)
        log_sigma = self.latent_logsigma(f_global)
        return mu, log_sigma


class MeshDecoder(nn.Module):
    """
    Decoder: (z, cond) -> per-edge xyz.
    If cond_dim>0:
      - embed cond
      - concat to z before FC (old behavior)
      - apply FiLM after FC and after every conv
    """
    def __init__(
        self,
        out_channels: int,
        channels: List[int],
        latent_dim: int,
        k: int = 5,
        opt=None,
        cond_dim: int = 0,
        cond_embed_dim: int = 16,
    ):
        super().__init__()
        self.E_cap = int(getattr(opt, "ninput_edges", 10000))

        self.cond_dim = int(cond_dim)
        self.cond_embed_dim = int(cond_embed_dim) if self.cond_dim > 0 else 0
        self.cond_proj = nn.Linear(self.cond_dim, self.cond_embed_dim) if self.cond_dim > 0 else None
        self.cond_scale = float(getattr(opt, "cond_scale", 1.0)) if (self.cond_dim > 0 and opt is not None) else 1.0

        c0 = channels[0]
        fc_in = latent_dim + (self.cond_embed_dim if self.cond_dim > 0 else 0)
        self.fc_z = nn.Linear(fc_in, c0 * self.E_cap)

        convs = []
        for c_in, c_out in zip(channels, channels[1:]):
            convs.append(nn.Conv1d(c_in, c_out, kernel_size=1, bias=True))
        self.convs = nn.ModuleList(convs)

        self.out_head = nn.Conv1d(channels[-1], out_channels, kernel_size=1, bias=True)

        # FiLM layers: one for fc output (channels[0]) + one per conv output (channels[1:])
        self.film = None
        if self.cond_dim > 0:
            self.film = nn.ModuleList([nn.Linear(self.cond_embed_dim, 2 * csz) for csz in channels])

    def forward(self, z: torch.Tensor, E: int, cond: torch.Tensor | None = None) -> torch.Tensor:
        B = z.shape[0]
        if E > self.E_cap:
            raise RuntimeError(f"[DEC] E={E} exceeds E_cap={self.E_cap}. Increase --ninput_edges.")

        c_embed = None
        if self.cond_proj is not None:
            if cond is None:
                raise ValueError("Model expects cond but got None.")
            c_embed = F.relu(self.cond_proj(cond.float())) * self.cond_scale  # (B, cond_embed_dim)
            z = torch.cat([z, c_embed], dim=1)

        f = self.fc_z(z).view(B, -1, self.E_cap)
        f = f[:, :, :E]

        # FiLM right after FC (channels[0])
        if (c_embed is not None) and (self.film is not None):
            f = _apply_film(f, self.film[0], c_embed)

        # conv stack (after conv i => channels[i+1], so FiLM index i+1)
        for i, conv in enumerate(self.convs):
            f = F.relu(conv(f))
            if (c_embed is not None) and (self.film is not None):
                f = _apply_film(f, self.film[i + 1], c_embed)

        return self.out_head(f)  # (B,3,E)


# ==============================================================================
# MeshVAE core
# ==============================================================================

class MeshVAE(nn.Module):
    """
    End-to-end VAE:
      - encoder = real mesh/graph encoder over verts+faces
      - decoder = keep existing point/vertex decoder
      - optional latent->plaque auxiliary head
    """
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        latent_dim: int,
        obs_log_std: float = math.log(0.1),
        k: int = 5,
        opt: Optional[object] = None,
    ):
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.boson_sampler_module = None
        self.boson_sampler = None

        self.latent_prior = getattr(opt, "latent_prior", "gaussian").lower()
        raw_boson_cfg = getattr(opt, "boson_cfg", None)
        self.boson_params = _normalize_boson_cfg(raw_boson_cfg, self.latent_dim) if raw_boson_cfg else None

        if self.latent_prior == "boson":
            if not _BOSON_AVAILABLE:
                raise ImportError("latent_prior='boson' requested but boson components are not available.")
            boson_module = BosonSamplerTorch(**self.boson_params)
            self.boson_sampler_module = boson_module
            self.boson_sampler = BosonLatentGenerator(self.latent_dim, boson_module)
            print("[VAE] Using BOSON latent prior.", flush=True)
        else:
            print("[VAE] Using GAUSSIAN latent prior.", flush=True)

        self.cond_dim = int(getattr(opt, "cond_dim", 6)) if getattr(opt, "use_cond", False) else 0
        self.cond_embed_dim = int(getattr(opt, "cond_embed_dim", 16))

        # keep metadata OUT of the encoder by default
        self.encoder_use_cond = bool(getattr(opt, "encoder_use_cond", False)) and (self.cond_dim > 0)
        enc_cond_dim = self.cond_dim if self.encoder_use_cond else 0

        self.encoder = MeshEncoder(
            in_channels, channels, self.latent_dim, k=k, opt=opt,
            cond_dim=enc_cond_dim, cond_embed_dim=self.cond_embed_dim
        )
        self.decoder = MeshDecoder(
            in_channels, channels, self.latent_dim, k=k, opt=opt,
            cond_dim=self.cond_dim, cond_embed_dim=self.cond_embed_dim
        )

        self.use_cond_prior = (
            bool(getattr(opt, "use_cond_prior", False))
            and (self.cond_dim > 0)
            and (self.latent_prior != "boson")
        )

        self.prior_net = None
        if self.use_cond_prior:
            self.prior_net = nn.Sequential(
                nn.Linear(self.cond_dim, self.cond_embed_dim),
                nn.ReLU(),
                nn.Linear(self.cond_embed_dim, 2 * self.latent_dim),
            )

        self.plaque_head = None
        if bool(getattr(opt, "aux_predict_plaque", False)):
            plaque_dim = len(getattr(opt, "aux_plaque_cols", [0, 0, 0, 0]))
            self.plaque_head = nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, plaque_dim),
            )

        self.aux_predict_athero = bool(getattr(opt, "aux_predict_athero", False))
        self.aux_athero_mode = str(getattr(opt, "aux_athero_mode", "any"))

        # cond vector layout:
        age_labels = list(getattr(opt, "age_labels", []))
        if len(age_labels) == 0:
            age_bins = list(getattr(opt, "age_bins", [18, 35, 50, 60, 200]))
            n_age = max(1, len(age_bins) - 1)
        else:
            n_age = len(age_labels)

        self.athero_offset = 2 + n_age
        self.cond_athero_enabled = bool(getattr(opt, "cond_athero", False))
        self.cond_athero_width = (
            1 if str(getattr(opt, "cond_athero_mode", "any")) == "any" else 4
        )

        # For binary diagnosis, use aux_athero_mode="any".
        self.aux_athero_dim = 4 if self.aux_athero_mode == "4bit" else 1

        self.athero_head = None
        if self.aux_predict_athero:
            self.athero_head = nn.Sequential(
                nn.Linear(self.latent_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(float(getattr(opt, "aux_athero_dropout", 0.15))),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(float(getattr(opt, "aux_athero_dropout", 0.15))),
                nn.Linear(128, self.aux_athero_dim),
            )

            if self.encoder_use_cond:
                print(
                    "[WARN] --aux_predict_athero is enabled together with "
                    "--encoder_use_cond. If cond contains atherosclerosis bits, "
                    "the classifier can leak labels through the encoder.",
                    flush=True,
                )

        self.obs_log_std = nn.Parameter(torch.tensor(float(obs_log_std)), requires_grad=False)

    @staticmethod
    def _diag_normal(mu: torch.Tensor, log_sigma: torch.Tensor) -> Independent:
        sigma = torch.exp(log_sigma)
        return Independent(Normal(mu, sigma), 1)

    def posterior(self, x: torch.Tensor, mesh_batch: dict | None, cond: torch.Tensor | None = None):
        mu, log_sigma = self.encoder(x, mesh_batch, cond=cond)
        log_sigma = log_sigma.clamp(min=-10.0, max=10.0)
        qz = self._diag_normal(mu, log_sigma)
        return qz, mu

    def prior(self, batch: int, latent_dim: int, device, cond: torch.Tensor | None = None):
        if self.latent_prior == "boson" and self.boson_sampler is not None:
            return BosonPrior(
                boson_sampler=self.boson_sampler,
                batch_size=batch,
                latent_features=latent_dim,
            )

        if self.use_cond_prior and (self.prior_net is not None) and (cond is not None):
            h = self.prior_net(cond.float().to(device))
            mu, log_sigma = h.chunk(2, dim=1)
            log_sigma = log_sigma.clamp(min=-6.0, max=2.0)
            return self._diag_normal(mu, log_sigma)

        mu = torch.zeros(batch, latent_dim, device=device)
        log_sigma = torch.zeros(batch, latent_dim, device=device)
        return self._diag_normal(mu, log_sigma)

    def sample_prior(self, batch: int, device, cond: torch.Tensor | None = None):
        if self.latent_prior == "boson" and self.boson_sampler is not None:
            return self.boson_sampler(batch).to(device)

        pz = self.prior(batch, self.latent_dim, device=device, cond=cond)
        return pz.rsample() if hasattr(pz, "rsample") else pz.sample()

    def athero_target_from_cond(self, cond: torch.Tensor | None) -> torch.Tensor | None:
        if cond is None:
            return None
        if not self.cond_athero_enabled:
            return None

        start = int(self.athero_offset)
        stop = start + int(self.cond_athero_width)
        if cond.shape[1] < stop:
            return None

        raw = cond[:, start:stop].float()

        if self.aux_athero_mode == "4bit":
            if raw.shape[1] == 4:
                return raw.clamp(0.0, 1.0)
            return raw.repeat(1, 4).clamp(0.0, 1.0)

        # Binary target: sick if any atherosclerosis bit is positive.
        return (raw.max(dim=1, keepdim=True).values > 0.5).float()

    def likelihood(self, x_mean: torch.Tensor) -> Independent:
        std = torch.exp(self.obs_log_std)
        return Independent(Normal(loc=x_mean, scale=std), reinterpreted_batch_ndims=2)

    def forward(self, x: torch.Tensor, mesh_batch: dict | None, cond: torch.Tensor | None = None):
        x_in = x.squeeze(-1) if (x.dim() == 4 and x.shape[-1] == 1) else x
        B, C, E = x_in.shape

        enc_cond = cond if self.encoder_use_cond else None
        dec_cond = cond if self.cond_dim > 0 else None

        qz, mu = self.posterior(x_in, mesh_batch, cond=enc_cond)
        z = qz.rsample() if self.training else mu

        x_mean = self.decoder(z, E, cond=dec_cond)
        if x_mean.dim() == 4 and x_mean.shape[-1] == 1:
            x_mean = x_mean.squeeze(-1)

        px = self.likelihood(x_mean)
        pz = self.prior(
            B,
            z.shape[-1],
            device=x.device,
            cond=(dec_cond if self.use_cond_prior else None),
        )

        out = {
            "px": px,
            "pz": pz,
            "qz": qz,
            "z": z,
            "x_mean": x_mean,
            "x_in": x_in,
        }

        if self.plaque_head is not None:
            out["plaque_pred"] = self.plaque_head(mu)

        if self.athero_head is not None:
            # Classifier uses mu, not z, so prediction is deterministic
            # and tied to the scan-derived latent location.
            out["athero_logits"] = self.athero_head(mu)

        return out

# ==============================================================================
# VI and training wrapper  (only MeshVAEModel init changes)
# ==============================================================================
def _subsample_points(P: torch.Tensor, n: int) -> torch.Tensor:
    """
    P: (B, N, 3)
    Returns: (B, n, 3) (with replacement if N < n)
    """
    B, N, _ = P.shape
    if N == n:
        return P
    if N > n:
        idx = torch.randint(0, N, (B, n), device=P.device)
        return P.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
    # N < n: sample with replacement
    idx = torch.randint(0, N, (B, n), device=P.device)
    return P.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))


def chamfer_distance_subsample(x: torch.Tensor, x_mean: torch.Tensor, n_points: int = 2048, chunk: int = 1024) -> torch.Tensor:
    """
    Order-invariant Chamfer distance between point clouds.
    x, x_mean: (B, 3, E)
    Returns: (B,) chamfer = mean_nn(x->x_mean) + mean_nn(x_mean->x)
    """
    assert x.dim() == 3 and x_mean.dim() == 3, f"expected (B,3,E), got {x.shape} and {x_mean.shape}"
    assert x.shape[1] == 3 and x_mean.shape[1] == 3, "Chamfer expects xyz channels (C=3)"

    # (B, E, 3)
    A = x.permute(0, 2, 1).contiguous()
    B = x_mean.permute(0, 2, 1).contiguous()

    # subsample to keep cdist reasonable
    A = _subsample_points(A, n_points)
    B = _subsample_points(B, n_points)

    # A -> B
    Bsz, Na, _ = A.shape
    Nb = B.shape[1]

    mins_ab = []
    for s in range(0, Na, chunk):
        e = min(s + chunk, Na)
        d = torch.cdist(A[:, s:e, :], B)          # (B, chunk, Nb)
        mins_ab.append(d.min(dim=2).values ** 2)
    d_ab = torch.cat(mins_ab, dim=1).mean(dim=1)  # (B,)

    # B -> A
    mins_ba = []
    for s in range(0, Nb, chunk):
        e = min(s + chunk, Nb)
        d = torch.cdist(B[:, s:e, :], A)          # (B, chunk, Na)
        mins_ba.append(d.min(dim=2).values ** 2)
    d_ba = torch.cat(mins_ba, dim=1).mean(dim=1)  # (B,)

    return d_ab + d_ba

def axis_std_loss(x: torch.Tensor, x_mean: torch.Tensor) -> torch.Tensor:
    """
    Penalize collapse by matching per-axis spread (std) between target and recon.
    x, x_mean: (B, 3, E)
    returns: (B,)
    """
    A = x.permute(0, 2, 1)       # (B, E, 3)
    B = x_mean.permute(0, 2, 1)  # (B, E, 3)
    std_A = A.std(dim=1)         # (B, 3)
    std_B = B.std(dim=1)         # (B, 3)
    return ((std_A - std_B) ** 2).mean(dim=1)  # (B,)

def repulsion_loss(x_mean: torch.Tensor, n_points: int = 1024, k: int = 8, radius: float = 0.02) -> torch.Tensor:
    """
    Encourage points to not collapse into a line/cluster by penalizing too-close neighbors.
    x_mean: (B, 3, E)
    returns: (B,)
    """
    P = x_mean.permute(0, 2, 1).contiguous()   # (B, E, 3)
    P = _subsample_points(P, n_points)         # (B, n, 3)

    # Pairwise distances
    D = torch.cdist(P, P)                      # (B, n, n)
    # Ignore self-distance
    n = D.shape[-1]
    eye = torch.eye(n, device=D.device).unsqueeze(0)
    D = D + eye * 1e6

    # kNN distances (smallest k)
    knn = D.topk(k, dim=-1, largest=False).values  # (B, n, k)

    # Penalize distances smaller than radius
    pen = F.relu(radius - knn) ** 2                # (B, n, k)
    return pen.mean(dim=(1, 2))                    # (B,)

class MeshVariationalInference(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        use_chamfer: bool = False,
        chamfer_npoints: int = 2048,
        w_spread: float = 0.2,
        w_rep: float = 0.05,
        rep_k: int = 8,
        rep_radius: float = 0.02,
        w_plaque_aux: float = 0.0,
        w_athero_aux: float = 0.0,
        athero_pos_weight: float = 1.0,
    ):
        super().__init__()
        self.beta = float(beta)
        self.use_chamfer = bool(use_chamfer)
        self.chamfer_npoints = int(chamfer_npoints)

        self.w_spread = float(w_spread)
        self.w_rep = float(w_rep)
        self.rep_k = int(rep_k)
        self.rep_radius = float(rep_radius)

        self.w_plaque_aux = float(w_plaque_aux)
        self.w_athero_aux = float(w_athero_aux)
        self.athero_pos_weight = float(athero_pos_weight)

    def forward(
        self,
        model: "MeshVAE",
        x: torch.Tensor,
        mesh_batch: dict | None,
        cond: torch.Tensor | None = None,
        plaque_target: torch.Tensor | None = None,
    ):
        out = model(x, mesh_batch, cond=cond)
        px, pz, qz, z = out["px"], out["pz"], out["qz"], out["z"]

        x_in = out["x_in"]
        x_mean = out["x_mean"]

        if self.use_chamfer:
            recon = chamfer_distance_subsample(
                x_in, x_mean, n_points=self.chamfer_npoints
            )

            spread = axis_std_loss(x_in, x_mean)

            rep = repulsion_loss(
                x_mean,
                n_points=min(self.chamfer_npoints, 1024),
                k=self.rep_k,
                radius=self.rep_radius,
            )

            total_recon = recon + self.w_spread * spread + self.w_rep * rep
            log_px = -total_recon
        else:
            recon = None
            spread = None
            rep = None
            log_px = px.log_prob(x_in)

        if getattr(model, "latent_prior", "gaussian") == "boson":
            z_prior = model.sample_prior(z.shape[0], x.device, cond=cond)
            prior_match = mmd_rbf(z, z_prior)
            elbo = log_px - self.beta * prior_match
            loss = -elbo.mean()

            diag = {
                "elbo": elbo.detach(),
                "log_px": log_px.detach(),
                "prior_match": prior_match.detach().expand_as(elbo),
            }
        else:
            log_pz = pz.log_prob(z)
            log_qz = qz.log_prob(z)
            kl = log_qz - log_pz
            elbo = log_px - self.beta * kl
            loss = -elbo.mean()

            diag = {
                "elbo": elbo.detach(),
                "log_px": log_px.detach(),
                "kl": kl.detach(),
            }

        if recon is not None:
            diag["chamfer"] = recon.detach()
            diag["spread"] = spread.detach()
            diag["repulsion"] = rep.detach()

        if (plaque_target is not None) and (out.get("plaque_pred") is not None):
            mask = torch.isfinite(plaque_target)
            target = torch.nan_to_num(plaque_target, nan=0.0)

            per_elem = F.smooth_l1_loss(out["plaque_pred"], target, reduction="none")
            per_sample = (per_elem * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp(min=1.0)

            loss = loss + self.w_plaque_aux * per_sample.mean()
            diag["plaque_loss"] = per_sample.detach()

        if self.w_athero_aux > 0.0 and out.get("athero_logits") is not None:
            target = model.athero_target_from_cond(cond)
            if target is not None:
                logits = out["athero_logits"]

                # Broadcast/trim target if needed.
                if target.shape != logits.shape:
                    if logits.shape[1] == 1:
                        target = (target.max(dim=1, keepdim=True).values > 0.5).float()
                    elif target.shape[1] == 1 and logits.shape[1] == 4:
                        target = target.repeat(1, 4)

                pos_weight = None
                if self.athero_pos_weight > 0 and not math.isclose(self.athero_pos_weight, 1.0):
                    pos_weight = torch.full(
                        (logits.shape[1],),
                        float(self.athero_pos_weight),
                        device=logits.device,
                        dtype=logits.dtype,
                    )

                per_elem = F.binary_cross_entropy_with_logits(
                    logits,
                    target.float(),
                    reduction="none",
                    pos_weight=pos_weight,
                )
                per_sample = per_elem.mean(dim=1)

                loss = loss + self.w_athero_aux * per_sample.mean()
                diag["athero_loss"] = per_sample.detach()

                with torch.no_grad():
                    prob = torch.sigmoid(logits)
                    if logits.shape[1] == 1:
                        pred_any = (prob[:, 0] >= 0.5).long()
                        true_any = (target[:, 0] >= 0.5).long()
                        diag["athero_acc"] = (pred_any == true_any).float().detach()
                        diag["athero_prob_sick"] = prob[:, 0].detach()
                    else:
                        pred_bits = (prob >= 0.5).long()
                        true_bits = (target >= 0.5).long()
                        diag["athero_bit_acc"] = (pred_bits == true_bits).float().mean(dim=1).detach()
                        pred_any = pred_bits.any(dim=1).long()
                        true_any = true_bits.any(dim=1).long()
                        diag["athero_acc"] = (pred_any == true_any).float().detach()
                        diag["athero_prob_sick"] = prob.max(dim=1).values.detach()

        return loss, diag, out


def get_scheduler(optimizer, opt):
    from torch.optim import lr_scheduler
    if opt.lr_policy == "lambda":
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(
                opt.niter_decay + 1
            )
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1
        )
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2,
            threshold=0.01, patience=5,
        )
    else:
        raise NotImplementedError(f"learning rate policy [{opt.lr_policy}] is not implemented")
    return scheduler


class MeshVAEModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gpu_ids = getattr(opt, "gpu_ids", [])
        self.is_train = getattr(opt, "is_train", True)

        self.register_buffer("_devref", torch.empty(0))
        device_target = torch.device(
            f"cuda:{self.gpu_ids[0]}"
            if self.gpu_ids and torch.cuda.is_available()
            else "cpu"
        )
        self._devref = self._devref.to(device_target)

        self.net = MeshVAE(
            in_channels=opt.input_nc,
            channels=list(opt.ncf),
            latent_dim=int(getattr(opt, "latent_dim", 64)),
            k=5,
            opt=opt,
        ).to(self.device)

        init_type = getattr(opt, "init_type", "xavier")
        init_gain = getattr(opt, "init_gain", 0.02)
        for m in self.net.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                if init_type == "xavier":
                    nn.init.xavier_normal_(m.weight, gain=init_gain)
                elif init_type == "normal":
                    nn.init.normal_(m.weight, 0.0, init_gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=init_gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        if isinstance(self.gpu_ids, (list, tuple)) and len(self.gpu_ids) > 1:
            self.net = nn.DataParallel(self.net, device_ids=self.gpu_ids)

        self.net.train(self.is_train)

        beta = float(getattr(opt, "beta", 1.0))
        use_chamfer = bool(getattr(opt, "use_pointcloud", False))
        chamfer_n = int(getattr(opt, "chamfer_npoints", 2048))
        aux_plaque_weight = (
            float(getattr(opt, "aux_plaque_weight", 0.0))
            if getattr(opt, "aux_predict_plaque", False)
            else 0.0
        )
        aux_athero_weight = (
            float(getattr(opt, "aux_athero_weight", 0.0))
            if getattr(opt, "aux_predict_athero", False)
            else 0.0
        )

        self.vi = MeshVariationalInference(
            beta=beta,
            use_chamfer=use_chamfer,
            chamfer_npoints=chamfer_n,
            w_plaque_aux=aux_plaque_weight,
            w_athero_aux=aux_athero_weight,
            athero_pos_weight=float(getattr(opt, "aux_athero_pos_weight", 1.0)),
        ).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.scheduler = get_scheduler(self.optimizer, opt)

        self.edge_features = None
        self.mesh = None
        self.mesh_batch = None
        self.cond = None
        self.plaque_target = None
        self.loss = torch.tensor(0.0, device=self.device)
        self.last_diag = {}
        self.task = "vae"

    @property
    def device(self):
        return self._devref.device

    def update_learning_rate(self) -> None:
        if not getattr(self, "is_train", False):
            return
        if not hasattr(self, "scheduler") or self.scheduler is None:
            print("[WARN] update_learning_rate called but no scheduler is set.", flush=True)
            return
        old_lr = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]["lr"]
        if lr != old_lr:
            print(f"learning rate: {old_lr:.7f} -> {lr:.7f}", flush=True)
        else:
            print(f"learning rate = {lr:.7f}", flush=True)

    def set_input(self, data: dict):
        def _to_dev(v, dtype=None):
            if v is None:
                return None
            if not isinstance(v, torch.Tensor):
                v = torch.from_numpy(v)
            if dtype is not None:
                v = v.to(dtype=dtype)
            return v.to(self.device)

        fe = data["edge_features"]
        if not isinstance(fe, torch.Tensor):
            fe = torch.from_numpy(fe).float()

        if fe.dim() == 4 and fe.shape[-1] == 1:
            fe = fe.squeeze(-1)
        elif fe.dim() != 3:
            raise ValueError(f"Unexpected edge_features shape {tuple(fe.shape)}")

        self.edge_features = fe.to(self.device)

        c = data.get("cond", None)
        if c is not None:
            if not isinstance(c, torch.Tensor):
                c = torch.from_numpy(c).float()
            c = c.to(self.device)
            if c.dim() == 1:
                c = c.unsqueeze(0)
            self.cond = c
        else:
            self.cond = None

        self.plaque_target = _to_dev(data.get("plaque_target", None), dtype=torch.float32)

        verts = _to_dev(data.get("verts", None), dtype=torch.float32)
        faces = _to_dev(data.get("faces", None), dtype=torch.long)
        verts_mask = _to_dev(data.get("verts_mask", None), dtype=torch.bool)
        faces_mask = _to_dev(data.get("faces_mask", None), dtype=torch.bool)

        self.mesh_batch = {
            "verts": verts,
            "faces": faces,
            "verts_mask": verts_mask,
            "faces_mask": faces_mask,
        }

        self.mesh = data.get("mesh", None)

        if not torch.isfinite(self.edge_features).all():
            self.edge_features = torch.nan_to_num(
                self.edge_features, nan=0.0, posinf=0.0, neginf=0.0
            )

    def forward(self):
        return self.net(self.edge_features, self.mesh_batch, cond=self.cond)

    def backward(self, outputs: dict):
        loss, _, _ = self.vi(
            self.net,
            self.edge_features,
            self.mesh_batch,
            self.cond,
            plaque_target=self.plaque_target,
        )
        self.loss = loss
        self.loss.backward()

    def optimize_parameters(self):
        self.net.train(True)
        self.optimizer.zero_grad(set_to_none=True)

        use_amp = bool(getattr(self.opt, "use_amp", False) and torch.cuda.is_available())

        if use_amp:
            if not hasattr(self, "scaler"):
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler(enabled=True)

            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                loss, diagnostics, _ = self.vi(
                    self.net,
                    self.edge_features,
                    self.mesh_batch,
                    self.cond,
                    plaque_target=self.plaque_target,
                )

            if not torch.isfinite(loss):
                print("[ERR] non-finite loss; skipping batch:", float(loss))
                return

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.loss = loss.detach()
            self.last_diag = {k: float(v.mean().detach().cpu()) for k, v in diagnostics.items()}
            self.last_diag["loss"] = float(self.loss.detach().cpu())
            self.last_diag["lr"] = float(self.optimizer.param_groups[0]["lr"])
        else:
            loss, diagnostics, _ = self.vi(
                self.net,
                self.edge_features,
                self.mesh_batch,
                self.cond,
                plaque_target=self.plaque_target,
            )

            if not torch.isfinite(loss):
                print("[ERR] non-finite loss; skipping batch:", float(loss))
                return

            self.loss = loss
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.last_diag = {k: float(v.mean().detach().cpu()) for k, v in diagnostics.items()}
            self.last_diag["loss"] = float(self.loss.detach().cpu())
            self.last_diag["lr"] = float(self.optimizer.param_groups[0]["lr"])

    @torch.no_grad()
    def test(self):
        self.net.eval()
        loss, diagnostics, outputs = self.vi(
            self.net,
            self.edge_features,
            self.mesh_batch,
            self.cond,
            plaque_target=self.plaque_target,
        )
        self.last_diag = {k: float(v.mean().detach().cpu()) for k, v in diagnostics.items()}
        self.last_diag["loss"] = float(loss.detach().cpu())

        if hasattr(self, "is_train") and self.is_train:
            self.net.train()

        return {
            "loss": loss.detach().item(),
            "diagnostics": diagnostics,
            "nexamples": int(self.edge_features.shape[0]),
            "x": self.edge_features.detach().cpu(),
            "x_mean": (
                outputs.get("x_mean").detach().cpu()
                if isinstance(outputs, dict) and outputs.get("x_mean") is not None
                else None
            ),
            "plaque_pred": (
                outputs.get("plaque_pred").detach().cpu()
                if isinstance(outputs, dict) and outputs.get("plaque_pred") is not None
                else None
            ),
            "athero_logits": (
                outputs.get("athero_logits").detach().cpu()
                if isinstance(outputs, dict) and outputs.get("athero_logits") is not None
                else None
            ),
            "athero_prob": (
                torch.sigmoid(outputs.get("athero_logits")).detach().cpu()
                if isinstance(outputs, dict) and outputs.get("athero_logits") is not None
                else None
            ),
        }

    def save_network(self, which_epoch: str):
        fname = f"{which_epoch}_vae_net.pth"
        path = os.path.join(self.opt.checkpoints_dir, self.opt.name, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.gpu_ids and torch.cuda.is_available():
            state = (
                self.net.module.cpu().state_dict()
                if isinstance(self.net, nn.DataParallel)
                else self.net.cpu().state_dict()
            )
            torch.save(state, path)
            self.net.to(self.device)
        else:
            torch.save(self.net.state_dict(), path)

    def load_network(self, which_epoch: str):
        fname = f"{which_epoch}_vae_net.pth"
        path = os.path.join(self.opt.checkpoints_dir, self.opt.name, fname)
        state = torch.load(path, map_location=str(self.device))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(state)
        else:
            self.net.load_state_dict(state)
