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

# ---- Mesh conv backbone ------------------------------------------------------
# Adjust the import if your project layout is different (e.g. remove 'models.')
from mesh_conv_model import MeshConv, MeshPool, MeshUnpool


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
    """
    Generate a random "unitary-like" matrix with losses, jitter, and phase noise.
    """
    random_matrix = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    Q, _ = np.linalg.qr(random_matrix)

    # global loss & jitter
    Q *= np.sqrt(bs_loss)
    Q += bs_jitter * np.random.randn(m, m)

    # per-element phase noise
    if phase_noise_std != 0.0:
        phase_noise = np.exp(1j * np.random.normal(0, phase_noise_std, size=(m, m)))
        Q *= phase_noise

    # systematic phase offset (scalar or array-like); always apply, zero is no-op
    phase = np.asarray(systematic_phase_offset, dtype=np.float32)
    Q *= np.exp(1j * phase)

    # optional per-mode loss
    if mode_loss is not None:
        ml = np.asarray(mode_loss, dtype=np.float32).reshape(m)
        Q = np.array([Q[i, :] * np.sqrt(ml[i]) for i in range(m)])

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
    p_eff = input_loss * coupling_efficiency * detector_inefficiency

    for _ in range(num_sources):
        if np.random.rand() < p_eff:
            n_emit = 2 if np.random.rand() < multi_photon_prob else 1
            total_photons += n_emit
            modes = np.random.choice(m, n_emit, replace=True)
            interfering_input_modes.extend(modes.tolist())

    interfering_input_modes = np.array(interfering_input_modes)
    return interfering_input_modes, total_photons


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
    """
    Wraps the NumPy boson_sampling_simulation in a PyTorch module.
    Returns a scalar probability expanded to (batch_size,).
    """
    def __init__(
        self,
        m: int,
        num_sources: int,
        num_loops: int,
        input_loss: float,
        coupling_efficiency: float,
        detector_inefficiency: float,
        multi_photon_prob: float,
        mu: float,
        temporal_mismatch: float,
        spectral_mismatch: float,
        arrival_time_jitter: float,
        bs_loss: float,
        bs_jitter: float,
        phase_noise_std: float,
        systematic_phase_offset,
        mode_loss,
        dark_count_rate: float,
        use_advanced_nonlinearity: bool = False,
        chi: float = 0.0,
    ):
        super().__init__()

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

        # helper: coerce any scalar/array to length-m numpy array
        def _to_len_m(x, m, dtype=np.float32):
            arr = np.asarray(x, dtype=dtype)
            if arr.ndim == 0:
                arr = np.full((m,), float(arr), dtype=dtype)
            elif arr.size != m:
                arr = np.resize(arr, (m,))
            return arr

        # mode_loss: Parameter tensor of shape (m,)
        self.mode_loss = torch.from_numpy(_to_len_m(mode_loss, m)).float()

        # systematic_phase_offset: Parameter of shape (m,)
        sys_phase_np = _to_len_m(systematic_phase_offset, m)
        self.systematic_phase_offset = nn.Parameter(
            torch.from_numpy(sys_phase_np).float()
        )

        self.dark_count_rate = dark_count_rate
        self.use_advanced_nonlinearity = use_advanced_nonlinearity
        self.chi = chi

        print(
            f"[BOSON] init m={self.m}  mode_loss={tuple(self.mode_loss.shape)}  "
            f"sys_phase_off={tuple(self.systematic_phase_offset.shape)}",
            flush=True,
        )

    def forward(self, batch_size: int):
        # Use current trainable parameters as NumPy arrays
        phase_offset_np = self.systematic_phase_offset.detach().cpu().numpy()
        mode_loss_np = self.mode_loss.detach().cpu().numpy()

        avg_probability = boson_sampling_simulation(
            self.m, self.num_sources, self.num_loops,
            self.input_loss, self.coupling_efficiency, self.detector_inefficiency,
            self.multi_photon_prob, self.mu, self.temporal_mismatch,
            self.spectral_mismatch, self.arrival_time_jitter,
            self.bs_loss, self.bs_jitter, self.phase_noise_std,
            phase_offset_np, mode_loss_np,
            self.dark_count_rate, self.use_advanced_nonlinearity, self.chi,
        )

        result = torch.tensor(
            avg_probability,
            dtype=torch.float32,
            device=self.systematic_phase_offset.device,
        )
        return result.expand(batch_size)


class BosonLatentGenerator(nn.Module):
    """
    Wrap BosonSamplerTorch (scalar output) into a latent vector of size latent_dim
    by simple replication.
    """
    def __init__(self, latent_dim: int, boson_sampler_module: BosonSamplerTorch):
        super().__init__()
        self.latent_dim = latent_dim
        self.boson_sampler_module = boson_sampler_module

    def forward(self, batch_size: int):
        latent_scalar = self.boson_sampler_module(batch_size)  # (batch,)
        latent_vector = latent_scalar.unsqueeze(1).repeat(1, self.latent_dim)
        return latent_vector


def _normalize_boson_cfg(raw: dict, latent_dim: int) -> dict:
    """
    Normalize CLI/JSON boson params into full config for BosonSamplerTorch.
    """
    p = dict(raw or {})

    # Required ints
    p.setdefault("m", latent_dim)
    p.setdefault("num_sources", max(1, latent_dim // 2))
    p.setdefault("num_loops", 1)

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


# ==============================================================================
# Mesh encoder / decoder
# ==============================================================================

class MeshEncoder(nn.Module):
    """
    Mesh encoder: MeshConv -> Pool ... -> global average -> (mu, log_sigma).

    Inputs:
        x:     [B, C, E]   (or [B, C, E, 1]; we squeeze the trailing 1)
        mesh:  list of B mesh objects (mutated in-place by pooling)
    Returns:
        mu:         [B, Z]
        log_sigma:  [B, Z]
        start_E:    int (edges after last pooling)
    """
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        pool_targets: List[int],
        latent_dim: int,
        k: int = 5,
        opt=None,
    ):
        super().__init__()
        assert len(channels) >= 1 and len(pool_targets) >= 1
        assert len(pool_targets) <= len(channels)

        convs, pools = [], []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            convs.append(MeshConv(c_in, c_out, k=k, bias=True))
            if i < len(pool_targets) and (opt is None or getattr(opt, "pool_mode", "target") != "skip"):
                pools.append(
                    MeshPool(
                        target=int(pool_targets[i]),
                        multi_thread=False,
                        mode=getattr(opt, "pool_mode", "target"),
                        keep_frac=float(getattr(opt, "pool_keep", 1.0)),
                    )
                )
            c_in = c_out

        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)
        self.latent_mu = nn.Linear(channels[-1], latent_dim)
        self.latent_logsigma = nn.Linear(channels[-1], latent_dim)
        self._config_start_E = int(pool_targets[-1])

    def forward(self, x: torch.Tensor, mesh: list) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # normalize to [B, C, E]
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        assert x.dim() == 3, f"[ENC] expected [B, C, E], got {tuple(x.shape)}"

        f = x
        for i, conv in enumerate(self.convs):
            f = F.relu(conv(f, mesh))
            if f.dim() == 4 and f.shape[-1] == 1:
                f = f.squeeze(-1)

            if i < len(self.pools):
                f = self.pools[i](f, mesh)
                if f.dim() == 4 and f.shape[-1] == 1:
                    f = f.squeeze(-1)

        if f.dim() == 4 and f.shape[-1] == 1:
            f = f.squeeze(-1)

        start_E = int(f.shape[-1])
        f_global = f.mean(dim=-1)

        assert f_global.dim() == 2
        assert f_global.shape[1] == self.latent_mu.in_features

        mu = self.latent_mu(f_global)
        log_sigma = self.latent_logsigma(f_global)
        return mu, log_sigma, start_E


class MeshDecoder(nn.Module):
    """
    Decoder: latent -> features on edges via MeshUnpool / MeshConv.

    We start from a small resolution and unpool back up to original edges.
    """
    def __init__(
        self,
        out_channels: int,
        channels: List[int],
        pool_targets: List[int],
        latent_dim: int,
        k: int = 5,
        opt=None,
    ):
        super().__init__()

        # capacity for E dimension – must be >= max start_E; usually opt.ninput_edges
        E_cap = int(getattr(opt, "ninput_edges", max(pool_targets) if len(pool_targets) else 600))
        self.max_E_cap = E_cap
        self.start_edges = E_cap

        c0 = channels[-1]
        self.fc = nn.Linear(latent_dim, c0 * self.max_E_cap)

        rev_targets = list(reversed(pool_targets))
        unpools, deconvs = [], []
        cur_channels = c0
        for up_target, next_c in zip(rev_targets, reversed(channels[:-1])):
            unpools.append(MeshUnpool(unroll_target=int(up_target)))
            deconvs.append(MeshConv(cur_channels, next_c, k=k, bias=True))
            cur_channels = next_c

        # Final overshoot unpool: last hop uses original E0 via unroll_target override
        unpools.append(MeshUnpool(unroll_target=10 ** 9))
        deconvs.append(MeshConv(cur_channels, cur_channels, k=k, bias=True))

        self.unpools = nn.ModuleList(unpools)
        self.deconvs = nn.ModuleList(deconvs)
        self.out_head = MeshConv(cur_channels, out_channels, k=k, bias=True)

    def forward(self, z: torch.Tensor, mesh: list, start_E: int) -> torch.Tensor:
        B = z.shape[0]
        assert start_E > 0, "[DEC] start_E must be > 0"
        if start_E > self.max_E_cap:
            raise RuntimeError(
                f"[DEC] start_E={start_E} > max_E_cap={self.max_E_cap} "
                f"(set --ninp_edges to >= {start_E})"
            )

        f = self.fc(z).view(B, -1, self.max_E_cap)[:, :, :start_E]

        for unpool, deconv in zip(self.unpools, self.deconvs):
            f = unpool(f, mesh)
            f = F.relu(deconv(f, mesh))

        x_hat = self.out_head(f, mesh)
        return x_hat


# ==============================================================================
# MeshVAE core
# ==============================================================================

class MeshVAE(nn.Module):
    """
    End-to-end VAE on meshes. Learns p(x|z) as a factorized Normal over edge features.

    Prior:
        - Gaussian (standard Normal)
        - Boson (via BosonPrior + BosonLatentGenerator)
    """
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        pool_targets: List[int],
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
                raise ImportError(
                    "latent_prior='boson' requested but boson components are not available."
                )
            boson_module = BosonSamplerTorch(**self.boson_params)
            self.boson_sampler_module = boson_module
            self.boson_sampler = BosonLatentGenerator(self.latent_dim, boson_module)
            print("[VAE] Using BOSON latent prior.", flush=True)
        else:
            print("[VAE] Using GAUSSIAN latent prior.", flush=True)

        self.encoder = MeshEncoder(in_channels, channels, pool_targets, self.latent_dim, k=k, opt=opt)
        self.decoder = MeshDecoder(in_channels, channels, pool_targets, self.latent_dim, k=k, opt=opt)

        self.obs_log_std = nn.Parameter(torch.tensor(float(obs_log_std)), requires_grad=False)

    @staticmethod
    def _diag_normal(mu: torch.Tensor, log_sigma: torch.Tensor) -> Independent:
        sigma = torch.exp(log_sigma)
        return Independent(Normal(mu, sigma), 1)

    def posterior(self, x: torch.Tensor, mesh: list):
        mu, log_sigma, start_E = self.encoder(x, mesh)
        log_sigma = log_sigma.clamp(min=-10.0, max=10.0)
        qz = self._diag_normal(mu, log_sigma)
        return qz, mu, start_E

    def prior(self, batch: int, latent_dim: int, device):
        if self.latent_prior == "boson" and self.boson_sampler is not None:
            return BosonPrior(
                boson_sampler=self.boson_sampler,
                batch_size=batch,
                latent_features=latent_dim,
            )
        else:
            mu = torch.zeros(batch, latent_dim, device=device)
            log_sigma = torch.zeros(batch, latent_dim, device=device)
            return self._diag_normal(mu, log_sigma)

    def likelihood(self, x_mean: torch.Tensor) -> Independent:
        std = torch.exp(self.obs_log_std)
        return Independent(Normal(loc=x_mean, scale=std), reinterpreted_batch_ndims=2)

    def forward(self, x: torch.Tensor, mesh: list):
        if x.dim() == 4 and x.shape[-1] == 1:
            x_in = x.squeeze(-1)
        else:
            x_in = x
        B, C, E0 = x_in.shape

        qz, mu, start_E = self.posterior(x_in, mesh)
        z = qz.rsample()

        self.decoder.start_edges = int(start_E)
        if hasattr(self.decoder, "unpools") and len(self.decoder.unpools) > 0:
            self.decoder.unpools[-1].unroll_target = int(E0)

        x_mean = self.decoder(z, mesh, start_E)
        if x_mean.dim() == 4 and x_mean.shape[-1] == 1:
            x_mean = x_mean.squeeze(-1)

        px = self.likelihood(x_mean)
        pz = self.prior(B, z.shape[-1], device=x.device)

        return {
            "px": px,
            "pz": pz,
            "qz": qz,
            "z": z,
            "x_mean": x_mean,
            "x_in": x_in,
        }


# ==============================================================================
# VI and training wrapper
# ==============================================================================

class MeshVariationalInference(nn.Module):
    """
    ELBO computation for MeshVAE with optional beta scaling of KL.
    """
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = float(beta)

    def forward(self, model: MeshVAE, x: torch.Tensor, mesh: list):
        out = model(x, mesh)
        px, pz, qz, z = out["px"], out["pz"], out["qz"], out["z"]

        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        log_px = px.log_prob(x)   # [B]
        log_pz = pz.log_prob(z)   # [B]
        log_qz = qz.log_prob(z)   # [B]

        kl = log_qz - log_pz
        elbo = log_px - self.beta * kl
        loss = -elbo.mean()

        diag = {
            "elbo": elbo.detach(),
            "log_px": log_px.detach(),
            "kl": kl.detach(),
        }
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
    """
    High-level wrapper used by your training script.

    Exposes:
        - set_input(data)
        - optimize_parameters()
        - test()
        - update_learning_rate()
        - save_network(), load_network()
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gpu_ids = getattr(opt, "gpu_ids", [])
        self.is_train = getattr(opt, "is_train", True)

        # device reference buffer
        self.register_buffer("_devref", torch.empty(0))
        device_target = torch.device(
            f"cuda:{self.gpu_ids[0]}"
            if self.gpu_ids and torch.cuda.is_available()
            else "cpu"
        )
        self._devref = self._devref.to(device_target)

        # build the VAE
        self.net = MeshVAE(
            in_channels=opt.input_nc,
            channels=list(opt.ncf),
            pool_targets=list(opt.pool_res),
            latent_dim=int(getattr(opt, "latent_dim", 64)),
            k=5,
            opt=opt,
        ).to(self.device)

        # init weights
        init_type = getattr(opt, "init_type", "xavier")
        init_gain = getattr(opt, "init_gain", 0.02)
        for m in self.net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
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
        self.vi = MeshVariationalInference(beta=beta).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.scheduler = get_scheduler(self.optimizer, opt)

        self.edge_features = None
        self.mesh = None
        self.loss = torch.tensor(0.0, device=self.device)
        self.task = "vae"

        if torch.cuda.is_available():
            main_dev = (
                self.net.device_ids[0]
                if isinstance(self.net, nn.DataParallel)
                else self.device
            )
            print(
                f"[Init] MeshVAEModel initialized on {main_dev}. "
                f"Model parameters on {next(self.net.parameters()).device}.",
                flush=True,
            )

    @property
    def device(self):
        return self._devref.device

    # ------------------------------------------------------------------ LR
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

    # ------------------------------------------------------------------ I/O
    def set_input(self, data: dict):
        fe = data["edge_features"]
        if not isinstance(fe, torch.Tensor):
            fe = torch.from_numpy(fe).float()

        if fe.dim() == 4 and fe.shape[-1] == 1:
            fe = fe.squeeze(-1)
        elif fe.dim() != 3:
            raise ValueError(f"Unexpected edge_features shape {tuple(fe.shape)}")

        dev = next(self.parameters()).device
        self.edge_features = fe.to(dev)
        self.mesh = data["mesh"]

        if not torch.isfinite(self.edge_features).all():
            self.edge_features = torch.nan_to_num(
                self.edge_features, nan=0.0, posinf=0.0, neginf=0.0
            )

        if torch.cuda.is_available():
            cur_dev = self.edge_features.device
            mem_alloc = torch.cuda.memory_allocated(cur_dev) / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved(cur_dev) / (1024 ** 2)
            print(
                f"[Debug] edge_features device: {cur_dev}, "
                f"GPU memory allocated: {mem_alloc:.1f} MB (reserved {mem_reserved:.1f} MB)",
                flush=True,
            )

    # ------------------------------------------------------------------ train
    def forward(self):
        return self.net(self.edge_features, self.mesh)

    def backward(self, outputs: dict):
        loss, _, _ = self.vi(self.net, self.edge_features, self.mesh)
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
            with autocast(device_type="cuda", dtype=torch.float16):
                loss, diagnostics, _ = self.vi(self.net, self.edge_features, self.mesh)

            if not torch.isfinite(loss):
                print("[ERR] non-finite loss; skipping batch:", float(loss))
                return

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.loss = loss.detach()
        else:
            loss, diagnostics, _ = self.vi(self.net, self.edge_features, self.mesh)
            if not torch.isfinite(loss):
                print("[ERR] non-finite loss; skipping batch:", float(loss))
                return
            self.loss = loss
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()

    # ------------------------------------------------------------------ test
    @torch.no_grad()
    def test(self):
        self.net.eval()
        loss, diagnostics, outputs = self.vi(self.net, self.edge_features, self.mesh)
        if hasattr(self, "is_train") and self.is_train:
            self.net.train()

        if torch.cuda.is_available():
            print(
                f"[Debug] Test run on device: {next(self.net.parameters()).device}. "
                f"Loss: {loss.item():.4f}",
                flush=True,
            )

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
        }

    # ------------------------------------------------------------------ save/load
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
