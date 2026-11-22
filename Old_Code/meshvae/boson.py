from __future__ import annotations

import itertools
import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution

_BOSON_AVAILABLE = True

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
