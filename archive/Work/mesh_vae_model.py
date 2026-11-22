import os
import math
from typing import List, Tuple
import numpy as np
import itertools
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Distribution
import torch.optim as optim
from typing import Optional, List, Tuple
import math

# -- Optional Boson imports (guarded) ---------------------------------
_BOSON_AVAILABLE = True

# ---- we reuse your MeshConv / MeshPool / MeshUnpool ----
from models.mesh_conv_model import MeshConv, MeshPool, MeshUnpool


class BosonPrior(Distribution):
    arg_constraints = {}
    has_rsample = True

    def __init__(self, boson_sampler, batch_size, latent_features, discriminator_iter=0, validate_args=None):
        super().__init__(validate_args=validate_args)
        self.boson_sampler = boson_sampler
        self.batch_size = batch_size
        self.latent_features = latent_features
        self.discriminator_iter = discriminator_iter

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    #def rsample(self, sample_shape=torch.Size()):
    #    total_samples = (1 + self.discriminator_iter) * self.batch_size
        #latent = self.boson_sampler.generate(total_samples).to(self.device)
    #    latent = self.boson_sampler(total_samples).to(self.device)
    #    latent = torch.chunk(latent, 1 + self.discriminator_iter, dim=0)
    #    return latent

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        total_samples = (1 + self.discriminator_iter) * self.batch_size
        latent = self.boson_sampler(total_samples).to(self.device)  # (total, latent_dim)

        if self.discriminator_iter and self.discriminator_iter > 0:
            # chunk into (1 + d_iter) segments but return the model’s segment
            chunks = torch.chunk(latent, 1 + self.discriminator_iter, dim=0)
            return chunks[0]  # (batch_size, latent_dim)
        else:
            return latent[: self.batch_size]  # (batch_size, latent_dim)


    def log_prob(self, z):
        mean = torch.zeros_like(z)
        std = torch.ones_like(z)
        log_scale = torch.log(std)
        log_prob = -0.5 * (
            ((z - mean) ** 2) / (std ** 2)
            + 2 * log_scale
            + torch.log(torch.tensor(2 * torch.pi, device=z.device))
        )
        return log_prob.sum(dim=-1)

# =============================================================================
# Boson Sampling Simulation Functions
# =============================================================================

def generate_unitary(m, bs_loss, bs_jitter, phase_noise_std=0.0, systematic_phase_offset=0.0, mode_loss=None):
    """
    Generate a modified random unitary matrix for boson sampling simulations.

    This function creates a random complex matrix, performs QR decomposition
    to obtain a unitary matrix, and then applies various modifications such as loss,
    jitter, phase noise, and a systematic phase offset. Optionally, mode-dependent losses
    can be applied.

    Parameters:
        m (int): The dimension of the unitary matrix.
        bs_loss (float): Overall loss factor applied to the matrix.
        bs_jitter (float): Jitter added to the matrix elements.
        phase_noise_std (float, optional): Standard deviation for phase noise. Defaults to 0.0.
        systematic_phase_offset (float, optional): Systematic phase offset to apply. Defaults to 0.0.
        mode_loss (array-like, optional): Array of loss factors for each mode. Defaults to None.

    Returns:
        np.ndarray: The modified unitary matrix of shape (m, m).
    """
    # random complex -> QR -> unitary-ish
    random_matrix = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    Q, _ = np.linalg.qr(random_matrix)

    # apply losses/jitter
    Q *= np.sqrt(bs_loss)
    Q += bs_jitter * np.random.randn(m, m)

    # per-element phase noise
    if phase_noise_std != 0.0:
        phase_noise = np.exp(1j * np.random.normal(0, phase_noise_std, size=(m, m)))
        Q *= phase_noise

    # CHANGED: always apply; if zeros it's a no-op (fixes ambiguous truth-value)
    phase = np.asarray(systematic_phase_offset, dtype=np.float32)
    Q *= np.exp(1j * phase)

    # optional per-mode loss
    if mode_loss is not None:
        ml = np.asarray(mode_loss, dtype=np.float32).reshape(m)
        Q = np.array([Q[i, :] * np.sqrt(ml[i]) for i in range(m)])

    return Q


def compute_permanent(A: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the permanent of a square matrix A (dtype should be torch.cfloat or torch.cdouble)
    by brute-force enumeration of permutations.

    Parameters:
        A (torch.Tensor): A 2D tensor (n x n) with a complex dtype.

    Returns:
        torch.Tensor: A single complex scalar (the permanent).
    """
    # Number of modes
    n = A.shape[0]

    # We'll accumulate the permanent in perm_sum
    perm_sum = torch.zeros([], dtype=A.dtype, device=A.device)

    # For each permutation sigma of {0,1,...,n-1}, multiply together A[i, sigma[i]] for i in [0..n-1].
    for sigma in itertools.permutations(range(n)):
        product_term = torch.tensor(1.0, dtype=A.dtype, device=A.device)
        for i in range(n):
            product_term = product_term * A[i, sigma[i]]
        perm_sum = perm_sum + product_term

    return perm_sum

def get_submatrix(U, input_modes, output_modes):
    """
    Extract a submatrix from a given matrix U.

    This function returns the submatrix defined by selecting the rows corresponding
    to input_modes and the columns corresponding to output_modes.

    Parameters:
        U (np.ndarray): The original matrix.
        input_modes (array-like): Indices for the rows.
        output_modes (array-like): Indices for the columns.

    Returns:
        np.ndarray: The extracted submatrix.
    """
    return U[np.ix_(input_modes, output_modes)]

def kerr_phase_factor(output_modes, m, chi):
    """
    Compute a Kerr phase factor based on the photon counts in output modes.
    
    Each mode contributes a factor exp(-1j * chi * n * (n-1)),
    where n is the photon number in that mode.
    """
    counts = np.bincount(output_modes, minlength=m)
    factors = [np.exp(-1j * chi * n * (n - 1)) for n in counts]
    return factors

def boson_sampling_probability(U, input_modes, output_modes, effective_mu, m):
    if len(input_modes) == 0 or len(output_modes) == 0:
        return 0
    U_sub = get_submatrix(U, input_modes, output_modes)

    A_sub_torch = torch.tensor(U_sub, dtype=torch.complex64)
    permanent_val = compute_permanent(A_sub_torch)

    classical_probability = np.prod(np.abs(U_sub)**2)

    # Convert permanent to numpy complex
    permanent_val_np = permanent_val.detach().cpu().numpy()  # shape (), complex

    # Compute probs
    probability = effective_mu * np.abs(permanent_val_np)**2 + (1 - effective_mu) * classical_probability

    return probability

def boson_sampling_probability_advanced(U, input_modes, output_modes, effective_mu, m, chi):
    if len(input_modes) == 0 or len(output_modes) == 0:
        return 0
    U_sub = get_submatrix(U, input_modes, output_modes)

    A_sub_torch = torch.tensor(U_sub, dtype=torch.complex64)
    permanent_val = compute_permanent(A_sub_torch)

    classical_probability = np.prod(np.abs(U_sub)**2)

    # Kerr Phase Factors
    nl_factor_list = kerr_phase_factor(output_modes, m, chi)  
    nl_factor_val = np.prod(nl_factor_list)  # single scalar

    # Convert permanent to numpy complex
    permanent_val_np = permanent_val.detach().cpu().numpy()  # shape (), complex

    # Compute probs
    combined_val = nl_factor_val * permanent_val_np
    probability = effective_mu * np.abs(combined_val)**2 + (1 - effective_mu) * classical_probability

    return probability

def sample_input_state(m, num_sources, input_loss, coupling_efficiency, detector_inefficiency, multi_photon_prob):
    """
    Simulate the sampling of input states for a boson sampling experiment.

    Each source is considered for photon emission based on a combined efficiency parameter.
    If a photon is emitted, a mode is randomly selected (with possible multi-photon events).

    Parameters:
        m (int): Total number of modes.
        num_sources (int): Number of photon sources.
        input_loss (float): Loss factor at the input.
        coupling_efficiency (float): Efficiency of photon coupling.
        detector_inefficiency (float): Detector inefficiency factor.
        multi_photon_prob (float): Probability of multi-photon emission from a source.

    Returns:
        tuple: A tuple (interfering_input_modes, total_photons) where:
            interfering_input_modes (np.ndarray): Array of mode indices where photons are present.
            total_photons (int): Total number of photons emitted.
    """
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
    """
    Apply a dark count correction to a computed probability.

    The correction is simply an additive term proportional to the dark count rate and the number of modes.

    Parameters:
        probability (float): The original computed probability.
        m (int): Total number of modes.
        dark_count_rate (float): The dark count rate per mode.

    Returns:
        float: The adjusted probability.
    """
    dark_counts = dark_count_rate * m
    return probability + dark_counts

def boson_sampling_simulation(
    m, num_sources, num_loops,
    input_loss, coupling_efficiency, detector_inefficiency, multi_photon_prob,
    mu, temporal_mismatch, spectral_mismatch, arrival_time_jitter,
    bs_loss, bs_jitter, phase_noise_std, systematic_phase_offset, mode_loss,
    dark_count_rate,
    use_advanced_nonlinearity=False, chi=0
):
    """
    Run multiple boson sampling simulation loops and return the average transition probability.

    For each loop, the function:
      1. Generates a unitary matrix with the specified losses, jitter, and noise.
      2. Samples an input state.
      3. Randomly selects output modes.
      4. Computes the transition probability using either the standard or an advanced nonlinearity model.
      5. Applies a dark count correction.

    Parameters:
        m (int): Total number of modes.
        num_sources (int): Number of photon sources.
        num_loops (int): Number of simulation loops.
        input_loss (float): Input loss factor.
        coupling_efficiency (float): Photon coupling efficiency.
        detector_inefficiency (float): Detector inefficiency.
        multi_photon_prob (float): Probability of multi-photon emission.
        mu (float): Weighting parameter between quantum and classical contributions.
        temporal_mismatch (float): Factor representing temporal mismatches.
        spectral_mismatch (float): Factor representing spectral mismatches.
        arrival_time_jitter (float): Factor for photon arrival time jitter.
        bs_loss (float): Loss factor for the boson sampler unitary.
        bs_jitter (float): Jitter factor for the boson sampler unitary.
        phase_noise_std (float): Standard deviation for phase noise.
        systematic_phase_offset (float): Systematic phase offset to apply.
        mode_loss (np.ndarray): Mode-dependent loss factors.
        dark_count_rate (float): Dark count rate per mode.
        use_advanced_nonlinearity (bool, optional): If True, use the advanced nonlinearity model. Defaults to False.
        chi (float): Entanglement strength. Default 0.0

    Returns:
        float: The average transition probability over all simulation loops.
    """
    probabilities = []
    effective_mu = mu * (1 - temporal_mismatch) * (1 - spectral_mismatch) * (1 - arrival_time_jitter)
    for _ in range(num_loops):
        U = generate_unitary(m, bs_loss, bs_jitter, phase_noise_std, systematic_phase_offset, mode_loss)
        input_modes, interfering_count = sample_input_state(
            m, num_sources, input_loss, coupling_efficiency, detector_inefficiency, multi_photon_prob
        )
        if interfering_count == 0:
            probabilities.append(0)
            continue
        output_modes = np.random.choice(m, interfering_count, replace=True)
        if use_advanced_nonlinearity:
            probability = boson_sampling_probability_advanced(U, input_modes, output_modes,
                                                               effective_mu, m, chi)
        else:
            probability = boson_sampling_probability(U, input_modes, output_modes, effective_mu)
        probability = apply_dark_counts(probability, m, dark_count_rate)
        probabilities.append(probability)
    return np.mean(probabilities)

# =============================================================================
# PyTorch Module: Wrap Boson Sampling Simulation
# =============================================================================

class BosonSamplerTorch(nn.Module):
    r"""A PyTorch module that wraps the boson sampling simulation.

    This module runs the boson sampling simulation (using NumPy operations) and returns a scalar probability,
    replicated over a batch. The systematic phase offset is a trainable parameter.

    Attributes:
        m (int): Total number of modes.
        num_sources (int): Number of photon sources.
        num_loops (int): Number of simulation loops.
        input_loss (float): Input loss factor.
        coupling_efficiency (float): Coupling efficiency.
        detector_inefficiency (float): Detector inefficiency.
        multi_photon_prob (float): Probability of multi-photon emission.
        mu (float): Weighting parameter for quantum versus classical contributions.
        temporal_mismatch (float): Temporal mismatch factor.
        spectral_mismatch (float): Spectral mismatch factor.
        arrival_time_jitter (float): Photon arrival time jitter factor.
        bs_loss (float): Loss factor for the boson sampler unitary.
        bs_jitter (float): Jitter factor for the boson sampler unitary.
        phase_noise_std (float): Standard deviation for phase noise.
        systematic_phase_offset (float): Systematic phase offset (trainable).
        mode_loss (np.ndarray): Array of mode-dependent loss factors.
        dark_count_rate (float): Dark count rate per mode.
        use_advanced_nonlinearity (bool): Flag to use the advanced nonlinearity model.
        chi (float): Entanglement Strength. Default 0.0
    """
    def __init__(self,
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
                systematic_phase_offset: float,      # scalar or array-like
                mode_loss: np.ndarray,               # scalar or array-like
                dark_count_rate: float,
                use_advanced_nonlinearity: bool = False,
                chi: float = 0.0):
        super(BosonSamplerTorch, self).__init__()

        # ---- scalar hyper-parameters ----
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

        # ---- helper: coerce any scalar/array to length-m numpy array ----
        def _to_len_m(x, m, dtype=np.float32):
            arr = np.asarray(x, dtype=dtype)
            if arr.ndim == 0:
                arr = np.full((m,), float(arr), dtype=dtype)  # scalar -> repeat m
            elif arr.size != m:
                arr = np.resize(arr, (m,))                    # short/long -> resize
            return arr

        # make tensors of shape (m,)
        self.mode_loss = torch.from_numpy(_to_len_m(mode_loss, m)).float()
        sys_phase_np = _to_len_m(systematic_phase_offset, m)
        # keep as trainable Parameter (length m)
        self.systematic_phase_offset = nn.Parameter(torch.from_numpy(sys_phase_np).float())
        self.dark_count_rate = dark_count_rate
        self.use_advanced_nonlinearity = use_advanced_nonlinearity
        self.chi = chi

        # --- lightweight debug so you can verify shapes once ---
        print(f"[BOSON] init m={self.m}  mode_loss={tuple(self.mode_loss.shape)}  "
            f"sys_phase_off={tuple(self.systematic_phase_offset.shape)}")

    def forward(self, batch_size: int):
        """
        Run the boson sampling simulation and return a tensor of probabilities
        replicated over the batch.
        """
        # Use current (trainable) phase offset and per-mode loss as NumPy
        phase_offset_np = self.systematic_phase_offset.detach().cpu().numpy()
        mode_loss_np    = self.mode_loss.detach().cpu().numpy()

        avg_probability = boson_sampling_simulation(
            self.m, self.num_sources, self.num_loops,
            self.input_loss, self.coupling_efficiency, self.detector_inefficiency,
            self.multi_photon_prob, self.mu, self.temporal_mismatch,
            self.spectral_mismatch, self.arrival_time_jitter,
            self.bs_loss, self.bs_jitter, self.phase_noise_std,
            phase_offset_np, mode_loss_np,
            self.dark_count_rate, self.use_advanced_nonlinearity, self.chi
        )

        # replicate scalar over batch and keep it on the same device as the Parameter
        result = torch.tensor(
            avg_probability, dtype=torch.float32, device=self.systematic_phase_offset.device
        )
        return result.expand(batch_size)


    #def forward(self, batch_size: int):
    #    """
    #    Run the boson sampling simulation and return a tensor of probabilities replicated over the batch.

    #    Parameters:
    #        batch_size (int): The number of samples in the batch.

    #    Returns:
    #        torch.Tensor: A tensor of shape (batch_size,) where each element is the simulated probability.
    #    """
    #    # Use the current (trainable) phase offset.
    #    phase_offset = self.systematic_phase_offset.detach().cpu().numpy().item()
    #    avg_probability = boson_sampling_simulation(
    #        self.m, self.num_sources, self.num_loops,
    #        self.input_loss, self.coupling_efficiency, self.detector_inefficiency, self.multi_photon_prob,
    #        self.mu, self.temporal_mismatch, self.spectral_mismatch, self.arrival_time_jitter,
    #        self.bs_loss, self.bs_jitter, self.phase_noise_std, phase_offset, self.mode_loss,
    #        self.dark_count_rate,
    #        self.use_advanced_nonlinearity, self.chi
    #    )
        # Return the scalar probability as a tensor replicated over the batch dimension.
    #    result = torch.tensor(avg_probability, dtype=torch.float32, device=self.systematic_phase_offset.device)
    #    return result.expand(batch_size)

# =============================================================================
# Boson Latent Generator: Wrap the boson sampler to produce a latent vector.
# =============================================================================

class BosonLatentGenerator(nn.Module):
    r"""A wrapper that converts the scalar output of BosonSamplerTorch into a latent vector.

    For demonstration, the scalar output is replicated to create a latent vector of a specified dimension.

    Attributes:
        latent_dim (int): Desired dimension of the latent vector.
        boson_sampler_module (BosonSamplerTorch): An instance of the boson sampler module.
    """
    def __init__(self, latent_dim: int, boson_sampler_module: BosonSamplerTorch):
        super(BosonLatentGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.boson_sampler_module = boson_sampler_module

    def forward(self, batch_size: int):
        """
        Convert the scalar output of the boson sampler into a latent vector.

        Parameters:
            batch_size (int): Number of latent vectors to generate.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, latent_dim) representing the latent vectors.
        """
        latent_scalar = self.boson_sampler_module(batch_size)  # shape: (batch_size,)
        latent_vector = latent_scalar.unsqueeze(1).repeat(1, self.latent_dim)
        return latent_vector

import numpy as np

def _normalize_boson_cfg(raw: dict, latent_dim: int) -> dict:
    """Coerce CLI/json dict to what BosonSamplerTorch expects; fill defaults."""
    p = dict(raw or {})
    # Required ints
    p.setdefault("m", latent_dim)               # number of modes; reasonable default
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


class MeshEncoder(nn.Module):
    """
    Mesh encoder: MeshConv -> Pool ... -> global average -> (mu, log_sigma).
    Inputs:
        x:     [B, C, E]   (or [B, C, E, 1]; we squeeze the trailing 1)
        mesh:  list of B mesh objects (mutated in-place by pooling)
    Returns:
        mu:         [B, Z]
        log_sigma:  [B, Z]
        start_E:    int (the **actual** E after the last pool op; equals input E if no pooling)
    """
    def __init__(self,
                 in_channels: int,
                 channels: List[int],      # e.g. [64, 128, 256, 256]
                 pool_targets: List[int],   # e.g. [600, 450, 300, 180]
                 latent_dim: int,
                 k: int = 5,
                 opt=None):
        super().__init__()
        assert len(channels) >= 1 and len(pool_targets) >= 1
        assert len(pool_targets) <= len(channels), "1 pool per conv block (last conv can be unpooled too)."

        convs, pools = [], []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            convs.append(MeshConv(c_in, c_out, k=k, bias=True))
            # only construct a pool op if we're not skipping
            if i < len(pool_targets) and (opt is None or getattr(opt, 'pool_mode', 'target') != 'skip'):
                pools.append(
                    MeshPool(
                        target=int(pool_targets[i]),
                        multi_thread=False,
                        mode=getattr(opt, 'pool_mode', 'target'),
                        keep_frac=float(getattr(opt, 'pool_keep', 1.0)),
                    )
                )
            c_in = c_out

        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)
        self.latent_mu      = nn.Linear(channels[-1], latent_dim)
        self.latent_logsigma= nn.Linear(channels[-1], latent_dim)

        # config hint only; we’ll compute the real start_E dynamically
        self._config_start_E = int(pool_targets[-1])

    def forward(self, x: torch.Tensor, mesh: list) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # normalize legacy shapes -> [B, C, E]
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        assert x.dim() == 3, f"[ENC] expected [B, C, E], got {tuple(x.shape)}"
        B, _, _ = x.shape
        #print(f"[ENC] start      {tuple(x.shape)}", flush=True)

        f = x
        for i, conv in enumerate(self.convs):
            f = F.relu(conv(f, mesh))
            # some convs may produce a trailing 1: [B, C, E, 1]
            if f.dim() == 4 and f.shape[-1] == 1:
                #print(f"[ENC] squeeze trailing 1 after hop {i:02d} from {tuple(f.shape)} -> ", end="", flush=True)
                f = f.squeeze(-1)
                #print(f"{tuple(f.shape)}", flush=True)
            #print(f"[ENC] after hop {i:02d}: {tuple(f.shape)}", flush=True)

            if i < len(self.pools):
                f = self.pools[i](f, mesh)   # mutates mesh; records groups
                # pool can also leave a trailing 1 in some paths
                if f.dim() == 4 and f.shape[-1] == 1:
                    #print(f"[ENC] squeeze trailing 1 after pool {i:02d} from {tuple(f.shape)} -> ", end="", flush=True)
                    f = f.squeeze(-1)
                    #print(f"{tuple(f.shape)}", flush=True)
                #print(f"[ENC] after pool {i:02d}: {tuple(f.shape)}", flush=True)

        # FINAL SAFETY: squeeze any leftover [B, C, E, 1]
        if f.dim() == 4 and f.shape[-1] == 1:
            #print(f"[ENC] final squeeze before mean from {tuple(f.shape)} -> ", end="", flush=True)
            f = f.squeeze(-1)
            #print(f"{tuple(f.shape)}", flush=True)

        # actual E that the decoder must start from
        start_E = int(f.shape[-1])

        # global average over **edges** -> [B, C_last]
        #print(f"[ENC] mean over E {tuple(f.shape)} -> ", end="", flush=True)
        f_global = f.mean(dim=-1)
        #print(f"{tuple(f_global.shape)}", flush=True)

        # hard guards so this never silently breaks again
        assert f_global.dim() == 2, f"[ENC] global features must be [B, C], got {tuple(f_global.shape)}"
        assert f_global.shape[1] == self.latent_mu.in_features, \
            f"[ENC] C={f_global.shape[1]} does not match Linear.in_features={self.latent_mu.in_features}"

        mu        = self.latent_mu(f_global)       # [B, Z]
        log_sigma = self.latent_logsigma(f_global) # [B, Z]
        return mu, log_sigma, start_E


class MeshDecoder(nn.Module):
    def __init__(self,
                 out_channels: int,
                 channels: List[int],
                 pool_targets: List[int],
                 latent_dim: int,
                 k: int = 5,
                 opt=None):
        super().__init__()

        # ---- capacity (very important) ----
        # This is the FC "canvas" width. It MUST be >= the largest start_E the encoder will produce.
        # We read it from opt.ninp_edges; fall back to the largest pool target or 600.
        E_cap = int(getattr(opt, "ninput_edges", max(pool_targets) if len(pool_targets) else 600))
        self.max_E_cap = E_cap
        self.start_edges = E_cap     # initial per-batch start; forward() will overwrite with actual start_E

        # We start from the *smallest* resolution (the encoder’s last conv out)
        c0 = channels[-1]
        self.fc = nn.Linear(latent_dim, c0 * self.max_E_cap)

        # Build unpool/conv blocks in reverse order
        rev_targets = list(reversed(pool_targets))
        unpools, deconvs = [], []
        cur_channels = c0
        for up_target, next_c in zip(rev_targets, reversed(channels[:-1])):
            unpools.append(MeshUnpool(unroll_target=int(up_target)))
            deconvs.append(MeshConv(cur_channels, next_c, k=k, bias=True))
            cur_channels = next_c

        # Final hop: overshoot back to ORIGINAL edges; last unpool clamps to history (or fallback)
        unpools.append(MeshUnpool(unroll_target=10**9))
        deconvs.append(MeshConv(cur_channels, cur_channels, k=k, bias=True))

        self.unpools = nn.ModuleList(unpools)
        self.deconvs = nn.ModuleList(deconvs)
        self.out_head = MeshConv(cur_channels, out_channels, k=k, bias=True)


    def forward(self, z: torch.Tensor, mesh: list, start_E: int) -> torch.Tensor:
        B = z.shape[0]

        # ---- Guards + helpful prints ----
        assert start_E > 0, "[DEC] start_E must be > 0"
        if start_E > self.max_E_cap:
            # If you see this: increase --ninp_edges to at least start_E
            raise RuntimeError(f"[DEC] start_E={start_E} > max_E_cap={self.max_E_cap} "
                            f"(set --ninp_edges to >= {start_E})")

        # Do not touch internals of MeshConv here (no .in_channels) — keep the print safe
        #print(f"[DEC] start (B={B}), cap={self.max_E_cap}", flush=True)

        # FC to capacity, then slice to the per-batch actual start_E
        f = self.fc(z).view(B, -1, self.max_E_cap)[:, :, :start_E]    # [B, C0, start_E]

        # Go up through unpools + deconvs
        for unpool, deconv in zip(self.unpools, self.deconvs):
            f = unpool(f, mesh)                 # increase E using recorded history or fallback
            f = F.relu(deconv(f, mesh))         # refine

        x_hat = self.out_head(f, mesh)          # [B, C_out, E_orig]
        return x_hat


# =====================================================================
# MeshVAE: unchanged encoder/decoder API; prior() can be Gaussian or Boson
# =====================================================================
class MeshVAE(nn.Module):
    """
    End-to-end VAE on meshes. Learns p(x|z) as a factorized Normal over edge features.
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

        # ---- config / options we pass through from MeshVAEModel ----
        self.latent_dim = int(latent_dim)

        # keep these attributes so forward() can use them
        self.boson_sampler_module = None   # the physics nn.Module (callable: batch_size -> scalar)
        self.boson_sampler        = None   # the latent vector wrapper (scalar -> latent_dim)

        self.latent_prior  = getattr(opt, "latent_prior", "gaussian").lower()
        self.boson_params  = getattr(opt, "boson_cfg", None)  # <- from BaseOptions.parse()

        if self.latent_prior == "boson":
            if not _BOSON_AVAILABLE:
                raise ImportError(
                    "latent_prior='boson' requested but BosonPrior/BosonLatentGenerator/BosonSamplerTorch "
                    "could not be imported. Check your GenAI.* modules."
                )
            # 1) instantiate your physics module with user-supplied kwargs
            boson_module = BosonSamplerTorch(**self.boson_params)
            # 2) keep a handle because your forward() calls module(batch_size)
            self.boson_sampler_module = boson_module
            # 3) wrap it to make latent vectors of size latent_dim when needed
            self.boson_sampler = BosonLatentGenerator(self.latent_dim, boson_module)
            print("[VAE] Using BOSON latent prior.", flush=True)
        else:
            print("[VAE] Using GAUSSIAN latent prior.", flush=True)


        # ---- encoder/decoder backbone (YOUR existing conv mesh nets) ----
        # NOTE: we don’t touch your existing MeshEncoder / MeshDecoder signatures.
        #       They should be defined above in this file as before.
        self.encoder = MeshEncoder(in_channels, channels, pool_targets, self.latent_dim, k=k, opt=opt)
        self.decoder = MeshDecoder(in_channels, channels, pool_targets, self.latent_dim, k=k, opt=opt)

        # observation scale (as before)
        self.obs_log_std = nn.Parameter(torch.tensor(float(obs_log_std)), requires_grad=False)

    # --------- small helper: diagonal Gaussian with reparam -----------
    @staticmethod
    def _diag_normal(mu: torch.Tensor, log_sigma: torch.Tensor) -> Independent:
        sigma = torch.exp(log_sigma)
        # interpret sum across last dim (latent features) for q(z|x) and across C*E for px
        return Independent(Normal(mu, sigma), 1)

    # ------------------------- q(z|x) ---------------------------------
    def posterior(self, x: torch.Tensor, mesh: list):
        mu, log_sigma, start_E = self.encoder(x, mesh)
        # **FIX 3:** Clamp log_sigma to avoid extreme variance values
        log_sigma = log_sigma.clamp(min=-10.0, max=10.0)
        qz = self._diag_normal(mu, log_sigma)
        return qz, mu, start_E

    # ------------------------- p(z) -----------------------------------
    def prior(self, batch: int, latent_dim: int, device):
        if getattr(self, "latent_prior", "gaussian") == "boson" and self.boson_sampler is not None:
            # Uses your sampler's .generate(n) (via the shim above if needed)
            return BosonPrior(
                boson_sampler=self.boson_sampler,
                batch_size=batch,
                latent_features=latent_dim
            )
        else:
            mu = torch.zeros(batch, latent_dim, device=device)
            log_sigma = torch.zeros(batch, latent_dim, device=device)
            return self._diag_normal(mu, log_sigma)


    # ------------------------ p(x|z) ----------------------------------
    def likelihood(self, x_mean: torch.Tensor) -> Independent:
        # treat C*E as factorized dims (sum over last 2 dims)
        std = torch.exp(self.obs_log_std)
        return Independent(Normal(loc=x_mean, scale=std), reinterpreted_batch_ndims=2)

    # ------------------------- forward --------------------------------
    def forward(self, x: torch.Tensor, mesh: list):
        # --- normalize legacy shapes -> [B, C, E] ---
        x_in = x.squeeze(-1) if (x.dim() == 4 and x.shape[-1] == 1) else x
        B, C, E0 = x_in.shape
        #print(f"[VAE] x_mean_? input check, B={B}, C={C}, E0={E0}", flush=True)

        # q(z|x)
        qz, mu, start_E = self.posterior(x_in, mesh)

        z = qz.rsample()

        # let decoder know the effective starting edges E
        self.decoder.start_edges = int(start_E)
        if hasattr(self.decoder, "unpools") and len(self.decoder.unpools) > 0:
            # last unpool overshoots back to original E0 when no history
            self.decoder.unpools[-1].unroll_target = int(E0)

        # decode to x_mean and normalize to [B, C, E]
        x_mean = self.decoder(z, mesh, start_E)
        if x_mean.dim() == 4 and x_mean.shape[-1] == 1:
            x_mean = x_mean.squeeze(-1)

        # print(f"[VAE] x_mean => {tuple(x_mean.shape)}", flush=True)

        # p(x|z)
        px = self.likelihood(x_mean)           # Independent Normal over [C, E]

        # p(z)
        pz = self.prior(B, z.shape[-1], device=x.device)

        # hand back everything the VI block needs
        return {"px": px, "pz": pz, "qz": qz, "z": z, "x_mean": x_mean, "x_in": x_in}


# =====================================================================
# MeshVariationalInference: unchanged math; works for both priors
# =====================================================================
class MeshVariationalInference(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = float(beta)

    def forward(self, model: MeshVAE, x: torch.Tensor, mesh: list):
        """
        x:     [B, C, E] (or [B, C, E, 1])
        mesh:  list of length B
        """
        out = model(x, mesh)
        px, pz, qz, z = out["px"], out["pz"], out["qz"], out["z"]

        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        # log-likelihoods aggregated to [B] by our Independent dists
        log_px = px.log_prob(x)      # [B]
        log_pz = pz.log_prob(z)      # [B]
        log_qz = qz.log_prob(z)      # [B]

        kl = (log_qz - log_pz)       # [B]
        elbo = log_px - self.beta * kl
        loss = -elbo.mean()

        diag = {
            "elbo": elbo.detach(),
            "log_px": log_px.detach(),
            "kl": kl.detach(),
        }
        return loss, diag, out


# =====================================================================
# LR scheduler helper (unchanged from your version)
# =====================================================================
def get_scheduler(optimizer, opt):
    from torch.optim import lr_scheduler
    if opt.lr_policy == "lambda":
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                   threshold=0.01, patience=5)
    else:
        return NotImplementedError(f"learning rate policy [{opt.lr_policy}] is not implemented")
    return scheduler


# =====================================================================
# MeshVAEModel: identical external behavior; we just pass new knobs
# =====================================================================
class MeshVAEModel(nn.Module):  # make sure to subclass nn.Module for register_buffer
    def __init__(self, opt):
        super().__init__()  # ensure proper nn.Module init
        self.opt = opt
        self.gpu_ids = getattr(opt, "gpu_ids", [])
        self.is_train = getattr(opt, "is_train", True)
        # Register a device reference buffer
        self.register_buffer("_devref", torch.empty(0))
        # **FIX 1:** Set _devref to GPU if available, else CPU
        device_target = torch.device(f"cuda:{self.gpu_ids[0]}" 
                                     if self.gpu_ids and torch.cuda.is_available() 
                                     else "cpu")
        self._devref = self._devref.to(device_target)  # ensure self.device is correct
        # Build the VAE (MeshVAE) and move to device
        # Build the VAE (MeshVAE) and move to device
        self.net = MeshVAE(
            in_channels=opt.input_nc,
            channels=list(opt.ncf),
            pool_targets=list(opt.pool_res),
            latent_dim=int(getattr(opt, "latent_dim", 64)),
            k=5,
            opt=opt,
        ).to(self.device)

        init_type = getattr(opt, "init_type", "xavier")
        init_gain = getattr(opt, "init_gain", 0.02)
        for m in self.net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if init_type == "xavier":
                    nn.init.xavier_normal_(m.weight, gain=init_gain)
                elif init_type == "normal":
                    nn.init.normal_(m.weight, 0.0, init_gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=init_gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # If multiple GPUs are specified, wrap model in DataParallel
        if isinstance(self.gpu_ids, (list, tuple)) and len(self.gpu_ids) > 1:
            self.net = nn.DataParallel(self.net, device_ids=self.gpu_ids)

        # Set network to training or eval mode based on phase
        self.net.train(self.is_train)

        # VI module on the same device
        beta = float(getattr(opt, "beta", 1.0))
        self.vi = MeshVariationalInference(beta=beta).to(self.device)

        # Optimizer / scheduler
        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = get_scheduler(self.optimizer, opt)

        # Runtime placeholders
        self.edge_features = None
        self.mesh = None
        self.loss = torch.tensor(0.0, device=self.device)
        self.task = "vae"

        # Debug
        if torch.cuda.is_available():
            main_dev = (self.net.device_ids[0] if isinstance(self.net, nn.DataParallel) else self.device)
            print(f"[Init] MeshVAEModel initialized on {main_dev}. "
                f"Model parameters on {next(self.net.parameters()).device}.",
                flush=True)


    @property
    def device(self):
        return self._devref.device

    # ----------------- train step plumbing (unchanged) -----------------
    def update_learning_rate(self) -> None:
        """Update LR once per epoch (mirrors ClassifierModel)."""
        if not getattr(self, "is_train", False):
            return
        if not hasattr(self, "scheduler") or self.scheduler is None:
            print("[WARN] update_learning_rate called but no scheduler is set.", flush=True)
            return
        old_lr = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()                                     # step scheduler
        lr = self.optimizer.param_groups[0]["lr"]                 # readback for print
        if lr != old_lr:
            print(f"learning rate: {old_lr:.7f} -> {lr:.7f}", flush=True)
        else:
            print(f"learning rate = {lr:.7f}", flush=True)

    # --------------------------- I/O -----------------------------------
    def set_input(self, data: dict):
        fe = torch.from_numpy(data['edge_features']).float() \
             if not isinstance(data['edge_features'], (list, tuple)) else data['edge_features']
        # Normalize shape to [B,C,E]
        if fe.dim() == 4 and fe.shape[-1] == 1:
            fe = fe.squeeze(-1)
        elif fe.dim() != 3:
            raise ValueError(f"Unexpected edge_features shape {tuple(fe.shape)}")

        # Move to model's device
        dev = next(self.parameters()).device  # after Fix 1, parameters exist on correct device
        self.edge_features = fe.to(dev)
        self.mesh = data['mesh']

        # **FIX 2:** Replace any NaN/Inf in input features to prevent propagation
        if not torch.isfinite(self.edge_features).all():
            self.edge_features = torch.nan_to_num(
                self.edge_features, nan=0.0, posinf=0.0, neginf=0.0
            )
        # Debug: verify device placement and memory
        if torch.cuda.is_available():
            cur_dev = self.edge_features.device
            mem_alloc = torch.cuda.memory_allocated(cur_dev) / (1024**2)
            mem_reserved = torch.cuda.memory_reserved(cur_dev) / (1024**2)
            print(f"[Debug] edge_features device: {cur_dev}, GPU memory allocated: {mem_alloc:.1f} MB (reserved {mem_reserved:.1f} MB)",
                  flush=True)

    # ------------------------- fwd/bwd ---------------------------------
    def forward(self):
        return self.net(self.edge_features, self.mesh)

    def backward(self, outputs: dict):
        loss, _, _ = self.vi(self.net, self.edge_features, self.mesh)
        self.loss = loss
        self.loss.backward()

    # mesh_vae_model.py  -- inside class MeshVAEModel
    def optimize_parameters(self):
        # make sure we’re in train mode
        self.net.train(True)

        self.optimizer.zero_grad(set_to_none=True)

        use_amp = bool(getattr(self.opt, "use_amp", False) and torch.cuda.is_available())

        if use_amp:
            # lazy-create scaler once
            if not hasattr(self, "scaler"):
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler(enabled=True)

            from torch.cuda.amp import autocast
            with autocast(device_type="cuda", dtype=torch.float16):
                loss, diagnostics, _ = self.vi(self.net, self.edge_features, self.mesh)

            if not torch.isfinite(loss):
                # guard against NaNs/Infs; skip this batch cleanly
                print("[ERR] non-finite loss; skipping batch:", float(loss))
                return

            # scale/backward; unscale before clipping; step; update
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.loss = loss.detach()

        else:
            # plain FP32 path (works on CPU or GPU)
            loss, diagnostics, _ = self.vi(self.net, self.edge_features, self.mesh)
            if not torch.isfinite(loss):
                print("[ERR] non-finite loss; skipping batch:", float(loss))
                return
            self.loss = loss
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()



    # -------------------------- eval / test ----------------------------
    @torch.no_grad()
    def test(self):
        # Ensure model is in eval mode during testing
        self.net.eval()
        # Forward pass on the current device
        loss, diagnostics, outputs = self.vi(self.net, self.edge_features, self.mesh)
        # Optionally, restore training mode after test if this method is called mid-training
        if hasattr(self, "is_train") and self.is_train:
            self.net.train()
        # Debug: print which device the evaluation ran on
        if torch.cuda.is_available():
            print(f"[Debug] Test run on device: {next(self.net.parameters()).device}. Loss: {loss.item():.4f}",
                  flush=True)
        # Prepare output dictionary (move large tensors to CPU for output)
        return {
            "loss": loss.detach().item(),
            "diagnostics": diagnostics,           # e.g. {'elbo': tensor([B]), 'log_px': tensor([B]), 'kl': tensor([B])}
            "nexamples": int(self.edge_features.shape[0]),
            "x": self.edge_features.detach().cpu(),
            "x_mean": (outputs.get("x_mean").detach().cpu() 
                       if isinstance(outputs, dict) and outputs.get("x_mean") is not None 
                       else None),
        }

    # ---------------------- save / load net ----------------------------
    def save_network(self, which_epoch: str):
        fname = f"{which_epoch}_vae_net.pth"
        path = os.path.join(self.opt.checkpoints_dir, self.opt.name, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.gpu_ids and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict() if isinstance(self.net, nn.DataParallel) else self.net.cpu().state_dict(), path)
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

