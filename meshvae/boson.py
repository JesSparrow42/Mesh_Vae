from __future__ import annotations

import importlib
import inspect
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution


_BOSON_AVAILABLE = True
_VENDORED_ROOT = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "nonlinear_photonic_sampler"
)


def _package_root(path: str | os.PathLike[str]) -> Path:
    root = Path(path).expanduser().resolve()
    if root.name == "bosonsampler" and (root / "__init__.py").is_file():
        root = root.parent
    if not (root / "bosonsampler" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"No bosonsampler package found below {root}. Expected "
            f"{root / 'bosonsampler' / '__init__.py'}."
        )
    return root


def _load_sparrow_sampler(sampler_path: str | None = None):
    """Import Sparrow's package without reproducing its simulator here."""
    requested = sampler_path or os.environ.get("SPARROW_BOSON_SAMPLER_PATH")
    roots: list[Path | None] = []
    if requested:
        roots.append(_package_root(requested))
    else:
        # Prefer the exact checked-in source over a potentially stale installed
        # package. An explicit path remains the way to select a newer checkout.
        if _VENDORED_ROOT.is_dir():
            roots.append(_VENDORED_ROOT)
        roots.append(None)

    errors: list[str] = []
    for root in roots:
        try:
            if root is not None:
                root_text = str(root)
                if root_text not in sys.path:
                    sys.path.insert(0, root_text)
            module = importlib.import_module("bosonsampler")
            if not hasattr(module, "BosonSamplerTorch"):
                raise ImportError("module does not export BosonSamplerTorch")
            if not hasattr(module, "BosonLatentGenerator"):
                raise ImportError("module does not export BosonLatentGenerator")
            return module
        except Exception as exc:
            errors.append(f"{root or 'installed package'}: {type(exc).__name__}: {exc}")

    detail = "\n  ".join(errors)
    raise ImportError(
        "Sparrow's Non-Linear Photonic Sampler is required for --latent_prior boson. "
        "Install bosonsampler-wrapper, set SPARROW_BOSON_SAMPLER_PATH, or pass "
        "--boson_sampler_path /path/to/Non_Linear_Photonic_Sampler."
        + (f"\nImport attempts:\n  {detail}" if detail else "")
    )


def _alternating_input_state(m: int, num_sources: int) -> list[int]:
    """Build the [1, 0, 1, 0, ...] input used by Sparrow's examples."""
    if num_sources < 1 or num_sources > m:
        raise ValueError(f"num_sources must be in [1, {m}], received {num_sources}.")
    state = [0] * m
    early_then_late = list(range(0, m, 2)) + list(range(1, m, 2))
    for index in early_then_late[:num_sources]:
        state[index] = 1
    return state


def _normalize_boson_cfg(raw: dict | None, latent_dim: int) -> dict:
    """Validate MeshVAE settings without changing Sparrow's physical model."""
    p = dict(raw or {})
    configured_latent_dim = int(p.pop("latent_dim", latent_dim))
    if configured_latent_dim != int(latent_dim):
        raise ValueError(
            f"Boson latent_dim={configured_latent_dim} disagrees with model "
            f"latent_dim={latent_dim}."
        )
    legacy_removed = sorted(
        set(p) & {"chi", "multi_photon_prob", "num_circuits", "pool_size", "seed"}
    )
    if legacy_removed:
        raise ValueError(
            "These parameters belong to the temporary replacement sampler and are not "
            f"part of Sparrow's current sampler: {legacy_removed}. Remove them. For the "
            "QD nonlinear model use use_advanced_nonlinearity, detuning, pulse_bw, "
            "QD_linewidth, phi, and g2_target."
        )

    defaults = {
        "m": 8,
        "num_sources": None,
        "input_state": None,
        "num_loops": 100,
        "input_loss": 0.0,
        "coupling_efficiency": 1.0,
        # The Sparrow implementation currently uses this value as an efficiency.
        "detector_inefficiency": 1.0,
        "mu": 1.0,
        "temporal_mismatch": 0.0,
        "spectral_mismatch": 0.0,
        "arrival_time_jitter": 0.0,
        # In Sparrow's generate_unitary, these are transmission factors.
        "bs_loss": 1.0,
        "bs_jitter": 0.0,
        "phase_noise_std": 0.0,
        "systematic_phase_offset": 0.0,
        "mode_loss": None,
        "dark_count_rate": 0.0,
        "use_advanced_nonlinearity": False,
        "detuning": 0.0,
        "pulse_bw": 0.0,
        "QD_linewidth": 1.0,
        "phi": 0.0,
        "g2_target": 0.0,
        "sampler_path": None,
        # Adapter-only performance settings. Every cached vector is still
        # produced by Sparrow's sampler.
        "cache_size": 4096,
        "cache_circuits": 8,
    }
    unknown = sorted(set(p) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown Sparrow Boson parameters: {unknown}")
    for key, value in defaults.items():
        p.setdefault(key, value)

    p["latent_dim"] = configured_latent_dim
    p["m"] = int(p["m"])
    if p["m"] < 2:
        raise ValueError("Sparrow Boson mode count m must be at least 2.")

    if p["input_state"] is None:
        if p["num_sources"] is None:
            p["num_sources"] = max(1, p["m"] // 2)
        p["num_sources"] = int(p["num_sources"])
        p["input_state"] = _alternating_input_state(p["m"], p["num_sources"])
    else:
        p["input_state"] = [int(value) for value in p["input_state"]]
        if len(p["input_state"]) != p["m"]:
            raise ValueError(
                f"input_state has length {len(p['input_state'])}, but m={p['m']}."
            )
        if any(value < 0 for value in p["input_state"]):
            raise ValueError("input_state must contain non-negative photon counts.")
        state_sources = int(sum(p["input_state"]))
        if p["num_sources"] is not None and int(p["num_sources"]) != state_sources:
            raise ValueError(
                f"num_sources={p['num_sources']} disagrees with sum(input_state)={state_sources}."
            )
        p["num_sources"] = state_sources

    if p["num_sources"] < 1:
        raise ValueError("The Boson input state must contain at least one photon.")
    if p["num_sources"] > 12:
        raise ValueError(
            "The supplied Sparrow simulator's exact permanent path is intended for at most "
            "12 photons. Use a smaller physical sampler and let the adapter concatenate "
            "independent physical samples to the MeshVAE latent dimension."
        )
    p["num_loops"] = int(p["num_loops"])
    if p["num_loops"] < 1:
        raise ValueError("num_loops must be positive.")
    p["cache_size"] = int(p["cache_size"])
    p["cache_circuits"] = int(p["cache_circuits"])
    if p["cache_size"] < 0:
        raise ValueError("cache_size must be non-negative.")
    if p["cache_size"] and p["cache_size"] < 64:
        raise ValueError("cache_size must be 0 (live sampling) or at least 64.")
    if p["cache_circuits"] < 1:
        raise ValueError("cache_circuits must be positive.")

    if p["mode_loss"] is None:
        p["mode_loss"] = [1.0] * p["m"]
    else:
        p["mode_loss"] = [float(value) for value in p["mode_loss"]]
        if len(p["mode_loss"]) != p["m"]:
            raise ValueError(
                f"mode_loss has length {len(p['mode_loss'])}, but m={p['m']}."
            )
    p["systematic_phase_offset"] = float(p["systematic_phase_offset"])
    p["sampler_path"] = str(p["sampler_path"]) if p["sampler_path"] else None
    return p


class BosonSamplerTorch(nn.Module):
    """MeshVAE adapter around Sparrow's unmodified BosonSamplerTorch.

    Sparrow returns one m-mode Fock sample. MeshVAE normally has a much wider
    latent vector, so independent Sparrow samples are concatenated and, only
    when necessary, truncated to latent_dim. No probabilities or QD physics are
    reimplemented in this adapter.
    """

    def __init__(self, **config):
        super().__init__()
        if "latent_dim" not in config:
            raise ValueError("BosonSamplerTorch requires latent_dim.")
        self.config = _normalize_boson_cfg(config, int(config["latent_dim"]))
        self.latent_dim = int(self.config["latent_dim"])
        self.m = int(self.config["m"])
        self.num_sources = int(self.config["num_sources"])
        self.num_blocks = int(math.ceil(self.latent_dim / self.m))

        package = _load_sparrow_sampler(self.config.get("sampler_path"))
        backend_class = package.BosonSamplerTorch
        signature = inspect.signature(backend_class.__init__)
        parameters = signature.parameters

        common = {
            "num_loops": self.config["num_loops"],
            "input_loss": self.config["input_loss"],
            "coupling_efficiency": self.config["coupling_efficiency"],
            "detector_inefficiency": self.config["detector_inefficiency"],
            "mu": self.config["mu"],
            "temporal_mismatch": self.config["temporal_mismatch"],
            "spectral_mismatch": self.config["spectral_mismatch"],
            "arrival_time_jitter": self.config["arrival_time_jitter"],
            "bs_loss": self.config["bs_loss"],
            "bs_jitter": self.config["bs_jitter"],
            "phase_noise_std": self.config["phase_noise_std"],
            "systematic_phase_offset": self.config["systematic_phase_offset"],
            "mode_loss": np.asarray(self.config["mode_loss"], dtype=np.float64),
            "dark_count_rate": self.config["dark_count_rate"],
            "use_advanced_nonlinearity": self.config["use_advanced_nonlinearity"],
            "detuning": self.config["detuning"],
            "pulse_bw": self.config["pulse_bw"],
            "QD_linewidth": self.config["QD_linewidth"],
            "phi": self.config["phi"],
            "g2_target": self.config["g2_target"],
        }
        if "early_late_pairs" in parameters and "input_state" in parameters:
            backend_kwargs = {
                "early_late_pairs": max(1, self.m // 2),
                "input_state": list(self.config["input_state"]),
                **common,
            }
            self.backend_api = "early_late_pairs+input_state"
            ignored_nonideal = {
                "coupling_efficiency": (self.config["coupling_efficiency"], 1.0),
                "detector_inefficiency": (self.config["detector_inefficiency"], 1.0),
                "mu": (self.config["mu"], 1.0),
                "temporal_mismatch": (self.config["temporal_mismatch"], 0.0),
                "spectral_mismatch": (self.config["spectral_mismatch"], 0.0),
                "arrival_time_jitter": (self.config["arrival_time_jitter"], 0.0),
                "g2_target": (self.config["g2_target"], 0.0),
            }
            active_ignored = sorted(
                key
                for key, (value, ideal) in ignored_nonideal.items()
                if not math.isclose(float(value), float(ideal))
            )
            if (
                not bool(self.config["use_advanced_nonlinearity"])
                and not math.isclose(float(self.config["dark_count_rate"]), 0.0)
            ):
                active_ignored.append("dark_count_rate")
            if active_ignored:
                raise ValueError(
                    "This Sparrow package revision does not apply these parameters in its "
                    f"explicit input_state path: {active_ignored}. Refusing to label an "
                    "ignored setting as part of the experiment. Use the ideal values or "
                    "update the Sparrow package itself before benchmarking those effects."
                )
        elif "m" in parameters and "num_sources" in parameters:
            backend_kwargs = {
                "m": self.m,
                "num_sources": self.num_sources,
                **common,
            }
            if "latent_dim" in parameters:
                backend_kwargs["latent_dim"] = self.m
            if "multi_photon_prob" in parameters:
                backend_kwargs["multi_photon_prob"] = 0.0
            if "chi" in parameters:
                backend_kwargs["chi"] = 0.0
            self.backend_api = "m+num_sources"
        else:
            raise TypeError(
                "Unsupported BosonSamplerTorch constructor in "
                f"{inspect.getfile(backend_class)}: {signature}"
            )

        # Support package revisions by passing only parameters that revision
        # declares. Required missing arguments still fail loudly in its class.
        backend_kwargs = {
            key: value for key, value in backend_kwargs.items() if key in parameters
        }
        self.backend = backend_class(**backend_kwargs)
        self.source_file = str(Path(inspect.getfile(backend_class)).resolve())
        self.source_version = str(getattr(package, "__version__", "unknown"))
        print(
            "[BOSON] Sparrow Non-Linear Photonic Sampler "
            f"version={self.source_version} api={self.backend_api} source={self.source_file}",
            flush=True,
        )
        print(
            f"[BOSON] physical modes={self.m} photons={self.num_sources} "
            f"loops={self.config['num_loops']} latent_blocks={self.num_blocks} "
            f"latent_dim={self.latent_dim} nonlinear={bool(self.config['use_advanced_nonlinearity'])}",
            flush=True,
        )

        cache_size = int(self.config["cache_size"])
        if cache_size:
            circuits = min(int(self.config["cache_circuits"]), cache_size)
            counts = [cache_size // circuits] * circuits
            for index in range(cache_size % circuits):
                counts[index] += 1
            with torch.no_grad():
                cached = torch.cat(
                    [self._sample_live(count) for count in counts if count > 0], dim=0
                )
            self.register_buffer("sample_pool", cached, persistent=True)
            print(
                f"[BOSON] cached {len(cached)} official Sparrow latent vectors "
                f"across {circuits} independently generated circuits.",
                flush=True,
            )
        else:
            self.register_buffer(
                "sample_pool",
                torch.empty((0, self.latent_dim), dtype=torch.float32),
                persistent=True,
            )

    def _sample_live(self, batch_size: int) -> torch.Tensor:
        batch_size = int(batch_size)
        physical_samples = batch_size * self.num_blocks
        output = self.backend(physical_samples)
        latent = output[1] if isinstance(output, (tuple, list)) else output
        latent = torch.as_tensor(latent, dtype=torch.float32)
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        if latent.shape[0] == 1 and physical_samples > 1:
            latent = latent.expand(physical_samples, -1)
        if tuple(latent.shape) != (physical_samples, self.m):
            raise RuntimeError(
                "Sparrow sampler returned shape "
                f"{tuple(latent.shape)}; expected {(physical_samples, self.m)}."
            )
        latent = latent.reshape(batch_size, self.num_blocks * self.m)
        return latent[:, : self.latent_dim]

    def forward(self, batch_size: int) -> torch.Tensor:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.sample_pool.shape[0] == 0:
            return self._sample_live(batch_size)
        indices = torch.randint(
            0,
            self.sample_pool.shape[0],
            (batch_size,),
            device=self.sample_pool.device,
        )
        return self.sample_pool.index_select(0, indices)


class BosonLatentGenerator(nn.Module):
    def __init__(self, latent_dim: int, boson_sampler_module: BosonSamplerTorch):
        super().__init__()
        if int(latent_dim) != int(boson_sampler_module.latent_dim):
            raise ValueError("Boson latent dimensions disagree.")
        self.latent_dim = int(latent_dim)
        self.boson_sampler_module = boson_sampler_module

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.boson_sampler_module(batch_size)


class BosonPrior(Distribution):
    """Implicit Sparrow prior; density evaluation is intentionally unavailable."""

    arg_constraints = {}
    has_rsample = False

    def __init__(self, boson_sampler, batch_size: int, latent_features: int, device):
        self.boson_sampler = boson_sampler
        self.batch_size = int(batch_size)
        self.latent_features = int(latent_features)
        self.device = torch.device(device)
        super().__init__(
            batch_shape=torch.Size([self.batch_size]),
            event_shape=torch.Size([self.latent_features]),
            validate_args=False,
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        multiplier = int(np.prod(sample_shape)) if len(sample_shape) else 1
        values = self.boson_sampler(multiplier * self.batch_size).to(self.device)
        if len(sample_shape):
            return values.reshape(*sample_shape, self.batch_size, self.latent_features)
        return values

    def log_prob(self, value):
        raise NotImplementedError(
            "The implicit Sparrow Boson prior has no tractable density. Use MMD prior matching."
        )
