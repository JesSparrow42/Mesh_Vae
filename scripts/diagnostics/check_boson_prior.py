#!/usr/bin/env python3
"""Fast integrity check for the official Sparrow Boson latent prior."""
from __future__ import annotations

# Allow direct execution as well as ``python -m`` from the repository checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
import importlib
import json
import time

import numpy as np
import torch

from meshvae.boson import (
    BosonSamplerTorch,
    _load_sparrow_sampler,
    _normalize_boson_cfg,
)


def _check_hong_ou_mandel(sampler_path: str | None) -> None:
    _load_sparrow_sampler(sampler_path)
    core = importlib.import_module("bosonsampler.core")
    unitary = np.asarray(
        [[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128
    ) / np.sqrt(2.0)
    states, probabilities = core._linear_distribution(
        unitary, np.asarray([1, 1], dtype=int)
    )
    observed = {
        tuple(int(value) for value in state): float(probability)
        for state, probability in zip(states, probabilities)
    }
    expected = {(2, 0): 0.5, (1, 1): 0.0, (0, 2): 0.5}
    for state, probability in expected.items():
        if not np.isclose(observed.get(state, 0.0), probability, atol=1e-6):
            raise RuntimeError(
                "Sparrow Hong-Ou-Mandel check failed: "
                f"state={state} observed={observed.get(state, 0.0)} "
                f"expected={probability}; full_distribution={observed}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--boson_params", default="{}")
    parser.add_argument("--boson_sampler_path", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    raw = json.loads(args.boson_params)
    if not isinstance(raw, dict):
        raise ValueError("--boson_params must be a JSON object.")
    if args.boson_sampler_path:
        raw["sampler_path"] = args.boson_sampler_path
    config = _normalize_boson_cfg(raw, args.latent_dim)

    np.random.seed(12345)
    torch.manual_seed(12345)
    _check_hong_ou_mandel(config.get("sampler_path"))

    started = time.perf_counter()
    sampler = BosonSamplerTorch(**config).to(torch.device(args.device))
    build_seconds = time.perf_counter() - started
    samples = sampler(args.samples).detach().cpu().numpy()
    if samples.shape != (args.samples, args.latent_dim):
        raise RuntimeError(f"Unexpected sample shape: {samples.shape}")
    if not np.isfinite(samples).all():
        raise RuntimeError("Sparrow Boson prior produced non-finite samples.")

    complete_blocks = args.latent_dim // sampler.m
    if complete_blocks:
        blocks = samples[:, : complete_blocks * sampler.m].reshape(
            args.samples, complete_blocks, sampler.m
        )
        photon_counts = blocks.sum(axis=2)
        if float(config["input_loss"]) == 0.0 and not np.allclose(
            photon_counts, sampler.num_sources
        ):
            raise RuntimeError(
                "Physical Fock blocks do not preserve the configured photon count: "
                f"observed range=({photon_counts.min()}, {photon_counts.max()}) "
                f"expected={sampler.num_sources}."
            )

    unique_rows = int(np.unique(samples, axis=0).shape[0])
    std_mean = float(samples.std(axis=0).mean())
    print("\n========== SPARROW BOSON PRIOR PREFLIGHT ==========", flush=True)
    print(f"source={sampler.source_file}", flush=True)
    print(f"version={sampler.source_version} api={sampler.backend_api}", flush=True)
    print(f"shape={samples.shape} unique_rows={unique_rows}/{len(samples)}", flush=True)
    print(
        f"physical_modes={sampler.m} photons={sampler.num_sources} "
        f"latent_blocks={sampler.num_blocks} mean_dimension_std={std_mean:.4f} "
        f"build_seconds={build_seconds:.2f}",
        flush=True,
    )
    printable_config = {key: value for key, value in config.items() if key != "sampler_path"}
    print(f"config={json.dumps(printable_config, sort_keys=True)}", flush=True)
    if unique_rows < max(16, args.samples // 10):
        raise RuntimeError(
            "Sparrow prior has too few unique latent vectors. Increase num_loops "
            "or use more physical modes."
        )
    print(
        "[PASS] Official Sparrow HOM probabilities, Fock photon counts, shape, "
        "finiteness, and diversity checks passed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
