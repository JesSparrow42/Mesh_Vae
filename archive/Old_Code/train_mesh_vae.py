#!/usr/bin/env python3
"""Backward-compatible entry point for the refactored MeshVAE training code."""
from meshvae.options import BaseOptions, TrainOptions, TestOptions  # noqa: F401
from meshvae.conditioning import *  # noqa: F401,F403
from meshvae.datasets import MeshVAEDataset, PreprocessedMeshVAEDataset, mesh_collate_fn  # noqa: F401
from meshvae.train_eval import evaluate_subset, split_dataset, redetail_to_simplified_edges  # noqa: F401
from meshvae.age_bins import *  # noqa: F401,F403
from meshvae.training import train, train_baseline_phase, run_test, sample_conditional  # noqa: F401
from meshvae.cli import main

if __name__ == "__main__":
    main()
