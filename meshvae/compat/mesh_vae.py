#!/usr/bin/env python3
"""Backward-compatible wrapper for the refactored MeshVAE model package."""

# Allow direct execution as well as ``python -m`` from the repository checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from meshvae.boson import *  # noqa: F401,F403
from meshvae.graph_blocks import *  # noqa: F401,F403
from meshvae.networks import *  # noqa: F401,F403
from meshvae.losses import *  # noqa: F401,F403
from meshvae.vi import *  # noqa: F401,F403
from meshvae.model import *  # noqa: F401,F403
