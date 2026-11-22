#!/usr/bin/env python3
"""Backward-compatible wrapper for the refactored geometry helpers."""

# Allow direct execution as well as ``python -m`` from the repository checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from meshvae.geometry import *  # noqa: F401,F403
