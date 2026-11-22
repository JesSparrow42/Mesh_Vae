#!/usr/bin/env bash
# Run from a GPU node, using the same interpreter as the established training command.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
exec /dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python \
  paper_analysis/body_composition_controls.py train "$@"
