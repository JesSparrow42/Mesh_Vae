#!/usr/bin/env python3
"""Export genuine held-out input/reconstruction pairs from a trained MeshVAE.

The default selection is deliberately non-cherry-picked: for each binary target
class, choose the held-out participant whose reconstruction Chamfer distance is
closest to that class's median.  The output bundle can be rendered by
``make_reconstruction_figure.py`` without loading PyTorch or the checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Direct execution puts paper_analysis/, not the repository root, on sys.path.
# Resolve relative to this file so imports also work outside the checkout cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class Candidate:
    dataset_index: int
    participant_id: str
    target: int
    probability: float
    chamfer: float
    source_path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Checkpoint experiment name.")
    parser.add_argument("--dataroot", required=True, type=Path)
    parser.add_argument("--checkpoints-dir", required=True, type=Path)
    parser.add_argument("--test-tsv", required=True, type=Path)
    parser.add_argument("--gender", choices=("F", "M"), required=True)
    parser.add_argument("--which-epoch", default="best")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--examples-per-class", type=int, default=1)
    parser.add_argument(
        "--participant-id",
        action="append",
        default=[],
        help="Export this held-out participant instead of median-by-class selection. Repeatable.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def scalar(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=float).reshape(-1)
    return float(array[0]) if array.size else default


def select_representatives(
    candidates: list[Candidate],
    examples_per_class: int,
    requested_ids: list[str] | None = None,
) -> tuple[list[Candidate], str]:
    requested = [str(value).strip().upper() for value in (requested_ids or []) if str(value).strip()]
    if requested:
        by_id = {candidate.participant_id.upper(): candidate for candidate in candidates}
        missing = [participant_id for participant_id in requested if participant_id not in by_id]
        if missing:
            raise ValueError(f"Requested held-out participant IDs were not found: {missing}")
        return [by_id[participant_id] for participant_id in requested], "requested_participant_ids"

    selected: list[Candidate] = []
    count = max(1, int(examples_per_class))
    for target in (0, 1):
        group = [
            candidate
            for candidate in candidates
            if candidate.target == target and np.isfinite(candidate.chamfer)
        ]
        if not group:
            raise ValueError(f"No finite held-out reconstruction candidates for target={target}.")
        median = float(np.median([candidate.chamfer for candidate in group]))
        group.sort(key=lambda candidate: (abs(candidate.chamfer - median), candidate.participant_id))
        selected.extend(group[:count])
    return selected, "closest_to_class_median_chamfer"


def write_ascii_ply(path: Path, points: np.ndarray) -> None:
    finite = np.asarray(points, dtype=np.float32)
    finite = finite[np.isfinite(finite).all(axis=1)]
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(finite)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("end_header\n")
        np.savetxt(handle, finite, fmt="%.7g %.7g %.7g")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args, passthrough = build_parser().parse_known_args()
    try:
        import torch
        from torch.utils.data import DataLoader

        from meshvae.conditioning import clean_stregkode
        from meshvae.datasets import PreprocessedMeshVAEDataset, mesh_collate_fn
        from meshvae.model import MeshVAEModel
        from meshvae.options import TestOptions
    except ImportError as error:
        raise ImportError(
            "Run this exporter in the same Python environment as training on Computerome or Gefion. "
            f"The import that failed was: {error}"
        ) from error

    saved_argv = sys.argv[:]
    sys.argv = [
        "export_model_reconstructions",
        "--phase",
        "test",
        "--name",
        args.name,
        "--dataroot",
        str(args.dataroot),
        "--checkpoints_dir",
        str(args.checkpoints_dir),
        "--cond_tsv",
        str(args.test_tsv),
        "--feature_source",
        "scan",
        "--only_gender",
        args.gender,
        "--gpu_ids",
        args.gpu_ids,
        "--which_epoch",
        args.which_epoch,
        "--batch_size",
        "1",
        "--num_threads",
        "0",
        "--use_preprocessed",
        "--use_pointcloud",
        "--no-use_cond",
        *passthrough,
    ]
    try:
        opt = TestOptions().parse()
    finally:
        sys.argv = saved_argv

    opt.cond_tsv = str(args.test_tsv)
    opt.is_train = False
    if not bool(getattr(opt, "use_pointcloud", False)):
        raise ValueError("This exporter expects a point-cloud checkpoint.")

    dataset = PreprocessedMeshVAEDataset(str(args.dataroot), opt)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=mesh_collate_fn,
    )
    model = MeshVAEModel(opt)
    model.load_network(args.which_epoch)
    model.eval()

    candidates: list[Candidate] = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            model.set_input(batch)
            result = model.test()
            diagnostics = result.get("diagnostics", {})
            path = str(batch["path"][0])
            participant_id = clean_stregkode(path) or Path(path).stem
            target = int(round(scalar(batch.get("binary_target"))))
            candidates.append(
                Candidate(
                    dataset_index=index,
                    participant_id=str(participant_id),
                    target=target,
                    probability=scalar(result.get("binary_prob")),
                    chamfer=scalar(diagnostics.get("chamfer")),
                    source_path=path,
                )
            )
            if (index + 1) % 50 == 0 or index + 1 == len(loader):
                print(f"[SCORE] {index + 1}/{len(loader)} held-out reconstructions", flush=True)

    selected, selection_method = select_representatives(
        candidates,
        args.examples_per_class,
        args.participant_id,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for example_index, candidate in enumerate(selected):
            batch = mesh_collate_fn([dataset[candidate.dataset_index]])
            model.set_input(batch)
            result = model.test()
            original = result["x"][0, :3, :].permute(1, 0).numpy().astype(np.float32)
            reconstruction = (
                result["x_mean"][0, :3, :].permute(1, 0).numpy().astype(np.float32)
            )
            key = f"example_{example_index:02d}"
            arrays[f"{key}_input"] = original
            arrays[f"{key}_reconstruction"] = reconstruction
            write_ascii_ply(output_dir / f"{key}_input.ply", original)
            write_ascii_ply(output_dir / f"{key}_reconstruction.ply", reconstruction)
            manifest_rows.append(
                {
                    "example_key": key,
                    "participant_id": candidate.participant_id,
                    "target": candidate.target,
                    "target_label": "plaque-positive" if candidate.target == 1 else "plaque-free",
                    "predicted_probability": candidate.probability,
                    "chamfer": candidate.chamfer,
                    "selection_method": selection_method,
                    "split": "held-out test",
                    "source_path": candidate.source_path,
                }
            )
            print(
                f"[EXPORT] {key}: target={candidate.target} "
                f"chamfer={candidate.chamfer:.6g} participant={candidate.participant_id}",
                flush=True,
            )

    bundle_path = output_dir / "reconstruction_bundle.npz"
    manifest_path = output_dir / "reconstruction_manifest.csv"
    np.savez_compressed(bundle_path, **arrays)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    checkpoint_path = args.checkpoints_dir / args.name / f"{args.which_epoch}_vae_net.pth"
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.name,
        "checkpoint": args.which_epoch,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "gender": args.gender,
        "test_tsv": str(args.test_tsv),
        "dataroot": str(args.dataroot),
        "selection_method": selection_method,
        "n_scored": len(candidates),
        "n_exported": len(selected),
        "deterministic_reconstruction": "posterior mean",
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    (output_dir / "reconstruction_export.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[OUT] point-cloud bundle -> {bundle_path}", flush=True)
    print(f"[OUT] selection manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
