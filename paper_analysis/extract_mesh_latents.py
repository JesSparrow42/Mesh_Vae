#!/usr/bin/env python3
"""Extract deterministic MeshVAE posterior means for train and held-out test.

Run this in the same Python environment as training on Computerome or Gefion. The output
CSV files are the inputs to ``run_modality_benchmark.py`` and
``analyze_latent_signal.py``.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="Checkpoint experiment name (single-model mode).")
    parser.add_argument("--dataroot", type=Path)
    parser.add_argument("--checkpoints-dir", type=Path)
    parser.add_argument("--train-tsv", type=Path)
    parser.add_argument("--test-tsv", type=Path)
    parser.add_argument("--gender", choices=("F", "M"))
    parser.add_argument("--study-root", action="append", type=Path,
                        help="Batch completed NEXT studies; repeat for checkpoint roots.")
    parser.add_argument("--platform", choices=("computerome", "gefion"),
                        help="Optional platform filter in batch mode; otherwise infer from run name.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Opt in to reusing existing train/test embeddings and their manifest without freshness checks.")
    parser.add_argument("--which-epoch", default="best")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        help="Single-model embeddings, or batch inventory directory (default: latent_extraction).")
    return parser


def extract_studies(args, passthrough):
    """Reuse single-model extraction for studies with known saved inputs only."""
    inventory, seen = [], set()
    for root in args.study_root:
        root = root.expanduser().resolve()
        if not root.is_dir():
            inventory.append({"study": str(root), "status": "failed", "error": "Missing checkpoint root"})
            continue
        for path in sorted(root.rglob("manifest.json")):
            run = path.parent.parent
            if path.parent.name != "study" or "_NEXT_" not in run.name or path in seen:
                continue
            seen.add(path)
            platform = run.name.split("_NEXT_", 1)[0].lower()
            if args.platform and platform != args.platform:
                continue
            row = {"study": str(run), "platform": platform, "status": "failed", "error": ""}
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                status = path.parent / "status.json"
                state = json.loads(status.read_text(encoding="utf-8")).get("status") if status.is_file() else "missing_status"
                if state != "completed":
                    row.update(status="skipped_incomplete", error=f"Study status: {state}")
                    inventory.append(row)
                    print(f"[SKIP] {run.name}: {row['error']}", flush=True)
                    continue
                dataroot = manifest["config"]["platforms"][platform]["dataroot"]
                gender = manifest["gender"]
                components = ("presence", "positive") if manifest["experiment"]["kind"] == "hurdle" else (None,)
                for component in components:
                    name = run.name + (f"_{component}" if component else "")
                    inputs = path.parent / component if component else path.parent
                    model_run = run.parent / name
                    output = model_run / "latents"
                    entry = {**row, "model": name, "train_tsv": str(inputs / "train.tsv"),
                             "test_tsv": str(inputs / "test.tsv"), "output_dir": str(output)}
                    required = [inputs / "train.tsv", inputs / "test.tsv", model_run / f"{args.which_epoch}_vae_net.pth"]
                    missing = [str(file) for file in required if not file.is_file()]
                    existing = [output / name for name in ("train_latent_embeddings.csv", "test_latent_embeddings.csv", "latent_manifest.json")]
                    if missing:
                        entry.update(status="skipped_missing_inputs", error="Missing: " + "; ".join(missing))
                    elif args.skip_existing and all(file.is_file() for file in existing):
                        entry.update(status="reused_explicitly", error="Freshness not checked (--skip-existing)")
                    else:
                        cmd = [sys.executable, str(Path(__file__).resolve()), "--name", name,
                               "--checkpoints-dir", str(run.parent), "--dataroot", str(dataroot),
                               "--train-tsv", entry["train_tsv"], "--test-tsv", entry["test_tsv"],
                               "--gender", gender, "--which-epoch", args.which_epoch, "--gpu-ids", args.gpu_ids,
                               "--batch-size", str(args.batch_size), "--output-dir", str(output),
                               "--cond_stregkode_col", "participant_id", "--cond_age_col", "study_age",
                               "--cond_gender_col", "study_gender", *passthrough]
                        print(f"[EXTRACT] {name}", flush=True)
                        result = subprocess.run(cmd, check=False)
                        entry.update(status="completed" if result.returncode == 0 else "failed",
                                     error="" if result.returncode == 0 else f"Extractor exited {result.returncode}")
                    inventory.append(entry)
                    print(f"[{entry['status'].upper()}] {name}: {entry['error']}", flush=True)
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                inventory.append(row)
                print(f"[FAILED] {run.name}: {row['error']}", flush=True)
    output = (args.output_dir or Path("latent_extraction")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory, columns=["study", "platform", "model", "status", "error", "train_tsv", "test_tsv", "output_dir"]).to_csv(output / "latent_extraction_inventory.csv", index=False)
    print(f"[OUT] {output / 'latent_extraction_inventory.csv'}", flush=True)
    if any(row["status"] == "failed" for row in inventory):
        raise SystemExit("Some extractions failed; inspect latent_extraction_inventory.csv.")
    if not inventory:
        print("[CHECK] No supported NEXT studies found in the requested roots.", flush=True)


def plotting_context(opt, net):
    """Carry the trained encoder inputs and plaque columns into portable exports."""
    conditioned = net.cond_dim > 0 and net.encoder_cond_mode != "none"
    label = "Scan + age latent representation" if conditioned else "Scan-only latent representation"
    if conditioned and getattr(opt, "cond_weight", False):
        label = "Scan + age + weight latent representation"
    target, volume = "anyplaq", None
    if getattr(opt, "aux_predict_binary", False):
        target = getattr(opt, "aux_binary_col", None) or target
        if target.startswith("presence_"):
            volume = "volume_" + target[len("presence_"):]
    elif getattr(opt, "aux_predict_regression", False):
        column = getattr(opt, "aux_regression_col", "") or ""
        if column.startswith("volume_"):
            volume, target = column, "presence_" + column[len("volume_"):]
        elif column.startswith("plaque_volume"):
            volume = column
    return {"representation_label": label, "id_col": getattr(opt, "cond_stregkode_col", None),
            "age_col": getattr(opt, "cond_age_col", None), "gender_col": getattr(opt, "cond_gender_col", None),
            "target_col": target, "volume_col": volume}


def main() -> None:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    single = ("name", "dataroot", "checkpoints_dir", "train_tsv", "test_tsv", "gender")
    if args.study_root:
        if any(getattr(args, field) is not None for field in single):
            parser.error("--study-root cannot be combined with single-model input options")
        extract_studies(args, passthrough)
        return
    required = [field for field in (*single, "output_dir") if getattr(args, field) is None]
    if required:
        parser.error("Single-model mode requires: " + ", ".join("--" + field.replace("_", "-") for field in required))
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        from meshvae.conditioning import clean_stregkode
        from meshvae.datasets import PreprocessedMeshVAEDataset, mesh_collate_fn
        from meshvae.model import MeshVAEModel
        from meshvae.options import TestOptions
    except ImportError as error:
        raise ImportError(
            "Run this extractor in the same Python environment as training on Computerome or Gefion. "
            f"The import that failed was: {error}"
        ) from error

    saved_argv = sys.argv[:]
    sys.argv = [
        "extract_mesh_latents",
        "--name",
        args.name,
        "--dataroot",
        str(args.dataroot),
        "--checkpoints_dir",
        str(args.checkpoints_dir),
        "--cond_tsv",
        str(args.train_tsv),
        "--feature_source",
        "scan",
        "--only_gender",
        args.gender,
        "--gpu_ids",
        args.gpu_ids,
        "--which_epoch",
        args.which_epoch,
        "--batch_size",
        str(args.batch_size),
        *passthrough,
    ]
    try:
        opt = TestOptions().parse()
    finally:
        sys.argv = saved_argv

    opt.cond_tsv = str(args.train_tsv)
    train_dataset = PreprocessedMeshVAEDataset(str(args.dataroot), opt)
    model = MeshVAEModel(opt)
    model.load_network(args.which_epoch)
    model.eval()
    net = model.net.module if isinstance(model.net, torch.nn.DataParallel) else model.net
    net.eval()

    def encoder_condition(full_condition):
        if net.cond_dim <= 0 or net.encoder_cond_mode == "none":
            return None
        if net.encoder_cond_mode == "demo":
            return full_condition[:, : net.demo_cond_dim]
        raise ValueError(f"Unknown encoder condition mode: {net.encoder_cond_mode}")

    @torch.no_grad()
    def extract(dataset, split_name: str) -> pd.DataFrame:
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=0,
            collate_fn=mesh_collate_fn,
        )
        rows = []
        for batch_index, batch in enumerate(loader, start=1):
            model.set_input(batch)
            condition = encoder_condition(model.cond)
            _, mean = net.posterior(model.edge_features, model.mesh_batch, cond=condition)
            latent = mean.detach().cpu().numpy()
            for path, vector in zip(batch["path"], latent):
                row = {
                    "participant_id": clean_stregkode(path) or str(path),
                    "split": split_name,
                }
                row.update({f"latent_{index:03d}": float(value) for index, value in enumerate(vector)})
                rows.append(row)
            if batch_index % 100 == 0 or batch_index == len(loader):
                print(
                    f"[{split_name}] batch {batch_index}/{len(loader)} "
                    f"({100.0 * batch_index / len(loader):.1f}%)",
                    flush=True,
                )
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError(f"No latent vectors were extracted for {split_name}.")
        if frame["participant_id"].duplicated().any():
            examples = frame.loc[
                frame["participant_id"].duplicated(False), "participant_id"
            ].head().tolist()
            raise ValueError(f"Duplicate participant IDs in {split_name}: {examples}")
        return frame

    train = extract(train_dataset, "train")
    del train_dataset
    gc.collect()

    opt.cond_tsv = str(args.test_tsv)
    test_dataset = PreprocessedMeshVAEDataset(str(args.dataroot), opt)
    test = extract(test_dataset, "test")
    del test_dataset
    gc.collect()

    overlap = sorted(set(train["participant_id"]) & set(test["participant_id"]))
    if overlap:
        raise ValueError(f"Train/test leakage in extracted IDs (examples: {overlap[:10]}).")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_latent_embeddings.csv"
    test_path = output_dir / "test_latent_embeddings.csv"
    combined_path = output_dir / "latent_embeddings.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    pd.concat([train, test], ignore_index=True).to_csv(combined_path, index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.name,
        "checkpoint": args.which_epoch,
        "gender": args.gender,
        "dataroot": str(args.dataroot),
        "train_tsv": str(args.train_tsv),
        "test_tsv": str(args.test_tsv),
        "n_train": len(train),
        "n_test": len(test),
        "latent_dimensions": len([column for column in train if column.startswith("latent_")]),
        "representation": "deterministic posterior mean (mu)",
        **plotting_context(opt, net),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    (output_dir / "latent_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[OUT] train latent -> {train_path}", flush=True)
    print(f"[OUT] test latent  -> {test_path}", flush=True)
    print(f"[OUT] combined     -> {combined_path}", flush=True)


if __name__ == "__main__":
    main()
