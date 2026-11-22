from __future__ import annotations

import os
from pathlib import Path

from .options import TrainOptions
from .geometry import preprocess_obj_folder
from .model import MeshVAEModel
from .datasets import MeshVAEDataset, PreprocessedMeshVAEDataset
from .training import train, run_test, sample_conditional, train_baseline_phase
from .age_bins import search_best_age_bins


def main() -> None:

    opt = TrainOptions().parse()

    os.makedirs(Path(opt.checkpoints_dir) / opt.name, exist_ok=True)

    if opt.phase == "preprocess":
        n_points = opt.n_points if getattr(opt, "n_points", 0) > 0 else None
        preprocess_roots = getattr(opt, "preprocess_roots", None) or [opt.dataroot]
        print("\n========== PREPROCESS (offline) ==========\n")
        print(f"[PREPROC] output dataroot: {opt.dataroot}")
        print(f"[PREPROC] input roots: {preprocess_roots}")
        preprocess_obj_folder(
            input_root=preprocess_roots,
            out_root=opt.dataroot,
            target_tris=opt.target_tris,
            save_npz=True,
            n_points=n_points,
            existing_npz=getattr(opt, "preprocess_existing_npz", None),
            dedupe_by_id=bool(getattr(opt, "preprocess_dedupe_by_id", True)),
            assume_simplified=bool(getattr(opt, "preprocess_assume_simplified", False)),
            point_sample_method=str(getattr(opt, "point_sample_method", "poisson")),
        )
        print("\nPreprocessing complete. You can now run with --use_preprocessed.\n")

    elif opt.phase == "train":
        opt.is_train = True
        train(opt)

    elif opt.phase == "test":
        opt.is_train = False
        model = MeshVAEModel(opt)
        dataset = PreprocessedMeshVAEDataset(opt.dataroot, opt) if getattr(opt, "use_preprocessed", False) else MeshVAEDataset(opt.dataroot, opt)
        model.load_network(opt.which_epoch)
        run_test(opt, model, dataset)

    elif opt.phase == "sample":
        opt.is_train = False
        sample_conditional(opt)

    elif opt.phase == "search_age_bins":
        opt.is_train = True
        search_best_age_bins(opt)

    elif opt.phase == "train_baseline":
        opt.is_train = False
        train_baseline_phase(opt)


if __name__ == "__main__":
    main()
