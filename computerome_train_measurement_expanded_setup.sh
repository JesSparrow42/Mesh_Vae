#!/usr/bin/env bash
(
set -euo pipefail
cd /data/projects/3D_scans/ML/MQVAE_new
PY=/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python

cat > .train_body_measurements_expanded.py.incoming <<'EXPANDED_MEASUREMENTS_SOURCE_20261027'
#!/usr/bin/env python3
"""Expand the labelled scan cohort using the existing measurement trainers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from body_measurements.data import align_training, clean_id, load_features, read_metadata
from run_body_measurements_pointcloud import EXCLUDED_IDS, METRES, load_baseline, raw_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--old-baseline", default="measurement_runs/n2894_baseline_01")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root, output = Path(args.obj_root).resolve(), Path(args.out_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Scan folder is missing: {root}")
    if output.exists():
        raise ValueError(f"Use a new output folder; already exists: {output}")
    if args.epochs < 1 or args.patience < 1 or args.points < 128 or args.bootstrap < 0:
        raise ValueError("epochs/patience must be positive, points >=128, bootstrap >=0")
    _, run, sources, old_X, _, old_groups, old_splits, _ = load_baseline(args.old_baseline)
    metadata = read_metadata(sources["metadata"])
    targets = run["targets"]
    inventory = pd.DataFrame(
        [(clean_id(p.name), str(p)) for p in sorted(root.rglob("*"))
         if p.is_file() and p.suffix.lower() == ".obj"], columns=["scan_id", "path"])
    if inventory.empty:
        raise ValueError(f"No OBJ files found under {root}")
    inventory["ambiguous_id"] = inventory.scan_id.duplicated(keep=False)
    unique = inventory.loc[inventory.scan_id.notna() & ~inventory.ambiguous_id
                           & ~inventory.scan_id.isin(EXCLUDED_IDS)]
    paths = dict(zip(unique.scan_id, unique.path))
    # Preserve the actual scan sources used in the old evaluation, including
    # scans whose other exports are duplicated in the new collection.
    paths.update(raw_paths(sources["features"], list(old_X.index), None))
    stub = pd.DataFrame({"inventory_only": 0.0}, index=pd.Index(sorted(paths), name="scan_id"))
    selected, _, _, _ = align_training(stub, metadata, targets, run["id_col"], run.get("participant_col"))
    output.mkdir(parents=True)
    inventory.to_csv(output / "obj_inventory.csv", index=False)
    view = output / "labelled_obj_sources"
    view.mkdir()
    for code in selected.index:
        (view / f"{code}.obj").symlink_to(Path(paths[code]).resolve())
    pd.DataFrame({"scan_id": selected.index,
                  "source_path": [str(Path(paths[c]).resolve()) for c in selected.index],
                  "original_cohort": selected.index.isin(old_X.index)}).to_csv(
                      output / "source_selection.csv", index=False)
    print(f"[SELECT] {len(inventory)} OBJ files; {len(selected)} labelled scan IDs; "
          f"{len(old_X)} original scan sources preserved", flush=True)

    code_root = Path(__file__).resolve().parent
    def execute(script, *arguments):
        command = [sys.executable, "-u", str(code_root / script), *map(str, arguments)]
        print("[RUN] " + " ".join(command), flush=True)
        subprocess.run(command, check=True)

    execute("run_body_measurements.py", "build-features", "--obj-root", view,
            "--workers", 2, "--out-dir", output / "features")
    features = output / "features" / "scan_geometry_features.npz"
    X, feature_audit = load_features(features)
    X, y, groups, alignment = align_training(X, metadata, targets, run["id_col"], run.get("participant_col"))
    missing = old_X.index.difference(X.index)
    if len(missing):
        raise ValueError(f"Original cohort lost scans; training stopped: {list(missing[:10])}")
    if list(X.columns) != list(old_X.columns) or not np.allclose(
            X.loc[old_X.index].to_numpy(), old_X.to_numpy(), rtol=1e-5, atol=1e-6):
        raise ValueError("Original scan geometry changed; training stopped before altering the comparison")
    if not groups.loc[old_groups.index].equals(old_groups):
        raise ValueError("Original participant identities changed")
    added = ~X.index.isin(old_X.index)
    axes = [f"bbox_extent_{axis}" for axis in "xyz"]
    expected_axis = old_X[axes].idxmax(axis=1).mode().iat[0]
    size_m = X.bbox_long * METRES[run["scan_units"]]
    unusual = added & (~size_m.between(0.8, 3.0) | X[axes].idxmax(axis=1).ne(expected_axis))
    if unusual.any():
        X.loc[unusual, axes + ["bbox_long"]].to_csv(output / "scan_format_review.csv")
        raise ValueError("Some added scans have unexpected physical size or upright axis. "
                         "Training stopped; inspect scan_format_review.csv. No automatic rescaling/rotation.")
    mapping = dict(zip(old_groups, old_splits))
    new_people = sorted(set(groups) - set(mapping))
    shuffled = np.random.default_rng(42).permutation(new_people)
    mapping.update({person: "validation" if i % 5 == 0 else "train" for i, person in enumerate(shuffled)})
    splits = groups.map(mapping)
    if not splits.loc[old_splits.index].equals(old_splits.rename(splits.name)):
        raise ValueError("Original split assignments changed")
    manifest = pd.DataFrame({"scan_id": X.index, "participant_id": groups.to_numpy(), "split": splits.to_numpy()})
    split_file = output / "expanded_splits.csv"
    manifest.to_csv(split_file, index=False)
    (output / "expansion_audit.json").write_text(json.dumps({
        "old_baseline": str(Path(args.old_baseline).resolve()), "scan_root": str(root),
        "selected_scan_ids": len(selected), "features": feature_audit, "alignment": alignment,
        "split_counts": splits.value_counts().to_dict(), "new_participants": len(new_people),
        "original_assignments_preserved": True, "scan_units": run["scan_units"],
        "format_check": "Broad size and dominant-axis screen only; cannot verify pose or left/right orientation"
    }, indent=2) + "\n")
    print("[COHORT]", len(X), "valid labelled scans; splits:", splits.value_counts().to_dict(), flush=True)
    common = ["--metadata", sources["metadata"], "--id-col", run["id_col"],
              "--config", sources["config"], "--features", features]
    if run.get("participant_col"):
        common += ["--participant-col", run["participant_col"]]
    baseline, cache, model = output / "baseline", output / "point_cache_2048", output / "pointcloud"
    execute("run_body_measurements.py", "train", *common, "--scan-units", run["scan_units"],
            "--splits", split_file, "--seed", 42, "--n-jobs", 2, "--out-dir", baseline)
    execute("run_body_measurements_pointcloud.py", "build-cache", "--baseline-run", baseline,
            "--points", args.points, "--workers", 2, "--out-dir", cache)
    execute("run_body_measurements_pointcloud.py", "train", "--baseline-run", baseline,
            "--cache", cache, "--device", args.device, "--batch-size", 8, "--threads", 2,
            "--epochs", args.epochs, "--patience", args.patience, "--bootstrap", args.bootstrap,
            "--out-dir", model)
    print("[DONE] New model:", model, flush=True)
    print("[RESULTS]", model / "test_metrics.csv", flush=True)


if __name__ == "__main__":
    main()
EXPANDED_MEASUREMENTS_SOURCE_20261027

"$PY" - <<'EXPANDED_MEASUREMENTS_VERIFY_20261027'
from pathlib import Path
from datetime import datetime
import hashlib
import os
import shutil

incoming = Path('.train_body_measurements_expanded.py.incoming')
if hashlib.sha256(incoming.read_bytes()).hexdigest() != '163ea0d8eb6ebd78949fd9c16f604178b6b0999fd29cc4f3bbf1bd51ed09eb34':
    raise SystemExit('The pasted code is incomplete or changed. Paste the entire setup file again.')
compile(incoming.read_text(encoding='utf-8'), 'train_body_measurements_expanded.py', 'exec')
destination = Path('train_body_measurements_expanded.py')
if destination.exists():
    shutil.copy2(destination, str(destination) + '.backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
os.replace(incoming, destination)
print('[INSTALLED]', destination.resolve())
EXPANDED_MEASUREMENTS_VERIFY_20261027

"$PY" train_body_measurements_expanded.py --help
)
