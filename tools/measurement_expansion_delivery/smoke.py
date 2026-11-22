"""Run the expansion workflow on synthetic scans, with an independent split check."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from build_scan_geometry_features import FEATURE_NAMES, parse_obj_vertices
from run_body_measurements import fingerprint
import train_body_measurements_expanded as workflow


def command(*args):
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True, cwd=REPO)


with tempfile.TemporaryDirectory(prefix="measurement_expansion_") as tmp:
    root = Path(tmp)
    original, collection = root / "original", root / "collection"
    original.mkdir()
    collection.mkdir()
    ids = [f"scan_{i:03}" for i in range(60)]
    rng = np.random.default_rng(319)
    height = rng.uniform(1.55, 1.95, 60)
    width = rng.uniform(.35, .6, 60)
    rows = []
    old_features, old_sources = [], []
    for i, code in enumerate(ids):
        cloud = rng.uniform(-.5, .5, (128, 3)) * [width[i], height[i], .3]
        cloud[:2] = [[-width[i]/2, -height[i]/2, -.15], [width[i]/2, height[i]/2, .15]]
        text = "".join(f"v {x:.12g} {y:.12g} {z:.12g}\n" for x, y, z in cloud)
        (collection / f"{code}.obj").write_text(text)
        if i < 40:
            path = original / f"{code}.obj"
            path.write_text(text)
            parsed = parse_obj_vertices(str(path))
            old_features.append([parsed[key] for key in FEATURE_NAMES])
            old_sources.append(str(path))
            # This other export must never replace the original test input.
            (collection / f"{code}.obj").write_text("".join(
                f"v {10*x:.12g} {10*y:.12g} {10*z:.12g}\n" for x, y, z in cloud))
        rows.append({"scan_id": code, "person_id": f"person_{i//2:03}",
                     "height_cm": height[i]*100, "waist_cm": 25 + width[i]*130})
    (collection / "another_export").mkdir()
    shutil.copy2(collection / "scan_059.obj", collection / "another_export" / "scan_059.obj")
    shutil.copy2(collection / "scan_058.obj", collection / "scan_999.obj")
    shutil.copy2(collection / "scan_058.obj", collection / "RIGDET050925.obj")
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
    (root / "targets.json").write_text(json.dumps({"targets": {
        "body_height": {"column": "height_cm", "unit": "cm"},
        "waist_circumference": {"column": "waist_cm", "unit": "cm"}}}))
    np.savez_compressed(root / "old_geometry.npz", names=np.asarray(ids[:40], dtype=object),
                        features=np.asarray(old_features), feature_names=np.asarray(FEATURE_NAMES, dtype=object),
                        source_paths=np.asarray(old_sources, dtype=object))
    old_splits = pd.DataFrame({"scan_id": ids[:40], "participant_id": [r["person_id"] for r in rows[:40]],
                               "split": ["train"]*24 + ["validation"]*8 + ["test"]*8})
    old_splits.to_csv(root / "old_splits.csv", index=False)
    baseline = root / "baseline"
    command(REPO / "run_body_measurements.py", "train", "--metadata", root / "metadata.csv",
            "--id-col", "scan_id", "--participant-col", "person_id", "--config", root / "targets.json",
            "--features", root / "old_geometry.npz", "--splits", root / "old_splits.csv",
            "--scan-units", "m", "--n-jobs", "2", "--out-dir", baseline)
    before = {file: fingerprint(str(baseline / file))["sha256"]
              for file in ("model.joblib", "splits.csv", "run.json")}
    output = root / "expanded"
    argv = ["train_body_measurements_expanded.py", "--obj-root", str(collection),
            "--old-baseline", str(baseline), "--out-dir", str(output),
            "--epochs", "1", "--points", "128", "--bootstrap", "20", "--device", "cpu"]
    # Windows symlink creation needs OS privileges unavailable in some setups.
    # Copies preserve all OBJ bytes for the local test; Computerome uses symlinks.
    def link_for_test(destination, target, *args, **kwargs):
        shutil.copy2(target, destination)
    with patch.object(sys, "argv", argv), patch.object(Path, "symlink_to", link_for_test):
        workflow.main()
    expanded = pd.read_csv(output / "expanded_splits.csv")
    joined = old_splits.merge(expanded, on="scan_id", validate="one_to_one", suffixes=("_old", "_new"))
    assert len(joined) == 40 and joined.split_old.equals(joined.split_new)
    assert joined.participant_id_old.equals(joined.participant_id_new)
    assert expanded.groupby("participant_id").split.nunique().max() == 1
    assert set(expanded.loc[expanded.split.eq("test"), "scan_id"]) == set(ids[32:40])
    assert len(expanded) == 59
    assert not {"scan_059", "scan_999", "RIGDET050925"}.intersection(expanded.scan_id)
    selection = pd.read_csv(output / "source_selection.csv")
    assert all(Path(path).resolve().parent == original.resolve()
               for path in selection.loc[selection.original_cohort, "source_path"])
    assert before == {file: fingerprint(str(baseline / file))["sha256"] for file in before}
    metrics = pd.read_csv(output / "pointcloud" / "test_metrics.csv")
    assert metrics.n.tolist() == [8, 8]
    assert np.isfinite(metrics.mae_cm).all()
    assert json.loads((output / "pointcloud" / "run.json").read_text())["status"] == "complete"
    receipt = {"status": "passed", "original_scans": 40, "expanded_scans": 59,
               "fixed_test_scans": 8, "same_person_separation": True,
               "duplicate_and_unlabelled_exclusion": True, "original_sources_and_files_preserved": True,
               "actual_baseline_cache_and_one_epoch_pointcloud_training": True}
    Path(__file__).with_name("smoke_result.json").write_text(json.dumps(receipt, indent=2))
    print("[SMOKE PASSED]", json.dumps(receipt))
