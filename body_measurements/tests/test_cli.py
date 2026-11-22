"""End-to-end synthetic checks: metadata -> held-out evaluation -> saved inference."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from scripts.measurements import run_body_measurements as cli
from body_measurements.data import read_metadata
from body_measurements.joins import link_validation_rows


class MeasurementCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(11)
        self.names = [f"RIGDET{100000 + i}" for i in range(100)]
        self.size = rng.uniform(25, 50, size=100)
        np.savez_compressed(
            self.root / "features.npz", names=np.array(self.names, dtype=object),
            features=np.column_stack([self.size, rng.normal(170, 10, 100)]),
            feature_names=np.array(["bbox_mid", "bbox_long"], dtype=object),
            valid=np.ones(100, dtype=bool), scale_proxy=np.full(100, 100),
        )
        metadata = pd.DataFrame({
            "scan": self.names, "person": [f"person/{i // 2:04}" for i in range(100)],
            "manual_waist_mm": 10 * (40 + 1.4 * self.size),
            "manual_hip_cm": 65 + self.size,
        })
        metadata.loc[:15, "manual_hip_cm"] = np.nan
        metadata.to_csv(self.root / "metadata.tsv", sep="\t", index=False)
        (self.root / "targets.json").write_text(json.dumps({"targets": {
            "waist": {"column": "manual_waist_mm", "unit": "mm"},
            "hip": {"column": "manual_hip_cm", "unit": "cm"},
        }}), encoding="utf-8")

    def call(self, *arguments):
        args = cli.build_parser().parse_args(list(map(str, arguments)))
        with redirect_stdout(io.StringIO()):
            args.handler(args)

    def train(self, output="trained", *extra):
        self.call("train", "--metadata", self.root / "metadata.tsv", "--id-col", "scan",
                  "--participant-col", "person", "--features", self.root / "features.npz",
                  "--config", self.root / "targets.json", "--scan-units", "cm",
                  "--n-estimators", "8", "--n-jobs", "1", "--out-dir", self.root / output, *extra)

    def test_audit_train_predict_and_split_roundtrip(self):
        self.call("audit", "--metadata", self.root / "metadata.tsv", "--id-col", "scan",
                  "--features", self.root / "features.npz", "--config", self.root / "targets.json",
                  "--participant-col", "person", "--out-dir", self.root / "audit")
        audit = json.loads((self.root / "audit" / "audit.json").read_text())
        self.assertEqual(audit["matching_unique_ids"], 100)
        self.assertEqual(audit["alignment"]["targets"]["hip"]["available"], 84)
        self.train()
        model_path = self.root / "trained" / "model.joblib"
        model = joblib.load(model_path)
        self.assertEqual(model["scan_units"], "cm")
        self.assertEqual(model["run"]["status"], "complete")
        manifest = pd.read_csv(self.root / "trained" / "splits.csv")
        self.assertTrue(manifest.groupby("participant_id")["split"].nunique().eq(1).all())
        selection = pd.read_csv(self.root / "trained" / "model_selection.csv")
        self.assertEqual(set(selection.status), {"trained"})
        self.assertTrue(selection.selected_model.str.startswith("ridge_").all())
        self.call("predict", "--features", self.root / "features.npz", "--model", model_path,
                  "--scan-units", "cm", "--out-dir", self.root / "predicted")
        predicted = pd.read_csv(self.root / "predicted" / "predictions_cm.csv")
        self.assertEqual(list(predicted), ["scan_id", "waist_cm", "hip_cm"])
        self.assertEqual(len(predicted), 100)
        np.testing.assert_allclose(predicted.waist_cm, 40 + 1.4 * self.size, atol=0.2)
        # Saved participant IDs must survive reuse, including punctuation/path-like values.
        self.train("reused", "--splits", self.root / "trained" / "splits.csv")
        pd.testing.assert_frame_equal(
            manifest, pd.read_csv(self.root / "reused" / "splits.csv"),
        )

    def test_inference_rejects_changed_units_and_overwrites(self):
        self.train()
        with self.assertRaisesRegex(ValueError, "Scan units differ"):
            self.call("predict", "--features", self.root / "features.npz", "--model",
                      self.root / "trained" / "model.joblib", "--scan-units", "mm",
                      "--out-dir", self.root / "wrong_units")
        self.assertFalse((self.root / "wrong_units").exists())
        with self.assertRaisesRegex(ValueError, "new or empty"):
            self.train()

    def test_manifest_rejects_cross_split_participant_and_missing_rows(self):
        groups = pd.Series(["same", "same", "other", "third"], index=["a", "b", "c", "d"])
        path = self.root / "splits.csv"
        pd.DataFrame({"scan_id": groups.index, "split": ["train", "test", "validation", "train"]}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "more than one split"):
            cli.load_split_file(str(path), groups)
        pd.DataFrame({"scan_id": ["a"], "split": ["train"]}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "lacks"):
            cli.load_split_file(str(path), groups)

    def test_manifest_preserves_people_and_normalizes_split_whitespace(self):
        groups = pd.Series(["clinic/001", "clinic/001", "BDET12345-person", "third"], index=["a", "b", "c", "d"])
        path = self.root / "splits.csv"
        pd.DataFrame({"scan_id": groups.index, "participant_id": groups.to_numpy(),
                      "split": [" train ", "train", "validation", "test"]}).to_csv(path, index=False)
        self.assertEqual(cli.load_split_file(str(path), groups).tolist(), ["train", "train", "validation", "test"])

    def test_failed_training_preserves_audit_and_failure_reason(self):
        with self.assertRaisesRegex(ValueError, "No measurement models"):
            self.train("failed", "--min-train-labels", "1000")
        run = json.loads((self.root / "failed" / "run.json").read_text())
        self.assertEqual(run["status"], "failed")
        self.assertIn("training labels", run["error"])
        self.assertTrue((self.root / "failed" / "label_coverage.csv").exists())

    def test_numbered_validation_csv_needs_explicit_obj_link(self):
        validation_path = self.root / "validation.csv"
        validation_path.write_text(
            '\"\",\"body_height_cm_3d\",\"waist_girth_cm_3d\"\n'
            '\"1\",173.8,96.1\n\"2\",168.8,100.7\n', encoding="utf-8"
        )
        validation = read_metadata(validation_path)
        self.assertEqual(list(validation), ["__row_number__", "body_height_cm_3d", "waist_girth_cm_3d"])
        self.call("audit", "--metadata", validation_path, "--out-dir", self.root / "numbered_audit")
        report = json.loads((self.root / "numbered_audit" / "audit.json").read_text())
        self.assertNotIn("unique_ids", report)
        mapping_path = self.root / "explicit_mapping.csv"
        pd.DataFrame({"row_number": ["1", "2"], "obj_file":
                      ["!RIGDET009225!.obj", "GL0DET054125.obj"]}).to_csv(mapping_path, index=False)
        self.call("link-labels", "--metadata", validation_path, "--mapping", mapping_path,
                  "--map-row-col", "row_number", "--map-obj-col", "obj_file",
                  "--out-dir", self.root / "linked")
        linked = read_metadata(self.root / "linked" / "metadata_with_scan_id.csv")
        self.assertEqual(linked.scan_id.tolist(), ["RIGDET009225", "GL0DET054125"])
        self.assertEqual(linked.body_height_cm_3d.tolist(), ["173.8", "168.8"])
        with self.assertRaisesRegex(ValueError, "lacks 1 validation row keys"):
            link_validation_rows(validation, pd.DataFrame({"row_number": ["1"], "obj_file": ["a.obj"]}),
                                 map_row_col="row_number", map_obj_col="obj_file")
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            link_validation_rows(validation, pd.DataFrame({"row_number": ["1", "2"],
                                                           "obj_file": ["a.obj", "a.obj"]}),
                                 map_row_col="row_number", map_obj_col="obj_file")

    def test_raw_obj_feature_builder_and_candidate_inspection(self):
        obj_root = self.root / "raw_objs"
        obj_root.mkdir()
        for index, name in enumerate(("GL0DET054125.obj", "!RIGDET009225!.obj")):
            scale = 100 + index * 20
            points = [(0, 0, 0), (scale, 0, 0), (0, scale / 3, 0), (0, 0, scale / 2),
                      (scale, scale / 3, scale / 2)]
            (obj_root / name).write_text("".join(f"v {x} {y} {z}\n" for x, y, z in points), encoding="utf-8")
        self.call("build-features", "--obj-root", obj_root, "--workers", "1",
                  "--checkpoint-every", "1", "--out-dir", self.root / "raw_features")
        feature_path = self.root / "raw_features" / "scan_geometry_features.npz"
        with np.load(feature_path, allow_pickle=True) as saved:
            self.assertEqual(saved["names"].tolist(), ["GL0DET054125", "RIGDET009225"])
            self.assertTrue(saved["valid"].all())
            self.assertEqual(saved["features"].shape, (2, 14))
        size = json.loads((self.root / "raw_features" / "build_audit.json").read_text())
        self.assertEqual(size["rows_valid"], 2)
        validation_path = self.root / "validation.csv"
        validation_path.write_text('"",body_height_cm_3d\n1,173.8\n2,168.8\n', encoding="utf-8")
        pd.DataFrame({"row_number": ["1", "2"], "obj_file":
                      ["GL0DET054125.obj", "!RIGDET009225!.obj"]}).to_csv(
            self.root / "candidate.csv", index=False,
        )
        self.call("inspect-sources", "--metadata", validation_path, "--obj-root", obj_root,
                  "--candidate-root", self.root, "--out-dir", self.root / "source_inspection")
        candidates = pd.read_csv(self.root / "source_inspection" / "candidate_tables.csv")
        row = candidates.loc[candidates.path.eq(str(self.root / "candidate.csv"))].iloc[0]
        self.assertEqual(row.best_obj_column, "obj_file")
        self.assertEqual(row.best_row_column, "row_number")
        self.assertEqual(row.best_obj_matches, 2)


if __name__ == "__main__":
    unittest.main()
