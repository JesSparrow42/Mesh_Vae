"""CPU checks for eligibility, frozen cohorts, baselines and scan-only controls."""
import argparse
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from paper_analysis import body_composition_controls as control


class BodyControlsTest(unittest.TestCase):
    def test_clean_id_join_and_duplicate_rejection(self):
        frame = control.identified(pd.DataFrame({"participant_id": ["!GLODET000125!.obj", "GLODET000325"], "x": [1, 2]}), "fixture")
        self.assertEqual(frame.index.tolist(), ["GLODET000125", "GLODET000325"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            control.identified(pd.DataFrame({"participant_id": ["GLODET000125.obj", "GLODET000125"]}), "fixture")

    def test_explicit_bmi_units_and_tissue_status(self):
        frame = pd.DataFrame({"weight": [80, 80], "height": [200, 0]})
        with self.assertRaisesRegex(ValueError, "units are never inferred"):
            control.bmi_values(frame, weight_col="weight", height_col="height")
        bmi, definition = control.bmi_values(frame, weight_col="weight", height_col="height", height_unit="cm")
        self.assertEqual(bmi.iloc[0], 20)
        self.assertTrue(np.isnan(bmi.iloc[1]))
        self.assertEqual(definition["weight_unit"], "kg")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tissue.csv"
            pd.DataFrame({"participant_id": ["GLODET000125", "GLODET000325", "GLODET000525", "GLODET000625"], "ts_status": ["ok", "error", "ok", "ok"], "skeletal_muscle_ml": [10, 10, 0, -1], "subcutaneous_fat_ml": [20, 20, 0, 4], "sat_ml": [20, 20, 0, 4], "vat_ml": [3, 4, 5, 6], "torso_fat_ml": [3, 4, 5, 6]}).to_csv(path, index=False)
            tissue, audit = control.tissue_labels(path)
            self.assertEqual(tissue.skeletal_muscle_ml.notna().sum(), 1)
            self.assertEqual(audit["skeletal_muscle_ml"]["zero"], 1)
            self.assertEqual(audit["possible_aliases"]["vat_ml==torso_fat_ml"]["equal_rows"], 4)
            self.assertNotIn("vat_ml", control.TARGETS)

    def test_freeze_without_plaque_or_weight_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = [f"GLODET{i:06d}" for i in range(440)]
            metadata = pd.DataFrame({"participant_id": ids, "age": np.tile(np.arange(20, 75), 8), "sex": ["F"] * 220 + ["M"] * 220, "source_bmi": np.linspace(18, 35, 440)})
            metadata_path = root / "metadata.tsv"
            metadata.to_csv(metadata_path, sep="\t", index=False)
            tissue_path = root / "tissue.csv"
            pd.DataFrame({"participant_id": ids, "ts_status": "ok", "skeletal_muscle_ml": np.arange(440) + 2000, "subcutaneous_fat_ml": np.arange(440) + 1000}).to_csv(tissue_path, index=False)
            for sex, start in (("F", 0), ("M", 220)):
                for split, lo, hi in (("train", start, start + 180), ("test", start + 180, start + 220)):
                    pd.DataFrame({"participant_id": ids[lo:hi]}).to_csv(root / f"{split}_{sex}.tsv", sep="\t", index=False)
            np.savez(root / "simplified_padded_tensors.npz", names=ids)
            np.savez(root / "scan_geometry_features.npz", names=ids, features=np.ones((440, 2)), feature_names=["bbox_long", "bbox_mid"], scale_proxy=np.ones(440))
            settings = {"metadata": str(metadata_path), "dataroot": str(root), "checkpoints": str(root / "checkpoints"), "train_template": str(root / "train_{gender}.tsv"), "test_template": str(root / "test_{gender}.tsv")}
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"platforms": {"computerome": settings}}))
            args = argparse.Namespace(platform="computerome", config=str(config_path), metadata=None, dataroot=None, checkpoints=None, targets=list(control.TARGETS), tissue=str(tissue_path), bmi_col="source_bmi", bmi_weight_col=None, bmi_height_col=None, bmi_height_unit=None, age_col=None, gender_col=None, seed=42, epochs=150, run_tag="test", output=str(root / "study"))
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = control.audit(args, freeze=True)
            self.assertTrue(manifest["passed"])
            self.assertEqual(len(manifest["tasks"]), 6)
            study, loaded = control.frozen(args)
            for task in loaded["tasks"]:
                frames = {key: pd.read_csv(study / path, sep="\t") for key, path in task["splits"].items()}
                self.assertFalse(set(frames["train"].participant_id) & set(frames["validation"].participant_id))
                self.assertFalse(set(frames["train"].participant_id) & set(frames["test"].participant_id))
                self.assertEqual(list(frames["train"]), ["participant_id", "study_age", "study_gender", "control_target"])
            split_path = study / loaded["tasks"][0]["splits"]["train"]
            split_path.write_text(split_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Frozen split changed"):
                control.frozen(args)

    def test_outer_overlap_and_target_specific_missingness(self):
        ids = [f"GLODET{i:06d}" for i in range(100)]
        metadata = control.identified(pd.DataFrame({"participant_id": ids, "age": 45, "sex": "F", "bmi": np.linspace(20, 30, 100), "subcutaneous_fat_ml": [np.nan] + list(range(1, 100))}), "metadata")
        train, test = metadata.iloc[:80], metadata.iloc[80:]
        train = train.copy()
        train.loc[ids[0], "subcutaneous_fat_ml"] = 999
        with self.assertRaisesRegex(ValueError, "overlap"):
            control.make_splits(metadata, train, metadata.iloc[79:], set(ids), "F", ["bmi"], "age", "sex")
        result = control.make_splits(metadata, train, test, set(ids), "F", ["bmi", "subcutaneous_fat_ml"], "age", "sex")
        for split in ("train", "validation", "test"):
            bmi_ids = set(result["bmi"][split].participant_id)
            tissue_ids = set(result["subcutaneous_fat_ml"][split].participant_id)
            self.assertTrue(tissue_ids <= bmi_ids)
        self.assertEqual(sum(len(f) for f in result["subcutaneous_fat_ml"].values()), 99)
        self.assertFalse(any(ids[0] in set(f.participant_id) for f in result["subcutaneous_fat_ml"].values()))

    def test_train_only_baselines_and_exact_prediction_alignment(self):
        train = pd.DataFrame({"study_age": [20, 30, 40], "control_target": [10, 20, 30]})
        test = pd.DataFrame({"participant_id": ["a", "b"], "study_age": [25, 35], "control_target": [999999, -99999]})
        predictions, fitted = control.baseline_predictions(train, test)
        self.assertEqual(fitted["age_mean"], 30)
        self.assertEqual(predictions["training_median"].tolist(), [20, 20])
        changed = test.copy(); changed["control_target"] = 0
        predictions2, fitted2 = control.baseline_predictions(train, changed)
        self.assertEqual(fitted, fitted2)
        np.testing.assert_allclose(predictions["age_linear"], predictions2["age_linear"])
        heldout = pd.DataFrame({"participant_id": ["b", "a"], "target": [-99999, 999999], "prediction": [3, 4]})
        control.validate_predictions(test, heldout)
        heldout.loc[0, "target"] = 10
        with self.assertRaisesRegex(ValueError, "target values differ"):
            control.validate_predictions(test, heldout)

    def test_commands_no_label_covariates_and_preserve_architecture(self):
        from meshvae.options import TrainOptions
        settings = {"checkpoints": "/checkpoints", "dataroot": "/data"}
        for target in control.TARGETS:
            cmd = control.command("fixture", {"train": "/train.tsv", "validation": "/val.tsv"}, target, "M", settings)
            parsed = TrainOptions().parser.parse_args(cmd[2:])
            self.assertFalse(parsed.cond_weight)
            self.assertTrue(parsed.aux_predict_regression)
            option = lambda flag: cmd[cmd.index(flag) + 1]
            self.assertEqual(option("--encoder_cond_mode"), "none")
            self.assertEqual(option("--encoder_arch"), "meshconv")
            self.assertEqual(option("--aux_regression_col"), "control_target")
            self.assertEqual(option("--aux_regression_transform"), "identity" if target == "bmi" else "log1p")
            self.assertEqual(option("--niter"), "150")
            for flag in ("--no-cond_weight", "--no-cond_athero", "--no-cond_plaque", "--restore_metric_scale", "--require_scan_geometry_features"):
                self.assertIn(flag, cmd)
            for flag in ("--weight_col", "--aux_regression_source_cols", "--cond_weight"):
                self.assertNotIn(flag, cmd)

    def test_real_target_preprocessing_fits_training_only(self):
        from meshvae.conditioning import build_age_gender_lookup
        from meshvae.options import TrainOptions
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path, val_path = root / "train.tsv", root / "val.tsv"
            pd.DataFrame({"participant_id": ["GLODET000125", "GLODET000325", "GLODET000525"], "study_age": [30, 40, 50], "study_gender": "M", "control_target": [20, 25, 30]}).to_csv(train_path, sep="\t", index=False)
            pd.DataFrame({"participant_id": ["GLODET000625"], "study_age": [40], "study_gender": "M", "control_target": [10000]}).to_csv(val_path, sep="\t", index=False)
            cmd = control.command("fixture", {"train": train_path, "validation": val_path}, "bmi", "M", {"checkpoints": str(root), "dataroot": str(root)})
            opt = TrainOptions().parser.parse_args(cmd[2:])
            opt.is_train = True
            with contextlib.redirect_stdout(io.StringIO()):
                build_age_gender_lookup(opt)
                validation = copy.deepcopy(opt)
                validation.is_train = False
                validation.cond_tsv = str(val_path)
                _, _, aux = build_age_gender_lookup(validation)
            self.assertEqual(opt.aux_regression_mean, 25)
            self.assertEqual(validation.aux_regression_mean, 25)
            self.assertAlmostEqual(aux["GLODET000625"]["regression_target"], (10000 - 25) / opt.aux_regression_std, places=3)

    def test_geometry_alignment_and_paired_participant_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root / "scan_geometry_features.npz", names=["GLODET000325", "GLODET000125", "GLODET000525"], features=[[30, 3], [10, 1], [50, 5]], feature_names=["bbox_long", "bbox_mid"], scale_proxy=np.ones(3))
            train = pd.DataFrame({"participant_id": ["GLODET000125", "GLODET000325"]})
            test = pd.DataFrame({"participant_id": ["GLODET000525"]})
            with contextlib.redirect_stdout(io.StringIO()):
                x, xt, names = control.geometry_arrays(train, test, root)
            self.assertEqual(x.tolist(), [[10, 1], [30, 3]])
            self.assertEqual(xt.tolist(), [[50, 5]])
            self.assertEqual(names, ["bbox_long", "bbox_mid"])
        interval = control.paired_mae_interval(np.array([10, 20, 30]), np.array([10, 20, 30]), np.array([11, 21, 31]), bootstraps=1000)
        self.assertEqual(interval["MeshVAEMinusComparatorMAE"], -1)
        self.assertEqual(interval["CI95Lower"], -1)
        self.assertEqual(interval["CI95Upper"], -1)


if __name__ == "__main__":
    unittest.main()
