"""CPU-only checks: python -m unittest paper_analysis.test_plaque_study -v"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd

from meshvae.conditioning import build_age_gender_lookup, cond_dim_from_opt, athero_slice_from_opt, age_slice_from_opt
from meshvae.options import TrainOptions, TestOptions
from paper_analysis.plaque_cohorts import derive_targets, load_splits, select_cohort, training_problem
from paper_analysis.plaque_reporting import metrics, fit_predict, combine_hurdle, report
from paper_analysis.run_plaque_study import Experiment, SUITES, tasks, training_command, run_task, study_name
from paper_analysis.study_config import load_config


def synthetic_frame(n=2000):
    idx = np.arange(n)
    frame = pd.DataFrame({"participant_id": [f"PERSON{i:05d}" for i in idx],
                          "sexMF": np.where(idx % 2, "F", "M"),
                          "age": 20 + (idx % 61), "weight_kg": 50 + (idx % 50)})
    for site in ("carotid", "femoral"):
        for side in ("right", "left"):
            frame[f"plaque_volume_{side}_{site}_corrected"] = np.where(idx % 5 == 0, 0, (idx % 101) + 1)
    frame["pv_coro"] = frame.plaque_volume_left_carotid_corrected
    return frame


class PlaqueStudyTests(unittest.TestCase):
    def test_explicit_checkpoint_root_is_preserved(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            config = json.loads(Path(__file__).with_name("next_steps_config.json").read_text())
            config["platforms"]["computerome"]["checkpoints"] = str(root/"custom_checkpoints")
            config_path = root/"config.json"
            config_path.write_text(json.dumps(config))
            resolved = load_config(config_path, "computerome", code_root=root/"MQVAE_new")
            self.assertEqual(resolved["platforms"]["computerome"]["checkpoints"], str((root/"custom_checkpoints").resolve()))

    def test_renamed_checkout_dry_run_and_collection_from_other_cwd(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            checkout = root/"MQVAE_new"
            data = root/"data"; data.mkdir()
            self.fixture(data)
            source = Path(__file__).resolve().parents[1]
            for folder in ("meshvae", "paper_analysis"):
                shutil.copytree(source/folder, checkout/folder,
                                ignore=shutil.ignore_patterns("__pycache__", "*.zip", "*.png", "*.pdf"))
            shutil.copy(source/"run_cardiovascular_target_benchmark.py", checkout)
            config_path = checkout/"paper_analysis/next_steps_config.json"
            config = json.loads(config_path.read_text())
            config["platforms"]["computerome"] = {
                "metadata": str(data/"metadata.tsv"), "dataroot": str(data),
                "train_template": str(data/"train.tsv"), "test_template": str(data/"test.tsv"),
                "checkpoints": "checkpoints",
            }
            config_path.write_text(json.dumps(config))
            # A foreign cwd catches accidental output routing to the caller or the old repo.
            for index, gender in ((0, "F"), (7, "M")):
                result = subprocess.run([
                    sys.executable, str(checkout/"paper_analysis/run_plaque_study.py"),
                    "--platform", "computerome", "--suite", "territory_classification",
                    "--task-index", str(index), "--dry-run",
                ], cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                study = next((checkout/"checkpoints").glob(f"*_{gender}_*/study"))
                status = json.loads((study/"status.json").read_text())
                self.assertEqual(status["status"], "dry_run")
                cmd = json.loads((study/"train_command.json").read_text())
                self.assertEqual(Path(cmd[cmd.index("--checkpoints_dir")+1]), (checkout/"checkpoints").resolve())
            result = subprocess.run([
                sys.executable, str(checkout/"paper_analysis/collect_plaque_study.py"),
                "--platform", "computerome",
            ], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            statuses = pd.read_csv(checkout/"next_steps_results/status.csv")
            self.assertEqual(set(statuses.gender), {"F", "M"})
            self.assertFalse((root/"checkpoints").exists())
            self.assertFalse((root/"next_steps_results").exists())

    def test_missing_is_not_negative(self):
        frame = synthetic_frame(3)
        frame.loc[0, "plaque_volume_left_carotid_corrected"] = np.nan
        frame.loc[1, "plaque_volume_left_carotid_corrected"] = np.nan
        frame.loc[2, "plaque_volume_right_femoral_corrected"] = -7
        out, _ = derive_targets(frame)
        self.assertTrue(np.isnan(out.loc[0, "presence_carotid_total"]))
        self.assertEqual(out.loc[1, "presence_carotid_total"], 1)
        self.assertTrue(np.isnan(out.loc[1, "volume_peripheral_total"]))
        self.assertTrue(np.isnan(out.loc[2, "volume_femoral_right"]))

    def test_total_excludes_coronary(self):
        frame = synthetic_frame(2)
        frame["pv_coro"] = 100000
        out, _ = derive_targets(frame)
        self.assertEqual(out.loc[1, "volume_peripheral_total"], 8)
        self.assertEqual(out.loc[0, "presence_peripheral_total"], 0)

    def test_thresholds_do_not_use_test_or_validation(self):
        train = pd.DataFrame({"study_age": [45]*100, "volume": np.arange(100)})
        split = {"train": train, "validation": train.copy(), "test": train.copy()}
        selected, rule = select_cohort(split, "volume", "top5")
        self.assertEqual(len(selected["train"]), 5)
        split["test"]["volume"] = 1e9
        split["validation"]["volume"] = 1e8
        _, altered = select_cohort(split, "volume", "top5")
        self.assertEqual(rule["threshold"], altered["threshold"])
        _, age_rule = select_cohort(split, "volume", "age_top")
        self.assertEqual(age_rule["thresholds"]["40"], float(train.volume.quantile(.8)))

    def test_zero_ties_and_unknown_age_bands(self):
        train = pd.DataFrame({"study_age": [25]*100, "volume": np.zeros(100)})
        test = pd.DataFrame({"study_age": [25, 45], "volume": [0, 5]})
        split = {"train": train, "validation": train, "test": test}
        selected, _ = select_cohort(split, "volume", "age_top")
        self.assertEqual(len(selected["test"]), 1)
        selected, _ = select_cohort(split, "volume", "top5")
        self.assertEqual(len(selected["train"]), 0)

    def test_model_eligibility_does_not_depend_on_test(self):
        train = pd.DataFrame({"y": [0, 1]*30})
        split = {"train": train, "validation": train, "test": pd.DataFrame({"y": []})}
        self.assertIsNone(training_problem(split, "y", "binary"))

    def fixture(self, root):
        frame = synthetic_frame()
        frame.to_csv(root/"metadata.tsv", sep="\t", index=False)
        frame.iloc[:1600].to_csv(root/"train.tsv", sep="\t", index=False)
        frame.iloc[1600:].to_csv(root/"test.tsv", sep="\t", index=False)
        np.savez(root/"simplified_padded_tensors.npz", names=frame.participant_id.to_numpy())
        np.savez(root/"scan_geometry_features.npz", names=frame.participant_id.to_numpy(),
                 features=np.c_[frame.weight_kg, frame.age], feature_names=["size", "height"],
                 scale_proxy=np.ones(len(frame)))
        return frame

    def test_sex_and_split_disjointness(self):
        with TemporaryDirectory() as td:
            root = Path(td); self.fixture(root)
            fs, _ = load_splits(root/"metadata.tsv", root/"train.tsv", root/"test.tsv", root, "F")
            ms, _ = load_splits(root/"metadata.tsv", root/"train.tsv", root/"test.tsv", root, "M")
            self.assertEqual(len(fs["train"])+len(fs["validation"]), 800)
            self.assertTrue(all(set(x.study_gender) == {"F"} for x in fs.values()))
            self.assertFalse(set(fs["train"].participant_id) & set(fs["validation"].participant_id))
            self.assertFalse(set(fs["train"].participant_id) & set(fs["test"].participant_id))
            self.assertFalse(set(pd.concat(fs.values()).participant_id) & set(pd.concat(ms.values()).participant_id))

    def test_duplicate_scans_rejected(self):
        with TemporaryDirectory() as td:
            root = Path(td); self.fixture(root)
            np.savez(root/"simplified_padded_tensors.npz", names=["PERSON00001", "PERSON00001"])
            with self.assertRaisesRegex(ValueError, "one nonempty ID"):
                load_splits(root/"metadata.tsv", root/"train.tsv", root/"test.tsv", root, "F")

    def test_overlap_and_missing_geometry_rejected(self):
        with TemporaryDirectory() as td:
            root = Path(td); frame = self.fixture(root)
            frame.iloc[:5].to_csv(root/"test.tsv", sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_splits(root/"metadata.tsv", root/"train.tsv", root/"test.tsv", root, "F")
            self.fixture(root)
            np.savez(root/"scan_geometry_features.npz", names=frame.participant_id.to_numpy(),
                     features=np.full((len(frame), 1), np.nan), feature_names=["size"], scale_proxy=np.ones(len(frame)))
            with self.assertRaisesRegex(ValueError, "physical geometry"):
                load_splits(root/"metadata.tsv", root/"train.tsv", root/"test.tsv", root, "F")

    def test_array_bounds_match_task_plan(self):
        import re
        root = Path(__file__).parent
        for suite in SUITES:
            source = (root/f"gefion_next_{suite}.sh").read_text()
            last = int(re.search(r"--array=0-(\d+)%2", source).group(1))
            self.assertEqual(last+1, len(tasks(suite)))
            self.assertNotIn("--export", source)

    def test_weight_ablation_has_identical_participants(self):
        with TemporaryDirectory() as td:
            root = Path(td); self.fixture(root)
            config = json.loads(Path(__file__).with_name("next_steps_config.json").read_text())
            config["platforms"]["gefion"] = {"metadata": str(root/"metadata.tsv"), "train_template": str(root/"train.tsv"),
                "test_template": str(root/"test.tsv"), "dataroot": td, "checkpoints": str(root/"checkpoints")}
            run_task("F", Experiment("regression_age", "peripheral_total", weight=False), "weight", config, "gefion", audit_only=True)
            run_task("F", Experiment("regression_age_weight", "peripheral_total", weight=True), "weight", config, "gefion", audit_only=True)
            manifests = list((root/"checkpoints").glob("*/study/manifest.json"))
            self.assertEqual(json.loads(manifests[0].read_text())["members"], json.loads(manifests[1].read_text())["members"])

    def test_weight_stats_are_frozen_for_validation(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            frame = pd.DataFrame({"participant_id": ["A", "B"], "age": [40, 50], "sexMF": ["F", "F"], "weight_kg": [60, 80]})
            frame.to_csv(root/"train.tsv", sep="\t", index=False)
            opt = SimpleNamespace(use_cond=True, cond_weight=True, weight_col="weight_kg", cond_tsv=str(root/"train.tsv"),
                                  cond_stregkode_col=None, cond_age_col=None, cond_gender_col=None, cond_age_source="age",
                                  age_bins=[18, 35, 50, 60, 200], only_gender="F", is_train=True, checkpoints_dir=td, name="test")
            lookup, _, _ = build_age_gender_lookup(opt)
            self.assertEqual(opt.weight_mean, 70)
            self.assertEqual(opt.weight_std, 10)
            self.assertEqual(lookup["A"][-1], -1)
            self.assertEqual(opt.demo_cond_dim, 7)
            self.assertEqual(cond_dim_from_opt(opt), 7)
            self.assertEqual(age_slice_from_opt(opt), slice(2, 6))
            opt.cond_athero = True
            self.assertEqual(athero_slice_from_opt(opt), slice(7, 8))
            opt.cond_athero = False
            frame.weight_kg = [100, 120]
            frame.to_csv(root/"validation.tsv", sep="\t", index=False)
            opt.cond_tsv, opt.is_train, opt.phase = str(root/"validation.tsv"), False, "test"
            Base = TestOptions()
            Base._load_cond_stats_for_eval(opt)
            lookup, _, _ = build_age_gender_lookup(opt)
            self.assertEqual(opt.weight_mean, 70)
            self.assertEqual(lookup["A"][-1], 3)

    def test_old_condition_shape_unchanged(self):
        opt = SimpleNamespace(use_cond=True, age_bins=[18, 35, 50, 60, 200])
        self.assertEqual(cond_dim_from_opt(opt), 6)

    def test_all_suites_include_both_sexes_and_valid_training_flags(self):
        config = json.loads(Path(__file__).with_name("next_steps_config.json").read_text())
        for suite in SUITES:
            plan = tasks(suite)
            self.assertEqual({g for g, _ in plan}, {"F", "M"})
            for gender, exp in plan:
                cmd = training_command("test", {"train": "train.tsv", "validation": "val.tsv"}, exp, gender, config, "gefion")
                parsed = TrainOptions().parser.parse_args(cmd[2:])
                self.assertEqual(parsed.only_gender, gender)
                self.assertEqual(parsed.cond_weight, exp.weight)
                self.assertEqual(parsed.num_threads, 0)
                self.assertFalse(parsed.cond_athero)
                self.assertFalse(parsed.cond_plaque)

    def test_metrics_and_baseline(self):
        self.assertAlmostEqual(metrics([0, 1], [.5, .5], "binary")["AUC"], .5)
        self.assertAlmostEqual(metrics([0, 1], [.5, .5], "binary")["AveragePrecision"], .5)
        self.assertEqual(metrics([0, 1, 2], [0, 1, 2])["R2"], 1)
        self.assertTrue(np.isnan(metrics([1, 1], [1, 1])["R2"]))
        x = np.arange(100, dtype=float).reshape(-1, 1)
        p = fit_predict(x, x[:, 0], x[:3], "regression")
        self.assertLess(np.max(abs(p-x[:3, 0])), 1)

    def test_hurdle_correction_uses_validation_only(self):
        with TemporaryDirectory() as td:
            binary = pd.DataFrame({"participant_id": ["A", "B"], "probability": [.2, .5]})
            positive = pd.DataFrame({"participant_id": ["A", "B"], "target": [0, 1e9], "prediction_transformed": np.log([11, 11])})
            validation = pd.DataFrame({"target_transformed": np.log([11]*10), "prediction_transformed": np.log([11]*10)})
            result = combine_hurdle(binary, positive, validation, td)
            np.testing.assert_allclose(result.prediction, [2, 5])

    def test_dry_run_and_report_end_to_end(self):
        with TemporaryDirectory() as td:
            root = Path(td); self.fixture(root)
            config = json.loads(Path(__file__).with_name("next_steps_config.json").read_text())
            config["platforms"]["gefion"] = {"metadata": str(root/"metadata.tsv"), "train_template": str(root/"train.tsv"),
                "test_template": str(root/"test.tsv"), "dataroot": td, "checkpoints": str(root/"checkpoints")}
            config["bootstraps"] = 5
            status = run_task("F", Experiment("all_log", "peripheral_total"), "plaque", config, "gefion", dry_run=True)
            self.assertEqual(status["status"], "dry_run")
            splits, _ = load_splits(root/"metadata.tsv", root/"train.tsv", root/"test.tsv", root, "F")
            predictions = pd.DataFrame({"participant_id": splits["test"].participant_id,
                                        "prediction": splits["test"].volume_peripheral_total})
            rows = report(splits["train"], splits["test"], predictions, "volume_peripheral_total", "regression", root/"report", root, bootstraps=5)
            self.assertEqual(rows[0]["R2"], 1)
            self.assertTrue((root/"report"/"exploratory_body_size_associations.csv").exists())

    def test_matched_prior_commands_keep_study_settings_and_names_distinct(self):
        config = json.loads(Path(__file__).with_name("next_steps_config.json").read_text())
        config.update(seed=43, split_seed=42, prior_arm="gaussian_mmd")
        exp = Experiment("all_log", "peripheral_total")
        paths = {"train": "train.tsv", "validation": "validation.tsv"}
        gaussian = training_command("gaussian", paths, exp, "F", config, "gefion")
        gaussian_name = study_name("F", exp, "plaque", config, "gefion")
        config["prior_arm"] = "boson_mmd"
        boson = training_command("boson", paths, exp, "F", config, "gefion")
        self.assertNotEqual(gaussian_name, study_name("F", exp, "plaque", config, "gefion"))
        for cmd in (gaussian, boson):
            opt = TrainOptions().parser.parse_args(cmd[2:])
            self.assertEqual(opt.seed, 43)
            self.assertEqual(opt.split_seed, 42)
            self.assertEqual(opt.aux_regression_transform, "log1p")
            self.assertEqual(opt.aux_regression_loss, "huber")
            self.assertTrue(opt.cond_weight)
            self.assertTrue(opt.use_point_labels)
            self.assertEqual(opt.prior_match_mode, "mmd")
        self.assertEqual(TrainOptions().parser.parse_args(boson[2:]).latent_prior, "boson")

    def test_reference_ablations_freeze_saved_participants_and_training_budget(self):
        with TemporaryDirectory() as td:
            root = Path(td); self.fixture(root)
            config = json.loads(Path(__file__).with_name("next_steps_config.json").read_text())
            config.update(niter=173, batch_size=3)
            config["platforms"]["gefion"] = {
                "metadata": str(root/"metadata.tsv"), "train_template": str(root/"train.tsv"),
                "test_template": str(root/"test.tsv"), "dataroot": td,
                "checkpoints": str(root/"checkpoints"),
            }
            exp = Experiment("all_log", "peripheral_total")
            run_task("F", exp, "plaque", config, "gefion", dry_run=True)
            reference_run = root/"checkpoints"/study_name("F", exp, "plaque", config, "gefion")
            reference = reference_run/"study"
            status = json.loads((reference/"status.json").read_text()); status["status"] = "completed"
            (reference/"status.json").write_text(json.dumps(status))
            base_command = json.loads((reference/"train_command.json").read_text())
            base_command[base_command.index("--encoder_max_points")+1] = "2048"
            (reference/"train_command.json").write_text(json.dumps(base_command))
            # Metadata has changed: ablations must use the original saved values and membership.
            (root/"metadata.tsv").unlink()
            original_members = json.loads((reference/"manifest.json").read_text())["members"]
            manifests = []
            for seed, arm in ((42, "gaussian_mmd"), (43, "boson_mmd"), (44, "gaussian_mmd")):
                result = subprocess.run([
                    sys.executable, str(Path(__file__).with_name("run_plaque_study.py")),
                    "--platform", "gefion", "--suite", "plaque", "--experiment", "all_log",
                    "--gender", "F", "--reference-run", str(reference_run),
                    "--run-tag", "matched", "--seed", str(seed), "--prior-arm", arm, "--dry-run",
                ], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                derived = root/"checkpoints"/f"GEFION_NEXT_plaque_all_log_{arm}_F_matched_seed{seed}"/"study"
                manifest = json.loads((derived/"manifest.json").read_text())
                self.assertEqual(manifest["members"], original_members)
                cmd = json.loads((derived/"train_command.json").read_text())
                opt = TrainOptions().parser.parse_args(cmd[2:])
                self.assertEqual(opt.niter, 173)
                self.assertEqual(opt.batch_size, 3)
                self.assertEqual(opt.encoder_max_points, 2048)
                self.assertEqual(opt.seed, seed)
                self.assertEqual(opt.split_seed, 42)
                for split in ("train", "validation", "test"):
                    pd.testing.assert_frame_equal(pd.read_csv(reference/f"{split}.tsv", sep="\t"),
                                                  pd.read_csv(derived/f"{split}.tsv", sep="\t"))
                manifests.append(manifest)
            self.assertEqual(manifests[0]["reference"], manifests[1]["reference"])
            self.assertNotEqual(manifests[0]["fingerprint"], manifests[1]["fingerprint"])
            # A reference TSV edited to change participants is rejected before any training.
            changed = pd.read_csv(reference/"train.tsv", sep="\t")
            changed.loc[0, "participant_id"] = "OTHER_PERSON"
            changed.to_csv(reference/"train.tsv", sep="\t", index=False)
            result = subprocess.run([
                sys.executable, str(Path(__file__).with_name("run_plaque_study.py")),
                "--platform", "gefion", "--suite", "plaque", "--experiment", "all_log",
                "--gender", "F", "--reference-run", str(reference_run),
                "--run-tag", "tampered", "--prior-arm", "gaussian_mmd", "--dry-run",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("participants differ", result.stderr)


if __name__ == "__main__":
    unittest.main()
