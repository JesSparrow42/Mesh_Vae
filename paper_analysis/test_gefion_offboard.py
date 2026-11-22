"""CPU tests for offboarding verification and streaming archive integrity."""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from paper_analysis import gefion_offboard as off


class OffboardTests(unittest.TestCase):
    def setUp(self):
        from paper_analysis.test_clean_gefion_comparison import ComparisonTests
        self.fixture = ComparisonTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        for group in ("F", "M"):
            self.fixture.fixture(group=group)
        with contextlib.redirect_stdout(io.StringIO()):
            self.plan = self.fixture.prepare()
        self.api = SimpleNamespace(COMPARISON=__import__("paper_analysis.test_clean_gefion_comparison", fromlist=["runner"]).runner,
                                   main=lambda args: None)
        self.project = self.fixture.root
        self.code = self.project/"MQVAE_new"
        self.code.mkdir()
        (self.code/"source.py").write_text("print('saved code')")
        self.args = argparse.Namespace(code_root=self.code, project_root=self.project,
            study_dir=self.fixture.output, output_dir=self.project/"offboarding",
            account="cu_0080", chunk_gib=8, allow_incomplete=False, extra=[])
        for task in self.plan["tasks"]:
            cohort = next(c for c in self.plan["cohorts"] if (c["target"], c["group"]) == (task["target"], task["group"]))
            run = Path(task["run_dir"]); run.mkdir(exist_ok=True)
            for name in ("best_vae_net.pth",):
                (run/name).write_bytes(b"fake checkpoint for CPU tests")
            off.save_json(run/"cond_stats.json", {"aux_predict_binary": True})
            off.save_json(run/"training_summary.json", {
                "train_samples": cohort["counts"]["train"], "validation_samples": cohort["counts"]["validation"],
                "validation_ids": cohort["members"]["validation"], "best_epoch": 2})
            if task["status"] != "reused":
                off.save_json(run/"clean_comparison_training_completed.json", {"fingerprint": task["fingerprint"]})
                off.save_json(Path(task["cohort_dir"])/(task["variant"]+"_status.json"), {
                    "status": "completed", "fingerprint": task["fingerprint"]})
                Path(task["predictions_path"]).write_bytes(Path(cohort["prediction_paths"]["age_only"]).read_bytes())
        self.bundle_patch = patch.object(off, "bundle", return_value=self.api)
        self.bundle_patch.start(); self.addCleanup(self.bundle_patch.stop)
        self.capture_patch = patch.object(off, "capture", return_value={"returncode": 0, "stdout": "", "stderr": ""})
        self.capture_patch.start(); self.addCleanup(self.capture_patch.stop)

    def quietly(self, func, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*args)

    def test_all_planned_fits_and_splits_verified(self):
        _, _, result = self.quietly(off.check, self.args)
        self.assertTrue(result["ready_for_collection"])
        self.assertEqual(result["passed_fits"], 6)
        report = self.args.output_dir/"verification"/"run_verification.csv"
        with report.open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 6)

    def test_prediction_mismatch_blocks_collection(self):
        task = next(t for t in self.plan["tasks"] if t["status"] != "reused")
        path = Path(task["predictions_path"])
        import pandas as pd
        frame = pd.read_csv(path); frame.loc[0, "target"] = 2; frame.to_csv(path, index=False)
        _, _, result = self.quietly(off.check, self.args)
        self.assertFalse(result["ready_for_collection"])
        with self.assertRaisesRegex(ValueError, "unresolved"):
            self.quietly(off.collect, self.args)
        self.assertFalse(list(self.args.output_dir.glob("results_*.tar")))

    def test_active_account_work_blocks_archive(self):
        with patch.object(off, "capture", return_value={"returncode": 0, "stdout": "999999|RUNNING|clean_models\n", "stderr": ""}):
            _, _, result = self.quietly(off.check, self.args)
        self.assertFalse(result["ready_for_collection"])
        self.assertTrue(result["active_jobs"])

    def test_scan_data_excluded_derived_latents_retained(self):
        data = self.code/"data"; data.mkdir()
        (data/"simplified_padded_tensors.npz").write_bytes(b"RAW SCANS")
        (data/"other_scans.npz").write_bytes(b"RAW SCANS")
        dataset = self.project/"datasets"; dataset.mkdir()
        (dataset/"raw.npz").write_bytes(b"RAW SCANS")
        (dataset/"membership.tsv").write_text("participant_id\nPERSON\n")
        run = Path(self.plan["tasks"][0]["run_dir"])
        (run/"computed_latents.npy").write_bytes(b"DERIVED RESULTS")
        files, _ = off.inventory(self.args, self.plan)
        names = [r["member"] for r in files]
        self.assertFalse(any(x.endswith("raw.npz") or x.endswith("other_scans.npz") or x.endswith("simplified_padded_tensors.npz") for x in names))
        self.assertTrue(any(x.endswith("computed_latents.npy") for x in names))
        self.assertTrue(any(x.endswith("membership.tsv") for x in names))

    def test_full_collection_and_tar_checksums(self):
        self.quietly(off.collect, self.args)
        self.assertEqual(off.read_json(self.args.output_dir/"COMPLETE.json")["planned_fits"], 6)
        for line in (self.args.output_dir/"SHA256SUMS.txt").read_text().splitlines():
            expected, filename = line.split("  ", 1)
            self.assertEqual(hashlib.sha256((self.args.output_dir/filename).read_bytes()).hexdigest(), expected)
        all_names = []
        for archive in self.args.output_dir.glob("results_*.tar"):
            with tarfile.open(archive) as tar:
                all_names.extend(tar.getnames())
        self.assertTrue(any(x.endswith("best_vae_net.pth") for x in all_names))
        self.assertTrue(any(x.endswith("source.py") for x in all_names))
        with (self.args.output_dir/"overview"/"training_data_and_splits.csv").open() as handle:
            overview = list(csv.DictReader(handle))
        self.assertEqual(len([r for r in overview if r["Source"] == "clean comparison plan"]), 6)
        self.assertEqual(len([r for r in overview if r["Source"] == "saved age-only baseline"]), 3)

    def test_mqvae_old_models_plots_and_run_overview_are_preserved(self):
        old = self.project/"MQVAE_old"
        run = old/"checkpoints"/"OLD_GEFION_RUN"
        run.mkdir(parents=True)
        (run/"best_vae_net.pth").write_bytes(b"old saved weights")
        (run/"old_auc.png").write_bytes(b"old plot")
        (old/"legacy_script.py").write_text("print('old training code')")
        (old/"plots").mkdir()
        (old/"plots"/"old_loss.pdf").write_bytes(b"old loss figure")
        (old/"data").mkdir()
        (old/"data"/"simplified_padded_tensors.npz").write_bytes(b"scan inputs must be excluded")
        off.save_json(run/"training_summary.json", {"train_samples": 20, "validation_samples": 5, "validation_ids": ["OLD_PERSON"]})
        off.save_json(run/"cond_stats.json", {"only_gender": "F", "aux_binary_col": "old_endpoint"})
        self.quietly(off.collect, self.args)
        names = []
        for archive in self.args.output_dir.glob("results_*.tar"):
            with tarfile.open(archive) as handle:
                names.extend(handle.getnames())
        self.assertIn("MQVAE_old/checkpoints/OLD_GEFION_RUN/best_vae_net.pth", names)
        self.assertIn("MQVAE_old/checkpoints/OLD_GEFION_RUN/old_auc.png", names)
        self.assertIn("MQVAE_old/plots/old_loss.pdf", names)
        self.assertIn("MQVAE_old/legacy_script.py", names)
        self.assertNotIn("MQVAE_old/data/simplified_padded_tensors.npz", names)
        with (self.args.output_dir/"overview"/"training_data_and_splits.csv").open() as handle:
            overview = list(csv.DictReader(handle))
        self.assertTrue(any(row["Run"] == "OLD_GEFION_RUN" for row in overview))
        inventory = off.read_json(self.args.output_dir/"collection_inventory.json")
        self.assertIn(str(old), inventory["source_directories"])

    def test_multiple_nested_checkpoint_trees_and_duplicate_names(self):
        directories = [
            self.project/"MQVAE"/"older_version"/"checkpoints"/"SAME_RUN",
            self.project/"MQVAE_new"/"experiments"/"checkpoints_v2"/"SAME_RUN",
            self.project/"MQVAE_old"/"backup"/"nested"/"checkpoints"/"SAME_RUN",
            self.project/"MQVAE_old"/"another_version"/"checkpoints_extra"/"WEIGHTS_ONLY"]
        for run in directories:
            run.mkdir(parents=True)
            (run/"best_vae_net.pth").write_bytes(b"nested weights")
            (run/"loss.png").write_bytes(b"nested plot")
            (run/"latents.npz").write_bytes(b"derived latent results")
            if run.name != "WEIGHTS_ONLY":
                off.save_json(run/"training_summary.json", {"train_samples": 8, "validation_samples": 2, "validation_ids": ["SAME_PERSON"]})
        self.quietly(off.collect, self.args)
        names = []
        for archive in self.args.output_dir.glob("results_*.tar"):
            with tarfile.open(archive) as handle:
                names.extend(handle.getnames())
        for run in directories:
            member = run.relative_to(self.project).as_posix()
            self.assertIn(member+"/best_vae_net.pth", names)
            self.assertIn(member+"/loss.png", names)
            self.assertIn(member+"/latents.npz", names)
        with (self.args.output_dir/"overview"/"training_data_and_splits.csv").open() as handle:
            overview = list(csv.DictReader(handle))
        found = [row for row in overview if row["Run"] == "SAME_RUN"]
        self.assertEqual(len(found), 3)
        self.assertEqual(len({row["RunDirectory"] for row in found}), 3)
        self.assertTrue(any(row["Run"] == "WEIGHTS_ONLY" and "unverified" in row["Status"] for row in overview))
        with (self.args.output_dir/"overview"/"participant_splits.csv").open() as handle:
            members = [row for row in csv.DictReader(handle) if row["Run"] == "SAME_RUN"]
        self.assertEqual(len({row["RunDirectory"] for row in members}), 3)

    def test_virtual_environments_are_skipped_before_external_paths_resolve(self):
        old = self.project/"MQVAE_old"
        for name in ("venv_dir", "env_dir", "custom_python_environment"):
            env = old/name
            (env/"bin").mkdir(parents=True)
            (env/"bin"/"python").write_bytes(b"placeholder for external interpreter link")
            (env/"site-packages").mkdir()
            (env/"site-packages"/"python_path.pth").write_text("package path, not a trained model")
            if name == "custom_python_environment":
                (env/"pyvenv.cfg").write_text("home = /software/python")
        from unittest.mock import patch
        original = Path.resolve
        def guarded(path, *args, **kwargs):
            if any(name in path.parts for name in ("venv_dir", "env_dir", "custom_python_environment")):
                raise AssertionError("Virtual environment should be excluded before resolving: "+str(path))
            return original(path, *args, **kwargs)
        with patch.object(Path, "resolve", guarded):
            files, _ = off.inventory(self.args, self.plan)
            runs = off.discover_run_directories(self.args, self.plan)
        self.assertFalse(any(any(name in r["member"] for name in ("venv_dir", "env_dir", "custom_python_environment")) for r in files))
        self.assertFalse(any(any(name in str(r) for name in ("venv_dir", "env_dir", "custom_python_environment")) for r in runs))
        self.assertTrue(any(r["member"].endswith("best_vae_net.pth") for r in files))

    def test_external_model_weights_still_trigger_boundary_guard(self):
        run = self.project/"MQVAE_old"/"checkpoints"/"LEGITIMATE_RUN"
        run.mkdir(parents=True)
        model = run/"best_vae_net.pth"
        model.write_bytes(b"model")
        original = Path.resolve
        def outside(path, *args, **kwargs):
            if path.name == model.name and "LEGITIMATE_RUN" in path.parts:
                return original(self.project.parent)/"outside_model.pth"
            return original(path, *args, **kwargs)
        with patch.object(Path, "resolve", outside):
            with self.assertRaisesRegex(ValueError, "outside project"):
                off.inventory(self.args, self.plan)

    def test_download_verification_detects_changed_archive(self):
        self.quietly(off.collect, self.args)
        self.quietly(off.verify_download, self.args)
        archive = next(self.args.output_dir.glob("results_*.tar"))
        with archive.open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            self.quietly(off.verify_download, self.args)

    def test_small_parts_remain_independent_valid_tar_files(self):
        source = self.project/"one"; source.write_bytes(b"abc")
        second = self.project/"two"; second.write_bytes(b"def")
        destination = self.project/"parts"; destination.mkdir()
        rows = [{"path": p, "member": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in (source, second)]
        parts = off.pack(rows, destination, 3)
        self.assertEqual(len(parts), 2)
        for expected, name in parts:
            self.assertEqual(hashlib.sha256((destination/name).read_bytes()).hexdigest(), expected)
            with tarfile.open(destination/name) as archive:
                self.assertEqual(len(archive.getnames()), 1)

    def test_changed_sources_are_not_silently_archived(self):
        source = self.project/"changing"; source.write_bytes(b"abc")
        row = {"path": source, "member": "changing", "bytes": 3, "mtime_ns": source.stat().st_mtime_ns}
        source.write_bytes(b"changed contents")
        destination = self.project/"parts"; destination.mkdir()
        with self.assertRaisesRegex(RuntimeError, "changed"):
            off.pack([row], destination, 1024)

    def test_collection_submission_is_cpu_only_and_has_no_export_override(self):
        import subprocess
        response = subprocess.CompletedProcess([], 0, "888888"+chr(10), "")
        with patch.object(off.subprocess, "run", return_value=response) as submitter:
            self.quietly(off.submit, self.args)
        command = submitter.call_args.args[0]
        self.assertIn("--account=cu_0080", command)
        self.assertIn("--cpus-per-task=2", command)
        self.assertFalse(any(x.startswith("--gpus") or x.startswith("--export") for x in command))
        self.assertIn(" collect ", (self.args.output_dir/"collect_offboard.sh").read_text())
        self.assertEqual(off.read_json(self.args.output_dir/"collection_job.json")["job_id"], "888888")

    def test_unavailable_comparisons_are_visible_not_counted_as_successful_fits(self):
        plan = off.read_json(self.args.study_dir/"plan.json")
        plan["cohorts"].append({"target": "coronary_volume", "group": "F", "status": "unavailable", "reason": "example missing reference"})
        off.save_json(self.args.study_dir/"plan.json", plan)
        _, _, result = self.quietly(off.check, self.args)
        self.assertEqual(result["passed_fits"], 6)
        self.assertEqual(len(result["unavailable_comparisons"]), 1)


if __name__ == "__main__":
    unittest.main()
