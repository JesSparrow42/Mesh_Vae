"""CPU integration checks for frozen cohorts, selected tasks, and paired comparison."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from meshvae.options import TrainOptions
from paper_analysis.run_plaque_study import experiments, training_command
from paper_analysis.selected_boson_study import (
    ARMS, ROOT, RUN_TAG, SELECTED, check_reference, check_runtime_compatibility, collect, paired_difference,
    run_name, same_training_settings, task_plan,
)


def reference_fixture(root):
    checkpoints = root/"checkpoints"
    for selection in SELECTED:
        exp = next(e for e in experiments(selection.suite) if e.key == selection.experiment)
        for gender in ("F", "M"):
            run = checkpoints/run_name(selection, gender, "next_v1")
            study = run/"study"; study.mkdir(parents=True)
            config = json.loads((ROOT/"paper_analysis"/"next_steps_config.json").read_text())
            config.update(niter=173, batch_size=3, split_seed=42)
            config["platforms"]["gefion"]["checkpoints"] = str(checkpoints)
            frames, paths = {}, {}
            for split, n in (("train", 60), ("validation", 12), ("test", 10)):
                idx = np.arange(n)
                volume = np.where(idx % 2, idx+3., 0.)
                frames[split] = pd.DataFrame({
                    "participant_id": [f"{gender}_{split}_{i}" for i in idx],
                    "study_gender": gender, "study_age": 30.+idx % 40, "study_weight_kg": 65.+idx % 20,
                    "presence_peripheral_total": (volume > 0).astype(float), "volume_peripheral_total": volume,
                })
                paths[split] = study/f"{split}.tsv"
                frames[split].to_csv(paths[split], sep="\t", index=False)
            manifest = {"experiment": vars(exp), "gender": gender, "config": config,
                        "audit": {}, "cohort_rule": {"cohort": "all"},
                        "members": {k: list(v.participant_id) for k, v in frames.items()},
                        "fingerprint": f"frozen_{selection.experiment}_{gender}"}
            (study/"manifest.json").write_text(json.dumps(manifest))
            (study/"status.json").write_text(json.dumps({"status": "completed"}))
            cmd = training_command(run.name, paths, exp, gender, config, "gefion")
            cmd[cmd.index("--encoder_max_points")+1] = "2048"
            cmd += ["--lr", "0.0004"]
            (study/"train_command.json").write_text(json.dumps(cmd))
            (run/"best_vae_net.pth").write_bytes(b"test fixture; not a trained checkpoint")
            heldout = study/"heldout"; heldout.mkdir()
            y = frames["test"][f"{'presence' if selection.kind == 'binary' else 'volume'}_peripheral_total"].to_numpy()
            pd.DataFrame({"participant_id": frames["test"].participant_id, "target": y,
                          "meshvae": np.full(len(y), .5) if selection.kind == "binary" else y+10}).to_csv(
                              heldout/"predictions.csv", index=False)
            pd.DataFrame({"Model": ["meshvae"]}).to_csv(heldout/"comparison.csv", index=False)
    return checkpoints


class SelectedBosonTests(unittest.TestCase):
    def test_legacy_runner_is_rejected_before_reference_checks(self):
        legacy_help = SimpleNamespace(returncode=0, stdout="--platform --suite --gender --audit-only --dry-run", stderr="")
        with patch("paper_analysis.selected_boson_study.subprocess.run", return_value=legacy_help):
            with self.assertRaisesRegex(ValueError, "Outdated paper_analysis/run_plaque_study.py"):
                check_runtime_compatibility()

    def test_current_runtime_interfaces_pass_without_gpu_or_participant_data(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            check_runtime_compatibility()
        self.assertIn("[PASS]", output.getvalue())

    def test_eight_tasks_and_slurm_bounds(self):
        plan = task_plan()
        self.assertEqual(len(plan), 8)
        self.assertEqual({s.experiment for s, _, _ in plan}, {"binary_age", "all_log"})
        self.assertEqual({(g, a) for _, g, a in plan}, {(g, a) for g in ("F", "M") for a in ARMS})
        script = (ROOT/"paper_analysis"/"gefion_best_boson.sh").read_text()
        self.assertIn("--array=0-7%2", script)
        self.assertNotIn("--export", script)

    def test_all_launches_preserve_frozen_cohorts_budget_and_covariates(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            checkpoints = reference_fixture(root)
            for index, (selection, gender, arm) in enumerate(task_plan()):
                completed = subprocess.run([
                    sys.executable, str(ROOT/"paper_analysis"/"selected_boson_study.py"), "run",
                    "--checkpoints-dir", str(checkpoints), "--task-index", str(index), "--dry-run",
                ], capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
                run, original, original_cmd, old_predictions = check_reference(checkpoints, selection, gender, "next_v1")
                study = checkpoints/run_name(selection, gender, RUN_TAG, arm)/"study"
                manifest = json.loads((study/"manifest.json").read_text())
                self.assertEqual(original["members"], manifest["members"])
                cmd = json.loads((study/"train_command.json").read_text())
                same_training_settings(original_cmd, cmd)
                opt = TrainOptions().parser.parse_args(cmd[2:])
                self.assertEqual((opt.niter, opt.batch_size, opt.encoder_max_points, opt.lr), (173, 3, 2048, .0004))
                self.assertEqual(opt.cond_weight, selection.kind == "regression")
                self.assertTrue(opt.use_cond)
                self.assertEqual(opt.seed, 42)
                self.assertEqual(opt.prior_match_mode, "mmd")
                for split in ("train", "validation", "test"):
                    pd.testing.assert_frame_equal(pd.read_csv(run/"study"/f"{split}.tsv", sep="\t"),
                                                  pd.read_csv(study/f"{split}.tsv", sep="\t"))
                # Supply fixture predictions after dry-run: collection must align shuffled participants.
                heldout = study/"heldout"; heldout.mkdir()
                new = old_predictions.copy()
                if selection.kind == "binary":
                    new.meshvae = .2+.6*new.target if arm == "boson_mmd" else .5
                else:
                    new.meshvae = new.target+(3 if arm == "boson_mmd" else 6)
                new.iloc[::-1].to_csv(heldout/"predictions.csv", index=False)
                (study/"status.json").write_text(json.dumps({"status": "completed"}))
            out = root/"comparison"
            with contextlib.redirect_stdout(io.StringIO()):
                collect(checkpoints, "next_v1", RUN_TAG, out, 20)
            results = pd.read_csv(out/"comparison.csv")
            delta = pd.read_csv(out/"paired_differences.csv")
            self.assertEqual(len(results), 12)
            self.assertEqual(len(delta), 8)
            np.testing.assert_allclose(delta[delta.Metric == "AUC"].BosonMinusComparator, .5)
            np.testing.assert_allclose(delta[(delta.Metric == "MAE") & (delta.Comparator == "gaussian_mmd")].BosonMinusComparator, -3)
            # Changing any core training setting invalidates the comparison.
            changed = cmd.copy(); changed[changed.index("--niter")+1] = "174"
            with self.assertRaisesRegex(ValueError, "training settings differ"):
                same_training_settings(original_cmd, changed)

    def test_missing_runs_remain_missing_and_reference_tampering_is_rejected(self):
        with TemporaryDirectory() as td:
            root = Path(td); checkpoints = reference_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                collect(checkpoints, "next_v1", RUN_TAG, root/"results", 5)
            statuses = pd.read_csv(root/"results"/"status.csv")
            self.assertEqual((statuses.Status == "not_started").sum(), 8)
            self.assertEqual(len(pd.read_csv(root/"results"/"comparison.csv")), 4)
            selection = SELECTED[0]
            run = checkpoints/run_name(selection, "F", "next_v1")
            frame = pd.read_csv(run/"study"/"train.tsv", sep="\t")
            frame.loc[0, "participant_id"] = "changed_person"
            frame.to_csv(run/"study"/"train.tsv", sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "membership/sex/target"):
                check_reference(checkpoints, selection, "F", "next_v1")

    def test_paired_mae_uses_same_participants(self):
        y = np.array([0., 10., 30., 60.])
        result = paired_difference(y, y+2, y+7, "regression", 20)
        self.assertEqual(result["BosonMinusComparator"], -5)
        self.assertEqual((result["CI95Lower"], result["CI95Upper"]), (-5, -5))
        self.assertEqual(result["BetterDirection"], "negative")


if __name__ == "__main__":
    unittest.main()
