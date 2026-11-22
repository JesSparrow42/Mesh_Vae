"""CPU-only synthetic checks for the standalone requested figure exporter."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from paper_analysis import make_requested_figures as figures
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


class RequestedFiguresTests(unittest.TestCase):
    def setUp(self):
        if not HAS_MATPLOTLIB:
            mocked = MagicMock()
            mocked.subplots.side_effect = lambda **kwargs: (MagicMock(), MagicMock())
            patcher = patch.object(figures, "pyplot", return_value=mocked)
            patcher.start(); self.addCleanup(patcher.stop)

    def arguments(self, root, action="auc"):
        return argparse.Namespace(action=action, platform="gefion", checkpoints_dir=root,
                                  output_dir=root/"figures", old_run_tag="paper_v1", next_run_tag="next_v1", bootstraps=100)

    def fixture(self, root):
        for gender in ("F", "M"):
            new = root/f"GEFION_NEXT_weight_binary_age_{gender}_next_v1_seed42"/"study"/"heldout"
            new.mkdir(parents=True)
            data = pd.DataFrame({"participant_id": [f"GLODET{i:06d}" for i in range(12)],
                                 "target": [0]*6+[1]*6, "meshvae": np.linspace(.1, .9, 12),
                                 "age": [.1, .4, .2, .5, .3, .6, .4, .8, .5, .7, .6, .9]})
            data.to_csv(new/"predictions.csv", index=False)
            old = root/f"GEFION_SCANONLY_ANYPLAQ_{gender}_paper_v1_seed42"
            diagnostic = old/"binary_diagnostics"/"best_recorded"
            diagnostic.mkdir(parents=True)
            historical = data[["participant_id", "target", "meshvae"]].rename(columns={"meshvae": "probability"})
            historical.iloc[::-1].to_csv(diagnostic/"predictions.csv", index=False)
            pd.DataFrame({"ResultsDir": [str(diagnostic)]}).to_csv(old/"binary_summary.csv", index=False)

    def test_auc_math_ties_and_deterministic_paired_resamples(self):
        self.assertEqual(figures.auc(np.array([0, 0, 1, 1]), [.5]*4), .5)
        self.assertEqual(figures.auc(np.array([0, 0, 1, 1]), [.1, .2, .7, .9]), 1)
        data = pd.DataFrame({"target": [0, 0, 0, 1, 1, 1], "a": [.2, .6, .4, .3, .8, .7]})
        data["b"] = data.a
        a = figures.auc_intervals(data, ["a", "b"], 120)
        self.assertEqual(a, figures.auc_intervals(data, ["a", "b"], 120))
        self.assertEqual(a[0]["CI95Lower"], a[1]["CI95Lower"])
        self.assertEqual(a[0]["CI95Upper"], a[1]["CI95Upper"])
        self.assertGreater(a[0]["CI95Upper"], a[0]["CI95Lower"])

    def test_id_and_probability_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"prediction.csv"
            base = pd.DataFrame({"participant_id": ["!GLODET000125!.obj", "GLODET000325"], "target": [0, 1], "age": [.3, .7]})
            base.to_csv(path, index=False)
            self.assertEqual(list(figures.read_predictions(path, ["age"]).index), ["GLODET000125", "GLODET000325"])
            for column, value in (("age", 1.1), ("age", np.nan), ("target", 2), ("participant_id", "GLODET000125")):
                data = base.copy(); data.loc[1, column] = value; data.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    figures.read_predictions(path, ["age"])

    def test_recorded_history_aggregation_and_timestamp_pairing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); metrics = root/"GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42"/"metrics"
            metrics.mkdir(parents=True)
            train = pd.DataFrame({"epoch": [1, 1, 2, 2], "loss": [4, 2, 3, 1], "binary_loss": [.8, .6, .4, .2]})
            val = pd.DataFrame({"epoch": [1, 2], "loss": [3.5, 2.5], "binary_loss": [.7, .5]})
            for stamp in ("20260901_100000", "20260902_100000"):
                train.to_csv(metrics/f"train_metrics_{stamp}.csv", index=False)
                val.to_csv(metrics/f"val_metrics_{stamp}.csv", index=False)
            train.to_csv(metrics/"train_metrics_20260903_100000.csv", index=False)
            combined, batches = figures.epoch_history(metrics/"train_metrics_20260901_100000.csv", metrics/"val_metrics_20260901_100000.csv")
            self.assertEqual(combined.train_loss.tolist(), [3, 2]); self.assertEqual(batches, 4)
            history = figures.make_losses(self.arguments(root, "losses"))
            self.assertEqual(sum(x["status"] == "plotted" for x in history), 2)
            self.assertEqual(sum(x["status"] == "plotted_training_only" for x in history), 1)
            self.assertEqual(sum(x["status"] == "missing" for x in history), 1)
            if HAS_MATPLOTLIB:
                self.assertEqual(len(list((root/"figures").glob("*_loss.png"))), 6)
            self.assertEqual(len(list((root/"figures").glob("*_epoch_metrics.csv"))), 3)

    def test_same_cohort_permuted_old_rows_three_way_and_no_ids_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            result = figures.make_auc(self.arguments(root))
            self.assertTrue(result["three_way_comparison"])
            self.assertTrue(all(x["status"] == "exact_match" for x in result["audit"].values()))
            table = pd.read_csv(root/"figures"/"auc_summary.csv")
            self.assertEqual(len(table), 6)
            self.assertNotIn("participant_id", table)
            self.assertNotIn("GLODET", (root/"figures"/"auc_audit.json").read_text())
            self.assertEqual(len(pd.read_csv(root/"figures"/"paired_auc_differences.csv")), 2)
            if HAS_MATPLOTLIB:
                for suffix in ("png", "pdf"):
                    self.assertGreater((root/"figures"/f"gefion_auc_comparison.{suffix}").stat().st_size, 1000)

    def test_different_ids_or_targets_never_intersect_or_three_way(self):
        for changed_column, value, expected in (("participant_id", "OTHER_PERSON", "different_participants"), ("target", 1, "target_mismatch")):
            with self.subTest(change=changed_column), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); self.fixture(root)
                old = root/"GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42"/"binary_diagnostics"/"best_recorded"/"predictions.csv"
                data = pd.read_csv(old); data.loc[data.index[-1], changed_column] = value; data.to_csv(old, index=False)
                with warnings.catch_warnings(record=True) as caught:
                    result = figures.make_auc(self.arguments(root))
                self.assertTrue(caught); self.assertFalse(result["three_way_comparison"])
                self.assertEqual(result["audit"]["F"]["status"], expected)
                table = pd.read_csv(root/"figures"/"auc_summary.csv")
                self.assertEqual(len(table[table.Cohort.eq("verified_comparison")]), 4)
                self.assertTrue((table.N == 12).all())
                if HAS_MATPLOTLIB:
                    self.assertTrue((root/"figures"/"gefion_scans_only_historical.png").exists())

    def test_missing_summary_never_chooses_latest_and_ambiguous_summary_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root)
            run = root/"GEFION_SCANONLY_ANYPLAQ_F_paper_v1_seed42"
            (run/"binary_summary.csv").unlink()
            with self.assertRaises(FileNotFoundError):
                figures.old_predictions_path(run)
            with warnings.catch_warnings(record=True):
                result = figures.make_auc(self.arguments(root))
            self.assertEqual(result["audit"]["F"]["status"], "historical_unavailable")
            pd.DataFrame({"ResultsDir": ["a", "b"]}).to_csv(run/"binary_summary.csv", index=False)
            with self.assertRaisesRegex(ValueError, "unambiguous"):
                figures.old_predictions_path(run)
            pd.DataFrame({"ResultsDir": ["missing"], "Experiment": ["OTHER_RUN"]}).to_csv(run/"binary_summary.csv", index=False)
            with self.assertRaisesRegex(ValueError, "expected run"):
                figures.old_predictions_path(run)

    @unittest.skipUnless(HAS_MATPLOTLIB, "Bundled runtime has no matplotlib; no dependencies installed.")
    def test_real_png_and_pdf_render(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.fixture(root); figures.make_auc(self.arguments(root))
            self.assertGreater((root/"figures"/"gefion_auc_comparison.png").stat().st_size, 1000)
            self.assertGreater((root/"figures"/"gefion_auc_comparison.pdf").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
