from __future__ import annotations

import unittest
import contextlib
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from paper_analysis.bootstrap_metrics import (
    average_precision,
    load_predictions,
    metric_values,
    paired_difference,
    roc_auc,
    summarize_model,
)
from paper_analysis.bootstrap_result_tree import main as bootstrap_tree_main
from paper_analysis.data_utils import derive_any_plaque


class BootstrapMetricTests(unittest.TestCase):
    def test_margin_scores_have_ranking_metrics_but_no_brier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_predictions.csv"
            source = pd.DataFrame({"participant_id": ["a", "b", "c", "d"], "target": [0, 0, 1, 1],
                                   "score_value": [-2, 0.4, -0.3, 2], "pred_sick": [0, 1, 0, 1]})
            source.to_csv(path, index=False)
            frame, _ = load_predictions(path, "binary", 0.5, "error")
            scores = metric_values(frame, "binary")
            self.assertAlmostEqual(scores["auc"], 0.75)
            self.assertAlmostEqual(scores["average_precision"], 5 / 6)
            self.assertTrue(np.isnan(scores["brier"]))
            paired = paired_difference("ref", frame, "cmp", frame, "binary", 100, 7, 0.95, "auto")
            brier = paired.set_index("metric").loc["brier"]
            self.assertEqual(brier["n_bootstrap_valid"], 0)
            source.rename(columns={"score_value": "probability"}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                load_predictions(path, "binary", 0.5, "error")

    def test_average_precision_groups_ties_and_is_row_order_independent(self) -> None:
        target = np.array([1, 0, 1, 0])
        self.assertAlmostEqual(average_precision(target, np.ones(4)), 0.5)
        scores = np.array([0.9, 0.8, 0.8, 0.1])
        self.assertAlmostEqual(average_precision(target, scores), 5 / 6)
        order = [3, 2, 1, 0]
        self.assertAlmostEqual(average_precision(target[order], scores[order]), 5 / 6)

    def test_loader_rejects_nonfinite_predictions_and_invalid_binary_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_predictions.csv"
            for task, column in (("binary", "probability"), ("regression", "prediction")):
                for invalid in (np.nan, np.inf, -np.inf, "invalid"):
                    with self.subTest(task=task, value=invalid):
                        pd.DataFrame({"participant_id": ["a", "b"], "target": [0, 1], column: [0.2, invalid]}).to_csv(path, index=False)
                        with self.assertRaisesRegex(ValueError, "Nonfinite"):
                            load_predictions(path, task, 0.5, "error")
            for invalid in (0.5, 2, -1, np.nan):
                with self.subTest(decision=invalid):
                    pd.DataFrame({"participant_id": ["a", "b"], "target": [0, 1],
                                  "probability": [0.2, 0.8], "prediction": [0, invalid]}).to_csv(path, index=False)
                    with self.assertRaisesRegex(ValueError, "Binary decisions"):
                        load_predictions(path, "binary", 0.5, "error")
            pd.DataFrame({"participant_id": ["a", "b"], "target": [0, 1], "prediction": [0, 1],
                          "target_transformed": [0, 1], "prediction_transformed": [0, np.inf]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                load_predictions(path, "regression", 0.5, "error")

    def test_loader_filters_split_before_validation_and_preserves_string_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_predictions.csv"
            pd.DataFrame({"participant_id": ["001", "002", "003", "004"],
                          "split": ["test", "held-out", "validation", "train"],
                          "target": [0, 1, 1, 0], "probability": [0.2, 0.8, np.inf, np.nan]}).to_csv(path, index=False)
            frame, task = load_predictions(path, "auto", 0.5, "error")
            self.assertEqual(task, "binary")
            self.assertEqual(frame._id.tolist(), ["001", "002"])
            self.assertEqual(frame.attrs["excluded_non_test_rows"], 2)
            pd.DataFrame({"participant_id": ["001"], "split": ["validation"],
                          "target": [1], "prediction": [1]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "No held-out"):
                load_predictions(path, "auto", 0.5, "error")

    def test_legacy_path_ids_are_limited_to_exact_test_predictions_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frame = pd.DataFrame({"path": [r"C:\ADET0000001\BDET0000007.obj", "/data/BDET0000008.npz"],
                                  "label_sick": [0, 1], "score_value": [0.1, 0.9], "pred_sick": [0, 1]})
            legacy = Path(temporary) / "test_predictions.csv"
            frame.to_csv(legacy, index=False)
            loaded, _ = load_predictions(legacy, "auto", 0.5, "error")
            self.assertEqual(loaded._id.tolist(), ["BDET0000007", "BDET0000008"])
            generic = Path(temporary) / "predictions.csv"
            frame.to_csv(generic, index=False)
            with self.assertRaisesRegex(ValueError, "No participant ID"):
                load_predictions(generic, "auto", 0.5, "error")

    def test_mean_duplicate_binary_scores_produce_valid_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_predictions.csv"
            pd.DataFrame({"participant_id": ["a", "a", "b"], "target": [0, 0, 1],
                          "probability": [0.1, 0.7, 0.9], "prediction": [0, 1, 1]}).to_csv(path, index=False)
            frame, _ = load_predictions(path, "binary", 0.5, "mean")
            self.assertEqual(frame._pred.tolist(), [0, 1])

    def test_default_tree_inventories_ambiguous_and_excludes_non_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = pd.DataFrame({"participant_id": ["a", "b", "c"], "target": [0, 1, 2], "prediction": [0, 1, 2]})
            files = ["run/study/test_predictions.csv", "run/study/validation_predictions.csv",
                     "run/calibration/predictions.csv", "run/study/heldout/predictions.csv",
                     "run/regression_diagnostics/latest_1/predictions.csv",
                     "run/regression_diagnostics/latest_0/predictions.csv",
                     "run/binary_diagnostics/latest_1/predictions.csv"]
            for relative in files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                source.to_csv(path, index=False)
            source.assign(split="test").to_csv(root / files[4], index=False)
            pd.DataFrame({"ResultsDir": ["/remote/run/regression_diagnostics/latest_1"]}).to_csv(root / "run/regression_summary.csv", index=False)
            pd.DataFrame({"ResultsDir": ["/remote/run/binary_diagnostics/latest_1"]}).to_csv(root / "run/binary_summary.csv", index=False)
            output = root / "ci"
            argv = ["bootstrap_result_tree", "--root", str(root), "--output-dir", str(output), "--bootstraps", "100"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                bootstrap_tree_main()
            inventory = pd.read_csv(output / "result_inventory.csv").set_index("relative_path")
            self.assertEqual(inventory.loc[files[0], "status"], "completed")
            self.assertEqual(inventory.loc[files[1], "status"], "excluded_non_test")
            self.assertEqual(inventory.loc[files[2], "status"], "excluded_non_test")
            self.assertEqual(inventory.loc[files[3], "status"], "ambiguous")
            self.assertEqual(inventory.loc[files[4], "status"], "completed")
            self.assertEqual(inventory.loc[files[5], "status"], "ambiguous")
            self.assertEqual(inventory.loc[files[6], "status"], "ambiguous")
            self.assertEqual(len(pd.read_csv(output / "all_result_intervals.csv")), 10)

    def test_auc_matches_hand_calculation(self) -> None:
        target = np.array([0, 0, 1, 1], dtype=float)
        score = np.array([0.1, 0.4, 0.35, 0.8], dtype=float)
        self.assertAlmostEqual(roc_auc(target, score), 0.75)

    def test_perfect_binary_bootstrap_interval_is_one(self) -> None:
        target = np.repeat([0.0, 1.0], 20)
        score = np.concatenate([np.linspace(0.01, 0.20, 20), np.linspace(0.80, 0.99, 20)])
        frame = pd.DataFrame(
            {
                "_id": [f"p{index:03d}" for index in range(40)],
                "_target": target,
                "_score": score,
                "_pred": (score >= 0.5).astype(float),
            }
        )
        summary = summarize_model("perfect", frame, "binary", 200, 7, 0.95, "auto")
        auc = summary.loc[summary["metric"] == "auc"].iloc[0]
        self.assertEqual(auc["estimate"], 1.0)
        self.assertEqual(auc["ci_low"], 1.0)
        self.assertEqual(auc["ci_high"], 1.0)

    def test_paired_comparison_matches_participants_by_id(self) -> None:
        ids = [f"p{index:03d}" for index in range(40)]
        target = np.repeat([0.0, 1.0], 20)
        reference_score = np.tile([0.25, 0.75], 20)
        comparison_score = np.concatenate([np.linspace(0.01, 0.20, 20), np.linspace(0.80, 0.99, 20)])
        reference = pd.DataFrame(
            {
                "_id": ids,
                "_target": target,
                "_score": reference_score,
                "_pred": (reference_score >= 0.5).astype(float),
            }
        )
        comparison = pd.DataFrame(
            {
                "_id": ids[::-1],
                "_target": target[::-1],
                "_score": comparison_score[::-1],
                "_pred": (comparison_score[::-1] >= 0.5).astype(float),
            }
        )
        result = paired_difference(
            "reference", reference, "comparison", comparison, "binary", 200, 11, 0.95, "auto"
        )
        auc = result.loc[result["metric"] == "auc"].iloc[0]
        self.assertEqual(auc["n_paired_participants"], 40)
        self.assertGreater(auc["delta_comparison_minus_reference"], 0.0)
        self.assertEqual(
            auc["directional_improvement"], auc["delta_comparison_minus_reference"]
        )

    def test_regression_metrics_are_exact_for_perfect_predictions(self) -> None:
        target = np.arange(1.0, 11.0)
        frame = pd.DataFrame(
            {
                "_id": [f"p{index:03d}" for index in range(10)],
                "_target": target,
                "_prediction": target.copy(),
            }
        )
        metrics = metric_values(frame, "regression")
        self.assertAlmostEqual(metrics["r2"], 1.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["pearson"], 1.0)
        self.assertAlmostEqual(metrics["spearman"], 1.0)

    def test_any_plaque_is_union_of_available_vascular_beds(self) -> None:
        frame = pd.DataFrame(
            {
                "plaque_volume_right_carotid_corrected": [0.0, 2.0, np.nan, 0.0],
                "plaque_volume_left_femoral_corrected": [0.0, 0.0, 4.0, np.nan],
            }
        )
        result = derive_any_plaque(frame)
        np.testing.assert_allclose(result.to_numpy(), np.array([0.0, 1.0, 1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
