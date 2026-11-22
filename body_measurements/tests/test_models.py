from __future__ import annotations

import io
import unittest

import joblib
import numpy as np
import pandas as pd

from body_measurements.models import _regression_metrics, fit_models, predict


class MeasurementModelsTests(unittest.TestCase):
    def setUp(self):
        values = np.arange(50, dtype=float)
        self.X = pd.DataFrame({"size": values, "shape": values % 3}, index=[f"scan_{i}" for i in range(50)])
        self.y = pd.DataFrame({"waist": 60 + 2 * values, "hip": 90 + values}, index=self.X.index)
        self.split = pd.Series(["train"] * 30 + ["validation"] * 10 + ["test"] * 10, index=self.X.index)

    def fit(self, y=None, **kwargs):
        return fit_models(self.X, self.y if y is None else y, self.split, n_estimators=12, n_jobs=1, **kwargs)

    def test_scaler_uses_only_training_features_and_test_labels_do_not_select(self):
        bundle, metrics, predictions, selection = self.fit()
        model = bundle["models"]["waist"]
        self.assertTrue(selection.set_index("target").loc["waist", "selected_model"].startswith("ridge_"))
        np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, self.X.iloc[:30].mean())
        changed = self.y.copy()
        changed.loc[self.split.eq("test")] = -50000
        other_bundle, other_metrics, other_predictions, other_selection = self.fit(changed)
        pd.testing.assert_frame_equal(selection, other_selection)
        pd.testing.assert_frame_equal(predict(bundle, self.X), predict(other_bundle, self.X))
        pd.testing.assert_series_equal(predictions.predicted_cm, other_predictions.predicted_cm)
        self.assertFalse(metrics.mae_cm.equals(other_metrics.mae_cm))

    def test_held_out_targets_do_not_change_candidate_fits(self):
        _, _, predictions, _ = self.fit()
        changed = self.y.copy()
        changed.loc[self.split.ne("train")] += 100000
        _, _, changed_predictions, _ = self.fit(changed)
        original_validation = predictions.query("split == 'validation'").set_index(["scan_id", "target", "model"])
        changed_validation = changed_predictions.query("split == 'validation'").set_index(["scan_id", "target", "model"])
        pd.testing.assert_series_equal(original_validation.predicted_cm, changed_validation.predicted_cm)

    def test_missing_labels_are_masked_independently_and_predictions_retained(self):
        y = self.y.copy()
        y.loc[self.X.index[:15], "hip"] = np.nan
        y.loc[self.X.index[30:37], "hip"] = np.nan
        y["sparse"] = np.nan
        y.loc[self.X.index[:3], "sparse"] = 45
        bundle, metrics, predictions, selection = self.fit(y)
        by_target = selection.set_index("target")
        self.assertEqual(by_target.loc["waist", "n_train"], 30)
        self.assertEqual(by_target.loc["hip", "n_train"], 15)
        self.assertEqual(by_target.loc["sparse", "status"], "skipped")
        self.assertEqual(set(bundle["target_names"]), {"waist", "hip"})
        hip_validation = predictions.query("target == 'hip' and split == 'validation'")
        self.assertEqual(len(hip_validation), 10 * 6)
        self.assertEqual(hip_validation.actual_cm.isna().sum(), 7 * 6)
        self.assertTrue((metrics.query("target == 'hip' and split == 'validation'").n == 3).all())

    def test_sparse_validation_uses_training_median(self):
        y = self.y.copy()
        y.loc[self.split.eq("validation"), "waist"] = np.nan
        bundle, metrics, _, selection = self.fit(y)
        waist = selection.set_index("target").loc["waist"]
        self.assertEqual(waist.status, "baseline_only_insufficient_validation")
        self.assertEqual(waist.selected_model, "train_median")
        self.assertTrue(metrics.query("target == 'waist' and split == 'validation'").mae_cm.isna().all())
        np.testing.assert_allclose(predict(bundle, self.X).waist, self.y.waist.iloc[:30].median())

    def test_all_targets_skipped_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "No measurement models.*training labels"):
            self.fit(min_train_labels=100)

    def test_serialization_and_reordered_features(self):
        bundle, _, _, _ = self.fit()
        stream = io.BytesIO()
        joblib.dump(bundle, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        pd.testing.assert_frame_equal(predict(bundle, self.X), predict(restored, self.X[self.X.columns[::-1]]))

    def test_schema_and_nonfinite_features_rejected(self):
        bundle, _, _, _ = self.fit()
        for X in (self.X.drop(columns="shape"), self.X.assign(metadata=1),
                  self.X.assign(shape=np.nan), self.X.assign(shape=np.inf),
                  self.X.rename(columns={"shape": "size"})):
            with self.subTest(columns=X.columns.tolist()):
                with self.assertRaises(ValueError):
                    predict(bundle, X)

    def test_metrics_mask_missing_truth_and_use_sample_limits_of_agreement(self):
        scores = _regression_metrics(np.array([1, 2, np.nan]), np.array([2, 1, 300]))
        self.assertEqual(scores["n"], 2)
        self.assertEqual(scores["mae_cm"], 1)
        self.assertEqual(scores["rmse_cm"], 1)
        self.assertEqual(scores["bias_cm"], 0)
        self.assertAlmostEqual(scores["loa_upper_cm"], 1.96 * np.sqrt(2))
        self.assertEqual(scores["within_1_cm"], 1)
        single = _regression_metrics(np.array([1]), np.array([2]))
        self.assertTrue(np.isnan(single["r2"]))
        self.assertTrue(np.isnan(single["loa_lower_cm"]))
        constant = _regression_metrics(np.array([1, 1]), np.array([1, 1]))
        self.assertTrue(np.isnan(constant["r2"]))


if __name__ == "__main__":
    unittest.main()
