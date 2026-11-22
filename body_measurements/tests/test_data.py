"""Checks for measurement-label alignment and leakage-safe participant splits."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from body_measurements.data import (
    align_training, clean_id, clean_participant_id, load_config, load_features, make_splits, read_metadata,
)


class MeasurementDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def features(self, ids):
        return pd.DataFrame({"height_feature": np.arange(len(ids), dtype=float)},
                            index=pd.Index(ids, name="scan_id"))

    def test_config_requires_units_and_skips_unmapped_targets(self):
        path = self.root / "targets.json"
        path.write_text(json.dumps({"targets": {
            "height": {"column": "height", "unit": "cm"},
            "waist": {"column": None, "unit": None},
        }}))
        self.assertEqual([row["name"] for row in load_config(path)], ["height"])
        path.write_text(json.dumps({"targets": {"height": {"column": "height"}}}))
        with self.assertRaisesRegex(ValueError, "explicit unit"):
            load_config(path)

    def test_config_rejects_duplicate_columns(self):
        path = self.root / "targets.json"
        path.write_text(json.dumps({"targets": {
            "waist": {"column": "circ", "unit": "cm"},
            "hip": {"column": "circ", "unit": "cm"},
        }}))
        with self.assertRaisesRegex(ValueError, "more than one"):
            load_config(path)

    def test_quoted_decimal_comma_export_preserves_ids(self):
        path = self.root / "manual.csv"
        path.write_bytes('scan;"waist, manual";note\n001;"84,5";"m\u00e5l; tape"\n'.encode("cp1252"))
        frame = read_metadata(path)
        self.assertEqual(frame.loc[0, "scan"], "001")
        self.assertEqual(frame.loc[0, "waist, manual"], "84,5")
        self.assertEqual(frame.loc[0, "note"], "m\u00e5l; tape")
        self.assertEqual(frame.attrs["encoding"], "cp1252")

    def test_quoted_csv_and_bom_tsv(self):
        for delimiter, encoding in ((",", "utf-8"), ("\t", "utf-8-sig")):
            with self.subTest(delimiter=delimiter):
                path = self.root / "table.csv"
                path.write_text(f'"scan"{delimiter}"waist"\n"001"{delimiter}"84,5"\n', encoding=encoding)
                self.assertEqual(read_metadata(path).loc[0, "waist"], "84,5")

    def test_malformed_rows_are_not_silently_skipped(self):
        path = self.root / "broken.csv"
        for text in ('id;waist\na;80;extra\n', 'id;waist\na\n', 'id;waist\na;"80\n'):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, "malformed"):
                    read_metadata(path)

    def test_alignment_units_missing_values_and_order(self):
        metadata = pd.DataFrame({
            "scan": ["c", "a", "b", "d", "e", None, "outside"],
            "person": ["p3", "p1", "p2", "p4", "p5", "p6", "p7"],
            "height_m": ["1,72", "1.80", "0", "nan", "inf", "1.7", "1.6"],
            "waist_mm": ["", "850", "910", "-2", "5000", "900", "800"],
        })
        targets = [
            {"name": "height", "column": "height_m", "unit": "m"},
            {"name": "waist", "column": "waist_mm", "unit": "mm", "max_cm": 200},
        ]
        X, y, groups, audit = align_training(self.features(["b", "a", "c", "d", "e", "unmatched"]),
                                               metadata, targets, "scan", "person")
        self.assertEqual(X.index.tolist(), ["b", "a", "c"])
        self.assertEqual(y.index.tolist(), X.index.tolist())
        self.assertEqual(groups.tolist(), ["p2", "p1", "p3"])
        self.assertAlmostEqual(y.loc["a", "height"], 180.0)
        self.assertAlmostEqual(y.loc["a", "waist"], 85.0)
        self.assertAlmostEqual(y.loc["c", "height"], 172.0)
        self.assertTrue(np.isnan(y.loc["b", "height"]))
        self.assertTrue(np.isnan(y.loc["c", "waist"]))
        self.assertEqual(audit["rows_without_any_target"], 2)
        self.assertEqual(audit["feature_rows_without_metadata"], 1)
        self.assertEqual(audit["metadata_rows_empty_id"], 1)
        self.assertEqual(audit["metadata_rows_without_scan"], 1)
        self.assertEqual(audit["targets"]["waist"]["above_max"], 1)
        self.assertEqual(audit["targets"]["height"]["invalid_numeric"], 1)

    def test_all_missing_labels_return_empty_alignment(self):
        metadata = pd.DataFrame({"scan": ["a", "b"], "waist": [None, ""]})
        X, y, groups, audit = align_training(self.features(["a", "b"]), metadata,
            [{"name": "waist", "column": "waist", "unit": "cm"}], "scan")
        self.assertEqual(len(X), 0)
        self.assertEqual(len(y), 0)
        self.assertEqual(len(groups), 0)
        self.assertEqual(audit["targets"]["waist"]["source_missing"], 2)

    def test_duplicate_cleaned_metadata_ids_fail(self):
        metadata = pd.DataFrame({"scan": ["ABDET12345.obj", "abdet12345"], "waist": [80, 85]})
        with self.assertRaisesRegex(ValueError, "Duplicate metadata"):
            align_training(self.features(["ABDET12345"]), metadata,
                [{"name": "waist", "column": "waist", "unit": "cm"}], "scan")

    def test_no_matching_scans_returns_audit_and_empty_frames(self):
        metadata = pd.DataFrame({"scan": ["b"], "waist": [80]})
        X, y, groups, audit = align_training(self.features(["a"]), metadata,
            [{"name": "waist", "column": "waist", "unit": "cm"}], "scan")
        self.assertTrue(X.empty)
        self.assertTrue(y.empty)
        self.assertTrue(groups.empty)
        self.assertEqual(audit["matched_rows"], 0)
        self.assertEqual(audit["targets"]["waist"]["coverage_fraction"], 0.0)

    def test_requested_missing_columns_and_missing_person_fail(self):
        metadata = pd.DataFrame({"scan": ["a"], "person": [None], "waist": [80]})
        targets = [{"name": "waist", "column": "waist", "unit": "cm"}]
        with self.assertRaisesRegex(ValueError, "missing requested columns"):
            align_training(self.features(["a"]), metadata, targets, "absent")
        with self.assertRaisesRegex(ValueError, "missing participant"):
            align_training(self.features(["a"]), metadata, targets, "scan", "person")

    def test_features_validate_and_filter_without_changing_columns(self):
        path = self.root / "features.npz"
        np.savez(path, names=np.array(["a", "b", "c", "d"], dtype=object),
                 feature_names=np.array(["height", "volume"], dtype=object),
                 features=[[170, 80], [np.nan, 70], [160, 90], [150, 60]],
                 valid=[True, True, False, True], scale_proxy=[1, 1, 1, 0])
        frame, audit = load_features(path)
        self.assertEqual(frame.index.tolist(), ["a"])
        self.assertEqual(frame.columns.tolist(), ["height", "volume"])
        self.assertEqual(audit["rows_total"], 4)
        self.assertEqual(audit["rows_invalid"], 3)

    def test_features_duplicate_ids_and_schema_fail(self):
        path = self.root / "features.npz"
        np.savez(path, names=["a.obj", "a"], feature_names=["height"], features=[[170], [160]])
        with self.assertRaisesRegex(ValueError, "duplicate scan IDs"):
            load_features(path)
        np.savez(path, names=["a"], feature_names=["height"], features=[[170, 160]])
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            load_features(path)

    def test_splits_keep_people_together_and_ignore_row_order(self):
        groups = pd.Series([f"person{i // 2:02d}" for i in range(40)],
                           index=[f"scan{i}" for i in range(40)])
        splits = make_splits(groups, seed=42)
        self.assertEqual(set(splits), {"train", "validation", "test"})
        for person in groups.unique():
            self.assertEqual(splits.loc[groups == person].nunique(), 1)
        reordered = make_splits(groups.sample(frac=1, random_state=1), seed=42)
        pd.testing.assert_series_equal(splits.sort_index(), reordered.sort_index())
        group_sets = {split: set(groups.loc[splits == split]) for split in splits.unique()}
        self.assertFalse(group_sets["train"] & group_sets["test"])
        self.assertFalse(group_sets["train"] & group_sets["validation"])
        self.assertFalse(group_sets["test"] & group_sets["validation"])

    def test_explicit_participant_ids_preserve_slashes_and_det_substrings(self):
        people = ["clinic/person001", "family_ABDET12345_member1", "family_ABDET12345_member2"]
        metadata = pd.DataFrame({"scan": ["a", "b", "c"], "person": people, "waist": [80, 90, 100]})
        _, _, groups, audit = align_training(self.features(["a", "b", "c"]), metadata,
            [{"name": "waist", "column": "waist", "unit": "cm"}], "scan", "person")
        self.assertEqual(groups.tolist(), people)
        self.assertEqual(audit["participants_retained"], 3)
        self.assertEqual(len(set(make_splits(groups, seed=42))), 3)
        self.assertEqual([clean_participant_id(value) for value in groups], people)

    def test_too_few_people_and_bad_fractions_fail(self):
        with self.assertRaisesRegex(ValueError, "At least three"):
            make_splits(pd.Series(["a", "a", "b"]), seed=42)
        with self.assertRaisesRegex(ValueError, "fractions"):
            make_splits(pd.Series(["a", "b", "c"]), seed=42, test_fraction=0.6, val_fraction=0.5)

    def test_clean_ids_preserves_generic_ids_and_refuses_nulls(self):
        self.assertEqual(clean_id(r"C:\scans\ABDET12345_simplified.obj"), "ABDET12345")
        self.assertEqual(clean_id("000012.obj"), "000012")
        self.assertEqual(clean_id("visit.001"), "visit.001")
        for value in (None, np.nan, "NaN", "NULL", "", pd.NA):
            self.assertIsNone(clean_id(value))


if __name__ == "__main__":
    unittest.main()
