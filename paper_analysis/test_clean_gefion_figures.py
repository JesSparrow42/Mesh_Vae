"""Numerical and alignment tests; no GPU, fitting, or display required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from paper_analysis import clean_gefion_figures as f


class FigureTests(unittest.TestCase):
    def fixture(self, root):
        cohorts = []
        for group in f.GROUPS:
            base = root/"peripheral_total"/group
            (base/"splits").mkdir(parents=True); (base/"predictions").mkdir()
            y = np.array([0, 1]*8)
            sex = [group]*16 if group != "combined" else ["F"]*8+["M"]*8
            test = pd.DataFrame({"participant_id": [f"GLODET{i:06d}" for i in range(16)],
                "study_gender": sex, "presence_peripheral_total": y, "volume_peripheral_total": y*20.})
            test.to_csv(base/"splits"/"test.tsv", sep="\t", index=False)
            for variant in f.VARIANTS:
                pd.DataFrame({"participant_id": test.participant_id, "target": y,
                    "probability": np.linspace(.05, .95, 16), "group": group}).iloc[::-1].to_csv(base/"predictions"/f"{variant}.csv", index=False)
            cohorts.append({"target": "peripheral_total", "group": group})
        (root/"plan.json").write_text(json.dumps({"combined_policy": "dedicated", "cohorts": cohorts}))

    def test_ties_auc_and_shared_paired_draws(self):
        y = np.array([0, 0, 1, 1]); p = np.array([.1, .6, .4, .9])
        self.assertEqual(f.roc_auc(y, p), .75)
        self.assertEqual(f.roc_auc(y, np.ones(4)), .5)
        rows, differences = f.intervals(y, {"age_only": p, "scans_age": p.copy()}, ["F"]*4, 100)
        self.assertEqual(rows[0]["CI95Lower"], rows[1]["CI95Lower"])
        self.assertEqual(differences[0]["AUCDifference"], 0.)
        self.assertEqual(differences[0]["CI95Lower"], 0.)
        self.assertEqual(differences[0]["CI95Upper"], 0.)

    def test_complete_collection_is_matched_and_has_no_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            table, differences, inventory, sources = f.collect(root, 100)
            self.assertEqual(len(table), 9); self.assertEqual(len(differences), 6)
            self.assertTrue(inventory.Status.eq("available").all())
            self.assertEqual(set(table.N), {16})
            self.assertNotIn("GLODET", table.to_csv(index=False))
            self.assertEqual(len(sources), 13)

    def test_missing_predictions_stay_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            (root/"peripheral_total"/"M"/"predictions"/"scans_only.csv").unlink()
            table, _, inventory, _ = f.collect(root, 100)
            self.assertEqual(len(table), 8)
            self.assertEqual(int(inventory.Status.eq("missing").sum()), 1)

    def test_target_or_membership_changes_are_not_silently_intersected(self):
        for column, value in (("target", 2), ("participant_id", "OTHER"), ("probability", 1.1)):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); self.fixture(root)
                path = root/"peripheral_total"/"F"/"predictions"/"scans_only.csv"
                data = pd.read_csv(path); data.loc[0, column] = value; data.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    f.collect(root, 100)

    def test_combined_is_not_pooled_sex_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            path = root/"plan.json"; data = json.loads(path.read_text()); data["combined_policy"] = "pooled"; path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "jointly"):
                f.collect(root, 100)

    def test_measured_volume_is_explicitly_a_definition_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            with patch.object(f, "pyplot", return_value=MagicMock()) as fake, patch.object(f, "save"):
                fake.return_value.subplots.return_value = (MagicMock(), MagicMock())
                table = f.volume_definition_check(root, root)
            self.assertEqual(len(table), 3)
            self.assertTrue(table.AUC.eq(1.).all())
            self.assertTrue(table.Meaning.str.contains("not a trained").all())
            path = root/"peripheral_total"/"F"/"splits"/"test.tsv"
            data = pd.read_csv(path, sep="\t"); data.loc[0, "volume_peripheral_total"] = 99; data.to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "definition disagrees"):
                f.volume_definition_check(root, root)

    def test_volume_figure_requires_invariance_and_uses_the_main_endpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            plan = json.loads((root/"plan.json").read_text()); plan["tasks"] = []
            import shutil
            for group in f.GROUPS:
                base = root/f.VOLUME_TARGET/group
                (base/"splits").mkdir(parents=True); (base/"predictions").mkdir()
                source = root/"peripheral_total"/group
                shutil.copyfile(source/"splits"/"test.tsv", base/"splits"/"test.tsv")
                plan["cohorts"].append({"target": f.VOLUME_TARGET, "group": group, "endpoint": "presence_peripheral_total"})
                for variant in f.VOLUME_VARIANTS:
                    shutil.copyfile(source/"predictions"/"scans_age.csv", base/"predictions"/f"{variant}.csv")
                    key = group+variant
                    plan["tasks"].append({"target": f.VOLUME_TARGET, "group": group, "variant": variant, "fingerprint": key})
                    (base/f"{variant}_inference_check.json").write_text(json.dumps({"status": "passed", "fingerprint": key}))
            (root/"plan.json").write_text(json.dumps(plan))
            table, differences, _, _ = f.collect(root, 100)
            self.assertEqual(len(table), 15); self.assertEqual(len(differences), 9)
            (root/f.VOLUME_TARGET/"M"/"volume_aux_inference_check.json").unlink()
            table, _, inventory, _ = f.collect(root, 100)
            self.assertEqual(len(table), 14)
            self.assertEqual(int(inventory.Status.eq("inference_check_pending").sum()), 1)


if __name__ == "__main__":
    unittest.main()
