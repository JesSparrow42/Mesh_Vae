"""Independent synthetic checks for the detailed-scan measurement experiment."""
from contextlib import redirect_stdout
import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import torch

import run_body_measurements_pointcloud as pc
from run_body_measurements import fingerprint


class DetailedMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        torch.set_num_threads(1)

    def make_fixture(self):
        root = self.root
        baseline = root / "baseline"
        baseline.mkdir()
        rng = np.random.default_rng(71)
        ids = [f"scan_{i:03}" for i in range(30)]
        X = pd.DataFrame({"height_m": np.linspace(1.5, 1.9, 30),
                          "width_m": rng.uniform(.3, .5, 30)}, index=ids)
        y = pd.DataFrame({"neck": 24 + 8 * X.height_m + 6 * X.width_m,
                          "height": 100 * X.height_m}, index=ids)
        y.loc[ids[:2] + [ids[-1]], "neck"] = np.nan
        split = pd.Series(["train"] * 18 + ["validation"] * 6 + ["test"] * 6, index=ids)
        people = pd.Series([f"person/{i // 2:03}" for i in range(30)], index=ids)
        sources = []
        points, a1, a2 = [], [], []
        for i, scan_id in enumerate(ids):
            cloud = rng.uniform(-.5, .5, (64, 3)) * np.array([X.iloc[i, 1], X.iloc[i, 0], .3])
            cloud[0], cloud[1] = [-.2, -X.iloc[i, 0] / 2, -.15], [.2, X.iloc[i, 0] / 2, .15]
            source = root / f"{scan_id}.obj"
            source.write_text("".join(f"v {x:.12g} {z:.12g} {w:.12g}\n" for x, z, w in cloud))
            sampled = pc.sample_obj((scan_id, str(source), 64, 8, 4, "m", 42))
            self.assertEqual(sampled["status"], "ok", sampled)
            sources.append(str(source))
            points.append(sampled["points"])
            a1.append(sampled["a1"])
            a2.append(sampled["a2"])
        feature_file = root / "geometry.npz"
        np.savez_compressed(feature_file, names=np.asarray(ids, dtype=object),
                            features=X.to_numpy(), feature_names=np.asarray(X.columns, dtype=object),
                            valid=np.ones(len(ids), dtype=bool), scale_proxy=np.ones(len(ids)),
                            source_paths=np.asarray(sources, dtype=object))
        metadata = y.assign(scan_id=ids, person_id=people.to_numpy())
        metadata.to_csv(root / "metadata.csv", index=False)
        config = {"targets": {name: {"column": name, "unit": "cm"} for name in y}}
        (root / "targets.json").write_text(json.dumps(config))
        models = {}
        for target in y:
            rows = split.eq("train") & y[target].notna()
            models[target] = Ridge(alpha=.01).fit(X.loc[rows], y.loc[rows, target])
        bundle = {"format_version": 1, "feature_names": list(X), "target_names": list(y),
                  "target_units": "cm", "models": models, "scan_units": "m", "seed": 42}
        joblib.dump(bundle, baseline / "model.joblib")
        manifest = pd.DataFrame({"scan_id": ids, "participant_id": people.to_numpy(), "split": split.to_numpy()})
        manifest.to_csv(baseline / "splits.csv", index=False)
        run = {"status": "complete", "scan_units": "m", "id_col": "scan_id", "participant_col": "person_id",
               "features": fingerprint(str(feature_file)), "metadata": fingerprint(str(root / "metadata.csv")),
               "config": fingerprint(str(root / "targets.json"))}
        (baseline / "run.json").write_text(json.dumps(run))
        old = pc.baseline_predict(bundle, X)
        prediction_rows = []
        for scan_id in split.index[split.ne("train")]:
            for target in y:
                prediction_rows.append({"scan_id": scan_id, "target": target, "split": split[scan_id],
                                        "actual_cm": y.loc[scan_id, target], "predicted_cm": old.loc[scan_id, target],
                                        "selected": True, "model": "ridge_alpha_0.01"})
        pd.DataFrame(prediction_rows).to_csv(baseline / "evaluation_predictions.csv", index=False)
        cache = root / "cache"
        cache.mkdir()
        settings = {"format_version": 1, "points": 64, "anchors1": 8, "anchors2": 4, "seed": 42,
                    "scan_units": "m", "features": fingerprint(str(feature_file)),
                    "split_sha256": fingerprint(str(baseline / "splits.csv"))["sha256"],
                    "scan_ids": ids, "feature_names": list(X), "centering": "full_OBJ_bounding_box_midpoint",
                    "coordinate_units": "m", "normalization": "translation_only_no_individual_scale_or_rotation"}
        (cache / "cache_settings.json").write_text(json.dumps(settings))
        np.savez_compressed(cache / "point_cache.npz", scan_ids=np.asarray(ids), points=np.stack(points),
                            a1=np.stack(a1), a2=np.stack(a2), features=X.to_numpy(np.float32),
                            feature_names=np.asarray(X.columns, dtype=str))
        return baseline, cache, X, y, split, people

    def train_args(self, baseline, cache, name):
        return argparse.Namespace(baseline_run=str(baseline), cache=str(cache), out_dir=str(self.root / name),
                                  device="cpu", epochs=1, patience=1, batch_size=6, threads=1, neighbors1=8,
                                  neighbors2=4, learning_rate=.001, weight_decay=.0001, bootstrap=30,
                                  seed=17, log_every=100)

    def test_missing_labels_have_finite_loss_and_zero_gradient(self):
        prediction = torch.tensor([[1., 2., 3.], [2., 4., 5.]], requires_grad=True)
        labels = torch.tensor([[float("nan"), 3., float("nan")], [2., 5., float("nan")]])
        loss = pc.masked_loss(prediction, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(prediction.grad[0, 0].item(), 0)
        self.assertTrue(torch.equal(prediction.grad[:, 2], torch.zeros(2)))
        self.assertGreater(prediction.grad[:, 1].abs().sum().item(), 0)

    def test_wholly_unlabelled_batch_has_differentiable_zero_loss(self):
        prediction = torch.randn(3, 2, requires_grad=True)
        loss = pc.masked_loss(prediction, torch.full((3, 2), float("nan")))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_centering_removes_translation_but_preserves_size_and_axis_signs(self):
        rng = np.random.default_rng(2)
        points = rng.uniform(-.5, .5, (128, 3)) * [1.0, 1.7, .4]
        results = []
        for name, cloud in (("original", points), ("moved", points + [5, -3, 2]),
                            ("larger", points * 1.1), ("millimetres", points * 1000)):
            source = self.root / f"{name}.obj"
            source.write_text("".join(f"v {x:.15g} {y:.15g} {z:.15g}\n" for x, y, z in cloud))
            result = pc.sample_obj(("same_id", str(source), 128, 8, 4,
                                    "mm" if name == "millimetres" else "m", 42))
            self.assertEqual(result["status"], "ok", result)
            results.append(result)
        np.testing.assert_allclose(results[0]["points"], results[1]["points"], atol=1e-7)
        np.testing.assert_allclose(results[0]["points"] * 1.1, results[2]["points"], atol=1e-7)
        np.testing.assert_allclose(results[0]["points"], results[3]["points"], atol=1e-7)
        centered = points - (points.min(0) + points.max(0)) / 2
        np.testing.assert_allclose(np.sort(results[0]["points"], axis=0), np.sort(centered, axis=0), atol=1e-7)

    def test_encoder_output_is_invariant_to_vertex_permutation(self):
        torch.manual_seed(12)
        model = pc.MeasurementPointNet(2, 2, 8, 4).eval()
        # A zero-initialized residual head would make this check trivial.
        torch.nn.init.normal_(model.head[-1].weight, std=.04)
        xyz, geometry, baseline = torch.randn(2, 64, 3), torch.randn(2, 2), torch.randn(2, 2)
        first = torch.tensor(np.stack([pc.farthest_points(cloud.numpy(), 8) for cloud in xyz]))
        second = torch.tensor(np.stack([pc.farthest_points(cloud.numpy()[anchors.numpy()], 4)
                                        for cloud, anchors in zip(xyz, first)]))
        permutation = torch.randperm(64)
        inverse = torch.argsort(permutation)
        with torch.no_grad():
            actual = model(xyz, geometry, baseline, first, second)
            permuted = model(xyz[:, permutation], geometry, baseline, inverse[first], second)
        torch.testing.assert_close(actual, permuted, rtol=1e-5, atol=1e-6)

    def test_load_baseline_rejects_source_changes_and_changed_cohort(self):
        baseline, _, _, _, _, _ = self.make_fixture()
        pc.load_baseline(baseline)
        source = self.root / "metadata.csv"
        contents = source.read_text()
        source.write_text(contents + "\n")
        with self.assertRaisesRegex(ValueError, "changed since fitting"):
            pc.load_baseline(baseline)
        source.write_text(contents)
        manifest = pd.read_csv(baseline / "splits.csv").iloc[:-1]
        manifest.to_csv(baseline / "splits.csv", index=False)
        with self.assertRaisesRegex(ValueError, "cohort differs"):
            pc.load_baseline(baseline)

    def test_manifest_rejects_same_person_in_different_partitions(self):
        baseline, _, _, _, _, _ = self.make_fixture()
        manifest = pd.read_csv(baseline / "splits.csv")
        manifest.loc[1, "split"] = "test"
        manifest.to_csv(baseline / "splits.csv", index=False)
        with self.assertRaisesRegex(ValueError, "participant.*multiple splits|more than one split"):
            pc.load_baseline(baseline)

    def test_saved_baseline_comparison_aligns_ids_and_rejects_changed_outputs(self):
        baseline, _, _, _, _, _ = self.make_fixture()
        path = baseline / "evaluation_predictions.csv"
        saved = pd.read_csv(path)
        saved.sample(frac=1, random_state=5).to_csv(path, index=False)
        # CSV row order must never determine a paired comparison.
        pc.load_baseline(baseline)
        saved.loc[0, "predicted_cm"] += 2
        saved.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "no longer reproduces saved predictions"):
            pc.load_baseline(baseline)

    def test_cache_rejects_changed_ids_and_out_of_bounds_anchors(self):
        _, cache, _, _, _, _ = self.make_fixture()
        path = cache / "point_cache.npz"
        with np.load(path, allow_pickle=False) as packed:
            original = {key: packed[key] for key in packed.files}
        changed = {key: values.copy() for key, values in original.items()}
        changed["scan_ids"][[0, 1]] = changed["scan_ids"][[1, 0]]
        np.savez_compressed(path, **changed)
        with self.assertRaisesRegex(ValueError, "scan IDs"):
            pc.read_cache(cache)
        changed = {key: values.copy() for key, values in original.items()}
        changed["a1"][0, 0] = 64
        np.savez_compressed(path, **changed)
        with self.assertRaisesRegex(ValueError, "anchor indices"):
            pc.read_cache(cache)

    def test_training_statistics_and_selection_do_not_use_test_labels(self):
        baseline, cache, X, _, split, _ = self.make_fixture()
        loaded = pc.load_baseline(baseline)
        original_y = loaded[4]
        args = self.train_args(baseline, cache, "first")
        with redirect_stdout(io.StringIO()):
            pc.train(args)
        changed_y = original_y.copy()
        changed_y.loc[split.eq("test")] += 500
        changed = (*loaded[:4], changed_y, *loaded[5:])
        # Deliberately bypass only the metadata integrity gate, to independently
        # verify train() never uses held-out labels to fit or select parameters.
        with patch.object(pc, "load_baseline", return_value=changed), redirect_stdout(io.StringIO()):
            pc.train(self.train_args(baseline, cache, "changed_test"))
        first, other = self.root / "first", self.root / "changed_test"
        a = torch.load(first / "best_model.pt", map_location="cpu", weights_only=False)
        b = torch.load(other / "best_model.pt", map_location="cpu", weights_only=False)
        for parameter in a["state_dict"]:
            torch.testing.assert_close(a["state_dict"][parameter], b["state_dict"][parameter], rtol=0, atol=0)
        self.assertEqual(a["epoch"], b["epoch"])
        self.assertEqual(a["blend_weights"], b["blend_weights"])
        pd.testing.assert_frame_equal(pd.read_csv(first / "model_selection.csv"),
                                      pd.read_csv(other / "model_selection.csv"))
        settings = a["settings"]
        np.testing.assert_allclose(settings["geometry_mean"], X.loc[split.eq("train")].mean())
        np.testing.assert_allclose(settings["target_mean_cm"], original_y.loc[split.eq("train")].mean(), rtol=2e-6)
        np.testing.assert_allclose(settings["target_std_cm"], original_y.loc[split.eq("train")].std(ddof=0), rtol=2e-6)
        self.assertFalse(pd.read_csv(first / "test_metrics.csv").mae_cm.equals(
            pd.read_csv(other / "test_metrics.csv").mae_cm))

        destination = self.root / "deployed.csv"
        with redirect_stdout(io.StringIO()):
            pc.predict_new(argparse.Namespace(model_run=str(first), cache=str(cache), output=str(destination),
                                              device="cpu", batch_size=6))
        deployed = pd.read_csv(destination, index_col="scan_id")
        reported = pd.read_csv(first / "evaluation_predictions.csv")
        for row in reported.itertuples():
            self.assertAlmostEqual(deployed.loc[row.scan_id, row.target], row.predicted_cm, places=4)

    def test_zero_residual_weight_preserves_baseline_and_paired_intervals(self):
        truth = np.array([[35.], [40.], [38.], [36.], [42.], [41.]])
        baseline = truth + np.array([[3.], [4.], [5.], [2.], [3.], [4.]])
        correction = np.arange(6, dtype=float)[:, None] - 100
        prediction = baseline + correction * 0
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        rows = pc.metrics_table(truth, prediction, ["neck"], groups, baseline, 200, 42)
        self.assertEqual(rows.iloc[0].participants, 3)
        self.assertEqual(rows.iloc[0].delta_mae_cm, 0)
        self.assertEqual(rows.iloc[0].delta_ci95_low_cm, 0)
        self.assertEqual(rows.iloc[0].delta_ci95_high_cm, 0)
        improved = pc.metrics_table(truth, baseline - 1, ["neck"], groups, baseline, 200, 42)
        self.assertAlmostEqual(improved.iloc[0].delta_mae_cm, -1)
        self.assertAlmostEqual(improved.iloc[0].delta_ci95_low_cm, -1)
        self.assertAlmostEqual(improved.iloc[0].delta_ci95_high_cm, -1)
        order = np.array([3, 0, 5, 1, 2, 4])
        shuffled = pc.metrics_table(truth[order], (baseline - 1)[order], ["neck"],
                                    groups[order], baseline[order], 200, 42)
        pd.testing.assert_frame_equal(improved, shuffled)


if __name__ == "__main__":
    unittest.main()
