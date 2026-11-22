from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from meshvae.options import TestOptions, TrainOptions
from paper_analysis.export_model_reconstructions import Candidate, select_representatives


class CheckpointWorkflowTests(unittest.TestCase):
    def test_cpu_request_on_cuda_capable_host(self) -> None:
        import torch
        from meshvae.model import MeshVAEModel

        with tempfile.TemporaryDirectory() as temporary:
            for options in (TestOptions, TrainOptions):
                with self.subTest(options=options.__name__), patch("sys.argv", [
                    "test", "--name", "cpu_test", "--checkpoints_dir", temporary,
                    "--gpu_ids", "-1",
                ]):
                    opt = options().parse()
                self.assertEqual(opt.gpu_ids, [])
                opt.is_train = False
                with patch("torch.cuda.is_available", return_value=True), patch(
                    "meshvae.model.MeshVAE", return_value=torch.nn.Linear(3, 3)
                ):
                    model = MeshVAEModel(opt)
                self.assertEqual(model.device.type, "cpu")
                self.assertEqual(next(model.net.parameters()).device.type, "cpu")
                self.assertNotIsInstance(model.net, torch.nn.DataParallel)

    def test_gpu_ids_preserve_requested_devices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for value, expected in (("0", [0]), ("0,1", [0, 1]), ("-1,0,1", [0, 1])):
                with self.subTest(gpu_ids=value), patch("sys.argv", [
                    "test", "--name", "gpu_test", "--checkpoints_dir", temporary,
                    "--gpu_ids=" + value,
                ]):
                    opt = TestOptions().parse()
                self.assertEqual(opt.gpu_ids, expected)

    def test_scan_only_checkpoint_restores_architecture_and_aux_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_root = Path(temporary)
            run_dir = checkpoint_root / "scan_only"
            run_dir.mkdir()
            (run_dir / "cond_stats.json").write_text(
                json.dumps(
                    {
                        "use_cond": False,
                        "cond_dim": 6,
                        "ninput_edges": 25000,
                        "input_nc": 23,
                        "ncf": [64, 128, 256, 256],
                        "latent_dim": 64,
                        "use_preprocessed": True,
                        "use_pointcloud": True,
                        "use_point_labels": True,
                        "point_label_regions": 6,
                        "point_label_pooling": True,
                        "use_scan_geometry_features": True,
                        "scan_geometry_feature_dim": 14,
                        "aux_predict_binary": True,
                        "aux_binary_col": "benchmark_anyplaq",
                    }
                ),
                encoding="utf-8",
            )
            opt = SimpleNamespace(
                phase="test",
                checkpoints_dir=str(checkpoint_root),
                name="scan_only",
                use_cond=False,
                use_cond_prior=False,
            )
            TestOptions()._load_cond_stats_for_eval(opt)
            self.assertFalse(opt.use_cond)
            self.assertEqual(opt.cond_dim, 0)
            self.assertEqual(opt.ninput_edges, 25000)
            self.assertEqual(opt.input_nc, 23)
            self.assertTrue(opt.use_preprocessed)
            self.assertTrue(opt.use_pointcloud)
            self.assertTrue(opt.aux_predict_binary)
            self.assertEqual(opt.aux_binary_col, "benchmark_anyplaq")

    def test_representatives_are_closest_to_each_class_median(self) -> None:
        candidates = [
            Candidate(0, "N1", 0, 0.1, 1.0, "n1"),
            Candidate(1, "N2", 0, 0.2, 2.0, "n2"),
            Candidate(2, "N3", 0, 0.3, 9.0, "n3"),
            Candidate(3, "P1", 1, 0.7, 3.0, "p1"),
            Candidate(4, "P2", 1, 0.8, 4.0, "p2"),
            Candidate(5, "P3", 1, 0.9, 20.0, "p3"),
        ]
        selected, method = select_representatives(candidates, examples_per_class=1)
        self.assertEqual([candidate.participant_id for candidate in selected], ["N2", "P2"])
        self.assertEqual(method, "closest_to_class_median_chamfer")


if __name__ == "__main__":
    unittest.main()
