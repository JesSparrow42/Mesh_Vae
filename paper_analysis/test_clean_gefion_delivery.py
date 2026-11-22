"""One-file integration: supported flags and mocked scheduler only, no GPU/jobs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from paper_analysis.build_clean_gefion_delivery import build
from paper_analysis.test_clean_gefion_comparison import ComparisonTests


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("gefion_delivery_test", build())
        cls.bundle = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.bundle)

    def fixture(self):
        source = ComparisonTests(); source.setUp(); self.addCleanup(source.tearDown)
        source.fixture(group="F"); source.fixture(group="M")
        return source

    def invoke(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return self.bundle.main(list(arguments))

    def test_one_file_prepare_long_and_actual_argument_preflight(self):
        source = self.fixture()
        self.invoke("prepare", "--study-dir", str(source.output), "--checkpoints-dir", str(source.checkpoints),
                    "--targets", "main", "--include-long")
        plan = json.loads((source.output/"plan.json").read_text())
        self.assertEqual(len(plan["tasks"]), 12)
        self.assertEqual(sum(t["status"] == "reused" for t in plan["tasks"]), 2)
        self.assertEqual(len([t for t in plan["tasks"] if t.get("configured_epochs") == 2000]), 6)
        first = [t["fingerprint"] for t in plan["tasks"]]
        plan = self.bundle.LONG.append_long_tasks(plan, self.bundle.COMPARISON)
        self.assertEqual(len(plan["tasks"]), 12)
        self.assertEqual(first, [t["fingerprint"] for t in plan["tasks"]])

    def test_submit_skips_reused_schedules_missing_and_writes_valid_bash(self):
        source = self.fixture()
        arguments = ["--study-dir", str(source.output), "--checkpoints-dir", str(source.checkpoints), "--targets", "main"]
        with patch.object(self.bundle, "sbatch", side_effect=["12345", "12346"]) as scheduler:
            self.invoke("submit", *arguments)
        self.assertEqual(scheduler.call_count, 2)
        scheduled = scheduler.call_args_list[0].args[0]
        array = next(x for x in scheduled if x.startswith("--array="))
        self.assertEqual(array, "--array=0,2,4,5%2")
        self.assertFalse(any(x.startswith("--export") for x in scheduled))
        followup = scheduler.call_args_list[1].args[0]
        self.assertIn("--dependency=afterany:12345", followup)
        text = (source.output/"gefion_clean_array.sh").read_text()
        self.assertIn("gefion_clean_results.py", text)
        self.assertIn('${SLURM_ARRAY_TASK_ID:?}', text)
        self.assertNotIn("study_check_gpu_environment", text)
        import subprocess
        for path in (source.output/"gefion_clean_array.sh", source.output/"gefion_clean_figures.sh"):
            result = subprocess.run(["C:/Program Files/Git/bin/bash.exe", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_preexisting_active_array_does_not_resubmit(self):
        source = self.fixture()
        arguments = ["--study-dir", str(source.output), "--checkpoints-dir", str(source.checkpoints), "--targets", "main"]
        self.invoke("prepare", *arguments)
        (source.output/"submission.json").write_text(json.dumps({"training_job": "12345", "task_indices": ["0", "2", "4", "5"]}))
        from subprocess import CompletedProcess
        with patch.object(self.bundle.subprocess, "run", return_value=CompletedProcess([], 0, "12345_0 R\n", "")), patch.object(self.bundle, "sbatch") as scheduler:
            self.invoke("submit", *arguments)
        scheduler.assert_not_called()

    def test_add_volume_to_active_array_only_submits_six_new_tasks(self):
        source = self.fixture()
        import pandas as pd
        for group in ("F", "M"):
            run = source.checkpoints/self.bundle.COMPARISON.reference_name("peripheral_total", group, "next_v1")
            for split in self.bundle.COMPARISON.SPLITS:
                path = run/"study"/f"{split}.tsv"
                frame = pd.read_csv(path, sep="\t"); frame["volume_peripheral_total"] = [0., 0., 10., 30.]; frame.to_csv(path, sep="\t", index=False)
        arguments = ["--study-dir", str(source.output), "--checkpoints-dir", str(source.checkpoints), "--targets", "main"]
        self.invoke("prepare", *arguments)
        (source.output/"submission.json").write_text(json.dumps({"training_job": "12345", "task_indices": ["0", "2", "4", "5"]}))
        from subprocess import CompletedProcess
        with patch.object(self.bundle.subprocess, "run", return_value=CompletedProcess([], 0, "12345_0 R\n", "")), patch.object(self.bundle, "sbatch", side_effect=["12347", "12348"]) as scheduler:
            self.invoke("submit", *arguments, "--include-volume")
        self.assertIn("--array=6,7,8,9,10,11%2", scheduler.call_args_list[0].args[0])
        self.assertIn("--dependency=afterany:12345:12347", scheduler.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
