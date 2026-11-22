"""Entry point template embedded in the standalone Gefion delivery file."""
from pathlib import Path
import argparse
import json
import re
import shlex
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def settings(args):
    return argparse.Namespace(output_dir=args.study_dir.resolve(), checkpoints_dir=args.checkpoints_dir.resolve(),
        reference_tag=args.reference_tag, run_tag=args.run_tag, targets=args.targets, group=None,
        combined_policy="dedicated", task_index=args.task_index)


def prepare_study(args):
    plan = COMPARISON.prepare(settings(args))
    if args.include_long:
        plan = LONG.append_long_tasks(plan, COMPARISON)
    if args.include_volume:
        plan = VOLUME.append_volume_tasks(plan, COMPARISON)
    return plan


def preflight(plan, active_indices=()):
    for relative in ("train_mesh_vae.py", "diagnose_mesh_binary.py", "paper_analysis/next_steps_environment.sh"):
        if not (ROOT/relative).is_file():
            raise ValueError(f"Missing existing repository file: {ROOT/relative}")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from meshvae.options import TrainOptions, TestOptions
    train_parser, test_parser = TrainOptions().parser, TestOptions().parser
    for task in plan["tasks"]:
        train_parser.parse_args(task["train_command"][2:])
        test_parser.parse_args(task["evaluate_command"][2:])
        cohort = next(row for row in plan["cohorts"] if row["target"] == task["target"] and row["group"] == task["group"])
        COMPARISON.verify_cohort(cohort)
        run = Path(task["run_dir"])
        if (COMPARISON.effective_status(task) not in ("completed", "reused") and run.is_dir()
                and str(task["task_index"]) not in set(active_indices)
                and any(run.iterdir()) and not (run/"clean_comparison_training_completed.json").is_file()):
            raise ValueError(f"Partial training exists in {run}; it will not be overwritten. Inspect it before starting another run.")
        if task["variant"].startswith("long_") and "boson" in task["variant"]:
            for relative in ("check_boson_prior.py", "third_party/nonlinear_photonic_sampler"):
                if not (ROOT/relative).exists():
                    raise ValueError(f"Missing existing Boson sampler dependency: {ROOT/relative}")
    print("[PASS] All generated train/evaluate flags are supported; saved splits and partial-run safeguards verified.")


def sbatch(arguments):
    result = subprocess.run(["sbatch", "--parsable", *arguments], check=True, capture_output=True, text=True)
    job = result.stdout.strip().split(";")[0]
    if not re.fullmatch(r"\d+", job):
        raise ValueError(f"Unexpected scheduler response: {result.stdout.strip()}")
    return job


def submit_study(args):
    plan_path = args.study_dir/"plan.json"
    previous_submission = args.study_dir/"submission.json"
    active_indices, active_jobs, history = set(), [], []
    if previous_submission.is_file():
        previous = json.loads(previous_submission.read_text())
        history = previous.get("submissions", [{key: value for key, value in previous.items() if key != "submissions"}])
        for submission in history:
            job = submission["training_job"]
            result = subprocess.run(["squeue", "-h", "-j", job], capture_output=True, text=True)
            if result.returncode and "invalid job id" not in result.stderr.lower():
                raise ValueError(f"Cannot check existing job {job}: {result.stderr.strip()}")
            if result.stdout.strip():
                active_jobs.append(job)
                active_indices.update(str(i) for i in submission.get("task_indices", []))
    if plan_path.is_file():
        plan = COMPARISON.read_json(plan_path)
        if plan["run_tag"] != args.run_tag or Path(plan["checkpoints_dir"]).resolve() != args.checkpoints_dir.resolve():
            raise ValueError("Existing plan tag/checkpoint root differs from this command")
        if args.include_long:
            plan = LONG.append_long_tasks(plan, COMPARISON)
        if args.include_volume:
            plan = VOLUME.append_volume_tasks(plan, COMPARISON)
    else:
        plan = prepare_study(args)
    if not plan["tasks"]:
        raise ValueError("No trainable cohorts were prepared. Read the [UNAVAILABLE] reasons; no jobs were submitted.")
    preflight(plan, active_indices)
    indices = [str(task["task_index"]) for task in plan["tasks"] if COMPARISON.effective_status(task) not in ("completed", "reused")
               and str(task["task_index"]) not in active_indices]
    if not indices:
        if active_jobs:
            print(f"[ALREADY QUEUED] Training jobs {', '.join(active_jobs)}. No duplicate tasks submitted.")
            return
        print("[DONE] No missing VAE runs. Generating figures from completed results.")
        make_figures(args)
        return
    logs = args.study_dir/"logs"; logs.mkdir(parents=True, exist_ok=True)
    script = args.study_dir/"gefion_clean_array.sh"
    command = shlex.join([str(Path(__file__).resolve()), "train", "--study-dir", str(args.study_dir.resolve())])
    script.write_text("#!/bin/bash -l\nset -euo pipefail\n"+
        f"cd {shlex.quote(str(ROOT))}\nsource paper_analysis/next_steps_environment.sh gefion\n"+
        '"$STUDY_PYTHON" -c "import torch,open3d,trimesh; assert torch.cuda.is_available(), \'No allocated GPU visible\'; print(torch.cuda.get_device_name(0))"\n'+
        f'exec "$STUDY_PYTHON" {command} --task-index "${{SLURM_ARRAY_TASK_ID:?}}"\n', encoding="utf-8", newline="\n")
    job = sbatch(["--account=cu_0080", "--partition=defq", "--nodes=1", "--ntasks=1", "--gpus=1", "--cpus-per-task=8",
        "--mem=128G", "--time=3-00:00:00", f"--array={','.join(indices)}%{args.parallel}", "--job-name=clean_models",
        f"--output={logs}/train_%A_%a.out", f"--error={logs}/train_%A_%a.err", str(script)])
    submission = {"training_job": job, "task_indices": indices, "figure_job": None}
    COMPARISON.write_json(previous_submission, {**submission, "submissions": [*history, submission.copy()]})
    print(f"Submitted training array {job}: {len(indices)} missing fits; at most {args.parallel} run simultaneously.")
    collector = args.study_dir/"gefion_clean_figures.sh"
    collector.write_text("#!/bin/bash -l\nset -euo pipefail\n"+
        f"cd {shlex.quote(str(ROOT))}\nsource paper_analysis/next_steps_environment.sh gefion\n"+
        f'exec "$STUDY_PYTHON" '+shlex.join([str(Path(__file__).resolve()), "export", "--study-dir", str(args.study_dir.resolve()),
             "--checkpoints-dir", str(args.checkpoints_dir.resolve()), "--bootstraps", str(args.bootstraps)])+"\n", encoding="utf-8", newline="\n")
    try:
        figure_job = sbatch(["--account=cu_0080", "--partition=defq", "--nodes=1", "--ntasks=1", "--cpus-per-task=2",
            "--mem=16G", "--time=02:00:00", f"--dependency=afterany:{':'.join([*active_jobs, job])}", "--job-name=clean_figures",
            f"--output={logs}/figures_%j.out", f"--error={logs}/figures_%j.err", str(collector)])
        submission["figure_job"] = figure_job
        COMPARISON.write_json(previous_submission, {**submission, "submissions": [*history, submission.copy()]})
        print(f"Figure/export job {figure_job} will run when the training array finishes, including after failed tasks.")
    except subprocess.CalledProcessError as exc:
        print(f"[NOTE] Training is submitted, but the automatic figure job was rejected: {exc.stderr}")
        print("Run the same Python file with action 'export' after training finishes.")


def make_figures(args, export=False):
    # Earlier and newly appended arrays may both have dependent export jobs.
    # Linux releases this process lock automatically even after a failed job.
    args.study_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.study_dir/".figure_export.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return render_figures(args, export)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def render_figures(args, export=False):
    errors = []
    output = args.study_dir/"figures"
    try:
        FIGURES.main(["--study-dir", str(args.study_dir), "--output-dir", str(output), "--bootstraps", str(args.bootstraps)])
    except Exception as exc:
        errors.append("AUC figures: "+str(exc)); print("[ERROR] "+errors[-1], file=sys.stderr)
    try:
        LOSSES.main(["--checkpoints-dir", str(args.checkpoints_dir), "--output-dir", str(output/"loss_curves"),
            "--comparison-plan", str(args.study_dir/"plan.json")])
    except Exception as exc:
        errors.append("Loss figures: "+str(exc)); print("[ERROR] "+errors[-1], file=sys.stderr)
    if export:
        target = args.study_dir/"clean_figures.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in output.rglob("*"):
                # Exclude superseded definition-check charts from earlier exports.
                if (path.is_file() and path.suffix in {".csv", ".json", ".png", ".pdf", ".svg"}
                        and "volume_presence_definition" not in path.name
                        and "plaque_training_supervision" not in path.name):
                    archive.write(path, path.relative_to(output))
        print(f"[DOWNLOAD] {target.resolve()} (plots, aggregate metrics and provenance; no participant-level predictions)")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"[FIGURES] {output.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="One-file Gefion workflow: matched AUC figures and normal/Boson loss curves. No dependency installations.")
    parser.add_argument("action", choices=["prepare", "submit", "train", "status", "figures", "losses", "export"])
    parser.add_argument("--study-dir", type=Path, default=ROOT/"clean_comparison")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("/dcai/projects/cu_0080/results/checkpoints"))
    parser.add_argument("--reference-tag", default="next_v1")
    parser.add_argument("--run-tag", default="clean_v1")
    parser.add_argument("--targets", choices=["main", "territories", "all"], default="all")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--include-long", action="store_true", help="Also prepare six new matched 2000-epoch Gaussian/KL, Gaussian/MMD and Boson/MMD fits for women/men.")
    parser.add_argument("--include-volume", action="store_true", help="Also prepare six matched classifier-control / classifier+volume-supervision fits for women, men and combined.")
    args = parser.parse_args(argv)
    if args.parallel < 1 or args.bootstraps < 100:
        parser.error("--parallel must be positive; --bootstraps must be at least 100")
    args.study_dir = args.study_dir.resolve(); args.checkpoints_dir = args.checkpoints_dir.resolve()
    if args.action == "prepare":
        plan = prepare_study(args); preflight(plan)
    elif args.action == "submit":
        submit_study(args)
    elif args.action == "train":
        plan = COMPARISON.read_json(args.study_dir/"plan.json")
        if args.task_index is None or not 0 <= args.task_index < len(plan["tasks"]):
            raise ValueError("A valid --task-index is required for a training worker")
        task = plan["tasks"][args.task_index]
        try:
            COMPARISON.train(settings(args))
            if task["target"] == "peripheral_total_volume_training":
                VOLUME.check_inference_invariance(task, COMPARISON)
        except Exception as exc:
            COMPARISON.write_json(Path(task["cohort_dir"])/f"{task['variant']}_status.json",
                {"status": "failed", "fingerprint": task["fingerprint"], "reason": str(exc)})
            raise
    elif args.action == "status":
        COMPARISON.audit(settings(args))
        path = args.study_dir/"submission.json"
        if path.is_file():
            submission = COMPARISON.read_json(path)
            jobs = [row["training_job"] for row in submission.get("submissions", [submission])]
            try:
                subprocess.run(["sacct", "-X", "-j", ",".join(jobs), "--format=JobID,State,ExitCode,Elapsed"], check=False)
            except FileNotFoundError:
                print("[NOTE] sacct is unavailable here; the table above reports saved model status.")
    elif args.action in ("figures", "export"):
        make_figures(args, export=args.action == "export")
    elif args.action == "losses":
        LOSSES.main(["--checkpoints-dir", str(args.checkpoints_dir), "--output-dir", str(args.study_dir/"figures"/"loss_curves"),
            "--comparison-plan", str(args.study_dir/"plan.json")])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        detail = (exc.stderr or str(exc)) if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise SystemExit("[ERROR] "+detail) from exc
