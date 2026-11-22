#!/usr/bin/env python3
"""Verify Gefion runs and preserve models/results/code without copying scan inputs.

Requires the existing paper_analysis/gefion_clean_results.py already on Gefion.
check is lightweight; submit schedules CPU-only collection. No training/deletion.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
INPUT_NAMES = {"simplified_padded_tensors.npz", "scan_geometry_features.npz"}
CACHE_NAMES = {".git", "__pycache__", ".venv", "venv", "venv_dir", "env_dir", "envs", "node_modules"}
SCAN_EXT = {".obj", ".ply", ".stl", ".h5", ".hdf5", ".npy", ".nii", ".dcm"}
META_EXT = {".tsv", ".csv", ".json", ".txt"}
GIB = 1024**3


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def csv_write(path, rows, columns=None):
    columns = columns or sorted({k for row in rows for k in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def bundle(code_root):
    path = Path(code_root)/"paper_analysis"/"gefion_clean_results.py"
    spec = importlib.util.spec_from_file_location("_offboard_existing_bundle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capture(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except OSError as exc:
        return {"command": command, "returncode": -1, "stdout": "", "stderr": str(exc)}


def options(command):
    result, key = {}, None
    for token in command[2:]:
        if token.startswith("--"):
            key, sep, val = token.partition("=")
            result[key] = [val] if sep else []
        elif key is not None:
            result[key].append(token)
    return result


def scalar(opts, name, default=""):
    return (opts.get("--"+name) or [default])[0]


def enabled(opts, name, default=False):
    return False if "--no-"+name in opts else ("--"+name in opts or default)


def job_history(study):
    path = study/"submission.json"
    if not path.is_file():
        return []
    data = read_json(path)
    return data.get("submissions", [data])


def scheduler(args):
    current = os.environ.get("SLURM_JOB_ID", "")
    query = capture(["squeue", "-h", "-u", getpass.getuser(), "--account", args.account,
                     "-o", "%i|%T|%j"])
    active = [line for line in query["stdout"].splitlines()
              if line.strip() and line.split("|", 1)[0] != current]
    ids = sorted({str(row[key]) for row in job_history(args.study_dir)
                  for key in ("training_job", "figure_job") if row.get(key)})
    accounting = capture(["sacct", "-X", "-j", ",".join(ids), "--starttime=2026-10-01",
                          "--parsable2", "--format=JobID,State,ExitCode,Elapsed"]) if ids else {
                              "returncode": 0, "stdout": "", "stderr": ""}
    return query, active, accounting


def check(args, existing=None):
    """Check saved artifacts and prediction memberships, not model performance."""
    import numpy as np
    api = existing or bundle(args.code_root)
    comparison = api.COMPARISON
    plan = read_json(args.study_dir/"plan.json")
    report = args.output_dir/"verification"
    report.mkdir(parents=True, exist_ok=True)
    cohort_frames, issues, unavailable, rows = {}, [], [], []
    cohorts = {(c["target"], c["group"]): c for c in plan["cohorts"]}
    for key, cohort in cohorts.items():
        if cohort.get("status") == "unavailable":
            unavailable.append({"target": key[0], "group": key[1], "reason": cohort.get("reason", "unknown")})
            continue
        try:
            cohort_frames[key] = comparison.verify_cohort(cohort)
            age_path = cohort.get("prediction_paths", {}).get("age_only")
            if age_path:
                if not Path(age_path).is_file():
                    raise ValueError("Missing fitted age-only baseline predictions")
                comparison.exact_predictions(age_path, cohort_frames[key]["test"], cohort["endpoint"])
        except Exception as exc:
            issues.append(f"cohort {key[0]}/{key[1]}: {exc}")
    for task in plan["tasks"]:
        record = {k: task.get(k, "") for k in ("task_index", "target", "group", "variant", "run_name")}
        try:
            state = comparison.effective_status(task)
            record["saved_status"] = state
            if state not in ("completed", "reused"):
                raise ValueError(f"Model is {state}")
            frames = cohort_frames[(task["target"], task["group"])]
            comparison.exact_predictions(task["predictions_path"], frames["test"], task["endpoint"])
            run = Path(task["run_dir"])
            for name in ("best_vae_net.pth", "cond_stats.json", "training_summary.json"):
                path = run/name
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Missing or empty {name}")
            training = read_json(run/"training_summary.json")
            if (training.get("train_samples") != len(frames["train"])
                    or training.get("validation_samples") != len(frames["validation"])
                    or set(training.get("validation_ids", [])) != set(frames["validation"].participant_id)):
                raise ValueError("Saved training/validation membership differs from the planned split")
            if state != "reused":
                marker = read_json(run/"clean_comparison_training_completed.json")
                if marker.get("fingerprint") != task["fingerprint"]:
                    raise ValueError("Training-completion fingerprint differs from the plan")
            if task.get("inference_check_required"):
                marker = read_json(Path(task["cohort_dir"])/(task["variant"]+"_inference_check.json"))
                if (marker.get("status") != "passed" or marker.get("fingerprint") != task["fingerprint"]
                        or marker.get("test_n") != len(frames["test"])
                        or not np.isfinite(marker.get("max_absolute_probability_difference", np.nan))):
                    raise ValueError("Volume-input independence check is absent or invalid")
            record.update(verification="PASS", train_n=len(frames["train"]),
                          validation_n=len(frames["validation"]), test_n=len(frames["test"]))
        except Exception as exc:
            record.update(verification="FAIL", reason=str(exc))
            issues.append(f"task {task['task_index']} {task['run_name']}: {exc}")
        rows.append(record)
    if not rows:
        issues.append("The saved plan contains no model tasks")
    query, active, accounting = scheduler(args)
    if query["returncode"]:
        issues.append("Could not verify that account jobs are stopped: "+query["stderr"].strip())
    elif active:
        issues.append(f"{len(active)} account job entries are still running or queued")
    result = {"utc_time": datetime.now(timezone.utc).isoformat(), "planned_fits": len(rows),
              "passed_fits": sum(r.get("verification") == "PASS" for r in rows),
              "ready_for_collection": not issues, "issues": issues,
              "unavailable_comparisons": unavailable, "active_jobs": active,
              "accounting_available": accounting["returncode"] == 0,
              "scope": "Verifies planned trained/reused models, saved splits, complete predictions and volume independence markers. Older experiments are inventoried separately, not certified by this check.",
              "raw_or_preprocessed_scans_included": False}
    csv_write(report/"run_verification.csv", rows)
    csv_write(report/"unavailable_comparisons.csv", unavailable, ["target", "group", "reason"])
    save_json(report/"verification.json", result)
    save_json(report/"scheduler_commands.json", {"queue": query, "accounting": accounting})
    (report/"slurm_accounting.txt").write_text(accounting["stdout"]+accounting["stderr"], encoding="utf-8")
    print(f"[CHECK] {result['passed_fits']}/{len(rows)} planned fits passed")
    for row in unavailable:
        print(f"[UNAVAILABLE] {row['target']}/{row['group']}: {row['reason']}")
    for issue in issues:
        print("[ISSUE] "+issue)
    print("[READY]" if not issues else "[NOT READY]", report.resolve())
    return api, plan, result


def discover_run_directories(args, plan):
    """Find metadata/weight-bearing runs at any depth in all main folders."""
    roots = [Path(plan["checkpoints_dir"]), args.code_root,
             args.project_root/"MQVAE", args.project_root/"MQVAE_new",
             args.project_root/"MQVAE_old", *args.extra]
    seen, candidates = set(), set()
    for source in roots:
        if not source.is_dir():
            continue
        for folder, directories, filenames in os.walk(source, followlinks=False):
            current = Path(folder)
            actual = current.resolve()
            if actual in seen or actual.is_relative_to(args.output_dir.resolve()):
                directories[:] = []
                continue
            seen.add(actual)
            directories[:] = sorted(d for d in directories if d not in CACHE_NAMES
                                    and not (current/d/"pyvenv.cfg").is_file()
                                    and not (current/d).resolve().is_relative_to(args.output_dir.resolve()))
            direct = {"training_summary.json", "cond_stats.json"} & set(filenames)
            weights = any(Path(name).suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors"} for name in filenames)
            study_metadata = (current/"study"/"manifest.json").is_file() or (current/"study"/"train_command.json").is_file()
            metric_history = current.name == "metrics" and any(name.startswith("train_metrics_") and name.endswith(".csv") for name in filenames)
            if direct or weights or study_metadata:
                candidates.add(actual)
            if metric_history:
                candidates.add(current.parent.resolve())
    return sorted(candidates)


def write_overviews(args, api, plan):
    """Readable run lineage plus exact saved partition membership; no scan inputs."""
    directory = args.output_dir/"overview"
    directory.mkdir(parents=True, exist_ok=True)
    rows, input_paths, planned_runs = [], {}, set()
    cohort_map = {(c["target"], c["group"]): c for c in plan["cohorts"]}
    membership_path = directory/"participant_splits.csv"
    with membership_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["Run", "Partition", "ParticipantID", "RunDirectory"])
        for task in plan["tasks"]:
            run = Path(task["run_dir"]); planned_runs.add(run.resolve())
            cohort = cohort_map[(task["target"], task["group"])]
            actual_command = task["train_command"]
            reused = api.COMPARISON.effective_status(task) == "reused"
            original_command = run/"study"/"train_command.json"
            if reused and original_command.is_file():
                actual_command = read_json(original_command)
            o = options(actual_command)
            training_summary = read_json(run/"training_summary.json") if (run/"training_summary.json").is_file() else {}
            row = {"Run": task["run_name"], "Source": "clean comparison plan", "Provenance": "saved exact command and split manifest",
                   "Group": task["group"], "Variant": task["variant"], "Endpoint": task["endpoint"],
                   "Inputs": ", ".join(task.get("training_inputs", [])), "DataRoot": scalar(o, "dataroot"),
                   "AuxiliaryTrainingLabels": ", ".join(task.get("training_labels", [])),
                   "AuxiliaryVolumeLossWeight": task.get("auxiliary_loss_weight", ""),
                   "BestEpoch": training_summary.get("best_epoch", ""),
                   "Seed": scalar(o, "seed"), "SplitSeed": scalar(o, "split_seed"),
                   "Epochs": task.get("configured_epochs", ""), "Prior": scalar(o, "latent_prior", "gaussian"),
                   "PriorMatch": scalar(o, "prior_match_mode", "auto"), "RunDirectory": str(run),
                   "Status": api.COMPARISON.effective_status(task)}
            for split in ("train", "validation", "test"):
                path = Path(cohort["splits_dir"])/(split+".tsv")
                actual_path = (Path(scalar(o, "cond_tsv")) if split == "train" else
                               Path(scalar(o, "val_cond_tsv")) if split == "validation" else
                               run/"study"/"test.tsv" if reused else path)
                row[split.title()+"TSV"] = str(actual_path)
                row["Audit"+split.title()+"TSV"] = str(path)
                row[split.title()+"N"] = cohort.get("counts", {}).get(split, "")
                row[split.title()+"SHA256"] = hashlib.sha256(actual_path.read_bytes()).hexdigest() if actual_path.is_file() else ""
                for participant in cohort.get("members", {}).get(split, []):
                    writer.writerow([task["run_name"], split, participant, str(run)])
            rows.append(row)
            if row["DataRoot"]:
                input_paths[row["DataRoot"]] = True
        for cohort in plan["cohorts"]:
            path = cohort.get("prediction_paths", {}).get("age_only")
            if not path or not Path(path).is_file():
                continue
            name = "AGE_ONLY::"+cohort["target"]+"::"+cohort["group"]
            row = {"Run": name, "Source": "saved age-only baseline", "Provenance": "training-fitted age-only logistic model",
                   "Group": cohort["group"], "Variant": "age_only", "Endpoint": cohort["endpoint"], "Inputs": "age",
                   "Status": "saved", "RunDirectory": str(Path(path).parents[1])}
            for split in ("train", "validation", "test"):
                row[split.title()+"TSV"] = str(Path(cohort["splits_dir"])/(split+".tsv"))
                row[split.title()+"N"] = cohort.get("counts", {}).get(split, "")
                for participant in cohort.get("members", {}).get(split, []):
                    writer.writerow([name, split, participant, str(Path(path).parents[1])])
            rows.append(row)
        seen = set(planned_runs)
        for run in discover_run_directories(args, plan):
            if not run.is_dir() or run.resolve() in seen:
                continue
            seen.add(run.resolve())
            summary = run/"training_summary.json"; manifest = run/"study"/"manifest.json"
            command = run/"study"/"train_command.json"; stats = run/"cond_stats.json"
            row = {"Run": run.name, "Source": "older checkpoint directory", "RunDirectory": str(run),
                   "Provenance": "partial; unrecorded values remain blank"}
            try:
                m = read_json(manifest) if manifest.is_file() else {}
                s = read_json(summary) if summary.is_file() else {}
                c = read_json(stats) if stats.is_file() else {}
                o = options(read_json(command)) if command.is_file() else {}
                row.update(Group=m.get("gender", c.get("only_gender", "")),
                           Endpoint=scalar(o, "aux_binary_col", scalar(o, "aux_regression_col", c.get("aux_binary_col", c.get("aux_regression_col", "")))),
                           DataRoot=scalar(o, "dataroot"), Seed=scalar(o, "seed", m.get("config", {}).get("seed", "")),
                           Prior=scalar(o, "latent_prior", s.get("latent_prior", "")),
                           PriorMatch=scalar(o, "prior_match_mode", s.get("prior_match_mode", "")),
                           TrainN=s.get("train_samples", ""), ValidationN=s.get("validation_samples", ""))
                if o:
                    row["Epochs"] = int(scalar(o, "niter", 0))+int(scalar(o, "niter_decay", 0))
                    row["TrainTSV"] = scalar(o, "cond_tsv"); row["ValidationTSV"] = scalar(o, "val_cond_tsv")
                    row["Provenance"] = "saved command; partition membership recorded where available"
                status = run/"study"/"status.json"
                row["Status"] = (read_json(status).get("status", "") if status.is_file() else
                                 "training summary saved; final pipeline status unverified" if summary.is_file() else
                                 "checkpoint/history found; training outcome unverified")
                for split, members in m.get("members", {}).items():
                    row[split.title()+"N"] = len(members)
                    row[split.title()+"TSV"] = str(run/"study"/(split+".tsv"))
                    for participant in members:
                        writer.writerow([run.name, split, participant, str(run)])
                if not m.get("members"):
                    for participant in s.get("validation_ids", []):
                        writer.writerow([run.name, "validation", participant, str(run)])
                if row["DataRoot"]:
                    input_paths[row["DataRoot"]] = True
            except Exception as exc:
                row["InventoryNote"] = str(exc)
            rows.append(row)
    csv_write(directory/"training_data_and_splits.csv", rows)
    inputs = []
    for root in sorted(input_paths):
        for name in sorted(INPUT_NAMES):
            path = Path(root)/name
            row = {"File": str(path), "Copied": False, "FingerprintAtTraining": "not recorded by this offboarding tool"}
            if path.is_file():
                stat = path.stat(); row.update(Bytes=stat.st_size, CurrentModifiedUTC=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())
            else:
                row["Note"] = "path currently missing"
            inputs.append(row)
    csv_write(directory/"input_data_references.csv", inputs, ["File", "Copied", "FingerprintAtTraining", "Bytes", "CurrentModifiedUTC", "Note"])
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for package in ("numpy", "pandas", "torch", "matplotlib", "scipy", "scikit-learn", "open3d", "trimesh"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed in collection environment"
    save_json(directory/"collection_environment.json", {
        "scope": "Environment at collection time, not reconstructed training-time versions",
        "python": sys.version, "executable": sys.executable,
        "loaded_modules": os.environ.get("LOADEDMODULES", ""), "packages": packages,
        "source_tree": str(args.code_root)})
    print("[OVERVIEW]", directory.resolve())


def inventory(args, plan):
    """Preserve checkpoints, predictions, metadata, code and logs; exclude scans."""
    root = args.project_root.resolve(); output = args.output_dir.resolve()
    selected, missing = {}, []
    def add(path):
        path = Path(path)
        if any(part in CACHE_NAMES for part in path.parts):
            return
        if path.is_dir():
            if (path/"pyvenv.cfg").is_file():
                return
            for folder, dirs, files in os.walk(path, followlinks=False):
                dirs[:] = sorted(d for d in dirs if d not in CACHE_NAMES
                    and not (Path(folder)/d/"pyvenv.cfg").is_file())
                for name in sorted(files):
                    add(Path(folder)/name)
            return
        if not path.is_file():
            missing.append(str(path)); return
        actual = path.resolve()
        try:
            relative = actual.relative_to(root)
        except ValueError:
            raise ValueError(f"Archive source outside project: {path}")
        if actual.is_relative_to(output) or any(x in CACHE_NAMES for x in relative.parts):
            return
        result_file = any(part.lower().startswith("checkpoint") for part in relative.parts) or "clean_comparison" in relative.parts
        if path.name in INPUT_NAMES:
            return
        if relative.parts[0] == "datasets" and path.suffix.lower() not in META_EXT:
            return
        if not result_file and path.suffix.lower() in SCAN_EXT | {".npz"}:
            return
        if not result_file and path.suffix.lower() in {".zip", ".7z"} and not path.name.startswith("clean_figures"):
            return
        if actual not in selected:
            stat = path.stat()
            selected[actual] = {"path": path, "member": relative.as_posix(),
                                "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    for candidate in (Path(plan["checkpoints_dir"]), args.code_root,
                      root/"MQVAE", root/"MQVAE_old", root/"MQVAE_new"/"checkpoints",
                      root/"datasets"):
        if candidate.exists():
            add(candidate)
        else:
            missing.append(str(candidate))
    metadata = root/"xfer"/"sftp"/"inbox"/"REACT-Q"
    if metadata.is_dir():
        for pattern in ("*.tsv", "*.csv"):
            for path in sorted(metadata.glob(pattern)):
                add(path)
    for pattern in ("*.out", "*.err", "*.log"):
        for path in sorted((root/"results").glob(pattern)):
            add(path)
    for path in args.extra:
        add(path)
    return sorted(selected.values(), key=lambda x: x["member"]), missing


class DigestReader:
    def __init__(self, handle):
        self.handle, self.hash = handle, hashlib.sha256()
    def read(self, n=-1):
        value = self.handle.read(n); self.hash.update(value); return value


class DigestWriter:
    def __init__(self, handle):
        self.handle, self.hash = handle, hashlib.sha256()
    def write(self, data):
        n = self.handle.write(data); self.hash.update(data[:n]); return n
    def flush(self):
        self.handle.flush()


def pack(files, output, chunk_bytes):
    """Streaming tar parts; each source and each whole tar receive SHA256."""
    parts, manifest = [], []
    handle = stream = archive = None
    payload = 0
    def close_part():
        nonlocal handle, stream, archive
        if archive is not None:
            archive.close(); stream.flush(); handle.close()
            parts.append((stream.hash.hexdigest(), Path(handle.name).name))
            archive = stream = handle = None
    try:
        for row in files:
            if archive is None or (payload and payload+row["bytes"] > chunk_bytes):
                close_part(); payload = 0
                path = output/f"results_{len(parts)+1:03d}.tar"
                handle = path.open("xb"); stream = DigestWriter(handle)
                archive = tarfile.open(fileobj=stream, mode="w|", dereference=True)
            source = row["path"]
            before = source.stat()
            if (before.st_size, before.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
                raise RuntimeError(f"Source changed since inventory: {source}")
            with source.open("rb") as raw:
                reader = DigestReader(raw)
                info = archive.gettarinfo(str(source), arcname=row["member"])
                archive.addfile(info, reader)
            after = source.stat()
            if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                raise RuntimeError(f"Source changed while archiving: {source}")
            manifest.append({"Archive": Path(handle.name).name, "Member": row["member"],
                             "Bytes": row["bytes"], "SHA256": reader.hash.hexdigest()})
            payload += row["bytes"]
        close_part()
    finally:
        if archive is not None:
            archive.close()
        if handle is not None:
            handle.close()
    csv_write(output/"file_manifest.csv", manifest, ["Archive", "Member", "Bytes", "SHA256"])
    return parts


def collect(args):
    if (args.output_dir/"COMPLETE.json").exists() or list(args.output_dir.glob("results_*.tar")):
        raise ValueError("This destination already has an archive. Choose a fresh --output-dir; existing results are not overwritten.")
    api, plan, audit = check(args)
    if not audit["ready_for_collection"] and not args.allow_incomplete:
        raise ValueError("Collection stopped because verification has unresolved issues. Read verification/run_verification.csv first.")
    try:
        api.main(["export", "--study-dir", str(args.study_dir), "--checkpoints-dir", plan["checkpoints_dir"]])
    except (Exception, SystemExit) as exc:
        audit["figure_export_warning"] = str(exc)
        print("[WARNING] Figures could not all be regenerated; original results will still be preserved:", exc)
    write_overviews(args, api, plan)
    files, missing = inventory(args, plan)
    total = sum(row["bytes"] for row in files)
    free = shutil.disk_usage(args.output_dir).free
    required = total+max(GIB, total//20)+len(files)*2048
    save_json(args.output_dir/"collection_inventory.json", {
        "files": len(files), "source_bytes": total, "free_bytes_before": free,
        "optional_paths_absent": missing, "raw_and_preprocessed_scan_inputs": "excluded",
        "archive_rule": "All checkpoint files, results, splits, code and logs under the selected project paths. Dataset binaries and unrelated input transfer archives are excluded.",
        "source_directories": [plan["checkpoints_dir"], str(args.code_root), str(args.project_root/"MQVAE"), str(args.project_root/"MQVAE_old")],
        "active_training_required_to_stop": True})
    print(f"[SIZE] {len(files)} files; {total/GIB:.2f} GiB to preserve; {free/GIB:.2f} GiB free")
    if free < required:
        raise ValueError(f"Insufficient staging space: need approximately {required/GIB:.2f} GiB free. No tar parts created.")
    parts = pack(files, args.output_dir, int(args.chunk_gib*GIB))
    # Reports/overview remain directly readable outside tar parts.
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.suffix in {".csv", ".json", ".txt"} and path.name not in {"SHA256SUMS.txt", "COMPLETE.json"}:
            parts.append((hashlib.sha256(path.read_bytes()).hexdigest(), path.relative_to(args.output_dir).as_posix()))
    (args.output_dir/"SHA256SUMS.txt").write_text("".join(f"{sha}  {name}\n" for sha, name in parts), encoding="utf-8")
    save_json(args.output_dir/"COMPLETE.json", {
        "status": "archive_complete", "utc_time": datetime.now(timezone.utc).isoformat(),
        "verified_planned_fits": audit["passed_fits"], "planned_fits": audit["planned_fits"],
        "unavailable_comparisons": audit["unavailable_comparisons"],
        "verification_issues": audit["issues"], "figure_export_warning": audit.get("figure_export_warning"),
        "archive_parts": sum(name.endswith(".tar") for _, name in parts), "copied_scan_inputs": False,
        "next_step": "Transfer the whole directory, verify SHA256SUMS.txt on the destination, and retain every archive part. No source data has been deleted."})
    print("[COMPLETE]", args.output_dir.resolve())
    print("[VERIFY AFTER DOWNLOAD] sha256sum -c SHA256SUMS.txt")


def submit(args):
    if (args.output_dir/"COMPLETE.json").exists() or list(args.output_dir.glob("results_*.tar")):
        raise ValueError("Archives already exist in this output folder; use verify, or choose a fresh --output-dir.")
    _, _, report = check(args)
    if not report["ready_for_collection"] and not args.allow_incomplete:
        raise ValueError("Some planned fits or queue checks failed. Review verification before scheduling collection.")
    script = args.output_dir/"collect_offboard.sh"
    argv = [str(Path(__file__).resolve()), "collect", "--project-root", str(args.project_root),
            "--code-root", str(args.code_root), "--study-dir", str(args.study_dir),
            "--output-dir", str(args.output_dir), "--chunk-gib", str(args.chunk_gib), "--account", args.account]
    for extra in args.extra:
        argv += ["--extra", str(extra)]
    if args.allow_incomplete:
        argv += ["--allow-incomplete"]
    script.write_text("#!/bin/bash -l\nset -euo pipefail\n"+
        "cd "+shlex.quote(str(args.code_root))+"\n"+
        "source paper_analysis/next_steps_environment.sh gefion\n"+
        'exec "$STUDY_PYTHON" '+shlex.join(argv)+"\n", encoding="utf-8", newline="\n")
    command = ["sbatch", "--parsable", "--account="+args.account, "--partition=defq",
               "--nodes=1", "--ntasks=1", "--cpus-per-task=2", "--mem=16G", "--time=12:00:00",
               "--job-name=meshvae_offboard", "--output="+str(args.output_dir/"collection_%j.out"),
               "--error="+str(args.output_dir/"collection_%j.err"), str(script)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("Unexpected Slurm response: "+result.stdout)
    save_json(args.output_dir/"collection_job.json", {"job_id": job, "output_dir": str(args.output_dir)})
    print("[SUBMITTED] CPU offboarding job", job)
    print("[FOLDER]", args.output_dir.resolve())


def verify_download(args):
    """Read-only integrity verification; works without MeshVAE, NumPy or GPU."""
    root = args.output_dir.resolve()
    complete = read_json(root/"COMPLETE.json")
    if complete.get("status") != "archive_complete":
        raise ValueError("Archive completion marker is absent or invalid")
    checked = 0
    for line in (root/"SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, filename = line.split("  ", 1)
        path = (root/filename).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or unsafe archive member: "+filename)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(2*1024*1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("Checksum mismatch: "+filename)
        checked += 1
    if not checked:
        raise ValueError("The checksum list is empty")
    print(f"[VERIFIED] {checked} downloaded archive parts/reports match their source checksums.")
    if complete.get("verification_issues") or complete.get("figure_export_warning"):
        print("[NOTE] File integrity passed; scientific-run or figure-export issues remain documented in COMPLETE.json.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "submit", "collect", "verify"])
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--project-root", type=Path, default=ROOT.parent)
    parser.add_argument("--study-dir", type=Path, default=ROOT/"clean_comparison")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chunk-gib", type=float, default=8)
    parser.add_argument("--account", default="cu_0080")
    parser.add_argument("--allow-incomplete", action="store_true", help="Preserve incomplete results, explicitly retaining audit failures.")
    parser.add_argument("--extra", action="append", type=Path, default=[], help="Additional result file/directory inside this project to preserve.")
    args = parser.parse_args(argv)
    for field in ("code_root", "project_root", "study_dir"):
        setattr(args, field, getattr(args, field).resolve())
    args.output_dir = (args.output_dir or args.project_root/"MQVAE_offboard").resolve()
    if args.chunk_gib <= 0 or not args.code_root.is_relative_to(args.project_root) or args.output_dir == args.project_root:
        parser.error("Positive chunk size and a project-local code folder / separate output folder are required")
    if any(args.output_dir == Path(p).resolve() for p in (args.code_root, args.study_dir, args.project_root/"results"/"checkpoints")):
        parser.error("Output folder must be separate from source directories")
    if args.action != "verify":
        args.output_dir.mkdir(parents=True, exist_ok=True)
    {"check": check, "submit": submit, "collect": collect, "verify": verify_download}[args.action](args)


if __name__ == "__main__":
    try:
        main()
    except (Exception, SystemExit) as exc:
        if isinstance(exc, SystemExit):
            raise
        raise SystemExit("[ERROR] "+str(exc)) from exc
