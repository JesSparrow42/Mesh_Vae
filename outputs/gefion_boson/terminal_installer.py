import argparse, ast, base64, hashlib, json, re, shutil, subprocess, sys, time, zlib
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--submit", action="store_true")
parser.add_argument("--checkpoints-dir", type=Path)
parser.add_argument("--bash", default="bash")
args = parser.parse_args()
root = Path.cwd().resolve()
folder = root/"paper_analysis"
assert (folder/"selected_boson_study.py").is_file(), "Run this from MQVAE_new"
if args.submit:
    active = subprocess.check_output(["squeue", "--me", "-h", "-n", "best_boson"], text=True)
    assert not active.strip(), "A best_boson array is already queued/running; no duplicate submitted"
packed = base64.b64decode("""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""".replace("\n", ""), validate=True)
assert hashlib.sha256(packed).hexdigest() == "5c4cd73bcabc5071533ebb5e5585be4c7321b2fecbb246c8c5bce80855546276", "Incomplete copy: checksum mismatch"
update = json.loads(zlib.decompress(packed))

selected_path = folder/"selected_boson_study.py"
selected = selected_path.read_text(encoding="utf-8")
nodes = ast.parse(selected).body
existing = next((n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "check_runtime_compatibility"), None)
lines = selected.splitlines(keepends=True)
if existing:
    lines[existing.lineno-1:existing.end_lineno] = [update["guard"].rstrip()+"\n"]
    selected = "".join(lines)
else:
    assert "def main():" in selected, "Unexpected selected_boson_study.py; stopped without queuing"
    selected = selected.replace("def main():", update["guard"].rstrip()+"\n\ndef main():", 1)
if not any(isinstance(n, ast.Import) and any(a.name == "ast" for a in n.names) for n in ast.parse(selected).body):
    assert "import argparse\n" in selected, "Unexpected imports; stopped without queuing"
    selected = selected.replace("import argparse\n", "import argparse\nimport ast\n", 1)
if "        check_runtime_compatibility()\n" not in selected:
    marker = '    if args.action == "check":\n'
    assert selected.count(marker) == 1, "Unexpected check action; stopped without queuing"
    selected = selected.replace(marker, marker+"        check_runtime_compatibility()\n", 1)

job_path = folder/"gefion_best_boson.sh"
job = job_path.read_text(encoding="utf-8")
gpu_check = '"$STUDY_PYTHON" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"'
job = re.sub(r"(?m)^[ \t]*study_check_gpu_environment[ \t]*$", lambda match: gpu_check, job)
assert "torch.cuda.is_available" in job, "GPU-check replacement failed; stopped without queuing"
assert "#SBATCH --gpus=1" in job and "#SBATCH --array=0-7%2" in job, "Unexpected job resources; stopped without queuing"
changes = {folder/"run_plaque_study.py": update["runner"], selected_path: selected, job_path: job}
for path, content in changes.items():
    if path.suffix == ".py":
        compile(content, str(path), "exec")
stamp = str(time.time_ns())
for path, content in changes.items():
    if path.read_text(encoding="utf-8") == content:
        continue
    backup = path.with_name(path.name+".before_terminal_fix_"+stamp)
    shutil.copy2(path, backup)
    temporary = path.with_name(path.name+".tmp_"+stamp)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    temporary.replace(path)
    print("[UPDATED]", path.name, "backup:", backup.name, flush=True)
subprocess.run([args.bash, "-n", str(job_path)], check=True)

sys.path.insert(0, str(root))
from paper_analysis import selected_boson_study as study
from meshvae.options import TrainOptions
checkpoints = args.checkpoints_dir or study.DEFAULT_CHECKPOINTS
study.check_runtime_compatibility()
for selection in study.SELECTED:
    for gender in ("F", "M"):
        study.check_reference(checkpoints, selection, gender, study.REFERENCE_TAG)
print("[PASS] All four original models and saved splits checked", flush=True)
plan = study.task_plan()
for selection, gender, arm in plan:
    run = checkpoints/study.run_name(selection, gender, study.RUN_TAG, arm)
    assert not any((run/name).exists() for name in ("best_vae_net.pth", "latest_vae_net.pth")), "Existing new-model weights found; stopped to preserve them: "+str(run)
for index, (selection, gender, arm) in enumerate(plan):
    cmd = [sys.executable, str(selected_path), "run", "--task-index", str(index),
           "--checkpoints-dir", str(checkpoints), "--dry-run"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout, flush=True)
        print(result.stderr, file=sys.stderr, flush=True)
        raise SystemExit("Check failed; no jobs submitted")
    derived = checkpoints/study.run_name(selection, gender, study.RUN_TAG, arm)/"study"
    command = study.read_json(derived/"train_command.json")
    TrainOptions().parser.parse_args(command[2:])
    original = checkpoints/study.run_name(selection, gender, study.REFERENCE_TAG)/"study"
    study.same_training_settings(study.read_json(original/"train_command.json"), command)
    print(f"[OK] Task {index}: {selection.experiment} {gender} {arm}", flush=True)
print("[PASS] All eight command checks passed", flush=True)
if args.submit:
    subprocess.run(["sbatch", str(job_path.relative_to(root))], check=True)
else:
    print("Check-only run: no jobs submitted", flush=True)
