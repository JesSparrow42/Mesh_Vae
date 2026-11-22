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
packed = base64.b64decode("""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
Ka5ajABXeRMlwWKcCipza0Syzvb/B8ww9fA=""".replace("\n", ""), validate=True)
assert hashlib.sha256(packed).hexdigest() == "0f295efc656d1add7e9392e41e6a3379b43fe402d7143cf247d500a7ce33398c", "Incomplete copy: checksum mismatch"
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
