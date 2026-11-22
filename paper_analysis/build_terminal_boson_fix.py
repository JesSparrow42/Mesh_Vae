#!/usr/bin/env python3
"""Generate a self-contained pasteable Gefion update, without a file upload."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

ROOT = Path(__file__).resolve().parents[1]

INSTALLER = r'''import argparse, ast, base64, hashlib, json, re, shutil, subprocess, sys, time, zlib
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
packed = base64.b64decode("""__PAYLOAD__""".replace("\n", ""), validate=True)
assert hashlib.sha256(packed).hexdigest() == "__PAYLOAD_SHA__", "Incomplete copy: checksum mismatch"
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
'''


def main():
    runner = (ROOT/"paper_analysis"/"run_plaque_study.py").read_text(encoding="utf-8")
    selected = (ROOT/"paper_analysis"/"selected_boson_study.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(selected).body if isinstance(n, ast.FunctionDef)
                and n.name == "check_runtime_compatibility")
    guard = "\n".join(selected.splitlines()[node.lineno-1:node.end_lineno])+"\n"
    payload = zlib.compress(json.dumps({"runner": runner, "guard": guard}).encode("utf-8"), 9)
    encoded = "\n".join(textwrap.wrap(base64.b64encode(payload).decode("ascii"), 100))
    code = INSTALLER.replace("__PAYLOAD__", encoded).replace("__PAYLOAD_SHA__", hashlib.sha256(payload).hexdigest())
    compile(code, "terminal_installer.py", "exec")
    output = ROOT/"outputs"/"gefion_boson"
    output.mkdir(parents=True, exist_ok=True)
    (output/"terminal_installer.py").write_text(code, encoding="utf-8", newline="\n")
    commands = ("cd /dcai/projects/cu_0080/MQVAE_new &&\n"
                "source paper_analysis/next_steps_environment.sh gefion &&\n"
                '"$STUDY_PYTHON" - --submit <<\'PY\'\n'+code+"PY\n")
    (output/"terminal_commands.txt").write_text(commands, encoding="utf-8", newline="\n")
    print(f"Generated terminal_commands.txt: {len(commands):,} characters")


if __name__ == "__main__":
    main()
