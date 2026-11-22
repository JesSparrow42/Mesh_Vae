"""Build pasteable measurement installers with their organized Python imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build_measurement_installer(root: Path, *, workflow: str) -> Path:
    """Package a measurement CLI and its project-local dependencies."""
    entries = {
        "baseline": ("scripts.measurements.run_body_measurements", "measurement_terminal_setup.sh"),
        "pointcloud": ("scripts.measurements.run_body_measurements_pointcloud", "computerome_measurement_pointcloud_setup.sh"),
        "expanded": ("scripts.training.train_body_measurements_expanded", "computerome_train_measurement_expanded_setup.sh"),
    }
    module, filename = entries[workflow]
    names = [
        "scripts/__init__.py", "scripts/_paths.py",
        "scripts/measurements/__init__.py", "scripts/preprocessing/__init__.py",
        "scripts/measurements/run_body_measurements.py",
        "scripts/preprocessing/build_scan_geometry_features.py",
    ]
    names += [path.relative_to(root).as_posix()
              for pattern in ("*.py", "*.json", "requirements.txt")
              for path in sorted((root / "body_measurements").glob(pattern))]
    if workflow != "baseline":
        names += ["scripts/measurements/run_body_measurements_pointcloud.py"]
    if workflow == "expanded":
        names += ["scripts/training/__init__.py", "scripts/training/train_body_measurements_expanded.py"]
    payload = {}
    for name in names:
        content = (root / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        if name.endswith(".py"):
            compile(content, name, "exec")
        payload[name] = {"content": content,
                         "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    delimiter = "MESHVAE_MEASUREMENT_INSTALLER_SOURCE"
    assert delimiter not in encoded
    header = '''#!/usr/bin/env bash
(
set -euo pipefail
TASK_CODE_DIR="${BODY_MEASUREMENTS_CODE_DIR:-/data/projects/3D_scans/ML/MQVAE_new}"
PY="${BODY_MEASUREMENTS_PYTHON:-/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python}"
cd "$TASK_CODE_DIR"
'''
    if workflow == "baseline":
        header += '''META="${BODY_MEASUREMENTS_METADATA:-/data/projects/3D_scans/bodyscan_variables_for_validation_n2894.csv}"
"$PY" -c 'import numpy, pandas, sklearn, joblib; from meshvae.conditioning import clean_stregkode'
test -f "$META" || { echo "[ERROR] Metadata file missing: $META"; exit 1; }
'''
    header += f'\n"$PY" - <<\'{delimiter}\'\n'
    header += '''from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

'''
    verify_and_install = '''
root = Path.cwd().resolve()
for name, item in payload.items():
    destination = root / name
    if not destination.resolve().is_relative_to(root):
        raise SystemExit(f"Unsafe installer path: {name}")
    if hashlib.sha256(item["content"].encode("utf-8")).hexdigest() != item["sha256"]:
        raise SystemExit(f"Incomplete installer copy: {name}")
    if name.endswith(".py"):
        compile(item["content"], name, "exec")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
for name, item in payload.items():
    destination = root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.copy2(destination, destination.with_name(destination.name + ".backup_" + stamp))
    temporary = destination.with_name(destination.name + ".incoming")
    temporary.write_bytes(item["content"].encode("utf-8"))
    temporary.replace(destination)
    print("[INSTALLED]", name)
'''
    content = (header + "payload = json.loads(" + repr(encoded) + ")\n" +
               verify_and_install + delimiter + "\n\n" +
               f'"$PY" -m {module} --help\n')
    if workflow == "baseline":
        content += '''mkdir -p measurement_runs
AUDIT_DIR=$(mktemp -d "$PWD/measurement_runs/n2894_metadata_audit_XXXXXXXX")
"$PY" -m scripts.measurements.run_body_measurements audit \\
    --metadata "$META" \\
    --id-col participant_id \\
    --config body_measurements/targets.validation_n1715.json \\
    --out-dir "$AUDIT_DIR"
cat "$AUDIT_DIR/audit.json"
'''
    content += ")\n"
    destination = root / "scripts" / "computerome" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8", newline="\n")
    print(f"{destination}: {len(payload)} source/configuration files")
    return destination
