from pathlib import Path
import hashlib

root = Path(__file__).resolve().parents[1]
source = (root / "train_body_measurements_expanded.py").read_text(encoding="utf-8").replace("\r\n", "\n")
source = source.rstrip() + "\n"
digest = hashlib.sha256(source.encode()).hexdigest()
delimiter = "EXPANDED_MEASUREMENTS_SOURCE_20261027"
assert delimiter not in source
header = f'''#!/usr/bin/env bash
(
set -euo pipefail
cd /data/projects/3D_scans/ML/MQVAE_new
PY=/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python

cat > .train_body_measurements_expanded.py.incoming <<'{delimiter}'
'''
trailer = f'''{delimiter}

"$PY" - <<'EXPANDED_MEASUREMENTS_VERIFY_20261027'
from pathlib import Path
from datetime import datetime
import hashlib
import os
import shutil

incoming = Path('.train_body_measurements_expanded.py.incoming')
if hashlib.sha256(incoming.read_bytes()).hexdigest() != '{digest}':
    raise SystemExit('The pasted code is incomplete or changed. Paste the entire setup file again.')
compile(incoming.read_text(encoding='utf-8'), 'train_body_measurements_expanded.py', 'exec')
destination = Path('train_body_measurements_expanded.py')
if destination.exists():
    shutil.copy2(destination, str(destination) + '.backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
os.replace(incoming, destination)
print('[INSTALLED]', destination.resolve())
EXPANDED_MEASUREMENTS_VERIFY_20261027

"$PY" train_body_measurements_expanded.py --help
)
'''
path = root / "computerome_train_measurement_expanded_setup.sh"
path.write_text(header + source + trailer, encoding="utf-8", newline="\n")
print(path)
print("Source SHA256:", digest)
print("Setup lines:", len((header + source + trailer).splitlines()))
