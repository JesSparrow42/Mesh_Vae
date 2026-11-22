from pathlib import Path
import hashlib

root = Path(__file__).resolve().parents[2]
source = root / 'run_body_measurements_pointcloud.py'
contents = source.read_text(encoding='utf-8').replace('\r\n', '\n')
if not contents.endswith('\n'):
    contents += '\n'
digest = hashlib.sha256(contents.encode('utf-8')).hexdigest()
delimiter = 'BODY_POINTCLOUD_SOURCE_20261014'
assert delimiter not in contents
header = '''#!/usr/bin/env bash
(
set -euo pipefail

cd /data/projects/3D_scans/ML/MQVAE_new
PY=/dpreact/home/jonsve/envs/mqvae_py312_cu121/bin/python

if [ -f run_body_measurements_pointcloud.py ]; then
  cp -- run_body_measurements_pointcloud.py "run_body_measurements_pointcloud.py.backup_$(date +%Y%m%d_%H%M%S)"
fi

cat > run_body_measurements_pointcloud.py <<'BODY_POINTCLOUD_SOURCE_20261014'
'''
trailer = '''BODY_POINTCLOUD_SOURCE_20261014

"$PY" - <<'BODY_POINTCLOUD_VERIFY_20261014'
from pathlib import Path
import hashlib
import py_compile

path = Path('run_body_measurements_pointcloud.py')
expected = '__DIGEST__'
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit('The pasted Python file is incomplete or changed. Paste the whole setup file again.')
py_compile.compile(str(path), doraise=True)
print('[INSTALLED]', path.resolve())
print('[SHA256]', actual)
BODY_POINTCLOUD_VERIFY_20261014

"$PY" run_body_measurements_pointcloud.py --help
)
'''.replace('__DIGEST__', digest)
destination = root / 'scripts' / 'computerome' / 'computerome_measurement_pointcloud_setup.sh'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(header + contents + trailer, encoding='utf-8', newline='\n')
print(destination)
print('Code SHA256:', digest)
print('Installer lines:', len((header + contents + trailer).splitlines()))
