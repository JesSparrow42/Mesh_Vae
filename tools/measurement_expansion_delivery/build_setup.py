"""Build the standalone Computerome expanded measurement installer."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools._terminal_installer import build_measurement_installer

if __name__ == "__main__":
    build_measurement_installer(ROOT, workflow='expanded')
