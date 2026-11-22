"""Resolve study outputs relative to this code checkout, not the caller's cwd."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("next_steps_config.json")


def load_config(path=DEFAULT_CONFIG, platform="computerome", code_root=ROOT):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    settings = config["platforms"][platform]
    checkpoints = Path(settings["checkpoints"]).expanduser()
    if not checkpoints.is_absolute():
        checkpoints = Path(code_root) / checkpoints
    settings["checkpoints"] = str(checkpoints.resolve())
    return config
