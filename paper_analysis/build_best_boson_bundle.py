#!/usr/bin/env python3
"""Build the code-only, standalone Gefion selected-boson upload archive."""
import ast
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"outputs"/"gefion_boson"


def main():
    files = {path.relative_to(ROOT).as_posix(): path for path in (ROOT/"meshvae").glob("*.py")}
    names = ["train_mesh_vae.py", "diagnose_mesh_binary.py", "diagnose_mesh_regression.py",
             "check_boson_prior.py", "run_cardiovascular_target_benchmark.py"]
    names += [f"paper_analysis/{name}" for name in (
        "run_plaque_study.py", "plaque_cohorts.py", "plaque_reporting.py", "study_config.py",
        "next_steps_config.json", "next_steps_environment.sh", "selected_boson_study.py",
        "gefion_best_boson.sh", "test_selected_boson_study.py")]
    names += ["third_party/nonlinear_photonic_sampler/UPSTREAM.md"]
    names += [path.relative_to(ROOT).as_posix() for path in
              (ROOT/"third_party"/"nonlinear_photonic_sampler"/"bosonsampler").glob("*.py")]
    files.update({name: ROOT/name for name in names})
    files["README_GEFION_BOSON.md"] = ROOT/"paper_analysis"/"BEST_BOSON_GEFION.md"
    payload = {}
    for name, path in sorted(files.items()):
        content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if name.endswith(".py"):
            ast.parse(content, filename=name)
        payload[name] = content.encode("utf-8")
    manifest = {"sha256": {name: hashlib.sha256(content).hexdigest() for name, content in payload.items()},
                "description": "Code-only bundle: two selected model families, F/M, matched Gaussian-MMD/Boson-MMD, seed42",
                "excluded": "No participant data, checkpoints, installed packages, caches or old code versions"}
    payload["bundle_manifest.json"] = json.dumps(manifest, indent=2).encode("utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT/"gefion_best_boson.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for name, content in payload.items():
            bundle.writestr(name, content)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (OUTPUT/"gefion_best_boson.zip.sha256").write_text(f"{checksum}  gefion_best_boson.zip\n", encoding="ascii")
    (OUTPUT/"README_GEFION_BOSON.md").write_bytes(payload["README_GEFION_BOSON.md"])
    print(f"{archive}: {len(payload)} files, {archive.stat().st_size:,} bytes")
    print(f"SHA256 {checksum}")


if __name__ == "__main__":
    main()
