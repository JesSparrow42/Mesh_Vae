#!/usr/bin/env python3
"""Small, matched Gefion comparison of the selected classifier and log-volume model."""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CHECKPOINTS = Path("/dcai/projects/cu_0080/results/checkpoints")
REFERENCE_TAG = "next_v1"
RUN_TAG = "best_boson_v1"
SEED = 42
ARMS = ("gaussian_mmd", "boson_mmd")


@dataclass(frozen=True)
class Selection:
    suite: str
    experiment: str
    kind: str


SELECTED = (Selection("weight", "binary_age", "binary"),
            Selection("plaque", "all_log", "regression"))


def task_plan():
    return [(selection, gender, arm) for selection in SELECTED
            for gender in ("F", "M") for arm in ARMS]


def run_name(selection, gender, tag, arm=None):
    suffix = f"_{arm}" if arm else ""
    return f"GEFION_NEXT_{selection.suite}_{selection.experiment}{suffix}_{gender}_{tag}_seed{SEED}"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_runtime_compatibility():
    """Read-only checks for mixed code versions; no GPU or training is needed."""
    from meshvae.options import TrainOptions

    help_result = subprocess.run([sys.executable, str(ROOT/"paper_analysis"/"run_plaque_study.py"), "--help"],
                                 capture_output=True, text=True)
    if help_result.returncode:
        raise ValueError("Cannot import the study runner. Its error was:\n"+help_result.stderr)
    flags = ("--experiment", "--seed", "--run-tag", "--prior-arm", "--reference-run",
             "--boson-params", "--boson-sampler-path")
    missing = [flag for flag in flags if flag not in help_result.stdout]
    if missing:
        raise ValueError("Outdated paper_analysis/run_plaque_study.py; replace that exact file. "
                         "Missing options: "+", ".join(missing)+". No jobs should be submitted yet.")
    for prior in ("gaussian", "boson"):
        probe = ["--name", "compatibility_check", "--latent_prior", prior, "--prior_match_mode", "mmd",
                 "--mmd_buffer_size", "64", "--mmd_prior_samples", "64", "--deterministic"]
        if prior == "boson":
            probe += ["--boson_params", "{}", "--boson_sampler_path",
                      str(ROOT/"third_party"/"nonlinear_photonic_sampler")]
        try:
            TrainOptions().parser.parse_args(probe)
        except SystemExit as exc:
            raise ValueError("Outdated meshvae/options.py: the training parser does not support "
                             "the matched MMD comparison. No jobs should be submitted yet.") from exc
    vi_path = ROOT/"meshvae"/"vi.py"
    vi_tree = ast.parse(vi_path.read_text(encoding="utf-8"))
    vi_class = next((node for node in vi_tree.body if isinstance(node, ast.ClassDef)
                     and node.name == "MeshVariationalInference"), None)
    constructor = next((node for node in (vi_class.body if vi_class else [])
                        if isinstance(node, ast.FunctionDef) and node.name == "__init__"), None)
    parameters = {arg.arg for arg in [*constructor.args.args, *constructor.args.kwonlyargs]} if constructor else set()
    if not {"prior_match_mode", "mmd_buffer_size", "mmd_prior_samples"} <= parameters:
        raise ValueError("Outdated meshvae/vi.py: the loss does not support the requested MMD settings. "
                         "No jobs should be submitted yet.")
    model_tree = ast.parse((ROOT/"meshvae"/"model.py").read_text(encoding="utf-8"))
    wired = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "MeshVariationalInference"
                and {"prior_match_mode", "mmd_buffer_size", "mmd_prior_samples"}
                <= {keyword.arg for keyword in node.keywords} for node in ast.walk(model_tree))
    if not wired:
        raise ValueError("Outdated meshvae/model.py: the requested MMD settings are not forwarded to the loss. "
                         "No jobs should be submitted yet.")
    adapter_tree = ast.parse((ROOT/"meshvae"/"boson.py").read_text(encoding="utf-8"))
    if not any(isinstance(node, ast.FunctionDef) and node.name == "_load_sparrow_sampler"
               for node in adapter_tree.body):
        raise ValueError("Outdated meshvae/boson.py: the Sparrow integration is missing. "
                         "No jobs should be submitted yet.")
    required = ["scripts/diagnostics/check_boson_prior.py", "third_party/nonlinear_photonic_sampler/bosonsampler/__init__.py",
                "third_party/nonlinear_photonic_sampler/bosonsampler/__version__.py",
                "third_party/nonlinear_photonic_sampler/bosonsampler/core.py"]
    missing_files = [name for name in required if not (ROOT/name).is_file()]
    if missing_files:
        raise ValueError("Missing sampler files: "+", ".join(missing_files)+". No jobs should be submitted yet.")
    print("[PASS] Runner CLI, both prior-option interfaces, MMD loss settings and sampler files are compatible.", flush=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_predictions(study, expected_ids, target, kind):
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(study/"heldout"/"predictions.csv", dtype={"participant_id": str})
    if frame.participant_id.duplicated().any() or set(frame.participant_id) != set(expected_ids):
        raise ValueError(f"Incomplete or duplicate held-out predictions: {study}")
    frame = frame.set_index("participant_id").loc[expected_ids].reset_index()
    values = frame[["target", "meshvae"]].to_numpy(float)
    if not np.isfinite(values).all() or not np.allclose(values[:, 0], target, rtol=1e-12, atol=1e-12):
        raise ValueError(f"Nonfinite predictions or mismatched targets: {study}")
    if kind == "binary" and (not np.isin(target, [0, 1]).all()
                            or not ((values[:, 1] >= 0) & (values[:, 1] <= 1)).all()):
        raise ValueError(f"Invalid binary targets/probabilities: {study}")
    return frame


def check_reference(checkpoints, selection, gender, reference_tag):
    """Read only: reject incomplete runs, changed memberships, leakage, and wrong model settings."""
    import numpy as np
    import pandas as pd
    from meshvae.options import TrainOptions
    from paper_analysis.run_plaque_study import experiments

    run = checkpoints/run_name(selection, gender, reference_tag)
    study = run/"study"
    required = [run/"best_vae_net.pth", *[study/name for name in (
        "manifest.json", "status.json", "train_command.json", "train.tsv", "validation.tsv",
        "test.tsv", "heldout/predictions.csv", "heldout/comparison.csv")]]
    for path in required:
        if not path.is_file():
            raise ValueError(f"Missing original result: {path}")
    if read_json(study/"status.json").get("status") != "completed":
        raise ValueError(f"Original study is not completed: {run}")
    manifest = read_json(study/"manifest.json")
    expected = next(exp for exp in experiments(selection.suite) if exp.key == selection.experiment)
    if manifest["experiment"] != asdict(expected) or manifest["gender"] != gender:
        raise ValueError(f"Original experiment/sex differs from the selected model: {run}")
    if manifest["config"]["seed"] != SEED or manifest["config"]["run_tag"] != reference_tag:
        raise ValueError(f"Original seed/tag differs from its expected name: {run}")
    cmd = read_json(study/"train_command.json")
    opt = TrainOptions().parser.parse_args(cmd[2:])
    match = "kl" if opt.prior_match_mode == "auto" else opt.prior_match_mode
    if (opt.latent_prior, match) != ("gaussian", "kl"):
        raise ValueError(f"Original result is not Gaussian-KL: {run}")
    if (opt.only_gender != gender or opt.seed != SEED or opt.feature_source != "scan"
            or not opt.use_cond or opt.cond_athero or opt.cond_plaque
            or opt.cond_weight != expected.weight):
        raise ValueError(f"Original sex, age/scan inputs, weight or disease-input settings differ: {run}")
    if selection.kind == "binary":
        valid = (opt.aux_predict_binary and not opt.aux_predict_regression
                 and opt.aux_binary_col == "presence_peripheral_total")
    else:
        valid = (opt.aux_predict_regression and not opt.aux_predict_binary
                 and opt.aux_regression_col == "volume_peripheral_total"
                 and opt.aux_regression_transform == "log1p" and opt.aux_regression_loss == "huber")
    if not valid:
        raise ValueError(f"Original task/target/loss differs from the selected model: {run}")
    frames = {}
    target = f"{'presence' if selection.kind == 'binary' else 'volume'}_peripheral_total"
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(study/f"{split}.tsv", sep="\t", dtype={"participant_id": str})
        if (frame.participant_id.duplicated().any()
                or list(frame.participant_id) != manifest["members"][split]
                or not (frame.study_gender == gender).all()
                or not np.isfinite(frame[target].to_numpy(float)).all()):
            raise ValueError(f"Original {split} membership/sex/target is invalid: {run}")
        frames[split] = frame
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(frames[a].participant_id) & set(frames[b].participant_id):
            raise ValueError(f"Original participant splits overlap: {run}")
    test = frames["test"]
    if test.empty:
        raise ValueError(f"Original run has no held-out participants: {run}")
    predictions = read_predictions(study, list(test.participant_id), test[target].to_numpy(float), selection.kind)
    return run, manifest, cmd, predictions


def launch_command(selection, gender, arm, reference_run, run_tag, dry_run=False):
    cmd = [sys.executable, str(ROOT/"paper_analysis"/"run_plaque_study.py"),
           "--platform", "gefion", "--suite", selection.suite, "--experiment", selection.experiment,
           "--gender", gender, "--seed", str(SEED), "--run-tag", run_tag, "--prior-arm", arm,
           "--reference-run", str(reference_run)]
    if arm == "boson_mmd":
        cmd += ["--boson-params", "{}", "--boson-sampler-path",
                str(ROOT/"third_party"/"nonlinear_photonic_sampler")]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def same_training_settings(original, derived):
    """Only prior/regularizer controls, determinism, run paths and executable may differ."""
    from meshvae.options import TrainOptions

    permitted = {"name", "checkpoints_dir", "cond_tsv", "val_cond_tsv", "latent_prior",
                 "prior_match_mode", "mmd_buffer_size", "mmd_prior_samples", "deterministic",
                 "boson_params", "boson_sampler_path"}
    def settings(command):
        return {key: value for key, value in vars(TrainOptions().parser.parse_args(command[2:])).items()
                if key not in permitted}
    if settings(original) != settings(derived):
        raise ValueError("Derived training settings differ from the original beyond the planned prior comparison")


def paired_difference(y, boson, comparator, kind, bootstraps=2000):
    import numpy as np
    from paper_analysis.plaque_reporting import metrics

    metric = "AUC" if kind == "binary" else "MAE"
    score = lambda idx, p: metrics(y[idx], p[idx], kind)[metric]
    idx = np.arange(len(y))
    delta = score(idx, boson)-score(idx, comparator)
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(y == value) for value in (0, 1)] if kind == "binary" else [idx]
    if any(not len(group) for group in groups):
        raise ValueError("AUC comparison requires both classes")
    samples = []
    for _ in range(bootstraps):
        resampled = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        samples.append(score(resampled, boson)-score(resampled, comparator))
    lower, upper = np.quantile(samples, [.025, .975])
    return {"Metric": metric, "BosonMinusComparator": delta, "CI95Lower": lower,
            "CI95Upper": upper, "BetterDirection": "positive" if kind == "binary" else "negative",
            "Bootstraps": bootstraps, "N": len(y)}


def collect(checkpoints, reference_tag, run_tag, output, bootstraps):
    import numpy as np
    import pandas as pd
    from paper_analysis.plaque_reporting import metrics

    statuses, results, deltas = [], [], []
    for selection in SELECTED:
        for gender in ("F", "M"):
            reference_run, original, original_cmd, original_predictions = check_reference(
                checkpoints, selection, gender, reference_tag)
            predictions = {"original_gaussian_kl": original_predictions}
            runs = {"original_gaussian_kl": reference_run}
            statuses.append({"Experiment": selection.experiment, "Gender": gender,
                             "Arm": "original_gaussian_kl", "Status": "completed", "Reason": ""})
            for arm in ARMS:
                run = checkpoints/run_name(selection, gender, run_tag, arm)
                study = run/"study"
                state = read_json(study/"status.json") if (study/"status.json").exists() else {"status": "not_started"}
                statuses.append({"Experiment": selection.experiment, "Gender": gender, "Arm": arm,
                                 "Status": state["status"], "Reason": state.get("reason") or ""})
                if state["status"] != "completed":
                    continue
                manifest = read_json(study/"manifest.json")
                if (manifest["experiment"] != original["experiment"] or manifest["gender"] != gender
                        or manifest["members"] != original["members"]
                        or manifest["config"]["prior_arm"] != arm or manifest["config"]["seed"] != SEED
                        or manifest["reference"]["fingerprint"] != original["fingerprint"]
                        or Path(manifest["reference"]["run"]).resolve() != reference_run.resolve()):
                    raise ValueError(f"Comparison is not matched to the original study: {run}")
                for split in ("train", "validation", "test"):
                    if manifest["reference"]["split_file_sha256"][split] != sha256(reference_run/"study"/f"{split}.tsv"):
                        raise ValueError(f"Original {split} values changed after training: {run}")
                trained_cmd = read_json(study/"train_command.json")
                same_training_settings(original_cmd, trained_cmd)
                from meshvae.options import TrainOptions
                opt = TrainOptions().parser.parse_args(trained_cmd[2:])
                if (opt.latent_prior != ("boson" if arm == "boson_mmd" else "gaussian")
                        or opt.prior_match_mode != "mmd" or opt.mmd_buffer_size != 64
                        or opt.mmd_prior_samples != 64 or not opt.deterministic):
                    raise ValueError(f"Unexpected trained prior/MMD settings: {run}")
                predictions[arm] = read_predictions(study, list(original_predictions.participant_id),
                                                    original_predictions.target.to_numpy(float), selection.kind)
                runs[arm] = run
            y = original_predictions.target.to_numpy(float)
            for arm, frame in predictions.items():
                results.append({"Experiment": selection.experiment, "Gender": gender, "Arm": arm,
                                **metrics(y, frame.meshvae.to_numpy(float), selection.kind), "Run": str(runs[arm])})
            if "boson_mmd" in predictions:
                boson = predictions["boson_mmd"].meshvae.to_numpy(float)
                for comparator in ("original_gaussian_kl", "gaussian_mmd"):
                    if comparator in predictions:
                        deltas.append({"Experiment": selection.experiment, "Gender": gender, "Comparator": comparator,
                                       **paired_difference(y, boson, predictions[comparator].meshvae.to_numpy(float),
                                                           selection.kind, bootstraps)})
    output.mkdir(parents=True, exist_ok=True)
    status_frame = pd.DataFrame(statuses)
    result_frame = pd.DataFrame(results)
    delta_frame = pd.DataFrame(deltas, columns=["Experiment", "Gender", "Comparator", "Metric",
        "BosonMinusComparator", "CI95Lower", "CI95Upper", "BetterDirection", "Bootstraps", "N"])
    status_frame.to_csv(output/"status.csv", index=False)
    result_frame.to_csv(output/"comparison.csv", index=False)
    delta_frame.to_csv(output/"paired_differences.csv", index=False)
    note = {"seed": SEED, "selected_models": [asdict(s) for s in SELECTED],
            "primary_classifier_metric": "AUC", "primary_volume_metric": "MAE on original volume scale",
            "comparison": "Boson-MMD vs Gaussian-MMD isolates the prior distribution; original Gaussian-KL is a historical reference",
            "intervals": "95% participant percentile bootstrap; binary resamples stratified by class; paired fixed fitted predictions",
            "interpretation": "Exploratory single-seed comparison. Models were selected after viewing these held-out results. Intervals do not include training-seed uncertainty or selection bias.",
            "sampler": "Bundled Sparrow simulator, existing linear settings; no photonic hardware job is submitted"}
    (output/"comparison_notes.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(status_frame.to_string(index=False), flush=True)
    columns = [c for c in ("Experiment", "Gender", "Arm", "N", "AUC", "AveragePrecision", "MAE", "R2", "Spearman") if c in result_frame]
    print("\nHELD-OUT RESULTS\n"+result_frame[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if deltas:
        print("\nPAIRED DIFFERENCES (Boson minus comparator)\n"+delta_frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    new_status = status_frame[status_frame.Arm != "original_gaussian_kl"]
    print(f"\nCompleted new runs: {(new_status.Status == 'completed').sum()}/8")
    print(f"Results saved in {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "check", "run", "collect"))
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--reference-tag", default=REFERENCE_TAG)
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--task-index", type=int, choices=range(8))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"best_boson_results")
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()
    if args.action == "list":
        for index, (selection, gender, arm) in enumerate(task_plan()):
            print(index, selection.experiment, gender, arm)
        return
    if args.action == "collect":
        if args.bootstraps < 1:
            parser.error("--bootstraps must be positive")
        collect(args.checkpoints_dir, args.reference_tag, args.run_tag, args.output_dir, args.bootstraps)
        return
    if args.action == "check":
        check_runtime_compatibility()
        for selection in SELECTED:
            for gender in ("F", "M"):
                run, _, cmd, predictions = check_reference(args.checkpoints_dir, selection, gender, args.reference_tag)
                budget = cmd[cmd.index("--niter")+1]
                print(f"[OK] {run.name}: {len(predictions)} held-out participants, niter={budget}", flush=True)
        print("[PASS] All four original models are available with saved splits and training settings.")
        return
    if args.task_index is None:
        parser.error("run requires --task-index; use the Slurm launcher for all eight tasks")
    selection, gender, arm = task_plan()[args.task_index]
    run, manifest, _, _ = check_reference(args.checkpoints_dir, selection, gender, args.reference_tag)
    if Path(manifest["config"]["platforms"]["gefion"]["checkpoints"]).resolve() != args.checkpoints_dir.resolve():
        parser.error("Checkpoint root differs from the saved reference configuration; use its original root")
    print(f"[TASK {args.task_index}] {selection.experiment} {gender} {arm}; reference={run}", flush=True)
    subprocess.run(launch_command(selection, gender, arm, run, args.run_tag, args.dry_run), check=True)
    derived = args.checkpoints_dir/run_name(selection, gender, args.run_tag, arm)/"study"
    status = read_json(derived/"status.json").get("status")
    if status != ("dry_run" if args.dry_run else "completed"):
        raise RuntimeError(f"Task ended without the expected completion state: {status}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
