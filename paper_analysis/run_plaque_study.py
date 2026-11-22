#!/usr/bin/env python3
"""Run one or all prespecified sex-specific MeshVAE plaque experiments."""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from paper_analysis.plaque_cohorts import TERRITORIES, load_splits, select_cohort, training_problem
from paper_analysis.plaque_reporting import report, combine_hurdle
from paper_analysis.study_config import load_config


@dataclass(frozen=True)
class Experiment:
    key: str
    territory: str
    kind: str = "regression"
    cohort: str = "all"
    transform: str = "log1p"
    loss: str = "huber"
    weight: bool = True


def experiments(suite):
    if suite == "territory_classification":
        return [Experiment(f"{t}_presence", t, "binary") for t in TERRITORIES]
    if suite == "territory_volume":
        return [Experiment(f"{t}_volume", t) for t in TERRITORIES]
    if suite == "plaque":
        return [Experiment("all_raw", "peripheral_total", transform="identity"),
                Experiment("all_log", "peripheral_total"),
                Experiment("positive_log", "peripheral_total", cohort="positive"),
                Experiment("positive_median", "peripheral_total", cohort="positive", loss="mae"),
                Experiment("hurdle", "peripheral_total", kind="hurdle", loss="mse")]
    if suite == "sensitivity":
        return [Experiment("age_top_log", "peripheral_total", cohort="age_top"),
                Experiment("top5_log", "peripheral_total", cohort="top5")]
    if suite == "weight":
        return [Experiment(f"{kind}_{cov}", "peripheral_total", kind=kind, weight=(cov == "age_weight"))
                for kind in ("binary", "regression") for cov in ("age", "age_weight")]
    raise ValueError(suite)


SUITES = ("territory_classification", "territory_volume", "plaque", "sensitivity", "weight")
PRIOR_ARMS = {"gaussian_kl": ("gaussian", "kl"),
              "gaussian_mmd": ("gaussian", "mmd"),
              "boson_mmd": ("boson", "mmd")}


def tasks(suite):
    return [(gender, experiment) for gender in ("F", "M") for experiment in experiments(suite)]


def command(args):
    print("[COMMAND] " + shlex.join([str(x) for x in args]), flush=True)
    subprocess.run([str(x) for x in args], check=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def study_name(gender, experiment, suite, config, platform):
    arm = f"_{config['prior_arm']}" if config.get("prior_arm") else ""
    return f"{platform.upper()}_NEXT_{suite}_{experiment.key}{arm}_{gender}_{config['run_tag']}_seed{config['seed']}"


def set_option(cmd, option, value):
    if option in cmd:
        cmd[cmd.index(option)+1] = str(value)
    else:
        cmd += [option, str(value)]


def training_command(name, paths, experiment, gender, config, platform):
    p = config["platforms"][platform]
    if config.get("reference_train_command"):
        cmd = list(config["reference_train_command"])
        cmd[:2] = [sys.executable, str(ROOT/"train_mesh_vae.py")]
        for option, value in [("--name", name), ("--checkpoints_dir", p["checkpoints"]),
                              ("--cond_tsv", paths["train"]), ("--val_cond_tsv", paths["validation"]),
                              ("--seed", config["seed"]),
                              ("--split_seed", config.get("split_seed", config["seed"]))]:
            set_option(cmd, option, value)
    else:
        cmd = default_training_command(name, paths, experiment, gender, config, platform)
    if config.get("prior_arm"):
        latent_prior, prior_match = PRIOR_ARMS[config["prior_arm"]]
        for option, value in [("--latent_prior", latent_prior), ("--prior_match_mode", prior_match),
                              ("--mmd_buffer_size", 64), ("--mmd_prior_samples", 64)]:
            set_option(cmd, option, value)
        if "--deterministic" not in cmd:
            cmd += ["--deterministic"]
        if latent_prior == "boson":
            set_option(cmd, "--boson_params", json.dumps(config.get("boson_params", {}), sort_keys=True))
            if config.get("boson_sampler_path"):
                set_option(cmd, "--boson_sampler_path", config["boson_sampler_path"])
    return cmd


def default_training_command(name, paths, experiment, gender, config, platform):
    p = config["platforms"][platform]
    cmd = [sys.executable, str(ROOT/"train_mesh_vae.py"), "--name", name,
           "--checkpoints_dir", p["checkpoints"], "--dataroot", p["dataroot"],
           "--cond_tsv", str(paths["train"]), "--val_cond_tsv", str(paths["validation"]),
           "--only_gender", gender, "--gpu_ids", "0", "--seed", str(config["seed"]),
           "--split_seed", str(config.get("split_seed", config["seed"])), "--niter", str(config["niter"]),
           "--niter_decay", str(config["niter_decay"]), "--batch_size", str(config["batch_size"]),
           "--num_threads", "0", "--save_epoch_freq", "250", "--beta", "0.001", "--beta_warmup_epochs", "30",
           "--feature_source", "scan", "--use_preprocessed", "--use_pointcloud", "--use_cond",
           "--cond_stregkode_col", "participant_id", "--cond_age_col", "study_age", "--cond_gender_col", "study_gender",
           "--cond_age_source", "age", "--age_bins", "18", "35", "50", "60", "200",
           "--cond_age_mode", "rbf", "--encoder_cond_mode", "demo", "--encoder_arch", "meshconv",
           "--encoder_max_points", "4096", "--mesh_use_knn", "--mesh_knn_k", "12", "--mesh_knn_max_points", "4096",
           "--use_scan_geometry_features", "--require_scan_geometry_features", "--restore_metric_scale",
           "--scan_geometry_features_file", "scan_geometry_features.npz",
           "--use_point_labels", "--point_label_pooling", "--point_label_regions", "6",
           "--no-use_amp", "--no-cond_athero", "--no-cond_plaque", "--no-aux_predict_athero",
           "--no-aux_predict_plaque", "--no-balance_athero"]
    cmd += ["--cond_weight", "--weight_col", "study_weight_kg"] if experiment.weight else ["--no-cond_weight"]
    if experiment.kind == "binary":
        # Use unweighted BCE: a hurdle requires probabilities, not cost-weighted scores.
        cmd += ["--aux_predict_binary", "--no-aux_predict_regression", "--aux_binary_col",
                f"presence_{experiment.territory}", "--aux_binary_weight", "10", "--aux_binary_pos_weight", "1"]
    else:
        cmd += ["--aux_predict_regression", "--no-aux_predict_binary", "--aux_regression_col",
                f"volume_{experiment.territory}", "--aux_regression_transform", experiment.transform,
                "--aux_regression_loss", experiment.loss, "--aux_regression_weight", "20"]
    return cmd


def diagnosis(name, gender, kind, frame, split, outdir, config, platform):
    if frame.empty:
        return pd.DataFrame()
    p = config["platforms"][platform]
    path = outdir/f"evaluate_{split}.tsv"
    frame.to_csv(path, sep="\t", index=False)
    diag = "binary" if kind == "binary" else "regression"
    command([sys.executable, ROOT/f"diagnose_mesh_{diag}.py", "--name", name,
             "--checkpoints_dir", p["checkpoints"], "--dataroot", p["dataroot"], "--cond_tsv", path,
             "--only_gender", gender, "--gpu_ids", "0", "--which_epoch", "best",
             "--batch_size", str(config["batch_size"]), "--seed", str(config["seed"]),
             "--cond_stregkode_col", "participant_id", "--cond_age_col", "study_age", "--cond_gender_col", "study_gender"])
    run = Path(p["checkpoints"])/name
    summary = pd.read_csv(run/f"{diag}_summary.csv")
    result_dir = Path(summary.iloc[0]["ResultsDir"])
    predictions = pd.read_csv(result_dir/"predictions.csv", dtype={"participant_id": str})
    if predictions.participant_id.duplicated().any() or set(predictions.participant_id) != set(frame.participant_id):
        raise ValueError("Missing or duplicate predictions: scan/metadata eligibility differs from the audit.")
    predictions.to_csv(outdir/f"{split}_predictions.csv", index=False)
    shutil.copyfile(result_dir/"metrics.json", outdir/f"{split}_metrics.json")
    return predictions


def execute_model(name, gender, exp, selected, evaluation, outdir, config, platform, dry_run=False):
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, frame in selected.items():
        paths[key] = outdir/f"{key}.tsv"
        frame.to_csv(paths[key], sep="\t", index=False)
    cmd = training_command(name, paths, exp, gender, config, platform)
    write_json(outdir/"train_command.json", cmd)
    if dry_run:
        print("[DRY RUN] " + shlex.join(cmd), flush=True)
        return {}
    run = Path(config["platforms"][platform]["checkpoints"])/name
    done = outdir/"training_completed.json"
    if not done.exists():
        if (run/"latest_vae_net.pth").exists() or (run/"best_vae_net.pth").exists():
            raise RuntimeError(f"Incomplete previous training in {run}. Change run_tag to restart; a partial checkpoint is not treated as finished.")
        if config.get("prior_arm") == "boson_mmd":
            latent_dim = cmd[cmd.index("--latent_dim")+1] if "--latent_dim" in cmd else "256"
            preflight = [sys.executable, str(ROOT/"check_boson_prior.py"), "--latent_dim", latent_dim,
                         "--samples", "512", "--boson_params", json.dumps(config.get("boson_params", {}), sort_keys=True)]
            if config.get("boson_sampler_path"):
                preflight += ["--boson_sampler_path", config["boson_sampler_path"]]
            command(preflight)
        command(cmd)
        if not (run/"best_vae_net.pth").is_file():
            raise RuntimeError("Training exited without a best checkpoint")
        write_json(done, {"status": "completed"})
    return {split: diagnosis(name, gender, exp.kind, frame, split, outdir, config, platform)
            for split, frame in evaluation.items()}


def run_task(gender, exp, suite, config, platform, audit_only=False, dry_run=False):
    p = config["platforms"][platform]
    name = study_name(gender, exp, suite, config, platform)
    run = Path(p["checkpoints"])/name
    out = run/"study"; out.mkdir(parents=True, exist_ok=True)
    print(f"\n[EXPERIMENT] {name}", flush=True)
    target = f"{'presence' if exp.kind == 'binary' else 'volume'}_{exp.territory}"
    if config.get("reference_run"):
        reference = Path(config["reference_run"])/"study"
        reference_manifest = json.loads((reference/"manifest.json").read_text())
        if reference_manifest["experiment"] != asdict(exp) or reference_manifest["gender"] != gender:
            raise ValueError("Reference experiment/sex does not match the requested task")
        selected = {}
        reference_files = {}
        for key in ("train", "validation", "test"):
            path = reference/f"{key}.tsv"
            frame = pd.read_csv(path, sep="\t", dtype={"participant_id": str})
            if frame.participant_id.duplicated().any() or list(frame.participant_id) != reference_manifest["members"][key]:
                raise ValueError(f"Reference {key} participants differ from its saved manifest")
            if not (frame.study_gender == gender).all() or target not in frame:
                raise ValueError(f"Reference {key} has incorrect sex/target data")
            selected[key] = frame
            reference_files[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        if any(set(selected[a].participant_id) & set(selected[b].participant_id)
               for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError("Reference participant splits overlap")
        audit, rule = reference_manifest["audit"], reference_manifest["cohort_rule"]
    else:
        splits, audit = load_splits(Path(p["metadata"]), Path(p["train_template"].format(gender=gender)),
                                   Path(p["test_template"].format(gender=gender)), Path(p["dataroot"]),
                                   gender, config.get("split_seed", config["seed"]), config["validation_fraction"],
                                   config.get("weight_col"), config.get("age_col"))
        selected, rule = select_cohort(splits, target, exp.cohort, config["age_top_fraction"], config["age_band_width"])
    # Hurdle components use identical outer eligibility: observed full peripheral volume.
    if exp.kind == "hurdle":
        for frame in selected.values():
            frame[f"presence_{exp.territory}"] = (frame[target] > 0).astype(float)
    manifest = {"experiment": asdict(exp), "gender": gender, "config": config, "audit": audit, "cohort_rule": rule,
                "members": {k: list(v.participant_id) for k, v in selected.items()},
                "interpretation": "outcome-selected exploratory subgroup" if exp.cohort in {"top5", "age_top"} else "prespecified cohort"}
    if config.get("reference_run"):
        manifest["reference"] = {"run": config["reference_run"], "fingerprint": reference_manifest["fingerprint"],
                                 "split_file_sha256": reference_files}
    # Fingerprint actual values too: never reuse a finished model after metadata changed.
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode())
    for key in sorted(selected):
        signature.update(selected[key].to_csv(index=False).encode())
    manifest["fingerprint"] = signature.hexdigest()
    prior = out/"manifest.json"
    if prior.exists() and json.loads(prior.read_text())["fingerprint"] != manifest["fingerprint"]:
        raise RuntimeError("Study inputs/settings changed under an existing run name. Set a new run_tag.")
    write_json(prior, manifest)
    for key, frame in selected.items():
        frame.to_csv(out/f"{key}.tsv", sep="\t", index=False)
    reason = training_problem(selected, target, "regression" if exp.kind == "hurdle" else exp.kind,
                              config["minimum_train"], config["minimum_validation"])
    positive, positive_rule = None, None
    if exp.kind == "hurdle" and reason is None:
        positive, positive_rule = select_cohort(selected, target, "positive")
        reason = training_problem(positive, target, "regression", config["minimum_train"], config["minimum_validation"])
        reason = reason or training_problem(selected, f"presence_{exp.territory}", "binary", config["minimum_train"], config["minimum_validation"])
        write_json(out/"positive_cohort_rule.json", positive_rule)
    status = {"name": name, "status": "unavailable" if reason else "audited", "reason": reason,
              "counts": {k: len(v) for k, v in selected.items()}, "gender": gender, "experiment": exp.key}
    write_json(out/"status.json", status)
    print("[AUDIT] " + json.dumps(status), flush=True)
    if reason or audit_only:
        return status
    if not dry_run:
        status["status"] = "running"
        write_json(out/"status.json", status)
    if exp.kind == "hurdle":
        binary = Experiment("presence", exp.territory, "binary", weight=exp.weight)
        burden = Experiment("positive", exp.territory, cohort="positive", loss="mse", weight=exp.weight)
        binary_pred = execute_model(name+"_presence", gender, binary, selected,
                                    {"test": selected["test"]}, out/"presence", config, platform, dry_run)
        volume_pred = execute_model(name+"_positive", gender, burden, positive,
                                    {"validation": positive["validation"], "test": selected["test"]},
                                    out/"positive", config, platform, dry_run)
        if not dry_run and len(selected["test"]):
            prediction = combine_hurdle(binary_pred["test"], volume_pred["test"], volume_pred["validation"], out)
            prediction.to_csv(out/"test_predictions.csv", index=False)
    else:
        predictions = execute_model(name, gender, exp, selected, {"test": selected["test"]}, out, config, platform, dry_run)
        if not dry_run:
            prediction = predictions["test"]
    if not dry_run and len(selected["test"]):
        report(selected["train"], selected["test"], prediction, target,
               "binary" if exp.kind == "binary" else "regression", out/"heldout", p["dataroot"],
               config["seed"], config["bootstraps"])
    status["status"] = "dry_run" if dry_run else "completed"
    if not len(selected["test"]):
        status["evaluation_note"] = "No eligible held-out participants; no performance estimate available."
    write_json(out/"status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["gefion", "computerome"], required=True)
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("next_steps_config.json"))
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--gender", choices=["F", "M"], help="Optional restriction; default always includes both sexes")
    parser.add_argument("--experiment", help="Restrict to an experiment key, e.g. all_log")
    parser.add_argument("--seed", type=int, help="Training seed; does not change the configured participant split")
    parser.add_argument("--split-seed", type=int, help="Fixed participant split seed; defaults to the config's original seed")
    parser.add_argument("--run-tag", help="Distinct tag for a new study variant")
    parser.add_argument("--prior-arm", choices=list(PRIOR_ARMS), help="Matched prior ablation; existing launches keep their defaults")
    parser.add_argument("--boson-params", help="JSON object for the Sparrow sampler; default uses its existing linear settings")
    parser.add_argument("--boson-sampler-path", help="Optional path to a newer Sparrow sampler checkout")
    parser.add_argument("--reference-run", type=Path,
                        help="Completed study run whose saved config, training command and participant TSVs must be reused")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.reference_run:
        reference = args.reference_run.resolve()/"study"
        if not (reference/"manifest.json").is_file() or not (reference/"train_command.json").is_file():
            parser.error(f"Reference study files missing in {reference}. Set --reference-run to a completed all_log run from the overview.")
        status_path = reference/"status.json"
        if not status_path.is_file() or json.loads(status_path.read_text()).get("status") != "completed":
            parser.error(f"Reference study is not completed: {reference}")
        config = json.loads((reference/"manifest.json").read_text())["config"]
        config["reference_run"] = str(args.reference_run.resolve())
        config["reference_train_command"] = json.loads((reference/"train_command.json").read_text())
        config.setdefault("split_seed", config["seed"])
        if args.split_seed is not None and args.split_seed != config["split_seed"]:
            parser.error("--split-seed cannot change the reference participant split")
    else:
        config = load_config(args.config, args.platform)
    if args.seed is not None:
        config.setdefault("split_seed", config["seed"])
        config["seed"] = args.seed
    if args.split_seed is not None:
        config["split_seed"] = args.split_seed
    if args.run_tag is not None:
        config["run_tag"] = args.run_tag
    if args.prior_arm is not None:
        config["prior_arm"] = args.prior_arm
        config["latent_prior"], config["prior_match_mode"] = PRIOR_ARMS[args.prior_arm]
    if args.boson_params is not None:
        config["boson_params"] = json.loads(args.boson_params)
    if not isinstance(config.get("boson_params", {}), dict):
        parser.error("--boson-params must be a JSON object")
    if args.boson_sampler_path is not None:
        config["boson_sampler_path"] = args.boson_sampler_path
    print(f"[PATH] Code: {ROOT}", flush=True)
    print(f"[PATH] Checkpoints: {config['platforms'][args.platform]['checkpoints']}", flush=True)
    if int(config["niter"])+int(config["niter_decay"]) < 1:
        raise ValueError("At least one training epoch required")
    plan = [(s, g, e) for s in (SUITES if args.suite == "all" else [args.suite]) for g, e in tasks(s)]
    if args.task_index is not None:
        if not 0 <= args.task_index < len(plan):
            parser.error(f"Task index must be between 0 and {len(plan)-1}")
        plan = [plan[args.task_index]]
    if args.gender:
        plan = [x for x in plan if x[1] == args.gender]
    if args.experiment:
        plan = [x for x in plan if x[2].key == args.experiment]
        if not plan:
            parser.error(f"No experiment {args.experiment!r} in the selected suite/tasks")
    if args.list:
        for i, (s, g, e) in enumerate(plan):
            print(i, s, g, e.key)
        return
    failed = []
    for suite, gender, exp in plan:
        try:
            run_task(gender, exp, suite, config, args.platform, args.audit_only, args.dry_run)
        except Exception as exc:
            failed.append(f"{suite}/{gender}/{exp.key}: {type(exc).__name__}: {exc}")
            print("[FAILED] " + failed[-1], file=sys.stderr, flush=True)
            name = study_name(gender, exp, suite, config, args.platform)
            status_path = Path(config["platforms"][args.platform]["checkpoints"])/name/"study"/"status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if status.get("status") != "completed":
                    status.update(status="failed", reason=failed[-1])
                    write_json(status_path, status)
    if failed:
        raise SystemExit("\n".join(failed))


if __name__ == "__main__":
    main()
