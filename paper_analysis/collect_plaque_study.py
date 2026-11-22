#!/usr/bin/env python3
"""Collect next-step study statuses and held-out baseline/model comparisons."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper_analysis.study_config import DEFAULT_CONFIG, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path)
    parser.add_argument("--platform", choices=["gefion", "computerome"],
                        help="Read checkpoint directory and run tag from the study configuration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-tag")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.platform:
        config = load_config(args.config, args.platform)
        if args.checkpoints_dir is None:
            args.checkpoints_dir = Path(config["platforms"][args.platform]["checkpoints"])
        if args.run_tag is None:
            args.run_tag = config["run_tag"]
    elif args.checkpoints_dir is None:
        parser.error("Pass --platform or --checkpoints-dir")
    args.run_tag = args.run_tag or "next_v1"
    if args.output_dir is None:
        args.output_dir = ROOT/"next_steps_results" if args.platform else Path("next_steps_results")
    statuses, comparisons = [], []
    for manifest_path in sorted(args.checkpoints_dir.glob("*_NEXT_*/study/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest["config"]["run_tag"] != args.run_tag:
            continue
        study = manifest_path.parent
        status_path = study/"status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "unknown"}
        for key, count in status.pop("counts", {}).items():
            status[f"{key}_n"] = count
        statuses.append(status)
        comparison = study/"heldout"/"comparison.csv"
        if comparison.exists():
            frame = pd.read_csv(comparison)
            frame.insert(0, "Experiment", manifest["experiment"]["key"])
            frame.insert(0, "Gender", manifest["gender"])
            frame.insert(0, "Run", manifest_path.parents[1].name)
            frame["Cohort"] = manifest["experiment"]["cohort"]
            comparisons.append(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(statuses).to_csv(args.output_dir/"status.csv", index=False)
    (pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()).to_csv(args.output_dir/"comparisons.csv", index=False)
    print(f"Collected {len(statuses)} experiments into {args.output_dir}")
    if statuses:
        print(pd.Series([s["status"] for s in statuses]).value_counts().to_string())


if __name__ == "__main__":
    main()
