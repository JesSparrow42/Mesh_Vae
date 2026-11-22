# Cluster scripts

Run these commands from the repository root in the appropriate cluster environment.

| Folder | Contents |
| --- | --- |
| `gefion/` | Gefion preprocessing and long-training scripts. |
| `computerome/` | Computerome setup, measurement training, and results-overview scripts. |

For example:

```bash
bash scripts/gefion/gefion_preprocess.sh
```

Read the script before running it: project paths, module loads, Python environments,
and compute-resource requirements are specific to the cluster. Follow the relevant
[workflow guide](../docs/README_BODY_MEASUREMENTS.md) for setup and training commands.
Additional paper-study jobs remain alongside their workflow in
[`paper_analysis/`](../paper_analysis/README.md).

Copy-and-paste command references are in [`docs/commands/`](../docs/commands/).
