#!/usr/bin/env python3
"""Clean figures from the explicit matched-study plan; no training or project imports."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

GROUPS = ("F", "M", "combined")
GROUP_LABELS = {"F": "Women", "M": "Men", "combined": "Combined model"}
COLORS = {"F": "#178D91", "M": "#D47560", "combined": "#7963A5"}
VARIANTS = ("age_only", "scans_only", "scans_age")
LABELS = {"age_only": "Age only", "scans_only": "Scans only", "scans_age": "Scans + age",
          "scans_age_plaque_aux": "Scans + age\n+ plaque supervision",
          "volume_control": "Classifier", "volume_aux": "Classifier\n+ volume supervision"}
VOLUME_TARGET = "peripheral_total_volume_training"
VOLUME_VARIANTS = ("volume_control", "volume_aux")
TERRITORIES = ("carotid_left", "carotid_right", "carotid_total", "femoral_left", "femoral_right", "femoral_total", "coronary_volume")
TITLES = {k: k.replace("_", " ").title() for k in TERRITORIES}
TITLES.update(coronary_volume="Coronary", peripheral_total="Overall peripheral plaque presence")
TITLES[VOLUME_TARGET] = "Does plaque-volume supervision help classification?"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(path):
    data = pd.read_csv(path, sep="\t" if str(path).endswith(".tsv") else ",", dtype={"participant_id": str})
    if "participant_id" not in data or data.participant_id.isna().any() or data.participant_id.duplicated().any():
        raise ValueError(f"Expected unique nonempty participant IDs: {path}")
    return data.set_index("participant_id").sort_index()


def roc_auc(y, score):
    y, score = np.asarray(y, int), np.asarray(score, float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        raise ValueError("ROC AUC requires both outcome classes")
    order = np.argsort(score, kind="stable")
    _, start, counts = np.unique(score[order], return_index=True, return_counts=True)
    ranks = np.empty(len(y)); ranks[order] = np.repeat(start + (counts+1)/2, counts)
    return float((ranks[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def intervals(y, values, sex, count, seed=42):
    """Same sex-by-outcome stratified participant draws for every predictor arm."""
    y, sex = np.asarray(y, int), np.asarray(sex, str)
    strata = [np.flatnonzero((sex == s) & (y == c)) for s in sorted(set(sex)) for c in (0, 1)]
    strata = [g for g in strata if len(g)]
    keys = list(values); scores = np.column_stack([values[k] for k in keys])
    point = np.array([roc_auc(y, scores[:, i]) for i in range(len(keys))])
    draws = np.empty((count, len(keys))); rng = np.random.default_rng(seed)
    for b in range(count):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in strata])
        for i in range(len(keys)):
            draws[b, i] = roc_auc(y[ix], scores[ix, i])
    low, high = np.quantile(draws, [.025, .975], axis=0)
    rows = [{"Variant": k, "AUC": point[i], "CI95Lower": low[i], "CI95Upper": high[i]} for i, k in enumerate(keys)]
    comparisons = []
    for a, b in (("scans_age", "age_only"), ("scans_age", "scans_only"), ("volume_aux", "volume_control")):
        if a in keys and b in keys:
            i, j = keys.index(a), keys.index(b)
            lo, hi = np.quantile(draws[:, i]-draws[:, j], [.025, .975])
            comparisons.append({"Contrast": f"{a} minus {b}", "AUCDifference": point[i]-point[j], "CI95Lower": lo, "CI95Upper": hi})
    return rows, comparisons


def collect(root, bootstraps=5000):
    root = Path(root); plan_path = root/"plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("combined_policy", "dedicated") != "dedicated":
        raise ValueError("These requested figures require a model trained jointly on women and men")
    rows, contrasts, inventory, sources = [], [], [], [{"file": str(plan_path), "sha256": digest(plan_path)}]
    cohorts = plan.get("cohorts", plan.get("studies", []))
    if not cohorts:
        raise ValueError("The plan contains no explicitly prepared cohorts")
    study_keys = [(s["target"], s["group"]) for s in cohorts]
    if len(study_keys) != len(set(study_keys)):
        raise ValueError("Duplicate target/group studies in the explicit plan")
    for cohort in cohorts:
        target, group = cohort["target"], cohort["group"]
        base = root/target/group
        test_path = base/"splits"/"test.tsv"
        if not test_path.is_file():
            inventory.append({"Target": target, "Group": group, "Variant": "all", "Status": "missing_split"})
            continue
        test = read_table(test_path); col = cohort.get("endpoint", "presence_"+target)
        if col not in test or "study_gender" not in test:
            raise ValueError(f"Missing canonical target or true sex: {test_path}")
        y = pd.to_numeric(test[col], errors="coerce").to_numpy(float)
        if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError(f"Invalid binary targets: {test_path}")
        true_sex = test.study_gender.astype(str).to_numpy()
        if not set(true_sex).issubset({"F", "M"}) or (group != "combined" and not (true_sex == group).all()):
            raise ValueError(f"Incorrect true sex for {target}/{group}")
        sources.append({"file": str(test_path), "sha256": digest(test_path)})
        values = {}
        variants = VOLUME_VARIANTS if target == VOLUME_TARGET else VARIANTS
        for variant in variants:
            path = base/"predictions"/f"{variant}.csv"
            if not path.is_file():
                inventory.append({"Target": target, "Group": group, "Variant": variant, "Status": "missing"})
                continue
            if target == VOLUME_TARGET:
                task = next((t for t in plan["tasks"] if t["target"] == target and t["group"] == group and t["variant"] == variant), None)
                marker = base/f"{variant}_inference_check.json"
                check = json.loads(marker.read_text()) if marker.is_file() else {}
                if not task or check.get("status") != "passed" or check.get("fingerprint") != task["fingerprint"]:
                    inventory.append({"Target": target, "Group": group, "Variant": variant, "Status": "inference_check_pending"})
                    continue
                sources.append({"file": str(marker), "sha256": digest(marker)})
            pred = read_table(path)
            if set(pred.index) != set(test.index):
                raise ValueError(f"Prediction membership differs from saved test split: {path}")
            pred = pred.loc[test.index]
            if not {"target", "probability"}.issubset(pred):
                raise ValueError(f"Missing target/probability columns: {path}")
            p = pred[["target", "probability"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            if not np.isfinite(p).all() or not np.array_equal(p[:, 0], y) or ((p[:, 1] < 0) | (p[:, 1] > 1)).any():
                raise ValueError(f"Incorrect targets or probabilities: {path}")
            if "group" in pred and not (pred.group == group).all():
                raise ValueError(f"Incorrect model group: {path}")
            values[variant] = p[:, 1]
            sources.append({"file": str(path), "sha256": digest(path)})
            inventory.append({"Target": target, "Group": group, "Variant": variant, "Status": "available"})
        if values and len(set(y)) == 2:
            estimates, differences = intervals(y, values, true_sex, bootstraps)
            common = {"Target": target, "Group": group, "N": len(y), "Positive": int(y.sum())}
            rows += [{**common, **row} for row in estimates]
            contrasts += [{**common, **row} for row in differences]
        elif values:
            inventory.append({"Target": target, "Group": group, "Variant": "all", "Status": "AUC_undefined_one_class"})
    return pd.DataFrame(rows), pd.DataFrame(contrasts), pd.DataFrame(inventory), sources


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def panel(ax, table, target, variants, small=False):
    x = np.arange(len(variants)); width = .23
    for gi, group in enumerate(GROUPS):
        selected = table[(table.Target == target) & (table.Group == group)] if len(table) else pd.DataFrame()
        for j, variant in enumerate(variants):
            at = x[j]+(gi-1)*width
            row = selected[selected.Variant == variant] if len(selected) else pd.DataFrame()
            if row.empty:
                ax.text(at, .035, "Pending", rotation=90, ha="center", va="bottom", color="#7A8790", fontsize=7)
                continue
            if len(row) != 1:
                raise ValueError(f"Duplicate plot result {target}/{group}/{variant}")
            r = row.iloc[0]
            ax.bar(at, r.AUC, width*.94, color=COLORS[group])
            ax.vlines(at, r.CI95Lower, r.CI95Upper, colors="#223847", linewidth=1)
            ax.hlines([r.CI95Lower, r.CI95Upper], at-.025, at+.025, colors="#223847", linewidth=1)
            if not small:
                ax.text(at, max(r.CI95Upper, r.AUC)+.018, f"{r.AUC:.3f}", ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=[LABELS[v] for v in variants], ylim=(0, 1.075), yticks=np.arange(0, 1.01, .25))
    ax.axhline(.5, color="#9DA9B2", linewidth=.8, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5EAEE", linewidth=.6); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, labelsize=8 if small else 11)
    ax.tick_params(axis="y", length=0, labelsize=8 if small else 10)
    ax.set_title(TITLES.get(target, target), fontsize=11 if small else 17, pad=12)


def save(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(str(stem)+"."+suffix, dpi=300, facecolor="white", bbox_inches="tight")
    pyplot().close(fig)


def plot(table, output):
    plt = pyplot(); output = Path(output)
    from matplotlib.patches import Patch
    legend = [Patch(color=COLORS[g], label=GROUP_LABELS[g]) for g in GROUPS]
    footer = "95% participant bootstrap intervals. Combined = separately trained joint-sex model.\nAge-only = logistic baseline; each panel uses identical held-out participants across available predictors."
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    panel(ax, table, "peripheral_total", VARIANTS)
    ax.set_ylabel("ROC AUC"); fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .96), ncol=3, frameon=False)
    fig.text(.5, .02, footer, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .085, 1, .89)); save(fig, output/"01_main_auc")
    fig, axes = plt.subplots(2, 4, figsize=(14, 8.5))
    for ax, target in zip(axes.flat, TERRITORIES):
        panel(ax, table, target, VARIANTS, small=True); ax.set_ylabel("ROC AUC", fontsize=9)
    axes.flat[-1].axis("off"); axes.flat[-1].legend(handles=legend, loc="center", frameon=False, fontsize=11)
    fig.suptitle("Plaque presence by territory", fontsize=19)
    fig.text(.5, .015, footer+"\nTerritory-specific missing outcomes mean different territories may contain different participants.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .095, 1, .94), h_pad=3, w_pad=2); save(fig, output/"02_territory_auc")
    if len(table) and table.Target.eq(VOLUME_TARGET).any():
        fig, ax = plt.subplots(figsize=(9, 5.7))
        panel(ax, table, VOLUME_TARGET, VOLUME_VARIANTS)
        ax.set_ylabel("ROC AUC"); ax.set_title("Plaque volume as additional training supervision", fontsize=16)
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .96), ncol=3, frameon=False)
        fig.text(.5, .02, "Both classifiers use identical development participants with observed volume.\nAt test time the classifier receives scans and age only. Error bars: 95% paired participant bootstrap intervals.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .105, 1, .89)); save(fig, output/"03_volume_supervision_auc")


def volume_definition_check(study_dir, output):
    """Observed volume is the outcome definition, not an independent predictor."""
    rows = []
    for group in GROUPS:
        path = Path(study_dir)/"peripheral_total"/group/"splits"/"test.tsv"
        if not path.is_file():
            continue
        frame = read_table(path)
        columns = ["volume_peripheral_total", "presence_peripheral_total"]
        if not set(columns).issubset(frame):
            continue
        values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        keep = np.isfinite(values).all(axis=1) & (values[:, 0] >= 0)
        v, y = values[keep, 0], values[keep, 1]
        if not np.isin(y, [0, 1]).all() or not np.array_equal((v > 0).astype(float), y):
            raise ValueError(f"Volume/presence definition disagrees in {path}")
        score = roc_auc(y, (v > 0).astype(float)) if len(set(y)) == 2 else None
        rows.append({"Group": group, "N": len(y), "MissingVolume": int((~keep).sum()),
                     "AUC": score, "Meaning": "Deterministic target-definition check; not a trained prediction model"})
    table = pd.DataFrame(rows)
    table.to_csv(Path(output)/"volume_presence_definition.csv", index=False)
    if rows:
        plt = pyplot(); fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for i, row in enumerate(rows):
            if row["AUC"] is not None:
                ax.bar(i, row["AUC"], color=COLORS[row["Group"]], width=.55)
                ax.text(i, 1.02, f"1.000\nN = {row['N']}", ha="center", fontsize=12)
        ax.set(xticks=np.arange(len(rows)), xticklabels=["All participants" if r["Group"] == "combined" else GROUP_LABELS[r["Group"]] for r in rows],
               ylim=(0, 1.17), yticks=np.arange(0, 1.01, .25), ylabel="ROC AUC", title="Measured plaque volume determines plaque presence")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E5EAEE"); ax.set_axisbelow(True)
        fig.text(.5, .025, "Presence is defined as measured volume > 0. No model is fitted.\nThis is a definition check, not evidence of scan-model performance; missing volumes are excluded.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .12, 1, 1)); save(fig, Path(output)/"03_volume_presence_definition_check")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parents[1]/"clean_comparison")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--export-zip", type=Path)
    args = parser.parse_args(argv)
    if args.bootstraps < 100:
        parser.error("At least100 bootstrap draws required")
    output = args.output_dir or args.study_dir/"figures"; output.mkdir(parents=True, exist_ok=True)
    table, differences, inventory, sources = collect(args.study_dir, args.bootstraps)
    table.to_csv(output/"auc_results.csv", index=False); differences.to_csv(output/"paired_auc_differences.csv", index=False)
    inventory.to_csv(output/"availability.csv", index=False)
    (output/"figure_audit.json").write_text(json.dumps({"inputs": sources, "bootstraps": args.bootstraps, "seed": 42,
        "confidence_level": .95, "method": "sex-by-outcome stratified paired participant percentile bootstrap",
        "scope": "Conditional on fixed trained models; does not include training-seed variability",
        "pending": int(inventory.Status.eq("missing").sum()) if len(inventory) else 0}, indent=2), encoding="utf-8")
    plot(table, output)
    if args.export_zip:
        args.export_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.export_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in output.iterdir():
                if path.suffix in {".csv", ".json", ".png", ".pdf", ".svg"}:
                    archive.write(path, path.name)
    print(table.to_string(index=False) if len(table) else "No completed matched comparisons yet.")
    print(f"[OUT] {output.resolve()}")
    return table


if __name__ == "__main__":
    main()
