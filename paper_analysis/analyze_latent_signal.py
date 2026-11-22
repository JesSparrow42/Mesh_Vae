#!/usr/bin/env python3
"""Quantify what a trained latent space contains.

t-SNE is used only as a visualization. All statistical tests in this script
operate on standardized original latent vectors (after train-fitted PCA):

* held-out local-neighbour plaque discrimination,
* plaque-label silhouette with a permutation test,
* train-selected unsupervised clusters and held-out plaque association,
* held-out prediction of age and BMI from the latent representation.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .bootstrap_metrics import summarize_model
    from .run_modality_benchmark import load_latent, load_metadata, merge_split
    from .data_utils import bmi_plot_data, clean_participant_id, numeric, read_table, resolve_id_column
except ImportError:
    from bootstrap_metrics import summarize_model
    from run_modality_benchmark import load_latent, load_metadata, merge_split
    from data_utils import bmi_plot_data, clean_participant_id, numeric, read_table, resolve_id_column


def cramers_v(cluster: np.ndarray, target: np.ndarray) -> float:
    table = pd.crosstab(pd.Series(cluster, name="cluster"), pd.Series(target, name="target"))
    observed = table.to_numpy(dtype=float)
    if observed.size == 0 or observed.sum() <= 0:
        return float("nan")
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(axis=0, keepdims=True) / observed.sum()
    valid = expected > 0
    chi_square = float(np.sum(((observed - expected) ** 2)[valid] / expected[valid]))
    denominator = observed.sum() * min(max(observed.shape[0] - 1, 0), max(observed.shape[1] - 1, 0))
    return float(np.sqrt(chi_square / denominator)) if denominator > 0 else float("nan")


def permutation_pvalue(
    statistic,
    features: np.ndarray,
    labels: np.ndarray,
    observed: float,
    permutations: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    exceed = 0
    valid = 0
    for _ in range(int(permutations)):
        shuffled = rng.permutation(labels)
        value = float(statistic(features, shuffled))
        if np.isfinite(value):
            valid += 1
            exceed += int(value >= observed - 1e-15)
    return float((exceed + 1) / (valid + 1)) if valid else float("nan")


def standardize_latent(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], seed: int):
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise ImportError("analyze_latent_signal.py requires scikit-learn.") from error
    train_values = train[columns].to_numpy(dtype=float)
    test_values = test[columns].to_numpy(dtype=float)
    variable = np.var(train_values, axis=0) > 1e-10
    if not variable.any():
        raise ValueError("No varying latent dimensions remain in the training split.")
    scaler = StandardScaler().fit(train_values[:, variable])
    train_scaled = scaler.transform(train_values[:, variable])
    test_scaled = scaler.transform(test_values[:, variable])
    dimensions = min(50, train_scaled.shape[1], len(train_scaled) - 1)
    pca = PCA(n_components=dimensions, random_state=seed).fit(train_scaled)
    return (
        pca.transform(train_scaled),
        pca.transform(test_scaled),
        {
            "input_dimensions": int(len(columns)),
            "variable_dimensions": int(variable.sum()),
            "pca_dimensions": int(dimensions),
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        },
    )


def choose_kmeans(train_values: np.ndarray, min_k: int, max_k: int, seed: int):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rows = []
    models = {}
    for clusters in range(min_k, min(max_k, len(train_values) - 1) + 1):
        model = KMeans(n_clusters=clusters, n_init=20, random_state=seed + clusters)
        labels = model.fit_predict(train_values)
        score = float(
            silhouette_score(
                train_values,
                labels,
                sample_size=min(2000, len(train_values)),
                random_state=seed,
            )
        )
        rows.append({"clusters": clusters, "train_silhouette": score})
        models[clusters] = model
    scores = pd.DataFrame(rows)
    selected = int(scores.sort_values(["train_silhouette", "clusters"], ascending=[False, True]).iloc[0]["clusters"])
    return models[selected], scores, selected


def heldout_knn(
    train_values: np.ndarray,
    train_target: np.ndarray,
    test_values: np.ndarray,
    test: pd.DataFrame,
    neighbors: int,
) -> pd.DataFrame:
    from sklearn.neighbors import KNeighborsClassifier

    neighbors = min(int(neighbors), max(1, len(train_values) - 1))
    model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance", metric="euclidean")
    model.fit(train_values, train_target)
    probability = model.predict_proba(test_values)[:, 1]
    return pd.DataFrame(
        {
            "_id": test["participant_id"].to_numpy(),
            "_target": test["target"].to_numpy(dtype=float),
            "_score": probability,
            "_pred": (probability >= 0.5).astype(float),
        }
    )


def latent_regression(
    train_values: np.ndarray,
    test_values: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, float]:
    from sklearn.linear_model import RidgeCV

    train_mask = train[target].notna().to_numpy()
    test_mask = test[target].notna().to_numpy()
    estimator = RidgeCV(alphas=np.logspace(-4, 4, 33), cv=5)
    estimator.fit(train_values[train_mask], train.loc[train_mask, target].to_numpy(dtype=float))
    prediction = estimator.predict(test_values[test_mask])
    frame = pd.DataFrame(
        {
            "_id": test.loc[test_mask, "participant_id"].to_numpy(),
            "_target": test.loc[test_mask, target].to_numpy(dtype=float),
            "_prediction": prediction,
        }
    )
    return frame, float(estimator.alpha_)


def label_silhouette(values: np.ndarray, target: np.ndarray, seed: int) -> float:
    from sklearn.metrics import silhouette_score

    if len(np.unique(target)) < 2:
        return float("nan")
    return float(
        silhouette_score(
            values,
            target,
            sample_size=min(1500, len(values)),
            random_state=seed,
        )
    )


def tsne_parameters(rows: int, seed: int, perplexity: float) -> dict[str, Any]:
    from sklearn.manifold import TSNE

    parameters = {
        "n_components": 2,
        "perplexity": min(float(perplexity), max(5.0, (rows - 1) / 3.0)),
        "init": "pca", "learning_rate": "auto", "random_state": int(seed), "verbose": 0,
    }
    parameters["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = 2000
    return parameters


def tsne_coordinate_fingerprint(values: np.ndarray, frame: pd.DataFrame, seed: int, perplexity: float) -> str:
    import sklearn

    matrix = np.ascontiguousarray(values, dtype="<f8")
    context = {
        "ordered_participants": list(zip(frame["participant_id"].astype(str), frame["split"].astype(str))),
        "shape": matrix.shape, "parameters": tsne_parameters(len(matrix), seed, perplexity),
        "requested_perplexity": float(perplexity), "sklearn_version": sklearn.__version__,
    }
    digest = hashlib.sha256(json.dumps(context, sort_keys=True).encode())
    digest.update(matrix.tobytes())
    return digest.hexdigest()


def make_tsne_figure(
    train_values: np.ndarray,
    test_values: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_cluster: np.ndarray,
    test_cluster: np.ndarray,
    output_dir: Path,
    seed: int,
    perplexity: float,
    gender: str,
    representation_label: str = "Latent representation",
    bmi_limits: tuple[float, float] = (15.0, 45.0),
    coordinates: np.ndarray | None = None,
) -> dict[str, float]:
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE, trustworthiness
    except ImportError:
        print("[WARN] matplotlib unavailable; skipping t-SNE figure.", flush=True)
        return {}

    values = np.concatenate([train_values, test_values], axis=0)
    frame = pd.concat(
        [train.assign(split="train"), test.assign(split="test")], ignore_index=True
    )
    frame["cluster"] = np.concatenate([train_cluster, test_cluster])
    kwargs = tsne_parameters(len(values), seed, perplexity)
    effective_perplexity = kwargs["perplexity"]
    if coordinates is None:
        coordinates = TSNE(**kwargs).fit_transform(values)
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (len(values), 2) or not np.isfinite(coordinates).all():
        raise ValueError("t-SNE coordinates must contain two finite values per participant.")
    frame["tsne_1"] = coordinates[:, 0]
    frame["tsne_2"] = coordinates[:, 1]
    (output_dir / "tsne_coordinate_cache.json").write_text(json.dumps({
        "fingerprint": tsne_coordinate_fingerprint(values, frame, seed, perplexity),
        "parameters": kwargs,
    }, indent=2), encoding="utf-8")

    trust_n = min(3000, len(values))
    rng = np.random.default_rng(seed)
    trust_indices = rng.choice(len(values), size=trust_n, replace=False)
    trust = float(
        trustworthiness(
            values[trust_indices],
            coordinates[trust_indices],
            n_neighbors=min(15, max(2, trust_n // 20)),
        )
    )

    panels = 4 if "plaque_volume" in frame else 3
    fig, axes_grid = plt.subplots(2, 2, figsize=(12.8, 10.2), constrained_layout=True)
    axes = axes_grid.ravel()
    x = frame["tsne_1"].to_numpy()
    y = frame["tsne_2"].to_numpy()
    plaque = frame["target"].to_numpy(dtype=int)
    for value, label, color in ((0, "No measured plaque", "#0F6F73"), (1, "Plaque present", "#D86D5A")):
        mask = plaque == value
        axes[0].scatter(x[mask], y[mask], s=10, alpha=0.58, color=color, linewidths=0, label=f"{label} (n={mask.sum()})")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("A  Measured plaque presence")

    bmi = frame["bmi"].to_numpy(dtype=float)
    lower, upper = map(float, bmi_limits)
    if not np.isfinite([lower, upper]).all() or lower <= 0 or lower >= upper:
        raise ValueError("BMI color limits must be positive and increasing.")
    missing = np.isnan(bmi)
    invalid = ~missing & (~np.isfinite(bmi) | (bmi <= 0))
    below = np.isfinite(bmi) & (bmi > 0) & (bmi < lower)
    above = np.isfinite(bmi) & (bmi > upper)
    in_range = np.isfinite(bmi) & (bmi >= lower) & (bmi <= upper)
    frame["bmi_plot_status"] = np.select(
        [missing, invalid, below, above],
        ["missing", "invalid/nonpositive", "below color scale", "above color scale"],
        default="within color scale",
    )
    color_points = axes[1].scatter(x[in_range], y[in_range], c=bmi[in_range], cmap="viridis",
                                  vmin=lower, vmax=upper, s=12, alpha=0.78, linewidths=0)
    for mask, label, marker, color in (
        (below, f"Below {lower:g}", "v", "#2463A3"),
        (above, f"Above {upper:g}", "^", "#C3328B"),
        (missing, "Missing", "x", "#8A949A"),
        (invalid, "Invalid/nonpositive", "x", "#D23C30"),
    ):
        if mask.any():
            axes[1].scatter(x[mask], y[mask], s=28, marker=marker, color=color,
                            linewidths=0.8, alpha=0.9, label=f"{label} (n={mask.sum()})", zorder=4)
    if (below | above | missing | invalid).any():
        axes[1].legend(frameon=False, fontsize=8, loc="best")
    extension = "both" if below.any() and above.any() else "min" if below.any() else "max" if above.any() else "neither"
    fig.colorbar(color_points, ax=axes[1], label="BMI (kg/m²)", extend=extension)
    bmi_audit_columns = [column for column in frame if column == "participant_id" or column == "split" or column.startswith("bmi")]
    frame[bmi_audit_columns].to_csv(output_dir / "bmi_audit.csv", index=False)
    bmi_summary = {
        "color_limits_kg_m2": [lower, upper], "rows": len(frame),
        "missing": int(missing.sum()), "invalid": int(invalid.sum()),
        "below_color_scale": int(below.sum()), "above_color_scale": int(above.sum()),
        "source_bmi_min": float(np.min(bmi[np.isfinite(bmi)])) if np.isfinite(bmi).any() else None,
        "source_bmi_max": float(np.max(bmi[np.isfinite(bmi)])) if np.isfinite(bmi).any() else None,
        "data_policy": "Source BMI retained; out-of-scale points use separate markers; no unit conversion inferred.",
    }
    (output_dir / "bmi_audit_summary.json").write_text(json.dumps(bmi_summary, indent=2), encoding="utf-8")
    frame.to_csv(output_dir / "tsne_coordinates.csv", index=False)
    axes[1].set_title("B  Body mass index")

    clusters = frame["cluster"].to_numpy(dtype=int)
    for cluster in sorted(np.unique(clusters)):
        mask = clusters == cluster
        axes[2].scatter(x[mask], y[mask], s=10, alpha=0.62, linewidths=0, label=f"Cluster {cluster}")
    axes[2].legend(frameon=False, fontsize=8, ncol=2)
    axes[2].set_title("C  Groups from training features")
    if panels == 4:
        volume = frame["plaque_volume"].to_numpy(dtype=float)
        valid_volume = np.isfinite(volume) & (volume >= 0)
        axes[3].scatter(x[~valid_volume], y[~valid_volume], s=8, alpha=0.35, color="#D5DDE1", linewidths=0)
        logged_volume = np.log1p(volume[valid_volume])
        maximum = max(float(logged_volume.max()) if len(logged_volume) else 0.0, 1.0)
        points = axes[3].scatter(x[valid_volume], y[valid_volume], c=logged_volume,
                                 cmap="magma", vmin=0.0, vmax=maximum, s=12, alpha=0.78, linewidths=0)
        colorbar = fig.colorbar(points, ax=axes[3], label="Measured volume (reported units; log1p scale)")
        ticks = np.array([0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0])
        ticks = ticks[np.log1p(ticks) <= maximum]
        colorbar.set_ticks(np.log1p(ticks), labels=[f"{tick:g}" for tick in ticks])
        axes[3].set_title("D  Measured plaque volume")
    else:
        axes[3].set_visible(False)
    for axis in axes[:panels]:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.set_aspect("equal", adjustable="box")
        axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.suptitle(
        f"{representation_label}, { {'F': 'women', 'M': 'men', 'ALL': 'pooled cohort'}[gender] }\n"
        f"{len(train)} training participants and {len(test)} test participants",
        fontsize=14,
    )
    fig.savefig(output_dir / "tsne_latent_signal.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "tsne_latent_signal.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"tsne_perplexity": effective_perplexity, "tsne_trustworthiness": trust}


def analyze_gender(
    train: pd.DataFrame,
    test: pd.DataFrame,
    latent_cols: list[str],
    gender: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_group = train.copy() if gender == "ALL" else train[train["gender"] == gender].reset_index(drop=True)
    test_group = test.copy() if gender == "ALL" else test[test["gender"] == gender].reset_index(drop=True)
    if train_group["target"].nunique() != 2 or test_group["target"].nunique() != 2:
        raise ValueError(f"Gender={gender} needs both plaque classes in train and test.")
    train_values, test_values, latent_info = standardize_latent(
        train_group, test_group, latent_cols, args.seed
    )
    train_target = train_group["target"].to_numpy(dtype=int)
    test_target = test_group["target"].to_numpy(dtype=int)

    knn_predictions = heldout_knn(
        train_values,
        train_target,
        test_values,
        test_group,
        args.neighbors,
    )
    knn_intervals = summarize_model(
        "latent_neighbourhood",
        knn_predictions,
        "binary",
        args.bootstraps,
        args.seed + 10,
        args.confidence,
        "auto",
    )
    knn_intervals.insert(0, "gender", gender)
    knn_intervals.to_csv(output_dir / "latent_neighbourhood_intervals.csv", index=False)

    model, k_scores, selected_k = choose_kmeans(
        train_values, args.min_clusters, args.max_clusters, args.seed
    )
    train_cluster = model.predict(train_values)
    test_cluster = model.predict(test_values)
    k_scores.to_csv(output_dir / "cluster_number_selection.csv", index=False)
    cluster_v = cramers_v(test_cluster, test_target)
    cluster_p = permutation_pvalue(
        lambda clusters, target: cramers_v(clusters, target),
        test_cluster,
        test_target,
        cluster_v,
        args.permutations,
        args.seed + 20,
    )
    plaque_silhouette = label_silhouette(test_values, test_target, args.seed)
    silhouette_p = permutation_pvalue(
        lambda values, target: label_silhouette(values, target, args.seed),
        test_values,
        test_target,
        plaque_silhouette,
        args.permutations,
        args.seed + 30,
    )

    cluster_frame = test_group[["participant_id", "target", "age", "bmi", "gender"]].copy()
    cluster_frame["cluster"] = test_cluster
    cluster_frame.to_csv(output_dir / "heldout_cluster_assignments.csv", index=False)
    profiles = (
        cluster_frame.groupby("cluster", as_index=False)
        .agg(
            n=("participant_id", "size"),
            plaque_prevalence=("target", "mean"),
            age_mean=("age", "mean"),
            age_sd=("age", "std"),
            bmi_mean=("bmi", "mean"),
            bmi_sd=("bmi", "std"),
        )
    )
    profiles.to_csv(output_dir / "heldout_cluster_profiles.csv", index=False)

    regression_frames = []
    regression_specs = []
    for offset, target in enumerate(("age", "bmi")):
        predictions, alpha = latent_regression(
            train_values, test_values, train_group, test_group, target
        )
        intervals = summarize_model(
            f"latent_to_{target}",
            predictions,
            "regression",
            args.bootstraps,
            args.seed + 100 + offset,
            args.confidence,
            "auto",
        )
        intervals.insert(0, "gender", gender)
        intervals.insert(2, "target_variable", target)
        intervals["ridge_alpha"] = alpha
        regression_frames.append(intervals)
        regression_specs.append({"target": target, "ridge_alpha": alpha, "n_test": len(predictions)})
    pd.concat(regression_frames, ignore_index=True).to_csv(
        output_dir / "latent_body_measurement_intervals.csv", index=False
    )

    tsne_info = {}
    if not args.skip_tsne:
        tsne_info = make_tsne_figure(
            train_values,
            test_values,
            train_group,
            test_group,
            train_cluster,
            test_cluster,
            output_dir,
            args.seed,
            args.perplexity,
            gender,
            args.representation_label,
        )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gender": gender,
        "n_train": len(train_group),
        "n_test": len(test_group),
        "test_positive": int(test_target.sum()),
        "test_prevalence": float(test_target.mean()),
        "representation_label": args.representation_label,
        "volume_color_column": args.volume_col,
        **latent_info,
        "knn_neighbors": min(args.neighbors, len(train_group) - 1),
        "selected_clusters": selected_k,
        "heldout_plaque_cramers_v": cluster_v,
        "heldout_plaque_cluster_permutation_p": cluster_p,
        "heldout_plaque_label_silhouette": plaque_silhouette,
        "heldout_plaque_silhouette_permutation_p": silhouette_p,
        "permutations": args.permutations,
        "body_measurement_models": regression_specs,
        **tsne_info,
        "interpretation_rule": (
            "Treat t-SNE as visualization only. Claim plaque clustering only if the held-out original-latent "
            "association survives the pre-specified permutation test."
        ),
    }
    (output_dir / "latent_signal_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    rows = [
        {"metric": "plaque_label_silhouette", "estimate": plaque_silhouette, "permutation_p": silhouette_p},
        {"metric": "cluster_plaque_cramers_v", "estimate": cluster_v, "permutation_p": cluster_p},
        {"metric": "selected_cluster_count", "estimate": selected_k, "permutation_p": np.nan},
    ]
    if "tsne_trustworthiness" in tsne_info:
        rows.append(
            {"metric": "tsne_trustworthiness", "estimate": tsne_info["tsne_trustworthiness"], "permutation_p": np.nan}
        )
    pd.DataFrame(rows).to_csv(output_dir / "latent_signal_tests.csv", index=False)
    print(
        f"[LATENT] gender={gender} k={selected_k} cluster V={cluster_v:.4f} p={cluster_p:.4g} "
        f"plaque silhouette={plaque_silhouette:.4f} p={silhouette_p:.4g}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-latent", required=True, type=Path)
    parser.add_argument("--test-latent", required=True, type=Path)
    parser.add_argument("--train-tsv", required=True, type=Path)
    parser.add_argument("--test-tsv", required=True, type=Path)
    parser.add_argument("--target-col", default="anyplaq")
    parser.add_argument("--volume-col", help="Optional measured volume column for an additional log1p color panel.")
    parser.add_argument("--representation-label", default="Latent representation",
                        help="Describe the checkpoint inputs, e.g. 'Scan + age + weight latent representation'.")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--age-col", default=None)
    parser.add_argument("--bmi-col", default=None)
    parser.add_argument("--gender-col", default=None)
    parser.add_argument("--genders", nargs="+", choices=("F", "M", "ALL"), default=("F", "M"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--neighbors", type=int, default=25)
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--seed", type=int, default=20260920)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_latent, train_columns = load_latent(args.train_latent, "train", args.id_col)
    test_latent, test_columns = load_latent(args.test_latent, "test", args.id_col)
    common_columns = sorted(set(train_columns) & set(test_columns))
    if not common_columns:
        raise ValueError("No common latent columns between train and test files.")
    train_metadata, _ = load_metadata(
        args.train_tsv,
        args.target_col,
        args.id_col,
        args.age_col,
        args.bmi_col,
        args.gender_col,
    )
    test_metadata, _ = load_metadata(
        args.test_tsv,
        args.target_col,
        args.id_col,
        args.age_col,
        args.bmi_col,
        args.gender_col,
    )
    enriched_bmi = []
    for metadata, path in ((train_metadata, args.train_tsv), (test_metadata, args.test_tsv)):
        source = read_table(path)
        audit = bmi_plot_data(source, args.bmi_col)
        audit["participant_id"] = source[resolve_id_column(source, args.id_col)].map(clean_participant_id)
        enriched_bmi.append(metadata.merge(audit.drop(columns="bmi"), on="participant_id", how="left", validate="one_to_one"))
    train_metadata, test_metadata = enriched_bmi
    if args.volume_col:
        enriched = []
        for metadata, path in ((train_metadata, args.train_tsv), (test_metadata, args.test_tsv)):
            source = read_table(path)
            if args.volume_col not in source:
                raise KeyError(f"Missing volume column {args.volume_col!r} in {path}")
            volume = numeric(source[args.volume_col])
            colors = pd.DataFrame({
                "participant_id": source[resolve_id_column(source, args.id_col)].map(clean_participant_id),
                "plaque_volume": volume.where(np.isfinite(volume) & (volume >= 0)),
            }).dropna(subset=["participant_id"])
            enriched.append(metadata.merge(colors, on="participant_id", how="left", validate="one_to_one"))
        train_metadata, test_metadata = enriched
    train, train_flow = merge_split(
        train_metadata,
        train_latent[["participant_id", *common_columns]],
        common_columns,
        "train",
    )
    test, test_flow = merge_split(
        test_metadata,
        test_latent[["participant_id", *common_columns]],
        common_columns,
        "test",
    )
    pd.DataFrame([train_flow, test_flow]).to_csv(output_dir / "cohort_flow.csv", index=False)
    for gender in args.genders:
        analyze_gender(
            train,
            test,
            common_columns,
            gender,
            output_dir / f"gender_{gender}",
            args,
        )
    print(f"[OUT] {output_dir}", flush=True)


if __name__ == "__main__":
    main()

