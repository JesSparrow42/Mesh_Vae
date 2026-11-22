"""Lightweight 15-design comparison components for a continuous outcome."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .comparison import (
    BenchmarkPointDataset,
    ConvFeatureVAEClassifier,
    Experiment,
    FeatureVAEClassifier,
    FlexiblePointNet,
    LinearClassifier,
    MetadataSpec,
    PointVAEClassifier,
    TabularConvClassifier,
    age_feature_matrix,
    apply_standardizer,
    experiments,
    fit_standardizer,
    load_points_from_npz,
    make_device,
    point_summary_features,
    read_condition_tsv,
)
from .conditioning import _pick_col, _to_numeric_series, clean_stregkode
from .train_eval import regression_metrics


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_rank = pd.Series(y_true).rank(method="average").to_numpy()
    p_rank = pd.Series(y_pred).rank(method="average").to_numpy()
    if len(y_rank) < 2 or np.std(y_rank) <= 0.0 or np.std(p_rank) <= 0.0:
        return float("nan")
    return float(np.corrcoef(y_rank, p_rank)[0, 1])


def build_regression_metadata(
    path: str,
    args,
    spec: MetadataSpec | None = None,
    target_mean: float | None = None,
    target_std: float | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, MetadataSpec, float, float]:
    df = read_condition_tsv(path, sep=getattr(args, "cond_sep", None))
    target_col = str(getattr(args, "target_col", "glycohemoglobin_hba1c"))
    if target_col not in df.columns:
        found = _pick_col(df, [target_col], fallback_contains="hba1c")
        if found is None:
            raise KeyError(f"Could not find {target_col!r}. Available: {list(df.columns)[:100]}")
        target_col = str(found)
        args.target_col = target_col

    if fit or spec is None:
        st_col = getattr(args, "cond_stregkode_col", None) or _pick_col(
            df, ["stregkode", "participant_id", "participantid", "id"], fallback_contains="stregkode"
        )
        if st_col is None:
            st_col = _pick_col(df, ["participant_id", "participantid"], fallback_contains="participant")
        gender_col = getattr(args, "cond_gender_col", None) or _pick_col(
            df, ["sex", "gender", "sexMF"], fallback_contains="sex"
        )
        agecut_col = getattr(args, "cond_agecut_col", None) or _pick_col(
            df, ["agecut", "age_cut", "age_bin", "agebin", "age_group", "agegroup"], fallback_contains="agecut"
        )
        age_col = getattr(args, "cond_age_col", None) or _pick_col(df, ["alder", "age"], fallback_contains="ald")
        if age_col is None:
            age_col = _pick_col(df, ["age"], fallback_contains="age")
    else:
        st_col = spec.st_col
        gender_col = spec.gender_col
        agecut_col = spec.agecut_col
        age_col = spec.age_col
    if st_col is None:
        raise KeyError(f"Could not find participant id in {path}.")

    use_cols = [st_col, target_col]
    for column in [gender_col, agecut_col, age_col]:
        if column is not None and column not in use_cols:
            use_cols.append(column)
    out = df[use_cols].copy()
    out["code"] = out[st_col].apply(clean_stregkode)
    out["target_raw"] = _to_numeric_series(out[target_col])
    before = len(out)
    out = out.dropna(subset=["code", "target_raw"]).copy()
    print(f"[TARGET] {path}: kept {len(out)}/{before} rows with finite {target_col}.", flush=True)

    if gender_col is not None:
        gender = out[gender_col].astype(str).str.strip().str.upper().str.strip('"').str.strip("'")
        gender = gender.replace({"MALE": "M", "FEMALE": "F"})
        out["gender_clean"] = gender
        if getattr(args, "only_gender", None):
            out = out[out["gender_clean"] == str(args.only_gender).upper()].copy()
    elif getattr(args, "only_gender", None):
        raise KeyError("--only_gender was set, but no gender column was found.")

    temp_spec = spec
    age_mat, _, age_mode, resolved_age_col, resolved_agecut_col, age_labels, age_mean, age_std = age_feature_matrix(
        out, args, temp_spec, fit
    )
    raw = out["target_raw"].to_numpy(np.float32)

    out["_position"] = np.arange(len(out), dtype=np.int64)
    out = out.drop_duplicates(subset=["code"]).reset_index(drop=True)
    keep = out["_position"].to_numpy(np.int64)
    age_mat = age_mat[keep]
    raw = raw[keep]
    out = out.drop(columns=["_position"])

    if fit:
        target_mean = float(np.mean(raw))
        target_std = float(np.std(raw))
        if not np.isfinite(target_std) or target_std < 1e-6:
            target_std = 1.0
    if target_mean is None or target_std is None or target_std <= 0.0:
        raise ValueError("Training target normalization is missing or invalid.")
    y = ((raw - target_mean) / target_std).astype(np.float32)
    new_spec = MetadataSpec(
        st_col=str(st_col),
        gender_col=None if gender_col is None else str(gender_col),
        age_col=None if resolved_age_col is None else str(resolved_age_col),
        agecut_col=None if resolved_agecut_col is None else str(resolved_agecut_col),
        age_mode=str(age_mode),
        age_labels=list(age_labels),
        age_mean=float(age_mean),
        age_std=float(age_std),
        target_cols=[target_col],
        clinical_cols=[],
        clinical_mean=[],
        clinical_std=[],
    )
    print(
        f"[TARGET] usable={len(out)} mean={target_mean:.6g} std={target_std:.6g}",
        flush=True,
    )
    return out, age_mat.astype(np.float32), y, new_spec, float(target_mean), float(target_std)


def build_arrays(args, fit: bool, spec=None, target_mean=None, target_std=None, scan_spec=None):
    meta, age, y, spec, target_mean, target_std = build_regression_metadata(
        args.cond_tsv,
        args,
        spec=spec,
        target_mean=target_mean,
        target_std=target_std,
        fit=fit,
    )
    points, age, y, names, scan_features, scan_spec = load_points_from_npz(
        args, meta, age, y, scan_spec=scan_spec, fit=fit
    )
    return points, age, y, names, scan_features, spec, target_mean, target_std, scan_spec


def quantile_split(y: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    try:
        q = min(10, max(2, len(y) // 20))
        labels = np.asarray(pd.qcut(y, q=q, labels=False, duplicates="drop"), dtype=np.int64)
    except Exception:
        labels = np.zeros(len(y), dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n_val = max(1, min(len(idx) - 1, int(round(len(idx) * val_ratio)))) if len(idx) > 1 else 0
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def feature_loaders(X_train, X_test, y_train, y_test, args):
    train_idx, val_idx = quantile_split(y_train, args.val_ratio, args.seed)
    mean, std = fit_standardizer(X_train[train_idx])
    train_x = apply_standardizer(X_train, mean, std)
    test_x = apply_standardizer(X_test, mean, std)

    def loader(x, y, idx, shuffle, batch):
        ds = TensorDataset(
            torch.from_numpy(x[idx]),
            torch.zeros((len(idx), 0), dtype=torch.float32),
            torch.from_numpy(y[idx].astype(np.float32)),
        )
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=args.num_threads)

    return (
        loader(train_x, y_train, train_idx, True, args.batch_size_tabular),
        loader(train_x, y_train, val_idx, False, args.batch_size_tabular),
        loader(test_x, y_test, np.arange(len(y_test)), False, args.batch_size_tabular),
        int(train_x.shape[1]),
    )


def point_loaders(points_train, points_test, scan_train, scan_test, meta_train, meta_test, y_train, y_test, exp, args):
    train_idx, val_idx = quantile_split(y_train, args.val_ratio, args.seed)
    common = {"n_points": args.n_points, "include_labels": bool(exp.point_label)}
    train_ds = BenchmarkPointDataset(
        points_train[train_idx], meta_train[train_idx], y_train[train_idx],
        scan_features=scan_train[train_idx], deterministic=False, seed=args.seed, **common,
    )
    val_ds = BenchmarkPointDataset(
        points_train[val_idx], meta_train[val_idx], y_train[val_idx],
        scan_features=scan_train[val_idx], deterministic=True, seed=args.seed + 1000, **common,
    )
    test_ds = BenchmarkPointDataset(
        points_test, meta_test, y_test,
        scan_features=scan_test, deterministic=True, seed=args.seed + 2000, **common,
    )
    return (
        DataLoader(train_ds, batch_size=args.batch_size_point, shuffle=True, num_workers=args.num_threads),
        DataLoader(val_ds, batch_size=args.batch_size_point, shuffle=False, num_workers=args.num_threads),
        DataLoader(test_ds, batch_size=args.batch_size_point, shuffle=False, num_workers=args.num_threads),
    )


def predict(model, loader, device, is_vae):
    model.eval()
    targets, predictions = [], []
    with torch.no_grad():
        for x, meta, y in loader:
            output = model(x.to(device), meta.to(device))
            pred = output[0] if is_vae else output
            targets.append(y.numpy())
            predictions.append(pred.detach().cpu().numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def metrics_raw(y_norm, pred_norm, mean, std):
    y = y_norm * std + mean
    pred = pred_norm * std + mean
    result = regression_metrics(y, pred)
    result["spearman"] = _spearman(y, pred)
    return result


def train_model(model, train_loader, val_loader, args, device, is_vae, target_mean, target_std):
    model.to(device)
    objective = nn.SmoothL1Loss()
    reconstruction = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_score = -float("inf")
    best: dict[str, Any] = {}
    patience = int(args.patience)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, meta, y in train_loader:
            x, meta, y = x.to(device), meta.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, meta)
            if is_vae:
                pred, recon, mu, logvar, recon_target = output
                reg_loss = objective(pred, y)
                recon_loss = reconstruction(recon, recon_target)
                kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
                loss = reg_loss + args.vae_recon_weight * recon_loss + args.vae_beta * kl
            else:
                loss = objective(output, y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_y, val_pred = predict(model, val_loader, device, is_vae)
        val_metrics = metrics_raw(val_y, val_pred, target_mean, target_std)
        score = float(val_metrics["r2"])
        if not np.isfinite(score):
            score = -float(val_metrics["rmse"])
        improved = score > best_score
        if improved:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best = {"epoch": epoch, "val": val_metrics}
            patience = int(args.patience)
        else:
            patience -= 1
        if epoch == 1 or epoch % args.print_every == 0 or improved:
            print(
                f"    epoch={epoch:03d} loss={np.mean(losses) if losses else np.nan:.4f} "
                f"val_r2={val_metrics['r2']:.4f} val_mae={val_metrics['mae']:.4f}"
                f"{' *' if improved else ''}",
                flush=True,
            )
        if patience <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def run_one(exp: Experiment, data, args, device, out_dir):
    print(f"\n========== {exp.slug} (regression) ==========", flush=True)
    if exp.point_label and not exp.use_pointcloud:
        return {
            "Experiment": exp.slug, "Model": "VAE", "Age": "Yes", "Pointcloud": "No",
            "Modification": exp.modification, "Status": "not_applicable", "Architecture": "no points to label",
        }
    meta_train = data["age_train"] if exp.use_age else np.zeros((len(data["y_train"]), 0), np.float32)
    meta_test = data["age_test"] if exp.use_age else np.zeros((len(data["y_test"]), 0), np.float32)
    is_vae = exp.family.startswith("vae")

    if exp.family in {"pointnet", "vae_point"} and exp.use_pointcloud:
        train_loader, val_loader, test_loader = point_loaders(
            data["points_train"], data["points_test"], data["scan_train"], data["scan_test"],
            meta_train, meta_test, data["y_train"], data["y_test"], exp, args,
        )
        channels = 3 + (6 if exp.point_label else 0) + data["scan_train"].shape[1]
        if exp.family == "pointnet":
            model = FlexiblePointNet(channels, meta_train.shape[1], hidden=args.hidden, dropout=args.dropout)
        else:
            model = PointVAEClassifier(channels, meta_train.shape[1], args.latent_dim, args.hidden, args.dropout)
    else:
        train_features, test_features = [], []
        if exp.use_pointcloud:
            train_features.extend([data["summary_train"], data["scan_train"]])
            test_features.extend([data["summary_test"], data["scan_test"]])
        if exp.use_age:
            train_features.append(data["age_train"])
            test_features.append(data["age_test"])
        X_train = np.concatenate(train_features, axis=1).astype(np.float32)
        X_test = np.concatenate(test_features, axis=1).astype(np.float32)
        train_loader, val_loader, test_loader, input_dim = feature_loaders(
            X_train, X_test, data["y_train"], data["y_test"], args
        )
        if exp.family == "logistic":
            model = LinearClassifier(input_dim)
        elif exp.family == "tabular_conv":
            model = TabularConvClassifier(input_dim, hidden=max(16, args.hidden // 4), dropout=args.dropout)
        elif exp.family == "vae_feature":
            cls = ConvFeatureVAEClassifier if exp.conv else FeatureVAEClassifier
            model = cls(input_dim, latent_dim=args.latent_dim, hidden=max(32, args.hidden // 2), dropout=args.dropout)
        else:
            raise ValueError(f"Unsupported regression family {exp.family}")

    model, best = train_model(
        model, train_loader, val_loader, args, device, is_vae,
        data["target_mean"], data["target_std"],
    )
    y_test, pred_test = predict(model, test_loader, device, is_vae)
    test = metrics_raw(y_test, pred_test, data["target_mean"], data["target_std"])
    checkpoint = out_dir / f"{exp.slug}.pt"
    torch.save({"experiment": exp.__dict__, "state_dict": model.state_dict(), "test": test, "best": best}, checkpoint)
    row = {
        "Experiment": exp.slug,
        "Model": "Linear regression" if exp.family == "logistic" else exp.display_model,
        "Age": "Yes" if exp.use_age else "No",
        "Pointcloud": "Yes" if exp.use_pointcloud else "No",
        "Modification": exp.modification,
        "Status": "completed",
        "Architecture": model.__class__.__name__,
        "R2": test["r2"], "MAE": test["mae"], "RMSE": test["rmse"],
        "Pearson": test["pearson"], "Spearman": test["spearman"],
        "BestEpoch": best.get("epoch", np.nan),
        "NTrain": len(data["y_train"]), "NTest": len(data["y_test"]),
        "Checkpoint": str(checkpoint),
    }
    print(
        f"[RESULT] r2={row['R2']:.4f} mae={row['MAE']:.4f} rmse={row['RMSE']:.4f} "
        f"pearson={row['Pearson']:.4f}",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--train_cond_tsv", required=True)
    parser.add_argument("--test_cond_tsv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_col", default="glycohemoglobin_hba1c")
    parser.add_argument("--only_gender", choices=["M", "F"], default="F")
    parser.add_argument("--npz_name", default="simplified_padded_tensors.npz")
    parser.add_argument("--cond_sep", default=None)
    parser.add_argument("--cond_stregkode_col", default=None)
    parser.add_argument("--cond_age_col", default=None)
    parser.add_argument("--cond_agecut_col", default=None)
    parser.add_argument("--cond_gender_col", default=None)
    parser.add_argument("--age_source", choices=["auto", "agecut", "age"], default="auto")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_points", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size_point", type=int, default=8)
    parser.add_argument("--batch_size_tabular", type=int, default=128)
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--print_every", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--vae_beta", type=float, default=0.01)
    parser.add_argument("--vae_recon_weight", type=float, default=0.10)
    parser.add_argument("--use_scan_geometry_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scan_geometry_features_file", default="scan_geometry_features.npz")
    parser.add_argument("--require_scan_geometry_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restore_metric_scale", action=argparse.BooleanOptionalAction, default=True)
    available = [exp.slug for exp in experiments()]
    parser.add_argument("--experiments", nargs="+", choices=available, default=available)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = make_device(args.gpu_ids)
    args.cond_tsv = args.train_cond_tsv
    train = build_arrays(args, fit=True)
    test_args = copy.copy(args)
    test_args.cond_tsv = args.test_cond_tsv
    test = build_arrays(
        test_args, fit=False, spec=train[5], target_mean=train[6], target_std=train[7], scan_spec=train[8]
    )
    data = {
        "points_train": train[0], "age_train": train[1], "y_train": train[2], "scan_train": train[4],
        "points_test": test[0], "age_test": test[1], "y_test": test[2], "scan_test": test[4],
        "target_mean": train[6], "target_std": train[7],
        "summary_train": point_summary_features(train[0]),
        "summary_test": point_summary_features(test[0]),
    }
    print(
        f"[DATA] train={len(train[2])} test={len(test[2])} target={args.target_col} "
        f"mean={train[6]:.4f} std={train[7]:.4f}",
        flush=True,
    )

    partial = out_dir / "regression_comparison_results_partial.csv"
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and partial.exists():
        existing = {str(row["Experiment"]): row for row in pd.read_csv(partial).to_dict("records")}
    rows = []
    by_key = {exp.slug: exp for exp in experiments()}
    for key in args.experiments:
        if key in existing and str(existing[key].get("Status")) in {"completed", "not_applicable"}:
            row = existing[key]
            print(f"[RESUME] {key}", flush=True)
        else:
            row = run_one(by_key[key], data, args, device, out_dir)
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial, index=False)
    final = out_dir / "regression_comparison_results.csv"
    pd.DataFrame(rows).to_csv(final, index=False)
    print(f"[OUT] {final}", flush=True)


if __name__ == "__main__":
    main()
