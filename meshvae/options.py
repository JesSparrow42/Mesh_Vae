from __future__ import annotations

import json
from pathlib import Path

from .conditioning import _build_age_labels, cond_dim_from_opt


class BaseOptions:
    default_phase = "train"
    is_train = False
    is_test = False

    def __init__(self):
        import argparse

        self.project_root = "/data/projects/3D_scans/ML/MQVAE"
        self.bool_action = argparse.BooleanOptionalAction

        parser = argparse.ArgumentParser()
        self.initialize(parser)
        self.parser = parser

    def initialize(self, parser):
        self.add_common_options(parser)
        self.add_geometry_options(parser)
        self.add_model_options(parser)
        self.add_conditioning_options(parser)
        self.add_age_search_options(parser)
        self.add_baseline_options(parser)
        self.add_phase_options(parser)

    def add_common_options(self, parser):
        group = parser.add_argument_group("Common options")

        group.add_argument("--dataroot", default=self.project_root)
        group.add_argument("--name", required=True)
        group.add_argument("--checkpoints_dir", default=f"{self.project_root}/checkpoints")
        group.add_argument("--outroot", type=str, default=".", help="Output root for test reconstructions.")

        group.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated GPU IDs; -1 selects CPU.")
        group.add_argument(
            "--phase",
            type=str,
            default=self.default_phase,
            choices=["preprocess", "train", "test", "sample", "search_age_bins", "train_baseline"],
        )

        group.add_argument("--batch_size", type=int, default=1)
        group.add_argument("--num_threads", type=int, default=0)
        group.add_argument("--max_dataset_size", type=int, default=99999)
        group.add_argument("--seed", type=int, default=42)
        group.add_argument(
            "--deterministic",
            action=self.bool_action,
            default=False,
            help="Prefer deterministic CUDA kernels for paired/repeated experiments.",
        )

        group.add_argument("--dump_padded_obj", action=self.bool_action, default=False)
        group.add_argument("--dump_padded_max", type=int, default=8)
        group.add_argument("--debug_kernels", action=self.bool_action, default=False)

    def add_geometry_options(self, parser):
        group = parser.add_argument_group("Geometry / data options")

        group.add_argument(
            "--feature_source",
            type=str,
            default="scan",
            choices=["scan", "ldl", "hba1c", "hbalc", "ldl_hba1c", "ldl_hbalc", "both"],
            help=(
                "Input representation. 'scan' uses the body scan as before. "
                "'ldl', 'hba1c', or 'ldl_hba1c' replace the scan with a tiny "
                "synthetic point cloud carrying those numeric TSV features. "
                "'hbalc' is accepted as an old alias for 'hba1c'. "
                "Use this as a target/model sanity check, not as the final scan model."
            ),
        )
        group.add_argument(
            "--clinical_points",
            type=int,
            default=32,
            help="Number of points used when --feature_source is ldl/hba1c/ldl_hba1c.",
        )
        group.add_argument(
            "--clinical_feature_cols",
            nargs="*",
            default=None,
            help=(
                "Optional explicit TSV column names for clinical feature mode. "
                "Normally not needed: ldl -> ldl_cholesterol; hba1c -> glycohemoglobin_hba1c."
            ),
        )

        group.add_argument(
            "--target_tris",
            type=int,
            default=20000,
            help="Target number of triangles per simplified mesh during offline preprocessing.",
        )
        group.add_argument(
            "--n_points",
            type=int,
            default=0,
            help="If > 0, also precompute point clouds with this many points per mesh.",
        )
        group.add_argument(
            "--point_sample_method",
            type=str,
            default="poisson",
            choices=["poisson", "uniform"],
            help="Point sampling method used during preprocessing when --n_points > 0.",
        )
        group.add_argument(
            "--preprocess_roots",
            nargs="+",
            default=None,
            help=(
                "One or more OBJ files/folders to include during --phase preprocess. "
                "Each folder is searched recursively. Default: --dataroot."
            ),
        )
        group.add_argument(
            "--preprocess_existing_npz",
            nargs="*",
            default=None,
            help=(
                "Existing simplified_padded_tensors.npz files to merge into the new combined NPZ. "
                "Useful for keeping the old preprocessed data while adding new OBJ folders."
            ),
        )
        group.add_argument(
            "--preprocess_dedupe_by_id",
            action=self.bool_action,
            default=True,
            help="During preprocessing, keep only the first scan for each cleaned RIGDET/GOEDET id.",
        )
        group.add_argument(
            "--preprocess_assume_simplified",
            action=self.bool_action,
            default=False,
            help=(
                "For --phase preprocess, treat input OBJ files as already cleaned/simplified. "
                "This skips normalization, decimation, and rewriting simplified OBJ files, "
                "and only packs tensors plus optional sampled points into the NPZ."
            ),
        )
        group.add_argument(
            "--use_preprocessed",
            action=self.bool_action,
            default=True,
            help="Load meshes from simplified_padded_tensors.npz instead of raw OBJs.",
        )
        group.add_argument(
            "--use_pointcloud",
            action=self.bool_action,
            default=True,
            help="Use precomputed point clouds as input features instead of mesh vertices.",
        )
        group.add_argument(
            "--use_scan_geometry_features",
            action=self.bool_action,
            default=False,
            help=(
                "Use physical-size features recovered from the original OBJ files. "
                "The sidecar is created by build_scan_geometry_features.py."
            ),
        )
        group.add_argument(
            "--scan_geometry_features_file",
            type=str,
            default="scan_geometry_features.npz",
            help="Geometry sidecar filename, relative to --dataroot unless absolute.",
        )
        group.add_argument(
            "--require_scan_geometry_features",
            action=self.bool_action,
            default=False,
            help="Fail if any selected scan is missing physical geometry features.",
        )
        group.add_argument(
            "--restore_metric_scale",
            action=self.bool_action,
            default=True,
            help=(
                "Restore between-person scale in legacy unit-sphere point clouds using "
                "the OBJ geometry sidecar, then divide all scans by one training-set scale."
            ),
        )
        group.add_argument(
            "--use_point_labels",
            action=self.bool_action,
            default=False,
            help=(
                "Append pseudo body-region one-hot labels to each point/vertex. "
                "This adds --point_label_regions channels after xyz."
            ),
        )
        group.add_argument(
            "--point_label_regions",
            type=int,
            default=6,
            help="Number of pseudo body-region bins used by --use_point_labels.",
        )

        group.add_argument(
            "--cohort_tsv",
            type=str,
            default=None,
            help="Path to cohort TSV containing a column with OBJ paths, e.g. obj_path.",
        )
        group.add_argument(
            "--cohort_obj_col",
            type=str,
            default="obj_path",
            help="Column name in cohort TSV that contains full paths to .obj files.",
        )
        group.add_argument(
            "--subset_group",
            type=str,
            default=None,
            help=(
                "Shortcut group name like 'F_18-40'. Looks for "
                "/data/projects/3D_scans/sorted_data/<group>/cohort_<group>.tsv."
            ),
        )
        group.add_argument(
            "--require_existing",
            action=self.bool_action,
            default=False,
            help="Drop cohort rows whose obj_path does not exist on disk.",
        )
        group.add_argument(
            "--chamfer_npoints",
            type=int,
            default=4096,
            help="Subsample size for Chamfer loss when --use_pointcloud is set.",
        )

    def add_model_options(self, parser):
        group = parser.add_argument_group("Model options")

        group.add_argument("--arch", type=str, default="meshvae")
        group.add_argument("--pool_mode", type=str, default="skip")

        group.add_argument("--input_nc", type=int, default=3)
        group.add_argument("--ncf", nargs="+", type=int, default=[64, 128, 256, 256])
        group.add_argument("--latent_dim", type=int, default=256)
        group.add_argument("--ninput_edges", type=int, default=25000)
        group.add_argument(
            "--encoder_arch",
            type=str,
            default="meshconv",
            choices=["pointnet", "meshconv"],
            help="pointnet uses permutation-invariant point MLPs; meshconv uses local edge convolution.",
        )
        group.add_argument(
            "--encoder_max_points",
            type=int,
            default=0,
            help="Deterministic encoder-only point cap (0 keeps every input point). The decoder still reconstructs all points.",
        )
        group.add_argument(
            "--point_label_pooling",
            action=self.bool_action,
            default=False,
            help="Pool latent features separately within each pseudo anatomical point-label region.",
        )

        group.add_argument(
            "--latent_prior",
            type=str,
            choices=["gaussian", "boson"],
            default="gaussian",
        )
        group.add_argument(
            "--boson_params",
            type=str,
            default=None,
            help="JSON parameters passed through the MeshVAE adapter to Sparrow's Boson sampler.",
        )
        group.add_argument(
            "--boson_sampler_path",
            type=str,
            default=None,
            help=(
                "Optional path to the Non_Linear_Photonic_Sampler repository. "
                "The installed package or checked-in Sparrow source is used when omitted."
            ),
        )
        group.add_argument(
            "--prior_match_mode",
            type=str,
            choices=["auto", "kl", "mmd"],
            default="auto",
            help="Latent regularizer. auto uses KL for Gaussian and MMD for the implicit Boson prior.",
        )
        group.add_argument("--mmd_buffer_size", type=int, default=64)
        group.add_argument("--mmd_prior_samples", type=int, default=64)

        group.add_argument(
            "--mesh_use_knn",
            action=self.bool_action,
            default=True,
            help="Augment face-adjacency edges with kNN edges in the mesh encoder.",
        )
        group.add_argument("--mesh_knn_k", type=int, default=8)
        group.add_argument("--mesh_knn_max_points", type=int, default=4096)

    def add_conditioning_options(self, parser):
        group = parser.add_argument_group("Conditioning options")

        group.add_argument(
            "--use_cond",
            action=self.bool_action,
            default=True,
            help="Train conditional VAE using metadata from a cohort TSV.",
        )
        group.add_argument(
            "--cond_tsv",
            type=str,
            default=None,
            help="TSV with stregkode + age + sex/gender columns.",
        )
        group.add_argument("--cond_sep", type=str, default=None)
        group.add_argument("--cond_stregkode_col", type=str, default=None)
        group.add_argument("--cond_age_col", type=str, default=None)
        group.add_argument("--cond_weight", action=self.bool_action, default=False,
                           help="Include training-standardized body weight in demographic conditioning.")
        group.add_argument("--weight_col", default=None, help="Body weight column in kg; otherwise resolve known aliases.")
        group.add_argument(
            "--cond_agecut_col",
            type=str,
            default=None,
            help="Column containing pre-binned age labels. Default: auto-detect agecut when present.",
        )
        group.add_argument(
            "--cond_age_source",
            type=str,
            default="auto",
            choices=["auto", "agecut", "age"],
            help=(
                "Which age representation to use in the condition vector. "
                "auto uses an agecut column when present; otherwise numeric age + --age_bins."
            ),
        )
        group.add_argument("--cond_gender_col", type=str, default=None)
        group.add_argument("--cond_embed_dim", type=int, default=64)
        group.add_argument("--cond_scale", type=float, default=1.0)

        group.add_argument(
            "--keep_missing_cond",
            action=self.bool_action,
            default=False,
            help="Keep samples missing metadata. Default: drop them.",
        )

        group.add_argument(
            "--only_gender",
            type=str,
            default=None,
            choices=["M", "F"],
            help="Optional: keep only one gender throughout dataset construction.",
        )

        group.add_argument(
            "--cond_age_mode",
            type=str,
            default="hard",
            choices=["hard", "soft", "rbf"],
            help="'hard' = one-hot, 'soft' = linear interpolation, 'rbf' = RBF over age-bin centers.",
        )
        group.add_argument(
            "--age_bins",
            nargs="+",
            type=float,
            default=None,
            help="Numeric age-bin edges. Not needed when the TSV has agecut and --cond_age_source is auto/agecut.",
        )
        group.add_argument("--age_rbf_gamma", type=float, default=0.015)

        group.add_argument(
            "--encoder_cond_mode",
            type=str,
            default="none",
            choices=["none", "demo"],
            help=(
                "'none' = scan-only encoder. "
                "'demo' = pass only gender+age to encoder. "
                "Atherosclerosis/plaque labels are never passed to the encoder."
            ),
        )
        group.add_argument(
            "--encoder_use_cond",
            action=self.bool_action,
            default=False,
            help="Deprecated and unsafe. Use --encoder_cond_mode demo instead.",
        )
        group.add_argument(
            "--use_cond_prior",
            action=self.bool_action,
            default=False,
            help="Use p(z|cond).",
        )

        # Atherosclerosis conditioning
        group.add_argument(
            "--cond_athero",
            action=self.bool_action,
            default=True,
            help="Include atherosclerosis flags in condition vector.",
        )
        group.add_argument(
            "--cond_athero_cols",
            nargs=4,
            default=["ho_carotis", "ve_carotis", "ho_fem", "ve_fem"],
            help="4 column names. Order: ho_carotis ve_carotis ho_fem ve_fem.",
        )
        group.add_argument(
            "--cond_athero_mode",
            type=str,
            default="4bit",
            choices=["any", "4bit"],
            help="'any' = one sick/healthy dim. '4bit' = one dim per location.",
        )
        group.add_argument(
            "--target_missing_policy",
            type=str,
            default="drop",
            choices=["drop", "zero", "error"],
            help=(
                "How to handle participants with incomplete plaque measurements and no observed positive site. "
                "The rigorous default is drop; zero reproduces the old behavior."
            ),
        )
        group.add_argument(
            "--balance_athero",
            action=self.bool_action,
            default=True,
            help="Oversample atherosclerosis-positive examples during training.",
        )

        # Plaque conditioning
        group.add_argument(
            "--cond_plaque",
            action=self.bool_action,
            default=False,
            help="Include plaque volume values in condition vector.",
        )
        group.add_argument(
            "--cond_plaque_cols",
            nargs=4,
            default=[
                "plaque_volume_right_carotid_corrected",
                "plaque_volume_left_carotid_corrected",
                "plaque_volume_right_femoral_corrected",
                "plaque_volume_left_femoral_corrected",
            ],
        )
        group.add_argument(
            "--cond_plaque_norm",
            type=str,
            default="log1p_z",
            choices=["none", "log1p_z"],
        )

        # Auxiliary atherosclerosis head
        group.add_argument(
            "--aux_predict_athero",
            action=self.bool_action,
            default=True,
            help="Add supervised classifier head on latent mu(X).",
        )
        group.add_argument(
            "--aux_athero_mode",
            type=str,
            default="any",
            choices=["any", "4bit"],
        )
        group.add_argument("--aux_athero_weight", type=float, default=10.0)
        group.add_argument("--aux_athero_pos_weight", type=float, default=2.0)
        group.add_argument("--aux_athero_dropout", type=float, default=0.15)

        # Generic auxiliary binary head
        group.add_argument(
            "--aux_predict_binary",
            action=self.bool_action,
            default=False,
            help="Predict one arbitrary binary TSV column from latent mu(X).",
        )
        group.add_argument(
            "--aux_binary_col",
            type=str,
            default="ccta_anycad",
            help="Binary 0/1 target column used by --aux_predict_binary.",
        )
        group.add_argument("--aux_binary_weight", type=float, default=10.0)
        group.add_argument(
            "--aux_binary_pos_weight",
            type=float,
            default=0.0,
            help="BCE positive-class weight. Zero calculates N_negative/N_positive from training metadata.",
        )
        group.add_argument("--aux_binary_dropout", type=float, default=0.15)

        # Auxiliary continuous regression head
        group.add_argument(
            "--aux_predict_regression",
            action=self.bool_action,
            default=False,
            help="Predict one continuous TSV value from latent mu(X).",
        )
        group.add_argument(
            "--aux_regression_col",
            type=str,
            default="glycohemoglobin_hba1c",
            help="Continuous target column used by --aux_predict_regression.",
        )
        group.add_argument(
            "--aux_regression_source_cols",
            nargs="+",
            default=None,
            help=(
                "Optional TSV columns to sum into the continuous target. "
                "When provided, --aux_regression_col is used as the derived target name."
            ),
        )
        group.add_argument(
            "--aux_regression_transform",
            type=str,
            default="identity",
            choices=["identity", "log1p"],
            help="Transformation applied before training-time z-normalization.",
        )
        group.add_argument("--aux_regression_weight", type=float, default=20.0)
        group.add_argument("--aux_regression_dropout", type=float, default=0.15)
        group.add_argument("--aux_regression_loss", choices=["huber", "mse", "mae"], default="huber")

        # Auxiliary plaque head
        group.add_argument(
            "--aux_predict_plaque",
            action=self.bool_action,
            default=False,
            help="Predict plaque volumes from latent as an auxiliary task.",
        )
        group.add_argument(
            "--aux_plaque_cols",
            nargs=4,
            default=[
                "plaque_volume_right_carotid_corrected",
                "plaque_volume_left_carotid_corrected",
                "plaque_volume_right_femoral_corrected",
                "plaque_volume_left_femoral_corrected",
            ],
        )
        group.add_argument(
            "--aux_plaque_norm",
            type=str,
            default="log1p_z",
            choices=["none", "log1p_z"],
        )
        group.add_argument("--aux_plaque_weight", type=float, default=0.25)

        # Adversarial age invariance
        group.add_argument(
            "--adv_predict_age",
            action=self.bool_action,
            default=False,
            help="Add gradient-reversal age classifier on latent mu(X).",
        )
        group.add_argument("--adv_age_weight", type=float, default=0.0)
        group.add_argument("--adv_age_hidden", type=int, default=128)
        group.add_argument("--adv_age_dropout", type=float, default=0.15)
        group.add_argument("--adv_age_grl_lambda", type=float, default=1.0)

    def add_age_search_options(self, parser):
        group = parser.add_argument_group("Age-bin search options")

        group.add_argument("--age_search_min_bins", type=int, default=4)
        group.add_argument("--age_search_max_bins", type=int, default=10)
        group.add_argument("--age_search_switch_age", type=float, default=50.0)
        group.add_argument("--age_search_step_young", type=float, default=10.0)
        group.add_argument("--age_search_step_old", type=float, default=5.0)
        group.add_argument(
            "--age_search_objective",
            type=str,
            default="diagnostic_bal_acc",
            choices=["diagnostic_bal_acc", "diagnostic_auc", "val_loss"],
        )
        group.add_argument(
            "--age_search_policy_mode",
            type=str,
            default="global",
            choices=["global", "age", "prevalence"],
        )
        group.add_argument(
            "--age_bin_candidates_json",
            type=str,
            default=None,
            help="JSON file with candidate age-bin sets for --phase search_age_bins.",
        )
        group.add_argument("--age_search_epochs", type=int, default=8)
        group.add_argument(
            "--age_search_source",
            type=str,
            default="json",
            choices=["json", "prevalence", "dynamic"],
        )
        group.add_argument("--age_auto_n_candidates", type=int, default=10)
        group.add_argument("--age_auto_min_bin_size", type=int, default=80)
        group.add_argument("--age_auto_smooth_window", type=int, default=5)

    def add_baseline_options(self, parser):
        group = parser.add_argument_group("Supervised baseline options")

        group.add_argument("--baseline_test_cond_tsv", type=str, default=None)
        group.add_argument(
            "--baseline_models",
            nargs="+",
            default=["logreg", "ridge", "random_forest", "hist_gbm", "mlp"],
            help="logreg ridge svm_rbf random_forest hist_gbm mlp cnn pointnet all_sklearn all",
        )
        group.add_argument(
            "--baseline_feature_set",
            type=str,
            default="shape_meta",
            choices=["shape", "meta", "shape_meta"],
        )
        group.add_argument("--baseline_calib_ratio", type=float, default=0.2)
        group.add_argument("--baseline_n_points", type=int, default=2048)
        group.add_argument("--baseline_epochs", type=int, default=40)
        group.add_argument("--baseline_lr", type=float, default=1e-3)

    def add_phase_options(self, parser):
        # BaseOptions itself has no phase-specific options.
        # TrainOptions and TestOptions override this.
        pass

    def _load_cond_stats_for_eval(self, opt):
        if getattr(opt, "phase", "") not in ("sample", "test"):
            return

        stats_path = Path(opt.checkpoints_dir) / opt.name / "cond_stats.json"
        if not stats_path.exists():
            return

        stats = json.loads(stats_path.read_text())

        opt.latent_prior = str(
            stats.get("latent_prior", getattr(opt, "latent_prior", "gaussian"))
        ).lower()
        saved_boson_cfg = stats.get("boson_cfg", getattr(opt, "boson_cfg", None))
        opt.boson_cfg = dict(saved_boson_cfg) if saved_boson_cfg else None
        opt.prior_match_mode = str(
            stats.get("prior_match_mode", getattr(opt, "prior_match_mode", "auto"))
        ).lower()
        opt.mmd_buffer_size = int(
            stats.get("mmd_buffer_size", getattr(opt, "mmd_buffer_size", 64))
        )
        opt.mmd_prior_samples = int(
            stats.get("mmd_prior_samples", getattr(opt, "mmd_prior_samples", 64))
        )
        opt.use_cond = bool(stats.get("use_cond", getattr(opt, "use_cond", False)))
        opt.cond_weight = bool(stats.get("cond_weight", False))
        opt.weight_col = stats.get("weight_col", getattr(opt, "weight_col", None))
        opt.weight_mean = float(stats.get("weight_mean", 0.0))
        opt.weight_std = float(stats.get("weight_std", 1.0))
        opt.aux_regression_loss = str(stats.get("aux_regression_loss", "huber"))
        opt.cond_dim = (
            int(stats.get("cond_dim", getattr(opt, "cond_dim", 0)))
            if opt.use_cond
            else 0
        )
        if int(stats.get("ninput_edges", 0) or 0) > 0:
            opt.ninput_edges = int(stats["ninput_edges"])
        opt.input_nc = int(stats.get("input_nc", getattr(opt, "input_nc", 3)))
        opt.ncf = [int(value) for value in stats.get("ncf", getattr(opt, "ncf", [64, 128, 256, 256]))]
        opt.latent_dim = int(stats.get("latent_dim", getattr(opt, "latent_dim", 256)))
        opt.use_preprocessed = bool(
            stats.get("use_preprocessed", getattr(opt, "use_preprocessed", False))
        )
        opt.use_pointcloud = bool(
            stats.get("use_pointcloud", getattr(opt, "use_pointcloud", False))
        )
        opt.use_point_labels = bool(stats.get("use_point_labels", getattr(opt, "use_point_labels", False)))
        opt.point_label_regions = int(stats.get("point_label_regions", getattr(opt, "point_label_regions", 6)))
        opt.point_label_pooling = bool(stats.get("point_label_pooling", getattr(opt, "point_label_pooling", False)))
        opt.use_scan_geometry_features = bool(
            stats.get("use_scan_geometry_features", getattr(opt, "use_scan_geometry_features", False))
        )
        opt.scan_geometry_features_file = str(
            stats.get("scan_geometry_features_file", getattr(opt, "scan_geometry_features_file", "scan_geometry_features.npz"))
        )
        opt.require_scan_geometry_features = bool(
            stats.get("require_scan_geometry_features", getattr(opt, "require_scan_geometry_features", False))
        )
        opt.restore_metric_scale = bool(
            stats.get("restore_metric_scale", getattr(opt, "restore_metric_scale", True))
        )
        opt.scan_geometry_feature_dim = int(stats.get("scan_geometry_feature_dim", 0))
        opt.target_missing_policy = str(
            stats.get("target_missing_policy", getattr(opt, "target_missing_policy", "drop"))
        )
        opt.aux_predict_athero = bool(
            stats.get("aux_predict_athero", getattr(opt, "aux_predict_athero", False))
        )
        opt.aux_athero_mode = str(
            stats.get("aux_athero_mode", getattr(opt, "aux_athero_mode", "any"))
        )
        opt.aux_athero_weight = float(
            stats.get("aux_athero_weight", getattr(opt, "aux_athero_weight", 0.0))
        )
        opt.aux_predict_binary = bool(
            stats.get("aux_predict_binary", getattr(opt, "aux_predict_binary", False))
        )
        opt.aux_binary_col = str(
            stats.get("aux_binary_col", getattr(opt, "aux_binary_col", "ccta_anycad"))
        )
        opt.aux_binary_weight = float(
            stats.get("aux_binary_weight", getattr(opt, "aux_binary_weight", 10.0))
        )
        opt.aux_binary_pos_weight = float(
            stats.get("aux_binary_pos_weight", getattr(opt, "aux_binary_pos_weight", 1.0))
        )
        opt.aux_binary_dropout = float(
            stats.get("aux_binary_dropout", getattr(opt, "aux_binary_dropout", 0.15))
        )
        opt.aux_predict_regression = bool(
            stats.get("aux_predict_regression", getattr(opt, "aux_predict_regression", False))
        )
        opt.aux_regression_col = str(
            stats.get("aux_regression_col", getattr(opt, "aux_regression_col", "glycohemoglobin_hba1c"))
        )
        source_cols = stats.get(
            "aux_regression_source_cols",
            getattr(opt, "aux_regression_source_cols", None),
        )
        opt.aux_regression_source_cols = list(source_cols) if source_cols else None
        opt.aux_regression_transform = str(
            stats.get(
                "aux_regression_transform",
                getattr(opt, "aux_regression_transform", "identity"),
            )
        )
        opt.aux_regression_weight = float(
            stats.get("aux_regression_weight", getattr(opt, "aux_regression_weight", 20.0))
        )
        opt.aux_regression_dropout = float(
            stats.get("aux_regression_dropout", getattr(opt, "aux_regression_dropout", 0.15))
        )
        opt.aux_regression_mean = float(stats.get("aux_regression_mean", 0.0))
        opt.aux_regression_std = float(stats.get("aux_regression_std", 1.0))
        opt.encoder_arch = str(stats.get("encoder_arch", getattr(opt, "encoder_arch", "meshconv")))
        opt.encoder_max_points = int(stats.get("encoder_max_points", getattr(opt, "encoder_max_points", 0)))
        opt.mesh_use_knn = bool(stats.get("mesh_use_knn", getattr(opt, "mesh_use_knn", True)))
        opt.mesh_knn_k = int(stats.get("mesh_knn_k", getattr(opt, "mesh_knn_k", 8)))
        opt.mesh_knn_max_points = int(stats.get("mesh_knn_max_points", getattr(opt, "mesh_knn_max_points", 4096)))
        stats_feature = str(stats.get("feature_source", getattr(opt, "feature_source", "scan"))).strip().lower().replace("-", "_")
        if stats_feature == "hbalc":
            stats_feature = "hba1c"
        if stats_feature in {"both", "ldl_hbalc"}:
            stats_feature = "ldl_hba1c"
        opt.feature_source = stats_feature
        opt.cond_athero = bool(opt.use_cond and stats.get("athero_cols", []))
        opt.cond_plaque = bool(opt.use_cond and stats.get("plaque_cols", []))

        opt.cond_athero_cols = stats.get("athero_cols", getattr(opt, "cond_athero_cols", None))
        opt.cond_plaque_cols = stats.get("plaque_cols", getattr(opt, "cond_plaque_cols", None))
        opt.cond_plaque_norm = stats.get(
            "plaque_norm",
            getattr(opt, "cond_plaque_norm", "log1p_z"),
        )
        opt.cond_athero_mode = stats.get(
            "athero_mode",
            getattr(opt, "cond_athero_mode", "4bit"),
        )

        opt.only_gender = stats.get("only_gender", getattr(opt, "only_gender", None))
        opt.age_bins = stats.get("age_bins", getattr(opt, "age_bins", None))
        opt.age_labels = stats.get("age_labels", getattr(opt, "age_labels", None))
        opt.cond_age_source = stats.get(
            "cond_age_source",
            getattr(opt, "cond_age_source", "auto"),
        )
        opt.cond_agecut_col = stats.get(
            "agecut_col",
            getattr(opt, "cond_agecut_col", None),
        )
        opt.cond_age_mode = stats.get(
            "cond_age_mode",
            getattr(opt, "cond_age_mode", "hard"),
        )
        opt.use_cond_prior = bool(
            opt.use_cond
            and stats.get("use_cond_prior", getattr(opt, "use_cond_prior", False))
        )
        opt.demo_cond_dim = (
            int(stats.get("demo_cond_dim", getattr(opt, "demo_cond_dim", 0)))
            if opt.use_cond
            else 0
        )
        opt.encoder_cond_mode = (
            stats.get("encoder_cond_mode", getattr(opt, "encoder_cond_mode", "none"))
            if opt.use_cond
            else "none"
        )

        opt.adv_predict_age = bool(
            stats.get("adv_predict_age", getattr(opt, "adv_predict_age", False))
        )
        opt.adv_age_weight = float(
            stats.get("adv_age_weight", getattr(opt, "adv_age_weight", 0.0))
        )
        opt.adv_age_hidden = int(
            stats.get("adv_age_hidden", getattr(opt, "adv_age_hidden", 128))
        )
        opt.adv_age_dropout = float(
            stats.get("adv_age_dropout", getattr(opt, "adv_age_dropout", 0.15))
        )
        opt.adv_age_grl_lambda = float(
            stats.get("adv_age_grl_lambda", getattr(opt, "adv_age_grl_lambda", 1.0))
        )

    def parse(self):
        opt = self.parser.parse_args()

        opt.is_train = self.is_train
        opt.is_test = self.is_test

        opt.gpu_ids = [
            int(i)
            for i in str(opt.gpu_ids).split(",")
            if str(i).strip() != "" and int(i) >= 0
        ]

        if opt.only_gender is not None:
            opt.only_gender = str(opt.only_gender).upper()

        fs = str(getattr(opt, "feature_source", "scan")).strip().lower().replace("-", "_")
        if fs == "hbalc":
            fs = "hba1c"
        if fs in {"both", "ldl_hbalc"}:
            fs = "ldl_hba1c"
        opt.feature_source = fs

        if fs != "scan":
            # Clinical sanity mode does not load body scans/NPZ files.  It still
            # uses the same MeshVAE code path by encoding the numeric feature(s)
            # as a small point cloud.
            opt.use_preprocessed = False
            opt.use_pointcloud = True
            opt.clinical_points = max(2, int(getattr(opt, "clinical_points", 32)))
            opt.ninput_edges = opt.clinical_points
            if hasattr(opt, "chamfer_npoints"):
                opt.chamfer_npoints = min(int(opt.chamfer_npoints), opt.clinical_points)
            if hasattr(opt, "metric_npoints"):
                opt.metric_npoints = min(int(opt.metric_npoints), opt.clinical_points)

        if opt.boson_params:
            opt.boson_cfg = json.loads(opt.boson_params)
            if not isinstance(opt.boson_cfg, dict):
                raise ValueError("--boson_params must be a JSON object.")
        else:
            opt.boson_cfg = {} if opt.latent_prior == "boson" else None
        # For test/sample, recover training-time conditioning setup if available.
        self._load_cond_stats_for_eval(opt)

        opt.latent_prior = str(getattr(opt, "latent_prior", "gaussian")).lower()
        if opt.latent_prior == "boson" and getattr(opt, "boson_sampler_path", None):
            if getattr(opt, "boson_cfg", None) is None:
                opt.boson_cfg = {}
            opt.boson_cfg["sampler_path"] = str(opt.boson_sampler_path)
        opt.prior_match_mode = str(getattr(opt, "prior_match_mode", "auto")).lower()
        if opt.latent_prior == "boson" and opt.prior_match_mode == "kl":
            raise ValueError(
                "The empirical Boson prior has no tractable density. "
                "Use --prior_match_mode mmd (or auto)."
            )
        if opt.latent_prior == "boson" and getattr(opt, "boson_cfg", None) is None:
            opt.boson_cfg = {}
        opt.mmd_buffer_size = max(0, int(getattr(opt, "mmd_buffer_size", 64)))
        opt.mmd_prior_samples = max(2, int(getattr(opt, "mmd_prior_samples", 64)))

        fs = str(getattr(opt, "feature_source", "scan")).strip().lower().replace("-", "_")
        if fs == "hbalc":
            fs = "hba1c"
        if fs in {"both", "ldl_hbalc"}:
            fs = "ldl_hba1c"
        opt.feature_source = fs
        if fs != "scan":
            opt.use_preprocessed = False
            opt.use_pointcloud = True
            opt.clinical_points = max(2, int(getattr(opt, "clinical_points", 32)))
            opt.ninput_edges = opt.clinical_points
            if hasattr(opt, "chamfer_npoints"):
                opt.chamfer_npoints = min(int(opt.chamfer_npoints), opt.clinical_points)
            if hasattr(opt, "metric_npoints"):
                opt.metric_npoints = min(int(opt.metric_npoints), opt.clinical_points)

        opt.point_label_regions = max(1, int(getattr(opt, "point_label_regions", 6)))
        base_input_nc = 3 + (opt.point_label_regions if getattr(opt, "use_point_labels", False) else 0)
        opt.input_nc = base_input_nc + int(getattr(opt, "scan_geometry_feature_dim", 0))
        if getattr(opt, "point_label_pooling", False) and not getattr(opt, "use_point_labels", False):
            raise ValueError("--point_label_pooling requires --use_point_labels")

        if getattr(opt, "aux_predict_regression", False) and getattr(opt, "aux_predict_binary", False):
            raise ValueError(
                "Choose one supervised target head per run: "
                "--aux_predict_regression or --aux_predict_binary."
            )

        if opt.only_gender is not None:
            opt.only_gender = str(opt.only_gender).upper()

        if getattr(opt, "encoder_use_cond", False):
            raise ValueError(
                "--encoder_use_cond is deprecated and unsafe for diagnostic training when "
                "--cond_athero is enabled, because it passes disease labels into the encoder. "
                "Use --encoder_cond_mode demo instead."
            )

        # Age conditioning can be driven either by explicit numeric --age_bins
        # or by a categorical TSV column called agecut.  In agecut mode the
        # labels are discovered when build_age_gender_lookup() reads the TSV,
        # so cond_dim is a temporary placeholder until the dataset is built.
        if getattr(opt, "use_cond", False):
            if opt.age_bins is not None:
                opt.age_bins = list(map(float, opt.age_bins))

            if not getattr(opt, "age_labels", None) and opt.age_bins:
                opt.age_labels = _build_age_labels(opt.age_bins)

            age_source = str(getattr(opt, "cond_age_source", "auto")).strip().lower()
            if not getattr(opt, "age_labels", None) and not opt.age_bins and age_source == "age":
                raise ValueError(
                    "--age_bins is required when --cond_age_source age is used. "
                    "For the 2026 TSV, omit --age_bins and use the default auto agecut mode."
                )

            if getattr(opt, "age_labels", None):
                opt.demo_cond_dim = 2 + len(opt.age_labels) + int(getattr(opt, "cond_weight", False))
                opt.cond_dim = cond_dim_from_opt(opt)
            else:
                opt.age_bins = []
                opt.age_labels = []
                opt.demo_cond_dim = 0
                opt.cond_dim = 0
        else:
            opt.age_bins = [] if opt.age_bins is None else list(map(float, opt.age_bins))
            opt.age_labels = []
            opt.demo_cond_dim = 0
            opt.cond_dim = 0
            opt.encoder_cond_mode = "none"

        return opt


class TrainOptions(BaseOptions):
    default_phase = "train"
    is_train = True
    is_test = False

    def add_phase_options(self, parser):
        group = parser.add_argument_group("Train options")

        group.add_argument("--lr", type=float, default=0.0002)
        group.add_argument("--beta1", type=float, default=0.5)
        group.add_argument("--lr_policy", type=str, default="lambda")
        group.add_argument("--lr_decay_iters", type=int, default=50)

        group.add_argument("--epoch_count", type=int, default=1)

        # Keep this at 50 as requested.
        group.add_argument("--niter", type=int, default=50)

        group.add_argument("--niter_decay", type=int, default=0)

        group.add_argument("--init_type", type=str, default="xavier")
        group.add_argument("--init_gain", type=float, default=1.0)

        group.add_argument("--split_ratio", type=float, default=0.1)
        group.add_argument("--split_seed", type=int, default=3)
        group.add_argument("--val_cond_tsv", default=None,
                           help="Explicit disjoint validation metadata; fit all normalization on cond_tsv only.")

        group.add_argument(
            "--use_amp",
            action=self.bool_action,
            default=False,
            help="Use mixed precision. Disabled by default because mesh graph encoding can produce fp16 NaNs.",
        )

        group.add_argument("--beta", type=float, default=0.1)
        group.add_argument(
            "--beta_warmup_epochs",
            type=int,
            default=30,
            help="Warm up the latent-prior regularizer from 0 to --beta over this many epochs.",
        )
        group.add_argument(
            "--test_reconstruction_every",
            type=int,
            default=0,
            help="Write expensive Open3D reconstruction previews every N epochs (0 disables during training).",
        )
        group.add_argument(
            "--save_epoch_freq",
            type=int,
            default=10,
            help="Save a numbered epoch checkpoint every N epochs; latest and best are still maintained.",
        )


class TestOptions(BaseOptions):
    default_phase = "test"
    is_train = False
    is_test = True

    def add_phase_options(self, parser):
        # --------------------------------------------------------------
        # Test / sample options
        # --------------------------------------------------------------
        group = parser.add_argument_group("Test / sample options")

        group.add_argument(
            "--which_epoch",
            type=str,
            default="latest",
            help="Checkpoint tag: 'latest' or e.g. 'epoch_50'.",
        )

        group.add_argument(
            "--sample_n",
            type=int,
            default=8,
            help="Number of samples to generate.",
        )
        group.add_argument(
            "--sample_gender",
            type=str,
            default=None,
            help="M or F.",
        )
        group.add_argument(
            "--sample_age",
            type=float,
            default=None,
            help="Numeric age. Will be binned.",
        )
        group.add_argument(
            "--sample_age_bin",
            type=str,
            default=None,
            help="Optional explicit age-bin label.",
        )
        group.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="Scale z. Higher gives more variety.",
        )
        group.add_argument(
            "--sample_athero",
            nargs=4,
            type=int,
            default=None,
            help="For --phase sample: 4 ints 0/1 for ho_carotis ve_carotis ho_fem ve_fem.",
        )
        group.add_argument(
            "--sample_athero_any",
            type=int,
            default=None,
            help="For --phase sample with --cond_athero_mode any: 0 or 1.",
        )
        group.add_argument(
            "--sample_plaque",
            nargs=4,
            type=float,
            default=None,
            help="For --phase sample: 4 raw plaque volumes.",
        )

        # --------------------------------------------------------------
        # Diagnose options
        # --------------------------------------------------------------
        diagnose = parser.add_argument_group("Diagnose options")

        diagnose.add_argument(
            "--calib_cond_tsv",
            type=str,
            default=None,
            help="Calibration/train cohort TSV used by diagnose_mesh_vae.py.",
        )
        diagnose.add_argument(
            "--test_cond_tsv",
            type=str,
            default=None,
            help="Held-out test cohort TSV used by diagnose_mesh_vae.py.",
        )

        diagnose.add_argument(
            "--metric_npoints",
            type=int,
            default=4096,
            help="Number of points used for metric computation.",
        )
        diagnose.add_argument(
            "--metric_chunk",
            type=int,
            default=4096,
            help="Chunk size used during metric computation.",
        )
        diagnose.add_argument(
            "--distance_batch_size",
            type=int,
            default=2,
            help="Batch size for distance computations.",
        )
        diagnose.add_argument(
            "--samples_per_class",
            type=int,
            default=64,
            help="Number of generated/reference samples per class.",
        )
        diagnose.add_argument(
            "--distance_temperature",
            type=float,
            default=0.05,
            help="Temperature used when turning distances into scores.",
        )

        diagnose.add_argument(
            "--score_method",
            type=str,
            default="auto",
            help="Diagnostic scoring method.",
        )
        diagnose.add_argument(
            "--prior_mode",
            type=str,
            default="global",
            help="Prior mode used for diagnostic sampling/scoring.",
        )
        diagnose.add_argument(
            "--policy_mode",
            type=str,
            default="global",
            help="Policy mode used for diagnostic sampling/scoring.",
        )
        diagnose.add_argument(
            "--prevalence_band_mode",
            type=str,
            default="global",
            help="Prevalence band mode used for diagnostic sampling/scoring.",
        )
