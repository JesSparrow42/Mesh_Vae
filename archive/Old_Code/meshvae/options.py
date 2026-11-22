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

        group.add_argument("--gpu_ids", type=str, default="0")
        group.add_argument(
            "--phase",
            type=str,
            default=self.default_phase,
            choices=["preprocess", "train", "test", "sample", "search_age_bins", "train_baseline"],
        )

        group.add_argument("--batch_size", type=int, default=1)
        group.add_argument("--num_threads", type=int, default=0)
        group.add_argument("--max_dataset_size", type=int, default=99999)

        group.add_argument("--dump_padded_obj", action=self.bool_action, default=False)
        group.add_argument("--dump_padded_max", type=int, default=8)
        group.add_argument("--debug_kernels", action=self.bool_action, default=False)

    def add_geometry_options(self, parser):
        group = parser.add_argument_group("Geometry / data options")

        group.add_argument(
            "--feature_source",
            type=str,
            default="scan",
            choices=["scan", "ldl", "hbalc", "hba1c", "ldl_hbalc", "ldl_hba1c", "both"],
            help=(
                "Input representation. 'scan' uses the body scan as before. "
                "'ldl', 'hbalc', or 'ldl_hbalc' replace the scan with a tiny "
                "synthetic point cloud carrying those numeric TSV features. "
                "Use this as a target/model sanity check, not as the final scan model."
            ),
        )
        group.add_argument(
            "--clinical_points",
            type=int,
            default=32,
            help="Number of points used when --feature_source is ldl/hbalc/ldl_hbalc.",
        )
        group.add_argument(
            "--clinical_feature_cols",
            nargs="*",
            default=None,
            help=(
                "Optional explicit TSV column names for clinical feature mode. "
                "Normally not needed: ldl -> ldl_cholesterol; hbalc -> glycohemoglobin_hba1c."
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

        group.add_argument("--latent_prior", type=str, default="gaussian")
        group.add_argument("--boson_params", type=str, default=None)

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
        if not getattr(opt, "use_cond", False):
            return

        if getattr(opt, "phase", "") not in ("sample", "test"):
            return

        stats_path = Path(opt.checkpoints_dir) / opt.name / "cond_stats.json"
        if not stats_path.exists():
            return

        stats = json.loads(stats_path.read_text())

        opt.cond_dim = int(stats.get("cond_dim", getattr(opt, "cond_dim", 0)))
        if int(stats.get("ninput_edges", 0) or 0) > 0:
            opt.ninput_edges = int(stats["ninput_edges"])
        opt.cond_athero = bool(stats.get("athero_cols", []))
        opt.cond_plaque = bool(stats.get("plaque_cols", []))

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
        opt.demo_cond_dim = int(
            stats.get("demo_cond_dim", getattr(opt, "demo_cond_dim", 0))
        )
        opt.encoder_cond_mode = stats.get(
            "encoder_cond_mode",
            getattr(opt, "encoder_cond_mode", "none"),
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
            if str(i).strip() != ""
        ]

        if opt.only_gender is not None:
            opt.only_gender = str(opt.only_gender).upper()

        fs = str(getattr(opt, "feature_source", "scan")).strip().lower().replace("-", "_")
        if fs == "hba1c":
            fs = "hbalc"
        if fs in {"both", "ldl_hba1c"}:
            fs = "ldl_hbalc"
        opt.feature_source = fs

        if fs != "scan":
            # Clinical sanity mode does not load body scans/NPZ files.  It still
            # uses the same MeshVAE code path by encoding the numeric feature(s)
            # as a small point cloud.
            opt.use_preprocessed = False
            opt.use_pointcloud = True
            opt.input_nc = 3
            opt.clinical_points = max(2, int(getattr(opt, "clinical_points", 32)))
            opt.ninput_edges = opt.clinical_points
            if hasattr(opt, "chamfer_npoints"):
                opt.chamfer_npoints = min(int(opt.chamfer_npoints), opt.clinical_points)
            if hasattr(opt, "metric_npoints"):
                opt.metric_npoints = min(int(opt.metric_npoints), opt.clinical_points)

        if opt.boson_params:
            opt.boson_cfg = json.loads(opt.boson_params)
        else:
            opt.boson_cfg = None

        # For test/sample, recover training-time conditioning setup if available.
        self._load_cond_stats_for_eval(opt)

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
                opt.demo_cond_dim = 2 + len(opt.age_labels)
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
            help="Warmup KL beta linearly from 0 -> --beta over this many epochs.",
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
