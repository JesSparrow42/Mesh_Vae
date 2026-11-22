from __future__ import annotations

import json
from pathlib import Path

from .conditioning import _build_age_labels, cond_dim_from_opt

class BaseOptions:
    def __init__(self):
        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument("--dataroot", required=True)
        parser.add_argument("--name", required=True)
        parser.add_argument("--checkpoints_dir", default="./checkpoints")

        parser.add_argument("--gpu_ids", type=str, default="0")
        parser.add_argument(
            "--phase",
            type=str,
            default="train",
            choices=["preprocess", "train", "test", "sample", "search_age_bins", "train_baseline"],
        )

        # Preprocessing / geometry
        parser.add_argument("--target_tris", type=int, default=20000,
                            help="Target number of triangles per simplified mesh (offline preprocessing).")
        parser.add_argument("--n_points", type=int, default=0,
                            help="If > 0, also precompute point clouds with this many points per mesh.")
        parser.add_argument("--use_preprocessed", action="store_true",
                            help="Load meshes from simplified_padded_tensors.npz instead of raw OBJs.")
        parser.add_argument("--use_pointcloud", action="store_true",
                            help="Use precomputed point clouds as input features instead of mesh vertices.")

        parser.add_argument("--arch", type=str, default="meshvae")
        parser.add_argument("--pool_mode", type=str, default="skip")

        # Model sizing
        parser.add_argument("--input_nc", type=int, default=3)  # xyz features
        parser.add_argument("--ncf", nargs="+", type=int, default=[64, 128, 256])
        parser.add_argument("--latent_dim", type=int, default=64)
        parser.add_argument("--ninput_edges", type=int, default=10000)

        parser.add_argument("--beta", type=float, default=1.0)
        parser.add_argument("--latent_prior", type=str, default="gaussian")
        parser.add_argument("--boson_params", type=str, default=None)

        parser.add_argument("--lr", type=float, default=0.0002)
        parser.add_argument("--beta1", type=float, default=0.5)
        parser.add_argument("--lr_policy", type=str, default="lambda")
        parser.add_argument("--lr_decay_iters", type=int, default=50)
        parser.add_argument("--epoch_count", type=int, default=1)
        parser.add_argument("--niter", type=int, default=100)
        parser.add_argument("--niter_decay", type=int, default=50)
        parser.add_argument("--init_type", type=str, default="xavier")
        parser.add_argument("--init_gain", type=float, default=0.02)

        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_threads", type=int, default=0)

        parser.add_argument("--max_dataset_size", type=int, default=99999)

        parser.add_argument("--split_ratio", type=float, default=0.2)
        parser.add_argument("--split_seed", type=int, default=0)

        parser.add_argument("--use_amp", action="store_true")
        parser.add_argument("--dump_padded_obj", action="store_true")
        parser.add_argument("--dump_padded_max", type=int, default=8)
        parser.add_argument("--debug_kernels", action="store_true")

        parser.add_argument(
            "--outroot", 
            type=str, 
            default=".",
            help="Output root for test reconstructions."
        )

        parser.add_argument("--cohort_tsv", type=str, default=None,
                    help="Path to cohort TSV containing a column with OBJ paths (e.g. obj_path).")
        parser.add_argument("--cohort_obj_col", type=str, default="obj_path",
                            help="Column name in cohort TSV that contains full paths to .obj files.")
        parser.add_argument("--subset_group", type=str, default=None,
                            help="Shortcut group name like 'F_18-40' -> will look for "
                                "/data/projects/3D_scans/sorted_data/<group>/cohort_<group>.tsv")
        parser.add_argument("--require_existing", action="store_true",
                            help="Drop cohort rows whose obj_path does not exist on disk.")
        parser.add_argument("--chamfer_npoints", type=int, default=2048,
                            help="Subsample size for Chamfer loss when --use_pointcloud is set.")

        # ----------------- Conditioning (age + gender) -----------------
        parser.add_argument("--use_cond", action="store_true",
                            help="Train conditional VAE using age+gender from a cohort TSV.")
        parser.add_argument("--cond_tsv", type=str, default=None,
                            help="TSV with stregkode + age + sex/gender columns (e.g. cohort_20251031.tsv).")
        parser.add_argument("--cond_stregkode_col", type=str, default=None,
                            help="Column name for stregkode (auto-detect if None).")
        parser.add_argument("--cond_age_col", type=str, default=None,
                            help="Column name for age (auto-detect if None).")
        parser.add_argument("--cond_gender_col", type=str, default=None,
                            help="Column name for sex/gender (auto-detect if None).")
        parser.add_argument("--cond_embed_dim", type=int, default=16,
                            help="Embedding size used inside the network for conditioning.")
        parser.add_argument("--keep_missing_cond", action="store_true",
                            help="Keep samples missing metadata (cond=all zeros). Default: drop them.")
        
        parser.add_argument(
            "--only_gender",
            type=str,
            default=None,
            choices=["M", "F"],
            help="Optional: keep only one gender throughout dataset construction. "
                "Use this for gender-separated training / diagnosis."
        )

        # ---- Bin Search for age-based sampling policy (diagnosis) ----
        parser.add_argument(
            "--cond_age_mode",
            type=str,
            default="hard",
            choices=["hard", "soft", "rbf"],
            help="Age encoding inside the 4 existing age dims: "
                "'hard' = one-hot, 'soft' = linear interpolation, 'rbf' = RBF over age-bin centers."
        )
        parser.add_argument(
            "--age_bins",
            nargs="+",
            type=float,
            default=[18, 35, 50, 60, 200],
            help="Age-bin edges. Number of bins is dynamic."
        )
        parser.add_argument("--age_search_min_bins", type=int, default=4)
        parser.add_argument("--age_search_max_bins", type=int, default=10)
        parser.add_argument("--age_search_switch_age", type=float, default=50.0)
        parser.add_argument("--age_search_step_young", type=float, default=10.0)
        parser.add_argument("--age_search_step_old", type=float, default=5.0)
        parser.add_argument("--age_search_objective", type=str, default="diagnostic_bal_acc",
                            choices=["diagnostic_bal_acc", "diagnostic_auc", "val_loss"])
        parser.add_argument("--age_search_policy_mode", type=str, default="global",
                            choices=["global", "age", "prevalence"])
        parser.add_argument(
            "--age_rbf_gamma",
            type=float,
            default=0.015,
            help="RBF gamma when --cond_age_mode=rbf."
        )

        parser.add_argument(
            "--encoder_use_cond",
            action="store_true",
            help="Allow metadata into encoder. Leave OFF for scan-driven diagnosis."
        )
        parser.add_argument(
            "--use_cond_prior",
            action="store_true",
            help="Use p(z|cond). Leave OFF for scan-driven diagnosis."
        )

        # ---- Adversarial age invariance ----
        parser.add_argument(
            "--adv_predict_age",
            action="store_true",
            help="Add a gradient-reversal age classifier on latent mu(X), so z is discouraged from carrying age."
        )
        parser.add_argument(
            "--adv_age_weight",
            type=float,
            default=0.0,
            help="Weight for adversarial age cross-entropy. Try 0.1-1.0 first."
        )
        parser.add_argument(
            "--adv_age_hidden",
            type=int,
            default=128,
            help="Hidden width of the adversarial age classifier."
        )
        parser.add_argument(
            "--adv_age_dropout",
            type=float,
            default=0.15,
            help="Dropout inside the adversarial age classifier."
        )
        parser.add_argument(
            "--adv_age_grl_lambda",
            type=float,
            default=1.0,
            help="Gradient-reversal strength for the age adversary."
        )

        # ---- Supervised latent atherosclerosis head ----
        parser.add_argument(
            "--aux_predict_athero",
            action="store_true",
            help="Add a supervised classifier head on latent mu(X). Keeps the model as a VAE but makes latent space disease-predictive."
        )
        parser.add_argument(
            "--aux_athero_mode",
            type=str,
            default="any",
            choices=["any", "4bit"],
            help="'any' = binary sick/healthy head. '4bit' = one auxiliary target per atherosclerosis location."
        )
        parser.add_argument(
            "--aux_athero_weight",
            type=float,
            default=0.0,
            help="Weight for supervised atherosclerosis BCE loss on latent mu(X). Suggested first try: 1.0 to 3.0."
        )
        parser.add_argument(
            "--aux_athero_pos_weight",
            type=float,
            default=1.0,
            help="Positive-class BCE weight for atherosclerosis classifier. Use >1 if positives are rare."
        )
        parser.add_argument(
            "--aux_athero_dropout",
            type=float,
            default=0.15,
            help="Dropout inside the latent atherosclerosis classifier head."
        )

        parser.add_argument(
            "--aux_predict_plaque",
            action="store_true",
            help="Predict plaque volumes from latent as an auxiliary task."
        )
        parser.add_argument(
            "--aux_plaque_cols",
            nargs=4,
            default=[
                "plaque_volume_right_carotid_corrected",
                "plaque_volume_left_carotid_corrected",
                "plaque_volume_right_femoral_corrected",
                "plaque_volume_left_femoral_corrected",
            ],
            help="4 plaque target columns for latent auxiliary regression."
        )
        parser.add_argument(
            "--aux_plaque_norm",
            type=str,
            default="log1p_z",
            choices=["none", "log1p_z"],
            help="Normalization for auxiliary plaque targets."
        )
        parser.add_argument(
            "--aux_plaque_weight",
            type=float,
            default=0.25,
            help="Weight for auxiliary plaque regression loss."
        )

        parser.add_argument(
            "--mesh_use_knn",
            action="store_true",
            help="Augment face-adjacency edges with kNN edges in the mesh encoder."
        )
        parser.add_argument(
            "--mesh_knn_k",
            type=int,
            default=8,
        )
        parser.add_argument(
            "--mesh_knn_max_points",
            type=int,
            default=4096,
        )

        parser.add_argument(
            "--age_bin_candidates_json",
            type=str,
            default=None,
            help="JSON file with candidate 5-edge age-bin sets for --phase search_age_bins."
        )
        parser.add_argument(
            "--age_search_epochs",
            type=int,
            default=8,
            help="Short training length per candidate during age-bin search."
        )

        parser.add_argument(
            "--age_search_source",
            type=str,
            default="json",
            choices=["json", "prevalence", "dynamic"],
            help="How to create age-bin candidates during --phase search_age_bins."
        )
        parser.add_argument(
            "--age_auto_n_candidates",
            type=int,
            default=10,
            help="Maximum number of automatically generated prevalence-based candidates."
        )
        parser.add_argument(
            "--age_auto_min_bin_size",
            type=int,
            default=80,
            help="Minimum number of samples required in each auto-generated age bin."
        )
        parser.add_argument(
            "--age_auto_smooth_window",
            type=int,
            default=5,
            help="Window size in years for smoothing the age-prevalence curve."
        )

        # ---- Conditioning extras (atherosclerosis / plaque volume) ----
        parser.add_argument("--cond_sep", type=str, default=None,
                            help="Optional: force separator for --cond_tsv. Default: auto-detect tab vs comma.")

        parser.add_argument("--cond_athero", action="store_true",
                            help="Include 4 binary atherosclerosis flags in cond vector.")
        parser.add_argument("--cond_athero_cols", nargs=4,
                            default=["ho_carotis", "ve_carotis", "ho_fem", "ve_fem"],
                            help="4 column names (0/1). Order: ho_carotis ve_carotis ho_fem ve_fem")

        parser.add_argument("--cond_plaque", action="store_true",
                            help="Include 4 plaque volume values in cond vector.")
        parser.add_argument("--cond_plaque_cols", nargs=4,
                            default=[
                                "plaque_volume_right_carotid_corrected",
                                "plaque_volume_left_carotid_corrected",
                                "plaque_volume_right_femoral_corrected",
                                "plaque_volume_left_femoral_corrected",
                            ],
                            help="4 plaque volume column names. Order: ho_carotis ve_carotis ho_fem ve_fem")
        parser.add_argument("--cond_plaque_norm", type=str, default="log1p_z",
                            choices=["none", "log1p_z"],
                            help="Normalize plaque volumes before feeding to net. Recommended: log1p_z")

        # --- Atherosclerosis encoding mode ---
        parser.add_argument(
            "--cond_athero_mode",
            type=str,
            default="any",
            choices=["any", "4bit"],
            help="How to encode atherosclerosis. "
                "'any' = single dim (1 if any of the 4 cols is 1). "
                "'4bit' = keep all 4 dims."
        )

        # --- Sampling helper for athero_mode=any ---
        parser.add_argument(
            "--sample_athero_any",
            type=int,
            default=None,
            help="For --phase sample when --cond_athero_mode=any: 0 or 1."
        )

        # --- Training imbalance fix ---
        parser.add_argument(
            "--balance_athero",
            action="store_true",
            help="Oversample atherosclerosis-positive examples during training (WeightedRandomSampler)."
        )

        # conditioning
        parser.add_argument("--cond_scale", type=float, default=1.0,
                            help="Scale factor applied to conditional embedding (encoder/decoder).")

        # Helps sampling quality (recommended when you want random z sampling to work)
        parser.add_argument("--beta_warmup_epochs", type=int, default=0,
                            help="Warmup KL beta linearly from 0->--beta over this many epochs.")

        # ----------------- Sampling from trained model -----------------
        parser.add_argument("--which_epoch", type=str, default="latest",
                            help="Checkpoint tag: 'latest' or e.g. 'epoch_50'")
        parser.add_argument("--sample_n", type=int, default=8,
                            help="Number of samples to generate.")
        parser.add_argument("--sample_gender", type=str, default=None,
                            help="M or F")
        parser.add_argument("--sample_age", type=float, default=None,
                            help="Numeric age (will be binned).")
        parser.add_argument("--sample_age_bin", type=str, default=None,
                            help="One of: 18-35, 35-50, 50-60, 60+")
        parser.add_argument("--temperature", type=float, default=1.0,
                            help="Scale z (higher => more variety).")
        parser.add_argument("--sample_athero", nargs=4, type=int, default=None,
                            help="For --phase sample: 4 ints (0/1) for ho_carotis ve_carotis ho_fem ve_fem.")

        parser.add_argument("--sample_plaque", nargs=4, type=float, default=None,
                            help="For --phase sample: 4 floats (raw plaque volumes) for ho_carotis ve_carotis ho_fem ve_fem.")

        # ----------------- Supervised baseline phase -----------------
        parser.add_argument(
            "--baseline_test_cond_tsv",
            type=str,
            default=None,
            help="Held-out test TSV for --phase train_baseline."
        )
        parser.add_argument(
            "--baseline_models",
            nargs="+",
            default=["logreg", "ridge", "random_forest", "hist_gbm", "mlp"],
            help="Baseline models: logreg ridge svm_rbf random_forest hist_gbm mlp cnn pointnet all_sklearn all"
        )
        parser.add_argument(
            "--baseline_feature_set",
            type=str,
            default="shape_meta",
            choices=["shape", "meta", "shape_meta"],
            help="Features for sklearn baselines."
        )
        parser.add_argument("--baseline_calib_ratio", type=float, default=0.2)
        parser.add_argument("--baseline_n_points", type=int, default=2048)
        parser.add_argument("--baseline_epochs", type=int, default=40)
        parser.add_argument("--baseline_lr", type=float, default=1e-3)


        self.parser = parser

    def parse(self):
        opt = self.parser.parse_args()
        opt.gpu_ids = [int(i) for i in opt.gpu_ids.split(",") if str(i).strip() != ""]

        if opt.only_gender is not None:
            opt.only_gender = str(opt.only_gender).upper()

        # Parse boson params JSON
        if opt.boson_params:
            opt.boson_cfg = json.loads(opt.boson_params)
        else:
            opt.boson_cfg = None

        # Always keep age config available
        opt.age_bins = list(map(float, getattr(opt, "age_bins", [18, 35, 50, 60, 200])))
        opt.age_labels = _build_age_labels(opt.age_bins)

        # ----------------- Compute cond_dim -----------------
        if getattr(opt, "use_cond", False):
            extra = 0
            if getattr(opt, "cond_athero", False):
                mode = getattr(opt, "cond_athero_mode", "any")
                extra += (1 if mode == "any" else 4)
            if getattr(opt, "cond_plaque", False):
                extra += 4
            opt.cond_dim = cond_dim_from_opt(opt)
        else:
            opt.cond_dim = 0

        # In sample/test: auto-load training-time cond config if it exists
        if getattr(opt, "use_cond", False) and getattr(opt, "phase", "") in ("sample", "test"):
            stats_path = Path(opt.checkpoints_dir) / opt.name / "cond_stats.json"
            if stats_path.exists():
                stats = json.loads(stats_path.read_text())
                opt.cond_dim = int(stats.get("cond_dim", opt.cond_dim))
                opt.cond_athero = bool(stats.get("athero_cols", []))
                opt.cond_plaque = bool(stats.get("plaque_cols", []))
                opt.cond_athero_cols = stats.get("athero_cols", getattr(opt, "cond_athero_cols", None))
                opt.cond_plaque_cols = stats.get("plaque_cols", getattr(opt, "cond_plaque_cols", None))
                opt.cond_plaque_norm = stats.get("plaque_norm", getattr(opt, "cond_plaque_norm", "log1p_z"))
                opt.cond_athero_mode = stats.get("athero_mode", getattr(opt, "cond_athero_mode", "4bit"))
                opt.only_gender = stats.get("only_gender", getattr(opt, "only_gender", None))

                opt.age_bins = list(map(float, stats.get("age_bins", getattr(opt, "age_bins", [18, 35, 50, 60, 200]))))
                opt.age_labels = stats.get("age_labels", getattr(opt, "age_labels", _build_age_labels(opt.age_bins)))
                opt.cond_age_mode = stats.get("cond_age_mode", getattr(opt, "cond_age_mode", "hard"))
                opt.adv_predict_age = bool(stats.get("adv_predict_age", getattr(opt, "adv_predict_age", False)))
                opt.adv_age_weight = float(stats.get("adv_age_weight", getattr(opt, "adv_age_weight", 0.0)))
                opt.adv_age_hidden = int(stats.get("adv_age_hidden", getattr(opt, "adv_age_hidden", 128)))
                opt.adv_age_dropout = float(stats.get("adv_age_dropout", getattr(opt, "adv_age_dropout", 0.15)))
                opt.adv_age_grl_lambda = float(stats.get("adv_age_grl_lambda", getattr(opt, "adv_age_grl_lambda", 1.0)))

        return opt



class TrainOptions(BaseOptions):
    pass


class TestOptions(BaseOptions):
    pass


# =========================================
# Conditioning helpers (age + gender)
# =========================================

AGE_BINS = [18, 35, 50, 60, 200]
