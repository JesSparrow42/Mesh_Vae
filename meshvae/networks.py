from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from .adversarial import grad_reverse
from .boson import BosonPrior, BosonSamplerTorch, BosonLatentGenerator, _normalize_boson_cfg, _BOSON_AVAILABLE
from .graph_blocks import _apply_film, MeshEdgeConvBlock, faces_to_edge_index, knn_edge_index, merge_edge_index

class MeshEncoder(nn.Module):
    """
    Real mesh encoder:
      - uses verts/faces/verts_mask/faces_mask from the dataset
      - face-adjacency message passing
      - optional kNN augmentation
      - global max + mean pooling

    Leave encoder conditioning OFF for scan-driven diagnosis.
    """
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        latent_dim: int,
        k: int = 5,
        opt=None,
        cond_dim: int = 0,
        cond_embed_dim: int = 16,
    ):
        super().__init__()
        assert len(channels) >= 1

        self.use_knn = bool(getattr(opt, "mesh_use_knn", False)) if opt is not None else False
        self.knn_k = int(getattr(opt, "mesh_knn_k", 8)) if opt is not None else 8
        self.knn_max_points = int(getattr(opt, "mesh_knn_max_points", 4096)) if opt is not None else 4096
        self.encoder_arch = str(getattr(opt, "encoder_arch", "meshconv")) if opt is not None else "meshconv"
        self.encoder_max_points = int(getattr(opt, "encoder_max_points", 0)) if opt is not None else 0
        self.use_point_labels = bool(getattr(opt, "use_point_labels", False)) if opt is not None else False
        self.point_label_regions = int(getattr(opt, "point_label_regions", 6)) if opt is not None else 6
        self.point_label_pooling = (
            bool(getattr(opt, "point_label_pooling", False)) and self.use_point_labels
            if opt is not None else False
        )
        self._reported_downsample = False

        if self.encoder_arch not in {"pointnet", "meshconv"}:
            raise ValueError(f"Unknown encoder_arch={self.encoder_arch!r}")

        self.input_proj = nn.Linear(in_channels, channels[0])

        blocks = []
        c_prev = channels[0]
        for c_out in channels[1:]:
            if self.encoder_arch == "meshconv":
                blocks.append(MeshEdgeConvBlock(c_prev, c_out))
            else:
                blocks.append(nn.Sequential(
                    nn.Linear(c_prev, c_out),
                    nn.LayerNorm(c_out),
                    nn.ReLU(inplace=True),
                    nn.Linear(c_out, c_out),
                    nn.LayerNorm(c_out),
                    nn.ReLU(inplace=True),
                ))
            c_prev = c_out
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = c_prev

        self.cond_dim = int(cond_dim)
        self.cond_embed_dim = int(cond_embed_dim) if self.cond_dim > 0 else 0
        self.cond_proj = nn.Linear(self.cond_dim, self.cond_embed_dim) if self.cond_dim > 0 else None
        self.cond_scale = float(getattr(opt, "cond_scale", 1.0)) if (self.cond_dim > 0 and opt is not None) else 1.0

        n_pool_groups = 1 + (self.point_label_regions if self.point_label_pooling else 0)
        pooled_dim = 2 * self.out_dim * n_pool_groups + (self.cond_embed_dim if self.cond_dim > 0 else 0)
        self.latent_mu = nn.Linear(pooled_dim, latent_dim)
        self.latent_logsigma = nn.Linear(pooled_dim, latent_dim)

        print(
            f"[ENCODER] arch={self.encoder_arch} max_points={self.encoder_max_points or 'all'} "
            f"knn={self.use_knn if self.encoder_arch == 'meshconv' else False} "
            f"point_label_pooling={self.point_label_pooling}",
            flush=True,
        )

    def _encode_one(self, features: torch.Tensor, coords: torch.Tensor, faces: torch.Tensor | None) -> torch.Tensor:
        features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        coords = torch.nan_to_num(coords.float(), nan=0.0, posinf=0.0, neginf=0.0)
        label_ids = None
        if self.point_label_pooling:
            start = 3
            stop = start + self.point_label_regions
            if features.shape[1] < stop:
                raise ValueError(
                    f"Point-label pooling needs {stop} input channels, got {features.shape[1]}"
                )
            label_ids = torch.argmax(features[:, start:stop], dim=1)

        h = F.relu(self.input_proj(features))
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        if self.encoder_arch == "meshconv":
            edge_index = faces_to_edge_index(faces, num_nodes=coords.shape[0], device=coords.device)
            if self.use_knn and coords.shape[0] <= self.knn_max_points:
                edge_index = merge_edge_index(edge_index, knn_edge_index(coords, self.knn_k))
            for block in self.blocks:
                h = block(h, coords, edge_index)
                h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            for block in self.blocks:
                h = block(h)
                h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        pooled_parts = [torch.cat([h.max(dim=0).values, h.mean(dim=0)], dim=0)]
        if label_ids is not None:
            zero_pool = torch.zeros(2 * self.out_dim, dtype=h.dtype, device=h.device)
            for region in range(self.point_label_regions):
                region_h = h[label_ids == region]
                if region_h.shape[0] == 0:
                    pooled_parts.append(zero_pool)
                else:
                    pooled_parts.append(
                        torch.cat([region_h.max(dim=0).values, region_h.mean(dim=0)], dim=0)
                    )

        pooled = torch.cat(pooled_parts, dim=0)
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        return pooled

    def forward(self, x: torch.Tensor, mesh_batch: dict | None, cond: torch.Tensor | None = None):
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        assert x.dim() == 3, f"[ENC] expected [B,C,E], got {tuple(x.shape)}"

        pooled_list = []
        B = x.shape[0]

        for b in range(B):
            faces = None
            features = x[b].permute(1, 0).contiguous().float()
            coords = features[:, :3].contiguous()

            if isinstance(mesh_batch, dict) and mesh_batch.get("verts") is not None:
                V_pad = mesh_batch["verts"][b]
                V_mask = mesh_batch.get("verts_mask", None)

                if V_mask is not None:
                    verts = V_pad[V_mask[b].bool()].float()
                else:
                    verts = V_pad.float()

                F_pad = mesh_batch.get("faces", None)
                if F_pad is not None and int(verts.shape[0]) == int(features.shape[0]):
                    F_mask = mesh_batch.get("faces_mask", None)
                    if F_mask is not None:
                        faces = F_pad[b][F_mask[b].bool()].long()
                    else:
                        faces = F_pad[b].long()
                else:
                    faces = None

                if features.numel() == 0 and verts.numel() > 0:
                    features = verts.float()
                    coords = verts[:, :3].contiguous()

            if self.encoder_max_points > 0 and features.shape[0] > self.encoder_max_points:
                if self.training:
                    take = torch.randperm(features.shape[0], device=features.device)[:self.encoder_max_points]
                else:
                    take = torch.linspace(
                        0,
                        features.shape[0] - 1,
                        steps=self.encoder_max_points,
                        device=features.device,
                    ).round().long()
                features = features[take]
                coords = features[:, :3].contiguous()
                # Subsampling invalidates original face indices; meshconv then
                # uses local kNN edges on the retained points.
                faces = None
                if not self._reported_downsample:
                    print(
                        f"[ENCODER] point cap: {x.shape[-1]} -> {features.shape[0]} "
                        "(random during training, deterministic during evaluation)",
                        flush=True,
                    )
                    self._reported_downsample = True

            pooled_list.append(self._encode_one(features, coords, faces))

        f_global = torch.stack(pooled_list, dim=0)

        if self.cond_proj is not None:
            if cond is None:
                raise ValueError("Model expects cond but got None.")
            c_embed = F.relu(self.cond_proj(cond.float())) * self.cond_scale
            f_global = torch.cat([f_global, c_embed], dim=1)

        mu = self.latent_mu(f_global)
        log_sigma = self.latent_logsigma(f_global)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=20.0, neginf=-20.0)
        log_sigma = torch.nan_to_num(log_sigma, nan=0.0, posinf=2.0, neginf=-10.0)
        return mu, log_sigma


class MeshDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: List[int],
        latent_dim: int,
        k: int = 5,
        opt=None,
        cond_dim: int = 0,
        cond_embed_dim: int = 16,
    ):
        super().__init__()
        self.E_cap = int(getattr(opt, "ninput_edges", 10000))

        self.cond_dim = int(cond_dim)
        self.cond_embed_dim = int(cond_embed_dim) if self.cond_dim > 0 else 0
        self.cond_proj = nn.Linear(self.cond_dim, self.cond_embed_dim) if self.cond_dim > 0 else None
        self.cond_scale = float(getattr(opt, "cond_scale", 1.0)) if (self.cond_dim > 0 and opt is not None) else 1.0

        grid_size = math.ceil(math.sqrt(self.E_cap))
        u = torch.linspace(-1.0, 1.0, grid_size)
        uu, vv = torch.meshgrid(u, u, indexing="ij")
        grid = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
        self.register_buffer("grid", grid)

        hidden = channels[0] if len(channels) > 0 else 64
        fold_in = latent_dim + self.cond_embed_dim + 2

        self.fold1 = nn.Sequential(
            nn.Linear(fold_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )

        fold2_in = latent_dim + self.cond_embed_dim + 3
        self.fold2 = nn.Sequential(
            nn.Linear(fold2_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_channels),
        )

    def forward(self, z: torch.Tensor, E: int, cond: torch.Tensor | None = None) -> torch.Tensor:
        B = z.shape[0]
        if E > self.E_cap:
            raise RuntimeError(f"[DEC] E={E} exceeds E_cap={self.E_cap}.")

        c_embed = None
        if self.cond_proj is not None:
            if cond is None:
                raise ValueError("Model expects cond but got None.")
            c_embed = F.relu(self.cond_proj(cond.float())) * self.cond_scale

        grid = self.grid[:E, :].unsqueeze(0).expand(B, -1, -1)
        z_exp = z.unsqueeze(1).expand(-1, E, -1)

        if c_embed is not None:
            c_exp = c_embed.unsqueeze(1).expand(-1, E, -1)
            fold1_in = torch.cat([z_exp, c_exp, grid], dim=-1)
        else:
            fold1_in = torch.cat([z_exp, grid], dim=-1)

        xyz_coarse = self.fold1(fold1_in)

        if c_embed is not None:
            fold2_in = torch.cat([z_exp, c_exp, xyz_coarse], dim=-1)
        else:
            fold2_in = torch.cat([z_exp, xyz_coarse], dim=-1)

        xyz_fine = self.fold2(fold2_in)

        return xyz_fine.permute(0, 2, 1).contiguous()


# ==============================================================================
# MeshVAE core
# ==============================================================================

class MeshVAE(nn.Module):
    """
    End-to-end VAE:
      - encoder = real mesh/graph encoder over verts+faces
      - decoder = keep existing point/vertex decoder
      - optional latent->plaque auxiliary head
    """
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        latent_dim: int,
        obs_log_std: float = math.log(0.1),
        k: int = 5,
        opt: Optional[object] = None,
    ):
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.boson_sampler = None

        self.latent_prior = getattr(opt, "latent_prior", "gaussian").lower()
        raw_boson_cfg = getattr(opt, "boson_cfg", None)
        self.boson_params = None

        if self.latent_prior == "boson":
            if not _BOSON_AVAILABLE:
                raise ImportError("latent_prior='boson' requested but boson components are not available.")
            self.boson_params = _normalize_boson_cfg(raw_boson_cfg, self.latent_dim)
            boson_module = BosonSamplerTorch(**self.boson_params)
            self.boson_sampler = BosonLatentGenerator(self.latent_dim, boson_module)
            print("[VAE] Using SPARROW BOSON latent prior.", flush=True)
        else:
            print("[VAE] Using GAUSSIAN latent prior.", flush=True)

        self.cond_dim = int(getattr(opt, "cond_dim", 0)) if getattr(opt, "use_cond", False) else 0
        self.cond_embed_dim = int(getattr(opt, "cond_embed_dim", 16))

        # Full condition layout:
        #   [gender_M, gender_F, age..., athero..., optional plaque...]
        #
        # Final diagnostic model:
        #   encoder gets only demo condition = gender + age
        #   decoder/prior get full condition = gender + age + disease
        age_labels = list(getattr(opt, "age_labels", []))
        if len(age_labels) == 0:
            age_bins = list(getattr(opt, "age_bins", [18, 35, 50, 60, 200]))
            n_age = max(1, len(age_bins) - 1)
        else:
            n_age = len(age_labels)

        self.n_age = int(n_age)
        self.age_offset = 2
        self.demo_cond_dim = int(getattr(opt, "demo_cond_dim", 2 + self.n_age))
        self.encoder_cond_mode = str(getattr(opt, "encoder_cond_mode", "none"))

        if bool(getattr(opt, "encoder_use_cond", False)):
            raise ValueError(
                "--encoder_use_cond is unsafe for diagnostic training when cond contains "
                "atherosclerosis labels. Use --encoder_cond_mode demo instead."
            )

        if self.cond_dim <= 0:
            self.encoder_cond_mode = "none"
            enc_cond_dim = 0
        elif self.encoder_cond_mode == "none":
            enc_cond_dim = 0
        elif self.encoder_cond_mode == "demo":
            enc_cond_dim = self.demo_cond_dim
        else:
            raise ValueError(
                f"Unknown encoder_cond_mode={self.encoder_cond_mode}. "
                "Use 'none' or 'demo'."
            )

        self.encoder_use_cond = self.encoder_cond_mode != "none"

        self.encoder = MeshEncoder(
            in_channels, channels, self.latent_dim, k=k, opt=opt,
            cond_dim=enc_cond_dim, cond_embed_dim=self.cond_embed_dim
        )
        self.decoder = MeshDecoder(
            in_channels, channels, self.latent_dim, k=k, opt=opt,
            cond_dim=self.cond_dim, cond_embed_dim=self.cond_embed_dim
        )

        self.use_cond_prior = (
            bool(getattr(opt, "use_cond_prior", False))
            and (self.cond_dim > 0)
            and (self.latent_prior != "boson")
        )

        self.prior_net = None
        if self.use_cond_prior:
            self.prior_net = nn.Sequential(
                nn.Linear(self.cond_dim, self.cond_embed_dim),
                nn.ReLU(),
                nn.Linear(self.cond_embed_dim, 2 * self.latent_dim),
            )

        self.plaque_head = None
        if bool(getattr(opt, "aux_predict_plaque", False)):
            plaque_dim = len(getattr(opt, "aux_plaque_cols", [0, 0, 0, 0]))
            self.plaque_head = nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, plaque_dim),
            )

        self.aux_predict_athero = bool(getattr(opt, "aux_predict_athero", False))
        self.aux_athero_mode = str(getattr(opt, "aux_athero_mode", "any"))

        # In the full condition vector, atherosclerosis starts right after
        # [gender_M, gender_F, age...]. This is also the end of the demo slice
        # that may be passed to the encoder.
        self.athero_offset = self.demo_cond_dim
        self.cond_athero_enabled = bool(getattr(opt, "cond_athero", False))
        self.cond_athero_width = (
            1 if str(getattr(opt, "cond_athero_mode", "any")) == "any" else 4
        )

        # Optional adversarial age head: the decoder can still receive age as
        # condition, but the scan-derived latent mu is trained not to retain age.
        self.adv_predict_age = (
            bool(getattr(opt, "adv_predict_age", False))
            or float(getattr(opt, "adv_age_weight", 0.0)) > 0.0
        )
        self.adv_age_grl_lambda = float(getattr(opt, "adv_age_grl_lambda", 1.0))
        self.age_adv_head = None
        if self.adv_predict_age and self.n_age > 1:
            h = int(getattr(opt, "adv_age_hidden", 128))
            d = float(getattr(opt, "adv_age_dropout", 0.15))
            self.age_adv_head = nn.Sequential(
                nn.Linear(self.latent_dim, h),
                nn.ReLU(inplace=True),
                nn.Dropout(d),
                nn.Linear(h, h),
                nn.ReLU(inplace=True),
                nn.Dropout(d),
                nn.Linear(h, self.n_age),
            )
            if self.encoder_cond_mode == "demo":
                print(
                    "[VAE] Age adversary enabled. Encoder receives gender+age only; "
                    "atherosclerosis labels are not passed to the encoder.",
                    flush=True,
                )

        # For binary diagnosis, use aux_athero_mode="any".
        self.aux_athero_dim = 4 if self.aux_athero_mode == "4bit" else 1

        self.athero_head = None
        if self.aux_predict_athero:
            self.athero_head = nn.Sequential(
                nn.Linear(self.latent_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(float(getattr(opt, "aux_athero_dropout", 0.15))),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(float(getattr(opt, "aux_athero_dropout", 0.15))),
                nn.Linear(128, self.aux_athero_dim),
            )

            if self.encoder_cond_mode == "demo":
                print(
                    "[VAE] Atherosclerosis auxiliary head enabled. Encoder conditioning is demo-only, "
                    "so disease labels are not passed to the encoder.",
                    flush=True,
                )

        self.binary_head = None
        if bool(getattr(opt, "aux_predict_binary", False)):
            binary_dropout = float(getattr(opt, "aux_binary_dropout", 0.15))
            self.binary_head = nn.Sequential(
                nn.Linear(self.latent_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(binary_dropout),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(binary_dropout),
                nn.Linear(128, 1),
            )
            print(
                f"[VAE] Generic binary head enabled for "
                f"{getattr(opt, 'aux_binary_col', 'target')!r}.",
                flush=True,
            )

        self.regression_head = None
        if bool(getattr(opt, "aux_predict_regression", False)):
            regression_dropout = float(getattr(opt, "aux_regression_dropout", 0.15))
            self.regression_head = nn.Sequential(
                nn.Linear(self.latent_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(regression_dropout),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(regression_dropout),
                nn.Linear(128, 1),
            )
            print(
                f"[VAE] Continuous regression head enabled for "
                f"{getattr(opt, 'aux_regression_col', 'target')!r}.",
                flush=True,
            )

        self.obs_log_std = nn.Parameter(torch.tensor(float(obs_log_std)), requires_grad=False)

    @staticmethod
    def _diag_normal(mu: torch.Tensor, log_sigma: torch.Tensor) -> Independent:
        sigma = torch.exp(log_sigma)
        return Independent(Normal(mu, sigma), 1)

    def posterior(self, x: torch.Tensor, mesh_batch: dict | None, cond: torch.Tensor | None = None):
        mu, log_sigma = self.encoder(x, mesh_batch, cond=cond)
        log_sigma = log_sigma.clamp(min=-10.0, max=10.0)
        qz = self._diag_normal(mu, log_sigma)
        return qz, mu

    def prior(self, batch: int, latent_dim: int, device, cond: torch.Tensor | None = None):
        if self.latent_prior == "boson" and self.boson_sampler is not None:
            return BosonPrior(
                boson_sampler=self.boson_sampler,
                batch_size=batch,
                latent_features=latent_dim,
                device=device,
            )

        if self.use_cond_prior and (self.prior_net is not None) and (cond is not None):
            h = self.prior_net(cond.float().to(device))
            mu, log_sigma = h.chunk(2, dim=1)
            log_sigma = log_sigma.clamp(min=-6.0, max=2.0)
            return self._diag_normal(mu, log_sigma)

        mu = torch.zeros(batch, latent_dim, device=device)
        log_sigma = torch.zeros(batch, latent_dim, device=device)
        return self._diag_normal(mu, log_sigma)

    def sample_prior(self, batch: int, device, cond: torch.Tensor | None = None):
        if self.latent_prior == "boson" and self.boson_sampler is not None:
            return self.boson_sampler(batch).to(device)

        pz = self.prior(batch, self.latent_dim, device=device, cond=cond)
        return pz.rsample() if hasattr(pz, "rsample") else pz.sample()

    def athero_target_from_cond(self, cond: torch.Tensor | None) -> torch.Tensor | None:
        if cond is None:
            return None
        if not self.cond_athero_enabled:
            return None

        start = int(self.athero_offset)
        stop = start + int(self.cond_athero_width)
        if cond.shape[1] < stop:
            return None

        raw = cond[:, start:stop].float()

        if self.aux_athero_mode == "4bit":
            if raw.shape[1] == 4:
                return raw.clamp(0.0, 1.0)
            return raw.repeat(1, 4).clamp(0.0, 1.0)

        # Binary target: sick if any atherosclerosis bit is positive.
        return (raw.max(dim=1, keepdim=True).values > 0.5).float()

    def age_target_from_cond(self, cond: torch.Tensor | None) -> torch.Tensor | None:
        if cond is None:
            return None
        stop = int(self.age_offset + self.n_age)
        if cond.shape[1] < stop:
            return None
        # Works for hard one-hot, soft age interpolation, and RBF age encodings.
        return torch.argmax(cond[:, self.age_offset:stop].float(), dim=1).long()

    def likelihood(self, x_mean: torch.Tensor) -> Independent:
        std = torch.exp(self.obs_log_std)
        return Independent(Normal(loc=x_mean, scale=std), reinterpreted_batch_ndims=2)

    def forward(self, x: torch.Tensor, mesh_batch: dict | None, cond: torch.Tensor | None = None):
        x_in = x.squeeze(-1) if (x.dim() == 4 and x.shape[-1] == 1) else x
        B, C, E = x_in.shape

        # Split condition safely.
        # Full cond layout:
        #   [gender_M, gender_F, age..., athero..., optional plaque...]
        # Encoder sees either:
        #   none -> scan only
        #   demo -> scan + gender/age only
        # Decoder/prior always receive the full condition when conditioning is enabled.
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("Model was built with conditioning, but cond=None was passed.")
            dec_cond = cond.float()
            if self.encoder_cond_mode == "demo":
                enc_cond = dec_cond[:, :self.demo_cond_dim]
            elif self.encoder_cond_mode == "none":
                enc_cond = None
            else:
                raise ValueError(f"Unknown encoder_cond_mode={self.encoder_cond_mode}")
        else:
            dec_cond = None
            enc_cond = None

        qz, mu = self.posterior(x_in, mesh_batch, cond=enc_cond)
        z = qz.rsample() if self.training else mu

        x_mean = self.decoder(z, E, cond=dec_cond)
        if x_mean.dim() == 4 and x_mean.shape[-1] == 1:
            x_mean = x_mean.squeeze(-1)

        px = self.likelihood(x_mean)
        pz = self.prior(
            B,
            z.shape[-1],
            device=x.device,
            cond=(dec_cond if self.use_cond_prior else None),
        )

        out = {
            "px": px,
            "pz": pz,
            "qz": qz,
            "z": z,
            "mu": mu,
            "posterior_sample": z if self.training else qz.rsample(),
            "x_mean": x_mean,
            "x_in": x_in,
        }

        if self.plaque_head is not None:
            out["plaque_pred"] = self.plaque_head(mu)

        if self.athero_head is not None:
            # Classifier uses mu from the encoder path. With encoder_cond_mode='demo',
            # mu can depend on scan + age/gender, but never on atherosclerosis labels.
            out["athero_logits"] = self.athero_head(mu)

        if self.binary_head is not None:
            out["binary_logits"] = self.binary_head(mu).squeeze(1)

        if self.regression_head is not None:
            out["regression_pred"] = self.regression_head(mu).squeeze(1)

        if self.age_adv_head is not None:
            out["age_adv_logits"] = self.age_adv_head(
                grad_reverse(mu, self.adv_age_grl_lambda)
            )

        return out
