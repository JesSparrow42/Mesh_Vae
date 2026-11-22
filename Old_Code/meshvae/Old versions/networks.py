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

        self.input_proj = nn.Linear(in_channels, channels[0])

        blocks = []
        c_prev = channels[0]
        for c_out in channels[1:]:
            blocks.append(MeshEdgeConvBlock(c_prev, c_out))
            c_prev = c_out
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = c_prev

        self.cond_dim = int(cond_dim)
        self.cond_embed_dim = int(cond_embed_dim) if self.cond_dim > 0 else 0
        self.cond_proj = nn.Linear(self.cond_dim, self.cond_embed_dim) if self.cond_dim > 0 else None
        self.cond_scale = float(getattr(opt, "cond_scale", 1.0)) if (self.cond_dim > 0 and opt is not None) else 1.0

        pooled_dim = 2 * self.out_dim + (self.cond_embed_dim if self.cond_dim > 0 else 0)
        self.latent_mu = nn.Linear(pooled_dim, latent_dim)
        self.latent_logsigma = nn.Linear(pooled_dim, latent_dim)

    def _encode_one(self, pos: torch.Tensor, faces: torch.Tensor | None) -> torch.Tensor:
        h = F.relu(self.input_proj(pos))

        edge_index = faces_to_edge_index(faces, num_nodes=pos.shape[0], device=pos.device)
        if self.use_knn and pos.shape[0] <= self.knn_max_points:
            edge_index = merge_edge_index(edge_index, knn_edge_index(pos, self.knn_k))

        for block in self.blocks:
            h = block(h, pos, edge_index)

        pooled = torch.cat([h.max(dim=0).values, h.mean(dim=0)], dim=0)
        return pooled

    def forward(self, x: torch.Tensor, mesh_batch: dict | None, cond: torch.Tensor | None = None):
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        assert x.dim() == 3, f"[ENC] expected [B,C,E], got {tuple(x.shape)}"

        pooled_list = []
        B = x.shape[0]

        for b in range(B):
            faces = None

            if isinstance(mesh_batch, dict) and mesh_batch.get("verts") is not None:
                V_pad = mesh_batch["verts"][b]
                V_mask = mesh_batch.get("verts_mask", None)

                if V_mask is not None:
                    pos = V_pad[V_mask[b].bool()].float()
                else:
                    pos = V_pad.float()

                F_pad = mesh_batch.get("faces", None)
                if F_pad is not None:
                    F_mask = mesh_batch.get("faces_mask", None)
                    if F_mask is not None:
                        faces = F_pad[b][F_mask[b].bool()].long()
                    else:
                        faces = F_pad[b].long()

                if pos.numel() == 0:
                    pos = x[b].permute(1, 0).contiguous().float()
                    faces = None
            else:
                pos = x[b].permute(1, 0).contiguous().float()

            pooled_list.append(self._encode_one(pos, faces))

        f_global = torch.stack(pooled_list, dim=0)

        if self.cond_proj is not None:
            if cond is None:
                raise ValueError("Model expects cond but got None.")
            c_embed = F.relu(self.cond_proj(cond.float())) * self.cond_scale
            f_global = torch.cat([f_global, c_embed], dim=1)

        mu = self.latent_mu(f_global)
        log_sigma = self.latent_logsigma(f_global)
        return mu, log_sigma


class MeshDecoder(nn.Module):
    """
    Decoder: (z, cond) -> per-edge xyz.
    If cond_dim>0:
      - embed cond
      - concat to z before FC (old behavior)
      - apply FiLM after FC and after every conv
    """
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

        c0 = channels[0]
        fc_in = latent_dim + (self.cond_embed_dim if self.cond_dim > 0 else 0)
        self.fc_z = nn.Linear(fc_in, c0 * self.E_cap)

        convs = []
        for c_in, c_out in zip(channels, channels[1:]):
            convs.append(nn.Conv1d(c_in, c_out, kernel_size=1, bias=True))
        self.convs = nn.ModuleList(convs)

        self.out_head = nn.Conv1d(channels[-1], out_channels, kernel_size=1, bias=True)

        # FiLM layers: one for fc output (channels[0]) + one per conv output (channels[1:])
        self.film = None
        if self.cond_dim > 0:
            self.film = nn.ModuleList([nn.Linear(self.cond_embed_dim, 2 * csz) for csz in channels])

    def forward(self, z: torch.Tensor, E: int, cond: torch.Tensor | None = None) -> torch.Tensor:
        B = z.shape[0]
        if E > self.E_cap:
            raise RuntimeError(f"[DEC] E={E} exceeds E_cap={self.E_cap}. Increase --ninput_edges.")

        c_embed = None
        if self.cond_proj is not None:
            if cond is None:
                raise ValueError("Model expects cond but got None.")
            c_embed = F.relu(self.cond_proj(cond.float())) * self.cond_scale  # (B, cond_embed_dim)
            z = torch.cat([z, c_embed], dim=1)

        f = self.fc_z(z).view(B, -1, self.E_cap)
        f = f[:, :, :E]

        # FiLM right after FC (channels[0])
        if (c_embed is not None) and (self.film is not None):
            f = _apply_film(f, self.film[0], c_embed)

        # conv stack (after conv i => channels[i+1], so FiLM index i+1)
        for i, conv in enumerate(self.convs):
            f = F.relu(conv(f))
            if (c_embed is not None) and (self.film is not None):
                f = _apply_film(f, self.film[i + 1], c_embed)

        return self.out_head(f)  # (B,3,E)


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
        self.boson_sampler_module = None
        self.boson_sampler = None

        self.latent_prior = getattr(opt, "latent_prior", "gaussian").lower()
        raw_boson_cfg = getattr(opt, "boson_cfg", None)
        self.boson_params = _normalize_boson_cfg(raw_boson_cfg, self.latent_dim) if raw_boson_cfg else None

        if self.latent_prior == "boson":
            if not _BOSON_AVAILABLE:
                raise ImportError("latent_prior='boson' requested but boson components are not available.")
            boson_module = BosonSamplerTorch(**self.boson_params)
            self.boson_sampler_module = boson_module
            self.boson_sampler = BosonLatentGenerator(self.latent_dim, boson_module)
            print("[VAE] Using BOSON latent prior.", flush=True)
        else:
            print("[VAE] Using GAUSSIAN latent prior.", flush=True)

        self.cond_dim = int(getattr(opt, "cond_dim", 6)) if getattr(opt, "use_cond", False) else 0
        self.cond_embed_dim = int(getattr(opt, "cond_embed_dim", 16))

        # keep metadata OUT of the encoder by default
        self.encoder_use_cond = bool(getattr(opt, "encoder_use_cond", False)) and (self.cond_dim > 0)
        enc_cond_dim = self.cond_dim if self.encoder_use_cond else 0

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

        # cond vector layout:
        age_labels = list(getattr(opt, "age_labels", []))
        if len(age_labels) == 0:
            age_bins = list(getattr(opt, "age_bins", [18, 35, 50, 60, 200]))
            n_age = max(1, len(age_bins) - 1)
        else:
            n_age = len(age_labels)

        self.athero_offset = 2 + n_age
        self.cond_athero_enabled = bool(getattr(opt, "cond_athero", False))
        self.cond_athero_width = (
            1 if str(getattr(opt, "cond_athero_mode", "any")) == "any" else 4
        )

        self.n_age = int(n_age)
        self.age_offset = 2

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
            if self.encoder_use_cond:
                print(
                    "[WARN] --adv_predict_age is enabled while --encoder_use_cond is also on. "
                    "This is a normal cVAE setting, but the age adversary must work against "
                    "age that enters the encoder directly; monitor age_adv_acc.",
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

            if self.encoder_use_cond:
                print(
                    "[WARN] --aux_predict_athero is enabled together with "
                    "--encoder_use_cond. If cond contains atherosclerosis bits, "
                    "the classifier can leak labels through the encoder.",
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

        enc_cond = cond if self.encoder_use_cond else None
        dec_cond = cond if self.cond_dim > 0 else None

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
            "x_mean": x_mean,
            "x_in": x_in,
        }

        if self.plaque_head is not None:
            out["plaque_pred"] = self.plaque_head(mu)

        if self.athero_head is not None:
            # Classifier uses mu, not z, so prediction is deterministic
            # and tied to the scan-derived latent location.
            out["athero_logits"] = self.athero_head(mu)

        if self.age_adv_head is not None:
            out["age_adv_logits"] = self.age_adv_head(
                grad_reverse(mu, self.adv_age_grl_lambda)
            )

        return out

# ==============================================================================
