from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .math_utils import mmd_rbf
from .losses import chamfer_distance_subsample, axis_std_loss, repulsion_loss

class MeshVariationalInference(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        use_chamfer: bool = False,
        chamfer_npoints: int = 2048,
        w_spread: float = 0.2,
        w_rep: float = 0.05,
        rep_k: int = 8,
        rep_radius: float = 0.02,
        w_plaque_aux: float = 0.0,
        w_athero_aux: float = 0.0,
        athero_pos_weight: float = 1.0,
        w_age_adv: float = 0.0,
    ):
        super().__init__()
        self.beta = float(beta)
        self.use_chamfer = bool(use_chamfer)
        self.chamfer_npoints = int(chamfer_npoints)

        self.w_spread = float(w_spread)
        self.w_rep = float(w_rep)
        self.rep_k = int(rep_k)
        self.rep_radius = float(rep_radius)

        self.w_plaque_aux = float(w_plaque_aux)
        self.w_athero_aux = float(w_athero_aux)
        self.athero_pos_weight = float(athero_pos_weight)
        self.w_age_adv = float(w_age_adv)

    def forward(
        self,
        model: "MeshVAE",
        x: torch.Tensor,
        mesh_batch: dict | None,
        cond: torch.Tensor | None = None,
        plaque_target: torch.Tensor | None = None,
    ):
        out = model(x, mesh_batch, cond=cond)
        px, pz, qz, z = out["px"], out["pz"], out["qz"], out["z"]

        x_in = out["x_in"]
        x_mean = out["x_mean"]

        if self.use_chamfer:
            recon = chamfer_distance_subsample(
                x_in, x_mean, n_points=self.chamfer_npoints
            )

            spread = axis_std_loss(x_in, x_mean)

            rep = repulsion_loss(
                x_mean,
                n_points=min(self.chamfer_npoints, 1024),
                k=self.rep_k,
                radius=self.rep_radius,
            )

            total_recon = recon + self.w_spread * spread + self.w_rep * rep
            log_px = -total_recon
        else:
            recon = None
            spread = None
            rep = None
            log_px = px.log_prob(x_in)

        if getattr(model, "latent_prior", "gaussian") == "boson":
            z_prior = model.sample_prior(z.shape[0], x.device, cond=cond)
            prior_match = mmd_rbf(z, z_prior)
            elbo = log_px - self.beta * prior_match
            loss = -elbo.mean()

            diag = {
                "elbo": elbo.detach(),
                "log_px": log_px.detach(),
                "prior_match": prior_match.detach().expand_as(elbo),
            }
        else:
            log_pz = pz.log_prob(z)
            log_qz = qz.log_prob(z)
            kl = log_qz - log_pz
            elbo = log_px - self.beta * kl
            loss = -elbo.mean()

            diag = {
                "elbo": elbo.detach(),
                "log_px": log_px.detach(),
                "kl": kl.detach(),
            }

        if recon is not None:
            diag["chamfer"] = recon.detach()
            diag["spread"] = spread.detach()
            diag["repulsion"] = rep.detach()

        if (plaque_target is not None) and (out.get("plaque_pred") is not None):
            mask = torch.isfinite(plaque_target)
            target = torch.nan_to_num(plaque_target, nan=0.0)

            per_elem = F.smooth_l1_loss(out["plaque_pred"], target, reduction="none")
            per_sample = (per_elem * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp(min=1.0)

            loss = loss + self.w_plaque_aux * per_sample.mean()
            diag["plaque_loss"] = per_sample.detach()

        if self.w_athero_aux > 0.0 and out.get("athero_logits") is not None:
            target = model.athero_target_from_cond(cond)
            if target is not None:
                logits = out["athero_logits"]

                # Broadcast/trim target if needed.
                if target.shape != logits.shape:
                    if logits.shape[1] == 1:
                        target = (target.max(dim=1, keepdim=True).values > 0.5).float()
                    elif target.shape[1] == 1 and logits.shape[1] == 4:
                        target = target.repeat(1, 4)

                pos_weight = None
                if self.athero_pos_weight > 0 and not math.isclose(self.athero_pos_weight, 1.0):
                    pos_weight = torch.full(
                        (logits.shape[1],),
                        float(self.athero_pos_weight),
                        device=logits.device,
                        dtype=logits.dtype,
                    )

                per_elem = F.binary_cross_entropy_with_logits(
                    logits,
                    target.float(),
                    reduction="none",
                    pos_weight=pos_weight,
                )
                per_sample = per_elem.mean(dim=1)

                loss = loss + self.w_athero_aux * per_sample.mean()
                diag["athero_loss"] = per_sample.detach()

                with torch.no_grad():
                    prob = torch.sigmoid(logits)
                    if logits.shape[1] == 1:
                        pred_any = (prob[:, 0] >= 0.5).long()
                        true_any = (target[:, 0] >= 0.5).long()
                        diag["athero_acc"] = (pred_any == true_any).float().detach()
                        diag["athero_prob_sick"] = prob[:, 0].detach()
                    else:
                        pred_bits = (prob >= 0.5).long()
                        true_bits = (target >= 0.5).long()
                        diag["athero_bit_acc"] = (pred_bits == true_bits).float().mean(dim=1).detach()
                        pred_any = pred_bits.any(dim=1).long()
                        true_any = true_bits.any(dim=1).long()
                        diag["athero_acc"] = (pred_any == true_any).float().detach()
                        diag["athero_prob_sick"] = prob.max(dim=1).values.detach()

        if self.w_age_adv > 0.0 and out.get("age_adv_logits") is not None:
            target = model.age_target_from_cond(cond)
            if target is not None:
                logits = out["age_adv_logits"]
                per_sample = F.cross_entropy(logits, target, reduction="none")
                loss = loss + self.w_age_adv * per_sample.mean()
                diag["age_adv_loss"] = per_sample.detach()

                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    diag["age_adv_acc"] = (pred == target).float().detach()

        return loss, diag, out
