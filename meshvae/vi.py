from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .math_utils import mmd_imq
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
        w_binary_aux: float = 0.0,
        binary_pos_weight: float = 1.0,
        w_regression_aux: float = 0.0,
        w_age_adv: float = 0.0,
        prior_match_mode: str = "auto",
        mmd_buffer_size: int = 64,
        mmd_prior_samples: int = 64,
        regression_loss: str = "huber",
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
        self.w_binary_aux = float(w_binary_aux)
        self.binary_pos_weight = float(binary_pos_weight)
        self.w_regression_aux = float(w_regression_aux)
        self.regression_loss = regression_loss
        if regression_loss not in {"huber", "mse", "mae"}:
            raise ValueError(f"Unknown regression loss: {regression_loss}")
        self.w_age_adv = float(w_age_adv)
        self.prior_match_mode = str(prior_match_mode).lower()
        self.mmd_buffer_size = max(0, int(mmd_buffer_size))
        self.mmd_prior_samples = max(2, int(mmd_prior_samples))
        self.register_buffer("_mmd_history", torch.empty(0), persistent=False)

    def _mmd_prior_match(self, model, z: torch.Tensor, cond):
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        training = bool(model.training)
        current_count = int(z.shape[0])
        match_samples = z
        if training and self.mmd_buffer_size > 0 and self._mmd_history.numel() > 0:
            history = self._mmd_history.to(device=z.device, dtype=z.dtype)
            match_samples = torch.cat([history, z], dim=0)
            match_samples = match_samples[-self.mmd_buffer_size :]

        prior_count = max(self.mmd_prior_samples, int(match_samples.shape[0]))
        prior_samples = base_model.sample_prior(prior_count, z.device, cond=cond).to(z.dtype)
        raw_match = mmd_imq(match_samples, prior_samples)

        # Historical samples are detached. Rescale so the current mini-batch's
        # gradient estimates the contribution of a full differentiable buffer.
        gradient_scale = float(match_samples.shape[0]) / max(current_count, 1)
        match_loss = raw_match * gradient_scale

        if training and self.mmd_buffer_size > 0:
            with torch.no_grad():
                history = self._mmd_history.to(device=z.device, dtype=z.dtype)
                updated = torch.cat([history, z.detach()], dim=0)
                self._mmd_history = updated[-self.mmd_buffer_size :]
        return match_loss, raw_match, int(match_samples.shape[0])

    def forward(
        self,
        model: "MeshVAE",
        x: torch.Tensor,
        mesh_batch: dict | None,
        cond: torch.Tensor | None = None,
        plaque_target: torch.Tensor | None = None,
        binary_target: torch.Tensor | None = None,
        regression_target: torch.Tensor | None = None,
    ):
        out = model(x, mesh_batch, cond=cond)
        px, pz, qz, z = out["px"], out["pz"], out["qz"], out["z"]

        x_in = out["x_in"]
        x_mean = out["x_mean"]

        if self.use_chamfer:
            x_xyz = x_in[:, :3, :]
            x_mean_xyz = x_mean[:, :3, :]
            recon = chamfer_distance_subsample(
                x_xyz, x_mean_xyz, n_points=self.chamfer_npoints
            )

            spread = axis_std_loss(x_xyz, x_mean_xyz)

            rep = repulsion_loss(
                x_mean_xyz,
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

        base_model = model.module if isinstance(model, nn.DataParallel) else model
        latent_prior = str(getattr(base_model, "latent_prior", "gaussian")).lower()
        match_mode = self.prior_match_mode
        if match_mode == "auto":
            match_mode = "mmd" if latent_prior == "boson" else "kl"

        if match_mode == "mmd":
            prior_match_loss, prior_match, match_count = self._mmd_prior_match(
                model, z, cond
            )
            elbo = log_px - self.beta * prior_match_loss
            loss = -elbo.mean()

            diag = {
                "elbo": elbo.detach(),
                "log_px": log_px.detach(),
                "prior_match": prior_match.detach().expand_as(elbo),
                "prior_match_samples": torch.full_like(elbo, float(match_count)),
            }
        else:
            if latent_prior == "boson":
                raise ValueError("The implicit Boson prior requires --prior_match_mode mmd.")
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
                        diag["athero_prob_sick"] = (1.0 - torch.prod(1.0 - prob, dim=1)).detach()
                    diag["athero_target_any"] = true_any.float().detach()

        if (
            self.w_binary_aux > 0.0
            and binary_target is not None
            and out.get("binary_logits") is not None
        ):
            logits = out["binary_logits"].reshape(-1)
            target = binary_target.float().reshape(-1)
            valid = torch.isfinite(target)
            if bool(valid.any()):
                pos_weight = torch.tensor(
                    self.binary_pos_weight,
                    device=logits.device,
                    dtype=logits.dtype,
                )
                per_sample = F.binary_cross_entropy_with_logits(
                    logits[valid],
                    target[valid],
                    reduction="none",
                    pos_weight=pos_weight,
                )
                loss = loss + self.w_binary_aux * per_sample.mean()
                probability = torch.sigmoid(logits[valid])
                diag["binary_loss"] = per_sample.detach()
                diag["binary_prob"] = probability.detach()
                diag["binary_target"] = target[valid].detach()
                diag["binary_acc"] = (
                    (probability >= 0.5) == (target[valid] >= 0.5)
                ).float().detach()

        if (
            self.w_regression_aux > 0.0
            and regression_target is not None
            and out.get("regression_pred") is not None
        ):
            pred = out["regression_pred"].reshape(-1)
            target = regression_target.float().reshape(-1)
            valid = torch.isfinite(target)
            if bool(valid.any()):
                objective = {"huber": F.smooth_l1_loss, "mse": F.mse_loss, "mae": F.l1_loss}[self.regression_loss]
                per_sample = objective(pred[valid], target[valid], reduction="none")
                loss = loss + self.w_regression_aux * per_sample.mean()
                diag["regression_loss"] = per_sample.detach()
                diag["regression_pred"] = pred[valid].detach()
                diag["regression_target"] = target[valid].detach()

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
