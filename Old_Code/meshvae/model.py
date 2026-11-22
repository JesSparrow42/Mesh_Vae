from __future__ import annotations

import os
import torch
from torch import nn
from torch.optim import lr_scheduler


from .networks import MeshVAE
from .vi import MeshVariationalInference

def get_scheduler(optimizer, opt):
    if opt.lr_policy == "lambda":
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(
                opt.niter_decay + 1
            )
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1
        )
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2,
            threshold=0.01, patience=5,
        )
    else:
        raise NotImplementedError(f"learning rate policy [{opt.lr_policy}] is not implemented")
    return scheduler


class MeshVAEModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gpu_ids = getattr(opt, "gpu_ids", [])
        self.is_train = getattr(opt, "is_train", True)

        self.register_buffer("_devref", torch.empty(0))
        device_target = torch.device(
            f"cuda:{self.gpu_ids[0]}"
            if self.gpu_ids and torch.cuda.is_available()
            else "cpu"
        )
        self._devref = self._devref.to(device_target)

        self.net = MeshVAE(
            in_channels=opt.input_nc,
            channels=list(opt.ncf),
            latent_dim=int(getattr(opt, "latent_dim", 64)),
            k=5,
            opt=opt,
        ).to(self.device)

        init_type = getattr(opt, "init_type", "xavier")
        init_gain = getattr(opt, "init_gain", 1.0)
        for m in self.net.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                if init_type == "xavier":
                    nn.init.xavier_normal_(m.weight, gain=init_gain)
                elif init_type == "normal":
                    nn.init.normal_(m.weight, 0.0, init_gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=init_gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        if isinstance(self.gpu_ids, (list, tuple)) and len(self.gpu_ids) > 1:
            self.net = nn.DataParallel(self.net, device_ids=self.gpu_ids)

        self.net.train(self.is_train)

        beta = float(getattr(opt, "beta", 1.0))
        use_chamfer = bool(getattr(opt, "use_pointcloud", False))
        chamfer_n = int(getattr(opt, "chamfer_npoints", 2048))
        aux_plaque_weight = (
            float(getattr(opt, "aux_plaque_weight", 0.0))
            if getattr(opt, "aux_predict_plaque", False)
            else 0.0
        )
        aux_athero_weight = (
            float(getattr(opt, "aux_athero_weight", 0.0))
            if getattr(opt, "aux_predict_athero", False)
            else 0.0
        )
        age_adv_weight = (
            float(getattr(opt, "adv_age_weight", 0.0))
            if (getattr(opt, "adv_predict_age", False) or float(getattr(opt, "adv_age_weight", 0.0)) > 0.0)
            else 0.0
        )

        self.vi = MeshVariationalInference(
            beta=beta,
            use_chamfer=use_chamfer,
            chamfer_npoints=chamfer_n,
            w_plaque_aux=aux_plaque_weight,
            w_athero_aux=aux_athero_weight,
            athero_pos_weight=float(getattr(opt, "aux_athero_pos_weight", 1.0)),
            w_age_adv=age_adv_weight,
        ).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.scheduler = get_scheduler(self.optimizer, opt)

        self.edge_features = None
        self.mesh = None
        self.mesh_batch = None
        self.cond = None
        self.plaque_target = None
        self.loss = torch.tensor(0.0, device=self.device)
        self.last_diag = {}
        self.task = "vae"

    @property
    def device(self):
        return self._devref.device

    def update_learning_rate(self) -> None:
        if not getattr(self, "is_train", False):
            return
        if not hasattr(self, "scheduler") or self.scheduler is None:
            print("[WARN] update_learning_rate called but no scheduler is set.", flush=True)
            return
        old_lr = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]["lr"]
        if lr != old_lr:
            print(f"learning rate: {old_lr:.7f} -> {lr:.7f}", flush=True)
        else:
            print(f"learning rate = {lr:.7f}", flush=True)

    def set_input(self, data: dict):
        def _to_dev(v, dtype=None):
            if v is None:
                return None
            if not isinstance(v, torch.Tensor):
                v = torch.from_numpy(v)
            if dtype is not None:
                v = v.to(dtype=dtype)
            return v.to(self.device)

        fe = data["edge_features"]
        if not isinstance(fe, torch.Tensor):
            fe = torch.from_numpy(fe).float()

        if fe.dim() == 4 and fe.shape[-1] == 1:
            fe = fe.squeeze(-1)
        elif fe.dim() != 3:
            raise ValueError(f"Unexpected edge_features shape {tuple(fe.shape)}")

        self.edge_features = fe.to(self.device)

        c = data.get("cond", None)
        if c is not None:
            if not isinstance(c, torch.Tensor):
                c = torch.from_numpy(c).float()
            c = c.to(self.device)
            if c.dim() == 1:
                c = c.unsqueeze(0)
            c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            self.cond = c
        else:
            self.cond = None

        self.plaque_target = _to_dev(data.get("plaque_target", None), dtype=torch.float32)
        if self.plaque_target is not None:
            self.plaque_target = torch.nan_to_num(
                self.plaque_target, nan=0.0, posinf=0.0, neginf=0.0
            )

        verts = _to_dev(data.get("verts", None), dtype=torch.float32)
        faces = _to_dev(data.get("faces", None), dtype=torch.long)
        verts_mask = _to_dev(data.get("verts_mask", None), dtype=torch.bool)
        faces_mask = _to_dev(data.get("faces_mask", None), dtype=torch.bool)
        if verts is not None:
            verts = torch.nan_to_num(verts, nan=0.0, posinf=0.0, neginf=0.0)

        self.mesh_batch = {
            "verts": verts,
            "faces": faces,
            "verts_mask": verts_mask,
            "faces_mask": faces_mask,
        }

        self.mesh = data.get("mesh", None)

        if not torch.isfinite(self.edge_features).all():
            self.edge_features = torch.nan_to_num(
                self.edge_features, nan=0.0, posinf=0.0, neginf=0.0
            )

    def forward(self):
        return self.net(self.edge_features, self.mesh_batch, cond=self.cond)

    def backward(self, outputs: dict):
        loss, _, _ = self.vi(
            self.net,
            self.edge_features,
            self.mesh_batch,
            self.cond,
            plaque_target=self.plaque_target,
        )
        self.loss = loss
        self.loss.backward()

    def optimize_parameters(self):
        self.net.train(True)
        self.optimizer.zero_grad(set_to_none=True)

        use_amp = bool(getattr(self.opt, "use_amp", False) and torch.cuda.is_available())

        if use_amp:
            if not hasattr(self, "scaler"):
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler(enabled=True)

            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                loss, diagnostics, _ = self.vi(
                    self.net,
                    self.edge_features,
                    self.mesh_batch,
                    self.cond,
                    plaque_target=self.plaque_target,
                )

            if not torch.isfinite(loss):
                print("[ERR] non-finite loss; skipping batch:", float(loss))
                return

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.loss = loss.detach()
            self.last_diag = {k: float(v.mean().detach().cpu()) for k, v in diagnostics.items()}
            self.last_diag["loss"] = float(self.loss.detach().cpu())
            self.last_diag["lr"] = float(self.optimizer.param_groups[0]["lr"])
        else:
            loss, diagnostics, _ = self.vi(
                self.net,
                self.edge_features,
                self.mesh_batch,
                self.cond,
                plaque_target=self.plaque_target,
            )

            if not torch.isfinite(loss):
                print("[ERR] non-finite loss; skipping batch:", float(loss))
                return

            self.loss = loss
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.last_diag = {k: float(v.mean().detach().cpu()) for k, v in diagnostics.items()}
            self.last_diag["loss"] = float(self.loss.detach().cpu())
            self.last_diag["lr"] = float(self.optimizer.param_groups[0]["lr"])

    @torch.no_grad()
    def test(self):
        self.net.eval()
        loss, diagnostics, outputs = self.vi(
            self.net,
            self.edge_features,
            self.mesh_batch,
            self.cond,
            plaque_target=self.plaque_target,
        )
        self.last_diag = {k: float(v.mean().detach().cpu()) for k, v in diagnostics.items()}
        self.last_diag["loss"] = float(loss.detach().cpu())

        if hasattr(self, "is_train") and self.is_train:
            self.net.train()

        return {
            "loss": loss.detach().item(),
            "diagnostics": diagnostics,
            "nexamples": int(self.edge_features.shape[0]),
            "x": self.edge_features.detach().cpu(),
            "x_mean": (
                outputs.get("x_mean").detach().cpu()
                if isinstance(outputs, dict) and outputs.get("x_mean") is not None
                else None
            ),
            "plaque_pred": (
                outputs.get("plaque_pred").detach().cpu()
                if isinstance(outputs, dict) and outputs.get("plaque_pred") is not None
                else None
            ),
            "athero_logits": (
                outputs.get("athero_logits").detach().cpu()
                if isinstance(outputs, dict) and outputs.get("athero_logits") is not None
                else None
            ),
            "athero_prob": (
                torch.sigmoid(outputs.get("athero_logits")).detach().cpu()
                if isinstance(outputs, dict) and outputs.get("athero_logits") is not None
                else None
            ),
            "age_adv_logits": (
                outputs.get("age_adv_logits").detach().cpu()
                if isinstance(outputs, dict) and outputs.get("age_adv_logits") is not None
                else None
            ),
        }

    def save_network(self, which_epoch: str):
        fname = f"{which_epoch}_vae_net.pth"
        path = os.path.join(self.opt.checkpoints_dir, self.opt.name, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.gpu_ids and torch.cuda.is_available():
            state = (
                self.net.module.cpu().state_dict()
                if isinstance(self.net, nn.DataParallel)
                else self.net.cpu().state_dict()
            )
            torch.save(state, path)
            self.net.to(self.device)
        else:
            torch.save(self.net.state_dict(), path)

    def load_network(self, which_epoch: str):
        fname = f"{which_epoch}_vae_net.pth"
        path = os.path.join(self.opt.checkpoints_dir, self.opt.name, fname)
        state = torch.load(path, map_location=str(self.device))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(state)
        else:
            self.net.load_state_dict(state)
