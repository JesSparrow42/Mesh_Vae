# ---------------------------------------------------------------------
# mesh_vae_model.py  (VAE section)  — Gaussian/Boson-selectable prior
# ---------------------------------------------------------------------

from typing import List, Tuple, Optional
import math
import torch
from torch import nn
from torch.distributions import Independent, Normal

# -- Optional Boson imports (guarded) ---------------------------------
_BOSON_AVAILABLE = False
try:
    # user-provided modules
    from GenAI.utils import BosonPrior
    from GenAI.BosonSamplerWrapper import BosonLatentGenerator
    _BOSON_AVAILABLE = True
except Exception:
    # we stay quiet; we will warn at runtime only if user selects boson
    _BOSON_AVAILABLE = False


# =====================================================================
# MeshVAE: unchanged encoder/decoder API; prior() can be Gaussian or Boson
# =====================================================================
class MeshVAE(nn.Module):
    """
    End-to-end VAE on meshes. Learns p(x|z) as a factorized Normal over edge features.
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        pool_targets: List[int],
        latent_dim: int,
        obs_log_std: float = math.log(0.1),
        k: int = 5,
        opt: Optional[object] = None,
    ):
        super().__init__()

        # ---- config / options we pass through from MeshVAEModel ----
        self.latent_dim = int(latent_dim)

        # prior selector (NEW)
        # - 'gaussian' (default)
        # - 'boson' (requires GenAI.* modules and boson params)
        self.latent_prior: str = getattr(opt, "latent_prior", "gaussian").lower() if opt else "gaussian"
        self.boson_params = getattr(opt, "boson_sampler_params", None) if opt else None

        # construct (optional) boson sampler
        self.boson_sampler = None
        if self.latent_prior == "boson":
            if not _BOSON_AVAILABLE:
                raise ImportError(
                    "latent_prior='boson' requested, but BosonPrior/BosonLatentGenerator "
                    "could not be imported. Check your GenAI.* modules."
                )
            # mirror your reference code: BosonLatentGenerator(latent_features, params)
            self.boson_sampler = BosonLatentGenerator(self.latent_dim, self.boson_params)
            print("[VAE] Using BOSON latent prior.", flush=True)
        else:
            print("[VAE] Using GAUSSIAN latent prior.", flush=True)

        # ---- encoder/decoder backbone (YOUR existing conv mesh nets) ----
        # NOTE: we don’t touch your existing MeshEncoder / MeshDecoder signatures.
        #       They should be defined above in this file as before.
        self.encoder = MeshEncoder(in_channels, channels, pool_targets, self.latent_dim, k=k, opt=opt)
        self.decoder = MeshDecoder(in_channels, channels, pool_targets, self.latent_dim, k=k, opt=opt)

        # observation scale (as before)
        self.obs_log_std = nn.Parameter(torch.tensor(float(obs_log_std)), requires_grad=False)

    # --------- small helper: diagonal Gaussian with reparam -----------
    @staticmethod
    def _diag_normal(mu: torch.Tensor, log_sigma: torch.Tensor) -> Independent:
        sigma = torch.exp(log_sigma)
        # interpret sum across last dim (latent features) for q(z|x) and across C*E for px
        return Independent(Normal(mu, sigma), 1)

    # ------------------------- q(z|x) ---------------------------------
    def posterior(self, x: torch.Tensor, mesh: list) -> Tuple[Independent, torch.Tensor, int]:
        mu, log_sigma, start_E = self.encoder(x, mesh)  # keep your encoder outputs
        qz = self._diag_normal(mu, log_sigma)
        return qz, mu, start_E

    # ------------------------- p(z) -----------------------------------
    def prior(self, batch: int, latent_dim: int, device: torch.device) -> Independent:
        if self.latent_prior == "boson":
            # returns Distribution over z with log_prob; sampling used only if you later need prior samples
            # BosonPrior will internally use the BosonLatentGenerator
            return BosonPrior(
                boson_sampler=self.boson_sampler,
                batch_size=batch,
                latent_features=latent_dim,
            )
        # Gaussian baseline (exactly what you had before)
        mu = torch.zeros(batch, latent_dim, device=device)
        log_sigma = torch.zeros(batch, latent_dim, device=device)
        return self._diag_normal(mu, log_sigma)

    # ------------------------ p(x|z) ----------------------------------
    def likelihood(self, x_mean: torch.Tensor) -> Independent:
        # treat C*E as factorized dims (sum over last 2 dims)
        std = torch.exp(self.obs_log_std)
        return Independent(Normal(loc=x_mean, scale=std), reinterpreted_batch_ndims=2)

    # ------------------------- forward --------------------------------
    def forward(self, x: torch.Tensor, mesh: list):
        # --- normalize legacy shapes -> [B, C, E] ---
        x_in = x.squeeze(-1) if (x.dim() == 4 and x.shape[-1] == 1) else x
        B, C, E0 = x_in.shape
        print(f"[VAE] x_mean_? input check, B={B}, C={C}, E0={E0}", flush=True)

        # q(z|x)
        qz, mu, start_E = self.posterior(x_in, mesh)
        z = qz.rsample()

        # let decoder know the effective starting edges E
        self.decoder.start_edges = int(start_E)
        if hasattr(self.decoder, "unpools") and len(self.decoder.unpools) > 0:
            # last unpool overshoots back to original E0 when no history
            self.decoder.unpools[-1].unroll_target = int(E0)

        # decode to x_mean and normalize to [B, C, E]
        x_mean = self.decoder(z, mesh, start_E)
        if x_mean.dim() == 4 and x_mean.shape[-1] == 1:
            x_mean = x_mean.squeeze(-1)

        print(f"[VAE] x_mean => {tuple(x_mean.shape)}", flush=True)

        # p(x|z)
        px = self.likelihood(x_mean)           # Independent Normal over [C, E]

        # p(z)
        pz = self.prior(B, z.shape[-1], device=x.device)

        # hand back everything the VI block needs
        return {"px": px, "pz": pz, "qz": qz, "z": z}


# =====================================================================
# MeshVariationalInference: unchanged math; works for both priors
# =====================================================================
class MeshVariationalInference(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = float(beta)

    def forward(self, model: MeshVAE, x: torch.Tensor, mesh: list):
        """
        x:     [B, C, E] (or [B, C, E, 1])
        mesh:  list of length B
        """
        out = model(x, mesh)
        px, pz, qz, z = out["px"], out["pz"], out["qz"], out["z"]

        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        # log-likelihoods aggregated to [B] by our Independent dists
        log_px = px.log_prob(x)      # [B]
        log_pz = pz.log_prob(z)      # [B]
        log_qz = qz.log_prob(z)      # [B]

        kl = (log_qz - log_pz)       # [B]
        elbo = log_px - self.beta * kl
        loss = -elbo.mean()

        diag = {
            "elbo": elbo.detach(),
            "log_px": log_px.detach(),
            "kl": kl.detach(),
        }
        return loss, diag, out


# =====================================================================
# LR scheduler helper (unchanged from your version)
# =====================================================================
def get_scheduler(optimizer, opt):
    from torch.optim import lr_scheduler
    if opt.lr_policy == "lambda":
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                   threshold=0.01, patience=5)
    else:
        return NotImplementedError(f"learning rate policy [{opt.lr_policy}] is not implemented")
    return scheduler


# =====================================================================
# MeshVAEModel: identical external behavior; we just pass new knobs
# =====================================================================
class MeshVAEModel:
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device(f"cuda:{opt.gpu_ids[0]}") if opt.gpu_ids and torch.cuda.is_available() \
                      else torch.device("cpu")

        # keep your original names
        C_in  = int(opt.input_nc)        # edge feature channels
        widths = list(opt.ncf)           # e.g. [64, 128, 256, 256]
        pools  = list(opt.pool_res)      # e.g. [600, 450, 300, 180]
        z_dim  = int(getattr(opt, "latent_dim", 64))
        beta   = float(getattr(opt, "beta", 1.0))

        # build the VAE and move to device
        self.net = MeshVAE(
            in_channels=opt.input_nc,
            channels=list(opt.ncf),
            pool_targets=list(opt.pool_res),
            latent_dim=int(opt.latent_dim),
            k=5,
            opt=opt,                      # <- critical so prior config is visible
        ).to(self.device)

        self.vi = MeshVariationalInference(beta=beta).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = get_scheduler(self.optimizer, opt)

        # runtime slots (unchanged)
        self.edge_features = None
        self.mesh = None
        self.loss = torch.tensor(0.0, device=self.device)
        self.task = "vae"

    # ----------------- train step plumbing (unchanged) -----------------
    def update_learning_rate(self) -> None:
        """Update LR once per epoch (mirrors ClassifierModel)."""
        if not getattr(self, "is_train", False):
            return
        if not hasattr(self, "scheduler") or self.scheduler is None:
            print("[WARN] update_learning_rate called but no scheduler is set.", flush=True)
            return
        old_lr = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()                                     # step scheduler
        lr = self.optimizer.param_groups[0]["lr"]                 # readback for print
        if lr != old_lr:
            print(f"learning rate: {old_lr:.7f} -> {lr:.7f}", flush=True)
        else:
            print(f"learning rate = {lr:.7f}", flush=True)

    # --------------------------- I/O -----------------------------------
    def set_input(self, data: dict):
        """
        Expects data['edge_features']: numpy or tensor, shape [B, C, E] or [B, C, E, 1]
                data['mesh']:        list of B Mesh objects
        """
        fe = torch.from_numpy(data['edge_features']).float() if isinstance(data['edge_features'], (list, tuple)) is False else data['edge_features']
        if fe.dim() == 3:
            pass
        elif fe.dim() == 4 and fe.shape[-1] == 1:
            fe = fe.squeeze(-1)
        else:
            raise ValueError(f"Unexpected edge_features shape {tuple(fe.shape)}")
        self.edge_features = fe.to(self.device)
        self.mesh = data['mesh']  # list; MeshPool/Unpool operate on these Python objects

    # ------------------------- fwd/bwd ---------------------------------
    def forward(self):
        return self.net(self.edge_features, self.mesh)

    def backward(self, outputs: dict):
        loss, _, _ = self.vi(self.net, self.edge_features, self.mesh)
        self.loss = loss
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.forward()
        self.backward(outputs)
        self.optimizer.step()

    # -------------------------- eval / test ----------------------------
    @torch.no_grad()
    def test(self):
        # forward pass
        loss, diagnostics, outputs = self.vi(self.net, self.edge_features, self.mesh)
        return {
            "loss": loss.detach().item(),
            "diagnostics": diagnostics,
            "nexamples": self.edge_features.shape[0],
        }

    # ---------------------- save / load net ----------------------------
    def save_network(self, which_epoch: str):
        fname = f"{which_epoch}_vae_net.pth"
        path = os.path.join(self.opt.checkpoints_dir, self.opt.name, fname)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.gpu_ids and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict() if isinstance(self.net, nn.DataParallel) else self.net.cpu().state_dict(), path)
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
