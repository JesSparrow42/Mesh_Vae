from __future__ import annotations

import torch

def _sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).t()
    return (x2 + y2 - 2.0 * (x @ y.t())).clamp_min(0.0)

def mmd_rbf(x: torch.Tensor, y: torch.Tensor, scales=(0.5, 1.0, 2.0, 4.0)) -> torch.Tensor:
    dxx = _sq_dist(x, x)
    dyy = _sq_dist(y, y)
    dxy = _sq_dist(x, y)

    kxx = 0.0
    kyy = 0.0
    kxy = 0.0
    for s in scales:
        gamma = 1.0 / (2.0 * s * s)
        kxx = kxx + torch.exp(-gamma * dxx)
        kyy = kyy + torch.exp(-gamma * dyy)
        kxy = kxy + torch.exp(-gamma * dxy)

    return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()


def mmd_imq(
    x: torch.Tensor,
    y: torch.Tensor,
    scales=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
) -> torch.Tensor:
    """Biased multi-scale inverse-multiquadratic MMD.

    The bandwidth scales with latent dimensionality, avoiding the vanishing
    RBF kernels produced by fixed widths in a 256-dimensional latent space.
    """
    dxx = _sq_dist(x, x)
    dyy = _sq_dist(y, y)
    dxy = _sq_dist(x, y)
    latent_dim = max(int(x.shape[1]), 1)
    kxx = torch.zeros_like(dxx)
    kyy = torch.zeros_like(dyy)
    kxy = torch.zeros_like(dxy)
    for scale in scales:
        c = 2.0 * float(latent_dim) * float(scale)
        kxx = kxx + c / (c + dxx)
        kyy = kyy + c / (c + dyy)
        kxy = kxy + c / (c + dxy)
    return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()


# ------------------------------------------
