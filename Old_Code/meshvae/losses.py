from __future__ import annotations

import torch
import torch.nn.functional as F

def _subsample_points(P: torch.Tensor, n: int) -> torch.Tensor:
    """
    P: (B, N, 3)
    Returns: (B, n, 3) (with replacement if N < n)
    """
    B, N, _ = P.shape
    if N == n:
        return P
    if N > n:
        idx = torch.randint(0, N, (B, n), device=P.device)
        return P.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
    # N < n: sample with replacement
    idx = torch.randint(0, N, (B, n), device=P.device)
    return P.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))


def chamfer_distance_subsample(x: torch.Tensor, x_mean: torch.Tensor, n_points: int = 2048, chunk: int = 1024) -> torch.Tensor:
    """
    Order-invariant Chamfer distance between point clouds.
    x, x_mean: (B, 3, E)
    Returns: (B,) chamfer = mean_nn(x->x_mean) + mean_nn(x_mean->x)
    """
    assert x.dim() == 3 and x_mean.dim() == 3, f"expected (B,3,E), got {x.shape} and {x_mean.shape}"
    assert x.shape[1] == 3 and x_mean.shape[1] == 3, "Chamfer expects xyz channels (C=3)"

    # (B, E, 3)
    A = x.permute(0, 2, 1).contiguous()
    B = x_mean.permute(0, 2, 1).contiguous()

    # subsample to keep cdist reasonable
    A = _subsample_points(A, n_points)
    B = _subsample_points(B, n_points)

    # A -> B
    Bsz, Na, _ = A.shape
    Nb = B.shape[1]

    mins_ab = []
    for s in range(0, Na, chunk):
        e = min(s + chunk, Na)
        d = torch.cdist(A[:, s:e, :], B)          # (B, chunk, Nb)
        mins_ab.append(d.min(dim=2).values ** 2)
    d_ab = torch.cat(mins_ab, dim=1).mean(dim=1)  # (B,)

    # B -> A
    mins_ba = []
    for s in range(0, Nb, chunk):
        e = min(s + chunk, Nb)
        d = torch.cdist(B[:, s:e, :], A)          # (B, chunk, Na)
        mins_ba.append(d.min(dim=2).values ** 2)
    d_ba = torch.cat(mins_ba, dim=1).mean(dim=1)  # (B,)

    return d_ab + d_ba

def axis_std_loss(x: torch.Tensor, x_mean: torch.Tensor) -> torch.Tensor:
    """
    Penalize collapse by matching per-axis spread (std) between target and recon.
    x, x_mean: (B, 3, E)
    returns: (B,)
    """
    A = x.permute(0, 2, 1)       # (B, E, 3)
    B = x_mean.permute(0, 2, 1)  # (B, E, 3)
    std_A = A.std(dim=1)         # (B, 3)
    std_B = B.std(dim=1)         # (B, 3)
    return ((std_A - std_B) ** 2).mean(dim=1)  # (B,)

def repulsion_loss(x_mean: torch.Tensor, n_points: int = 1024, k: int = 8, radius: float = 0.02) -> torch.Tensor:
    """
    Encourage points to not collapse into a line/cluster by penalizing too-close neighbors.
    x_mean: (B, 3, E)
    returns: (B,)
    """
    P = x_mean.permute(0, 2, 1).contiguous()   # (B, E, 3)
    P = _subsample_points(P, n_points)         # (B, n, 3)

    # Pairwise distances
    D = torch.cdist(P, P)                      # (B, n, n)
    # Ignore self-distance
    n = D.shape[-1]
    eye = torch.eye(n, device=D.device).unsqueeze(0)
    D = D + eye * 1e6

    # kNN distances (smallest k)
    knn = D.topk(k, dim=-1, largest=False).values  # (B, n, k)

    # Penalize distances smaller than radius
    pen = F.relu(radius - knn) ** 2                # (B, n, k)
    return pen.mean(dim=(1, 2))                    # (B,)
