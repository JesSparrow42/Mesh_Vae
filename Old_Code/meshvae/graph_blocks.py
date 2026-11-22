from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

def _apply_film(feat: torch.Tensor, film_layer: nn.Module, c_embed: torch.Tensor) -> torch.Tensor:
    """
    FiLM: feat -> feat * (1 + gamma) + beta
    feat: (B, C, E)
    c_embed: (B, D)
    film_layer outputs (B, 2C) => gamma,beta each (B,C)
    """
    gb = film_layer(c_embed)                 # (B, 2C)
    gamma, beta = gb.chunk(2, dim=1)         # (B, C), (B, C)
    return feat * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

# ==============================================================================
# Mesh Encoder Helpers
# ==============================================================================

def faces_to_edge_index(faces: torch.Tensor | None, num_nodes: int, device=None) -> torch.Tensor:
    if device is None:
        if faces is not None:
            device = faces.device
        else:
            device = torch.device("cpu")

    if num_nodes <= 1:
        idx = torch.zeros(1, dtype=torch.long, device=device)
        return torch.stack([idx, idx], dim=0)

    if faces is None or faces.numel() == 0:
        idx = torch.arange(num_nodes, device=device)
        src = idx[:-1]
        dst = idx[1:]
        return torch.cat([
            torch.stack([src, dst], dim=0),
            torch.stack([dst, src], dim=0),
        ], dim=1)

    tri = faces.long().view(-1, 3)
    e = torch.cat([
        tri[:, [0, 1]], tri[:, [1, 0]],
        tri[:, [1, 2]], tri[:, [2, 1]],
        tri[:, [2, 0]], tri[:, [0, 2]],
    ], dim=0)

    valid = (
        (e[:, 0] >= 0) & (e[:, 1] >= 0) &
        (e[:, 0] < num_nodes) & (e[:, 1] < num_nodes)
    )
    e = e[valid]

    if e.numel() == 0:
        idx = torch.arange(num_nodes, device=device)
        src = idx[:-1]
        dst = idx[1:]
        return torch.cat([
            torch.stack([src, dst], dim=0),
            torch.stack([dst, src], dim=0),
        ], dim=1)

    e = torch.unique(e, dim=0)
    return e.t().contiguous()


def knn_edge_index(points: torch.Tensor, k: int) -> torch.Tensor:
    n = int(points.shape[0])
    if n <= 1:
        idx = torch.zeros(1, dtype=torch.long, device=points.device)
        return torch.stack([idx, idx], dim=0)

    k_eff = max(1, min(int(k), n - 1))
    with torch.no_grad():
        d = torch.cdist(points, points)
        d.fill_diagonal_(float("inf"))
        nbr = d.topk(k_eff, largest=False).indices

    dst = torch.arange(n, device=points.device).repeat_interleave(k_eff)
    src = nbr.reshape(-1)

    e = torch.stack([src, dst], dim=0)
    rev = torch.stack([dst, src], dim=0)
    e = torch.cat([e, rev], dim=1)
    e = torch.unique(e.t(), dim=0).t().contiguous()
    return e


def merge_edge_index(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a is None or a.numel() == 0:
        return b
    if b is None or b.numel() == 0:
        return a
    return torch.unique(torch.cat([a, b], dim=1).t(), dim=0).t().contiguous()


class MeshEdgeConvBlock(nn.Module):
    """
    Message passing over mesh edges:
      msg(i <- j) = MLP([x_i, x_j - x_i, p_j - p_i])
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 3, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
        self.self_lin = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        if src.numel() == 0:
            out = self.self_lin(x)
            return F.relu(self.norm(out))

        rel_x = x[src] - x[dst]
        rel_p = pos[src] - pos[dst]
        msg_in = torch.cat([x[dst], rel_x, rel_p], dim=1)
        msg = self.msg_mlp(msg_in)

        out = torch.zeros(x.shape[0], msg.shape[1], device=x.device, dtype=msg.dtype)
        out.index_add_(0, dst, msg)

        deg = torch.bincount(dst, minlength=x.shape[0]).to(msg.dtype).unsqueeze(1).clamp(min=1.0)
        out = out / deg
        out = out + self.self_lin(x)
        out = self.norm(out)
        return F.relu(out)

# ==============================================================================
# Mesh encoder / decoder  (encoder stays as before, decoder fixed)
# ==============================================================================
