"""Refactored MeshVAE package.

Heavy torch modules are loaded lazily so metadata and preprocessing utilities
can run in a lightweight Python environment.
"""

__all__ = [
    "MeshVAEModel",
    "get_scheduler",
    "MeshVAE",
    "MeshEncoder",
    "MeshDecoder",
    "MeshEdgeConvBlock",
    "chamfer_distance_subsample",
    "axis_std_loss",
    "repulsion_loss",
]


def __getattr__(name):
    if name in {"MeshVAEModel", "get_scheduler"}:
        from .model import MeshVAEModel, get_scheduler
        return {"MeshVAEModel": MeshVAEModel, "get_scheduler": get_scheduler}[name]
    if name in {"MeshVAE", "MeshEncoder", "MeshDecoder", "MeshEdgeConvBlock"}:
        from .networks import MeshVAE, MeshEncoder, MeshDecoder, MeshEdgeConvBlock
        return {
            "MeshVAE": MeshVAE,
            "MeshEncoder": MeshEncoder,
            "MeshDecoder": MeshDecoder,
            "MeshEdgeConvBlock": MeshEdgeConvBlock,
        }[name]
    if name in {"chamfer_distance_subsample", "axis_std_loss", "repulsion_loss"}:
        from .losses import chamfer_distance_subsample, axis_std_loss, repulsion_loss
        return {
            "chamfer_distance_subsample": chamfer_distance_subsample,
            "axis_std_loss": axis_std_loss,
            "repulsion_loss": repulsion_loss,
        }[name]
    raise AttributeError(name)
