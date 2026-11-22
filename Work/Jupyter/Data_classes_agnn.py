"""Data object classes."""

from collections.abc import Sequence

import torch


class BaseData:
    """A dict-like base class for data objects.

    Store all tensors in a dict for easy access and enumeration.
    """

    def __init__(self, **kwargs):
        # TODO: Setting return type None here reveals a series of type issues with int or tensor.
        self.tensors = dict()
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __getattr__(self, key):
        # try to get from self.tensors and get everything else as normal
        try:
            return self.tensors[key]
        except KeyError:
            return super().__getattr__(key)

    def __setattr__(self, key, value) -> None:
        # store tensors in self.tensors and everything else in self.__dict__
        # while ensuring the same key is not in both self.tensors and self.__dict__
        if isinstance(value, torch.Tensor):
            self.tensors[key] = value
            self.__dict__.pop(key, None)  # remove key from __dict__ if it exists
        else:
            super().__setattr__(key, value)
            self.tensors.pop(key, None)  # remove key from tensors if it exists

    def __getstate__(self) -> dict:
        # pickle
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        # unpickle
        self.__dict__ = state

    def validate(self) -> bool:
        """Validate the data."""
        for key, tensor in self.tensors.items():
            assert isinstance(tensor, torch.Tensor), f"'{key}' is not a tensor!"
        return True

    def to(self, device: torch.device) -> None:
        """Move all tensors to the given device."""
        self.tensors = {k: v.to(device) for k, v in self.tensors.items()}


class Data(BaseData):
    """A data object describing a homogeneous graph.

    Includes general graph information about: nodes, edges, target labels and global features.
    """

    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor = torch.tensor([]),
        edge_features: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        global_features: torch.Tensor | None = None,
        **kwargs
    ) -> None:
        """Initialize the data object.

        Args:
            node_features: Node feature tensor with shape (num_nodes, num_node_features).
            edge_index: Edge index (adjacency list) tensor with shape (num_edges, 2).
                        Each edge has the form [source_node_index, target_node_index].
            edge_features: Edge feature tensor with shape (num_edges, num_edge_features).
            targets: Graph-level or node-level target tensor with arbitrary shape.
            global_features: Graph-level features.
        """
        super().__init__(**kwargs)
        # nodes
        self.node_features = node_features
        # TODO: could add another type of node feature specifically for embedding
        # (node_embed_index or node_type)
        # edges
        self.edge_index = edge_index
        self.edge_features = edge_features
        # global
        self.global_features = global_features
        # arbitrary
        self.targets = targets
        # TODO: could also do node_targets, edge_targets, global_targets
        # Use kwargs to the tensors dict for arbitrary properties

    def validate(self) -> bool:
        """Validate the data."""
        super().validate()
        assert self.num_nodes > 0
        assert self.node_features.shape[0] == self.num_nodes
        assert self.node_features.ndim >= 2
        assert self.edge_index.shape[0] == self.num_edges
        assert self.edge_index.shape[0] == 0 or self.edge_index.shape[1] == 2
        assert self.edge_index.shape[0] == 0 or self.edge_index.max() < self.num_nodes
        if self.edge_features is not None:
            assert self.edge_features.shape[0] == self.num_edges
            assert self.num_edges == 0 or self.edge_features.ndim >= 2
        return True

    @property
    def num_nodes(self) -> torch.Tensor:
        """Get number of nodes in the data."""
        # try to get num_nodes from tensors, else get it from node_features
        return self.tensors.get("num_nodes", self.node_features.shape[0])

    @property
    def num_edges(self) -> torch.Tensor:
        """Get number of edges in the data."""
        # try to get num_edges from tensors, else get it from edge_index
        return self.tensors.get("num_edges", self.edge_index.shape[0])

    @property
    def edge_index_source(self) -> torch.Tensor:
        """Get indices of source (sender) nodes."""
        return self.edge_index[:, 0]

    @property
    def edge_index_target(self) -> torch.Tensor:
        """Get indices of target (receiver) nodes."""
        return self.edge_index[:, 1]


class AtomsData(Data):
    """A data object describing atoms as a graph with spatial information.

    The AtomsData object can represent an isolated molecule, or a periodically repeated structure.
    Loosely based on: https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms
    """

    def __init__(
        self,
        node_positions: torch.Tensor,
        energy: torch.Tensor | None = None,
        forces: torch.Tensor | None = None,
        magmoms: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        volume: torch.Tensor | None = None,
        stress: torch.Tensor | None = None,
        pbc: torch.Tensor | None = None,
        edge_shift: torch.Tensor | None = None,
        **kwargs
    ) -> None:
        """Initialize the data object.

        Args:
            node_positions: Tensor of node positions with shape (num_nodes, num_dimensions).
            energy: Tensor of the total potential energy with shape (1,).
            forces: Tensor of the forces acting on the atoms with same shape as node_positions.
            magmoms: Tensor of the magnetic moments with shape (num_nodes, 1).
            cell: Tensor of the simulation cell with shape (num_dimensions, num_dimensions).
            volume: Tensor of the cell volume with shape (1,).
            stress: Tensor of the stress with shape (num_dimensions, num_dimensions).
            pbc: Tensor of the periodic boundary conditions with shape (num_dimensions,).
            edge_shift: Tensor of cell shift vectors with shape (num_edges, num_dimensions).
        """
        super().__init__(**kwargs)
        self.node_positions = node_positions
        self.energy = energy
        self.forces = forces  # TODO: could add option to init forces with zeros
        self.magmoms = magmoms
        self.cell = cell
        self.volume = volume
        self.stress = stress
        self.pbc = pbc
        self.edge_shift = edge_shift

    def validate(self) -> bool:
        """Validate the data."""
        super().validate()
        assert self.node_positions.shape[0] == self.num_nodes
        assert self.node_positions.ndim == 2
        spatial_dim = self.node_positions.shape[1]
        if self.energy is not None:
            assert self.energy.shape == (1,)
        if self.forces is not None:
            assert self.forces.shape == (self.num_nodes, spatial_dim)
        if self.magmoms is not None:
            assert self.magmoms.shape == (self.num_nodes, 1)
        if self.cell is not None or self.pbc is not None:
            assert self.cell is not None
            assert self.pbc is not None
            assert self.cell.shape == (spatial_dim, spatial_dim)
            assert self.pbc.shape == (spatial_dim,)
        if self.volume is not None:
            assert self.cell is not None
            assert self.volume.shape == (1,)
            assert torch.isclose(self.volume, torch.linalg.det(self.cell))
        if self.stress is not None:
            assert self.cell is not None
            assert self.stress.shape == (spatial_dim, spatial_dim)
        if self.edge_shift is not None:
            assert self.edge_index is not None
            assert self.edge_shift.shape == (self.num_edges, spatial_dim)
        return True

    def any_pbc(self) -> bool:
        """Return True if any of the periodic boundary conditions are True."""
        return self.pbc is not None and bool(torch.any(self.pbc))


class GeometricData(Data):
    """A data object describing a geometric graph with spatial information."""

    def __init__(
        self,
        node_positions: torch.Tensor,
        node_velocities: torch.Tensor | None = None,
        node_accelerations: torch.Tensor | None = None,
        **kwargs
    ) -> None:
        """Initialize the data object.

        Args:
            node_positions: Tensor of node positions with shape (num_nodes, num_dimensions).
            node_velocities: Tensor of node velocities with same shape as node_positions.
            node_accelerations: Tensor of node accelerations with same shape as node_positions.
        """
        super().__init__(**kwargs)
        self.node_positions = node_positions
        self.node_velocities = node_velocities
        self.node_accelerations = node_accelerations

    def validate(self) -> bool:
        """Validate the data."""
        super().validate()
        assert self.node_positions.shape[0] == self.num_nodes
        assert self.node_positions.ndim == 2
        spatial_dim = self.node_positions.shape[1]
        if self.node_velocities is not None:
            assert self.node_velocities.shape[0] == self.num_nodes
            assert self.node_velocities.shape[1] == spatial_dim
        if self.node_accelerations is not None:
            assert self.node_accelerations.shape[0] == self.num_nodes
            assert self.node_accelerations.shape[1] == spatial_dim
        return True


# Batch


class Batch(Data):
    """An object representing a batch of data.

    Typically a disjoint union of graphs. See collate_data function.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the batch object."""
        super().__init__(**kwargs)
        self._node_data_index: torch.Tensor | None = None
        self._edge_data_index: torch.Tensor | None = None

    def validate(self) -> bool:
        """Validate the data."""
        for key, tensor in self.tensors.items():
            assert isinstance(tensor, torch.Tensor), f"'{key}' is not a tensor!"
        assert self.node_features.shape[0] == torch.sum(self.num_nodes)
        assert self.node_features.shape[0] == 0 or self.node_features.ndim >= 2
        assert self.edge_index.shape[0] == torch.sum(self.num_edges)
        assert self.edge_index.shape[0] == 0 or self.edge_index.shape[1] == 2
        assert self.edge_index.shape[0] == 0 or self.edge_index.max() < self.node_features.shape[0]
        assert self.num_data >= 1
        if self.edge_features is not None:
            assert self.edge_features.shape[0] == self.edge_index.shape[0]
            assert self.edge_features.ndim >= 2
        if self.global_features is not None:
            assert self.global_features.shape[0] == self.num_data
        if self.targets is not None:
            assert self.targets.ndim >= 2
        return True

    @property
    def num_data(self) -> int:
        """Get the number of data in the batch."""
        return self.num_nodes.shape[0]

    @property
    def node_data_index(self) -> torch.Tensor:
        """Get the data index of each node in the batch."""
        if self._node_data_index is None:
            self._node_data_index = torch.repeat_interleave(
                torch.arange(self.num_nodes.shape[0], device=self.num_nodes.device), self.num_nodes)
        return self._node_data_index

    @property
    def edge_data_index(self) -> torch.Tensor:
        """Get the data index of each edge in the batch."""
        if self._edge_data_index is None:
            self._edge_data_index = torch.repeat_interleave(
                torch.arange(self.num_edges.shape[0], device=self.num_edges.device), self.num_edges)
        return self._edge_data_index


def collate_data(list_of_data: Sequence[Data]) -> Batch:
    """Collate a list of data objects into a batch object.

    The input graphs are combined into a single graph as a disjoint union by
    concatenation of all data and appropriate adjustment of the edge_index.

    It is assumed that all the input data objects have the same keys/tensors
    and that tensor shapes line up.

    This function can be used with torch.utils.data.DataLoader.

    Args:
        list_of_data: A list of data objects.
    Returns:
        A Batch object representing a disjoint union of the input data.
    """
    batch = dict()
    # Add num_nodes and num_edges to the batch
    batch["num_nodes"] = torch.tensor([d.num_nodes for d in list_of_data])
    batch["num_edges"] = torch.tensor([d.num_edges for d in list_of_data])
    # Compute edge index offset from num_nodes
    offset = torch.cumsum(batch["num_nodes"], dim=0) - batch["num_nodes"]
    # Add offset edge_index to the batch
    batch["edge_index"] = torch.cat([d.edge_index + offset[i] for i, d in enumerate(list_of_data)])
    # Concatenate and include remaining tensors in the batch
    for k in list_of_data[0].tensors.keys():  # Use keys from the first data object
        if k not in batch.keys():  # Avoid overwriting for example the edge_index
            try:
                if k == "cell" or k == "stress":  # Add batch dimension to cell and stress tensors
                    batch[k] = torch.cat([d.tensors[k].unsqueeze(0) for d in list_of_data])
                else:
                    batch[k] = torch.cat([torch.atleast_2d(d.tensors[k]) for d in list_of_data])
            except Exception as e:
                raise Exception(f"Failed to add '{k}' to batch:", e)
    # Create a Batch object from the batch dict and return
    return Batch(**batch)