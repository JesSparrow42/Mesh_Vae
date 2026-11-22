import math
import torch

@torch.jit.script
def reduce_splits(
    values: torch.Tensor,
    splits: torch.Tensor,
    out: torch.Tensor | None = None,
    reduction: str = "sum",
) -> torch.Tensor:
    """Reduce values in splits.

    Can be used to reduce from node-level values to graph-level values in a batch
    when values are node_states and splits is a tensor of num_nodes.

     Args:
        values: The input values to be reduced with shape (num_values, ...).
        splits: The split sizes with shape (num_splits,) and sum(splits) = num_values.
        out: The tensor used to store the output with shape (num_splits, ...).
        reduction: The reduction method.
    Returns:
        A tensor with size (num_splits, ...).
    """
    if reduction == "sum":
        return sum_splits(values, splits, out=out)
    elif reduction == "mean":
        return mean_splits(values, splits, out=out)
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")


# Reduce with index


@torch.jit.script
def sum_index(
    values: torch.Tensor,
    index: torch.Tensor,
    out: torch.Tensor | None = None,
    num_out: int = 0
) -> torch.Tensor:
    """Sum values by index into out.

    If out is not provided, it is created with shape (num_out, ...).

    Args:
        values: The input values to be reduced with shape (num_values, ...).
        index: The indices in the output each value is reduced to (num_values,).
        out: The tensor used to store the output with shape (num_out, ...).
        num_out: The first dimension of the output tensor if out is None.
    Returns:
        A tensor with size (num_out, ...).
    """
    assert out is not None or num_out > 0
    if out is None:
        out_shape = torch.Size([num_out]) + values.shape[1:]
        out = torch.zeros(out_shape, dtype=values.dtype, device=values.device)
    out.index_add_(0, index, values)
    return out

@torch.jit.script
def compute_edge_vectors_and_norms(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor | None = None,
    edge_cell: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute edge vectors and norms from positions.

    Consider periodic boundary conditions (pbc) if edge_shift and edge_cell are provided.

    Args:
        positions: (num_nodes, num_dimensions)
        edge_index: Edge index (adjacency list) with shape (num_edges, 2)
        edge_shift: Cell shift tensor for each edge (num_edges, num_dimensions).
        edge_cell: Cell tensor for each edge (num_edges, num_dimensions, num_dimensions).
    Returns:
        Tensor of edge vectors (num_edges, num_dimensions) and norms (num_edges, 1).
    """
    source_positions = positions[edge_index[:, 0]]
    target_positions = positions[edge_index[:, 1]]
    if edge_shift is not None:
        assert edge_cell is not None
        shift = torch.squeeze(edge_shift.unsqueeze(1) @ edge_cell, dim=1)
        target_positions = target_positions + shift
    vectors = target_positions - source_positions
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
    return vectors, norms


class SparseOneHot(torch.nn.Module):
    """Sparse one-hot encoding.

    One-hot encoding that only keeps the relevant classes.
    Useful when some classes in [0, class_index_max] are not in the data.
    """

    def __init__(self, classes: torch.Tensor):
        """Initialize SparseOneHot.

        Args:
            classes: Tensor of unique class indices with shape (num_classes, ).
        """
        super().__init__()
        assert classes.dim() == 1
        assert len(set(list(classes))) == len(classes)  # Must be unique
        self.register_buffer("classes", classes)
        self.num_classes_hidden = int(classes.max() + 1)
        self.num_classes = len(classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One-hot encode input tensor.

        Args:
            x: Input tensor with class indices of shape (num_nodes, ).
        Returns:
            One-hot encoded tensor of shape (num_nodes, num_classes)."""
        assert torch.all(torch.isin(x, self.classes))
        z = torch.nn.functional.one_hot(x.squeeze(), num_classes=self.num_classes_hidden)
        z = torch.atleast_2d(z)
        z = z[:, self.classes]  # Only keep the specified class indices
        z = z.to(device=x.device, dtype=torch.get_default_dtype())
        assert z.shape == (x.shape[0], self.num_classes)
        return z


class BesselExpansion(torch.nn.Module):
    """Expand distances with Bessel radial basis functions (RBFs).

    Resources:
        DimeNet: https://arxiv.org/abs/2003.03123
        PaiNN: https://arxiv.org/abs/2102.03150
        NequIP: https://www.nature.com/articles/s41467-022-29939-5
    """

    def __init__(self, size: int, cutoff: float = 5.0, trainable: bool = False) -> None:
        """Initialize the Bessel expansion.

        Args:
            size: Size of the expansion.
            cutoff: Cutoff distance.
            trainable: Train the weights.
        """
        super().__init__()
        self.size = size
        self.b_pi_over_c = torch.nn.parameter.Parameter(
            (torch.arange(size) + 1) * math.pi / cutoff, requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the expansion.

        Args:
            x: Tensor of non-negative input values with shape (num_values, 1).
        Returns:
            Tensor with shape (num_values, size).
        """
        x = x + 1e-10  # Avoid division by zero
        return torch.sin(self.b_pi_over_c * x) / x


@torch.jit.script
def cosine_cutoff(x: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Cosine cutoff function.

    Resources:
        [1]: https://doi.org/10.1063/1.3553717
    Args:
        x: Tensor of non-negative input values.
        cutoff: Cutoff distance.
    Returns:
        Tensor with same shape as the input.
    """
    assert cutoff > 0.0
    # Eq. 4 in [1]
    return 0.5 * (torch.cos(math.pi * x / cutoff) + 1) * (x <= cutoff)


@torch.jit.script
def polynomial_cutoff(x: torch.Tensor, cutoff: float, p: float) -> torch.Tensor:
    """Polynomial cutoff function.

    Resources:
        DimeNet: https://arxiv.org/abs/2003.03123
    Args:
        x: Tensor of non-negative input values.
        cutoff: Cutoff distance.
        p: Power of the polynomial.
    Returns:
        Tensor with same shape as the input.
    """
    assert cutoff > 0.0 and p > 0.0
    # Eq. 8 in DimeNet
    x = x / cutoff
    out = 1.0 - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))
    return out * (x <= 1.0)


##########################
# Symmetric Contractions
##########################

###########################################################################################
# Implementation of the symmetric contraction algorithm presented in the MACE paper
# (Batatia et al, MACE: Higher Order Equivariant Message Passing Neural Networks for Fast
# and Accurate Force Fields , Eq.10 and 11)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

# This code was modified for atomgnn. Primarely to resolve typing issues and remove unused code.

import collections

import opt_einsum_fx  # type: ignore
import torch
import torch.fx
import e3nn  # type: ignore
from e3nn.util.codegen import CodeGenMixin  # type: ignore
from e3nn.util.jit import compile_mode  # type: ignore


BATCH_EXAMPLE = 10
ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


@compile_mode("script")
class SymmetricContraction(CodeGenMixin, torch.nn.Module):

    def __init__(
        self,
        irreps_in: e3nn.o3.Irreps,
        irreps_out: e3nn.o3.Irreps,
        correlation: int | dict[str, int],
        # irrep_normalization: str = "component",  # Not used
        # path_normalization: str = "element",  # Not used
        # internal_weights: bool | None = None,  # Not used
        # shared_weights: bool | None = None,  # Not used
        num_elements: int | None = None,
    ) -> None:
        super().__init__()

        # if irrep_normalization is None:
        #     irrep_normalization = "component"

        # if path_normalization is None:
        #     path_normalization = "element"

        # assert irrep_normalization in ["component", "norm", "none"]
        # assert path_normalization in ["element", "path", "none"]

        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.irreps_out = e3nn.o3.Irreps(irreps_out)

        del irreps_in, irreps_out

        # if not isinstance(correlation, tuple):
        #     corr = correlation
        #     correlation = {}
        #     for irrep_out in self.irreps_out:
        #         correlation[irrep_out] = corr
        if isinstance(correlation, int):
            correlation = {ir: correlation for ir in self.irreps_out}

        # assert shared_weights or not internal_weights

        # if internal_weights is None:
        #     internal_weights = True

        # self.internal_weights = internal_weights
        # self.shared_weights = shared_weights

        # del internal_weights, shared_weights

        self.contractions = torch.nn.ModuleList()
        for irrep_out in self.irreps_out:
            self.contractions.append(
                Contraction(
                    irreps_in=self.irreps_in,
                    irrep_out=e3nn.o3.Irreps(str(irrep_out.ir)),
                    correlation=correlation[irrep_out],
                    # internal_weights=self.internal_weights,
                    num_elements=num_elements,
                    # weights=self.shared_weights,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.cat([contraction(x, y) for contraction in self.contractions], dim=-1)


@compile_mode("script")
class Contraction(torch.nn.Module):

    def __init__(
        self,
        irreps_in: e3nn.o3.Irreps,
        irrep_out: e3nn.o3.Irreps,
        correlation: int,
        # internal_weights: bool = True,
        num_elements: int | None = None,
        # weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        # TODO: Added for typing. Seems like num_elements must be int.
        assert isinstance(num_elements, int)

        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = e3nn.o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
        dtype = torch.get_default_dtype()
        for nu in range(1, correlation + 1):
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                dtype=dtype,
            )[-1]
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

        # Tensor contraction equations
        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        # Create weight for product basis
        self.weights = torch.nn.ParameterList([])

        for i in range(correlation, 0, -1):
            # Shapes definying
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * irrep_out.lmax + 1
            num_ell = self.U_tensors(i).size()[-2]

            if i == correlation:
                parse_subscript_main = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                    + ["ik,ekc,bci,be -> bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                )
                graph_module_main = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum(
                        "".join(parse_subscript_main), x, y, w, z
                    )
                )

                # Optimizing the contractions
                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights_max = w
            else:
                # Generate optimized contractions equations
                parse_subscript_weighting = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                    + ["k,ekc,be->bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                )
                parse_subscript_features = (
                    ["bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    + ["i,bci->bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                )

                # Symbolic tracing of contractions
                graph_module_weighting = torch.fx.symbolic_trace(
                    lambda x, y, z: torch.einsum(
                        "".join(parse_subscript_weighting), x, y, z
                    )
                )
                graph_module_features = torch.fx.symbolic_trace(
                    lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                )

                # Optimizing the contractions
                graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_weighting,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                graph_opt_features = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_features,
                    example_inputs=(
                        torch.randn(
                            [BATCH_EXAMPLE, self.num_features, num_equivariance]
                            + [num_ell] * i
                        ).squeeze(2),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                    ),
                )
                self.contractions_weighting.append(graph_opt_weighting)
                self.contractions_features.append(graph_opt_features)
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights.append(w)
        # if not internal_weights:
        #     self.weights = weights[:-1]
        #     self.weights_max = weights[-1]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = self.graph_opt_main(
            self.U_tensors(self.correlation),
            self.weights_max,
            x,
            y,
        )
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = contract_weights(
                self.U_tensors(self.correlation - i - 1),
                weight,
                y,
            )
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)

        return out.view(out.shape[0], -1)

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]


_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")


def _wigner_nj(
    irrepss: list[e3nn.o3.Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [e3nn.o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [e3nn.o3.Irrep(ir) for ir in filter_ir_mid]
    ret: list[tuple] = []
    if len(irrepss) == 1:
        (irreps,) = irrepss
        # ret = []
        e = torch.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    # ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = e3nn.o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.dim**0.5
                if normalization == "norm":
                    C *= ir_left.dim**0.5 * ir.dim**0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(
                    ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim
                )
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.dim,
                        *(irreps.dim for irreps in irrepss_left),
                        irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [(
                        ir_out,
                        _TP(op=(ir_left, ir, ir_out),
                            args=(path_left, _INPUT(len(irrepss_left), sl.start, sl.stop))),
                        E,
                    )]
            i += mul * ir.dim
    return sorted(ret, key=lambda x: x[0])


# Copied: from mace.tools.cg import U_matrix_real
def U_matrix_real(
    irreps_in: str | e3nn.o3.Irreps,
    irreps_out: str | e3nn.o3.Irreps,
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irreps_out = e3nn.o3.Irreps(irreps_out)
    irrepss = [e3nn.o3.Irreps(irreps_in)] * correlation
    if correlation == 4:
        filter_ir_mid = [
            (0, 1),
            (1, -1),
            (2, 1),
            (3, -1),
            (4, 1),
            (5, -1),
            (6, 1),
            (7, -1),
            (8, 1),
            (9, -1),
            (10, 1),
            (11, -1),
        ]
    wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)
    current_ir = wigners[0][0]
    out = []
    stack = torch.tensor([])

    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    out += [last_ir, stack]
    return out

############
# Other
############

def get_tp_irreps_out_with_instructions(
    irreps_in1: e3nn.o3.Irreps, irreps_in2: e3nn.o3.Irreps, target_irreps: e3nn.o3.Irreps
) -> tuple[e3nn.o3.Irreps, list]:
    # Collect possible irreps and their instructions
    tp_irreps_out_list: list[tuple[int, e3nn.o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps_in1):
        for j, (_, ir_edge) in enumerate(irreps_in2):
            for ir_out in ir_in * ir_edge:
                if ir_out in target_irreps:
                    k = len(tp_irreps_out_list)  # instruction index
                    tp_irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))
    # Sort the output irreps of the tensor product so they can be simplified
    tp_irreps_out = e3nn.o3.Irreps(tp_irreps_out_list)
    tp_irreps_out, permut, _ = tp_irreps_out.sort()
    # Permute the output indexes of the instructions to match the sorted irreps
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]
    instructions = sorted(instructions, key=lambda x: x[2])
    return tp_irreps_out, instructions