"""Typed latent transport with a differentiable, consistent sparse mass solve.

Geometry/operators are fixed inputs. Only right-hand sides need gradients; the
adjoint of the symmetric mass solve is another solve with the same matrix.
"""

from __future__ import annotations

import torch
from torch import nn


def _mass_cg(mass, diagonal, rhs, max_iterations=64):
    """Independent Jacobi-preconditioned CG solves for all feature columns.

    The diagonal is a preconditioner, never a replacement for the full M0.
    Reject a failed true-residual check rather than silently changing delta1.
    """
    if rhs.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(rhs).all()):
        raise ValueError("mass solve requires finite float32/float64 right-hand sides")
    rtol = 1e-10 if rhs.dtype == torch.float64 else 1e-6
    tolerance = rtol * torch.linalg.vector_norm(rhs, dim=0)
    x, residual = torch.zeros_like(rhs), rhs.clone()
    z = residual / diagonal[:, None]
    direction = z.clone()
    rz = (residual * z).sum(dim=0)
    for _ in range(max_iterations):
        active = torch.linalg.vector_norm(residual, dim=0) > tolerance
        if not bool(active.any()):
            break
        applied = torch.sparse.mm(mass, direction)
        denominator = (direction * applied).sum(dim=0)
        if bool((active & ((denominator <= 0) | ~torch.isfinite(denominator))).any()):
            raise RuntimeError("consistent mass CG breakdown")
        alpha = torch.where(active, rz / torch.where(active, denominator, 1), 0)
        x = x + direction * alpha
        residual = residual - applied * alpha
        z = residual / diagonal[:, None]
        next_rz = (residual * z).sum(dim=0)
        beta = torch.where(active, next_rz / torch.where(active, rz, 1), 0)
        direction = z + direction * beta
        rz = next_rz
    true_residual = torch.linalg.vector_norm(rhs - torch.sparse.mm(mass, x), dim=0)
    if not bool(torch.isfinite(x).all()) or bool((true_residual > 5 * tolerance).any()):
        raise RuntimeError("consistent mass CG failed its relative residual tolerance")
    return x


class _MassSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rhs, mass, diagonal):
        ctx.save_for_backward(mass, diagonal)
        return _mass_cg(mass, diagonal, rhs)

    @staticmethod
    def backward(ctx, gradient):
        mass, diagonal = ctx.saved_tensors
        return _mass_cg(mass, diagonal, gradient), None, None


def mass_solve(rhs, mass, diagonal):
    """Solve fixed SPD M0 x=rhs on the tensor device, without a dense inverse."""
    return _MassSolve.apply(rhs, mass, diagonal)


def typed_messages(vertex, edge, batch):
    """Return (delta1 edge, d0 vertex) in the adapter's latent coordinates.

    Edge latent channels are canonical edge densities, as in u0/length at input.
    S*L lifts them to requested-orientation integral cochains before delta1;
    L^-1*S maps d0 output back. Hidden channels are learned dimensionless features,
    not normalized physical p/u, so no physical field-scale ratio is inserted.
    """
    edges, xyz = batch["edges"], batch["points"]
    sign = torch.where(edges[:, 0] < edges[:, 1], 1.0, -1.0).to(vertex.dtype)
    length = torch.linalg.vector_norm(xyz[edges[:, 1]] - xyz[edges[:, 0]], dim=1)
    d0 = batch["d0"]
    gradient = torch.sparse.mm(d0, vertex) * (sign / length)[:, None]
    cochain = edge * (sign * length)[:, None]
    rhs = torch.sparse.mm(d0.transpose(0, 1), torch.sparse.mm(batch["m1"], cochain))
    divergence = mass_solve(rhs, batch["m0"], batch["m0_diagonal"])
    return divergence, gradient


class TypedTransport(nn.Module):
    """Residual learned channel mixing of the two fixed differential messages."""

    def __init__(self, width):
        super().__init__()
        self.vertex_norm = nn.LayerNorm(width)
        self.edge_norm = nn.LayerNorm(width)
        self.to_vertex = nn.Linear(width, width, bias=False)
        self.to_edge = nn.Linear(width, width, bias=False)
        # The paired attention baseline and this model start at the same function.
        nn.init.zeros_(self.to_vertex.weight)
        nn.init.zeros_(self.to_edge.weight)

    def forward(self, tokens, batch):
        count = len(batch["points"])
        vertex, edge = tokens[:count], tokens[count:]
        divergence, gradient = typed_messages(vertex, edge, batch)
        vertex = vertex + self.to_vertex(self.vertex_norm(divergence))
        edge = edge + self.to_edge(self.edge_norm(gradient))
        return torch.cat((vertex, edge))
