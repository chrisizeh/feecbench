"""Small Transolver-style slice-attention adaptation, direct query time prediction.

Vertex inputs: xyz, p0, c, t. Edge inputs: midpoint xyz, canonical tangent,
length, u0/length, c, t. The optional vector_features adapter appends the
edge-direction projection (u0/length)*unit_tangent; it is not a full vector
reconstruction. No learned incidence/operator transport is claimed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def forward_features(adapter="scalar"):
    """Record the exact learner-visible feature list for checkpoint provenance."""
    if adapter not in ("scalar", "vector_features"):
        raise ValueError("adapter must be scalar or vector_features")
    result = {
        "vertex": ["xyz", "p0/scalep", "c", "t"],
        "edge": ["midpoint", "canonical_tangent", "length", "canonical_u0/(length*scaleu)", "c", "t"],
    }
    if adapter == "vector_features":
        result["edge"].append("(canonical_u0/(length*scaleu))*canonical_unit_tangent [3 components]")
    return result


class SliceAttention(nn.Module):
    def __init__(self, width, slices, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.assignment = nn.Linear(width, slices)
        self.values = nn.Linear(width, width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0)
        self.output = nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, tokens):
        x = self.norm(tokens)
        weights = self.assignment(x).softmax(dim=-1)
        slices = (weights.T @ self.values(x)) / weights.sum(dim=0)[:, None].clamp_min(1e-6)
        attended = self.attention(slices[None], slices[None], slices[None], need_weights=False)[0][0]
        tokens = tokens + self.output(weights @ attended)
        return tokens + self.mlp(tokens)


class SurfaceTransolver(nn.Module):
    def __init__(self, width=128, blocks=4, slices=32, adapter="scalar"):
        super().__init__()
        if adapter not in ("scalar", "vector_features"):
            raise ValueError("adapter must be scalar or vector_features")
        self.adapter = adapter
        self.settings = {"width": width, "blocks": blocks, "slices": slices, "adapter": adapter}
        self.vertex_encoder = nn.Linear(6, width)
        self.edge_encoder = nn.Linear(10, width)
        self.blocks = nn.ModuleList([SliceAttention(width, slices) for _ in range(blocks)])
        self.pressure_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.velocity_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        if adapter == "vector_features":
            # Initialize all shared parameters in the legacy RNG order. At the
            # same seed the added features start with zero influence, so the
            # two adapters have the same initial function (up to GEMM roundoff).
            with torch.no_grad():
                weight = self.edge_encoder.weight
                self.edge_encoder.weight = nn.Parameter(torch.cat((weight, weight.new_zeros(width, 3)), dim=1))
                self.edge_encoder.in_features = 13

    def forward(self, batch, query_time):
        """One mesh per call. Canonicalize input and map velocity back to requested signs."""
        xyz, edges = batch["points"], batch["edges"]
        sign = torch.where(edges[:, 0] < edges[:, 1], 1.0, -1.0)
        tangent = (xyz[edges[:, 1]] - xyz[edges[:, 0]]) * sign[:, None]
        length = torch.linalg.vector_norm(tangent, dim=-1)
        midpoint = (xyz[edges[:, 0]] + xyz[edges[:, 1]]) / 2
        t = torch.as_tensor(query_time, dtype=xyz.dtype, device=xyz.device).reshape(1, 1)
        c = torch.ones_like(t) * batch.get("coefficient", 1.0)
        pv = batch["p0"][:, None]
        uv = (batch["u0"] * sign / length)[:, None]
        v = torch.cat((xyz, pv, c.expand(len(xyz), 1), t.expand(len(xyz), 1)), dim=1)
        e = torch.cat(
            (
                midpoint,
                tangent / length[:, None],
                length[:, None],
                uv,
                c.expand(len(edges), 1),
                t.expand(len(edges), 1),
            ),
            dim=1,
        )
        if self.adapter == "vector_features":
            e = torch.cat((e, uv * (tangent / length[:, None])), dim=1)
        tokens = torch.cat((self.vertex_encoder(v), self.edge_encoder(e)))
        for block in self.blocks:
            tokens = block(tokens)
        p = batch["p0"] + t.squeeze() * self.pressure_head(tokens[: len(xyz)]).squeeze(-1)
        u = batch["u0"] + t.squeeze() * self.velocity_head(tokens[len(xyz) :]).squeeze(-1) * length * sign
        return p, u


def sparse_tensor(matrix, device="cpu", dtype=torch.float32):
    coo = matrix.tocoo()
    index = torch.tensor(np.stack((coo.row, coo.col)), dtype=torch.long, device=device)
    return torch.sparse_coo_tensor(
        index, torch.as_tensor(coo.data, dtype=dtype, device=device), coo.shape, device=device, check_invariants=True
    ).coalesce()


def tensor_sample(sample, scales, device="cpu"):
    op = sample.ops

    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    return {
        "points": tensor(op.points),
        "edges": torch.as_tensor(op.edges, dtype=torch.long, device=device),
        "p0": tensor(sample.pressure[0] / scales["pressure"]),
        "u0": tensor(sample.velocity[0] / scales["velocity"]),
        "pressure": tensor(sample.pressure / scales["pressure"]),
        "velocity": tensor(sample.velocity / scales["velocity"]),
        "m0": sparse_tensor(op.m0, device),
        "m1": sparse_tensor(op.m1, device),
        "area": op.area,
        "projection": tensor(sample.vectors.T @ op.m0),
        "eigenvalues": tensor(sample.values),
        "times": tensor(sample.times),
        "coefficient": 1.0,
    }


def quadratic(error, mass):
    return torch.dot(error, torch.sparse.mm(mass, error[:, None]).squeeze(1))


def loss(prediction, target, batch, objective="mass", bands=None, spectral_weight=1.0, field_balance=None):
    ep, eu = (p - t for p, t in zip(prediction, target, strict=True))
    if objective == "mse":
        return ep.square().mean() + eu.square().mean()
    if objective == "mse_balanced":
        if field_balance is None:
            raise ValueError("mse_balanced requires fixed training-derived weights")
        weights = [field_balance["weights"][f] for f in ("pressure", "velocity")]
        if any(not np.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("positive finite field weights required")
        return weights[0] * ep.square().mean() + weights[1] * eu.square().mean()
    if objective not in ("mass", "spectral"):
        raise ValueError("unknown loss")
    spatial = (quadratic(ep, batch["m0"]) + quadratic(eu, batch["m1"])) / batch["area"]
    if objective == "mass":
        return spatial
    if not bands or not any(b["resolved"] and b["scale"] > 0 for b in bands):
        raise ValueError("spectral objective requires frozen, resolved training-derived bands")
    coefficients = batch["projection"] @ ep
    terms = []
    for band in bands:
        if not band["resolved"] or band["scale"] <= 0:
            continue
        mask = (batch["eigenvalues"] > band["lower"]) & (batch["eigenvalues"] <= band["upper"])
        if bool(mask.any()):
            terms.append(coefficients[mask].square().sum() / batch["area"] / band["scale"] ** 2)
    if not terms:
        raise ValueError("no represented resolved bands on this mesh")
    return spatial + spectral_weight * torch.stack(terms).mean()
