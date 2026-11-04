"""Small Transolver-style slice-attention adaptation, direct query time prediction.

Vertex inputs: xyz, p0, c, t. Edge inputs: midpoint xyz, canonical tangent,
length, u0/length, c, t. No learned incidence/operator transport is claimed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


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
    def __init__(self, width=128, blocks=4, slices=32):
        super().__init__()
        self.settings = {"width": width, "blocks": blocks, "slices": slices}
        self.vertex_encoder = nn.Linear(6, width)
        self.edge_encoder = nn.Linear(10, width)
        self.blocks = nn.ModuleList([SliceAttention(width, slices) for _ in range(blocks)])
        self.pressure_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.velocity_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

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


def loss(prediction, target, batch, objective="mass", bands=None, spectral_weight=1.0):
    ep, eu = (p - t for p, t in zip(prediction, target, strict=True))
    if objective == "mse":
        return ep.square().mean() + eu.square().mean()
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
