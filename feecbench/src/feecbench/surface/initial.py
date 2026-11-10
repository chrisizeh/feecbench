"""Mesh-paired initial fields defined on a recorded physical reference mesh."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .operators import Derived

FAMILIES = ("pulse", "two_pulses", "pulse_harmonic", "exact_velocity")


def scalar_transfer(source, target_points):
    """Nearest-triangle barycentric interpolation, with explicit distance diagnostics.

    This is a scalar sampling map, not a commuting FE transfer. Triangles are
    queried by centroid, then clamped barycentric coordinates give local convex
    interpolation. Projection distance is recorded to expose geometry error.
    """
    triangles = source.points[source.tris]
    _, candidates = cKDTree(triangles.mean(axis=1)).query(target_points, k=min(16, len(triangles)))
    candidates = np.atleast_2d(candidates).reshape(len(target_points), -1)
    p = triangles[candidates]
    e1, e2 = p[:, :, 1] - p[:, :, 0], p[:, :, 2] - p[:, :, 0]
    v = target_points[:, None] - p[:, :, 0]
    aa, ab, bb = (np.einsum("nki,nki->nk", a, b) for a, b in ((e1, e1), (e1, e2), (e2, e2)))
    va, vb = np.einsum("nki,nki->nk", v, e1), np.einsum("nki,nki->nk", v, e2)
    denom = aa * bb - ab * ab
    s, t = (bb * va - ab * vb) / denom, (aa * vb - ab * va) / denom
    weights = np.stack((1 - s - t, s, t), axis=-1).clip(0)
    weights /= weights.sum(axis=-1, keepdims=True)
    projected = np.einsum("nkj,nkji->nki", weights, p)
    distance = np.linalg.norm(projected - target_points[:, None], axis=-1)
    best = distance.argmin(axis=1)
    rows = np.arange(len(target_points))
    tri_ids, bary = source.tris[candidates[rows, best]], weights[rows, best]
    matrix = sp.csr_matrix(
        (bary.ravel(), (np.repeat(rows, 3), tri_ids.ravel())), shape=(len(target_points), len(source.points))
    )
    return matrix, {
        "max_projection_distance": float(distance[rows, best].max()),
        "rms_projection_distance": float(np.sqrt(np.mean(distance[rows, best] ** 2))),
    }


def angular_harmonic(ops, centers, coefficients):
    """Circulations around planar holes, in radians, followed by exact projection.

    Wrapped angular differences are a closed cochain with prescribed periods.
    Reject underresolved edges/triangles instead of silently losing a cycle.
    These g independent periods cover a documented subspace of the 2g harmonic
    space. The complementary g periods are not varied in this first generator.
    """
    closed = np.zeros(len(ops.edges))
    for cx, amplitude in zip(centers, coefficients, strict=True):
        angle = np.arctan2(ops.points[:, 1], ops.points[:, 0] - cx)
        diff = angle[ops.edges[:, 1]] - angle[ops.edges[:, 0]]
        wrapped = (diff + np.pi) % (2 * np.pi) - np.pi
        if np.max(np.abs(ops.d1 @ wrapped)) > 1e-10:
            raise ValueError("mesh underresolves angular circulation")
        closed += amplitude * wrapped
    return closed - Derived(ops).exact(closed)


def reference_initial(ops, values, vectors, family, seed, centers):
    """Record pulse/spectral parameters once; paired meshes sample these scalars."""
    if family not in FAMILIES:
        raise ValueError("unknown IC family")
    rng = np.random.default_rng(seed)
    edges = ops.edges
    lengths = np.linalg.norm(ops.points[edges[:, 1]] - ops.points[edges[:, 0]], axis=1)
    graph = sp.coo_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=ops.m0.shape)
    graph = graph + graph.T
    p = np.zeros(len(ops.points))
    psi = np.zeros_like(p)
    params = {"family": family, "seed": seed, "pulses": [], "harmonic_periods": "planar-hole angles"}
    if family != "exact_velocity":
        for _ in range(2 if family == "two_pulses" else 1):
            center = int(rng.integers(len(p)))
            width = float(rng.uniform(0.45, 0.7))
            amplitude = float(rng.uniform(0.7, 1.3))
            distance = dijkstra(graph, directed=False, indices=center)
            p += amplitude * np.exp(-0.5 * (distance / width) ** 2)
            params["pulses"].append(
                {
                    "center_xyz": ops.points[center].tolist(),
                    "width": width,
                    "amplitude": amplitude,
                    "width_over_max_edge": width / float(lengths.max()),
                }
            )
        # Intrinsic heat smoothing suppresses shortest-path nonsmoothness.
        from scipy.sparse.linalg import spsolve

        p = spsolve(ops.m0 + 0.02 * ops.stiffness, ops.m0 @ p)
    else:
        selected = np.where((values > 1e-8) & (values < 35))[0]
        if not len(selected):
            raise ValueError("no resolved potential modes")
        coeff = rng.normal(size=len(selected)) / np.sqrt(values[selected])
        psi = vectors[:, selected] @ coeff
        psi /= np.sqrt((ops.d0 @ psi) @ (ops.m1 @ (ops.d0 @ psi)) / ops.area)
        params["potential_mode_ids"] = selected.tolist()
        params["potential_coefficients"] = coeff.tolist()
        params["potential_eigenvalues"] = values[selected].tolist()
        params["zero_pressure"] = bool(seed % 2 == 0)
        if seed % 2:
            pressure_coeff = rng.normal(size=len(selected)) / (1 + values[selected])
            p = vectors[:, selected] @ pressure_coeff
            params["pressure_coefficients"] = pressure_coeff.tolist()
    coeff = np.zeros(len(centers))
    if family == "pulse_harmonic":
        coeff = rng.normal(size=len(centers))
        h = angular_harmonic(ops, centers, coeff)
        mean = float(np.ones(len(p)) @ (ops.m0 @ p) / ops.area)
        dynamic = (p - mean) @ (ops.m0 @ (p - mean))
        ratio = float(rng.uniform(0.05, 0.25))
        coeff *= np.sqrt(ratio * dynamic / (h @ (ops.m1 @ h)))
        params["harmonic_to_dynamic_energy"] = ratio
    params["circulation_coefficients"] = coeff.tolist()
    return p, psi, coeff, params
