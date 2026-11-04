"""Smooth, embedded genus 1--3 surfaces: doubles of perforated ellipses.

The regular zero set z²/t²=q(x,y) doubles an ellipse with g disjoint circular
holes. Its two sheets join smoothly at q=0. This explicit construction has
no self intersections; meshing and projection must still pass manifold checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ..mesh.refine import refine


@dataclass(frozen=True)
class Geometry:
    genus: int = 1
    radius: float = 0.32
    spacing: float = 1.1
    thickness: float = 0.55
    stretch: float = 1.0
    area: float = 16.0

    def __post_init__(self):
        if self.genus not in (1, 2, 3):
            raise ValueError("genus must be 1, 2, or 3")
        if not (0.15 <= self.radius <= 0.45 and 1.0 <= self.spacing <= 1.4):
            raise ValueError("radius/spacing outside the validated disjoint-hole construction")
        if not (0.3 <= self.thickness <= 0.8 and 0.85 <= self.stretch <= 1.2 and self.area > 0):
            raise ValueError("invalid thickness, stretch, or area")

    @property
    def axes(self):
        return (0.8 + (self.genus - 1) * self.spacing / 2, self.stretch)

    @property
    def centers(self):
        return (np.arange(self.genus) - (self.genus - 1) / 2) * self.spacing

    def implicit(self, points):
        x, y, z = np.asarray(points).T
        a, b = self.axes
        q = 1 - (x / a) ** 2 - (y / b) ** 2
        for cx in self.centers:
            r2 = (x - cx) ** 2 + y**2
            q = q * (r2 - self.radius**2) / (r2 + self.radius**2)
        return (z / self.thickness) ** 2 - q

    def project(self, points):
        points = np.array(points, dtype=np.float64, copy=True)
        for _ in range(15):
            val = self.implicit(points)
            if np.max(np.abs(val)) < 2e-13:
                break
            grad = np.column_stack(
                [
                    (self.implicit(points + np.eye(3)[i] * 1e-6) - self.implicit(points - np.eye(3)[i] * 1e-6)) / 2e-6
                    for i in range(3)
                ]
            )
            points -= (val / np.sum(grad * grad, axis=1))[:, None] * grad
        if np.max(np.abs(self.implicit(points))) > 1e-10:
            raise ValueError("surface projection did not converge")
        return points


def areas(points, tris):
    p = points[tris]
    return np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2


def complex_arrays(points, tris):
    local = tris[:, [[0, 1], [1, 2], [2, 0]]]
    edges, inverse = np.unique(np.sort(local.reshape(-1, 2), axis=1), axis=0, return_inverse=True)
    d0 = sp.csr_matrix(
        (np.tile(np.array([-1, 1], np.int8), len(edges)), (np.repeat(np.arange(len(edges)), 2), edges.ravel())),
        shape=(len(edges), len(points)),
    )
    signs = np.where(local[..., 0] < local[..., 1], 1, -1).astype(np.int8)
    d1 = sp.csr_matrix((signs.ravel(), (np.repeat(np.arange(len(tris)), 3), inverse)), shape=(len(tris), len(edges)))
    return edges, d0, d1


def validate_mesh(points, tris, genus):
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("expected finite embedded coordinates (V,3)")
    if tris.ndim != 2 or tris.shape[1] != 3 or tris.min() < 0 or tris.max() >= len(points):
        raise ValueError("invalid triangles")
    if len(np.unique(np.sort(tris, axis=1), axis=0)) != len(tris):
        raise ValueError("duplicate triangles")
    if len(np.unique(tris)) != len(points) or np.min(areas(points, tris)) <= 1e-14:
        raise ValueError("unused vertices or degenerate elements")
    edges, d0, d1 = complex_arrays(points, tris)
    if not np.all(np.asarray(abs(d1).sum(axis=0)).ravel() == 2):
        raise ValueError("surface must be closed with two triangles per edge")
    if np.any(np.asarray(d1.sum(axis=0))):
        raise ValueError("inconsistent triangle orientation")
    graph = abs(d0).T @ abs(d0)
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise ValueError("surface must be connected")
    # A pinched vertex can pass the edge-manifold test. Its link must be one cycle.
    links = [[] for _ in points]
    for tri in tris:
        for i in range(3):
            links[tri[i]].append((tri[(i + 1) % 3], tri[(i + 2) % 3]))
    for link in links:
        verts, inv = np.unique(link, return_inverse=True)
        pairs = inv.reshape(-1, 2)
        adj = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(verts), len(verts)))
        degree = np.bincount(inv.ravel(), minlength=len(verts))
        if np.any(degree != 2) or connected_components(adj, directed=False, return_labels=False) != 1:
            raise ValueError("nonmanifold vertex link")
    chi = len(points) - len(edges) + len(tris)
    if chi != 2 - 2 * genus or (d1 @ d0).nnz:
        raise ValueError("wrong genus or invalid chain complex")
    length = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    a = areas(points, tris)
    local = points[tris]
    sides = np.linalg.norm(local[:, [1, 2, 0]] - local, axis=2)
    shape = 4 * np.sqrt(3) * a / np.sum(sides**2, axis=1)
    return {
        "betti": [1, 2 * genus, 1],
        "euler": chi,
        "area": float(a.sum()),
        "edge_min": float(length.min()),
        "edge_median": float(np.median(length)),
        "edge_max": float(length.max()),
        "area_min": float(a.min()),
        "area_max": float(a.max()),
        "shape_quality_min": float(shape.min()),
        "shape_quality_median": float(np.median(shape)),
    }


def improve_mesh(geometry, points, tris):
    """Collapse very short extraction edges with the manifold link condition.

    Every accepted collapse preserves triangle orientation locally and is checked
    by the full manifold validator afterwards. This removes marching-cubes slivers
    before Whitney assembly; it does not smooth the target geometry.
    """
    points, tris = points.copy(), tris.copy()
    for _ in range(len(points)):
        edges, _, _ = complex_arrays(points, tris)
        lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
        local = points[tris]
        sides = np.linalg.norm(local[:, [1, 2, 0]] - local, axis=2)
        quality = 4 * np.sqrt(3) * areas(points, tris) / np.sum(sides**2, axis=1)
        bad = np.flatnonzero(quality < 0.08)
        candidates = edges[lengths < 0.08 * np.median(lengths)].tolist()
        for tri in tris[bad]:
            pairs = tri[[[0, 1], [1, 2], [2, 0]]]
            candidates.append(
                pairs[np.argmin(np.linalg.norm(points[pairs[:, 1]] - points[pairs[:, 0]], axis=1))].tolist()
            )
        changed = False
        for a, b in candidates:
            ta, tb = np.any(tris == a, axis=1), np.any(tris == b, axis=1)
            na, nb = set(tris[ta].ravel()) - {a, b}, set(tris[tb].ravel()) - {a, b}
            if len(na & nb) != 2:
                continue
            affected = (ta | tb) & ~(ta & tb)
            before = points[tris[affected]]
            after_tris = np.where(tris[affected] == b, a, tris[affected])
            position = geometry.project(((points[a] + points[b]) / 2)[None])[0]
            after = points[after_tris].copy()
            after[after_tris == a] = position
            normals0 = np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0])
            normals1 = np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0])
            if np.any(np.sum(normals0 * normals1, axis=1) <= 0):
                continue
            points[a] = position
            tris = np.where(tris[~(ta & tb)] == b, a, tris[~(ta & tb)])
            changed = True
            break
        if not changed:
            break
    used, inverse = np.unique(tris, return_inverse=True)
    return points[used], inverse.reshape(-1, 3).astype(np.int64)


def generate(geometry: Geometry, grid: int = 24, levels: int = 1, phase: float = 0.0):
    """Marching-cubes base and projected refinements; scale fixed across all levels.

    Independent remeshes use a changed grid count/phase. No FE-space nesting is
    claimed after projection. Area normalization uses a fixed high-resolution
    quadrature mesh, independent of the requested triangulation.
    """
    from skimage.measure import marching_cubes

    if grid < 12 or levels < 1 or not 0 <= phase < 1:
        raise ValueError("grid >=12, levels >=1, and phase in [0,1) required")

    def raw(n, offset=0.0):
        a, b = geometry.axes
        step = 2.4 / n
        axes = [np.arange(-r - step, r + 2 * step, step) + offset * step for r in (a, b, geometry.thickness)]
        xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        field = geometry.implicit(xyz.reshape(-1, 3)).reshape(xyz.shape[:-1])
        vertices, triangles, _, _ = marching_cubes(field, 0, spacing=(step,) * 3, allow_degenerate=False)
        vertices = geometry.project(vertices.astype(np.float64) + [v[0] for v in axes])
        return vertices, triangles.astype(np.int64)

    qp, qt = raw(64)
    scale = np.sqrt(geometry.area / areas(qp, qt).sum())
    points, tris = raw(grid, phase)
    points, tris = improve_mesh(geometry, points, tris)
    result = []
    for level in range(levels):
        scaled = points * scale
        quality = validate_mesh(scaled, tris, geometry.genus)
        result.append(
            (
                scaled,
                tris.copy(),
                {
                    "geometry": asdict(geometry),
                    "generator": "implicit_double_v1",
                    "grid": grid,
                    "phase": phase,
                    "level": level,
                    "scale": float(scale),
                    "projection_residual": float(np.max(np.abs(geometry.implicit(points)))),
                    "geometry_error": float(np.max(np.abs(geometry.implicit(points[tris].mean(axis=1))))),
                    "quality": quality,
                    "transfer": "projected refinement; no FE commuting map claimed",
                },
            )
        )
        if level + 1 < levels:
            points, tris, _ = refine(points, tris)
            points = geometry.project(points)
    return result
