"""Discrete de Rham complex: orientations, incidence matrices, Hodge stars, derived operators.

Conventions:

* vertices ``0..n0-1`` with ``points (n0, 2)``; edges are sorted pairs ``(i, j)``, ``i < j``,
  oriented low -> high, lexicographically sorted; triangles are counter-clockwise;
* ``d0 (n1, n0)``: row of edge (i, j) has -1 at i, +1 at j (gradient);
  ``d1 (n2, n1)``: row of a triangle has +1 per edge whose stored orientation agrees with
  the CCW traversal, else -1 (curl). Both int8; ``d1 @ d0 == 0`` exactly;
* boundary edges belong to exactly one triangle; boundary vertices are their endpoints;
* ``star0``: lumped P1 mass (diagonal, 1/3 of the adjacent area per vertex);
  ``star1``: Galerkin mass of lowest-order Whitney 1-forms
  ``w_ij = lambda_i grad(lambda_j) - lambda_j grad(lambda_i)`` with the exact quadrature
  ``int_T lambda_a lambda_b = |T| (1 + delta_ab) / 12``; ``star2``: triangle areas;
* derived (never stored): ``delta1 = star0^{-1} d0^T star1``, ``Delta0 = delta1 d0``,
  ``Delta1 = d0 delta1 + star1^{-1} d1^T star2 d1``; weighted norms ``sqrt(z^T star_k z)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from .mesh.quality import triangle_areas


@dataclass
class Complex:
    """Oriented 2D simplicial complex with its incidence matrices."""

    points: np.ndarray
    tris: np.ndarray
    edges: np.ndarray
    d0: sp.csr_matrix
    d1: sp.csr_matrix
    boundary_vertices: np.ndarray
    boundary_edges: np.ndarray

    @property
    def n_cells(self) -> tuple[int, int, int]:
        return (len(self.points), len(self.edges), len(self.tris))


def build_complex(points: np.ndarray, tris: np.ndarray) -> Complex:
    """Enforce CCW triangles, extract edges, assemble d0/d1 and the boundary masks."""
    points = np.ascontiguousarray(points, dtype=np.float64)
    tris = np.ascontiguousarray(tris, dtype=np.int64)
    area = triangle_areas(points, tris)
    if np.any(area == 0.0):
        raise ValueError("degenerate (zero-area) triangle in mesh")
    tris = np.where((area < 0.0)[:, None], tris[:, [0, 2, 1]], tris)
    local = tris[:, [[0, 1], [1, 2], [2, 0]]]  # (n2, 3, 2), CCW traversal
    edges, inverse = np.unique(np.sort(local.reshape(-1, 2), axis=1), axis=0, return_inverse=True)
    n0, n1, n2 = len(points), len(edges), len(tris)
    d0 = sp.csr_matrix(
        (np.tile(np.array([-1, 1], np.int8), n1), (np.repeat(np.arange(n1), 2), edges.ravel())),
        shape=(n1, n0),
    )
    sign = np.where(local[..., 0] < local[..., 1], 1, -1).astype(np.int8)
    d1 = sp.csr_matrix((sign.ravel(), (np.repeat(np.arange(n2), 3), inverse.ravel())), shape=(n2, n1))
    boundary_edges = np.bincount(inverse.ravel(), minlength=n1) == 1
    boundary_vertices = np.zeros(n0, dtype=bool)
    boundary_vertices[edges[boundary_edges].ravel()] = True
    return Complex(points, tris, edges.astype(np.int64), d0, d1, boundary_vertices, boundary_edges)


def betti_numbers(cx: Complex) -> tuple[int, int]:
    """(beta0, beta1): connected components + Euler characteristic (beta2 = 0 with boundary)."""
    beta0 = _components(cx.edges, cx.n_cells[0])
    return beta0, beta0 - (cx.n_cells[0] - cx.n_cells[1] + cx.n_cells[2])


def n_boundary_components(cx: Complex) -> int:
    """Connected components of the boundary curve."""
    n0 = cx.n_cells[0]
    isolated = n0 - int(cx.boundary_vertices.sum())
    return _components(cx.edges[cx.boundary_edges], n0) - isolated


def _components(edges: np.ndarray, n: int) -> int:
    graph = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return int(connected_components(graph, directed=False, return_labels=False))


def build_stars(cx: Complex) -> dict[str, sp.csr_matrix]:
    """star0 (lumped P1), star1 (Whitney Galerkin, SPD), star2 (areas); vectorized assembly."""
    p, tris = cx.points[cx.tris], cx.tris
    area = triangle_areas(cx.points, tris)
    n2 = len(tris)
    star0 = np.bincount(tris.ravel(), np.repeat(area / 3.0, 3), minlength=cx.n_cells[0])
    # grad(lambda_i) = perp(p_k - p_j) / (2A), (i, j, k) cyclic, perp(x, y) = (-y, x)
    e = p[:, [2, 0, 1]] - p[:, [1, 2, 0]]  # (n2, 3, 2)
    grads = np.stack([-e[..., 1], e[..., 0]], axis=-1) / (2.0 * area)[:, None, None]
    gg = np.einsum("tid,tjd->tij", grads, grads)
    lam = area[:, None, None] * (1.0 + np.eye(3)) / 12.0
    i, j = np.array([0, 1, 2]), np.array([1, 2, 0])  # local edge a = (i[a], j[a])
    m = (
        lam[:, i[:, None], i] * gg[:, j[:, None], j]
        - lam[:, i[:, None], j] * gg[:, j[:, None], i]
        - lam[:, j[:, None], i] * gg[:, i[:, None], j]
        + lam[:, j[:, None], j] * gg[:, i[:, None], i]
    )
    local = tris[:, [[0, 1], [1, 2], [2, 0]]]
    sign = np.where(local[..., 0] < local[..., 1], 1.0, -1.0)  # global (low -> high) orientation
    m *= sign[:, :, None] * sign[:, None, :]
    edge_id = _local_edge_ids(cx)
    rows = np.broadcast_to(edge_id[:, :, None], (n2, 3, 3)).ravel()
    cols = np.broadcast_to(edge_id[:, None, :], (n2, 3, 3)).ravel()
    n1 = cx.n_cells[1]
    return {
        "star0": sp.diags(star0, format="csr"),
        "star1": sp.csr_matrix((m.ravel(), (rows, cols)), shape=(n1, n1)),
        "star2": sp.diags(area, format="csr"),
    }


def _local_edge_ids(cx: Complex) -> np.ndarray:
    """Edge id of each triangle's local edges ab, bc, ca (n2, 3), via the sorted edge table."""
    local = np.sort(cx.tris[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    n0 = cx.n_cells[0]
    keys, query = cx.edges[:, 0] * n0 + cx.edges[:, 1], local[:, 0] * n0 + local[:, 1]
    return np.searchsorted(keys, query).reshape(-1, 3)


# -- derived operators (reconstructed, never stored) ---------------------------------------


def codifferential1(d0, star0, star1) -> sp.csr_matrix:
    """delta1 = star0^{-1} d0^T star1 (star0 is diagonal, so this stays sparse)."""
    if (star0 - sp.diags(star0.diagonal())).nnz:
        raise ValueError("consistent star0 requires the surface solve-based backend")
    return (sp.diags(1.0 / star0.diagonal()) @ (d0.T.astype(np.float64) @ star1)).tocsr()


def laplacian0(d0, star0, star1) -> sp.csr_matrix:
    """Delta0 = delta1 d0."""
    return (codifferential1(d0, star0, star1) @ d0.astype(np.float64)).tocsr()


def laplacian1(d0, d1, star0, star1, star2) -> spla.LinearOperator:
    """Delta1 = d0 delta1 + star1^{-1} d1^T star2 d1, with star1^{-1} applied via sparse LU."""
    d0f, d1f = d0.astype(np.float64), d1.astype(np.float64)
    delta1, curl = codifferential1(d0, star0, star1), (d1f.T @ star2 @ d1f).tocsc()
    lu = spla.splu(star1.tocsc())
    n1 = star1.shape[0]
    return spla.LinearOperator((n1, n1), matvec=lambda u: d0f @ (delta1 @ u) + lu.solve(curl @ u))


def norm_star(z: np.ndarray, star: sp.spmatrix) -> float:
    """Weighted norm sqrt(z^T star z), summed over the leading axis of a (T, n) series."""
    z = np.atleast_2d(z)
    return float(np.sqrt(max(np.einsum("ti,ti->", z, (star @ z.T).T), 0.0)))


def project_to_vertices(points: np.ndarray, tris: np.ndarray, per_tri: np.ndarray) -> np.ndarray:
    """Area-weighted projection of a per-triangle coefficient onto vertices (used for c^2)."""
    area = triangle_areas(points, tris)
    idx, n0 = tris.ravel(), len(points)
    return np.bincount(idx, np.repeat(area * per_tri, 3), n0) / np.bincount(idx, np.repeat(area, 3), n0)


def d1_operator_norm(d1, star1, star2) -> float:
    """Operator norm of d1 : (C^1, star1) -> (C^2, star2), via the largest generalized eigenvalue."""
    d1f = d1.astype(np.float64)
    lam = spla.eigsh((d1f.T @ star2 @ d1f).tocsc(), k=1, M=star1.tocsc(), return_eigenvectors=False)
    return float(np.sqrt(max(lam[0], 0.0)))


# -- Hodge decomposition helpers (evaluator support) ----------------------------------------


def harmonic_1forms(d0, d1, star0, star1, star2, beta1: int, tol: float = 1e-8) -> np.ndarray:
    """star1-orthonormal basis (n1, beta1) of the harmonic 1-forms.

    Null space of the SPSD matrix ``A = d1^T star2 d1 + star1 d0 star0^{-1} d0^T star1``,
    solved as ``A h = lambda star1 h`` by shift-invert eigsh (sigma < 0 keeps A - sigma star1
    SPD). The null-space dimension must equal beta1 or a RuntimeError is raised. The basis
    is defined up to rotation; projections onto it are well defined.
    """
    if beta1 == 0:
        return np.zeros((star1.shape[0], 0))
    d0f, d1f = d0.astype(np.float64), d1.astype(np.float64)
    if (star0 - sp.diags(star0.diagonal())).nnz:
        raise ValueError("consistent star0 requires the surface harmonic backend")
    inv0 = sp.diags(1.0 / star0.diagonal())
    a = (d1f.T @ star2 @ d1f + star1 @ d0f @ inv0 @ d0f.T @ star1).tocsc()
    vals, vecs = spla.eigsh(a, k=min(beta1 + 3, a.shape[0] - 2), M=star1.tocsc(), sigma=-1e-2)
    order = np.argsort(np.abs(vals))
    vals, vecs = vals[order], vecs[:, order]
    n_null = int(np.sum(np.abs(vals) < tol * max(np.abs(vals).max(), 1.0)))
    if n_null != beta1:
        raise RuntimeError(f"harmonic dimension {n_null} != beta1 = {beta1}; eigenvalues {vals}")
    h = vecs[:, :beta1]
    return h @ np.linalg.inv(np.linalg.cholesky(h.T @ (star1 @ h))).T  # guard roundoff


def hodge_components(u: np.ndarray, d0, star1, harmonic: np.ndarray) -> dict[str, np.ndarray]:
    """star1-orthogonal exact / harmonic / coexact split of a 1-cochain.

    Exact part by a least-squares potential solve (vertex 0 pinned; connected meshes),
    harmonic part by projection onto the given basis, coexact part as the remainder.
    """
    d0f = d0.astype(np.float64)
    k, rhs = (d0f.T @ star1 @ d0f).tocsc(), d0f.T @ (star1 @ u)
    alpha = np.zeros(k.shape[0])
    alpha[1:] = spla.spsolve(k[1:, 1:], rhs[1:])
    exact, harm = d0f @ alpha, harmonic @ (harmonic.T @ (star1 @ u))
    return {"exact": exact, "harmonic": harm, "coexact": u - exact - harm}
