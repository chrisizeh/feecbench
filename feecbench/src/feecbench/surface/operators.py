"""PyDEC Whitney masses with canonical edges and solve-based derived operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from .geometry import complex_arrays, validate_mesh

PYDEC_REVISION = "c181202c70e79794dcc8fa4577537a4eb4987b2a"


@dataclass
class SurfaceOperators:
    points: np.ndarray
    tris: np.ndarray
    edges: np.ndarray
    d0: sp.csr_matrix
    d1: sp.csr_matrix
    m0: sp.csr_matrix
    m1: sp.csr_matrix
    m2: sp.csr_matrix
    genus: int

    @property
    def area(self):
        return float(self.m0.sum())

    @property
    def stiffness(self):
        return (self.d0.T.astype(float) @ self.m1 @ self.d0.astype(float)).tocsr()

    def flipped(self, signs):
        signs = np.asarray(signs)
        if signs.shape != (len(self.edges),) or not np.all(np.abs(signs) == 1):
            raise ValueError("one +/-1 orientation sign per edge required")
        s = sp.diags(signs, format="csr", dtype=np.float64)
        edges = self.edges.copy()
        edges[signs < 0] = edges[signs < 0, ::-1]
        return SurfaceOperators(
            self.points,
            self.tris,
            edges,
            (s @ self.d0).tocsr(),
            (self.d1 @ s).tocsr(),
            self.m0,
            (s @ self.m1 @ s).tocsr(),
            self.m2,
            self.genus,
        )


def assemble(points, tris, genus):
    """Assemble from hirani/PyDEC; never use its diagonal circumcentric star."""
    from pydec import simplicial_complex, whitney_innerproduct

    points, tris = np.asarray(points, dtype=np.float64), np.asarray(tris, dtype=np.int64)
    validate_mesh(points, tris, genus)
    edges, d0, d1 = complex_arrays(points, tris)
    sc = simplicial_complex((points.copy(), tris.copy()))
    if not np.array_equal(sc[0].simplices.ravel(), np.arange(len(points))):
        raise RuntimeError("unexpected PyDEC vertex ordering")
    if not np.array_equal(sc[1].simplices, edges):
        raise RuntimeError("unexpected PyDEC edge ordering")
    m0, m1, m2 = [whitney_innerproduct(sc, k).tocsr() for k in range(3)]
    # PyDEC sorts top simplex rows. Top-form signs cancel in the diagonal mass.
    tri_lookup = {tuple(sorted(tri)): i for i, tri in enumerate(sc[2].simplices)}
    perm = np.array([tri_lookup[tuple(sorted(tri))] for tri in tris])
    m2 = m2[perm][:, perm].tocsr()
    for mass in (m0, m1, m2):
        if not np.isfinite(mass.data).all() or np.any(mass.diagonal() <= 0):
            raise ValueError("invalid Whitney mass")
    if (sc[0].d - d0).nnz:
        raise RuntimeError("PyDEC incidence mismatch")
    return SurfaceOperators(points, tris, edges, d0, d1, m0, m1, m2, genus)


class Derived:
    """Reusable sparse factorizations; no diagonal approximation to consistent M0."""

    def __init__(self, ops):
        self.ops = ops
        self.solve0 = sla.factorized(ops.m0.tocsc())
        self.solve1 = sla.factorized(ops.m1.tocsc())
        self.solve_k = sla.factorized(ops.stiffness[1:, 1:].tocsc())

    def delta1(self, u):
        return self.solve0(self.ops.d0.T @ (self.ops.m1 @ u))

    def laplacian0(self, p):
        return self.delta1(self.ops.d0 @ p)

    def laplacian1(self, u):
        op = self.ops
        return op.d0 @ self.delta1(u) + self.solve1(op.d1.T @ (op.m2 @ (op.d1 @ u)))

    def exact(self, u):
        op = self.ops
        rhs = op.d0.T @ (op.m1 @ u)
        potential = np.zeros((len(op.points),) + np.asarray(u).shape[1:])
        potential[1:] = self.solve_k(rhs[1:])
        return op.d0 @ potential

    def hodge(self, u, harmonic):
        exact = self.exact(u)
        harm = harmonic @ (harmonic.T @ (self.ops.m1 @ u))
        return {"exact": exact, "harmonic": harm, "coexact": u - exact - harm}


def harmonic_basis(ops, tol=1e-7):
    """Kernel of curl and divergence, using a sparse positive surrogate metric.

    A lumped auxiliary 0-metric is used only to find the common nullspace;
    it is never used for acoustic dynamics or codifferential application.
    """
    b = ops.d0.T.astype(float) @ ops.m1
    a = ops.d1.T.astype(float) @ ops.m2 @ ops.d1 + b.T @ sp.diags(1 / np.asarray(ops.m0.sum(axis=1)).ravel()) @ b
    count = 2 * ops.genus
    vals, vecs = sla.eigsh(
        a.tocsc(), k=count + 2, M=ops.m1.tocsc(), sigma=-1e-4, v0=np.random.default_rng(0).normal(size=len(ops.edges))
    )
    order = np.argsort(np.abs(vals))
    vals, h = vals[order], vecs[:, order[:count]]
    if np.max(np.abs(vals[:count])) > tol or abs(vals[count]) < tol:
        raise ValueError(f"incorrect harmonic dimension: eigenvalues {vals}")
    h = h @ la.solve_triangular(la.cholesky(h.T @ (ops.m1 @ h), lower=True).T, np.eye(count))
    if max(np.linalg.norm(ops.d1 @ h), np.linalg.norm(b @ h)) > tol:
        raise ValueError("harmonic residual too large")
    return h
