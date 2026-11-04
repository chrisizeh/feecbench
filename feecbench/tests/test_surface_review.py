"""Independent review regressions for embedded geometry and consistent operators."""

import numpy as np
import pytest
import scipy.linalg as la
from scipy.spatial import ConvexHull

pytest.importorskip("pydec")

from feecbench.surface.geometry import areas, validate_mesh
from feecbench.surface.operators import Derived, assemble
from feecbench.surface.oracle import Midpoint, pressure_modes


def tetrahedron():
    points = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    points /= np.sqrt(3)
    tris = ConvexHull(points).simplices.copy()
    p = points[tris]
    signs = np.einsum("ij,ij->i", np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), p.mean(axis=1))
    tris[signs < 0] = tris[signs < 0, ::-1]
    return points, tris


def test_embedded_mass_normalizations_and_rigid_invariance():
    points, tris = tetrahedron()
    ops = assemble(points, tris, 0)
    np.testing.assert_allclose(ops.m2.diagonal(), 1 / areas(points, tris), rtol=1e-13)
    np.testing.assert_allclose(ops.m0.sum(), areas(points, tris).sum(), rtol=1e-13)
    for mass in (ops.m0, ops.m1, ops.m2):
        assert la.eigvalsh(mass.toarray())[0] > 0
    rotation = la.expm(np.array([[0, 0.3, -0.2], [-0.3, 0, 0.7], [0.2, -0.7, 0]]))
    rotated = assemble(points @ rotation + np.array([1.0, -2.0, 0.7]), tris, 0)
    for name in ("m0", "m1", "m2", "d0", "d1"):
        np.testing.assert_allclose(getattr(ops, name).toarray(), getattr(rotated, name).toarray(), atol=2e-14)


def test_consistent_mass_solve_and_orientation_through_dynamics():
    points, tris = tetrahedron()
    ops = assemble(points, tris, 0)
    rng = np.random.default_rng(62)
    p0, u0 = rng.normal(size=len(points)), rng.normal(size=len(ops.edges))
    rhs = ops.d0.T @ (ops.m1 @ u0)
    delta = Derived(ops).delta1(u0)
    np.testing.assert_allclose(ops.m0 @ delta, rhs, atol=1e-14)
    assert np.linalg.norm(delta-rhs/ops.m0.diagonal()) > 0.1
    signs = rng.choice([-1, 1], size=len(ops.edges))
    flipped = ops.flipped(signs)
    np.testing.assert_allclose(Derived(flipped).delta1(signs*u0), delta, atol=1e-14)
    times = [0.013, 0.04, 0.091]
    p, u = Midpoint(ops).solve(p0, u0, times, 0.004)
    fp, fu = Midpoint(flipped).solve(p0, signs*u0, times, 0.004)
    np.testing.assert_allclose(fp, p, atol=2e-14)
    np.testing.assert_allclose(fu, u*signs, atol=2e-14)
    # Additional nearby queries must not change the regular-grid trajectory.
    pp, uu = Midpoint(ops).solve(p0, u0, [0.013, 0.025, 0.04, 0.075, 0.091], 0.004)
    np.testing.assert_array_equal(pp[[0, 2, 4]], p)
    np.testing.assert_array_equal(uu[[0, 2, 4]], u)


def test_vertex_link_rejects_pinched_closed_surface():
    points, tris = tetrahedron()
    # Two individually closed tetrahedra share only vertex zero: every edge has
    # two faces and the graph is connected, but the common vertex is nonmanifold.
    other = 2*points[0]-points[1:]
    joined_points = np.r_[points, other]
    mapping = np.array([0, 4, 5, 6])
    joined_tris = np.r_[tris, mapping[tris]]
    with pytest.raises(ValueError, match="vertex link"):
        validate_mesh(joined_points, joined_tris, genus=0)


def test_inconsistent_face_orientation_rejected():
    points, tris = tetrahedron()
    tris[0] = tris[0, ::-1]
    with pytest.raises(ValueError, match="orientation"):
        validate_mesh(points, tris, genus=0)


def test_full_spectrum_parseval_and_nonuniform_band_gradient():
    """Spectral balancing changes the objective; its sparse gradient is correct."""
    import torch

    from feecbench.surface.model import loss, sparse_tensor

    points, tris = tetrahedron()
    ops = assemble(points, tris, 0)
    values, vectors = pressure_modes(ops, len(points))
    rng = np.random.default_rng(21)
    error = rng.normal(size=len(points))
    projection = vectors.T @ ops.m0
    np.testing.assert_allclose(np.linalg.norm(projection @ error)**2, error @ (ops.m0 @ error), rtol=1e-13)
    # Constant and positive modes receive different weights. All three equal
    # positive eigenvalues stay in one band, avoiding arbitrary basis dependence.
    bands = [{"lower": -1, "upper": 1e-8, "scale": 0.5, "resolved": True},
             {"lower": 1e-8, "upper": float(values[-1])+1, "scale": 2.0, "resolved": True}]
    ep = torch.tensor(error, dtype=torch.float64, requires_grad=True)
    eu = torch.zeros(len(ops.edges), dtype=torch.float64, requires_grad=True)
    batch = {"m0": sparse_tensor(ops.m0, dtype=torch.float64),
             "m1": sparse_tensor(ops.m1, dtype=torch.float64), "area": ops.area,
             "projection": torch.tensor(projection, dtype=torch.float64),
             "eigenvalues": torch.tensor(values, dtype=torch.float64)}
    target = torch.zeros_like(ep), torch.zeros_like(eu)
    spatial = loss((ep, eu), target, batch, "mass")
    objective = loss((ep, eu), target, batch, "spectral", bands=bands, spectral_weight=0.7)
    objective.backward()
    weights = np.array([4, 0.25, 0.25, 0.25]) / len(bands)
    expected = (2*(ops.m0 @ error) + 1.4*projection.T @ (weights*(projection @ error))) / ops.area
    np.testing.assert_allclose(ep.grad.numpy(), expected, rtol=1e-12, atol=1e-13)
    assert float(objective.detach()) > float(spatial.detach())
    # Rotate the degenerate eigenspace: aggregate band loss must not change.
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated_projection = projection.copy()
    rotated_projection[1:] = rotation @ projection[1:]
    batch["projection"] = torch.tensor(rotated_projection, dtype=torch.float64)
    rotated_loss = loss((ep, eu), target, batch, "spectral", bands=bands, spectral_weight=0.7)
    torch.testing.assert_close(rotated_loss, objective, rtol=1e-12, atol=1e-13)
