"""Closed-surface numerical contracts, including both acoustic coupling directions."""

import json

import numpy as np
import pytest

pytest.importorskip("pydec")
pytest.importorskip("skimage")

from feecbench.surface.data import SurfaceSample, load, make_manifest, save, training_scales, validate_manifest
from feecbench.surface.geometry import Geometry, areas, generate, validate_mesh
from feecbench.surface.initial import angular_harmonic, scalar_transfer
from feecbench.surface.operators import Derived, assemble, harmonic_basis
from feecbench.surface.oracle import Midpoint, ModalControl, energy, pressure_modes


@pytest.fixture(scope="module")
def surface():
    points, tris, _ = generate(Geometry(), grid=12)[0]
    op = assemble(points, tris, 1)
    values, vectors = pressure_modes(op, 12)
    return op, values, vectors, harmonic_basis(op)


@pytest.mark.parametrize("genus", [1, 2, 3])
def test_surface_topology(genus):
    points, tris, metadata = generate(Geometry(genus), grid=12)[0]
    assert validate_mesh(points, tris, genus)["betti"] == [1, 2 * genus, 1]
    assert metadata["projection_residual"] < 1e-10
    op = assemble(points, tris, genus)
    np.testing.assert_allclose(op.m2.diagonal(), 1 / areas(points, tris), rtol=1e-10)
    h = harmonic_basis(op)
    assert h.shape[1] == 2 * genus
    np.testing.assert_allclose(h.T @ (op.m1 @ h), np.eye(2 * genus), atol=1e-9)
    assert np.linalg.norm(op.d1 @ h) < 1e-7


def test_consistent_solve_and_midpoint(surface):
    op, lam, phi, h = surface
    derived = Derived(op)
    np.testing.assert_allclose(derived.laplacian0(phi[:, 1]), lam[1] * phi[:, 1], atol=1e-9)
    p0 = phi[:, 1] + 0.3
    u0 = 0.7 * op.d0 @ phi[:, 2] + h[:, 0]
    times = np.array([0, 0.137, 0.2, 0.413, 0.5])
    p, u = Midpoint(op).solve(p0, u0, times, 0.01)
    np.testing.assert_allclose(energy(op, p, u), energy(op, p0, u0)[0], rtol=1e-10)
    np.testing.assert_allclose(p @ (op.m0 @ np.ones(len(p0))), p0 @ (op.m0 @ np.ones(len(p0))), atol=1e-10)
    np.testing.assert_allclose(u @ (op.m1 @ h), np.broadcast_to(u0 @ (op.m1 @ h), (len(times), 2)), atol=1e-9)
    np.testing.assert_allclose(op.d1 @ u.T, 0, atol=1e-8)
    control = ModalControl(op, lam, phi, h)
    exact = control.solve(p0, u0, times)
    errors = []
    for dt in (0.02, 0.01, 0.005):
        pp, uu = Midpoint(op).solve(p0, u0, times, dt)
        errors.append(np.sqrt(energy(op, pp - exact[0], uu - exact[1]).sum()))
    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8
    for initial in ((phi[:, 1], np.zeros(len(op.edges))), (np.zeros(len(op.points)), op.d0 @ phi[:, 1])):
        pp, uu = control.solve(*initial, [0, 0.1])
        assert np.linalg.norm(pp[1] - pp[0]) > 1e-3
        assert np.linalg.norm(uu[1] - uu[0]) > 1e-3


def test_orientation_and_stationary_fields(surface):
    op, lam, phi, h = surface
    signs = np.random.default_rng(8).choice([-1, 1], len(op.edges))
    flipped = op.flipped(signs)
    p0, u0 = phi[:, 1], op.d0 @ phi[:, 2] + h[:, 0]
    p, u = Midpoint(op).solve(p0, u0, [0, 0.2], 0.01)
    fp, fu = Midpoint(flipped).solve(p0, signs * u0, [0, 0.2], 0.01)
    np.testing.assert_allclose(fp, p, atol=1e-10)
    np.testing.assert_allclose(fu, u * signs, atol=1e-10)
    np.testing.assert_allclose(Derived(flipped).delta1(signs * u0), Derived(op).delta1(u0), atol=1e-10)
    pp, uu = Midpoint(op).solve(np.ones(len(p0)), h[:, 0], [0, 0.2], 0.01)
    np.testing.assert_allclose(pp, 1, atol=1e-10)
    np.testing.assert_allclose(uu[1], h[:, 0], atol=1e-10)
    assert np.max(np.abs((op.m0 - __import__("scipy").sparse.diags(op.m0.diagonal())).data)) > 0


def test_scalar_and_circulation_transfer(surface):
    op, _, _, _ = surface
    matrix, diagnostics = scalar_transfer(op, op.points)
    np.testing.assert_allclose(matrix @ op.points, op.points, atol=1e-10)
    np.testing.assert_allclose(matrix @ np.ones(len(op.points)), 1, atol=1e-12)
    assert diagnostics["max_projection_distance"] < 1e-10
    h = angular_harmonic(op, [0], [0.5])
    assert np.linalg.norm(op.d1 @ h) < 1e-10
    assert np.linalg.norm(op.d0.T @ (op.m1 @ h)) < 1e-10
    assert h @ (op.m1 @ h) > 0.1


def test_surface_roundtrip_and_scales(surface, tmp_path):
    op, lam, phi, h = surface
    p, u = Midpoint(op).solve(phi[:, 1], h[:, 0], [0, 0.1], 0.01)
    sample = SurfaceSample(
        op,
        np.array([0, 0.1]),
        p,
        u,
        lam,
        phi,
        h,
        {"split": "train", "resolution": "medium", "config_id": "a"},
        np.zeros(2, bool),
    )
    path = tmp_path / "sample.h5"
    save(path, sample)
    reread = load(path)
    for name in ("pressure", "velocity", "times", "vectors", "harmonic"):
        np.testing.assert_array_equal(getattr(sample, name), getattr(reread, name))
    for name in ("d0", "d1", "m0", "m1", "m2"):
        np.testing.assert_array_equal(getattr(op, name).data, getattr(reread.ops, name).data)
    scales = training_scales([path])
    signs = np.random.default_rng(0).choice([-1, 1], len(op.edges))
    sample.ops, sample.velocity = op.flipped(signs), u * signs
    save(path, sample)
    assert training_scales([path]) == scales
    assert not scales["centering"]


def test_manifest_leakage_and_balance():
    manifest = make_manifest({"train": 32, "validation": 8, "topology_ood": 8}, seed=7)
    validate_manifest(manifest)
    for genus in (1, 2):
        cases = [
            c
            for c in manifest["configurations"]
            if c["split"] == "train" and c["geometry"]["genus"] == genus and c["family"] == "exact_velocity"
        ]
        assert {c["ic_seed"] % 2 for c in cases} == {0, 1}
    bad = json.loads(json.dumps(manifest))
    bad["configurations"][32]["geometry_id"] = bad["configurations"][0]["geometry_id"]
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(bad)
