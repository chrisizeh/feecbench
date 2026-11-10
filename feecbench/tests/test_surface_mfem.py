"""Independent block integration checks and optional actual MFEM comparison."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from feecbench.surface.mfem_validation import (
    _signed_permutation,
    assemble_mfem,
    block_midpoint,
    compare_operators,
    matrix_discrepancy,
)


def test_block_midpoint_convergence_and_energy():
    """Both coupling directions converge to a separate matrix exponential."""
    ops = SimpleNamespace(
        m0=sp.csr_matrix([[2.0, 0.5], [0.5, 1.0]]),
        m1=sp.csr_matrix([[1.7]]),
        d0=sp.csr_matrix([[-1.0, 1.0]]),
    )
    mass = sp.block_diag((ops.m0, ops.m1)).toarray()
    coupling = ops.d0.T @ ops.m1
    generator = sp.bmat([[None, coupling], [-coupling.T, None]]).toarray()
    for initial in (np.array([1.0, -0.5, 0.0]), np.array([0.0, 0.0, 1.0])):
        exact = la.expm(la.solve(mass, generator) * 0.7) @ initial
        errors = []
        for step in (0.07, 0.035, 0.0175):
            p, u = block_midpoint(ops, initial[:2], initial[2:], np.array([0.0, 0.7]), step)
            state = np.c_[p, u]
            errors.append(la.norm(state[-1] - exact))
            energy = np.einsum("ti,ij,tj->t", state, mass, state)
            np.testing.assert_allclose(energy, energy[0], rtol=2e-14)
            np.testing.assert_allclose(
                np.ones(2) @ ops.m0 @ p[-1], np.ones(2) @ ops.m0 @ initial[:2], atol=2e-14)
        assert errors[0] / errors[1] > 3.9
        assert errors[1] / errors[2] > 3.9


@pytest.mark.parametrize("times,dt", [([-0.1, 0.2], 0.1), ([0, 0], 0.1), ([0, 1], 0), ([0, np.nan], 0.1)])
def test_block_midpoint_rejects_invalid_times(times, dt):
    with pytest.raises(ValueError):
        block_midpoint(None, np.zeros(2), np.zeros(1), times, dt)


def test_signed_permutation_and_zero_entry_discrepancy():
    permutation = _signed_permutation([-3, 0, -2], 3)
    np.testing.assert_array_equal(permutation @ [10, 20, 30], [20, -30, -10])
    np.testing.assert_array_equal((permutation.T @ permutation).toarray(), np.eye(3))
    with pytest.raises(ValueError):
        _signed_permutation([0, 0], 2)
    report = matrix_discrepancy(sp.eye(2, format="csr"), sp.csr_matrix([[1, 0.25], [0, 1]]))
    assert report["max_at_reference_zeros"] == 0.25
    assert report["max_absolute"] == 0.25


def test_mfem_embedded_torus_assembly_and_trajectory():
    """Runs real independent assembly when the optional backend is installed."""
    if not os.environ.get("FEECBENCH_MFEM_EXECUTABLE"):
        pytest.importorskip("mfem.ser", reason="requires optional MFEM backend")
    pytest.importorskip("pydec")
    from feecbench.surface.operators import assemble
    from feecbench.surface.oracle import Midpoint

    n, m = 8, 6
    angles = [(2 * np.pi * i / n, 2 * np.pi * j / m) for i in range(n) for j in range(m)]
    points = np.array([[(2 + 0.6 * np.cos(b)) * np.cos(a),
                        (2 + 0.6 * np.cos(b)) * np.sin(a), 0.6 * np.sin(b)] for a, b in angles])
    tris = []
    for i in range(n):
        for j in range(m):
            a, b, c, d = i * m + j, ((i + 1) % n) * m + j, ((i + 1) % n) * m + (j + 1) % m, i * m + (j + 1) % m
            tris.extend([(a, b, c), (a, c, d)])
    reference = assemble(points, np.asarray(tris), genus=1)
    candidate = assemble_mfem(reference.points, reference.tris, reference.edges)
    report = compare_operators(reference, candidate)
    for name in ("M0", "M1", "D0", "K", "MFEM_K_vs_D0T_M1_D0"):
        assert report[name]["relative_frobenius"] < 2e-12, (name, report[name])
        assert report[name]["max_at_reference_zeros"] < 2e-12, (name, report[name])
    rng = np.random.default_rng(42)
    signs = rng.choice([-1, 1], size=len(reference.edges))
    flipped = reference.flipped(signs)
    flipped_candidate = assemble_mfem(flipped.points, flipped.tris, flipped.edges)
    flipped_report = compare_operators(flipped, flipped_candidate)
    for name in ("M0", "M1", "D0", "K"):
        assert flipped_report[name]["relative_frobenius"] < 2e-12, (name, flipped_report[name])
    for p0, u0 in ((rng.normal(size=len(points)), np.zeros(len(reference.edges))),
                   (np.zeros(len(points)), reference.d0 @ rng.normal(size=len(points)))):
        times = np.array([0.0, 0.04, 0.1])
        expected_p, expected_u = Midpoint(reference).solve(p0, u0, times, 0.002)
        actual_p, actual_u = block_midpoint(candidate, p0, u0, times, 0.002)
        np.testing.assert_allclose(actual_p, expected_p, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(actual_u, expected_u, rtol=1e-10, atol=1e-11)
