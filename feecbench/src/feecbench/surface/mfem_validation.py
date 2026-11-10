"""Optional independent MFEM assembly and full-block midpoint validation.

MFEM uses H1(order=1) scalars and ND(order=1) edge elements on exactly the
supplied piecewise-flat surface. Its own gradient interpolator and diffusion
integrator provide D0 and K. Signed permutation matrices align the assembled
DoFs to the supplied vertex/edge order; no PyDEC assembly is used here.

Install the pinned MFEM validation environment separately. PyMFEM 4.8.0.1
binary wheels are available for Linux x86_64. Alternatively build the bridge
with ``scripts/build_mfem_bridge.sh`` against serial MFEM 4.8 and set
``FEECBENCH_MFEM_EXECUTABLE`` to its binary.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.io import mmread


@dataclass
class MFEMOperators:
    """Operators in the benchmark's DoF order, copied out of MFEM-owned memory."""

    m0: sp.csr_matrix
    m1: sp.csr_matrix
    d0: sp.csr_matrix
    stiffness: sp.csr_matrix
    metadata: dict


def _csr(matrix) -> sp.csr_matrix:
    return sp.csr_matrix(
        (matrix.GetDataArray().copy(), matrix.GetJArray().copy(), matrix.GetIArray().copy()),
        shape=(matrix.Height(), matrix.Width()),
    )


def _signed_permutation(dofs: list[int], size: int) -> sp.csr_matrix:
    """Map benchmark cochains to MFEM coefficients (negative ids encode signs)."""
    dofs = np.asarray(dofs, dtype=np.int64)
    rows = np.where(dofs < 0, -1 - dofs, dofs)
    if len(dofs) != size or not np.array_equal(np.sort(rows), np.arange(size)):
        raise ValueError("MFEM DoF mapping is not a signed permutation")
    return sp.csr_matrix((np.where(dofs < 0, -1.0, 1.0), (rows, np.arange(size))), shape=(size, size))


def assemble_mfem(points: np.ndarray, tris: np.ndarray, edges: np.ndarray) -> MFEMOperators:
    """Assemble independent consistent masses, gradient, and scalar stiffness.

    ``edges`` may have arbitrary order and orientation. Its unordered edge set
    must equal the mesh's edges. Pressure follows the input vertex indices.
    """
    if executable := os.environ.get("FEECBENCH_MFEM_EXECUTABLE"):
        return _assemble_cpp(points, tris, edges, executable)
    try:
        import mfem.ser as mfem
    except ImportError as exc:
        raise ImportError("MFEM validation requires its separate environment or FEECBENCH_MFEM_EXECUTABLE") from exc

    start = perf_counter()
    points, tris, edges = np.asarray(points), np.asarray(tris), np.asarray(edges)
    if points.ndim != 2 or points.shape[1] != 3 or tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("expected embedded triangles with points (V, 3) and tris (F, 3)")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("expected edges (E, 2)")
    mesh = mfem.Mesh(2, len(points), len(tris), 0, 3)
    for point in points:
        mesh.AddVertex(point.tolist())
    for triangle in tris:
        mesh.AddTriangle(triangle.tolist(), 1)
    # Preserve the supplied surface orientation; planar orientation fixing is
    # not appropriate when the topological and embedding dimensions differ.
    mesh.FinalizeTriMesh(1, 0, False)
    h1, nd = mfem.H1_FECollection(1, 2), mfem.ND_FECollection(1, 2)
    scalar, velocity = mfem.FiniteElementSpace(mesh, h1), mfem.FiniteElementSpace(mesh, nd)
    if scalar.GetVSize() != len(points) or velocity.GetVSize() != len(edges):
        raise ValueError("unexpected MFEM lowest-order finite-element dimensions")

    vertex_dofs = []
    for vertex in range(len(points)):
        dofs = scalar.GetVertexDofs(vertex)
        if len(dofs) != 1:
            raise ValueError("expected one H1 DoF per vertex")
        vertex_dofs.append(dofs[0])
    edge_dofs = {}
    for edge in range(mesh.GetNEdges()):
        a, b = mesh.GetEdgeVertices(edge)
        dofs = velocity.GetEdgeDofs(edge)
        if len(dofs) != 1:
            raise ValueError("expected one ND DoF per edge")
        dof = dofs[0]
        edge_dofs[tuple(sorted((a, b)))] = dof if a < b else -1 - dof
    mapped = [edge_dofs[tuple(sorted((a, b)))] for a, b in edges]
    mapped = [dof if a < b else -1 - dof for dof, (a, b) in zip(mapped, edges, strict=True)]
    p0 = _signed_permutation(vertex_dofs, scalar.GetVSize())
    p1 = _signed_permutation(mapped, velocity.GetVSize())

    def bilinear(space, integrator):
        form = mfem.BilinearForm(space)
        form.AddDomainIntegrator(integrator)
        form.Assemble()
        form.Finalize()
        return _csr(form.SpMat())

    m0 = bilinear(scalar, mfem.MassIntegrator())
    m1 = bilinear(velocity, mfem.VectorFEMassIntegrator())
    stiffness = bilinear(scalar, mfem.DiffusionIntegrator())
    gradient = mfem.DiscreteLinearOperator(scalar, velocity)
    gradient.AddDomainInterpolator(mfem.GradientInterpolator())
    gradient.Assemble()
    gradient.Finalize()
    d0 = _csr(gradient.SpMat())
    return MFEMOperators(
        (p0.T @ m0 @ p0).tocsr(),
        (p1.T @ m1 @ p1).tocsr(),
        (p1.T @ d0 @ p0).tocsr(),
        (p0.T @ stiffness @ p0).tocsr(),
        {
            "backend": "MFEM",
            "version": version("mfem"),
            "assembly_seconds": perf_counter() - start,
            "spaces": "H1_1 / ND_1",
            "geometry": "piecewise_flat_embedded_triangles",
        },
    )


def _assemble_cpp(points, tris, edges, executable):
    """Use the committed C++ bridge when PyMFEM cannot be installed locally."""
    start = perf_counter()
    executable = Path(executable).expanduser().resolve(strict=True)
    with TemporaryDirectory(prefix="feecbench-mfem-") as directory:
        root = Path(directory)
        with (root / "mesh.txt").open("w") as stream:
            stream.write(f"{len(points)} {len(tris)}\n")
            np.savetxt(stream, points, fmt="%.17g")
            np.savetxt(stream, tris, fmt="%d")
        result = subprocess.run(
            [str(executable), str(root / "mesh.txt"), str(root / "out")], capture_output=True, text=True, check=False
        )
        if result.returncode:
            raise RuntimeError(f"MFEM C++ assembly failed ({result.returncode}): {result.stderr[-4000:]}")
        matrices = {name: sp.csr_matrix(mmread(root / f"out_{name}.mtx")) for name in ("m0", "m1", "d0", "stiffness")}
        p0 = _signed_permutation(np.loadtxt(root / "out_vertices.txt", dtype=int).tolist(), len(points))
        dofs = {}
        for a, b, dof in np.loadtxt(root / "out_edges.txt", dtype=int, ndmin=2):
            dofs[tuple(sorted((a, b)))] = int(dof if a < b else -1 - dof)
        mapped = [dofs[tuple(sorted((a, b)))] for a, b in edges]
        mapped = [dof if a < b else -1 - dof for dof, (a, b) in zip(mapped, edges, strict=True)]
        p1 = _signed_permutation(mapped, len(edges))
        backend_version = (root / "out_version.txt").read_text().strip()
    return MFEMOperators(
        (p0.T @ matrices["m0"] @ p0).tocsr(),
        (p1.T @ matrices["m1"] @ p1).tocsr(),
        (p1.T @ matrices["d0"] @ p0).tocsr(),
        (p0.T @ matrices["stiffness"] @ p0).tocsr(),
        {
            "backend": "MFEM C++",
            "version": backend_version,
            "assembly_seconds": perf_counter() - start,
            "spaces": "H1_1 / ND_1",
            "geometry": "piecewise_flat_embedded_triangles",
            "executable": str(executable),
        },
    )


def block_midpoint(ops, p0: np.ndarray, u0: np.ndarray, times: np.ndarray, dt: float):
    """Independent coupled solve, without the oracle's pressure elimination.

    Regular steps are fixed at ``dt``. An off-grid output uses a separate final
    remainder step, without altering the regular trajectory. The block matrix
    is the weak first-order acoustic system, testing both signs and masses.
    """
    times = np.asarray(times, dtype=np.float64)
    if (
        times.ndim != 1
        or not len(times)
        or np.any(times < 0)
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("times must be nonnegative and increase strictly; dt must be positive")
    n0, n1 = ops.m0.shape[0], ops.m1.shape[0]
    p0, u0 = np.asarray(p0, dtype=float), np.asarray(u0, dtype=float)
    if p0.shape != (n0,) or u0.shape != (n1,) or not np.all(np.isfinite(np.r_[p0, u0])):
        raise ValueError("initial fields must have finite values and match operator dimensions")
    mass = sp.block_diag((ops.m0, ops.m1), format="csc")
    coupling = ops.d0.T @ ops.m1
    generator = sp.bmat([[None, coupling], [-coupling.T, None]], format="csc")
    state = np.r_[p0, u0]
    factor = spla.splu(mass - 0.5 * dt * generator)
    right = mass + 0.5 * dt * generator
    snapshots, count = [], 0
    for time in times:
        required = int(np.floor(time / dt + 1e-12))
        while count < required:
            state = factor.solve(right @ state)
            count += 1
        remainder = time - count * dt
        if remainder > 1e-12 * dt:
            final_factor = spla.splu(mass - 0.5 * remainder * generator)
            query = final_factor.solve((mass + 0.5 * remainder * generator) @ state)
        else:
            query = state
        snapshots.append(query.copy())
    result = np.asarray(snapshots)
    return result[:, :n0], result[:, n0:]


def matrix_discrepancy(reference, candidate) -> dict[str, float]:
    """Frobenius relative error plus absolute checks, including reference zeros."""
    reference, candidate = reference.tocsr(), candidate.tocsr()
    if reference.shape != candidate.shape:
        raise ValueError("matrix shapes differ")
    difference = (candidate - reference).tocsr()
    norm = float(spla.norm(reference))
    on_support = candidate.multiply(reference.astype(bool))
    zero_error = candidate - on_support
    return {
        "relative_frobenius": float(spla.norm(difference)) / max(norm, np.finfo(float).tiny),
        "max_absolute": float(abs(difference).max()),
        "max_at_reference_zeros": float(abs(zero_error).max()),
    }


def compare_operators(reference, candidate: MFEMOperators) -> dict:
    """Report comparisons; tolerances belong to the frozen validation protocol."""
    stiffness = reference.d0.T @ reference.m1 @ reference.d0
    return {
        "M0": matrix_discrepancy(reference.m0, candidate.m0),
        "M1": matrix_discrepancy(reference.m1, candidate.m1),
        "D0": matrix_discrepancy(reference.d0, candidate.d0),
        "K": matrix_discrepancy(stiffness, candidate.stiffness),
        "MFEM_K_vs_D0T_M1_D0": matrix_discrepancy(candidate.stiffness, candidate.d0.T @ candidate.m1 @ candidate.d0),
        "metadata": candidate.metadata,
    }


def compare_trajectories(reference, candidate, p0, u0, times, dt, harmonic) -> dict:
    """Mass errors, energy histories, and basis-independent harmonic errors.

    Relative field errors use one trajectory-level denominator, so zero initial
    pressure/velocity snapshots do not cause singular relative errors.
    """
    from .oracle import Midpoint

    start = perf_counter()
    p_ref, u_ref = Midpoint(reference).solve(p0, u0, times, dt)
    reference_seconds = perf_counter() - start
    start = perf_counter()
    p_other, u_other = block_midpoint(candidate, p0, u0, times, dt)
    candidate_seconds = perf_counter() - start

    def squared(x, mass):
        return np.maximum(np.einsum("ti,it->t", x, mass @ x.T), 0)

    result = {}
    for name, expected, actual, mass in (
        ("pressure", p_ref, p_other, reference.m0),
        ("velocity", u_ref, u_other, reference.m1),
    ):
        error = np.sqrt(squared(actual - expected, mass))
        scale = max(float(np.sqrt(np.mean(squared(expected, mass)))), 1e-14)
        result[name] = {"absolute_mass_error": error.tolist(), "relative_trajectory_error": (error / scale).tolist()}
    energy_ref = 0.5 * (squared(p_ref, reference.m0) + squared(u_ref, reference.m1))
    energy_other = 0.5 * (squared(p_other, candidate.m0) + squared(u_other, candidate.m1))
    initial_energy = float(0.5 * (p0 @ (reference.m0 @ p0) + u0 @ (reference.m1 @ u0)))
    for name, values in (("reference", energy_ref), ("mfem", energy_other)):
        result[f"{name}_energy"] = values.tolist()
        result[f"{name}_energy_drift_relative"] = float(np.max(np.abs(values - initial_energy))) / max(
            initial_energy, 1e-14
        )
    # The norm of the projected difference is invariant to rotations/signs of H.
    projected_difference = (harmonic @ (harmonic.T @ (reference.m1 @ (u_other - u_ref).T))).T
    result["harmonic_projection_error"] = np.sqrt(squared(projected_difference, reference.m1)).tolist()
    initial_harmonic = harmonic @ (harmonic.T @ (reference.m1 @ u0))
    for name, u in (("reference", u_ref), ("mfem", u_other)):
        drift = (harmonic @ (harmonic.T @ (reference.m1 @ u.T))).T - initial_harmonic
        result[f"{name}_harmonic_drift"] = np.sqrt(squared(drift, reference.m1)).tolist()
    result["reference_seconds"], result["mfem_seconds"] = reference_seconds, candidate_seconds
    result["times"], result["dt"] = np.asarray(times).tolist(), float(dt)
    return result
