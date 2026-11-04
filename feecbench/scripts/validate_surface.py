#!/usr/bin/env python3
"""Independent backend agreement and separate continuum/convergence evidence.

Examples (from feecbench/):
  uv run --extra surface python scripts/validate_surface.py --convergence-only
  uv run --project validation python scripts/validate_surface.py
  FEECBENCH_MFEM_EXECUTABLE=/path/to/bridge uv run --extra surface python scripts/validate_surface.py

Default backend subset: 3 genera x 2 geometries x 3 resolutions x 2 ICs.
Reports are development evidence, not a frozen scientific protocol.
"""
from __future__ import annotations

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "../reports/mpl-cache"))

import argparse
import importlib.util
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.spatial import ConvexHull

from feecbench.mesh.refine import refine
from feecbench.surface.data import atomic_json
from feecbench.surface.geometry import Geometry, generate
from feecbench.surface.initial import angular_harmonic, reference_initial, scalar_transfer
from feecbench.surface.mfem_validation import assemble_mfem, compare_operators, compare_trajectories
from feecbench.surface.operators import PYDEC_REVISION, assemble, harmonic_basis
from feecbench.surface.oracle import Midpoint, ModalControl, energy, pressure_modes


def sphere_meshes(levels=4):
    """Projected icosahedral refinements, used only for continuum verification."""
    golden = (1 + np.sqrt(5)) / 2
    points = np.array([(0, a, b * golden) for a in (-1, 1) for b in (-1, 1)]
                      + [(a, b * golden, 0) for a in (-1, 1) for b in (-1, 1)]
                      + [(b * golden, 0, a) for a in (-1, 1) for b in (-1, 1)], dtype=float)
    points /= np.linalg.norm(points, axis=1)[:, None]
    triangles = ConvexHull(points).simplices.copy()
    p = points[triangles]
    outward = np.einsum("ij,ij->i", np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), p.mean(axis=1))
    triangles[outward < 0] = triangles[outward < 0, ::-1]
    for _ in range(levels):
        yield points, triangles
        points, triangles, _ = refine(points, triangles)
        points /= np.linalg.norm(points, axis=1)[:, None]


def convergence_report(grid=12):
    """Separate temporal error, sphere continuum accuracy, and genus geometry studies."""
    points, tris, _ = generate(Geometry(genus=1), grid)[0]
    ops = assemble(points, tris, 1)
    values, vectors = pressure_modes(ops, min(16, len(points)-2))
    h = harmonic_basis(ops)
    mode = min(4, len(values)-1)
    p0 = vectors[:, mode]
    u0 = np.zeros(len(ops.edges))
    final_time = 0.32
    expected_p, expected_u = ModalControl(ops, values, vectors, h).solve(p0, u0, [final_time])
    rows = []
    for dt in (0.04, 0.02, 0.01, 0.005):
        p, u = Midpoint(ops).solve(p0, u0, [0, final_time], dt)
        ep, eu = p[-1]-expected_p[0], u[-1]-expected_u[0]
        error = float(np.sqrt(ep @ (ops.m0 @ ep) + eu @ (ops.m1 @ eu)))
        energies = energy(ops, p, u)
        rows.append({"dt": dt, "mass_error": error,
                     "energy_drift": float(np.max(np.abs(energies-energies[0])) / energies[0])})
    rates = [float(np.log2(a["mass_error"] / b["mass_error"])) for a, b in zip(rows[:-1], rows[1:], strict=True)]

    # On the unit sphere: -Delta x=2x, p=x*cos(sqrt(2)t),
    # u=-d(x)*sin(sqrt(2)t)/sqrt(2). Edge integral d(x) is exactly x_j-x_i.
    # This checks continuum spatial/geometric accuracy, independently of MFEM.
    sphere = []
    omega, time, temporal_step = np.sqrt(2.0), 0.4, 0.0005
    for level, (points, tris) in enumerate(sphere_meshes()):
        op = assemble(points, tris, genus=0)
        lam, _ = pressure_modes(op, 4)
        initial = points[:, 0]
        p, u = Midpoint(op).solve(initial, np.zeros(len(op.edges)), [time], temporal_step)
        ep = p[0] - initial * np.cos(omega*time)
        eu = u[0] + (op.d0 @ initial) * np.sin(omega*time) / omega
        error = float(np.sqrt(ep @ (op.m0 @ ep) + eu @ (op.m1 @ eu)))
        lengths = np.linalg.norm(points[op.edges[:, 1]]-points[op.edges[:, 0]], axis=1)
        sphere.append({"level": level, "vertices": len(points), "max_edge": float(lengths.max()),
                       "area_absolute_error": float(abs(op.area-4*np.pi)),
                       "first_positive_eigenvalues": lam[1:].tolist(),
                       "eigenvalue_max_error": float(np.max(np.abs(lam[1:]-2))),
                       "trajectory_mass_error": error})
    spatial_rates = [float(np.log(a["trajectory_mass_error"] / b["trajectory_mass_error"])
                           / np.log(a["max_edge"] / b["max_edge"]))
                     for a, b in zip(sphere[:-1], sphere[1:], strict=True)]
    # Surface-specific geometry/spectrum changes are self-convergence evidence;
    # unlike the sphere case, they do not have an analytical truth here.
    surface = []
    for genus in (1, 2, 3):
        for points, tris, metadata in generate(Geometry(genus=genus), grid, levels=3):
            op = assemble(points, tris, genus)
            lam, _ = pressure_modes(op, 8)
            surface.append({"genus": genus, "level": metadata["level"], "vertices": len(points),
                            "area": op.area, "geometric_implicit_residual": metadata["geometry_error"],
                            "low_eigenvalues": lam.tolist()})
    return {
        "temporal": {"case": "single semidiscrete eigenmode", "eigenvalue": float(values[mode]),
                     "rows": rows, "rates": rates, "second_order_observed": min(rates) > 1.8},
        "continuum_sphere": {"case": "unit-sphere spherical harmonic x", "time": time,
                             "dt": temporal_step, "rows": sphere, "rates": spatial_rates,
                             "errors_decrease": all(a["trajectory_mass_error"] > b["trajectory_mass_error"]
                                                     for a, b in zip(sphere[:-1], sphere[1:], strict=True)),
                             "interpretation": "Combined spatial/geometric error; sphere is verification only"},
        "surface_self_convergence": {"rows": surface,
                                     "interpretation": "Area/spectrum changes; no analytic multi-torus accuracy claim"},
    }


def mfem_subset(report_path, report, grid=12, levels=3, geometries=2, genera=(1, 2, 3)):
    """Run paired physical ICs at three resolutions and persist after each case."""
    rows = report.setdefault("mfem", {}).setdefault("cases", [])
    for genus in genera:
        for geometry_id in range(geometries):
            geometry = Geometry(genus=genus, radius=0.28+0.05*geometry_id,
                                thickness=0.5+0.08*geometry_id, stretch=0.95+0.1*geometry_id)
            meshes = generate(geometry, grid, levels=levels)
            points, tris, mesh_meta = meshes[-1]
            reference = assemble(points, tris, genus)
            values, vectors = pressure_modes(reference, min(32, len(points)-2))
            centers = geometry.centers * mesh_meta["scale"]
            initial_cases = {}
            for family in ("pulse_harmonic", "exact_velocity"):
                # Even seed enforces p0=0 for the velocity-driven comparison.
                initial_cases[family] = reference_initial(reference, values, vectors, family,
                                                         2*(100*genus+geometry_id), centers)
            for level, (points, tris, mesh_meta) in enumerate(meshes):
                ops = assemble(points, tris, genus)
                other = assemble_mfem(points, tris, ops.edges)
                harmonic = harmonic_basis(ops)
                transfer, transfer_meta = scalar_transfer(reference, points)
                operators = compare_operators(ops, other)
                for family, (p_ref, psi_ref, coeff, ic_meta) in initial_cases.items():
                    p0 = transfer @ p_ref
                    u0 = ops.d0 @ (transfer @ psi_ref) + angular_harmonic(ops, centers, coeff)
                    trajectory = compare_trajectories(ops, other, p0, u0,
                                                     np.array([0, 0.037, 0.08, 0.131, 0.2]), 0.002, harmonic)
                    rows.append({"genus": genus, "geometry": geometry_id, "level": level, "family": family,
                                 "operators": operators, "trajectory": trajectory, "mesh": mesh_meta,
                                 "initial": ic_meta, "transfer": transfer_meta})
                    atomic_json(report_path, report)
                    print(f"MFEM comparison {len(rows)}: genus={genus} geometry={geometry_id} level={level} {family}",
                          flush=True)
                # Sanity cases are explicitly separate from the requested 36.
                if genus == 1 and geometry_id == level == 0:
                    report["mfem"]["sanity"] = {
                        "pressure_only": compare_trajectories(ops, other, p0 + points[:, 0], np.zeros(len(ops.edges)),
                                                               [0, 0.1], 0.002, harmonic),
                        "stationary_harmonic": compare_trajectories(ops, other, np.zeros(len(points)), harmonic[:, 0],
                                                                    [0, 0.1], 0.002, harmonic),
                    }
    report["mfem"]["case_count"] = len(rows)
    report["mfem"]["complete_36_case_subset"] = len(rows) == 36
    # Development tolerances are reported, never silently promoted to a frozen protocol.
    matrix_errors = [row["operators"][name]["relative_frobenius"] for row in rows
                     for name in ("M0", "M1", "D0", "K")]
    trajectory_errors = [max(row["trajectory"][name]["relative_trajectory_error"]) for row in rows
                         for name in ("pressure", "velocity")]
    report["mfem"]["development_checks"] = {
        "matrix_relative_tolerance": 1e-9, "trajectory_relative_tolerance": 1e-8,
        "max_matrix_relative_error": max(matrix_errors), "max_trajectory_relative_error": max(trajectory_errors),
        "passed": max(matrix_errors) < 1e-9 and max(trajectory_errors) < 1e-8,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "reports/surface_validation.json")
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--convergence-only", action="store_true")
    parser.add_argument("--prototype", action="store_true", help="one geometry/level/genus, not the 36-case gate")
    parser.add_argument("--skip-convergence", action="store_true")
    args = parser.parse_args()
    start = perf_counter()
    report = {"status": "development validation; protocol not frozen", "pydec_revision": PYDEC_REVISION,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "platform": platform.platform()},
              "cross_backend_agreement_is_not_continuum_accuracy": True,
              "remaining_protocol_gates": ["reference error by spectral band", "paired-IC transfer uncertainty",
                                           "time-horizon and off-grid sampling freeze", "GPU agreement"]}
    atomic_json(args.output, report)
    if not args.skip_convergence:
        report["convergence"] = convergence_report(args.grid)
        atomic_json(args.output, report)
    if not args.convergence_only:
        try:
            available = (bool(os.environ.get("FEECBENCH_MFEM_EXECUTABLE"))
                         or importlib.util.find_spec("mfem.ser") is not None)
        except ModuleNotFoundError:
            available = False
        if not available:
            report["mfem"] = {"status": "not run: optional MFEM backend is unavailable", "case_count": 0}
            atomic_json(args.output, report)
            parser.exit(2, "MFEM backend unavailable; use the validation environment or compiled bridge. "
                           f"Partial report: {args.output}\n")
        mfem_subset(args.output, report, args.grid, levels=1 if args.prototype else 3,
                    geometries=1 if args.prototype else 2, genera=(1,) if args.prototype else (1, 2, 3))
    report["runtime_seconds"] = perf_counter()-start
    report["scientific_protocol_ready"] = False
    report["requested_checks_passed"] = (
        (args.skip_convergence or (report["convergence"]["temporal"]["second_order_observed"]
                                  and report["convergence"]["continuum_sphere"]["errors_decrease"]))
        and (args.convergence_only or report["mfem"]["development_checks"]["passed"]))
    atomic_json(args.output, report)
    print(args.output)
    if not report["requested_checks_passed"]:
        parser.exit(1, "Numerical development checks failed; inspect the report.\n")


if __name__ == "__main__":
    main()
