"""Compatibility with the shared sample API; no corpus migration.

Conversion is explicit. Existing surface generation, storage and experiment readers
remain unchanged. Reference arrays are auxiliary data, never input/target cochains.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from ..evaluator import register_energy
from ..schema import FEECSample, Field, IdBlock, MeshBlock, SolverMeta, TopologyBlock
from .data import SurfaceSample
from .geometry import validate_mesh
from .operators import SurfaceOperators


@register_energy("surface_acoustic")
def acoustic_energy(sample: FEECSample, fields: dict) -> np.ndarray:
    """Unit-speed physical acoustic energy using the full consistent masses."""
    p, u = np.atleast_2d(fields["p"]), np.atleast_2d(fields["u"])
    return 0.5 * sum(
        np.einsum("ti,it->t", values, sample.operators[star] @ values.T)
        for values, star in ((p, "star0"), (u, "star1"))
    )


def to_feec(surface: SurfaceSample) -> FEECSample:
    """Represent a validated canonical surface sample without changing any arrays.

    Solver provenance must be present: this adapter does not invent missing data.
    The Courant number is descriptive; implicit midpoint has no CFL stability bound.
    """
    op, meta = surface.ops, deepcopy(surface.metadata)
    if meta.get("coefficient", 1.0) != 1.0 or meta["solver"] != "implicit_midpoint":
        raise ValueError("surface adapter supports unit-speed implicit-midpoint samples only")
    topology = validate_mesh(op.points, op.tris, op.genus)
    if not np.all(op.edges[:, 0] < op.edges[:, 1]):
        raise ValueError("conversion requires canonical low-to-high edge orientation")
    label = meta["resolution"]
    sid = meta.get('sample_id', f"{meta['config_id']}-{label}")
    dt = meta["dt"]
    return FEECSample(
        mesh=MeshBlock(
            op.points, op.edges, op.tris, np.zeros(len(op.points), bool), np.zeros(len(op.edges), bool),
            orientation_convention="edges_low_to_high; tris_consistently_oriented_closed_surface; "
                                   "d_k in R^{n_{k+1} x n_k}",
            source=deepcopy(meta.get("geometry_source")),
        ),
        fields={
            "p": Field(surface.pressure, 0, "target"),
            "u": Field(surface.velocity, 1, "target"),
            "p0": Field(surface.pressure[0], 0, "input"),
            "u0": Field(surface.velocity[0], 1, "input"),
        },
        operators={"d0": op.d0, "d1": op.d1, "star0": op.m0, "star1": op.m1, "star2": op.m2},
        star_types={"star0": "consistent_P1", "star1": "whitney_galerkin", "star2": "inverse_area"},
        params={
            "pde": "surface_acoustic", "c": 1.0, "genus": op.genus,
            "betti2": topology["betti"][2], "t_snapshots": surface.times.tolist(),
            "offgrid": surface.offgrid.tolist(), "surface_metadata": meta,
        },
        ids=IdBlock(meta["config_id"], sid, "medium" if label == "remeshed" else label,
                    refinement="remeshed" if label == "remeshed" else "nested", split=meta["split"]),
        topology=TopologyBlock(1, 2 * op.genus, *topology["betti"][:2], 0, 1, topology["betti"][2]),
        solver_meta=SolverMeta(
            meta["solver"], meta.get("solver_version", "unversioned"), dt,
            None,
            meta["seed"], meta["integration_seconds"], None, meta["energy_drift"],
            bc_type="closed_surface", notes="Unit speed; implicit midpoint has no CFL stability restriction.",
        ),
        auxiliary={"pressure_eigenvalues": surface.values, "pressure_eigenvectors": surface.vectors,
                   "harmonic_basis": surface.harmonic},
    )


def from_feec(sample: FEECSample) -> SurfaceSample:
    """Restore the surface view for numerical comparisons, including reference arrays."""
    if sample.params.get("pde") != "surface_acoustic":
        raise ValueError("expected a surface acoustic sample")
    mesh, ops, aux = sample.mesh, sample.operators, sample.auxiliary
    return SurfaceSample(
        SurfaceOperators(mesh.points, mesh.tris, mesh.edges, ops["d0"], ops["d1"],
                         ops["star0"], ops["star1"], ops["star2"], sample.params["genus"]),
        np.asarray(sample.params["t_snapshots"]), sample.fields["p"].data, sample.fields["u"].data,
        aux["pressure_eigenvalues"], aux["pressure_eigenvectors"], aux["harmonic_basis"],
        deepcopy(sample.params["surface_metadata"]), np.asarray(sample.params["offgrid"], dtype=bool),
    )
