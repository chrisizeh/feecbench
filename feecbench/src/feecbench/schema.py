"""FEECSample: the typed, FEEC-native sample contract.

Seven blocks: ``mesh`` (cells, boundary masks, orientation, quality, generator provenance),
``fields`` (named cochains with form degree / role / units), ``operators`` (sparse d0, d1,
star0..2 with star-type tags), ``params``, ``ids`` (config / sample identity, nesting,
prolongation, split), ``topology`` (analytic + computed Betti numbers) and ``solver_meta``.
Derived operators (codifferential, Hodge Laplacians) are never stored -- see operators.py.
Unused cochain degrees are simply absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass

import numpy as np
import scipy.sparse as sp

SCHEMA_VERSION = "0.2.0"
SUPPORTED_VERSIONS = ("0.1.0", SCHEMA_VERSION)
LEVELS = ("coarse", "medium", "fine")
FIELD_ROLES = ("input", "target", "coefficient")
ORIENTATION = "edges_low_to_high; tris_ccw; d_k in R^{n_{k+1} x n_k}"
STAR_TYPES = {"star0": "lumped_P1", "star1": "whitney_galerkin", "star2": "diagonal_area"}


@dataclass
class Field:
    """One typed cochain field; ``data`` is ``(n_k,)`` or ``(T, n_k)`` with k = form_degree."""

    data: np.ndarray
    form_degree: int
    role: str  # "input" | "target" | "coefficient"
    units: str = "1"

    def __post_init__(self):
        if self.role not in FIELD_ROLES:
            raise ValueError(f"bad field role {self.role!r}")


@dataclass
class MeshBlock:
    points: np.ndarray  # (n0, 2) planar or (n0, 3) embedded surface
    edges: np.ndarray  # (n1, 2) int64, sorted pairs i < j, lexicographic
    tris: np.ndarray  # (n2, 3) int64, counter-clockwise
    boundary_vertices: np.ndarray  # (n0,) bool
    boundary_edges: np.ndarray  # (n1,) bool
    orientation_convention: str = ORIENTATION
    quality: dict | None = None  # mesh-quality stats (mesh.quality)
    source: dict | None = None  # generator provenance: gmsh version/options/seed, derivation

    @property
    def n_cells(self) -> tuple[int, int, int]:
        return (len(self.points), len(self.edges), len(self.tris))


@dataclass
class IdBlock:
    config_id: str  # one physical configuration, shared across discretizations
    sample_id: str  # unique per discretization
    resolution_level: str  # "coarse" | "medium" | "fine"
    refinement: str = "nested"  # "nested" | "remeshed"
    parent_sample_id: str | None = None
    prolongation: np.ndarray | None = None  # parent vertex id -> this sample's vertex id
    split: str | None = None


@dataclass
class TopologyBlock:
    betti0: int
    betti1: int
    betti0_computed: int
    betti1_computed: int
    n_boundary_components: int
    betti2: int = 0
    betti2_computed: int = 0


@dataclass
class SolverMeta:
    solver_name: str
    solver_version: str
    dt: float
    cfl: float | None
    seed: int
    runtime_seconds: float
    cfl_satisfied: bool | None
    energy_drift_rel: float  # max relative drift of the exactly conserved discrete energy
    bc_type: str = "sound_hard"
    notes: str = ""


@dataclass
class FEECSample:
    mesh: MeshBlock
    fields: dict[str, Field]
    operators: dict[str, sp.csr_matrix]  # d0, d1, star0, star1, star2
    star_types: dict[str, str]
    params: dict
    ids: IdBlock
    topology: TopologyBlock
    solver_meta: SolverMeta
    auxiliary: dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> str:
        n0, n1, n2 = self.mesh.n_cells
        i, t, sm = self.ids, self.topology, self.solver_meta
        lines = [
            f"FEECSample {i.sample_id}  (config {i.config_id}, {i.resolution_level}, {i.refinement})",
            f"  mesh: n0={n0} n1={n1} n2={n2}   beta=({t.betti0},{t.betti1})  "
            f"boundary components={t.n_boundary_components}",
            f"  orientation: {self.mesh.orientation_convention}",
            "  fields:",
            *(
                f"    {k:8s} degree={f.form_degree} role={f.role:11s} "
                f"shape={tuple(f.data.shape)} units={f.units}"
                for k, f in self.fields.items()
            ),
            "  operators:",
            *(
                f"    {k:6s} shape={op.shape} nnz={op.nnz} dtype={op.dtype} "
                f"type={self.star_types.get(k, '-')}"
                for k, op in self.operators.items()
            ),
            f"  solver: {sm.solver_name} v{sm.solver_version}  dt={sm.dt:.3e} cfl={sm.cfl}  "
            f"seed={sm.seed}  energy_drift={sm.energy_drift_rel:.2e}",
        ]
        return "\n".join(lines)


def first_difference(a, b, path: str = "sample") -> str | None:
    """Path of the first mismatch between two nested sample structures, or None if equal.

    Arrays compare bit-exactly (dtype included), sparse matrices by CSR triplets; lists,
    tuples and arrays compare element-wise so JSON round-trips of ``params`` are tolerated.
    """
    if sp.issparse(a) or sp.issparse(b):
        a, b = a.tocsr(), b.tocsr()
        if a.shape != b.shape or a.dtype != b.dtype:
            return f"{path} (sparse shape/dtype)"
        parts = ("indptr", "indices", "data")
        return next((f"{path}.{k}" for k in parts if not np.array_equal(getattr(a, k), getattr(b, k))), None)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype != b.dtype:
            return f"{path} (dtype)"
        return None if np.array_equal(np.asarray(a), np.asarray(b)) else path
    if is_dataclass(a):
        pairs = ((getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}") for f in fields(a))
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return f"{path} (keys)"
        pairs = ((a[k], b[k], f"{path}[{k!r}]") for k in a)
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return f"{path} (length)"
        pairs = ((x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b, strict=True)))
    else:
        return None if a == b else path
    return next((d for x, y, p in pairs if (d := first_difference(x, y, p))), None)


def assert_samples_equal(a: FEECSample, b: FEECSample) -> None:
    """Bit-exact equality of two samples (used by the IO round-trip test)."""
    diff = first_difference(a, b)
    assert diff is None, f"samples differ at {diff}"
