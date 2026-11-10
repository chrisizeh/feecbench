"""HDF5 persistence (one file per sample, exact round-trip) and the dataset index.

Layout of ``<root>/<config_id>/<sample_id>.h5``: root attr ``schema_version``; ``mesh/``
arrays with attrs ``orientation_convention``, ``quality_json``, ``source_json``;
``fields/<name>`` datasets with attrs ``form_degree``, ``role``, ``units``;
``operators/<name>/{indptr,indices,data}`` with attrs ``shape`` and ``star_type``;
``params_json`` dataset (attrs are capped at 64 KB); ``ids/`` attrs (+ ``prolongation``
dataset; None-valued attrs are absent, never aliased to ""); ``topology/`` and
``solver_meta/`` attrs. ``load_sample(save_sample(s)) == s`` bit-exactly (tests/test_io.py).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import scipy.sparse as sp

from .schema import (
    SCHEMA_VERSION,
    SUPPORTED_VERSIONS,
    FEECSample,
    Field,
    IdBlock,
    MeshBlock,
    SolverMeta,
    TopologyBlock,
)

MESH_ARRAYS = ("points", "edges", "tris", "boundary_vertices", "boundary_edges")


def _py(v):
    """h5py attribute -> plain Python scalar/str."""
    return v.item() if isinstance(v, np.generic) else v


def save_sample(sample: FEECSample, root: str | Path) -> Path:
    path = Path(root) / sample.ids.config_id / f"{sample.ids.sample_id}.h5"
    _save_atomic(sample, path)
    return path


def _save_atomic(sample: FEECSample, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
        tmp = Path(stream.name)
    try:
        _write_sample(sample, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_sample(sample: FEECSample, path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        g = f.create_group("mesh")
        g.attrs["orientation_convention"] = sample.mesh.orientation_convention
        for key in ("quality", "source"):
            if getattr(sample.mesh, key) is not None:
                g.attrs[f"{key}_json"] = json.dumps(getattr(sample.mesh, key))
        for name in MESH_ARRAYS:
            g.create_dataset(name, data=getattr(sample.mesh, name))
        g = f.create_group("fields")
        for name, fld in sample.fields.items():
            d = g.create_dataset(name, data=fld.data)
            d.attrs.update(form_degree=fld.form_degree, role=fld.role, units=fld.units)
        g = f.create_group("operators")
        for name, op in sample.operators.items():
            csr, og = op.tocsr(), g.create_group(name)
            for part in ("indptr", "indices", "data"):
                og.create_dataset(part, data=getattr(csr, part))
            og.attrs["shape"] = csr.shape
            if name in sample.star_types:
                og.attrs["star_type"] = sample.star_types[name]
        f.create_dataset("params_json", data=json.dumps(sample.params))
        g = f.create_group("ids")
        for key, val in vars(sample.ids).items():
            if key == "prolongation" and val is not None:
                g.create_dataset(key, data=val)
            elif key != "prolongation" and val is not None:
                g.attrs[key] = val
        f.create_group("topology").attrs.update(vars(sample.topology))
        f.create_group("solver_meta").attrs.update(
            {k: v for k, v in vars(sample.solver_meta).items() if v is not None}
        )
        g = f.create_group("auxiliary")
        for name, values in sample.auxiliary.items():
            g.create_dataset(name, data=values)
    return path


def load_sample(path: str | Path) -> FEECSample:
    with h5py.File(path, "r") as f:
        version = f.attrs.get("schema_version")
        if version is None and f.attrs.get("version") == "surface-0.2.0":
            from .surface.compat import to_feec
            from .surface.data import load

            return to_feec(load(path))
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"{path}: unsupported schema version {version or f.attrs.get('version')!r}")
        g = f["mesh"]
        mesh = MeshBlock(
            *(g[name][()] for name in MESH_ARRAYS[:3]),
            *(g[name][()].astype(bool) for name in MESH_ARRAYS[3:]),
            orientation_convention=str(g.attrs["orientation_convention"]),
            **{key: json.loads(g.attrs[f"{key}_json"]) for key in ("quality", "source") if f"{key}_json" in g.attrs},
        )
        fields = {
            name: Field(d[()], int(d.attrs["form_degree"]), str(d.attrs["role"]), str(d.attrs["units"]))
            for name, d in f["fields"].items()
        }
        operators, star_types = {}, {}
        for name, og in f["operators"].items():
            shape = tuple(int(s) for s in og.attrs["shape"])
            operators[name] = sp.csr_matrix((og["data"][()], og["indices"][()], og["indptr"][()]), shape=shape)
            operators[name].indices = og["indices"][()]
            operators[name].indptr = og["indptr"][()]
            if "star_type" in og.attrs:
                star_types[name] = str(og.attrs["star_type"])
        params = json.loads(f["params_json"][()].decode())
        g = f["ids"]
        ids = IdBlock(**{k: _py(v) for k, v in g.attrs.items()})
        ids.prolongation = g["prolongation"][()] if "prolongation" in g else None
        topology = TopologyBlock(**{k: _py(v) for k, v in f["topology"].attrs.items()})
        solver_meta = SolverMeta(**{
            "cfl": None, "cfl_satisfied": None, **{k: _py(v) for k, v in f["solver_meta"].attrs.items()}
        })
        auxiliary = {k: v[()] for k, v in f.get("auxiliary", {}).items()}
    return FEECSample(mesh, fields, operators, star_types, params, ids, topology, solver_meta, auxiliary)


def write_index(root: str | Path, splits: dict[str, str] | None = None) -> Path:
    """Scan ``root`` for sample files and write ``index.json`` (``splits`` overrides stored labels)."""
    root, entries = Path(root), []
    existing = root / "index.json"
    if existing.exists() and "fingerprint" in json.loads(existing.read_text()):
        raise ValueError("cannot replace a fingerprinted corpus index with a directory scan")
    for path in sorted(root.glob("*/*.h5")):
        with h5py.File(path, "r") as f:
            a = {k: _py(v) for k, v in f["ids"].attrs.items()}
        entries.append(
            {
                "path": str(path.relative_to(root)),
                **{k: a[k] for k in ("config_id", "sample_id", "resolution_level", "refinement")},
                "split": (splits or {}).get(a["sample_id"], a.get("split")),
            }
        )
    index_path = root / "index.json"
    index_path.write_text(json.dumps({"samples": entries}, indent=2))
    return index_path
