"""Surface-only versioned, atomic HDF5 format and leakage-safe manifests."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import scipy.sparse as sp

from .operators import SurfaceOperators

VERSION = "surface-0.2.0"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, prefix=".pending-", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


@dataclass
class SurfaceSample:
    ops: SurfaceOperators
    times: np.ndarray
    pressure: np.ndarray
    velocity: np.ndarray
    values: np.ndarray
    vectors: np.ndarray
    harmonic: np.ndarray
    metadata: dict
    offgrid: np.ndarray


def save(path, sample):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
        tmp = Path(stream.name)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["version"] = VERSION
            f.attrs["genus"] = sample.ops.genus
            f.create_dataset("metadata", data=json.dumps(sample.metadata, allow_nan=False))
            for name in ("points", "tris", "edges"):
                f.create_dataset(name, data=getattr(sample.ops, name))
            for name in ("d0", "d1", "m0", "m1", "m2"):
                matrix = getattr(sample.ops, name).tocsr()
                group = f.create_group(name)
                group.attrs["shape"] = matrix.shape
                for key in ("data", "indices", "indptr"):
                    group.create_dataset(key, data=getattr(matrix, key))
            for name in ("times", "pressure", "velocity", "values", "vectors", "harmonic", "offgrid"):
                f.create_dataset(name, data=getattr(sample, name), compression="gzip")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load(path):
    with h5py.File(path, "r") as f:
        if f.attrs["version"] != VERSION:
            raise ValueError("incompatible surface schema")
        arrays = {name: f[name][()] for name in ("points", "tris", "edges")}
        for name in ("d0", "d1", "m0", "m1", "m2"):
            group = f[name]
            arrays[name] = sp.csr_matrix(
                tuple(group[k][()] for k in ("data", "indices", "indptr")), shape=tuple(group.attrs["shape"])
            )
        ops = SurfaceOperators(**arrays, genus=int(f.attrs["genus"]))
        return SurfaceSample(
            ops,
            **{
                name: f[name][()]
                for name in ("times", "pressure", "velocity", "values", "vectors", "harmonic", "offgrid")
            },
            metadata=json.loads(f["metadata"][()].decode()),
        )


def make_manifest(counts, seed=0, development=False):
    """Split physical configurations before meshing; geometry ids cannot cross splits."""
    from dataclasses import asdict

    from .geometry import Geometry
    from .initial import FAMILIES

    allowed = {"train", "validation", "id_test", "geometry_ood", "topology_ood", "verification"}
    if set(counts) - allowed or any(n < 0 or n % 4 for n in counts.values()):
        raise ValueError("known splits and nonnegative counts divisible by four required")
    rng = np.random.default_rng(seed)
    configurations = []
    for split, count in counts.items():
        for index in range(count):
            genus = 3 if split in ("topology_ood", "verification") else 1 + (index // 4) % 2
            radius = float(rng.uniform(0.38, 0.44) if split == "geometry_ood" else rng.uniform(0.25, 0.35))
            geometry = asdict(
                Geometry(
                    genus=genus,
                    radius=radius,
                    spacing=float(rng.uniform(1.05, 1.3)),
                    thickness=float(rng.uniform(0.45, 0.65)),
                    stretch=float(rng.uniform(0.9, 1.1)),
                )
            )
            gid = digest(geometry)[:20]
            item = {
                "geometry_id": gid,
                "geometry": geometry,
                "split": split,
                "family": FAMILIES[index % 4],
                "seed": seed * 100000 + len(configurations),
                "development": development,
                "coefficient": 1.0,
            }
            # Alternate zero-pressure exact cases independently of family index.
            item["ic_seed"] = item["seed"] * 2 + (index // 8) % 2
            item["config_id"] = digest(item)[:20]
            configurations.append(item)
    result = {"version": VERSION, "seed": seed, "development": development, "configurations": configurations}
    validate_manifest(result)
    return result


def validate_manifest(manifest):
    from .geometry import Geometry
    from .initial import FAMILIES

    if manifest.get("version") != VERSION:
        raise ValueError("unsupported manifest version")
    allowed = {"train", "validation", "id_test", "geometry_ood", "topology_ood", "verification"}
    configs, geometries = {}, {}
    for item in manifest["configurations"]:
        cid, gid, split = item["config_id"], item["geometry_id"], item["split"]
        geometry = Geometry(**item["geometry"])
        if split not in allowed or item["family"] not in FAMILIES or item["coefficient"] != 1.0:
            raise ValueError("invalid split, initial family, or non-unit acoustic coefficient")
        if split == "topology_ood" and geometry.genus != 3:
            raise ValueError("topology OOD must have genus 3")
        if cid in configs:
            raise ValueError("duplicate configuration")
        configs[cid] = split
        if gid in geometries and geometries[gid] != split:
            raise ValueError("geometry leakage")
        geometries[gid] = split
        if split in ("train", "validation", "id_test", "geometry_ood") and item["geometry"]["genus"] == 3:
            raise ValueError("genus 3 cannot enter learning development")
        if gid != digest(item["geometry"])[:20]:
            raise ValueError("geometry content hash mismatch")
        if cid != digest({k: v for k, v in item.items() if k != "config_id"})[:20]:
            raise ValueError("configuration content hash mismatch")


def training_scales(paths, floor=1e-8):
    """Streaming, equal configuration/time weighting, medium training data only."""
    totals, count = np.zeros(2), 0
    seen = set()
    for path in paths:
        sample = load(path)
        meta, op = sample.metadata, sample.ops
        if meta["split"] != "train" or meta["resolution"] != "medium":
            continue
        if op.genus == 3:
            raise ValueError("genus 3 cannot contribute to training scales")
        if meta["config_id"] in seen:
            raise ValueError("duplicate training configuration")
        seen.add(meta["config_id"])
        selected = ~sample.offgrid
        for i, (field, mass) in enumerate(((sample.pressure, op.m0), (sample.velocity, op.m1))):
            x = field[selected]
            totals[i] += np.einsum("ti,it->", x, mass @ x.T) / len(x) / op.area
        count += 1
    if not count:
        raise ValueError("no training medium data")
    return {
        "pressure": float(max(np.sqrt(totals[0] / count), floor)),
        "velocity": float(max(np.sqrt(totals[1] / count), floor)),
        "floor": floor,
        "training_configurations": count,
        "centering": False,
    }
