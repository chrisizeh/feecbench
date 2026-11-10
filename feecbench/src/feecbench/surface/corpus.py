"""Resumable, bounded-memory CPU generation with configuration hash checks."""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np

from .data import SurfaceSample, atomic_json, digest, load, validate_manifest
from .geometry import Geometry, generate
from .initial import angular_harmonic, reference_initial, scalar_transfer
from .operators import PYDEC_REVISION, assemble, harmonic_basis
from .oracle import Midpoint, energy, pressure_modes


def generate_corpus(
    root, manifest, grid=16, dt=0.005, horizon=1.0, snapshots=51, modes=32, paired=False, progress=False
):
    """One configuration at a time; output snapshots do not count as configurations."""
    if dt <= 0 or horizon <= 0 or snapshots < 2 or modes < 2:
        raise ValueError("invalid temporal/spectral configuration")
    validate_manifest(manifest)
    root = Path(root)
    settings = {
        "grid": grid,
        "dt": dt,
        "horizon": horizon,
        "snapshots": snapshots,
        "modes": modes,
        "paired": paired,
        "pydec_revision": PYDEC_REVISION,
        "generator_version": 3,
        "storage_schema": "0.2.0",
        "schema": manifest["version"],
    }
    fingerprint = digest({"manifest": manifest, "settings": settings})
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        import json

        existing = json.loads(manifest_path.read_text())
        if existing["fingerprint"] != fingerprint:
            raise ValueError("existing corpus configuration hash mismatch; use a new output directory")
    atomic_json(manifest_path, {"fingerprint": fingerprint, "settings": settings, **manifest})
    entries = []
    atomic_json(root / "index.json", {"fingerprint": fingerprint, "samples": [], "complete": False})
    for number, item in enumerate(manifest["configurations"], 1):
        start = time.perf_counter()
        geometry = Geometry(**item["geometry"])
        variants = ["medium"]
        if paired and item["split"] == "id_test":
            variants = ["coarse", "medium", "fine", "remeshed"]
        pending = []
        for label in variants:
            path = root / item["config_id"] / f"{label}.h5"
            entry = {"path": str(path.relative_to(root)), **item, "resolution": label,
                     "resolution_level": "medium" if label == "remeshed" else label,
                     "refinement": "remeshed" if label == "remeshed" else "nested"}
            entries.append(entry)
            if path.exists():
                from ..io import load_sample

                entry["sample_id"] = load_sample(path).ids.sample_id
                if load(path).metadata.get("fingerprint") != fingerprint:
                    raise ValueError(f"sample hash mismatch: {path}")
            else:
                pending.append((label, path))
        if not pending:
            continue
        if progress:
            print(f"Generating configuration {number}/{len(manifest['configurations'])}: {item['split']}", flush=True)
        # A common finer physical reference is used even when only medium is requested.
        meshes = generate(geometry, grid, levels=3)
        rp, rt, rmeta = meshes[-1]
        reference = assemble(rp, rt, geometry.genus)
        spectrum_start = time.perf_counter()
        rv, rphi = pressure_modes(reference, min(modes, len(rp) - 2))
        reference_spectral_seconds = time.perf_counter() - spectrum_start
        centers = geometry.centers * rmeta["scale"]
        p_ref, psi_ref, coeff, ic = reference_initial(reference, rv, rphi, item["family"], item["ic_seed"], centers)
        ref_hash = digest({"points": rp.tolist(), "triangles": rt.tolist(), "ic": ic})
        for label, path in pending:
            entry = next(e for e in entries if e["path"] == str(path.relative_to(root)))
            if label == "remeshed":
                points, tris, mesh_meta = generate(geometry, 2 * grid + 1, phase=0.37)[0]
            else:
                points, tris, mesh_meta = meshes[{"coarse": 0, "medium": 1, "fine": 2}[label]]
            ops = assemble(points, tris, geometry.genus)
            transfer, transfer_meta = scalar_transfer(reference, points)
            p0, psi = transfer @ p_ref, transfer @ psi_ref
            u0 = ops.d0 @ psi + angular_harmonic(ops, centers, coeff)
            ts = time.perf_counter()
            values, vectors = pressure_modes(ops, min(modes, len(points) - 2))
            harmonic = harmonic_basis(ops)
            spectral_seconds = time.perf_counter() - ts
            regular = np.linspace(0, horizon, snapshots)
            extra = np.array([]) if item["split"] == "train" else horizon * np.array([0.137, 0.413, 0.829])
            extra = extra[~np.isclose(extra[:, None], regular[None, :], atol=1e-12).any(axis=1)]
            times = np.sort(np.concatenate((regular, extra)))
            offgrid = np.isin(times, extra)
            ts = time.perf_counter()
            pressure, velocity = Midpoint(ops).solve(p0, u0, times, dt)
            integration_seconds = time.perf_counter() - ts
            energies = energy(ops, pressure, velocity)
            drift = float(np.max(np.abs(energies - energies[0])) / max(energies[0], 1e-30))
            if not np.isfinite(energies).all() or drift > 1e-8:
                raise ValueError(f"reference energy conservation failed: {drift}")
            metadata = {
                **item,
                "resolution": label,
                "fingerprint": fingerprint,
                "geometry_source": mesh_meta,
                "reference_mesh_hash": ref_hash,
                "initial": ic,
                "transfer": transfer_meta,
                "solver": "implicit_midpoint",
                "dt": dt,
                "coefficient": 1.0,
                "pydec_revision": PYDEC_REVISION,
                "python": platform.python_version(),
                "energy_drift": drift,
                "generation_seconds": time.perf_counter() - start,
                "integration_seconds": integration_seconds,
                "spectral_seconds": spectral_seconds,
                "reference_spectral_seconds": reference_spectral_seconds,
                "status": "development; reference-resolution/protocol gates required",
            }
            from ..io import _save_atomic
            from .compat import to_feec

            shared = to_feec(SurfaceSample(
                ops, times, pressure, velocity, values, vectors, harmonic, metadata, offgrid
            ))
            _save_atomic(shared, path)
            entry["sample_id"] = shared.ids.sample_id
        atomic_json(root / "index.json", {"fingerprint": fingerprint, "samples": entries, "complete": False})
    atomic_json(root / "index.json", {"fingerprint": fingerprint, "samples": entries, "complete": True})
    return entries
