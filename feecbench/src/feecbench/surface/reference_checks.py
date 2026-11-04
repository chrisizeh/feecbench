"""Measured development reference uncertainty; no inference from energy alone."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import atomic_json, load
from .geometry import Geometry, generate
from .initial import angular_harmonic, reference_initial, scalar_transfer
from .metrics import norm2
from .operators import Derived, assemble
from .oracle import Midpoint, pressure_modes


def _errors(sample, p, u, thresholds, floor):
    op = sample.ops
    selected = sample.times > 0
    target = sample.pressure[selected]
    difference = sample.pressure[selected] - p[selected]
    projection = op.m0 @ sample.vectors
    coefficients, reference = difference @ projection, target @ projection
    result = {}
    for name, low, high in zip(("low", "middle"), (1e-8, thresholds[0]), thresholds, strict=True):
        mask = (sample.values > low) & (sample.values <= high)
        target_energy = float(np.mean(np.sum(reference[:, mask] ** 2, axis=1)) / op.area)
        error = float(np.sqrt(np.mean(np.sum(coefficients[:, mask] ** 2, axis=1)) / op.area))
        result[name] = {
            "covered": bool(sample.values[-1] >= high),
            "target_energy": target_energy,
            "mass_rmse": error,
            "normalized_rmse": error / max(np.sqrt(target_energy), floor),
        }
    velocity_scale = max(np.sqrt(np.mean(norm2(sample.velocity[selected], op.m1)) / op.area), floor)
    result["velocity"] = {
        "normalized_rmse": float(
            np.sqrt(np.mean(norm2(sample.velocity[selected] - u[selected], op.m1)) / op.area) / velocity_scale
        )
    }
    return result


def check_references(root, output, thresholds=(10.0, 30.0), limit=None, tolerance=0.05, floor=0.01):
    """Compare dt/2 and projected fine reference against each own-mesh target.

    Velocity transfer is valid only on this corpus's exact-plus-harmonic subspace:
    scalar potential interpolation commutes with D0 by construction, and the
    angular period prescription fixes the harmonic part. This is not a general
    Whitney 1-form transfer and is never used to score arbitrary model predictions.
    """
    root = Path(root)
    index = json.loads((root / "index.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if index.get("complete") is not True or {e["config_id"] for e in index["samples"]} != {
        c["config_id"] for c in manifest["configurations"]
    }:
        raise ValueError("reference checks require a complete corpus")
    settings = manifest["settings"]
    rows = []
    for entry in index["samples"]:
        if entry["resolution"] != "medium":
            continue
        sample = load(root / entry["path"])
        op, meta = sample.ops, sample.metadata
        geometry = Geometry(**meta["geometry"])
        points, tris, source = generate(geometry, settings["grid"], levels=3)[-1]
        fine = assemble(points, tris, geometry.genus)
        lam, phi = pressure_modes(fine, min(settings["modes"], len(points) - 2))
        centers = geometry.centers * source["scale"]
        p0, psi0, coefficients, _ = reference_initial(fine, lam, phi, meta["family"], meta["ic_seed"], centers)
        fh = angular_harmonic(fine, centers, coefficients)
        fp, fu = Midpoint(fine).solve(p0, fine.d0 @ psi0 + fh, sample.times, settings["dt"] / 2)
        transfer, transfer_meta = scalar_transfer(fine, op.points)
        derived = Derived(fine)
        rhs = fine.d0.T @ (fine.m1 @ (fu - fh).T)
        potentials = np.zeros((len(points), len(sample.times)))
        potentials[1:] = derived.solve_k(rhs[1:])
        mapped_pressure = (transfer @ fp.T).T
        mh = angular_harmonic(op, centers, coefficients)
        mapped_velocity = (op.d0 @ (transfer @ potentials)).T + mh
        fine_coexact = (fu - fh).T - fine.d0 @ potentials
        if np.linalg.norm(fine_coexact) > 1e-7 * max(np.linalg.norm(fu), 1):
            raise ValueError("reference is not in exact-plus-prescribed-harmonic transfer subspace")
        tp, tu = Midpoint(op).solve(sample.pressure[0], sample.velocity[0], sample.times, settings["dt"] / 2)
        maximum_edge = meta["geometry_source"]["quality"]["edge_max"]
        pulse_ratios = [pulse["width"] / maximum_edge for pulse in meta["initial"]["pulses"]]
        active = lam[(lam > 1e-8) & (lam <= thresholds[-1])]
        period_min = float(2 * np.pi / np.sqrt(active.max())) if len(active) else None
        interval = settings["horizon"] / (settings["snapshots"] - 1)
        rows.append(
            {
                "config_id": meta["config_id"],
                "genus": op.genus,
                "family": meta["family"],
                "split": meta["split"],
                "temporal": _errors(sample, tp, tu, thresholds, floor),
                "spatial_plus_geometry": _errors(sample, mapped_pressure, mapped_velocity, thresholds, floor),
                "initial_pressure_rmse": float(
                    np.sqrt(norm2(sample.pressure[0] - mapped_pressure[0], op.m0)[0] / op.area)
                ),
                "initial_velocity_rmse": float(
                    np.sqrt(norm2(sample.velocity[0] - mapped_velocity[0], op.m1)[0] / op.area)
                ),
                "transfer": transfer_meta,
                "pulse_width_over_max_edge": pulse_ratios,
                "minimum_evaluated_period": period_min,
                "samples_per_minimum_period": period_min / interval if period_min else None,
                "horizon_over_area_length": settings["horizon"] / np.sqrt(op.area),
                "mesh_quality": meta["geometry_source"]["quality"],
                "low_spectrum": lam[:12].tolist(),
                "hole_diameter": 2 * geometry.radius * source["scale"],
                "inter_hole_gap": (geometry.spacing - 2 * geometry.radius) * source["scale"] if op.genus > 1 else None,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    development = [row for row in rows if row["genus"] in (1, 2)]
    if not development:
        raise ValueError("genus1--2 development references required")
    gates = []
    for band in ("low", "middle"):
        gates.append(
            all(
                row[kind][band]["covered"] and row[kind][band]["normalized_rmse"] <= tolerance
                for row in development
                for kind in ("temporal", "spatial_plus_geometry")
            )
        )
    report = {
        "fingerprint": index["fingerprint"],
        "settings": settings,
        "thresholds": list(thresholds),
        "relative_tolerance": tolerance,
        "absolute_scale_floor": floor,
        "resolved_bands": gates,
        "complete_corpus": limit is None,
        "cases": rows,
        "interpretation": "fine-grid differences estimate uncertainty; they are not continuum error bounds",
    }
    atomic_json(output, report)
    return report
