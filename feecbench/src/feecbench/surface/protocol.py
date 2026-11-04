"""Evidence-bound protocol freeze and train-only pressure band scales."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import atomic_json, digest, load

GATES = (
    "mfem_36_cases",
    "spatial_convergence",
    "temporal_convergence",
    "paired_initial_conditions",
    "pulse_resolution",
    "time_sampling",
    "orientation",
    "learning_smoke",
    "reference_uncertainty",
)


def freeze_protocol(root, evidence_path, output, thresholds=(10.0, 30.0), floor=0.01):
    """Freeze only evidence explicitly marked passed; do not infer scientific readiness.

    Evidence entries carry a report path and sha256, so merely setting a boolean
    without its underlying report is rejected. Numerical tolerance selection is
    recorded in each reviewed report rather than silently chosen by this function.
    """
    root, evidence_path = Path(root), Path(evidence_path)
    evidence = json.loads(evidence_path.read_text())
    if floor <= 0 or len(thresholds) != 2 or not 0 < thresholds[0] < thresholds[1]:
        raise ValueError("positive floor and increasing physical band thresholds required")
    for gate in GATES:
        record = evidence.get(gate, {})
        if record.get("passed") is not True:
            raise ValueError(f"protocol gate not passed: {gate}")
        path = evidence_path.parent / record["report"]
        import hashlib

        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"evidence hash mismatch: {gate}")
    reference_path = evidence_path.parent / evidence["reference_uncertainty"]["report"]
    reference_report = json.loads(reference_path.read_text())
    index = json.loads((root / "index.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if index.get("complete") is not True or {e["config_id"] for e in index["samples"]} != {
        c["config_id"] for c in manifest["configurations"]
    }:
        raise ValueError("protocol freeze requires a complete corpus")
    if (
        reference_report["fingerprint"] != index["fingerprint"]
        or reference_report["settings"] != manifest["settings"]
        or reference_report["thresholds"] != list(thresholds)
        or not reference_report["complete_corpus"]
    ):
        raise ValueError("reference uncertainty report does not cover this corpus, settings, or thresholds")
    resolved = reference_report["resolved_bands"]
    if not isinstance(resolved, list) or len(resolved) != 2 or not all(isinstance(x, bool) for x in resolved):
        raise ValueError("two explicit reference-resolution band gates required")
    energies = []
    for entry in index["samples"]:
        if entry["split"] != "train" or entry["resolution"] != "medium":
            continue
        sample = load(root / entry["path"])
        if sample.ops.genus == 3:
            raise ValueError("genus 3 cannot set spectral thresholds")
        if sample.values[-1] < thresholds[-1]:
            raise ValueError("increase eigensolver budget to cover physical middle-band cutoff")
        coeff = sample.pressure[(sample.times > 0) & ~sample.offgrid] @ (sample.ops.m0 @ sample.vectors)
        energies.append(
            [
                float(
                    np.mean(np.sum(coeff[:, (sample.values > lo) & (sample.values <= hi)] ** 2, axis=1))
                    / sample.ops.area
                )
                for lo, hi in zip((1e-8, thresholds[0]), thresholds, strict=True)
            ]
        )
    if not energies:
        raise ValueError("training data required")
    mean_energy = np.mean(energies, axis=0)
    protocol = {
        "version": 1,
        "development_fingerprint": index["fingerprint"],
        "numerical_settings": manifest["settings"],
        "development_configurations": [c["config_id"] for c in manifest["configurations"]],
        "development_geometries": [c["geometry_id"] for c in manifest["configurations"]],
        "evidence": evidence,
        "thresholds": list(thresholds),
        "floor": floor,
        "bands": [
            {
                "lower": lo,
                "upper": hi,
                "resolved": bool(ok and value > floor**2),
                "scale_raw": float(max(np.sqrt(value), floor)),
                "training_energy": float(value),
            }
            for lo, hi, ok, value in zip((1e-8, thresholds[0]), thresholds, resolved, mean_energy, strict=True)
        ],
    }
    protocol["hash"] = digest(protocol)
    atomic_json(output, protocol)
    return protocol


def read_protocol(path):
    protocol = json.loads(Path(path).read_text())
    expected = protocol.pop("hash")
    if digest(protocol) != expected:
        raise ValueError("protocol hash mismatch")
    protocol["hash"] = expected
    if not all(protocol["evidence"][gate]["passed"] for gate in GATES):
        raise ValueError("incomplete protocol")
    return protocol
