"""Star-weighted normalization statistics (The Well style, FEEC-native).

Computed over the train split only, per field, in the field's own ``star_k`` inner product
(mesh-size independent). For a field z of degree k over all train samples s and snapshots t
(static fields count as one snapshot):

    vol = sum 1^T star_k 1,  mean = sum 1^T star_k z / vol,
    rms = sqrt(sum z^T star_k z / vol),  std = sqrt(rms^2 - mean^2)

``delta`` holds the same moments of the snapshot differences ``z_{t+1} - z_t`` (time series
only). ``Normalizer`` applies ``zscore: (z - mean) / std`` or ``rms: z / rms`` and the exact
inverses; ``FEECDataset(normalize=...)`` uses it on load and the evaluator always receives
raw fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EPS = 1e-30


def _accumulate(acc: dict, z: np.ndarray, star, ones: np.ndarray, vol: float) -> None:
    acc["w"] += vol * len(z)
    acc["wz"] += float(ones @ (star @ z.sum(axis=0)))
    acc["wz2"] += float(np.einsum("ti,ti->", z, (star @ z.T).T))


def _moments(acc: dict) -> dict:
    mean = acc["wz"] / (acc["w"] + EPS)
    rms = float(np.sqrt(max(acc["wz2"] / (acc["w"] + EPS), 0.0)))
    return {"mean": mean, "std": float(np.sqrt(max(rms**2 - mean**2, 0.0))), "rms": rms}


def compute_stats(samples, *, policy=None) -> dict:
    """Star-weighted mean/std/rms (+ ``delta`` moments for time series) per field."""
    if policy == "surface_energy":
        from .surface.compat import from_feec

        totals, seen = np.zeros(2), set()
        for shared in samples:
            if (shared.ids.split != "train" or shared.ids.resolution_level != "medium"
                    or shared.ids.refinement == "remeshed"):
                continue
            s = from_feec(shared)
            cid = shared.ids.config_id
            if s.ops.genus == 3 or cid in seen:
                raise ValueError("genus-3 or duplicate training configuration")
            seen.add(cid)
            selected = ~s.offgrid
            for i, (values, mass) in enumerate(((s.pressure, s.ops.m0), (s.velocity, s.ops.m1))):
                x = values[selected]
                totals[i] += np.einsum("ti,it->", x, mass @ x.T) / len(x) / s.ops.area
        if not seen:
            raise ValueError("no training medium data")
        scales = np.maximum(np.sqrt(totals / len(seen)), 1e-8)
        return {"version": 1, "policy": policy, "provenance": {"configurations": sorted(seen)},
                "fields": {n: {"rms": float(scales[i])} for n, i in (("p", 0), ("p0", 0), ("u", 1), ("u0", 1))}}
    if policy not in (None, "star_weighted"):
        raise ValueError(f"unknown normalization policy {policy!r}")
    acc: dict[str, dict] = {}
    for s in samples:
        if policy is not None and s.ids.split != "train":
            continue
        for name, fld in s.fields.items():
            star = s.operators[f"star{fld.form_degree}"]
            ones = np.ones(star.shape[0])
            vol = float(ones @ (star @ ones))
            a = acc.setdefault(name, {"degree": fld.form_degree, "z": _zero(), "dz": _zero(), "series": False})
            z = np.atleast_2d(fld.data)
            _accumulate(a["z"], z, star, ones, vol)
            if fld.data.ndim == 2:
                a["series"] = True
                _accumulate(a["dz"], np.diff(z, axis=0), star, ones, vol)
    return {
        name: {"form_degree": a["degree"], **_moments(a["z"]), **({"delta": _moments(a["dz"])} if a["series"] else {})}
        for name, a in acc.items()
    }


def _zero() -> dict:
    return {"w": 0.0, "wz": 0.0, "wz2": 0.0}


def write_stats(root: str | Path, stats: dict) -> Path:
    path = Path(root) / "stats.json"
    path.write_text(json.dumps(stats, indent=2))
    return path


def load_stats(root: str | Path) -> dict:
    return json.loads((Path(root) / "stats.json").read_text())


class Normalizer:
    """Applies / inverts per-field normalization from a stats dict (fields without stats pass through)."""

    MODES = ("zscore", "rms")

    def __init__(self, stats: dict, mode: str):
        if mode not in self.MODES:
            raise ValueError(f"normalize mode {mode!r} not in {self.MODES}")
        if "policy" in stats:
            if stats.get("version") != 1 or stats["policy"] not in ("star_weighted", "surface_energy"):
                raise ValueError("unsupported statistics version or policy")
            if stats["policy"] == "surface_energy" and mode != "rms":
                raise ValueError("surface_energy supports rms mode only")
            stats = stats["fields"]
        self.stats, self.mode = stats, mode

    def _scale_shift(self, name: str) -> tuple[float, float]:
        st = self.stats.get(name)
        if st is None:
            return 1.0, 0.0
        return (st["std"] + EPS, st["mean"]) if self.mode == "zscore" else (st["rms"] + EPS, 0.0)

    def normalize(self, name: str, data):
        scale, shift = self._scale_shift(name)
        return (data - shift) / scale

    def denormalize(self, name: str, data):
        scale, shift = self._scale_shift(name)
        return data * scale + shift
