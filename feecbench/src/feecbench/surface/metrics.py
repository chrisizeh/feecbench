"""Raw-unit trajectory, spectral, Hodge, and PDE residual diagnostics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .operators import Derived
from .oracle import energy


def norm2(x, mass):
    x = np.atleast_2d(x)
    return np.maximum(np.einsum("ti,it->t", x, mass @ x.T), 0)


def score(sample, pressure, velocity, selected=None, thresholds=(10.0, 30.0), floor=1e-8):
    op = sample.ops
    selected = (sample.times > 0) if selected is None else np.asarray(selected)
    if not np.any(selected):
        raise ValueError("no evaluation queries")
    pt, ut = sample.pressure[selected], sample.velocity[selected]
    pp, up = np.asarray(pressure)[selected], np.asarray(velocity)[selected]
    if pp.shape != pt.shape or up.shape != ut.shape or not np.isfinite(pp).all() or not np.isfinite(up).all():
        raise ValueError("nonfinite or incompatible predictions")
    result = {"times": sample.times[selected].tolist()}
    for label, pred, target, mass in (("pressure", pp, pt, op.m0), ("velocity", up, ut, op.m1)):
        error = pred - target
        curve = np.sqrt(norm2(error, mass) / op.area)
        scale = max(np.sqrt(np.mean(norm2(target, mass)) / op.area), floor)
        result[label] = {
            "mass_rmse": float(np.sqrt(np.mean(curve**2))),
            "normalized_rmse": float(np.sqrt(np.mean(curve**2)) / scale),
            "final_normalized_rmse": float(curve[-1] / scale),
            "curve": (curve / scale).tolist(),
            "mse_cochain": float(np.mean(error**2)),
            "max_cochain": float(np.max(np.abs(error))),
        }
    ep = pp - pt
    coefficients = ep @ (op.m0 @ sample.vectors)
    target_coeff = pt @ (op.m0 @ sample.vectors)
    result["pressure_constant_error"] = float(np.sqrt(np.mean(coefficients[:, 0] ** 2) / op.area))
    represented = sample.values[-1]
    bands = {}
    lo = 1e-8
    for name, hi in zip(("low", "middle"), thresholds, strict=True):
        mask = (sample.values > lo) & (sample.values <= hi)
        bands[name] = {
            "lower": lo,
            "upper": hi,
            "covered": bool(represented >= hi),
            "error_energy": float(np.mean(np.sum(coefficients[:, mask] ** 2, axis=1))),
            "target_energy": float(np.mean(np.sum(target_coeff[:, mask] ** 2, axis=1))),
        }
        lo = hi
    # Only label a physical high-band complement if all modes through cutoff are represented.
    mask = sample.values <= thresholds[-1]
    if represented >= thresholds[-1]:
        bands["high"] = {
            "lower": thresholds[-1],
            "covered": True,
            "error_energy": float(
                np.mean(np.maximum(norm2(ep, op.m0) - np.sum(coefficients[:, mask] ** 2, axis=1), 0))
            ),
            "target_energy": float(
                np.mean(np.maximum(norm2(pt, op.m0) - np.sum(target_coeff[:, mask] ** 2, axis=1), 0))
            ),
        }
    result["pressure_bands"] = bands
    derived = Derived(op)
    errors = derived.hodge((up - ut).T, sample.harmonic)
    targets = derived.hodge(ut.T, sample.harmonic)
    result["velocity_components"] = {}
    for name, component in errors.items():
        absolute = np.sqrt(np.mean(norm2(component.T, op.m1)) / op.area)
        target_scale = max(np.sqrt(np.mean(norm2(targets[name].T, op.m1)) / op.area), floor)
        result["velocity_components"][name] = {"mass_rmse": float(absolute)}
        if name != "coexact":
            result["velocity_components"][name]["normalized_rmse"] = float(absolute / target_scale)
    initial_energy = float(energy(op, sample.pressure[0], sample.velocity[0])[0])
    pred_energy, target_energy = energy(op, pp, up), energy(op, pt, ut)
    result["energy"] = {
        "predicted": pred_energy.tolist(),
        "reference": target_energy.tolist(),
        "relative_error_max": float(np.max(np.abs(pred_energy - target_energy)) / max(initial_energy, floor)),
        "drift_max": float(np.max(np.abs(pred_energy - initial_energy)) / max(initial_energy, floor)),
        "reference_drift_max": float(np.max(np.abs(target_energy - initial_energy)) / max(initial_energy, floor)),
    }
    volume = op.m0 @ np.ones(len(op.points))
    result["mean_pressure_error"] = float(np.max(np.abs((pp - pt) @ volume)) / op.area)
    result["closedness_error"] = float(np.sqrt(np.mean(norm2((op.d1 @ (up - ut).T).T, op.m2)) / op.area))
    h0 = sample.harmonic.T @ (op.m1 @ sample.velocity[0])
    ht = up @ (op.m1 @ sample.harmonic)
    result["harmonic_drift"] = float(np.sqrt(np.mean(np.sum((ht - h0) ** 2, axis=1)) / op.area))
    return result


def residuals(ops, predictor, times, steps=(0.002, 0.001)):
    """Centered query derivatives, with step refinement; caller also scores reference."""
    derived, result = Derived(ops), {}
    times = np.asarray(times)
    for step in steps:
        if np.any(times <= step):
            raise ValueError("residual queries must exceed differentiation step")
        p, u = predictor(times)
        pp, up = predictor(times + step)
        pm, um = predictor(times - step)
        rp = (pp - pm) / (2 * step) - derived.delta1(u.T).T
        ru = (up - um) / (2 * step) + (ops.d0 @ p.T).T
        result[str(step)] = {
            "pressure": float(np.sqrt(np.mean(norm2(rp, ops.m0)) / ops.area)),
            "velocity": float(np.sqrt(np.mean(norm2(ru, ops.m1)) / ops.area)),
        }
    return result


def aggregate(rows):
    """Equal weight per configuration, first averaging any repeated mesh entries."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["method"], row["split"], row["resolution"], row["queries"], row["genus"], row["family"])
        groups[key][row["config_id"]].append(row["metrics"])
    output = []
    for key, configurations in groups.items():
        entry = dict(zip(("method", "split", "resolution", "queries", "genus", "family"), key, strict=True))
        entry["configurations"] = len(configurations)
        for field in ("pressure", "velocity"):
            values = [np.mean([m[field]["normalized_rmse"] for m in items]) for items in configurations.values()]
            entry[field] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.9)),
            }
        output.append(entry)
    return output
