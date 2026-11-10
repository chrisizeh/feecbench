"""Metric-profile evaluator: a nested dict, never one scalar.

Driven by the sample's typed fields: every target-role field present in the prediction gets
the metrics of its form degree; DEC metrics (closedness, Hodge split) apply to degree-1
targets; PDE energy functionals come from a registry keyed by ``params["pde"]``.

Conventions: all weighted norms are ``sqrt(z^T star_k z)`` with the sample's stars; relative
errors are normalized by the target's star-norm; time series aggregate over the whole
series, the rollout curve is per snapshot. Closedness is reported both raw
(``norm_star2(d1 z)``, resolution-dependent) and scale-free as
``rel_opnorm = norm_star2(d1 z) / (|||d1||| norm_star1(z))`` with ``|||d1|||`` the
star1 -> star2 operator norm of d1 on this mesh (a dimensionless number in [0, 1]), for the
prediction and the target (reference scale). The Hodge split projects the final-snapshot
error onto exact / coexact / harmonic parts in the star1 inner product. Harmonic bases and
operator norms are cached per ``sample_id``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from .operators import d1_operator_norm, harmonic_1forms, hodge_components, norm_star
from .schema import FEECSample

EPS = 1e-30
ENERGY_REGISTRY: dict[str, Callable[[FEECSample, dict], np.ndarray]] = {}
_HARMONIC_CACHE: dict[str, np.ndarray] = {}
_D1_NORM_CACHE: dict[str, float] = {}


def register_energy(pde_name: str):
    """Decorator registering ``energy(sample, fields) -> (T,)`` for a PDE tag."""

    def deco(fn):
        ENERGY_REGISTRY[pde_name] = fn
        return fn

    return deco


def harmonic_basis(sample: FEECSample) -> np.ndarray:
    """star1-orthonormal harmonic 1-form basis (n1, beta1), cached per sample_id."""
    key = sample.ids.sample_id
    if key not in _HARMONIC_CACHE:
        ops = sample.operators
        _HARMONIC_CACHE[key] = harmonic_1forms(
            ops["d0"], ops["d1"], ops["star0"], ops["star1"], ops["star2"], sample.topology.betti1
        )
    return _HARMONIC_CACHE[key]


def _d1_norm(sample: FEECSample) -> float:
    key = sample.ids.sample_id
    if key not in _D1_NORM_CACHE:
        ops = sample.operators
        _D1_NORM_CACHE[key] = d1_operator_norm(ops["d1"], ops["star1"], ops["star2"])
    return _D1_NORM_CACHE[key]


def clear_caches() -> None:
    _HARMONIC_CACHE.clear()
    _D1_NORM_CACHE.clear()


def _rel(err: np.ndarray, ref: np.ndarray, star) -> float:
    return norm_star(err, star) / (norm_star(ref, star) + EPS)


def _field_profile(pred: np.ndarray, tgt: np.ndarray, sample: FEECSample, degree: int, harmonic) -> dict:
    star, err = sample.operators[f"star{degree}"], pred - tgt
    abs_err = np.abs(err).ravel()
    entry = {
        "form_degree": degree,
        "rel_l2_star": _rel(err, tgt, star),
        "mse": float(np.mean(abs_err**2)),
        "mae": float(np.mean(abs_err)),
        "max": float(abs_err.max(initial=0.0)),
        "p99": float(np.percentile(abs_err, 99)) if abs_err.size else 0.0,
        "rollout_rel_l2": [_rel(e, t, star) for e, t in zip(np.atleast_2d(err), np.atleast_2d(tgt), strict=True)],
    }
    if degree in (0, 1):  # boundary- vs interior-restricted errors
        mask = sample.mesh.boundary_vertices if degree == 0 else sample.mesh.boundary_edges
        for name, m in (("boundary", mask), ("interior", ~mask)):
            w = sp.diags(m.astype(float))
            entry[name] = _rel(err, tgt, w @ star @ w)
    if degree == 1:
        d1 = sample.operators["d1"].astype(np.float64)
        star1, star2, opnorm = sample.operators["star1"], sample.operators["star2"], _d1_norm(sample)
        entry["closedness"] = {}
        for name, series in (("pred", pred), ("target", tgt)):
            z = np.atleast_2d(series)[-1]
            a = norm_star(d1 @ z, star2)
            entry["closedness"][name] = {"abs": a, "rel_opnorm": a / (opnorm * norm_star(z, star1) + EPS)}
        if harmonic is not None:
            comps = hodge_components(np.atleast_2d(err)[-1], sample.operators["d0"], star1, harmonic)
            denom = norm_star(np.atleast_2d(tgt)[-1], star1) + EPS
            entry["hodge_error"] = {k: norm_star(v, star1) / denom for k, v in comps.items()}
    return entry


def evaluate_profile(
    pred: dict[str, np.ndarray],
    sample: FEECSample,
    target: dict[str, np.ndarray] | None = None,
    include_hodge: bool = True,
    harmonic: np.ndarray | None = None,
    *,
    profile: str | None = None,
    selected: np.ndarray | None = None,
    thresholds: tuple[float, float] = (10.0, 30.0),
    floor: float = 1e-8,
) -> dict:
    """Metric profile of ``pred`` (keyed like the sample's target fields) against ``target``
    (default: the sample's role == "target" fields). ``include_hodge=False`` skips the
    harmonic-basis solve; ``harmonic`` supplies a precomputed basis."""
    if profile is None:
        profile = "surface_acoustic" if sample.params.get("pde") == "surface_acoustic" else "planar"
    if profile == "surface_acoustic":
        from .surface.compat import from_feec
        from .surface.metrics import score

        if target is not None:
            from dataclasses import replace

            sample = replace(sample, fields={**sample.fields, **{
                n: replace(sample.fields[n], data=v) for n, v in target.items()
            }})
        return score(from_feec(sample), pred["p"], pred["u"], selected=selected, thresholds=thresholds, floor=floor)
    if profile != "planar":
        raise ValueError(f"unknown evaluation profile {profile!r}")
    if sample.params.get("pde") == "surface_acoustic":
        from .surface.compat import acoustic_energy

        register_energy("surface_acoustic")(acoustic_energy)
    if target is None:
        target = {n: f.data for n, f in sample.fields.items() if f.role == "target"}
    profile: dict = {"fields": {}}
    for name, tgt in target.items():
        if name in pred:
            degree = sample.fields[name].form_degree
            basis = None
            if degree == 1 and include_hodge:
                basis = harmonic if harmonic is not None else harmonic_basis(sample)
            profile["fields"][name] = _field_profile(np.asarray(pred[name]), np.asarray(tgt), sample, degree, basis)
    energy = ENERGY_REGISTRY.get(sample.params.get("pde"))
    if energy is not None and all(n in pred for n in target):
        e_pred, e_tgt = energy(sample, pred), energy(sample, target)
        e0 = abs(e_tgt[0]) + EPS
        profile["energy"] = {
            "target_drift_rel": float(np.max(np.abs(e_tgt - e_tgt[0])) / e0),
            "pred_vs_target_rel": float(np.max(np.abs(e_pred - e_tgt)) / e0),
            "curve_pred": e_pred.tolist(),
            "curve_target": e_tgt.tolist(),
        }
    return profile
