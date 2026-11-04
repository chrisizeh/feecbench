"""Fixed-query fit and source-removal diagnostics for acoustic coupling.

Source-removal tests use pure pressure or pure exact-velocity initial conditions.
Removing their sole initial field gives the exact zero solution. Joint overfit
includes those zero controls, preventing a geometry-only memorizer from passing.
Passing this diagnostic demonstrates finite-sample fitting, not generalization.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import atomic_json, load
from .metrics import norm2
from .model import SurfaceTransolver, loss, tensor_sample
from .training import _save_checkpoint, device_for


def query_indices(sample, count=3):
    """Deterministic interior regular times; exclude initial and off-grid outputs."""
    if count < 1:
        raise ValueError("query count must be positive")
    regular = np.flatnonzero((sample.times > 0) & ~sample.offgrid)
    if len(regular) < count:
        raise ValueError("not enough positive regular snapshots")
    locations = np.rint(np.linspace(0, len(regular) - 1, count + 2)[1:-1]).astype(int)
    if len(np.unique(locations)) != count:
        locations = np.rint(np.linspace(0, len(regular) - 1, count)).astype(int)
    return regular[locations]


def coupling_direction(sample):
    """Return the isolated coupling direction only when the other initial field vanishes."""
    meta, op = sample.metadata, sample.ops
    p2 = norm2(sample.pressure[0], op.m0)[0]
    u2 = norm2(sample.velocity[0], op.m1)[0]
    if meta["family"] in ("pulse", "two_pulses") and u2 < 1e-24 and p2 > 1e-12:
        return "pressure_to_velocity"
    if meta["family"] == "exact_velocity" and p2 < 1e-24 and u2 > 1e-12:
        return "velocity_to_pressure"
    return None


def _predict(model, batch, indices, scales, zero=False):
    source = dict(batch)
    if zero:
        source["p0"] = torch.zeros_like(batch["p0"])
        source["u0"] = torch.zeros_like(batch["u0"])
    with torch.no_grad():
        predictions = [model(source, batch["times"][q]) for q in indices]
    return tuple(
        np.asarray([pair[i].detach().cpu().numpy() * scales[field] for pair in predictions])
        for i, field in enumerate(("pressure", "velocity"))
    )


def inspect_sample(model, sample, batch, indices, scales, floor=1e-8):
    """Errors use identical fixed queries for training and validation configurations."""
    prediction = _predict(model, batch, indices, scales)
    targets = (sample.pressure[indices], sample.velocity[indices])
    op = sample.ops
    row = {
        "config_id": sample.metadata["config_id"],
        "split": sample.metadata["split"],
        "family": sample.metadata["family"],
        "genus": op.genus,
        "times": sample.times[indices].tolist(),
    }
    for field, pred, target, mass in zip(("pressure", "velocity"), prediction, targets, (op.m0, op.m1), strict=True):
        target_norm = max(float(np.sqrt(norm2(target, mass).mean())), floor * np.sqrt(op.area))
        row[field] = {
            "normalized_rmse": float(np.sqrt(norm2(pred - target, mass).mean()) / target_norm),
            "predicted_to_target_rms": float(np.sqrt(norm2(pred, mass).mean()) / target_norm),
        }
    ones = np.ones(len(op.points))
    volume = op.m0 @ ones
    pt, pp = targets[0], prediction[0]
    target_free = pt - (pt @ volume / op.area)[:, None]
    pred_free = pp - (pp @ volume / op.area)[:, None]
    row["pressure_mean_free_nrmse"] = float(
        np.sqrt(norm2(pred_free - target_free, op.m0).mean())
        / max(np.sqrt(norm2(target_free, op.m0).mean()), floor * np.sqrt(op.area))
    )
    direction = coupling_direction(sample)
    if direction is not None:
        zero_prediction = _predict(model, batch, indices, scales, zero=True)
        output = 1 if direction == "pressure_to_velocity" else 0
        mass = op.m1 if output else op.m0
        target = targets[output]
        response = prediction[output] - zero_prediction[output]
        target_norm = max(float(np.sqrt(norm2(target, mass).mean())), floor * np.sqrt(op.area))
        response_norm = float(np.sqrt(norm2(response, mass).mean()))
        inner = float(np.einsum("ti,it->", response, mass @ target.T) / len(indices))
        row["coupling"] = {
            "direction": direction,
            "response_nrmse": float(np.sqrt(norm2(response - target, mass).mean()) / target_norm),
            "response_rms_ratio": response_norm / target_norm,
            "response_alignment": inner / max(response_norm * target_norm, floor**2 * op.area),
            "zero_input_output_nrmse": float(np.sqrt(norm2(zero_prediction[output], mass).mean()) / target_norm),
        }
    return row


def _context(root, run, device):
    root, run = Path(root), Path(run)
    index = json.loads((root / "index.json").read_text())
    if index.get("complete") is not True:
        raise ValueError("diagnostics require a complete corpus")
    checkpoint_bytes = (run / "best.pt").read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    checkpoint["diagnostic_source_sha256"] = hashlib.sha256(checkpoint_bytes).hexdigest()
    settings = checkpoint["settings"]
    if settings["fingerprint"] != index["fingerprint"]:
        raise ValueError("checkpoint/corpus fingerprint mismatch")
    architecture = {k: settings[k] for k in ("width", "blocks", "slices")}
    model = SurfaceTransolver(**architecture).to(device)
    model.load_state_dict(checkpoint["model"])
    return index, checkpoint, model


def inspect_checkpoint(root, run, output, device="cpu", queries=3):
    device = device_for(device)
    root = Path(root)
    index, checkpoint, model = _context(root, run, device)
    model.eval()
    scales = checkpoint["settings"]["scales"]
    rows = []
    for entry in index["samples"]:
        if entry["split"] not in ("train", "validation") or entry["resolution"] != "medium":
            continue
        sample = load(root / entry["path"])
        batch = tensor_sample(sample, scales, device)
        rows.append(inspect_sample(model, sample, batch, query_indices(sample, queries), scales))
    summary = []
    for split in ("train", "validation"):
        for family in sorted({r["family"] for r in rows if r["split"] == split}):
            group = [r for r in rows if r["split"] == split and r["family"] == family]
            summary.append(
                {
                    "split": split,
                    "family": family,
                    "configurations": len(group),
                    **{
                        field: float(np.mean([r[field]["normalized_rmse"] for r in group]))
                        for field in ("pressure", "velocity")
                    },
                    "pressure_mean_free": float(np.mean([r["pressure_mean_free_nrmse"] for r in group])),
                }
            )
    report = {
        "fingerprint": index["fingerprint"],
        "checkpoint_epoch": checkpoint["epoch"],
        "source_checkpoint_sha256": checkpoint["diagnostic_source_sha256"],
        "queries_per_configuration": queries,
        "scales": scales,
        "summary": summary,
        "cases": rows,
        "interpretation": "fixed-query fitting errors; training and validation use the same scoring rule",
    }
    atomic_json(output, report)
    for row in summary:
        print(
            f"{row['split']:10s} {row['family']:16s} p={row['pressure']:.4f} "
            f"u={row['velocity']:.4f} p_mean_free={row['pressure_mean_free']:.4f}",
            flush=True,
        )
    return report


def select_overfit_samples(root, entries):
    """Two training configurations, one for each pure coupling direction."""
    chosen = {}
    for entry in sorted(entries, key=lambda e: e["config_id"]):
        if (
            entry["split"] != "train"
            or entry["resolution"] != "medium"
            or entry["family"] not in ("pulse", "exact_velocity")
        ):
            continue
        sample = load(Path(root) / entry["path"])
        if sample.metadata["split"] != "train" or sample.ops.genus not in (1, 2):
            raise ValueError("overfit diagnostics must use genus1--2 training data")
        direction = coupling_direction(sample)
        if direction is not None and direction not in chosen:
            chosen[direction] = sample
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise ValueError("need training pulse/u0=0 and exact_velocity/p0=0 configurations")
    return [chosen[key] for key in sorted(chosen)]


def overfit_coupling(root, run, output, device="cpu", steps=500, queries=3, seed=0, report_every=50):
    """Fresh model, unchanged architecture/loss/scales, joint pure-source and zero controls.

    Each optimizer update averages both configurations and their zero-input cases.
    Query times rotate deterministically. Reports evaluate every selected time;
    the separate best checkpoint uses this fixed training-set mass score only.
    """
    if steps < 1 or report_every < 1:
        raise ValueError("steps and report_every must be positive")
    device = device_for(device)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("diagnostic output must be empty; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    index, checkpoint, source_model = _context(root, run, device)
    settings = checkpoint["settings"]
    source_sha256 = checkpoint["diagnostic_source_sha256"]
    del source_model, checkpoint
    scales = settings["scales"]
    architecture = {k: settings[k] for k in ("width", "blocks", "slices")}
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = SurfaceTransolver(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"])
    samples = select_overfit_samples(root, index["samples"])
    batches = [tensor_sample(sample, scales, device) for sample in samples]
    indices = [query_indices(sample, queries) for sample in samples]
    zero_batches = [
        {**batch, "p0": torch.zeros_like(batch["p0"]), "u0": torch.zeros_like(batch["u0"])} for batch in batches
    ]
    history, best, best_state = [], float("inf"), None
    start = time.perf_counter()

    def measure(step):
        nonlocal best, best_state
        model.eval()
        terms = []
        with torch.no_grad():
            for batch, zero, qs in zip(batches, zero_batches, indices, strict=True):
                for q in qs:
                    terms.append(
                        float(
                            loss(model(batch, batch["times"][q]), (batch["pressure"][q], batch["velocity"][q]), batch)
                        )
                    )
                    terms.append(float(loss(model(zero, batch["times"][q]), (zero["p0"], zero["u0"]), zero)))
        cases = [inspect_sample(model, s, b, qs, scales) for s, b, qs in zip(samples, batches, indices, strict=True)]
        score = float(np.mean(terms))
        if not np.isfinite(score):
            raise RuntimeError("nonfinite diagnostic score")
        history.append({"step": step, "mass_score": score, "cases": cases, "seconds": time.perf_counter() - start})
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_checkpoint(
                output / "best.pt",
                {
                    "model": best_state,
                    "settings": {**settings, "objective": "mass"},
                    "step": step,
                    "diagnostic_only": True,
                    "mass_score": score,
                },
            )
        print(
            f"overfit step={step} mass={score:.6g} "
            + " ".join(
                f"{r['coupling']['direction']} response={r['coupling']['response_nrmse']:.4f} "
                f"gain={r['coupling']['response_rms_ratio']:.4f} "
                f"zero={r['coupling']['zero_input_output_nrmse']:.4f}"
                for r in cases
            ),
            flush=True,
        )
        atomic_json(output / "history.json", history)

    measure(0)
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        for batch, zero, qs in zip(batches, zero_batches, indices, strict=True):
            q = qs[(step - 1) % len(qs)]
            for inputs, target in (
                (batch, (batch["pressure"][q], batch["velocity"][q])),
                (zero, (zero["p0"], zero["u0"])),
            ):
                value = loss(model(inputs, batch["times"][q]), target, inputs) / (2 * len(samples))
                if not torch.isfinite(value):
                    raise RuntimeError("nonfinite overfit loss")
                value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == steps:
            measure(step)
    model.load_state_dict(best_state)
    model.eval()
    cases = [inspect_sample(model, s, b, qs, scales) for s, b, qs in zip(samples, batches, indices, strict=True)]
    report = {
        "diagnostic_only": True,
        "initialization": "fresh",
        "seed": seed,
        "steps": steps,
        "architecture": architecture,
        "learning_rate": settings["lr"],
        "objective": "mass",
        "source_objective": settings["objective"],
        "scales": scales,
        "fingerprint": index["fingerprint"],
        "source_checkpoint_sha256": source_sha256,
        "query_indices": [q.tolist() for q in indices],
        "best_mass_score": best,
        "cases": cases,
        "history": history,
        "seconds": time.perf_counter() - start,
        "interpretation": "joint finite-sample fitting with exact zero controls; not generalization evidence",
    }
    # A diagnostic criterion, not a benchmark protocol acceptance threshold.
    report["coupling_fit_below_10_percent"] = all(
        r["coupling"]["response_nrmse"] < 0.1
        and r["coupling"]["zero_input_output_nrmse"] < 0.1
        and r["velocity" if r["coupling"]["direction"] == "pressure_to_velocity" else "pressure"]["normalized_rmse"]
        < 0.1
        for r in cases
    )
    atomic_json(output / "report.json", report)
    return report
