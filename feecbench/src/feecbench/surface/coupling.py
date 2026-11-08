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
from .model import SurfaceTransolver, forward_features, loss, quadratic, tensor_sample
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


def _context(root, run, device, checkpoint_path=None):
    root, run = Path(root), Path(run)
    index = json.loads((root / "index.json").read_text())
    if index.get("complete") is not True:
        raise ValueError("diagnostics require a complete corpus")
    checkpoint_bytes = (Path(checkpoint_path) if checkpoint_path is not None else run / "best.pt").read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    checkpoint["diagnostic_source_sha256"] = hashlib.sha256(checkpoint_bytes).hexdigest()
    settings = checkpoint["settings"]
    if settings["fingerprint"] != index["fingerprint"]:
        raise ValueError("checkpoint/corpus fingerprint mismatch")
    architecture = {k: settings[k] for k in ("width", "blocks", "slices")}
    architecture["adapter"] = settings.get("adapter", "scalar")
    architecture["transport"] = settings.get("transport", "none")
    model = SurfaceTransolver(**architecture).to(device)
    model.load_state_dict(checkpoint["model"])
    return index, checkpoint, model


def inspect_checkpoint(root, run, output, device="cpu", queries=3, checkpoint_path=None):
    device = device_for(device)
    root = Path(root)
    index, checkpoint, model = _context(root, run, device, checkpoint_path=checkpoint_path)
    model.eval()
    scales = checkpoint["settings"]["scales"]
    rows = []
    for entry in index["samples"]:
        if entry["split"] not in ("train", "validation") or entry["resolution"] != "medium":
            continue
        sample = load(root / entry["path"])
        batch = tensor_sample(sample, scales, device, transport=model.transport)
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
        "architecture": model.settings,
        "parameter_count": sum(p.numel() for p in model.parameters()),
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


def fixed_field_energies(batch, indices, floor=1e-6):
    """Fixed per-field target energies in normalized units, from fitting queries only.

    Both the nonzero case and its zero-input control must reuse these denominators.
    They are never recomputed from zero targets, predictions, or validation data.
    """
    if not np.isfinite(floor) or floor <= 0 or not len(indices):
        raise ValueError("positive finite energy floor and fitting queries required")
    raw = torch.stack(
        [
            torch.stack([quadratic(batch[field][q], batch[mass]) for q in indices]).mean() / batch["area"]
            for field, mass in (("pressure", "m0"), ("velocity", "m1"))
        ]
    ).detach()
    if not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
        raise ValueError("invalid target mass energy")
    return raw, raw.clamp_min(floor)


def coupling_loss(prediction, target, batch, objective="mass", energies=None):
    """Diagnostic-only relative mass loss; original benchmark objectives are unchanged."""
    if objective == "mass":
        return loss(prediction, target, batch)
    if objective != "balanced" or energies is None:
        raise ValueError("balanced diagnostic requires fixed field-energy denominators")
    ep, eu = (pred - ref for pred, ref in zip(prediction, target, strict=True))
    return (quadratic(ep, batch["m0"]) / energies[0] + quadratic(eu, batch["m1"]) / energies[1]) / batch["area"]


def _fit_passes(cases):
    return all(
        r["coupling"]["response_nrmse"] < 0.1
        and r["coupling"]["zero_input_output_nrmse"] < 0.1
        and r["velocity" if r["coupling"]["direction"] == "pressure_to_velocity" else "pressure"]["normalized_rmse"]
        < 0.1
        for r in cases
    )


def overfit_coupling(
    root,
    run,
    output,
    device="cpu",
    steps=500,
    queries=3,
    seed=0,
    report_every=50,
    objective="mass",
    energy_floor=1e-6,
    adapter=None,
    lr_schedule="constant",
    final_lr_ratio=0.01,
    transport=None,
):
    """Fresh model, fixed architecture/scales, joint pure-source and zero controls.

    The optional balanced objective equalizes relative target-energy errors by
    case and field. It is a diagnostic, not another benchmark loss variant.
    Both final-budget and objective-selected best fits are reported separately.
    """
    if steps < 1 or report_every < 1:
        raise ValueError("steps and report_every must be positive")
    if lr_schedule not in ("constant", "cosine") or not 0 < final_lr_ratio <= 1:
        raise ValueError("schedule must be constant/cosine and final_lr_ratio in (0,1]")
    if objective not in ("mass", "balanced"):
        raise ValueError("diagnostic objective must be mass or balanced")
    if not np.isfinite(energy_floor) or energy_floor <= 0:
        raise ValueError("energy floor must be finite and positive")
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
    architecture["adapter"] = adapter if adapter is not None else settings.get("adapter", "scalar")
    architecture["transport"] = transport if transport is not None else settings.get("transport", "none")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = SurfaceTransolver(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=settings["lr"] * final_lr_ratio)
        if lr_schedule == "cosine"
        else None
    )
    samples = select_overfit_samples(root, index["samples"])
    batches = [tensor_sample(sample, scales, device, transport=model.transport) for sample in samples]
    indices = [query_indices(sample, queries) for sample in samples]
    energy_pairs = [fixed_field_energies(batch, qs, energy_floor) for batch, qs in zip(batches, indices, strict=True)]
    denominators = [pair[1] for pair in energy_pairs]
    energy_records = []
    for sample, (raw, denominator) in zip(samples, energy_pairs, strict=True):
        record = {"config_id": sample.metadata["config_id"]}
        for i, field in enumerate(("pressure", "velocity")):
            value, denom = float(raw[i]), float(denominator[i])
            record[field] = {
                "target_energy_normalized": value,
                "target_energy_physical": value * scales[field] ** 2,
                "denominator": denom,
                "floor_applied": value < energy_floor,
                "objective_weight": 1 / denom if objective == "balanced" else 1.0,
            }
        energy_records.append(record)
    zero_batches = [
        {**batch, "p0": torch.zeros_like(batch["p0"]), "u0": torch.zeros_like(batch["u0"])} for batch in batches
    ]
    history, best, best_state, best_mass, best_step = [], float("inf"), None, None, None
    gradient_sum, gradient_max, clipped_updates, updates = 0.0, 0.0, 0, 0
    start = time.perf_counter()

    def gradient_statistics():
        return {
            "updates": updates,
            "preclip_norm_mean": gradient_sum / max(updates, 1),
            "preclip_norm_max": gradient_max,
            "clip_threshold": 1.0,
            "clipped_fraction": clipped_updates / max(updates, 1),
        }

    def measure(step):
        nonlocal best, best_state, best_mass, best_step
        model.eval()
        terms, mass_terms = [], []
        with torch.no_grad():
            for batch, zero, qs, denom in zip(batches, zero_batches, indices, denominators, strict=True):
                for q in qs:
                    for inputs, target in (
                        (batch, (batch["pressure"][q], batch["velocity"][q])),
                        (zero, (zero["p0"], zero["u0"])),
                    ):
                        prediction = model(inputs, batch["times"][q])
                        terms.append(float(coupling_loss(prediction, target, inputs, objective, denom)))
                        mass_terms.append(float(loss(prediction, target, inputs)))
        cases = [inspect_sample(model, s, b, qs, scales) for s, b, qs in zip(samples, batches, indices, strict=True)]
        score, mass_score = float(np.mean(terms)), float(np.mean(mass_terms))
        if not np.isfinite(score) or not np.isfinite(mass_score):
            raise RuntimeError("nonfinite diagnostic score")
        history.append(
            {
                "step": step,
                "objective_score": score,
                "mass_score": mass_score,
                "cases": cases,
                "gradient_statistics": gradient_statistics(),
                "next_learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - start,
            }
        )
        if score < best:
            best, best_mass, best_step = score, mass_score, step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_checkpoint(
                output / "best.pt",
                {
                    "model": best_state,
                    "settings": {
                        **settings,
                        "objective": objective,
                        "adapter": architecture["adapter"],
                        "transport": architecture["transport"],
                        "features": forward_features(architecture["adapter"], architecture["transport"]),
                    },
                    "step": step,
                    "diagnostic_only": True,
                    "objective_score": score,
                    "mass_score": mass_score,
                    "energy_records": energy_records,
                    "energy_floor": energy_floor,
                    "lr_schedule": lr_schedule,
                    "final_lr_ratio": final_lr_ratio if scheduler is not None else 1.0,
                },
            )
        print(
            f"overfit step={step} objective={objective} score={score:.6g} mass={mass_score:.6g} "
            f"lr={optimizer.param_groups[0]['lr']:.3g} "
            + " ".join(
                f"{r['coupling']['direction']} response={r['coupling']['response_nrmse']:.4f} "
                f"gain={r['coupling']['response_rms_ratio']:.4f} zero={r['coupling']['zero_input_output_nrmse']:.4f}"
                for r in cases
            ),
            flush=True,
        )
        atomic_json(output / "history.json", history)

    measure(0)
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        for batch, zero, qs, denom in zip(batches, zero_batches, indices, denominators, strict=True):
            q = qs[(step - 1) % len(qs)]
            for inputs, target in (
                (batch, (batch["pressure"][q], batch["velocity"][q])),
                (zero, (zero["p0"], zero["u0"])),
            ):
                value = coupling_loss(model(inputs, batch["times"][q]), target, inputs, objective, denom) / (
                    2 * len(samples)
                )
                if not torch.isfinite(value):
                    raise RuntimeError("nonfinite overfit loss")
                value.backward()
        preclip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
        gradient_sum += preclip
        gradient_max = max(gradient_max, preclip)
        clipped_updates += int(preclip > 1.0)
        updates += 1
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step % report_every == 0 or step == steps:
            measure(step)
    final_cases = history[-1]["cases"]
    model.load_state_dict(best_state)
    model.eval()
    cases = [inspect_sample(model, s, b, qs, scales) for s, b, qs in zip(samples, batches, indices, strict=True)]
    group = optimizer.param_groups[0]
    report = {
        "diagnostic_only": True,
        "initialization": "fresh",
        "seed": seed,
        "steps": steps,
        "architecture": architecture,
        "source_adapter": settings.get("adapter", "scalar"),
        "source_transport": settings.get("transport", "none"),
        "features": forward_features(architecture["adapter"], architecture["transport"]),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "paired_initialization": "legacy shared weights; vector columns and transport output projections start at zero",
        "learning_rate": settings["lr"],
        "lr_schedule": lr_schedule,
        "final_lr_ratio": final_lr_ratio if scheduler is not None else 1.0,
        # Scheduler steps after each update; this is the rate for a next update.
        "final_learning_rate": group["lr"],
        "objective": objective,
        "source_objective": settings["objective"],
        "scales": scales,
        "energy_records": energy_records,
        "energy_floor": energy_floor,
        "gradient_statistics": gradient_statistics(),
        "optimizer": {
            "name": "AdamW",
            "betas": list(group["betas"]),
            "eps": group["eps"],
            "weight_decay": group["weight_decay"],
        },
        "fingerprint": index["fingerprint"],
        "source_checkpoint_sha256": source_sha256,
        "query_indices": [q.tolist() for q in indices],
        "best_objective_score": best,
        "best_mass_score": best_mass,
        "best_step": best_step,
        "cases": cases,
        "final_cases": final_cases,
        "history": history,
        "seconds": time.perf_counter() - start,
        "interpretation": "joint finite-sample fitting with exact zero controls; not generalization evidence",
        "weighting_caveat": "balancing changes gradient magnitudes and clipping; compare fixed-budget final cases",
        "coupling_fit_below_10_percent": _fit_passes(cases),
        "final_coupling_fit_below_10_percent": _fit_passes(final_cases),
    }
    atomic_json(output / "report.json", report)
    return report
