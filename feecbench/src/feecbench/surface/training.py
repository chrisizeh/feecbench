"""Single-mesh training, deterministic paired seeds, resumable common-score selection."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import atomic_json, digest, load, training_scales
from .metrics import aggregate, residuals, score
from .model import SurfaceTransolver, forward_features, loss, tensor_sample
from .oracle import Midpoint, ModalControl


def device_for(request):
    if request not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    return torch.device(request)


def _save_checkpoint(path, checkpoint):
    temp = path.with_suffix(".pending")
    torch.save(checkpoint, temp)
    os.replace(temp, path)


def train(
    root,
    output,
    device="cpu",
    seed=0,
    objective="mass",
    epochs=20,
    width=128,
    blocks=4,
    slices=32,
    lr=0.001,
    accumulation=4,
    resume=False,
    bands=None,
    spectral_weight=1.0,
    adapter="scalar",
):
    device = device_for(device)
    if epochs < 1 or accumulation < 1:
        raise ValueError("epochs and accumulation must be positive")
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((root / "index.json").read_text())
    if index.get("complete") is not True:
        raise ValueError("corpus generation is incomplete; resume generation before training")
    entries = index["samples"]
    train_paths = [root / e["path"] for e in entries if e["split"] == "train" and e["resolution"] == "medium"]
    validation_paths = [root / e["path"] for e in entries if e["split"] == "validation" and e["resolution"] == "medium"]
    if not train_paths or not validation_paths:
        raise ValueError("training and independent validation configurations required")
    scales = training_scales(train_paths)
    settings = {
        "seed": seed,
        "objective": objective,
        "width": width,
        "blocks": blocks,
        "slices": slices,
        "lr": lr,
        "accumulation": accumulation,
        "fingerprint": index["fingerprint"],
        "scales": scales,
        "bands": bands,
        "spectral_weight": spectral_weight,
        "selection_score": "mean configuration normalized-field mass squared error / area, p+u, regular t>0",
        "features": forward_features(adapter),
    }
    # Preserve the exact legacy scalar settings/hash so old runs still resume.
    if adapter != "scalar":
        settings["adapter"] = adapter
    identity = digest(settings)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = SurfaceTransolver(width, blocks, slices, adapter=adapter).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    start_epoch, best, history = 0, float("inf"), []
    latest = output / "latest.pt"
    if resume:
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint["identity"] != identity:
            raise ValueError("resume configuration/data hash mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best, history = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["history"]
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
    elif latest.exists():
        raise ValueError("run exists; use --resume or a new output directory")
    atomic_json(
        output / "configuration.json",
        {
            **settings,
            "epochs_requested": epochs,
            "device": str(device),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    )
    atomic_json(output / "normalization.json", scales)
    for epoch in range(start_epoch, epochs):
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        running = 0.0
        order = rng.permutation(len(train_paths))
        for i, index_ in enumerate(order):
            sample = load(train_paths[index_])
            batch = tensor_sample(sample, scales, device)
            choices = np.flatnonzero((sample.times > 0) & ~sample.offgrid)
            query = int(rng.choice(choices))
            prediction = model(batch, batch["times"][query])
            value = loss(
                prediction,
                (batch["pressure"][query], batch["velocity"][query]),
                batch,
                objective,
                bands,
                spectral_weight,
            )
            if not torch.isfinite(value):
                raise RuntimeError("nonfinite training loss")
            # Last partial accumulation group has its own correct denominator.
            group_size = min(accumulation, len(order) - (i // accumulation) * accumulation)
            (value / group_size).backward()
            running += float(value.detach())
            if (i + 1) % accumulation == 0 or i + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        model.eval()
        validation = []
        with torch.no_grad():
            for path in validation_paths:
                sample = load(path)
                batch = tensor_sample(sample, scales, device)
                scores = [
                    float(
                        loss(
                            model(batch, batch["times"][q]), (batch["pressure"][q], batch["velocity"][q]), batch, "mass"
                        )
                    )
                    for q in np.flatnonzero((sample.times > 0) & ~sample.offgrid)
                ]
                validation.append(float(np.mean(scores)))
        val = float(np.mean(validation))
        if not np.isfinite(val):
            raise RuntimeError("nonfinite validation score")
        improved = val < best
        best = min(val, best)
        history.append(
            {
                "epoch": epoch,
                "training_loss": running / len(order),
                "validation_mass_score": val,
                "seconds": time.perf_counter() - start,
            }
        )
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best": best,
            "settings": settings,
            "identity": identity,
            "history": history,
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        _save_checkpoint(latest, checkpoint)
        if improved:
            _save_checkpoint(output / "best.pt", checkpoint)
        atomic_json(output / "history.json", history)
    return output / "best.pt"


def evaluate(root, checkpoint_path, output, device="cpu", residual_subset=1):
    device = device_for(device)
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    settings = checkpoint["settings"]
    index = json.loads((root / "index.json").read_text())
    if index["fingerprint"] != settings["fingerprint"]:
        raise ValueError("evaluation corpus differs from checkpoint manifest")
    model = SurfaceTransolver(
        **{k: settings[k] for k in ("width", "blocks", "slices")}, adapter=settings.get("adapter", "scalar")
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scales, rows, residual_reports = settings["scales"], [], []
    for entry in index["samples"]:
        if entry["split"] in ("train", "verification"):
            continue
        sample = load(root / entry["path"])
        batch = tensor_sample(sample, scales, device)

        def neural(times, batch=batch):
            with torch.no_grad():
                outputs = [model(batch, float(t)) for t in times]
            return (
                np.array([p.cpu().numpy() * scales["pressure"] for p, _ in outputs]),
                np.array([u.cpu().numpy() * scales["velocity"] for _, u in outputs]),
            )

        modal = ModalControl(sample.ops, sample.values, sample.vectors, sample.harmonic)
        start = time.perf_counter()
        predicted = neural(sample.times)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start
        start = time.perf_counter()
        modal_pred = modal.solve(sample.pressure[0], sample.velocity[0], sample.times)
        modal_seconds = time.perf_counter() - start
        predictions = {
            "neural": predicted,
            "modal": modal_pred,
            "persistence": (
                np.broadcast_to(sample.pressure[0], sample.pressure.shape),
                np.broadcast_to(sample.velocity[0], sample.velocity.shape),
            ),
            "zero": (np.zeros_like(sample.pressure), np.zeros_like(sample.velocity)),
        }
        destination = output / "predictions" / entry["config_id"]
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / f"{entry['resolution']}.npz", times=sample.times, pressure=predicted[0], velocity=predicted[1]
        )
        for method, (p, u) in predictions.items():
            for queries, selected in (("regular", (sample.times > 0) & ~sample.offgrid), ("offgrid", sample.offgrid)):
                if not np.any(selected):
                    continue
                rows.append(
                    {
                        "method": method,
                        "queries": queries,
                        "split": entry["split"],
                        "config_id": entry["config_id"],
                        "genus": sample.ops.genus,
                        "family": entry["family"],
                        "resolution": entry["resolution"],
                        "inference_seconds": inference_seconds
                        if method == "neural"
                        else modal_seconds
                        if method == "modal"
                        else None,
                        "modal_omitted_energy": modal.omitted_energy(sample.pressure[0], sample.velocity[0]),
                        "metrics": score(
                            sample,
                            p,
                            u,
                            selected,
                            thresholds=tuple(b["upper"] for b in settings["bands"])
                            if settings["bands"]
                            else (10.0, 30.0),
                        ),
                    }
                )
        if len(residual_reports) < residual_subset:
            ts = sample.times[-1] * np.array([0.2, 0.5, 0.8])
            solver = Midpoint(sample.ops)

            def reference(times, solver=solver, sample=sample):
                return solver.solve(sample.pressure[0], sample.velocity[0], times, sample.metadata["dt"])

            residual_reports.append(
                {
                    "config_id": entry["config_id"],
                    "neural": residuals(sample.ops, neural, ts),
                    "reference": residuals(sample.ops, reference, ts),
                }
            )
    atomic_json(output / "per_sample.json", rows)
    atomic_json(output / "summary.json", aggregate(rows))
    atomic_json(output / "residuals.json", residual_reports)
    return rows
