"""Matched architecture/loss/budget study on an immutable existing surface corpus."""

from __future__ import annotations

import json
import math
import platform
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import torch

from .campaign import _stage, file_hash, source_identity
from .coupling import inspect_checkpoint
from .data import atomic_json, digest, load, training_mse_balance, training_scales, validate_manifest
from .loss_balance import _checked_path, _read
from .model import SurfaceTransolver, loss, tensor_sample
from .report import _table_metrics
from .training import device_for, evaluate, train

TRANSPORTS = ("none", "d0_delta1")
OBJECTIVES = ("mse_balanced", "mass")
SNAPSHOT_FILES = ("best.pt", "latest.pt", "history.json", "configuration.json", "resources.json")


def _sources():
    package = Path(__file__).resolve().parents[3]
    result = source_identity()
    for name in ("scripts/transport_study.py", "scripts/run_transport_study.sh"):
        result[name] = file_hash(package / name)
    return result


def _corpus_snapshot(root):
    index, manifest = _read(root / "index.json"), _read(root / "manifest.json")
    base = {k: v for k, v in manifest.items() if k not in ("settings", "fingerprint")}
    validate_manifest(base)
    fingerprint = digest({"manifest": base, "settings": manifest["settings"]})
    if (index.get("complete") is not True or index["fingerprint"] != fingerprint
            or manifest["fingerprint"] != fingerprint):
        raise ValueError("source corpus is incomplete or its manifest fingerprint differs")
    configs = {c["config_id"]: c for c in base["configurations"]}
    expected = {
        (cid, resolution)
        for cid, c in configs.items()
        for resolution in (("coarse", "medium", "fine", "remeshed")
                           if manifest["settings"]["paired"] and c["split"] == "id_test" else ("medium",))
    }
    entries = index["samples"]
    if len(entries) != len(expected) or {(e["config_id"], e["resolution"]) for e in entries} != expected:
        raise ValueError("source index has missing or duplicate mesh samples")
    counts = Counter(c["split"] for c in configs.values())
    if not counts["train"] or not counts["validation"]:
        raise ValueError("independent training and validation configurations required")
    files = {name: file_hash(root / name) for name in ("index.json", "manifest.json")}
    seen_paths = set()
    for entry in entries:
        config = configs[entry["config_id"]]
        path = _checked_path(root, entry["path"])
        if path in seen_paths:
            raise ValueError("duplicate source sample path")
        seen_paths.add(path)
        with h5py.File(path, "r") as stream:
            metadata = json.loads(stream["metadata"][()].decode())
            if int(stream.attrs["genus"]) != config["geometry"]["genus"]:
                raise ValueError("sample genus differs from manifest")
            if config["split"] in ("train", "validation"):
                if np.count_nonzero((stream["times"][()] > 0) & ~stream["offgrid"][()]) < 3:
                    raise ValueError("three positive regular queries per train/validation case required")
        for key in ("config_id", "geometry_id", "split", "family"):
            if entry[key] != config[key] or metadata[key] != config[key]:
                raise ValueError(f"sample/index {key} differs from manifest")
        if metadata["resolution"] != entry["resolution"] or metadata["fingerprint"] != fingerprint:
            raise ValueError("sample resolution or fingerprint mismatch")
        files[entry["path"]] = file_hash(path)
    train_paths = [root / e["path"] for e in entries if e["split"] == "train" and e["resolution"] == "medium"]
    scales = training_scales(train_paths)
    return {
        "fingerprint": fingerprint, "files": files, "counts": dict(counts), "mesh_samples": len(entries),
        "numerical": manifest["settings"], "scales": scales,
        "field_balance": training_mse_balance(train_paths, scales),
    }


def prepare_transport_study(root, output, *, epochs=400, first_budget=200, smoke=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("output must be separate from the read-only source corpus")
    if not 1 <= first_budget < epochs:
        raise ValueError("budgets must satisfy 1 <= first_budget < epochs")
    snapshot = _corpus_snapshot(root)
    architecture = {"width": 16 if smoke else 128, "blocks": 1 if smoke else 4, "slices": 4 if smoke else 32}
    # Parameter accounting must not perturb RNG state used by a caller.
    with torch.random.fork_rng(devices=[]):
        parameters = {t: sum(p.numel() for p in SurfaceTransolver(**architecture, transport=t).parameters())
                      for t in TRANSPORTS}
    specification = {
        "version": 1, "status": "smoke only" if smoke else "exploratory architecture/loss/budget study",
        "root": str(root), "corpus": snapshot, "budgets": [first_budget, epochs], "seeds": [0, 1, 2],
        "transports": list(TRANSPORTS), "objectives": list(OBJECTIVES),
        "training": {**architecture, "lr": .001, "accumulation": 4, "adapter": "scalar"},
        "schedule": "constant", "initialization": "fresh; paired shared weights; zero transport projections",
        "updates_per_epoch": math.ceil(snapshot["counts"]["train"] / 4), "parameters": parameters,
        "fit_queries": 3, "residual_subset": 1 if smoke else 4,
        "validation_splits": ["validation"],
        "heldout_splits": [s for s in ("id_test", "geometry_ood", "topology_ood") if snapshot["counts"].get(s)],
        "sources": _sources(),
        "interpretation": {
            "primary": "paired transport-minus-attention at fixed loss, seed, budget and checkpoint role",
            "secondary": "paired later-minus-earlier budget; balanced MSE versus mass within architecture",
            "selection": "same validation mass score; best and final reported separately",
            "capacity": "matched updates and shared initialization, not matched parameters or wall time",
            "heldout": "explicit stage after full validation suite; both budgets/roles evaluated without selection",
            "reference": "inherits existing exploratory corpus; does not attest a numerical protocol",
            "residuals": "first eligible mesh samples only, not exhaustive or stratified coverage",
            "timing": "epoch time includes validation; inference time covers saved trajectory, excludes preprocessing",
        },
    }
    specification["hash"] = digest(specification)
    path = output / "study.json"
    if path.exists():
        if _read(path) != specification:
            raise ValueError("prepared settings/source/data changed; use a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("new output directory must be empty")
    else:
        atomic_json(path, specification)
    return specification


def read_transport_study(output, verify_data=True):
    specification = _read(Path(output) / "study.json")
    if specification["hash"] != digest({k: v for k, v in specification.items() if k != "hash"}):
        raise ValueError("study configuration hash mismatch")
    if specification["sources"] != _sources():
        raise ValueError("study source/lockfile changed; retain this checkout or prepare a new study")
    if verify_data:
        root = Path(specification["root"])
        for name, expected in specification["corpus"]["files"].items():
            if file_hash(_checked_path(root, name)) != expected:
                raise ValueError(f"source corpus changed: {name}")
    return specification


def _run_name(transport, objective, seed):
    return f"{transport}-{objective}-seed{seed}"


def _combinations(spec):
    for seed in spec["seeds"]:
        for objective in spec["objectives"]:
            for transport in spec["transports"]:
                yield transport, objective, seed


def _verify_record(marker):
    record = _read(marker)
    for name, expected in record["artifacts"].items():
        if file_hash(_checked_path(marker.parent, name)) != expected:
            raise ValueError(f"completed study artifact changed: {marker.parent / name}")
    return record


def _preflight(spec, output, device):
    """Exercise active transport and its adjoint before a long CUDA run."""
    root = Path(spec["root"])
    entry = next(e for e in _read(root / "index.json")["samples"] if e["split"] == "train")
    sample = load(root / entry["path"])
    q = int(np.flatnonzero((sample.times > 0) & ~sample.offgrid)[0])
    torch.manual_seed(901)
    cpu = SurfaceTransolver(**{k: spec["training"][k] for k in ("width", "blocks", "slices")}, transport="d0_delta1")
    with torch.no_grad():
        for block in cpu.transports:
            block.to_vertex.weight.normal_(std=.01)
            block.to_edge.weight.normal_(std=.01)
    batch = tensor_sample(sample, spec["corpus"]["scales"], transport="d0_delta1")
    pred = cpu(batch, batch["times"][q])
    loss(pred, (batch["pressure"][q], batch["velocity"][q]), batch).backward()
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in cpu.parameters()):
        raise RuntimeError("preflight produced missing/nonfinite CPU gradients")
    if device == "cuda":
        gpu = SurfaceTransolver(**cpu.settings).cuda()
        gpu.load_state_dict(cpu.state_dict())
        gb = tensor_sample(sample, spec["corpus"]["scales"], device, transport="d0_delta1")
        gp = gpu(gb, gb["times"][q])
        loss(gp, (gb["pressure"][q], gb["velocity"][q]), gb).backward()
        for a, b in zip(pred, gp, strict=True):
            torch.testing.assert_close(a, b.cpu(), rtol=5e-4, atol=2e-5)
        for a, b in zip(cpu.parameters(), gpu.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad.cpu(), rtol=1e-3, atol=1e-4)
    atomic_json(output / f"preflight-{device}.json", {
        "passed": True, "device": device, "config_id": entry["config_id"],
        "cpu_cuda_agreement": device == "cuda", "scope": "one training mesh, active forward/backward",
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
    })
    print(f"Preflight passed on {device}; one training mesh, active transport and gradients", flush=True)


def _train_budget(spec, run, snapshot, budget, transport, objective, seed, device):
    if (run / "latest.pt").exists():
        latest = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
        if latest["epoch"] + 1 > budget:
            raise ValueError("an earlier milestone was damaged after training advanced; restore its original artifacts")
        del latest
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    train(spec["root"], run, device=device, seed=seed, objective=objective, transport=transport,
          epochs=budget, resume=(run / "latest.pt").exists(), progress=True,
          field_balance=spec["corpus"]["field_balance"] if objective == "mse_balanced" else None,
          **spec["training"])
    if device == "cuda":
        torch.cuda.synchronize()
    history = _read(run / "history.json")
    if [r["epoch"] for r in history] != list(range(budget)):
        raise ValueError("training did not complete the fixed milestone budget")
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in SNAPSHOT_FILES[:-1]:
        pending = snapshot / f"{name}.pending"
        shutil.copyfile(run / name, pending)
        pending.replace(snapshot / name)
    atomic_json(snapshot / "resources.json", {
        "parameters": spec["parameters"][transport], "epochs": budget,
        "updates": budget * spec["updates_per_epoch"],
        "training_epoch_seconds": sum(r["seconds"] for r in history),
        "last_invocation_seconds": time.perf_counter() - start,
        "peak_cuda_allocated_bytes_last_invocation": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
    })


def _evaluate_role(spec, snapshot, role, phase, transport, objective, seed, budget, device):
    checkpoint = snapshot / ("best.pt" if role == "best" else "latest.pt")
    destination = snapshot / role / phase
    splits = spec["validation_splits" if phase == "validation" else "heldout_splits"]
    evaluate(spec["root"], checkpoint, destination / "evaluation", device=device, resume=True,
             residual_subset=spec["residual_subset"], splits=splits)
    if phase == "validation":
        fit = destination / "fit.json"
        if not fit.exists() or _read(fit)["source_checkpoint_sha256"] != file_hash(checkpoint):
            inspect_checkpoint(spec["root"], snapshot, fit, device=device, queries=spec["fit_queries"],
                               checkpoint_path=checkpoint)
    names = ["evaluation/per_sample.json", "evaluation/summary.json", "evaluation/residuals.json",
             "evaluation/identity.json"] + (["fit.json"] if phase == "validation" else [])
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    atomic_json(destination / "complete.json", {
        "study_hash": spec["hash"], "transport": transport, "objective": objective, "seed": seed,
        "budget": budget, "role": role, "phase": phase, "checkpoint_epoch": checkpoint_data["epoch"] + 1,
        "checkpoint_sha256": file_hash(checkpoint), "resources": _read(snapshot / "resources.json"),
        "artifacts": {name: file_hash(destination / name) for name in names},
    })


def run_transport_study(output, device="cuda", stage="suite"):
    if stage not in ("pilot", "suite", "heldout"):
        raise ValueError("stage must be pilot, suite or heldout")
    device_for(device)
    output = Path(output)
    spec = read_transport_study(output)
    if stage == "heldout":
        if not spec["heldout_splits"]:
            raise ValueError("source corpus has no held-out partitions")
        for transport, objective, seed in _combinations(spec):
            for budget in spec["budgets"]:
                snap = output / "runs" / _run_name(transport, objective, seed) / f"epoch-{budget}"
                for role in ("best", "final"):
                    marker = snap / role / "validation/complete.json"
                    if not marker.exists():
                        raise ValueError("held-out evaluation requires the completed validation suite")
                    record = _verify_record(marker)
                    cp = snap / ("best.pt" if role == "best" else "latest.pt")
                    if record["study_hash"] != spec["hash"] or file_hash(cp) != record["checkpoint_sha256"]:
                        raise ValueError("validation checkpoint or study identity changed")
    _preflight(spec, output, device)
    executions = _read(output / "executions.json") if (output / "executions.json").exists() else []
    executions.append({"started_unix": time.time(), "stage": stage, **_read(output / f"preflight-{device}.json")})
    atomic_json(output / "executions.json", executions)
    for transport, objective, seed in _combinations(spec):
        if stage == "pilot" and (seed != 0 or objective != "mse_balanced"):
            continue
        run = output / "runs" / _run_name(transport, objective, seed)
        budgets = spec["budgets"][:1] if stage == "pilot" else spec["budgets"]
        for budget in budgets:
            snapshot = run / f"epoch-{budget}"
            if stage != "heldout":
                _stage(output, f"train-{run.name}-{budget}",
                       partial(_train_budget, spec, run, snapshot, budget, transport, objective, seed, device),
                       [snapshot / name for name in SNAPSHOT_FILES])
            phase = "heldout" if stage == "heldout" else "validation"
            for role in ("best", "final"):
                _evaluate_role(spec, snapshot, role, phase, transport, objective, seed, budget, device)
            summarize_transport_study(output)
            print(f"Completed {run.name} budget={budget} phase={phase}", flush=True)
    return summarize_transport_study(output)


def _stats(values):
    return {"mean": float(np.mean(values)), "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def _summaries(rows, dimensions):
    """Equal configuration means, paired seeds; never pool query sets or meshes."""
    cells = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in dimensions)
        if row["seed"] in cells[key]:
            raise ValueError("duplicate seed/cell result")
        cells[key][row["seed"]] = row
    output = []
    for key, by_seed in sorted(cells.items(), key=lambda item: str(item[0])):
        entry = dict(zip(dimensions, key, strict=True))
        entry["seeds"] = sorted(by_seed)
        entry["configurations_per_seed"] = [
            {"seed": s, "configurations": by_seed[s].get("configurations")} for s in entry["seeds"]
        ]
        baseline_key = tuple("none" if name == "transport" else value
                             for name, value in zip(dimensions, key, strict=True))
        baseline = cells.get(baseline_key, {})
        earlier = [k[dimensions.index("budget")] for k in cells
                   if k[dimensions.index("budget")] < entry["budget"]]
        earlier_budget = min(earlier) if earlier else None
        budget_key = tuple(earlier_budget if name == "budget" else value
                           for name, value in zip(dimensions, key, strict=True))
        loss_key = tuple("mse_balanced" if name == "objective" else value
                         for name, value in zip(dimensions, key, strict=True))
        entry["metrics"] = {}
        for field in sorted(set.intersection(*(set(r["metrics"]) for r in by_seed.values()))):
            values = [by_seed[s]["metrics"][field] for s in entry["seeds"]]
            differences = [{"seed": s, "value": by_seed[s]["metrics"][field] - baseline[s]["metrics"][field]}
                           for s in entry["seeds"] if s in baseline and field in baseline[s]["metrics"]]
            entry["metrics"][field] = {
                **_stats(values),
                "per_seed": [{"seed": s, "value": by_seed[s]["metrics"][field]} for s in entry["seeds"]],
                "paired_minus_attention": differences,
                "paired_mean": float(np.mean([d["value"] for d in differences])) if differences else None,
            }
            for label, reference in (("earlier_budget", cells.get(budget_key, {})),
                                     ("balanced_mse", cells.get(loss_key, {}))):
                paired = [{"seed": s, "value": by_seed[s]["metrics"][field] - reference[s]["metrics"][field]}
                          for s in entry["seeds"] if s in reference and field in reference[s]["metrics"]]
                entry["metrics"][field][f"paired_minus_{label}"] = paired
        output.append(entry)
    return output


def summarize_transport_study(output):
    output = Path(output)
    spec = read_transport_study(output, verify_data=False)
    records, evaluation, fitting = [], [], []
    for transport, objective, seed in _combinations(spec):
        for budget in spec["budgets"]:
            snapshot = output / "runs" / _run_name(transport, objective, seed) / f"epoch-{budget}"
            for role in ("best", "final"):
                for phase in ("validation", "heldout"):
                    destination = snapshot / role / phase
                    marker = destination / "complete.json"
                    if not marker.exists():
                        continue
                    record = _verify_record(marker)
                    cp = snapshot / ("best.pt" if role == "best" else "latest.pt")
                    if record["study_hash"] != spec["hash"] or file_hash(cp) != record["checkpoint_sha256"]:
                        raise ValueError("completed checkpoint or study changed")
                    records.append(record)
                    labels = {k: record[k] for k in ("transport", "objective", "seed", "budget", "role", "phase")}
                    groups = defaultdict(list)
                    for row in _read(destination / "evaluation/per_sample.json"):
                        metrics = _table_metrics(row["metrics"])
                        metrics.update({f"{f}_final": row["metrics"][f]["final_normalized_rmse"]
                                        for f in ("pressure", "velocity")})
                        metrics.update({k: row["metrics"][k] for k in ("closedness_error", "harmonic_drift")})
                        if row["inference_seconds"] is not None:
                            metrics["trajectory_inference_seconds"] = row["inference_seconds"]
                        for genus, family in (("all", "all"), ("all", row["family"]),
                                              (row["genus"], "all"), (row["genus"], row["family"])):
                            key = (row["split"], row["resolution"], row["queries"], row["method"], genus, family)
                            groups[key].append(metrics)
                    for key, values in groups.items():
                        dimensions = ("split", "resolution", "queries", "method", "genus", "family")
                        metrics = {}
                        for field in set.intersection(*(set(v) for v in values)):
                            data = [v[field] for v in values]
                            metrics[field] = float(np.mean(data))
                            if field in ("pressure", "velocity"):
                                metrics[f"{field}_median"] = float(np.median(data))
                                metrics[f"{field}_p90"] = float(np.quantile(data, .9))
                        evaluation.append({**labels, **dict(zip(dimensions, key, strict=True)),
                                           "configurations": len(values), "metrics": metrics})
                    if phase == "validation":
                        fit_groups = defaultdict(list)
                        for row in _read(destination / "fit.json")["cases"]:
                            metrics = {f: row[f]["normalized_rmse"] for f in ("pressure", "velocity")}
                            metrics["pressure_mean_free"] = row["pressure_mean_free_nrmse"]
                            fit_groups[(row["split"], row["family"], "all")].append(metrics)
                            if "coupling" in row:
                                c = row["coupling"]
                                fit_groups[(row["split"], row["family"], c["direction"])].append(
                                    {k: v for k, v in c.items() if k != "direction"})
                        for (split, family, direction), values in fit_groups.items():
                            fitting.append({**labels, "split": split, "family": family, "direction": direction,
                                            "configurations": len(values),
                                            "metrics": {f: float(np.mean([v[f] for v in values])) for f in values[0]}})
    common = ("transport", "objective", "budget", "role", "phase")
    report = {
        "study_hash": spec["hash"], "status": spec["status"], "interpretation": spec["interpretation"],
        "expected_validation_records": len(spec["seeds"]) * 4 * len(spec["budgets"]) * 2,
        "completed_validation_records": sum(r["phase"] == "validation" for r in records),
        "records": records,
        "evaluation": _summaries(evaluation, common + ("split", "resolution", "queries", "method", "genus", "family")),
        "fitting": _summaries(fitting, common + ("split", "family", "direction")),
    }
    atomic_json(output / "analysis.json", report)
    return report


def export_transport_study(output):
    """Portable review bundle; omit source data, checkpoints and raw predictions."""
    output = Path(output)
    summarize_transport_study(output)  # Verify completed artifacts before export.
    files = [output / name for name in ("study.json", "analysis.json", "progress.json", "executions.json")]
    files += list(output.glob("preflight-*.json"))
    for pattern in ("runs/*/epoch-*/history.json", "runs/*/epoch-*/resources.json",
                    "runs/*/epoch-*/configuration.json", "runs/*/epoch-*/*/*/fit.json",
                    "runs/*/epoch-*/*/*/evaluation/residuals.json"):
        files += list(output.glob(pattern))
    pending = output / "report.zip.pending"
    with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if path.exists():
                archive.write(path, path.relative_to(output))
    destination = output / "report.zip"
    pending.replace(destination)
    return destination
