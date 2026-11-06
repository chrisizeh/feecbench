"""Three paired balanced-MSE runs, with immutable historical controls and validation only."""

from __future__ import annotations

import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .campaign import _stage, file_hash, source_identity
from .coupling import inspect_checkpoint
from .data import atomic_json, digest, training_mse_balance, training_scales
from .training import device_for, evaluate, train


def _read(path):
    return json.loads(Path(path).read_text())


def _sources():
    package = Path(__file__).resolve().parents[3]
    result = source_identity()
    for name in ("scripts/loss_balance.py", "scripts/run_loss_balance.sh"):
        result[name] = file_hash(package / name)
    return result


def _checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes its source directory")
    return path


def _parent_snapshot(parent):
    """Import completed evidence, not resume an old campaign under changed sources.

    The old campaign source guard is deliberately untouched. This new study retains
    the old source record, checks its content hash and completed artifact hashes,
    and binds its own current implementation separately.
    """
    config = _read(parent / "campaign.json")
    if config["hash"] != digest({k: v for k, v in config.items() if k != "hash"}):
        raise ValueError("parent campaign hash mismatch")
    if config["sources"]["uv.lock"] != _sources()["uv.lock"]:
        raise ValueError("parent dependency lock differs from the current study")
    if config["training"]["adapter"] != "scalar" or config["schedule"] != "constant":
        raise ValueError("this study requires the scalar constant-rate controls")
    root = parent / "data"
    index, manifest = _read(root / "index.json"), _read(root / "manifest.json")
    expected = config["manifest"]
    if (
        index.get("complete") is not True
        or {k: manifest[k] for k in expected} != expected
        or index["fingerprint"] != manifest["fingerprint"]
        or index["fingerprint"] != digest({"manifest": expected, "settings": manifest["settings"]})
        or any(manifest["settings"][k] != v for k, v in config["numerical"].items())
    ):
        raise ValueError("parent corpus is incomplete or differs from its campaign")
    entries = [e for e in index["samples"] if e["split"] in ("train", "validation") and e["resolution"] == "medium"]
    expected_ids = {c["config_id"] for c in expected["configurations"] if c["split"] in ("train", "validation")}
    if len(entries) != len(expected_ids) or {e["config_id"] for e in entries} != expected_ids:
        raise ValueError("parent train/validation samples are incomplete or duplicated")
    expected_splits = {c["config_id"]: c["split"] for c in expected["configurations"]}
    if any(e["split"] != expected_splits[e["config_id"]] for e in entries):
        raise ValueError("parent index split labels differ from the frozen manifest")
    files = {name: file_hash(parent / name) for name in ("campaign.json", "data/index.json", "data/manifest.json")}
    for entry in entries:
        path = _checked_path(root, entry["path"])
        files[str(path.relative_to(parent))] = file_hash(path)
    paths = [_checked_path(root, e["path"]) for e in entries if e["split"] == "train"]
    scales = training_scales(paths)
    calibration = training_mse_balance(paths, scales)
    updates = config["training"]["epochs"] * math.ceil(len(paths) / config["training"]["accumulation"])
    common, controls = None, []
    for objective in ("mse", "mass"):
        for seed in (0, 1, 2):
            relative = Path("runs") / f"{objective}-seed{seed}"
            run = parent / relative
            complete = _read(run / "complete.json")
            required = {"best.pt", "history.json", "configuration.json", "fit.json", "evaluation/summary.json"}
            if not required <= set(complete["artifacts"]):
                raise ValueError("parent control is missing required completed artifacts")
            files[str(relative / "complete.json")] = file_hash(run / "complete.json")
            for name, expected_hash in complete["artifacts"].items():
                if file_hash(_checked_path(run, name)) != expected_hash:
                    raise ValueError(f"parent control artifact changed: {relative / name}")
                files[str(relative / name)] = expected_hash
            settings = _read(run / "configuration.json")
            if (
                settings["objective"] != objective
                or settings["seed"] != seed
                or complete["objective"] != objective
                or complete["seed"] != seed
                or settings["fingerprint"] != index["fingerprint"]
                or settings["scales"] != scales
                or any(
                    settings[k] != config["training"][k] for k in ("width", "blocks", "slices", "lr", "accumulation")
                )
                or settings.get("adapter", "scalar") != "scalar"
                or settings["epochs_requested"] != config["training"]["epochs"]
                or complete["updates"] != updates
            ):
                raise ValueError("parent controls have unmatched data, normalization or training settings")
            shared = {k: v for k, v in settings.items() if k not in ("objective", "seed", "device", "torch", "numpy")}
            if common is not None and common != shared:
                raise ValueError("parent control configurations are not matched")
            common = shared
            history, fit = _read(run / "history.json"), _read(run / "fit.json")
            if [r["epoch"] for r in history] != list(range(config["training"]["epochs"])):
                raise ValueError("parent control has an incomplete training budget")
            if (
                fit["source_checkpoint_sha256"] != complete["artifacts"]["best.pt"]
                or fit["fingerprint"] != index["fingerprint"]
                or fit["queries_per_configuration"] != 3
            ):
                raise ValueError("parent fitting diagnostics do not match checkpoint/data/query rule")
            controls.append(
                {
                    "objective": objective,
                    "seed": seed,
                    "fit_summary": fit["summary"],
                    "validation_evaluation": [
                        r for r in _read(run / "evaluation/summary.json") if r["split"] == "validation"
                    ],
                    "checkpoint_sha256": complete["artifacts"]["best.pt"],
                    "checkpoint_epoch": fit["checkpoint_epoch"],
                    "historical_gradient_diagnostics": None,
                }
            )
    return {
        "campaign_hash": config["hash"],
        "original_sources": config["sources"],
        "files": files,
        "fingerprint": index["fingerprint"],
        "training": config["training"],
        "updates_per_run": updates,
        "scales": scales,
        "bands": common["bands"],
        "spectral_weight": common["spectral_weight"],
        "controls": controls,
        "field_balance": calibration,
    }


def prepare_study(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("study output must be separate from the read-only parent campaign")
    snapshot = _parent_snapshot(parent)
    specification = {
        "version": 1,
        "status": "training/validation loss study; no held-out test evaluation",
        "parent": str(parent),
        "parent_snapshot": snapshot,
        "objective": "mse_balanced",
        "seeds": [0, 1, 2],
        "evaluation_splits": ["validation"],
        "fit_queries": 3,
        "sources": _sources(),
        "calibration_limit": "matches expected zero loss, not initial-model loss, gradients or optimizer trajectory",
    }
    specification["hash"] = digest(specification)
    path = output / "study.json"
    if path.exists():
        if _read(path) != specification:
            raise ValueError("study settings, sources or parent changed; use a new directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("new study directory must be empty")
    else:
        atomic_json(path, specification)
    return specification


def read_study(output, verify_parent=True):
    specification = _read(Path(output) / "study.json")
    if specification["hash"] != digest({k: v for k, v in specification.items() if k != "hash"}):
        raise ValueError("study configuration hash mismatch")
    if specification["sources"] != _sources():
        raise ValueError("study source/lockfile changed; retain its checkout or prepare another study")
    if verify_parent:
        parent = Path(specification["parent"])
        for name, expected in specification["parent_snapshot"]["files"].items():
            if file_hash(_checked_path(parent, name)) != expected:
                raise ValueError(f"parent data or artifact changed: {name}")
    return specification


def summarize_study(output):
    output = Path(output)
    specification = read_study(output, verify_parent=False)
    records = list(specification["parent_snapshot"]["controls"])
    for marker in sorted((output / "runs").glob("*/complete.json")):
        record = _read(marker)
        for name, expected in record["artifacts"].items():
            if file_hash(_checked_path(marker.parent, name)) != expected:
                raise ValueError(f"study artifact changed: {marker.parent / name}")
        records.append(record)
    cells, validation = defaultdict(list), []
    for record in records:
        for row in record["fit_summary"]:
            if row["split"] not in ("train", "validation"):
                raise ValueError("held-out scores cannot enter the loss study comparison")
            cells[(row["split"], row["family"])].append(
                {**row, "objective": record["objective"], "seed": record["seed"]}
            )
        validation.extend(
            {**r, "objective": record["objective"], "seed": record["seed"]} for r in record["validation_evaluation"]
        )
    comparison = []
    for (split, family), rows in sorted(cells.items()):
        paired = {(r["objective"], r["seed"]): r for r in rows}
        if len(paired) != len(rows):
            raise ValueError("duplicate objective/seed/family fitting score")
        for objective in sorted({r["objective"] for r in rows}):
            selected = sorted((r for r in rows if r["objective"] == objective), key=lambda r: r["seed"])
            entry = {"split": split, "family": family, "objective": objective, "seeds": [r["seed"] for r in selected]}
            for field in ("pressure", "velocity", "pressure_mean_free"):
                values = [r[field] for r in selected]
                entry[field] = {
                    "mean": float(np.mean(values)),
                    "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "per_seed": [{"seed": r["seed"], "value": r[field]} for r in selected],
                    "paired_differences": {
                        control: [
                            {"seed": r["seed"], "value": r[field] - paired[(control, r["seed"])][field]}
                            for r in selected
                        ]
                        for control in ("mse", "mass")
                    },
                }
            comparison.append(entry)
    report = {
        "study_hash": specification["hash"],
        "parent_campaign_hash": specification["parent_snapshot"]["campaign_hash"],
        "status": specification["status"],
        "field_balance": specification["parent_snapshot"]["field_balance"],
        "calibration_limit": specification["calibration_limit"],
        "fit_queries_per_configuration": specification["fit_queries"],
        "runs": records,
        "fit_comparison": comparison,
        "validation_evaluation": validation,
        "interpretation": "paired fixed-query fits; validation evaluation uses separate regular/offgrid query sets",
    }
    atomic_json(output / "analysis.json", report)
    return report


def run_study(output, device="cuda"):
    device_for(device)
    output = Path(output)
    specification = read_study(output)
    parent, snapshot = Path(specification["parent"]), specification["parent_snapshot"]
    root = parent / "data"
    executions_path = output / "executions.json"
    executions = _read(executions_path) if executions_path.exists() else []
    executions.append(
        {
            "started_unix": time.time(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        }
    )
    atomic_json(executions_path, executions)
    for seed in specification["seeds"]:
        run = output / "runs" / f"mse_balanced-seed{seed}"
        _stage(
            output,
            f"train-mse_balanced-{seed}",
            lambda run=run, seed=seed: train(
                root,
                run,
                device=device,
                seed=seed,
                objective="mse_balanced",
                bands=snapshot["bands"],
                spectral_weight=snapshot["spectral_weight"],
                field_balance=snapshot["field_balance"],
                resume=(run / "latest.pt").exists(),
                progress=True,
                **snapshot["training"],
            ),
            [run / name for name in ("best.pt", "latest.pt", "history.json", "configuration.json")],
        )
        evaluate(root, run / "best.pt", run / "evaluation", device=device, resume=True, splits=["validation"])
        fit_path = run / "fit.json"
        checkpoint_hash = file_hash(run / "best.pt")
        if not fit_path.exists() or _read(fit_path)["source_checkpoint_sha256"] != checkpoint_hash:
            inspect_checkpoint(root, run, fit_path, device=device, queries=specification["fit_queries"])
        fit = _read(fit_path)
        names = (
            "best.pt",
            "history.json",
            "configuration.json",
            "fit.json",
            "evaluation/per_sample.json",
            "evaluation/summary.json",
            "evaluation/residuals.json",
        )
        atomic_json(
            run / "complete.json",
            {
                "objective": "mse_balanced",
                "seed": seed,
                "updates": snapshot["updates_per_run"],
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_epoch": fit["checkpoint_epoch"],
                "fit_summary": fit["summary"],
                "validation_evaluation": _read(run / "evaluation/summary.json"),
                "training_diagnostics": [
                    {"epoch": r["epoch"], **r["training_diagnostics"]} for r in _read(run / "history.json")
                ],
                "artifacts": {name: file_hash(run / name) for name in names},
            },
        )
        summarize_study(output)
        print(f"Completed balanced MSE seed {seed}; analysis: {output / 'analysis.json'}", flush=True)
    return summarize_study(output)
