"""Fixed, resumable exploratory generalization campaigns; never attest protocol gates."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .corpus import generate_corpus
from .coupling import inspect_checkpoint
from .data import atomic_json, digest, make_manifest, training_scales
from .protocol import read_protocol
from .reference_checks import check_references
from .training import device_for, evaluate, train


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_identity():
    package = Path(__file__).resolve().parents[3]
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [
        package / "uv.lock",
        package / "scripts/acoustic_campaign.py",
        package / "scripts/run_acoustic_campaign.sh",
    ]
    return {str(p.relative_to(package)): file_hash(p) for p in paths}


def prepare_campaign(directory, *, epochs=None, smoke=False, protocol_path=None, transport="none"):
    """Create the run specification before looking at any campaign outcomes."""
    if transport not in ("none", "d0_delta1"):
        raise ValueError("transport must be none or d0_delta1")
    directory = Path(directory)
    epochs = epochs if epochs is not None else (1 if smoke else 200)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    protocol = read_protocol(protocol_path) if protocol_path else None
    if smoke and protocol:
        raise ValueError("smoke does not consume a scientific protocol")
    numerical = {"grid": 16, "dt": 0.002, "horizon": 1.0, "snapshots": 51, "modes": 64}
    if smoke:
        numerical.update(grid=12, dt=0.02, horizon=0.12, snapshots=4, modes=8)
    if protocol:
        numerical = {k: protocol["numerical_settings"][k] for k in numerical}
        if not any(b["resolved"] for b in protocol["bands"]):
            raise ValueError("spectral campaign requires a resolved protocol band")
    counts = {"train": 512, "validation": 64, "id_test": 64, "geometry_ood": 64, "topology_ood": 64}
    if smoke:
        counts = {k: 8 if k == "train" else 4 for k in counts}
    manifest = make_manifest(counts, seed=202)
    audit = make_manifest(
        {"id_test": 4 if smoke else 8, "geometry_ood": 4 if smoke else 8, "topology_ood": 4 if smoke else 8},
        seed=203,
        development=True,
    )
    main_geometries = {c["geometry_id"] for c in manifest["configurations"]}
    audit_geometries = {c["geometry_id"] for c in audit["configurations"]}
    if main_geometries & audit_geometries:
        raise ValueError("audit and campaign geometries overlap")
    if protocol and (main_geometries | audit_geometries) & set(protocol["development_geometries"]):
        raise ValueError("campaign or audit overlaps protocol development")
    config = {
        "version": 1,
        "status": "smoke only" if smoke else "exploratory generalization campaign",
        "numerical": numerical,
        "paired_id_meshes": True,
        "manifest": manifest,
        "audit_manifest": audit,
        "protocol": protocol,
        "training": {
            "epochs": epochs,
            "width": 32 if smoke else 128,
            "blocks": 1 if smoke else 4,
            "slices": 8 if smoke else 32,
            "lr": 0.001,
            "accumulation": 4,
            "adapter": "scalar",
        },
        "schedule": "constant",
        "updates_per_run": epochs * math.ceil(counts["train"] / 4),
        "seeds": [0, 1, 2],
        "objectives": ["mass", "mse", "spectral"] if protocol else ["mass", "mse"],
        "sources": source_identity(),
        "interpretation": {
            "time": "interpolation within the horizon only",
            "geometry": "held-out hole-radius range; other geometry ranges shared",
            "topology": "unseen genus with accompanying geometry changes",
            "initial_conditions": "same four IC families; no unseen-family or coefficient test",
            "audit": "independent preselected reference subset, not a continuum error bound or protocol approval",
            "selection": "validation mass score only; no automatic adjustment using test results",
        },
    }
    if transport != "none":
        config["training"]["transport"] = transport
    config["hash"] = digest(config)
    path = directory / "campaign.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("campaign settings/source mismatch; use a new directory")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError("new campaign directory must be empty")
    else:
        atomic_json(path, config)
    return config


def read_campaign(directory):
    config = json.loads((Path(directory) / "campaign.json").read_text())
    expected = config.pop("hash")
    if digest(config) != expected:
        raise ValueError("campaign configuration hash mismatch")
    config["hash"] = expected
    if config["sources"] != source_identity():
        raise ValueError("campaign source/lockfile changed; use its original checkout or prepare a new directory")
    return config


def _stage(directory, name, function, artifacts):
    """Commit stage completion only after success and bind it to artifact bytes."""
    path = directory / "progress.json"
    progress = json.loads(path.read_text()) if path.exists() else {}
    record = progress.get(name, {})
    if record.get("state") == "complete" and all(
        p.exists() and record.get("artifacts", {}).get(str(p.relative_to(directory))) == file_hash(p) for p in artifacts
    ):
        print(f"Skipping completed stage: {name}", flush=True)
        return
    start = time.time()
    progress[name] = {"state": "running", "started_unix": start}
    atomic_json(path, progress)
    print(f"Starting stage: {name}", flush=True)
    try:
        function()
        progress[name].update(
            state="complete",
            seconds=time.time() - start,
            artifacts={str(p.relative_to(directory)): file_hash(p) for p in artifacts},
        )
    except BaseException as exc:
        progress[name].update(state="failed", error=f"{type(exc).__name__}: {exc}", seconds=time.time() - start)
        raise
    finally:
        atomic_json(path, progress)


def summarize_campaign(directory):
    """Keep split/family/genus and all controls separate, with seed variability."""
    directory = Path(directory)
    config = read_campaign(directory)
    runs, grouped = [], defaultdict(list)
    for marker in sorted((directory / "runs").glob("*/complete.json")):
        completed = json.loads(marker.read_text())
        run = marker.parent
        for name, expected in completed["artifacts"].items():
            if file_hash(run / name) != expected:
                raise ValueError(f"completed run artifact changed: {run / name}")
        runs.append(completed)
        rows = json.loads((run / "evaluation" / "per_sample.json").read_text())
        # Each table cell has one entry per configuration; meshes and query sets stay separate.
        cells = defaultdict(list)
        for row in rows:
            for family, genus in (("all", "all"), (row["family"], row["genus"])):
                key = (row["method"], row["split"], row["resolution"], row["queries"], family, genus)
                cells[key].append(row)
        for key, items in cells.items():
            entry = dict(zip(("method", "split", "resolution", "queries", "family", "genus"), key, strict=True))
            entry.update(objective=completed["objective"], seed=completed["seed"], configurations=len(items))
            for field in ("pressure", "velocity"):
                values = [r["metrics"][field]["normalized_rmse"] for r in items]
                entry[field] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p90": float(np.quantile(values, 0.9)),
                }
            grouped[(completed["objective"], *key)].append(entry)
    summaries = []
    for entries in grouped.values():
        first = entries[0]
        row = {k: v for k, v in first.items() if k not in ("seed", "pressure", "velocity")}
        row["seeds"] = [e["seed"] for e in entries]
        for field in ("pressure", "velocity"):
            values = [e[field]["mean"] for e in entries]
            row[field] = {
                "mean": float(np.mean(values)),
                "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "per_seed": [{"seed": e["seed"], **e[field]} for e in entries],
            }
        summaries.append(row)
    report = {"campaign_hash": config["hash"], "status": config["status"], "runs": runs, "summary": summaries}
    audit_path = directory / "audit" / "reference_uncertainty.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        report["reference_uncertainty"] = {
            "path": str(audit_path.relative_to(directory)),
            "sha256": file_hash(audit_path),
            "interpretation": audit["interpretation"],
            "cases": audit["cases"],
        }
    atomic_json(directory / "analysis.json", report)
    return report


def run_campaign(directory, device="cuda", stage="baseline"):
    """Baseline first; suite explicitly expands without selecting from held-out scores."""
    device_for(device)  # Reject unavailable CUDA before spending time generating data.
    if stage not in ("baseline", "suite"):
        raise ValueError("stage must be baseline or suite")
    directory = Path(directory)
    config = read_campaign(directory)
    environment_path = directory / "executions.json"
    executions = json.loads(environment_path.read_text()) if environment_path.exists() else []
    executions.append(
        {
            "started_unix": time.time(),
            "stage": stage,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        }
    )
    atomic_json(environment_path, executions)
    audit = directory / "audit"
    root = directory / "data"
    numerical = config["numerical"]
    # Generation itself is sample-resumable and validates existing sample fingerprints.
    print("Generating/resuming independent numerical-audit data", flush=True)
    generate_corpus(audit / "data", config["audit_manifest"], **numerical, progress=True)
    audit_report = audit / "reference_uncertainty.json"
    thresholds = tuple(config["protocol"]["thresholds"]) if config["protocol"] else (10.0, 30.0)
    _stage(
        directory,
        "reference-audit",
        lambda: check_references(audit / "data", audit_report, thresholds=thresholds),
        [audit_report],
    )
    print("Generating/resuming campaign data and paired ID meshes", flush=True)
    generate_corpus(root, config["manifest"], **numerical, paired=True, progress=True)
    bands = None
    if config["protocol"]:
        index = json.loads((root / "index.json").read_text())
        scales = training_scales([root / e["path"] for e in index["samples"]])
        bands = [{**b, "scale": b["scale_raw"] / scales["pressure"]} for b in config["protocol"]["bands"]]
    pairs = [("mass", 0)]
    if stage == "suite":
        pairs += [(o, s) for s in config["seeds"] for o in config["objectives"] if (o, s) != ("mass", 0)]
    for objective, seed in pairs:
        run = directory / "runs" / f"{objective}-seed{seed}"
        _stage(
            directory,
            f"train-{objective}-{seed}",
            lambda run=run, objective=objective, seed=seed: train(
                root,
                run,
                device=device,
                seed=seed,
                objective=objective,
                bands=bands,
                resume=(run / "latest.pt").exists(),
                progress=True,
                **config["training"],
            ),
            [run / name for name in ("best.pt", "latest.pt", "history.json", "configuration.json")],
        )
        # Reuses per-case caches only for the exact checkpoint and intact prediction bytes.
        print(f"Evaluating {objective} seed {seed}", flush=True)
        evaluate(root, run / "best.pt", run / "evaluation", device=device, resume=True)
        fit = run / "fit.json"
        checkpoint_hash = file_hash(run / "best.pt")
        if not fit.exists() or json.loads(fit.read_text())["source_checkpoint_sha256"] != checkpoint_hash:
            inspect_checkpoint(root, run, fit, device=device)
        artifacts = [
            "best.pt",
            "history.json",
            "configuration.json",
            "fit.json",
            "evaluation/per_sample.json",
            "evaluation/summary.json",
            "evaluation/residuals.json",
        ]
        atomic_json(
            run / "complete.json",
            {
                "objective": objective,
                "seed": seed,
                "updates": config["updates_per_run"],
                "fit_summary": json.loads(fit.read_text()).get("summary", []),
                "artifacts": {name: file_hash(run / name) for name in artifacts},
            },
        )
        summarize_campaign(directory)
        print(f"Completed {objective} seed {seed}; analysis: {directory / 'analysis.json'}", flush=True)
    return summarize_campaign(directory)
