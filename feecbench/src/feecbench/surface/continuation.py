"""Matched extra-budget constant/cosine branches from completed balanced-MSE checkpoints."""

from __future__ import annotations

import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .campaign import _stage, file_hash
from .coupling import inspect_checkpoint
from .data import atomic_json, digest
from .loss_balance import _checked_path, _read
from .loss_balance import _sources as balance_sources
from .training import _save_checkpoint, device_for, evaluate, train


def _sources():
    package = Path(__file__).resolve().parents[3]
    sources = balance_sources()
    for name in ("scripts/continue_loss_balance.py", "scripts/run_loss_continuation.sh"):
        sources[name] = file_hash(package / name)
    return sources


def _verify_files(root, files):
    for name, expected in files.items():
        if file_hash(_checked_path(root, name)) != expected:
            raise ValueError(f"source data or artifact changed: {root / name}")


def initialize_branch(source_checkpoint, output, policy):
    """Commit one bootstrap checkpoint; never recopy over a resumed branch."""
    output = Path(output)
    if file_hash(source_checkpoint) != policy["source_checkpoint_sha256"]:
        raise ValueError("continuation source checkpoint hash mismatch")
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if source["identity"] != policy["source_identity"] or digest(source["settings"]) != source["identity"]:
        raise ValueError("continuation source identity mismatch")
    if source["epoch"] + 1 != policy["start_epoch"] or "continuation" in source["settings"]:
        raise ValueError("continuation must fork the declared original latest epoch")
    settings = {**source["settings"], "continuation": policy}
    identity = digest(settings)
    latest = output / "latest.pt"
    if latest.exists():
        if torch.load(latest, map_location="cpu", weights_only=False)["identity"] != identity:
            raise ValueError("branch already exists with different continuation settings")
        return
    if output.exists() and any(
        p.name != "latest.pending" or not p.is_file() or p.is_symlink() for p in output.iterdir()
    ):
        raise ValueError("branch output must be empty before initialization")
    output.mkdir(parents=True, exist_ok=True)
    # Model, optimizer, RNG, history and embedded best state are unchanged.
    source.update(settings=settings, identity=identity, continuation_updates=0, scheduler=None)
    _save_checkpoint(latest, source)
    # train(resume=True) reconstructs best.pt and history.json from this authoritative commit.


def prepare_continuation(source_study, output, additional_epochs=200, final_lr_ratio=0.01):
    source_study, output = Path(source_study).resolve(), Path(output).resolve()
    if additional_epochs < 1 or not np.isfinite(final_lr_ratio) or not 0 < final_lr_ratio < 1:
        raise ValueError("positive additional epochs and final LR ratio in (0,1) required")
    original = _read(source_study / "study.json")
    if original["hash"] != digest({k: v for k, v in original.items() if k != "hash"}):
        raise ValueError("source study hash mismatch")
    ancestor = Path(original["parent"]).resolve()
    for source in (source_study, ancestor):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("continuation output must be separate from every source directory")
    if original["sources"]["uv.lock"] != _sources()["uv.lock"]:
        raise ValueError("dependency lock differs from source study")
    snapshot = original["parent_snapshot"]
    _verify_files(ancestor, snapshot["files"])
    files = {"study.json": file_hash(source_study / "study.json")}
    seeds = []
    common = None
    for seed in (0, 1, 2):
        relative = Path("runs") / f"mse_balanced-seed{seed}"
        run = source_study / relative
        complete = _read(run / "complete.json")
        _verify_files(run, complete["artifacts"])
        for name, value in complete["artifacts"].items():
            files[str(relative / name)] = value
        files[str(relative / "complete.json")] = file_hash(run / "complete.json")
        latest_path = run / "latest.pt"
        latest_hash = file_hash(latest_path)  # Source complete.json did not include latest.pt.
        files[str(relative / "latest.pt")] = latest_hash
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        settings = checkpoint["settings"]
        saved_settings = _read(run / "configuration.json")
        expected_settings = {
            k: v for k, v in saved_settings.items() if k not in ("epochs_requested", "device", "torch", "numpy")
        }
        history = _read(run / "history.json")
        epochs = snapshot["training"]["epochs"]
        if (
            settings != expected_settings
            or checkpoint["identity"] != digest(settings)
            or settings["objective"] != "mse_balanced"
            or settings["seed"] != seed
            or complete["objective"] != "mse_balanced"
            or complete["seed"] != seed
            or settings["fingerprint"] != snapshot["fingerprint"]
            or settings["scales"] != snapshot["scales"]
            or settings["field_balance"] != snapshot["field_balance"]
            or "continuation" in settings
            or checkpoint["epoch"] + 1 != epochs
            or saved_settings["epochs_requested"] != epochs
            or history != checkpoint["history"]
            or [r["epoch"] for r in history] != list(range(epochs))
            or complete["updates"] != snapshot["updates_per_run"]
            or any(g["lr"] != settings["lr"] for g in checkpoint["optimizer"]["param_groups"])
        ):
            raise ValueError("source latest checkpoint/settings/history do not match the completed study")
        best_row = min(history, key=lambda r: r["validation_mass_score"])
        if (
            checkpoint["best_epoch"] != best_row["epoch"]
            or checkpoint["best"] != best_row["validation_mass_score"]
            or best["epoch"] != checkpoint["best_epoch"]
            or best["best"] != checkpoint["best"]
            or set(checkpoint["best_model"]) != set(best["model"])
            or any(not torch.equal(v, best["model"][k]) for k, v in checkpoint["best_model"].items())
        ):
            raise ValueError("source embedded best state does not match its selected checkpoint")
        for key in ("numpy_rng", "torch_rng", "cuda_rng", "optimizer"):
            if key not in checkpoint:
                raise ValueError(f"source continuation state missing: {key}")
        shared = {k: v for k, v in settings.items() if k != "seed"}
        if common is not None and shared != common:
            raise ValueError("source seeds have different training settings")
        common = shared
        count = len(settings["field_balance"]["training_configurations"])
        updates_per_epoch = math.ceil(count / settings["accumulation"])
        if updates_per_epoch * epochs != complete["updates"]:
            raise ValueError("source optimizer update budget mismatch")
        fit = _read(run / "fit.json")
        if fit["source_checkpoint_sha256"] != files[str(relative / "best.pt")] or fit["queries_per_configuration"] != 3:
            raise ValueError("source fit diagnostics mismatch")
        seeds.append(
            {
                "seed": seed,
                "source_latest": str(relative / "latest.pt"),
                "source_checkpoint_sha256": latest_hash,
                "source_identity": checkpoint["identity"],
                "start_epoch": epochs,
                "updates_per_epoch": updates_per_epoch,
                "parent_best_epoch": checkpoint["best_epoch"],
                "parent_best_score": checkpoint["best"],
                "parent_best_fit": fit["summary"],
                "source_history": history,
            }
        )
    kwargs = {
        k: common[k]
        for k in (
            "objective",
            "width",
            "blocks",
            "slices",
            "lr",
            "accumulation",
            "bands",
            "spectral_weight",
            "field_balance",
        )
    }
    kwargs["adapter"] = common.get("adapter", "scalar")
    specification = {
        "version": 1,
        "status": "paired continuation; train/validation only",
        "source_study": str(source_study),
        "source_study_hash": original["hash"],
        "source_files": files,
        "ancestor": str(ancestor),
        "ancestor_files": snapshot["files"],
        "original_sources": original["sources"],
        "sources": _sources(),
        "training": kwargs,
        "seeds": seeds,
        "additional_epochs": additional_epochs,
        "cosine_final_lr_ratio": final_lr_ratio,
        "schedules": ["constant", "cosine"],
        "selection": "best validation mass over full history including parent; separate final-budget scores",
    }
    specification["hash"] = digest(specification)
    path = output / "continuation.json"
    if path.exists():
        if _read(path) != specification:
            raise ValueError("continuation settings/source changed; use a new directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("new continuation directory must be empty")
    else:
        atomic_json(path, specification)
    return specification


def read_continuation(output, verify_sources=True):
    specification = _read(Path(output) / "continuation.json")
    if specification["hash"] != digest({k: v for k, v in specification.items() if k != "hash"}):
        raise ValueError("continuation configuration hash mismatch")
    if specification["sources"] != _sources():
        raise ValueError("continuation source/lockfile changed; retain its checkout or prepare another study")
    if verify_sources:
        _verify_files(Path(specification["source_study"]), specification["source_files"])
        _verify_files(Path(specification["ancestor"]), specification["ancestor_files"])
    return specification


def summarize_continuation(output):
    output = Path(output)
    specification = read_continuation(output, verify_sources=False)
    runs, groups = [], defaultdict(dict)
    for path in sorted((output / "runs").glob("*/complete.json")):
        record = _read(path)
        _verify_files(path.parent, record["artifacts"])
        runs.append(record)
        for role in ("best", "final"):
            for row in record[role]["fit_summary"]:
                key = (role, row["split"], row["family"])
                pair = (record["schedule"], record["seed"])
                if pair in groups[key]:
                    raise ValueError("duplicate continuation comparison cell")
                groups[key][pair] = row
    comparison = []
    for (role, split, family), pairs in sorted(groups.items()):
        row = {"checkpoint_role": role, "split": split, "family": family}
        for field in ("pressure", "velocity", "pressure_mean_free"):
            row[field] = {}
            for schedule in specification["schedules"]:
                values = [
                    {"seed": seed, "value": r[field]} for (arm, seed), r in sorted(pairs.items()) if arm == schedule
                ]
                if values:
                    row[field][schedule] = {
                        "mean": float(np.mean([v["value"] for v in values])),
                        "seed_std": float(np.std([v["value"] for v in values], ddof=1)) if len(values) > 1 else None,
                        "per_seed": values,
                    }
            row[field]["paired_cosine_minus_constant"] = [
                {"seed": seed, "value": pairs[("cosine", seed)][field] - pairs[("constant", seed)][field]}
                for seed in (0, 1, 2)
                if ("cosine", seed) in pairs and ("constant", seed) in pairs
            ]
        comparison.append(row)
    starting = []
    progress_path = output / "progress.json"
    progress = _read(progress_path) if progress_path.exists() else {}
    for seed in specification["seeds"]:
        path = output / "initial" / f"seed{seed['seed']}.json"
        summary = None
        stage = progress.get(f"initial-fit-{seed['seed']}", {})
        if stage.get("state") == "complete":
            expected_hash = stage.get("artifacts", {}).get(str(path.relative_to(output)))
            if not path.exists() or expected_hash != file_hash(path):
                raise ValueError("initial fit artifact hash mismatch")
            fit = _read(path)
            if (
                fit["source_checkpoint_sha256"] != seed["source_checkpoint_sha256"]
                or fit["checkpoint_epoch"] + 1 != seed["start_epoch"]
                or fit["queries_per_configuration"] != 3
            ):
                raise ValueError("initial fit does not match source latest checkpoint")
            summary = fit["summary"]
        starting.append({**seed, "latest_fit_summary": summary})
    report = {
        "continuation_hash": specification["hash"],
        "source_study_hash": specification["source_study_hash"],
        "status": specification["status"],
        "selection": specification["selection"],
        "additional_epochs": specification["additional_epochs"],
        "source_seeds": starting,
        "runs": runs,
        "fit_comparison": comparison,
        "learning_rate_semantics": "cosine steps after optimizer updates; next_learning_rate is not the last rate used",
    }
    atomic_json(output / "analysis.json", report)
    return report


def run_continuation(output, device="cuda"):
    device_for(device)
    output = Path(output)
    specification = read_continuation(output)
    root, source = Path(specification["ancestor"]) / "data", Path(specification["source_study"])
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
        source_latest = source / seed["source_latest"]
        initial = output / "initial" / f"seed{seed['seed']}.json"
        _stage(
            output,
            f"initial-fit-{seed['seed']}",
            lambda initial=initial, source_latest=source_latest: inspect_checkpoint(
                root,
                source_latest.parent,
                initial,
                device=device,
                checkpoint_path=source_latest,
            ),
            [initial],
        )
        for schedule in specification["schedules"]:
            run = output / "runs" / f"{schedule}-seed{seed['seed']}"
            policy = {
                k: seed[k] for k in ("source_identity", "source_checkpoint_sha256", "start_epoch", "updates_per_epoch")
            }
            policy.update(
                schedule=schedule,
                additional_epochs=specification["additional_epochs"],
                final_lr_ratio=specification["cosine_final_lr_ratio"] if schedule == "cosine" else 1.0,
            )
            initialize_branch(source_latest, run, policy)
            _stage(
                output,
                f"train-{schedule}-{seed['seed']}",
                lambda run=run, seed=seed, policy=policy: train(
                    root,
                    run,
                    device=device,
                    seed=seed["seed"],
                    resume=True,
                    progress=True,
                    epochs=seed["start_epoch"] + specification["additional_epochs"],
                    continuation=policy,
                    **specification["training"],
                ),
                [run / name for name in ("best.pt", "latest.pt", "history.json", "configuration.json")],
            )
            result = {}
            artifacts = ["best.pt", "latest.pt", "history.json", "configuration.json"]
            history = _read(run / "history.json")
            for role, filename in (("best", "best.pt"), ("final", "latest.pt")):
                destination = run / role
                checkpoint_path = run / filename
                evaluate(
                    root, checkpoint_path, destination / "evaluation", device=device, resume=True, splits=["validation"]
                )
                fit_path = destination / "fit.json"
                checkpoint_hash = file_hash(checkpoint_path)
                if not fit_path.exists() or _read(fit_path)["source_checkpoint_sha256"] != checkpoint_hash:
                    inspect_checkpoint(root, run, fit_path, device=device, checkpoint_path=checkpoint_path)
                fit = _read(fit_path)
                result[role] = {
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_epoch": fit["checkpoint_epoch"],
                    "selected_from_parent": fit["checkpoint_epoch"] < seed["start_epoch"],
                    "validation_mass_score": history[fit["checkpoint_epoch"]]["validation_mass_score"],
                    "fit_summary": fit["summary"],
                    "validation_evaluation": _read(destination / "evaluation/summary.json"),
                }
                artifacts += [
                    f"{role}/{name}"
                    for name in (
                        "fit.json",
                        "evaluation/per_sample.json",
                        "evaluation/summary.json",
                        "evaluation/residuals.json",
                    )
                ]
            atomic_json(
                run / "complete.json",
                {
                    "seed": seed["seed"],
                    "schedule": schedule,
                    "additional_updates": specification["additional_epochs"] * seed["updates_per_epoch"],
                    "continuation_history": history[seed["start_epoch"] :],
                    **result,
                    "artifacts": {name: file_hash(run / name) for name in artifacts},
                },
            )
            summarize_continuation(output)
            print(f"Completed {schedule} seed {seed['seed']}; analysis: {output / 'analysis.json'}", flush=True)
    return summarize_continuation(output)
