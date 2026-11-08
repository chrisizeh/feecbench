"""Offline figures and paired-seed summaries from saved raw-unit evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import atomic_json, load


def plot_run(root, evaluation):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    root, evaluation = Path(root), Path(evaluation)
    rows = json.loads((evaluation / "per_sample.json").read_text())
    figures = evaluation / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for field, axis in zip(("pressure", "velocity"), axes, strict=True):
        curves = defaultdict(list)
        for row in rows:
            if row["queries"] == "regular" and row["resolution"] == "medium":
                curves[row["method"]].append((row["metrics"]["times"], row["metrics"][field]["curve"]))
        for method, data in curves.items():
            axis.plot(data[0][0], np.mean([curve for _, curve in data], axis=0), label=method)
        axis.set(xlabel="Query time", ylabel="Normalized mass error", title=field.capitalize())
        axis.legend()
    fig.savefig(figures / "error_time.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    resolutions = ("coarse", "medium", "fine", "remeshed")
    for field, axis in zip(("pressure", "velocity"), axes, strict=True):
        for method in sorted({row["method"] for row in rows}):
            x, y = [], []
            for resolution in resolutions:
                values = [
                    row["metrics"][field]["normalized_rmse"]
                    for row in rows
                    if row["method"] == method
                    and row["resolution"] == resolution
                    and row["split"] == "id_test"
                    and row["queries"] == "regular"
                ]
                if values:
                    x.append(resolution)
                    y.append(np.mean(values))
            if x:
                axis.plot(x, y, "o-", label=method)
        axis.set(ylabel="Normalized mass error", title=field.capitalize())
        if axis.lines:
            axis.legend()
    fig.savefig(figures / "error_resolution.png", dpi=150)
    plt.close(fig)
    index = json.loads((root / "index.json").read_text())
    for entry in index["samples"]:
        prediction_path = evaluation / "predictions" / entry["config_id"] / f"{entry['resolution']}.npz"
        if not prediction_path.exists():
            continue
        sample = load(root / entry["path"])
        with np.load(prediction_path) as pred:
            pressure = pred["pressure"][-1]
        fig = plt.figure(figsize=(12, 4), layout="constrained")
        values = (sample.pressure[-1], pressure, pressure - sample.pressure[-1])
        titles = ("Reference pressure", "Predicted pressure", "Pressure error")
        maximum = max(float(np.max(np.abs(v))) for v in values[:2])
        for i, (value, title) in enumerate(zip(values, titles, strict=True)):
            axis = fig.add_subplot(1, 3, i + 1, projection="3d")
            cell = value[sample.ops.tris].mean(axis=1)
            limit = maximum if i < 2 else max(float(np.max(np.abs(cell))), 1e-8)
            mesh = Poly3DCollection(sample.ops.points[sample.ops.tris], cmap="coolwarm", linewidth=0)
            mesh.set_array(cell)
            mesh.set_clim(-limit, limit)
            axis.add_collection3d(mesh)
            xyz = sample.ops.points
            axis.auto_scale_xyz(xyz[:, 0], xyz[:, 1], xyz[:, 2])
            axis.set_box_aspect(np.ptp(xyz, axis=0))
            axis.set_title(title)
            axis.set_axis_off()
            fig.colorbar(mesh, ax=axis, shrink=0.6)
        fig.savefig(figures / "pressure_example.png", dpi=150)
        plt.close(fig)
        break
    return figures


def _table_metrics(metrics):
    result = {field: metrics[field]["normalized_rmse"] for field in ("pressure", "velocity")}
    result.update(
        energy=metrics["energy"]["relative_error_max"],
        exact_velocity=metrics["velocity_components"]["exact"]["mass_rmse"],
        harmonic_velocity=metrics["velocity_components"]["harmonic"]["mass_rmse"],
        coexact_velocity=metrics["velocity_components"]["coexact"]["mass_rmse"],
        constant_pressure=metrics["pressure_constant_error"],
    )
    for band, values in metrics["pressure_bands"].items():
        if values["covered"]:
            result[f"pressure_{band}_squared_error"] = values["error_energy"]
            result[f"pressure_{band}_target_energy"] = values["target_energy"]
    return result


def compare_runs(run_directories, output):
    """Seed variability and paired differences; reject mismatched experiment inputs."""
    runs, shared = [], None
    for directory in map(Path, run_directories):
        settings = json.loads((directory / "configuration.json").read_text())
        identity = {
            key: settings[key]
            for key in (
                "fingerprint",
                "scales",
                "width",
                "blocks",
                "slices",
                "lr",
                "accumulation",
                "epochs_requested",
                "selection_score",
                "bands",
            )
        }
        identity["adapter"] = settings.get("adapter", "scalar")
        identity["transport"] = settings.get("transport", "none")
        if shared is not None and shared != identity:
            raise ValueError("comparison requires matched data, architecture, field scales and budgets")
        shared = identity
        history = json.loads((directory / "history.json").read_text())
        if len(history) != settings["epochs_requested"]:
            raise ValueError("comparison requires completed training budgets")
        rows = json.loads((directory / "evaluation" / "per_sample.json").read_text())
        groups = defaultdict(list)
        for row in rows:
            if row["method"] == "neural":
                groups[(row["split"], row["resolution"], row["queries"])].append(row)
        for key, items in groups.items():
            metrics = [_table_metrics(row["metrics"]) for row in items]
            names = set.intersection(*(set(m) for m in metrics))
            runs.append(
                {
                    "objective": settings["objective"],
                    "seed": settings["seed"],
                    "split": key[0],
                    "resolution": key[1],
                    "queries": key[2],
                    **{field: float(np.mean([m[field] for m in metrics])) for field in names},
                }
            )
    grouped = defaultdict(dict)
    for row in runs:
        key = (row["split"], row["resolution"], row["queries"])
        pair = (row["objective"], row["seed"])
        if pair in grouped[key]:
            raise ValueError("duplicate objective/seed run")
        grouped[key][pair] = row
    summary = []
    for key, paired in grouped.items():
        for objective in sorted({obj for obj, _ in paired}):
            seeds = sorted(seed for obj, seed in paired if obj == objective)
            entry = {
                "split": key[0],
                "resolution": key[1],
                "queries": key[2],
                "objective": objective,
                "seeds": seeds,
                "reported_comparison_ready": (
                    len(seeds) >= 3
                    and all((obj, seed) in paired for obj in ("mse", "mass", "spectral") for seed in seeds)
                ),
            }
            fields = set.intersection(*(set(paired[(objective, seed)]) for seed in seeds))
            for field in sorted(fields - {"objective", "seed", "split", "resolution", "queries"}):
                values = [paired[(objective, seed)][field] for seed in seeds]
                differences = [
                    paired[(objective, seed)][field] - paired[("mass", seed)][field]
                    for seed in seeds
                    if ("mass", seed) in paired
                ]
                entry[field] = {
                    "mean": float(np.mean(values)),
                    "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "paired_minus_mass": differences,
                    "paired_mean": float(np.mean(differences)) if differences else None,
                }
            summary.append(entry)
    atomic_json(output, {"matched_settings": shared, "runs": runs, "summary": summary})
    return summary
