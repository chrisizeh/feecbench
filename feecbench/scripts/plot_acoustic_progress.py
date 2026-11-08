#!/usr/bin/env python3
"""Build report figures from exported results, without rerunning experiments.

Run from any directory. Inputs default to repository-root reports/acoustic;
generated figures and their provenance remain under the ignored package reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO / "feecbench/reports/mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OBJECTIVES = ("mse", "mass", "mse_balanced")
LABELS = ("Ordinary MSE", "Mass-weighted", "Balanced MSE")
COLORS = ("#a34f37", "#377da5", "#237a69")
FAMILIES = ("pulse", "two_pulses", "pulse_harmonic", "exact_velocity")
FAMILY_LABELS = ("One pulse", "Two pulses", "Pulse + harmonic", "Exact velocity")


def unique(rows, **criteria):
    matches = [r for r in rows if all(r[k] == v for k, v in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {criteria}; found {len(matches)}")
    return matches[0]


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def loss_figure(study, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), layout="constrained", sharey=True)
    values = {}
    for ax, field in zip(axes, ("pressure", "velocity"), strict=True):
        for j, objective in enumerate(OBJECTIVES):
            means = []
            for seed in range(3):
                rows = [r for r in study["validation_evaluation"] if r["objective"] == objective
                        and r["seed"] == seed and r["method"] == "neural" and r["queries"] == "regular"
                        and r["split"] == "validation" and r["resolution"] == "medium"]
                assert len(rows) == 8 and sum(r["configurations"] for r in rows) == 64
                means.append(100 * np.average([r[field]["mean"] for r in rows],
                                             weights=[r["configurations"] for r in rows]))
            values[f"{objective}/{field}"] = means
            ax.plot([j - .09, j, j + .09], means, "o", color=COLORS[j], ms=5)
            ax.plot([j - .2, j + .2], [np.mean(means)] * 2, color=COLORS[j], lw=2)
            ax.text(j, max(means) + 5, f"{np.mean(means):.1f}%", ha="center", weight="bold")
        ax.set(title=field.capitalize(), xticks=range(3), xticklabels=LABELS, ylim=(0, 105))
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Validation trajectory error (%)")
    save(fig, output, "loss_comparison")
    return values


def family_figure(study, output):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), layout="constrained")
    for i, field in enumerate(("pressure", "velocity")):
        for j, split in enumerate(("train", "validation")):
            data = np.array([[100 * unique(study["fit_comparison"], split=split, family=family,
                                          objective=obj)[field]["mean"] for obj in OBJECTIVES]
                             for family in FAMILIES])
            ax = axes[i, j]
            im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=110, aspect="auto")
            ax.set(title=f"{field.capitalize()} | {split.capitalize()}", xticks=range(3),
                   xticklabels=("MSE", "Mass", "Balanced"), yticks=range(4),
                   yticklabels=FAMILY_LABELS if j == 0 else [""] * 4)
            for (row, col), value in np.ndenumerate(data):
                ax.text(col, row, f"{value:.1f}", ha="center", va="center",
                        color="white" if value > 65 else "#17212b", fontsize=11)
    fig.colorbar(im, ax=axes, label="Fixed-query mass-normalized error (%)", shrink=.85, pad=.02)
    save(fig, output, "family_response")


def transfer_figure(campaign, output):
    settings = (("id_test", "medium", "ID medium"), ("geometry_ood", "medium", "Geometry OOD"),
                ("topology_ood", "medium", "Unseen genus 3"), ("id_test", "coarse", "ID coarse"),
                ("id_test", "fine", "ID fine"), ("id_test", "remeshed", "ID remeshed"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained", sharey=True)
    rows = [unique(campaign["summary"], method="neural", objective="mass", queries="regular",
                   family="all", genus="all", split=s, resolution=r) for s, r, _ in settings]
    for ax, field, color in zip(axes, ("pressure", "velocity"), COLORS[1:], strict=True):
        base = 100 * rows[0][field]["mean"]
        for i, row in enumerate(rows):
            value = 100 * row[field]["mean"]
            ax.plot([base, value], [i, i], color=color, lw=2)
            ax.plot(value, i, "o", color=color)
            ax.text(value + .6, i, f"{value:.1f}%" + (f"  (+{value-base:.1f} pp)" if i else ""),
                    va="center", fontsize=9)
        ax.axvline(base, color="#737b83", lw=1, ls="--")
        ax.set(title=field.capitalize(), xlabel="Mass-normalized trajectory error (%)",
               yticks=range(6), yticklabels=[x[2] for x in settings], xlim=(24, 60))
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    save(fig, output, "transfer")


def case_figure(source, output):
    all_rows = []
    for seed in range(3):
        path = source / f"loss-balance-g16/runs/mse_balanced-seed{seed}/evaluation/per_sample.json"
        rows = json.loads(path.read_text())
        all_rows.append([r for r in rows if r["method"] == "neural" and r["queries"] == "regular"
                         and r["split"] == "validation" and r["resolution"] == "medium"])
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), layout="constrained")
    selections = []
    # Rank each family by seed-averaged pressure trajectory error; then illustrate seed 0.
    for col, (family, quantile, label) in enumerate((("pulse", .5, "Typical pulse"),
                                                    ("exact_velocity", .9, "Difficult exact velocity"))):
        ids = sorted(r["config_id"] for r in all_rows[0] if r["family"] == family)
        scored = [(float(np.mean([unique(rows, config_id=cid)["metrics"]["pressure"]["normalized_rmse"]
                                 for rows in all_rows])), cid) for cid in ids]
        scored.sort()
        rank = int(np.ceil(quantile * len(scored))) - 1
        score, cid = scored[rank]
        row = unique(all_rows[0], config_id=cid)
        m = row["metrics"]
        times = m["times"]
        for field, color in zip(("pressure", "velocity"), COLORS[1:], strict=True):
            axes[0, col].plot(times, 100 * np.array(m[field]["curve"]), color=color, label=field.capitalize())
        axes[0, col].set(title=f"{label} | genus {row['genus']}", ylim=(0, 140),
                         ylabel="Normalized field error (%)", xlabel="Query time")
        energy = m["energy"]
        reference = np.array(energy["reference"])
        axes[1, col].plot(times, np.array(energy["predicted"]) / reference[0], color="#a34f37", label="Prediction")
        axes[1, col].plot(times, reference / reference[0], "--", color="#344351", label="Reference")
        axes[1, col].set(xlabel="Query time", ylabel="Total energy / reference at t=0.02", ylim=(0, 1.6))
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
            ax.legend(loc="best", fontsize=8)
        selections.append({"family": family, "config_id": cid, "quantile": quantile,
                           "rank": rank + 1, "family_count": len(scored), "shown_seed": 0,
                           "selection_metric": "three-seed mean pressure trajectory NRMSE",
                           "selection_score": score,
                           "shown_pressure_error": m["pressure"]["normalized_rmse"],
                           "shown_velocity_error": m["velocity"]["normalized_rmse"]})
    save(fig, output, "case_dynamics")
    return selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "reports/acoustic")
    parser.add_argument("--output", type=Path, default=REPO / "feecbench/reports/acoustic-progress/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "pdf.fonttype": 42})
    campaign = json.loads((args.source / "analysis.json").read_text())
    study = json.loads((args.source / "loss-balance-g16/analysis.json").read_text())
    assert study["parent_campaign_hash"] == campaign["campaign_hash"]
    assert {(r["objective"], r["seed"]) for r in study["runs"]} == {
        (obj, seed) for obj in OBJECTIVES for seed in range(3)}
    means = loss_figure(study, args.output)
    family_figure(study, args.output)
    transfer_figure(campaign, args.output)
    cases = case_figure(args.source, args.output)
    paths = [args.source / "analysis.json", args.source / "loss-balance-g16/analysis.json"]
    paths += [args.source / f"loss-balance-g16/runs/mse_balanced-seed{s}/evaluation/per_sample.json"
              for s in range(3)]
    provenance = {"sources": {str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths}, "validation_per_seed_percent": means, "cases": cases,
                  "missing_inputs": "Full-run reference meshes/fields and separate field energies are absent."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
