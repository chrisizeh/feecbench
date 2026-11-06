#!/usr/bin/env python3
"""CLI for development smoke, gated pilot runs, training and raw-unit evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "reports" / "mpl-cache"))

from feecbench.surface.corpus import generate_corpus
from feecbench.surface.data import make_manifest, training_scales
from feecbench.surface.protocol import read_protocol
from feecbench.surface.training import device_for, evaluate, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "development", "pilot", "generate", "train", "evaluate"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective", choices=("mse", "mse_balanced", "mass", "spectral"), default="mass")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--grid", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--snapshots", type=int)
    parser.add_argument("--modes", type=int)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument(
        "--adapter",
        choices=("scalar", "vector_features"),
        default="scalar",
        help="training adapter; evaluation restores the checkpoint adapter",
    )
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("epochs must be positive")
    device_for(args.device)  # Fail before generation if explicit CUDA cannot be honored.
    smoke = args.stage == "smoke"
    base = Path(__file__).resolve().parents[1] / "reports" / "acoustic"
    root = args.root or base / (
        "smoke-data" if smoke else "development-data" if args.stage == "development" else "pilot-data"
    )
    adapter_suffix = "" if args.adapter == "scalar" else f"-{args.adapter}"
    output = args.output or base / f"{'smoke' if smoke else 'run'}{adapter_suffix}-{args.objective}-seed{args.seed}"
    protocol = read_protocol(args.protocol) if args.protocol else None
    if args.stage == "pilot" and protocol is None:
        parser.error("pilot requires --protocol from completed numerical/protocol gates; use development first")
    if args.objective == "spectral" and protocol is None:
        parser.error("spectral training requires a frozen --protocol with resolved bands")
    if args.objective == "spectral" and not any(b["resolved"] for b in protocol["bands"]):
        parser.error("no pressure bands passed the reference-resolution gate")
    if args.stage in ("smoke", "development", "pilot", "generate"):
        if args.manifest:
            manifest = json.loads(args.manifest.read_text())
        elif smoke:
            manifest = make_manifest({"train": 8, "validation": 4, "id_test": 4}, seed=101, development=True)
        elif args.stage == "development":
            manifest = make_manifest({"train": 16, "validation": 8, "verification": 8}, seed=102, development=True)
        else:
            manifest = make_manifest(
                {"train": 512, "validation": 64, "id_test": 64, "geometry_ood": 64, "topology_ood": 64}, seed=202
            )
        numerical = {
            "grid": args.grid if args.grid is not None else (12 if smoke else 16),
            "dt": args.dt if args.dt is not None else (0.01 if smoke else 0.002),
            "snapshots": args.snapshots if args.snapshots is not None else (6 if smoke else 51),
            "modes": args.modes if args.modes is not None else 32,
            "horizon": 1.0,
        }
        if args.stage == "pilot":
            frozen = protocol["numerical_settings"]
            for key in ("grid", "dt", "snapshots", "modes"):
                requested = getattr(args, key)
                if requested is not None and requested != frozen[key]:
                    parser.error(f"{key} differs from the validated protocol; revalidate before changing it")
                numerical[key] = frozen[key]
            numerical["horizon"] = frozen["horizon"]
            if any(
                c["config_id"] in protocol["development_configurations"]
                or c["geometry_id"] in protocol["development_geometries"]
                for c in manifest["configurations"]
            ):
                parser.error("pilot configurations/geometries overlap numerical development")
        generate_corpus(root, manifest, **numerical, paired=args.paired or args.stage == "pilot")
    if args.stage in ("generate", "development"):
        print(f"Corpus: {root}")
        return
    bands = None
    if protocol:
        index = json.loads((root / "index.json").read_text())
        scales = training_scales([root / e["path"] for e in index["samples"]])
        bands = [{**b, "scale": b["scale_raw"] / scales["pressure"]} for b in protocol["bands"]]
    if args.stage in ("smoke", "pilot", "train"):
        train(
            root,
            output,
            device=args.device,
            seed=args.seed,
            objective=args.objective,
            epochs=args.epochs if args.epochs is not None else (3 if smoke else 200),
            width=32 if smoke else 128,
            blocks=1 if smoke else 4,
            slices=8 if smoke else 32,
            resume=args.resume,
            bands=bands,
            adapter=args.adapter,
        )
    if args.stage in ("smoke", "pilot", "evaluate"):
        evaluate(root, output / "best.pt", output / "evaluation", device=args.device)
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
