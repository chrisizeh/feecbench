#!/usr/bin/env python3
"""Generate a surface corpus or score raw-unit predictions without a model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

from feecbench import evaluate_profile, load_sample
from feecbench.surface.corpus import generate_corpus
from feecbench.surface.data import atomic_json, load, make_manifest
from feecbench.surface.protocol import read_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    generate = stages.add_parser("generate")
    generate.add_argument("--root", type=Path, required=True)
    generate.add_argument("--manifest", type=Path, required=True)
    generate.add_argument("--grid", type=int, default=16)
    generate.add_argument("--dt", type=float, default=0.002)
    generate.add_argument("--snapshots", type=int, default=51)
    generate.add_argument("--modes", type=int, default=32)
    generate.add_argument("--horizon", type=float, default=1.0)
    generate.add_argument("--paired", action="store_true")
    manifest = stages.add_parser("manifest")
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--counts", required=True, help='JSON split counts, e.g. {"train": 4, "validation": 4}')
    manifest.add_argument("--seed", type=int, default=202)
    manifest.add_argument("--development", action="store_true")
    evaluate = stages.add_parser("score")
    evaluate.add_argument("--sample", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    if args.stage == "manifest":
        atomic_json(args.output, make_manifest(json.loads(args.counts), args.seed, args.development))
    elif args.stage == "generate":
        generate_corpus(args.root, json.loads(args.manifest.read_text()), grid=args.grid, dt=args.dt,
                        snapshots=args.snapshots, modes=args.modes, horizon=args.horizon, paired=args.paired)
    else:
        import hashlib

        sample = load(args.sample)
        protocol = read_protocol(args.protocol) if args.protocol else None
        with np.load(args.predictions, allow_pickle=False) as prediction:
            if not np.array_equal(prediction["times"], sample.times):
                raise ValueError("prediction times must exactly match sample times")
            metrics = evaluate_profile({"p": prediction["pressure"], "u": prediction["velocity"]},
                                       load_sample(args.sample), profile="surface_acoustic",
                            thresholds=protocol["thresholds"] if protocol else (10.0, 30.0))
        atomic_json(args.output, {
            "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
            "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
            "protocol_hash": protocol["hash"] if protocol else None,
            "metadata": sample.metadata, "metrics": metrics,
        })


if __name__ == "__main__":
    main()
