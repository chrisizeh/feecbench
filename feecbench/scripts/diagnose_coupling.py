#!/usr/bin/env python3
"""Inspect fixed-query train/validation fits and jointly overfit both coupling directions."""

import argparse
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "reports" / "mpl-cache"))

from feecbench.surface.coupling import inspect_checkpoint, overfit_coupling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "overfit", "both"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective", choices=("mass", "balanced"), default="mass")
    parser.add_argument("--energy-floor", type=float, default=1e-6)
    parser.add_argument("--adapter", choices=("source", "scalar", "vector_features"), default="source")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--final-lr-ratio", type=float, default=0.01)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty; use a new directory")
    if args.stage in ("inspect", "both"):
        inspect_checkpoint(args.root, args.run, args.output / "checkpoint_fit.json", args.device, args.queries)
    if args.stage in ("overfit", "both"):
        report = overfit_coupling(
            args.root,
            args.run,
            args.output / "overfit",
            args.device,
            args.steps,
            args.queries,
            args.seed,
            objective=args.objective,
            energy_floor=args.energy_floor,
            adapter=None if args.adapter == "source" else args.adapter,
            lr_schedule=args.lr_schedule,
            final_lr_ratio=args.final_lr_ratio,
        )
        print("Best-selected coupling fit below 10%:", report["coupling_fit_below_10_percent"])
        print("Final-budget coupling fit below 10%:", report["final_coupling_fit_below_10_percent"])
    print("Diagnostics:", args.output)


if __name__ == "__main__":
    main()
