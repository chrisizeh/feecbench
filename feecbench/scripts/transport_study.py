#!/usr/bin/env python3
"""Prepare and run matched attention/transport experiments on an existing corpus."""

import argparse
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "reports" / "mpl-cache"))

from feecbench.surface.transport_study import (
    export_transport_study,
    prepare_transport_study,
    read_transport_study,
    run_transport_study,
    summarize_transport_study,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize", "export"))
    parser.add_argument("--root", type=Path, help="prepare only; existing corpus directory")
    parser.add_argument("--output", type=Path, default=Path("reports/acoustic/transport-study-g16"))
    parser.add_argument("--epochs", type=int, help="prepare only; final budget, default 400")
    parser.add_argument("--first-budget", type=int, help="prepare only; intermediate budget, default 200")
    parser.add_argument("--smoke", action="store_true", help="prepare only; tiny architecture, default budgets 1/2")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--stage", choices=("pilot", "suite", "heldout"), default="suite")
    args = parser.parse_args()
    if args.command != "prepare" and any((args.root is not None, args.epochs is not None,
                                          args.first_budget is not None, args.smoke)):
        parser.error("data, budgets and architecture are fixed at prepare; use a new directory for changes")
    if args.command == "prepare":
        root = args.root or Path("reports/acoustic/generalization-g16/data")
        if not (root / "index.json").exists() or not (root / "manifest.json").exists():
            parser.error(f"complete corpus required at {root}; an exported analysis.json is insufficient")
        config = prepare_transport_study(
            root, args.output, epochs=args.epochs if args.epochs is not None else (2 if args.smoke else 400),
            first_budget=args.first_budget if args.first_budget is not None else (1 if args.smoke else 200),
            smoke=args.smoke,
        )
        print(f"Prepared {args.output}: 2 architectures x 2 losses x 3 seeds = 12 fresh runs.")
        print(f"Budgets: {config['budgets']} epochs; {config['updates_per_epoch']} updates/epoch.")
        print(f"Total suite budget: {12 * config['budgets'][-1] * config['updates_per_epoch']:,} updates.")
        print("Constant LR; best/final checkpoints at each budget; training/validation only in pilot/suite.")
        print("Held-out evaluation requires an explicit run --stage heldout after the full suite.")
        print("Preparation hashes existing data; it does not generate a corpus or start training.")
    elif args.command == "run":
        import fcntl

        if not (args.output / "study.json").exists():
            parser.error("run prepare first, using the machine containing the complete corpus")
        read_transport_study(args.output, verify_data=False)
        with (args.output / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("another process is running this transport study")
            run_transport_study(args.output, args.device, args.stage)
    elif args.command == "export":
        print(f"Review bundle: {export_transport_study(args.output)}")
    else:
        report = summarize_transport_study(args.output)
        print(f"Validation records: {report['completed_validation_records']}/{report['expected_validation_records']}")
        for row in report["evaluation"]:
            if (row["method"], row["genus"], row["family"], row["resolution"], row["queries"]) == (
                "neural", "all", "all", "medium", "regular"
            ):
                print(f"{row['transport']:9s} {row['objective']:12s} budget={row['budget']} {row['role']:5s} "
                      f"{row['split']:13s} seeds={row['seeds']} "
                      f"p={row['metrics']['pressure']['mean']:.4f} u={row['metrics']['velocity']['mean']:.4f}")
        print(f"Analysis: {args.output / 'analysis.json'}")


if __name__ == "__main__":
    main()
