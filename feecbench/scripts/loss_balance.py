#!/usr/bin/env python3
"""Prepare, run or summarize the train/validation-only field-balanced MSE study."""

import argparse
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "reports" / "mpl-cache"))

from feecbench.surface.loss_balance import prepare_study, read_study, run_study, summarize_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--source-campaign", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/acoustic/loss-balance-g16"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.source_campaign is not None and args.command != "prepare":
        parser.error("source campaign is fixed at prepare")
    if args.command == "prepare":
        specification = prepare_study(args.source_campaign or Path("reports/acoustic/generalization-g16"), args.output)
        snapshot = specification["parent_snapshot"]
        print(f"Prepared {args.output}: three fresh balanced-MSE runs; six completed controls reused read-only.")
        print(f"Per run: {snapshot['training']['epochs']} epochs, {snapshot['updates_per_run']} optimizer updates")
        print(f"Fixed weights: {snapshot['field_balance']['weights']}")
        print("Training/validation only. No generation or OOD evaluation.")
    elif args.command == "run":
        import fcntl

        read_study(args.output, verify_parent=False)
        with (args.output / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("another process is running this study")
            run_study(args.output, args.device)
    else:
        report = summarize_study(args.output)
        for row in report["fit_comparison"]:
            print(
                f"{row['objective']:12s} {row['split']:10s} {row['family']:16s} seeds={row['seeds']} "
                f"p={row['pressure']['mean']:.4f} u={row['velocity']['mean']:.4f}"
            )
        print(f"Analysis: {args.output / 'analysis.json'}")


if __name__ == "__main__":
    main()
