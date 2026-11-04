#!/usr/bin/env python3
"""Prepare or run a fixed exploratory acoustic generalization campaign."""

import argparse
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "reports" / "mpl-cache"))

from feecbench.surface.campaign import prepare_campaign, read_campaign, run_campaign, summarize_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--output", type=Path, default=Path("reports/acoustic/generalization-g16"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--stage", choices=("baseline", "suite"), default="baseline")
    parser.add_argument("--epochs", type=int, help="prepare only; default 200, or 1 for smoke")
    parser.add_argument("--smoke", action="store_true", help="prepare only; tiny CPU-compatible plumbing check")
    parser.add_argument("--protocol", type=Path, help="prepare only; genuine frozen protocol enables spectral runs")
    args = parser.parse_args()
    if args.command != "prepare" and (args.epochs is not None or args.smoke or args.protocol):
        parser.error("settings are fixed at prepare; use a separate directory for a different campaign")
    if args.command == "prepare":
        config = prepare_campaign(args.output, epochs=args.epochs, smoke=args.smoke, protocol_path=args.protocol)
        print(f"Prepared {config['status']}: {args.output}")
        print(f"Per run: {config['training']['epochs']} epochs, {config['updates_per_run']} optimizer updates")
        print(f"Baseline: mass seed 0. Suite: {len(config['seeds']) * len(config['objectives'])} runs total.")
        print("Reference audit reports numerical uncertainty; it does not approve scientific protocol gates.")
    elif args.command == "run":
        # OS lock is released on exit/crash. No stale PID files or competing writers.
        import fcntl

        read_campaign(args.output)
        with (args.output / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("another process is running this campaign")
            run_campaign(args.output, args.device, args.stage)
    else:
        report = summarize_campaign(args.output)
        print(f"Completed runs: {len(report['runs'])}; analysis: {args.output / 'analysis.json'}")
        for row in report["summary"]:
            if row["family"] == "all" and row["resolution"] == "medium" and row["queries"] == "regular":
                print(
                    f"{row['objective']:8s} seeds={row['seeds']} {row['split']:13s} {row['method']:11s} "
                    f"p={row['pressure']['mean']:.4f} u={row['velocity']['mean']:.4f}"
                )


if __name__ == "__main__":
    main()
