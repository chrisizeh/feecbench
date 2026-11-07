#!/usr/bin/env python3
"""Prepare/run paired constant and cosine continuations of a completed loss study."""

import argparse
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "reports" / "mpl-cache"))

from feecbench.surface.continuation import (
    prepare_continuation,
    read_continuation,
    run_continuation,
    summarize_continuation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--source-study", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/acoustic/loss-continuation-g16"))
    parser.add_argument("--additional-epochs", type=int, help="prepare only; default 200")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.command != "prepare" and (args.source_study is not None or args.additional_epochs is not None):
        parser.error("source study and continuation budget are fixed at prepare")
    if args.command == "prepare":
        config = prepare_continuation(
            args.source_study or Path("reports/acoustic/loss-balance-g16"),
            args.output,
            additional_epochs=args.additional_epochs if args.additional_epochs is not None else 200,
        )
        seed = config["seeds"][0]
        print(f"Prepared {args.output}: 3 seeds x constant/cosine; source latest checkpoints preserved.")
        print(
            f"Each branch adds {config['additional_epochs']} epochs / "
            f"{config['additional_epochs'] * seed['updates_per_epoch']} updates."
        )
        print(
            f"Learning rate: constant {config['training']['lr']:g}; "
            f"cosine to {config['training']['lr'] * config['cosine_final_lr_ratio']:g}."
        )
        print("Selected-best and final-budget training/validation diagnostics; no OOD evaluation.")
    elif args.command == "run":
        import fcntl

        read_continuation(args.output, verify_sources=False)
        with (args.output / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("another process is running this continuation")
            run_continuation(args.output, args.device)
    else:
        report = summarize_continuation(args.output)
        for run in report["runs"]:
            print(
                f"{run['schedule']:8s} seed={run['seed']} best={run['best']['validation_mass_score']:.6g} "
                f"final={run['final']['validation_mass_score']:.6g} "
                f"inherited_best={run['best']['selected_from_parent']}"
            )
        print(f"Analysis: {args.output / 'analysis.json'}")


if __name__ == "__main__":
    main()
