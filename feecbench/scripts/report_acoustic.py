#!/usr/bin/env python3
"""Create plots or a matched paired-seed comparison from offline results."""

import argparse
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "reports" / "mpl-cache"))

from feecbench.surface.report import compare_runs, plot_run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=("plot", "compare"))
parser.add_argument("--root")
parser.add_argument("--evaluation")
parser.add_argument("--runs", nargs="+")
parser.add_argument("--output")
args = parser.parse_args()
if args.stage == "plot":
    if not args.root or not args.evaluation:
        parser.error("plot requires --root and --evaluation")
    print(plot_run(args.root, args.evaluation))
else:
    if not args.runs or not args.output:
        parser.error("compare requires --runs and --output")
    compare_runs(args.runs, args.output)
