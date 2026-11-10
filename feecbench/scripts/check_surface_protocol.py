#!/usr/bin/env python3
"""Measure reference uncertainty or freeze reviewed protocol evidence."""

import argparse
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "reports" / "mpl-cache"))

from feecbench.surface.protocol import freeze_protocol
from feecbench.surface.reference_checks import check_references

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=("check", "freeze"))
parser.add_argument("--root", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--evidence")
parser.add_argument("--limit", type=int)
parser.add_argument("--tolerance", type=float, default=0.05)
parser.add_argument("--thresholds", type=float, nargs=2, default=(10.0, 30.0))
args = parser.parse_args()
if args.stage == "check":
    check_references(args.root, args.output, thresholds=args.thresholds, limit=args.limit, tolerance=args.tolerance)
else:
    if not args.evidence:
        parser.error("freeze requires --evidence naming reviewed, hashed reports for every gate")
    freeze_protocol(args.root, args.evidence, args.output, thresholds=args.thresholds)
