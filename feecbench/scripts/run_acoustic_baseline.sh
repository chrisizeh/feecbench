#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PACKAGE_DIR/reports/mpl-cache}"
mkdir -p "$PACKAGE_DIR/reports/logs"
LOG="$PACKAGE_DIR/reports/logs/acoustic-$(date +%Y%m%d-%H%M%S)-$$.log"
uv run --locked --project "$PACKAGE_DIR" --extra surface --extra dev \
    python "$PACKAGE_DIR/scripts/acoustic_baseline.py" "$@" 2>&1 | tee "$LOG"
