#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
mkdir -p "$PACKAGE_DIR/reports/logs"
LOG="$PACKAGE_DIR/reports/logs/campaign-$(date +%Y%m%d-%H%M%S)-$$.log"
cd "$PACKAGE_DIR"
uv run --locked --project "$PACKAGE_DIR" --extra surface \
    python "$PACKAGE_DIR/scripts/acoustic_campaign.py" "$@" 2>&1 | tee "$LOG"
