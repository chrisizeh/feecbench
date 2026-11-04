#!/usr/bin/env bash
# Same architecture, budgets, data and seed pairing; validation score is fixed.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for argument in "$@"; do
    case "$argument" in
        --seed|--seed=*|--objective|--objective=*|--output|--output=*)
            printf 'The comparison controls seed, objective and output per run; use the single-run launcher for overrides.\n' >&2
            exit 2
            ;;
    esac
done
for seed in 0 1 2; do
    for objective in mse mass spectral; do
        bash "$SCRIPT_DIR/run_acoustic_baseline.sh" pilot --seed "$seed" --objective "$objective" "$@"
    done
done
