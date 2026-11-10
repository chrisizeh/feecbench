#!/usr/bin/env bash
# Build only the bridge, against an existing serial MFEM 4.8 installation.
# Usage: bash scripts/build_mfem_bridge.sh /path/to/mfem/prefix [/path/to/output]
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mfem_prefix="${1:?Pass the MFEM installation prefix containing include/ and lib/}"
output="${2:-$repo_root/reports/bin/mfem_surface_assembly}"
mkdir -p "$(dirname "$output")"
"${CXX:-c++}" -std=c++17 -O2 -I"$mfem_prefix/include" \
    "$repo_root/scripts/mfem_surface_assembly.cpp" \
    -L"$mfem_prefix/lib" -lmfem "-Wl,-rpath,$mfem_prefix/lib" -o "$output"
printf 'Built %s\nSet FEECBENCH_MFEM_EXECUTABLE to this path when running validate_surface.py.\n' "$output"
