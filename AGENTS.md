# Repository Guidelines

## Project Structure & Module Organization

The installable Python package lives in `feecbench/`, separate from the repository root.
- `feecbench/src/feecbench/`: schema, acoustic oracle, sparse operators, HDF5 I/O, dataset loading, normalization, evaluation, and visualization; `mesh/` contains meshing and refinement.
- `feecbench/tests/`: acceptance and regression tests, with shared fixtures in `conftest.py`.
- `feecbench/scripts/`: pilot-corpus generation and cost profiling.
- `feecbench_demo.py` and `feecbench_demo.ipynb`: end-to-end reference demo.
- `feecbench/DECISIONS.md`: numerical conventions and architectural rationale.

Generated corpora, figures, and reports under `feecbench/demo/` and `feecbench/reports/` are ignored; keep generated outputs out of commits.

## Build, Test, and Development Commands

Create and activate a virtual environment, then run from the repository root:

```bash
python -m pip install -e './feecbench[dev]'
cd feecbench
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m ruff check .                  # lint package, scripts, and tests
python -m pytest -q                    # full acceptance suite
python scripts/build_pilot.py          # generate the 84-sample pilot corpus
python scripts/profile_pilot.py        # generate cost-model reports
```

Meshing requires working `gmsh`; Linux installations may need `libGLU.so.1`. Set thread limits before importing NumPy to avoid slow sparse-kernel execution.

## Coding Style & Naming Conventions

Target Python 3.10+, use four-space indentation, and respect Ruff's 120-character line limit. Use `snake_case` for functions/modules, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Follow existing type annotations and dataclass-based schema blocks. Ruff checks imports, style, unused code, modernization, and common bugs; no separate formatter is configured.

## Testing Guidelines

Name files `test_*.py` and functions `test_*`. Reuse deterministic oracle fixtures and temporary directories for generated data. Run focused checks with `python -m pytest tests/test_io.py -q` from `feecbench/`, then the full suite before submitting. No coverage percentage is configured. Add regression tests for changed behavior, especially exact HDF5 round trips, conservation, convergence, splits, and batching; keep the golden-path test CPU-compatible.

## Numerical Contracts

Preserve low-to-high edge orientation, counterclockwise triangles, and exact `d1 @ d0 == 0`. Compute normalization statistics from training data only; evaluate raw fields. Consult `DECISIONS.md` before changing these contracts.

## Commit & Pull Request Guidelines

History currently contains only `feecbench 0.1.0`; no recurring commit convention is established. Use concise, descriptive commit subjects. PRs should explain the change, relevant numerical or schema implications, and validation commands/results. Link related issues when applicable and include plots for visualization changes.
