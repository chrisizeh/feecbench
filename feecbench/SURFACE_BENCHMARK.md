# Closed-surface acoustic benchmark

The surface benchmark provides geometry, consistent Whitney operators, acoustic
reference solutions, HDF5 samples, disjoint manifests, training-only normalization,
and raw-unit evaluation. Models and experiment orchestration live on the
`experiment/acoustic-baseline` branch and consume these interfaces.

Install from this directory with `uv sync --extra surface --extra dev`.
Use the pinned PyDEC revision in the lockfile. The planar API remains unchanged.

## Generate data

```sh
uv run --extra surface python scripts/surface_benchmark.py manifest \
  --counts '{"train":4,"validation":4}' --seed 102 --development \
  --output reports/development-manifest.json
uv run --extra surface python scripts/surface_benchmark.py generate \
  --manifest reports/development-manifest.json --root reports/development-data \
  --grid 12 --dt 0.01 --snapshots 3 --horizon 0.1 --modes 16
```

This is development data, not a scientifically validated release. Generation is
resumable and rejects changed settings for an existing corpus. Use `--paired` for
paired mesh resolutions. Preserve the manifest and index with published data.
Use `surface.data.training_scales` for normalization; it selects training-medium
samples only. Never normalize evaluation targets using held-out statistics.

## Evaluate predictions

Save an NPZ containing `times`, `pressure` and `velocity`: arrays must follow the
sample's complete time ordering, vertex ordering and low-to-high edge orientation.
Pressure and velocity must be in raw physical units, with shapes matching the
sample arrays. Evaluation excludes time zero.

```sh
uv run --extra surface python scripts/surface_benchmark.py score \
  --sample path/to/sample.h5 --predictions predictions.npz --output scores.json
```

Supply `--protocol protocol.json` to use frozen spectral thresholds. The output
records sample and prediction SHA256 hashes and the protocol hash. A single-sample
score is not a full benchmark submission. For corpus evaluation use
`surface.metrics.score` per sample and `surface.metrics.aggregate` with configuration,
split, resolution, query subset, genus and initial-condition family labels.
Record the benchmark Git revision, corpus fingerprint, protocol hash, model
configuration and seed alongside experiment results.

## Numerical validation

`scripts/validate_surface.py` runs independent numerical checks; MFEM has a separate
validation environment and optional compiled bridge. `scripts/check_surface_protocol.py`
checks reference uncertainty and freezes a protocol from reviewed, hashed evidence.
The existing gates, including learning-smoke evidence supplied by an external model,
are preserved. Freezing does not itself establish scientific validity.

Generated corpora, predictions and results belong outside version control. Distribute
released corpora through artifact hosting with checksums and a persistent identifier.
Model checkpoints, optimizer choices, loss balancing and experiment reports are not
part of the benchmark package.


## Shared planar and surface workflow

Use `load_sample(path)`, `FEECDataset(root, split="train")`,
`collate_feec(items)`, and `evaluate_profile(predictions, sample)` for either family.
Surface predictions use `p` and `u`. The evaluator selects the surface acoustic
profile from the sample PDE; `profile="planar"` explicitly selects original metrics,
and `profile="surface_acoustic"` retains the complete surface score structure.
Surface geometry, full consistent mass solves and midpoint dynamics stay in `surface/`.

Shared writes use schema 0.2.0 and atomic replacement; readers retain planar 0.1.0
and surface-0.2.0 compatibility. Legacy corpora and indexes are adapted in memory.
Legacy sample identities use `<config_id>-<resolution>` unless a sample ID is stored.
Remeshed surfaces are medium resolution, with refinement tracked independently.
Fingerprint indexes cannot be replaced by directory scans, and incomplete indexes fail.
New surface generation uses generator contract 3 and rejects older resume settings.

`compute_stats(samples, policy="surface_energy")` fits training-medium regular-query
energy per area with equal configuration weights and no centering. Use RMS mode.
The original `compute_stats(samples)` and field-keyed statistics remain supported;
`policy="star_weighted"` explicitly filters training data. `load_full()` stays raw.
Batch compatible field schemas and query shapes together; query coordinates, operator
types and auxiliary spectral arrays remain available per sample.

### Portable migration regressions

`tests/fixtures/unification/` contains compact, frozen planar 0.1.0 and
surface-0.2.0 samples captured from revision `21b97a7`. `capture.json` records
settings, the full source revision, and SHA-256 checksums. Normal tests read these
files and generate round-trip working files only under pytest temporary directories.
Missing or changed references fail tests. Numerical metrics use `rtol=1e-10` and
`atol=1e-12` to allow floating-point solver variation; stored arrays and CSR
representations are checked exactly. These tolerances are substantially below the
perturbation magnitude (0.03) and the harmonic solver tolerance (1e-7).

To reproduce the capture in a **new** directory using the supported surface/dev
environment (substitute absolute paths for the variables):

```sh
git worktree add --detach /tmp/feecbench-legacy 21b97a7
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=/tmp/feecbench-legacy/feecbench/src python "$FIX_WORKTREE/feecbench/scripts/capture_unification_baseline.py" "$NEW_BASELINE"
cd "$FIX_WORKTREE/feecbench"
PYTHONPATH=src python -m pytest tests/test_unified_api.py tests/test_unified_review.py -q
```

Capture verifies the imported checkout's revision and tracked source contents,
and refuses any existing destination. Do not replace committed references during
normal testing. The capture script lives in the fix/current checkout but must
import the pinned legacy checkout through `PYTHONPATH`.

For `profile="planar"` on surfaces, shared final-time Hodge metrics use an explicit
`harmonic` basis first, otherwise `auxiliary["harmonic_basis"]`. A missing,
nonfinite, or incorrectly shaped surface basis raises a specific error; callers
may supply a valid basis or disable Hodge metrics with `include_hodge=False`.
This does not change the default surface acoustic metric definitions.
Legacy index normalization uses index metadata only; excluded unavailable files
are never opened during dataset construction. Split labels and fingerprints are
preserved, and `get_config` still resolves against the full index.
