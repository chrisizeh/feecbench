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
