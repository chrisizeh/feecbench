# Closed-surface acoustic experiment

The `feecbench.surface` package implements the first experiment in the repository's
`plan.md`. It is separate from the existing planar package interfaces. The
scientific pilot is gated: implementing and testing the pipeline does not establish
that all proposed mesh/frequency ranges are adequately resolved.

## Executed validation (2026-11-04)

- Original planar acceptance suite: 40 tests pass.
- Combined acceptance suite: 78 passed; one CUDA comparison skipped on this Mac.
- Independent MFEM 4.8 assembly and independently stepped trajectories: all 36 cases
  passed (3 genera × 2 geometries × 3 resolutions × 2 ICs), plus pressure-only and
  stationary-harmonic checks. Maximum relative matrix discrepancy: `6.14e-14`;
  maximum normalized trajectory discrepancy: `1.26e-13`.
- Midpoint temporal orders: 1.999, 2.000, 2.000. A separate analytical unit-sphere
  spherical-harmonic case has decreasing spatial/geometric error, with observed
  orders 1.82–1.89. Backend agreement and continuum evidence are separate reports.
- CPU smoke: generation, three training epochs, best/latest checkpoints, regular
  and off-grid evaluation, modal/persistence/zero controls, and plots completed.
  Separate tiny-overfit tests pass for both pressure-driven and velocity-driven
  dynamics. Resume reproduces uninterrupted CPU training exactly at epoch boundaries.
- A 32-configuration development corpus was generated: 16 training, 8 validation,
  8 genus-3 numerical verification. With grid 12, 64 modes, `dt=0.002`, `T=1`, and
  51 regular snapshots, medium meshes have 676–1960 vertices, occupy 107 MB total,
  and took about 62 seconds of recorded generation work on this machine.
  Maximum reference energy drift: `1.18e-14`. Timings are measurements, not GPU estimates.
- Complete development uncertainty check: maximum genus-1/2 pressure discrepancies
  against the finer projected reference were 2.04% (low band) and 6.51% (middle
  band), with a 5% provisional gate and a 0.01 absolute scale floor. **Only the low
  band passes.** The largest velocity discrepancy was 9.87%. Time-refinement
  errors were smaller. These are fine-grid differences, not continuum error bounds.

Generated evidence is ignored under `reports/`: `surface_validation_mfem.json`,
`development_reference_uncertainty.json`, and `acoustic/`. No scientific protocol
has been frozen and no full 768-configuration corpus or GPU comparison has run.
Refine the development meshes/pulse ranges and review velocity error, pulse
resolution, horizon, and remesh transfer before freezing a protocol. The present
minimum pulse-width/maximum-edge ratio is 1.75; the regular output grid provides
at least 57 samples per evaluated modal period. These measurements inform that
review; they do not automatically establish adequate handle exploration.

## Environment and commands

Run from `feecbench/`, or use repository-relative paths to the launchers. The shell
launcher resolves its own location and can be invoked from any directory.

```bash
uv sync --locked --extra surface --extra dev
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --locked --extra surface --extra dev python -m ruff check .
uv run --locked --extra surface --extra dev python -m pytest -q
bash scripts/run_acoustic_baseline.sh smoke --device cpu
bash scripts/run_acoustic_baseline.sh smoke --device cpu --epochs 6 --resume
bash scripts/run_acoustic_baseline.sh development --grid 12 --modes 64
uv run --locked --extra surface python scripts/check_surface_protocol.py check \
  --root reports/acoustic/development-data \
  --output reports/development_reference_uncertainty.json
```

Python 3.12 is selected by `.python-version`; `uv.lock` pins the environment.
`surface` installs **hirani/PyDEC** at revision
`c181202c70e79794dcc8fa4577537a4eb4987b2a`, never the unrelated PyPI package.
NumPy 1.26.4 is used on Python 3.12. PyDEC's upstream scalar conversion fails on
newer NumPy releases; Python 3.13 is constrained below NumPy 2.3. The tested 3.12
OpenBLAS environment also avoids spurious floating-point flags observed with the
Mac Accelerate NumPy 2.2 wheel. Only that known upstream deprecation is filtered in
tests; numerical warnings are not globally suppressed.

A root directory is bound to its manifest and numerical-settings hash. Generation
resumes existing matching files, writes each HDF5 sample atomically, and rejects
mismatches. Use a new `--root` when changing numerical settings. Training rejects
partial indexes and existing runs without `--resume`; resumption checks settings,
data identity, optimizer state and RNG state. `--epochs` is the total epoch budget.
An explicit `--device cuda` fails before data generation if CUDA is unavailable.

### MFEM validation

The independent Python backend uses MFEM's `H1_FECollection(1,2)` and
`ND_FECollection(1,2)` with ambient dimension 3. MFEM assembles its own gradient,
masses and diffusion matrix; signed DoF maps align its bases. The comparison stepper
solves the full coupled weak block system, independently of the oracle's reduced
pressure solve. Matrix reports include Frobenius, maximum-entry and zero-entry checks.

On supported Linux machines, use the separate locked environment:

```bash
uv run --locked --project validation python scripts/validate_surface.py
```

MFEM's Python wheel is unavailable on this arm64 Mac, and upstream SWIG bindings
failed to build. The executed validation instead used the included native C++
bridge against the same independently built MFEM 4.8 library:

```bash
bash scripts/build_mfem_bridge.sh /path/to/mfem/install
export FEECBENCH_MFEM_EXECUTABLE="$PWD/reports/bin/mfem_surface_assembly"
uv run --locked --extra surface python scripts/validate_surface.py \
  --output reports/surface_validation_mfem.json
uv run --locked --extra surface --extra dev python -m pytest tests/test_surface_mfem.py -q
```

The bridge selects MFEM's own assembly; it is not an alternative hand-coded FEEC
assembler. Neither backend nor any compiled binary is vendored. Without either
MFEM route, optional integration tests skip and the full validation command fails
clearly. `--convergence-only` runs numerical convergence evidence without MFEM.

### Protocol freeze and GPU comparison

`check_surface_protocol.py freeze` accepts a reviewed evidence JSON. It requires
entries named `mfem_36_cases`, `spatial_convergence`, `temporal_convergence`,
`paired_initial_conditions`, `pulse_resolution`, `time_sampling`, `orientation`,
`learning_smoke`, and `reference_uncertainty`. Each entry has this form:

```json
{"passed": true, "report": "relative/report.json", "sha256": "SHA256_OF_REPORT_BYTES"}
```

These entries are explicit scientific-review attestations, not an automatic claim
that arbitrary report contents prove a gate. Report paths are relative to the
evidence file. The reference-uncertainty entry must additionally name a complete
`check` report matching the corpus fingerprint, numerical settings and physical
thresholds. Its measured `resolved_bands` decide band eligibility. Hashes preserve
provenance; they are not authentication. Threshold choices must use genus 1–2 only.

```bash
uv run --locked --extra surface python scripts/check_surface_protocol.py freeze \
  --root reports/acoustic/development-data --evidence reports/reviewed-evidence.json \
  --output reports/protocol.json
bash scripts/run_acoustic_baseline.sh pilot --device cuda --seed 0 \
  --protocol reports/protocol.json
bash scripts/run_acoustic_comparison.sh --device cuda --protocol reports/protocol.json
```

The pilot inherits frozen grid, timestep, horizon, output count and mode budget;
conflicting overrides and development geometry/configuration overlap are rejected.
The comparison runs MSE, mass loss, and mass-plus-spectral loss with three paired
seeds, common field scales, architecture and training budgets. All checkpoints use
the same validation mass score. An unresolved middle band is excluded rather than
silently trained. Epoch-boundary resume is supported; CPU/GPU agreement remains a
hardware-dependent acceptance test.

```bash
uv run --locked --extra surface python scripts/report_acoustic.py plot \
  --root reports/acoustic/smoke-data \
  --evaluation reports/acoustic/smoke-mass-seed0/evaluation
uv run --locked --extra surface python scripts/report_acoustic.py compare \
  --runs reports/acoustic/run-*-seed* --output reports/acoustic/comparison.json
```

Reports retain per-configuration/time errors, ordinary cochain MSE, normalized and
absolute mass errors, constant pressure, physical spectral bands, total energy,
mean pressure, exact/harmonic/coexact velocity, closedness, harmonic drift and
finite-difference PDE residuals (including reference residuals at both step sizes).
Regular and off-grid queries are separate. Comparison tables include seed
variability and paired differences; incompatible settings or incomplete budgets
are rejected. Predictions are saved in raw units for independent CPU analysis.
Plots show time/resolution errors and reference/predicted/error pressure fields.

## Numerical and data conventions

- Geometry is the smooth double of a perforated ellipse:
  `z²/thickness² = (1-x²/a²-y²/b²) product_i((r_i²-radius²)/(r_i²+radius²))`.
  Disjoint interior holes give genus 1–3 without intersecting target sheets.
  Marching-cubes extraction is projected onto the zero set; short-edge collapse
  enforces the link condition and preserves local orientation. Every result passes
  closedness, connectedness, vertex-link, orientation, Euler and chain checks.
- Area scale comes from one fixed finer quadrature mesh, independent of requested
  resolution. Coarse/medium/fine are projected subdivisions of a common base;
  curved projection is **not** claimed to give exactly nested FE spaces. Independent
  remeshes use a separate extraction grid and phase.
- Low-to-high edges; consistently oriented surface triangles, without a global
  planar CCW claim. Full consistent Whitney `M0`, `M1`; top-form **integral** mass
  `M2=diag(1/area)`. Derived operators apply sparse mass solves.
- `M0 p_dot=D0.T M1 u`, `u_dot=-D0 p`, `c=1`. Midpoint reuses bounded pressure
  factorizations. Off-grid outputs take a remainder step from the regular solver
  grid without perturbing later regular outputs.
- Harmonics satisfy both `D1 h=0` and `D0.T M1 h=0`; dimension `2g` is checked.
  A lumped auxiliary metric is used only to find this common kernel, never in
  acoustic dynamics. Projections are invariant under rotations of the harmonic basis.
- Pressure pulses use intrinsic edge-graph distance and heat smoothing on the
  physical reference mesh. Potentials and mixed pressures use recorded reference
  eigenmodes. One reference scalar field is sampled on all paired meshes; independent
  eigenvector coordinates on different meshes are never treated as the same IC.
- Harmonic ICs use wrapped planar-hole angle cochains and removal of their exact
  component. Coefficients are fixed using reference dynamic energy. This varies a
  documented **g-dimensional subspace of the full 2g harmonic space**. Complete
  harmonic coverage is an extension, not a present result. Paired exact-velocity
  transfer commutes with `D0` through scalar potentials; no general Whitney
  edge-transfer operator is claimed or used on arbitrary model predictions.
- `surface-0.2.0` HDF5 stores geometry, full CSR operators, reference/query fields,
  times, spectra, harmonics, configuration/geometry identities, splits, provenance,
  accuracy diagnostics and timings. Legacy `0.1.0` planar files remain unchanged.
  Normalization streams medium training data only, uses mass energy divided by
  physical area, preserves pressure mean, and never shifts edge cochains.
- The model has separate vertex/edge encoders and heads around shared slice attention,
  following the [Transolver idea](https://github.com/thuml/Transolver). It is an
  independently implemented adaptation, not the full Transformer–TNO architecture.
  Inputs are recorded in each run. Edge inputs and outputs are canonicalized with
  length-aware cochain scaling. Operators enter reference generation, losses,
  diagnostics and the modal numerical control, not learned forward transport.
- The modal control retains both pressure and exact-velocity coefficients and full
  harmonic velocity. Omitted pressure/exact content is reported; out-of-corpus
  coexact initial content is discarded and separately reported. It is a numerical
  control for known linear dynamics, not another learned baseline.

## Remaining scientific work

Review/freeze numerical ranges after refining the unresolved middle-band and
velocity cases; validate independent-remesh transfer at the frozen settings;
choose the propagation horizon from geometric/modal evidence; then generate the
full corpus and run the three-seed GPU comparison. GPU runtime/memory, topology
OOD learning performance and loss-variant conclusions have not been measured.
The generation implementation is serial and bounded by one physical configuration;
parallel generation scheduling is a future performance improvement. No Slurm is
required. Boundary physics, variable coefficients, general commuting transfers,
full harmonic IC coverage and additional architectures remain outside this run.

## Diagnosing weak pressure–velocity coupling

A small overall pressure error can coexist with failure to generate pressure from
initial velocity or velocity from initial pressure. Before extending a stalled run,
inspect identical fixed positive query times on both the training and validation
configurations. The diagnostic also reports mean-free pressure error and removes
the sole initial source field in pure-source cases. Its cross-field prediction
change is compared with the exact corresponding oracle change, including amplitude
and sign. Correct dynamics give response error zero and response RMS ratio one;
a model insensitive to the source has response error one and response ratio zero.

From the package directory, after completing development training:

```bash
uv run --locked --extra surface python scripts/diagnose_coupling.py both --device cuda --root reports/acoustic/development-g16 --run reports/acoustic/dev-g16-mass-seed0 --output reports/acoustic/coupling-g16 --steps 500
```

Use `--device cpu` without CUDA. `inspect` runs only checkpoint diagnostics;
`overfit` runs only the controlled fitting experiment. Choose a fresh output
directory for every invocation. Original data and training checkpoints are read-only.

The overfit experiment initializes a fresh model with the source run's architecture,
learning rate and fixed training-derived scales. It jointly fits one training pulse
with `u0=0` and one training exact-velocity case with `p0=0`, at three fixed regular
times. On each of the same meshes it also trains the exact all-zero solution.
Thus, fitting by recognizing geometry alone cannot pass the source-response test.
The default objective is explicitly the original mass loss, regardless of the source run's
objective. The optional balanced diagnostic described below changes only the fitting
weights; neither diagnostic changes the model or the benchmark training objectives.
Every step is an optimizer update, unlike the main loop's accumulated epochs.

`checkpoint_fit.json` contains comparable train/validation errors and source-removal
responses. `overfit/report.json` records all fixed-query scores, zero-input errors,
selected configuration identities, checkpoint provenance, and the best diagnostic
fit. Actual cross-field, source-response, and zero-input errors below 10% in both directions define a
**diagnostic criterion**, not a scientific protocol gate. Success establishes
finite-sample fitting only; it does not establish generalization or correct dynamics
for arbitrary initial conditions. Failure calls for inspecting optimization and the
adapter before expanding the corpus or reporting a loss comparison.

### Controlled weighting diagnostic

If the original-loss overfit fails, repeat the **same** finite fitting problem with
`--objective balanced`, using the same source run, seed, query count and update
budget, and a new output directory:

```bash
uv run --locked --extra surface python scripts/diagnose_coupling.py overfit --device cuda --root reports/acoustic/development-g16 --run reports/acoustic/dev-g16-mass-seed0 --output reports/acoustic/coupling-g16-balanced --steps 500 --objective balanced
```

The balanced objective divides each configuration/field's squared mass error per
area by its fixed mean target energy over the selected nonzero training queries.
The energy is measured after the original shared field scaling. A documented
`--energy-floor` (default `1e-6` in these normalized squared units) bounds weights.
Each zero-input control **inherits its paired nonzero case's denominators**; its
zero targets never determine a new denominator. Targets, model inputs, architecture,
shared scales, optimizer settings and initialization remain the same as in the
mass diagnostic. With no floors active, all-zero predictions have balanced score
one after averaging the two source cases and their zero controls.

Reports preserve physical and normalized energies, denominators, floor activation,
weights, pre-clipping gradient norms and clipping frequency. Weight changes also
change gradient magnitude and clipping, so improvement indicates sensitivity to
weighting/optimization; it does not uniquely establish the cause of a failure.
Compare `final_cases` after the **same number of updates** across objectives.
`cases` contains the separately selected best checkpoint and may come from an
earlier step. Both outcomes are printed and saved.

This is a diagnostic-only relative loss on two fitting configurations. It is not
silently added as a fourth benchmark objective, and a successful fit still does
not demonstrate generalization to new amplitudes, signs, or spatial initial
conditions. The original full-corpus MSE/mass/spectral comparison is unchanged.

In a local grid-12 matched CPU check (fresh 128-wide, four-block model, three query
times, seed 0, 500 updates), final source-response errors were:

| Coupling direction | Original mass | Balanced diagnostic |
| --- | ---: | ---: |
| Pressure to velocity | 0.9461 | 0.2459 |
| Velocity to pressure | 0.2534 | 0.4540 |

Neither fit met the 10% diagnostic criterion. Gradient clipping was active on
93.8% and 95.4% of updates respectively. The effect is a tradeoff: weighting helps
the weak pressure-to-velocity response while slowing the other direction. This
supports sensitivity to weighting/optimization, not a completed model fix or a
claim about grid-16 GPU results. The experiment used the existing grid-12 corpus;
repeat with the user's own corpus and source-run configuration for a matched local
comparison. Reports are ignored under `reports/acoustic/coupling-{mass,balanced}-500`.
