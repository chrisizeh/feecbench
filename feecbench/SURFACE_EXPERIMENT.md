# Closed-surface acoustic experiment

The `feecbench.surface` package implements the first experiment in the repository's
`plan.md`. It is separate from the existing planar package interfaces. The
scientific pilot is gated: implementing and testing the pipeline does not establish
that all proposed mesh/frequency ranges are adequately resolved.

## Extended transport experiment

The new runner reuses the complete existing corpus and runs a fixed factorial study:

| Axis | Values |
| --- | --- |
| Architecture | Existing attention baseline; attention plus D0/delta1 transport |
| Objective | Balanced MSE; consistent mass loss |
| Seed | 0, 1, 2, paired across architecture and objective |
| Budget | Snapshots at 200 and 400 epochs, each with best and final checkpoints |
| Optimizer | AdamW, constant LR 0.001, accumulation four, unchanged sampler |
| Model | Scalar adapter, width 128, four blocks, 32 slices |

There are **12 fresh runs**, not 24: each 400-epoch run continues its own 200-epoch
state. With 512 training configurations, each epoch has 128 optimizer updates;
the full suite totals **614,400 updates**. Existing historical runs are read-only
context, not automatically resumed or substituted for these matched fresh runs.
The primary comparison is transport versus attention at fixed loss/budget/seed;
secondary comparisons examine extra budget and mass versus balanced MSE. Ordinary
MSE's known imbalance is not rerun. Constant LR keeps scheduling out of this comparison.
The new transport has 814,722 parameters versus 681,602; this is not a capacity-
or wall-time-matched ablation. No GPU runtime estimate is assumed.

**Launch on the GPU machine containing the full HDF5 corpus.** From the repository
root (the wrapper changes into the package directory):

```bash
bash feecbench/scripts/run_transport_study.sh prepare
bash feecbench/scripts/run_transport_study.sh run --device cuda --stage suite
```

Run the second command in tmux. Defaults use the package-relative source
`reports/acoustic/generalization-g16/data` and destination
`reports/acoustic/transport-study-g16`. An exported `analysis.json` alone is not a
corpus. Preparation checks/hash-binds the manifest, all sample files, training-only
scales and balanced-MSE weights; it starts no generation/training. Files in the
source corpus are not changed. For a different location or budget, prepare a new
output directory and supply that same output to every subsequent command:

```bash
bash feecbench/scripts/run_transport_study.sh prepare \
  --root /absolute/path/to/generalization-g16/data \
  --output reports/acoustic/transport-custom --first-budget 200 --epochs 400
```

For a smaller first launch, replace `--stage suite` with `--stage pilot`. It runs
balanced MSE, seed 0, both architectures, through the first budget only. The suite
then reuses those stages and finishes the remaining budgets/runs. A preflight
exercises active transport and backward on one training mesh; CUDA runs also compare
CPU/GPU outputs and parameter gradients. A failure stops before a long training job.
It is not exhaustive validation of every mesh or a runtime forecast.

**Resume and review.** Rerun the identical `run` command after interruption; no
`--resume` flag is needed. Training resumes committed epochs, evaluation repairs
missing/corrupt prediction caches, and completed training milestones are skipped.
An OS lock prevents concurrent writers. Data/code/lockfile/setting changes are
rejected; retain the prepared checkout or prepare another study. The snapshots
under `runs/<transport>-<loss>-seedN/epoch-200` and `epoch-400` are independent of
the mutable latest training checkpoint. Damaged earlier snapshots after training
has advanced require restoring those original artifacts, not silently retraining.

```bash
bash feecbench/scripts/run_transport_study.sh summarize
bash feecbench/scripts/run_transport_study.sh export
```

The default suite evaluates **training/validation only**, including fixed-query
source-removal coupling checks and full regular/off-grid validation. `analysis.json`
contains paired per-seed transport, loss and budget differences; mean/median/p90
field errors; spectral energies; exact/harmonic/coexact errors; energy and closedness;
checkpoint roles and resources. Errors are compared at fixed configuration/query/mesh
groups; seed standard deviation is not a confidence interval. Harmonic/coexact
summary errors are absolute, avoiding division by zero target components. Raw
checkpoint records use one-based `checkpoint_epoch`: a 400-epoch best selected at
epoch 200 or earlier is inherited from the first budget, so compare its final-role
result before concluding that the additional training helped. Raw
predictions, per-configuration time curves and residual reports remain in each
evaluation directory. Residuals cover the first four eligible mesh samples (one in
smoke), not the full corpus or a stratified sample. Timing reports distinguish
epoch time including validation, last training invocation time, whole-trajectory
inference time excluding preprocessing, and last-invocation CUDA peak allocation.

`report.zip` packages the specification, summaries, histories, fit diagnostics,
residuals and resource records for review; it omits HDF5 data, checkpoints and raw
predictions. Timestamped launcher logs live under `reports/logs/`.

**Held-out evaluation is a separate explicit stage**, available only after every
validation run and both budgets/roles are complete:

```bash
bash feecbench/scripts/run_transport_study.sh run --device cuda --stage heldout
bash feecbench/scripts/run_transport_study.sh export
```

It scores all prepared variants/budgets/roles on available ID, geometry OOD and
genus-3 partitions, including coarse/fine/remeshed ID variants already in the source
corpus. It neither regenerates data nor selects new settings. These previously
inspected test sets remain exploratory; they are not fresh confirmatory evidence.
Reference-resolution gates, unseen-family/coefficient/long-time tests, spectral
training, capacity-matched controls and full Transformer–TNO remain separate work.

For a local orchestration smoke with an existing small corpus, use a new output:

```bash
bash feecbench/scripts/run_transport_study.sh prepare --smoke \
  --root reports/acoustic/campaign-smoke-v2/data \
  --output reports/acoustic/transport-study-smoke
bash feecbench/scripts/run_transport_study.sh run --device cpu \
  --output reports/acoustic/transport-study-smoke
```

Smoke keeps all 12 combinations but uses width 16, one block, four slices and
budgets 1/2 epochs. It tests orchestration, not scientific performance. Create a
small campaign corpus first if that example source path is not present.
Executed locally: all 12 smoke runs, 48 validation and 48 held-out checkpoint-role
records, both query types and all four mesh variants completed. The full acceptance
suite passed 111 tests with three hardware/backend skips; Ruff passed. Recovery,
immutable milestones, source checks and paired-summary calculations have regression
coverage. The full-size GPU experiment remains to be run on the user's GPU host.

## Typed transport and current results (2026-11-08)

The exported full campaign and subsequent balanced-MSE study supersede the historical
"no GPU comparison" status below. Three-seed regular-query validation mass-normalized
pressure/velocity errors are 31.11%/92.11% (MSE), 28.61%/39.69% (mass), and
27.60%/40.36% (balanced MSE). Balancing recovers most of the aggregate velocity benefit;
it does not establish superiority of FEEC loss or typed transport. All those models
used the scalar attention adapter **without differential transport**. Balanced MSE
has only training/validation evaluation. The original grid-16 campaign's 24-case
audit has maximum low/middle/velocity spatial-plus-geometry differences of
0.94%/4.18%/4.99%; these sampled differences do not freeze a scientific protocol.
See `docs/acoustic_progress.tex` for the concise report, source locations and limitations.

`--transport d0_delta1` now adds explicit vertex-to-edge D0 and edge-to-vertex
`M0^-1 D0.T M1` messages before each attention block. The consistent sparse M0
solve is differentiable; canonical edge densities are converted to/from integral
cochains around the operators. See decision 24 for the precise convention and solver
tolerances. The existing baseline remains `--transport none`. Shared parameters
have identical initialization; the new residual projections start at zero. Transport
adds 133,120 parameters at the standard size. Conservation and full orientation-
equivariant latent learning are not enforced by this canonicalization-based adapter.

Run the small end-to-end check from the package directory:

```bash
bash scripts/run_acoustic_baseline.sh smoke --device cpu --transport d0_delta1 --objective mse_balanced
```

For the next full experiment, reuse the original corpus and create fresh runs for
seeds 0, 1 and 2 with the same objective and update budget as the baseline:

```bash
bash scripts/run_acoustic_baseline.sh train --device cuda --transport d0_delta1 \
  --objective mse_balanced --seed 0 --epochs 200 \
  --root reports/acoustic/generalization-g16/data \
  --output reports/acoustic/transport-g16/mse_balanced-seed0
uv run --locked --extra surface python scripts/diagnose_coupling.py inspect \
  --device cuda --root reports/acoustic/generalization-g16/data \
  --run reports/acoustic/transport-g16/mse_balanced-seed0 \
  --output reports/acoustic/transport-g16/fit-seed0
```

These commands train and inspect train/validation fits; they do not launch a new
held-out evaluation. The two-case diagnostic also accepts `--transport d0_delta1`
for a fresh transport fit or `--transport none` for its attention control; its default
restores the source architecture. New independent campaigns can select transport
at `prepare --transport d0_delta1 --output NEW_DIRECTORY`; their existing suite
remains mass/MSE (plus spectral only with a frozen protocol). Do not resume an old
campaign after changing source code or architecture. Loss-only comparison utilities
reject mixed transports so an architecture change cannot masquerade as a loss effect.

CPU verification covers operator agreement, adjoint gradients, active orientation
flips, both coupling-direction overfits, checkpoint/resume and evaluation. Full
GPU transport training, CUDA agreement, efficiency and generalization remain unmeasured.

## Executed validation (2026-11-04)

- Original planar acceptance suite: 40 tests pass.
- Combined acceptance suite: 100 passed; one CUDA comparison skipped on this Mac.
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

### Prepared exploratory generalization campaign

This launcher prepares an immutable experiment specification, runs an independent
numerical audit, generates the full held-out partitions, trains, and evaluates.
It does not create protocol attestations. The default is an **exploratory** mass
loss run at seed 0, with the scalar adapter and the original constant learning
rate. It does not use the balanced/cosine two-case diagnostic objective.

Run from the package directory (`feecbench/feecbench` in the Linux checkout):

```bash
bash scripts/run_acoustic_campaign.sh prepare
nohup bash scripts/run_acoustic_campaign.sh run --device cuda > reports/acoustic/generalization-g16/launcher.log 2>&1 &
tail -f reports/acoustic/generalization-g16/launcher.log
```

Preparation is cheap and starts no generation or training. The default specification
has grid 16, `dt=0.002`, `T=1`, 51 regular snapshots, 64 pressure modes, 512 training
configurations, and 64 each for validation, ID test, geometry OOD and genus-3 OOD.
The 64 ID test configurations also have coarse, fine and independent remeshed
variants: **960 HDF5 samples** in the main corpus. Training is on medium meshes
only, for **200 epochs / 25,600 optimizer updates** with accumulation four.
Each new run starts from its own fixed seed. All four IC families are included.

Before main-corpus generation, a separate 24-configuration audit covers ID,
geometry OOD and genus 3, all four families and both familiar genera. Its geometry
IDs are disjoint from every main partition. It computes timestep-halving and
fine-grid differences using the same numerical settings. This is a finite sample,
not exhaustive coverage of the IC parameter range (the audit exact-velocity cases
use zero initial pressure). The runner records large numerical discrepancies
without treating them as model failures or manufacturing a passed protocol gate.
Inspect these errors alongside learning results, especially on shifted shapes.

Every run saves zero/persistence/modal comparisons, regular and off-grid results,
per-configuration time curves, physical/spectral/Hodge diagnostics, raw predicted
fields and fixed-query train/validation fitting diagnostics. PDE residuals use
the existing small subset (one evaluation sample), not every test case. Mean,
median and p90 configuration errors remain separated by family, genus, split,
resolution and query type; seed variability is added when multiple seeds exist.
The comparison is within `T=1`, with fixed `c=1` and the same four IC families.
It does not test long-time extrapolation, unseen families or new coefficients.
The genus-3 shift also changes geometry; it is not an isolated topology effect.

After examining training fit and ID controls, expand the **same fixed campaign**:

```bash
nohup bash scripts/run_acoustic_campaign.sh run --device cuda --stage suite >> reports/acoustic/generalization-g16/launcher.log 2>&1 &
```

This reuses mass/seed-0 and runs the remaining five mass/MSE seed pairs (seeds 0,
1, 2). It never changes the budget or selects settings from held-out performance.
Poor scores are preserved for analysis rather than causing automatic retuning.
For a nine-run MSE/mass/spectral campaign, prepare a **new directory** using
`--output ... --protocol reports/protocol.json` with an actual frozen protocol.
The protocol's numerical settings and eligible pressure bands are applied to
every objective. The exploratory audit cannot substitute for that protocol.

Useful outputs under `reports/acoustic/generalization-g16/`:

- `campaign.json`: settings, manifests, optimizer update budget, source and lockfile hashes.
- `audit/reference_uncertainty.json`: numerical differences on the independent audit cases.
- `progress.json` and `executions.json`: stage completion and execution environments.
- `runs/mass-seed0/fit.json`: comparable train/validation family scores at fixed queries.
- `runs/*/evaluation/`: per-sample metrics, physics diagnostics and raw `.npz` predictions.
- `analysis.json`: completed-run records, family/control tables and seed variability.

Print a compact medium-mesh comparison after any completed run:

```bash
bash scripts/run_acoustic_campaign.sh summarize
```

Rerun the same `run` command after interruption; no `--resume` flag is needed.
Generation resumes at samples, training at committed epochs, evaluation at completed
cases. Checkpoint hashes and prediction hashes prevent stale evaluation reuse.
The latest checkpoint embeds the best model and repairs best/history files after
interrupted writes. A process lock prevents concurrent writers. Logs are also
kept under `reports/logs/`. Settings, source and lockfile changes are rejected on
resume; retain the original checkout or prepare another directory for changes.
Set a different epoch budget only with `prepare --epochs N --output NEW_DIRECTORY`.

For a CPU plumbing check, use `prepare --smoke --output reports/acoustic/campaign-smoke`
then `run --device cpu --output reports/acoustic/campaign-smoke`. Its shortened
horizon and tiny model/budget are not scientific performance evidence.
The local six-run smoke suite completed its independent 12-configuration audit,
24-configuration campaign (36 mesh samples), training, all evaluation partitions,
raw predictions, and analysis. Regression tests cover interrupted final-epoch
checkpoint writes, damaged prediction-cache recovery and changed-checkpoint
rejection. These checks do not establish a GPU runtime or performance result.

### Field-balanced MSE: reuse the six completed controls

The separate loss study tests global pressure/velocity weighting before adding
spectral losses. It references a completed six-run campaign read-only and trains
three fresh `mse_balanced` models, using the same seeds, scalar architecture,
normalization, constant learning rate, accumulation, epoch budget and validation
mass checkpoint-selection score. For the full campaign this is 200 epochs and
25,600 updates per model. It generates no new corpus and evaluates only training
fits and validation cases. Existing ID and OOD results do not select the weights.

From the package directory on the machine containing the full campaign:

```bash
bash scripts/run_loss_balance.sh prepare
bash scripts/run_loss_balance.sh run --device cuda
```

Run the second command inside tmux for an unattended job. Both commands save
timestamped logs under `reports/logs/`. The default source is
`reports/acoustic/generalization-g16`, and the new study is saved separately in
`reports/acoustic/loss-balance-g16`. Override these during preparation with
`--source-campaign PATH --output NEW_DIRECTORY`; use that same output for run and
summarize. Preparation needs the full source campaign, its HDF5 files and all six
completed run directories, not just the exported `analysis.json`.

For each field, calculate `E_f`, the equal-configuration mean of squared normalized
target coefficients, averaging equally over positive regular times and coefficients
within each mesh. Use only medium-resolution training samples. Set
`d_f = max(E_f, 1e-6)` and
`a = (E_p + E_u) / (E_p / d_p + E_u / d_u)`. The new objective is
`a * (mean(ep**2) / d_p + mean(eu**2) / d_u)`.

When neither floor is active, the two fields have equal expected contributions
for zero prediction. The common multiplier preserves the original MSE's expected
zero-prediction total, including when a floor is active. This calibration does not
match the initial model loss, gradients, clipping or optimizer trajectories.
Weights are fixed globally, without per-case or per-family normalization. The
existing physical scales and raw-field evaluation remain unchanged. This objective
is distinct from `--objective balanced` in the two-case coupling diagnostic.

Preparation verifies the parent campaign hash, all six completed artifact chains,
matching budgets/settings/scales and the fixed-query fitting rule. The new study
records the original sources and hashes of its training/validation data and control
artifacts, plus its own current source identity. The old campaign's source guard
remains unchanged: importing its completed evidence is not resuming it under new
code. Source artifacts are never rewritten. Changed parent data or study settings
are rejected on resume. The study output must not overlap the parent directory.

Rerun the same `run` command after interruption. All three seeds resume automatically,
with checkpoint-bound, validation-only prediction caches. Print the paired results:

```bash
bash scripts/run_loss_balance.sh summarize
```

The new `analysis.json` contains:

- Fixed calibration energies, floors, weights and training-configuration provenance.
- `fit_comparison`: identical three-query train/validation family scores for the
  three objectives, per-seed values, seed variability and paired differences from
  each historical control. Negative differences indicate lower error.
- `validation_evaluation`: regular/off-grid metrics by family, genus and method;
  these use their own query sets and are separate from the fixed-query fits.
- Per-epoch unweighted/weighted field MSEs and pre-clipping gradient statistics for
  the new models. Historical gradient diagnostics are explicitly unavailable.

The criterion is whether balanced MSE improves velocity over MSE and approaches
mass-loss accuracy, with any pressure trade-off reported. A negative result is still
valid evidence. No Fourier term, architecture change or new OOD evaluation is added.
The pre-existing `spectral` objective remains separately gated by a frozen protocol.
The three-run CPU loss-study smoke and completed-run resume both passed. Replaying
all six original CPU smoke controls under the updated implementation reproduced
their model tensors, settings identities and validation scores exactly. These are
compatibility/plumbing checks, not evidence that balanced MSE improves full-run fit.

### Paired extra-budget continuation

Continue the three completed balanced-MSE seeds with two schedules each. Both
branches start from the same original `latest.pt`, preserving AdamW state, RNG
state, history and the embedded best model. The default adds 200 epochs / 25,600
optimizer updates per branch: 400 epochs / 51,200 updates including the parent.
One branch keeps LR 0.001; the other applies cosine decay toward 0.00001 over the
additional updates. The model, objective, field weights and query sampler stay fixed.

On the machine containing the complete loss study and its source campaign, from
the package directory:

```bash
bash scripts/run_loss_continuation.sh prepare
bash scripts/run_loss_continuation.sh run --device cuda
```

The source defaults to `reports/acoustic/loss-balance-g16`; the six new runs go to
`reports/acoustic/loss-continuation-g16`. Preparation accepts `--source-study PATH`,
`--output NEW_DIRECTORY` and `--additional-epochs N`. Use the same output for run
and summarize. Preparation needs all three latest checkpoints and complete source
artifacts, not just the exported analysis. It verifies and snapshots their hashes;
the source directories are read-only inputs. Earlier studies retain their source
guards. The new configuration fixes its own source identity and continuation budget.

Run inside tmux; the wrapper saves timestamped logs under `reports/logs/`. Rerunning
the identical command resumes automatically. Cosine progress is checkpointed per
epoch together with optimizer state, and interruption recovery repeats any
uncommitted epoch. Changing the prepared budget or source code is rejected.

```bash
bash scripts/run_loss_continuation.sh summarize
```

The resulting `analysis.json` contains the parent histories and latest-checkpoint
fits, each continuation's history and LR trajectory, and paired per-seed constant
versus cosine differences. It reports two checkpoint roles separately:

- **Best:** the lowest validation mass score over the entire history, including
  the parent. `selected_from_parent` identifies an inherited best checkpoint.
- **Final:** the checkpoint after the full additional budget, regardless of its
  validation score.

Both roles receive the same three-query train/validation family diagnostics and
full validation regular/off-grid evaluation. No OOD evaluation is run. Negative
cosine-minus-constant differences indicate lower error; compare both roles to avoid
mistaking an inherited best for successful continuation. LR steps occur after each
optimizer update, so the final stored next LR is the decay endpoint, not the last
LR actually used. Histories record both.

Tests verify exact constant-continuation equivalence to uninterrupted CPU training,
exact interrupted/resumed cosine state, interrupted initial checkpoint recovery,
and initial-fit provenance. A six-branch CPU smoke checks orchestration; it does not
establish a scientific advantage for either schedule.

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
  --runs reports/acoustic/run-mse-seed* reports/acoustic/run-mass-seed* reports/acoustic/run-spectral-seed* --output reports/acoustic/comparison.json
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

### Longer fits and optional directional features

The 500-update failure is not sufficient evidence of an architecture limitation.
Extending the same grid-12 balanced scalar experiment to 1,500 updates gave final
source-response errors 0.0904 (pressure to velocity) and 0.0993 (velocity to
pressure), with zero-input errors 0.0185 and 0.0076. Actual cross-field errors also
passed the 10% diagnostic criterion. This is one seed on two fitting configurations,
not a generalization result or a guarantee for the grid-16 corpus.

`--lr-schedule cosine` optionally decays the diagnostic learning rate from the
source run's rate toward `--final-lr-ratio` times that rate (default 0.01).
Scheduling occurs after each update: `next_learning_rate` in history and
`final_learning_rate` in the report describe the rate scheduled for the next
update. The default remains constant. A comparison that changes both budget and
schedule cannot identify the effect of scheduling alone. This option does not
alter the main benchmark training loop.

A local grid-12 scalar run with 2,000 updates and cosine decay reached final
response errors 0.0630 and 0.0441, actual cross-field errors 0.0624 and 0.0441,
and zero-input errors 0.0105 and 0.0012. Both best-selected and final-budget
coupling criteria passed. The same-field velocity error on the exact-velocity
case was still 0.1765, so this is not a claim that every field error is below 10%.
The run took about 300 seconds on CPU; its ignored report is
`reports/acoustic/coupling-scalar-cosine-2000/overfit/report.json`.

To check the same scalar model with a longer, decaying-rate fitting budget on
the grid-16 corpus, use a fresh output directory:

```bash
uv run --locked --extra surface python scripts/diagnose_coupling.py overfit --device cuda --root reports/acoustic/development-g16 --run reports/acoustic/dev-g16-mass-seed0 --output reports/acoustic/coupling-g16-cosine-2000 --steps 2000 --objective balanced --adapter scalar --lr-schedule cosine
```

The optional `--adapter vector_features` appends three edge-direction projection
components `(canonical_u0 / (length * scaleu)) * canonical_unit_tangent` to the
existing scalar edge features. It is neither a full velocity reconstruction nor a
rotation-equivariant architecture. For pure pressure inputs these additional
features vanish. All shared weights and the RNG state match the scalar adapter
at initialization; the added columns start at zero, adding `3 * width` parameters
(384 at width 128). No incidence or stiffness operator enters the forward pass.

At the same 1,500-update grid-12 budget, this variant gave response errors 0.0690
and 0.0883, but failed the joint diagnostic criterion because its actual
pressure-to-velocity error remained above 10%. It is an experimental comparison,
not an established fix. Keep `scalar` for the next optimization check.

The diagnostic defaults to the source checkpoint's adapter; `--adapter scalar`
or `--adapter vector_features` explicitly selects a fresh fitting model.
Benchmark training also supports `--adapter vector_features` with separate default
output paths. Evaluation restores the saved adapter, resume rejects a changed
adapter, and loss comparisons reject mixed adapters. Existing scalar checkpoint
identities and resume behavior are preserved. Keep comparison globs specific to
one adapter. Local reports are ignored under
`reports/acoustic/coupling-{scalar,vector_features}-1500`.
