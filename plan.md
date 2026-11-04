# Closed-Surface FEEC Acoustic Benchmark Plan

## 1. Objective and status

Build a reproducible first experiment for learning acoustic dynamics on closed torus
and multi-torus surfaces. Generate reference data with a lowest-order Whitney FEEC
solver using PyDEC, validate it against MFEM, and run learning experiments through a
shell script on a GPU node.

This document records the agreed direction and proposed implementation details. It
does not imply that the new surface solver, data generator, metrics, or training
scripts already exist. Implement in reviewable commits on
`experiment/acoustic-baseline`; writing this plan does not start implementation.

Context:

- Proposal: `/Users/chrisizeh/Downloads/PDE_Benchmark.pdf`, especially Section 1.3
  (metrics), Section 4 (learning problem), and Sections 5–6 (meshes and experiments).
- The user's clarification fixes the initial scope as **closed surfaces without
  boundaries**, rather than the planar domains in the existing implementation.
- The existing CPU acceptance suite passed: **40 tests**, with two PyTorch sparse
  CSR warnings. This validates the current checkout, not the proposed surface oracle.
- Use `uv` for reproducible Python environments. The eventual GPU job runs from a
  direct terminal through a shell script; no Slurm integration is required.

## 2. Scientific scope

### Initial benchmark

- Connected, closed, orientable two-dimensional surfaces embedded in three dimensions.
- Torus, double torus, and triple torus: genus 1, 2, and 3.
- Intrinsic, linear surface acoustics with constant sound speed `c = 1`.
- Pressure is a scalar 0-form; velocity is a tangential 1-form represented by
  oriented edge integrals at lowest order.
- No boundaries, boundary conditions, forcing, or damping in the initial corpus.
- Train on genus 1 and 2; hold out genus 3 for topology transfer.

The learning task is

`(surface mesh, discrete operators, initial pressure, initial velocity, c, query time)`
`-> (pressure at query time, velocity at query time)`.

The full reference sample retains physical parameters and solver metadata separately
from learner-visible inputs. This preserves a future hidden-physics track without
introducing it into the first experiment.

### Deferred extensions

Variable material coefficients, diffusion/Darcy/Navier–Stokes, hidden constitutive
physics, boundaries, higher-order elements, curved finite elements, adaptive mesh
regimes, and large model comparisons are later work. The existing planar examples
remain useful regression tests but do not substitute for the surface benchmark.

## 3. Repository gaps to address

Audit and extend the existing code rather than treating its naming as a mathematical
classification. It already contains Whitney 1-form Galerkin assembly, but its current
geometry and solver do not implement this planned benchmark.

| Current behavior | Required change |
| --- | --- |
| Planar signed areas and globally counterclockwise triangles | Embedded-surface areas, tangent geometry, and globally consistent surface orientation |
| Betti-number calculation assumes `beta2 = 0` | Closed-surface topology, with expected Betti numbers `(1, 2g, 1)` |
| Lumped pressure mass and existing toy acoustic solver | Explicitly specified Whitney FEEC assembly and a validated reference integrator |
| Zero initial velocity | Support harmonic initial velocity and nonzero circulation |
| Fixed five training configurations per family | Configurable split counts and explicit manifests |
| Loader omits coordinates/connectivity; tensors are float64 | Geometry-aware training adapter with explicit dtype handling |
| Vertex-inclusion map only | Proper degree-specific transfer operators where needed |
| Boundary metrics and no spectral bands | Closed-surface evaluation and spectral diagnostics |

Audit field conventions throughout the schema. In particular, distinguish top-form
integral degrees of freedom from scalar cell values: their mass matrices have
different normalizations. Do not reuse the existing `star2 = triangle areas`
convention blindly for Whitney 2-form cochains.

Update numerical documentation to distinguish consistent orientation on a curved
surface from the planar counterclockwise convention. Keep old tests and formats
working where feasible; version incompatible schema changes explicitly.

## 4. Surface geometry and meshes

Choose and prototype a reproducible generator for genus 1–3 surfaces. A parametric
torus is straightforward; the multi-torus construction and meshing backend remain
an implementation decision. Require smooth target geometry, no self-intersections,
and a known genus. Validate the multi-torus construction before committing to it.

Vary handle thickness, spacing, and shape within each genus. Normalize surface area
to control global scale, while recording all rescaling. Match local geometric
scales and mesh density across splits as closely as possible.

For selected configurations, generate coarse, medium, and fine nested triangulations
of the same target surface, plus independent remeshes. Start with piecewise-flat
triangles in both solver backends. Record refinement/projection provenance and
geometric approximation error; refinement of curved geometry is not automatically
an exact nesting of finite-element function spaces.

Mesh acceptance checks:

- Every edge belongs to exactly two triangles; no boundary edges or nonmanifold vertices.
- One connected component, consistent triangle orientation, and no degenerate elements.
- Euler characteristic `V - E + F = 2 - 2g` and independently checked homology where practical.
- Exact `d1 @ d0 == 0`, expected harmonic dimension `2g`, and valid mass matrices.
- Mesh quality, area, edge-length distribution, aspect ratios, and pulse resolution.

## 5. PyDEC-backed FEEC oracle

Use **Whitney FEEC assembly from `hirani/pydec`**, not its separate diagonal DEC
Hodge-star construction. Pin a tested repository revision: the unrelated package
named `pydec` on PyPI must not be installed by mistake. Verify compatibility with
the selected NumPy/SciPy versions in a small prototype.

PyDEC supplies complexes and finite-element inner products; the project supplies
the acoustic integrator, configuration interface, and dataset serialization.

Initial discretization:

- Continuous piecewise-linear scalar pressure space.
- Lowest-order Whitney edge 1-form velocity space.
- Consistent finite-element mass matrices `M0` and `M1`.
- Incidence-based exterior derivatives with documented ordering and orientation.

For `c = 1`, the semidiscrete acoustic system is

```text
M0 dp/dt = D0.T M1 u
   du/dt = -D0 p
E = 0.5 * (p.T M0 p + u.T M1 u)
```

Use implicit midpoint as the proposed initial reference integrator. Validate its
energy behavior and temporal convergence. Reuse the reduced pressure-system
factorization for a fixed mesh, coefficient, and timestep. Choose timesteps for
accuracy even though the integrator avoids an explicit CFL stability restriction.

Keep generation and reference calculations in float64. Generate data on CPU,
parallelizing independent configurations with bounded memory and controlled BLAS
thread counts. Training remains a separate GPU workload.

## 6. Independent PyDEC–MFEM validation

MFEM is a reference backend for a validation subset, not a requirement for every
data-generation or training run. First establish a working embedded-surface
assembly prototype in both backends; API support has not yet been demonstrated in
this repository.

### A. Equivalent operator assembly

Use identical vertices, triangles, piecewise-flat geometry, physical coefficients,
and equivalent lowest-order finite-element spaces. Independently assemble operators
in each backend. Align vertex/edge ordering, edge signs, and basis normalization.

Compare `M0`, `M1`, `D0`, and `K = D0.T @ M1 @ D0`. Report relative matrix norms and
maximum entrywise discrepancies, with appropriate absolute checks for zero entries.
Equivalent assembly on well-conditioned test meshes should agree near floating-point
accuracy; determine acceptance tolerances from conditioning and measured residuals.

### B. Equivalent trajectories

Use identical initial cochains, coefficients, timesteps, output times, and initially
the same midpoint scheme. Independently implement or invoke the stepping paths so
the test does not merely run one shared solver twice.

Compare pressure/velocity mass-weighted errors over time, energy histories, and
harmonic projections. Comparing arbitrary harmonic eigenvector coordinates is
invalid because basis signs and rotations are not unique.

Start with one small torus, then use:

`3 genera × 2 geometries × 3 resolutions × 2 IC families = 36 paired trajectories`.

The two IC families are a pressure pulse and a pulse plus harmonic velocity.

### C. Accuracy beyond agreement

Two implementations can agree while sharing discretization error. Also require:

- Spatial refinement and geometric convergence studies.
- Temporal refinement with spatial error controlled.
- Analytical or manufactured verification cases, separated from the unforced corpus.
- Conservation and stationary-harmonic-field checks.
- A different time-integration reference on a small subset where useful.

Report cross-backend agreement and continuum-accuracy evidence separately. Reproduce
relevant validation for these closed surfaces rather than importing the PDF's
reported results as evidence for this implementation.

## 7. Initial conditions and temporal sampling

Provide three labelled IC families:

1. One localized pressure pulse with zero initial velocity.
2. Two pressure pulses with zero initial velocity, testing interference/superposition.
3. A pressure pulse plus harmonic initial velocity, testing circulation preservation.

Vary pulse centers, widths, amplitudes, and circulation strengths. Use intrinsic,
smooth surface-localized pulses; choose the precise construction during the geometry
prototype. Avoid inadvertently exciting nearby but geodesically distant handles.
Record physical IC parameters independently of mesh IDs.

All paired meshes must represent the same physical initial condition. For harmonic
velocity, use a documented cycle/circulation prescription or validated transfer;
do not reuse arbitrary eigenbasis coordinates across meshes.

Proposed starting temporal grid: `T = 1` and 51 uniformly spaced snapshots. Inspect
development trajectories before freezing this choice to ensure propagation explores
the handles. Solver steps are independent of output sampling and selected through
accuracy studies. More snapshots do not count as more independent configurations.

## 8. Corpus sizes, splits, and leakage controls

First build a roughly 32-configuration development corpus spanning all genera and
IC families. Use it to validate correctness, generation cost, storage, and spectral
preprocessing. Freeze the protocol before generating the first full corpus.

Proposed full-run counts, subject to measured cost:

| Partition | Configurations | Purpose |
| --- | ---: | --- |
| Training, genus 1 and 2 | 512 | Balanced topology and IC coverage |
| Validation, genus 1 and 2 | 64 | Hyperparameters and checkpoint selection |
| ID test, genus 1 and 2 | 64 | Unseen configurations from training distributions |
| Geometry OOD, genus 1 and 2 | 64 | Held-out shape parameter ranges |
| Topology OOD, genus 3 | 64 | Transfer to an unseen genus |

A configuration comprises geometry, initial conditions, and physical parameters.
Assign splits before meshing or temporal sampling. Keep all resolutions, remeshes,
and snapshots of a configuration in the same partition. Track `geometry_id` as
well as `config_id`; by default group shared geometries across train/validation/test
to prevent accidental geometry leakage. Any intentional IC-only holdout on familiar
geometry must be explicitly labelled.

Train on medium resolution. Generate coarse, fine, and independently remeshed
versions of the 64 ID test configurations. Normalize using training medium data
only. Balance IC families deterministically in each applicable partition.

Include all three IC families in the initial experiment. Define an additional
IC-transfer manifest later that withholds one family; keep this separate from the
topology holdout. Do not combine unseen circulation and unseen genus without a
corresponding controlled comparison.

Match pulse wavelengths, area, mesh density, and local geometric scales across
topology splits. A genus change necessarily changes geometry; document remaining
confounding factors rather than claiming a perfectly isolated topological change.

## 9. Storage and generation workflow

Retain the existing typed HDF5 sample approach where suitable. Store:

- Mesh coordinates, connectivity, orientation, topology, and quality diagnostics.
- Sparse FEEC operators, mass matrices, and their exact basis/DoF conventions.
- Initial conditions, coefficient data, pressure/velocity snapshots, and actual times.
- Configuration, geometry, resolution, split, and sample identifiers.
- Transfer metadata and operators where validated.
- Solver/backend versions, dependency revisions, numerical tolerances, seed,
  runtime, and conservation diagnostics.

Use manifests, atomic sample writes, and resumable generation. Validate configuration
hashes before reusing outputs. Stream normalization statistics and bound mesh/operator
caches rather than retaining the full corpus in memory. Record generation and
spectral-preprocessing costs separately. Keep generated corpora out of Git.

## 10. Evaluation metrics

Compute metrics per configuration, then aggregate over configurations with equal
weight. Report pressure and velocity separately, split by genus, IC family, and
resolution. Include mean, median, and an upper-tail statistic; preserve per-sample
results and time curves.

### Spatial accuracy

- Ordinary MSE/RMSE as familiar reference metrics.
- Mass-weighted normalized RMSE as the primary cross-mesh accuracy metric.
- Error versus time and final-time error.
- Maximum error as a supplementary local diagnostic.

Use trajectory-level normalization and safe floors for fields that vanish at some
times. Report absolute errors alongside normalized errors. Raw edge-cochain MSE is
resolution-sensitive because edge integrals change with edge length; label it as
supplementary for resolution comparisons. Treat the supplied initial snapshot
separately from predictions at positive times.

### Surface spectral accuracy

For pressure, solve `K phi = lambda M0 phi` with an `M0`-orthonormal basis and project
using `a = Phi.T @ M0 @ p`. This is the surface analogue of Fourier analysis.

- Report the constant pressure mode separately.
- Define low/middle/high bands using fixed physical wavelength or nondimensional
  eigenvalue thresholds; select them from development/training data and freeze them.
- Do not use a fraction of each mesh's largest eigenvalue as the frequency definition.
- Compare predicted and reference coefficients, retaining phase/sign information.
- Keep degenerate eigenspaces together when selecting cutoffs where practical.
- Compute modes through the middle-band cutoff. Obtain high-band squared error as
  total mass-weighted error minus the resolved-band error, accounting for the
  constant mode and numerical orthogonality. If using only truncated modes without
  this complement, explicitly label the represented frequency range.

Each mesh uses its own basis for comparing its prediction and target. Do not compare
individual eigenvectors between different geometries without an alignment method.
Cache bases on CPU and validate orthogonality/eigensolver residuals.

Velocity spectral analysis requires a 1-form Hodge-Laplacian basis. Initially report
velocity error decomposed into exact, coexact, and harmonic components; add its
frequency bands after validating the full surface complex.

### Physics and topology

- Predicted energy versus the reference energy curve, plus drift from initial energy.
- Reference solver energy drift beside prediction drift.
- Conserved mean pressure/integral for the constant-coefficient closed system.
- Closedness error, comparing predictions with reference behavior.
- Harmonic-component error and drift, particularly for circulation ICs.

Use projection-based harmonic comparisons within each mesh to avoid arbitrary basis
signs/rotations. Boundary metrics do not apply to this corpus.

### Resolution and remeshing

First compare predictions against the reference solution on their own mesh. Report
paired degradation from medium to coarse/fine/remeshed discretizations.

Only add cross-resolution consistency and commuting-transfer metrics after defining
and validating degree-specific transfer operators. Vertex slicing is not a general
transfer rule for edge forms. Separate reference discretization error from learned
prediction error and from transfer error.

## 11. Training-loss experiments

Compare three objectives with otherwise matched training conditions:

1. Ordinary MSE, with separate pressure and velocity scaling.
2. Mass-weighted MSE.
3. Mass-weighted MSE plus band-balanced pressure spectral error.

The weighted spatial objective uses `ep.T @ M0 @ ep` and `eu.T @ M1 @ eu`, with
fixed training-derived scales and consistent per-sample normalization. Retain the
full sparse velocity mass matrix. Choose scale floors for zero/near-zero fields.

The spectral term needs nonuniform weighting or band balancing: a complete,
uniformly weighted spectral squared error equals mass-weighted spatial squared
error by Parseval and adds no distinct objective. Use training-derived band scales
with floors and select the spectral coefficient on validation data only.

Initially apply spectral loss to pressure. Keep energy and circulation as evaluation
diagnostics; physics penalties are subsequent ablations. Pressure spectral projection
is differentiable and can run on GPU after CPU preprocessing.

## 12. Baseline model and training infrastructure

Model work begins after the data and validation gates. The current recommendation
is a small **Transolver-based** model for direct initial-state-plus-query-time
prediction. This is a proposed adaptation, not an established SOTA result on this
new benchmark.

Proposed adapter: separate vertex/edge encoders, shared slice attention, and separate
pressure/velocity output heads. Provide surface coordinates, edge midpoints,
directions, lengths, initial fields, coefficient, and query time. Handle cochain
orientation and edge-length scaling explicitly; do not treat edge integrals as
ordinary point values. Specify which operators are actually consumed by each model.

Proposed development defaults: four blocks, width 128, 32 slices, one mesh per batch
with gradient accumulation. Confirm memory use on real meshes before fixing these.
Prevent attention across unrelated samples when implementing batching.

Use float32 initially, with explicit conversion from float64 references. Validate
sparse loss gradients before enabling mixed precision. Provide fixed seeds,
checkpoint/resume, best-validation checkpoint selection, configuration capture, and
machine-readable metrics. Run three seeds for the first reported comparison.

Include persistence and zero-prediction controls. A MeshGraphNet autoregressive
baseline is follow-up work; distinguish recursive rollout metrics from direct
query-time prediction. GINO and Transolver++ remain optional later comparisons.

## 13. CPU checks and GPU shell workflow

The proposed interface below is illustrative; these scripts do not yet exist:

```bash
bash scripts/run_acoustic_baseline.sh smoke --device cpu
bash scripts/run_acoustic_baseline.sh pilot --device cuda --seed 0
bash scripts/run_acoustic_baseline.sh pilot --device cuda --seed 0 --resume
```

The launcher should resolve repository-relative paths, use the pinned environment,
generate or validate data, train, evaluate, and save logs/artifacts. An explicit
CUDA request must fail clearly if CUDA is unavailable. Keep MFEM validation an
optional separate stage with its own environment/dependency setup.

CPU acceptance includes schema round trips, split leakage checks, orientation
handling, sparse weighted-loss gradients, spectral identities/bands, and a tiny
overfit test. Verify CPU/GPU agreement on a small case when CUDA becomes available;
CPU success alone does not establish CUDA support.

Profile generation, eigensolves, training steps, inference, and evaluation separately.
Do not promise a GPU runtime or memory budget before measurement. Save predictions
for offline CPU reference evaluation, including expensive Hodge diagnostics.

## 14. Milestones and completion gates

1. **Environment and geometry prototype:** pinned PyDEC, reproducible genus 1–3
   meshes, manifold/topology checks, and embedded-surface assembly feasibility.
2. **FEEC oracle:** consistent operators, midpoint stepping, nonzero harmonic ICs,
   serialization, and numerical verification cases.
3. **MFEM comparison:** aligned independent assembly, the 36-case validation subset,
   reported residuals/errors, and separate convergence evidence.
4. **Development corpus and metrics:** configurable splits, validated IC transfer,
   spatial/spectral/physical diagnostics, and measured generation/storage costs.
5. **Full pilot corpus:** frozen configuration manifests, validation reports, and
   train-only normalization/spectral settings.
6. **Learning smoke test:** adapted baseline, meaningful gradient checks, tiny-data
   overfit, checkpoints, and CPU shell execution.
7. **GPU experiments:** three loss variants, three seeds, reproducible evaluation,
   and separate follow-up model/IC ablations.

Commit each milestone in reviewable units on the experiment branch. Run the existing
acceptance suite and focused new checks appropriate to each change. Preserve the
existing `AGENTS.md` unless a later task explicitly authorizes editing it.

## 15. First deliverables

- Reproducible PyDEC-backed FEEC acoustic oracle for closed surfaces.
- Independent MFEM comparison report with operator and trajectory discrepancies.
- Versioned dataset manifests and generation/quality/convergence diagnostics.
- CPU smoke and direct-terminal GPU launch scripts.
- Checkpoints, configurations, dependency revisions, seeds, and per-sample results.
- Main table with ID, geometry OOD, topology OOD, coarse, fine, and remeshed rows;
  pressure/velocity error, spectral bands, energy, and harmonic-error columns.
- Error-versus-time and error-versus-resolution plots, representative field/error
  visualizations, and measured cost/memory results.

## 16. References

- [PyDEC repository and installation instructions](https://github.com/hirani/pydec)
- [PyDEC paper](https://arxiv.org/abs/1103.3076)
- [FEEC foundations: Arnold, Falk, and Winther](https://sites.math.rutgers.edu/~falk/papers/bulletin-8-13-09.pdf)
- [MFEM finite-element integrators](https://mfem.org/integration/)
- [MFEM embedded-surface example](https://docs.mfem.org/4.9/ex29_8cpp_source.html)
- [PDEBench](https://arxiv.org/abs/2210.07182)
- [Generalized surface Fourier analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC4401437/)
- [Transolver paper](https://arxiv.org/abs/2402.02366) and [implementation](https://github.com/thuml/Transolver)
- [MeshGraphNets implementation](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
