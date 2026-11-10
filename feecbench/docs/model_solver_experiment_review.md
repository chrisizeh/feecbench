# Acoustic benchmark: model, solver, evidence, and relation to the TNO proposal

Reviewed 10 November 2026 against the current code, available local reports, and the previously extracted text of *Typed Physics Benchmark and Transformer–TNO Foundation Model* (`tno_bench.pdf`, 4 November 2026). This is a description of the closed-surface experiment, not every older planar component in this repository. Remote GPU progress has not been inspected directly.

## 1. What we have built

We have a closed-surface acoustic benchmark with a numerically checked FEEC reference solver, a Transolver-style neural baseline, and an optional differential-transport extension. The historical full experiments establish a strong effect of balancing pressure and velocity losses. They also expose substantial remaining errors in learning the coupled dynamics. The extended experiment tests whether explicit transport improves those results.

This is an acoustic prototype of the PDF's broader program. It is not yet the proposed multi-PDE Transformer–TNO foundation model. Three kinds of evidence must remain separate:

- **Numerical validation:** whether the reference solver and its discretization are trustworthy within the tested regime.
- **Implementation validation:** whether the neural operators, gradients, orientation handling, and experiment orchestration behave as intended.
- **Learning results:** how accurately trained models fit and generalize. Passing implementation tests does not establish predictive benefit.

## 2. Physical problem and data

Pressure is a vertex 0-cochain, `p ∈ C⁰`, and velocity is an oriented edge 1-cochain, `u ∈ C¹`. An edge value is an integrated tangential quantity, not a three-component velocity vector attached to a vertex.

The experiment uses connected, closed, orientable triangulated surfaces of genus 1–3. Sound speed is fixed at `c = 1`; there is no forcing, damping, boundary, or material variation. The prediction horizon is `0 ≤ t ≤ 1`.

The semidiscrete equations are

```math
M_0 \dot p = D_0^\top M_1 u, \qquad \dot u = -D_0 p,
\qquad \delta_1 = M_0^{-1}D_0^\top M_1.
```

The four initial-condition families are one pressure pulse, two pressure pulses, pressure plus harmonic velocity, and exact velocity `u₀ = D₀ψ` with zero or independently varied initial pressure. Harmonic velocity is stationary in this system. Exact velocity is therefore essential for testing whether velocity generates the correct pressure response.

The full corpus contains 512 training configurations and 64 each for validation, ID test, geometry OOD, and genus-3 OOD: 768 physical configurations. Paired coarse, fine, and independent remesh variants of the ID test set bring this to 960 mesh samples. Training uses medium meshes, grid parameter 16; this parameter is not a fixed vertex count. Reference generation uses timestep 0.002, 51 regular snapshots, and 64 pressure eigenmodes for spectral diagnostics/control.

Training sees genus 1 and 2. Geometry OOD changes the geometry distribution, while genus-3 OOD changes both topology and geometry. Physical configurations and geometry groups are split together to limit leakage. Normalization scales and balancing weights are derived only from training-medium data. Off-grid time queries remain inside the training horizon; they are temporal interpolation tests, not long-time extrapolation.

## 3. Reference solver

### Spatial discretization

The surface oracle uses lowest-order Whitney FEEC: continuous piecewise-linear scalar pressure and Whitney edge 1-forms. PyDEC supplies consistent mass matrices `M₀`, `M₁`, and `M₂`. These are not the diagonal circumcentric DEC stars. Edges use a canonical low-to-high vertex orientation, triangles are consistently oriented, and the incidence operators obey `D₁D₀ = 0`.

The pressure stiffness is `K = D₀ᵀ M₁ D₀`. Derived codifferentials use mass solves rather than replacing the consistent masses with diagonal approximations. A lumped auxiliary metric is used when finding the harmonic nullspace; it is not the acoustic dynamics metric.

### Time integration

The reference solver uses implicit midpoint. For timestep `h`, the implementation solves

```math
(M_0 + h^2K/4)p_{n+1}
 = (M_0-h^2K/4)p_n + hD_0^\top M_1u_n,
```

then updates

```math
u_{n+1}=u_n-\frac h2D_0(p_n+p_{n+1}).
```

SciPy sparse factorization is cached for each step size. Off-grid outputs use remainder steps from a common regular trajectory, so requesting an additional query does not change the regular solution. Midpoint is second-order and preserves the quadratic discrete energy, up to solve/roundoff error:

```math
E=\tfrac12(p^\top M_0p+u^\top M_1u).
```

This solver generates the targets. The neural network does not execute these time steps.

### Numerical controls

Zero prediction and initial-state persistence provide simple reference scores. A truncated modal propagator uses known FEEC eigenmodes and analytical time evolution, including the velocity-driven pressure response and stationary harmonic content. It is a numerical control with access to the known operator, not a learned competing model or a continuum ground truth.

Implementation: [oracle.py](../src/feecbench/surface/oracle.py), [operators.py](../src/feecbench/surface/operators.py).

## 4. Neural model

### Attention baseline

`SurfaceTransolver` is a small Transolver-style slice-attention adaptation for direct time-conditioned prediction:

```math
(K,p_0,u_0,t)\longmapsto(\widehat p(t),\widehat u(t)).
```

Each forward call processes one mesh. Separate linear encoders construct vertex and edge features:

- Vertices: coordinates, normalized initial pressure, coefficient, and query time.
- Edges: midpoint, canonical unit tangent, length, normalized canonical initial velocity divided by length, coefficient, and query time.

Although the feature interface includes a coefficient, the current data path supplies `c = 1`. It does not demonstrate variable-coefficient learning.

Vertex and edge tokens are concatenated into a shared attention stream. Each block learns soft assignments to 32 slices, pools token features onto those slices, applies four-head attention between slices, and distributes the result back to tokens. Residual updates, LayerNorm, and a pointwise MLP complete the block. The standard model has width 128, four blocks, and **681,602 parameters**.

Separate pressure and velocity heads produce corrections multiplied by query time. Velocity corrections are converted back to oriented integral cochains using edge length and sign. Consequently, the output exactly recovers the supplied initial state at `t = 0` by construction.

This is direct prediction at each requested time, not an autoregressive integrator. Exact initial-state recovery does not guarantee a correct time derivative, conservation, or consistent trajectories. Canonical edge handling supports orientation changes; it does not establish general rotation, intrinsic-coordinate, or arbitrary relabeling equivariance. Coordinates and tangents are explicit inputs.

An optional directional-feature adapter exists, but the matched extended study uses the scalar adapter. The directional experiments did not establish a general remedy for the observed fitting problem.

### Added differential transport

The optional `d0_delta1` variant adds simultaneous vertex-to-edge and edge-to-vertex residual messages before every attention block:

```math
m_{0\to1}=L^{-1}S D_0 Z_0,\qquad
m_{1\to0}=M_0^{-1}D_0^\top M_1 S L Z_1.
```

Here `Z₀` and `Z₁` are learned hidden features; `Z₁` uses canonical edge-density coordinates. `L` contains edge lengths and `S` maps to the requested edge orientation. These conversion factors make the messages compatible with integral cochain operators. The hidden features are not physical pressure and velocity, and the residual updates do not impose the acoustic equations exactly.

Each message passes through channel LayerNorm and learned projections. Output projections start at zero, so paired baseline/transport models initially represent the same function. The transport model has **814,722 parameters**, approximately **19.5% more** than the baseline.

Applying `M₀⁻¹` inside the network uses device-local, Jacobi-preconditioned, multi-right-hand-side conjugate gradients on the full consistent mass matrix. Tolerances are `1e-6` in float32 and `1e-10` in float64, with 64 iterations maximum and a final true-residual check allowing five times the tolerance. Failure raises an error. Backpropagation uses an implicit adjoint mass solve. Geometry/operator differentiation and mixed precision are unsupported.

This mass solve is distinct from the reference solver's sparse midpoint factorization. The new model now contains the proposal's central `C⁰ ↔ C¹` differential communication mechanism, but only within this limited acoustic architecture.

Implementation: [model.py](../src/feecbench/surface/model.py), [transport.py](../src/feecbench/surface/transport.py).

## 5. What we checked

| Check | Available evidence | What it establishes / does not establish |
| --- | --- | --- |
| Independent reference backend | 36 MFEM comparison cases; maximum relative matrix discrepancy `6.14e-14`, trajectory discrepancy `1.26e-13` | Strong agreement on the tested discretizations; not continuum accuracy |
| Convergence | Temporal order approximately 2; analytical sphere spatial/geometric orders 1.82–1.89 | Expected convergence on those tests; not every corpus geometry or field |
| Resolution audit | Disjoint 24-configuration grid-16 audit: maximum low-band pressure difference 0.94%, middle-band 4.18%, velocity 4.99%; temporal velocity difference 0.52% | Sampled finer-reference differences; not rigorous error bounds |
| Older resolution diagnostic | Grid-12 middle-band discrepancy 6.51% and velocity 9.87% | Earlier regime failed its provisional gate; those findings must not be presented as grid-16 results |
| Structural/transport implementation | Operator agreement, adjoint gradient checks, active edge flips, exact initial state, both coupling-direction local fits, checkpoint/resume/evaluation integration | Correctness checks for implemented operations; not full-run accuracy |
| Repository tests | Previously completed full suite: 111 passed, 3 skipped; Ruff passed | Recorded CPU validation; skipped checks are not passes and these tests were not rerun for this document |
| Study orchestration | Complete 12-run CPU smoke with tiny models/budgets; 48 validation and 48 held-out checkpoint records | Training, snapshots, evaluation and export work end to end; scientifically uninformative accuracy |

The extended runner also implements a CUDA preflight comparing active transport outputs and parameter gradients against CPU on one training mesh. Its implementation is not evidence that all GPU meshes, tolerances, or performance have been validated. The current remote job's preflight/resource files have not been inspected for this review.

The evaluation code measures raw-field mass-norm errors, pressure spectral quantities, exact/harmonic/coexact velocity errors, energy behavior, closedness, harmonic drift, time curves, and sampled PDE residuals. Fixed-query source-removal, zero-input, and mean-free-pressure diagnostics probe the coupling response. They complement ordinary whole-field errors, which can hide a stationary harmonic component.

Residual coverage is limited: the extended full study uses the first four eligible mesh samples, one in smoke. The older report describes one-sample residual diagnostics. Neither is a corpus-wide or stratified PDE-fidelity assessment.

## 6. What the completed learning experiments found

The historical full runs used **attention without differential transport**. Ordinary MSE, consistent mass loss, and balanced MSE were compared using the same corpus, seeds 0–2, AdamW, constant learning rate `0.001`, accumulation four, and 200 epochs / 25,600 optimizer updates per run. All use a common validation mass-score selection rule.

Regular-query validation trajectory errors, mean ± seed standard deviation:

| Objective | Pressure | Velocity |
| --- | ---: | ---: |
| Ordinary MSE | 31.11 ± 2.76% | 92.11 ± 1.85% |
| Consistent mass loss | 28.61 ± 2.53% | 39.69 ± 2.87% |
| Balanced MSE | 27.60 ± 1.52% | 40.36 ± 4.40% |

These are configuration-averaged, trajectory-normalized physical mass errors, not training-loss values. Seed standard deviations are not confidence intervals.

Ordinary MSE has approximately **173:1 pressure-to-velocity zero-prediction contributions** after normalization. Training-derived weights correct this imbalance. Balanced MSE recovers most of mass loss's aggregate velocity improvement. Therefore, the earlier mass-versus-MSE improvement cannot be attributed solely to geometric mass weighting. This does not show that mass geometry is irrelevant; balanced MSE does not uniformly beat mass loss.

The remaining limitation is already visible on training configurations. For the exact-velocity family, balanced-MSE fixed-query pressure error is **60.91% on training and 67.85% on validation**. For the pulse family, velocity errors are **36.71% and 37.81%**. These diagnostics use a different query set from the trajectory table and should not be compared as identical scores.

Energy behavior also remains imperfect: the balanced validation mean of each configuration's maximum relative energy error is **27.63%**, rising to **66.55%** for the exact-velocity family. Energy preservation alone would still not establish correct pressure–velocity exchange: persistence can preserve total energy while missing the dynamics.

Small two-case grid-12 fitting studies show that the architecture can fit both coupling directions with a longer budget. One 2,000-update cosine run reached source-response errors of **6.30% / 4.41%**. These single-seed local fits do not establish full-corpus generalization, and changing schedule and budget together does not isolate either effect. A separate matched constant/cosine continuation workflow exists, but no full result from it is included in the available report.

### Existing OOD evidence

The original mass-loss campaign already evaluated geometry OOD, genus-3 OOD, and coarse/fine/remeshed transfer. Performance degrades under these shifts. Its ID-medium pressure/velocity errors are **28.23% / 38.34%**, versus **1.33% / 1.72%** for the truncated same-mesh modal control.

The balanced-MSE study evaluated training and validation only. No completed full transport OOD result is available locally. The smoke study's held-out records are pipeline checks, not replacements for these missing scientific results.

Evidence: [progress report source](acoustic_progress.tex), [historical campaign export](../../reports/acoustic/analysis.json), [balanced-loss export](../../reports/acoustic/loss-balance-g16/analysis.json). Export files are local, untracked evidence and may be absent in another checkout.

### Measured held-out results: attention-only models

The exported historical campaign contains held-out results for **both mass loss and ordinary MSE**. Neither model uses differential transport. Each row below uses 64 configurations and seeds 0–2, evaluated at regular query times within `0 < t ≤ 1`. Values are percentage trajectory-normalized mass errors, averaged equally over configurations and then seeds; lower is better. The ± values are the standard deviation of the three seed means, not confidence intervals or variation across individual cases.

**Mass-loss model: absolute errors and degradation relative to its ID-medium baseline.** Differences are calculated from unrounded means and expressed in percentage points (pp), not relative percent increases.

| Evaluation | Pressure error | Velocity error | Pressure change vs ID | Velocity change vs ID |
| --- | ---: | ---: | ---: | ---: |
| ID reference, medium | 28.23 ± 2.32% | 38.34 ± 3.19% | +0.00 pp | +0.00 pp |
| Geometry OOD, medium | 31.11 ± 3.32% | 42.11 ± 4.13% | +2.88 pp | +3.78 pp |
| Genus-3 OOD, medium | 33.40 ± 1.00% | 44.39 ± 3.42% | +5.17 pp | +6.05 pp |
| Resolution transfer, coarse | 30.03 ± 1.80% | 44.34 ± 2.94% | +1.80 pp | +6.00 pp |
| Resolution transfer, fine | 28.72 ± 2.35% | 39.77 ± 3.31% | +0.49 pp | +1.43 pp |
| Independent remeshing | 29.51 ± 1.68% | 42.99 ± 2.78% | +1.27 pp | +4.66 pp |

**Ordinary-MSE model: the corresponding held-out control.** Its poor velocity accuracy is consistent with the field imbalance observed on validation. These numbers do not replace the still-missing balanced-MSE comparison.

| Evaluation | Pressure error | Velocity error |
| --- | ---: | ---: |
| ID reference, medium | 30.55 ± 3.11% | 91.08 ± 1.98% |
| Geometry OOD, medium | 34.30 ± 4.31% | 93.51 ± 2.11% |
| Genus-3 OOD, medium | 35.97 ± 3.80% | 92.18 ± 1.58% |
| Resolution transfer, coarse | 35.67 ± 5.50% | 96.75 ± 7.49% |
| Resolution transfer, fine | 31.28 ± 3.94% | 91.29 ± 1.60% |
| Independent remeshing | 35.25 ± 6.04% | 92.16 ± 4.07% |

**What these results show.** For the mass-loss model, genus-3 transfer has the largest pressure degradation (+5.17 pp), and velocity increases by +6.05 pp. Coarsening also raises velocity error substantially (+6.00 pp); independent remeshing raises it by +4.66 pp. Fine-mesh transfer changes the aggregate scores relatively little (+0.49 pp pressure, +1.43 pp velocity). These are descriptive differences, not statistical-significance claims or evidence that the absolute errors are acceptable.

The mesh variants reuse the same ID physical configurations, allowing paired mesh comparisons. Geometry and genus results use separate partitions, so their differences from ID also reflect the distribution of those configurations. Genus 3 changes geometry as well as topology. Scores are measured against each mesh's numerical oracle; small cross-mesh changes do not prove convergence to a continuum solution.

| Model / study | Full held-out results available in the reviewed local exports? |
| --- | --- |
| Historical attention-only, mass loss | Yes: the table above |
| Historical attention-only, ordinary MSE | Yes: the table above |
| Historical attention-only, balanced MSE | No: training/validation only |
| New matched 200/400-epoch attention-only runs | No completed full held-out export reviewed |
| New matched 200/400-epoch transport runs | No completed full held-out export reviewed |
| Tiny CPU smoke models | Pipeline checks only; excluded from scientific tables |

Source: `reports/acoustic/analysis.json` at the repository root, `summary` entries filtered by `method=neural`, `queries=regular`, `family=all`, `genus=all`, and the indicated objective/split/resolution. These are historical 200-epoch results, not interim results from the currently running extended study. Off-grid queries are deliberately not pooled into these tables.

## 7. The extended experiment currently being run

The new study separates architecture, objective, and training budget:

| Axis | Settings |
| --- | --- |
| Architecture | Attention only; attention plus `D₀/δ₁` transport |
| Objective | Balanced MSE; consistent mass loss |
| Seeds | 0, 1, 2, paired |
| Budget | 200 and 400 epochs |
| Saved checkpoint roles | Best validation checkpoint and final checkpoint at each budget |
| Optimization | AdamW, constant LR 0.001, accumulation four |

There are **12 fresh training runs**, each continued from its own 200-epoch state to 400 epochs. This is not 24 independent trainings. At 128 updates per epoch, the total budget is **614,400 optimizer updates**. Historical checkpoints are context, not substituted into the matched study.

The primary question is whether transport improves results at fixed loss, seed, budget, and checkpoint role. Secondary comparisons test whether another 200 epochs help and whether mass loss differs from balanced MSE within each architecture. Ordinary MSE is omitted because its imbalance is already demonstrated. The schedule stays constant to avoid adding another experimental axis.

The suite evaluates training diagnostics and validation. **Held-out evaluation is a separate command and currently requires every validation cell to be complete.** It then scores all saved budgets/roles on the available ID, geometry-OOD, topology-OOD, and paired mesh variants. There is no current CLI option for evaluating only completed non-transport runs early.

```bash
# From the repository root, on the machine containing the prepared study:
bash feecbench/scripts/run_transport_study.sh run --device cuda --stage suite
bash feecbench/scripts/run_transport_study.sh run --device cuda --stage heldout
bash feecbench/scripts/run_transport_study.sh export
```

Repeat the suite command to resume after interruption. Completed milestones are reused; training resumes saved epochs. Source, data, lockfile, and configuration hashes protect the prepared study, and a process lock prevents concurrent writers. Editing the runner in an active prepared checkout can invalidate its source identity.

Runs are ordered by seed, then objective, then architecture. The reported label `train-d0_delta1-mse_balanced-seed1-400` identifies run 6 of 12, continuing toward 400 epochs. It is not a live epoch count and does not imply that all non-transport runs are finished. Without the remote logs, this document cannot establish subsequent progress or completion.

Interpretation cautions: transport has more parameters and costs more per update; matching epochs does not match capacity or compute. A best checkpoint at the 400-epoch milestone can still come from epoch 200 or earlier. Inspect checkpoint epoch and final-role results before attributing a change to additional training.

Implementation and operational details: [transport_study.py](../src/feecbench/surface/transport_study.py), [launch guide](../SURFACE_EXPERIMENT.md).

## 8. Connection to the TNO PDF

The proposal separates a typed physics benchmark from a broadly reusable Transformer–TNO. Our experiment develops one acoustic slice of both. It also makes deliberate choices that differ from the proposed architecture.

| PDF section / idea | Present implementation | Remaining gap |
| --- | --- | --- |
| §3–4: typed cochains, incidence and metric operators | Pressure on vertices, velocity on edges; FEEC operators, metadata and stored reference fields | No demonstrated unified multi-PDE, multi-dimension corpus |
| §5.1: acoustic coupling | Both pressure-to-velocity and velocity-to-pressure dynamics, including nonzero initial velocity | Constant speed only; no boundary/source diversity or broad frequency regimes |
| §8: separate generalization axes | ID, geometry, genus, resolution and remeshing partitions | No coefficient, frequency, long-time, cross-PDE or cross-dimension studies |
| §9: trusted references | Independent MFEM comparison, analytical convergence and sampled refinement audits | Complete mixed-pressure exact-velocity/remesh audits and freeze numerical acceptance criteria |
| §10: typed/physics metrics | Physical mass errors, orientation checks, Hodge content, energy, closedness and sampled residuals | Limited residual coverage; no comprehensive multi-PDE metric validation |
| §11.2: typed token streams and embeddings | Separate vertex/edge encoders and readouts | No explicit learned degree/PDE/task embeddings; no face/volume streams |
| §11.3: differential transport | Optional `D₀` and consistent-mass `δ₁` latent messages | No `D₁/δ₂` or higher-degree neural transport |
| §11.4: Transformer block | Shared slice attention after differential messages | Not the proposed per-degree self-attention plus optional incidence-restricted cross-degree attention |
| §11.5–6: context and outputs | Time/coefficient features and two typed output heads | No global context token carrying PDE/task/topology/resolution and no task-mask system |
| §12: training tasks | Direct initial-state-plus-time prediction | No autoregressive, masked-cochain, denoising, inverse, super-resolution or cross-PDE pretraining tasks |
| §13: losses | MSE, balanced MSE and consistent-mass supervision; a gated spectral-loss implementation exists | No residual/conservation/DEC/pretraining penalties in the current study; spectral training not established |
| §14: variable-size batching | Different mesh sizes supported one mesh per forward call, with gradient accumulation | No block-diagonal multi-mesh batching in this learner |
| §15: baselines and ablations | Attention versus attention-plus-transport; two losses; zero/persistence/modal controls | No full GNN/MLP/MeshGraphNet/FNO baseline suite, transport-only model, node-only study or matched-capacity controls |

The PDF's typed supervised loss uses target-relative per-degree normalization. Our mass training objective is the sum of consistent-mass quadratic errors in normalized fields, divided by surface area. Physical trajectory-relative mass errors are used for evaluation. These objectives are related but not identical.

The current experiment directly addresses the PDF's question of whether differential transport improves typed prediction, and later whether those gains survive geometry/topology/resolution shifts. It cannot establish the PDF's broader claims about foundation pretraining, sample efficiency, higher-degree coupling, or cross-PDE transfer.

## 9. Limits on conclusions and outstanding work

1. **Reference uncertainty remains.** The largest sampled grid-16 velocity difference, 4.99%, is close to the provisional 5% margin. Mixed-pressure exact-velocity and independent-remesh reference checks remain incomplete. Agreement between solver backends does not eliminate discretization error.
2. **The physical regime is narrow.** Closed surfaces, constant `c`, linear acoustics and a fixed horizon exclude heterogeneous materials, forcing, boundaries, nonlinear dynamics, and long-time behavior.
3. **The initial-condition span is restricted.** All four families appear in training. Harmonic initial conditions span only a `g`-dimensional subspace of the full `2g` harmonic space. Reference velocity contains no coexact component, so coexact scores chiefly detect spurious leakage rather than successful coexact dynamics.
4. **Topology is not isolated from geometry.** Genus-3 evaluation changes both. A performance difference cannot be attributed solely to topology.
5. **The architecture does not enforce the dynamics.** Typed messages and orientation handling do not guarantee conservation, closedness, correct energy exchange, or intrinsic equivariance after nonlinear updates.
6. **Architecture comparisons are not capacity/compute matched.** A transport gain could partly reflect extra learned parameters or computation. A transport-only/no-attention ablation is also needed to isolate attention's role.
7. **Scientific transport results are pending locally.** CPU tests and smoke runs establish implementation behavior. Final predictive gains, GPU resource tradeoffs, and transport OOD performance require reviewing the completed full-study exports.
8. **Existing test sets are exploratory evidence.** Historical held-out scores already informed design decisions. Confirmatory claims need a frozen protocol and fresh held-out evidence, rather than repeatedly tuning to the same partitions.
9. **Some diagnostics and visualizations are incomplete.** Residuals cover a small subset. Matched spatial reference/prediction/error panels and separate pressure/velocity energy-exchange plots need the matching reference meshes/fields, which were absent from the reviewed full-run export.

## 10. Source map and status caveats

- [Model and loss definitions](../src/feecbench/surface/model.py)
- [Typed transport and differentiable mass solve](../src/feecbench/surface/transport.py)
- [Reference integrator and modal control](../src/feecbench/surface/oracle.py)
- [FEEC assembly and derived operators](../src/feecbench/surface/operators.py)
- [Independent backend validation](../src/feecbench/surface/mfem_validation.py)
- [Reference-resolution checks](../src/feecbench/surface/reference_checks.py)
- [Metrics](../src/feecbench/surface/metrics.py) and [coupling diagnostics](../src/feecbench/surface/coupling.py)
- [Extended study implementation](../src/feecbench/surface/transport_study.py), [experiment guide](../SURFACE_EXPERIMENT.md), and [numerical decisions](../DECISIONS.md)
- [Existing report source](acoustic_progress.tex), incorporating full-run results from 5–7 November and the subsequent implementation checks
- Original proposal: `/Users/chrisizeh/Downloads/tno_bench.pdf`, especially §§3, 5.1, 8–15; reviewed through the available prior extraction `/private/tmp/feecbench-tno.txt`

Older planning text can describe transport or OOD work as wholly deferred. The current code and dated evidence take precedence: transport is now implemented, and historical mass-loss OOD results exist. Conversely, the existence of a runner or metric does not establish that its full experiment has completed. This review adds documentation only; it does not change the active study or report new measurements.
