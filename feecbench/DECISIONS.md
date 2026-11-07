# DECISIONS — choices made

Design and implementation choices behind feecbench, with the reasoning for each.

1. **gmsh is a hard dependency, imported lazily.** `feecbench.mesh.gmsh_backend`
   imports `gmsh` inside `generate()` so that data consumers (dataset, evaluator) can
   load HDF5 corpora on machines without a working gmsh; meshing raises an `ImportError`
   naming the fix (`pip install gmsh`, plus a system `libGLU.so.1` that the manylinux
   wheel dlopens — e.g. `conda install -c conda-forge libglu`). There is **no**
   hand-written fallback generator.

2. **CAD circles are re-projected on refinement.** The coarse level is gmsh-native
   (boundary vertices lie on the OpenCASCADE circles to ~1e-16). Medium and fine levels
   are uniform midpoint subdivisions of it; new vertices on boundary edges whose
   endpoints both lie on one CAD circle are projected radially back onto that circle, so
   *every* level has on-circle boundary vertices (tested). Parent vertices keep their ids,
   so the vertex-inclusion prolongation stays an exact index map. The levels triangulate
   slightly different polygonal approximations of the same curved domain, which is the
   behaviour an MFEM oracle with curved refinement would also show.

3. **Boundary masks: gmsh classification, cross-checked.** For gmsh-native meshes the
   boundary vertex mask comes from gmsh's physical group "boundary" (all boundary curves
   of the surface); the oracle asserts it equals the "edge in exactly one triangle" mask
   and raises otherwise. Refined and stored samples use the incidence rule.

4. **Determinism.** `Mesh.Algorithm = 6` (Frontal-Delaunay), uniform size field
   (`MeshSizeMin = MeshSizeMax = h`, no curvature adaptation), `Mesh.RandomSeed` pinned;
   the full option set, gmsh version, h and seed are stored in `mesh.source` of every
   sample (plus the level derivation). Verified bit-reproducible across runs.

5. **Remeshing = gmsh at a seeded perturbed size.** `mesh.remesh` multiplies h by a
   seeded factor in [0.9, 1.1] and bumps `Mesh.RandomSeed`; Frontal-Delaunay is
   insensitive to the seed alone, so the size perturbation is what guarantees different
   connectivity. Same `config_id`, `refinement = "remeshed"`, no parent, no prolongation.

6. **multi_hole is the square [-1, 1]² minus circles.** Hole boundaries are true CAD
   circles, so there is no `h ≤ r/2` snapping constraint. β₁ = number of holes.

7. **`operators.py` holds the full complex.** Incidence matrices, stars, derived
   operators, and Hodge helpers live together; all assembly is vectorized (edges via
   `np.unique(..., return_inverse=True)`, star1 via one einsum over triangles). The
   Whitney mass matrix agrees with a per-cell loop assembly to 1e-16 on the same mesh.

8. **Evaluator metrics are plain functions, not per-metric classes.** The profile keys are
   `rel_l2_star`, `mse/mae/max/p99`, `rollout_rel_l2`, `boundary/interior`,
   `closedness{pred,target}{abs,rel_opnorm}`, `hodge_error{exact,harmonic,coexact}`,
   `energy{...}`; `evaluate_profile(pred, sample, target=None, include_hodge=True,
   harmonic=None)`. The energy functional lives in `oracle.py` and registers itself in the
   evaluator's registry (a per-PDE plugin alongside the oracle).

9. **Time integration and mass lumping.** feecbench uses explicit **leapfrog with a CFL
   bound** rather than an implicit midpoint rule; both are second order and symplectic, and
   the leapfrog energy is conserved exactly in its staggered form (`energy_drift_rel`
   ~1e-16). It always uses the **lumped** P1 pressure mass (`star0_type = "lumped_P1"`),
   which makes the explicit p-update a diagonal solve; eigenmode convergence stays second
   order (rate 2.02). Governing equations `p_t = c² δ₁ u`, `u_t = -d₀ p`; edge orientation
   low → high index; `D0` (-1 at i, +1 at j); the Whitney 1-form mass matrix and the
   standard boundary rule. `star1` is a Whitney–Galerkin (SPD, non-diagonal) Hodge star.

10. **Staggered energy diagnostic.** The leapfrog conserves
    `E = ½ uᵀ⋆₁u + ½ pⁿᵀ⋆₀(pⁿ⁺¹/c²)` to machine precision; any drift above ~1e-12 flags
    a bug or CFL violation. The evaluator's energy metric uses the physical energy of the
    stored snapshots (`u` time-centered), whose O(dt²) oscillation is reported as
    `target_drift_rel`.

11. **Hodge split, harmonic basis, closedness normalization, SPD test.** Exact part by
    pinned least-squares potential solve, harmonic by projection onto the shift-invert
    `eigsh` null-space basis (dimension must equal β₁), coexact as the remainder;
    closedness reported raw and as `rel_opnorm` with the per-mesh ⋆₁→⋆₂ operator norm of
    d1; SPD checked with shift-invert `eigsh` (`which='SA'` does not converge on these mass
    matrices).

12. **Split semantics.** Configuration-level splits, `test_topo_ood` = the whole
    multi_hole family, `test_mesh_ood` = the remeshed variants (sample-level rule).
    Multi_hole remeshed variants are both topology- and mesh-OOD.

13. **Normalization.** ⋆-weighted mean/std/rms over the train split with
    per-snapshot-delta moments; applied in `FEECDataset.__getitem__`, inverted by
    `denormalize`; `load_full` and the evaluator always see raw fields.

14. **BLAS threads capped to 1** at every entry point: multi-threaded OpenBLAS on
    many-core nodes is dramatically slower for these small sparse kernels.

15. **Generic sample comparison.** `schema.first_difference` walks dataclasses / dicts /
    lists / arrays / sparse matrices and returns the path of the first mismatch,
    tolerating the list/tuple coercion of JSON round-trips.

16. **HDF5 attrs are round-tripped generically** (`vars()` → attrs, `np.generic.item()`
    back), so schema blocks can grow without touching `io.py`. `schema_version = "0.1.0"`;
    files written under an older schema_version are refused — regenerate.

17. **Mesh quality is checked from gmsh stats** — min angle > 20° on every level, with
    quality statistics stored per sample.

18. **Closed surfaces are an additive, separately versioned experiment.** The
    `surface/` package and `surface-0.2.0` HDF5 format preserve planar 0.1.0 APIs.
    Surface triangles have consistent manifold orientation, not global planar CCW;
    edges retain low-to-high orientation. Whitney FEEC assembly comes from pinned
    hirani/PyDEC, with consistent M0/M1 and inverse-area mass for top-form integrals.
    All surface codifferential applications solve the full M0 system. The auxiliary
    lumped metric in harmonic nullspace extraction does not enter dynamics.

19. **Surface dynamics and query sampling.** Implicit midpoint uses a reduced
    pressure factorization and conserves physical discrete energy for c=1. Query
    remainder steps do not alter the regular reference trajectory. Surface
    normalization uses training-medium mass energy per physical area, without
    centering pressure or edge forms. Random orientation flips transform cochains,
    incidence and M1 consistently; model outputs map back to requested signs.

20. **Scientific readiness requires measured reference uncertainty.** MFEM operator
    and trajectory agreement, temporal convergence, sphere continuum verification,
    and multi-torus geometric/spectral self-convergence are distinct checks. A
    hashed protocol binds reviewed evidence, development identities, physical band
    thresholds and numerical settings. Unresolved bands cannot enter the spectral
    objective. See `SURFACE_EXPERIMENT.md` for executed measurements and remaining
    research gates; CPU smoke is not a scientific loss comparison.

21. **Exploratory campaigns retain failures and separate numerical evidence.**
    Campaign manifests, budgets and source identities are fixed before generation.
    A disjoint preselected audit covers familiar and shifted geometry distributions;
    it reports uncertainty without automatically approving scientific gates. Default
    execution is one scalar/mass seed, with explicit expansion to paired seeds and
    losses. Spectral expansion requires a supplied frozen protocol. Held-out scores
    never automatically alter training. The authoritative latest checkpoint contains
    the best model and history to recover interrupted epoch commits; evaluation
    caches are bound to checkpoint and prediction hashes. See the campaign commands
    in `SURFACE_EXPERIMENT.md`.

22. **Balanced MSE isolates global field weighting.** Fixed weights come only from
    medium training targets at positive regular times, with equal configuration,
    time and coefficient averaging. A common multiplier preserves the original
    MSE zero-prediction expectation; it does not equalize gradients or optimization.
    This is separate from the per-case balanced coupling diagnostic. The loss study
    imports immutable completed MSE/mass controls, preserves the original sampler
    and model, and selects checkpoints using the existing validation mass score.
    Its automatic evaluation is restricted to validation; a future generalization
    study requires a separate decision after fitting behavior is understood.

23. **Continuation comparisons pair optimizer state and additional budget.** Each
    balanced-MSE seed forks its original latest checkpoint into constant and cosine
    LR branches, preserving optimizer, RNG, sampler epoch and history. Cosine steps
    after optimizer updates across a fixed additional budget; its state is saved
    with the authoritative latest checkpoint. Original runs remain immutable.
    Best selection includes the parent history and explicitly labels inherited
    winners; final-budget checkpoints are evaluated separately. Both roles use
    train/validation diagnostics only, with no new OOD evaluation.
