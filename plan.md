# Unified Benchmark Review Follow-up

## Completion status — 10 November 2026

Completed and independently reviewed. All three findings below are resolved:

- Shared-profile surface evaluation selects the correct harmonic basis and honors
  explicit overrides and disabled Hodge metrics.
- Legacy index adaptation uses metadata without reading excluded sample arrays.
- Portable legacy fixtures are committed with provenance and checksums; baseline
  capture rejects the wrong source revision and existing destinations.

Implementation commits: `63dea5b` and `54cfba2` on `main`; integrated into the
experiment branch by `21d7e33`.

Final verification:

- Main: **86 passed, 1 skipped** (optional MFEM backend unavailable).
- Experiment branch: **134 passed, 3 skipped** (optional MFEM and two CUDA checks).
- Focused migration/review tests: **22 passed**, with access to the old machine-local
  reference directory blocked by a Python audit hook in the test process.
- Ruff and Git whitespace checks passed.

No outstanding findings from this review. GPU and optional MFEM execution were not
revalidated in this environment. The sections below preserve the completed work's
requirements and acceptance criteria; they are not pending implementation tasks.

## Purpose and current state

The benchmark unification is implemented and committed locally. Its overall
architecture matches the intended design: one shared sample, storage, dataset,
normalization, and evaluation interface, with specialized surface numerical solvers.
Do not redo the migration or restructure the repository.

This plan replaces the previous implementation plan. Address the three findings
below before considering the integration complete.

Review baseline:

- Local `main`: `40e1059` — shared benchmark integration.
- Local `experiment/acoustic-baseline`: `850f917` — integrated benchmark plus
  experiment compatibility changes.
- Main tests: 72 passed, 1 skipped.
- Experiment tests: 120 passed, 3 skipped.
- Ruff passed.
- Existing passing tests did not cover the two reproduced failures below, and some
  migration tests rely on local temporary reference files.

Verify current Git state before starting. Read `AGENTS.md` and
`feecbench/DECISIONS.md`. Preserve user changes and generated experiment artifacts.
Work on a fix branch from current local `main`, preferably in a separate worktree.
Keep model implementations on the experiment branch. Do not rewrite existing
commits, regenerate experiment corpora, or weaken checkpoint/source hash guards.

## 1. Fix shared-profile evaluation on surface samples

### Reproduced problem

In `feecbench/src/feecbench/evaluator.py`, this documented call fails:

```python
from feecbench import load_sample, evaluate_profile

sample = load_sample(surface_path)
pred = {name: field.data for name, field in sample.fields.items()
        if field.role == "target"}
evaluate_profile(pred, sample, profile="planar")
```

With the default `include_hodge=True`, the evaluator calls the original harmonic
basis routine, which now correctly rejects a non-diagonal pressure mass:

```text
ValueError: consistent star0 requires the surface harmonic backend
```

The numerical guard is correct. The evaluator's backend selection is wrong.

### Required implementation

1. Retain the existing default surface acoustic profile and all its metric meanings.
2. Make the explicit `profile="planar"` path work on surface samples as documented.
   This profile requests the original shared metrics; it must not force the use of
   planar numerical assumptions.
3. Preserve an explicitly supplied `harmonic` argument as the highest-priority basis.
4. Otherwise, use the stored surface harmonic basis from the sample's auxiliary
   arrays for surface samples. Validate its dimensions against the sample.
5. If a surface basis is missing, use the correct surface backend or raise a clear,
   specific error. Do not silently use a diagonal approximation or omit metrics.
6. Preserve `include_hodge=False`: no harmonic-basis calculation should occur.
7. Keep planar sample behavior and the consistent-mass guards unchanged.
8. Preserve fresh-process energy registration; evaluation must not depend on a
   conversion helper having been imported earlier.

Consider whether the existing public `harmonic_basis(sample)` helper should handle
surface samples centrally, so dispatch is not duplicated. Keep explicit-basis
precedence in `evaluate_profile` regardless of where backend dispatch lives.

### Acceptance tests

Use deterministic surface fixtures independent of external temporary directories:

- Exact surface predictions with `profile="planar"` return field, Hodge, and energy
  metrics successfully.
- Perturbed surface predictions agree with the correctly supplied-basis path.
- An explicit harmonic argument is honored.
- `include_hodge=False` avoids basis computation and omits Hodge metrics.
- Missing or incorrectly shaped auxiliary bases have the chosen documented behavior.
- Fresh-process shared-profile evaluation works without a prior adapter import.
- Existing planar and default surface-profile outputs remain unchanged.

Do not compare unlike surface/shared Hodge or closedness metrics as though their
normalization and time aggregation were identical.

## 2. Avoid eager corpus loading during legacy index adaptation

### Reproduced problem

`FEECDataset.__init__` currently calls `load_sample` for every legacy index entry
missing canonical identity fields, before applying split or other filters.

Consequences:

- `FEECDataset(root, split="train")` fails if an excluded held-out file is not
  downloaded, even when all training files are available.
- Dataset construction reads all trajectories, operators, and spectral arrays,
  making initialization unnecessarily expensive for large corpora.

The failure was reproduced using an index containing one available training sample
and one unavailable `id_test` sample.

### Required implementation

Primary file: `feecbench/src/feecbench/dataset.py`.

1. Normalize known legacy surface index fields without reading sample arrays:
   - `resolution` maps to `resolution_level` and `refinement` using the established
     migration convention.
   - Preserve an existing sample ID; otherwise derive the documented stable ID.
   - Preserve paths, configuration IDs, split labels, and extra provenance fields.
2. Share the identity-mapping logic with the compatibility adapter where practical,
   so dataset entries and loaded samples cannot disagree.
3. If file inspection is genuinely needed for a nonstandard entry, defer it until
   the entry is requested. Read metadata only where possible, not full arrays.
4. Apply filters using normalized index metadata. Support resolution/refinement
   filters without opening unrelated samples.
5. Keep the full normalized index for `get_config`; split filtering must not remove
   its ability to find other discretizations of a requested configuration.
6. Continue rejecting an index explicitly marked incomplete.
7. Preserve a useful error when the caller actually requests a missing file.
8. Do not rewrite legacy indexes or change corpus fingerprints as a side effect of
   constructing a dataset.

An unavailable file excluded by the requested view must not prevent construction
or access to available samples. This does not authorize ignoring an explicitly
incomplete corpus marker.

### Acceptance tests

- A legacy index with an available training sample and an unavailable held-out
  sample supports constructing and reading the training view.
- Constructing a dataset from ordinary legacy index metadata performs no full
  `load_sample` calls; use a spy or monkeypatch to verify this directly.
- Resolution and refinement filters work for coarse, medium, fine, and remeshed
  legacy entries.
- Derived IDs agree with `load_sample(...).ids`; explicit stored IDs are preserved.
- `get_config` uses the full index and retains nested/remeshed behavior.
- Requesting a missing selected sample raises a clear error.
- Explicitly incomplete indexes are still rejected.
- Legacy index bytes and fingerprints remain unchanged after dataset access.

## 3. Make migration regression tests portable and dependable

### Current problem

`feecbench/tests/test_unified_api.py` defines:

```python
REFERENCE = Path('/private/tmp/feecbench-unification-reference')
```

Five tests skip when `expected.json` is absent. These cover important migration
parity, dataset/scaling, planar compatibility, and fresh-process behavior. Therefore
a fresh checkout or CI can pass while failing to validate the migration.

`feecbench/scripts/capture_unification_baseline.py` also writes to that hard-coded
directory. Its instruction to run against the old revision is only a docstring;
there is no safeguard against accidentally replacing the baseline with outputs
from the migrated implementation.

### Required implementation

1. Remove hard-coded machine-specific paths from ordinary regression tests.
2. Make essential compatibility tests self-contained. Use deterministic fixtures
   and pytest temporary directories for generated working files.
3. Preserve independent pre-migration numerical references. Do not generate expected
   outputs using the current implementation under test.
4. Use compact committed fixture data under `feecbench/tests/` if practical, or a
   reproducible, pinned fixture-artifact mechanism available to CI. Do not commit
   a full generated corpus, large reports, or experiment outputs.
5. Include old planar 0.1.0 and old surface-0.2.0 representations so compatibility
   readers are actually exercised. A fixture saved by the new writer is not a
   substitute for a legacy-format fixture.
6. Record the baseline source revision (`21b97a7`), capture settings, and checksums
   for frozen artifacts. Keep the baseline immutable during normal test runs.
7. Update the capture script to accept an explicit output directory and verify that
   it imports the intended pre-migration implementation before capture. Fail on
   a wrong implementation or unsafe overwrite of an existing baseline.
8. Document the exact baseline creation and validation commands in the existing
   benchmark guide. Do not introduce another competing developer workflow.
9. Missing required fixture data must fail clearly, not silently skip core tests.
   Skips for genuinely optional numerical dependencies remain acceptable if reported.

### Acceptance tests

With the supported development/surface dependencies installed, a clean checkout
without `/private/tmp/feecbench-unification-reference` must execute the migration
checks successfully:

- Legacy planar load and new-format round trip.
- Legacy surface load and new-format round trip, including sparse operators and
  auxiliary arrays.
- Frozen planar and surface numerical parity for exact and perturbed predictions.
- Positive-time, regular-query, and off-grid surface subsets.
- Surface normalization parity and held-out-data exclusion.
- Fresh-process evaluation without accidental import-order dependencies.
- The evaluator and index-adaptation regressions from steps 1 and 2.

Use scientifically justified numerical tolerances where bit-exact floating-point
results are not portable. Require exact preservation for stored arrays, metadata,
and sparse representation where the persistence contract demands it.

## Validation and integration sequence

1. Add regression coverage for the reproduced failures before or alongside each fix.
2. Run focused evaluator, dataset, I/O, normalization, and migration tests.
3. Verify which worktree the Python interpreter imports. An editable installation
   can point at another checkout; use an explicit `PYTHONPATH` when needed.
4. Run the complete benchmark suite and Ruff from the integration worktree:

   ```sh
   cd feecbench
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   python -m ruff check .
   python -m pytest -q
   ```

5. Confirm the tests do not access the old machine-local reference directory. Do
   not delete existing user reference artifacts to test this; use an isolated
   test configuration or instrument path access.
6. Review changes for numerical-contract, identity, and fingerprint preservation.
7. Commit benchmark fixes in reviewable units. Merge validated fixes into local
   `main`, then merge `main` into the experiment branch without rewriting history.
8. Run the full experiment suite after integration. Preserve checkpoint behavior
   and source/corpus hash guards.
9. Report commits, test results, skip reasons, and any remaining limitations.
   Do not push remote branches unless requested.

## Completion criteria

- The documented shared evaluation profile works on surface samples.
- Legacy dataset construction is metadata-driven and does not load excluded data.
- Migration regression tests run reproducibly on a fresh checkout or CI.
- Existing planar and surface numerical results remain unchanged.
- Benchmark fixes live on `main`; experiment code remains on its branch.
- No corpus, fingerprint, checkpoint, or unrelated user file is silently changed.
