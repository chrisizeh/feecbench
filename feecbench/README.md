# feecbench — FEEC-native PDE benchmark prototype

A typed sample schema that keeps every field's **form degree** (0/1/2-cochains on
vertices/edges/triangles), **orientation**, and the **sparse operators** (`d0`, `d1`,
`star0..2`) of the discrete de Rham complex — plus a toy DEC acoustic oracle, exact HDF5
round-trip storage, a PyTorch dataset with variable-size-mesh batching and star-weighted
normalization, a metric-profile evaluator with DEC-specific metrics, and a small pilot
corpus with OOD splits and a measured cost model.

Pilot physics: the 2D acoustic wave (pressure `p` on vertices, velocity `u` on edges) on
**three geometry families** meshed by **gmsh (OpenCASCADE)** with curved boundaries —
`unit_square` (simply connected), `annulus` (one hole, β₁ = 1), `multi_hole` (square with
two circular holes, β₁ = 2) — each configuration at three nested resolutions (gmsh coarse
mesh + two midpoint refinements with an exact prolongation map), plus independently
remeshed variants.

## Install

```bash
cd feecbench
python -m pip install -e ".[dev]"                       # installs gmsh from PyPI as well
```

gmsh ships manylinux wheels with no system dependencies except `libGLU.so.1`. If
`import gmsh` fails with that name, add it to the env (`mamba install -c conda-forge
libglu`) or use `pip install --user gmsh` with a system libGLU. Without a working gmsh the
package still imports (datasets/evaluator work on saved corpora) but meshing raises an
`ImportError` that names the fix — there is no hand-written fallback.

## Quickstart — the three golden-path calls

```python
from feecbench import FEECDataset, collate_feec, evaluate_profile
ds = FEECDataset("pilot_data", split="train", normalize="zscore")
batch = collate_feec([ds[i] for i in idxs]).to("cuda")   # cpu when no GPU; test on cpu
profile = evaluate_profile(model_predictions, sample)
```

`model_predictions` is a dict of arrays keyed like the sample's target fields (`p`, `u`),
`sample = ds.load_full(i)` is the raw `FEECSample`. `tests/test_golden_path.py` runs these
lines verbatim on a mini corpus.

Everything else, end to end:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # see DECISIONS.md #14
python -m ruff check . && python -m pytest -q          # lint + acceptance tests (~1 min)
python scripts/build_pilot.py                       # pilot corpus (84 samples, ~2 min)
python scripts/profile_pilot.py                     # cost model -> reports/cost_model.md
```

Generating samples directly:

```python
from feecbench import ToyAcousticOracle
oracle = ToyAcousticOracle()
sample = oracle.generate({"config_id": "demo", "geometry": "annulus"}, "coarse")
print(sample.summary())
remeshed = oracle.generate({"config_id": "demo", "geometry": "annulus"}, "medium", remesh_seed=0)
```

## Layout

```
feecbench/
  pyproject.toml            PEP 621 metadata, deps (incl. gmsh), [dev] extra, ruff + pytest config
  src/feecbench/
    __init__.py             curated public API
    schema.py               FEECSample + block dataclasses, schema version, generic equality
    mesh/gmsh_backend.py    OpenCASCADE geometry for the three families, gmsh meshing,
                            boundary physical group, determinism, remeshing
    mesh/refine.py          midpoint refinement (+ re-projection onto CAD circles), prolongation
    mesh/quality.py         areas, edge lengths, quality stats
    operators.py            oriented complex, d0/d1 (int8, d1@d0 = 0), stars, derived operators,
                            harmonic basis + Hodge split
    oracle.py               Oracle protocol, ToyAcousticOracle (leapfrog, CFL on c_max,
                            exact staggered energy), acoustic energy functional
    sampler.py              deterministic per-family configuration sampler
    io.py                   HDF5 save/load (bit-exact), index.json
    normalization.py        ⋆-weighted train-split stats, zscore/rms normalizer
    dataset.py              FEECDataset (split/filters/normalize), collate_feec, FEECBatch
    evaluator.py            metric profile, per-PDE energy registry, per-sample caches
    viz.py                  mesh / 0-cochain / 1-cochain plots
  scripts/                  build_pilot.py, profile_pilot.py
  tests/                    ported acceptance tests + gmsh tests + golden-path test
  demo/                     pilot_data output location (generated at build time; git-ignored)
  reports/                  cost_model.md, cost_scaling.png (generated)
```

## Conventions (short)

Edges are sorted pairs oriented low → high index; triangles are CCW; `d0 (n1×n0)` has −1/+1
per edge, `d1 (n2×n1)` ±1 per edge (sign = agreement with the CCW traversal); both int8
with `d1 @ d0 == 0` exactly. `star0` = lumped P1 mass, `star1` = Whitney 1-form Galerkin
mass (SPD), `star2` = triangle areas. Derived operators (`delta1`, `Delta0`, `Delta1`) are
rebuilt from these, never stored. All evaluator norms are `sqrt(zᵀ ⋆ₖ z)`.

## The pilot corpus and its splits

`scripts/build_pilot.py` builds 26 configurations (8 per family + 2 geometry-OOD annuli;
deterministic in `(family, seed, index)`), each at coarse/medium/fine nested resolutions,
plus 6 independently remeshed medium variants — 84 samples. Splits are assigned at the
**configuration** level and written to `splits.json`, each sample, and `index.json`:

| split | contents | axis tested |
|---|---|---|
| `train` | 5 configs/family, unit_square + annulus | — |
| `val` | 1 config/family, unit_square + annulus | in-distribution |
| `test_id` | 2 configs/family, unit_square + annulus | in-distribution |
| `test_geom_ood` | 2 annulus configs, hole radius 0.5–0.55 (train: 0.25–0.45) | unseen geometry params |
| `test_topo_ood` | the entire multi_hole family (never trained on) | unseen topology (β₁ = 2) |
| `test_mesh_ood` | remeshed variants (sample-level rule) | unseen connectivity |

`normalize="zscore"|"rms"` applies the ⋆-weighted train-split statistics from `stats.json`;
`dataset.denormalize` inverts them and the evaluator always receives raw fields.

## Cost model

`reports/cost_model.md` holds measured numbers (median of 3 single-core runs per
family/level including gmsh meshing; sizes from the built corpus) and extrapolations to
1 000 and 10 000 configurations.

See `DECISIONS.md` for the design choices and conventions.

## Closed-surface benchmark

See [SURFACE_BENCHMARK.md](SURFACE_BENCHMARK.md) for standalone data generation,
raw-unit prediction scoring and numerical validation. Models remain on the
`experiment/acoustic-baseline` branch.
