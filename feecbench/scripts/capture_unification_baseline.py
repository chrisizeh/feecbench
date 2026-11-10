"""Run against revision 21b97a7 to freeze migration references (never update from migrated code)."""

import json
from pathlib import Path

import numpy as np

from feecbench import ToyAcousticOracle, evaluate_profile, save_sample
from feecbench.normalization import compute_stats
from feecbench.surface.corpus import generate_corpus
from feecbench.surface.data import load, make_manifest, training_scales
from feecbench.surface.metrics import score

root = Path("/private/tmp/feecbench-unification-reference")
root.mkdir(exist_ok=True)
s = ToyAcousticOracle().generate(
    dict(
        config_id="baseline",
        geometry="unit_square",
        h_coarse=0.4,
        T=0.1,
        n_snapshots=3,
        seed=1,
        ic=dict(type="gaussian", x0=[0.5, 0.5], width=0.15),
    ),
    "coarse",
)
s.ids.split = "train"
save_sample(s, root / "planar")
result = {"planar_stats": compute_stats([s]), "planar_scores": []}
rng = np.random.default_rng(31)
for noise in (0, 0.03):
    result["planar_scores"].append(
        evaluate_profile(
            {k: f.data + noise * rng.normal(size=f.data.shape) for k, f in s.fields.items() if f.role == "target"}, s
        )
    )
manifest = make_manifest({"train": 4, "id_test": 4}, seed=91, development=True)
generate_corpus(root / "surface", manifest, grid=12, dt=0.01, horizon=0.2, snapshots=5, modes=8, paired=True)
paths = sorted((root / "surface").glob("*/*.h5"))
result["surface_scales"] = training_scales(paths)
result["surface_scores"] = {}
for path in paths:
    s = load(path)
    rng = np.random.default_rng(31)
    result["surface_scores"][str(path.relative_to(root))] = [
        score(
            s,
            s.pressure + n * rng.normal(size=s.pressure.shape),
            s.velocity + n * rng.normal(size=s.velocity.shape),
            selected=sel,
        )
        for n in (0, 0.03)
        for sel in (s.times > 0, (s.times > 0) & ~s.offgrid, s.offgrid)
        if sel.any()
    ]
(root / "expected.json").write_text(json.dumps(result))
