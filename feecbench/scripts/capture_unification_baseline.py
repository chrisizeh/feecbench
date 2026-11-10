"""Capture compact immutable legacy fixtures using only the pinned pre-migration source."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BASELINE = "21b97a7"


def verify_source():
    import feecbench

    source = Path(feecbench.__file__).resolve().parents[3]
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    pinned = subprocess.check_output(["git", "-C", str(source), "rev-parse", BASELINE], text=True).strip()
    if revision != pinned:
        raise RuntimeError(f"baseline capture requires {BASELINE}; imported {feecbench.__file__} at {revision}")
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--", "feecbench/src"], check=True)
    return revision


def capture(output):
    revision = verify_source()
    # Refuse even an empty existing directory: baseline replacement is always explicit.
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np

    from feecbench import ToyAcousticOracle, evaluate_profile, save_sample
    from feecbench.normalization import compute_stats
    from feecbench.surface.data import SurfaceSample, save, training_scales
    from feecbench.surface.geometry import Geometry, generate
    from feecbench.surface.metrics import score
    from feecbench.surface.operators import assemble, harmonic_basis
    from feecbench.surface.oracle import Midpoint, energy, pressure_modes

    config = dict(config_id="baseline", geometry="unit_square", h_coarse=.4, T=.1, n_snapshots=3,
                  seed=1, ic=dict(type="gaussian", x0=[.5, .5], width=.15))
    sample = ToyAcousticOracle().generate(config, "coarse")
    sample.ids.split = "train"
    save_sample(sample, output / "planar")
    result = {"planar_stats": compute_stats([sample]), "planar_scores": []}
    rng = np.random.default_rng(31)
    for noise in (0, .03):
        result["planar_scores"].append(evaluate_profile(
            {n: f.data + noise*rng.normal(size=f.data.shape) for n, f in sample.fields.items() if f.role == "target"},
            sample))
    entries, paths = [], []
    for grid in (12, 14):
        points, tris, source = generate(Geometry(), grid)[0]
        op = assemble(points, tris, 1)
        values, vectors = pressure_modes(op, 8)
        harmonic = harmonic_basis(op)
        times = np.array([0., .037, .1, .163, .2])
        p, u = Midpoint(op).solve(vectors[:, 1] + .1, op.d0 @ vectors[:, 2] + harmonic[:, 0], times, .01)
        e = energy(op, p, u)
        meta = dict(config_id=f"baseline-{grid}", resolution="medium", split="train", seed=grid,
                    dt=.01, solver="implicit_midpoint", integration_seconds=0.,
                    energy_drift=float(np.max(np.abs(e-e[0]))/e[0]), geometry_source=source, coefficient=1.)
        sample = SurfaceSample(op, times, p, u, values, vectors, harmonic, meta,
                               np.array([False, True, False, True, False]))
        relative = f"surface/{meta['config_id']}/medium.h5"
        path = output / relative
        save(path, sample)
        paths.append(path)
        entries.append(dict(path=str(Path(relative).relative_to('surface')), **meta))
        rng = np.random.default_rng(31)
        result.setdefault("surface_scores", {})[relative] = [
            score(sample, p+noise*rng.normal(size=p.shape), u+noise*rng.normal(size=u.shape), selected=selected)
            for noise in (0, .03)
            for selected in (times > 0, (times > 0) & ~sample.offgrid, sample.offgrid)]
    result["surface_scales"] = training_scales(paths)
    (output / "surface/index.json").write_text(json.dumps(dict(samples=entries, complete=True,
                                                               fingerprint="frozen-legacy-reference"), indent=2))
    (output / "expected.json").write_text(json.dumps(result, indent=2))
    checksums = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(output.rglob('*')) if p.is_file()}
    (output / "capture.json").write_text(json.dumps(dict(
        revision=revision, planar_config=config, surface_settings=dict(grids=[12, 14], modes=8, dt=.01,
        times=[0., .037, .1, .163, .2], noise_seed=31, noise=[0., .03]), sha256=checksums), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory; existing directories are never overwritten")
    capture(parser.parse_args().output)
