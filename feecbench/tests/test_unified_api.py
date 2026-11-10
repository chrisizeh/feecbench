"""Regression checks against frozen pre-migration samples and numerical results."""
import json
from pathlib import Path

import numpy as np
import pytest

from feecbench import FEECDataset, assert_samples_equal, collate_feec, evaluate_profile, load_sample, save_sample
from feecbench.normalization import Normalizer, compute_stats
from feecbench.surface.compat import from_feec
from feecbench.surface.data import load

REFERENCE = Path('/private/tmp/feecbench-unification-reference')


@pytest.fixture
def reference():
    if not (REFERENCE / 'expected.json').exists():
        pytest.skip('run the documented baseline capture before numerical parity checks')
    return json.loads((REFERENCE / 'expected.json').read_text())


def test_surface_reference_parity(reference, tmp_path):
    for relative, expected in reference['surface_scores'].items():
        path = REFERENCE / relative
        sample = load_sample(path)
        source = load(path)
        assert sample.solver_meta.cfl is None
        assert sample.topology.betti2 == 1
        assert_samples_equal(sample, load_sample(save_sample(sample, tmp_path)))
        np.testing.assert_array_equal(from_feec(sample).pressure, source.pressure)
        rng = np.random.default_rng(31)
        actual = []
        for noise in (0, .03):
            for selected in (source.times > 0, (source.times > 0) & ~source.offgrid, source.offgrid):
                if selected.any():
                    pred = dict(p=source.pressure + noise*rng.normal(size=source.pressure.shape),
                                u=source.velocity + noise*rng.normal(size=source.velocity.shape))
                    actual.append(evaluate_profile(pred, sample, profile='surface_acoustic', selected=selected))
        assert actual == expected


def test_surface_dataset_and_scales(reference):
    ds = FEECDataset(REFERENCE / 'surface', split='train')
    batch = collate_feec([ds[0], ds[1]])
    assert len(batch.auxiliary) == 2
    stats = compute_stats((ds.load_full(i) for i in range(len(ds))), policy='surface_energy')
    for name, long in [('p', 'pressure'), ('u', 'velocity')]:
        assert stats['fields'][name]['rms'] == reference['surface_scales'][long]
    normalizer = Normalizer(stats, 'rms')
    x = ds.load_full(0).fields['p'].data
    np.testing.assert_allclose(normalizer.denormalize('p', normalizer.normalize('p', x)), x, atol=1e-14)
    with pytest.raises(ValueError, match='rms'):
        Normalizer(stats, 'zscore')


def test_atomic_failure(square_samples, tmp_path, monkeypatch):
    import feecbench.io as io

    sample = square_samples['coarse']
    path = save_sample(sample, tmp_path)
    before = path.read_bytes()
    def fail(*args):
        raise RuntimeError('injected write failure')
    monkeypatch.setattr(io, '_write_sample', fail)
    with pytest.raises(RuntimeError):
        save_sample(sample, tmp_path)
    assert path.read_bytes() == before
    assert not list(path.parent.glob('.pending-*'))


def test_incomplete_index(tmp_path):
    (tmp_path / 'index.json').write_text('{"complete": false, "samples": []}')
    with pytest.raises(ValueError, match='incomplete'):
        FEECDataset(tmp_path)


def test_query_rejection(square_samples, tmp_path):
    from feecbench import write_index

    for s in square_samples.values():
        save_sample(s, tmp_path)
    write_index(tmp_path)
    ds = FEECDataset(tmp_path)
    a, b = ds[0], ds[1]
    b['fields']['p'] = b['fields']['p'][:-1]
    with pytest.raises(ValueError, match='query lengths'):
        collate_feec([a, b])


def test_planar_reference(reference):
    path = next((REFERENCE / 'planar').glob('*/*.h5'))
    sample = load_sample(path)
    assert compute_stats([sample]) == reference['planar_stats']
    rng = np.random.default_rng(31)
    for noise, expected in zip((0, .03), reference['planar_scores'], strict=True):
        pred = {n: f.data + noise*rng.normal(size=f.data.shape)
                for n, f in sample.fields.items() if f.role == 'target'}
        assert_metrics_close(evaluate_profile(pred, sample), expected)


def test_fresh_process(reference):
    import subprocess
    import sys

    path = next((REFERENCE / 'surface').glob('*/*.h5'))
    subprocess.run([sys.executable, '-c',
                    'from feecbench import load_sample,evaluate_profile; import sys; '
                    's=load_sample(sys.argv[1]); '
                    'r=evaluate_profile({n:f.data for n,f in s.fields.items() if f.role=="target"},s); '
                    'assert "energy" in r', str(path)], check=True)


def test_unknown_version(square_samples, tmp_path):
    import h5py

    path = save_sample(square_samples['coarse'], tmp_path)
    with h5py.File(path, 'r+') as stream:
        stream.attrs['schema_version'] = 'unknown'
    with pytest.raises(ValueError, match='unsupported schema'):
        load_sample(path)


def assert_metrics_close(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            assert_metrics_close(actual[key], expected[key])
    elif isinstance(expected, (float, int, list, tuple)):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
    else:
        assert actual == expected
