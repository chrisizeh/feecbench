"""Shared-profile and metadata-only dataset regressions using frozen legacy files."""
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from feecbench import FEECDataset, evaluate_profile, load_sample
from feecbench.evaluator import harmonic_basis
from feecbench.normalization import compute_stats
from feecbench.schema import assert_samples_equal

FIXTURES = Path(__file__).parent / 'fixtures/unification'


@pytest.fixture
def surface():
    return load_sample(next((FIXTURES / 'surface').glob('*/*.h5')))


def test_shared_surface_profile(surface):
    pred = {n: f.data for n, f in surface.fields.items() if f.role == 'target'}
    profile = evaluate_profile(pred, surface, profile='planar')
    assert {'fields', 'energy'} == set(profile)
    assert profile['fields']['u']['rel_l2_star'] == 0
    assert max(profile['fields']['u']['hodge_error'].values()) == 0
    rng = np.random.default_rng(29)
    pred = {n: x + .03*rng.normal(size=x.shape) for n, x in pred.items()}
    a = evaluate_profile(pred, surface, profile='planar')
    b = evaluate_profile(pred, surface, profile='planar', harmonic=surface.auxiliary['harmonic_basis'])
    assert a == b


def test_basis_precedence_and_errors(surface, monkeypatch):
    pred = {n: f.data for n, f in surface.fields.items() if f.role == 'target'}
    missing = replace(surface, auxiliary={})
    with pytest.raises(ValueError, match='requires auxiliary harmonic_basis'):
        harmonic_basis(missing)
    malformed = replace(surface, auxiliary={'harmonic_basis': np.zeros((1, 1))})
    with pytest.raises(ValueError, match='shape'):
        evaluate_profile(pred, malformed, profile='planar')
    # Explicit argument must win over malformed stored data.
    evaluate_profile(pred, malformed, profile='planar', harmonic=surface.auxiliary['harmonic_basis'])
    def fail(*args):
        raise AssertionError('basis computation must be bypassed')
    monkeypatch.setattr('feecbench.evaluator.harmonic_basis', fail)
    result = evaluate_profile(pred, missing, profile='planar', include_hodge=False)
    assert 'hodge_error' not in result['fields']['u']
    evaluate_profile(pred, missing, profile='planar', harmonic=surface.auxiliary['harmonic_basis'])


@pytest.fixture
def legacy_index(tmp_path):
    source = next((FIXTURES / 'surface').glob('*/*.h5'))
    sample = load_sample(source)
    shutil.copyfile(source, tmp_path / 'available.h5')
    entries = [dict(path='available.h5', config_id=sample.ids.config_id, split='train', resolution='medium', extra=42),
               dict(path='absent.h5', config_id='heldout', split='id_test', resolution='fine')]
    path = tmp_path / 'index.json'
    path.write_text(json.dumps(dict(samples=entries, fingerprint='unchanged', complete=True)))
    return tmp_path, sample, path.read_bytes()


def test_index_is_metadata_only(legacy_index, monkeypatch):
    root, sample, before = legacy_index
    def fail(*args):
        raise AssertionError('constructor must not load arrays')
    with monkeypatch.context() as scoped:
        scoped.setattr('feecbench.dataset.load_sample', fail)
        ds = FEECDataset(root, split='train')
    assert ds.entries[0]['sample_id'] == sample.ids.sample_id
    assert ds.entries[0]['extra'] == 42
    assert_samples_equal(ds.load_full(0), sample)
    assert ds.get_config(sample.ids.config_id)['medium'].ids == sample.ids
    missing = FEECDataset(root, split='id_test')
    with pytest.raises(FileNotFoundError):
        missing.load_full(0)
    assert (root / 'index.json').read_bytes() == before


@pytest.mark.parametrize('label,level,refinement', [('coarse', 'coarse', 'nested'), ('medium', 'medium', 'nested'),
                                                  ('fine', 'fine', 'nested'), ('remeshed', 'medium', 'remeshed')])
def test_legacy_filters(legacy_index, label, level, refinement):
    root, sample, _ = legacy_index
    entries = [dict(path=f'{x}.h5', config_id='config', split='train', resolution=x)
               for x in ('coarse', 'medium', 'fine', 'remeshed')]
    (root / 'index.json').write_text(json.dumps(dict(samples=entries)))
    ds = FEECDataset(root, filters=dict(resolution_level=level, refinement=refinement))
    assert len(ds) == 1
    assert ds.entries[0]['sample_id'] == f'config-{label}'


def test_explicit_identity_and_full_index(legacy_index):
    root, sample, _ = legacy_index
    from feecbench.surface.compat import from_feec, to_feec
    original = from_feec(sample)
    original.metadata['sample_id'] = 'explicit'
    assert to_feec(original).ids.sample_id == 'explicit'
    entries = [dict(path='available.h5', config_id=sample.ids.config_id, sample_id='explicit',
                    split=split, resolution=label) for split, label in [('train', 'coarse'), ('id_test', 'medium'),
                                                                       ('id_test', 'remeshed')]]
    (root / 'index.json').write_text(json.dumps(dict(samples=entries)))
    ds = FEECDataset(root, split='train')
    assert ds.entries[0]['sample_id'] == 'explicit'
    assert set(ds.get_config(sample.ids.config_id)) == {'coarse', 'medium'}
    assert set(ds.get_config(sample.ids.config_id, 'remeshed')) == {'explicit'}


def test_heldout_exclusion(surface):
    heldout = replace(surface, ids=replace(surface.ids, split='id_test'), fields={
        n: replace(f, data=f.data*1e8) for n, f in surface.fields.items()})
    before = compute_stats([surface], policy='surface_energy')
    assert before == compute_stats([surface, heldout], policy='surface_energy')


def test_capture_rejects_migrated_source(tmp_path):
    import subprocess
    import sys

    output = tmp_path / 'must-not-exist'
    script = Path(__file__).parents[1] / 'scripts/capture_unification_baseline.py'
    result = subprocess.run([sys.executable, str(script), str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'baseline capture requires 21b97a7' in result.stderr
    assert not output.exists()


def test_legacy_versions():
    import h5py

    with h5py.File(next((FIXTURES / 'planar').glob('*/*.h5'))) as stream:
        assert stream.attrs['schema_version'] == '0.1.0'
    with h5py.File(next((FIXTURES / 'surface').glob('*/*.h5'))) as stream:
        assert stream.attrs['version'] == 'surface-0.2.0'


def test_capture_refuses_existing_destination(tmp_path):
    import runpy

    script = Path(__file__).parents[1] / 'scripts/capture_unification_baseline.py'
    namespace = runpy.run_path(str(script))
    capture = namespace['capture']
    capture.__globals__['verify_source'] = lambda: '21b97a7'
    sentinel = tmp_path / 'sentinel'
    sentinel.write_text('unchanged')
    with pytest.raises(FileExistsError):
        capture(tmp_path)
    assert sentinel.read_text() == 'unchanged'


def test_nonstandard_index_identity_is_deferred(legacy_index, monkeypatch):
    root, sample, _ = legacy_index
    (root / 'index.json').write_text(json.dumps(dict(samples=[
        dict(path='available.h5', config_id=sample.ids.config_id, split='train'),
        dict(path='missing.h5', config_id='other', split='id_test')])))
    ds = FEECDataset(root, split='train')
    def fail(*args):
        raise AssertionError('metadata resolution must not load sample arrays')
    # Inspection is confined to identity metadata for the requested configuration.
    with monkeypatch.context() as scoped:
        scoped.setattr('feecbench.dataset.load_sample', fail)
        with pytest.raises(AssertionError, match='metadata resolution'):
            ds.get_config(sample.ids.config_id)
    assert ds._all_entries[0]['sample_id'] == sample.ids.sample_id
    assert ds.get_config(sample.ids.config_id)['medium'].ids.sample_id == sample.ids.sample_id
