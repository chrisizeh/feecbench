"""Persisted workflow contracts: resumability, incomplete-data guards, evidence binding."""

import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("pydec")
pytest.importorskip("skimage")

from feecbench.surface.corpus import generate_corpus
from feecbench.surface.data import atomic_json, digest, load, make_manifest, save, validate_manifest
from feecbench.surface.protocol import GATES, freeze_protocol, read_protocol
from feecbench.surface.reference_checks import check_references
from feecbench.surface.training import train


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("surface-corpus")
    manifest = make_manifest({"train": 4, "validation": 4}, seed=19, development=True)
    settings = {"grid": 12, "modes": 16, "dt": 0.01, "horizon": 0.1, "snapshots": 3}
    entries = generate_corpus(root, manifest, **settings)
    return root, manifest, entries, settings


def test_resumable_corpus_and_offgrid(corpus):
    root, manifest, entries, settings = corpus
    mtimes = [(root / e["path"]).stat().st_mtime_ns for e in entries]
    assert generate_corpus(root, manifest, **settings) == entries
    assert mtimes == [(root / e["path"]).stat().st_mtime_ns for e in entries]
    for entry in entries:
        sample = load(root / entry["path"])
        assert sample.ops.points.shape[1] == 3
        assert sample.metadata["geometry_source"]["level"] == 1
        assert sample.metadata["energy_drift"] < 1e-8
        assert bool(sample.offgrid.any()) == (entry["split"] != "train")
    with pytest.raises(ValueError, match="hash mismatch"):
        generate_corpus(root, manifest, **{**settings, "dt": 0.02})


def test_atomic_save_preserves_existing_file(corpus):
    root, _, entries, _ = corpus
    path = root / entries[0]["path"]
    sample, original = load(path), path.read_bytes()
    sample.metadata["unserializable"] = object()
    with pytest.raises(TypeError):
        save(path, sample)
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".pending-*"))


def test_manifest_rejects_forged_ids():
    manifest = make_manifest({"train": 4, "validation": 4}, seed=5)
    item = manifest["configurations"][4]
    item["geometry"] = manifest["configurations"][0]["geometry"].copy()
    item["config_id"] = digest({k: v for k, v in item.items() if k != "config_id"})[:20]
    with pytest.raises(ValueError, match="geometry content hash"):
        validate_manifest(manifest)


def test_training_rejects_partial_generation(tmp_path):
    atomic_json(tmp_path / "index.json", {"complete": False, "samples": [], "fingerprint": "partial"})
    with pytest.raises(ValueError, match="incomplete"):
        train(tmp_path, tmp_path / "output")


def test_reference_and_protocol_binding(corpus, tmp_path):
    root, _, _, _ = corpus
    report = check_references(root, tmp_path / "reference.json", thresholds=(1, 4), limit=1)
    assert not report["complete_corpus"]
    assert report["cases"][0]["initial_pressure_rmse"] < 1e-10
    assert report["cases"][0]["initial_velocity_rmse"] < 1e-10
    atomic_json(tmp_path / "review.json", {"notes": "test fixture reviewed attestation"})
    evidence = {}
    for gate in GATES:
        filename = "reference.json" if gate == "reference_uncertainty" else "review.json"
        evidence[gate] = {
            "passed": True,
            "report": filename,
            "sha256": hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest(),
        }
    atomic_json(tmp_path / "evidence.json", evidence)
    with pytest.raises(ValueError, match="does not cover"):
        freeze_protocol(root, tmp_path / "evidence.json", tmp_path / "protocol.json", thresholds=(1, 4))
    # A complete measured report can freeze settings, but changing thresholds invalidates it.
    report = check_references(root, tmp_path / "reference.json", thresholds=(1, 4))
    evidence["reference_uncertainty"]["sha256"] = hashlib.sha256((tmp_path / "reference.json").read_bytes()).hexdigest()
    atomic_json(tmp_path / "evidence.json", evidence)
    protocol = freeze_protocol(root, tmp_path / "evidence.json", tmp_path / "protocol.json", thresholds=(1, 4))
    assert read_protocol(tmp_path / "protocol.json") == protocol
    assert protocol["numerical_settings"]["grid"] == 12
    with pytest.raises(ValueError, match="does not cover"):
        freeze_protocol(root, tmp_path / "evidence.json", tmp_path / "protocol2.json", thresholds=(1, 5))
    bad = json.loads((tmp_path / "protocol.json").read_text())
    bad["numerical_settings"]["grid"] = 99
    atomic_json(tmp_path / "protocol.json", bad)
    with pytest.raises(ValueError, match="hash mismatch"):
        read_protocol(tmp_path / "protocol.json")
    assert np.isfinite(report["cases"][0]["temporal"]["low"]["mass_rmse"])


def test_reference_check_rejects_partial_index(corpus, tmp_path):
    root, _, _, _ = corpus
    index = json.loads((root / "index.json").read_text())
    index["complete"] = False
    atomic_json(tmp_path / "index.json", index)
    atomic_json(tmp_path / "manifest.json", json.loads((root / "manifest.json").read_text()))
    with pytest.raises(ValueError, match="complete corpus"):
        check_references(tmp_path, tmp_path / "reference.json")


def test_standalone_prediction_scoring(corpus, tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root, _, entries, _ = corpus
    sample_path = root / entries[0]["path"]
    sample = load(sample_path)
    predictions = tmp_path / "predictions.npz"
    output = tmp_path / "scores.json"
    np.savez(predictions, times=sample.times, pressure=sample.pressure, velocity=sample.velocity)
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/surface_benchmark.py"),
               "score", "--sample", str(sample_path), "--predictions", str(predictions), "--output", str(output)]
    subprocess.run(command, check=True)
    result = json.loads(output.read_text())
    assert result["metrics"]["pressure"]["mass_rmse"] == 0
    assert result["metrics"]["velocity"]["mass_rmse"] == 0
    assert result["sample_sha256"] == hashlib.sha256(sample_path.read_bytes()).hexdigest()
    np.savez(predictions, times=sample.times + 0.01, pressure=sample.pressure, velocity=sample.velocity)
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "prediction times must exactly match" in rejected.stderr
