"""Frozen corpus, checkpoint milestones, paired summaries and safe resumption."""

import json
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("pydec")
pytest.importorskip("skimage")

from feecbench.surface import transport_study as study
from feecbench.surface.corpus import generate_corpus
from feecbench.surface.data import atomic_json, make_manifest


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("transport-study-data")
    manifest = make_manifest({"train": 4, "validation": 4, "id_test": 4}, seed=703, development=True)
    generate_corpus(root, manifest, grid=12, modes=8, dt=.02, horizon=.12, snapshots=4)
    return root


def test_prepare_binds_data_sources_budgets_and_architecture(corpus, tmp_path, monkeypatch):
    output = tmp_path / "study"
    config = study.prepare_transport_study(corpus, output)
    assert config["budgets"] == [200, 400]
    assert config["parameters"] == {"none": 681602, "d0_delta1": 814722}
    assert config["objectives"] == ["mse_balanced", "mass"]
    assert config["seeds"] == [0, 1, 2]
    assert config["corpus"]["counts"] == {"train": 4, "validation": 4, "id_test": 4}
    assert study.read_transport_study(output) == config
    assert study.prepare_transport_study(corpus, output) == config
    with pytest.raises(ValueError, match="changed"):
        study.prepare_transport_study(corpus, output, epochs=600)
    with pytest.raises(ValueError, match="budgets"):
        study.prepare_transport_study(corpus, tmp_path / "bad", epochs=200, first_budget=200)
    with pytest.raises(ValueError, match="separate"):
        study.prepare_transport_study(corpus, corpus / "study")
    monkeypatch.setattr(study, "_sources", lambda: {"changed": True})
    with pytest.raises(ValueError, match="source/lockfile"):
        study.read_transport_study(output)


def test_source_mutation_and_split_forgery_rejected(corpus, tmp_path, monkeypatch):
    output = tmp_path / "study"
    study.prepare_transport_study(corpus, output, smoke=True, epochs=2, first_budget=1)
    original_hash = study.file_hash
    monkeypatch.setattr(study, "file_hash", lambda p: "changed" if str(p).endswith(".h5") else original_hash(p))
    with pytest.raises(ValueError, match="source corpus changed"):
        study.read_transport_study(output)
    monkeypatch.setattr(study, "file_hash", original_hash)
    original_read = study._read

    def forged(path):
        obj = original_read(path)
        if Path(path).name == "index.json":
            obj["samples"][0]["split"] = "validation"
        return obj

    monkeypatch.setattr(study, "_read", forged)
    with pytest.raises(ValueError, match="split differs"):
        study.prepare_transport_study(corpus, tmp_path / "forged", smoke=True, epochs=2, first_budget=1)


def test_summary_pairs_only_matching_cells():
    rows = []
    for budget in (200, 400):
        for transport in ("none", "d0_delta1"):
            for objective in ("mass", "mse_balanced"):
                for seed in (0, 1, 2):
                    value = 1 + seed + (budget == 400) * -.1 + (transport == "d0_delta1") * -.3
                    value += (objective == "mass") * .2
                    rows.append({"transport": transport, "objective": objective, "budget": budget,
                                 "seed": seed, "metrics": {"velocity": value}})
    summary = study._summaries(rows, ("transport", "objective", "budget"))
    target = next(r for r in summary if r["transport"] == "d0_delta1" and r["objective"] == "mass"
                  and r["budget"] == 400)["metrics"]["velocity"]
    assert target["paired_mean"] == pytest.approx(-.3)
    assert [p["value"] for p in target["paired_minus_earlier_budget"]] == pytest.approx([-.1] * 3)
    assert [p["value"] for p in target["paired_minus_balanced_mse"]] == pytest.approx([.2] * 3)
    with pytest.raises(ValueError, match="duplicate"):
        study._summaries(rows + [rows[0]], ("transport", "objective", "budget"))


def test_resume_milestones_and_explicit_heldout(corpus, tmp_path, monkeypatch):
    output = tmp_path / "study"
    study.prepare_transport_study(corpus, output, smoke=True, epochs=2, first_budget=1)
    # Exercise the real train/evaluate pipeline with one pair; the full 12-run
    # smoke is also runnable via the CLI. No mocked checkpoints or metric files.
    monkeypatch.setattr(study, "_combinations", lambda spec: iter((t, "mse_balanced", 0) for t in study.TRANSPORTS))
    with pytest.raises(ValueError, match="completed validation suite"):
        study.run_transport_study(output, device="cpu", stage="heldout")
    evaluate = study.evaluate
    observed_splits = []

    def interrupted(*args, **kwargs):
        raise RuntimeError("evaluation interruption")

    monkeypatch.setattr(study, "evaluate", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        study.run_transport_study(output, device="cpu", stage="pilot")
    snapshot = output / "runs/none-mse_balanced-seed0/epoch-1"
    original = {p.name: study.file_hash(p) for p in snapshot.iterdir() if p.is_file()}

    def tracked(*args, **kwargs):
        observed_splits.append(kwargs["splits"])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(study, "evaluate", tracked)
    report = study.run_transport_study(output, device="cpu", stage="suite")
    assert report["completed_validation_records"] == 8
    assert observed_splits and all(s == ["validation"] for s in observed_splits)
    assert {name: study.file_hash(snapshot / name) for name in original} == original
    fields = {r["direction"] for r in report["fitting"]}
    assert {"pressure_to_velocity", "velocity_to_pressure"} <= fields
    assert {r["budget"] for r in report["evaluation"]} == {1, 2}
    assert {r["role"] for r in report["evaluation"]} == {"best", "final"}
    assert all("harmonic_velocity" in r["metrics"] for r in report["evaluation"])
    latest = output / "runs/none-mse_balanced-seed0/latest.pt"
    latest_hash = study.file_hash(latest)
    # A corrupted prediction is repaired, while completed training is reused.
    predicted = next((snapshot / "best/validation/evaluation/predictions").rglob("*.npz"))
    expected = study.file_hash(predicted)
    predicted.write_bytes(b"broken")
    study.run_transport_study(output, device="cpu", stage="pilot")
    assert study.file_hash(predicted) == expected
    assert study.file_hash(latest) == latest_hash
    study.run_transport_study(output, device="cpu", stage="heldout")
    assert ["id_test"] in observed_splits
    assert study.file_hash(latest) == latest_hash
    assert {r["phase"] for r in study.summarize_transport_study(output)["evaluation"]} == {"validation", "heldout"}
    with zipfile.ZipFile(study.export_transport_study(output)) as archive:
        assert "analysis.json" in archive.namelist()
        assert not any(p.endswith((".pt", ".h5", ".npz")) for p in archive.namelist())
    # Saved checkpoint mutation cannot silently reuse a complete evaluation.
    (snapshot / "best.pt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        study.summarize_transport_study(output)
    progress = json.loads((output / "progress.json").read_text())
    assert all(r["state"] == "complete" for r in progress.values())


def test_reject_forged_specification(corpus, tmp_path):
    output = tmp_path / "study"
    config = study.prepare_transport_study(corpus, output, smoke=True, epochs=2, first_budget=1)
    config["budgets"][-1] = 3
    atomic_json(output / "study.json", config)
    with pytest.raises(ValueError, match="configuration hash"):
        study.read_transport_study(output)
