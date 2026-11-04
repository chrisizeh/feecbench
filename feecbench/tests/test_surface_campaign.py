"""Campaign identities, split separation, failure recovery and analysis grouping."""

import json

import pytest

pytest.importorskip("pydec")
pytest.importorskip("skimage")

from feecbench.surface import campaign
from feecbench.surface.data import atomic_json


def test_campaign_fixed_inputs_and_disjoint_audit(tmp_path, monkeypatch):
    config = campaign.prepare_campaign(tmp_path / "run")
    assert config["updates_per_run"] == 25600
    assert config["objectives"] == ["mass", "mse"]
    assert len(config["manifest"]["configurations"]) == 768
    assert len(config["audit_manifest"]["configurations"]) == 24
    main = {c["geometry_id"] for c in config["manifest"]["configurations"]}
    audit = config["audit_manifest"]["configurations"]
    assert not main & {c["geometry_id"] for c in audit}
    assert {(c["split"], c["geometry"]["genus"]) for c in audit} == {
        ("id_test", 1),
        ("id_test", 2),
        ("geometry_ood", 1),
        ("geometry_ood", 2),
        ("topology_ood", 3),
    }
    assert campaign.prepare_campaign(tmp_path / "run") == config
    with pytest.raises(ValueError, match="mismatch"):
        campaign.prepare_campaign(tmp_path / "run", epochs=300)
    monkeypatch.setattr(campaign, "source_identity", lambda: {"changed": "source"})
    with pytest.raises(ValueError, match="source/lockfile"):
        campaign.read_campaign(tmp_path / "run")


def test_campaign_config_tampering(tmp_path):
    config = campaign.prepare_campaign(tmp_path / "run")
    config["training"]["epochs"] = 1
    atomic_json(tmp_path / "run" / "campaign.json", config)
    with pytest.raises(ValueError, match="configuration hash"):
        campaign.read_campaign(tmp_path / "run")


def test_campaign_recovers_after_evaluation_failure(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    campaign.prepare_campaign(directory, smoke=True)
    trained, evaluated = [], []

    def generate(root, manifest, **kwargs):
        atomic_json(root / "index.json", {"complete": True})

    def reference(root, output, **kwargs):
        atomic_json(output, {"cases": [], "interpretation": "test"})

    def train(root, run, **kwargs):
        trained.append((kwargs["objective"], kwargs["seed"]))
        for name in ("best.pt", "latest.pt", "history.json", "configuration.json"):
            atomic_json(run / name, {"seed": kwargs["seed"]})

    def evaluate(root, checkpoint, output, **kwargs):
        evaluated.append(output)
        if len(evaluated) == 1:
            raise RuntimeError("simulated interruption")
        rows = []
        for family, value in (("pulse", 0.1), ("exact_velocity", 0.9)):
            for method, factor in (("neural", 1), ("zero", 2)):
                rows.append(
                    {
                        "method": method,
                        "split": "id_test",
                        "resolution": "medium",
                        "queries": "regular",
                        "family": family,
                        "genus": 1,
                        "config_id": family,
                        "metrics": {f: {"normalized_rmse": value * factor} for f in ("pressure", "velocity")},
                    }
                )
        atomic_json(output / "per_sample.json", rows)
        atomic_json(output / "summary.json", {})
        atomic_json(output / "residuals.json", {})

    def inspect(root, run, output, **kwargs):
        atomic_json(output, {"source_checkpoint_sha256": campaign.file_hash(run / "best.pt")})

    for name, function in (
        ("generate_corpus", generate),
        ("check_references", reference),
        ("train", train),
        ("evaluate", evaluate),
        ("inspect_checkpoint", inspect),
    ):
        monkeypatch.setattr(campaign, name, function)
    with pytest.raises(RuntimeError, match="interruption"):
        campaign.run_campaign(directory, device="cpu")
    assert not (directory / "runs/mass-seed0/complete.json").exists()
    report = campaign.run_campaign(directory, device="cpu")
    assert trained == [("mass", 0)]  # Training checkpoint survives the failed evaluation.
    assert len(report["runs"]) == 1
    assert {r["family"] for r in report["summary"]} == {"all", "pulse", "exact_velocity"}
    assert {r["method"] for r in report["summary"]} == {"neural", "zero"}
    total = next(r for r in report["summary"] if r["family"] == "all" and r["method"] == "neural")
    assert total["pressure"]["mean"] == pytest.approx(0.5)
    assert total["pressure"]["seed_std"] is None
    report = campaign.run_campaign(directory, device="cpu", stage="suite")
    assert len(trained) == len(set(trained)) == len(report["runs"]) == 6
    assert all(len(r["seeds"]) == 3 for r in report["summary"])
    (directory / "runs/mass-seed0/fit.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        campaign.summarize_campaign(directory)
    assert json.loads((directory / "progress.json").read_text())["train-mass-0"]["state"] == "complete"
