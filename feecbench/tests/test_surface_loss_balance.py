"""Global training-only field calibration and read-only loss-study provenance."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("pydec")
pytest.importorskip("skimage")

from feecbench.surface import data, loss_balance
from feecbench.surface.data import atomic_json, mse_balance_from_energies, training_mse_balance
from feecbench.surface.model import loss


def test_field_balance_matches_zero_loss_and_equalizes_fields():
    energies = {"pressure": 4.0, "velocity": 0.25}
    result = mse_balance_from_energies(energies)
    weighted = [energies[f] * result["weights"][f] for f in energies]
    assert weighted == pytest.approx([2.125, 2.125])
    assert sum(weighted) == pytest.approx(sum(energies.values()))
    floored = mse_balance_from_energies({"pressure": 0.0, "velocity": 2.0})
    assert floored["floor_applied"] == {"pressure": True, "velocity": False}
    assert floored["weights"]["velocity"] * 2 == pytest.approx(2)
    assert np.isfinite(floored["weights"]["pressure"])


@pytest.mark.parametrize("values", [(0, 0), (-1, 1), (float("nan"), 1), (1, float("inf"))])
def test_invalid_balance_energies(values):
    with pytest.raises(ValueError):
        mse_balance_from_energies(dict(zip(("pressure", "velocity"), values, strict=True)))


def test_calibration_uses_equal_configurations_and_only_positive_regular_training(monkeypatch):
    def sample(cid, width, p, u, split="train", resolution="medium"):
        return SimpleNamespace(
            ops=SimpleNamespace(genus=1),
            metadata={"config_id": cid, "split": split, "resolution": resolution, "fingerprint": "fixed"},
            times=np.array([0, 0.1, 0.2, 0.3]),
            offgrid=np.array([False, False, False, True]),
            pressure=np.array([[1e9] * width, [p] * width, [p] * width, [1e9] * width]),
            velocity=np.array([[1e9] * (3 * width), [u] * (3 * width), [u] * (3 * width), [1e9] * (3 * width)]),
        )

    first, second = sample("a", 2, 2, 4), sample("b", 10, 6, 8)
    # Different query and DoF counts must not give the second configuration more weight.
    second.offgrid[2] = True
    samples = {
        "a": first,
        "b": second,
        "val": sample("v", 2, 1e8, 1e8, split="validation"),
        "fine": sample("a", 20, 1e8, 1e8, resolution="fine"),
    }
    monkeypatch.setattr(data, "load", samples.__getitem__)
    scales = {"pressure": 2.0, "velocity": 4.0}
    result = training_mse_balance(samples, scales)
    assert result["energies"] == {"pressure": 5.0, "velocity": 2.5}
    assert result["training_configurations"]["a"]["queries"] == 2
    assert result["training_configurations"]["b"]["queries"] == 1
    first.velocity *= np.array([1, -1, 1, -1, 1, -1])
    assert training_mse_balance(samples, scales) == result
    # Changing normalization is accounted for explicitly in normalized target energies.
    changed = training_mse_balance(samples, {"pressure": 4.0, "velocity": 4.0})
    assert changed["energies"]["pressure"] == pytest.approx(1.25)
    with pytest.raises(ValueError, match="duplicate"):
        training_mse_balance(["a", "a"], scales)
    second.ops.genus = 3
    with pytest.raises(ValueError, match="genus-3"):
        training_mse_balance(["b"], scales)


def test_balanced_mse_gradient_orientation_and_legacy_formula():
    calibration = mse_balance_from_energies({"pressure": 4, "velocity": 0.25})
    prediction = (torch.tensor([1.0, 3.0], requires_grad=True), torch.tensor([2.0, -1.0, 0.5], requires_grad=True))
    target = (torch.tensor([0.5, 2.0]), torch.tensor([1.0, 1.0, -0.5]))
    value = loss(prediction, target, {}, "mse_balanced", field_balance=calibration)
    value.backward()
    for field, pred, truth in zip(("pressure", "velocity"), prediction, target, strict=True):
        torch.testing.assert_close(
            pred.grad, 2 * calibration["weights"][field] * (pred.detach() - truth) / pred.numel()
        )
    signs = torch.tensor([-1, 1, -1])
    flipped = loss(
        (prediction[0], prediction[1] * signs),
        (target[0], target[1] * signs),
        {},
        "mse_balanced",
        field_balance=calibration,
    )
    torch.testing.assert_close(value, flipped, rtol=0, atol=0)
    torch.testing.assert_close(
        loss(prediction, target, {}, "mse"),
        sum((p - t).square().mean() for p, t in zip(prediction, target, strict=True)),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="requires"):
        loss(prediction, target, {}, "mse_balanced")


def test_study_provenance_and_paired_comparison(tmp_path, monkeypatch):
    parent, output = tmp_path / "parent", tmp_path / "study"
    parent.mkdir()
    sentinel = parent / "control.json"
    atomic_json(sentinel, {"untouched": True})
    sources = {"test": "source"}
    monkeypatch.setattr(loss_balance, "_sources", lambda: sources.copy())
    controls = [
        {
            "objective": objective,
            "seed": seed,
            "fit_summary": [
                {
                    "split": "validation",
                    "family": "pulse",
                    "configurations": 4,
                    "pressure": 0.2,
                    "velocity": value,
                    "pressure_mean_free": 0.3,
                }
            ],
            "validation_evaluation": [],
        }
        for objective, value in (("mse", 1.0), ("mass", 0.4))
        for seed in (0, 1, 2)
    ]
    snapshot = {
        "campaign_hash": "old-campaign",
        "files": {"control.json": loss_balance.file_hash(sentinel)},
        "controls": controls,
        "field_balance": mse_balance_from_energies({"pressure": 1, "velocity": 0.01}),
    }
    monkeypatch.setattr(loss_balance, "_parent_snapshot", lambda _: copy.deepcopy(snapshot))
    specification = loss_balance.prepare_study(parent, output)
    assert loss_balance.prepare_study(parent, output) == specification
    assert loss_balance.read_study(output) == specification
    before = sentinel.read_bytes()
    record = copy.deepcopy(controls[0])
    record.update(objective="mse_balanced", artifacts={})
    record["fit_summary"][0]["velocity"] = 0.5
    atomic_json(output / "runs/mse_balanced-seed0/complete.json", record)
    report = loss_balance.summarize_study(output)
    row = next(r for r in report["fit_comparison"] if r["objective"] == "mse_balanced")
    assert row["velocity"]["paired_differences"] == {
        "mse": [{"seed": 0, "value": -0.5}],
        "mass": [{"seed": 0, "value": pytest.approx(0.1)}],
    }
    assert sentinel.read_bytes() == before
    sources["test"] = "changed"
    with pytest.raises(ValueError, match="source/lockfile"):
        loss_balance.read_study(output)
    sources["test"] = "source"
    atomic_json(sentinel, {"changed": True})
    with pytest.raises(ValueError, match="parent data or artifact"):
        loss_balance.read_study(output)
    with pytest.raises(ValueError, match="separate"):
        loss_balance.prepare_study(parent, parent / "new-study")
