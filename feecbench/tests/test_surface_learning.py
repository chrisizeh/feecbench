"""Adapter orientation, real sparse gradients, bidirectional overfit and resume."""

import numpy as np
import pytest
import torch

pytest.importorskip("pydec")
pytest.importorskip("skimage")

from feecbench.surface.data import SurfaceSample, atomic_json, load, save
from feecbench.surface.geometry import Geometry, generate
from feecbench.surface.metrics import residuals, score
from feecbench.surface.model import SurfaceTransolver, loss, quadratic, sparse_tensor, tensor_sample
from feecbench.surface.operators import assemble, harmonic_basis
from feecbench.surface.oracle import Midpoint, ModalControl, pressure_modes
from feecbench.surface.training import device_for, train


@pytest.fixture(scope="module")
def samples():
    torch.set_num_threads(1)
    p, t, _ = generate(Geometry(), 12)[0]
    op = assemble(p, t, 1)
    lam, phi = pressure_modes(op, 12)
    h = harmonic_basis(op)
    samples = []
    for family in ("pressure", "velocity"):
        p0 = phi[:, 1] if family == "pressure" else np.zeros(len(p))
        u0 = op.d0 @ phi[:, 1] if family == "velocity" else np.zeros(len(op.edges))
        ps, us = Midpoint(op).solve(p0, u0, [0, 0.2, 0.4], 0.005)
        samples.append(
            SurfaceSample(
                op,
                np.array([0, 0.2, 0.4]),
                ps,
                us,
                lam,
                phi,
                h,
                {"config_id": family, "split": "train", "resolution": "medium"},
                np.zeros(3, bool),
            )
        )
    return samples


def test_sparse_weighted_gradient(samples):
    mass = sparse_tensor(samples[0].ops.m1, dtype=torch.float64)
    x = torch.randn(mass.shape[0], dtype=torch.float64, requires_grad=True)
    quadratic(x, mass).backward()
    np.testing.assert_allclose(x.grad.numpy(), 2 * samples[0].ops.m1 @ x.detach().numpy(), atol=1e-12)
    direction = torch.randn_like(x)
    eps = 1e-5
    numerical = (quadratic(x.detach() + eps * direction, mass) - quadratic(x.detach() - eps * direction, mass)) / (
        2 * eps
    )
    torch.testing.assert_close(numerical, torch.dot(x.grad, direction), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("adapter", ["scalar", "vector_features"])
def test_end_to_end_orientation(samples, adapter):
    sample = samples[1]
    scales = {"pressure": 2.3, "velocity": 0.7}
    batch = tensor_sample(sample, scales)
    signs = np.random.default_rng(42).choice([-1, 1], len(sample.ops.edges))
    flipped = SurfaceSample(
        sample.ops.flipped(signs),
        sample.times,
        sample.pressure,
        sample.velocity * signs,
        sample.values,
        sample.vectors,
        sample.harmonic * signs[:, None],
        sample.metadata,
        sample.offgrid,
    )
    torch.manual_seed(4)
    model = SurfaceTransolver(32, 1, 8, adapter=adapter).eval()
    p, u = model(batch, 0.2)
    fp, fu = model(tensor_sample(flipped, scales), 0.2)
    torch.testing.assert_close(fp, p)
    torch.testing.assert_close(fu, u * torch.tensor(signs))
    bp, bu = model(batch, 0)
    torch.testing.assert_close(bp, batch["p0"], rtol=0, atol=0)
    torch.testing.assert_close(bu, batch["u0"], rtol=0, atol=0)
    base_loss = loss((p, u), (batch["pressure"][1], batch["velocity"][1]), batch)
    fb = tensor_sample(flipped, scales)
    torch.testing.assert_close(loss((fp, fu), (fb["pressure"][1], fb["velocity"][1]), fb), base_loss)


@pytest.mark.parametrize("which", [0, 1])
def test_overfit_both_coupling_directions(samples, which):
    torch.manual_seed(12)
    batch = tensor_sample(samples[which], {"pressure": 1, "velocity": 1})
    model = SurfaceTransolver(32, 1, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    target = (batch["pressure"][2], batch["velocity"][2])
    initial = float(loss(model(batch, 0.4), target, batch).detach())
    for _ in range(160):
        optimizer.zero_grad()
        value = loss(model(batch, 0.4), target, batch)
        value.backward()
        optimizer.step()
    assert float(loss(model(batch, 0.4), target, batch).detach()) < initial * 0.08


def test_modal_and_residual_diagnostics(samples):
    sample = samples[1]
    control = ModalControl(sample.ops, sample.values, sample.vectors, sample.harmonic)

    def predictor(ts):
        return control.solve(sample.pressure[0], sample.velocity[0], ts)

    result = residuals(sample.ops, predictor, [0.15, 0.3], steps=(0.01, 0.005))
    assert result["0.01"]["pressure"] / result["0.005"]["pressure"] > 3.9
    p, u = predictor(sample.times)
    metrics = score(sample, p, u)
    assert metrics["pressure"]["normalized_rmse"] < 1e-4
    frozen_p = np.broadcast_to(sample.pressure[0], sample.pressure.shape)
    frozen_u = np.broadcast_to(sample.velocity[0], sample.velocity.shape)
    assert score(sample, frozen_p, frozen_u)["pressure"]["normalized_rmse"] > 0.9


def test_resume_exact_epoch_trajectory(samples, tmp_path):
    root = tmp_path / "data"
    entries = []
    for split in ("train", "validation"):
        for i, sample in enumerate(samples):
            sample = SurfaceSample(**vars(sample))
            sample.metadata = {**sample.metadata, "split": split, "config_id": f"{split}-{i}"}
            path = root / f"{split}-{i}.h5"
            save(path, sample)
            entries.append({"path": path.name, "split": split, "resolution": "medium"})
    atomic_json(root / "index.json", {"fingerprint": "test", "samples": entries, "complete": True})
    kwargs = {"width": 16, "blocks": 1, "slices": 4, "accumulation": 3}
    train(root, tmp_path / "uninterrupted", epochs=2, **kwargs)
    train(root, tmp_path / "resumed", epochs=1, **kwargs)
    train(root, tmp_path / "resumed", epochs=2, resume=True, **kwargs)
    a = torch.load(tmp_path / "uninterrupted" / "latest.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed" / "latest.pt", weights_only=False)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert a["best"] == b["best"]
    assert "adapter" not in a["settings"]  # Legacy scalar checkpoints retain their settings identity.
    with pytest.raises(ValueError, match="hash mismatch"):
        train(root, tmp_path / "resumed", epochs=3, resume=True, seed=9, **kwargs)
    assert load(root / "train-0.h5").metadata["split"] == "train"


def test_cuda_request_fails_when_unavailable():
    if torch.cuda.is_available():
        pytest.skip("CUDA available")
    with pytest.raises(RuntimeError, match="explicitly requested"):
        device_for("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA hardware")
def test_cpu_cuda_predictions_and_sparse_gradients(samples):
    torch.manual_seed(17)
    cpu = SurfaceTransolver(32, 1, 8)
    gpu = SurfaceTransolver(32, 1, 8).cuda()
    gpu.load_state_dict(cpu.state_dict())
    scales = {"pressure": 1.2, "velocity": 0.8}
    cb, gb = tensor_sample(samples[1], scales), tensor_sample(samples[1], scales, "cuda")
    cp, gp = cpu(cb, 0.2), gpu(gb, 0.2)
    for a, b in zip(cp, gp, strict=True):
        torch.testing.assert_close(a, b.cpu(), atol=2e-5, rtol=2e-5)
    loss(cp, (cb["pressure"][1], cb["velocity"][1]), cb).backward()
    loss(gp, (gb["pressure"][1], gb["velocity"][1]), gb).backward()
    for a, b in zip(cpu.parameters(), gpu.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad.cpu(), atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("which", [0, 1])
def test_coupling_diagnostic_distinguishes_response_from_memorization(samples, which):
    from feecbench.surface.coupling import inspect_sample, query_indices

    sample = SurfaceSample(**vars(samples[which]))
    sample.metadata = {**sample.metadata, "family": "pulse" if which == 0 else "exact_velocity"}
    scales = {"pressure": 1.3, "velocity": 0.7}
    batch = tensor_sample(sample, scales)
    indices = query_indices(sample, 2)
    assert np.all(sample.times[indices] > 0)

    class Replay(torch.nn.Module):
        def __init__(self, responsive):
            super().__init__()
            self.responsive = responsive

        def forward(self, inputs, time):
            q = int(torch.argmin(torch.abs(batch["times"] - time)))
            source = "p0" if which == 0 else "u0"
            amplitude = torch.dot(inputs[source], batch[source]) / torch.dot(batch[source], batch[source])
            factor = amplitude if self.responsive else 1.0
            return batch["pressure"][q] * factor, batch["velocity"][q] * factor

    responsive = inspect_sample(Replay(True), sample, batch, indices, scales)
    memorized = inspect_sample(Replay(False), sample, batch, indices, scales)
    for row in (responsive, memorized):
        assert row["pressure"]["normalized_rmse"] < 1e-6
        assert row["velocity"]["normalized_rmse"] < 1e-6
    assert responsive["coupling"]["response_nrmse"] < 1e-6
    assert responsive["coupling"]["zero_input_output_nrmse"] == 0
    assert responsive["coupling"]["response_alignment"] == pytest.approx(1, abs=1e-6)
    assert memorized["coupling"]["response_nrmse"] == pytest.approx(1)
    assert memorized["coupling"]["zero_input_output_nrmse"] == pytest.approx(1)
    assert memorized["coupling"]["response_rms_ratio"] == 0


def test_actual_sample_diagnostic_preserves_run_and_uses_training_only(samples, tmp_path):
    import hashlib

    from feecbench.surface.coupling import inspect_checkpoint, overfit_coupling

    root, run = tmp_path / "data", tmp_path / "run"
    entries = []
    for split in ("train", "validation"):
        for i, sample in enumerate(samples):
            sample = SurfaceSample(**vars(sample))
            sample.metadata = {
                **sample.metadata,
                "config_id": f"{split}-{i}",
                "split": split,
                "family": "pulse" if i == 0 else "exact_velocity",
            }
            path = root / f"{split}-{i}.h5"
            save(path, sample)
            entries.append({"path": path.name, **sample.metadata})
    atomic_json(root / "index.json", {"fingerprint": "diagnostic-test", "complete": True, "samples": entries})
    train(root, run, epochs=1, width=16, blocks=1, slices=4)
    original = (run / "best.pt").read_bytes()
    inspection = inspect_checkpoint(root, run, tmp_path / "inspection.json", queries=2)
    assert {r["split"] for r in inspection["cases"]} == {"train", "validation"}
    report = overfit_coupling(root, run, tmp_path / "overfit", steps=2, queries=2, report_every=1)
    assert report["initialization"] == "fresh"
    assert len(report["history"]) == 3
    assert {r["split"] for r in report["cases"]} == {"train"}
    assert {r["coupling"]["direction"] for r in report["cases"]} == {"pressure_to_velocity", "velocity_to_pressure"}
    assert report["source_checkpoint_sha256"] == hashlib.sha256(original).hexdigest()
    balanced = overfit_coupling(
        root, run, tmp_path / "balanced", steps=2, queries=2, report_every=1, objective="balanced"
    )
    assert balanced["objective"] == "balanced"
    assert balanced["query_indices"] == report["query_indices"]
    assert balanced["history"][0]["cases"] == report["history"][0]["cases"]
    assert balanced["history"][0]["mass_score"] == report["history"][0]["mass_score"]
    assert balanced["history"][0]["objective_score"] != report["history"][0]["objective_score"]
    assert balanced["final_cases"] == balanced["history"][-1]["cases"]
    assert balanced["lr_schedule"] == "constant"
    assert balanced["final_learning_rate"] == balanced["learning_rate"]
    cosine = overfit_coupling(
        root,
        run,
        tmp_path / "cosine",
        steps=2,
        queries=2,
        report_every=1,
        objective="balanced",
        lr_schedule="cosine",
        final_lr_ratio=0.01,
    )
    assert cosine["history"][0]["cases"] == balanced["history"][0]["cases"]
    assert cosine["lr_schedule"] == "cosine"
    assert cosine["final_learning_rate"] == pytest.approx(cosine["learning_rate"] * 0.01)
    assert cosine["history"][0]["next_learning_rate"] == cosine["learning_rate"]
    assert cosine["history"][1]["next_learning_rate"] == pytest.approx(cosine["learning_rate"] * 0.505)
    assert cosine["history"][-1]["next_learning_rate"] == cosine["final_learning_rate"]
    assert balanced["gradient_statistics"]["updates"] == 2
    assert 0 <= balanced["gradient_statistics"]["clipped_fraction"] <= 1
    assert len(balanced["energy_records"]) == 2
    for record in balanced["energy_records"]:
        for field in ("pressure", "velocity"):
            assert record[field]["objective_weight"] == 1 / record[field]["denominator"]
    assert (run / "best.pt").read_bytes() == original
    from feecbench.surface.training import evaluate

    with pytest.raises(ValueError, match="hash mismatch"):
        train(root, run, epochs=2, width=16, blocks=1, slices=4, resume=True, adapter="vector_features")
    vector_run = tmp_path / "vector-run"
    train(root, vector_run, epochs=1, width=16, blocks=1, slices=4, adapter="vector_features")
    checkpoint = torch.load(vector_run / "best.pt", weights_only=False)
    assert checkpoint["settings"]["adapter"] == "vector_features"
    assert len(checkpoint["settings"]["features"]["edge"]) == 7
    results = evaluate(root, vector_run / "best.pt", tmp_path / "vector-evaluation", residual_subset=0)
    assert {row["method"] for row in results} == {"neural", "modal", "zero", "persistence"}
    vector_fit = inspect_checkpoint(root, vector_run, tmp_path / "vector-inspection.json", queries=2)
    assert vector_fit["architecture"]["adapter"] == "vector_features"
    with pytest.raises(ValueError, match="empty"):
        overfit_coupling(root, run, tmp_path / "overfit", steps=1, queries=2)


@pytest.mark.parametrize("which", [0, 1])
def test_balanced_coupling_loss_energies_scaling_and_gradients(samples, which):
    from feecbench.surface.coupling import coupling_loss, fixed_field_energies

    sample = samples[which]
    batch = tensor_sample(sample, {"pressure": 1.3, "velocity": 0.7})
    qs = [1, 2]
    raw, energies = fixed_field_energies(batch, qs, floor=1e-12)
    assert not energies.requires_grad
    assert torch.equal(raw, energies)
    zero = (torch.zeros_like(batch["p0"]), torch.zeros_like(batch["u0"]))
    objectives = [
        coupling_loss(zero, (batch["pressure"][q], batch["velocity"][q]), batch, "balanced", energies) for q in qs
    ]
    torch.testing.assert_close(torch.stack(objectives).mean(), torch.tensor(2.0))
    changed = {**batch, "pressure": batch["pressure"] * 7, "velocity": batch["velocity"] * 0.03}
    _, changed_energies = fixed_field_energies(changed, qs, floor=1e-12)
    for q, value in zip(qs, objectives, strict=True):
        torch.testing.assert_close(
            value,
            coupling_loss(
                zero, (changed["pressure"][q], changed["velocity"][q]), changed, "balanced", changed_energies
            ),
        )
    target = (batch["pressure"][2], batch["velocity"][2])
    prediction = tuple((0.5 * field).detach().requires_grad_() for field in target)
    value = coupling_loss(prediction, target, batch, "balanced", energies)
    value.backward()
    for i, mass in enumerate((batch["m0"], batch["m1"])):
        expected = 2 * torch.sparse.mm(mass, (prediction[i].detach() - target[i])[:, None]).squeeze(1)
        torch.testing.assert_close(prediction[i].grad, expected / batch["area"] / energies[i])
    # The zero control borrows the source case's denominators, not a floor recomputed on zero targets.
    torch.testing.assert_close(coupling_loss(target, zero, batch, "balanced", energies), objectives[1])
    torch.testing.assert_close(
        coupling_loss(prediction, target, batch, "mass", energies), loss(prediction, target, batch), rtol=0, atol=0
    )
    zero_batch = {
        **batch,
        "pressure": torch.zeros_like(batch["pressure"]),
        "velocity": torch.zeros_like(batch["velocity"]),
    }
    zero_raw, floored = fixed_field_energies(zero_batch, qs, floor=0.01)
    torch.testing.assert_close(zero_raw, torch.zeros(2))
    torch.testing.assert_close(floored, torch.full((2,), 0.01))


def test_directional_feature_initialization_and_gradients(samples):
    torch.manual_seed(31)
    scalar = SurfaceTransolver(32, 1, 8)
    scalar_rng = torch.get_rng_state().clone()
    torch.manual_seed(31)
    vector = SurfaceTransolver(32, 1, 8, adapter="vector_features")
    assert torch.equal(torch.get_rng_state(), scalar_rng)
    for key, value in scalar.state_dict().items():
        other = vector.state_dict()[key]
        if key == "edge_encoder.weight":
            torch.testing.assert_close(value, other[:, :10], rtol=0, atol=0)
            assert torch.count_nonzero(other[:, 10:]) == 0
        else:
            torch.testing.assert_close(value, other, rtol=0, atol=0)
    for i, sample in enumerate(samples):
        batch = tensor_sample(sample, {"pressure": 1.3, "velocity": 0.7})
        left, right = scalar(batch, 0.4), vector(batch, 0.4)
        for a, b in zip(left, right, strict=True):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        vector.zero_grad()
        loss(right, (batch["pressure"][2], batch["velocity"][2]), batch).backward()
        gradient = vector.edge_encoder.weight.grad[:, 10:]
        if i == 0:  # A pure pressure input has u0=0, so these features vanish exactly.
            assert torch.count_nonzero(gradient) == 0
        else:
            assert torch.linalg.vector_norm(gradient) > 1e-7
    assert sum(p.numel() for p in vector.parameters()) - sum(p.numel() for p in scalar.parameters()) == 3 * 32


def test_directional_features_recover_edge_measurement(samples):
    model = SurfaceTransolver(16, 1, 4, adapter="vector_features")
    sample = samples[1]
    batch = tensor_sample(sample, {"pressure": 1.3, "velocity": 0.7})
    captured = []
    hook = model.edge_encoder.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0].detach()))
    model(batch, 0.2)
    hook.remove()
    features = captured[0]
    torch.testing.assert_close(features[:, 10:], features[:, 7:8] * features[:, 3:6])
    torch.testing.assert_close((features[:, 10:] * features[:, 3:6]).sum(dim=1), features[:, 7])
    assert features.shape[1] == 13
    with pytest.raises(ValueError, match="adapter"):
        SurfaceTransolver(adapter="unknown")


def test_evaluation_resume_checks_checkpoint_and_prediction_bytes(samples, tmp_path, monkeypatch):
    import json

    import feecbench.surface.training as training

    root, run, output = tmp_path / "data", tmp_path / "run", tmp_path / "evaluation"
    entries = []
    for split in ("train", "validation"):
        sample = SurfaceSample(**vars(samples[0]))
        sample.metadata = {**sample.metadata, "config_id": split, "split": split, "family": "pulse", "dt": 0.005}
        save(root / f"{split}.h5", sample)
        entries.append({"path": f"{split}.h5", **sample.metadata})
    atomic_json(root / "index.json", {"fingerprint": "resume-test", "complete": True, "samples": entries})
    train(root, run, epochs=1, width=16, blocks=1, slices=4)
    first = training.evaluate(root, run / "best.pt", output, resume=True)
    original_load = training.load
    loaded = []

    def tracked_load(path):
        loaded.append(path)
        return original_load(path)

    monkeypatch.setattr(training, "load", tracked_load)
    assert training.evaluate(root, run / "best.pt", output, resume=True) == first
    assert loaded == []
    prediction = output / "predictions/validation/medium.npz"
    prediction.write_bytes(b"interrupted prediction write")
    repaired = training.evaluate(root, run / "best.pt", output, resume=True)
    assert len(loaded) == 1
    assert repaired[0]["metrics"] == first[0]["metrics"]
    assert len(json.loads((output / "residuals.json").read_text())) == 1
    with prediction.open("rb") as stream:
        assert np.load(stream)["pressure"].shape == samples[0].pressure.shape
    # A valid but different checkpoint cannot reuse old prediction caches.
    checkpoint = torch.load(run / "best.pt", weights_only=False)
    checkpoint["epoch"] += 1
    torch.save(checkpoint, run / "changed.pt")
    with pytest.raises(ValueError, match="identity mismatch"):
        training.evaluate(root, run / "changed.pt", output, resume=True)


@pytest.mark.parametrize("boundary", ["before_latest", "after_latest", "after_best", "after_history"])
def test_training_recovers_interrupted_final_epoch(samples, tmp_path, monkeypatch, boundary):
    import json

    import feecbench.surface.training as training

    root, run = tmp_path / "data", tmp_path / "interrupted"
    entries = []
    for split in ("train", "validation"):
        sample = SurfaceSample(**vars(samples[0]))
        sample.metadata = {**sample.metadata, "split": split}
        save(root / f"{split}.h5", sample)
        entries.append({"path": f"{split}.h5", **sample.metadata})
    atomic_json(root / "index.json", {"fingerprint": "crash-test", "complete": True, "samples": entries})
    kwargs = {"epochs": 1, "width": 16, "blocks": 1, "slices": 4}
    train(root, tmp_path / "reference", **kwargs)
    save_checkpoint, save_json = training._save_checkpoint, training.atomic_json

    def interrupt_checkpoint(path, value):
        if boundary == "before_latest" and path.name == "latest.pt":
            raise RuntimeError("interruption")
        save_checkpoint(path, value)
        if (boundary, path.name) in (("after_latest", "latest.pt"), ("after_best", "best.pt")):
            raise RuntimeError("interruption")

    def interrupt_json(path, value):
        save_json(path, value)
        if boundary == "after_history" and path.name == "history.json":
            raise RuntimeError("interruption")

    monkeypatch.setattr(training, "_save_checkpoint", interrupt_checkpoint)
    monkeypatch.setattr(training, "atomic_json", interrupt_json)
    with pytest.raises(RuntimeError, match="interruption"):
        train(root, run, **kwargs)
    monkeypatch.setattr(training, "_save_checkpoint", save_checkpoint)
    monkeypatch.setattr(training, "atomic_json", save_json)
    train(root, run, resume=(run / "latest.pt").exists(), **kwargs)
    best = torch.load(run / "best.pt", weights_only=False)
    expected = torch.load(tmp_path / "reference/best.pt", weights_only=False)
    assert best["epoch"] == expected["epoch"] == 0
    assert best["best"] == expected["best"]
    for key, value in best["model"].items():
        torch.testing.assert_close(value, expected["model"][key], rtol=0, atol=0)
    assert len(json.loads((run / "history.json").read_text())) == 1
