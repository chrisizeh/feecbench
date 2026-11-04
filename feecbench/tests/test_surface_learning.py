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


def test_end_to_end_orientation(samples):
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
    model = SurfaceTransolver(32, 1, 8).eval()
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
