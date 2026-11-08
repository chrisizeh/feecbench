"""Consistent typed messages, adjoints, orientation and persisted training."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

pytest.importorskip("pydec")
pytest.importorskip("skimage")

from feecbench.surface.campaign import prepare_campaign, read_campaign
from feecbench.surface.coupling import inspect_checkpoint, overfit_coupling
from feecbench.surface.data import SurfaceSample, atomic_json, save
from feecbench.surface.geometry import Geometry, generate
from feecbench.surface.model import SurfaceTransolver, loss, sparse_tensor, tensor_sample
from feecbench.surface.operators import Derived, assemble, harmonic_basis
from feecbench.surface.oracle import Midpoint, pressure_modes
from feecbench.surface.training import evaluate, train
from feecbench.surface.transport import _mass_cg, mass_solve, typed_messages


@pytest.fixture(scope="module")
def samples():
    torch.set_num_threads(1)
    points, tris, _ = generate(Geometry(), 12)[0]
    op = assemble(points, tris, 1)
    values, vectors = pressure_modes(op, 12)
    harmonic = harmonic_basis(op)
    output = []
    for family in ("pulse", "exact_velocity"):
        p0 = vectors[:, 1] if family == "pulse" else np.zeros(len(points))
        u0 = op.d0 @ vectors[:, 1] if family == "exact_velocity" else np.zeros(len(op.edges))
        times = np.array([0, 0.1, 0.2])
        p, u = Midpoint(op).solve(p0, u0, times, 0.005)
        output.append(SurfaceSample(
            op, times, p, u, values, vectors, harmonic,
            {"config_id": family, "family": family, "genus": 1, "split": "train",
             "resolution": "medium", "dt": 0.005}, np.zeros(3, bool),
        ))
    return output


def test_consistent_mass_solve_gradient_and_failure():
    dense = torch.tensor([[2., .4, .1], [.4, 3., .3], [.1, .3, 1.]], dtype=torch.float64)
    mass = sparse_tensor(sp.csr_matrix(dense.numpy()), dtype=torch.float64)
    diagonal = dense.diag()
    rhs = torch.tensor([[1., 0.], [-2., 0.], [.5, 0.]], dtype=torch.float64, requires_grad=True)
    actual = mass_solve(rhs, mass, diagonal)
    torch.testing.assert_close(actual, torch.linalg.solve(dense, rhs), rtol=1e-9, atol=1e-12)
    assert torch.autograd.gradcheck(lambda r: mass_solve(r, mass, diagonal), (rhs,))
    assert not torch.allclose(actual, rhs / diagonal[:, None])
    with pytest.raises(RuntimeError, match="residual tolerance"):
        _mass_cg(mass, diagonal, rhs.detach(), max_iterations=1)


def test_messages_match_whitney_operators_and_adjoint(samples):
    op = samples[0].ops
    batch = tensor_sample(samples[0], {"pressure": 2.3, "velocity": .7}, transport="d0_delta1")
    batch = {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in batch.items()}
    # Rebuild in float64, rather than comparing rounded float32 operators.
    batch.update(points=torch.tensor(op.points), d0=sparse_tensor(op.d0, dtype=torch.float64),
                 m0=sparse_tensor(op.m0, dtype=torch.float64), m1=sparse_tensor(op.m1, dtype=torch.float64),
                 m0_diagonal=torch.tensor(op.m0.diagonal()))
    rng = np.random.default_rng(44)
    vertex = torch.tensor(rng.normal(size=(len(op.points), 3)), requires_grad=True)
    edge = torch.tensor(rng.normal(size=(len(op.edges), 3)), requires_grad=True)
    lengths = np.linalg.norm(op.points[op.edges[:, 1]] - op.points[op.edges[:, 0]], axis=1)
    delta, gradient = typed_messages(vertex, edge, batch)
    expected = Derived(op).delta1(edge.detach().numpy() * lengths[:, None])
    np.testing.assert_allclose(delta.detach().numpy(), expected, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(gradient.detach().numpy() * lengths[:, None], op.d0 @ vertex.detach().numpy())
    left = vertex.detach().numpy().T @ (op.m0 @ delta.detach().numpy())
    right = (op.d0 @ vertex.detach().numpy()).T @ (op.m1 @ (edge.detach().numpy() * lengths[:, None]))
    np.testing.assert_allclose(left, right, rtol=1e-8, atol=1e-8)
    delta.square().sum().backward()
    expected_grad = 2 * lengths[:, None] * (op.m1 @ (op.d0 @ Derived(op).solve0(expected)))
    np.testing.assert_allclose(edge.grad.numpy(), expected_grad, rtol=1e-7, atol=1e-6)
    _, constant_gradient = typed_messages(torch.ones_like(vertex), torch.zeros_like(edge), batch)
    assert torch.count_nonzero(constant_gradient) == 0


@pytest.mark.parametrize("adapter", ["scalar", "vector_features"])
def test_active_transport_orientation_and_paired_initialization(samples, adapter):
    sample = samples[1]
    scales = {"pressure": 2.3, "velocity": .7}
    batch = tensor_sample(sample, scales, transport="d0_delta1")
    torch.manual_seed(4)
    baseline = SurfaceTransolver(16, 2, 4, adapter=adapter)
    torch.manual_seed(4)
    model = SurfaceTransolver(16, 2, 4, adapter=adapter, transport="d0_delta1")
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    for a, b in zip(baseline(batch, .1), model(batch, .1), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with torch.no_grad():
        for block in model.transports:
            block.to_vertex.weight.normal_(std=.05)
            block.to_edge.weight.normal_(std=.05)
    signs = np.random.default_rng(51).choice([-1, 1], len(sample.ops.edges))
    flipped = SurfaceSample(**{**vars(sample), "ops": sample.ops.flipped(signs),
                               "velocity": sample.velocity * signs, "harmonic": sample.harmonic * signs[:, None]})
    p, u = model(batch, .1)
    fp, fu = model(tensor_sample(flipped, scales, transport="d0_delta1"), .1)
    torch.testing.assert_close(p, fp)
    torch.testing.assert_close(u * torch.tensor(signs), fu)
    for actual, expected in zip(model(batch, 0), (batch["p0"], batch["u0"]), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    objective = loss((p, u), (batch["pressure"][1], batch["velocity"][1]), batch)
    objective.backward()
    for block in model.transports:
        assert block.to_vertex.weight.grad.norm() > 0
        assert block.to_edge.weight.grad.norm() > 0
    assert all(torch.isfinite(v.grad).all() for v in model.parameters())
    with pytest.raises(ValueError, match="transport"):
        SurfaceTransolver(transport="invalid")


def test_transport_training_resume_evaluation_and_diagnostics(samples, tmp_path):
    root = tmp_path / "data"
    entries = []
    for split in ("train", "validation"):
        for i, original in enumerate(samples):
            sample = SurfaceSample(**vars(original))
            sample.metadata = {**sample.metadata, "split": split, "config_id": f"{split}-{i}"}
            path = root / f"{split}-{i}.h5"
            save(path, sample)
            entries.append({"path": path.name, **sample.metadata})
    atomic_json(root / "index.json", {"fingerprint": "typed-test", "complete": True, "samples": entries})
    settings = {"width": 16, "blocks": 1, "slices": 4, "transport": "d0_delta1", "objective": "mse_balanced"}
    run = tmp_path / "resumed"
    train(root, tmp_path / "full", epochs=2, **settings)
    train(root, run, epochs=1, **settings)
    train(root, run, epochs=2, resume=True, **settings)
    a = torch.load(tmp_path / "full/latest.pt", weights_only=False)
    b = torch.load(run / "latest.pt", weights_only=False)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert a["best"] == b["best"]
    assert b["settings"]["transport"] == "d0_delta1"
    assert "operators" in b["settings"]["features"]
    with pytest.raises(ValueError, match="hash mismatch"):
        train(root, run, epochs=3, resume=True, **{**settings, "transport": "none"})
    rows = evaluate(root, run / "best.pt", tmp_path / "evaluation", splits=["validation"])
    assert {r["method"] for r in rows} == {"neural", "modal", "zero", "persistence"}
    fit = inspect_checkpoint(root, run, tmp_path / "fit.json", queries=2)
    assert fit["architecture"]["transport"] == "d0_delta1"
    report = overfit_coupling(root, run, tmp_path / "overfit", steps=2, queries=2, report_every=1)
    assert report["architecture"]["transport"] == "d0_delta1"
    assert report["features"]["operators"] == b["settings"]["features"]["operators"]
    config = prepare_campaign(tmp_path / "campaign", smoke=True, transport="d0_delta1")
    assert read_campaign(tmp_path / "campaign") == config
    assert config["training"]["transport"] == "d0_delta1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA hardware")
def test_transport_cuda_forward_and_backward(samples):
    torch.manual_seed(7)
    cpu = SurfaceTransolver(16, 1, 4, transport="d0_delta1")
    with torch.no_grad():
        cpu.transports[0].to_vertex.weight.normal_(std=.05)
        cpu.transports[0].to_edge.weight.normal_(std=.05)
    gpu = SurfaceTransolver(16, 1, 4, transport="d0_delta1").cuda()
    gpu.load_state_dict(cpu.state_dict())
    gradients = []
    predictions = []
    for model, device in ((cpu, "cpu"), (gpu, "cuda")):
        batch = tensor_sample(samples[0], {"pressure": 1., "velocity": 1.}, device, transport="d0_delta1")
        pred = model(batch, .1)
        loss(pred, (batch["pressure"][1], batch["velocity"][1]), batch).backward()
        predictions.append(pred)
        gradients.append([p.grad.cpu() for p in model.parameters()])
    for a, b in zip(predictions[0], predictions[1], strict=True):
        torch.testing.assert_close(a, b.cpu(), rtol=1e-4, atol=1e-5)
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
