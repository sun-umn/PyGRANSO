"""The solver with the on-device factorization, compared with pinned C."""

import pytest
import torch
from pinned_osqp_c_oracle import paired_settings, pinned_dll_path
from test_pinned_double_sparse_input import (
    _assert_same,
    _problem,
    _StoredSession,
    _torch_solve,
)

import pygranso.private.torchQDLDLSolve as qdldl_solve
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace

pytestmark = pytest.mark.skipif(
    not pinned_dll_path().is_file(), reason="pinned OSQP C library is unavailable"
)


@pytest.mark.parametrize("polishing", [0, 1])
@pytest.mark.parametrize("seed", range(12))
def test_device_programs_on_cpu_match_native(seed, polishing, monkeypatch):
    data, stored = _problem(seed)
    c, t = paired_settings(polishing=polishing)
    with _StoredSession(data, stored, monkeypatch, **c) as session:
        reference = session.solve()
    monkeypatch.setattr(qdldl_solve, "DEVICE_PROGRAM", True)
    workspace = TorchOSQPWorkspace()
    x, info = _torch_solve(data, stored, "csc", "cpu", t, workspace)
    assert torch.is_tensor(workspace.linear_solver.factors[2])
    _assert_same(reference, x, info)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("seed", range(6))
def test_cuda_solve_keeps_the_values_on_the_gpu(seed, monkeypatch):
    data, stored = _problem(seed)
    c, t = paired_settings(polishing=1)
    with _StoredSession(data, stored, monkeypatch, **c) as session:
        reference = session.solve()

    def forbid(name):
        original = getattr(torch.Tensor, name)

        def guarded(self, *args, **kwargs):
            if self.is_cuda and self.is_floating_point() and self.numel() > 1:
                raise AssertionError(f"a CUDA value tensor was copied with {name}()")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, guarded)

    for name in ("cpu", "tolist", "numpy"):
        forbid(name)
    workspace = TorchOSQPWorkspace()
    x, info = _torch_solve(data, stored, "csc", "cuda", t, workspace)
    monkeypatch.undo()
    assert x.is_cuda and workspace.linear_solver.factors[2].is_cuda
    with _StoredSession(data, stored, monkeypatch, **c) as session:
        _assert_same(session.solve(), x, info)
    assert reference["iter"] == info["admm_iterations"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("seed", range(6))
def test_gpu_factorization_setting_selects_the_device(seed, monkeypatch):
    data, stored = _problem(seed)
    c, t = paired_settings(polishing=1)
    with _StoredSession(data, stored, monkeypatch, **c) as session:
        reference = session.solve()
    for on_gpu in (True, False):
        workspace = TorchOSQPWorkspace()
        settings = dict(t, gpu_factorization=on_gpu)
        x, info = _torch_solve(data, stored, "csc", "cuda", settings, workspace)
        assert torch.is_tensor(workspace.linear_solver.factors[2]) == on_gpu
        _assert_same(reference, x, info)
    with pytest.raises(ValueError, match="gpu_factorization must be a bool"):
        _torch_solve(data, stored, "csc", "cuda", dict(t, gpu_factorization=1), None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("change", ["none", "q"])
@pytest.mark.parametrize("first", [True, False])
@pytest.mark.parametrize("seed", range(3))
def test_gpu_factorization_switch_on_a_reused_workspace(seed, first, change):
    """A changed setting takes effect although the KKT matrix is unchanged."""

    data, stored = _problem(seed)
    _, t = paired_settings()
    second = dict(data)
    if change == "q":
        second["q"] = data["q"] * 1.01

    def run(settings_first, settings_second):
        workspace = TorchOSQPWorkspace()
        _torch_solve(data, stored, "csc", "cuda", settings_first, workspace)
        made_on_gpu = torch.is_tensor(workspace.linear_solver.factors[2])
        x, info = _torch_solve(second, stored, "csc", "cuda", settings_second, workspace)
        return made_on_gpu, torch.is_tensor(workspace.linear_solver.factors[2]), x, info

    before, after = dict(t, gpu_factorization=first), dict(t, gpu_factorization=not first)
    made, used, x, info = run(before, after)
    assert (made, used) == (first, not first)
    # Same numbers as without the switch; that sequence is compared with C above.
    _, kept, x_ref, info_ref = run(before, before)
    assert kept == first
    assert info["status_val"] == info_ref["status_val"]
    assert info["admm_iterations"] == info_ref["admm_iterations"]
    assert x.cpu().numpy().tobytes() == x_ref.cpu().numpy().tobytes()
    assert (
        info["public_y"].cpu().numpy().tobytes()
        == info_ref["public_y"].cpu().numpy().tobytes()
    )
