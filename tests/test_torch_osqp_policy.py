import warnings

import pytest
import torch

from pygranso.private import osqpTorchAdapter as adapter
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace


def _problem(n=2, device="cpu"):
    device = torch.device(device)
    dtype = torch.float64
    return (
        torch.eye(n, device=device, dtype=dtype),
        -torch.ones((n, 1), device=device, dtype=dtype),
        None,
        None,
        torch.zeros((n, 1), device=device, dtype=dtype),
        torch.ones((n, 1), device=device, dtype=dtype),
    )


def test_auto_cpu_follows_requested_device_and_uses_builtin():
    H, f, A, b, lower, upper = _problem()
    _, info = adapter.solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        lower,
        upper,
        "cpu",
        True,
        {"algebra": "auto", "settings": {"return_info": True}},
        TorchOSQPWorkspace(),
    )
    assert info["backend"] == "builtin"
    assert info["selection_reason"] == "cpu_target_uses_builtin"


def test_builtin_workspace_reuses_structure_for_vector_update():
    workspace = TorchOSQPWorkspace()
    problem = _problem()
    _, first = adapter.solve_osqp_torch_qp(
        *problem,
        "cpu",
        True,
        {"algebra": "builtin", "settings": {"return_info": True}},
        workspace,
    )
    changed = list(problem)
    changed[1] = -2.0 * torch.ones_like(changed[1])
    _, second = adapter.solve_osqp_torch_qp(
        *changed,
        "cpu",
        True,
        {"algebra": "builtin", "settings": {"return_info": True}},
        workspace,
    )
    assert first["builtin_workspace"]["setups"] == 1
    assert second["builtin_workspace"]["updates"] == 1
    assert second["builtin_workspace"]["last_cache_hit"]


def test_legacy_solver_settings_warn_and_raise():
    H, f, A, b, lower, upper = _problem()
    with pytest.warns(FutureWarning, match="archived"):
        with pytest.raises(ValueError, match="no longer executable"):
            adapter.solve_osqp_torch_qp(
                H,
                f,
                A,
                b,
                lower,
                upper,
                "cpu",
                True,
                {"algebra": "torch", "settings": {"linear_solver": "sparse_cg"}},
            )


def test_explicit_oversize_warns_but_remains_torch_selection():
    n = 1201
    H = torch.empty((n, n), device="meta")
    f = torch.empty((n, 1), device="meta")
    with warnings.catch_warnings(record=True) as caught:
        selection = adapter._select_backend(
            "torch", torch.device("cpu"), torch.float64, H, f, None, None
        )
    assert selection["backend"] == "torch"
    assert selection["estimated_kkt_dim"] > adapter.MAX_SUPPORTED_KKT_DIM
    assert any("exceeds the validated" in str(item.message) for item in caught)


def test_auto_failure_records_full_fallback(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to select the automatic Torch route.")
    problem = _problem(device="cuda")
    monkeypatch.setitem(adapter.PROMOTED_ACCELERATOR_BACKENDS, "cuda", True)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic torch failure")

    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", fail)
    with pytest.warns(RuntimeWarning, match="synthetic torch failure"):
        _, info = adapter.solve_osqp_torch_qp(
            *problem,
            torch.device("cuda"),
            True,
            {"algebra": "auto", "settings": {"return_info": True}},
            TorchOSQPWorkspace(),
        )
    fallback = info["fallback"]
    assert fallback["occurred"]
    assert fallback["exception_type"] == "RuntimeError"
    assert fallback["fallback_backend"] == "builtin"
    assert fallback["device_transfer"]


def test_auto_unsolved_status_records_status_and_falls_back(monkeypatch):
    problem = _problem()
    monkeypatch.setattr(
        adapter,
        "_select_backend",
        lambda *args, **kwargs: {
            "requested_backend": "auto",
            "backend": "torch",
            "selection_reason": "synthetic_validated_target",
            "estimated_kkt_dim": 4,
            "estimated_dense_working_memory_mb": 1.0,
            "fallback": {"occurred": False},
        },
    )

    def unsolved(*args, **kwargs):
        return torch.zeros((2, 1), dtype=torch.float64), {
            "status": "max_iter_reached",
            "status_compatible": False,
        }

    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", unsolved)
    with pytest.warns(RuntimeWarning, match="solved-compatible"):
        _, info = adapter.solve_osqp_torch_qp(
            *problem,
            torch.device("cpu"),
            True,
            {"algebra": "auto", "settings": {"return_info": True}},
            TorchOSQPWorkspace(),
        )
    assert info["backend"] == "builtin"
    assert info["fallback"]["trigger"] == "unsolved_status"
    assert info["fallback"]["status"] == "max_iter_reached"
    assert info["fallback"]["torch_info"]["status_compatible"] is False


def test_unpromoted_cuda_auto_fallback_is_observable():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("An NVIDIA CUDA device is required.")
    H, f, A, b, lower, upper = _problem(device="cuda")
    with pytest.warns(RuntimeWarning, match="cuda_not_promoted"):
        _, info = adapter.solve_osqp_torch_qp(
            H,
            f,
            A,
            b,
            lower,
            upper,
            "cuda",
            True,
            {"algebra": "auto", "settings": {"return_info": True}},
            TorchOSQPWorkspace(),
        )
    assert info["backend"] == "builtin"
    assert info["selection_reason"] == "cuda_not_promoted"
    assert info["fallback"]["occurred"]
    assert info["fallback"]["trigger"] == "selection_policy"


def test_auto_memory_preflight_selects_observable_builtin_fallback(monkeypatch):
    monkeypatch.setattr(adapter, "_accelerator_capability", lambda *args: (True, "ok"))
    monkeypatch.setattr(adapter, "_memory_limit_mb", lambda *args: 0.0001)
    H, f, A, b, *_ = _problem(n=10)
    with pytest.warns(RuntimeWarning, match="memory"):
        selection = adapter._select_backend(
            "auto", torch.device("cuda"), torch.float64, H, f, A, b
        )
    assert selection["backend"] == "builtin"
    assert selection["selection_reason"] == "memory_preflight"
    assert selection["fallback"]["occurred"]


def test_mps_float64_auto_fallback_returns_cpu_solution():
    problem = _problem()
    with pytest.warns(RuntimeWarning, match="mps_float64_unsupported"):
        solution, info = adapter.solve_osqp_torch_qp(
            *problem,
            torch.device("mps"),
            True,
            {"algebra": "auto", "settings": {"return_info": True}},
            TorchOSQPWorkspace(),
        )
    assert solution.device.type == "cpu"
    assert info["requested_device"] == "mps"
    assert info["device"] == "cpu"
    assert info["selection_reason"] == "mps_float64_unsupported"
    assert info["fallback"]["occurred"]


def test_only_three_public_algebra_values_are_accepted():
    with pytest.raises(ValueError, match="auto.*builtin.*torch"):
        adapter._normalize_options({"algebra": "cuda"}, torch.float64)


def test_explicit_torch_unsolved_status_raises(monkeypatch):
    problem = _problem()

    def unsolved(*args, **kwargs):
        return torch.zeros((2, 1), dtype=torch.float64), {
            "status": "max_iter_reached",
            "status_compatible": False,
        }

    monkeypatch.setattr(adapter, "_solve_torch_osqp_path", unsolved)
    with pytest.raises(RuntimeError, match="max_iter_reached"):
        adapter.solve_osqp_torch_qp(
            *problem,
            torch.device("cpu"),
            True,
            {"algebra": "torch", "settings": {"return_info": True}},
            TorchOSQPWorkspace(),
        )


def test_backend_change_invalidates_complete_workspace():
    workspace = TorchOSQPWorkspace()
    workspace.state = {"x": torch.ones(1), "z": torch.ones(1), "y": torch.ones(1)}
    workspace.active_backend = "torch"
    assert workspace.ensure_backend("builtin")
    assert workspace.state is None
    assert workspace.builtin_cache is None
    assert workspace.active_backend == "builtin"


def test_unindexed_cuda_target_is_canonicalized_to_the_current_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    canonical = adapter._canonical_device(torch.device("cuda"))
    assert canonical.type == "cuda"
    assert canonical.index == torch.cuda.current_device()
