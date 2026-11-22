import json
from pathlib import Path

import numpy as np
import osqp
import pytest
import torch
from pinned_osqp_c_oracle import (
    default_settings as pinned_c_default_settings,
)
from pinned_osqp_c_oracle import (
    load_pinned_library,
    pinned_dll_path,
    solve_saved_pygranso_problem,
)
from pinned_osqp_c_oracle import (
    solve_dense_qp as solve_pinned_c_dense_qp,
)
from scipy import sparse

from pygranso.private import torchOSQP as torch_osqp_module
from pygranso.private.osqpTorchAdapter import (
    DEFAULT_OSQP_SETTINGS,
    estimate_dense_kkt,
    solve_osqp_torch_qp,
)
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchLinearSolve import TorchLinearSolveError
from pygranso.private.torchOSQP import solve_torch_osqp_direct


def _solve(
    H,
    f,
    A,
    b,
    lower,
    upper,
    algebra,
    settings_override=None,
    workspace=None,
):
    settings = {
        "return_info": True,
        "rho_update_safeguard": False,
        "eps_abs": 1e-8,
        "eps_rel": 1e-8,
        "scaling": 10,
        "adaptive_rho": True,
        "polishing": True,
    }
    settings.update(settings_override or {})
    return solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        lower,
        upper,
        torch.device("cpu"),
        True,
        {"algebra": algebra, "settings": settings},
        workspace or TorchOSQPWorkspace(),
    )


def _direct_settings(**updates):
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "rho_update_safeguard": False,
        "return_info": True,
        "return_state": True,
        "eps_abs": 1e-6,
        "eps_rel": 1e-6,
        "polish_refine_iter": 4,
    }
    settings.update(updates)
    return settings


@pytest.mark.skipif(not pinned_dll_path().is_file(), reason="pinned C DLL unavailable")
def test_retained_pinned_c_oracle_abi_and_defaults():
    defaults = pinned_c_default_settings()
    assert defaults == {
        "version": "1.0.0",
        "rho": 0.1,
        "sigma": 1e-6,
        "alpha": 1.6,
        "max_iter": 4000,
        "eps_abs": 1e-3,
        "eps_rel": 1e-3,
        "eps_prim_inf": 1e-4,
        "eps_dual_inf": 1e-4,
        "scaling": 10,
        "polishing": 0,
        "adaptive_rho": 1,
        "adaptive_rho_interval": 50,
        "adaptive_rho_fraction": 0.4,
        "adaptive_rho_tolerance": 5.0,
        "check_termination": 25,
        "check_dualgap": 1,
        "time_limit": 1e10,
        "delta": 1e-6,
        "polish_refine_iter": 3,
    }


@pytest.mark.skipif(not pinned_dll_path().is_file(), reason="pinned C DLL unavailable")
def test_compiled_pinned_time_mode_uses_executable_interval_clamp():
    result = solve_pinned_c_dense_qp(
        np.ones((1, 1)),
        np.array([-1.0]),
        np.ones((1, 1)),
        np.zeros(1),
        np.ones(1),
        settings_overrides={
            "verbose": 0,
            "scaling": 0,
            "polishing": 0,
            "adaptive_rho": 2,
            "adaptive_rho_interval": 0,
            "adaptive_rho_fraction": 1e-12,
            "adaptive_rho_tolerance": 1.0,
            "check_termination": 0,
            "max_iter": 4000,
            "eps_abs": 1e-20,
            "eps_rel": 1e-20,
        },
    )
    assert result["setup_code"] == result["solve_code"] == 0
    assert result["adaptive_rho"] == 2
    assert result["adaptive_rho_interval"] == 1


@pytest.mark.skipif(not pinned_dll_path().is_file(), reason="pinned C DLL unavailable")
def test_compiled_pinned_kkt_mode_zero_interval_and_update_branch():
    result = solve_pinned_c_dense_qp(
        np.ones((1, 1)),
        np.array([-1.0]),
        np.ones((1, 1)),
        np.zeros(1),
        np.ones(1),
        settings_overrides={
            "verbose": 0,
            "scaling": 0,
            "polishing": 0,
            "adaptive_rho": 3,
            "adaptive_rho_interval": 0,
            "adaptive_rho_fraction": 0.5,
            "adaptive_rho_tolerance": 1.0,
            "check_termination": 0,
            "max_iter": 5,
            "eps_abs": 1e-12,
            "eps_rel": 1e-12,
        },
    )
    assert result["setup_code"] == result["solve_code"] == 0
    assert result["adaptive_rho"] == 3
    assert result["adaptive_rho_interval"] == 1
    assert result["rho_updates"] > 0


@pytest.mark.skipif(not pinned_dll_path().is_file(), reason="pinned C DLL unavailable")
@pytest.mark.parametrize(
    "seed,expected_polish,expected_external_pass",
    [
        (3, 1, False),
        (19, -1, True),
        (27, 1, False),
        (47, 1, False),
        (55, -1, True),
        (59, 1, False),
        (79, 1, False),
        (95, 1, False),
    ],
)
def test_retained_pinned_c_oracle_reproduces_polishing_disposition(
    seed,
    expected_polish,
    expected_external_pass,
):
    cases = json.loads(Path(__file__).with_name("pinned_polish_problems.json").read_text())
    problem = [torch.tensor(values, dtype=torch.float64) for values in cases[str(seed)]]
    result = solve_saved_pygranso_problem(
        problem,
        library=load_pinned_library(),
        settings_overrides={
            "verbose": 0,
            "scaling": 10,
            "polishing": 1,
            "rho": 0.1,
            "sigma": 1e-6,
            "alpha": 1.6,
            "adaptive_rho": 1,
            "eps_abs": 1e-8,
            "eps_rel": 1e-8,
            "check_dualgap": 1,
            "polish_refine_iter": 3,
        },
    )

    assert result["setup_code"] == 0
    assert result["solve_code"] == 0
    assert result["status_val"] == 1
    assert result["status_polish"] == expected_polish
    external_pass = (
        result["independent_primal_residual"] <= result["eps_primal"]
        and result["independent_dual_residual"] <= result["eps_dual"]
    )
    assert external_pass is expected_external_pass


@pytest.mark.parametrize("check_dualgap,expected_iterations", [(True, 13), (False, 5)])
def test_duality_gap_prevents_residual_only_early_stop(check_dualgap, expected_iterations):
    """F03: pinned auxil.c checks the gap as well as both residuals."""
    P = torch.tensor([[1e-6]], dtype=torch.float64)
    q = torch.tensor([-1.0], dtype=torch.float64)
    A = torch.empty((0, 1), dtype=torch.float64)
    bounds = torch.empty(0, dtype=torch.float64)
    settings = _direct_settings(
        scaling=0, max_iter=100, eps_abs=1e-3, eps_rel=0.0,
        eps_dual_inf=1e-12, check_termination=1, check_dualgap=check_dualgap,
        adaptive_rho=False, polishing=False,
    )
    reference = osqp.OSQP()
    reference.setup(
        P=sparse.csc_matrix(P.numpy()), q=q.numpy(),
        A=sparse.csc_matrix((0, 1)), l=bounds.numpy(), u=bounds.numpy(),
        scaling=0, max_iter=100, eps_abs=1e-3, eps_rel=0.0,
        eps_dual_inf=1e-12, check_termination=1, check_dualgap=check_dualgap,
        adaptive_rho=False, polishing=False, verbose=False,
    )
    official = reference.solve(raise_error=False)
    result, info = solve_torch_osqp_direct(P, q, A, bounds, bounds, settings)
    assert official.info.status_val == info["status_val"] == 1
    assert official.info.iter == info["admm_iterations"] == expected_iterations
    np.testing.assert_allclose(result.flatten().numpy(), official.x, rtol=1e-14)
    assert info["duality_gap"] == pytest.approx(official.info.duality_gap, abs=2e-9)
    assert info["check_dualgap"] is check_dualgap
    assert (abs(info["duality_gap"]) < info["eps_duality_gap"]) is check_dualgap


def test_adaptive_rho_zero_residual_matches_pinned_source():
    """Pinned auxil.c updates 0/0 to OSQP_RHO_MIN instead of skipping it."""
    dtype = torch.float64
    P = torch.tensor([[1.0]], dtype=dtype)
    q = torch.zeros(1, dtype=dtype)
    A = P.clone()
    l = torch.tensor([-1.0], dtype=dtype)
    u = torch.tensor([1.0], dtype=dtype)
    settings = _direct_settings(
        rho=0.1,
        sigma=1e-6,
        scaling=0,
        max_iter=1,
        eps_abs=1e-3,
        eps_rel=1e-3,
        check_termination=0,
        check_dualgap=False,
        adaptive_rho=True,
        rho_update_interval=1,
        rho_update_tolerance=5.0,
        polishing=False,
    )

    reference = osqp.OSQP()
    reference.setup(
        P=sparse.csc_matrix(P.numpy()),
        q=q.numpy(),
        A=sparse.csc_matrix(A.numpy()),
        l=l.numpy(),
        u=u.numpy(),
        rho=0.1,
        sigma=1e-6,
        scaling=0,
        max_iter=1,
        eps_abs=1e-3,
        eps_rel=1e-3,
        check_termination=0,
        check_dualgap=False,
        adaptive_rho=True,
        adaptive_rho_interval=1,
        adaptive_rho_tolerance=5.0,
        polishing=False,
        verbose=False,
    )
    official = reference.solve(raise_error=False)
    result, info = solve_torch_osqp_direct(P, q, A, l, u, settings)

    assert official.info.status_val == info["status_val"] == 1
    assert official.info.iter == info["admm_iterations"] == 1
    assert official.info.rho_updates == info["rho_updates"] == 1
    assert official.info.rho_estimate == pytest.approx(info["rho_bar"], abs=1e-12)
    assert info["rho_bar"] == pytest.approx(1e-6, abs=1e-12)
    np.testing.assert_allclose(result.flatten().numpy(), official.x, rtol=1e-14)


def test_final_rho_estimate_matches_pinned_source_without_an_update():
    dtype = torch.float64
    P = torch.ones((1, 1), dtype=dtype)
    q = -torch.ones(1, dtype=dtype)
    A = P.clone()
    l, u = -torch.ones(1, dtype=dtype), torch.ones(1, dtype=dtype)
    settings = _direct_settings(
        rho=0.1,
        sigma=1e-6,
        scaling=0,
        max_iter=1,
        check_termination=0,
        adaptive_rho=False,
        polishing=False,
    )

    reference = osqp.OSQP()
    reference.setup(
        P=sparse.csc_matrix(P.numpy()),
        q=q.numpy(),
        A=sparse.csc_matrix(A.numpy()),
        l=l.numpy(),
        u=u.numpy(),
        rho=0.1,
        sigma=1e-6,
        scaling=0,
        max_iter=1,
        check_termination=0,
        adaptive_rho=False,
        polishing=False,
        verbose=False,
    )
    official = reference.solve(raise_error=False)
    result, info = solve_torch_osqp_direct(P, q, A, l, u, settings)

    assert official.info.status_val == info["status_val"] == 7
    # Pinned osqp_api.c performs KKT solve -> relaxed x -> projected z -> y.
    x_tilde = 1.0 / (1.0 + 1e-6 + 0.1)
    expected_x = 1.6 * x_tilde
    expected_z = 1.0
    expected_y = 0.1 * (expected_x - expected_z)
    np.testing.assert_allclose(result.reshape(-1).numpy(), [expected_x], rtol=1e-14)
    np.testing.assert_allclose(info["state"]["z"].numpy(), [expected_z], rtol=0.0)
    np.testing.assert_allclose(info["state"]["y"].numpy(), [expected_y], rtol=1e-14)
    np.testing.assert_allclose(official.x, [expected_x], rtol=1e-14)
    np.testing.assert_allclose(official.y, [expected_y], rtol=1e-14)
    assert official.info.rho_estimate != pytest.approx(info["rho_bar"])
    assert info["rho_estimate"] == pytest.approx(
        official.info.rho_estimate, abs=1e-15
    )


@pytest.mark.parametrize(
    "check_termination,max_iter,expected_updates",
    [(0, 99, 0), (0, 100, 1), (2, 7, 0), (2, 8, 1)],
)
def test_adaptive_rho_zero_interval_uses_pinned_automatic_schedule(
    check_termination, max_iter, expected_updates
):
    dtype = torch.float64
    P = torch.ones((1, 1), dtype=dtype)
    q = -torch.ones(1, dtype=dtype)
    A = P.clone()
    l, u = -torch.ones(1, dtype=dtype), torch.ones(1, dtype=dtype)
    settings = _direct_settings(
        rho=0.1,
        sigma=1e-6,
        scaling=0,
        max_iter=max_iter,
        eps_abs=1e-12,
        eps_rel=1e-12,
        check_termination=check_termination,
        check_dualgap=False,
        adaptive_rho=True,
        rho_update_interval=0,
        rho_update_tolerance=1.0,
        polishing=False,
    )

    reference = osqp.OSQP()
    reference.setup(
        P=sparse.csc_matrix(P.numpy()),
        q=q.numpy(),
        A=sparse.csc_matrix(A.numpy()),
        l=l.numpy(),
        u=u.numpy(),
        rho=0.1,
        sigma=1e-6,
        scaling=0,
        max_iter=max_iter,
        eps_abs=1e-12,
        eps_rel=1e-12,
        check_termination=check_termination,
        check_dualgap=False,
        adaptive_rho=True,
        adaptive_rho_interval=0,
        adaptive_rho_tolerance=1.0,
        polishing=False,
        verbose=False,
    )
    official = reference.solve(raise_error=False)
    _, info = solve_torch_osqp_direct(P, q, A, l, u, settings)

    assert official.info.status_val == info["status_val"] == 7
    assert official.info.iter == info["admm_iterations"] == max_iter
    assert official.info.rho_updates == info["rho_updates"] == expected_updates
    assert info["factorizations_this_solve"] == 1 + expected_updates


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scaled_termination_does_not_change_scaled_adaptive_rho_path(
    device, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    P = torch.diag(torch.tensor([1e-8, 1.0], device=device, dtype=dtype))
    q = torch.tensor([-1e4, -1.0], device=device, dtype=dtype)
    A = torch.tensor(
        [[1e6, 1.0], [1.0, 1e-3]], device=device, dtype=dtype
    )
    l = torch.tensor([-1.0, -2.0], device=device, dtype=dtype)
    u = torch.tensor([2.0, 3.0], device=device, dtype=dtype)
    torch_info = {}
    workspaces = {}
    official_info = {}

    for scaled_termination in (False, True):
        settings = _direct_settings(
            rho=0.1,
            sigma=1e-6,
            alpha=1.6,
            scaling=10,
            max_iter=3,
            eps_abs=1e-12,
            eps_rel=1e-12,
            check_termination=0,
            check_dualgap=False,
            adaptive_rho=True,
            rho_update_interval=3,
            rho_update_tolerance=1.0,
            polishing=False,
            warm_start=False,
            scaled_termination=scaled_termination,
        )
        if device == "cpu" and dtype == torch.float64:
            reference = osqp.OSQP()
            reference.setup(
                P=sparse.csc_matrix(P.numpy()),
                q=q.numpy(),
                A=sparse.csc_matrix(A.numpy()),
                l=l.numpy(),
                u=u.numpy(),
                rho=0.1,
                sigma=1e-6,
                alpha=1.6,
                scaling=10,
                max_iter=3,
                eps_abs=1e-12,
                eps_rel=1e-12,
                check_termination=0,
                check_dualgap=False,
                adaptive_rho=True,
                adaptive_rho_interval=3,
                adaptive_rho_tolerance=1.0,
                polishing=False,
                warm_starting=False,
                scaled_termination=scaled_termination,
                verbose=False,
            )
            official_info[scaled_termination] = reference.solve(
                raise_error=False
            ).info
        workspaces[scaled_termination] = TorchOSQPWorkspace()
        _, torch_info[scaled_termination] = solve_torch_osqp_direct(
            P, q, A, l, u, settings, workspaces[scaled_termination]
        )

    def residuals_and_candidate(P_value, q_value, A_value, x, z, y):
        Ax = A_value @ x
        Px = P_value @ x
        ATy = A_value.T @ y
        primal = torch.max(torch.abs(Ax - z))
        dual = torch.max(torch.abs(Px + q_value + ATy))
        primal_normalizer = torch.maximum(
            torch.max(torch.abs(Ax)), torch.max(torch.abs(z))
        )
        dual_normalizer = torch.maximum(
            torch.maximum(torch.max(torch.abs(Px)), torch.max(torch.abs(ATy))),
            torch.max(torch.abs(q_value)),
        )
        division_tolerance = torch.as_tensor(
            1e-17 if dtype == torch.float32 and device == "cuda" else 1e-30,
            device=device,
            dtype=dtype,
        )
        ratio = (primal / (primal_normalizer + division_tolerance)) / (
            dual / (dual_normalizer + division_tolerance)
        )
        return primal, dual, (0.1 * torch.sqrt(ratio)).clamp(1e-6, 1e6)

    workspace = workspaces[False]
    D, E, cost = (
        workspace.scaling["D"],
        workspace.scaling["E"],
        workspace.scaling["cost"],
    )
    assert not torch.equal(D, torch.ones_like(D))
    assert not torch.equal(E, torch.ones_like(E))
    state = torch_info[False]["state"]
    original = residuals_and_candidate(
        P, q, A, state["x"], state["z"], state["y"]
    )
    scaled = residuals_and_candidate(
        cost * (D[:, None] * P * D[None, :]),
        cost * D * q,
        E[:, None] * A * D[None, :],
        state["x"] / D,
        state["z"] * E,
        state["y"] * cost / E,
    )
    relative_tolerance = 2e-5 if dtype == torch.float32 else 1e-12
    assert scaled[2].item() != pytest.approx(original[2].item(), rel=1e-3)

    if official_info:
        assert official_info[False].rho_updates == official_info[True].rho_updates == 1
        assert official_info[False].rho_estimate == pytest.approx(
            official_info[True].rho_estimate, rel=1e-14
        )
    assert torch_info[False]["rho_updates"] == torch_info[True]["rho_updates"] == 1
    assert all(
        info["factorizations_this_solve"] == 2 for info in torch_info.values()
    )
    assert torch_info[False]["rho_bar"] == pytest.approx(
        scaled[2].item(), rel=relative_tolerance
    )
    assert torch_info[False]["rho_bar"] != pytest.approx(
        original[2].item(), rel=1e-3
    )
    assert torch_info[False]["rho_bar"] == pytest.approx(
        torch_info[True]["rho_bar"], rel=relative_tolerance
    )
    assert torch_info[False]["rho_estimate"] == pytest.approx(
        torch_info[True]["rho_estimate"], rel=relative_tolerance
    )
    if official_info:
        assert torch_info[False]["rho_estimate"] == pytest.approx(
            official_info[False].rho_estimate, rel=1e-13
        )
    assert torch_info[False]["primal_residual"] == pytest.approx(
        original[0].item(), rel=relative_tolerance
    )
    assert torch_info[False]["dual_residual"] == pytest.approx(
        original[1].item(), rel=relative_tolerance
    )
    assert torch_info[True]["primal_residual"] == pytest.approx(
        scaled[0].item(), rel=relative_tolerance
    )
    assert torch_info[True]["dual_residual"] == pytest.approx(
        scaled[1].item(), rel=relative_tolerance
    )
    assert any(
        original_value.item()
        != pytest.approx(scaled_value.item(), rel=1e-3)
        for original_value, scaled_value in zip(original[:2], scaled[:2])
    )
    assert all(info["admm_iterations"] == 3 for info in torch_info.values())
    assert all(info["scaling_applied"] for info in torch_info.values())


def test_multi_update_adaptive_rho_prefix_trajectory_matches_official_osqp():
    dtype = torch.float64
    P = torch.diag(torch.tensor([1e-8, 1.0], dtype=dtype))
    q = torch.tensor([-1e4, -1.0], dtype=dtype)
    A = torch.tensor([[1e6, 1.0], [1.0, 1e-3]], dtype=dtype)
    l = torch.tensor([-1.0, -2.0], dtype=dtype)
    u = torch.tensor([2.0, 3.0], dtype=dtype)
    active_rho_trajectory = []

    for expected_updates, max_iter in enumerate((3, 6, 9, 12, 15), start=1):
        settings = _direct_settings(
            rho=0.1,
            sigma=1e-6,
            alpha=1.6,
            scaling=10,
            max_iter=max_iter,
            eps_abs=1e-12,
            eps_rel=1e-12,
            check_termination=0,
            check_dualgap=False,
            adaptive_rho=True,
            rho_update_interval=3,
            rho_update_tolerance=1.0,
            polishing=False,
            warm_start=False,
        )
        reference = osqp.OSQP()
        reference.setup(
            P=sparse.csc_matrix(P.numpy()),
            q=q.numpy(),
            A=sparse.csc_matrix(A.numpy()),
            l=l.numpy(),
            u=u.numpy(),
            rho=0.1,
            sigma=1e-6,
            alpha=1.6,
            scaling=10,
            max_iter=max_iter,
            eps_abs=1e-12,
            eps_rel=1e-12,
            check_termination=0,
            check_dualgap=False,
            adaptive_rho=True,
            adaptive_rho_interval=3,
            adaptive_rho_tolerance=1.0,
            polishing=False,
            warm_starting=False,
            verbose=False,
        )
        official = reference.solve(raise_error=False)
        official_active_rho = reference._solver.get_settings().rho
        _, info = solve_torch_osqp_direct(P, q, A, l, u, settings)

        assert official.info.status_val == info["status_val"] == 7
        assert official.info.iter == info["admm_iterations"] == max_iter
        assert official.info.rho_updates == info["rho_updates"] == expected_updates
        assert info["factorizations_this_solve"] == 1 + expected_updates
        assert info["rho_bar"] == pytest.approx(official_active_rho, rel=1e-12)
        assert info["rho_estimate"] == pytest.approx(
            official.info.rho_estimate, rel=1e-12
        )
        active_rho_trajectory.append(info["rho_bar"])

    assert active_rho_trajectory == pytest.approx(
        [
            0.10067045271879548,
            0.25898952633858163,
            1.7176878305116952,
            4.91762758460343,
            21.60932638807736,
        ],
        rel=1e-12,
    )


def _independent_rho_observation(P, q, A, x, z, y, rho, division_tolerance):
    Ax = A @ x
    Px = P @ x
    ATy = A.T @ y
    primal = torch.linalg.vector_norm(Ax - z, ord=float("inf"))
    dual = torch.linalg.vector_norm((q + Px) + ATy, ord=float("inf"))
    primal_normalizer = torch.maximum(
        torch.linalg.vector_norm(Ax, ord=float("inf")),
        torch.linalg.vector_norm(z, ord=float("inf")),
    )
    dual_normalizer = torch.maximum(
        torch.maximum(
            torch.linalg.vector_norm(q, ord=float("inf")),
            torch.linalg.vector_norm(ATy, ord=float("inf")),
        ),
        torch.linalg.vector_norm(Px, ord=float("inf")),
    )
    estimate_raw = (
        rho
        * torch.sqrt(
            (primal / (primal_normalizer + division_tolerance))
            / (dual / (dual_normalizer + division_tolerance))
        )
    )
    rho_min = torch.as_tensor(1e-6, device=rho.device, dtype=rho.dtype)
    rho_max = torch.as_tensor(1e6, device=rho.device, dtype=rho.dtype)
    candidate = torch.where(torch.isnan(estimate_raw), rho_min, estimate_raw)
    candidate = torch.minimum(torch.maximum(candidate, rho_min), rho_max)
    return {
        "primal": primal,
        "dual": dual,
        "primal_normalizer": primal_normalizer,
        "dual_normalizer": dual_normalizer,
        "candidate": candidate,
    }


def _independent_dense_kkt(P, A, sigma, rho_vec):
    n = P.shape[0]
    m = A.shape[0]
    kkt = torch.empty((n + m, n + m), device=P.device, dtype=P.dtype)
    kkt[:n, :n] = P + float(sigma) * torch.eye(
        n, device=P.device, dtype=P.dtype
    )
    kkt[:n, n:] = A.T
    kkt[n:, :n] = A
    kkt[n:, n:] = -torch.diag(rho_vec.reciprocal())
    return kkt


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_multi_update_adaptive_rho_uses_refreshed_factors_on_all_devices(
    monkeypatch, device, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")

    admm_states = []
    rho_decisions = []
    refactorizations = []
    linear_solves = []
    original_admm_update = torch_osqp_module.admm_vector_update
    original_rho_update = torch_osqp_module._adaptive_rho_update
    original_refactorize = torch_osqp_module.QDLDLSolver.factorize
    original_solve = torch_osqp_module.QDLDLSolver.solve

    def capture_admm_update(*args, **kwargs):
        updated_state = original_admm_update(*args, **kwargs)
        admm_states.append(tuple(value.detach().clone() for value in updated_state[:3]))
        return updated_state

    def capture_rho_update(
        rho_bar,
        equality,
        primal,
        dual,
        eps_primal,
        eps_dual,
        tolerance,
        *,
        loose=None,
        primal_normalizer=None,
        dual_normalizer=None,
    ):
        outcome = original_rho_update(
            rho_bar,
            equality,
            primal,
            dual,
            eps_primal,
            eps_dual,
            tolerance,
            loose=loose,
            primal_normalizer=primal_normalizer,
            dual_normalizer=dual_normalizer,
        )
        rho_decisions.append(
            {
                "rho": rho_bar.detach().clone(),
                "primal": primal.detach().clone(),
                "dual": dual.detach().clone(),
                "primal_normalizer": primal_normalizer.detach().clone(),
                "dual_normalizer": dual_normalizer.detach().clone(),
                "tolerance": tolerance,
                "updated": outcome[0],
                "committed_rho": outcome[1].detach().clone(),
                "rho_vec": outcome[2].detach().clone(),
            }
        )
        return outcome

    def capture_refactorization(self, matrix, **kwargs):
        if self.factorization_count:
            refactorizations.append(matrix.detach().clone())
        return original_refactorize(self, matrix, **kwargs)

    def capture_solve(self, rhs, *, calculate_residual=False):
        matrix = self.matrix.detach().clone()
        solution, diagnostics = original_solve(
            self, rhs, calculate_residual=calculate_residual
        )
        linear_solves.append(
            {
                "matrix": matrix,
                "rhs": rhs.detach().clone(),
                "solution": solution.detach().clone(),
            }
        )
        return solution, diagnostics

    monkeypatch.setattr(
        torch_osqp_module, "admm_vector_update", capture_admm_update
    )
    monkeypatch.setattr(
        torch_osqp_module, "_adaptive_rho_update", capture_rho_update
    )
    monkeypatch.setattr(
        torch_osqp_module.QDLDLSolver, "factorize", capture_refactorization
    )
    monkeypatch.setattr(
        torch_osqp_module.QDLDLSolver, "solve", capture_solve
    )

    P = torch.diag(torch.tensor([1e-8, 1.0], device=device, dtype=dtype))
    q = torch.tensor([-1e4, -1.0], device=device, dtype=dtype)
    A = torch.tensor(
        [[1e6, 1.0], [1.0, 1e-3]], device=device, dtype=dtype
    )
    l = torch.tensor([-1.0, -2.0], device=device, dtype=dtype)
    u = torch.tensor([2.0, 3.0], device=device, dtype=dtype)
    relative_tolerance = 2e-5 if dtype == torch.float32 else 1e-12
    absolute_tolerance = torch.finfo(dtype).eps * 10
    linear_residual_limit = torch.finfo(dtype).eps * 1000

    for max_iter in (3, 6, 9, 12, 15, 16):
        admm_states.clear()
        rho_decisions.clear()
        refactorizations.clear()
        linear_solves.clear()
        settings = _direct_settings(
            rho=0.1,
            sigma=1e-6,
            alpha=1.6,
            scaling=10,
            max_iter=max_iter,
            eps_abs=1e-12,
            eps_rel=1e-12,
            check_termination=0,
            check_dualgap=False,
            adaptive_rho=True,
            rho_update_interval=3,
            rho_update_tolerance=1.0,
            polishing=False,
            warm_start=False,
        )
        workspace = TorchOSQPWorkspace()
        _, info = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)

        P_scaled = workspace.scaling["scaled_P"]
        q_scaled = workspace.scaling["scaled_q"]
        A_scaled = workspace.scaling["scaled_A"]
        division_tolerance = torch.as_tensor(
            1e-17 if dtype == torch.float32 and device == "cuda" else 1e-30,
            device=device,
            dtype=dtype,
        )

        assert info["status_val"] == 7
        assert info["admm_iterations"] == max_iter
        assert len(admm_states) == len(linear_solves) == max_iter
        assert len(rho_decisions) == max_iter // 3

        expected_refactorization = 0
        for decision_index, observed in enumerate(rho_decisions):
            iteration = 3 * (decision_index + 1)
            x, z, y = admm_states[iteration - 1]
            source = _independent_rho_observation(
                P_scaled,
                q_scaled,
                A_scaled,
                x,
                z,
                y,
                observed["rho"],
                division_tolerance,
            )
            candidate = source["candidate"]
            tolerance = torch.as_tensor(
                observed["tolerance"], device=device, dtype=dtype
            )
            upper_margin = candidate - tolerance * observed["rho"]
            lower_margin = observed["rho"] / tolerance - candidate
            should_update = bool(
                ((upper_margin > 0) | (lower_margin > 0)).item()
            )
            branch_distance = torch.minimum(
                torch.abs(upper_margin), torch.abs(lower_margin)
            )
            branch_scale = torch.maximum(
                torch.maximum(torch.abs(candidate), torch.abs(observed["rho"])),
                torch.ones((), device=device, dtype=dtype),
            )
            assert bool(
                (
                    branch_distance
                    > 100 * torch.finfo(dtype).eps * branch_scale
                ).item()
            )

            torch.testing.assert_close(
                observed["primal"],
                source["primal"],
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
            torch.testing.assert_close(
                observed["dual"],
                source["dual"],
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
            torch.testing.assert_close(
                observed["primal_normalizer"],
                source["primal_normalizer"],
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
            torch.testing.assert_close(
                observed["dual_normalizer"],
                source["dual_normalizer"],
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
            assert observed["updated"] is should_update

            expected_rho = candidate if should_update else observed["rho"]
            torch.testing.assert_close(
                observed["committed_rho"],
                expected_rho,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
            expected_rho_vec = torch.full_like(l, expected_rho.item())
            torch.testing.assert_close(
                observed["rho_vec"],
                expected_rho_vec,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )

            if should_update:
                expected_kkt = _independent_dense_kkt(
                    P_scaled, A_scaled, 1e-6, expected_rho_vec
                )
                torch.testing.assert_close(
                    refactorizations[expected_refactorization],
                    expected_kkt,
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                )
                expected_refactorization += 1

        assert expected_refactorization == len(refactorizations)
        assert info["rho_updates"] == expected_refactorization
        assert info["factorizations_this_solve"] == 1 + expected_refactorization
        torch.testing.assert_close(
            torch.as_tensor(info["rho_bar"], device=device, dtype=dtype),
            rho_decisions[-1]["committed_rho"],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )

        x, z, y = admm_states[-1]
        active_rho = torch.as_tensor(
            info["rho_bar"], device=device, dtype=dtype
        )
        final_observation = _independent_rho_observation(
            P_scaled,
            q_scaled,
            A_scaled,
            x,
            z,
            y,
            active_rho,
            division_tolerance,
        )
        torch.testing.assert_close(
            torch.as_tensor(info["rho_estimate"], device=device, dtype=dtype),
            final_observation["candidate"],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        assert info["rho_estimate"] != pytest.approx(
            info["rho_bar"],
            rel=relative_tolerance,
            abs=float(absolute_tolerance),
        )

        if max_iter == 16:
            assert len(refactorizations) == 5
            for update_index, refreshed_kkt in enumerate(refactorizations):
                subsequent_solve = linear_solves[3 * (update_index + 1)]
                torch.testing.assert_close(
                    subsequent_solve["matrix"],
                    refreshed_kkt,
                    rtol=0,
                    atol=0,
                )
                residual = (
                    refreshed_kkt @ subsequent_solve["solution"]
                    - subsequent_solve["rhs"]
                )
                denominator = (
                    torch.linalg.matrix_norm(
                        refreshed_kkt, ord=float("inf")
                    )
                    * torch.linalg.vector_norm(
                        subsequent_solve["solution"], ord=float("inf")
                    )
                    + torch.linalg.vector_norm(
                        subsequent_solve["rhs"], ord=float("inf")
                    )
                )
                assert bool((denominator > 0).item())
                relative_residual = (
                    torch.linalg.vector_norm(residual, ord=float("inf"))
                    / denominator
                )
                assert bool((relative_residual <= linear_residual_limit).item())


@pytest.mark.parametrize(
    "scenario",
    ["decrease", "rejected", "minimum_clamp", "maximum_clamp", "mixed_rows"],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adaptive_rho_decision_families_control_kkt_refresh(
    monkeypatch, scenario, device, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")

    admm_states = []
    rho_decisions = []
    refactorizations = []
    linear_solves = []
    original_admm_update = torch_osqp_module.admm_vector_update
    original_rho_update = torch_osqp_module._adaptive_rho_update
    original_refactorize = torch_osqp_module.QDLDLSolver.factorize
    original_solve = torch_osqp_module.QDLDLSolver.solve

    def capture_admm_update(*args, **kwargs):
        updated_state = original_admm_update(*args, **kwargs)
        admm_states.append(tuple(value.detach().clone() for value in updated_state[:3]))
        return updated_state

    def capture_rho_update(*args, **kwargs):
        outcome = original_rho_update(*args, **kwargs)
        rho_decisions.append(
            {
                "rho": args[0].detach().clone(),
                "tolerance": args[6],
                "updated": outcome[0],
                "committed_rho": outcome[1].detach().clone(),
                "rho_vec": outcome[2].detach().clone(),
            }
        )
        return outcome

    def capture_refactorization(self, matrix, **kwargs):
        if self.factorization_count:
            refactorizations.append(matrix.detach().clone())
        return original_refactorize(self, matrix, **kwargs)

    def capture_solve(self, rhs, *, calculate_residual=False):
        matrix = self.matrix.detach().clone()
        solution, diagnostics = original_solve(
            self, rhs, calculate_residual=calculate_residual
        )
        linear_solves.append(
            {
                "matrix": matrix,
                "rhs": rhs.detach().clone(),
                "solution": solution.detach().clone(),
            }
        )
        return solution, diagnostics

    monkeypatch.setattr(
        torch_osqp_module, "admm_vector_update", capture_admm_update
    )
    monkeypatch.setattr(
        torch_osqp_module, "_adaptive_rho_update", capture_rho_update
    )
    monkeypatch.setattr(
        torch_osqp_module.QDLDLSolver, "factorize", capture_refactorization
    )
    monkeypatch.setattr(
        torch_osqp_module.QDLDLSolver, "solve", capture_solve
    )

    if scenario in {"decrease", "rejected"}:
        P = torch.tensor(
            [
                [3.321136216667415, 4.3951008944071095],
                [4.3951008944071095, 6.113382299571413],
            ],
            device=device,
            dtype=dtype,
        )
        q = torch.tensor(
            [0.6280013943561004, -2.2095355495051727],
            device=device,
            dtype=dtype,
        )
        A = torch.tensor(
            [
                [-1.4313642691810498, 0.6807253321045634],
                [-0.802474394712645, 1.0439603382853593],
            ],
            device=device,
            dtype=dtype,
        )
        l = -torch.ones(2, device=device, dtype=dtype)
        u = -l
        initial_rho = 1.0 if scenario == "decrease" else 0.1
    elif scenario == "minimum_clamp":
        P = torch.ones((1, 1), device=device, dtype=dtype)
        q = torch.zeros(1, device=device, dtype=dtype)
        A = P.clone()
        l = torch.tensor([-1.0], device=device, dtype=dtype)
        u = -l
        initial_rho = 0.1
    elif scenario == "maximum_clamp":
        P = torch.zeros((1, 1), device=device, dtype=dtype)
        q = torch.zeros(1, device=device, dtype=dtype)
        A = P.clone()
        l = torch.tensor([1.0], device=device, dtype=dtype)
        u = torch.tensor([2.0], device=device, dtype=dtype)
        initial_rho = 0.1
    else:
        P = torch.ones((1, 1), device=device, dtype=dtype)
        q = torch.tensor([-1.0], device=device, dtype=dtype)
        A = torch.ones((3, 1), device=device, dtype=dtype)
        l = torch.tensor(
            [0.0, -1.0, -torch.inf], device=device, dtype=dtype
        )
        u = torch.tensor(
            [0.0, 1.0, torch.inf], device=device, dtype=dtype
        )
        initial_rho = 0.1

    rho_tolerance = 1.0 if scenario == "mixed_rows" else 5.0

    settings = _direct_settings(
        rho=initial_rho,
        sigma=1e-6,
        alpha=1.6,
        scaling=0,
        max_iter=2,
        eps_abs=1e-10,
        eps_rel=1e-10,
        eps_prim_inf=1e-12,
        eps_dual_inf=1e-12,
        check_termination=0,
        check_dualgap=False,
        adaptive_rho=True,
        rho_update_interval=1,
        rho_update_tolerance=rho_tolerance,
        polishing=False,
        warm_start=False,
    )
    workspace = TorchOSQPWorkspace()
    _, info = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)

    assert len(admm_states) == len(rho_decisions) == len(linear_solves) == 2
    division_tolerance = torch.as_tensor(
        1e-17 if dtype == torch.float32 and device == "cuda" else 1e-30,
        device=device,
        dtype=dtype,
    )
    relative_tolerance = 3e-5 if dtype == torch.float32 else 1e-12
    absolute_tolerance = 10 * torch.finfo(dtype).eps
    initial_rho_vec = torch.full_like(l, initial_rho)
    if scenario == "mixed_rows":
        initial_rho_vec[0] *= 1000.0
        initial_rho_vec[2] = 1e-6
    active_kkt = _independent_dense_kkt(
        P,
        A,
        settings["sigma"],
        initial_rho_vec,
    )
    torch.testing.assert_close(linear_solves[0]["matrix"], active_kkt, rtol=0, atol=0)

    expected_refactorizations = 0
    expected_updates = []
    for decision_index, observed in enumerate(rho_decisions):
        x, z, y = admm_states[decision_index]
        source = _independent_rho_observation(
            P,
            q,
            A,
            x,
            z,
            y,
            observed["rho"],
            division_tolerance,
        )
        candidate = source["candidate"]
        tolerance = torch.as_tensor(
            observed["tolerance"], device=device, dtype=dtype
        )
        upper_margin = candidate - tolerance * observed["rho"]
        lower_margin = observed["rho"] / tolerance - candidate
        branch_distance = torch.minimum(
            torch.abs(upper_margin), torch.abs(lower_margin)
        )
        branch_scale = torch.maximum(
            torch.maximum(torch.abs(candidate), torch.abs(observed["rho"])),
            torch.as_tensor(
                torch.finfo(dtype).tiny, device=device, dtype=dtype
            ),
        )
        assert bool(
            (
                branch_distance
                > 100 * torch.finfo(dtype).eps * branch_scale
            ).item()
        )
        should_update = bool(
            (
                (upper_margin > 0)
                | (lower_margin > 0)
            ).item()
        )
        expected_updates.append(should_update)
        assert observed["updated"] is should_update

        expected_rho = candidate if should_update else observed["rho"]
        expected_rho_vec = torch.full_like(l, expected_rho.item())
        if scenario == "mixed_rows":
            expected_rho_vec[0] *= 1000.0
            expected_rho_vec[2] = 1e-6
        torch.testing.assert_close(
            observed["committed_rho"],
            expected_rho,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        torch.testing.assert_close(
            observed["rho_vec"],
            expected_rho_vec,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )

        if should_update:
            active_kkt = _independent_dense_kkt(
                P, A, settings["sigma"], expected_rho_vec
            )
            torch.testing.assert_close(
                refactorizations[expected_refactorizations],
                active_kkt,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
            expected_refactorizations += 1
        if decision_index + 1 < len(linear_solves):
            torch.testing.assert_close(
                linear_solves[decision_index + 1]["matrix"],
                active_kkt,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )

    if scenario == "decrease":
        assert expected_updates == [True, False]
        assert rho_decisions[0]["committed_rho"] < initial_rho
        assert rho_decisions[0]["committed_rho"] > 1e-6
    elif scenario == "rejected":
        assert expected_updates == [False, False]
    elif scenario == "minimum_clamp":
        assert expected_updates == [True, False]
        assert rho_decisions[0]["committed_rho"] == torch.as_tensor(
            1e-6, device=device, dtype=dtype
        )
    elif scenario == "maximum_clamp":
        assert expected_updates == [True, False]
        assert rho_decisions[0]["committed_rho"] == torch.as_tensor(
            1e6, device=device, dtype=dtype
        )
        assert info["status"] == "primal_infeasible"
        assert info["primal_infeasibility_certificate"] is not None
    else:
        assert expected_updates == [True, True]
        for observed in rho_decisions:
            torch.testing.assert_close(
                observed["rho_vec"][0],
                1000.0 * observed["committed_rho"],
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
            torch.testing.assert_close(
                observed["rho_vec"][1],
                observed["committed_rho"],
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
            assert observed["rho_vec"][2] == torch.as_tensor(
                1e-6, device=device, dtype=dtype
            )

    assert info["rho_updates"] == expected_refactorizations
    assert len(refactorizations) == expected_refactorizations
    assert info["factorizations_this_solve"] == 1 + expected_refactorizations

    for linear_solve in linear_solves:
        residual = (
            linear_solve["matrix"] @ linear_solve["solution"]
            - linear_solve["rhs"]
        )
        numerator = torch.linalg.vector_norm(residual, ord=float("inf"))
        denominator = (
            torch.linalg.matrix_norm(linear_solve["matrix"], ord=float("inf"))
            * torch.linalg.vector_norm(
                linear_solve["solution"], ord=float("inf")
            )
            + torch.linalg.vector_norm(linear_solve["rhs"], ord=float("inf"))
        )
        if bool((denominator == 0).item()):
            assert numerator == 0
        else:
            assert bool(
                (numerator / denominator <= 1000 * torch.finfo(dtype).eps).item()
            )

    if device == "cpu" and dtype == torch.float64:
        reference = osqp.OSQP()
        reference.setup(
            P=sparse.csc_matrix(P.numpy()),
            q=q.numpy(),
            A=sparse.csc_matrix(A.numpy()),
            l=l.numpy(),
            u=u.numpy(),
            rho=initial_rho,
            sigma=1e-6,
            alpha=1.6,
            scaling=0,
            max_iter=2,
            eps_abs=1e-10,
            eps_rel=1e-10,
            eps_prim_inf=1e-12,
            eps_dual_inf=1e-12,
            check_termination=0,
            check_dualgap=False,
            adaptive_rho=True,
            adaptive_rho_interval=1,
            adaptive_rho_tolerance=rho_tolerance,
            polishing=False,
            warm_starting=False,
            verbose=False,
        )
        official = reference.solve(raise_error=False)
        assert official.info.status_val == info["status_val"]
        assert official.info.iter == info["admm_iterations"] == 2
        assert official.info.rho_updates == info["rho_updates"]
        assert reference._solver.get_settings().rho == pytest.approx(
            info["rho_bar"], rel=1e-12
        )
        assert official.info.rho_estimate == pytest.approx(
            info["rho_estimate"], rel=1e-12
        )


@pytest.mark.parametrize("scaled_termination,expected_iterations", [(False, 1), (True, 63)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scaled_termination_matches_pinned_observable(
    scaled_termination, expected_iterations, device, dtype
):
    """F03: pinned auxil.c selects working- or original-coordinate checks."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    P = torch.tensor([[1e-12]], device=device, dtype=dtype)
    q = torch.tensor([-1e-6], device=device, dtype=dtype)
    A = torch.tensor([[1e-3]], device=device, dtype=dtype)
    l = torch.tensor([-1e-6], device=device, dtype=dtype)
    u = -l
    settings = _direct_settings(
        scaling=10,
        scaled_termination=scaled_termination,
        max_iter=500,
        eps_abs=1e-5,
        eps_rel=1e-5,
        check_termination=1,
        check_dualgap=True,
        adaptive_rho=False,
        polishing=False,
    )
    reference = osqp.OSQP()
    reference.setup(
        P=sparse.csc_matrix(P.cpu().double().numpy()),
        q=q.cpu().double().numpy(),
        A=sparse.csc_matrix(A.cpu().double().numpy()),
        l=l.cpu().double().numpy(),
        u=u.cpu().double().numpy(),
        scaling=10,
        scaled_termination=scaled_termination,
        max_iter=500,
        eps_abs=1e-5,
        eps_rel=1e-5,
        check_termination=1,
        check_dualgap=True,
        adaptive_rho=False,
        polishing=False,
        verbose=False,
    )
    official = reference.solve(raise_error=False)
    result, info = solve_torch_osqp_direct(P, q, A, l, u, settings)
    assert official.info.status_val == info["status_val"] == 1
    assert official.info.iter == info["admm_iterations"] == expected_iterations
    if dtype == torch.float32:
        # Single-precision native reference. The installed Python OSQP uses double.
        values = ((0.0010098546044901013, 3.1162926461547613e-7, 6.324633432086557e-6, -2.098570206499062e-10)
                  if scaled_termination else (0.0, 1.5999995994775418e-8, 9.999999974752427e-7, 0.0))
    else:
        values = (official.x.item(), official.info.prim_res, official.info.dual_res, official.info.duality_gap)
    tolerance = 2e-4 if dtype == torch.float32 else 2e-8
    assert info["scaled_termination"] is scaled_termination
    for observed, expected in zip(
        (result.item(), info["primal_residual"], info["dual_residual"], info["duality_gap"]), values
    ):
        assert observed == pytest.approx(expected, rel=tolerance, abs=1e-14)



@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_polishing_retains_duplicate_active_rows_in_source_order(device, dtype):
    """F04: pinned polish.c keeps both duplicate equality rows."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    P = torch.tensor([[1.0]], device=device, dtype=dtype)
    q = torch.zeros(1, device=device, dtype=dtype)
    A = torch.ones((2, 1), device=device, dtype=dtype)
    l = torch.ones(2, device=device, dtype=dtype)
    u = l.clone()
    settings = _direct_settings(
        scaling=0,
        max_iter=4000,
        eps_abs=1e-5,
        eps_rel=1e-5,
        check_termination=25,
        check_dualgap=False,
        adaptive_rho=False,
        polishing=True,
        polish_refine_iter=3,
    )
    result, info = solve_torch_osqp_direct(
        P, q, A, l, u, settings, TorchOSQPWorkspace()
    )
    reference = osqp.OSQP()
    reference.setup(
        P=sparse.csc_matrix(P.cpu().double().numpy()),
        q=q.cpu().double().numpy(),
        A=sparse.csc_matrix(A.cpu().double().numpy()),
        l=l.cpu().double().numpy(),
        u=u.cpu().double().numpy(),
        scaling=0,
        max_iter=4000,
        eps_abs=1e-5,
        eps_rel=1e-5,
        check_termination=25,
        check_dualgap=False,
        adaptive_rho=False,
        polishing=True,
        delta=1e-6,
        polish_refine_iter=3,
        verbose=False,
    )
    official = reference.solve(raise_error=False)
    assert official.info.status_val == info["status_val"] == 1
    assert official.info.status_polish == 1
    assert info["polishing_status"] in {"accepted", "rejected_no_improvement"}
    assert info["polishing_active_constraints"] == 2
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    np.testing.assert_allclose(
        result.cpu().flatten().numpy(), official.x, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        info["state"]["y"].cpu().numpy(), official.y,
        rtol=tolerance, atol=tolerance,
    )


def test_polishing_builds_active_rows_in_original_order(monkeypatch):
    """F04: mixed active rows are not reordered to put equalities first."""
    captured = []
    original_factorize = torch_osqp_module.QDLDLSolver.factorize

    def capture_factorization(self, matrix, **kwargs):
        captured.append(matrix.detach().clone())
        return original_factorize(self, matrix, **kwargs)

    monkeypatch.setattr(
        torch_osqp_module.QDLDLSolver, "factorize", capture_factorization
    )
    P = torch.eye(2, dtype=torch.float64)
    q = torch.tensor([-2.0, 0.0], dtype=torch.float64)
    A = torch.eye(2, dtype=torch.float64)
    l = torch.tensor([-torch.inf, 1.0], dtype=torch.float64)
    u = torch.ones(2, dtype=torch.float64)
    settings = _direct_settings(
        scaling=0,
        check_termination=25,
        check_dualgap=False,
        adaptive_rho=False,
        polishing=True,
        polish_refine_iter=0,
    )
    _, info = solve_torch_osqp_direct(
        P, q, A, l, u, settings, TorchOSQPWorkspace()
    )
    assert info["polishing_status"] in {"accepted", "rejected_no_improvement"}
    assert info["polishing_active_constraints"] == 2
    assert captured
    np.testing.assert_allclose(captured[-1][2:, :2].numpy(), A.numpy(), rtol=0, atol=0)


def test_polishing_uses_scaled_problem_coordinates(monkeypatch):
    """F05: pinned polish.c receives the internally scaled problem and state."""
    captured = []
    original_factorize = torch_osqp_module.QDLDLSolver.factorize

    def capture_factorization(self, matrix, **kwargs):
        captured.append(matrix.detach().clone())
        return original_factorize(self, matrix, **kwargs)

    monkeypatch.setattr(
        torch_osqp_module.QDLDLSolver, "factorize", capture_factorization
    )
    P = torch.tensor([[1e-8, 0.0], [0.0, 10.0]], dtype=torch.float64)
    q = torch.tensor([-1.0, -20.0], dtype=torch.float64)
    A = torch.tensor([[1e-4, 0.0], [0.0, 1.0]], dtype=torch.float64)
    l = torch.tensor([-torch.inf, 0.0], dtype=torch.float64)
    u = torch.tensor([1e-4, 1.0], dtype=torch.float64)
    settings = _direct_settings(
        scaling=10,
        max_iter=4000,
        eps_abs=1e-5,
        eps_rel=1e-5,
        check_termination=25,
        check_dualgap=False,
        adaptive_rho=False,
        polishing=True,
        polish_refine_iter=3,
    )
    expected_scaling = torch_osqp_module._scaling_for_problem(
        TorchOSQPWorkspace(), P, q, A, settings
    )
    expected_A = torch_osqp_module._scale_problem(
        P, q, A, l, u, expected_scaling
    )[2]
    _, info = solve_torch_osqp_direct(
        P, q, A, l, u, settings, TorchOSQPWorkspace()
    )
    assert info["polishing_active_constraints"] == 2
    assert captured
    np.testing.assert_allclose(
        captured[-1][2:, :2].numpy(), expected_A.numpy(), rtol=0, atol=0
    )
    reference = osqp.OSQP()
    reference.setup(
        P=sparse.csc_matrix(P.numpy()),
        q=q.numpy(),
        A=sparse.csc_matrix(A.numpy()),
        l=l.numpy(),
        u=u.numpy(),
        scaling=10,
        max_iter=4000,
        eps_abs=1e-5,
        eps_rel=1e-5,
        check_termination=25,
        adaptive_rho=False,
        polishing=True,
        delta=1e-6,
        polish_refine_iter=3,
        check_dualgap=False,
        verbose=False,
    )
    official = reference.solve(raise_error=False)
    assert official.info.status_val == info["status_val"] == 1
    np.testing.assert_allclose(
        info["state"]["x"].numpy(), official.x, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        info["state"]["y"].numpy(), official.y, rtol=1e-12, atol=1e-12
    )


def test_polishing_uses_sigma_for_direct_source_regularization(monkeypatch):
    """F06: pinned direct polishing uses sigma in both signed KKT blocks."""
    captured = []
    original_factorize = torch_osqp_module.QDLDLSolver.factorize

    def capture_factorization(self, matrix, **kwargs):
        captured.append(matrix.detach().clone())
        return original_factorize(self, matrix, **kwargs)

    monkeypatch.setattr(
        torch_osqp_module.QDLDLSolver, "factorize", capture_factorization
    )
    P = torch.ones((1, 1), dtype=torch.float64)
    q = torch.zeros(1, dtype=torch.float64)
    A = torch.ones((1, 1), dtype=torch.float64)
    l = torch.ones(1, dtype=torch.float64)
    u = l.clone()
    settings = _direct_settings(
        scaling=0,
        max_iter=4000,
        eps_abs=1e-3,
        eps_rel=1e-3,
        check_termination=25,
        sigma=1e-2,
        polish_delta=1e-6,
        check_dualgap=False,
        adaptive_rho=False,
        polishing=True,
        polish_refine_iter=0,
    )
    _, info = solve_torch_osqp_direct(
        P, q, A, l, u, settings, TorchOSQPWorkspace()
    )
    assert info["polishing_active_constraints"] == 1
    assert captured
    np.testing.assert_allclose(captured[-1][0, 0].item(), 1.01, rtol=0, atol=0)
    np.testing.assert_allclose(captured[-1][1, 1].item(), -1e-2, rtol=0, atol=0)
    assert info["polishing_status"] == "rejected_no_improvement"


def test_builtin_adapter_preserves_raw_official_polish_rejection():
    """F08: builtin output is not post-processed by the local Torch polish."""
    dtype = torch.float64
    H = torch.ones((1, 1), dtype=dtype)
    f = torch.zeros((1, 1), dtype=dtype)
    A = torch.ones((1, 1), dtype=dtype)
    b = torch.ones((1, 1), dtype=dtype)
    lower = torch.tensor([[-torch.inf]], dtype=dtype)
    upper = torch.tensor([[torch.inf]], dtype=dtype)
    settings = {
        "return_info": True,
        "polishing": True,
        "polish_refine_iter": 0,
        "sigma": 1e-2,
        "polish_delta": 1e-6,
        "scaling": 0,
        "max_iter": 4000,
        "eps_abs": 1e-3,
        "eps_rel": 1e-3,
        "check_termination": 25,
        "check_dualgap": False,
        "adaptive_rho": False,
        "warm_start": False,
    }
    result, info = solve_osqp_torch_qp(
        H, f, A, b, lower, upper, torch.device("cpu"), True,
        {"algebra": "builtin", "settings": settings}, TorchOSQPWorkspace(),
    )

    A_reference = sparse.vstack(
        (sparse.csc_matrix([[1.0]]), sparse.eye(1, format="csc")), format="csc"
    )
    reference = osqp.OSQP()
    reference.setup(
        sparse.csc_matrix([[1.0]]), np.array([0.0]), A_reference,
        np.array([1.0, -np.inf]), np.array([1.0, np.inf]),
        rho=0.1, sigma=1e-2, scaling=0, max_iter=4000,
        eps_abs=1e-3, eps_rel=1e-3, check_termination=25,
        check_dualgap=False, adaptive_rho=False, polishing=True,
        delta=1e-6, polish_refine_iter=0, warm_starting=False, verbose=False,
    )
    official = reference.solve(raise_error=False)

    assert info["status_val"] == official.info.status_val == 1
    assert info["admm_iterations"] == official.info.iter == 25
    assert info["polishing_status"] == official.info.status_polish == -1
    assert info["polishing_fallback"] is None
    np.testing.assert_allclose(result.flatten().numpy(), official.x, rtol=1e-14)
    np.testing.assert_allclose(info["solver_objective"], official.info.obj_val, rtol=0)
    np.testing.assert_allclose(info["objective"], official.info.obj_val, rtol=0)


def test_polishing_acceptance_uses_original_coordinate_residuals():
    """Pinned update_info unscales residuals before polish acceptance."""
    seed = 67
    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = 2 + seed % 7
    eigenvalues = torch.logspace(0, 6.0, n, dtype=dtype)
    basis, _ = torch.linalg.qr(
        torch.randn((n, n), generator=generator, dtype=dtype)
    )
    H = basis @ torch.diag(eigenvalues) @ basis.T
    f = torch.randn((n, 1), generator=generator, dtype=dtype)
    A = torch.ones((1, n), dtype=dtype)
    b = torch.tensor(0.25, dtype=dtype)
    lower = -torch.ones((n, 1), dtype=dtype)
    upper = torch.ones((n, 1), dtype=dtype)

    admm_x, admm_info = _solve(
        H, f, A, b, lower, upper, "torch", {"polishing": False}
    )
    polished_x, polished_info = _solve(
        H, f, A, b, lower, upper, "torch", {"polishing": True}
    )
    _, builtin_info = _solve(
        H, f, A, b, lower, upper, "builtin", {"polishing": True}
    )

    assert builtin_info["polishing_status"] == -1
    assert polished_info["polishing_status"] == "rejected_no_improvement"
    assert not polished_info["polishing_success"]
    torch.testing.assert_close(polished_x, admm_x, rtol=0, atol=0)
    assert polished_info["primal_residual"] == admm_info["primal_residual"]
    assert polished_info["dual_residual"] == admm_info["dual_residual"]
    assert polished_info["primal_residual"] <= polished_info["eps_primal"]
    assert polished_info["dual_residual"] <= polished_info["eps_dual"]

    _, scaled_info = _solve(
        H,
        f,
        A,
        b,
        lower,
        upper,
        "torch",
        {"polishing": True, "scaled_termination": True},
    )
    _, scaled_builtin_info = _solve(
        H,
        f,
        A,
        b,
        lower,
        upper,
        "builtin",
        {"polishing": True, "scaled_termination": True},
    )
    assert scaled_builtin_info["polishing_status"] == 1
    assert scaled_info["polishing_status"] == "accepted"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scaling", [0, 10])
@pytest.mark.parametrize("polishing", [False, True])
def test_duality_gap_scaled_objectives_and_polish_metadata(device, dtype, scaling, polishing):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    # A well-conditioned fixture keeps floating-point error separate from F03.
    P = torch.diag(torch.tensor([2.0, 3.0], device=device, dtype=dtype))
    q = torch.tensor([-4.0, -6.0], device=device, dtype=dtype)
    A = torch.eye(2, device=device, dtype=dtype)
    l, u = torch.zeros_like(q), torch.ones_like(q)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-8
    settings = _direct_settings(
        scaling=scaling, polishing=polishing, adaptive_rho=False, check_termination=1,
        eps_abs=tolerance, eps_rel=tolerance, check_dualgap=True,
    )
    _, info = solve_torch_osqp_direct(P, q, A, l, u, settings)
    reference = osqp.OSQP()
    reference.setup(
        P=sparse.csc_matrix(P.cpu().double().numpy()), q=q.cpu().double().numpy(),
        A=sparse.eye(2, format="csc"), l=np.zeros(2), u=np.ones(2),
        scaling=scaling, polishing=polishing, adaptive_rho=False, check_termination=1,
        eps_abs=tolerance, eps_rel=tolerance, check_dualgap=True, verbose=False,
    )
    official = reference.solve(raise_error=False)
    assert info["status_val"] == official.info.status_val == 1
    if polishing and dtype == torch.float64:
        assert info["polishing_success"]
    elif polishing and not info["polishing_success"]:
        # Float32 may reject refinement; gap metadata must retain the ADMM values.
        _, unpolished = solve_torch_osqp_direct(
            P, q, A, l, u, {**settings, "polishing": False}
        )
        assert info["polishing_status"] == "rejected_no_improvement"
        for key in ("objective", "dual_objective", "duality_gap", "eps_duality_gap"):
            assert info[key] == unpolished[key]
        assert all(torch.equal(info["state"][key], unpolished["state"][key])
                   for key in ("x", "z", "y"))
    elif not polishing:
        assert info["polishing_status"] == "disabled"
    x, y = info["state"]["x"], info["state"]["y"]
    # Analytic original-coordinate expressions, independent of the new helper.
    quad = torch.dot(x, P @ x).item()
    lin = torch.dot(q, x).item()
    support = torch.dot(u, y.clamp_min(0)).item() + torch.dot(l, y.clamp_max(0)).item()
    # Dot products plus the Ruiz round trip: bound error by precision and magnitudes.
    rounding = 32 * torch.finfo(dtype).eps * max(abs(quad), abs(lin), abs(support), 1)
    assert info["duality_gap"] == pytest.approx(quad + lin + support, abs=rounding)
    assert info["dual_objective"] == pytest.approx(-.5 * quad - support, abs=rounding)
    assert info["eps_duality_gap"] == pytest.approx(
        tolerance * (1 + max(abs(quad), abs(lin), abs(support))), abs=rounding * tolerance,
    )
    assert info["objective"] == pytest.approx(official.info.obj_val, abs=10 * tolerance)
    assert info["duality_gap"] == pytest.approx(official.info.duality_gap, abs=10 * tolerance)


def _dense_from_csc(matrix, shape):
    indptr = matrix["indptr"]
    indices = torch.tensor(matrix["indices"], dtype=torch.int64)
    values = torch.tensor(matrix["data"], dtype=torch.float64)
    dense = torch.zeros(shape, dtype=torch.float64)
    for column in range(shape[1]):
        start, stop = indptr[column : column + 2]
        dense[indices[start:stop], column] = values[start:stop]
    return dense


def _pinned_large_qp_fixture():
    fixture_path = Path(__file__).with_name("pinned_osqp_large_qp_fixture.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    n, m = fixture["n"], fixture["m"]
    P_upper = _dense_from_csc(fixture["P"], (n, n))
    P = P_upper + torch.triu(P_upper, diagonal=1).T
    A = _dense_from_csc(fixture["A"], (m, n))
    return {
        **fixture,
        "P_csc": fixture["P"],
        "A_csc": fixture["A"],
        "P": P,
        "A": A,
        "q": torch.tensor(fixture["q"], dtype=torch.float64),
        "l": torch.tensor(fixture["l"], dtype=torch.float64),
        "u": torch.tensor(fixture["u"], dtype=torch.float64),
    }


def _pinned_basic_qp_fixture():
    dtype = torch.float64
    return {
        "P": torch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=dtype),
        "q": torch.ones(2, dtype=dtype),
        "A": torch.tensor(
            [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=dtype,
        ),
        "l": torch.tensor([1.0, 0.0, 0.0, -torch.inf], dtype=dtype),
        "u": torch.tensor([1.0, 0.7, 0.7, torch.inf], dtype=dtype),
        "q_new": torch.tensor([2.5, 3.2], dtype=dtype),
        "l_new": torch.tensor([0.8, -3.4, -torch.inf, 0.5], dtype=dtype),
        "u_new": torch.tensor([1.6, 1.0, torch.inf, 0.5], dtype=dtype),
    }


def _pinned_update_matrices_fixture():
    """Reproduce the pinned tests/update_matrices generated fixture."""
    generator = np.random.Generator(np.random.PCG64(1))
    n, m = 5, 8
    A = sparse.random(m, n, density=0.7, format="csc", random_state=generator)
    P_seed = sparse.random(n, n, density=0.7, random_state=generator)
    P = (P_seed @ P_seed.T).tocsc() + sparse.eye(n, format="csc")
    P_upper = sparse.triu(P, format="csc")

    A_new = A.copy()
    A_new.data += generator.standard_normal(A_new.nnz)
    P_upper_new = P_upper.copy()
    P_upper_new.data += 0.1 * generator.standard_normal(P_upper_new.nnz)
    P_new = P_upper_new + P_upper_new.T - sparse.diags(P_upper_new.diagonal())

    dtype = torch.float64
    return {
        "P": torch.from_numpy(P.toarray()).to(dtype=dtype),
        "P_upper": P_upper,
        "P_new": torch.from_numpy(P_new.toarray()).to(dtype=dtype),
        "P_upper_new": P_upper_new,
        "q": torch.from_numpy(generator.standard_normal(n)).to(dtype=dtype),
        "A": torch.from_numpy(A.toarray()).to(dtype=dtype),
        "A_csc": A,
        "A_new": torch.from_numpy(A_new.toarray()).to(dtype=dtype),
        "A_new_csc": A_new,
        "l": torch.from_numpy(-30 + generator.standard_normal(m)).to(dtype=dtype),
        "u": torch.from_numpy(30 + generator.standard_normal(m)).to(dtype=dtype),
        "expected_x": torch.tensor(
            [-0.34282147, 0.0260615, 0.27197987, -0.01202531, 0.44052732],
            dtype=dtype,
        ),
        "expected_x_p_new": torch.tensor(
            [-0.36033252, 0.10729745, 0.22965285, 0.06323582, 0.42301398],
            dtype=dtype,
        ),
    }


def _pinned_basic_qp2_fixture():
    dtype = torch.float64
    return {
        "P": torch.diag(torch.tensor([11.0, 0.0], dtype=dtype)),
        "q": torch.tensor([3.0, 4.0], dtype=dtype),
        "A": torch.tensor(
            [
                [-1.0, 0.0],
                [0.0, -1.0],
                [-1.0, 3.0],
                [2.0, 5.0],
                [3.0, 4.0],
            ],
            dtype=dtype,
        ),
        "l": torch.full((5,), -torch.inf, dtype=dtype),
        "u": torch.tensor([0.0, 0.0, -15.0, 100.0, 80.0], dtype=dtype),
    }


def _assert_relative_fixture_observables(
    info, solution, expected_x, expected_y, objective
):
    state_y = info["state"]["y"]
    assert info["status"] == "solved"
    assert info["status_val"] == 1
    assert (
        torch.linalg.vector_norm(solution.reshape(-1) - expected_x, ord=float("inf"))
        / torch.linalg.vector_norm(expected_x, ord=float("inf"))
        < 1e-4
    )
    assert (
        torch.linalg.vector_norm(state_y - expected_y, ord=float("inf"))
        / torch.linalg.vector_norm(expected_y, ord=float("inf"))
        < 1e-4
    )
    assert abs(info["objective"] - objective) / abs(objective) < 1e-4


@pytest.mark.parametrize("with_equality", [False, True])
def test_builtin_and_torch_agree_on_observable_outcomes(with_equality):
    dtype = torch.float64
    H = torch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=dtype)
    f = torch.tensor([[-1.0], [-1.0]], dtype=dtype)
    lower = torch.zeros((2, 1), dtype=dtype)
    upper = torch.ones((2, 1), dtype=dtype)
    A = torch.tensor([[1.0, 1.0]], dtype=dtype) if with_equality else None
    b = torch.tensor(0.75, dtype=dtype) if with_equality else None
    torch_solution, torch_info = _solve(H, f, A, b, lower, upper, "torch")
    builtin_solution, builtin_info = _solve(H, f, A, b, lower, upper, "builtin")
    assert torch_info["status_compatible"]
    assert builtin_info["status_compatible"]
    assert torch.allclose(torch_solution, builtin_solution, atol=2e-6, rtol=2e-6)
    objective_scale = max(1.0, abs(builtin_info["objective"]))
    assert (
        abs(torch_info["objective"] - builtin_info["objective"]) / objective_scale
        < 1e-7
    )


@pytest.mark.parametrize(
    "H,f,expected_objective",
    [
        (torch.zeros((2, 2)), torch.tensor([[1.1], [0.95]]), 0.995),
        (
            torch.tensor([[4.0, 1.0], [1.0, 2.0]]),
            torch.ones((2, 1)),
            1.88,
        ),
    ],
    ids=["pinned-basic-lp", "pinned-basic-qp"],
)
def test_pinned_basic_problem_observable_parity(H, f, expected_objective):
    dtype = torch.float64
    H = H.to(dtype=dtype)
    f = f.to(dtype=dtype)
    A = torch.tensor([[1.0, 1.0]], dtype=dtype)
    b = torch.tensor(1.0, dtype=dtype)
    lower = torch.zeros((2, 1), dtype=dtype)
    upper = torch.full((2, 1), 0.7, dtype=dtype)
    expected = torch.tensor([[0.3], [0.7]], dtype=dtype)

    torch_solution, torch_info = _solve(
        H, f, A, b, lower, upper, "torch", {"polishing": False}
    )
    builtin_solution, builtin_info = _solve(
        H, f, A, b, lower, upper, "builtin", {"polishing": False}
    )

    for solution, info in (
        (torch_solution, torch_info),
        (builtin_solution, builtin_info),
    ):
        assert info["status_compatible"]
        assert info["status"] == "solved"
        assert torch.allclose(solution, expected, atol=2e-6, rtol=2e-6)
        assert info["objective"] == pytest.approx(expected_objective, abs=2e-6)


@pytest.mark.parametrize(
    "polishing,expected_polishing_status",
    [(False, "disabled"), (True, "accepted")],
)
def test_pinned_basic_lp_polishing_branches(polishing, expected_polishing_status):
    dtype = torch.float64
    solution, info = solve_torch_osqp_direct(
        torch.zeros((2, 2), dtype=dtype),
        torch.tensor([1.1, 0.95], dtype=dtype),
        torch.tensor(
            [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=dtype,
        ),
        torch.tensor([1.0, 0.0, 0.0, -torch.inf], dtype=dtype),
        torch.tensor([1.0, 0.7, 0.7, torch.inf], dtype=dtype),
        _direct_settings(
            max_iter=2000,
            eps_abs=1e-5,
            eps_rel=1e-5,
            scaling=1,
            polishing=polishing,
            polish_refine_iter=4,
        ),
        TorchOSQPWorkspace(),
    )

    assert info["status"] == "solved"
    assert info["status_val"] == 1
    assert torch.allclose(
        solution.reshape(-1),
        torch.tensor([0.3, 0.7], dtype=dtype),
        atol=1e-4,
        rtol=0,
    )
    assert torch.allclose(
        info["state"]["y"],
        torch.tensor([-1.1, 0.0, 0.15, 0.0], dtype=dtype),
        atol=1e-4,
        rtol=0,
    )
    assert info["objective"] == pytest.approx(0.995, abs=1e-4)
    assert info["polishing_status"] == expected_polishing_status


def test_pinned_basic_qp_solve_combination():
    problem = _pinned_basic_qp_fixture()
    solution, info = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        _direct_settings(
            max_iter=2000,
            eps_abs=1e-5,
            eps_rel=1e-5,
            scaling=0,
            warm_start=False,
            polishing=True,
            polish_refine_iter=3,
        ),
        TorchOSQPWorkspace(),
    )

    assert info["status"] == "solved"
    assert info["status_val"] == 1
    assert torch.allclose(
        solution.reshape(-1),
        torch.tensor([0.3, 0.7], dtype=torch.float64),
        atol=1e-4,
        rtol=0,
    )
    assert torch.allclose(
        info["state"]["y"],
        torch.tensor([-2.9, 0.0, 0.2, 0.0], dtype=torch.float64),
        atol=1e-4,
        rtol=0,
    )
    assert info["objective"] == pytest.approx(1.88, abs=1e-4)
    assert not info["scaling_applied"]
    assert not info["factorization_reused"]
    assert info["polishing_status"] == "accepted"


def test_pinned_derivative_fixture_forward_solve_observables():
    """Port the five forward-solve assertions that precede adjoint API calls."""
    problem = _pinned_basic_qp_fixture()
    solution, info = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        _direct_settings(
            max_iter=2000,
            eps_abs=1e-5,
            eps_rel=1e-5,
            scaling=1,
            polishing=False,
            polish_refine_iter=3,
        ),
        TorchOSQPWorkspace(),
    )

    assert info["status"] == "solved"
    assert info["status_val"] == 1
    assert torch.allclose(
        solution.reshape(-1),
        torch.tensor([0.3, 0.7], dtype=torch.float64),
        atol=1e-4,
        rtol=0,
    )
    assert torch.allclose(
        info["state"]["y"],
        torch.tensor([-2.9, 0.0, 0.2, 0.0], dtype=torch.float64),
        atol=1e-4,
        rtol=0,
    )
    assert info["objective"] == pytest.approx(1.88, abs=1e-4)
    assert info["polishing_status"] == "disabled"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pinned_basic_qp_cuda_output_observables(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    problem = {
        name: value.to(device="cuda", dtype=dtype)
        for name, value in _pinned_basic_qp_fixture().items()
        if name in {"P", "q", "A", "l", "u"}
    }
    solution, info = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        _direct_settings(
            rho=0.1,
            alpha=1.6,
            max_iter=2000,
            eps_abs=1e-5,
            eps_rel=1e-5,
            scaling=0,
            warm_start=False,
            polishing=True,
            polish_refine_iter=3,
        ),
        TorchOSQPWorkspace(),
    )

    tolerance = 1e-3 if dtype == torch.float32 else 1e-4
    assert info["status"] == "solved"
    assert info["status_val"] == 1
    assert solution.device.type == "cuda"
    assert solution.dtype == dtype
    assert info["state"]["y"].device.type == "cuda"
    assert info["state"]["y"].dtype == dtype
    assert torch.allclose(
        solution.reshape(-1),
        torch.tensor([0.3, 0.7], device="cuda", dtype=dtype),
        atol=tolerance,
        rtol=0,
    )
    assert torch.allclose(
        info["state"]["y"],
        torch.tensor([-2.9, 0.0, 0.2, 0.0], device="cuda", dtype=dtype),
        atol=tolerance,
        rtol=0,
    )
    assert info["objective"] == pytest.approx(1.88, abs=tolerance)
    assert info["polishing_status"] == "accepted"


def test_pinned_basic_qp_disabled_periodic_checks_still_run_final_termination():
    problem = _pinned_basic_qp_fixture()
    settings = _direct_settings(
        max_iter=400,
        alpha=1.6,
        eps_abs=1e-3,
        eps_rel=1e-3,
        scaling=0,
        check_termination=0,
        warm_start=False,
        polishing=False,
    )
    solution, info = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        settings,
        TorchOSQPWorkspace(),
    )

    official_solver = osqp.OSQP()
    official_solver.setup(
        P=sparse.csc_matrix(np.triu(problem["P"].numpy())),
        q=problem["q"].numpy(),
        A=sparse.csc_matrix(problem["A"].numpy()),
        l=problem["l"].numpy(),
        u=problem["u"].numpy(),
        max_iter=400,
        alpha=1.6,
        eps_abs=1e-3,
        eps_rel=1e-3,
        scaling=0,
        check_termination=0,
        warm_starting=False,
        polishing=False,
        verbose=False,
    )
    official_result = official_solver.solve(raise_error=False)

    expected_x = torch.tensor([0.3, 0.7], dtype=torch.float64)
    expected_y = torch.tensor([-2.9, 0.0, 0.2, 0.0], dtype=torch.float64)
    assert info["admm_iterations"] == 400
    assert official_result.info.iter == 400
    assert info["status"] == "solved"
    assert info["status_val"] == 1
    assert info["status_compatible"]
    assert info["primal_infeasibility_certificate"] is None
    assert info["dual_infeasibility_certificate"] is None
    assert official_result.info.status == "solved"
    assert official_result.info.status_val == 1
    assert torch.allclose(solution.reshape(-1), expected_x, atol=1e-4, rtol=0)
    assert np.allclose(official_result.x, expected_x.numpy(), atol=1e-4, rtol=0)
    assert torch.allclose(info["state"]["y"], expected_y, atol=1e-4, rtol=0)
    assert np.allclose(official_result.y, expected_y.numpy(), atol=1e-4, rtol=0)
    assert info["objective"] == pytest.approx(1.88, abs=1e-4)
    assert official_result.info.obj_val == pytest.approx(1.88, abs=1e-4)


def test_pinned_basic_qp_time_limit_status_matches_official_wall_clock():
    problem = _pinned_basic_qp_fixture()
    settings = _direct_settings(
        rho=20.0,
        adaptive_rho=False,
        time_limit=1e-12,
        max_iter=2_000_000_000,
        eps_abs=1e-12,
        eps_rel=1e-12,
        check_termination=0,
        warm_start=False,
        polishing=False,
    )
    solution, info = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        settings,
        TorchOSQPWorkspace(),
    )

    official_solver = osqp.OSQP()
    official_solver.setup(
        P=sparse.csc_matrix(np.triu(problem["P"].numpy())),
        q=problem["q"].numpy(),
        A=sparse.csc_matrix(problem["A"].numpy()),
        l=problem["l"].numpy(),
        u=problem["u"].numpy(),
        rho=20.0,
        adaptive_rho=False,
        time_limit=1e-12,
        max_iter=2_000_000_000,
        eps_abs=1e-12,
        eps_rel=1e-12,
        check_termination=0,
        warm_starting=False,
        polishing=False,
        verbose=False,
    )
    official_result = official_solver.solve(raise_error=False)

    assert info["status"] == "time_limit_reached"
    assert info["status_val"] == official_result.info.status_val == 8
    assert info["admm_iterations"] == 0
    assert info["linear_solves_this_solve"] == 1
    assert not info["status_compatible"]
    assert torch.all(torch.isfinite(solution))
    assert np.all(np.isfinite(official_result.x))
    assert info["primal_infeasibility_certificate"] is None
    assert info["dual_infeasibility_certificate"] is None


def test_pinned_basic_qp_warm_start_iteration_invariants():
    problem = _pinned_basic_qp_fixture()
    settings = _direct_settings(
        eps_abs=1e-3,
        eps_rel=1e-3,
        check_termination=1,
        adaptive_rho=False,
        polishing=False,
    )
    workspace = TorchOSQPWorkspace()
    _, first = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        settings,
        workspace,
    )

    workspace.state = {
        "x": torch.zeros(2, dtype=torch.float64),
        "z": torch.zeros(4, dtype=torch.float64),
        "y": torch.zeros(4, dtype=torch.float64),
    }
    _, cold = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        settings,
        workspace,
    )
    assert cold["admm_iterations"] == first["admm_iterations"]

    x_opt = torch.tensor([0.3, 0.7], dtype=torch.float64)
    workspace.state = {
        "x": x_opt,
        "z": problem["A"] @ x_opt,
        "y": torch.tensor([-2.9, 0.0, 0.2, 0.0], dtype=torch.float64),
    }
    solution, warm = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        settings,
        workspace,
    )
    assert warm["admm_iterations"] == 1
    assert torch.allclose(solution.reshape(-1), x_opt, atol=1e-8, rtol=0)


def test_pinned_basic_qp_requested_rho_matches_fresh_setup_iterations():
    problem = _pinned_basic_qp_fixture()
    common = {
        "eps_abs": 5e-5,
        "eps_rel": 5e-5,
        "check_termination": 1,
        "adaptive_rho": False,
        "polishing": False,
        "warm_start": False,
    }
    _, fresh = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        _direct_settings(rho=0.7, **common),
        TorchOSQPWorkspace(),
    )

    workspace = TorchOSQPWorkspace()
    solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        _direct_settings(rho=0.1, **common),
        workspace,
    )
    solution, updated = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        _direct_settings(rho=0.7, **common),
        workspace,
    )

    assert updated["admm_iterations"] == fresh["admm_iterations"]
    assert updated["factorizations_this_solve"] >= 1
    assert not updated["factorization_reused"]
    assert torch.allclose(
        solution.reshape(-1), torch.tensor([0.3, 0.7], dtype=torch.float64), atol=1e-4
    )


def test_pinned_unconstrained_problem_observable_parity():
    dtype = torch.float64
    diagonal = torch.tensor(
        [0.617022, 0.92032449, 0.20011437, 0.50233257, 0.34675589],
        dtype=dtype,
    )
    H = torch.diag(diagonal)
    f = torch.tensor(
        [[-1.10593508], [-1.65451545], [-2.3634686], [1.13534535], [-1.01701414]],
        dtype=dtype,
    )
    lower = torch.full((5, 1), -torch.inf, dtype=dtype)
    upper = torch.full((5, 1), torch.inf, dtype=dtype)
    expected = -f.reshape(-1) / diagonal

    torch_solution, torch_info = _solve(
        H, f, None, None, lower, upper, "torch", {"polishing": False}
    )
    builtin_solution, builtin_info = _solve(
        H, f, None, None, lower, upper, "builtin", {"polishing": False}
    )

    for solution, info in (
        (torch_solution, torch_info),
        (builtin_solution, builtin_info),
    ):
        assert info["status_compatible"]
        assert torch.allclose(solution.reshape(-1), expected, atol=2e-6, rtol=2e-6)
        assert info["objective"] == pytest.approx(-19.209752026813277, abs=2e-6)


@pytest.mark.parametrize(
    "polishing,expected_polishing_status",
    [(False, "disabled"), (True, "skipped_no_active_constraints")],
)
def test_pinned_unconstrained_zero_row_fixture_and_polishing_branches(
    polishing, expected_polishing_status
):
    dtype = torch.float64
    P = torch.diag(
        torch.tensor(
            [0.617022, 0.92032449, 0.20011437, 0.50233257, 0.34675589],
            dtype=dtype,
        )
    )
    q = torch.tensor(
        [-1.10593508, -1.65451545, -2.3634686, 1.13534535, -1.01701414],
        dtype=dtype,
    )
    expected = torch.tensor(
        [1.79237542, 1.79775228, 11.81058885, -2.26014678, 2.93293975],
        dtype=dtype,
    )

    solution, info = solve_torch_osqp_direct(
        P,
        q,
        torch.empty((0, 5), dtype=dtype),
        torch.empty(0, dtype=dtype),
        torch.empty(0, dtype=dtype),
        _direct_settings(polishing=polishing, polish_refine_iter=4),
        TorchOSQPWorkspace(),
    )

    assert info["status"] == "solved"
    assert info["status_val"] == 1
    assert torch.allclose(solution.reshape(-1), expected, atol=1e-5, rtol=0)
    assert info["objective"] == pytest.approx(-19.209752026813277, abs=1e-6)
    assert info["primal_residual"] == 0.0
    assert info["state"]["z"].numel() == 0
    assert info["state"]["y"].numel() == 0
    assert info["polishing_status"] == expected_polishing_status
    assert info["polishing_active_constraints"] == 0


def test_pinned_large_qp_observable_parity_and_dense_capacity():
    problem = _pinned_large_qp_fixture()
    n, m = problem["n"], problem["m"]
    P, A = problem["P"], problem["A"]
    q, l, u = problem["q"], problem["l"], problem["u"]

    assert problem["source_commit"] == "1572ae068e9ce9ca723cf8223548ade1ff7acc29"
    assert problem["source_blob"] == "0c7075774c99bd39161b173a362c4c157845725c"
    assert (n, m) == (160, 270)
    for matrix, shape, nnz in (
        (problem["P_csc"], (n, n), 155),
        (problem["A_csc"], (m, n), 1770),
    ):
        assert len(matrix["indptr"]) == shape[1] + 1
        assert len(matrix["indices"]) == len(matrix["data"]) == nnz
        assert matrix["indptr"][0] == 0
        assert matrix["indptr"][-1] == nnz
        assert all(
            start <= stop for start, stop in zip(matrix["indptr"], matrix["indptr"][1:])
        )
        assert all(0 <= row < shape[0] for row in matrix["indices"])
        for column in range(shape[1]):
            start, stop = matrix["indptr"][column : column + 2]
            rows = matrix["indices"][start:stop]
            assert rows == sorted(set(rows))
    assert torch.count_nonzero(q) == 0
    assert torch.all(torch.isfinite(P))
    assert torch.all(torch.isfinite(A))
    assert torch.all(torch.isfinite(l))
    assert torch.all(torch.isfinite(u))
    assert torch.all(l <= u)
    assert torch.count_nonzero(l == u) == 110
    assert torch.all(l[:110] == u[:110])
    assert torch.count_nonzero(A[110:], dim=1).equal(torch.ones(n, dtype=torch.int64))
    bound_diagonal = A[110 + torch.arange(n), torch.arange(n)]
    assert torch.all(bound_diagonal > 0)
    assert not torch.equal(bound_diagonal, torch.ones_like(bound_diagonal))

    eigenvalues = torch.linalg.eigvalsh(P)
    assert eigenvalues.min() >= -1e-12
    assert torch.count_nonzero(eigenvalues > 1e-12) == 110
    assert torch.count_nonzero(torch.abs(eigenvalues) <= 1e-12) == 50

    estimated_kkt_dim, estimated_memory_mb = estimate_dense_kkt(
        P,
        q,
        A[:110],
        l[:110],
        torch.float64,
    )
    assert estimated_kkt_dim == 430
    assert estimated_memory_mb == pytest.approx(5.08651733398438)
    assert estimated_memory_mb < 512.0

    P_data = problem["P_csc"]
    A_data = problem["A_csc"]
    official_solver = osqp.OSQP()
    official_solver.setup(
        P=sparse.csc_matrix(
            (
                np.asarray(P_data["data"]),
                np.asarray(P_data["indices"]),
                np.asarray(P_data["indptr"]),
            ),
            shape=(n, n),
        ),
        q=q.numpy(),
        A=sparse.csc_matrix(
            (
                np.asarray(A_data["data"]),
                np.asarray(A_data["indices"]),
                np.asarray(A_data["indptr"]),
            ),
            shape=(m, n),
        ),
        l=l.numpy(),
        u=u.numpy(),
        rho=0.1,
        alpha=1.6,
        max_iter=2000,
        scaling=1,
        eps_abs=1e-5,
        eps_rel=1e-5,
        polishing=False,
        verbose=False,
    )
    official_result = official_solver.solve(raise_error=False)

    solution, info = solve_torch_osqp_direct(
        P,
        q,
        A,
        l,
        u,
        _direct_settings(
            rho=0.1,
            alpha=1.6,
            max_iter=2000,
            scaling=1,
            eps_abs=1e-5,
            eps_rel=1e-5,
            check_termination=25,
            polishing=False,
        ),
        TorchOSQPWorkspace(),
    )

    expected_objective = problem["expected_objective"]
    assert official_result.info.status == "solved"
    assert official_result.info.status_val == 1
    assert info["status"] == "solved"
    assert info["status_val"] == 1
    assert info["backend"] == "torch"
    assert info["estimated_kkt_dim"] == 430
    assert info["factorization_count"] >= 1
    assert info["primal_infeasibility_certificate"] is None
    assert info["dual_infeasibility_certificate"] is None
    assert torch.all(torch.isfinite(solution))
    assert torch.all(torch.isfinite(info["state"]["y"]))
    assert np.isfinite(official_result.x).all()
    assert np.isfinite(official_result.y).all()

    for objective in (official_result.info.obj_val, info["objective"]):
        assert abs(objective - expected_objective) / expected_objective < 1e-4
    assert info["objective"] == pytest.approx(official_result.info.obj_val, rel=1e-8)

    x = solution.reshape(-1)
    Ax = A @ x
    primal_violation = torch.maximum(
        torch.clamp(l - Ax, min=0).max(),
        torch.clamp(Ax - u, min=0).max(),
    )
    dual_residual = torch.linalg.vector_norm(
        P @ x + q + A.T @ info["state"]["y"], ord=float("inf")
    )
    assert primal_violation < info["eps_primal"]
    assert dual_residual < info["eps_dual"]
    assert info["primal_residual"] == pytest.approx(
        official_result.info.prim_res, abs=1e-10
    )
    assert info["dual_residual"] == pytest.approx(
        official_result.info.dual_res, abs=1e-10
    )


@pytest.mark.skipif(not pinned_dll_path().is_file(), reason="pinned C DLL unavailable")
def test_b3_seed1_matches_pinned_acceptance_without_extra_diagonal_guard(
    available_device,
):
    """Translate source acceptance without claiming nonconvex optimality."""
    dtype = torch.float64
    sigma = 1e-6
    P = torch.tensor(
        [[-1.0684888085421562e-8]], device=available_device, dtype=dtype
    )
    q = torch.tensor(
        [-5.44810519187422e-6], device=available_device, dtype=dtype
    )
    A = torch.ones((1, 1), device=available_device, dtype=dtype)
    l = torch.tensor([-1.0], device=available_device, dtype=dtype)
    u = torch.tensor([1.0], device=available_device, dtype=dtype)
    assert float((P[0, 0] + sigma).item()) > 0.0

    reference = osqp.OSQP()
    reference.setup(
        P=sparse.csc_matrix(P.detach().cpu().numpy()),
        q=q.detach().cpu().numpy(),
        A=sparse.eye(1, format="csc"),
        l=l.detach().cpu().numpy(),
        u=u.detach().cpu().numpy(),
        sigma=sigma,
        max_iter=4000,
        eps_abs=1e-5,
        eps_rel=1e-5,
        check_termination=25,
        scaling=10,
        adaptive_rho=True,
        polishing=False,
        verbose=False,
    )
    official = reference.solve(raise_error=False)
    assert official.info.status_val == 1
    source_settings = dict(
        sigma=sigma, max_iter=4000, eps_abs=1e-5, eps_rel=1e-5,
        check_termination=25, scaling=10, adaptive_rho=1, polishing=0,
    )
    pinned = solve_pinned_c_dense_qp(
        *(t.detach().cpu().numpy() for t in (P, q, A, l, u)),
        settings_overrides=source_settings,
    )
    solution, info = solve_torch_osqp_direct(
        P, q, A, l, u, _direct_settings(**source_settings, check_convexity=False),
    )
    assert pinned["status_val"] == 1
    assert info["status"] == "solved"
    assert info["admm_iterations"] == pinned["iter"] == official.info.iter
    np.testing.assert_allclose(solution.detach().cpu().reshape(-1).numpy(),
                               pinned["x"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("sigma", [1e-6, 5.0])
def test_pinned_nonconvex_setup_sign_check_without_optional_psd_check(
    sigma, floating_dtype, available_device
):
    problem = _pinned_basic_qp2_fixture()
    problem = {
        name: value.to(device=available_device, dtype=floating_dtype)
        if torch.is_tensor(value)
        else value
        for name, value in problem.items()
    }
    problem["P"] = torch.tensor(
        [[2.0, 5.0], [5.0, 1.0]],
        device=available_device,
        dtype=floating_dtype,
    )
    settings = _direct_settings(
        sigma=sigma,
        scaling=1,
        eps_abs=1e-5,
        eps_rel=1e-5,
        adaptive_rho=False,
        polishing=False,
        max_iter=1,
        check_convexity=False,
    )
    if sigma == 1e-6:
        with pytest.raises(TorchLinearSolveError, match="KKT inertia"):
            solve_torch_osqp_direct(
                problem["P"], problem["q"], problem["A"],
                problem["l"], problem["u"], settings
            )
    else:
        _, info = solve_torch_osqp_direct(
            problem["P"], problem["q"], problem["A"],
            problem["l"], problem["u"], settings
        )
        assert info["factorizations_this_solve"] == 1
        assert info["admm_iterations"] == 1


@pytest.mark.parametrize("sigma", [1e-6, 5.0])
def test_pinned_nonconvex_matrix_is_rejected_by_optional_torch_check(sigma):
    dtype = torch.float64
    H = torch.tensor([[2.0, 5.0], [5.0, 1.0]], dtype=dtype)
    f = torch.tensor([[3.0], [4.0]], dtype=dtype)
    lower = torch.zeros((2, 1), dtype=dtype)
    upper = torch.full((2, 1), torch.inf, dtype=dtype)

    with pytest.raises(ValueError, match="positive semidefinite"):
        _solve(
            H,
            f,
            None,
            None,
            lower,
            upper,
            "torch",
            {"check_convexity": True, "sigma": sigma, "polishing": False},
        )


@pytest.mark.parametrize("sigma", [1e-6, 1.0])
def test_reference_zero_pivot_rejection_changes_with_duplicate_rows(sigma):
    # Diagnostic reference behavior, not a claim that this nonconvex QP is valid.
    setup_codes = []
    for rows in (1, 2):
        reference = osqp.OSQP()
        try:
            reference.setup(
                P=sparse.csc_matrix([[-sigma]]), q=np.zeros(1),
                A=sparse.csc_matrix(np.ones((rows, 1))),
                l=-np.ones(rows), u=np.ones(rows),
                sigma=sigma, rho=0.1, scaling=0,
                adaptive_rho=False, polishing=False, verbose=False,
            )
        except osqp.OSQPException as exc:
            setup_codes.append(int(exc.args[0]))
        else:
            setup_codes.append(0)
    # The variable block is zero in both; a blanket zero-block rejection
    # would disagree with the reference's second, successful setup.
    assert setup_codes == [4, 0]


@pytest.mark.parametrize("sigma", [1e-6, 1.0])
def test_reference_zero_pivot_depends_on_explicit_sparse_zero_storage(sigma):
    implicit = sparse.csc_matrix([[1.0], [0.0]])
    explicit = sparse.csc_matrix(
        (np.array([1.0, 0.0]), np.array([0, 1]), np.array([0, 2])),
        shape=(2, 1),
    )
    assert implicit.nnz == 1 and explicit.nnz == 2
    assert np.array_equal(implicit.toarray(), explicit.toarray())
    dense_inputs = []
    setup_codes = []
    for A in (implicit, explicit):
        coo = A.tocoo()
        tensor = torch.sparse_coo_tensor(
            np.stack((coo.row, coo.col)), coo.data,
            size=coo.shape, dtype=torch.float64, check_invariants=True,
        )
        dense_inputs.append(tensor.to_dense())
        reference = osqp.OSQP()
        try:
            reference.setup(
                P=sparse.csc_matrix([[-sigma]]), q=np.zeros(1), A=A,
                l=-np.ones(2), u=np.ones(2),
                sigma=sigma, rho=0.1, scaling=0,
                adaptive_rho=False, polishing=False, verbose=False,
            )
        except osqp.OSQPException as exc:
            setup_codes.append(int(exc.args[0]))
        else:
            setup_codes.append(0)
    # Densification loses information on which reference setup depends.
    assert torch.equal(dense_inputs[0], dense_inputs[1])
    assert setup_codes == [4, 0]
    # Both dense inputs encode the implicit-zero pattern seen by AMD.
    for A in dense_inputs:
        with pytest.raises(TorchLinearSolveError, match="zeros in the diagonal"):
            solve_torch_osqp_direct(
                A.new_tensor([[-sigma]]), A.new_zeros(1), A,
                -A.new_ones(2), A.new_ones(2),
                _direct_settings(sigma=sigma, scaling=0, check_convexity=False),
            )


@pytest.mark.parametrize("sigma", [1e-6, 1.0])
def test_convex_zero_hessian_accepts_both_sparse_zero_encodings(sigma):
    implicit = sparse.csc_matrix([[1.0], [0.0]])
    explicit = sparse.csc_matrix(
        (np.array([1.0, 0.0]), np.array([0, 1]), np.array([0, 2])),
        shape=(2, 1),
    )
    dense_inputs = []
    for A in (implicit, explicit):
        reference = osqp.OSQP()
        reference.setup(
            P=sparse.csc_matrix((1, 1)),
            q=np.zeros(1),
            A=A,
            l=-np.ones(2),
            u=np.ones(2),
            sigma=sigma,
            rho=0.1,
            scaling=0,
            adaptive_rho=False,
            polishing=False,
            verbose=False,
        )
        result = reference.solve(raise_error=False)
        assert int(result.info.status_val) == 1
        assert np.array_equal(result.x, np.zeros(1))

        coo = A.tocoo()
        tensor = torch.sparse_coo_tensor(
            np.stack((coo.row, coo.col)),
            coo.data,
            size=coo.shape,
            dtype=torch.float64,
            check_invariants=True,
        ).to_dense()
        dense_inputs.append(tensor)
        solution, info = solve_torch_osqp_direct(
            tensor.new_zeros((1, 1)),
            tensor.new_zeros(1),
            tensor,
            -tensor.new_ones(2),
            tensor.new_ones(2),
            _direct_settings(
                sigma=sigma,
                rho=0.1,
                scaling=0,
                adaptive_rho=False,
                polishing=False,
            ),
        )
        assert info["status"] == "solved"
        assert torch.equal(solution.reshape(-1), tensor.new_zeros(1))
    assert torch.equal(dense_inputs[0], dense_inputs[1])


def test_builtin_nonconvex_status_never_exposes_the_osqp_no_solution_sentinel():
    dtype = torch.float64
    H = torch.tensor([[2.0, 5.0], [5.0, 1.0]], dtype=dtype)
    f = torch.tensor([[3.0], [4.0]], dtype=dtype)
    lower = torch.full((2, 1), -torch.inf, dtype=dtype)
    upper = torch.full((2, 1), torch.inf, dtype=dtype)

    solution, info = _solve(
        H,
        f,
        None,
        None,
        lower,
        upper,
        "builtin",
        {
            "sigma": 5.0,
            "adaptive_rho": False,
            "polishing": False,
        },
    )

    assert info["status"] == "non_convex"
    assert info["status_val"] == 9
    assert not info["status_compatible"]
    assert torch.all(torch.isnan(solution))
    assert info["objective"] != info["objective"]
    assert info["primal_infeasibility_certificate"] is None
    assert info["dual_infeasibility_certificate"] is None


def test_pinned_nonconvex_solve_returns_source_status_without_false_certificate():
    problem = _pinned_basic_qp2_fixture()
    settings = _direct_settings(
        rho=0.1,
        sigma=5.0,
        alpha=1.6,
        adaptive_rho=False,
        max_iter=2000,
        scaling=1,
        eps_abs=1e-5,
        eps_rel=1e-5,
        check_termination=25,
        warm_start=True,
        polishing=True,
    )
    workspace = TorchOSQPWorkspace()
    _, valid_info = solve_torch_osqp_direct(
        torch.eye(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        -torch.ones(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        settings,
        workspace,
    )
    assert valid_info["status_compatible"]
    assert workspace.work_iterates is not None

    problem["P"] = torch.tensor(
        [[2.0, 5.0], [5.0, 1.0]], dtype=torch.float64
    )
    eigenvalues = torch.linalg.eigvalsh(problem["P"])
    shifted_eigenvalues = torch.linalg.eigvalsh(
        problem["P"] + 5.0 * torch.eye(2, dtype=torch.float64)
    )
    assert eigenvalues[0] < 0.0
    assert shifted_eigenvalues[0] > 0.0

    solution, info = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        settings,
        workspace,
    )

    official_solver = osqp.OSQP()
    official_solver.setup(
        P=sparse.csc_matrix(np.triu(problem["P"].numpy())),
        q=problem["q"].numpy(),
        A=sparse.csc_matrix(problem["A"].numpy()),
        l=problem["l"].numpy(),
        u=problem["u"].numpy(),
        rho=0.1,
        sigma=5.0,
        alpha=1.6,
        adaptive_rho=False,
        max_iter=2000,
        scaling=1,
        eps_abs=1e-5,
        eps_rel=1e-5,
        check_termination=25,
        warm_starting=True,
        polishing=True,
        verbose=False,
    )
    official_result = official_solver.solve(raise_error=False)

    assert official_result.info.status_val == 9
    assert np.isnan(official_result.info.obj_val)
    assert info["status"] == "non_convex"
    assert info["status_val"] == 9
    assert info["polishing_status"] == "skipped_unsolved"
    assert not info["status_compatible"]
    assert info["primal_residual"] > 1e30 or info["dual_residual"] > 1e30
    assert info["factorizations_this_solve"] >= 1
    assert info["linear_solves_this_solve"] >= 1
    assert torch.all(solution == 0x7FC00000)
    assert info["objective"] == 0x7FC00000
    assert info["primal_infeasibility_certificate"] is None
    assert info["dual_infeasibility_certificate"] is None
    assert "state" not in info
    # store_solution cold-starts iterates while keeping the current problem.
    assert workspace.state is None and workspace.work_iterates is None
    assert workspace.linear_solver.ready
    assert workspace.rho_bar == settings["rho"]
    assert torch.equal(workspace.scaling_source_p, problem["P"])
    assert torch.equal(workspace.scaling_source_a, problem["A"])
    assert workspace.last_info["status"] == "non_convex"


def test_bound_value_update_reuses_existing_backend_workspace():
    dtype = torch.float64
    H = torch.eye(2, dtype=dtype)
    f = -2.0 * torch.ones((2, 1), dtype=dtype)
    lower = torch.zeros((2, 1), dtype=dtype)
    initial_upper = torch.ones((2, 1), dtype=dtype)
    updated_upper = torch.tensor([[0.5], [0.75]], dtype=dtype)
    expected = updated_upper
    results = {}

    for algebra in ("torch", "builtin"):
        workspace = TorchOSQPWorkspace()
        _solve(
            H,
            f,
            None,
            None,
            lower,
            initial_upper,
            algebra,
            {"polishing": False},
            workspace,
        )
        results[algebra] = _solve(
            H,
            f,
            None,
            None,
            lower,
            updated_upper,
            algebra,
            {"polishing": False},
            workspace,
        )

    for solution, info in results.values():
        assert info["status_compatible"]
        assert torch.allclose(solution, expected, atol=2e-6, rtol=2e-6)
    assert results["torch"][1]["factorizations_this_solve"] == 0
    assert results["torch"][1]["factorization_reused"]
    assert results["builtin"][1]["builtin_workspace"]["last_cache_hit"]


def test_pinned_basic_qp_invalid_bound_update_is_transactional():
    problem = _pinned_basic_qp_fixture()
    P, q, A = problem["P"], problem["q"], problem["A"]
    original_l, original_u = problem["l"], problem["u"]
    updated_l, updated_u = problem["l_new"], problem["u_new"]
    settings = _direct_settings(
        max_iter=200,
        scaling=0,
        warm_start=False,
        polishing=True,
        polish_refine_iter=3,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )
    workspace = TorchOSQPWorkspace()

    solve_torch_osqp_direct(
        P,
        q,
        A,
        original_l,
        original_u,
        settings,
        workspace,
    )
    state_before = {key: value.clone() for key, value in workspace.state.items()}
    matrix_before = workspace.linear_solver.matrix.clone()
    constraint_types_before = workspace.constraint_types.clone()
    last_info_before = workspace.last_info
    factorization_count_before = workspace.linear_solver.factorization_count
    solve_count_before = workspace.linear_solver.solve_count
    rho_bar_before = workspace.rho_bar
    problem_signature_before = workspace.problem_signature

    invalid_updates = (
        (updated_u, updated_l),
        (updated_u, original_u),
        (original_l, updated_l),
    )
    for invalid_l, invalid_u in invalid_updates:
        with pytest.raises(ValueError, match="lower bound"):
            solve_torch_osqp_direct(
                P,
                q,
                A,
                invalid_l,
                invalid_u,
                settings,
                workspace,
            )
        assert workspace.last_info is last_info_before
        assert workspace.problem_signature == problem_signature_before
        assert workspace.rho_bar == rho_bar_before
        assert workspace.linear_solver.factorization_count == factorization_count_before
        assert workspace.linear_solver.solve_count == solve_count_before
        assert torch.equal(workspace.linear_solver.matrix, matrix_before)
        assert torch.equal(workspace.constraint_types, constraint_types_before)
        for key, value in state_before.items():
            assert torch.equal(workspace.state[key], value)

    solution, info = solve_torch_osqp_direct(
        P,
        q,
        A,
        updated_l,
        updated_u,
        settings,
        workspace,
    )

    official_solver = osqp.OSQP()
    official_solver.setup(
        P=sparse.csc_matrix(np.triu(P.numpy())),
        q=q.numpy(),
        A=sparse.csc_matrix(A.numpy()),
        l=updated_l.numpy(),
        u=updated_u.numpy(),
        max_iter=200,
        scaling=0,
        warm_starting=False,
        polishing=True,
        eps_abs=1e-5,
        eps_rel=1e-5,
        verbose=False,
    )
    official_result = official_solver.solve(raise_error=False)

    assert info["status"] == "solved"
    assert info["status_val"] == 1
    assert not info["factorization_reused"]
    assert info["factorizations_this_solve"] >= 1
    assert official_result.info.status == "solved"
    assert official_result.info.status_val == 1
    assert torch.allclose(
        solution.reshape(-1),
        torch.from_numpy(official_result.x),
        atol=1e-5,
        rtol=1e-5,
    )
    assert info["objective"] == pytest.approx(
        official_result.info.obj_val,
        abs=1e-5,
    )


@pytest.mark.parametrize(
    "update_name,expect_factor_reuse",
    [("q", True), ("l", False), ("u", False)],
    ids=["linear-cost", "lower-bounds", "upper-bounds"],
)
def test_pinned_basic_qp_valid_vector_updates_match_official(
    update_name, expect_factor_reuse
):
    problem = _pinned_basic_qp_fixture()
    settings = _direct_settings(
        max_iter=500,
        scaling=0,
        adaptive_rho=False,
        warm_start=False,
        polishing=True,
        polish_refine_iter=3,
        eps_abs=1e-5,
        eps_rel=1e-5,
    )
    workspace = TorchOSQPWorkspace()
    solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        settings,
        workspace,
    )

    updated_problem = {
        "q": problem["q"],
        "l": problem["l"],
        "u": problem["u"],
    }
    updated_problem[update_name] = problem[f"{update_name}_new"]
    solution, info = solve_torch_osqp_direct(
        problem["P"],
        updated_problem["q"],
        problem["A"],
        updated_problem["l"],
        updated_problem["u"],
        settings,
        workspace,
    )

    official_solver = osqp.OSQP()
    official_solver.setup(
        P=sparse.csc_matrix(np.triu(problem["P"].numpy())),
        q=problem["q"].numpy(),
        A=sparse.csc_matrix(problem["A"].numpy()),
        l=problem["l"].numpy(),
        u=problem["u"].numpy(),
        max_iter=500,
        scaling=0,
        adaptive_rho=False,
        warm_starting=False,
        polishing=True,
        polish_refine_iter=3,
        eps_abs=1e-5,
        eps_rel=1e-5,
        verbose=False,
    )
    official_solver.update(**{update_name: problem[f"{update_name}_new"].numpy()})
    official_result = official_solver.solve(raise_error=False)

    assert info["status"] == "solved"
    assert info["status_val"] == 1
    assert info["factorization_reused"] is expect_factor_reuse
    assert info["factorizations_this_solve"] == (0 if expect_factor_reuse else 1)
    assert official_result.info.status == "solved"
    assert official_result.info.status_val == 1
    assert torch.allclose(
        solution.reshape(-1),
        torch.from_numpy(official_result.x),
        atol=1e-5,
        rtol=1e-5,
    )
    assert info["objective"] == pytest.approx(
        official_result.info.obj_val,
        abs=1e-5,
    )


def test_pinned_update_matrices_kkt_formation_and_update():
    problem = _pinned_update_matrices_fixture()
    sigma = 0.1
    rho_vec = torch.full((8,), 1.6, dtype=torch.float64)

    for P, A in (
        (problem["P"], problem["A"]),
        (problem["P_new"], problem["A_new"]),
    ):
        expected = torch.zeros((13, 13), dtype=torch.float64)
        expected[:5, :5] = P + sigma * torch.eye(5, dtype=torch.float64)
        expected[:5, 5:] = A.T
        expected[5:, :5] = A
        expected[5:, 5:] = -torch.eye(8, dtype=torch.float64) / 1.6

        actual = torch_osqp_module.build_kkt_matrix(P, A, sigma, rho_vec)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "update_p,update_a",
    [(True, False), (False, True), (True, True)],
    ids=["P-values", "A-values", "P-and-A-values"],
)
def test_pinned_matrix_value_updates_match_official(update_p, update_a):
    problem = _pinned_update_matrices_fixture()
    settings = _direct_settings(
        max_iter=1000,
        eps_abs=1e-5,
        eps_rel=1e-5,
        scaling=1,
        adaptive_rho=True,
        warm_start=True,
        polishing=False,
    )
    workspace = TorchOSQPWorkspace()

    initial_solution, initial_info = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        settings,
        workspace,
    )
    initial_kkt = workspace.linear_solver.matrix.clone()
    initial_scaling = workspace.scaling
    initial_factorization_count = workspace.linear_solver.factorization_count

    official_solver = osqp.OSQP()
    official_solver.setup(
        P=problem["P_upper"],
        q=problem["q"].numpy(),
        A=problem["A_csc"],
        l=problem["l"].numpy(),
        u=problem["u"].numpy(),
        max_iter=1000,
        eps_abs=1e-5,
        eps_rel=1e-5,
        scaling=1,
        adaptive_rho=True,
        warm_starting=True,
        polishing=False,
        verbose=False,
    )
    initial_official = official_solver.solve(raise_error=False)

    assert initial_info["status"] == "solved"
    assert initial_info["status_val"] == 1
    assert initial_official.info.status == "solved"
    assert initial_official.info.status_val == 1
    assert torch.allclose(
        initial_solution.reshape(-1), problem["expected_x"], atol=1e-4, rtol=0
    )
    assert np.allclose(
        initial_official.x, problem["expected_x"].numpy(), atol=1e-4, rtol=0
    )
    assert torch.allclose(
        initial_info["state"]["y"], torch.zeros(8, dtype=torch.float64), atol=1e-4
    )
    assert np.allclose(initial_official.y, np.zeros(8), atol=1e-4, rtol=0)

    updated_p = problem["P_new"] if update_p else problem["P"]
    updated_a = problem["A_new"] if update_a else problem["A"]
    assert torch.equal(problem["P"].ne(0), problem["P_new"].ne(0))
    assert torch.equal(problem["A"].ne(0), problem["A_new"].ne(0))
    solution, info = solve_torch_osqp_direct(
        updated_p,
        problem["q"],
        updated_a,
        problem["l"],
        problem["u"],
        settings,
        workspace,
    )

    official_update = {}
    if update_p:
        official_update["Px"] = problem["P_upper_new"].data
    if update_a:
        official_update["Ax"] = problem["A_new_csc"].data
    official_solver.update(**official_update)
    official_result = official_solver.solve(raise_error=False)

    expected_x = problem["expected_x_p_new"] if update_p else problem["expected_x"]
    assert info["status"] == "solved"
    assert info["status_val"] == 1
    assert official_result.info.status == "solved"
    assert official_result.info.status_val == 1
    assert torch.allclose(solution.reshape(-1), expected_x, atol=1e-4, rtol=0)
    assert np.allclose(official_result.x, expected_x.numpy(), atol=1e-4, rtol=0)
    assert torch.allclose(
        info["state"]["y"], torch.zeros(8, dtype=torch.float64), atol=1e-4, rtol=0
    )
    assert np.allclose(official_result.y, np.zeros(8), atol=1e-4, rtol=0)
    assert torch.allclose(
        solution.reshape(-1), torch.from_numpy(official_result.x), atol=1e-4, rtol=0
    )
    assert torch.allclose(
        info["state"]["y"], torch.from_numpy(official_result.y), atol=1e-4, rtol=0
    )
    assert info["objective"] == pytest.approx(official_result.info.obj_val, abs=1e-4)

    assert not info["factorization_reused"]
    assert info["factorizations_this_solve"] >= 1
    assert workspace.linear_solver.factorization_count > initial_factorization_count
    assert not torch.equal(workspace.linear_solver.matrix, initial_kkt)
    assert torch.equal(workspace.scaling_source_p, updated_p)
    assert torch.equal(workspace.scaling_source_a, updated_a)
    assert workspace.scaling is not initial_scaling
    assert torch.all(torch.isfinite(workspace.scaling["D"]))
    assert torch.all(torch.isfinite(workspace.scaling["E"]))
    assert torch.isfinite(workspace.scaling["cost"])
    assert workspace.state is not None


@pytest.mark.parametrize(
    "polishing,expected_polishing_status",
    [(False, "disabled"), (True, "accepted")],
)
def test_pinned_basic_qp2_solve_observables_and_polishing(
    polishing, expected_polishing_status
):
    problem = _pinned_basic_qp2_fixture()
    solution, info = solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        _direct_settings(
            max_iter=2000,
            eps_abs=1e-6,
            eps_rel=1e-6,
            scaling=1,
            polishing=polishing,
            polish_refine_iter=4,
        ),
        TorchOSQPWorkspace(),
    )

    _assert_relative_fixture_observables(
        info,
        solution,
        torch.tensor([15.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 508.0, 168.0, 0.0, 0.0], dtype=torch.float64),
        1282.5,
    )
    assert info["polishing_status"] == expected_polishing_status


def test_pinned_basic_qp2_vector_update_reuses_dense_factors():
    problem = _pinned_basic_qp2_fixture()
    workspace = TorchOSQPWorkspace()
    settings = _direct_settings(
        max_iter=4000,
        eps_abs=1e-3,
        eps_rel=1e-3,
        scaling=10,
        warm_start=True,
        polishing=True,
        polish_refine_iter=3,
    )
    updated_q = torch.tensor([1.0, 1.0], dtype=torch.float64)
    updated_u = torch.tensor(
        [-2.0, 0.0, -20.0, 100.0, 80.0], dtype=torch.float64
    )
    expected_x = torch.tensor([20.0, 0.0], dtype=torch.float64)
    expected_y = torch.tensor(
        [0.0, 664.0, 221.0, 0.0, 0.0], dtype=torch.float64
    )

    fresh_solution, fresh_info = solve_torch_osqp_direct(
        problem["P"],
        updated_q,
        problem["A"],
        problem["l"],
        updated_u,
        settings,
        TorchOSQPWorkspace(),
    )
    _assert_relative_fixture_observables(
        fresh_info, fresh_solution, expected_x, expected_y, 2220.0
    )
    assert fresh_info["polishing_status"] == "accepted"

    solve_torch_osqp_direct(
        problem["P"],
        problem["q"],
        problem["A"],
        problem["l"],
        problem["u"],
        settings,
        workspace,
    )

    solution, info = solve_torch_osqp_direct(
        problem["P"],
        updated_q,
        problem["A"],
        problem["l"],
        updated_u,
        settings,
        workspace,
    )

    _assert_relative_fixture_observables(
        info, solution, expected_x, expected_y, 2220.0
    )
    assert info["polishing_status"] == "accepted"
    assert info["factorization_reused"]
    assert info["factorizations_this_solve"] == 0


@pytest.mark.parametrize("family", ["primal", "dual"])
def test_builtin_and_torch_agree_on_certificate_backed_infeasibility(family):
    dtype = torch.float64
    if family == "primal":
        problem = (
            torch.ones((1, 1), dtype=dtype),
            torch.zeros((1, 1), dtype=dtype),
            torch.ones((1, 1), dtype=dtype),
            torch.tensor(2.0, dtype=dtype),
            torch.zeros((1, 1), dtype=dtype),
            torch.ones((1, 1), dtype=dtype),
        )
        certificate_key = "primal_infeasibility_certificate"
        nonmatching_key = "dual_infeasibility_certificate"
        expected_codes = {3, 4}
    else:
        problem = (
            torch.zeros((1, 1), dtype=dtype),
            -torch.ones((1, 1), dtype=dtype),
            None,
            None,
            torch.full((1, 1), -torch.inf, dtype=dtype),
            torch.full((1, 1), torch.inf, dtype=dtype),
        )
        certificate_key = "dual_infeasibility_certificate"
        nonmatching_key = "primal_infeasibility_certificate"
        expected_codes = {5, 6}

    torch_workspace = TorchOSQPWorkspace()
    with pytest.raises(RuntimeError, match=rf"{family}_infeasible"):
        _solve(
            *problem,
            "torch",
            {"polishing": False},
            torch_workspace,
        )
    torch_info = torch_workspace.last_info
    builtin_workspace = TorchOSQPWorkspace()
    builtin_solution, builtin_info = _solve(
        *problem,
        "builtin",
        {"polishing": False},
        builtin_workspace,
    )

    for info in (torch_info, builtin_info):
        certificate = info[certificate_key]
        assert info["status"].startswith(f"{family}_infeasible")
        assert info["status_val"] in expected_codes
        assert not info["status_compatible"]
        assert certificate is not None
        assert torch.all(torch.isfinite(certificate))
        assert torch.linalg.vector_norm(certificate, ord=float("inf")) == pytest.approx(
            1.0
        )
        assert info[nonmatching_key] is None
    assert torch.all(torch.isnan(builtin_solution))
    assert builtin_workspace.builtin_cache is None
