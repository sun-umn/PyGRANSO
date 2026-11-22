import pytest
import torch

from pygranso.private import torchOSQP as torch_osqp_module
from pygranso.private.osqpTorchAdapter import (
    DEFAULT_OSQP_SETTINGS,
    solve_osqp_torch_qp,
)
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchLinearSolve import TorchLinearSolveError
from pygranso.private.torchQDLDLSolve import QDLDLSolver
from pygranso.private.torchOSQP import (
    _adaptive_rho_update,
    _compute_obj_val_dual_gap,
    _polish_solution,
    _rho_vector,
    _scale_problem,
    _scaling_for_problem,
    _unscale_state,
    solve_torch_osqp_direct,
)


@pytest.mark.parametrize(
    "diagonal_hex",
    [
        "0x1.0000000000000p+2",
        "0x1.fffffffffffffp+1021",
        "0x1.ffffffffffffep+1022",
        "0x1.fffffffffffffp+1022",
        "0x1.0000000000000p+1023",
        "0x1.ffffffffffffep+1023",
        "0x1.fffffffffffffp+1023",
    ],
)
def test_finite_upper_hessian_keeps_float64_diagonal(diagonal_hex):
    p = torch.tensor([[float.fromhex(diagonal_hex)]], dtype=torch.float64)
    q = p.new_zeros(1)
    a = p.new_zeros((0, 1))
    l = u = p.new_empty(0)

    reconstructed, *_ = torch_osqp_module._validate_qp(p, q, a, l, u, {})

    assert reconstructed.item().hex() == diagonal_hex


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_upper_hessian_mirrors_off_diagonal_entries(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    upper = torch.tensor(
        [[torch.finfo(dtype).max, -1.0], [0.0, 3.0]], dtype=dtype, device=device
    )
    expected = upper.clone()
    expected[1, 0] = upper[0, 1]
    q = upper.new_zeros(2)
    a = upper.new_zeros((0, 2))
    l = u = upper.new_empty(0)

    for p in (upper, upper.to_sparse_coo()):
        reconstructed, *_ = torch_osqp_module._validate_qp(p, q, a, l, u, {})
        assert reconstructed.dtype == dtype and reconstructed.device == upper.device
        assert torch.equal(reconstructed, expected)

    ordinary_full = expected.clone()
    ordinary_full[0, 0] = 4.0
    reconstructed, *_ = torch_osqp_module._validate_qp(ordinary_full, q, a, l, u, {})
    assert torch.equal(reconstructed, ordinary_full)


@pytest.mark.parametrize("scaling", [0, 1])
def test_extreme_hessian_reaches_finite_dense_lu_setup(scaling):
    p = torch.tensor([[float.fromhex("0x1.0000000000000p+1023")]], dtype=torch.float64)
    z = p.new_zeros(1)
    a = p.new_ones((1, 1))
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": scaling,
        "adaptive_rho": 0,
        "polishing": False,
        "check_termination": 1,
        "check_dualgap": False,
        "max_iter": 1,
        "return_info": True,
    }
    workspace = TorchOSQPWorkspace()

    bounds = a.flatten()
    x, info = solve_torch_osqp_direct(p, z, a, -bounds, bounds, settings, workspace)

    assert info["status"] == "solved"
    assert torch.equal(x.flatten(), z)
    assert info["factorizations_this_solve"] == 1
    assert torch.isfinite(workspace.linear_solver.matrix).all()
    if not scaling:
        assert workspace.linear_solver.matrix[0, 0].item().hex() == p.item().hex()


@pytest.fixture
def coordinate_qp():
    dtype = torch.float64
    matrix = torch.full((1, 1), 9.0, dtype=dtype)
    q = torch.tensor([-10.0], dtype=dtype)
    bounds = torch.zeros(1, dtype=dtype)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "scaling": 1,
        "adaptive_rho": 0,
        "check_termination": 1,
        "check_dualgap": False,
        "eps_abs": 1e4,
        "max_iter": 1,
        "polishing": False,
        "return_info": True,
        "return_state": True,
    }
    return (matrix, q, matrix, bounds, bounds), settings


@pytest.mark.parametrize("scaled_termination", [False, True])
def test_public_coordinates_use_the_pinned_cost_reciprocal(
    coordinate_qp, scaled_termination, monkeypatch
):
    qp, settings = coordinate_qp
    settings["scaled_termination"] = scaled_termination
    x = y = torch.full((1,), 0.5625, dtype=torch.float64)
    zero = torch.zeros_like(x)
    monkeypatch.setattr(
        torch_osqp_module, "admm_vector_update", lambda *args, **kw: (x, zero, y, zero)
    )

    solution, info = solve_torch_osqp_direct(*qp, settings, TorchOSQPWorkspace())

    assert solution.item().hex() == "0x1.8000000000000p-3"
    assert info["state"]["y"].item().hex() == "0x1.4000000000000p-1"


def test_warm_coordinates_use_pinned_inverse_factors(coordinate_qp, monkeypatch):
    qp, settings = coordinate_qp
    workspace = TorchOSQPWorkspace()
    solve_torch_osqp_direct(*qp, settings, workspace)
    dtype = torch.float64
    workspace.state = {
        "x": torch.tensor([float.fromhex("0x1.7ffffffffffffp-3")], dtype=dtype),
        "z": torch.tensor([float.fromhex("0x1.affffffffffffp+0")], dtype=dtype),
        "y": torch.tensor([0.1875], dtype=dtype),
    }
    original = torch_osqp_module._initial_scaled_state
    observed = []

    def capture_state(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(tuple(value.item().hex() for value in result))
        return result

    monkeypatch.setattr(torch_osqp_module, "_initial_scaled_state", capture_state)
    _, info = solve_torch_osqp_direct(*qp, settings, workspace)

    assert info["factorization_reused"]
    assert observed == [
        ("0x1.1ffffffffffffp-1", "0x1.1ffffffffffffp-1", "0x1.599999999999ap-3")
    ]


@pytest.mark.parametrize(
    "mode,requested,check_termination,expected",
    [
        (0, 0, 25, 0),
        (1, 0, 25, 100),
        (1, 0, 0, 100),
        (1, 7, 25, 7),
        (2, 0, 25, 0),
        (2, 7, 25, 7),
        (3, 0, 25, 1),
        (3, 7, 25, 7),
    ],
)
def test_source_rho_setup_interval_modes(
    mode, requested, check_termination, expected,
):
    assert torch_osqp_module._setup_rho_interval(
        mode, requested, check_termination
    ) == expected


@pytest.mark.parametrize(
    "iteration,check_termination,elapsed,expected",
    [
        (100, 25, 0.39, 0),
        (100, 25, 0.4, 0),
        (1, 25, 0.4000000001, 1),
        (12, 0, 0.4000000001, 0),
        (13, 0, 0.4000000001, 1),
    ],
)
def test_time_rho_auto_interval_uses_strict_threshold_and_executable_clamp(
    iteration, check_termination, elapsed, expected,
):
    assert torch_osqp_module._time_rho_interval(
        iteration=iteration,
        check_termination=check_termination,
        elapsed=elapsed,
        setup_time=1.0,
        fraction=0.4,
    ) == expected


@pytest.mark.parametrize(
    "primal,dual,gap,expected",
    [
        (1.0, 2.0, 3.0, 3.0),
        (1.0, 2.0, 0.0, 2.0),
        (1.0, 2.0, -100.0, 2.0),
    ],
)
def test_relative_kkt_error_uses_signed_gap_without_normalization(
    primal, dual, gap, expected,
):
    values = [torch.tensor(value, dtype=torch.float64) for value in (primal, dual, gap)]
    assert torch_osqp_module._relative_kkt_error(*values) == values[0].new_tensor(expected)


@pytest.mark.parametrize(
    "relative_kkt,expected",
    [(1.999999, True), (2.0, True), (2.000001, False)],
)
def test_kkt_rho_eligibility_uses_inclusive_fraction_threshold(
    relative_kkt, expected,
):
    assert torch_osqp_module._kkt_rho_eligible(
        torch.tensor(relative_kkt, dtype=torch.float64), 0.5, 4.0
    ) is expected


@pytest.mark.parametrize(
    "current_error, expected_calls",
    [
        ("0x1.fffffffffffffp-2", 1),
        ("0x1.0000000000000p-1", 1),
        ("0x1.0000000000001p-1", 0),
    ],
    ids=["immediately-below", "equal", "immediately-above"],
)
def test_kkt_rho_first_eligibility_uses_pinned_finite_history(
    current_error, expected_calls, monkeypatch
):
    dtype = torch.float64
    P = A = torch.ones((1, 1), dtype=dtype)
    q = torch.tensor([-1.0], dtype=dtype)
    l = torch.zeros(1, dtype=dtype)
    u = torch.full((1,), 0.25, dtype=dtype)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "adaptive_rho": 3,
        # This fraction times the pinned binary64 OSQP_INFTY rounds to 0.5.
        "adaptive_rho_fraction": float.fromhex("0x1.4484bfeebc2a0p-101"),
        "rho_update_interval": 0,
        "rho_update_tolerance": 1e12,
        "rho_update_safeguard": False,
        "rho": 0.125,
        "sigma": 0.125,
        "alpha": 1.0,
        "check_termination": 0,
        "check_dualgap": False,
        "max_iter": 1,
        "scaling": 0,
        "polishing": False,
        "warm_start": False,
        "return_info": True,
    }
    error = float.fromhex(current_error)
    primed_residuals = False
    observed_errors = []
    adaptation_calls = []
    original_residuals = torch_osqp_module._residuals
    original_eligibility = torch_osqp_module._kkt_rho_eligible
    original_adaptation = torch_osqp_module._adaptive_rho_update

    def fixed_initial_residuals(*args, **kwargs):
        nonlocal primed_residuals
        result = original_residuals(*args, **kwargs)
        if primed_residuals:
            return result
        primed_residuals = True
        # Leave the later residual inputs to actual rho computation intact.
        return result[0].new_tensor(error), result[1].new_tensor(0.125), *result[2:]

    def record_eligibility(current, fraction, history):
        observed_errors.append(current.item().hex())
        return original_eligibility(current, fraction, history)

    def record_adaptation(*args, **kwargs):
        adaptation_calls.append(True)
        return original_adaptation(*args, **kwargs)

    monkeypatch.setattr(torch_osqp_module, "_residuals", fixed_initial_residuals)
    monkeypatch.setattr(torch_osqp_module, "_kkt_rho_eligible", record_eligibility)
    monkeypatch.setattr(torch_osqp_module, "_adaptive_rho_update", record_adaptation)
    _, info = solve_torch_osqp_direct(P, q, A, l, u, settings, TorchOSQPWorkspace())

    assert observed_errors == [current_error]
    assert len(adaptation_calls) == expected_calls
    assert info["rho_updates"] == 0
    assert info["factorizations_this_solve"] == 1


def test_kkt_rho_mode_updates_history_only_after_an_actual_rho_change(monkeypatch):
    P = torch.ones((1, 1), dtype=torch.float64)
    q = torch.tensor([-1.0], dtype=torch.float64)
    A = torch.ones((1, 1), dtype=torch.float64)
    l, u = torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "rho_update_safeguard": False,
        "adaptive_rho": 3,
        "adaptive_rho_fraction": 0.5,
        "rho_update_interval": 0,
        "check_termination": 0,
        "max_iter": 3,
        "scaling": 0,
        "polishing": False,
        "warm_start": False,
        "return_info": True,
    }
    observed_kkt = iter((4.0, 3.0, 1.0, 1.0))
    update_calls = []

    def relative_kkt(*values):
        return values[0].new_tensor(next(observed_kkt))

    def adaptive_update(rho, equality, *args, loose=None, **kwargs):
        update_calls.append(float(rho.item()))
        if len(update_calls) == 1:
            candidate = 2.0 * rho
            return True, candidate, _rho_vector(candidate, equality, loose)
        return False, rho, _rho_vector(rho, equality, loose)

    monkeypatch.setattr(torch_osqp_module, "_relative_kkt_error", relative_kkt)
    monkeypatch.setattr(torch_osqp_module, "_adaptive_rho_update", adaptive_update)
    workspace = TorchOSQPWorkspace()
    _, info = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)

    # Iteration 1 is eligible against the initial sentinel. Iteration 2 fails
    # 3 <= .5*4. Iteration 3 is eligible, but a rejected rho candidate does not
    # replace the history value.
    assert update_calls == [0.1, 0.2]
    assert workspace.last_rel_kkt == 4.0
    assert workspace.adaptive_rho_interval == 1
    assert info["rho_updates"] == 1
    assert info["factorizations_this_solve"] == 2
    assert info["linear_solves_this_solve"] == 3


def test_kkt_rho_mode_checks_termination_before_adaptation(monkeypatch):
    P = torch.ones((1, 1), dtype=torch.float64)
    q = torch.tensor([-1.0], dtype=torch.float64)
    A = torch.ones((1, 1), dtype=torch.float64)
    l, u = torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "rho_update_safeguard": False,
        "adaptive_rho": 3,
        "rho_update_interval": 1,
        "check_termination": 1,
        "max_iter": 2,
        "scaling": 0,
        "polishing": False,
        "return_info": True,
    }

    monkeypatch.setattr(
        torch_osqp_module,
        "_termination_status",
        lambda *args, **kwargs: ("solved", None, None),
    )
    monkeypatch.setattr(
        torch_osqp_module,
        "_adaptive_rho_update",
        lambda *args, **kwargs: pytest.fail(
            "termination must precede KKT-error rho adaptation"
        ),
    )
    _, info = solve_torch_osqp_direct(P, q, A, l, u, settings)
    assert info["status"] == "solved"
    assert info["admm_iterations"] == 1
    assert info["rho_updates"] == 0


def test_time_rho_mode_resolves_automatic_interval_from_mocked_solve_clock(
    monkeypatch,
):
    P = torch.ones((1, 1), dtype=torch.float64)
    q = torch.tensor([-1.0], dtype=torch.float64)
    A = torch.ones((1, 1), dtype=torch.float64)
    l, u = torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "rho_update_safeguard": False,
        "adaptive_rho": 2,
        "adaptive_rho_fraction": 0.4,
        "rho_update_interval": 0,
        "check_termination": 0,
        "max_iter": 15,
        "time_limit": 100.0,
        "scaling": 0,
        "polishing": False,
        "warm_start": False,
        "return_info": True,
    }
    clock_index = 0
    update_calls = []
    schedule_observations = []

    def clock():
        nonlocal clock_index
        value = 0.0 if clock_index == 0 else 1.0 if clock_index == 1 else 2.0
        clock_index += 1
        return value

    monkeypatch.setattr(torch_osqp_module, "_clock", clock)
    original_schedule = torch_osqp_module._time_rho_interval

    def capture_schedule(**kwargs):
        result = original_schedule(**kwargs)
        schedule_observations.append((kwargs, result))
        return result

    monkeypatch.setattr(
        torch_osqp_module, "_time_rho_interval", capture_schedule
    )

    def no_change(rho, equality, *args, loose=None, **kwargs):
        update_calls.append(float(rho.item()))
        return False, rho, _rho_vector(rho, equality, loose)

    monkeypatch.setattr(torch_osqp_module, "_adaptive_rho_update", no_change)
    workspace = TorchOSQPWorkspace()
    _, info = solve_torch_osqp_direct(P, q, A, l, u, settings, workspace)

    assert workspace.adaptive_rho_setup_time == 1.0
    assert schedule_observations
    assert schedule_observations[0][1] == 0
    assert schedule_observations[-1][1] == 1
    assert workspace.adaptive_rho_interval == 1
    assert update_calls == [0.1, 0.1, 0.1]
    assert info["admm_iterations"] == 15
    assert info["rho_updates"] == 0


@pytest.mark.parametrize("mode", [2, 3])
@pytest.mark.parametrize("scaled_termination", [False, True])
def test_extended_rho_modes_execute_on_supported_device_and_dtype(
    mode, scaled_termination, floating_dtype, available_device,
):
    P = torch.ones((1, 1), dtype=floating_dtype, device=available_device)
    q = torch.tensor([-1.0], dtype=floating_dtype, device=available_device)
    A = torch.ones((1, 1), dtype=floating_dtype, device=available_device)
    l, u = P.new_zeros(1), P.new_ones(1)
    settings = {
        **DEFAULT_OSQP_SETTINGS,
        "rho_update_safeguard": False,
        "adaptive_rho": mode,
        "adaptive_rho_fraction": 1.0,
        "rho_update_interval": 1,
        "rho_update_tolerance": 5.0,
        "check_termination": 0,
        "max_iter": 3,
        "scaled_termination": scaled_termination,
        "scaling": 1,
        "polishing": False,
        "warm_start": False,
        "return_info": True,
    }
    solution, info = solve_torch_osqp_direct(P, q, A, l, u, settings)
    assert solution.device == P.device and solution.dtype == P.dtype
    assert info["device"] == str(P.device)
    assert info["dtype"] == str(P.dtype)
    assert info["admm_iterations"] == 3
    assert "rel_kkt_error" in info


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_duality_gap_round_to_zero_and_strict_deadzone(device, dtype):
    """Port Vector: Round-to-zero, then exercise both exact cutoff signs."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    deadzone = torch.tensor(1e-10 if dtype == torch.float32 else 1e-15,
                           device=device, dtype=dtype)
    # Pinned six-entry fixture has four sub-deadzone entries and two retained.
    y = deadzone * torch.tensor([2, .1, -.1, -2, .01, -.01], device=device, dtype=dtype)
    y = torch.cat((y, torch.stack((
        torch.nextafter(deadzone, torch.zeros_like(deadzone)), deadzone,
        -deadzone, torch.nextafter(-deadzone, torch.zeros_like(deadzone)),
    ))))
    before = y.clone()
    expected_y = deadzone * torch.tensor([2, 0, 0, -2, 0, 0, 0, 1, -1, 0],
                                         device=device, dtype=dtype)
    # Each row is isolated so cancelling small terms cannot hide a failure.
    for index in range(y.numel()):
        zero = torch.zeros(1, device=device, dtype=dtype)
        values = _compute_obj_val_dual_gap(
            zero.reshape(1, 1), zero, zero, torch.ones_like(zero),
            zero, y[index:index + 1], torch.ones((), device=device, dtype=dtype), 0.1, 0.0,
        )
        assert values["duality_gap"].item() == expected_y[index].clamp_min(0).item()
        # Lower-only contribution validates the negative cutoff too.
        values = _compute_obj_val_dual_gap(
            zero.reshape(1, 1), zero, -torch.ones_like(zero), zero,
            zero, y[index:index + 1], torch.ones((), device=device, dtype=dtype), 0.1, 0.0,
        )
        assert values["duality_gap"].item() == -expected_y[index].clamp_max(0).item()
        assert all(value.device.type == device and value.dtype == dtype
                   for value in values.values())
    assert torch.equal(y, before)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cost", [1.0, 0.125, 8.0])
def test_duality_gap_support_projection_and_cost_unscaling(device, dtype, cost):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    def tensor(values):
        return torch.tensor(values, device=device, dtype=dtype)

    # Two-sided, lower-only, upper-only, free, and wrong one-sided dual signs.
    l = tensor([-2, 3, -torch.inf, -torch.inf, 3, -torch.inf, 2])
    u = tensor([4, torch.inf, 5, torch.inf, torch.inf, 5, 2])
    y = tensor([2, -2, 3, 100, 7, -9, -3])
    values = _compute_obj_val_dual_gap(
        tensor([[3.0]]), tensor([-4.0]), l, u, tensor([2.0]), y, tensor(cost), .2, .1,
    )
    # xPx=12; qx=-8; support=8-6+15+0+0+0-6=11.
    expected = {"objective": -2 / cost, "dual_objective": -17 / cost,
                "duality_gap": 15 / cost, "eps_duality_gap": .2 + .1 * 12 / cost,
                "scaled_duality_gap": 15, "eps_scaled_duality_gap": 1.4}
    for key, value in expected.items():
        assert values[key].item() == pytest.approx(value, rel=4 * torch.finfo(dtype).eps)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_duality_gap_projection_preserves_exact_infinity_cutoff(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    zero = torch.zeros(1, device=device, dtype=dtype)
    cutoff = torch.full_like(zero, torch_osqp_module._osqp_infinity(zero) * 1e-4)
    for sign in (-1, 1):
        for bound, retained in ((cutoff, True), (torch.nextafter(cutoff, cutoff * 2), False)):
            l, u = (zero, bound) if sign > 0 else (-bound, zero)
            values = _compute_obj_val_dual_gap(
                zero.reshape(1, 1), zero, l, u, zero, torch.full_like(zero, sign),
                torch.ones((), device=device, dtype=dtype), 1.0, 0.0,
            )
            assert values["duality_gap"].item() == (cutoff.item() if retained else 0.0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ruiz_scaling_round_trips_primal_and_dual_state(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    P = torch.diag(torch.tensor([1e-4, 1e4], device=device, dtype=dtype))
    q = torch.tensor([2.0, -3.0], device=device, dtype=dtype)
    A = torch.tensor([[1e3, 0.0], [0.0, 1e-3]], device=device, dtype=dtype)
    l = torch.tensor([-1.0, -2.0], device=device, dtype=dtype)
    u = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
    workspace = TorchOSQPWorkspace()
    scaling = _scaling_for_problem(workspace, P, q, A, {"scaling": 10})
    P_s, q_s, A_s, l_s, u_s = _scale_problem(P, q, A, l, u, scaling)
    assert all(torch.all(torch.isfinite(value)) for value in (P_s, q_s, A_s, l_s, u_s))

    x = torch.tensor([0.25, -0.5], device=device, dtype=dtype)
    z = A @ x
    y = torch.tensor([0.3, -0.2], device=device, dtype=dtype)
    D, E, cost = scaling["D"], scaling["E"], scaling["cost"]
    x_o, z_o, y_o = _unscale_state(x / D, z * E, y * cost / E, scaling)
    assert torch.allclose(x_o, x)
    assert torch.allclose(z_o, z)
    assert torch.allclose(y_o, y)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ruiz_scaling_matches_pinned_per_pass_factors_and_data(device, dtype):
    """Translate scaling.c's two-pass D/E/c and data transformations."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")

    P = torch.tensor([[4.0, 2.0], [2.0, 9.0]], device=device, dtype=dtype)
    q = torch.tensor([8.0, -3.0], device=device, dtype=dtype)
    A = torch.tensor([[1.0, 0.0], [0.0, 16.0]], device=device, dtype=dtype)
    l = torch.tensor([-torch.inf, -2.0], device=device, dtype=dtype)
    u = torch.tensor([torch.inf, -2.0], device=device, dtype=dtype)
    originals = tuple(value.clone() for value in (P, q, A, l, u))

    P_work, q_work, A_work = P.clone(), q.clone(), A.clone()
    D = torch.ones(2, device=device, dtype=dtype)
    E = torch.ones(2, device=device, dtype=dtype)
    cost = torch.ones((), device=device, dtype=dtype)
    expected = []
    for _ in range(2):
        p_norm = torch.amax(torch.abs(torch.triu(P_work)), dim=0)
        a_col = torch.amax(torch.abs(A_work), dim=0)
        a_row = torch.amax(torch.abs(A_work), dim=1)

        # Literal limit_scaling_vector behavior from pinned src/scaling.c.
        one_col = torch.ones_like(p_norm)
        one_row = torch.ones_like(a_row)
        d_norm = torch.maximum(p_norm, a_col)
        d_norm = torch.where(d_norm < 1e-4, one_col, d_norm)
        d_norm = torch.minimum(d_norm, torch.full_like(d_norm, 1e4))
        e_norm = torch.where(a_row < 1e-4, one_row, a_row)
        e_norm = torch.minimum(e_norm, torch.full_like(e_norm, 1e4))
        d_step, e_step = torch.rsqrt(d_norm), torch.rsqrt(e_norm)

        P_work = d_step[:, None] * P_work * d_step[None, :]
        A_work = e_step[:, None] * A_work * d_step[None, :]
        q_work = d_step * q_work
        D, E = D * d_step, E * e_step

        average_p_norm = torch.mean(
            torch.amax(torch.abs(torch.triu(P_work)), dim=0)
        )
        q_norm = torch.linalg.vector_norm(q_work, ord=float("inf"))
        q_norm = torch.where(q_norm < 1e-4, torch.ones_like(q_norm), q_norm)
        q_norm = torch.minimum(q_norm, torch.full_like(q_norm, 1e4))
        cost_measure = torch.maximum(average_p_norm, q_norm)
        cost_measure = torch.where(
            cost_measure < 1e-4, torch.ones_like(cost_measure), cost_measure
        )
        cost_measure = torch.minimum(
            cost_measure, torch.full_like(cost_measure, 1e4)
        )
        cost_step = cost_measure.reciprocal()
        P_work, q_work = cost_step * P_work, cost_step * q_work
        cost = cost * cost_step
        expected.append(
            tuple(value.clone() for value in (D, E, cost, P_work, q_work, A_work))
        )

    sqrt_two = torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    literal_factors = (
        (
            torch.tensor([0.5, 0.25], device=device, dtype=dtype),
            torch.tensor([1.0, 0.25], device=device, dtype=dtype),
            torch.tensor(0.25, device=device, dtype=dtype),
        ),
        (
            torch.stack((sqrt_two / 2, torch.as_tensor(0.25, device=device, dtype=dtype))),
            torch.stack((sqrt_two, torch.as_tensor(0.25, device=device, dtype=dtype))),
            1 / (4 * sqrt_two),
        ),
    )
    tolerance = 20 * torch.finfo(dtype).eps

    for passes, (reference, literal) in enumerate(
        zip(expected, literal_factors), start=1
    ):
        expected_D, expected_E, expected_cost, expected_P, expected_q, expected_A = reference
        literal_D, literal_E, literal_cost = literal
        torch.testing.assert_close(expected_D, literal_D, rtol=tolerance, atol=0)
        torch.testing.assert_close(expected_E, literal_E, rtol=tolerance, atol=0)
        torch.testing.assert_close(expected_cost, literal_cost, rtol=tolerance, atol=0)

        scaling = _scaling_for_problem(
            TorchOSQPWorkspace(), P, q, A, {"scaling": passes}
        )
        for actual, wanted in (
            (scaling["D"], expected_D),
            (scaling["E"], expected_E),
            (scaling["cost"], expected_cost),
        ):
            torch.testing.assert_close(actual, wanted, rtol=tolerance, atol=0)

        scaled = _scale_problem(P, q, A, l, u, scaling)
        for actual, wanted in zip(
            scaled,
            (expected_P, expected_q, expected_A, expected_E * l, expected_E * u),
        ):
            torch.testing.assert_close(actual, wanted, rtol=tolerance, atol=0)

        P_scaled, q_scaled, A_scaled, l_scaled, u_scaled = scaled
        D_actual, E_actual, cost_actual = (
            scaling["D"], scaling["E"], scaling["cost"]
        )
        restored = (
            P_scaled / cost_actual / D_actual[:, None] / D_actual[None, :],
            q_scaled / cost_actual / D_actual,
            A_scaled / E_actual[:, None] / D_actual[None, :],
            l_scaled / E_actual,
            u_scaled / E_actual,
        )
        for actual, wanted in zip(restored, originals):
            torch.testing.assert_close(actual, wanted, rtol=tolerance, atol=0)
        assert torch.isneginf(l_scaled[0]) and torch.isposinf(u_scaled[0])
        assert l_scaled[1] == u_scaled[1]

    for actual, wanted in zip((P, q, A, l, u), originals):
        assert torch.equal(actual, wanted)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_limit_scaling_matches_pinned_strict_thresholds(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    lower = torch.tensor(1e-4, device=device, dtype=dtype)
    upper = torch.tensor(1e4, device=device, dtype=dtype)
    positive_infinity = torch.tensor(torch.inf, device=device, dtype=dtype)
    values = torch.stack(
        (
            zero,
            torch.nextafter(lower, zero),
            lower,
            torch.nextafter(lower, positive_infinity),
            upper,
            torch.nextafter(upper, positive_infinity),
            positive_infinity,
        )
    )
    expected = torch.stack(
        (
            torch.ones_like(zero),
            torch.ones_like(zero),
            lower,
            values[3],
            upper,
            upper,
            upper,
        )
    )
    assert torch.equal(torch_osqp_module._limit_scaling(values), expected)
    nan = torch.tensor(torch.nan, device=device, dtype=dtype)
    assert torch.isnan(torch_osqp_module._limit_scaling(nan))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ruiz_zero_tiny_and_extreme_norms_use_pinned_steps(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    P = torch.zeros((3, 3), device=device, dtype=dtype)
    q = torch.zeros(3, device=device, dtype=dtype)
    A = torch.tensor(
        [[0.0, 1e-8, 1e8], [0.0, 0.0, 0.0]], device=device, dtype=dtype
    )
    scaling = _scaling_for_problem(
        TorchOSQPWorkspace(), P, q, A, {"scaling": 1}
    )
    torch.testing.assert_close(
        scaling["D"],
        torch.tensor([1.0, 1.0, 0.01], device=device, dtype=dtype),
        rtol=10 * torch.finfo(dtype).eps,
        atol=0,
    )
    torch.testing.assert_close(
        scaling["E"],
        torch.tensor([0.01, 1.0], device=device, dtype=dtype),
        rtol=10 * torch.finfo(dtype).eps,
        atol=0,
    )
    assert scaling["cost"] == torch.ones((), device=device, dtype=dtype)


def test_adaptive_rho_update_is_deterministic_and_keeps_equality_boost():
    dtype = torch.float64
    rho = torch.tensor(0.1, dtype=dtype)
    equality = torch.tensor([False, True])
    updated, rho_next, rho_vector = _adaptive_rho_update(
        rho,
        equality,
        torch.tensor(10.0, dtype=dtype),
        torch.tensor(0.1, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
        5.0,
    )
    assert updated
    assert rho_next > rho
    assert torch.allclose(rho_vector, _rho_vector(rho_next, equality))
    assert rho_vector[1] == 1000.0 * rho_vector[0]


def test_rho_loose_rows_and_clamped_candidates_do_not_refactorize():
    dtype = torch.float64
    rho = torch.tensor(1e6, dtype=dtype)
    equality = torch.tensor([False, True, False])
    loose = torch.tensor([False, False, True])
    updated, rho_next, rho_vector = _adaptive_rho_update(
        rho,
        equality,
        torch.tensor(1e12, dtype=dtype),
        torch.tensor(1e-12, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
        5.0,
        loose=loose,
    )
    assert not updated
    assert rho_next == rho
    assert rho_vector[0] == 1e6
    assert rho_vector[1] == 1e9
    assert rho_vector[2] == 1e-6


def test_adaptive_rho_both_zero_uses_pinned_nan_clamp():
    """Pinned auxil.c maps the 0/0 estimate to OSQP_RHO_MIN."""
    rho = torch.tensor(0.1, dtype=torch.float64)
    zero = torch.zeros((), dtype=rho.dtype)
    updated, rho_next, rho_vector = _adaptive_rho_update(
        rho,
        torch.tensor([False]),
        zero,
        zero,
        torch.ones_like(zero),
        torch.ones_like(zero),
        5.0,
    )
    assert updated
    assert rho_next == torch.tensor(1e-6, dtype=rho.dtype)
    assert rho_vector[0] == rho_next


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "primal_multiplier,primal_normalizer_multiplier,dual_multiplier,"
    "dual_normalizer_multiplier,expected_ratio,expected_update",
    [
        (1.0, 0.0, 2.0, 0.0, 0.5, True),
        (4.0, 1.0, 1.0, 0.0, 2.0, True),
        (1.0, 1.0, 1.0, 1.0, 1.0, False),
    ],
)
def test_adaptive_rho_tiny_nonzero_residuals_use_source_division_tolerance(
    device,
    dtype,
    primal_multiplier,
    primal_normalizer_multiplier,
    dual_multiplier,
    dual_normalizer_multiplier,
    expected_ratio,
    expected_update,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    division_tolerance = torch.as_tensor(
        1e-17 if dtype == torch.float32 and device == "cuda" else 1e-30,
        device=device,
        dtype=dtype,
    )
    rho = torch.as_tensor(0.1, device=device, dtype=dtype)
    primal = primal_multiplier * division_tolerance
    primal_normalizer = primal_normalizer_multiplier * division_tolerance
    dual = dual_multiplier * division_tolerance
    dual_normalizer = dual_normalizer_multiplier * division_tolerance
    expected = rho * torch.sqrt(
        torch.as_tensor(expected_ratio, device=device, dtype=dtype)
    )

    updated, rho_next, rho_vector = _adaptive_rho_update(
        rho,
        torch.tensor([False], device=device),
        primal,
        dual,
        torch.ones((), device=device, dtype=dtype),
        torch.ones((), device=device, dtype=dtype),
        1.0,
        primal_normalizer=primal_normalizer,
        dual_normalizer=dual_normalizer,
    )

    assert updated is expected_update
    torch.testing.assert_close(
        rho_next,
        expected if expected_update else rho,
        rtol=10 * torch.finfo(dtype).eps,
        atol=0,
    )
    torch.testing.assert_close(rho_vector[0], rho_next, rtol=0, atol=0)


def test_adaptive_rho_uses_strict_threshold_and_clamp_boundaries():
    dtype = torch.float64
    equality = torch.tensor([False])
    one = torch.ones((), dtype=dtype)
    zero = torch.zeros((), dtype=dtype)
    positive_infinity = torch.full((), torch.inf, dtype=dtype)

    rho = torch.tensor(0.125, dtype=dtype)
    upper_cases = (
        (torch.tensor(16.0, dtype=dtype), False),
        (
            torch.nextafter(
                torch.nextafter(torch.tensor(16.0, dtype=dtype), positive_infinity), positive_infinity
            ),
            True,
        ),
    )
    lower_cases = (
        (torch.tensor(0.0625, dtype=dtype), False),
        (
            torch.nextafter(torch.nextafter(torch.tensor(0.0625, dtype=dtype), zero), zero),
            True,
        ),
    )
    for primal, expected_update in upper_cases + lower_cases:
        updated, _, _ = _adaptive_rho_update(
            rho, equality, primal, one, one, one, 4.0
        )
        assert updated is expected_update

    clamp_cases = (
        (
            torch.tensor(125000.0, dtype=dtype),
            positive_infinity,
            False,
            125000.0,
        ),
        (
            torch.nextafter(torch.tensor(125000.0, dtype=dtype), zero),
            positive_infinity,
            True,
            1e6,
        ),
        (torch.tensor(8e-6, dtype=dtype), zero, False, 8e-6),
        (
            torch.nextafter(
                torch.tensor(8e-6, dtype=dtype), positive_infinity
            ),
            zero,
            True,
            1e-6,
        ),
    )
    for rho_value, primal, expected_update, expected_rho in clamp_cases:
        updated, rho_next, _ = _adaptive_rho_update(
            rho_value, equality, primal, one, one, one, 8.0
        )
        assert updated is expected_update
        assert rho_next == torch.tensor(expected_rho, dtype=dtype)


def test_ruiz_scaling_supports_empty_constraint_matrix():
    dtype = torch.float64
    P = torch.diag(torch.tensor([1e-12, 1e12], dtype=dtype))
    q = torch.tensor([1.0, -1.0], dtype=dtype)
    A = torch.empty((0, 2), dtype=dtype)
    scaling = _scaling_for_problem(TorchOSQPWorkspace(), P, q, A, {"scaling": 10})
    assert scaling["E"].numel() == 0
    assert torch.all(torch.isfinite(scaling["D"]))
    assert torch.isfinite(scaling["cost"])


def test_ruiz_cost_scaling_treats_zero_q_norm_as_one_like_pinned_source():
    dtype = torch.float64
    P = 0.25 * torch.eye(2, dtype=dtype)
    q = torch.zeros(2, dtype=dtype)
    A = torch.eye(2, dtype=dtype)
    scaling = _scaling_for_problem(
        TorchOSQPWorkspace(), P, q, A, {"scaling": 1}
    )
    assert scaling["cost"] == torch.tensor(1.0, dtype=dtype)


def test_ruiz_p_norms_match_pinned_upper_triangle_storage():
    dtype = torch.float64
    P = torch.tensor([[1.0, 10.0], [10.0, 100.0]], dtype=dtype)
    scaling = _scaling_for_problem(
        TorchOSQPWorkspace(),
        P,
        torch.zeros(2, dtype=dtype),
        torch.empty((0, 2), dtype=dtype),
        {"scaling": 1},
    )
    assert torch.allclose(scaling["D"], torch.tensor([1.0, 0.1], dtype=dtype))
    assert scaling["cost"] == torch.tensor(1.0, dtype=dtype)


def test_polishing_handles_duplicate_equality_and_bound_rows():
    dtype = torch.float64
    H = torch.ones((1, 1), dtype=dtype)
    f = torch.tensor([[-2.0]], dtype=dtype)
    A = torch.ones((1, 1), dtype=dtype)
    b = torch.tensor(1.0, dtype=dtype)
    lower = torch.zeros((1, 1), dtype=dtype)
    upper = torch.ones((1, 1), dtype=dtype)
    solution, info = solve_osqp_torch_qp(
        H,
        f,
        A,
        b,
        lower,
        upper,
        "cpu",
        True,
        {
            "algebra": "torch",
            "settings": {"return_info": True, "polishing": True},
        },
        TorchOSQPWorkspace(),
    )
    assert torch.allclose(solution, torch.ones_like(solution), atol=1e-8, rtol=1e-8)
    assert info["polishing_status"] == "accepted"
    assert info["polishing_active_constraints"] == 2


def test_pinned_no_active_set_fixture_polishing_behavior():
    dtype = torch.float64
    P = torch.diag(torch.tensor([11.0, 5.0], dtype=dtype))
    q = torch.zeros(2, dtype=dtype)
    A = torch.tensor(
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=dtype
    )
    l = torch.tensor([-1.0, -0.5, -0.25], dtype=dtype)
    u = torch.tensor([1.0, 0.7, 0.7], dtype=dtype)

    for polishing, expected_status in (
        (False, "disabled"),
        (True, "skipped_no_active_constraints"),
    ):
        settings = {
            **DEFAULT_OSQP_SETTINGS,
            "rho_update_safeguard": False,
            "eps_abs": 1e-3,
            "eps_rel": 1e-3,
            "polishing": polishing,
            "polish_refine_iter": 4,
            "return_info": True,
            "return_state": True,
        }
        solution, info = solve_torch_osqp_direct(
            P, q, A, l, u, settings, TorchOSQPWorkspace()
        )

        assert info["status"] == "solved"
        assert info["status_val"] == 1
        assert torch.equal(solution, torch.zeros((2, 1), dtype=dtype))
        assert torch.equal(info["state"]["y"], torch.zeros(3, dtype=dtype))
        assert info["objective"] == 0.0
        assert info["polishing_status"] == expected_status
        assert info["polishing_active_constraints"] == 0


@pytest.mark.parametrize("refine_iter", [0, 1, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_polishing_honors_zero_one_and_three_refinement_passes(
    refine_iter, device, dtype, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    instances = []

    class RecordingQDLDLSolver(QDLDLSolver):
        def __init__(self):
            super().__init__()
            self.solutions = []
            instances.append(self)

        def solve(self, rhs, *, calculate_residual=False):
            solution, diagnostics = super().solve(
                rhs, calculate_residual=calculate_residual
            )
            self.solutions.append(solution.detach().clone())
            return solution, diagnostics

    monkeypatch.setattr(
        torch_osqp_module,
        "QDLDLSolver",
        RecordingQDLDLSolver,
    )
    x_result, z_result, y_result, info = _polish_solution(
        torch.ones((1, 1), device=device, dtype=dtype),
        torch.tensor([-2.0], device=device, dtype=dtype),
        torch.ones((1, 1), device=device, dtype=dtype),
        torch.zeros(1, device=device, dtype=dtype),
        torch.ones(1, device=device, dtype=dtype),
        torch.tensor([0.99], device=device, dtype=dtype),
        torch.tensor([0.99], device=device, dtype=dtype),
        torch.tensor([0.9], device=device, dtype=dtype),
        {
            "eps_abs": 1e-6,
            "eps_rel": 1e-6,
            "polishing": True,
            "polish_delta": 1e-6,
            "polish_refine_iter": refine_iter,
        },
    )
    assert info["polishing_status"] == "accepted"
    assert info["polishing_refine_iter"] == refine_iter
    tolerance = max(2e-6, 100 * torch.finfo(dtype).eps)
    assert torch.allclose(
        x_result, torch.ones_like(x_result), atol=tolerance, rtol=tolerance
    )
    assert torch.allclose(
        z_result, torch.ones_like(z_result), atol=tolerance, rtol=tolerance
    )
    assert torch.allclose(
        y_result, torch.ones_like(y_result), atol=tolerance, rtol=tolerance
    )
    assert len(instances) == 1
    assert instances[0].factorization_count == 1
    assert instances[0].solve_count == 1 + refine_iter

    exact_kkt = torch.tensor(
        [[1.0, 1.0], [1.0, 0.0]], device=device, dtype=dtype
    )
    rhs = torch.tensor([2.0, 1.0], device=device, dtype=dtype)
    initial = instances[0].solutions[0]
    refined = initial + sum(
        instances[0].solutions[1:],
        start=torch.zeros_like(initial),
    )
    initial_residual = torch.linalg.vector_norm(rhs - exact_kkt @ initial)
    refined_residual = torch.linalg.vector_norm(rhs - exact_kkt @ refined)
    if refine_iter == 0:
        assert torch.equal(refined, initial)
    else:
        assert refined_residual < initial_residual


@pytest.mark.parametrize("reduced_dual", [1.0, 3.0])
def test_dense_polishing_refinement_subtracts_blocks_in_source_order(
    reduced_dual, monkeypatch
):
    dtype = torch.float64
    x = torch.tensor([2.0**53], dtype=dtype)
    A = P = torch.ones((1, 1), dtype=dtype)
    initial_candidate = torch.cat((x, torch.tensor([reduced_dual], dtype=dtype)))
    seen_rhs = []

    class FixedCandidateQDLDLSolver(QDLDLSolver):
        def solve(self, rhs, *, calculate_residual=False):
            seen_rhs.append(rhs.clone())
            result, diagnostics = super().solve(
                rhs, calculate_residual=calculate_residual
            )
            if len(seen_rhs) == 1:
                return initial_candidate.clone(), diagnostics
            return result, diagnostics

    monkeypatch.setattr(torch_osqp_module, "QDLDLSolver", FixedCandidateQDLDLSolver)
    _polish_solution(
        P, -x, A, x, x, x, x, torch.zeros_like(x),
        {
            "eps_abs": 1e-8,
            "eps_rel": 1e-8,
            "polishing": True,
            "sigma": 0.125,
            "polish_refine_iter": 1,
        },
    )

    assert len(seen_rhs) == 2
    assert torch.equal(seen_rhs[0], torch.cat((x, x)))
    # In pinned C, the separate subtraction of P @ x cancels the top RHS
    # exactly before the contribution from A.T @ y is applied.
    expected = torch.tensor([-reduced_dual, 0.0], dtype=dtype)
    assert torch.equal(seen_rhs[1].view(torch.int64), expected.view(torch.int64))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_polishing_active_set_rhs_and_normal_cone_match_pinned_source(
    device, dtype, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")

    P = torch.eye(2, device=device, dtype=dtype)
    q = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
    A = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
        device=device,
        dtype=dtype,
    )
    l = torch.tensor([1.0, 0.0, -torch.inf, -torch.inf], device=device, dtype=dtype)
    u = torch.tensor([1.0, torch.inf, 2.0, torch.inf], device=device, dtype=dtype)
    x = torch.tensor([9.0, 9.0], device=device, dtype=dtype)
    z = torch.tensor([1.0, 0.0, 2.0, 0.0], device=device, dtype=dtype)
    y = torch.tensor([0.0, -1.0, 1.0, 100.0], device=device, dtype=dtype)
    polished = torch.tensor(
        [1.0, 0.0, -0.5, -1.0, 2.0], device=device, dtype=dtype
    )

    class RecordingSolver:
        matrix = None
        rhs = None

        def factorize(self, matrix, **kwargs):
            self.positive_d_count = kwargs["P"].shape[0]
            type(self).matrix = matrix.detach().clone()

        def solve(self, rhs, *, calculate_residual=False):
            type(self).rhs = rhs.detach().clone()
            return polished.detach().clone(), {}

    residual_values = iter(
        (
            tuple(torch.tensor(value, device=device, dtype=dtype) for value in (10, 10, 1, 1)),
            tuple(torch.tensor(value, device=device, dtype=dtype) for value in (1, 1, 1, 1)),
        )
    )
    monkeypatch.setattr(torch_osqp_module, "QDLDLSolver", RecordingSolver)
    monkeypatch.setattr(
        torch_osqp_module,
        "_residuals",
        lambda *args, **kwargs: next(residual_values),
    )

    x_result, z_result, y_result, info = _polish_solution(
        P,
        q,
        A,
        l,
        u,
        x,
        z,
        y,
        {
            "eps_abs": 1e-6,
            "eps_rel": 1e-6,
            "polishing": True,
            "sigma": 0.25,
            "polish_refine_iter": 0,
        },
    )

    active_a = A[:3]
    exact_kkt = torch.cat(
        (
            torch.cat((P, active_a.T), dim=1),
            torch.cat((active_a, torch.zeros((3, 3), device=device, dtype=dtype)), dim=1),
        ),
        dim=0,
    )
    regularizer = torch.diag(
        torch.tensor([0.25, 0.25, -0.25, -0.25, -0.25], device=device, dtype=dtype)
    )
    assert torch.equal(RecordingSolver.matrix, exact_kkt + regularizer)
    assert torch.equal(
        RecordingSolver.rhs,
        torch.tensor([-1.0, -2.0, 1.0, 0.0, 2.0], device=device, dtype=dtype),
    )
    assert info["polishing_active_constraints"] == 3
    assert info["polishing_status"] == "accepted"
    assert torch.equal(x_result, torch.tensor([1.0, 0.0], device=device, dtype=dtype))
    assert torch.equal(z_result, torch.tensor([1.0, 0.0, 2.0, 1.0], device=device, dtype=dtype))
    assert torch.equal(y_result, torch.tensor([-0.5, -1.0, 1.0, 0.0], device=device, dtype=dtype))


@pytest.mark.parametrize(
    "old_primal,old_dual,new_primal,new_dual,accepted",
    [
        (2.0, 2.0, 1.0, 1.0, True),
        (2.0, 5e-11, 1.0, 3.0, True),
        (5e-11, 2.0, 3.0, 1.0, True),
        (2.0, 2.0, 2.0, 1.0, False),
        (2.0, 1e-10, 1.0, 2.0, False),
        (1e-10, 2.0, 2.0, 1.0, False),
    ],
    ids=[
        "both-improve",
        "primal-improves-old-dual-tiny",
        "dual-improves-old-primal-tiny",
        "strict-equal-primal-rejects",
        "strict-dual-tiny-cutoff-rejects",
        "strict-primal-tiny-cutoff-rejects",
    ],
)
def test_polishing_uses_pinned_three_branch_strict_acceptance(
    old_primal, old_dual, new_primal, new_dual, accepted, monkeypatch
):
    dtype = torch.float64
    original = (
        torch.tensor([0.5], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
        torch.tensor([0.0], dtype=dtype),
    )
    candidate = torch.tensor([1.0, 1.0], dtype=dtype)

    class CandidateSolver:
        def factorize(self, matrix, **kwargs):
            self.positive_d_count = kwargs["P"].shape[0]
            return None

        def solve(self, rhs, *, calculate_residual=False):
            return candidate.clone(), {}

    residual_values = iter(
        (
            tuple(torch.tensor(value, dtype=dtype) for value in (old_primal, old_dual, 1, 1)),
            tuple(torch.tensor(value, dtype=dtype) for value in (new_primal, new_dual, 1, 1)),
        )
    )
    monkeypatch.setattr(torch_osqp_module, "QDLDLSolver", CandidateSolver)
    monkeypatch.setattr(
        torch_osqp_module,
        "_residuals",
        lambda *args, **kwargs: next(residual_values),
    )

    result = _polish_solution(
        torch.ones((1, 1), dtype=dtype),
        torch.tensor([-2.0], dtype=dtype),
        torch.ones((1, 1), dtype=dtype),
        torch.ones(1, dtype=dtype),
        torch.ones(1, dtype=dtype),
        *original,
        {
            "eps_abs": 1e-6,
            "eps_rel": 1e-6,
            "polishing": True,
            "sigma": 1e-6,
            "polish_refine_iter": 0,
        },
    )

    x_result, z_result, y_result, info = result
    assert info["polishing_success"] is accepted
    assert info["polishing_status"] == (
        "accepted" if accepted else "rejected_no_improvement"
    )
    if accepted:
        assert torch.equal(x_result, torch.ones(1, dtype=dtype))
        assert torch.equal(z_result, torch.ones(1, dtype=dtype))
        assert torch.equal(y_result, torch.ones(1, dtype=dtype))
    else:
        assert torch.equal(x_result, original[0])
        assert torch.equal(z_result, original[1])
        assert torch.equal(y_result, original[2])


@pytest.mark.parametrize("fail_on_solve", [1, 2], ids=["initial", "refinement"])
def test_polishing_solve_failure_preserves_admm_state(
    fail_on_solve, monkeypatch
):
    dtype = torch.float64
    x = torch.tensor([0.99], dtype=dtype)
    z = torch.tensor([0.99], dtype=dtype)
    y = torch.tensor([0.9], dtype=dtype)

    class FailingSolve(QDLDLSolver):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def solve(self, rhs, *, calculate_residual=False):
            self.calls += 1
            if self.calls == fail_on_solve:
                raise TorchLinearSolveError("synthetic polish solve failure")
            return super().solve(rhs, calculate_residual=calculate_residual)

    monkeypatch.setattr(torch_osqp_module, "QDLDLSolver", FailingSolve)
    before = (x.clone(), z.clone(), y.clone())
    with pytest.raises(TorchLinearSolveError, match="synthetic polish"):
        _polish_solution(
            torch.ones((1, 1), dtype=dtype),
            torch.tensor([-2.0], dtype=dtype),
            torch.ones((1, 1), dtype=dtype),
            torch.zeros(1, dtype=dtype),
            torch.ones(1, dtype=dtype),
            x,
            z,
            y,
            {
                "eps_abs": 1e-6,
                "eps_rel": 1e-6,
                "polishing": True,
                "sigma": 1e-6,
                "polish_refine_iter": 1,
            },
        )
    for actual, saved in zip((x, z, y), before):
        assert torch.equal(actual, saved)


def test_polishing_linear_failure_preserves_admm_state(monkeypatch):
    dtype = torch.float64
    P = torch.ones((1, 1), dtype=dtype)
    q = torch.tensor([-1.0], dtype=dtype)
    A = torch.ones((1, 1), dtype=dtype)
    l = torch.zeros(1, dtype=dtype)
    u = torch.ones(1, dtype=dtype)
    x = torch.ones(1, dtype=dtype)
    z = torch.ones(1, dtype=dtype)
    y = torch.ones(1, dtype=dtype)

    def fail_factorization(self, matrix, **kwargs):
        raise TorchLinearSolveError("synthetic polish factorization failure")

    monkeypatch.setattr(
        torch_osqp_module.QDLDLSolver,
        "factorize",
        fail_factorization,
    )
    before = (x.clone(), z.clone(), y.clone())
    with pytest.raises(TorchLinearSolveError, match="synthetic polish"):
        _polish_solution(
            P,
            q,
            A,
            l,
            u,
            x,
            z,
            y,
            {
                "eps_abs": 1e-6,
                "eps_rel": 1e-6,
                "polishing": True,
                "polish_delta": 1e-6,
                "polish_refine_iter": 3,
            },
        )
    for actual, saved in zip((x, z, y), before):
        assert torch.equal(actual, saved)


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, MemoryError],
    ids=["runtime", "allocation"],
)
def test_polishing_runtime_failure_preserves_admm_state(monkeypatch, error_type):
    dtype = torch.float64
    P = torch.ones((1, 1), dtype=dtype)
    q = -torch.ones(1, dtype=dtype)
    A = torch.ones((1, 1), dtype=dtype)
    l = torch.ones(1, dtype=dtype)
    u = torch.ones(1, dtype=dtype)
    x = torch.ones(1, dtype=dtype)
    z = torch.ones(1, dtype=dtype)
    y = torch.ones(1, dtype=dtype)

    def fail_active_rows(*args, **kwargs):
        raise error_type("synthetic polish construction failure")

    monkeypatch.setattr(torch, "nonzero", fail_active_rows)
    before = (x.clone(), z.clone(), y.clone())
    with pytest.raises(error_type, match="synthetic polish"):
        _polish_solution(
            P,
            q,
            A,
            l,
            u,
            x,
            z,
            y,
            {
                "eps_abs": 1e-6,
                "eps_rel": 1e-6,
                "polishing": True,
                "polish_delta": 1e-6,
                "polish_refine_iter": 3,
            },
        )
    for actual, saved in zip((x, z, y), before):
        assert torch.equal(actual, saved)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_residual_unscaling_preserves_scaled_cancellation(device):
    """Pinned auxil.c unscales the residual, not its large cancelling terms."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64, device=device)

    # In scaled coordinates the exact dual residual is 1 and the primal
    # residual is 2**-50. Dividing the input terms first loses this evidence.
    residuals = torch_osqp_module._residuals(
        tensor([[2.0**52]]), tensor([-2.0**52]), tensor([[1.0]]),
        tensor([1.0]), tensor([1.0 + 2.0**-50]), tensor([1.0]), 0.0, 0.0,
        scaling={"D": tensor([3.0]), "E": tensor([5.0]), "cost": tensor(7.0)},
    )
    assert float(residuals[0]) == pytest.approx(2.0**-50 / 5.0, rel=1e-15, abs=0.0)
    assert float(residuals[1]) == pytest.approx(1.0 / 21.0, rel=1e-15, abs=0.0)
