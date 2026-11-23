"""CPU Float64 controls for the PyGRANSO solver integration."""

import torch
from pinned_osqp_c_oracle import paired_settings

from pygranso.private import osqpTorchAdapter as adapter
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchOSQP import solve_torch_osqp_direct
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct


def test_pygranso_default_torch_route_solves_constrained_problem(monkeypatch):
    # The analytic optimum is x = 0 and y = 2, with objective 2.
    def objective(variables):
        x, y = variables.x[0], variables.x[1]
        inequalities, equalities = pygransoStruct(), pygransoStruct()
        inequalities.nonnegative_x = -x
        equalities.fixed_y = y - 2
        return (x + 1) ** 2 + (y - 3) ** 2, inequalities, equalities

    core = adapter.solve_torch_osqp_direct
    statuses = []

    def observe(P, q, A, lower, upper, settings, workspace=None):
        for value in (P, q, A, lower, upper):
            assert value.dtype == torch.float64 and value.device.type == "cpu"
        x, info = core(P, q, A, lower, upper, settings, workspace)
        statuses.append(info["status_val"])
        return x, info

    monkeypatch.setattr(adapter, "solve_torch_osqp_direct", observe)
    options = pygransoStruct()
    options.torch_device = torch.device("cpu")
    options.double_precision = True
    options.x0 = torch.tensor([[-2.0], [-2.0]], dtype=torch.float64)
    options.QPsolver = "osqp"
    options.print_level = 0
    options.quadprog_info_msg = False
    options.maxit = 100
    tolerance = 1e-7
    options.opt_tol = options.viol_eq_tol = options.viol_ineq_tol = tolerance
    options.osqp_settings = {
        "eps_abs": 1e-8, "eps_rel": 1e-8, "max_iter": 4000, "polishing": False,
    }

    solution = pygranso({"x": [2, 1]}, objective, options)

    assert solution.termination_code == 0
    assert solution.quadprog_failure_rate == 0.0
    assert len(statuses) > 2
    assert set(statuses) <= {1, 2}
    assert solution.final.x.dtype == torch.float64 and solution.final.x.device.type == "cpu"
    expected = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    torch.testing.assert_close(solution.final.x, expected, atol=tolerance, rtol=0)
    assert abs(float(solution.final.f) - 2.0) <= 4 * tolerance
    assert float(solution.final.tv) <= 2 * tolerance
    assert float(solution.stat_value) <= tolerance


def test_interleaved_float64_workspaces_keep_independent_state_and_factors():
    P = torch.ones((1, 1), dtype=torch.float64)
    lower, upper = P.new_zeros(1), P.new_ones(1)
    _, settings = paired_settings(scaling=0, adaptive_rho=0, check_termination=1)
    first, second = TorchOSQPWorkspace(), TorchOSQPWorkspace()

    def solve(workspace, q_value):
        return solve_torch_osqp_direct(
            P, P.new_tensor([q_value]), P, lower, upper, settings, workspace,
        )[1]

    assert solve(first, -0.25)["status_val"] == 1
    assert solve(second, -0.75)["status_val"] == 1
    second_state = {key: value.clone() for key, value in second.state.items()}
    second_factors = second.linear_solver.factors
    second_info = second.last_info
    second_counts = (second.linear_solver.factorization_count, second.linear_solver.solve_count)

    updated = solve(first, -0.5)

    assert updated["status_val"] == 1
    assert updated["factorization_reused"]
    assert first.linear_solver is not second.linear_solver
    assert first.linear_solver.factors is not second_factors
    assert second.linear_solver.factors is second_factors
    assert second.last_info is second_info
    assert (second.linear_solver.factorization_count, second.linear_solver.solve_count) == second_counts
    for key, value in second_state.items():
        assert torch.equal(second.state[key], value)
