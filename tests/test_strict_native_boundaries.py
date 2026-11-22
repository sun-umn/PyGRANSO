"""Differential checks for the pinned builtin binary64 publication and bound arithmetic."""
import ctypes
import math
import struct

import numpy as np
import pytest
import torch
from pinned_osqp_c_internals import VecF
from pinned_osqp_c_oracle import load_pinned_library, pinned_dll_path
from pinned_osqp_c_session import PinnedOSQPSession, paired_settings

from pygranso.private import torchOSQP as direct
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchOSQP import admm_vector_update, solve_torch_osqp_direct

needs_dll = pytest.mark.skipif(
    not pinned_dll_path().is_file(), reason="pinned OSQP C library is unavailable"
)


def _same_bits(actual, expected):
    a, b = (np.asarray(value, dtype="<f8") for value in (actual, expected))
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes()


def _solve(problem, settings, workspace=None):
    return solve_torch_osqp_direct(
        *(torch.as_tensor(a, dtype=torch.float64) for a in problem),
        settings, workspace,
    )


@needs_dll
@pytest.mark.parametrize("scaling", [0, 10])
@pytest.mark.parametrize("check_dualgap", [0, 1])
def test_raw_ieee_bound_gap_decision_matches_retained_c(scaling, check_dualgap):
    # Phase 3.24 pinned input representations. Keep raw infinities.
    bound = struct.unpack("<d", bytes.fromhex("666666666666e63f"))[0]
    problem = (
        [[4.0, 1.0], [1.0, 2.0]], [1.0, 1.0],
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        [1.0, 0.0, 0.0, -math.inf], [1.0, bound, bound, math.inf],
    )
    c_settings, t_settings = paired_settings(
        scaling=scaling, adaptive_rho=0, adaptive_rho_interval=0, warm_starting=0,
        max_iter=2000, eps_abs=1e-8, eps_rel=1e-8, check_dualgap=check_dualgap,
    )
    with PinnedOSQPSession(*problem, **c_settings) as reference:
        assert reference.setup_code == 0
        c = reference.solve()
    x, t = _solve(problem, t_settings)
    assert c["status_val"] == (7 if check_dualgap else 1)
    assert t["status_val"] == c["status_val"]
    assert t["admm_iterations"] == c["iter"]
    for name in ("dual_objective", "duality_gap", "rel_kkt_error"):
        assert math.isnan(c[name]) and math.isnan(t[name])
    _same_bits(t["objective"], c["objective"])
    _same_bits(x.reshape(-1), c["x"])
    _same_bits(t["public_y"], c["y"])


@needs_dll
@pytest.mark.parametrize("scaling", [0, 10])
@pytest.mark.parametrize("kind", ["primal", "dual"])
def test_infeasible_public_buffers_and_certificate_match_retained_c(scaling, kind):
    if kind == "primal":
        problem = ([[1.0]], [0.0], [[1.0], [1.0]], [0.0, 1.0], [0.0, 1.0])
        status, cert = 3, "prim_inf_cert"
    else:
        problem = ([[0.0]], [-1.0], [[0.0]], [-1e30], [1e30])
        status, cert = 5, "dual_inf_cert"
    c_settings, t_settings = paired_settings(scaling=scaling, adaptive_rho=0)
    ws = TorchOSQPWorkspace()
    with PinnedOSQPSession(*problem, **c_settings) as reference:
        assert reference.setup_code == 0
        c = reference.solve()
    x, t = _solve(problem, t_settings, ws)
    assert c["status_val"] == t["status_val"] == status
    assert t["admm_iterations"] == c["iter"]
    assert np.all(c["x"] == 0x7FC00000)
    _same_bits(x.reshape(-1), c["x"])
    _same_bits(t["public_y"], c["y"])
    _same_bits(t[cert], c[cert])
    _same_bits(t["objective"], c["objective"])
    assert ws.work_iterates is None


@needs_dll
def test_admm_projection_tie_matches_retained_c_sign_bit():
    # Phase 3.50 verified this private vector ABI and routine RVA for the
    # hash-checked DLL. Swapping the equal zero operands changes the result.
    class Vector(ctypes.Structure):
        _fields_ = [("values", ctypes.POINTER(ctypes.c_double)), ("length", ctypes.c_int64)]

    library = load_pinned_library()
    signature = ctypes.CFUNCTYPE(None, *([ctypes.POINTER(Vector)] * 4))
    project = signature(library._handle + 0xB5C0)
    values = [np.array(a, dtype="<f8") for a in (
        [0., 0., 0.], [-0., -0., -0.], [0., 0., -0.], [0.75, -0., 0.],
    )]
    vectors = [Vector(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), a.size) for a in values]
    project(*(ctypes.byref(v) for v in vectors))

    argument = torch.from_numpy(values[1])
    ones = torch.ones_like(argument)
    _, result, _ = admm_vector_update(
        argument, argument, argument, argument, argument, ones,
        torch.from_numpy(values[2]), torch.from_numpy(values[3]), 1.0,
        rho_inv_vec=ones,
    )
    _same_bits(result, values[0])


@needs_dll
def test_polishing_projection_equality_tie_matches_retained_c(monkeypatch):
    # With an equality expressed as [+0, -0], the upper operand wins the
    # native second comparison. Observe the real polishing candidate before
    # acceptance, since equal residuals cause it to be rejected.
    library = load_pinned_library()
    project = ctypes.CFUNCTYPE(None, *([ctypes.POINTER(VecF)] * 4))(
        library._handle + 0xB5C0
    )
    projected, shifted, lower, upper = (
        np.array([a], dtype="<f8") for a in (+0., +0., +0., -0.)
    )
    vectors = [
        VecF(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), a.size)
        for a in (projected, shifted, lower, upper)
    ]
    project(*(ctypes.byref(v) for v in vectors))
    _same_bits(projected, [-0.])

    observed = []
    residuals = direct._residuals

    def record(P, q, A, x, z, y, *args, **kwargs):
        observed.append((z.clone(), (y + z).clone()))
        return residuals(P, q, A, x, z, y, *args, **kwargs)

    _, settings = paired_settings(scaling=0, polishing=True, polish_refine_iter=0)
    matrix = torch.ones((1, 1), dtype=torch.float64)
    zero = torch.zeros(1, dtype=torch.float64)
    monkeypatch.setattr(direct, "_residuals", record)
    _, _, _, info = direct._polish_solution(
        matrix, zero, matrix, torch.from_numpy(lower), torch.from_numpy(upper),
        zero, zero, zero, settings,
    )
    assert info["polishing_status"] == "rejected_no_improvement"
    assert len(observed) == 2
    _same_bits(observed[1][1], shifted)
    _same_bits(observed[1][0], projected)


@needs_dll
@pytest.mark.parametrize("values", [[float("nan"), -3.0], [-0., float("nan")],
                                   [float("nan"), -3., 1., float("nan"), -4.]])
def test_infinity_norm_uses_native_strict_comparison(values):
    data = np.array(values, dtype="<f8")
    vector = VecF(data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), data.size)
    library = load_pinned_library()
    # OSQPVectorf_norm_inf, identified in the hash-checked DLL by matching
    # the named vector.c.obj routine's instructions, excluding relocations.
    norm = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(VecF))(
        library._handle + 0xCB60
    )
    _same_bits(direct._inf_norm(torch.from_numpy(data)), norm(ctypes.byref(vector)))


@pytest.mark.parametrize(("primal", "dual", "gap", "expected"), [
    (2., float("nan"), -7., 2.),
    (float("nan"), 3., -7., -7.),
    (+0., -0., -0., -0.),
    (2., 3., float("nan"), float("nan")),
])
def test_relative_kkt_uses_pinned_nested_max_operand_order(primal, dual, gap, expected):
    # Expected values follow the strict ternary c_max macro and the signed
    # gap operand in auxil.c::update_info.
    result = direct._relative_kkt_error(
        *(torch.tensor(v, dtype=torch.float64) for v in (primal, dual, gap))
    )
    _same_bits(result, expected)


def test_gap_normalizer_uses_native_operand_order_with_nan_cost_terms():
    # auxil.c::compute_duality_gap_tol compares the support term last.
    # The actual gap remains NaN and cannot pass the strict stopping test.
    one = torch.ones(1, dtype=torch.float64)
    values = direct._compute_obj_val_dual_gap(
        one.reshape(1, 1), one, torch.zeros_like(one), 3. * one,
        one.new_full((1,), float("nan")), one, one.squeeze(), 1e-4, 1e-4,
    )
    assert torch.isnan(values["duality_gap"])
    _same_bits(values["scaled_gap_normalizer"], 3.)
    _same_bits(values["eps_scaled_duality_gap"], 1e-4 + 1e-4 * 3.)


@needs_dll
@pytest.mark.parametrize("path", ["projection", "certificate", "duality_gap"])
def test_polar_projection_uses_native_tie_and_nan_branches(monkeypatch, path):
    sentinel = 1e30
    direction, lower, upper = (
        np.array(values, dtype="<f8")
        for values in (
            [-0., -0., float("nan"), float("nan"), -0., -0., -2., 3.],
            [0., -sentinel, 0., -sentinel, -sentinel, 0., 0., -sentinel],
            [sentinel, 1., sentinel, 1., sentinel, 1., sentinel, 1.],
        )
    )
    expected = direction.copy()
    vectors = [
        VecF(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), a.size)
        for a in (expected, lower, upper)
    ]
    library = load_pinned_library()
    project = ctypes.CFUNCTYPE(
        None, *([ctypes.POINTER(VecF)] * 3), ctypes.c_double
    )(library._handle + 0xCF30)
    project(*(ctypes.byref(v) for v in vectors), sentinel * 1e-4)
    d, l, u = map(torch.from_numpy, (direction, lower, upper))
    if path == "certificate":
        observed = []
        norm = direct._inf_norm

        def record(value):
            observed.append(value.clone())
            return norm(value)

        monkeypatch.setattr(direct, "_inf_norm", record)
        direct._primal_infeasibility_certificate(
            torch.zeros((d.numel(), 1), dtype=d.dtype), l, u, d, 1e-4,
        )
        actual = observed[0]
    elif path == "duality_gap":
        observed = []
        support = direct._signed_bound_support

        def record_support(lower, upper, value):
            observed.append(value.clone())
            return support(lower, upper, value)

        monkeypatch.setattr(direct, "_signed_bound_support", record_support)
        zero = d.new_zeros(1)
        direct._compute_obj_val_dual_gap(
            zero.reshape(1, 1), zero, l, u, zero, d, d.new_tensor(1.), 1e-4, 1e-4,
        )
        expected[np.abs(expected) < 1e-15] = 0.
        actual = observed[0]
    else:
        actual = direct._project_polar_reccone(d, l, u)
    _same_bits(actual, expected)



@needs_dll
def test_nonconvex_solve_uses_native_objective_and_public_buffers():
    problem = ([[-0.1]], [1.0], np.empty((0, 1)), [], [])
    c_settings, t_settings = paired_settings(sigma=0.5, scaling=0, adaptive_rho=0)
    with PinnedOSQPSession(*problem, **c_settings) as reference:
        assert reference.setup_code == 0
        c = reference.solve()
    ws = TorchOSQPWorkspace()
    x, t = _solve(problem, t_settings, ws)
    assert c["status_val"] == t["status_val"] == 9
    assert t["admm_iterations"] == c["iter"]
    _same_bits(t["objective"], c["objective"])
    _same_bits(x.reshape(-1), c["x"])
    _same_bits(t["public_y"], c["y"])
    assert ws.work_iterates is None


@needs_dll
def test_changed_q_naturally_accepts_rho_and_consumes_rebuilt_factors(monkeypatch):
    from pygranso.private.torchQDLDLSolve import QDLDLSolver

    original = (
        [[4.,1.],[1.,2.]], [1.,1.], [[1.,1.],[1.,0.],[0.,1.],[0.,1.]],
        [1.,0.,0.,-1e30], [1.,.7,.7,1e30],
    )
    updated = (original[0], [0., 1.], *original[2:])
    c_settings, t_settings = paired_settings(
        rho=1e-4, scaling=0, adaptive_rho=1, adaptive_rho_interval=25,
        eps_abs=1e-8, eps_rel=1e-8, max_iter=2000,
    )
    ws = TorchOSQPWorkspace()
    with PinnedOSQPSession(*original, **c_settings) as reference:
        assert reference.setup_code == 0
        c_initial = reference.solve()
        _, t_initial = _solve(original, t_settings, ws)
        assert c_initial["rho_updates"] == t_initial["rho_updates"] == 2
        assert reference.update(q=updated[1]) == 0
        c_resumed = reference.solve()
        original_solve = QDLDLSolver.solve
        uses = []
        def observe(self, rhs, **kwargs):
            result = original_solve(self, rhs, **kwargs)
            uses.append(self.factorization_count)
            return result

        monkeypatch.setattr(QDLDLSolver, "solve", observe)
        x, t = _solve(updated, t_settings, ws)

    assert c_resumed["status_val"] == t["status_val"] == 1
    assert c_resumed["rho_updates"] == t["rho_updates"] == 1
    assert c_resumed["iter"] == t["admm_iterations"] == 50
    assert uses == [3] * 25 + [4] * 25
    _same_bits(t["rho_bar"], c_resumed["rho"])
    _same_bits(x.reshape(-1), c_resumed["x"])
    _same_bits(t["public_y"], c_resumed["y"])
