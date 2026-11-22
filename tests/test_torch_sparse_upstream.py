"""Pinned C fixture contracts on CPU/CUDA tensors and opt-in CUDA cuDSS.

TESTS_TOL is unchanged from tests/osqp_tester.h (1e-4 double, 1e-3
float). Solver tolerances are separate and come from the original cases.
These are observable/algorithm checks, not bitwise QDLDL/cuDSS equivalence.
"""

import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pinned_osqp_c_oracle import pinned_dll_path, solve_dense_qp
from scipy import sparse as scipy_sparse
from test_osqp_backend_agreement import (
    _pinned_basic_qp2_fixture,
    _pinned_basic_qp_fixture,
    _pinned_large_qp_fixture,
    _pinned_update_matrices_fixture,
)

from pygranso.private import torchOSQP as runtime
from pygranso.private.osqpTorchAdapter import DEFAULT_OSQP_SETTINGS
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace
from pygranso.private.torchOSQP import solve_torch_osqp_direct


@pytest.fixture(params=["cpu-qdldl", "cuda-qdldl", "cuda-cudss"])
def execution(request):
    device = "cpu" if request.param == "cpu-qdldl" else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    use_sparse = request.param == "cuda-cudss"
    if use_sparse:
        matrix = torch.eye(1, device=device).to_sparse_csr()
        try:
            torch.sparse.spsolve(matrix, torch.ones(1, device=device))
        except RuntimeError as exc:
            if "requires compiling PyTorch with CUDA cuDSS" in str(exc):
                pytest.skip("cuDSS unavailable")
            raise
    return device, use_sparse


def fixture(name):
    opts = dict(max_iter=2000, scaling=1, eps_abs=1e-5, eps_rel=1e-5)
    if name == "basic_qp":
        problem = _pinned_basic_qp_fixture()
        opts.update(scaling=0)
    elif name == "basic_qp2":
        problem = _pinned_basic_qp2_fixture()
        opts.update(eps_abs=1e-6, eps_rel=1e-6)
    elif name == "basic_lp":
        problem = _pinned_basic_qp_fixture()
        problem.update(
            P=torch.zeros_like(problem["P"]),
            q=torch.tensor([1.1, 0.95], dtype=torch.float64),
        )
    elif name == "unconstrained":
        problem = dict(
            P=torch.diag(
                torch.tensor(
                    [0.617022, 0.92032449, 0.20011437, 0.50233257, 0.34675589],
                    dtype=torch.float64,
                )
            ),
            q=torch.tensor(
                [-1.10593508, -1.65451545, -2.3634686, 1.13534535, -1.01701414],
                dtype=torch.float64,
            ),
            A=torch.empty((0, 5)),
            l=torch.empty(0),
            u=torch.empty(0),
        )
    elif name == "no_active_set":
        problem = dict(
            P=torch.diag(torch.tensor([11.0, 5.0], dtype=torch.float64)),
            q=torch.zeros(2, dtype=torch.float64),
            A=torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
            l=torch.tensor([-1.0, -0.5, -0.25], dtype=torch.float64),
            u=torch.tensor([1.0, 0.7, 0.7], dtype=torch.float64),
        )
    elif name == "update_matrices":
        problem = _pinned_update_matrices_fixture()
        opts.update(max_iter=1000)
    elif name == "large_qp":
        problem = _pinned_large_qp_fixture()
    else:
        raise AssertionError(name)
    return problem, opts


def arguments(problem, device, dtype):
    return tuple(
        problem[k].to(device=device, dtype=dtype) for k in ("P", "q", "A", "l", "u")
    )


@pytest.fixture(scope="module")
def float_c_oracle():
    """Reuse the C bridge with an isolated float/long ABI and float buffers.

    The original bridge is double-only. An isolated module keeps its existing
    callers unchanged; every pointer-bearing structure is rebuilt for float.
    The DLL is built from the archived, unmodified pinned C sources.
    """
    path = Path(__file__).with_name("pinned_osqp_c_oracle.py")
    spec = importlib.util.spec_from_file_location("pinned_float_oracle", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    replacements = {
        ct.c_double: ct.c_float,
        ct.POINTER(ct.c_double): ct.POINTER(ct.c_float),
    }
    for name in (
        "OSQPSettings",
        "OSQPInfo",
        "OSQPSolution",
        "OSQPSolver",
        "OSQPCscMatrix",
    ):
        original = getattr(oracle, name)
        clone = type(
            name,
            (ct.Structure,),
            {
                "_fields_": [
                    (key, replacements.get(kind, kind))
                    for key, kind in original._fields_
                ]
            },
        )
        replacements[original] = clone
        replacements[ct.POINTER(original)] = ct.POINTER(clone)
        setattr(oracle, name, clone)
    oracle.OSQPFloat = ct.c_float
    oracle._as_float = lambda values: np.ascontiguousarray(values, dtype=np.float32)
    dll = (
        Path(__file__).resolve().parents[1]
        / "artifacts/torch-sparse-upstream-20261007/native/build-float/upstream/out/osqp.dll"
    )
    if not dll.is_file():
        pytest.skip("Build the pinned float C validation library first")
    manifest = json.loads(
        (dll.parents[3] / "source-and-build-manifest.json").read_text()
    )
    oracle.OSQP_DLL_SHA256 = manifest["builds"]["float"]["dll_sha256"]
    assert hashlib.sha256(dll.read_bytes()).hexdigest() == oracle.OSQP_DLL_SHA256
    library = oracle.load_pinned_library(dll)
    return oracle, library


def c_reference(qp, opts, float_oracle=None):
    # The raw C API uses a finite OSQP_INFTY sentinel, not IEEE infinity.
    values = [v.detach().cpu().double().numpy() for v in qp]
    values[3:] = [np.clip(v, -1e30, 1e30) for v in values[3:]]
    if float_oracle:
        oracle, library = float_oracle
        return oracle.solve_dense_qp(*values, settings_overrides=opts, library=library)
    if not pinned_dll_path().is_file():
        pytest.skip("pinned double C DLL unavailable")
    return solve_dense_qp(*values, settings_overrides=opts)


def assert_observables(qp, x, info, reference, dtype, use_sparse, relative=False):
    tol = 1e-3 if dtype == torch.float32 else 1e-4
    assert reference["setup_code"] == reference["solve_code"] == 0
    assert info["status_val"] == reference["status_val"] == 1
    for actual, expected in (
        (x.flatten(), reference["x"]),
        (info["state"]["y"], reference["y"]),
    ):
        # Absolute C TESTS_TOL except QP2's explicitly relative comparison.
        expected = actual.new_tensor(expected)
        error = (
            torch.max(torch.abs(actual - expected))
            if actual.numel()
            else actual.new_tensor(0.0)
        )
        denominator = (
            float(torch.max(torch.abs(expected)))
            if relative and expected.numel()
            else 1.0
        )
        assert float(error) / denominator < tol
    assert (
        abs(info["objective"] - reference["objective"])
        / (abs(reference["objective"]) if relative else 1.0)
        < tol
    )
    P, q, A, l, u = [v.cpu().double() for v in qp]
    x, y = x.flatten().cpu().double(), info["state"]["y"].cpu().double()
    Ax, Px, Aty = A @ x, P @ x, A.T @ y

    def norm(v):
        return float(torch.max(torch.abs(v))) if v.numel() else 0.0

    primal = norm(Ax - torch.maximum(torch.minimum(Ax, u), l))
    dual = norm(Px + q + Aty)
    # This checks the independent solution against the reported stopping limits;
    # it does not replace solver tolerances with the looser fixture tolerance.
    assert primal < info["eps_primal"]
    assert dual < info["eps_dual"]
    if use_sparse:
        assert "spsolve" in info["linear_solver"]
        assert not info["factorization_reused"]
        assert info["factorization_count"] > info["linear_solve_count"]


@pytest.mark.parametrize(
    "name",
    [
        "basic_lp",
        "basic_qp",
        "basic_qp2",
        "unconstrained",
        "no_active_set",
        "update_matrices",
        "large_qp",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("polishing", [False, True])
def test_original_c_fixture(execution, name, dtype, polishing, request):
    device, use_sparse = execution
    problem, common = fixture(name)
    # The float C build disables the gap check by default, as in its constants.
    common.update(
        polishing=polishing, polish_refine_iter=4, check_dualgap=dtype == torch.float64
    )
    qp = arguments(problem, device, dtype)
    float_oracle = (
        request.getfixturevalue("float_c_oracle") if dtype == torch.float32 else None
    )
    reference = c_reference(qp, common, float_oracle)
    opts = {
        **DEFAULT_OSQP_SETTINGS,
        **common,
        "use_sparse": use_sparse,
        "return_state": True,
        "return_info": True,
    }
    x, info = solve_torch_osqp_direct(*qp, opts)
    assert_observables(
        qp,
        x,
        info,
        reference,
        dtype,
        use_sparse,
        relative=name in {"basic_qp2", "large_qp"},
    )
    polish_codes = {
        "disabled": 0,
        "accepted": 1,
        "skipped_no_active_constraints": 2,
    }
    request.node.user_properties.extend(
        [
            ("torch_polish_status", info["polishing_status"]),
            ("c_polish_status", reference["status_polish"]),
        ]
    )
    if name in {"basic_lp", "basic_qp2", "unconstrained", "no_active_set"}:
        # These original C cases require a particular polishing disposition.
        assert polish_codes[info["polishing_status"]] == reference["status_polish"]
    # Independent constants in the original generate_problem.py files prevent
    # a shared fixture/bridge mistake from agreeing with itself.
    expected = {
        "basic_lp": ([0.3, 0.7], [-1.1, 0.0, 0.15, 0.0], 0.995),
        "basic_qp": ([0.3, 0.7], [-2.9, 0.0, 0.2, 0.0], 1.88),
        "basic_qp2": ([15.0, 0.0], [0.0, 508.0, 168.0, 0.0, 0.0], 1282.5),
        "unconstrained": (
            [1.79237542, 1.79775228, 11.81058885, -2.26014678, 2.93293975],
            [],
            -19.209752026813277,
        ),
        "no_active_set": ([0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
    }.get(name)
    if expected:
        tol = 1e-3 if dtype == torch.float32 else 1e-4
        for actual, target in (
            (x.flatten(), expected[0]),
            (info["state"]["y"], expected[1]),
        ):
            if not target:
                assert actual.numel() == 0
                continue
            target = actual.new_tensor(target)
            denominator = float(target.abs().max()) if name == "basic_qp2" else 1.0
            assert float((actual - target).abs().max()) / denominator < tol
        denominator = abs(expected[2]) if name == "basic_qp2" else 1.0
        assert abs(info["objective"] - expected[2]) / denominator < tol
    if name == "large_qp":
        assert abs(info["objective"] - problem["expected_objective"]) / problem[
            "expected_objective"
        ] < (1e-3 if dtype == torch.float32 else 1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("change", ["q", "bounds", "P", "A", "P+A"])
def test_original_c_update_fixtures(execution, dtype, change, request):
    device, use_sparse = execution
    problem, common = fixture(
        "update_matrices" if change in {"P", "A", "P+A"} else "basic_qp"
    )
    common.update(
        polishing=True, polish_refine_iter=4, check_dualgap=dtype == torch.float64
    )
    if change in {"q", "bounds"}:
        common.update(max_iter=200, warm_starting=False)
    opts = {
        **DEFAULT_OSQP_SETTINGS,
        **common,
        "use_sparse": use_sparse,
        "return_state": True,
        "return_info": True,
    }
    if "warm_starting" in common:
        opts["warm_start"] = bool(common["warm_starting"])
    workspace = TorchOSQPWorkspace()
    solve_torch_osqp_direct(*arguments(problem, device, dtype), opts, workspace)
    for key in {
        "q": ("q",),
        "bounds": ("l", "u"),
        "P": ("P",),
        "A": ("A",),
        "P+A": ("P", "A"),
    }[change]:
        problem[key] = problem[key + "_new"]
    qp = arguments(problem, device, dtype)
    actual, info = solve_torch_osqp_direct(*qp, opts, workspace)
    float_oracle = (
        request.getfixturevalue("float_c_oracle") if dtype == torch.float32 else None
    )
    assert_observables(
        qp, actual, info, c_reference(qp, common, float_oracle), dtype, use_sparse
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_c_reciprocal_and_recovery_association(execution, dtype):
    """Non-dyadic inputs detect algebraically equal but differently rounded code.

    auxil.c::compute_rhs/update_z and qdldl_interface.c::solve_linsys_qdldl
    use rho_inv*y and then add rho_inv*nu to the existing lower RHS.
    """
    device, _ = execution

    def t(values):
        return torch.tensor(values, device=device, dtype=dtype)

    rho = t([0.1, 0.3, 0.7, 0.11])
    x, z, y, nu = (
        t([0.123, -2.34, 5.67, -0.891]),
        t([1.123, 2.7, -0.345, 9.81]),
        t([2.777, -0.434, 5.66, 7.899]),
        t([3.55, 4.31, -0.22, 6.7]),
    )
    q, reciprocal = t([0.3, -0.2, 0.7, -0.1]), rho.reciprocal()
    rhs = runtime.build_kkt_rhs(x, z, y, q, 1e-6, reciprocal)
    torch.testing.assert_close(
        rhs, torch.cat((1e-6 * x - q, z - reciprocal * y)), rtol=0, atol=0
    )
    recovered = runtime.recover_z_tilde(rhs[4:], nu, reciprocal)
    torch.testing.assert_close(recovered, rhs[4:] + reciprocal * nu, rtol=0, atol=0)
    alpha = 1.6
    lower, upper = t([-100] * 4), t([100] * 4)
    relaxed = alpha * recovered + (1 - alpha) * z
    expected_z = torch.maximum(torch.minimum(reciprocal * y + relaxed, upper), lower)
    actual = runtime.admm_vector_update(
        x, x, recovered, z, y, rho, lower, upper, alpha, rho_inv_vec=reciprocal
    )
    torch.testing.assert_close(actual[1], expected_z, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_each_admm_step_uses_source_arithmetic(
    execution, dtype, monkeypatch
):
    device, use_sparse = execution
    calls = []
    for name in ("build_kkt_rhs", "recover_z_tilde", "admm_vector_update"):
        original = getattr(runtime, name)

        def record(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            result = _original(*args, **kwargs)
            if _name == "build_kkt_rhs":
                x, z, y, q, sigma, inv = args
                target = torch.cat((sigma * x - q, -(inv * y) + z))
            elif _name == "recover_z_tilde":
                rhs, nu, inv = args
                target = rhs + inv * nu
            else:
                xt, x, zt, z, y, rho, lower, upper, alpha = args
                relaxed = alpha * zt + (1.0 - alpha) * z
                projected = (kwargs["rho_inv_vec"] * y + relaxed).clamp(lower, upper)
                delta = rho * (relaxed - projected)
                target = (alpha * xt + (1.0 - alpha) * x, projected, y + delta, delta)
            torch.testing.assert_close(result, target, rtol=0, atol=0)
            return result

        monkeypatch.setattr(runtime, name, record)
    problem, common = fixture("basic_qp")
    solve_torch_osqp_direct(
        *arguments(problem, device, dtype),
        {
            **DEFAULT_OSQP_SETTINGS,
            **common,
            "max_iter": 3,
            "check_termination": 0,
            "adaptive_rho": False,
            "use_sparse": use_sparse,
        },
    )
    assert calls == ["build_kkt_rhs", "recover_z_tilde", "admm_vector_update"] * 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "case", ["feasible", "primal", "dual", "both", "random-primal"]
)
def test_original_c_infeasibility_families(execution, dtype, case, request):
    device, use_sparse = execution
    P, q = (
        torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64)),
        torch.tensor([1.0, -1.0], dtype=torch.float64),
    )
    A = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    l = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    bounds = {
        "feasible": [5.0, 3.0, 3.0],
        "primal": [0.0, 3.0, 3.0],
        "dual": [2.0, 3.0, torch.inf],
        "both": [0.0, 3.0, torch.inf],
    }
    common = dict(
        max_iter=2000,
        scaling=0,
        eps_abs=1e-5,
        eps_rel=1e-5,
        polishing=case == "feasible",
        check_dualgap=dtype == torch.float64,
    )
    if case in {"dual", "both"}:
        A[0] = torch.tensor([1.0, 0.0])
    if case == "random-primal":
        # Exact PCG64(2) generator and draw order in the pinned C fixture.
        rng = np.random.Generator(np.random.PCG64(2))
        seed = scipy_sparse.random(50, 50, random_state=rng)
        P = torch.from_numpy((seed.T @ seed + scipy_sparse.eye(50)).toarray())
        q = torch.from_numpy(rng.standard_normal(50))
        a = scipy_sparse.random(150, 50, random_state=rng).tolil()
        upper, lower = 3 + rng.standard_normal(150), -3 + rng.standard_normal(150)
        a[25, :] = a[26, :]
        lower[25] = upper[26] + 10 * rng.random()
        upper[25] = lower[25] + 0.5
        A, l, u = (
            torch.from_numpy(a.toarray()),
            torch.from_numpy(lower),
            torch.from_numpy(upper),
        )
        common.update(scaling=1)
    else:
        u = torch.tensor(bounds[case], dtype=torch.float64)
    qp = tuple(v.to(device=device, dtype=dtype) for v in (P, q, A, l, u))
    float_oracle = (
        request.getfixturevalue("float_c_oracle") if dtype == torch.float32 else None
    )
    reference = c_reference(qp, common, float_oracle)
    opts = {
        **DEFAULT_OSQP_SETTINGS,
        **common,
        "return_info": True,
        "return_state": True,
        "use_sparse": use_sparse,
    }
    x, info = solve_torch_osqp_direct(*qp, opts)
    allowed = {
        "feasible": {1},
        "primal": {3},
        "dual": {5},
        "both": {3, 5},
        "random-primal": {3},
    }[case]
    assert info["status_val"] in allowed
    assert reference["status_val"] in allowed
    if case == "feasible":
        assert_observables(qp, x, info, reference, dtype, use_sparse)
    else:
        assert (x == 0x7FC00000).all()
        kind = "primal" if info["status_val"] == 3 else "dual"
        certificate = info[kind + "_infeasibility_certificate"]
        assert torch.isfinite(certificate).all()
        assert abs(float(certificate.abs().max()) - 1) < (
            1e-3 if dtype == torch.float32 else 1e-4
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "case", ["final-check", "rho-update", "warm-start", "invalid-bounds"]
)
def test_original_c_basic_qp_lifecycle(execution, dtype, case, request):
    device, use_sparse = execution
    problem, common = fixture("basic_qp")
    qp = arguments(problem, device, dtype)
    common.update(polishing=False, adaptive_rho=0, check_dualgap=dtype == torch.float64)
    opts = {
        **DEFAULT_OSQP_SETTINGS,
        **common,
        "return_info": True,
        "return_state": True,
        "use_sparse": use_sparse,
    }
    workspace = TorchOSQPWorkspace()
    if case == "final-check":
        common.update(max_iter=400, check_termination=0)
        opts.update(common)
        x, info = solve_torch_osqp_direct(*qp, opts, workspace)
        float_oracle = (
            request.getfixturevalue("float_c_oracle")
            if dtype == torch.float32
            else None
        )
        reference = c_reference(qp, common, float_oracle)
        assert info["admm_iterations"] == reference["iter"] == 400
        assert_observables(qp, x, info, reference, dtype, use_sparse)
    elif case == "rho-update":
        opts.update(
            scaling=10,
            rho=0.7,
            eps_abs=5e-5,
            eps_rel=5e-5,
            check_termination=1,
            warm_start=False,
        )
        fresh, fresh_info = solve_torch_osqp_direct(*qp, opts)
        solve_torch_osqp_direct(*qp, {**opts, "rho": 0.1}, workspace)
        updated, updated_info = solve_torch_osqp_direct(*qp, opts, workspace)
        assert updated_info["admm_iterations"] == fresh_info["admm_iterations"]
        for x, info in ((fresh, fresh_info), (updated, updated_info)):
            assert info["status_val"] == 1
            tol = 1e-3 if dtype == torch.float32 else 1e-4
            assert (
                float((x.flatten() - x.new_tensor([0.3, 0.7])).abs().max()) / 0.7 < tol
            )
            assert (
                float(
                    (info["state"]["y"] - x.new_tensor([-2.9, 0.0, 0.2, 0.0]))
                    .abs()
                    .max()
                )
                / 2.9
                < tol
            )
            assert abs(info["objective"] - 1.88) < tol
    elif case == "warm-start":
        opts.update(scaling=1, check_termination=1)
        _, cold = solve_torch_osqp_direct(*qp, opts, workspace)
        workspace.state = {
            "x": qp[1].new_zeros(2),
            "y": qp[1].new_zeros(4),
            "z": qp[1].new_zeros(4),
        }
        _, again = solve_torch_osqp_direct(*qp, opts, workspace)
        assert again["admm_iterations"] == cold["admm_iterations"]
        optimum = qp[1].new_tensor([0.3, 0.7])
        workspace.state = {
            "x": optimum,
            "y": qp[1].new_tensor([-2.9, 0.0, 0.2, 0.0]),
            "z": qp[2] @ optimum,
        }
        _, warm = solve_torch_osqp_direct(*qp, opts, workspace)
        assert warm["admm_iterations"] == 1
    else:
        solve_torch_osqp_direct(*qp, opts, workspace)
        state, solver = workspace.state, workspace.linear_solver
        bad = list(qp)
        bad[3] = bad[4] + 1
        with pytest.raises(ValueError, match="bounds|lower"):
            solve_torch_osqp_direct(*bad, opts, workspace)
        assert workspace.state is state
        assert workspace.linear_solver is solver
        assert solver.ready


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_original_c_qp2_vector_update(execution, dtype, request):
    device, use_sparse = execution
    problem = _pinned_basic_qp2_fixture()
    common = dict(
        scaling=10,
        eps_abs=1e-3,
        eps_rel=1e-3,
        polishing=True,
        check_dualgap=dtype == torch.float64,
    )
    opts = {
        **DEFAULT_OSQP_SETTINGS,
        **common,
        "return_info": True,
        "return_state": True,
        "use_sparse": use_sparse,
    }
    workspace = TorchOSQPWorkspace()
    solve_torch_osqp_direct(*arguments(problem, device, dtype), opts, workspace)
    problem.update(
        q=torch.tensor([1.0, 1.0], dtype=torch.float64),
        u=torch.tensor([-2.0, 0.0, -20.0, 100.0, 80.0], dtype=torch.float64),
    )
    qp = arguments(problem, device, dtype)
    x, info = solve_torch_osqp_direct(*qp, opts, workspace)
    float_oracle = (
        request.getfixturevalue("float_c_oracle") if dtype == torch.float32 else None
    )
    assert_observables(
        qp,
        x,
        info,
        c_reference(qp, common, float_oracle),
        dtype,
        use_sparse,
        relative=True,
    )
    assert info["polishing_status"] == "accepted"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_original_c_nonconvex_solve(execution, dtype, request):
    device, use_sparse = execution
    problem = _pinned_basic_qp2_fixture()
    problem["P"] = torch.tensor([[2.0, 5.0], [5.0, 1.0]], dtype=torch.float64)
    qp = arguments(problem, device, dtype)
    common = dict(
        sigma=5.0,
        adaptive_rho=0,
        scaling=1,
        max_iter=2000,
        eps_abs=1e-5,
        eps_rel=1e-5,
        check_dualgap=dtype == torch.float64,
    )
    float_oracle = (
        request.getfixturevalue("float_c_oracle") if dtype == torch.float32 else None
    )
    reference = c_reference(qp, common, float_oracle)
    opts = {
        **DEFAULT_OSQP_SETTINGS,
        **common,
        "use_sparse": use_sparse,
        "return_info": True,
    }
    x, info = solve_torch_osqp_direct(*qp, opts)
    assert info["status_val"] == reference["status_val"] == 9
    # This pinned build's OSQP_NAN macro is a numeric cast, not IEEE NaN.
    assert info["objective"] == reference["objective"] == 2143289344
    assert (x == 0x7FC00000).all()
    assert info["primal_infeasibility_certificate"] is None
    assert info["dual_infeasibility_certificate"] is None
