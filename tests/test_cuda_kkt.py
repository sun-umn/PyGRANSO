"""Independent double-precision CUDA KKT, cache and synchronization checks."""

import pytest
import torch
from pinned_osqp_c_oracle import paired_settings

import pygranso.private.torchOSQP as osqp
from pygranso.private.osqpWorkspace import TorchOSQPWorkspace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def problem(n):
    generator = torch.Generator().manual_seed(n)

    def rand(size):
        return torch.rand(size, generator=generator, dtype=torch.float64)

    p = torch.diag(2 + rand(n)) + torch.diag(0.1 * rand(n - 1), 1)
    p = p + torch.triu(p, 1).T
    a = torch.eye(n, dtype=p.dtype) + torch.diag(0.3 * rand(n - 1) - 0.15, -1)
    q = 4 * rand(n) - 2
    return p, q, a, torch.full_like(q, -0.4), torch.full_like(q, 0.4)


def inputs(data, device):
    return [
        v.to_sparse_csc().to(device) if i in (0, 2) else v.to(device)
        for i, v in enumerate(data)
    ]


@pytest.mark.parametrize("n", [100, 300, 1000])
def test_cuda_random_sparse_kkt(n):
    data = problem(n)
    p, q, a, l, u = data
    _, opts = paired_settings(
        rho=0.5, scaling=0, adaptive_rho=0, eps_abs=1e-7, eps_rel=1e-7
    )
    opts["sparse"] = True
    reference, cpu = osqp.solve_torch_osqp_direct(*inputs(data, "cpu"), opts)
    ws = TorchOSQPWorkspace()
    x, gpu = osqp.solve_torch_osqp_direct(*inputs(data, "cuda"), opts, ws)
    assert cpu["status_val"] == gpu["status_val"] == 1
    torch.testing.assert_close(x.cpu(), reference, atol=1e-7, rtol=1e-7)
    x, y = x.cpu().flatten(), gpu["public_y"].cpu().flatten()
    ax = a @ x
    assert (ax - ax.clamp(l, u)).abs().max() < 5e-7
    assert (p @ x + q + a.T @ y).abs().max() < 5e-7
    assert (ax - (ax + y).clamp(l, u)).abs().max() < 5e-7

    # Check both block equations against the assembled full KKT with two RHS.
    k = ws.linear_solver.matrix.to_dense()
    rhs = torch.randn(
        (2 * n, 2), generator=torch.Generator().manual_seed(n), dtype=x.dtype
    ).cuda()
    result, _ = ws.linear_solver.solve(rhs)
    scale = k.abs().sum(1).max() * result.abs().amax(0) + rhs.abs().amax(0)
    assert ((k @ result - rhs).abs().amax(0) / scale).max() < 1e-12


def test_gpu_factors_reused_after_linear_term_change():
    data = inputs(problem(12), "cuda")
    _, opts = paired_settings(scaling=0, adaptive_rho=0)
    opts["sparse"] = True
    ws = TorchOSQPWorkspace()
    osqp.solve_torch_osqp_direct(*data, opts, ws)
    lower = ws.linear_solver._factors[2]
    data[1] = data[1] + 0.1
    _, info = osqp.solve_torch_osqp_direct(*data, opts, ws)
    assert info["factorizations_this_solve"] == 0
    assert ws.linear_solver._factors[2] is lower
    for change in ("rho", "matrix"):
        if change == "rho":
            opts["rho"] *= 2
        else:
            data[0] = data[0] * 2
        _, info = osqp.solve_torch_osqp_direct(*data, opts, ws)
        assert info["factorizations_this_solve"] == 1
        assert ws.linear_solver._factors[2] is not lower
        lower = ws.linear_solver._factors[2]


@pytest.mark.parametrize("interval", [0, 3])
def test_cuda_iterations_capture_without_host_sync(monkeypatch, interval):
    # CUDA forbids synchronous device-to-host copies or host waits during
    # stream capture. Capture iterations 2-7, including rho checks at 3 and 6.
    data = problem(16)
    _, opts = paired_settings(
        rho=1e-4 if interval else 0.5,
        scaling=0,
        adaptive_rho=int(bool(interval)),
        adaptive_rho_interval=interval,
        adaptive_rho_tolerance=1.01,
        eps_abs=1e-7,
        eps_rel=1e-7,
    )
    opts["sparse"] = True
    cpu, ref = osqp.solve_torch_osqp_direct(*inputs(data, "cpu"), opts)
    original = osqp.build_kkt_rhs
    graph = torch.cuda.CUDAGraph()
    calls = 0

    def capture(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            graph.capture_begin()
        if calls == 8:
            graph.capture_end()
            graph.replay()
        return original(*args)

    monkeypatch.setattr(osqp, "build_kkt_rhs", capture)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x, info = osqp.solve_torch_osqp_direct(*inputs(data, "cuda"), opts)
    torch.cuda.current_stream().wait_stream(stream)
    assert calls >= 8
    assert info["status_val"] == ref["status_val"] == 1
    assert bool(info["rho_updates"]) == bool(interval)
    torch.testing.assert_close(x.cpu(), cpu, rtol=1e-6, atol=1e-6)
