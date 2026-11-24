"""Replay double-precision public outputs recorded from the pinned C oracle."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pinned_osqp_c_oracle import PinnedOSQPSession, paired_settings, pinned_dll_path

from pygranso.private.torchOSQP import solve_torch_osqp_direct

INFO_FIELDS = (
    ("status_val", "status_val"),
    ("iter", "admm_iterations"),
    ("rho_updates", "rho_updates"),
    ("rho", "rho_bar"),
    ("rho_estimate", "rho_estimate"),
    ("objective", "objective"),
    ("dual_objective", "dual_objective"),
    ("duality_gap", "duality_gap"),
    ("primal_residual", "primal_residual"),
    ("dual_residual", "dual_residual"),
    ("rel_kkt_error", "rel_kkt_error"),
)


CASES = json.loads(Path(__file__).with_name("pinned_double_results.json").read_text())


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("polish", [0, 1])
@pytest.mark.parametrize("device", ["cpu", "cuda", "native"])
def test_pinned_double_result(name, polish, device):
    case = CASES[name]
    run = case["runs"][polish]
    inputs = [np.asarray(a, dtype=np.float64) for a in case["qp"]]
    settings, opts = paired_settings(**run["settings"])
    assert settings["polishing"] == polish
    if device == "native":
        if not pinned_dll_path().is_file():
            pytest.skip("retained native DLL unavailable")
        with PinnedOSQPSession(*inputs, **settings) as session:
            assert session.setup_code == 0
            actual = session.solve()
    else:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        x, info = solve_torch_osqp_direct(
            *(torch.from_numpy(a).to(device) for a in inputs), opts
        )
        assert x.device.type == device and x.dtype == torch.float64
        actual = {key: info[field] for key, field in INFO_FIELDS}
        actual.update(status_polish=info["status_polish"], x=x)
        if "y" in run["native"]:
            actual["y"] = info["state"]["y"]
        for cert in ("prim_inf_cert", "dual_inf_cert"):
            if cert in run["native"]:
                actual[cert] = info[cert]
    if device == "cuda" and info["status_val"] == 1:
        P, q, A, l, u = inputs
        point = x.cpu().numpy().ravel()
        y = info["public_y"].cpu().numpy().ravel()
        z = info["state"]["z"].cpu().numpy().ravel()
        ax = A @ point
        assert np.max(np.abs(ax - np.clip(ax, l, u)), initial=0) <= info["eps_primal"]
        assert np.max(np.abs(ax - z), initial=0) <= info["eps_primal"]
        assert np.max(np.abs(P @ point + q + A.T @ y)) <= info["eps_dual"]
    for field, expected in run["native"].items():
        value = actual[field]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        if device == "cuda":
            reference = np.frombuffer(bytes.fromhex(expected), dtype="<f8")
            if field == "rho_estimate":
                # A residual ratio at convergence may have a nearly zero denominator.
                assert 1e-6 <= value <= 1e6
            else:
                np.testing.assert_allclose(np.asarray(value).ravel(), reference,
                                           rtol=1e-7, atol=1e-7, err_msg=field)
        else:
            assert np.asarray(value, dtype="<f8").tobytes().hex() == expected, field
