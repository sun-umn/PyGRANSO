"""Replay double-precision public outputs recorded from the pinned C oracle."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from c_parity_support import INFO_FIELDS, paired_settings
from pinned_osqp_c_oracle import pinned_dll_path
from pinned_osqp_c_session import PinnedOSQPSession

from pygranso.private.torchOSQP import solve_torch_osqp_direct


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
    for field, expected in run["native"].items():
        value = actual[field]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        assert np.asarray(value, dtype="<f8").tobytes().hex() == expected, field
