import importlib

import pytest
import torch

from pygranso.private.qpSteeringStrategy import qpSS
from pygranso.private.qpTerminationCondition import qpTC


def test_stationarity_qp_exhaustion_returns_outer_fallback_sentinel():
    termination = qpTC()
    termination.H = torch.eye(2, dtype=torch.float64)
    termination.all_grads = torch.eye(2, dtype=torch.float64)

    def fail(_matrix):
        raise RuntimeError("synthetic stationarity QP failure")

    termination.solveQP_fn = fail
    x, lambdas, stat_type, errors = termination.solveQPRobust(torch.float64)
    assert x is None
    assert lambdas is None
    assert stat_type == 0
    assert len(errors) == 3
    assert all("synthetic" in str(error) for error in errors)


def test_steering_qp_preserves_original_failure(monkeypatch):
    steering_module = importlib.import_module(
        "pygranso.private.qpSteeringStrategy"
    )
    steering = qpSS()
    steering.QPsolver = "osqp"
    steering.H = torch.eye(1, dtype=torch.float64)
    steering.f = torch.zeros((1, 1), dtype=torch.float64)
    steering.LB = -torch.ones((1, 1), dtype=torch.float64)
    steering.UB = torch.ones((1, 1), dtype=torch.float64)
    steering.device = torch.device("cpu")
    steering.double_precision = True
    steering.osqp_options = {}

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic steering QP failure")

    monkeypatch.setattr(steering_module, "solveQP", fail)
    with pytest.raises(RuntimeError, match="synthetic steering QP failure"):
        steering.solveSteeringDualQP()
